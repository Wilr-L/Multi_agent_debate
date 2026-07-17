"""
Three-Agent VLM Debate for Multi-Robot Task Planning  (Three_debate variant)
============================================================================
Built on VIKI-Bench. **THREE** VLMs debate to produce a consensus task plan
for three heterogeneous robots (R1, R2, R3 — any VIKI embodiment combo).
This is the 3-agent extension of the 2-agent debate one directory up.

Pipeline:
  Phase 1: Independent Proposal  — each of 3 VLMs proposes a joint plan
  Phase 2: Debate Loop           — VLM1 → VLM2 → VLM3 rotation; consensus
                                   reached when 3 CONSECUTIVE turns ACCEPT
  Phase 3: Execution             — run consensus plan in symbolic simulator
  Phase 4: Reflection & Re-Debate — on failure, all 3 reflect, merge revisions

run:
$env:APIMART_API_KEY = "sk-..."
E:\\anaconda3\\python.exe multi_agent_debate.py
"""

import json
import copy
import base64
import os
import re
import sys
import time
import random
import mimetypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import requests

# ─────────────────────────────────────────────
# 0a. Bring VIKI-Bench symbolic-verification package onto the import path.
#     The `eval` subpackage lives at:
#         data-pipeline/RoboFactory/utils/eval/
#     and uses *relative* imports internally (`from .entities import ...`),
#     so we register its parent directory and import via package syntax.
#     `import eval` would shadow the builtin only in *this* module's globals
#     (the builtin `eval()` lives in `builtins` and is unaffected for other
#     modules); we never call the builtin here.
# ─────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# `data-pipeline/RoboFactory/utils/eval/` may sit alongside this file OR
# one directory up (the Three_debate variant lives in a subdir of the
# main repo; data-pipeline is only at the root). Walk up at most 2 levels.
_VIKI_EVAL_PARENT = None
for base in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    cand = os.path.join(base, "data-pipeline", "RoboFactory", "utils")
    if os.path.isdir(os.path.join(cand, "eval")):
        _VIKI_EVAL_PARENT = cand
        break
if _VIKI_EVAL_PARENT is None:
    raise ImportError("Could not locate data-pipeline/RoboFactory/utils/eval "
                      "under this file's dir or its parents.")
if _VIKI_EVAL_PARENT not in sys.path:
    sys.path.insert(0, _VIKI_EVAL_PARENT)

from eval.eval import Eval as _VIKIEval                     # noqa: E402
from eval.entities import Position as _VIKIPosition         # noqa: E402
from eval.entities import Asset as _VIKIAsset               # noqa: E402
from eval.checker import Checker as _VIKICheckerBase        # noqa: E402
from eval.env import SimEnv as _VIKISimEnv                  # noqa: E402
from eval.eval_viki_2 import filter_none_values as _filter_none_values  # noqa: E402

from itertools import combinations as _combinations         # noqa: E402

# ─────────────────────────────────────────────
# 0. Configuration & Data Structures
# ─────────────────────────────────────────────

class DebateRole(Enum):
    VLM1_R1_ADVOCATE = "vlm1"   # advocates for R1
    VLM2_R2_ADVOCATE = "vlm2"   # advocates for R2
    VLM3_R3_ADVOCATE = "vlm3"   # advocates for R3 — Three_debate variant


@dataclass
class RobotProfile:
    """Embodiment description. `name` must be one of VIKI-Bench's robot types
    (`stompy`, `fetch`, `unitree_h1`, `panda`, `unitree_go2`, `anymal_c`) —
    used as the key into ROBOT_DESCRIPTION / AGENT_AVAIL_ACTIONS below."""
    name: str           # VIKI robot type, e.g. "fetch"
    robot_id: str       # role label in the plan, e.g. "R1"

    @property
    def description(self) -> str:
        return ROBOT_DESCRIPTION.get(self.name, "")

    @property
    def available_actions(self) -> list[str]:
        return AGENT_AVAIL_ACTIONS.get(self.name, [])

    @property
    def end_effector_num(self) -> int:
        return AGENT_END_EFFECTOR_NUM.get(self.name, 0)


@dataclass
class TaskPlan:
    """A complete multi-robot task plan."""
    steps: list[dict]   # [{"step": 1, "R1_action": "...", "R2_action": "..."}, ...]
    reasoning: str      # chain-of-thought
    raw_text: str       # original VLM output


@dataclass
class DebateMessage:
    """One turn in the debate."""
    role: DebateRole
    content: str
    proposed_plan: Optional[TaskPlan] = None
    accepts_current_plan: bool = False


@dataclass
class DebateState:
    """Tracks the full debate history and current consensus."""
    task_description: str
    scene_image_path: str
    robot_profiles: list[RobotProfile]
    messages: list[DebateMessage] = field(default_factory=list)
    current_plan: Optional[TaskPlan] = None
    round_num: int = 0
    consensus_reached: bool = False
    execution_feedback: Optional[str] = None  # from simulator on failure
    initial_world_state: str = ""             # rendered by simulator before Phase 1
    skip_vlm1_first_critique: bool = False    # set by _merge_proposals; consumed by phase2_debate


# ─────────────────────────────────────────────
# 1. Robot Profiles (Example: VIKI-Bench)
# ─────────────────────────────────────────────

# ── VIKI-Bench canonical embodiment + action vocabulary ──
# Verbatim from VIKI-R/eval/VIKI-L2/qwen.py. Keep these in sync if VIKI updates
# its prompt so the symbolic checker and the VLM share one vocabulary.

ROBOT_DESCRIPTION = {
    'stompy':      'A bipedal robot designed for dynamic walking and stomping tasks, featuring articulated arms. Color: Light blue body with yellow and orange accents.',
    'fetch':       'A wheeled robot with a flexible arm for object manipulation, designed for mobility and dexterity. Color: White with blue and black accents.',
    'unitree_h1':  'A humanoid robot with arms and legs designed for human-like movements and tasks. Color: Black.',
    'panda':       'A fixed robotic arm designed for precise and delicate manipulation tasks. Color: White with black accents.',
    'anymal_c':    'A quadrupedal robot built for navigating rough terrains and performing complex tasks with four articulated legs. Color: Red and black with some accents.',
    'unitree_go2': 'A compact quadrupedal robot optimized for agile movement and stability with four legs for efficient locomotion. Color: White.',
}

ACTION_DESCRIPTION = {
    'Move':     "Command ['Move', 'object']: Robot R moves to the specified object.",
    'Open':     "Command ['Open', 'object']: Open the object held by the Robot R's end effector.",
    'Close':    "Command ['Close', 'object']: Close the object held by the Robot R's end effector.",
    'Reach':    "Command ['Reach', 'object']: Robot R reaches the specified object.",
    'Grasp':    "Command ['Grasp', 'object']: Robot R's end effector performs a grasping operation on a specified object.",
    'Place':    "Command ['Place', 'object']: Place the object held by the Robot R's end effector at a specified location (the release point, not the object itself).",
    'Push':     "Command ['Push', 'object', 'R1']: Robot R pushes the object to robot R1.",
    'Interact': "Command ['Interact', 'object']: A general interaction operation, flexible for representing interactions with any asset.",
}

AGENT_AVAIL_ACTIONS = {
    'panda':       ['Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'fetch':       ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'unitree_go2': ['Move', 'Push', 'Interact'],
    'unitree_h1':  ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'stompy':      ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'anymal_c':    ['Move', 'Push', 'Interact'],
}

AGENT_END_EFFECTOR_NUM = {
    'panda': 1, 'fetch': 1, 'unitree_go2': 0,
    'unitree_h1': 2, 'stompy': 2, 'anymal_c': 0,
}

FETCH_PROFILE  = RobotProfile(name="fetch",  robot_id="R1")
STOMPY_PROFILE = RobotProfile(name="stompy", robot_id="R2")


# ─────────────────────────────────────────────
# 2. Prompt Templates
# ─────────────────────────────────────────────

# ── 2.1 System Prompts ──

SYSTEM_PROMPT = """\
You are a plan creator for a multi-robot team in VIKI-Bench. You are the ADVOCATE for \
robot {robot_id} ({robot_name}). I will provide you with an image of the robots in a scene, \
the available robots and their action primitives, and a task description. You need to debate \
with the advocates for {partner_a_id} ({partner_a_name}) and {partner_b_id} ({partner_b_name}) \
to jointly create a plan that completes the task.

You must first analyze the image to fully understand the scene depicted. Then, analyze the task \
description. Finally, propose / critique / revise the plan accordingly. Your reasoning must \
strictly adhere to the visual content of the image and the task description — no assumptions, \
hypotheses, or guesses are allowed.

## Available robots and their action primitives
- {robot_id} ({robot_name}): {robot_description}
  Available actions: {robot_avail_actions}
- {partner_a_id} ({partner_a_name}): {partner_a_description}
  Available actions: {partner_a_avail_actions}
- {partner_b_id} ({partner_b_name}): {partner_b_description}
  Available actions: {partner_b_avail_actions}

## VIKI-Bench action primitives and rules
Action must follow the following format as a JSON list, for example [\"Move\", \"plate\"] or [\"grasp\", \"banana\"]. It describes the single action that robot will perform in this step, with the following format: action_type, target_object_or_location\nAction primitives and descriptions: {{'Move': \"Command ['Move', 'object']: Robot R moves to the specified object.(Move to the object! Not move the object to other place!)\", 'Reach': \"Command ['Reach', 'object']: Robot R reaches the specified object.\", 'Grasp': \"Command ['Grasp', 'object']: Robot R's end effector performs a grasping operation on a specified object.\", 'Place': \"Command ['Place', 'object']: Place the thing held by the Robot R's end effector at a specified location ('object' means location).\", 'Open': \"Command ['Open', 'object']: Open the object held by the Robot R's end effector.\", 'Close': \"Command ['Close', 'object']: Close the object held by the Robot R's end effector.\", 'Push': \"Command ['Push', 'object', 'R1']: Robot R pushes the object to robot R1.\", 'Interact': \"Command ['Interact', 'object']: A general interaction operation, flexible for representing interactions with any asset.\"}}
Use exact object and location names from the task, relevant assets, and world state. Do not invent new entity names.
Choose the primitive that advances the current object state, not just the task name.
  - If the robot is not near the object, use Move on the object. Note that the robot should move to the object like apple but not the location like table or cabinet.
  - If the robot is at/near the object but has not reached it, use Reach on the object.
  - If the robot has reached the object and is not carrying it, use Grasp on the object.
  - If the robot is carrying an object and it is not at the target, use Move on the target location or target object.
  - If the robot is carrying an object at the target area, use Place on the target location.
  - If an appliance or device must be started or activated, use Interact on that appliance after the required object is placed or available.
  - If the robot needs to cut an object on a cutting borad, the robot should hold a knife, and move to the cutting board and then interact with the knife to cut the object on the cutting board.
  - Note that the robot should not always wait throughout all steps.

## Your role
You understand ALL THREE robots, but your primary responsibility is to ensure that {robot_id}'s \
actions in the plan are feasible, efficient, and well-coordinated with {partner_a_id} and \
{partner_b_id}. When proposing or critiquing plans, pay special attention to:
1. Whether {robot_id}'s assigned actions are within its **available action set** above.
2. Whether the timing/sequencing avoids conflicts (e.g., two robots reaching for the same \
object simultaneously, or one robot blocking another's path).
3. Whether {robot_id} could do certain tasks better than {partner_a_id} / {partner_b_id}, or vice versa.
4. **Each robot can only perform ONE action per time step.** All three robots may work in \
parallel but each is limited to one action per step.

## Required output format (strict JSON)
Every plan you output MUST follow this exact structure:
```json
{{
  "reasoning": "step-by-step chain of thought...",
  "steps": [
    {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Move", "apple"], "R3": ["Move", "box"]}}}},
    {{"step": 2, "actions": {{"R1": ["Reach", "pumpkin"], "R2": ["Reach", "apple"], "R3": ["Reach", "box"]}}}}
  ]
}}
```
Rules:
- `step` is the time step number (starts at 1, increments sequentially).
- `actions` is a dict mapping robot id ("R1", "R2", "R3") to a list \
  `[action_type, target_object_or_location, (optional: extra_argument)]`.
- Only use action primitives that are in that robot's "Available actions" list above.
- If a robot has no action in a step, set its value to `["Wait"]`.
"""


# ── 2.2 Phase 1: Independent Proposal Prompts ──

PROPOSAL_PROMPT = """\
## Task
Look at the scene image carefully. Here is the task description:
{task_description}

## Initial world state (from the symbolic simulator)
{world_state}

## Instructions
Propose a sub-plan focusing on what {robot_id} ({robot_name}) should do to help accomplish \
this task. Also suggest what the partner robots {partner_a_id} ({partner_a_name}) and \
{partner_b_id} ({partner_b_name}) should do, based on your understanding of their capabilities. \
The initial world state above is the ground truth for object positions, container open/closed \
status, and what each robot is currently holding — your plan's pre-conditions must be \
consistent with it.

Think step by step:
1. What objects are visible in the scene? Where are they (cross-check against the world state)?
2. What is the goal state?
3. What actions does {robot_id} need to perform?
4. What actions should {partner_a_id} and {partner_b_id} perform in parallel?
5. Are there any dependencies or ordering constraints between the three robots?

Output the complete joint plan in the required JSON format.
"""


MERGE_PROMPT = """\
## Your task — integrate three independent proposals into one joint plan
You and the advocates for {partner_a_id} and {partner_b_id} have each independently proposed \
a plan for the same task. Your job now is to combine the three proposals into a single \
coherent joint plan that will be the starting point for debate.

## Your own proposal (for {robot_id} = {robot_name})
{plan_self_json}

## {partner_a_id}'s advocate's proposal (for {partner_a_id} = {partner_a_name})
{plan_a_json}

## {partner_b_id}'s advocate's proposal (for {partner_b_id} = {partner_b_name})
{plan_b_json}

Output ONLY the merged plan as a fresh plan in the standard JSON schema (do NOT use an \
ACCEPT/REVISE verdict wrapper). Every step MUST contain an entry for every robot \
(`["Wait"]` if idle):
```json
{{
  "reasoning": "why you merged this way, including any conflict resolutions across the three plans...",
  "steps": [
    {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Move", "apple"], "R3": ["Move", "box"]}}}}
  ]
}}
```
"""


# ── 2.3 Phase 2: Debate Prompts ──

CRITIQUE_PROMPT = """\
## Current joint plan under review
{current_plan_json}

## The most recent critique (from the advocate who spoke just before you)
{other_critique}

## Debate history
{debate_history}

## Your task
You are developing a task plan by debating with the advocates for {partner_a_id} and \
{partner_b_id}. Only a task plan that all THREE advocates ACCEPT will be implemented.
Now review the current plan and the other advocates' critiques from the perspective of \
{robot_id} ({robot_name}). Evaluate:

1. **Feasibility**: Can {robot_id} physically execute its assigned actions? \
   Check reachability, payload, manipulation type.
2. **Efficiency**: Is there a better task allocation to save steps? Could {robot_id} take on \
   something currently assigned to {partner_a_id} or {partner_b_id} (or vice versa) more efficiently?
3. **Coordination**: Are there timing conflicts? Two robots reaching for the same object \
   simultaneously? One robot blocking another's path?
4. **Completeness**: Does the plan achieve the task goal? Are any steps missing?

If the plan is acceptable, respond with:
```json
{{"verdict": "ACCEPT", "reasoning": "why the plan is good"}}
```

If you want to revise, respond with (steps follow the SYSTEM_PROMPT's nested `actions` schema \
— every step must include all three robots):
```json
{{
  "verdict": "REVISE",
  "issues": ["issue 1", "issue 2", ...],
  "revised_plan": {{
    "reasoning": "...",
    "steps": [
      {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Wait"], "R3": ["Move", "box"]}}}}
    ]
  }}
}}
```
"""

# Placeholder used the very first time CRITIQUE_PROMPT is rendered (round 1,
# VLM1's turn): no advocate has spoken yet in Phase 2.
NO_PRIOR_CRITIQUE = "No other critique because this is the first turn in this debate round."


# ── 2.4 Phase 4: Reflection Prompt (after execution failure) ──

REFLECTION_PROMPT = """\
## Execution result: FAILED
The previously agreed plan was executed in the simulator and FAILED.

## Execution feedback
{execution_feedback}

## Failed plan
{failed_plan_json}

## Previous debate history (summary)
{debate_summary}

## Your task
Analyze the failure from {robot_id}'s perspective:
1. Which step failed and why?
2. Was it a planning error (wrong action sequence) or an execution error \
   (action was correct but physically failed)?
3. What should be changed?

Propose a revised plan that addresses the failure. Be specific about what changed and why in "reasoning".
Output in the standard JSON plan format.
"""


# ─────────────────────────────────────────────
# 3. VLM Interface — APIMart Chat Completions
# ─────────────────────────────────────────────
#
# Client for APIMart's OpenAI-compatible chat-completions endpoint.
#   Endpoint:  POST {base_url}/chat/completions   (base_url defaults to
#              https://api.apimart.ai/v1)
#   Auth:      Bearer token via constructor `api_key=` or $APIMART_API_KEY
#   Default model: gpt-4o (override via `model_name=`; APIMart proxies many
#              providers — GPT-4o, Claude, Qwen-VL, etc. — all via the same
#              OpenAI-style payload, including `image_url` parts for vision)
#
# Each VLMInterface instance keeps its own multi-turn conversation_history
# so all three debaters maintain independent memories across rounds.


class RunLogger:
    """
    Dump every successful VLM call into a numbered `.txt` under `log_dir`,
    one file per call. Two VLMs may share a single logger — the counter is
    shared so files interleave in chronological order (001_vlm1.txt,
    002_vlm2.txt, 003_vlm1.txt, ...).

    Each file contains:
      • header (call #, role, model, timestamp)
      • every message in the API request (system / user / assistant history),
        with images noted as `[image: <path or size>]` instead of dumping
        raw base64
      • the assistant response text
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        # Write a small README so the folder is self-explanatory.
        readme = self.log_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "VLM call log. One file per chat-completion request, numbered\n"
                "chronologically across all three debaters (vlm1 = R1, vlm2 = R2, vlm3 = R3).\n"
                "Each file shows the full message stack sent to the API plus the\n"
                "model's reply. Images are summarized (not dumped as base64).\n",
                encoding="utf-8",
            )

    def log_call(self, *, role: str, model_name: str,
                 messages: list, response: str,
                 reasoning: Optional[str] = None) -> Path:
        self._counter += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.log_dir / f"{self._counter:03d}_{role}.txt"

        lines = [
            f"=== Call #{self._counter} | role={role} | model={model_name} | {ts} ===",
            "",
        ]
        for i, msg in enumerate(messages):
            lines.append(f"--- [{i}] {msg['role']} ---")
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        lines.append(part.get("text", ""))
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            lines.append(f"[image: data URL ({len(url)} chars, truncated from log)]")
                        else:
                            lines.append(f"[image: {url}]")
                    else:
                        lines.append(f"[unsupported content part: {ptype}]")
            else:
                lines.append(str(content))
            lines.append("")

        # Optional: thinking-mode reasoning, separate from the final answer.
        if reasoning:
            lines.append("--- assistant (reasoning_content) ---")
            lines.append(reasoning)
            lines.append("")

        lines.append("--- assistant (response) ---")
        lines.append(response)
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


class VLMInterface:
    """APIMart Chat-Completions client (gpt-4o by default; override
    via `model_name=` for any other model APIMart hosts — e.g.
    `gpt-4o`, `claude-3-5-sonnet`, `qwen-vl-max`, etc.)."""

    DEFAULT_MODEL    = "gpt-4o"
    # DEFAULT_MODEL    = "Qwen/Qwen3-VL-32B-Instruct"   # if APIMart hosts it
    DEFAULT_BASE_URL = "https://api.apimart.ai/v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        role: DebateRole = DebateRole.VLM1_R1_ADVOCATE,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        top_p: float = 0.7,
        timeout: float = 180.0,
        max_retries: int = 3,
        image_detail: str = "auto",          # "auto" | "low" | "high"
        extra_params: Optional[dict] = None, # e.g. {"enable_thinking": True, "thinking_budget": 4096}
        logger: Optional[RunLogger] = None,  # shared across both VLMs to merge logs
    ):
        self.model_name   = model_name
        self.role         = role
        self.api_key      = api_key or os.environ.get("APIMART_API_KEY")
        self.base_url     = base_url.rstrip("/")
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.top_p        = top_p
        self.logger       = logger
        self.timeout      = timeout
        self.max_retries  = max_retries
        self.image_detail = image_detail
        self.extra_params = extra_params or {}
        self.conversation_history: list[dict] = []

    # ── public API ──

    def set_system_prompt(self, system_prompt: str):
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

    def query(self, user_prompt: str, image_path: Optional[str] = None) -> str:
        """
        Send a (text + optional image) message to the model and return the
        assistant's text. Updates conversation_history so multi-turn debate
        works across calls.
        """
        if not self.api_key:
            raise RuntimeError(
                "APIMart API key not set. Pass `api_key=` to VLMInterface "
                "or export APIMART_API_KEY in the environment."
            )

        user_msg = self._build_user_message(user_prompt, image_path)
        # Stateless: send only system + current user_msg. The debate
        # engine already embeds all relevant state in each prompt, so
        # replaying conversation_history just duplicates info and was
        # blowing past --max-model-len in long retry loops. See
        # _build_stateless_messages docstring for the full rationale.
        messages = self._build_stateless_messages(self.conversation_history, user_msg)

        payload = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "top_p":       self.top_p,
            **self.extra_params,
        }

        # ── Outer loop: retry the whole completion if APIMart returns
        #    an empty body with no truncation (transient model issue). The
        #    HTTP-level retry inside `_post_with_retry` only handles network
        #    errors and 5xx — empty-but-200 needs its own retry. ──
        max_empty_retries = 3
        last_choice = None
        last_usage  = {}
        assistant_text  = ""
        reasoning_text  = ""
        finish_reason: Optional[str] = None

        for attempt in range(max_empty_retries):
            data = self._post_with_retry("/chat/completions", payload)

            # APIMart / Qwen3 thinking-mode models put the user-visible
            # answer in `content`; if reasoning mode is on, it can land in
            # `reasoning_content` while `content` is empty. Read both.
            try:
                choice = data["choices"][0]
                msg = choice["message"]
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f"Unexpected response shape from APIMart: {data!r}"
                ) from e
            finish_reason   = choice.get("finish_reason")
            last_choice     = choice
            last_usage      = data.get("usage", {}) or {}
            content_text    = (msg.get("content") or "").strip()
            reasoning_text  = (msg.get("reasoning_content") or "").strip()

            if content_text:
                assistant_text = content_text
                break
            if reasoning_text:
                # Thinking-mode fallback: the model only emitted reasoning.
                if not getattr(self, "_reasoning_fallback_warned", False):
                    print(f"[INFO] {self.role.value}: empty `content`; using "
                          f"`reasoning_content` ({len(reasoning_text)} chars). "
                          f"(Pass extra_params={{'enable_thinking': False}} to disable.)")
                    self._reasoning_fallback_warned = True
                assistant_text = reasoning_text
                break

            # Both empty. Decide whether to retry.
            ct = last_usage.get("completion_tokens")
            transient = (
                finish_reason in (None, "stop")
                and ct == 0
                and attempt < max_empty_retries - 1
            )
            if transient:
                backoff = 2.0 * (attempt + 1)
                print(f"[WARN] {self.role.value}: empty response "
                      f"(finish_reason={finish_reason!r}, completion_tokens=0). "
                      f"Retrying in {backoff:.0f}s "
                      f"(attempt {attempt + 2}/{max_empty_retries})...")
                time.sleep(backoff)
                continue

            # Not a transient case — produce the targeted error.
            hint = ""
            if finish_reason == "length":
                hint = (f" finish_reason='length' AND completion_tokens={ct} "
                        f">= max_tokens={self.max_tokens} → bump --max-tokens.")
            elif finish_reason == "content_filter":
                hint = (" finish_reason='content_filter' → output was blocked "
                        "by a safety filter. Inspect the request/image.")
            elif finish_reason in (None, "stop") and ct == 0:
                hint = (f" Already retried {max_empty_retries} times — model "
                        f"still emitted 0 tokens. Common causes: image too "
                        f"large / unsupported format, prompt too long for the "
                        f"model's context window, or a persistent model issue.")
            elif finish_reason in (None, "stop"):
                hint = (" The model stopped on its own but emitted no visible "
                        "text. If this is a Qwen3 thinking-mode model that "
                        "supports `enable_thinking`, try setting it to False; "
                        "otherwise bump --max-tokens.")
            raise RuntimeError(
                f"APIMart returned empty content AND empty reasoning_content. "
                f"finish_reason={finish_reason!r}, usage={last_usage}.{hint} "
                f"Raw choice: {last_choice!r}"
            )

        # Dump the full request stack + response to disk if a logger is wired.
        if self.logger is not None:
            try:
                self.logger.log_call(
                    role=self.role.value,
                    model_name=self.model_name,
                    messages=messages,
                    response=assistant_text,
                    reasoning=reasoning_text if reasoning_text and reasoning_text != assistant_text else None,
                )
            except Exception as e:                       # never let logging break a run
                print(f"[WARN] RunLogger failed: {e}")

        self.conversation_history.append(user_msg)
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text}
        )
        return assistant_text

    def reset_history(self):
        """Drop everything except the system prompt."""
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []

    # ── internals ──

    @staticmethod
    def _encode_image_as_data_url(image_ref: str) -> str:
        """
        Convert a local path to a `data:image/...;base64,...` URL.
        HTTPS / data: URLs are returned unchanged (APIMart accepts both).
        """
        if image_ref.startswith(("http://", "https://", "data:")):
            return image_ref
        path = Path(image_ref)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_ref}")
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _strip_images_from_history(history: list[dict]) -> list[dict]:
        """Return a copy of `history` with every `image_url` content part
        removed. Used by `query()` to avoid re-sending the same scene
        image on every debate turn — duplicates trip vLLM's
        `limit_mm_per_prompt` cap and waste KV tokens on API backends
        that don't error out (e.g. APIMart). The image stays on the
        CURRENT user turn, which is enough for the model to attend to."""
        out: list[dict] = []
        for msg in history:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [p for p in content if p.get("type") != "image_url"]
                if not text_parts:
                    text_parts = [{"type": "text", "text": ""}]
                out.append({**msg, "content": text_parts})
            else:
                out.append(msg)
        return out

    @staticmethod
    def _build_stateless_messages(conversation_history: list[dict],
                                  user_msg: dict) -> list[dict]:
        """Return `[system_prompt, user_msg]` (or just `[user_msg]` if
        no system prompt was set).

        Why stateless: MultiAgentDebateEngine already embeds the full
        debate state explicitly in every prompt (CRITIQUE_PROMPT carries
        current_plan_json + other_critique + a 6-message debate_history;
        REFLECTION_PROMPT carries failed_plan_json + debate_summary; …).
        Carrying conversation_history as well duplicates the same info
        in raw form and is what blew past --max-model-len in long
        retry loops. conversation_history is still kept on the
        VLMInterface instance for logger / inspection use — just not
        replayed back to the model."""
        system_msg = None
        if conversation_history and conversation_history[0].get("role") == "system":
            system_msg = conversation_history[0]
        return ([system_msg] if system_msg else []) + [user_msg]

    def _build_user_message(self, prompt: str, image_path: Optional[str]) -> dict:
        if not image_path:
            return {"role": "user", "content": prompt}
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":    self._encode_image_as_data_url(image_path),
                        "detail": self.image_detail,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with simple exponential backoff on 429 / 5xx / network errors."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"network error: {e}"
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            # Non-retryable (400, 401, 403, 404, ...) — surface immediately.
            raise RuntimeError(
                f"APIMart API error {resp.status_code}: {resp.text[:500]}"
            )
        raise RuntimeError(
            f"APIMart API failed after {self.max_retries} attempts. Last error: {last_err}"
        )


# ─────────────────────────────────────────────
# 4. Plan Parser
# ─────────────────────────────────────────────

def _normalize_action(action) -> list:
    """
    Coerce a single action into VIKI's canonical list form
    `[Primitive, target, (optional extra)]`.
      ['Move', 'pear']  → ['Move', 'pear']         (already canonical)
      'Move pear'       → ['Move', 'pear']         (legacy string form)
      'Wait' / '' / None → ['Wait']                (idle marker)
    """
    if action is None:
        return ["Wait"]
    if isinstance(action, (list, tuple)):
        return [str(x) for x in action] or ["Wait"]
    if isinstance(action, str):
        s = action.strip()
        if not s or s.lower() == "wait":
            return ["Wait"]
        return s.replace(",", " ").split()
    return ["Wait"]


def _normalize_step(step: dict, fallback_idx: int) -> dict:
    """Coerce a step into `{step: N, actions: {R1: [...], R2: [...]}}`."""
    out = {"step": step.get("step", fallback_idx + 1)}
    if isinstance(step.get("actions"), dict):
        out["actions"] = {k: _normalize_action(v) for k, v in step["actions"].items()}
    else:
        # Legacy flat schema: R1_action / R2_action at the top of the step dict.
        actions = {}
        for rid in ("R1", "R2"):
            v = step.get(f"{rid}_action")
            if v is not None:
                actions[rid] = _normalize_action(v)
        out["actions"] = actions
    return out


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _iter_balanced_json_objects(text: str):
    """Yield every top-level, balanced ``{...}`` substring in `text`, in
    order. Brace-tracking is string-aware — braces that appear INSIDE a
    JSON string literal (including escaped quotes) do NOT open/close an
    object, so reasoning prose like ``"reasoning": "move to {pandaB}"``
    does not corrupt the scan."""
    depth, start = 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _extract_json_dicts(response: str) -> list[dict]:
    """Return every parseable JSON dict in `response`. Considers both
    ```json fenced blocks AND raw balanced ``{...}`` spans, so it works
    whether or not the model wrapped its answer in a code fence. Order is
    preserved so callers can take the LAST (the model's final answer,
    after any chain-of-thought)."""
    out: list[dict] = []
    seen: set[str] = set()
    candidates: list[str] = list(_JSON_FENCE_RE.findall(response))
    candidates.extend(_iter_balanced_json_objects(response))
    for c in candidates:
        c = c.strip()
        if c in seen:
            continue
        seen.add(c)
        try:
            # strict=False tolerates literal newlines / control chars INSIDE
            # string values — the single most common malformed-JSON mistake
            # VLMs make (they write a multi-line `"reasoning": "..."` with
            # real line breaks instead of escaped `\n`). Default strict=True
            # rejects those with "Invalid control character" and the whole
            # ACCEPT/REVISE verdict would be lost.
            data = json.loads(c, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _last_answer_dict(response: str) -> Optional[dict]:
    """Return the LAST JSON dict that actually carries an answer — a
    `verdict`, `steps`, or `revised_plan` key. Ignores stray/echoed JSON
    fragments in the model's reasoning. Falls back to the last dict of
    any shape, else None.

    This is the key robustness fix over the old `first-{ ... last-}` slice,
    which broke whenever the model emitted ANY brace before its real
    answer (echoed schema, prose with `{}`, a worked example, …) — a
    misparse there silently flipped a genuine ACCEPT into a non-accept and
    reset the debate's consecutive-ACCEPT streak, so consensus was never
    detected and the loop ran to max rounds."""
    dicts = _extract_json_dicts(response)
    answers = [d for d in dicts
               if "verdict" in d or "steps" in d or "revised_plan" in d]
    if answers:
        return answers[-1]
    return dicts[-1] if dicts else None


def parse_plan_from_response(response: str) -> Optional[TaskPlan]:
    """Extract structured plan from VLM response and normalize to VIKI schema."""
    data = _last_answer_dict(response)
    if data is None:
        return None

    # Handle both ACCEPT and REVISE formats
    if "verdict" in data:
        if str(data["verdict"]).upper() == "ACCEPT":
            return None  # No new plan, current one is accepted
        rp = data.get("revised_plan")
        if isinstance(rp, dict):
            data = rp

    if "steps" not in data or not isinstance(data["steps"], list):
        return None

    try:
        steps = [_normalize_step(s if isinstance(s, dict) else {}, i)
                 for i, s in enumerate(data["steps"])]
    except Exception as e:
        print(f"[WARN] Failed to normalize plan steps: {e}")
        return None
    return TaskPlan(
        steps=steps,
        reasoning=data.get("reasoning", ""),
        raw_text=response,
    )


def is_accept_response(response: str) -> bool:
    """Check if a VLM response is an ACCEPT verdict. Robust to models that
    emit chain-of-thought (and stray braces) before the final JSON."""
    data = _last_answer_dict(response)
    if data is not None and "verdict" in data:
        return str(data.get("verdict", "")).upper() == "ACCEPT"
    # Regex fallback for when no clean JSON object could be isolated but the
    # verdict is still stated literally. Only treat as ACCEPT if an ACCEPT
    # verdict appears and no REVISE verdict does.
    has_accept = re.search(r'["\']?verdict["\']?\s*:\s*["\']ACCEPT["\']',
                           response, re.IGNORECASE)
    has_revise = re.search(r'["\']?verdict["\']?\s*:\s*["\']REVISE["\']',
                           response, re.IGNORECASE)
    return bool(has_accept and not has_revise)


# ─────────────────────────────────────────────
# 5. Simulator Interface — VIKI-Bench Symbolic Verification
# ─────────────────────────────────────────────
#
# Replaces physical simulation with VIKI-Bench's symbolic mechanism:
#   • Checker C   — verifies each action's pre-conditions (reachability,
#                   free end-effectors, action-target type, container
#                   isolation, payload, etc.).
#   • World Sim S — applies the abstract state transition produced by each
#                   action; afterwards goal / temporal constraints are
#                   evaluated against the resulting world state.
#
# Source modules (cloned VIKI repo):
#   data-pipeline/RoboFactory/utils/eval/{checker.py, env.py, eval.py,
#                                         entities.py, eval_viki_2.py}

# Assets that may serve as containers (mirrors eval_viki_2.CONTAINER_ASSETS).
_CONTAINER_ASSETS = {
    "plate", "cabinet", "drawer", "bowl", "sink",
    "toaster", "tray", "cardboardbox",
}

# Plan-key ↔ VIKI robot-id mapping. Extend if you use more than two robots.


class _PatchedSimEnv(_VIKISimEnv):
    """
    Bug-fix overlay on VIKI's `SimEnv.sim_step`.

    Upstream's `place` branch ([env.py:170-171] of the VIKI clone) does
        elif isinstance(params[1], Asset):    # asset as position
            new_env_status["assets"][carried_object.name]["pos"] = params[1]
    when the place target is a non-container Asset (e.g. `cutting_board`):
    it stores the Asset *object itself* as the carried object's `.pos`.
    Any later check that does `<asset>.pos.isolated` / `.pos.name` then
    crashes with `'Asset' object has no attribute 'isolated'`.

    We post-process the env after every `sim_step`: any asset whose `.pos`
    is not a Position gets re-wrapped as a fresh `Position(name=<that
    asset's name>)` so downstream Checker code sees the type it expects.
    """

    def sim_step(self, commands: list):
        super().sim_step(commands)
        for asset in self.assets.values():
            if not isinstance(asset.pos, _VIKIPosition):
                # `asset.pos` got set to an Asset (the buggy `place` branch).
                # Re-wrap as a Position so `pos.name` / `pos.isolated` work.
                asset.pos = _VIKIPosition(name=asset.pos.name)


class _PatchedChecker(_VIKICheckerBase):
    """
    Bug-fix overlay on VIKI's `Checker.check_compatible_constraints`.

    Upstream ([checker.py:160] of the VIKI clone) does:
        target_container = params[commands.index('close')][0]
    but `params[i]` is `[Agent, Target1, ...]` — index [0] is the *agent
    performing the close*, not the container being closed. The container
    is at index [1]. As written, the line raises
        AttributeError: 'Agent' object has no attribute 'container_position'
    on every plan containing a `close` command.

    This subclass keeps the rest of the method identical and only fixes
    the indexing (plus defensive guards for malformed close commands).
    """

    def check_compatible_constraints(self, step_commands: list,
                                     assets: dict = None, agents: dict = None):
        if not assets:
            assets = {}
        if not agents:
            agents = {}
        commands = [c[0] for c in step_commands if c]
        params   = [c[1:] for c in step_commands if c]
        target_agents = [p[0].name for p in params]
        if len(target_agents) != len(set(target_agents)):
            return False

        target_entities: dict = {}
        for idx, inst_params in enumerate(params):
            for param in inst_params:
                if param.name in assets:
                    target_entities.setdefault(param.name, []).append(idx)

        for asset, inst_idx in target_entities.items():
            if len(inst_idx) < 2:
                continue
            operation_names = [commands[i] for i in inst_idx]
            for op1, op2 in _combinations(operation_names, 2):
                if not self.check_compatible_paired_actions(op1, op2):
                    return False

        # `close` must not collide with anything else still touching the
        # container's interior position.
        if 'close' in commands:
            close_params = params[commands.index('close')]
            if len(close_params) < 2:
                return True                                # malformed; skip
            target_container = close_params[1]             # ← FIXED ([1] not [0])
            if not hasattr(target_container, 'container_position'):
                return True                                # not a container; skip
            for idx, inst_params in enumerate(params):
                for param in inst_params:
                    if (isinstance(param, _VIKIAsset)
                            and param.pos == target_container.container_position
                            and commands[idx] not in ['move', 'close']):
                        return False
        return True


class DebateEval(_VIKIEval):
    """
    Extends VIKI's `Eval` with **structured per-step error detail** so Phase 4
    reflection can feed the failing step + a natural-language reason back into
    the debate.

    On failure, populates:
      • `error_desc_code`: machine-readable category (INVALID_COMMAND, ...
        NOT_FOUND_ENTITY, ACTION_NOT_FEASIBLE, ACTION_NOT_COMPATIBLE,
        FAILED_TEMPORAL_CONSTRAINT, FAILED_GOAL_CONSTRAINT)
      • `error_detail`:   {step, robot, command, reason, ...} — exact fields
        depend on the failure type
    `get_feedback_for_debate()` formats both into the multi-line block the
    VLMs see in the reflection prompt.
    """

    def __init__(self):
        super().__init__()
        # Swap in the bug-fixed Checker (see _PatchedChecker for details).
        self.checker = _PatchedChecker()
        self.error_detail: dict = {}

    def set_env(self, env_metadata):
        # Use the bug-fixed SimEnv (see _PatchedSimEnv for details) instead
        # of VIKI's stock one. Same constructor signature.
        self.env = _PatchedSimEnv(metadata=env_metadata)

    def eval(self, command_records: list):
        all_commands = []
        for step_idx, command_record in enumerate(command_records):
            commands = []
            for robot_name, command_desc in command_record.items():
                if not self.is_valid_sequence(command_desc):
                    self.error_desc_code = "INVALID_COMMAND"
                    self.error_detail = {
                        "step": step_idx + 1,
                        "robot": robot_name,
                        "command": command_desc,
                        "reason": f"'{command_desc}' is not a valid command format.",
                    }
                    return False
                parsed_command = self.parse_command(command_desc)
                parsed_command.insert(1, robot_name)
                commands.append((robot_name, command_desc, parsed_command))
            all_commands.append(commands)

        satisfied_temporal_constraints = [False] * len(
            self.env.metadata.get("temporal_constraints", []) or []
        )

        for step_idx, commands in enumerate(all_commands):
            step_commands = []
            for robot_name, raw_cmd, command in commands:
                operation_name = command[0]
                operation_params = command[1:]
                operation_entities = []
                for p in operation_params:
                    if p in self.env.agents:
                        operation_entities.append(self.env.agents[p])
                    elif p in self.env.assets:
                        operation_entities.append(self.env.assets[p])
                    elif operation_name in ["move", "place"]:
                        operation_entities.append(_VIKIPosition(name=p))
                    else:
                        # `p` is neither an agent nor an asset, and the current
                        # operation is one of {reach, grasp, open, close,
                        # handover, interact, push} — none of which fall back
                        # to a synthetic Position. `p` could legitimately be a
                        # location name (e.g. "kitchen work area"), just not a
                        # valid target for this op. Be explicit about that
                        # distinction so the VLM can fix the right thing.
                        self.error_desc_code = "NOT_FOUND_ENTITY"
                        self.error_detail = {
                            "step": step_idx + 1,
                            "robot": robot_name,
                            "command": raw_cmd,
                            "reason": (
                                f"Entity '{p}' is not an asset/agent and cannot "
                                f"be the target of '{operation_name}'. Only "
                                f"'move' and 'place' accept location names."
                            ),
                        }
                        return False

                if not self.checker.check_operation(
                    operation_name=operation_name,
                    params=operation_entities,
                    assets=self.env.assets,
                    agents=self.env.agents,
                ):
                    self.error_desc_code = "ACTION_NOT_FEASIBLE"
                    self.error_detail = {
                        "step": step_idx + 1,
                        "robot": robot_name,
                        "command": raw_cmd,
                        "reason": self._diagnose_infeasibility(
                            operation_name, operation_entities
                        ),
                    }
                    return False

                step_commands.append([operation_name] + operation_entities)

            if not self.checker.check_compatible_constraints(
                step_commands=step_commands,
                assets=self.env.assets,
                agents=self.env.agents,
            ):
                self.error_desc_code = "ACTION_NOT_COMPATIBLE"
                self.error_detail = {
                    "step": step_idx + 1,
                    "commands": [c[1] for c in commands],
                    "reason": "Actions in this step conflict with each other.",
                }
                return False

            self.env.sim_step(step_commands)

            # temporal constraint check (same as original)
            for idx, tc in enumerate(
                self.env.metadata.get("temporal_constraints", []) or []
            ):
                if satisfied_temporal_constraints[idx]:
                    continue
                satisfied = True
                for ts in tc:
                    if self.check_constraint(ts):
                        if not satisfied:
                            self.error_desc_code = "FAILED_TEMPORAL_CONSTRAINT"
                            self.error_detail = {
                                "step": step_idx + 1,
                                "reason": f"Temporal order violated at step {step_idx + 1}.",
                            }
                            return False
                    else:
                        satisfied = False
                if satisfied:
                    satisfied_temporal_constraints[idx] = True

        if satisfied_temporal_constraints and not all(satisfied_temporal_constraints):
            self.error_desc_code = "FAILED_TEMPORAL_CONSTRAINT"
            self.error_detail = {"reason": "Not all temporal constraints met."}
            return False

        for gc in self.env.metadata.get("goal_constraints", []) or []:
            if not self.check_constraint(gc):
                self.error_desc_code = "FAILED_GOAL_CONSTRAINT"
                self.error_detail = {"reason": f"Goal not achieved: {gc}"}
                return False
        return True

    def _diagnose_infeasibility(self, op_name, entities):
        """Map a Checker rejection to a precise natural-language reason."""
        agent = entities[0]
        if op_name not in agent.avail_actions:
            return f"{agent.name}({agent.type}) cannot perform '{op_name}'."
        if op_name == "grasp":
            target = entities[1]
            if target.is_grasped_by:
                return (
                    f"'{target.name}' is already held by "
                    f"{[a.name for a in target.is_grasped_by]}."
                )
            if not self.checker.check_agent_has_free_end_effector(agent):
                return f"{agent.name} has no free end effector."
            if not agent.is_reached_objects(target):
                return f"{agent.name} has not reached '{target.name}' yet."
        if op_name == "place" and not agent.get_carried_objects():
            return f"{agent.name} is not carrying anything to place."
        if op_name == "open":
            target = entities[1]
            if not hasattr(target, "container_position"):
                return f"'{target.name}' is not a container."
            if not target.container_position.isolated:
                return f"'{target.name}' is already open."
        return f"Preconditions for '{op_name}' not met."

    def get_feedback_for_debate(self) -> str:
        """Format the structured error into the block the VLMs see verbatim."""
        if not self.error_desc_code:
            return "Plan executed successfully."
        d = self.error_detail
        lines = [f"Error type: {self.error_desc_code}"]
        if "step" in d:
            lines.append(f"Failed at: Step {d['step']}")
        if "robot" in d:
            lines.append(f"Robot: {d['robot']}")
        if "command" in d:
            lines.append(f"Command: {d['command']}")
        lines.append(f"Reason: {d['reason']}")
        return "\n".join(lines)


class SimulatorInterface:
    """
    VIKI-Bench symbolic verifier: checker C + world simulator S.

    Plans produced by the multi-VLM debate are translated into VIKI's
    `<Op,arg1,...>` command syntax, then verified symbolically. The Checker
    rejects any step whose pre-conditions are not met by the current abstract
    world; the World Simulator applies state transitions; goal and temporal
    constraints determine final task success.

    `scene_config` schema (VIKI-Bench format):
        {
            "robots":               {"R1": "fetch", "R2": "stompy", ...},
            "init_pos":             {"<asset>_<i>": [<pos_name>, ...] | None, ...},
            "goal_constraints":     [...],   # VIKI goal-constraint list
            "temporal_constraints": [...],   # VIKI temporal-constraint list
        }
    """

    def __init__(self, scene_seed: Optional[int] = None):
        self.scene_seed = scene_seed

    # ── Plan → VIKI command-record translation ──

    @staticmethod
    def _action_to_command(action) -> Optional[str]:
        """
        VIKI canonical action list → VIKI command syntax:
            ['Move', 'pear']         → '<Move,pear>'
            ['Push', 'box', 'R1']    → '<Push,box,R1>'   (3-arg primitive)
            ['Wait']  / [] / None    → None              (robot idle this step)
        Legacy string form ('Move pear') is tolerated for robustness.
        """
        if action is None:
            return None
        if isinstance(action, str):
            action = _normalize_action(action)
        if not isinstance(action, (list, tuple)) or len(action) == 0:
            return None
        op = str(action[0]).strip()
        if not op or op.lower() == "wait":
            return None
        params = [str(p).strip() for p in action[1:] if str(p).strip()]
        if not params:
            return f"<{op}>"
        return "<" + ",".join([op, *params]) + ">"

    def _plan_to_command_records(self, plan: TaskPlan) -> list[dict]:
        """
        TaskPlan with VIKI nested schema → list of per-step command dicts:
            {"R1": "<Move,pear>", "R2": "<Move,apple>"}
        Steps in which every robot is idle ("Wait") are dropped.
        """
        records = []
        for step in plan.steps:
            record = {}
            for robot_id, action in (step.get("actions") or {}).items():
                cmd = self._action_to_command(action)
                if cmd is not None:
                    record[robot_id] = cmd
            if record:
                records.append(record)
        return records

    # ── Scene-config → VIKI env metadata ──

    @staticmethod
    def _deep_strip_none(obj):
        """
        Recursively drop None values from ANY nested dict/list. Used to clean
        `goal_constraints` and `temporal_constraints` whose status dicts often
        carry `is_activated: None` (meaning "don't care") — VIKI's stock
        `filter_none_values` does NOT recurse into nested lists, so those
        `None`s survive and break the strict `==` check in `Eval.check_constraint`,
        causing spurious FAILED_GOAL_CONSTRAINT errors on otherwise-valid plans.
        """
        if isinstance(obj, dict):
            return {k: SimulatorInterface._deep_strip_none(v)
                    for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [SimulatorInterface._deep_strip_none(x)
                    for x in obj if x is not None]
        return obj

    @staticmethod
    def _build_env_metadata(scene_config: dict) -> dict:
        """Mirrors `eval_viki_2.eval_single`'s metadata construction."""
        sc = _filter_none_values(scene_config)
        metadata = {"agents": {}, "assets": {}}

        for rid, rtype in sc["robots"].items():
            metadata["agents"][rid] = {"type": rtype, "pos": {"name": rid}}

        for asset_name, positions in sc.get("init_pos", {}).items():
            asset_type = asset_name.rsplit("_", 1)[0]
            chosen_pos = random.choice(positions) if positions else asset_type
            metadata["assets"][asset_type] = {"pos": {"name": chosen_pos}}
            if asset_type in _CONTAINER_ASSETS:
                metadata["assets"][asset_type]["params"] = {
                    "is_container": True,
                    "position_kwargs": {
                        "name": asset_type,
                        "isolated": asset_type in {"cabinet"},
                    },
                }

        # Deep-strip None values from constraint trees (works around VIKI's
        # filter_none_values bug that doesn't recurse into nested lists).
        metadata["goal_constraints"] = SimulatorInterface._deep_strip_none(
            sc.get("goal_constraints", [])
        )
        metadata["temporal_constraints"] = SimulatorInterface._deep_strip_none(
            sc.get("temporal_constraints", [])
        )
        return metadata

    @staticmethod
    def _render_world_state(judger: DebateEval, header: str) -> str:
        """Pretty-print the abstract world held by `judger.env`.

        When two robots share the same VIKI type (e.g. two `panda` arms),
        VIKI disambiguates them by a fixed BASE POSITION stored as an asset
        keyed by the robot id — e.g. `init_pos["R1"] == ["pandaB"]` becomes
        `assets["R1"].pos.name == "pandaB"`. Without surfacing this, the two
        agent lines render identically as `R1(panda)` / `R3(panda)` and the
        VLM cannot tell which arm is where (critical for reachability).

        So we fold the base position into the agent's type label
        (`R1(panda@pandaB)`) and suppress the now-redundant base-marker
        asset line. This is a DISPLAY-only change — the underlying metadata
        (and therefore the symbolic checker) is untouched."""
        lines = [header]
        agent_ids = set(judger.env.agents.keys())
        for name, agent in judger.env.agents.items():
            reached = [o.name for o in agent.get_reached_objects()]
            carried = [o.name for o in agent.get_carried_objects()]
            # Base-position marker for same-type disambiguation, if any.
            base_asset = judger.env.assets.get(name)
            base = f"@{base_asset.pos.name}" if base_asset is not None else ""
            lines.append(
                f"  - {name}({agent.type}{base}) pos={agent.pos.name} "
                f"reached={reached} carried={carried}"
            )
        for name, asset in judger.env.assets.items():
            # Skip base-position markers already folded into the agent line.
            if name in agent_ids:
                continue
            extras = []
            if asset.is_grasped_by:
                extras.append(f"grasped_by={[a.name for a in asset.is_grasped_by]}")
            if asset.is_activated:
                extras.append("activated")
            if getattr(asset, "is_container", False):
                extras.append(f"isolated={asset.container_position.isolated}")
            tail = (" " + " ".join(extras)) if extras else ""
            lines.append(f"  - {name} pos={asset.pos.name}{tail}")
        return "\n".join(lines)

    @staticmethod
    def _observation_at_failure(judger: DebateEval) -> str:
        """Snapshot the abstract world state, for Phase-4 reflection feedback."""
        return SimulatorInterface._render_world_state(judger, "World state at failure:")

    def get_initial_world_state(self, scene_config: dict) -> str:
        """
        Build the symbolic env from `scene_config` *without* running any plan,
        and return its initial-state snapshot. Used by Phase-1 to seed both
        VLMs with ground-truth positions / container status before they propose.
        """
        if self.scene_seed is not None:
            random.seed(self.scene_seed)
        metadata = self._build_env_metadata(scene_config)
        judger = DebateEval()
        judger.set_env(metadata)
        return self._render_world_state(judger, "Initial world state:")

    # ── Main entry — called by MultiAgentDebateEngine.phase3_execute ──

    def execute_plan(self, plan: TaskPlan, scene_config: dict) -> dict:
        if self.scene_seed is not None:
            random.seed(self.scene_seed)

        metadata = self._build_env_metadata(scene_config)
        command_records = self._plan_to_command_records(plan)
        total_steps = len(command_records)

        judger = DebateEval()
        judger.set_env(metadata)
        success = judger.eval(command_records)

        # Structured failure: pulled from DebateEval.error_detail and rendered
        # into the multi-line block VLMs read during Phase-4 reflection.
        failure_step:   Optional[int] = None
        failure_reason: Optional[str] = None
        if not success:
            failure_step   = judger.error_detail.get("step")
            failure_reason = judger.get_feedback_for_debate()

        # `completed_steps` is a rough heuristic for trajectory_score: actions
        # in the failing step didn't run for action-level errors; the step DID
        # run for post-step errors (temporal / goal). When the step number is
        # unknown (goal-end failure) we count everything that executed.
        if success:
            completed = total_steps
        elif failure_step is None:
            completed = total_steps
        elif judger.error_desc_code in {
            "INVALID_COMMAND", "NOT_FOUND_ENTITY",
            "ACTION_NOT_FEASIBLE", "ACTION_NOT_COMPATIBLE",
        }:
            completed = failure_step - 1
        else:
            completed = failure_step

        # Metrics (symbolic counterparts of the VIKI-R scores).
        trajectory_score    = completed / total_steps if total_steps else 0.0
        task_planning_score = 1.0 if success else 0.0
        goal_constraints = metadata.get("goal_constraints", []) or []
        if goal_constraints:
            satisfied = sum(1 for gc in goal_constraints if judger.check_constraint(gc))
            activation_score = satisfied / len(goal_constraints)
        else:
            activation_score = float(success)

        return {
            "success":         success,
            "completed_steps": completed,
            "total_steps":     total_steps,
            "failure_step":    failure_step,
            "failure_reason":  failure_reason,
            "error_desc_code": judger.error_desc_code,
            "error_detail":    dict(judger.error_detail),
            "observation_at_failure": (
                None if success else self._observation_at_failure(judger)
            ),
            "metrics": {
                "agent_activation_score": activation_score,
                "task_planning_score":    task_planning_score,
                "trajectory_score":       trajectory_score,
            },
        }


# ─────────────────────────────────────────────
# 6. Core Debate Engine
# ─────────────────────────────────────────────

class MultiAgentDebateEngine:
    """
    Three-agent variant of the debate pipeline:
    Phase 1 (3 proposals + merge) → Phase 2 (3-way rotating debate) →
    Phase 3 → (Phase 4 with 3-way reflection if needed) → loop
    """

    def __init__(
        self,
        vlm1: VLMInterface,
        vlm2: VLMInterface,
        vlm3: VLMInterface,
        simulator: SimulatorInterface,
        robot1: RobotProfile,
        robot2: RobotProfile,
        robot3: RobotProfile,
        max_debate_rounds: int = 5,
        max_retry_rounds: int = 3,
    ):
        self.vlm1 = vlm1
        self.vlm2 = vlm2
        self.vlm3 = vlm3
        self.simulator = simulator
        self.robot1 = robot1
        self.robot2 = robot2
        self.robot3 = robot3
        self.max_debate_rounds = max_debate_rounds
        self.max_retry_rounds = max_retry_rounds

    # ── Helper: format system prompts ──

    @staticmethod
    def _render_action_descriptions(*robot_types: str) -> str:
        """Build the bulleted ACTION_DESCRIPTION block restricted to the
        primitives any of the listed robot types can perform."""
        relevant = set()
        for rt in robot_types:
            relevant.update(AGENT_AVAIL_ACTIONS.get(rt, []))
        # Preserve VIKI's documented order.
        lines = [f"- {ACTION_DESCRIPTION[a]}" for a in ACTION_DESCRIPTION if a in relevant]
        return "\n".join(lines)

    def _init_vlm_system_prompts(self):
        """Render and install the SYSTEM_PROMPT into each of the three VLMs,
        with the advocate role rotated (each VLM is `self`, the other two
        are `partner_a` and `partner_b`)."""
        action_block = self._render_action_descriptions(
            self.robot1.name, self.robot2.name, self.robot3.name)

        def render(self_robot: RobotProfile,
                   partner_a: RobotProfile,
                   partner_b: RobotProfile) -> str:
            return SYSTEM_PROMPT.format(
                robot_id=self_robot.robot_id,
                robot_name=self_robot.name,
                robot_description=self_robot.description,
                robot_avail_actions=self_robot.available_actions,
                partner_a_id=partner_a.robot_id,
                partner_a_name=partner_a.name,
                partner_a_description=partner_a.description,
                partner_a_avail_actions=partner_a.available_actions,
                partner_b_id=partner_b.robot_id,
                partner_b_name=partner_b.name,
                partner_b_description=partner_b.description,
                partner_b_avail_actions=partner_b.available_actions,
                action_descriptions=action_block,
            )

        self.vlm1.set_system_prompt(render(self.robot1, self.robot2, self.robot3))
        self.vlm2.set_system_prompt(render(self.robot2, self.robot1, self.robot3))
        self.vlm3.set_system_prompt(render(self.robot3, self.robot1, self.robot2))

    @staticmethod
    def _compact_history_msg(content: str,
                             max_reasoning_chars: int = 400,
                             fallback_chars: int = 2500) -> str:
        """Reformat a debate-message body so the plan `steps` are kept
        IN FULL (those are what the next round needs to decide
        ACCEPT/REVISE) and only the verbose `reasoning` prose is
        truncated. Falls back to plain char-truncation if the content
        doesn't parse as JSON (e.g. the implicit-ACCEPT placeholder, or
        a model that forgot the JSON wrapper).

        Handles both response shapes the debate engine produces:
          • Plain plan : {reasoning, steps}
          • Critique   : {verdict, issues?, revised_plan: {reasoning, steps}}
                         or {verdict: "ACCEPT", reasoning}"""
        # Reuse the robust extractor (strict=False, last-answer object) so
        # this stays consistent with is_accept_response / parse_plan_from_
        # response and survives literal-newline JSON.
        data = _last_answer_dict(content)
        if data is None:
            return (content[:fallback_chars] + "..."
                    if len(content) > fallback_chars else content)

        def trunc(text):
            text = str(text)
            return (text[:max_reasoning_chars] + "..."
                    if len(text) > max_reasoning_chars else text)

        compact: dict = {}
        if "verdict" in data:
            compact["verdict"] = data["verdict"]
            if data.get("verdict") == "ACCEPT":
                if "reasoning" in data:
                    compact["reasoning"] = trunc(data["reasoning"])
            else:                                          # REVISE
                if "issues" in data:
                    compact["issues"] = data["issues"]
                rp = data.get("revised_plan")
                if isinstance(rp, dict):
                    compact["revised_plan"] = {
                        "reasoning": trunc(rp.get("reasoning", "")),
                        "steps":     rp.get("steps", []),  # FULL — never trimmed
                    }
        elif "steps" in data:
            compact["reasoning"] = trunc(data.get("reasoning", ""))
            compact["steps"]     = data["steps"]           # FULL — never trimmed
        else:
            return (content[:fallback_chars] + "..."
                    if len(content) > fallback_chars else content)

        return json.dumps(compact, indent=2)

    def _format_debate_history(self, state: DebateState) -> str:
        """Format debate history for inclusion in prompts.

        Since VLMInterface.query() is now stateless, this summary is the
        ONLY way the model sees prior turns. Per-message compaction
        keeps the actionable parts (verdict + steps) in full and only
        trims the prose `reasoning` field — see _compact_history_msg."""
        if not state.messages:
            return "(No previous debate messages)"

        # Bumped from 8 → 10 in the 3-agent variant: a single Phase-2 round
        # has 3 turns instead of 2, so a 3-round window now needs 9 slots
        # plus a couple for Phase-1 proposals.
        MAX_MESSAGES = 10

        _role_to_label = {
            DebateRole.VLM1_R1_ADVOCATE: f"VLM1({self.robot1.robot_id})",
            DebateRole.VLM2_R2_ADVOCATE: f"VLM2({self.robot2.robot_id})",
            DebateRole.VLM3_R3_ADVOCATE: f"VLM3({self.robot3.robot_id})",
        }
        lines = []
        for msg in state.messages[-MAX_MESSAGES:]:
            role_label = _role_to_label.get(msg.role, str(msg.role))
            content = self._compact_history_msg(msg.content)
            lines.append(f"[{role_label}]: {content}")
        return "\n\n".join(lines)

    # ── Phase 1: Independent Proposals ──

    def phase1_independent_proposals(
        self, state: DebateState
    ) -> tuple[TaskPlan, TaskPlan, TaskPlan]:
        """Each of the three VLMs independently proposes a plan from its
        robot's perspective (with the other two robots framed as partners)."""
        print("\n" + "=" * 60)
        print("PHASE 1: Independent Proposals (3 agents)")
        print("=" * 60)

        world_state = state.initial_world_state or "(unavailable)"

        def propose(self_robot: RobotProfile,
                    partner_a: RobotProfile,
                    partner_b: RobotProfile,
                    vlm: VLMInterface, label: str,
                    role: DebateRole) -> Optional[TaskPlan]:
            prompt = PROPOSAL_PROMPT.format(
                task_description=state.task_description,
                world_state=world_state,
                robot_id=self_robot.robot_id,
                robot_name=self_robot.name,
                partner_a_id=partner_a.robot_id,
                partner_a_name=partner_a.name,
                partner_b_id=partner_b.robot_id,
                partner_b_name=partner_b.name,
            )
            response = vlm.query(prompt, image_path=state.scene_image_path)
            plan = parse_plan_from_response(response)
            print(f"[{label}] Proposed plan with "
                  f"{len(plan.steps) if plan else 0} steps")
            state.messages.append(DebateMessage(
                role=role, content=response, proposed_plan=plan,
            ))
            return plan

        plan1 = propose(self.robot1, self.robot2, self.robot3,
                        self.vlm1, "VLM1", DebateRole.VLM1_R1_ADVOCATE)
        plan2 = propose(self.robot2, self.robot1, self.robot3,
                        self.vlm2, "VLM2", DebateRole.VLM2_R2_ADVOCATE)
        plan3 = propose(self.robot3, self.robot1, self.robot2,
                        self.vlm3, "VLM3", DebateRole.VLM3_R3_ADVOCATE)
        return plan1, plan2, plan3

    def _merge_proposals(
        self,
        plan1: TaskPlan,
        plan2: TaskPlan,
        plan3: TaskPlan,
        state: Optional[DebateState] = None,
    ) -> TaskPlan:
        """
        Ask VLM1 (who will also speak first in Phase 2) to integrate the
        THREE independent proposals into a single coherent joint plan via
        an LLM call. Falls back to `_mechanical_merge` if VLM1's response
        can't be parsed.

        VLM1 just authored `current_plan` — its first Phase-2 critique
        would almost certainly be ACCEPT, so we set
        `state.skip_vlm1_first_critique` to seed the consecutive-ACCEPT
        counter at 1.
        """
        plan_self_json = json.dumps({"steps": plan1.steps}, indent=2)
        plan_a_json    = json.dumps({"steps": plan2.steps}, indent=2)
        plan_b_json    = json.dumps({"steps": plan3.steps}, indent=2)

        prompt = MERGE_PROMPT.format(
            plan_self_json=plan_self_json,
            plan_a_json=plan_a_json,
            plan_b_json=plan_b_json,
            robot_id=self.robot1.robot_id,
            robot_name=self.robot1.name,
            partner_a_id=self.robot2.robot_id,
            partner_a_name=self.robot2.name,
            partner_b_id=self.robot3.robot_id,
            partner_b_name=self.robot3.name,
        )

        # Stateless query() doesn't carry the image from prior turns,
        # so we re-attach the scene image here.
        response = self.vlm1.query(
            prompt,
            image_path=state.scene_image_path if state else None,
        )
        merged = parse_plan_from_response(response)

        if merged is not None:
            print(f"[VLM1] integrated 3 proposals → {len(merged.steps)} steps")
            if state is not None:
                state.messages.append(DebateMessage(
                    role=DebateRole.VLM1_R1_ADVOCATE,
                    content=response,
                    proposed_plan=merged,
                ))
                state.skip_vlm1_first_critique = True
            return merged

        print("[WARN] VLM1 merge produced an unparseable plan; "
              "falling back to mechanical merge.")
        return self._mechanical_merge(plan1, plan2, plan3)

    def _mechanical_merge(self, plan1: TaskPlan, plan2: TaskPlan,
                          plan3: TaskPlan) -> TaskPlan:
        """
        Fallback used only when the LLM-driven merge in `_merge_proposals`
        returns an unparseable response. Take R1's action from plan1, R2's
        from plan2, R3's from plan3, aligned by step index. Missing entries
        become `["Wait"]`.
        """
        r1_id = self.robot1.robot_id
        r2_id = self.robot2.robot_id
        r3_id = self.robot3.robot_id
        max_steps = max(len(plan1.steps), len(plan2.steps), len(plan3.steps))

        merged_steps = []
        for i in range(max_steps):
            step1_actions = plan1.steps[i].get("actions", {}) if i < len(plan1.steps) else {}
            step2_actions = plan2.steps[i].get("actions", {}) if i < len(plan2.steps) else {}
            step3_actions = plan3.steps[i].get("actions", {}) if i < len(plan3.steps) else {}
            merged_steps.append({
                "step": i + 1,
                "actions": {
                    r1_id: step1_actions.get(r1_id, ["Wait"]),
                    r2_id: step2_actions.get(r2_id, ["Wait"]),
                    r3_id: step3_actions.get(r3_id, ["Wait"]),
                },
            })

        return TaskPlan(
            steps=merged_steps,
            reasoning=(
                f"Mechanical fallback merge from VLM1's {len(plan1.steps)}-step plan, "
                f"VLM2's {len(plan2.steps)}-step plan, and "
                f"VLM3's {len(plan3.steps)}-step plan."
            ),
            raw_text=json.dumps({"steps": merged_steps}, indent=2),
        )

    # ── Phase 2: Debate Loop ──

    def phase2_debate(self, state: DebateState) -> bool:
        """
        Three-way rotating debate.
        **Consensus = THREE CONSECUTIVE turns of ACCEPT** (one full sweep
        of all three agents), counted across round boundaries (any REVISE
        from any agent resets the streak to 0).

        Returns True if consensus reached (or max rounds exhausted, in which
        case the latest plan is taken as best-effort consensus).

        If `state.skip_vlm1_first_critique` is set (by `_merge_proposals`),
        round 1 starts directly with VLM2 — VLM1's first critique is treated
        as an implicit ACCEPT (it just authored `current_plan`). The implicit
        ACCEPT seeds the consecutive-ACCEPT counter to 1, so consensus is
        reachable on round 1 if VLM2 and VLM3 both ACCEPT.
        From round 2 onward, normal VLM1→VLM2→VLM3 rotation resumes.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Debate Loop (3-way rotation)")
        print("=" * 60)

        # `last_critique` carries the IMMEDIATELY-previous advocate's full
        # response into the next turn's `{other_critique}` slot — persists
        # across rounds. The full debate history is also visible via
        # `_format_debate_history` so a VLM can see what the OTHER partners
        # said too, not just the most recent.
        last_critique: Optional[str] = None

        # Cross-round counter — consensus iff this hits 3.
        consecutive_accepts = 0
        CONSENSUS_REQUIRED = 3

        # Consume the skip flag set by _merge_proposals.
        skip_vlm1_round1 = state.skip_vlm1_first_critique
        state.skip_vlm1_first_critique = False

        if skip_vlm1_round1:
            # Seed last_critique with VLM1's most-recent message (the merge
            # response) so VLM2's first turn sees it in `{other_critique}`.
            for msg in reversed(state.messages):
                if msg.role == DebateRole.VLM1_R1_ADVOCATE:
                    last_critique = msg.content
                    break

        # Rotating turn order: (VLM1 → VLM2 → VLM3). Each tuple is
        # (vlm, self_robot, partner_a, partner_b, role, label).
        turn_order = [
            (self.vlm1, self.robot1, self.robot2, self.robot3,
             DebateRole.VLM1_R1_ADVOCATE, "VLM1"),
            (self.vlm2, self.robot2, self.robot1, self.robot3,
             DebateRole.VLM2_R2_ADVOCATE, "VLM2"),
            (self.vlm3, self.robot3, self.robot1, self.robot2,
             DebateRole.VLM3_R3_ADVOCATE, "VLM3"),
        ]

        for round_idx in range(self.max_debate_rounds):
            state.round_num = round_idx + 1
            print(f"\n--- Debate Round {state.round_num} ---")

            # Round 1 only: if VLM1 just authored current_plan via merge,
            # skip its turn and seed the streak at 1.
            if round_idx == 0 and skip_vlm1_round1:
                this_round_turns = turn_order[1:]      # VLM2 + VLM3
                consecutive_accepts = 1                # VLM1's implicit accept
                print("[VLM1] (implicit ACCEPT — just authored the merged plan)")
                state.messages.append(DebateMessage(
                    role=DebateRole.VLM1_R1_ADVOCATE,
                    content="(implicit ACCEPT — VLM1 just authored the integrated plan)",
                    proposed_plan=None,
                    accepts_current_plan=True,
                ))
            else:
                this_round_turns = turn_order

            for (current_vlm, current_robot, partner_a, partner_b,
                 role, label) in this_round_turns:
                # Re-serialize the *latest* plan — if the previous turn
                # REVISEd, this advocate sees the updated draft.
                current_plan_json = json.dumps(
                    {"steps": state.current_plan.steps}, indent=2
                )
                debate_history = self._format_debate_history(state)
                other_critique = last_critique if last_critique is not None else NO_PRIOR_CRITIQUE

                prompt = CRITIQUE_PROMPT.format(
                    current_plan_json=current_plan_json,
                    other_critique=other_critique,
                    debate_history=debate_history,
                    robot_id=current_robot.robot_id,
                    robot_name=current_robot.name,
                    partner_a_id=partner_a.robot_id,
                    partner_b_id=partner_b.robot_id,
                )
                response = current_vlm.query(prompt, image_path=state.scene_image_path)
                accepts = is_accept_response(response)
                new_plan = parse_plan_from_response(response)

                state.messages.append(DebateMessage(
                    role=role,
                    content=response,
                    proposed_plan=new_plan,
                    accepts_current_plan=accepts,
                ))
                print(f"[{label}] {'ACCEPTS' if accepts else 'REVISES'}  "
                      f"(streak {consecutive_accepts + 1 if accepts else 0}/"
                      f"{CONSENSUS_REQUIRED})")

                # On REVISE, swap in the new plan immediately so the *next*
                # turn sees the revised version in `{current_plan_json}`.
                if new_plan and not accepts:
                    state.current_plan = new_plan

                last_critique = response

                # ── Consensus: THREE CONSECUTIVE ACCEPTs (any REVISE resets) ──
                if accepts:
                    consecutive_accepts += 1
                    if consecutive_accepts >= CONSENSUS_REQUIRED:
                        state.consensus_reached = True
                        print(f"\n>>> CONSENSUS REACHED "
                              f"({CONSENSUS_REQUIRED} consecutive ACCEPTs) <<<")
                        return True
                else:
                    consecutive_accepts = 0

        print(f"\n>>> Max debate rounds ({self.max_debate_rounds}) reached without consensus.")
        print(">>> Using latest plan as best-effort consensus.")
        state.consensus_reached = True  # forced consensus
        return True

    # ── Phase 3: Execution ──

    def phase3_execute(self, state: DebateState, scene_config: dict) -> dict:
        """Execute the consensus plan in the simulator."""
        print("\n" + "=" * 60)
        print("PHASE 3: Execution in Simulator")
        print("=" * 60)
        print(f"Executing plan with {len(state.current_plan.steps)} steps...")

        result = self.simulator.execute_plan(state.current_plan, scene_config)

        if result["success"]:
            print(">>> TASK COMPLETED SUCCESSFULLY <<<")
        else:
            print(f">>> EXECUTION FAILED at step {result.get('failure_step', '?')} <<<")
            print(f"    Reason: {result.get('failure_reason', 'unknown')}")

        return result

    # ── Phase 4: Reflection & Re-Debate ──

    def phase4_reflection(self, state: DebateState, exec_result: dict):
        """All THREE VLMs reflect on execution failure and propose fixes;
        the three revisions are then merged via VLM1 (same path as the
        Phase-1 merge)."""
        print("\n" + "=" * 60)
        print("PHASE 4: Reflection on Failure (3-way)")
        print("=" * 60)

        # Build execution feedback string
        feedback = (
            f"Plan failed at step {exec_result.get('failure_step', '?')}.\n"
            f"Completed {exec_result.get('completed_steps', 0)}/{exec_result.get('total_steps', '?')} steps.\n"
            f"Failure reason: {exec_result.get('failure_reason', 'unknown')}\n"
        )
        if exec_result.get("observation_at_failure"):
            feedback += f"Observation: {exec_result['observation_at_failure']}\n"

        state.execution_feedback = feedback

        # Build debate summary (condensed)
        debate_summary = f"Previous debate had {state.round_num} rounds. "
        accept_count = sum(1 for m in state.messages if m.accepts_current_plan)
        debate_summary += f"{accept_count} acceptance messages were recorded."

        failed_plan_json = json.dumps({"steps": state.current_plan.steps}, indent=2)

        def reflect(vlm: VLMInterface, robot: RobotProfile,
                    role: DebateRole, label: str) -> Optional[TaskPlan]:
            prompt = REFLECTION_PROMPT.format(
                execution_feedback=feedback,
                failed_plan_json=failed_plan_json,
                debate_summary=debate_summary,
                robot_id=robot.robot_id,
            )
            response = vlm.query(prompt, image_path=state.scene_image_path)
            new_plan = parse_plan_from_response(response)
            state.messages.append(DebateMessage(
                role=role, content=response, proposed_plan=new_plan,
            ))
            print(f"[{label}] Reflection complete, proposed revision: "
                  f"{new_plan is not None}")
            return new_plan

        new_plan1 = reflect(self.vlm1, self.robot1,
                            DebateRole.VLM1_R1_ADVOCATE, "VLM1")
        new_plan2 = reflect(self.vlm2, self.robot2,
                            DebateRole.VLM2_R2_ADVOCATE, "VLM2")
        new_plan3 = reflect(self.vlm3, self.robot3,
                            DebateRole.VLM3_R3_ADVOCATE, "VLM3")

        # Merge whatever subset parsed. We need >=1 plan; if all 3 parsed,
        # do a full 3-way merge; if only 2 parsed, mechanical-merge the
        # missing one's slot from the still-failed plan; if only 1 parsed,
        # use it directly.
        survivors = [p for p in (new_plan1, new_plan2, new_plan3) if p is not None]
        if len(survivors) == 3:
            state.current_plan = self._merge_proposals(
                new_plan1, new_plan2, new_plan3, state)
        elif len(survivors) == 2:
            # Stand in for the missing one with the prior failed plan so
            # the merge still sees 3 inputs (this preserves that robot's
            # actions if no one revised them).
            fill = state.current_plan
            p1 = new_plan1 or fill
            p2 = new_plan2 or fill
            p3 = new_plan3 or fill
            state.current_plan = self._merge_proposals(p1, p2, p3, state)
        elif len(survivors) == 1:
            state.current_plan = survivors[0]
        # else: keep the old plan (shouldn't happen)

    # ── Main Pipeline ──

    def run(self, task_description: str, scene_image_path: str, scene_config: dict) -> dict:
        """
        Run the full pipeline:
        Phase 1 → Phase 2 → Phase 3 → (Phase 4 → Phase 2 → Phase 3)* → Done
        
        Returns:
            {
                "success": bool,
                "final_plan": TaskPlan,
                "total_debate_rounds": int,
                "total_retry_rounds": int,
                "execution_results": list[dict],
                "full_debate_log": list[DebateMessage],
            }
        """
        # Initialize
        self._init_vlm_system_prompts()
        # Pre-render the symbolic simulator's initial world state once, so both
        # Phase-1 proposals are anchored to the same ground-truth state.
        try:
            initial_world_state = self.simulator.get_initial_world_state(scene_config)
        except Exception as e:                          # never let it break the run
            print(f"[WARN] could not render initial world state: {e}")
            initial_world_state = ""

        state = DebateState(
            task_description=task_description,
            scene_image_path=scene_image_path,
            robot_profiles=[self.robot1, self.robot2, self.robot3],
            initial_world_state=initial_world_state,
        )
        execution_results = []
        # Per-loop bookkeeping for evaluator scripts. One entry per Phase 2
        # invocation: the number of debate rounds it took to (forcibly) reach
        # consensus before being handed to Phase 3.
        debate_rounds_per_loop: list[int] = []

        # Phase 1: Independent proposals (3 of them) + merge
        plan1, plan2, plan3 = self.phase1_independent_proposals(state)
        survivors = [p for p in (plan1, plan2, plan3) if p is not None]
        if len(survivors) == 0:
            print("[ERROR] All three VLMs failed to produce a valid plan.")
            return {
                "success": False, "error": "All VLMs failed to produce plans",
                "debate_loop_count": 0, "debate_rounds_per_loop": [],
            }
        elif len(survivors) < 3:
            print(f"[WARN] Only {len(survivors)}/3 VLMs produced parseable plans; "
                  f"using survivors as starting point (no full 3-way merge).")
            state.current_plan = survivors[0]
        else:
            state.current_plan = self._merge_proposals(plan1, plan2, plan3, state)

        print(f"\nMerged plan has {len(state.current_plan.steps)} steps")

        # Retry loop
        for retry in range(self.max_retry_rounds + 1):
            # Phase 2: Debate
            self.phase2_debate(state)
            debate_rounds_per_loop.append(state.round_num)

            # Phase 3: Execute
            exec_result = self.phase3_execute(state, scene_config)
            execution_results.append(exec_result)

            if exec_result["success"]:
                return {
                    "success": True,
                    "final_plan": state.current_plan,
                    "total_debate_rounds": state.round_num,
                    "total_retry_rounds": retry,
                    "debate_loop_count": len(debate_rounds_per_loop),
                    "debate_rounds_per_loop": debate_rounds_per_loop,
                    "execution_results": execution_results,
                    "full_debate_log": state.messages,
                }

            # Phase 4: Reflection (if not last retry)
            if retry < self.max_retry_rounds:
                print(f"\n{'='*60}")
                print(f"RETRY {retry + 1}/{self.max_retry_rounds}")
                print(f"{'='*60}")
                self.phase4_reflection(state, exec_result)
                state.consensus_reached = False
                state.round_num = 0

        # All retries exhausted
        return {
            "success": False,
            "final_plan": state.current_plan,
            "total_debate_rounds": state.round_num,
            "total_retry_rounds": self.max_retry_rounds,
            "debate_loop_count": len(debate_rounds_per_loop),
            "debate_rounds_per_loop": debate_rounds_per_loop,
            "execution_results": execution_results,
            "full_debate_log": state.messages,
        }


# ─────────────────────────────────────────────
# 7. Example Usage
# ─────────────────────────────────────────────

def main():
    """Run the debate pipeline on one task drawn from VIKI-L2/test.parquet."""
    import argparse

    parser = argparse.ArgumentParser(description="VIKI-L2 multi-agent-debate demo")
    parser.add_argument(
        "--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
        help="path to the VIKI-L2 parquet (default: VIKI_data/viki/VIKI-L2/test.parquet)",
    )
    parser.add_argument(
        "--idx", type=int, default=None,
        help="row index to run; default = first 3-robot (R1+R2+R3) task in the parquet",
    )
    parser.add_argument(
        "--max-debate-rounds", type=int, default=3,
        help="max rounds per debate session (default 3 to keep API cost down)",
    )
    parser.add_argument(
        "--max-retry-rounds", type=int, default=2,
        help="max execution retries on simulator failure (default 2)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="folder to dump per-call VLM logs to; defaults to logs/<timestamp>/. "
             "Use --no-log to disable.",
    )
    parser.add_argument("--no-log", action="store_true",
                        help="disable per-call VLM logging entirely")
    args = parser.parse_args()

    # ── Pre-flight: API key ──
    if not os.environ.get("APIMART_API_KEY"):
        print("[ERROR] APIMART_API_KEY is not set.")
        print("        On PowerShell:  $env:APIMART_API_KEY = 'sk-...'")
        return

    # ── Load the task from the parquet ──
    from viki_loader import load_viki_task, find_task_indices

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).resolve().parent / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] Parquet not found: {parquet_path}")
        return

    if args.idx is None:
        candidates = find_task_indices(parquet_path, n_robots=3,
                                       required_ids=("R1", "R2", "R3"), limit=1)
        if not candidates:
            print("[ERROR] No 3-robot (R1+R2+R3) tasks found in the parquet.")
            return
        idx = candidates[0]
    else:
        idx = args.idx

    task = load_viki_task(parquet_path, idx)
    robots = task["scene_config"]["robots"]
    print(f"=== VIKI-L2 task #{idx} (task_id={task['task_id']}) ===")
    print(f"name        : {task['task_name']}")
    print(f"robots      : {robots}")
    print(f"description : {task['task_description']}")
    print(f"image       : {task['image_path']}")
    print()

    # ── Build RobotProfiles dynamically from the parquet's `robots` dict ──
    # (R1/R2/R3 may be any VIKI embodiment — not hardcoded.)
    if not all(robots.get(rid) for rid in ("R1", "R2", "R3")):
        print(f"[ERROR] Task #{idx} does not have all of R1/R2/R3; got {robots}")
        return
    robot1 = RobotProfile(name=robots["R1"], robot_id="R1")
    robot2 = RobotProfile(name=robots["R2"], robot_id="R2")
    robot3 = RobotProfile(name=robots["R3"], robot_id="R3")

    # ── Set up VLM call logger (shared by all three debaters) ──
    logger = None
    if not args.no_log:
        log_dir = Path(args.log_dir) if args.log_dir else (
            Path(__file__).resolve().parent / "logs" / time.strftime("%Y%m%d_%H%M%S")
        )
        logger = RunLogger(log_dir)
        print(f"logging VLM calls → {log_dir}")

    # ── Instantiate VLMs + simulator + engine ──
    vlm1 = VLMInterface(role=DebateRole.VLM1_R1_ADVOCATE, logger=logger)
    vlm2 = VLMInterface(role=DebateRole.VLM2_R2_ADVOCATE, logger=logger)
    vlm3 = VLMInterface(role=DebateRole.VLM3_R3_ADVOCATE, logger=logger)
    simulator = SimulatorInterface(scene_seed=0)

    engine = MultiAgentDebateEngine(
        vlm1=vlm1, vlm2=vlm2, vlm3=vlm3, simulator=simulator,
        robot1=robot1, robot2=robot2, robot3=robot3,
        max_debate_rounds=args.max_debate_rounds,
        max_retry_rounds=args.max_retry_rounds,
    )

    # ── Run ──
    result = engine.run(
        task["task_description"], task["image_path"], task["scene_config"]
    )

    # ── Report ──
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Success:        {result['success']}")
    print(f"Debate rounds:  {result.get('total_debate_rounds')}")
    print(f"Retry rounds:   {result.get('total_retry_rounds')}")
    print(f"Final plan:")
    for step in result["final_plan"].steps:
        actions = step.get("actions", {})
        parts = ", ".join(f"{rid}={act}" for rid, act in actions.items())
        print(f"  Step {step['step']}: {parts}")
    last_exec = result.get("execution_results", [{}])[-1] if result.get("execution_results") else {}
    if last_exec:
        m = last_exec.get("metrics", {})
        print(f"\nMetrics: activation={m.get('agent_activation_score', 0):.2f}  "
              f"planning={m.get('task_planning_score', 0):.2f}  "
              f"trajectory={m.get('trajectory_score', 0):.2f}")
        if not result["success"] and last_exec.get("failure_reason"):
            print(f"Failure: step {last_exec.get('failure_step')} — {last_exec['failure_reason']}")


if __name__ == "__main__":
    main()
