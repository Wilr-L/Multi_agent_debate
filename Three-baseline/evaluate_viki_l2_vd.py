"""
evaluate_viki_l2_vd  (Three-baseline / 3-embodiment variant)
============================================================
Multi-agent **Vanilla Debate** ablation for VIKI-L2 3-robot (R1+R2+R3)
tasks: THREE VLMs debate using IDENTICAL, ROLE-NEUTRAL prompts. Mechanics
are otherwise identical to the 3-agent debate engine:
  Phase 1 triple propose → merge
  → Phase 2 3-way debate ⟲ → Phase 3 execute
  → Phase 4 reflection on failure → retry.

The ablation isolates whether the per-robot advocate role differentiation
in the main engine contributes to performance:
  - SAME 3-agent engine code path (subclass of MultiAgentDebateEngine)
  - SAME flow / call count / max-rounds / retry budget
  - the ONLY change is the prompt content — all three VLMs see an
    identical neutral system prompt, and every per-phase prompt drops the
    "advocate for {robot_id}" framing.

Per-phase prompt constants in `multi_agent_debate.py` are monkey-patched
to neutral versions on entry. The neutral templates reference only the
CONTENT kwargs the 3-agent engine passes (task_description, world_state,
plan_self/a/b_json, current_plan_json, other_critique, debate_history,
execution_feedback, failed_plan_json, debate_summary); the advocate-role
kwargs (robot_id, partner_a_id, …) are still passed but simply ignored by
str.format, so the engine's call sites stay unchanged.

Example:
  python evaluate_viki_l2_vd.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --max-debate-rounds 3 --max-retry-rounds 2 --limit 20 \\
      --trust-remote-code --tensor-parallel-size 2
"""

import argparse

import multi_agent_debate as mad
from multi_agent_debate import (
    MultiAgentDebateEngine, SimulatorInterface, RobotProfile, DebateRole,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import add_common_args, run_evaluation


# ─── Role-neutral prompts ─────────────────────────────────────────────
# Each template below accepts the SAME kwargs as the original it replaces
# (so the engine's `.format(...)` call sites need no changes). Kwargs
# that distinguish advocate-vs-partner (`robot_id`, `partner_id`, …) are
# accepted but not referenced — str.format silently ignores extras.

VANILLA_SYSTEM_PROMPT = """\
You are one of THREE independent AI planners working together on a multi-robot task in \
VIKI-Bench. I will provide you with an image of the robots in a scene, the available robots \
and their action primitives, and a task description. Your job is to produce a joint plan for \
ALL robots; you and the other planners will each propose a plan independently, then debate \
back-and-forth to converge on a single joint plan that all planners ACCEPT.

You must first analyze the image to fully understand the scene depicted. Then, analyze the \
task description. Finally, produce the plan. Your reasoning must strictly adhere to the visual \
content of the image and the task description — no assumptions, hypotheses, or guesses are \
allowed.

## Available robots and their action primitives
{robot_block}

## VIKI-Bench action primitives and rules
Action must follow the following format as a JSON list, for example ["Move", "plate"] or \
["grasp", "banana"]. Each step has format: action_type, target_object_or_location.
Action primitives and descriptions:
{action_descriptions}
Use exact object and location names from the task, relevant assets, and world state. Do not \
invent new entity names.
Choose the primitive that advances the current object state, not just the task name:
  - If the robot is not near the object, use Move on the object.
  - If the robot is at/near the object but has not reached it, use Reach on the object.
  - If the robot has reached the object and is not carrying it, use Grasp on the object.
  - If the robot is carrying an object and it is not at the target, use Move on the target.
  - If the robot is carrying an object at the target area, use Place on the target location.
  - If an appliance or device must be started or activated, use Interact on that appliance.
  - **Each robot can only perform ONE action per time step.** All robots may work in \
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
- `step` starts at 1 and increments sequentially.
- `actions` maps robot id ("R1", "R2", "R3") to `[action_type, target_or_location]`.
- Only use action primitives in that robot's "Available actions" list above.
- Every step MUST include an entry for every robot (use `["Wait"]` for idle).
"""


VANILLA_PROPOSAL_PROMPT = """\
## Task
Look at the scene image carefully. Here is the task description:
{task_description}

## Initial world state (from the symbolic simulator)
{world_state}

## Instructions
Propose a complete joint plan that controls ALL robots in the team to accomplish this task. \
The other planners will independently propose theirs; you will then debate to converge on a \
single joint plan. Every step must include an entry for every robot (`["Wait"]` if idle).

Think step by step:
1. What objects are visible in the scene? Where are they (cross-check against the world state)?
2. What is the goal state?
3. What actions does each robot need to perform? Can any be parallelized?
4. Are there ordering / dependency constraints?

Output the complete joint plan in the required JSON format.
"""


VANILLA_MERGE_PROMPT = """\
## Your task — integrate three independent proposals into one joint plan
Three planners independently proposed plans for the same multi-robot task. Combine the three \
proposals into a single coherent joint plan that will be the starting point for debate. \
Every step MUST include an entry for every robot (`["Wait"]` if idle).

## Proposal A
{plan_self_json}

## Proposal B
{plan_a_json}

## Proposal C
{plan_b_json}

Output ONLY the merged plan in the standard JSON schema (do NOT use an ACCEPT/REVISE verdict \
wrapper):
```json
{{
  "reasoning": "why you merged this way, including any conflict resolutions across the three plans...",
  "steps": [
    {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Move", "apple"], "R3": ["Wait"]}}}}
  ]
}}
```
"""


VANILLA_CRITIQUE_PROMPT = """\
## Current joint plan under review
{current_plan_json}

## The other planner's most recent critique
{other_critique}

## Debate history
{debate_history}

## Your task
You and the other planners are debating to produce a joint plan. Only a plan that ALL \
planners ACCEPT will be implemented. Review the current plan and the most recent critique. \
Evaluate:
1. **Feasibility**: Can each robot physically execute its assigned actions? Are pre-conditions \
met (Move/Reach/Grasp before Place)?
2. **Coordination**: Are there timing conflicts? Will two robots try to access the same \
space or object simultaneously?
3. **Completeness**: Does the plan achieve the task goal? Any steps missing?
4. **Efficiency**: Is there a better task allocation to save steps?

If the plan is acceptable, respond:
```json
{{"verdict": "ACCEPT", "reasoning": "why the plan is good"}}
```
If you want to revise, respond (every step must include all three robots):
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


VANILLA_REFLECTION_PROMPT = """\
## Execution failed
{execution_feedback}

## Failed plan
{failed_plan_json}

## Previous debate history (summary)
{debate_summary}

## Your task
Analyze the failure:
1. Which step failed and why?
2. Was it a planning error (wrong action sequence) or a pre-condition error (missing setup)?
3. What should be changed?

Propose a revised plan that addresses the failure. Be specific about what changed and why in \
"reasoning". Output in the standard JSON plan format.
"""


def _install_vanilla_prompts():
    """Replace MAD's module-level prompt constants. Methods reference these
    as free variables, so the replacement takes effect on the NEXT call —
    perfectly fine since we patch before instantiating the engine. The
    SYSTEM_PROMPT swap is handled by `VanillaDebateEngine` overriding
    `_init_vlm_system_prompts` (the inherited method's SYSTEM_PROMPT use
    bakes in the advocate-vs-partner swap that we explicitly want gone)."""
    mad.PROPOSAL_PROMPT   = VANILLA_PROPOSAL_PROMPT
    mad.MERGE_PROMPT      = VANILLA_MERGE_PROMPT
    mad.CRITIQUE_PROMPT   = VANILLA_CRITIQUE_PROMPT
    mad.REFLECTION_PROMPT = VANILLA_REFLECTION_PROMPT


# ─── Engine override: install one identical system prompt on both VLMs ──

class VanillaDebateEngine(MultiAgentDebateEngine):
    """Same as the 3-agent MultiAgentDebateEngine but installs a SINGLE
    neutral system prompt (the exact same string) on ALL THREE VLMs. The
    parent class instead rotates robot/partner perspective per VLM — that's
    the advocate framing we're ablating out."""

    def _init_vlm_system_prompts(self):
        robots = [self.robot1, self.robot2, self.robot3]
        action_block = self._render_action_descriptions(*(r.name for r in robots))
        robot_block = "\n".join(
            f"- {r.robot_id} ({r.name}): {r.description}\n"
            f"  Available actions: {r.available_actions}"
            for r in robots
        )
        rendered = VANILLA_SYSTEM_PROMPT.format(
            robot_block=robot_block,
            action_descriptions=action_block,
        )
        self.vlm1.set_system_prompt(rendered)
        self.vlm2.set_system_prompt(rendered)
        self.vlm3.set_system_prompt(rendered)


# ─── Per-task strategy ────────────────────────────────────────────────

def run_one_task(task, llm, sampling_params, args, logger, stats):
    _install_vanilla_prompts()         # idempotent; safe to call every task

    robots = task["scene_config"]["robots"]
    robot1 = RobotProfile(name=robots["R1"], robot_id="R1")
    robot2 = RobotProfile(name=robots["R2"], robot_id="R2")
    robot3 = RobotProfile(name=robots["R3"], robot_id="R3")

    vlm1 = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    vlm2 = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM2_R2_ADVOCATE, logger=logger, stats=stats,
    )
    vlm3 = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM3_R3_ADVOCATE, logger=logger, stats=stats,
    )
    sim = SimulatorInterface(scene_seed=0)
    eng = VanillaDebateEngine(
        vlm1, vlm2, vlm3, sim, robot1, robot2, robot3,
        max_debate_rounds=args.max_debate_rounds,
        max_retry_rounds=args.max_retry_rounds,
    )

    result = eng.run(
        task["task_description"], task["image_path"], task["scene_config"]
    )

    last_exec = (result.get("execution_results") or [{}])[-1]
    return {
        "success":               bool(result.get("success")),
        "debate_loops":          int(result.get("debate_loop_count", 0)),
        "debate_rounds_per_loop": list(result.get("debate_rounds_per_loop", [])),
        "last_failure":          last_exec.get("failure_reason"),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-agent Vanilla Debate (VD) ablation on VIKI-L2")
    add_common_args(p)
    p.add_argument("--max-debate-rounds", type=int, default=3,
                   help="max rounds per Phase-2 invocation (default 3).")
    p.add_argument("--max-retry-rounds", type=int, default=2,
                   help="max retries on execution failure (default 2).")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(args, strategy_label="vd", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
