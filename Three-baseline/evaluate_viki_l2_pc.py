"""
evaluate_viki_l2_pc
===================
Multi-agent **Proposer-Critic** ablation: TWO VLMs with ASYMMETRIC roles.
  VLM1 (PROPOSER) generates and revises plans.
  VLM2 (CRITIC)   reviews plans and lists issues — but NEVER writes a plan.

Flow per task:
  1) PROPOSER generates the initial plan.
  2) Refinement loop (up to --max-iters):
     a) Try execution. If success → exit.
     b) CRITIC reviews the current plan (with the simulator's failure
        feedback from the last execution as primary evidence).
     c) If CRITIC verdicts ACCEPT despite the failure, we force a
        revision by passing the raw simulator feedback as the issue
        (the symbolic ground truth trumps a permissive critic).
     d) PROPOSER revises based on CRITIC's issues.
  3) Final execution.

Comparison axes:
  - vs SR-C: same iteration structure and same checker feedback, but the
    critique and revise come from TWO different models instead of one
    self-critiquing.
  - vs Vanilla Debate: symmetric (both critique each other) vs asymmetric
    (one always critiques, one always proposes).

Records produced match evaluate_viki_l2_localmodel.py's schema so
aggregate_metrics.py works unchanged:
  - debate_loops          = 1 + #refine iterations that actually ran
  - debate_rounds_per_loop = [1] (initial) + [2]*K (each refine = critic + proposer)

Example:
  python evaluate_viki_l2_pc.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --max-iters 6 --limit 20 --trust-remote-code --tensor-parallel-size 2
"""

import argparse
import json

from multi_agent_debate import (
    SimulatorInterface, DebateRole, parse_plan_from_response,
    ROBOT_DESCRIPTION, ACTION_DESCRIPTION, AGENT_AVAIL_ACTIONS,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import (
    PROPOSAL_PROMPT_SA, REVISE_PROMPT_SA,
    add_common_args, run_evaluation,
    parse_critique_issues, format_execution_feedback,
)


# ─── Asymmetric system prompts ────────────────────────────────────────
# 3-embodiment note: the robot block is now built DYNAMICALLY from the
# task's robots dict (any number of robots), so R3 is no longer dropped.

_ROBOT_BLOCK_TEMPLATE = """\
## Available robots and their action primitives
{robot_lines}

## VIKI-Bench action primitives and rules
Each action is a JSON list `[action_type, target_object_or_location]`. \
**Each robot can only perform ONE action per time step.** Every step must include an \
entry for EVERY robot (use `["Wait"]` for idle).
Action primitives and descriptions:
{action_descriptions}
Use exact object and location names from the task. Do not invent new entity names. Choose the \
primitive that advances the current object state, not just the task name (Move → Reach → \
Grasp → Move → Place is the canonical pick-and-place sequence; Interact activates appliances)."""


PROPOSER_SYSTEM_PROMPT = """\
You are the PROPOSER agent for a multi-robot team in VIKI-Bench. Your job is to GENERATE and \
REVISE joint plans for the team. A separate CRITIC agent will review your plans and give you \
feedback; you must revise based on their critique.

You must first analyze the image to fully understand the scene. Then analyze the task. Then \
produce or revise the plan. Your reasoning must strictly adhere to the visual content of the \
image and the task description — no assumptions, hypotheses, or guesses allowed.

{robot_block}

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


CRITIC_SYSTEM_PROMPT = """\
You are the CRITIC agent for a multi-robot team in VIKI-Bench. Your job is to REVIEW joint \
plans produced by the PROPOSER agent and identify issues that need fixing. You DO NOT write \
plans yourself — you only critique. Be concrete and actionable: when you find issues, name \
the specific step and explain why it violates feasibility / coordination / completeness / \
efficiency.

You must first analyze the image and the task before reviewing. Your reasoning must strictly \
adhere to the visual content and task description — no assumptions, hypotheses, or guesses.

{robot_block}

## Required output format (strict JSON)
Respond ONLY in one of these two forms.

If the plan is correct:
```json
{{"verdict": "ACCEPT", "reasoning": "why the plan is good"}}
```
If the plan has issues:
```json
{{"verdict": "REVISE", "issues": ["concrete issue 1", "concrete issue 2", ...]}}
```
Do NOT include a revised plan — only issues. The proposer will write the revision.
"""


CRITIQUE_PROMPT_PC = """\
## Current plan from the proposer
{plan_json}

## Symbolic simulator feedback (ground truth from the last execution attempt)
{execution_feedback}

## Your task
Review the plan above. The simulator already executed it and reported the feedback shown — \
use that as PRIMARY evidence for what went wrong. Identify concrete, actionable issues that \
the proposer must fix in the next revision. If you genuinely believe the plan is correct \
(rare given the feedback says it failed), respond ACCEPT.
"""


# ─── Render helper ────────────────────────────────────────────────────

def _render_pc_system(template: str, robots: dict) -> str:
    """Fill in the robot/action block of a PC system prompt template.
    Enumerates ALL robots in the task (R1, R2, R3, …), not just R1/R2."""
    robot_lines = []
    relevant = set()
    for rid, rname in robots.items():
        if not rname:
            continue
        relevant.update(AGENT_AVAIL_ACTIONS.get(rname, []))
        robot_lines.append(
            f"- {rid} ({rname}): {ROBOT_DESCRIPTION.get(rname, '')}\n"
            f"  Available actions: {AGENT_AVAIL_ACTIONS.get(rname, [])}"
        )
    action_block = "\n".join(f"- {ACTION_DESCRIPTION[a]}"
                             for a in ACTION_DESCRIPTION if a in relevant)
    robot_block = _ROBOT_BLOCK_TEMPLATE.format(
        robot_lines="\n".join(robot_lines),
        action_descriptions=action_block,
    )
    return template.format(robot_block=robot_block)


# ─── Per-task strategy ────────────────────────────────────────────────

def run_one_task(task, llm, sampling_params, args, logger, stats):
    robots = task["scene_config"]["robots"]
    sim = SimulatorInterface(scene_seed=0)

    proposer = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    critic = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM2_R2_ADVOCATE, logger=logger, stats=stats,
    )
    proposer.set_system_prompt(_render_pc_system(PROPOSER_SYSTEM_PROMPT, robots))
    critic.set_system_prompt(_render_pc_system(CRITIC_SYSTEM_PROMPT, robots))

    try:
        world_state = sim.get_initial_world_state(task["scene_config"])
    except Exception as e:
        print(f"  [WARN] could not render initial world state: {e}")
        world_state = ""
    img = task["image_path"]

    # ── Initial plan from the proposer (1 VLM call) ──
    print("\n--- PC initial proposal ---")
    response = proposer.query(
        PROPOSAL_PROMPT_SA.format(
            task_description=task["task_description"],
            world_state=world_state or "(unavailable)",
        ),
        image_path=img,
    )
    current_plan = parse_plan_from_response(response)
    rounds_per_loop = [1]
    if current_plan is None:
        return {"success": False, "debate_loops": 1,
                "debate_rounds_per_loop": rounds_per_loop,
                "last_failure": "initial plan parse failed"}

    last_failure: str | None = None
    last_exec: dict = {}

    for attempt_idx in range(args.max_iters + 1):
        try:
            exec_result = sim.execute_plan(current_plan, task["scene_config"])
        except Exception as e:
            return {"success": False, "debate_loops": attempt_idx + 1,
                    "debate_rounds_per_loop": rounds_per_loop,
                    "last_failure": f"simulator crash: {e}"}

        if exec_result.get("success"):
            print(f"  [SIM] attempt {attempt_idx + 1}: SUCCESS")
            return {"success": True, "debate_loops": attempt_idx + 1,
                    "debate_rounds_per_loop": rounds_per_loop,
                    "last_failure": None}

        last_exec = exec_result
        last_failure = exec_result.get("failure_reason") or "unknown failure"
        first_line  = str(last_failure).splitlines()[0]
        print(f"  [SIM] attempt {attempt_idx + 1}: FAIL — {first_line}")

        if attempt_idx == args.max_iters:
            break

        # ── Critic reviews (1 call to critic) + Proposer revises (1 call to proposer) ──
        print(f"\n--- PC refine iteration {attempt_idx + 1}/{args.max_iters} ---")
        plan_json = json.dumps({"steps": current_plan.steps}, indent=2)
        feedback  = format_execution_feedback(last_exec)

        crit_response = critic.query(
            CRITIQUE_PROMPT_PC.format(plan_json=plan_json, execution_feedback=feedback),
            image_path=img,
        )
        issues = parse_critique_issues(crit_response)
        if not issues:
            # Critic verdicted ACCEPT (or its response didn't parse), but the
            # simulator says the plan failed. Trust the simulator: feed the
            # raw failure back to the proposer as the only "issue".
            issues = [f"Simulator feedback (must fix): {feedback.strip()}"]
        issues_list = "\n".join(f"- {x}" for x in issues)

        rev_response = proposer.query(
            REVISE_PROMPT_SA.format(plan_json=plan_json, issues_list=issues_list),
            image_path=img,
        )
        rounds_per_loop.append(2)               # 1 critic + 1 proposer call per iter
        new_plan = parse_plan_from_response(rev_response)
        if new_plan is None:
            return {"success": False, "debate_loops": attempt_idx + 2,
                    "debate_rounds_per_loop": rounds_per_loop,
                    "last_failure": f"revise parse failed; prev: {last_failure}"}
        current_plan = new_plan

    return {"success": False,
            "debate_loops": len(rounds_per_loop),
            "debate_rounds_per_loop": rounds_per_loop,
            "last_failure": last_failure}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-agent Proposer-Critic (PC) ablation on VIKI-L2")
    add_common_args(p)
    p.add_argument("--max-iters", type=int, default=6,
                   help="max refine iterations after the initial plan (default 6).")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(args, strategy_label="pc", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
