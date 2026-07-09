"""
evaluate_viki_l2_baseline_local
================================
VIKI-Bench L2 official single-VLM **ZERO-SHOT** baseline, calling a
LOCAL VLM (via vLLM) instead of the SiliconFlow API.

Prompt + answer parsing are IDENTICAL to `evaluate_viki_l2_baseline.py`
— i.e. VIKI's own `VIKI-R/eval/VIKI-L2/qwen.py` verbatim, minus the
`<think>` requirement (which crashes non-thinking Instruct models — see
the docstring in `evaluate_viki_l2_baseline.py` for the diagnosis). The
`<answer>[...]</answer>` block is parsed by VIKI's `parse_viki_answer`.

One VLM call per task, no debate / refinement / retry. This is the
faithful VIKI zero-shot reference point that the other 8 scripts
(main debate, VD, PC, MAD, SC, SR, SR-C, plus the API baseline) all
compare against.

Records produced match evaluate_viki_l2_localmodel.py's schema so
aggregate_metrics.py works unchanged:
  debate_loops           = 1
  debate_rounds_per_loop = [1]

Example
-------
  python evaluate_viki_l2_baseline_local.py \\
      --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --tensor-parallel-size 2 --max-model-len 8192\\
      --enforce-eager --max-num-seqs 4 --temperature 0.0 \\
      --trust-remote-code
"""

import argparse

from multi_agent_debate import (
    SimulatorInterface, DebateRole,
    ROBOT_DESCRIPTION, ACTION_DESCRIPTION, AGENT_AVAIL_ACTIONS,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import add_common_args, run_evaluation
# Reuse VIKI's stock prompt + <answer>...</answer> parser directly from
# the API-based baseline so the two scripts stay in lockstep on the
# VIKI-faithful contract. Only the transport (vLLM vs SiliconFlow) differs.
from evaluate_viki_l2_baseline import (
    VIKI_L2_SYSTEM_PROMPT, parse_viki_answer,
)


def run_one_task(task, llm, sampling_params, args, logger, stats):
    robots = task["scene_config"]["robots"]
    sim = SimulatorInterface(scene_seed=0)

    # Build VIKI's stock system prompt for this task's specific robot team.
    # Same formatting call as evaluate_viki_l2_baseline.py — swap nothing.
    robot_types = list(robots.values())
    available_actions_view = {r: AGENT_AVAIL_ACTIONS.get(r, []) for r in robot_types}
    available_robots_view  = {r: ROBOT_DESCRIPTION.get(r, "")   for r in robot_types}
    system_prompt = VIKI_L2_SYSTEM_PROMPT.format(
        ACTION_DESCRIPTION=ACTION_DESCRIPTION,
        robots=robots,
        available_robots=available_robots_view,
        available_actions=available_actions_view,
    )

    vlm = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params,
        model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    vlm.set_system_prompt(system_prompt)

    # ── ONE VLM call — VIKI's exact user-message shape ──
    # "Task Description: <text>" with the scene image attached. That's it.
    response = vlm.query(
        f"Task Description: {task['task_description']}",
        image_path=task["image_path"],
    )

    # ── Parse VIKI's <answer>...</answer> block into a TaskPlan ──
    plan = parse_viki_answer(response)
    if plan is None:
        print("  [PARSE] failed to extract a plan from <answer>...</answer>")
        return {
            "success":                False,
            "debate_loops":           1,
            "debate_rounds_per_loop": [1],
            "last_failure":           "parse_failed: no <answer>[...]</answer> block",
        }

    # ── Execute in symbolic simulator ──
    try:
        exec_result = sim.execute_plan(plan, task["scene_config"])
    except Exception as e:
        return {
            "success":                False,
            "debate_loops":           1,
            "debate_rounds_per_loop": [1],
            "last_failure":           f"simulator crash: {e}",
        }

    success      = bool(exec_result.get("success"))
    last_failure = None if success else exec_result.get("failure_reason")
    if success:
        print("  [SIM] SUCCESS")
    else:
        first_line = str(last_failure).splitlines()[0] if last_failure else "unknown"
        print(f"  [SIM] FAIL — {first_line}")

    return {
        "success":                success,
        "debate_loops":           1,
        "debate_rounds_per_loop": [1],
        "last_failure":           last_failure,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="VIKI-L2 ZERO-SHOT single-VLM baseline on a LOCAL vLLM "
                    "(faithful to VIKI's official qwen.py; one call per task)")
    add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    # VIKI's official reference uses temperature=0.0 (greedy). Our shared
    # default via add_common_args is 0.3. Nudge the user if they didn't
    # explicitly set it — 0.0 is the most faithful reproduction.
    if args.temperature == 0.3:
        print("[INFO] --temperature defaulted to 0.3. VIKI's official "
              "zero-shot uses 0.0 (greedy). Pass --temperature 0.0 for "
              "the most faithful reproduction.")
    run_evaluation(args, strategy_label="baseline_local", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
