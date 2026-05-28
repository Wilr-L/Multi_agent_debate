"""
evaluate_viki_l2_src
====================
Single-agent **Self-Refine + Checker** ablation: identical to SR
(evaluate_viki_l2_sr.py) except the critique step receives the
**symbolic simulator's failure feedback** as the primary signal for
what went wrong, instead of asking the VLM to introspect on the plan
alone.

This is the "verification-driven" baseline — same model, same number
of iterations, only difference is whether the checker error is fed
back. The SR vs SR-C delta isolates how much of the gain comes from
external grounding vs pure self-critique.

Records produced match evaluate_viki_l2_localmodel.py's schema so
aggregate_metrics.py works unchanged.

Example:
  python evaluate_viki_l2_src.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --max-iters 6 --limit 20 --trust-remote-code --tensor-parallel-size 2
"""

import argparse
import json

from multi_agent_debate import (
    SimulatorInterface, DebateRole, parse_plan_from_response,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import (
    PROPOSAL_PROMPT_SA, CRITIQUE_PROMPT_SA_WITH_CHECKER, REVISE_PROMPT_SA,
    add_common_args, render_system_prompt, run_evaluation,
    parse_critique_issues, format_execution_feedback,
)


def run_one_task(task, llm, sampling_params, args, logger, stats):
    sim = SimulatorInterface(scene_seed=0)
    vlm = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params,
        model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    vlm.set_system_prompt(render_system_prompt(task["scene_config"]["robots"]))

    try:
        world_state = sim.get_initial_world_state(task["scene_config"])
    except Exception as e:
        print(f"  [WARN] could not render initial world state: {e}")
        world_state = ""
    img = task["image_path"]

    # ── Initial plan (1 VLM call) ──
    print("\n--- SR-C initial proposal ---")
    response = vlm.query(
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
    last_exec_result: dict = {}

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

        last_exec_result = exec_result
        last_failure = exec_result.get("failure_reason") or "unknown failure"
        first_line  = str(last_failure).splitlines()[0]
        print(f"  [SIM] attempt {attempt_idx + 1}: FAIL — {first_line}")

        if attempt_idx == args.max_iters:
            break

        # ── Critique with checker feedback + Revise = 2 VLM calls ──
        print(f"\n--- SR-C refine iteration {attempt_idx + 1}/{args.max_iters} ---")
        plan_json = json.dumps({"steps": current_plan.steps}, indent=2)
        feedback  = format_execution_feedback(last_exec_result)

        crit_response = vlm.query(
            CRITIQUE_PROMPT_SA_WITH_CHECKER.format(
                plan_json=plan_json, execution_feedback=feedback,
            ),
            image_path=img,
        )
        issues = parse_critique_issues(crit_response)
        if not issues:
            # Critique didn't parse — fall back to handing the raw
            # simulator feedback to the reviser as the only "issue".
            issues = [f"Simulator feedback: {feedback.strip()}"]
        issues_list = "\n".join(f"- {x}" for x in issues)

        rev_response = vlm.query(
            REVISE_PROMPT_SA.format(plan_json=plan_json, issues_list=issues_list),
            image_path=img,
        )
        rounds_per_loop.append(2)
        new_plan = parse_plan_from_response(rev_response)
        if new_plan is None:
            return {"success": False,
                    "debate_loops": attempt_idx + 2,
                    "debate_rounds_per_loop": rounds_per_loop,
                    "last_failure": f"revise parse failed; prev: {last_failure}"}
        current_plan = new_plan

    return {"success": False,
            "debate_loops": len(rounds_per_loop),
            "debate_rounds_per_loop": rounds_per_loop,
            "last_failure": last_failure}


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-agent Self-Refine + Checker (SR-C) ablation on VIKI-L2")
    add_common_args(p)
    p.add_argument("--max-iters", type=int, default=6,
                   help="max self-refinement iterations after the initial "
                        "plan (default 6).")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(args, strategy_label="src", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
