"""
evaluate_viki_l2_sr
===================
Single-agent **Self-Refine** ablation (Madaan et al. 2023 style):

  1) Generate an initial plan from one VLM.
  2) Execute it in the simulator. If success → exit.
  3) Otherwise the SAME VLM self-critiques its own plan (NO simulator
     feedback — that's the SR-C variant), then revises based on its
     critique. Execute the revised plan.
  4) Repeat (3) up to `--max-iters` times.

This is the "introspection-only" baseline — the model must spot its
own errors from the plan alone. Contrast against SR-C, which feeds
the symbolic simulator's error message back in.

Records produced match evaluate_viki_l2_localmodel.py's schema so
aggregate_metrics.py works unchanged:
  - debate_loops          = # attempts (1 = initial plan succeeded;
                            >1 = how many refinement iterations were used)
  - debate_rounds_per_loop = per-attempt VLM call count
                            (initial=1, each refine iter=2: critique + revise)

Example:
  python evaluate_viki_l2_sr.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --max-iters 6 --limit 20 --trust-remote-code --tensor-parallel-size 2
"""

import argparse
import json

from multi_agent_debate import (
    SimulatorInterface, DebateRole, parse_plan_from_response,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import (
    PROPOSAL_PROMPT_SA, CRITIQUE_PROMPT_SA, REVISE_PROMPT_SA,
    add_common_args, render_system_prompt, run_evaluation,
    parse_critique_issues,
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
    print("\n--- SR initial proposal ---")
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

    # Attempt = 1 initial + up to max_iters refinements. Each attempt is
    # executed; on success we exit early.
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

        last_failure = exec_result.get("failure_reason") or "unknown failure"
        first_line  = str(last_failure).splitlines()[0]
        print(f"  [SIM] attempt {attempt_idx + 1}: FAIL — {first_line}")

        if attempt_idx == args.max_iters:
            break        # used up the refinement budget; the last exec sets last_failure

        # ── Self-critique (no checker feedback) + Revise = 2 VLM calls ──
        print(f"\n--- SR refine iteration {attempt_idx + 1}/{args.max_iters} ---")
        plan_json = json.dumps({"steps": current_plan.steps}, indent=2)

        crit_response = vlm.query(
            CRITIQUE_PROMPT_SA.format(plan_json=plan_json),
            image_path=img,
        )
        issues = parse_critique_issues(crit_response)
        if not issues:
            # Model said ACCEPT or couldn't parse — but the plan demonstrably
            # failed, so we have to push it. Use a generic prod.
            issues = ["The plan failed when executed. Identify the broken "
                      "pre-conditions and reorder / add the necessary "
                      "Move / Reach / Grasp setup steps."]
        issues_list = "\n".join(f"- {x}" for x in issues)

        rev_response = vlm.query(
            REVISE_PROMPT_SA.format(plan_json=plan_json, issues_list=issues_list),
            image_path=img,
        )
        rounds_per_loop.append(2)               # 2 calls per refine iteration
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
        description="Single-agent Self-Refine (SR) ablation on VIKI-L2")
    add_common_args(p)
    p.add_argument("--max-iters", type=int, default=6,
                   help="max self-refinement iterations after the initial "
                        "plan (default 6).")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(args, strategy_label="sr", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
