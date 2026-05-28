"""
evaluate_viki_l2_sc
===================
Single-agent **Self-Consistency** ablation. Always draws ALL N samples
from ONE VLM (no early exit), then **votes by plan content** to pick a
single winning plan:

  - Group the N parsed plans by their canonical step sequence.
  - The group with the most votes wins (plurality).
  - Ties are broken in favor of a group with at least one successful
    simulator execution.
  - When every plan is unique (N groups of 1), the tie-break picks the
    first successful sample if any, else sample 0.
  - `success` = the winning plan's simulator execution succeeded.

Higher sampling temperature (default bumped to 0.7) is needed because SC
relies on sample-to-sample diversity; at temperature 0.3 vLLM would emit
near-identical plans and SC would degenerate to a single attempt.

Records produced (additions on top of the shared schema):
  - debate_loops          = N (every task runs all samples — by design)
  - debate_rounds_per_loop = [1] * N
  - first_pass_success    = bool, did SAMPLE 0 succeed
                            (aggregate_metrics.py uses this for the SC
                            First-Pass Success metric, since the usual
                            `debate_loops == 1` heuristic doesn't apply)
  - sc_n_successful_samples = how many of N samples passed simulator
  - sc_winning_plan_votes   = vote count of the winning plan group

Example:
  python evaluate_viki_l2_sc.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --n-samples 5 --temperature 0.7 --limit 20 --trust-remote-code --tensor-parallel-size 2
"""

import argparse
import json
from collections import defaultdict
from typing import Optional

from multi_agent_debate import (
    SimulatorInterface, DebateRole, parse_plan_from_response, TaskPlan,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import (
    PROPOSAL_PROMPT_SA,
    add_common_args, render_system_prompt, run_evaluation,
)


def _plan_key(plan: Optional[TaskPlan]) -> Optional[str]:
    """Canonical hash for an SC vote: JSON of the step list with sorted
    inner keys. None means the response didn't parse into a plan."""
    if plan is None:
        return None
    try:
        return json.dumps(plan.steps, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def _vote_winner(plans: list[Optional[TaskPlan]],
                 execs: list[dict]) -> tuple[Optional[int], int]:
    """Plurality vote on plan content. Returns (winner_sample_idx, votes).
    Ties broken by groups containing at least one successful execution,
    then by lowest sample index. Returns (None, 0) if no parseable plan."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(plans):
        k = _plan_key(p)
        if k is not None:
            groups[k].append(i)
    if not groups:
        return None, 0

    def group_score(idxs):
        n_votes   = len(idxs)
        any_ok    = int(any(execs[i].get("success") for i in idxs))
        first_idx = -min(idxs)             # higher = earlier sample (we negate so max() prefers it)
        return (n_votes, any_ok, first_idx)

    winner_key = max(groups, key=lambda k: group_score(groups[k]))
    winner_idxs = groups[winner_key]
    # Within the winning group, prefer a successful sample as the "representative".
    rep = next((i for i in winner_idxs if execs[i].get("success")),
               winner_idxs[0])
    return rep, len(winner_idxs)


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

    prompt = PROPOSAL_PROMPT_SA.format(
        task_description=task["task_description"],
        world_state=world_state or "(unavailable)",
    )

    plans: list[Optional[TaskPlan]] = []
    execs: list[dict] = []

    # ── ALWAYS draw all N samples (no early exit) ──
    for sample_idx in range(args.n_samples):
        print(f"\n--- SC sample {sample_idx + 1}/{args.n_samples} ---")
        response = vlm.query(prompt, image_path=task["image_path"])
        plan = parse_plan_from_response(response)
        plans.append(plan)

        if plan is None:
            execs.append({"success": False, "failure_reason": "parse_failed"})
            print(f"  [VLM] sample {sample_idx + 1}: parse failed")
            continue
        try:
            r = sim.execute_plan(plan, task["scene_config"])
        except Exception as e:
            r = {"success": False, "failure_reason": f"simulator crash: {e}"}
        execs.append(r)
        if r.get("success"):
            print(f"  [SIM] sample {sample_idx + 1}: SUCCESS")
        else:
            fl = (r.get("failure_reason") or "unknown").splitlines()[0]
            print(f"  [SIM] sample {sample_idx + 1}: FAIL — {fl}")

    # ── Vote ──
    winner_idx, votes = _vote_winner(plans, execs)
    n_succ = sum(1 for r in execs if r.get("success"))

    if winner_idx is None:
        success      = False
        last_failure = "all samples failed to parse"
        win_label    = "n/a"
    else:
        win_exec     = execs[winner_idx]
        success      = bool(win_exec.get("success"))
        last_failure = None if success else (win_exec.get("failure_reason")
                                             or "unknown failure")
        win_label    = f"sample {winner_idx + 1} (votes={votes})"

    first_pass = bool(execs[0].get("success")) if execs else False

    print(f"\n  [VOTE] winning plan: {win_label}   "
          f"successful samples: {n_succ}/{args.n_samples}   "
          f"task → {'SUCCESS' if success else 'FAIL'}")

    return {
        "success":               success,
        "debate_loops":          args.n_samples,
        "debate_rounds_per_loop": [1] * args.n_samples,
        "last_failure":          last_failure,
        # SC-specific fields (aggregate_metrics.py reads first_pass_success):
        "first_pass_success":    first_pass,
        "sc_n_successful_samples": n_succ,
        "sc_winning_plan_votes":   votes,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-agent Self-Consistency (SC) ablation on VIKI-L2")
    add_common_args(p)
    p.add_argument("--n-samples", type=int, default=5,
                   help="number of independent plan samples per task "
                        "(default 5). All N are drawn — no early exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.temperature == 0.3:               # the common default
        print("[INFO] temperature=0.3 will give near-identical SC samples; "
              "bumping to 0.7 for diversity. Pass --temperature explicitly "
              "to override.")
        args.temperature = 0.7
    run_evaluation(args, strategy_label="sc", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
