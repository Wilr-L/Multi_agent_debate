"""
recompute_sc_metrics
====================
Re-score Self-Consistency (SC) eval runs under the NEW random-tiebreak
voting rule WITHOUT re-running the VLM.

Background
----------
The OLD SC tiebreak preferred groups containing at least one successful
simulator execution — a soft form of "peeking at the simulator" that
isn't faithful to a real SC system (which has no oracle at inference
time). The NEW rule picks among tied-at-max-votes groups uniformly at
RANDOM (see evaluate_viki_l2_sc.py).

How this script reconstructs the new metric
-------------------------------------------
The SC log dir already contains every sampled plan response. The
symbolic simulator is deterministic (scene_seed=0), so:
  1) Parse every sample's plan from its NNN_vlm1.txt log.
  2) Re-execute each plan in SimulatorInterface → per-sample success.
  3) Group samples by canonical plan-step hash.
  4) Find tied-at-max-votes groups. Under uniform-random tiebreak,
     P(task success) = (# tied groups whose plan succeeded) / (# tied)
     — an exact expectation, no Monte-Carlo noise.

So the script gives you the SAME number you'd get by re-running the
eval with random.choice averaged over infinite trials, but instantly
and with no GPU.

Usage
-----
  python recompute_sc_metrics.py evaluate_results/sc_20260529_001348
  python recompute_sc_metrics.py evaluate_results          # auto-discover sc_*
  python recompute_sc_metrics.py dirA dirB ... --verbose
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from multi_agent_debate import SimulatorInterface, parse_plan_from_response
from viki_loader import load_viki_task


_RESP_MARKER = "--- assistant (response) ---"


# ─── Log parsing ──────────────────────────────────────────────────────

def _read_response(txt_path: Path) -> str:
    """Return the assistant-response body of a single NNN_vlmN.txt file."""
    txt = txt_path.read_text(encoding="utf-8", errors="replace")
    i = txt.rfind(_RESP_MARKER)
    return txt[i + len(_RESP_MARKER):].strip() if i != -1 else ""


def _plan_key(plan) -> Optional[str]:
    """Canonical hash for a parsed plan (matches evaluate_viki_l2_sc.py)."""
    if plan is None:
        return None
    try:
        return json.dumps(plan.steps, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


# ─── Per-task analysis ────────────────────────────────────────────────

def process_task(task_dir: Path, parquet_path: Path) -> Optional[dict]:
    """Re-execute the SC samples in `task_dir` and compute both the OLD
    (success-preferring) and NEW (uniform-random) tiebreak outcomes."""
    m = re.match(r"task_(\d+)_", task_dir.name)
    if not m:
        return None
    idx = int(m.group(1))

    try:
        task = load_viki_task(parquet_path, idx)
    except Exception as e:
        print(f"  [SKIP] idx={idx}: load_viki_task failed: {e}", file=sys.stderr)
        return None

    sim = SimulatorInterface(scene_seed=0)
    scene_config = task["scene_config"]

    sample_files = sorted(task_dir.glob("[0-9]*_vlm*.txt"))
    plans = [parse_plan_from_response(_read_response(f)) for f in sample_files]

    succs: list[bool] = []
    for p in plans:
        if p is None:
            succs.append(False)
            continue
        try:
            r = sim.execute_plan(p, scene_config)
            succs.append(bool(r.get("success")))
        except Exception:
            succs.append(False)

    # Group by plan content
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(plans):
        k = _plan_key(p)
        if k is not None:
            groups[k].append(i)

    if not groups:                          # every sample failed to parse
        return {
            "idx": idx, "n_samples": len(plans), "n_succ": 0,
            "first_pass_success": False,
            "old_success": False, "new_expected_success": 0.0,
            "tied_groups": 0, "max_votes": 0, "n_groups": 0,
        }

    max_votes  = max(len(idxs) for idxs in groups.values())
    tied_keys  = [k for k, idxs in groups.items() if len(idxs) == max_votes]

    def group_succeeded(k):
        # All samples in a group share the SAME plan, hence the same
        # deterministic simulator outcome — any() == all() here.
        return any(succs[i] for i in groups[k])

    old_success         = any(group_succeeded(k) for k in tied_keys)
    n_succ_tied         = sum(1 for k in tied_keys if group_succeeded(k))
    new_expected_success = n_succ_tied / len(tied_keys)

    return {
        "idx":                idx,
        "n_samples":          len(plans),
        "n_succ":             sum(succs),
        "first_pass_success": succs[0] if succs else False,
        "old_success":        old_success,
        "new_expected_success": new_expected_success,
        "tied_groups":        len(tied_keys),
        "max_votes":          max_votes,
        "n_groups":           len(groups),
    }


# ─── Directory discovery ──────────────────────────────────────────────

def discover_sc_dirs(paths: list[Path]) -> list[Path]:
    """Each input path is either an SC leaf dir (has task_* children) or
    a parent dir holding multiple sc_* subdirs. De-dup, preserve order."""
    leaves, seen = [], set()
    for p in paths:
        if any(p.glob("task_*")):
            cands = [p]
        else:
            cands = sorted(c for c in p.glob("sc_*") if any(c.glob("task_*")))
        for c in cands:
            rp = c.resolve()
            if rp not in seen:
                seen.add(rp)
                leaves.append(c)
    return leaves


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Re-score SC evaluation under "
                                            "the new random-tiebreak rule")
    p.add_argument("results_dirs", nargs="+",
                   help="one or more SC result dirs (or a parent like "
                        "`evaluate_results` to auto-discover all sc_* runs).")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--verbose", action="store_true",
                   help="print per-task analysis line before the summary.")
    return p.parse_args()


def main():
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).resolve().parent / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    leaf_dirs = discover_sc_dirs([Path(p) for p in args.results_dirs])
    if not leaf_dirs:
        print(f"[ERROR] no SC result dirs with task_* folders under "
              f"{args.results_dirs}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(leaf_dirs)} SC dir(s):")
    for d in leaf_dirs:
        print(f"  {d}")

    rows = []
    for d in leaf_dirs:
        for task_dir in sorted(d.glob("task_*")):
            r = process_task(task_dir, parquet_path)
            if r is None:
                continue
            rows.append(r)
            if args.verbose:
                old_s = "Y" if r["old_success"] else "N"
                fp_s  = "Y" if r["first_pass_success"] else "N"
                print(f"  idx={r['idx']:>4}  "
                      f"n_succ={r['n_succ']}/{r['n_samples']}  "
                      f"groups={r['n_groups']}  "
                      f"tied={r['tied_groups']}@{r['max_votes']}v  "
                      f"old={old_s}  new_E={r['new_expected_success']:.2f}  "
                      f"fp={fp_s}")

    n = len(rows)
    if n == 0:
        print("No tasks processed.")
        return

    n_old   = sum(1 for r in rows if r["old_success"])
    n_fp    = sum(1 for r in rows if r["first_pass_success"])
    new_E   = statistics.mean(r["new_expected_success"] for r in rows)

    n_tie_changes = sum(1 for r in rows
                        if r["tied_groups"] > 1
                        and r["old_success"] != (r["new_expected_success"] == 1.0))
    n_real_ties   = sum(1 for r in rows if r["tied_groups"] > 1)

    print()
    print("=" * 64)
    print(f"SC RE-SCORING — {n} tasks")
    print("=" * 64)
    print(f"  Old voting (peek tiebreak) : {n_old}/{n} = {n_old/n:.2%}")
    print(f"  New voting (random)        : {new_E:.2%}  (expected, exact)")
    print(f"  Δ (new − old)              : {new_E - n_old/n:+.2%}")
    print(f"  First-pass success         : {n_fp}/{n} = {n_fp/n:.2%}")
    print("-" * 64)
    print(f"  Tasks with real ties       : {n_real_ties}/{n}")
    print(f"  Tasks where tiebreak flips : {n_tie_changes}/{n}")
    print("=" * 64)
    print()
    print("'new' is the analytic expectation under uniform-random tiebreak.")
    print("Re-running the eval with random.choice would give numbers")
    print("fluctuating around this value (variance shrinks ~1/sqrt(N_tasks)).")


if __name__ == "__main__":
    main()
