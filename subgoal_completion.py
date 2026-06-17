"""
subgoal_completion
==================
Compute SUBGOAL COMPLETION RATE — a partial-credit metric.

  rate = Σ_task n_satisfied_goals(task)  /  Σ_task n_total_goals(task)

A task with 2 goals where only 1 is satisfied contributes 1/2, not 0.
This complements the binary task-success rate that aggregate_metrics.py
already reports.

How per-task n_satisfied is computed
------------------------------------
The eval-time `SimulatorInterface.execute_plan()` already counts goal
satisfaction (it powers `agent_activation_score` in the metrics block),
but that value was never persisted to results.json. So we recover it
post-hoc:

  - task.success == True  → n_satisfied = n_total_goals (trust the flag).
  - task.success == False → parse the LAST parseable plan from the task's
                             call logs, re-run it through the simulator,
                             and round (activation_score × n_total_goals).
  - if no log dir / no parseable plan / simulator crash
                             → n_satisfied = 0 (conservative).

Multi-dir support mirrors stratified_acc.py / aggregate_metrics.py:
each input path can be a leaf result dir or a parent like
`evaluate_results` (auto-discovered). Records from multiple dirs are
merged unless `--per-dir` is set.

Usage
-----
  python subgoal_completion.py evaluate_results
  python subgoal_completion.py evaluate_results --per-dir --verbose
"""

import argparse
import contextlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from multi_agent_debate import (
    SimulatorInterface, parse_plan_from_response, TaskPlan,
)
from viki_loader import load_viki_task
from aggregate_metrics import discover_result_dirs


_RESP_MARKER = "--- assistant (response) ---"


# ─── Log → final plan parsing ─────────────────────────────────────────

def _read_response(txt_path: Path) -> str:
    """Return the assistant-response body of one call log file."""
    txt = txt_path.read_text(encoding="utf-8", errors="replace")
    i = txt.rfind(_RESP_MARKER)
    return txt[i + len(_RESP_MARKER):].strip() if i != -1 else ""


def final_plan_from_logs(task_dir: Path) -> Optional[TaskPlan]:
    """Walk the task's call logs in order; return the LAST response that
    parses to a valid TaskPlan. Same heuristic as
    aggregate_metrics.final_generated_steps — the executed plan is the
    debate/refinement loop's last consensus.

    `parse_plan_from_response` print()s a `[WARN] Failed to parse plan: ...`
    on every JSON parse failure; we walk dozens of logs per task and many
    contain ACCEPT verdicts / prose / critique JSONs that are not plans,
    so its noise drowns out the actual progress output. We redirect stdout
    into a throwaway buffer just around the parse to mute it — the only
    print inside the function is that WARN, so nothing useful is lost."""
    last = None
    for f in sorted(task_dir.glob("[0-9]*_vlm*.txt")):
        with contextlib.redirect_stdout(io.StringIO()):
            plan = parse_plan_from_response(_read_response(f))
        if plan is not None:
            last = plan
    return last


def find_task_dir(source_dir: Path, idx: int) -> Optional[Path]:
    """Find task_{idx:05d}_* under source_dir."""
    hits = sorted(source_dir.glob(f"task_{idx:05d}_*"))
    return hits[0] if hits else None


# ─── Goal counting ────────────────────────────────────────────────────

def n_goals_of(scene_config: dict) -> int:
    """Atomic goal-predicate count for a viki_loader-built scene_config.
    `viki_loader._denumpy` already converted ndarrays to native lists,
    so plain len()/sum() is safe here (unlike on raw parquet rows)."""
    gc = scene_config.get("goal_constraints", [])
    try:
        return sum(len(g) for g in gc) if len(gc) > 0 else 0
    except TypeError:
        return 0


# ─── Per-task subgoal counting ────────────────────────────────────────

def per_task_satisfied(record: dict, source_dir: Path,
                       sim: SimulatorInterface,
                       parquet_path: Path) -> tuple[int, int, str]:
    """Return (n_satisfied, n_total_goals, reason_tag) for one task.

    reason_tag is a short label for grouping in the verbose breakdown:
      - "ok-success"         success=True → full credit
      - "ok-replay-partial"  failed, replay yielded partial satisfaction
      - "ok-replay-zero"     failed, replay yielded zero satisfaction
      - "no-plan"            failed, no parseable plan in logs
      - "no-task-dir"        failed, no task_* folder on disk
      - "load-failed"        parquet load failed for this idx
      - "sim-crash"          simulator threw during replay
    """
    idx = record["idx"]
    success = bool(record.get("success"))

    try:
        task = load_viki_task(parquet_path, idx)
    except Exception:
        return 0, 0, "load-failed"

    scene = task["scene_config"]
    n_total = n_goals_of(scene)
    if n_total == 0:
        # Pathological / no-goal task — nothing to credit.
        return 0, 0, "ok-success" if success else "no-plan"

    # Success path: trust the recorded flag.
    if success:
        return n_total, n_total, "ok-success"

    # Failure path: re-execute the final plan to count partial credit.
    tdir = find_task_dir(source_dir, idx)
    if tdir is None:
        return 0, n_total, "no-task-dir"

    plan = final_plan_from_logs(tdir)
    if plan is None:
        return 0, n_total, "no-plan"

    try:
        exec_result = sim.execute_plan(plan, scene)
    except Exception:
        return 0, n_total, "sim-crash"

    activation = (exec_result.get("metrics") or {}).get("agent_activation_score", 0.0)
    n_sat = int(round(activation * n_total))
    # Clamp to [0, n_total] just in case of float rounding edges.
    n_sat = max(0, min(n_sat, n_total))
    tag = "ok-replay-partial" if n_sat > 0 else "ok-replay-zero"
    return n_sat, n_total, tag


# ─── Aggregation across one or more leaf dirs ─────────────────────────

def _load_records_from(leaf_dir: Path) -> list[dict]:
    """Load results.json records; tag each with `_source_dir` so the
    scorer knows which dir to glob for its task_* folder later."""
    rj = leaf_dir / "results.json"
    if not rj.is_file():
        print(f"[WARN] {leaf_dir}/results.json missing — skipping",
              file=sys.stderr)
        return []
    recs = json.loads(rj.read_text(encoding="utf-8"))
    return [{**r, "_source_dir": leaf_dir}
            for r in recs if "success" in r]


def _score_records(records: list[dict], parquet_path: Path,
                   sim: SimulatorInterface, verbose: bool = False
                   ) -> tuple[int, int, dict[str, int]]:
    """Score a list of records (each carrying its `_source_dir`).
    Returns (total_satisfied, total_goals, reason_tag_counts)."""
    total_sat = 0
    total_goals = 0
    tag_counts: dict[str, int] = defaultdict(int)

    for r in records:
        n_sat, n_total, tag = per_task_satisfied(
            r, r["_source_dir"], sim, parquet_path
        )
        total_sat   += n_sat
        total_goals += n_total
        tag_counts[tag] += 1
        if verbose:
            ok = "Y" if r.get("success") else "N"
            src = r["_source_dir"].name
            print(f"    idx={r['idx']:>5}  ok={ok}  "
                  f"goals={n_sat}/{n_total}  [{tag}]  "
                  f"src={src:<24s}  {r.get('task_id', '')}")

    return total_sat, total_goals, dict(tag_counts)


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Subgoal-completion rate (partial credit) over VIKI-L2 "
                    "evaluation runs")
    p.add_argument("results_dirs", nargs="+",
                   help="leaf result dir(s) or a parent like "
                        "`evaluate_results` to auto-discover children.")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--per-dir", action="store_true",
                   help="report one rate per dir instead of merging.")
    p.add_argument("--verbose", action="store_true",
                   help="print per-task replay decisions.")
    return p.parse_args()


def _print_summary(label: str, total_sat: int, total_goals: int,
                   tag_counts: dict[str, int]) -> None:
    print()
    print("=" * 64)
    print(f"SUBGOAL COMPLETION  —  {label}")
    print("=" * 64)
    rate = total_sat / total_goals if total_goals else 0.0
    n_tasks = sum(tag_counts.values())
    n_success = tag_counts.get("ok-success", 0)
    success_rate = n_success / n_tasks if n_tasks else 0.0
    print(f"  Goals satisfied / total : {total_sat} / {total_goals}")
    print(f"  Subgoal completion rate : {rate:.2%}   (partial credit)")
    print(f"  Binary task success rate: {success_rate:.2%}   "
          f"({n_success}/{n_tasks}, for comparison)")
    if rate > success_rate:
        delta = rate - success_rate
        print(f"  → +{delta:.2%} comes from partial credit on failed tasks "
              f"(plans that achieved some goals before failing).")
    if tag_counts:
        print()
        print("  per-task contribution breakdown:")
        order = ["ok-success", "ok-replay-partial", "ok-replay-zero",
                 "no-plan", "no-task-dir", "load-failed", "sim-crash"]
        for tag in order:
            n = tag_counts.get(tag, 0)
            if n:
                print(f"    {tag:<22s} {n:>5}")


def main():
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).resolve().parent / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    leaf_dirs = discover_result_dirs([Path(p) for p in args.results_dirs])
    if not leaf_dirs:
        print(f"[ERROR] no results.json found under: {args.results_dirs}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(leaf_dirs)} result dir(s):")
    for d in leaf_dirs:
        print(f"  {d}")

    sim = SimulatorInterface(scene_seed=0)

    if args.per_dir:
        for d in leaf_dirs:
            print(f"\nProcessing {d.name}...")
            records = _load_records_from(d)
            sat, goals, tags = _score_records(records, parquet_path, sim,
                                              args.verbose)
            _print_summary(d.name, sat, goals, tags)
    else:
        # Merge across all input dirs, dedup by idx — important for
        # resumed runs where the same baseline's output got split across
        # multiple `local_*` dirs. Matches aggregate_metrics.py's
        # "later run wins" convention.
        by_idx: dict[int, dict] = {}
        n_dup = 0
        for d in leaf_dirs:
            for r in _load_records_from(d):
                if r["idx"] in by_idx:
                    n_dup += 1
                by_idx[r["idx"]] = r              # later dir wins
        if n_dup:
            print(f"  [WARN] {n_dup} duplicate idx across dirs — kept "
                  f"the later dir's record for each.")
        records = sorted(by_idx.values(), key=lambda x: x["idx"])
        print(f"\nProcessing {len(records)} unique tasks "
              f"merged from {len(leaf_dirs)} dir(s)...")
        sat, goals, tags = _score_records(records, parquet_path, sim,
                                          args.verbose)
        _print_summary(f"MERGED ({len(leaf_dirs)} dirs, {len(records)} tasks)",
                       sat, goals, tags)


if __name__ == "__main__":
    main()
