"""
subgoal_completion_3embodi
==========================
Subgoal-completion rate for the VIKI-L2 **3-robot** transport tasks
(`dog_push_box_for_two_panda_transport` family), with ONE EXTRA subgoal
per task on top of the parquet's `goal_constraints`:

    "was the goal object placed INTO the cardboardbox at some point
     during plan execution?"

For these tasks the object's journey is
    R1 grasps object → Places it into the box → box pushed to R3 →
    R3 grasps object from the box → Places it in the sink.
The parquet's only goal_constraint checks the FINAL location (object in
sink). But getting the object into the box is a meaningful intermediate
milestone that binary success and even the final-state activation score
both miss. This script credits it as a +1 subgoal and recomputes the
partial-credit rate:

    new_rate = Σ (n_satisfied_goals + inbox_hit)
             / Σ (n_total_goals    + 1)

where `inbox_hit ∈ {0,1}` is whether any goal object's position ever
equalled the box during a step-by-step replay of the executed plan.

How the extra subgoal is detected
----------------------------------
The final executed plan is recovered from the task's call logs (same
heuristic as subgoal_completion.py: last parseable plan). We then
PREFIX-REPLAY it — build the symbolic env and run steps[:k] for
k = 1..N — and after each prefix check whether any goal object's
`pos.name == <box>`. Confirmed empirically: a `Place cardboardbox`
executed while carrying the object sets the object's `pos.name` to
`"cardboardbox"`. A prefix that would fail simply leaves the world at
its last successful step, so box entry that happened before a later
failure is still caught (partial credit for plans that boxed the object
but then botched the sink placement).

The existing goal_constraint satisfaction is computed by REUSING
`subgoal_completion.per_task_satisfied` verbatim, so the "existing
subgoal rate" this script prints is byte-identical to subgoal_completion.py.

Multi-dir support and CLI mirror subgoal_completion.py.

Usage
-----
  python subgoal_completion_3embodi.py Three_debate/logs --verbose
  python subgoal_completion_3embodi.py Three_debate/logs Three-baseline/logs
  python subgoal_completion_3embodi.py <dir> --box-asset cardboardbox
"""

import argparse
import contextlib
import io
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from multi_agent_debate import SimulatorInterface, TaskPlan, DebateEval
from viki_loader import load_viki_task
from aggregate_metrics import discover_result_dirs
# Reuse the existing-goal scorer + log/plan helpers verbatim.
from subgoal_completion import (
    per_task_satisfied, final_plan_from_logs, find_task_dir, _load_records_from,
)


# ─── Goal-object + box helpers ────────────────────────────────────────

def goal_object_names(scene_config: dict) -> set:
    """Asset names referenced by the task's goal_constraints — the
    transported object(s). `viki_loader._denumpy` already made these
    native lists/dicts."""
    names: set = set()
    for g in scene_config.get("goal_constraints", []):
        for pred in g:
            if (isinstance(pred, dict) and pred.get("type") == "asset"
                    and pred.get("name")):
                names.add(pred["name"])
    return names


def scene_has_box(scene_config: dict, box_asset: str) -> bool:
    """True if the box asset is present in the scene (init_pos keys are
    `<asset_type>_<n>` or bare robot/base ids)."""
    asset_types = {k.rsplit("_", 1)[0] for k in scene_config.get("init_pos", {})}
    return box_asset in asset_types


# ─── Box-entry detection via prefix replay ────────────────────────────

def detect_box_entry(plan: TaskPlan, scene_config: dict, goal_objs: set,
                     box_asset: str, sim: SimulatorInterface,
                     max_steps: int = 40) -> bool:
    """Prefix-replay `plan`; return True iff any goal object's position
    ever equals `box_asset` after some prefix. Deterministic: re-seeds
    before each env build so all prefixes share identical initial asset
    positions (matching SimulatorInterface.execute_plan's single-seed
    behavior)."""
    if plan is None or not goal_objs:
        return False
    seed = sim.scene_seed if sim.scene_seed is not None else 0
    n = min(len(plan.steps), max_steps)
    for k in range(1, n + 1):
        random.seed(seed)
        meta = sim._build_env_metadata(scene_config)
        prefix = TaskPlan(steps=plan.steps[:k], reasoning="", raw_text="")
        recs = sim._plan_to_command_records(prefix)
        judger = DebateEval()
        judger.set_env(meta)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                judger.eval(recs)
            except Exception:
                continue
        for obj in goal_objs:
            asset = judger.env.assets.get(obj)
            if asset is not None and getattr(asset, "pos", None) is not None \
                    and asset.pos.name == box_asset:
                return True
    return False


# ─── Per-task scoring ─────────────────────────────────────────────────

def score_task(record: dict, source_dir: Path, sim: SimulatorInterface,
               parquet_path: Path, box_asset: str) -> dict:
    """Return per-task counts: existing goals (reused verbatim) + the
    extra in-box subgoal."""
    idx = record["idx"]
    # Existing-goal satisfaction — identical to subgoal_completion.py.
    n_sat, n_total, tag = per_task_satisfied(record, source_dir, sim, parquet_path)

    # Extra in-box subgoal.
    inbox_total = 0
    inbox_sat = 0
    box_tag = "no-box"
    try:
        task = load_viki_task(parquet_path, idx)
        scene = task["scene_config"]
        goal_objs = goal_object_names(scene)
        if scene_has_box(scene, box_asset) and goal_objs:
            inbox_total = 1
            tdir = find_task_dir(source_dir, idx)
            plan = final_plan_from_logs(tdir) if tdir else None
            if plan is None:
                box_tag = "box-no-plan"
            else:
                entered = detect_box_entry(plan, scene, goal_objs, box_asset, sim)
                inbox_sat = 1 if entered else 0
                box_tag = "box-in" if entered else "box-out"
    except Exception:
        box_tag = "box-load-failed"

    return {
        "idx": idx,
        "success": bool(record.get("success")),
        "task_id": record.get("task_id", ""),
        "n_sat": n_sat, "n_total": n_total, "goal_tag": tag,
        "inbox_sat": inbox_sat, "inbox_total": inbox_total, "box_tag": box_tag,
        "source": source_dir.name,
    }


# ─── Aggregation ──────────────────────────────────────────────────────

def _score_records(records: list[dict], parquet_path: Path,
                   sim: SimulatorInterface, box_asset: str,
                   verbose: bool = False) -> list[dict]:
    rows = []
    for r in records:
        row = score_task(r, r["_source_dir"], sim, parquet_path, box_asset)
        rows.append(row)
        if verbose:
            ok = "Y" if row["success"] else "N"
            print(f"    idx={row['idx']:>5}  ok={ok}  "
                  f"goals={row['n_sat']}/{row['n_total']} [{row['goal_tag']}]  "
                  f"inbox={row['inbox_sat']}/{row['inbox_total']} [{row['box_tag']}]  "
                  f"src={row['source']:<20s}  {row['task_id']}")
    return rows


def _print_summary(label: str, rows: list[dict], box_asset: str) -> None:
    print()
    print("=" * 68)
    print(f"SUBGOAL COMPLETION (3-embodiment, +in-box)  —  {label}")
    print("=" * 68)
    n_tasks = len(rows)

    # Existing goals only (== subgoal_completion.py).
    g_sat   = sum(r["n_sat"]   for r in rows)
    g_total = sum(r["n_total"] for r in rows)
    g_rate  = g_sat / g_total if g_total else 0.0

    # In-box subgoal only.
    b_sat   = sum(r["inbox_sat"]   for r in rows)
    b_total = sum(r["inbox_total"] for r in rows)
    b_rate  = b_sat / b_total if b_total else 0.0

    # Combined (existing + in-box) — the recomputed subgoal rate.
    c_sat   = g_sat + b_sat
    c_total = g_total + b_total
    c_rate  = c_sat / c_total if c_total else 0.0

    n_success = sum(1 for r in rows if r["success"])
    success_rate = n_success / n_tasks if n_tasks else 0.0

    print(f"  Tasks:                        {n_tasks}")
    print(f"  Binary task success:          {n_success}/{n_tasks} = {success_rate:.2%}")
    print("-" * 68)
    print(f"  Existing goal subgoals:       {g_sat}/{g_total} = {g_rate:.2%}   "
          f"(== subgoal_completion.py)")
    print(f"  In-box subgoal ('{box_asset}'):  {b_sat}/{b_total} = {b_rate:.2%}   "
          f"(object entered the box at some point)")
    print(f"  COMBINED subgoal rate:        {c_sat}/{c_total} = {c_rate:.2%}   "
          f"(recomputed with +1 in-box subgoal per task)")
    print("-" * 68)

    # Box-entry breakdown.
    box_counts: dict = defaultdict(int)
    for r in rows:
        box_counts[r["box_tag"]] += 1
    print("  in-box detection breakdown:")
    order = ["box-in", "box-out", "box-no-plan", "no-box", "box-load-failed"]
    for tag in order:
        n = box_counts.get(tag, 0)
        if n:
            print(f"    {tag:<18s} {n:>5}")


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Subgoal-completion rate for VIKI-L2 3-robot transport "
                    "tasks, with an extra 'object placed into the box' subgoal")
    p.add_argument("results_dirs", nargs="+",
                   help="leaf result dir(s) or a parent to auto-discover children.")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--box-asset", default="cardboardbox",
                   help="asset name the object must enter to satisfy the extra "
                        "subgoal (default cardboardbox).")
    p.add_argument("--per-dir", action="store_true",
                   help="report one rate per dir instead of merging.")
    p.add_argument("--verbose", action="store_true",
                   help="print per-task decisions.")
    return p.parse_args()


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
            rows = _score_records(records, parquet_path, sim, args.box_asset,
                                  args.verbose)
            _print_summary(d.name, rows, args.box_asset)
    else:
        by_idx: dict[int, dict] = {}
        n_dup = 0
        for d in leaf_dirs:
            for r in _load_records_from(d):
                if r["idx"] in by_idx:
                    n_dup += 1
                by_idx[r["idx"]] = r              # later dir wins
        if n_dup:
            print(f"  [WARN] {n_dup} duplicate idx across dirs — kept the "
                  f"later dir's record for each.")
        records = sorted(by_idx.values(), key=lambda x: x["idx"])
        print(f"\nProcessing {len(records)} unique tasks "
              f"merged from {len(leaf_dirs)} dir(s)...")
        rows = _score_records(records, parquet_path, sim, args.box_asset,
                              args.verbose)
        _print_summary(f"MERGED ({len(leaf_dirs)} dirs, {len(records)} tasks)",
                       rows, args.box_asset)


if __name__ == "__main__":
    main()
