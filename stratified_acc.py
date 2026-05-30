"""
stratified_acc
==============
Compute success rate (accuracy) STRATIFIED by task-complexity dimensions:

  - n_steps       : length of ground-truth `time_steps` (the reference plan)
  - n_robots      : number of active robots in ground-truth `robots`
  - n_constraints : number of atomic goal + temporal predicates

Each dimension produces its own bucketed table; one TOTAL row at the
bottom of each. Tables are sorted by bucket key ascending so harder
buckets sit at the bottom.

Multi-dir support
-----------------
Like `aggregate_metrics.py`, each input path is either a LEAF results
dir (containing results.json + task_* folders) or a PARENT (e.g.
`evaluate_results`) auto-expanded to every subdir with a results.json.
Records from multiple dirs are MERGED into one big set by default; pass
`--per-dir` to print one table per dir instead.

Note on constraint counting
---------------------------
`goal_constraints` is an ndarray of ndarrays-of-dicts → atomic count =
sum(len(g) for g in goal_constraints). `temporal_constraints` is a
3-level nesting (wrapper-of-steps-of-predicates), so atomic count =
sum(len(t) for t in temporal_constraints[0]) (if non-empty). Total
constraint count = goal + temporal. See `count_constraints()` below.

Usage
-----
  python stratified_acc.py evaluate_results
  python stratified_acc.py evaluate_results/local_20260524_020504 --verbose
  python stratified_acc.py dirA dirB dirC --per-dir
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Reuse aggregate_metrics' dir discovery so behavior matches that script.
from aggregate_metrics import discover_result_dirs


# ─── Ground-truth complexity extraction ──────────────────────────────

def count_constraints(gt: dict) -> tuple[int, int]:
    """Return (n_goal, n_temporal) atomic predicate counts for a row's
    ground-truth dict. Tolerant of missing / empty / wrapped fields.

    NOTE: must NOT use `gt.get("x", []) or []` here — `or` triggers
    bool(array) on a non-empty ndarray and crashes with "truth value
    of an array with more than one element is ambiguous". Use explicit
    `is None` checks instead."""
    gc = gt.get("goal_constraints")
    if gc is None:
        gc = []
    tc = gt.get("temporal_constraints")
    if tc is None:
        tc = []
    try:
        n_goal = sum(len(g) for g in gc) if len(gc) > 0 else 0
    except TypeError:
        n_goal = 0
    try:
        # `temporal_constraints` is wrapped in an outer length-1 array
        # (or length-0 if no temporal constraints). The actual list of
        # "temporal steps" sits at index [0], and each step is an
        # ndarray of dicts. We flatten one level to count predicates.
        if len(tc) > 0 and hasattr(tc[0], "__len__"):
            n_temp = sum(len(t) for t in tc[0])
        else:
            n_temp = 0
    except TypeError:
        n_temp = 0
    return n_goal, n_temp


def load_gt_complexity(parquet_path: Path,
                       indices: list[int]) -> dict[int, Optional[dict]]:
    """Map row idx → {n_steps, n_robots, n_goal, n_temporal, n_constraints}.
    None for malformed rows."""
    import pandas as pd
    df = pd.read_parquet(str(parquet_path))
    out: dict[int, Optional[dict]] = {}
    for idx in indices:
        try:
            gt = df.iloc[idx]["reward_model"]["ground_truth"]
            ts = gt.get("time_steps")
            robots = gt.get("robots")
            if robots is None:
                robots = {}
            n_goal, n_temp = count_constraints(gt)
            out[idx] = {
                "n_steps":       len(ts) if ts is not None and hasattr(ts, "__len__") else None,
                "n_robots":      sum(1 for v in robots.values() if v is not None),
                "n_goal":        n_goal,
                "n_temporal":    n_temp,
                "n_constraints": n_goal + n_temp,
            }
        except Exception as e:
            print(f"[WARN] idx={idx} GT load failed: {e}", file=sys.stderr)
            out[idx] = None
    return out


# ─── Stratification + printing ────────────────────────────────────────

def stratify(rows: list[dict], dim_name: str) -> dict[int, list[dict]]:
    """Bucket records by row[dim_name]. Skip rows missing that dim."""
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        v = r.get(dim_name)
        if v is None:
            continue
        out[v].append(r)
    return out


def print_strat_table(title: str, buckets: dict[int, list[dict]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not buckets:
        print("  (no data)")
        return
    print(f"  {'bucket':>7}  {'n':>5}  {'succ':>5}  {'rate':>7}")
    total_n = total_succ = 0
    for k in sorted(buckets):
        recs   = buckets[k]
        n      = len(recs)
        n_succ = sum(1 for r in recs if r["success"])
        rate   = n_succ / n if n else 0.0
        print(f"  {k:>7}  {n:>5}  {n_succ:>5}  {rate:>6.1%}")
        total_n    += n
        total_succ += n_succ
    overall = total_succ / total_n if total_n else 0.0
    print(f"  {'TOTAL':>7}  {total_n:>5}  {total_succ:>5}  {overall:>6.1%}")


def print_per_task_table(rows: list[dict]) -> None:
    print(f"\n  {'idx':>5}  {'src':>14}  {'ok':>2}  "
          f"{'steps':>5}  {'rob':>3}  {'cons':>4}  "
          f"{'goal':>4}  {'temp':>4}  task_id")
    for r in sorted(rows, key=lambda x: x["idx"]):
        src = r.get("source_dir", "")[:14]
        ok  = "Y" if r["success"] else "N"
        print(f"  {r['idx']:>5}  {src:>14}  {ok:>2}  "
              f"{str(r.get('n_steps')):>5}  {str(r.get('n_robots')):>3}  "
              f"{str(r.get('n_constraints')):>4}  "
              f"{str(r.get('n_goal')):>4}  {str(r.get('n_temporal')):>4}  "
              f"{r.get('task_id', '')}")


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stratify VIKI-L2 eval accuracy by task complexity")
    p.add_argument("results_dirs", nargs="+",
                   help="leaf results dir(s) or a parent like "
                        "`evaluate_results` to auto-discover children.")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--per-dir", action="store_true",
                   help="print one set of stratification tables PER dir "
                        "instead of merging all dirs.")
    p.add_argument("--by", choices=["steps", "robots", "constraints",
                                    "goal", "temporal", "all"],
                   default="all",
                   help="which complexity dimension to stratify by "
                        "(default: all of steps/robots/constraints).")
    p.add_argument("--verbose", action="store_true",
                   help="print a per-task table before the strat tables.")
    return p.parse_args()


def _load_records_from(leaf_dir: Path) -> list[dict]:
    """Load results.json from a leaf dir; tag each record with source_dir.
    Skip records lacking 'success' (load/runtime errors)."""
    rj = leaf_dir / "results.json"
    if not rj.is_file():
        print(f"[WARN] {leaf_dir}/results.json missing — skipping that dir",
              file=sys.stderr)
        return []
    recs = json.loads(rj.read_text(encoding="utf-8"))
    out = []
    for r in recs:
        if "success" not in r:
            continue
        out.append({**r, "source_dir": leaf_dir.name})
    return out


def _attach_complexity(records: list[dict], parquet_path: Path) -> None:
    """In-place: add n_steps / n_robots / n_constraints / n_goal / n_temporal
    to each record using one parquet read for all unique idxs."""
    idxs = sorted({r["idx"] for r in records})
    complexity = load_gt_complexity(parquet_path, idxs)
    for r in records:
        c = complexity.get(r["idx"])
        if c is None:
            continue
        r.update(c)


def _print_strat_block(label: str, rows: list[dict], which: str) -> None:
    print()
    print("=" * 70)
    print(f"{label}   (n={len(rows)} tasks)")
    print("=" * 70)
    if which in ("steps", "all"):
        print_strat_table("Accuracy by # plan steps (gt time_steps)",
                          stratify(rows, "n_steps"))
    if which in ("robots", "all"):
        print_strat_table("Accuracy by # active robots",
                          stratify(rows, "n_robots"))
    if which in ("constraints", "all"):
        print_strat_table("Accuracy by # total constraints (goal + temporal)",
                          stratify(rows, "n_constraints"))
    if which in ("goal", "all"):
        if which == "goal":
            print_strat_table("Accuracy by # goal constraints",
                              stratify(rows, "n_goal"))
    if which in ("temporal", "all"):
        if which == "temporal":
            print_strat_table("Accuracy by # temporal constraints",
                              stratify(rows, "n_temporal"))


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

    if args.per_dir:
        for d in leaf_dirs:
            records = _load_records_from(d)
            if not records:
                continue
            _attach_complexity(records, parquet_path)
            if args.verbose:
                print_per_task_table(records)
            _print_strat_block(d.name, records, args.by)
    else:
        all_records: list[dict] = []
        for d in leaf_dirs:
            all_records.extend(_load_records_from(d))
        if not all_records:
            print("No records to analyze.")
            return
        # Dedup by idx (last seen wins — matches aggregate_metrics).
        by_idx: dict[int, dict] = {}
        n_dup = 0
        for r in all_records:
            if r["idx"] in by_idx:
                n_dup += 1
            by_idx[r["idx"]] = r
        if n_dup:
            print(f"  [WARN] {n_dup} duplicate idx across dirs — kept later.")
        records = sorted(by_idx.values(), key=lambda x: x["idx"])
        _attach_complexity(records, parquet_path)
        if args.verbose:
            print_per_task_table(records)
        _print_strat_block(f"MERGED ({len(leaf_dirs)} dirs)",
                           records, args.by)


if __name__ == "__main__":
    main()
