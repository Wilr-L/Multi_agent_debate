"""
vlm_call_count
==============
Count the average number of VLM calls per task across one or more
evaluation runs of the SAME baseline. Counts ACTUAL log files
(task_xxxxx_*/NNN_vlm*.txt under each leaf dir) — that's the ground
truth of how many times the VLM was actually called.

Why not sum debate_rounds_per_loop from results.json?
  - The semantics drift across baselines:
    * multi-agent debate: rounds count Phase-2 turns only; Phase 1 (3
      calls: 2 proposals + 1 merge) and Phase 4 reflections (2 calls
      per retry) are NOT in that field.
    * SC: stored as [1]*N (one call per sample).
    * SR / SR-C: [1] for the initial + [2]*k for each refine iter.
    * PC / VD: same scheme as SR-C / debate respectively.
  - The on-disk file count IS the call count by construction (RunLogger
    writes exactly one NNN_vlm*.txt per query). Use that.

Multi-dir + dedup-by-idx (later dir wins) mirrors aggregate_metrics.py
and subgoal_completion.py. Meant for ONE baseline at a time — pass
folders that all belong to the same method.

Usage
-----
  python vlm_call_count.py evaluate_results
  python vlm_call_count.py evaluate_results --per-dir --verbose

  # resumed baseline split across 3 dirs:
  python vlm_call_count.py \\
      /scratch/.../debate_20260530_run1 \\
      /scratch/.../debate_20260530_run2 \\
      /scratch/.../debate_20260530_run3
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from aggregate_metrics import discover_result_dirs


# ─── Per-task call counting ───────────────────────────────────────────

def find_task_dir(source_dir: Path, idx: int) -> Optional[Path]:
    """Match task_{idx:05d}_* under source_dir."""
    hits = sorted(source_dir.glob(f"task_{idx:05d}_*"))
    return hits[0] if hits else None


def count_vlm_calls(task_dir: Path) -> int:
    """One NNN_vlm*.txt log file == one VLM query. README.txt and other
    non-numbered files are filtered out by the leading-digit glob."""
    return len(list(task_dir.glob("[0-9]*_vlm*.txt")))


# ─── Aggregation ──────────────────────────────────────────────────────

def _load_records_from(leaf_dir: Path) -> list[dict]:
    """Load results.json, tag each record with `_source_dir` so the
    scorer knows which dir to glob for its task_* folder."""
    rj = leaf_dir / "results.json"
    if not rj.is_file():
        print(f"[WARN] {leaf_dir}/results.json missing — skipping",
              file=sys.stderr)
        return []
    recs = json.loads(rj.read_text(encoding="utf-8"))
    return [{**r, "_source_dir": leaf_dir}
            for r in recs if "success" in r]


def _score_records(records: list[dict], verbose: bool = False
                   ) -> tuple[list[dict], dict[str, int]]:
    """For each record, count its on-disk VLM call logs. Returns the
    per-task list (idx, success, n_calls) and a small tag-count dict
    for tasks we had to skip."""
    per_task: list[dict] = []
    tag_counts: dict[str, int] = defaultdict(int)

    for r in records:
        idx     = r["idx"]
        src     = r["_source_dir"]
        success = bool(r.get("success"))

        tdir = find_task_dir(src, idx)
        if tdir is None:
            tag_counts["no-task-dir"] += 1
            if verbose:
                print(f"    idx={idx:>5}  ok={'Y' if success else 'N'}  "
                      f"calls= N/A  [no-task-dir]  {r.get('task_id', '')}")
            continue

        n_calls = count_vlm_calls(tdir)
        per_task.append({"idx": idx, "success": success,
                         "n_calls": n_calls})
        if n_calls == 0:
            tag_counts["zero-calls"] += 1
        else:
            tag_counts["counted"] += 1

        if verbose:
            src_label = src.name[:24]
            print(f"    idx={idx:>5}  ok={'Y' if success else 'N'}  "
                  f"calls={n_calls:>4}  src={src_label:<24s}  "
                  f"{r.get('task_id', '')}")

    return per_task, dict(tag_counts)


# ─── Summary printing ─────────────────────────────────────────────────

def _stats_line(label: str, vals: list[int]) -> str:
    if not vals:
        return f"  {label:<22s} (no data)"
    return (f"  {label:<22s} "
            f"avg={statistics.mean(vals):6.2f}   "
            f"med={statistics.median(vals):>4}   "
            f"min/max={min(vals)}/{max(vals)}   "
            f"(n={len(vals)})")


def _print_summary(label: str, per_task: list[dict],
                   tag_counts: dict[str, int]) -> None:
    print()
    print("=" * 64)
    print(f"VLM CALL COUNT  —  {label}")
    print("=" * 64)
    if not per_task:
        print("  No tasks to analyze.")
        if tag_counts.get("no-task-dir"):
            print(f"  ({tag_counts['no-task-dir']} record(s) had no matching "
                  f"task_* folder on disk.)")
        return

    all_calls = [t["n_calls"] for t in per_task]
    print(_stats_line("calls per task", all_calls))
    print(f"  total VLM calls       : {sum(all_calls)}")
    print()

    succ = [t["n_calls"] for t in per_task if t["success"]]
    fail = [t["n_calls"] for t in per_task if not t["success"]]
    print(_stats_line("successful tasks", succ))
    print(_stats_line("failed tasks",     fail))

    if tag_counts.get("no-task-dir"):
        print()
        print(f"  [WARN] {tag_counts['no-task-dir']} record(s) had no "
              f"matching task_* folder — skipped.")
    if tag_counts.get("zero-calls"):
        print(f"  [WARN] {tag_counts['zero-calls']} task folder(s) "
              f"contained zero call logs (counted as 0 calls).")


# ─── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Average VLM-call count per task over one or more "
                    "VIKI-L2 evaluation dirs (one baseline at a time).")
    p.add_argument("results_dirs", nargs="+",
                   help="leaf result dir(s) or a parent like "
                        "`evaluate_results` to auto-discover children.")
    p.add_argument("--per-dir", action="store_true",
                   help="report one summary per dir instead of merging.")
    p.add_argument("--verbose", action="store_true",
                   help="print a per-task call-count line.")
    return p.parse_args()


def main():
    args = parse_args()

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
            print(f"\nProcessing {d.name}...")
            records = _load_records_from(d)
            per_task, tags = _score_records(records, args.verbose)
            _print_summary(d.name, per_task, tags)
    else:
        # Dedup by idx — later dir wins. Matches aggregate_metrics convention.
        by_idx: dict[int, dict] = {}
        n_dup = 0
        for d in leaf_dirs:
            for r in _load_records_from(d):
                if r["idx"] in by_idx:
                    n_dup += 1
                by_idx[r["idx"]] = r
        if n_dup:
            print(f"  [WARN] {n_dup} duplicate idx across dirs — kept "
                  f"the later dir's record for each.")
        records = sorted(by_idx.values(), key=lambda x: x["idx"])
        print(f"\nProcessing {len(records)} unique tasks "
              f"merged from {len(leaf_dirs)} dir(s)...")
        per_task, tags = _score_records(records, args.verbose)
        _print_summary(f"MERGED ({len(leaf_dirs)} dirs, {len(records)} tasks)",
                       per_task, tags)


if __name__ == "__main__":
    main()
