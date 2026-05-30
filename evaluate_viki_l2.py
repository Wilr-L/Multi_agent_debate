"""
evaluate_viki_l2
================
Run the multi-agent-debate pipeline on every 2-robot (R1+R2) task in
`VIKI_data/viki/VIKI-L2/test.parquet` and aggregate:

  - task success rate
  - max / min / avg of debate-loop count per task
        (= number of Phase-2 invocations, i.e. 1 + actual retry count)
  - max / min / avg of debate-round count per Phase-2 invocation

TensorBoard scalars are written live after each task. Launch in another
terminal:
    tensorboard --logdir tb_logs

Example:
    $env:APIMART_API_KEY = "sk-..."
    E:\\anaconda3\\python.exe evaluate_viki_l2.py --limit 20
"""

import argparse
import json
import os
import sys
import time
import statistics
import traceback
from pathlib import Path

# TensorBoard writer — prefer tensorboardX (no torch dep), fall back to torch.
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        SummaryWriter = None  # type: ignore[assignment]

from viki_loader import load_viki_task, find_task_indices
from multi_agent_debate import (
    MultiAgentDebateEngine, VLMInterface, SimulatorInterface,
    DebateRole, RobotProfile, RunLogger,
)


# ─── CLI ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VIKI-L2 2-robot tasks")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
                   help="path to the VIKI-L2 parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N 2-robot tasks (default: all)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first OFFSET tasks (default: 0)")
    p.add_argument("--max-debate-rounds", type=int, default=3,
                   help="max rounds per Phase-2 invocation (default 3)")
    p.add_argument("--max-retry-rounds", type=int, default=2,
                   help="max retries on execution failure (default 2)")
    p.add_argument("--log-dir", default=None,
                   help="root dir for per-task VLM-call logs; "
                        "defaults to logs/eval_<timestamp>")
    p.add_argument("--tb-dir", default=None,
                   help="TensorBoard log dir; defaults to tb_logs/eval_<timestamp>")
    p.add_argument("--results-json", default=None,
                   help="path to dump per-task results JSON "
                        "(defaults to <log-dir>/results.json)")
    p.add_argument("--no-vlm-log", action="store_true",
                   help="disable per-task VLM call logging entirely")
    p.add_argument("--continue-on-error", action="store_true",
                   help="if a task crashes, log + continue instead of aborting")
    return p.parse_args()


# ─── Stat helpers ───────────────────────────────────────────────────

def fmt_stats(label: str, values: list) -> str:
    if not values:
        return f"  {label:<30s} (no data)"
    return (
        f"  {label:<30s} "
        f"min={min(values)}  avg={statistics.mean(values):6.2f}  max={max(values)}  "
        f"(n={len(values)})"
    )


# ─── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.environ.get("APIMART_API_KEY"):
        print("[ERROR] APIMART_API_KEY is not set.", file=sys.stderr)
        print("        On PowerShell:  $env:APIMART_API_KEY = 'sk-...'", file=sys.stderr)
        sys.exit(1)

    # ── Resolve paths ──
    root = Path(__file__).resolve().parent
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = root / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] Parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_dir) if args.log_dir else (root / "logs" / f"eval_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (root / "tb_logs" / f"eval_{stamp}")

    # ── Find tasks ──
    print(f"Scanning parquet for 2-robot (R1+R2) tasks: {parquet_path}")
    all_indices = find_task_indices(parquet_path, n_robots=2, required_ids=("R1", "R2"))
    indices = all_indices[args.offset:]
    if args.limit is not None:
        indices = indices[:args.limit]
    print(f"  → {len(all_indices)} total 2-robot tasks; evaluating {len(indices)} "
          f"(offset={args.offset}, limit={args.limit})")

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is None:
        print("[WARN] No TensorBoard writer available (install `tensorboardX` "
              "or `tensorboard`+`torch`). Skipping TB logging.")
    else:
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard logs    → {tb_dir}")
        print(f"  view:  tensorboard --logdir {tb_dir.parent}")

    print(f"Per-task logs       → {log_root}")
    results_json = (Path(args.results_json) if args.results_json
                    else log_root / "results.json")
    print(f"Per-task JSON       → {results_json}")
    print()

    # ── Evaluate ──
    results: list[dict] = []
    n_success = 0
    t0 = time.time()

    for i, idx in enumerate(indices):
        step = i + 1
        print(f"\n{'='*70}\n[{step}/{len(indices)}] task #{idx}\n{'='*70}")

        # Load task
        try:
            task = load_viki_task(parquet_path, idx)
        except Exception as e:
            print(f"  [SKIP] load failed: {e}")
            results.append({"idx": idx, "error": f"load failed: {e}"})
            continue

        robots = task["scene_config"]["robots"]
        print(f"  task_id={task['task_id']}  robots={robots}")
        print(f"  desc: {task['task_description'][:120]}")

        # Per-task RunLogger
        rlog = None
        if not args.no_vlm_log:
            rlog = RunLogger(log_root / f"task_{idx:05d}_{task['task_id']}")

        # Engine
        robot1 = RobotProfile(name=robots["R1"], robot_id="R1")
        robot2 = RobotProfile(name=robots["R2"], robot_id="R2")
        vlm1 = VLMInterface(role=DebateRole.VLM1_R1_ADVOCATE, logger=rlog)
        vlm2 = VLMInterface(role=DebateRole.VLM2_R2_ADVOCATE, logger=rlog)
        sim  = SimulatorInterface(scene_seed=0)
        eng  = MultiAgentDebateEngine(
            vlm1, vlm2, sim, robot1, robot2,
            max_debate_rounds=args.max_debate_rounds,
            max_retry_rounds=args.max_retry_rounds,
        )

        # Run
        task_t0 = time.time()
        try:
            result = eng.run(
                task["task_description"], task["image_path"], task["scene_config"]
            )
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            raise
        except Exception as e:
            print(f"  [ERROR] task crashed: {e}")
            traceback.print_exc()
            if not args.continue_on_error:
                raise
            results.append({
                "idx": idx, "task_id": task["task_id"],
                "error": f"runtime: {e}",
            })
            continue
        task_elapsed = time.time() - task_t0

        success         = bool(result.get("success"))
        loops           = int(result.get("debate_loop_count", 0))
        rounds_per_loop = list(result.get("debate_rounds_per_loop", []))
        last_exec       = (result.get("execution_results") or [{}])[-1]
        last_failure    = last_exec.get("failure_reason")

        if success:
            n_success += 1

        record = {
            "idx":             idx,
            "task_id":         task["task_id"],
            "task_name":       task["task_name"],
            "robots":          robots,
            "success":         success,
            "debate_loops":    loops,
            "debate_rounds_per_loop": rounds_per_loop,
            "elapsed_seconds": round(task_elapsed, 2),
            "last_failure":    last_failure,
        }
        results.append(record)

        # ── Stats so far ──
        succ_records = [r for r in results if "success" in r]
        n_done       = len(succ_records)
        success_rate = n_success / n_done if n_done else 0.0
        loops_list   = [r["debate_loops"] for r in succ_records]
        all_rounds   = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]
        avg_loops    = statistics.mean(loops_list)  if loops_list else 0.0
        avg_rounds   = statistics.mean(all_rounds) if all_rounds else 0.0

        print(f"  → success={success}  loops={loops}  rounds={rounds_per_loop}  "
              f"({task_elapsed:.1f}s)")
        if last_failure and not success:
            print(f"  → failure: {last_failure.splitlines()[0]}")
        print(f"  running: success={n_success}/{n_done} ({success_rate:.1%})  "
              f"avg_loops={avg_loops:.2f}  avg_rounds={avg_rounds:.2f}")

        # ── TensorBoard ──
        if writer is not None:
            writer.add_scalar("rate/success",          success_rate,     step)
            writer.add_scalar("rate/success_this",     int(success),     step)
            writer.add_scalar("loops/this_task",       loops,            step)
            writer.add_scalar("loops/avg_so_far",      avg_loops,        step)
            writer.add_scalar("rounds/avg_so_far",     avg_rounds,       step)
            writer.add_scalar("timing/seconds_per_task", task_elapsed,   step)
            if rounds_per_loop:
                writer.add_scalar("rounds/last_loop",  rounds_per_loop[-1], step)
            writer.flush()

        # Dump results.json after every task so a crash doesn't lose data.
        results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    total_elapsed = time.time() - t0

    # ── Final report ──
    print()
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    print(f"Tasks evaluated:    {n_done}")
    print(f"Tasks succeeded:    {n_success}")
    print(f"Success rate:       {n_success}/{n_done} = {n_success/n_done:.2%}")
    print()
    print(fmt_stats("debate loops per task",   loops_list))
    print(fmt_stats("debate rounds per loop",  all_rounds))
    print()
    print(f"Total wall time:    {total_elapsed:.1f}s "
          f"({total_elapsed/n_done:.1f}s per task)")
    print()
    print(f"Per-task records  : {results_json}")
    if writer is not None:
        print(f"TensorBoard logs  : {tb_dir}")
        print(f"  view:  tensorboard --logdir {tb_dir.parent}")
        writer.close()


if __name__ == "__main__":
    main()
