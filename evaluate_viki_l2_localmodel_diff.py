"""
evaluate_viki_l2_localmodel_diff
================================
Same multi-agent debate pipeline as `evaluate_viki_l2_localmodel.py`,
but VLM1 and VLM2 use **DIFFERENT** local models (e.g. a 32B agent
debating a 72B agent, or two same-size but different-checkpoint models).

GPU layout
----------
vLLM cannot retarget GPUs after process start (it reads CUDA_VISIBLE_DEVICES
once at import). So both `LLM` instances share THE SAME GPU set, each
claiming `--gpu-mem-util` of every GPU. The two fractions must sum below
~0.9 to leave room for activations and CUDA context — that's why the
default is 0.45 here (was 0.90 for the single-model script).

Rough memory budget (per GPU, bf16):
  - 2× 7B on 1 GPU 80GB    → 14 + 14 = 28 GB weights, gpu_mem_util 0.45 each
  - 2× 32B with TP=2 on 2 GPUs 80GB → 16 + 16 GB weights/GPU, tight
  - 32B + 72B with TP=4 on 4 GPUs 80GB → 8 + 18 GB weights/GPU, doable
  - For tighter setups switch to --quantization int8 for both models.

If you need each model on its OWN GPU subset, run two separate vLLM
servers (one per model) and have a thin proxy script hit them via HTTP —
that's out of scope for this script.

Results.json schema matches evaluate_viki_l2_localmodel.py exactly so
aggregate_metrics.py works unchanged; an extra pair of fields
`model_path_1` / `model_path_2` is recorded per task for analysis.

Example:
  python evaluate_viki_l2_localmodel_diff.py \\
      --model-path-1 /path/to/Qwen2.5-VL-32B-Instruct \\
      --model-path-2 /path/to/Qwen2.5-VL-72B-Instruct \\
      --tensor-parallel-size 4 --max-model-len 8192 --gpu-mem-util 0.45 \\
      --enforce-eager --max-num-seqs 4 \\
      --limit 40 --trust-remote-code
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, TYPE_CHECKING

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        SummaryWriter = None  # type: ignore[assignment]

from viki_loader import load_viki_task, find_task_indices
from multi_agent_debate import (
    MultiAgentDebateEngine, SimulatorInterface,
    DebateRole, RobotProfile, RunLogger,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface, TaskStats

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# ─── vLLM loader (one per model, shared kwargs) ───────────────────────

def _resolve_quant(args):
    """Map our friendly --quantization name to vLLM's (quant, load_format)."""
    quant       = args.quantization
    load_format = args.load_format
    if quant in (None, "bf16"):
        quant = None
    elif quant == "int8":
        os.environ.setdefault("BNB_QUANTIZATION", "int8")
        quant = "bitsandbytes"
        if load_format == "auto":
            load_format = "bitsandbytes"
    elif quant == "int4":
        os.environ.setdefault("BNB_QUANTIZATION", "nf4")
        quant = "bitsandbytes"
        if load_format == "auto":
            load_format = "bitsandbytes"
    return quant, load_format


def _load_one_llm(model_path: str, args, label: str, gpu_mem_util: float):
    """Construct one vLLM LLM with the shared inference config and a
    per-model GPU memory fraction (which is where the symmetric default
    breaks down for asymmetric pairs like 7B + 72B)."""
    try:
        from vllm import LLM
    except ImportError as e:
        print(f"[ERROR] vLLM not installed: {e}\n"
              f"        Install it on Linux/WSL: pip install vllm",
              file=sys.stderr)
        sys.exit(1)

    quant, load_format = _resolve_quant(args)

    print(f"Loading {label}: {model_path}")
    print(f"  max_model_len={args.max_model_len}  "
          f"gpu_mem_util={gpu_mem_util}  "
          f"dtype={args.dtype}  tp={args.tensor_parallel_size}  "
          f"quantization={args.quantization or 'bf16'}  "
          f"load_format={load_format}")

    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
    )
    if quant is not None:
        llm_kwargs["quantization"] = quant
    if load_format != "auto":
        llm_kwargs["load_format"] = load_format
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    print(f"  {label} loaded in {time.time() - t0:.1f}s\n")
    return llm


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VIKI-L2 multi-agent debate with TWO DIFFERENT local "
                    "VLM models (one per debater)")

    # Two model paths — the whole point of this script
    p.add_argument("--model-path-1", required=True,
                   help="local path or HF hub id for VLM1 (R1-advocate).")
    p.add_argument("--model-path-2", required=True,
                   help="local path or HF hub id for VLM2 (R2-advocate).")

    # vLLM args — applied to BOTH model loads (must be compatible with each)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.45,
                   help="DEFAULT per-model GPU memory fraction, used for BOTH "
                        "models unless --gpu-mem-util-1 / --gpu-mem-util-2 "
                        "override. The two models share the same GPUs, so "
                        "this value is claimed by EACH model — total = 2 × "
                        "this. Default 0.45 (symmetric pair fits under 0.90).")
    p.add_argument("--gpu-mem-util-1", type=float, default=None,
                   help="override --gpu-mem-util for MODEL 1 (VLM1). Use this "
                        "for asymmetric pairs — e.g. 7B + 72B on 4×80GB at "
                        "TP=4 needs ~0.12 for 7B and ~0.60 for 72B.")
    p.add_argument("--gpu-mem-util-2", type=float, default=None,
                   help="override --gpu-mem-util for MODEL 2 (VLM2).")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--quantization", default=None,
                   choices=[None, "bf16", "int8", "int4", "fp8",
                            "awq", "gptq", "bitsandbytes"])
    p.add_argument("--load-format", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=None)

    # Sampling — shared. Both models get the same SamplingParams. If you
    # really need different temperatures per debater, edit main().
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)

    # Eval set
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)

    # Debate
    p.add_argument("--max-debate-rounds", type=int, default=3)
    p.add_argument("--max-retry-rounds",  type=int, default=2)

    # Logging
    p.add_argument("--log-dir", default=None,
                   help="defaults to logs/diff_<timestamp>")
    p.add_argument("--tb-dir",  default=None,
                   help="defaults to tb_logs/diff_<timestamp>")
    p.add_argument("--results-json", default=None)
    p.add_argument("--no-vlm-log", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    return p.parse_args()


# ─── Per-task stats helper (same as evaluate_viki_l2_localmodel.py) ──

def _print_task_stats_summary(args, task_stats: TaskStats):
    if not task_stats.call_count:
        return
    max_out   = task_stats.max_completion_tokens
    max_in    = task_stats.max_prompt_tokens
    max_total = task_stats.max_total_tokens
    out_pct   = max_out   / args.max_tokens    * 100 if args.max_tokens    else 0.0
    in_pct    = max_in    / args.max_model_len * 100 if args.max_model_len else 0.0
    total_pct = max_total / args.max_model_len * 100 if args.max_model_len else 0.0
    remaining = (args.max_model_len - max_in) if max_in else 0

    print(f"  → calls={task_stats.call_count}  "
          f"avg_call={task_stats.avg_seconds:.2f}s")
    print(f"  → output: max_completion={max_out}/{int(args.max_tokens)} "
          f"({out_pct:.0f}% of --max-tokens)")
    if max_in:
        print(f"  → input : max_prompt={max_in}/{int(args.max_model_len)} "
              f"({in_pct:.0f}%)   max_total={max_total}/{int(args.max_model_len)} "
              f"({total_pct:.0f}%)   gen_budget_left={remaining}")
    if task_stats.truncated_count:
        print(f"  [WARN] {task_stats.truncated_count}/{task_stats.call_count} "
              f"calls hit max_tokens={int(args.max_tokens)} — bump --max-tokens")
    elif out_pct >= 90:
        print(f"  [WARN] output close to --max-tokens cap ({out_pct:.0f}%)")
    if max_in and remaining < args.max_tokens:
        print(f"  [WARN] prompt grew to {max_in} tokens, leaving only "
              f"{remaining} tokens of generation budget "
              f"(< --max-tokens={int(args.max_tokens)}).")
    elif in_pct >= 80:
        print(f"  [WARN] prompt reached {in_pct:.0f}% of --max-model-len")


def fmt_stats(label, values):
    if not values:
        return f"  {label:<30s} (no data)"
    return (f"  {label:<30s} "
            f"min={min(values)}  avg={statistics.mean(values):6.2f}  max={max(values)}  "
            f"(n={len(values)})")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Lazy SamplingParams import (vLLM optional at --help time) ──
    try:
        from vllm import SamplingParams
    except ImportError as e:
        print(f"[ERROR] vLLM not installed: {e}", file=sys.stderr)
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
    log_root = (Path(args.log_dir) if args.log_dir
                else root / "logs" / f"diff_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = (Path(args.tb_dir) if args.tb_dir
              else root / "tb_logs" / f"diff_{stamp}")

    # ── Find tasks ──
    print(f"Scanning parquet for 2-robot (R1+R2) tasks: {parquet_path}")
    all_indices = find_task_indices(parquet_path, n_robots=2,
                                    required_ids=("R1", "R2"))
    indices = all_indices[args.offset:]
    if args.limit is not None:
        indices = indices[:args.limit]
    print(f"  → {len(all_indices)} total 2-robot tasks; evaluating "
          f"{len(indices)} (offset={args.offset}, limit={args.limit})")

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is None:
        print("[WARN] No TensorBoard writer — skipping TB logging.")
    else:
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard logs    → {tb_dir}")
        print(f"  view:  tensorboard --logdir {tb_dir.parent}")

    print(f"Per-task logs       → {log_root}")
    results_json = (Path(args.results_json) if args.results_json
                    else log_root / "results.json")
    print(f"Per-task JSON       → {results_json}")
    print()

    # ── Load BOTH models ──
    # Shared GPU set; each claims --gpu-mem-util. Same TP for both. If the
    # models have different tensor-parallel compatibility (head counts not
    # both divisible by TP), pick a TP that works for both — or run two
    # separate vLLM servers off-script.
    # Per-model GPU memory shares (each defaults to --gpu-mem-util).
    mem1 = args.gpu_mem_util_1 if args.gpu_mem_util_1 is not None else args.gpu_mem_util
    mem2 = args.gpu_mem_util_2 if args.gpu_mem_util_2 is not None else args.gpu_mem_util

    print("=" * 70)
    print(f"Loading 2 models (they will share the same GPU set)")
    print(f"  per-model GPU memory: model1={mem1:.2f}  model2={mem2:.2f}  "
          f"total={mem1 + mem2:.2f}")
    print("=" * 70)
    if mem1 + mem2 > 0.95:
        print(f"[WARN] mem1 + mem2 = {mem1 + mem2:.2f} > 0.95 — likely to "
              f"OOM loading the second model.")

    llm1 = _load_one_llm(args.model_path_1, args,
                         label="model 1 (VLM1, R1)", gpu_mem_util=mem1)
    llm2 = _load_one_llm(args.model_path_2, args,
                         label="model 2 (VLM2, R2)", gpu_mem_util=mem2)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=int(args.max_tokens),
    )

    # ── Evaluate ──
    results: list[dict] = []
    n_success = 0
    t0 = time.time()

    for i, idx in enumerate(indices):
        step = i + 1
        print(f"\n{'=' * 70}\n[{step}/{len(indices)}] task #{idx}\n{'=' * 70}")

        try:
            task = load_viki_task(parquet_path, idx)
        except Exception as e:
            print(f"  [SKIP] load failed: {e}")
            results.append({"idx": idx, "error": f"load failed: {e}"})
            continue

        robots = task["scene_config"]["robots"]
        print(f"  task_id={task['task_id']}  robots={robots}")
        print(f"  desc: {task['task_description'][:120]}")

        rlog = None
        if not args.no_vlm_log:
            rlog = RunLogger(log_root / f"task_{idx:05d}_{task['task_id']}")

        robot1 = RobotProfile(name=robots["R1"], robot_id="R1")
        robot2 = RobotProfile(name=robots["R2"], robot_id="R2")
        task_stats = TaskStats(max_tokens=int(args.max_tokens),
                               max_model_len=int(args.max_model_len))

        # ── KEY DIFFERENCE FROM evaluate_viki_l2_localmodel.py ──
        # VLM1 wraps llm1 (model 1), VLM2 wraps llm2 (model 2). They
        # also pass model_name distinctly so the RunLogger header shows
        # which model produced each call.
        vlm1 = LocalVLMInterface(
            llm=llm1, sampling_params=sampling_params,
            model_name=args.model_path_1,
            role=DebateRole.VLM1_R1_ADVOCATE, logger=rlog, stats=task_stats,
        )
        vlm2 = LocalVLMInterface(
            llm=llm2, sampling_params=sampling_params,
            model_name=args.model_path_2,
            role=DebateRole.VLM2_R2_ADVOCATE, logger=rlog, stats=task_stats,
        )
        sim = SimulatorInterface(scene_seed=0)
        eng = MultiAgentDebateEngine(
            vlm1, vlm2, sim, robot1, robot2,
            max_debate_rounds=args.max_debate_rounds,
            max_retry_rounds=args.max_retry_rounds,
        )

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
            # ── extra fields specific to the diff-model setup ──
            "model_path_1":    args.model_path_1,
            "model_path_2":    args.model_path_2,
            "success":         success,
            "debate_loops":    loops,
            "debate_rounds_per_loop": rounds_per_loop,
            "elapsed_seconds": round(task_elapsed, 2),
            "last_failure":    last_failure,
        }
        results.append(record)

        # ── Running aggregates ──
        succ_records = [r for r in results if "success" in r]
        n_done       = len(succ_records)
        success_rate = n_success / n_done if n_done else 0.0
        loops_list   = [r["debate_loops"] for r in succ_records]
        all_rounds   = [r for rec in succ_records
                          for r in rec["debate_rounds_per_loop"]]
        avg_loops    = statistics.mean(loops_list)  if loops_list else 0.0
        avg_rounds   = statistics.mean(all_rounds) if all_rounds else 0.0

        print(f"  → success={success}  loops={loops}  rounds={rounds_per_loop}  "
              f"({task_elapsed:.1f}s)")
        if last_failure and not success:
            print(f"  → failure: {last_failure.splitlines()[0]}")

        _print_task_stats_summary(args, task_stats)

        print(f"  running: success={n_success}/{n_done} ({success_rate:.1%})  "
              f"avg_loops={avg_loops:.2f}  avg_rounds={avg_rounds:.2f}")

        # ── TensorBoard (SAME tag names as the other evaluators) ──
        if writer is not None:
            writer.add_scalar("rate/success",            success_rate,     step)
            writer.add_scalar("rate/success_this",       int(success),     step)
            writer.add_scalar("loops/this_task",         loops,            step)
            writer.add_scalar("loops/avg_so_far",        avg_loops,        step)
            writer.add_scalar("rounds/avg_so_far",       avg_rounds,       step)
            writer.add_scalar("timing/seconds_per_task", task_elapsed,     step)
            if rounds_per_loop:
                writer.add_scalar("rounds/last_loop",    rounds_per_loop[-1], step)
            writer.flush()

        # ── Dump after every task so a crash doesn't lose data ──
        results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    total_elapsed = time.time() - t0

    # ── Final report ──
    print()
    print("=" * 70)
    print("FINAL STATISTICS — multi-agent debate (DIFFERENT models)")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    print(f"Model 1 (VLM1):     {args.model_path_1}")
    print(f"Model 2 (VLM2):     {args.model_path_2}")
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
        writer.close()


if __name__ == "__main__":
    main()
