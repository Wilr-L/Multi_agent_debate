"""
evaluate_viki_l2_localmodel
===========================
Same debate pipeline and reporting as `evaluate_viki_l2.py`, but the two
debaters talk to a LOCAL VLM loaded with vLLM instead of the SiliconFlow
API. A single `vllm.LLM` instance is shared between VLM1 and VLM2 so the
weights are only loaded into GPU memory once.

vLLM is the inference backend. Native Windows support is spotty — run this
on Linux / WSL with a CUDA GPU. vLLM is imported lazily so `--help` still
works on a Windows box without it installed.

Example:
    python evaluate_viki_l2_localmodel.py \\
        --model-path /models/Qwen3-VL-32B-Instruct \\
        --limit 20 \\
        --max-model-len 8192 \\
        --gpu-mem-util 0.9 \\
        --trust-remote-code

    python evaluate_viki_l2_localmodel.py  --model-path /scratch/users/k25159491/WORK/Model/Qwen2.5-VL-3B-Instruct  --limit 20  --max-model-len 8192  --gpu-mem-util 0.9  --trust-remote-code

TensorBoard scalars use the SAME tag names as evaluate_viki_l2.py /
evaluate_viki_l2_baseline.py so all three runs overlay cleanly.
"""

import argparse
import json
import os
import sys
import time
import statistics
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
    MultiAgentDebateEngine, VLMInterface, SimulatorInterface,
    DebateRole, RobotProfile, RunLogger,
)

if TYPE_CHECKING:                           # vLLM only for type hints — never
    from vllm import LLM, SamplingParams    # imported at module top so --help
                                            # works without vLLM installed.


# ─── Per-task stats accumulator ───────────────────────────────────────

class TaskStats:
    """Per-task counters fed by LocalVLMInterface.query. Tracks two
    independent ceilings separately because they have different fixes:

      • Output ceiling : SamplingParams.max_tokens (--max-tokens). Triggers
                         finish_reason == 'length' when hit. Fix: bump
                         --max-tokens (cheap, no model reload).
      • Context ceiling: LLM.max_model_len (--max-model-len), covers
                         input + output. vLLM either raises on request
                         OR silently clips max_tokens to
                         (max_model_len - prompt_tokens). Fix: bump
                         --max-model-len (model reload + more KV cache),
                         or reduce --max-debate-rounds so history stays
                         shorter."""

    def __init__(self, max_tokens: int, max_model_len: int):
        self.max_tokens                  = max_tokens
        self.max_model_len               = max_model_len
        self.call_count                  = 0
        self.total_seconds               = 0.0
        self.truncated_count             = 0
        self.completion_tokens_list: list[int] = []
        self.prompt_tokens_list:     list[int] = []
        self.total_tokens_list:      list[int] = []   # prompt + completion per call

    def record(self, elapsed_s: float, finish_reason: Optional[str],
               completion_tokens: Optional[int],
               prompt_tokens: Optional[int] = None):
        self.call_count    += 1
        self.total_seconds += elapsed_s
        if finish_reason == "length":
            self.truncated_count += 1
        if completion_tokens is not None:
            self.completion_tokens_list.append(completion_tokens)
        if prompt_tokens is not None:
            self.prompt_tokens_list.append(prompt_tokens)
            if completion_tokens is not None:
                self.total_tokens_list.append(prompt_tokens + completion_tokens)

    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / self.call_count if self.call_count else 0.0

    @property
    def max_completion_tokens(self) -> int:
        return max(self.completion_tokens_list) if self.completion_tokens_list else 0

    @property
    def max_prompt_tokens(self) -> int:
        return max(self.prompt_tokens_list) if self.prompt_tokens_list else 0

    @property
    def max_total_tokens(self) -> int:
        return max(self.total_tokens_list) if self.total_tokens_list else 0


# ─── Local VLM adapter ────────────────────────────────────────────────

class LocalVLMInterface:
    """
    vLLM-backed drop-in replacement for `VLMInterface`. Implements the
    duck-typed surface that `MultiAgentDebateEngine` uses:
      • role : DebateRole
      • conversation_history : list of OpenAI-format messages
      • set_system_prompt(text)
      • query(user_prompt, image_path=None) -> str
      • reset_history()

    Both VLM1 and VLM2 share the same `vllm.LLM` instance (one set of
    weights on GPU). They keep independent `conversation_history` so the
    two debaters remember their own turns.
    """

    def __init__(
        self,
        llm: "LLM",
        sampling_params: "SamplingParams",
        model_name: str,
        role: DebateRole = DebateRole.VLM1_R1_ADVOCATE,
        logger: Optional[RunLogger] = None,
        image_detail: str = "auto",
        stats: Optional[TaskStats] = None,
    ):
        self.llm             = llm
        self.sampling_params = sampling_params
        self.model_name      = model_name
        self.role            = role
        self.logger          = logger
        self.image_detail    = image_detail
        self.stats           = stats
        self.conversation_history: list[dict] = []

    def set_system_prompt(self, system_prompt: str):
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def reset_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []

    def _build_user_message(self, prompt: str, image_path: Optional[str]) -> dict:
        if not image_path:
            return {"role": "user", "content": prompt}
        # Reuse VLMInterface's base64 data-URL encoder so the same shape
        # works through vLLM's OpenAI-format chat path without needing
        # `allowed_local_media_path` configuration.
        url = VLMInterface._encode_image_as_data_url(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": url, "detail": self.image_detail}},
                {"type": "text", "text": prompt},
            ],
        }

    def query(self, user_prompt: str, image_path: Optional[str] = None) -> str:
        user_msg = self._build_user_message(user_prompt, image_path)
        # Stateless: just system + current user_msg. Matches the API-side
        # VLMInterface, and keeps each request well under --max-model-len
        # in long debate loops. See VLMInterface._build_stateless_messages
        # for why this is safe (debate engine embeds state in prompts).
        messages = VLMInterface._build_stateless_messages(self.conversation_history, user_msg)

        # vLLM's offline `chat()` applies the model's chat template AND
        # extracts multimodal payloads from OpenAI-format `image_url`
        # parts. One call per query — sequential matches the debate flow.
        call_t0 = time.time()
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        call_elapsed = time.time() - call_t0

        request_out       = outputs[0]
        completion        = request_out.outputs[0]
        response          = (completion.text or "").strip()
        finish_reason     = getattr(completion, "finish_reason", None)
        token_ids         = getattr(completion, "token_ids", None) or []
        completion_tokens = len(token_ids) if token_ids else None
        # prompt_token_ids lives on the RequestOutput (one prompt per request),
        # NOT on each CompletionOutput.
        prompt_token_ids  = getattr(request_out, "prompt_token_ids", None) or []
        prompt_tokens     = len(prompt_token_ids) if prompt_token_ids else None

        # Immediate warning if the model ran into --max-tokens. Easier to
        # diagnose mid-run than waiting for the per-task summary, and the
        # downstream JSON parser will probably fail on a truncated answer
        # so the user wants to know NOW.
        if finish_reason == "length":
            print(f"  [WARN] {self.role.value}: hit max_tokens"
                  f"={self.sampling_params.max_tokens} "
                  f"(generated {completion_tokens} tokens — response truncated). "
                  f"Bump --max-tokens.")

        if self.stats is not None:
            self.stats.record(call_elapsed, finish_reason,
                              completion_tokens, prompt_tokens)

        if self.logger is not None:
            try:
                self.logger.log_call(
                    role=self.role.value,
                    model_name=self.model_name,
                    messages=messages,
                    response=response,
                )
            except Exception as e:
                print(f"[WARN] RunLogger failed: {e}")

        self.conversation_history.append(user_msg)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VIKI-L2 2-robot tasks against a LOCAL VLM (vLLM)")

    # Model / inference backend
    p.add_argument("--model-path", required=True,
                   help="HuggingFace local path or hub id of the VLM "
                        "(e.g. /models/Qwen3-VL-32B-Instruct).")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM context length (KV-cache size). Lower this if "
                        "the model OOMs on init (default 8192).")
    p.add_argument("--gpu-mem-util", type=float, default=0.90,
                   help="vLLM `gpu_memory_utilization` (default 0.90).")
    p.add_argument("--dtype", default="auto",
                   help="vLLM dtype: auto | bfloat16 | float16 | float32 "
                        "(default auto).")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="number of GPUs for tensor parallelism (default 1).")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="pass trust_remote_code=True to vLLM (needed for "
                        "models whose modeling code lives in the HF repo).")

    # Sampling
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature (default 0.3, matches "
                        "VLMInterface's API-side default).")
    p.add_argument("--top-p", type=float, default=0.7,
                   help="top-p (default 0.7).")
    p.add_argument("--max-tokens", type=float, default=4096,
                   help="max generated tokens per call (default 4096).")

    # Evaluation set
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
                   help="path to the VIKI-L2 parquet.")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N 2-robot tasks "
                        "(default: all).")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first OFFSET tasks (default 0).")

    # Debate
    p.add_argument("--max-debate-rounds", type=int, default=3,
                   help="max rounds per Phase-2 invocation (default 3).")
    p.add_argument("--max-retry-rounds", type=int, default=2,
                   help="max retries on execution failure (default 2).")

    # Logging / output
    p.add_argument("--log-dir", default=None,
                   help="root dir for per-task VLM-call logs; defaults to "
                        "logs/local_<timestamp>.")
    p.add_argument("--tb-dir", default=None,
                   help="TensorBoard log dir; defaults to "
                        "tb_logs/local_<timestamp>.")
    p.add_argument("--results-json", default=None,
                   help="path to dump per-task results JSON "
                        "(defaults to <log-dir>/results.json).")
    p.add_argument("--no-vlm-log", action="store_true",
                   help="disable per-task VLM call logging.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="if a task crashes, log + continue.")

    return p.parse_args()


# ─── Stat helper ──────────────────────────────────────────────────────

def fmt_stats(label: str, values: list) -> str:
    if not values:
        return f"  {label:<30s} (no data)"
    return (
        f"  {label:<30s} "
        f"min={min(values)}  avg={statistics.mean(values):6.2f}  max={max(values)}  "
        f"(n={len(values)})"
    )


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Lazy vLLM import so --help works without it installed ──
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[ERROR] vLLM not installed: {e}\n"
              f"        Install it on Linux/WSL: pip install vllm",
              file=sys.stderr)
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
    log_root = Path(args.log_dir) if args.log_dir else (root / "logs" / f"local_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (root / "tb_logs" / f"local_{stamp}")

    # ── Find tasks ──
    print(f"Scanning parquet for 2-robot (R1+R2) tasks: {parquet_path}")
    all_indices = find_task_indices(parquet_path, n_robots=2,
                                    required_ids=("R1", "R2"))
    indices = all_indices[args.offset:]
    if args.limit is not None:
        indices = indices[:args.limit]
    print(f"  → {len(all_indices)} total 2-robot tasks; evaluating {len(indices)} "
          f"(offset={args.offset}, limit={args.limit})")

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is None:
        print("[WARN] No TensorBoard writer available — skipping TB logging.")
    else:
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard logs    → {tb_dir}")
        print(f"  view:  tensorboard --logdir {tb_dir.parent}")

    print(f"Per-task logs       → {log_root}")
    results_json = (Path(args.results_json) if args.results_json
                    else log_root / "results.json")
    print(f"Per-task JSON       → {results_json}")
    print()

    # ── Load model ONCE — both debaters share this LLM instance ──
    print(f"Loading local model with vLLM: {args.model_path}")
    print(f"  max_model_len={args.max_model_len}  "
          f"gpu_mem_util={args.gpu_mem_util}  "
          f"dtype={args.dtype}  tp={args.tensor_parallel_size}")
    llm_t0 = time.time()
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": 1},   # VIKI-L2 = one scene image per call
    )
    print(f"  loaded in {time.time() - llm_t0:.1f}s")

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
        # One shared stats accumulator — both debaters' calls feed it so
        # the per-task summary covers the whole debate.
        task_stats = TaskStats(
            max_tokens=int(args.max_tokens),
            max_model_len=int(args.max_model_len),
        )
        vlm1 = LocalVLMInterface(
            llm=llm, sampling_params=sampling_params,
            model_name=args.model_path,
            role=DebateRole.VLM1_R1_ADVOCATE, logger=rlog,
            stats=task_stats,
        )
        vlm2 = LocalVLMInterface(
            llm=llm, sampling_params=sampling_params,
            model_name=args.model_path,
            role=DebateRole.VLM2_R2_ADVOCATE, logger=rlog,
            stats=task_stats,
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
            "success":         success,
            "debate_loops":    loops,
            "debate_rounds_per_loop": rounds_per_loop,
            "elapsed_seconds": round(task_elapsed, 2),
            "last_failure":    last_failure,
        }
        results.append(record)

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

        # ── Per-task model-call stats ──
        if task_stats.call_count:
            max_out    = task_stats.max_completion_tokens
            max_in     = task_stats.max_prompt_tokens
            max_total  = task_stats.max_total_tokens
            out_pct    = max_out   / args.max_tokens    * 100 if args.max_tokens    else 0.0
            in_pct     = max_in    / args.max_model_len * 100 if args.max_model_len else 0.0
            total_pct  = max_total / args.max_model_len * 100 if args.max_model_len else 0.0
            remaining  = (args.max_model_len - max_in)        if max_in            else 0

            print(f"  → calls={task_stats.call_count}  "
                  f"avg_call={task_stats.avg_seconds:.2f}s")
            print(f"  → output: max_completion={max_out}/{int(args.max_tokens)} "
                  f"({out_pct:.0f}% of --max-tokens)")
            if max_in:
                print(f"  → input : max_prompt={max_in}/{int(args.max_model_len)} "
                      f"({in_pct:.0f}% of --max-model-len)   "
                      f"max_total={max_total}/{int(args.max_model_len)} "
                      f"({total_pct:.0f}%)   "
                      f"gen_budget_left={remaining}")

            # ── Output-side: --max-tokens warnings ──
            if task_stats.truncated_count:
                print(f"  [WARN] {task_stats.truncated_count}/{task_stats.call_count} "
                      f"calls hit max_tokens={int(args.max_tokens)} — "
                      f"bump --max-tokens (current cap leaks into success/parse rate)")
            elif out_pct >= 90:
                print(f"  [WARN] output close to --max-tokens cap "
                      f"({out_pct:.0f}%) — consider bumping --max-tokens.")

            # ── Input/context-side: --max-model-len warnings ──
            # Most useful to catch BEFORE vLLM raises or silently clips
            # generation budget. The "gen_budget_left vs --max-tokens"
            # comparison is the one that actually matters in practice.
            if max_in and remaining < args.max_tokens:
                print(f"  [WARN] prompt grew to {max_in} tokens, leaving only "
                      f"{remaining} tokens of generation budget "
                      f"(< --max-tokens={int(args.max_tokens)}). Next debate "
                      f"round may exceed --max-model-len={int(args.max_model_len)}. "
                      f"Bump --max-model-len, or lower --max-debate-rounds.")
            elif in_pct >= 80:
                print(f"  [WARN] prompt reached {in_pct:.0f}% of "
                      f"--max-model-len — debate history is getting heavy, "
                      f"watch for vLLM context errors on longer runs.")

        print(f"  running: success={n_success}/{n_done} ({success_rate:.1%})  "
              f"avg_loops={avg_loops:.2f}  avg_rounds={avg_rounds:.2f}")

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

        results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    total_elapsed = time.time() - t0

    # ── Final report ──
    print()
    print("=" * 70)
    print("FINAL STATISTICS — local VLM debate")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    print(f"Model:              {args.model_path}")
    print(f"Tasks evaluated:    {n_done}")
    print(f"Tasks succeeded:    {n_success}")
    print(f"Success rate:       {n_success}/{n_done} = {n_success/n_done:.2%}")
    print()
    print(fmt_stats("debate loops per task",  loops_list))
    print(fmt_stats("debate rounds per loop", all_rounds))
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
