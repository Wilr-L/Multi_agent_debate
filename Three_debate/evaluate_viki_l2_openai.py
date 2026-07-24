"""
evaluate_viki_l2_openai  (Three_debate / 3-embodiment variant)
==============================================================
Three-agent version of the official-OpenAI-API debate evaluator. THREE
debater VLMs (advocates for R1 / R2 / R3) run the full 4-phase pipeline
(Phase 1 triple proposals+merge → Phase 2 3-way debate, consensus =
3 consecutive ACCEPTs → Phase 3 symbolic execution → Phase 4 3-way
reflection+retry) against the **official OpenAI API** via the official
`openai` Python SDK.

Filters the parquet for VIKI-L2 3-robot (R1+R2+R3) tasks — there are 46.

Reference: https://developers.openai.com/api/docs/quickstart

Design (same as the 2-agent sibling one directory up)
-----------------------------------------------------
`OpenAIVLMInterface` subclasses the local (3-agent-aware) `VLMInterface`
and overrides ONLY the HTTP transport (`_post_with_retry`): the SDK
response is converted back to a plain chat-completions dict via
`.model_dump()`, so every inherited behavior — stateless message
building, base64 image encoding, empty-content retry, RunLogger
integration — keeps working unchanged.

Two retry layers guard against unstable networks (e.g. reaching
api.openai.com through a proxy/VPN): the SDK's fast internal retries,
plus an outer 3s/9s/27s exponential backoff for connection errors that
outlast them.

Setup:
    pip install openai
    # PowerShell:
    $env:OPENAI_API_KEY = "sk-..."

Example:
    E:\\anaconda3\\python.exe evaluate_viki_l2_openai.py --limit 5
    E:\\anaconda3\\python.exe evaluate_viki_l2_openai.py \\
        --model-name gpt-4o-mini --temperature 0.3 --limit 46

Notes:
  - The default --parquet resolves against this folder FIRST and falls
    back to the parent repo root, so running from Three_debate/ works
    without passing ../VIKI_data/... explicitly.
  - results.json schema and TensorBoard tag names match the other
    evaluators, so aggregate_metrics.py / stratified_acc.py work
    unchanged and runs overlay cleanly in TB.
"""

import argparse
import json
import os
import sys
import time
import statistics
import traceback
from pathlib import Path
from typing import Optional

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


# ─── Official-SDK transport for VLMInterface ─────────────────────────

class OpenAIVLMInterface(VLMInterface):
    """`VLMInterface` with the raw-`requests` transport swapped for the
    official `openai` SDK. Everything else (stateless messages, image
    data-URLs, empty-content retry loop, logging) is inherited.

    The SDK client is created once per instance with the inherited
    `timeout` / `max_retries` settings, so SDK-level retry/backoff
    replaces the hand-rolled HTTP retry of the parent class."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, *args, base_url: Optional[str] = None, **kwargs):
        kwargs.setdefault("model_name", self.DEFAULT_MODEL)
        kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY"))
        super().__init__(*args, base_url=base_url or "https://api.openai.com/v1",
                         **kwargs)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The official OpenAI SDK is required: pip install openai"
            ) from e
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,            # None → official endpoint
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        """Send the chat completion through the official SDK and return
        the plain-dict response the inherited `query()` expects.

        `max_tokens` → `max_completion_tokens`: the official API renamed
        the parameter; the new name is accepted by gpt-4o-family models
        and REQUIRED by o-series / gpt-5 reasoning models.

        Two retry layers:
          - SDK-internal (client max_retries): fast back-to-back retries
            for 429/5xx/connect errors — good for momentary hiccups.
          - This outer loop: LONG exponential backoff (3s/9s/27s) for
            connection errors that outlast the SDK's quick retries —
            typical when api.openai.com is reached through an unstable
            proxy/VPN and the tunnel drops for several seconds
            (symptom: `WinError 10054` connection-reset during the TLS
            handshake inside httpcore's http_proxy transport)."""
        kwargs = dict(payload)
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        from openai import (APIConnectionError, RateLimitError,
                            InternalServerError)
        outer_attempts = 4
        last_err: Optional[Exception] = None
        for attempt in range(outer_attempts):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.model_dump()
            except (APIConnectionError, RateLimitError,
                    InternalServerError) as e:
                last_err = e
                if attempt == outer_attempts - 1:
                    break
                backoff = 3.0 ** (attempt + 1)          # 3s, 9s, 27s
                print(f"[WARN] {self.role.value}: {type(e).__name__} — "
                      f"retrying in {backoff:.0f}s "
                      f"(outer attempt {attempt + 2}/{outer_attempts})...")
                time.sleep(backoff)
        raise RuntimeError(
            f"OpenAI API failed after {outer_attempts} outer attempts "
            f"(each with SDK-internal retries). Last error: {last_err}. "
            f"If this is APIConnectionError / WinError 10054 and your "
            f"traffic goes through a proxy/VPN, the tunnel is unstable — "
            f"switch proxy node or check its TLS handling."
        ) from last_err


# ─── CLI ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VIKI-L2 3-robot tasks — 4-phase 3-agent "
                    "debate via the official OpenAI API")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
                   help="path to the VIKI-L2 parquet (resolved against this "
                        "folder, then the parent repo root)")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N 3-robot tasks (default: all)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first OFFSET tasks (default: 0)")
    p.add_argument("--max-debate-rounds", type=int, default=3,
                   help="max rounds per Phase-2 invocation (default 3)")
    p.add_argument("--max-retry-rounds", type=int, default=2,
                   help="max retries on execution failure (default 2)")
    p.add_argument("--model-name", default="gpt-4o",
                   help="OpenAI model id (default gpt-4o; must be "
                        "vision-capable for the scene image)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature (default 0.3; reasoning "
                        "models like o-series/gpt-5 only accept 1)")
    p.add_argument("--top-p", type=float, default=0.7,
                   help="top-p (default 0.7; reasoning models only accept 1)")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="max completion tokens per call (default 4096; sent "
                        "as `max_completion_tokens`)")
    p.add_argument("--base-url", default=None,
                   help="override the API base URL (default: official "
                        "api.openai.com; use for Azure/OpenAI-compatible proxies)")
    p.add_argument("--log-dir", default=None,
                   help="root dir for per-task VLM-call logs; "
                        "defaults to logs/openai3_<timestamp>")
    p.add_argument("--tb-dir", default=None,
                   help="TensorBoard log dir; defaults to tb_logs/openai3_<timestamp>")
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


def resolve_parquet(parquet_arg: str, script_root: Path) -> Optional[Path]:
    """Resolve --parquet against the script dir first, then the parent
    repo root (VIKI_data usually lives at the repo root, one level up
    from Three_debate/)."""
    parquet_path = Path(parquet_arg)
    if parquet_path.is_absolute():
        return parquet_path if parquet_path.is_file() else None
    for base in (script_root, script_root.parent):
        cand = base / parquet_path
        if cand.is_file():
            return cand
    return None


# ─── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
        print("        On PowerShell:  $env:OPENAI_API_KEY = 'sk-...'",
              file=sys.stderr)
        sys.exit(1)

    # ── Resolve paths ──
    root = Path(__file__).resolve().parent
    parquet_path = resolve_parquet(args.parquet, root)
    if parquet_path is None:
        print(f"[ERROR] Parquet not found: {args.parquet} "
              f"(searched {root} and {root.parent})", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_dir) if args.log_dir else (root / "logs" / f"openai3_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (root / "tb_logs" / f"openai3_{stamp}")

    # ── Find tasks (3-robot variant) ──
    print(f"Scanning parquet for 3-robot (R1+R2+R3) tasks: {parquet_path}")
    all_indices = find_task_indices(parquet_path, n_robots=3,
                                    required_ids=("R1", "R2", "R3"))
    indices = all_indices[args.offset:]
    if args.limit is not None:
        indices = indices[:args.limit]
    print(f"  → {len(all_indices)} total 3-robot tasks; evaluating {len(indices)} "
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

    print(f"Model               → {args.model_name} (official OpenAI API, 3-agent debate)")
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

        # Per-task RunLogger (shared by all three debaters)
        rlog = None
        if not args.no_vlm_log:
            rlog = RunLogger(log_root / f"task_{idx:05d}_{task['task_id']}")

        # Engine — three official-SDK debaters
        robot1 = RobotProfile(name=robots["R1"], robot_id="R1")
        robot2 = RobotProfile(name=robots["R2"], robot_id="R2")
        robot3 = RobotProfile(name=robots["R3"], robot_id="R3")
        vlm_kwargs = dict(
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            logger=rlog,
        )
        vlm1 = OpenAIVLMInterface(role=DebateRole.VLM1_R1_ADVOCATE, **vlm_kwargs)
        vlm2 = OpenAIVLMInterface(role=DebateRole.VLM2_R2_ADVOCATE, **vlm_kwargs)
        vlm3 = OpenAIVLMInterface(role=DebateRole.VLM3_R3_ADVOCATE, **vlm_kwargs)
        sim  = SimulatorInterface(scene_seed=0)
        eng  = MultiAgentDebateEngine(
            vlm1, vlm2, vlm3, sim, robot1, robot2, robot3,
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
            "model_name":      args.model_name,
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
    print("FINAL STATISTICS — 3-agent debate via official OpenAI API")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    print(f"Model:              {args.model_name}")
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
