"""
evaluate_viki_l2_baseline_openai  (Three-baseline / 3-embodiment variant)
=========================================================================
The **no-debate VIKI-L2 baseline** (stock single-VLM prompt, ONE call per
task) on the **3-robot (R1+R2+R3)** task subset, evaluated through the
**official OpenAI API** — the 3-embodiment counterpart of the root
`evaluate_viki_l2_baseline_openai.py`.

Reference: https://developers.openai.com/api/docs/quickstart

Self-contained by design (Three-baseline has no importable baseline /
openai sibling modules), so this file carries its own copies of:
  - `VIKI_L2_SYSTEM_PROMPT` — byte-identical to the root baseline's
    (VIKI stock prompt minus the <think> requirement). The template is
    robot-count-agnostic: `{robots}` / `{available_robots}` /
    `{available_actions}` enumerate however many robots the task has,
    which is exactly how VIKI's own eval prompts 3-robot tasks.
  - `parse_viki_answer` — the <answer>…</answer> → TaskPlan parser.
  - `OpenAIVLMInterface` — official `openai` SDK transport subclassing
    the LOCAL (fix-carrying) `VLMInterface`, with
    max_tokens→max_completion_tokens translation and dual-layer retry
    (SDK-internal fast retries + outer 3s/9s/27s backoff for proxy/VPN
    blips).

Setup:
    pip install openai
    $env:OPENAI_API_KEY = "sk-..."          # PowerShell

Example (from inside Three-baseline/; the parquet default resolves
against this folder first, then the parent repo root):
    E:\\anaconda3\\python.exe evaluate_viki_l2_baseline_openai.py --limit 5
    E:\\anaconda3\\python.exe evaluate_viki_l2_baseline_openai.py \\
        --model-name gpt-4o-mini --limit 46 --continue-on-error

TensorBoard tags and the results.json schema match the other evaluators
(loops/rounds flat 1s, `parse_failed` recorded), so aggregate_metrics.py
/ stratified_acc.py work unchanged.
"""

import argparse
import ast
import json
import os
import re
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
    VLMInterface, SimulatorInterface, DebateRole, RunLogger, TaskPlan,
    ROBOT_DESCRIPTION, ACTION_DESCRIPTION, AGENT_AVAIL_ACTIONS,
    _normalize_step,
)


# ─── VIKI's L2 prompt (byte-identical to the root baseline's copy) ─────
# Deviation from VIKI-R/eval/VIKI-L2/qwen.py: the <think>...</think>
# requirement is removed (see the root baseline for the full history).
# The <answer>...</answer> JSON contract is kept verbatim. The example
# shows R1/R2 but the `{robots}` / `{available_robots}` /
# `{available_actions}` lines list ALL of this task's robots (R1..R3),
# matching how VIKI's own eval prompts 3-robot tasks.

VIKI_L2_SYSTEM_PROMPT = """You are a plan creator. I will provide you with an image of robots in a scene, available robots and their action primitives, and a task description. You need to create a plan to complete the task.
You must first analyze the image to fully understand the scene depicted. Then, analyze the task description. Finally, create a plan to complete the task.
Your reasoning must strictly adhere to the visual content of the image and the task description—no assumptions, hypotheses, or guesses are allowed.
1. Create a plan to complete the task, noting:
   - Each robot can only perform ONE action per time step.
   - Multiple robots can work in parallel, but each robot is limited to one action at a time.
2. Your final answer must be within <answer> and </answer> tags, and **strictly follow the JSON format specified below**.

Output Format Requirements(please comply strictly, do not output any additional content):
<answer>
  [
    {{
      "step": 1,
      "actions": {{'R1': ['Move', 'pumpkin'], 'R2': ['Move', 'apple']}}
    }},
    {{
      "step": 2,
      "actions": {{'R1': ['Reach', 'pumpkin'], 'R2': ['Reach', 'apple']}}
    }}
    # ... subsequent steps ...
  ]
</answer>
Where:
- step is the time step number (starting from 1, incrementing sequentially).
- Each robot can only have ONE action per time step.
- "actions" is a dictionary that specifies the action for each robot during a single time step. Each key (e.g., "R1", "R2") represents a robot. Each value is a list describing the single action that robot will perform in this step, with the following format: action_type, target_object_or_location, (optional: extra_argument)
Action primitives and descriptions: {ACTION_DESCRIPTION}
Available robot set: {robots}
Robot characteristics: {available_robots}
Their available operation APIs: {available_actions}
"""


# ─── <answer>...</answer> parsing ──────────────────────────────────

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def parse_viki_answer(response: str) -> Optional[TaskPlan]:
    """
    Extract the <answer>…</answer> block from a VIKI-L2 response and parse
    its step list into a TaskPlan that SimulatorInterface understands.
    Tolerates both strict JSON and Python-literal output (VIKI's own example
    uses single-quoted strings inside `actions`).
    """
    if not response:
        return None
    m = _ANSWER_RE.search(response)
    if not m:
        # Some models forget the tag — fall back to "first [...] block".
        l = response.find("[")
        r = response.rfind("]") + 1
        if l == -1 or r == 0:
            return None
        body = response[l:r]
    else:
        body = m.group(1).strip()
        # Strip trailing comments like "# ... subsequent steps ..."
        body = re.sub(r"#[^\n]*", "", body)
        body = body.strip()

    steps_raw = None
    try:
        steps_raw = json.loads(body)
    except json.JSONDecodeError:
        try:
            steps_raw = ast.literal_eval(body)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(steps_raw, list) or not steps_raw:
        return None

    normalized = [_normalize_step(s if isinstance(s, dict) else {}, i)
                  for i, s in enumerate(steps_raw)]
    return TaskPlan(steps=normalized, reasoning="", raw_text=response)


# ─── Official-SDK transport for VLMInterface ─────────────────────────

class OpenAIVLMInterface(VLMInterface):
    """`VLMInterface` with the raw-`requests` transport swapped for the
    official `openai` SDK. Everything else (stateless messages, image
    data-URLs, empty-content retry loop, logging) is inherited."""

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
        """Official-SDK call returning the plain chat-completions dict the
        inherited `query()` expects. `max_tokens` is sent as
        `max_completion_tokens` (required by o-series / gpt-5, accepted
        by gpt-4o). Outer 3s/9s/27s backoff covers connection errors that
        outlast the SDK's fast internal retries (unstable proxy/VPN)."""
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


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stock-VIKI-L2 single-VLM baseline on 3-robot tasks "
                    "via the official OpenAI API")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
                   help="path to the VIKI-L2 parquet (resolved against this "
                        "folder, then the parent repo root)")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N 3-robot tasks (default: all)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first OFFSET tasks (default: 0)")
    p.add_argument("--model-name", default="gpt-4o",
                   help="OpenAI model id (default gpt-4o; must be "
                        "vision-capable for the scene image)")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="max completion tokens per call (default 8192; sent "
                        "as `max_completion_tokens`)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="VIKI eval uses 0.0; defaults to 0.0 here too "
                        "(reasoning models like o-series/gpt-5 only accept 1)")
    p.add_argument("--base-url", default=None,
                   help="override the API base URL (default: official "
                        "api.openai.com)")
    p.add_argument("--log-dir", default=None,
                   help="root dir for per-task VLM-call logs; "
                        "defaults to logs/baseline_openai3_<timestamp>")
    p.add_argument("--tb-dir", default=None,
                   help="TensorBoard log dir; defaults to "
                        "tb_logs/baseline_openai3_<timestamp>")
    p.add_argument("--results-json", default=None,
                   help="path to dump per-task results JSON "
                        "(defaults to <log-dir>/results.json)")
    p.add_argument("--no-vlm-log", action="store_true",
                   help="disable per-task VLM call logging")
    p.add_argument("--continue-on-error", action="store_true",
                   help="if a task crashes, log + continue")
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────

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
    from Three-baseline/)."""
    parquet_path = Path(parquet_arg)
    if parquet_path.is_absolute():
        return parquet_path if parquet_path.is_file() else None
    for base in (script_root, script_root.parent):
        cand = base / parquet_path
        if cand.is_file():
            return cand
    return None


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
        print("        On PowerShell:  $env:OPENAI_API_KEY = 'sk-...'",
              file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent
    parquet_path = resolve_parquet(args.parquet, root)
    if parquet_path is None:
        print(f"[ERROR] Parquet not found: {args.parquet} "
              f"(searched {root} and {root.parent})", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = (Path(args.log_dir) if args.log_dir
                else root / "logs" / f"baseline_openai3_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = (Path(args.tb_dir) if args.tb_dir
              else root / "tb_logs" / f"baseline_openai3_{stamp}")

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
        print("[WARN] No TensorBoard writer available — skipping TB logging.")
    else:
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard logs    → {tb_dir}")
        print(f"  view:  tensorboard --logdir {tb_dir.parent}")

    print(f"Model               → {args.model_name} (official OpenAI API)")
    print(f"Per-task logs       → {log_root}")
    results_json = (Path(args.results_json) if args.results_json
                    else log_root / "results.json")
    print(f"Per-task JSON       → {results_json}")
    print()

    # ── Single-shot evaluation loop ──
    results: list[dict] = []
    n_success = 0
    t0 = time.time()
    simulator = SimulatorInterface(scene_seed=0)

    for i, idx in enumerate(indices):
        step = i + 1
        print(f"\n{'='*70}\n[{step}/{len(indices)}] task #{idx}\n{'='*70}")

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

        # Build VIKI's stock system prompt for this task's 3-robot team.
        robot_types = list(robots.values())
        available_actions_view = {r: AGENT_AVAIL_ACTIONS.get(r, []) for r in robot_types}
        available_robots_view  = {r: ROBOT_DESCRIPTION.get(r, "")   for r in robot_types}
        system_prompt = VIKI_L2_SYSTEM_PROMPT.format(
            ACTION_DESCRIPTION=ACTION_DESCRIPTION,
            robots=robots,
            available_robots=available_robots_view,
            available_actions=available_actions_view,
        )

        vlm = OpenAIVLMInterface(
            role=DebateRole.VLM1_R1_ADVOCATE,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            logger=rlog,
        )
        vlm.set_system_prompt(system_prompt)

        task_t0 = time.time()
        # ── ONE API call — VIKI's user message shape, <answer>-only nudge ──
        try:
            response = vlm.query(
                f"Task Description: {task['task_description']}"
                f"\n\nNow produce the plan in the exact <answer>…</answer> "
                f"JSON format required above.",
                image_path=task["image_path"],
            )
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            raise
        except Exception as e:
            print(f"  [ERROR] API call crashed: {e}")
            traceback.print_exc()
            if not args.continue_on_error:
                raise
            results.append({"idx": idx, "task_id": task["task_id"],
                            "error": f"api: {e}"})
            continue

        # ── Parse <answer>...</answer> → TaskPlan ──
        plan = parse_viki_answer(response)
        parse_failed = plan is None

        success = False
        exec_failure = None
        if not parse_failed:
            try:
                exec_result = simulator.execute_plan(plan, task["scene_config"])
                success = bool(exec_result.get("success"))
                if not success:
                    exec_failure = exec_result.get("failure_reason")
            except Exception as e:
                print(f"  [ERROR] simulator crashed: {e}")
                traceback.print_exc()
                if not args.continue_on_error:
                    raise
                exec_failure = f"simulator crash: {e}"

        task_elapsed = time.time() - task_t0
        if success:
            n_success += 1

        # Baseline analogs to debate's loop/round counts (flat 1s).
        loops_this_task  = 0 if parse_failed else 1
        rounds_this_loop = 0 if parse_failed else 1

        record = {
            "idx":             idx,
            "task_id":         task["task_id"],
            "task_name":       task["task_name"],
            "robots":          robots,
            "model_name":      args.model_name,
            "success":         success,
            "parse_failed":    parse_failed,
            "debate_loops":    loops_this_task,
            "debate_rounds_per_loop": [rounds_this_loop] if loops_this_task else [],
            "elapsed_seconds": round(task_elapsed, 2),
            "last_failure":    exec_failure or ("parse_failed" if parse_failed else None),
        }
        results.append(record)

        succ_records = [r for r in results if "success" in r]
        n_done       = len(succ_records)
        success_rate = n_success / n_done if n_done else 0.0
        loops_list   = [r["debate_loops"] for r in succ_records]
        all_rounds   = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]
        avg_loops    = statistics.mean(loops_list)  if loops_list else 0.0
        avg_rounds   = statistics.mean(all_rounds) if all_rounds else 0.0

        print(f"  → success={success}  parsed={not parse_failed}  ({task_elapsed:.1f}s)")
        if exec_failure and not success:
            print(f"  → failure: {exec_failure.splitlines()[0]}")
        print(f"  running: success={n_success}/{n_done} ({success_rate:.1%})  "
              f"avg_loops={avg_loops:.2f}  avg_rounds={avg_rounds:.2f}")

        # ── TensorBoard (same tag names as the other evaluators) ──
        if writer is not None:
            writer.add_scalar("rate/success",            success_rate,     step)
            writer.add_scalar("rate/success_this",       int(success),     step)
            writer.add_scalar("loops/this_task",         loops_this_task,  step)
            writer.add_scalar("loops/avg_so_far",        avg_loops,        step)
            writer.add_scalar("rounds/last_loop",        rounds_this_loop, step)
            writer.add_scalar("rounds/avg_so_far",       avg_rounds,       step)
            writer.add_scalar("timing/seconds_per_task", task_elapsed,     step)
            writer.flush()

        # Dump after every task so a crash doesn't lose data.
        results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    total_elapsed = time.time() - t0

    # ── Final report ──
    print()
    print("=" * 70)
    print("FINAL STATISTICS — VIKI-L2 3-robot single-VLM baseline "
          "(official OpenAI API)")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    n_parsed = sum(1 for r in succ_records if not r.get("parse_failed", False))
    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    print(f"Model:              {args.model_name}")
    print(f"Tasks evaluated:    {n_done}")
    print(f"Parse-OK rate:      {n_parsed}/{n_done} = {n_parsed/n_done:.2%}  "
          f"(plans the model produced in a parseable shape)")
    print(f"Tasks succeeded:    {n_success}")
    print(f"Success rate:       {n_success}/{n_done} = {n_success/n_done:.2%}")
    print()
    print(fmt_stats("loops per task (always 1)", loops_list))
    print(fmt_stats("rounds per loop (always 1)", all_rounds))
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
