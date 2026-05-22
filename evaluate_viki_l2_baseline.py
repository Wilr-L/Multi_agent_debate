"""
evaluate_viki_l2_baseline
=========================
Run the **stock VIKI-Bench L2 single-VLM prompt** (verbatim from
`VIKI-R/eval/VIKI-L2/qwen.py`) on the first N 2-robot tasks of
`VIKI_data/viki/VIKI-L2/test.parquet`, calling SiliconFlow via the same
`VLMInterface` used by the debate pipeline. One API call per task; no
debate, no reflection, no retry — this is the **no-debate baseline** to
compare against `evaluate_viki_l2.py` (multi-agent debate).

TensorBoard scalars are written under the SAME tag names as the debate
evaluator so the two runs overlay cleanly:
    rate/success, rate/success_this,
    loops/this_task, loops/avg_so_far,
    rounds/last_loop, rounds/avg_so_far,
    timing/seconds_per_task

For the baseline `loops` and `rounds` are flat 1's (or 0 on parse failure),
making the contrast against debate's variable counts obvious in TB.

Example:
    $env:SILICONFLOW_API_KEY = "sk-..."
    E:\\anaconda3\\python.exe evaluate_viki_l2_baseline.py --limit 20
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


# ─── VIKI's L2 prompt, adapted for a non-thinking instruct model ───────
# Deviation from VIKI-R/eval/VIKI-L2/qwen.py: the <think>...</think>
# requirement (numbered step 2 in the original) is removed. The default
# model here is Qwen/Qwen3-VL-32B-Instruct — a non-thinking variant —
# and being told to emit <think> tokens caused it to stop on the first
# token (completion_tokens=0, finish_reason='stop'). Confirmed by
# diagnose_baseline.py Test C. Keeping the same Instruct model as the
# debate engine is what makes the baseline-vs-debate comparison clean.
# The <answer>...</answer> JSON contract is kept verbatim.

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


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stock-VIKI-L2 single-VLM baseline evaluator")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N 2-robot tasks (default: all)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first OFFSET tasks (default: 0)")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="max tokens per chat completion (default 16384 — "
                        "VIKI prompts + image are large and the model writes "
                        "both <think>...</think> AND <answer>...</answer>; "
                        "bump higher if you see empty-content errors)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="VIKI eval uses 0.0; defaults to 0.0 here too")
    p.add_argument("--enable-thinking", action="store_true",
                   help="explicitly opt in to Qwen3 API-level thinking mode. "
                        "OFF by default and NOT sent at all unless this flag is "
                        "passed (Qwen3-VL-32B-Instruct rejects the parameter; "
                        "smaller Qwen3 variants may accept it)")
    p.add_argument("--log-dir", default=None,
                   help="root dir for per-task VLM-call logs; "
                        "defaults to logs/baseline_<timestamp>")
    p.add_argument("--tb-dir", default=None,
                   help="TensorBoard log dir; defaults to "
                        "tb_logs/baseline_<timestamp>")
    p.add_argument("--results-json", default=None,
                   help="path to dump per-task results JSON "
                        "(defaults to <log-dir>/results.json)")
    p.add_argument("--no-vlm-log", action="store_true",
                   help="disable per-task VLM call logging")
    p.add_argument("--continue-on-error", action="store_true",
                   help="if a task crashes, log + continue")
    return p.parse_args()


# ─── Stat helper ─────────────────────────────────────────────────────

def fmt_stats(label: str, values: list) -> str:
    if not values:
        return f"  {label:<30s} (no data)"
    return (
        f"  {label:<30s} "
        f"min={min(values)}  avg={statistics.mean(values):6.2f}  max={max(values)}  "
        f"(n={len(values)})"
    )


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.environ.get("SILICONFLOW_API_KEY"):
        print("[ERROR] SILICONFLOW_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = root / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] Parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_dir) if args.log_dir else (root / "logs" / f"baseline_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (root / "tb_logs" / f"baseline_{stamp}")

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

        # Per-task RunLogger (one folder per task — single .txt file each)
        rlog = None
        if not args.no_vlm_log:
            rlog = RunLogger(log_root / f"task_{idx:05d}_{task['task_id']}")

        # Build VIKI's stock system prompt for this task's specific robot team.
        robot_types = list(robots.values())
        available_actions_view = {r: AGENT_AVAIL_ACTIONS.get(r, []) for r in robot_types}
        available_robots_view  = {r: ROBOT_DESCRIPTION.get(r, "")   for r in robot_types}
        system_prompt = VIKI_L2_SYSTEM_PROMPT.format(
            ACTION_DESCRIPTION=ACTION_DESCRIPTION,
            robots=robots,
            available_robots=available_robots_view,
            available_actions=available_actions_view,
        )

        # Only forward `enable_thinking` when the user opted in — many models
        # (incl. Qwen3-VL-32B-Instruct) 400 on the parameter outright.
        extra: dict = {}
        if args.enable_thinking:
            extra["enable_thinking"] = True

        vlm = VLMInterface(
            role=DebateRole.VLM1_R1_ADVOCATE,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logger=rlog,
            extra_params=extra,
        )
        vlm.set_system_prompt(system_prompt)

        task_t0 = time.time()
        # ── ONE API call — VIKI's exact user message shape: ──
        # "Task Description: <text>" with the scene image attached.
        try:
            response = vlm.query(
                f"Task Description: {task['task_description']}"
                f"\n\nNow produce the plan in the exact <think>…</think><answer>…</answer> format required above.",
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

        # Baseline analogs to debate's loop/round counts.
        # 1 if we got a parseable plan, else 0 — gives a flat reference line in TB.
        loops_this_task  = 0 if parse_failed else 1
        rounds_this_loop = 0 if parse_failed else 1

        record = {
            "idx":             idx,
            "task_id":         task["task_id"],
            "task_name":       task["task_name"],
            "robots":          robots,
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

        # ── TensorBoard (same tag names as the debate evaluator) ──
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
    print("FINAL STATISTICS — VIKI-L2 single-VLM baseline")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    n_parsed = sum(1 for r in succ_records if not r.get("parse_failed", False))
    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

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
