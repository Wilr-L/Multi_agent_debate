"""
single_agent_common  (Three-baseline / 3-embodiment variant)
============================================================
Shared scaffolding for the single-agent ablation evaluators, filtered to
the VIKI-L2 **3-robot (R1+R2+R3)** task subset (the sibling copy one dir
up filters for 2-robot tasks). The single-agent prompt machinery is
robot-count-agnostic (`render_system_prompt` enumerates whatever robots
the task has), so the ONLY functional change here vs the 2-robot version
is the task filter in `run_evaluation` (n_robots=3).

Evaluators sharing this module:
  - SC   (evaluate_viki_l2_sc.py)   : self-consistency, N samples + vote
  - SR   (evaluate_viki_l2_sr.py)   : self-refine (no checker feedback)
  - SR-C (evaluate_viki_l2_src.py)  : self-refine + checker error feedback

Each ablation script defines a `run_one_task(...)` callback and calls
`run_evaluation(args, strategy_label, run_one_task)`. The runner owns:
  - CLI parsing (common args; ablation adds its own on top)
  - vLLM model load (one shared LLM across all tasks)
  - parquet task discovery
  - per-task RunLogger + TaskStats
  - per-task summary + max_tokens / max_model_len warnings
  - TensorBoard scalars (SAME tag names as evaluate_viki_l2_localmodel.py
    so all four runs overlay cleanly)
  - incremental results.json writes (so a crash doesn't lose data)
  - final aggregate stats

Results.json schema matches evaluate_viki_l2_localmodel.py exactly so the
existing `aggregate_metrics.py` works unchanged on the ablations' output:
  success / debate_loops / debate_rounds_per_loop / elapsed_seconds /
  last_failure  — plus idx / task_id / task_name / robots.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        SummaryWriter = None  # type: ignore[assignment]

from viki_loader import load_viki_task, find_task_indices
from multi_agent_debate import (
    SimulatorInterface, RunLogger, DebateRole,
    ROBOT_DESCRIPTION, ACTION_DESCRIPTION, AGENT_AVAIL_ACTIONS,
    parse_plan_from_response, TaskPlan, _last_answer_dict,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface, TaskStats

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# ─── Single-agent prompt templates ────────────────────────────────────

SINGLE_AGENT_SYSTEM_PROMPT = """\
You are a plan creator for a multi-robot team in VIKI-Bench. I will provide you with \
an image of the robots in a scene, the available robots and their action primitives, and \
a task description. Your job is to produce a single JOINT plan that controls ALL robots \
to complete the task.

You must first analyze the image to fully understand the scene depicted. Then, analyze the \
task description. Finally, produce the plan. Your reasoning must strictly adhere to the visual \
content of the image and the task description — no assumptions, hypotheses, or guesses are allowed.

## Available robots and their action primitives
{robot_block}

## VIKI-Bench action primitives and rules
Action must follow the following format as a JSON list, for example ["Move", "plate"] or \
["grasp", "banana"]. It describes the single action that robot will perform in this step, \
with the format: action_type, target_object_or_location.
Action primitives and descriptions:
{action_block}
Use exact object and location names from the task, relevant assets, and world state. Do not \
invent new entity names.
Choose the primitive that advances the current object state, not just the task name:
  - If the robot is not near the object, use Move on the object.
  - If the robot is at/near the object but has not reached it, use Reach on the object.
  - If the robot has reached the object and is not carrying it, use Grasp on the object.
  - If the robot is carrying an object and it is not at the target, use Move on the target.
  - If the robot is carrying an object at the target area, use Place on the target location.
  - If an appliance or device must be started or activated, use Interact on that appliance \
after the required object is placed or available.
  - **Each robot can only perform ONE action per time step.** Multiple robots may work in \
parallel but each is limited to one action per step.

## Required output format (strict JSON)
Every plan you output MUST follow this exact structure:
```json
{{
  "reasoning": "step-by-step chain of thought...",
  "steps": [
    {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Move", "apple"], "R3": ["Move", "box"]}}}},
    {{"step": 2, "actions": {{"R1": ["Reach", "pumpkin"], "R2": ["Reach", "apple"], "R3": ["Reach", "box"]}}}}
  ]
}}
```
Rules:
- `step` starts at 1 and increments sequentially.
- `actions` is a dict mapping robot id ("R1", "R2", "R3") to a list \
`[action_type, target_object_or_location, (optional: extra_argument)]`.
- Only use action primitives that are in that robot's "Available actions" list above.
- Every step MUST include an entry for every robot (use `["Wait"]` for idle).
"""


PROPOSAL_PROMPT_SA = """\
## Task
Look at the scene image carefully. Here is the task description:
{task_description}

## Initial world state (from the symbolic simulator)
{world_state}

## Instructions
Produce the complete joint plan in the required JSON format. Think step by step:
1. What objects are visible in the scene? Where are they (cross-check against the world state)?
2. What is the goal state?
3. What actions does each robot need to perform? Can any be parallelized?
4. Are there ordering / dependency constraints?

Output ONLY the JSON plan.
"""


CRITIQUE_PROMPT_SA = """\
## Plan under self-review
{plan_json}

## Your task
Critically review YOUR OWN plan above. Identify any issues:
1. Feasibility — are actions within each robot's available action set? Are pre-conditions met \
(e.g. did the robot Reach + Grasp before Place)?
2. Coordination — timing conflicts? Both robots trying to access the same object simultaneously?
3. Completeness — does the plan achieve the task goal? Any steps missing?
4. Efficiency — wasted Wait steps? Better task allocation possible?

Respond in strict JSON. If you genuinely believe the plan is correct, respond:
```json
{{"verdict": "ACCEPT", "reasoning": "why the plan is good"}}
```
Otherwise list every concrete issue:
```json
{{
  "verdict": "REVISE",
  "issues": ["concrete issue 1", "concrete issue 2", ...]
}}
```
"""


CRITIQUE_PROMPT_SA_WITH_CHECKER = """\
## Plan under review
{plan_json}

## Symbolic simulator feedback (ground truth — trust this)
{execution_feedback}

## Your task
The plan was executed by the symbolic simulator and FAILED with the feedback above. Use \
that feedback as the primary signal for what went wrong. Identify:
1. Which step failed and why? (cross-check against the simulator's reason)
2. Was it a planning error (wrong action sequence) or a pre-condition error (missing setup step)?
3. What concrete changes should fix it?

Respond in strict JSON with the issues to address in the next revision:
```json
{{
  "verdict": "REVISE",
  "issues": ["concrete issue 1", "concrete issue 2", ...]
}}
```
"""


REVISE_PROMPT_SA = """\
## Previous plan
{plan_json}

## Issues identified in the critique
{issues_list}

## Your task
Produce a REVISED joint plan that addresses every issue above. Output ONLY the new plan in \
the standard JSON format:
```json
{{
  "reasoning": "what changed and why",
  "steps": [...]
}}
```
"""


# ─── Prompt rendering helpers ─────────────────────────────────────────

def render_system_prompt(robots_dict: dict) -> str:
    """Build SINGLE_AGENT_SYSTEM_PROMPT for this task's specific robot team."""
    robot_lines = []
    for rid, rname in robots_dict.items():
        if rname is None:
            continue
        desc    = ROBOT_DESCRIPTION.get(rname, "")
        actions = AGENT_AVAIL_ACTIONS.get(rname, [])
        robot_lines.append(f"- {rid} ({rname}): {desc}\n  Available actions: {actions}")
    robot_block = "\n".join(robot_lines)

    # Restrict the action descriptions to the primitives this task's robots can use.
    relevant = set()
    for rname in robots_dict.values():
        if rname:
            relevant.update(AGENT_AVAIL_ACTIONS.get(rname, []))
    action_block = "\n".join(f"- {ACTION_DESCRIPTION[a]}"
                             for a in ACTION_DESCRIPTION if a in relevant)

    return SINGLE_AGENT_SYSTEM_PROMPT.format(
        robot_block=robot_block, action_block=action_block,
    )


def parse_critique_issues(response: str) -> list[str]:
    """Extract `issues` list from a critique JSON response. Returns []
    when the verdict was ACCEPT or the response can't be parsed.

    Uses the robust extractor from multi_agent_debate (strict=False +
    last-answer object) so it survives the literal-newline-in-reasoning
    malformed JSON that VLMs routinely emit — the old first-{ / last-}
    slice with strict json.loads would silently return [] on those."""
    data = _last_answer_dict(response)
    if not isinstance(data, dict):
        return []
    if str(data.get("verdict", "")).upper() == "ACCEPT":
        return []
    issues = data.get("issues", [])
    if isinstance(issues, list):
        return [str(x) for x in issues if x]
    return []


def format_execution_feedback(exec_result: dict) -> str:
    """Mirror MultiAgentDebateEngine.phase4_reflection's feedback shape."""
    fb = (
        f"Plan failed at step {exec_result.get('failure_step', '?')}.\n"
        f"Completed {exec_result.get('completed_steps', 0)}/"
        f"{exec_result.get('total_steps', '?')} steps.\n"
        f"Failure reason: {exec_result.get('failure_reason', 'unknown')}\n"
    )
    if exec_result.get("observation_at_failure"):
        fb += f"Observation: {exec_result['observation_at_failure']}\n"
    return fb


# ─── Shared CLI ───────────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser):
    """Append the CLI args shared by all single-agent ablations."""
    # Model / inference backend
    p.add_argument("--model-path", required=True,
                   help="HuggingFace local path or hub id of the VLM.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--quantization", default=None,
                   choices=[None, "bf16", "int8", "int4", "fp8",
                            "awq", "gptq", "bitsandbytes"])
    p.add_argument("--load-format", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=None)

    # Sampling
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature (default 0.3; SC scripts "
                        "typically want 0.7-1.0 for sample diversity).")
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)

    # Evaluation set
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)

    # Logging / output
    p.add_argument("--log-dir", default=None)
    p.add_argument("--tb-dir",  default=None)
    p.add_argument("--results-json", default=None)
    p.add_argument("--no-vlm-log", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")


# ─── vLLM loader ──────────────────────────────────────────────────────

def load_llm(args):
    """Construct (llm, sampling_params). Lazy-import vllm so --help works
    without vLLM installed. Same quantization → vLLM-arg translation as
    evaluate_viki_l2_localmodel.py."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[ERROR] vLLM not installed: {e}\n"
              f"        Install it on Linux/WSL: pip install vllm",
              file=sys.stderr)
        sys.exit(1)

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
    # awq / gptq / fp8 / bitsandbytes pass through

    print(f"Loading local model with vLLM: {args.model_path}")
    print(f"  max_model_len={args.max_model_len}  "
          f"gpu_mem_util={args.gpu_mem_util}  "
          f"dtype={args.dtype}  tp={args.tensor_parallel_size}  "
          f"quantization={args.quantization or 'bf16'}  "
          f"load_format={load_format}")
    t0 = time.time()
    llm_kwargs = dict(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
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
    print(f"  loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=int(args.max_tokens),
    )
    return llm, sampling_params


# ─── Per-task stats summary helper (same format as the multi-agent runner) ──

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
        print(f"  [WARN] output close to --max-tokens cap "
              f"({out_pct:.0f}%) — consider bumping --max-tokens")
    if max_in and remaining < args.max_tokens:
        print(f"  [WARN] prompt grew to {max_in} tokens, leaving only "
              f"{remaining} tokens of generation budget "
              f"(< --max-tokens={int(args.max_tokens)}).")
    elif in_pct >= 80:
        print(f"  [WARN] prompt reached {in_pct:.0f}% of --max-model-len")


# ─── Main runner ──────────────────────────────────────────────────────

# `run_one_task` signature:
#     fn(task, llm, sampling_params, args, logger, stats) -> dict with keys
#         success            : bool
#         debate_loops       : int            (interpretation: # attempts)
#         debate_rounds_per_loop : list[int]  ([1]*N or per-loop call count)
#         last_failure       : str | None
TaskRunFn = Callable[[dict, "LLM", "SamplingParams", argparse.Namespace,
                      Optional[RunLogger], TaskStats], dict]


def run_evaluation(args: argparse.Namespace, strategy_label: str,
                   run_one_task: TaskRunFn):
    """Generic main loop shared by SC / SR / SR-C. `strategy_label` is a
    short slug (e.g. "sc") used in default log/tb dir names."""

    # ── Resolve paths ──
    root = Path(__file__).resolve().parent
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = root / parquet_path
    if not parquet_path.is_file():
        print(f"[ERROR] Parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_dir) if args.log_dir else (
        root / "logs" / f"{strategy_label}_{stamp}")
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (
        root / "tb_logs" / f"{strategy_label}_{stamp}")

    # ── Find tasks (3-robot variant) ──
    print(f"Scanning parquet for 3-robot (R1+R2+R3) tasks: {parquet_path}")
    all_indices = find_task_indices(parquet_path, n_robots=3,
                                    required_ids=("R1", "R2", "R3"))
    indices = all_indices[args.offset:]
    if args.limit is not None:
        indices = indices[:args.limit]
    print(f"  → {len(all_indices)} total 3-robot tasks; evaluating "
          f"{len(indices)} (offset={args.offset}, limit={args.limit})")

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is None:
        print("[WARN] No TensorBoard writer available — skipping TB logging.")
    else:
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard logs    → {tb_dir}")

    print(f"Per-task logs       → {log_root}")
    results_json = (Path(args.results_json) if args.results_json
                    else log_root / "results.json")
    print(f"Per-task JSON       → {results_json}")
    print()

    # ── Load model ONCE ──
    llm, sampling_params = load_llm(args)

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
        task_stats = TaskStats(max_tokens=int(args.max_tokens),
                               max_model_len=int(args.max_model_len))

        task_t0 = time.time()
        try:
            out = run_one_task(task, llm, sampling_params, args, rlog, task_stats)
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

        success         = bool(out.get("success"))
        loops           = int(out.get("debate_loops", 0))
        rounds_per_loop = list(out.get("debate_rounds_per_loop", []))
        last_failure    = out.get("last_failure")
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

        # ── Running aggregates ──
        succ_records = [r for r in results if "success" in r]
        n_done       = len(succ_records)
        success_rate = n_success / n_done if n_done else 0.0
        loops_list   = [r["debate_loops"] for r in succ_records]
        all_rounds   = [r for rec in succ_records
                          for r in rec["debate_rounds_per_loop"]]
        avg_loops    = statistics.mean(loops_list)  if loops_list else 0.0
        avg_rounds   = statistics.mean(all_rounds) if all_rounds else 0.0

        print(f"  → success={success}  loops={loops}  "
              f"rounds={rounds_per_loop}  ({task_elapsed:.1f}s)")
        if last_failure and not success:
            print(f"  → failure: {str(last_failure).splitlines()[0]}")

        _print_task_stats_summary(args, task_stats)

        print(f"  running: success={n_success}/{n_done} ({success_rate:.1%})  "
              f"avg_loops={avg_loops:.2f}  avg_rounds={avg_rounds:.2f}")

        # ── TensorBoard ──
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

        # ── Dump JSON after every task so a crash doesn't lose data ──
        results_json.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    total_elapsed = time.time() - t0

    # ── Final report ──
    print()
    print("=" * 70)
    print(f"FINAL STATISTICS — single-agent {strategy_label.upper()}")
    print("=" * 70)

    succ_records = [r for r in results if "success" in r]
    n_done = len(succ_records)
    if n_done == 0:
        print("No tasks evaluated successfully.")
        return

    loops_list = [r["debate_loops"] for r in succ_records]
    all_rounds = [r for rec in succ_records for r in rec["debate_rounds_per_loop"]]

    def fmt_stats(label, values):
        if not values:
            return f"  {label:<30s} (no data)"
        return (f"  {label:<30s} "
                f"min={min(values)}  avg={statistics.mean(values):6.2f}  "
                f"max={max(values)}  (n={len(values)})")

    print(f"Model:              {args.model_path}")
    print(f"Tasks evaluated:    {n_done}")
    print(f"Tasks succeeded:    {n_success}")
    print(f"Success rate:       {n_success}/{n_done} = {n_success/n_done:.2%}")
    print()
    print(fmt_stats("attempts per task",  loops_list))
    print(fmt_stats("calls per attempt",  all_rounds))
    print()
    print(f"Total wall time:    {total_elapsed:.1f}s "
          f"({total_elapsed/n_done:.1f}s per task)")
    print()
    print(f"Per-task records  : {results_json}")
    if writer is not None:
        print(f"TensorBoard logs  : {tb_dir}")
        writer.close()
