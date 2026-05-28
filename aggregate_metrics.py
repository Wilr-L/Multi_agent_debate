"""
aggregate_metrics
=================
Aggregate evaluation metrics over a results directory produced by
`evaluate_viki_l2_localmodel.py` / `evaluate_viki_l2.py`.

Reports the average of:
  - delta_steps          : (generated plan steps) - (ground-truth steps),
                           averaged over SUCCESSFUL tasks only (a failed
                           plan's length is not comparable to ground truth)
  - Convergence Rounds   : debate rounds to first consensus (rounds of loop 1)
  - First-Pass Success   : fraction where the first consensus plan passed
                           validation directly (no retry)
  - Debate Loops         : number of Phase-2 invocations per task
  - Error Reduction Rate : N/A — see note below
  - Token Cost           : ESTIMATED total tokens (input + output) per task

Data sources
------------
  results.json            -> debate_loops, debate_rounds_per_loop, success
  per-task NNN_vlmN.txt    -> final generated plan (step count) + token estimate
  parquet `time_steps`     -> ground-truth step count (keyed by row idx)

Best-effort caveats (the original run did NOT persist these directly):
  * delta_steps  — the generated step count is recovered by parsing the LAST
    parseable plan in the task's call logs. This is the plan the debate
    converged to, i.e. what Phase 3 executed.
  * Token Cost   — the logs store message TEXT but truncate the base64 image
    and never recorded exact token usage. We re-tokenize the text and add a
    fixed per-image token budget (--image-tokens). Treat as ±10% estimate.
  * Error Reduction Rate — needs per-loop execution error COUNTS, which were
    never written to results.json or the logs. Reported as N/A. To get it,
    re-run with the evaluator persisting execution_results per loop.

Usage
-----
  python aggregate_metrics.py evaluate_results/local_20260524_020504
  python aggregate_metrics.py <dir> \\
      --parquet VIKI_data/viki/VIKI-L2/test.parquet \\
      --tokenizer-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --image-tokens 1280 \\
      --verbose
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Optional


# ─── Plan-step extraction from a logged response ──────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level balanced {...} substring in `text`."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def _steps_in_obj(data) -> Optional[int]:
    """Step count from a parsed plan/critique dict, or None."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("steps"), list):
        return len(data["steps"])
    rp = data.get("revised_plan")
    if isinstance(rp, dict) and isinstance(rp.get("steps"), list):
        return len(rp["steps"])
    return None


def count_plan_steps(response_text: str) -> Optional[int]:
    """Return the step count of the plan encoded in a single response, or
    None if it carries no plan (e.g. an ACCEPT-only verdict, or prose)."""
    # Prefer ```json fenced blocks; take the LAST one (the final plan).
    candidates = _FENCE_RE.findall(response_text) or _balanced_objects(response_text)
    for blob in reversed(candidates):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        n = _steps_in_obj(data)
        if n is not None:
            return n
    return None


# ─── Per-task log parsing ─────────────────────────────────────────────

_RESP_MARKER = "--- assistant (response) ---"


def _read_response_section(txt: str) -> str:
    """Return the assistant-response body of a single NNN_vlmN.txt file."""
    i = txt.rfind(_RESP_MARKER)
    if i == -1:
        return ""
    return txt[i + len(_RESP_MARKER):].strip()


def final_generated_steps(task_dir: Path) -> Optional[int]:
    """Walk the task's call logs in order; return the step count of the
    LAST response that carried a parseable plan (= the executed plan)."""
    last = None
    for f in sorted(task_dir.glob("[0-9]*_vlm*.txt")):
        resp = _read_response_section(f.read_text(encoding="utf-8", errors="replace"))
        n = count_plan_steps(resp)
        if n is not None:
            last = n
    return last


# ─── Token estimation ─────────────────────────────────────────────────

_HEADER_RE   = re.compile(r"^=== Call #")
_SECTION_RE  = re.compile(r"^--- (\[\d+\] \w+|assistant \(response\))")
_IMAGE_RE    = re.compile(r"^\[image:")


def _make_token_counter(tokenizer_path: Optional[str], chars_per_token: float):
    """Return (count_fn, label). Prefer an HF tokenizer, then tiktoken,
    then a chars/token heuristic."""
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
                    f"hf:{tokenizer_path}")
        except Exception as e:
            print(f"[WARN] could not load tokenizer at {tokenizer_path}: {e}\n"
                  f"       falling back to heuristic.", file=sys.stderr)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s)), "tiktoken:cl100k_base (approx)")
    except Exception:
        pass
    return (lambda s: int(len(s) / chars_per_token),
            f"heuristic:{chars_per_token:.1f}chars/token (rough)")


def task_token_cost(task_dir: Path, count_fn, image_tokens: int) -> int:
    """Estimate total input+output tokens for one task: sum over all call
    logs of (tokenized message + response text) + image_tokens per image."""
    total = 0
    for f in sorted(task_dir.glob("[0-9]*_vlm*.txt")):
        body_lines, n_images = [], 0
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if _HEADER_RE.match(line) or _SECTION_RE.match(line):
                continue                       # drop log scaffolding
            if _IMAGE_RE.match(line):
                n_images += 1                  # count, don't tokenize the blob
                continue
            body_lines.append(line)
        total += count_fn("\n".join(body_lines)) + n_images * image_tokens
    return total


# ─── Ground-truth step counts from the parquet ────────────────────────

def load_gt_steps(parquet_path: Path, indices: list[int]) -> dict[int, int]:
    """Map row idx -> len(ground_truth['time_steps'])."""
    import pandas as pd
    df = pd.read_parquet(str(parquet_path))
    out = {}
    for idx in indices:
        try:
            gt = df.iloc[idx]["reward_model"]["ground_truth"]
            ts = gt.get("time_steps")
            out[idx] = len(ts) if ts is not None else None
        except Exception:
            out[idx] = None
    return out


# ─── Main ─────────────────────────────────────────────────────────────

def discover_result_dirs(paths: list[Path]) -> list[Path]:
    """Expand the given paths into a list of leaf result dirs (each holding
    a results.json + task_* folders). A path is taken as a leaf if it
    contains results.json directly; otherwise it's treated as a parent and
    every `*/results.json` child is collected (so you can pass the
    `evaluate_results` parent and pick up every local_* run). Duplicates
    are removed, order is preserved."""
    leaves, seen = [], set()
    for p in paths:
        if (p / "results.json").is_file():
            cands = [p]
        else:
            cands = sorted(c.parent for c in p.glob("*/results.json"))
        for c in cands:
            rp = c.resolve()
            if rp not in seen:
                seen.add(rp)
                leaves.append(c)
    return leaves


def find_task_dir(source_dir: Path, idx: int) -> Optional[Path]:
    hits = sorted(source_dir.glob(f"task_{idx:05d}_*"))
    return hits[0] if hits else None


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate VIKI debate eval metrics")
    p.add_argument("results_dirs", nargs="+",
                   help="one or more result dirs (each with results.json + "
                        "task_* folders), OR a parent dir like "
                        "`evaluate_results` to auto-discover every local_* "
                        "run inside it. Records from interrupted/resumed runs "
                        "are merged; duplicate idx keeps the last one seen.")
    p.add_argument("--parquet", default="VIKI_data/viki/VIKI-L2/test.parquet",
                   help="VIKI-L2 parquet for ground-truth step counts")
    p.add_argument("--tokenizer-path", default=None,
                   help="HF tokenizer dir/id for exact-ish token counting "
                        "(e.g. a local Qwen2.5-VL path). Falls back to "
                        "tiktoken then a chars/token heuristic.")
    p.add_argument("--image-tokens", type=int, default=1280,
                   help="token budget charged per scene image (Qwen2.5-VL "
                        "default ~1280; tune to your min/max_pixels).")
    p.add_argument("--chars-per-token", type=float, default=3.8,
                   help="heuristic ratio used only if no tokenizer/tiktoken "
                        "is available (default 3.8).")
    p.add_argument("--verbose", action="store_true",
                   help="print a per-task table before the averages.")
    return p.parse_args()


def main():
    args = parse_args()

    leaf_dirs = discover_result_dirs([Path(p) for p in args.results_dirs])
    if not leaf_dirs:
        print(f"[ERROR] no results.json found under: {args.results_dirs}",
              file=sys.stderr)
        sys.exit(1)

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).resolve().parent / parquet_path

    # ── Merge records across all runs, tagging each with its source dir. ──
    # idx -> (record, source_dir). On duplicate idx (a resumed run re-did a
    # task) the LATER source dir wins, since leaf_dirs is in discovery order
    # and resumed runs come later.
    by_idx: dict[int, tuple[dict, Path]] = {}
    n_err = 0
    n_dup = 0
    print(f"Merging {len(leaf_dirs)} result dir(s):")
    for d in leaf_dirs:
        recs = json.loads((d / "results.json").read_text(encoding="utf-8"))
        n_ok = sum(1 for r in recs if "success" in r)
        print(f"  {d}  ({n_ok} evaluated / {len(recs)} records)")
        for r in recs:
            if "success" not in r:           # skip load/runtime error records
                n_err += 1
                continue
            idx = r["idx"]
            if idx in by_idx:
                n_dup += 1
            by_idx[idx] = (r, d)
    if n_dup:
        print(f"  [WARN] {n_dup} duplicate idx across dirs — kept the "
              f"later run's record for each.")

    evaluated = [by_idx[i] for i in sorted(by_idx)]
    if not evaluated:
        print("No evaluated tasks (every record is an error). Nothing to do.")
        return

    # Ground-truth step counts (one parquet read for all idxs).
    gt_steps = load_gt_steps(parquet_path, [r["idx"] for r, _ in evaluated])

    count_fn, tok_label = _make_token_counter(args.tokenizer_path, args.chars_per_token)

    rows = []
    for r, src_dir in evaluated:
        idx          = r["idx"]
        loops        = r.get("debate_loops")
        rounds_list  = r.get("debate_rounds_per_loop") or []
        conv_rounds  = rounds_list[0] if rounds_list else None
        success      = bool(r.get("success"))
        # First-Pass Success: a record may set `first_pass_success` directly
        # (SC needs this because it always draws all N samples so
        # `debate_loops == 1` would never hold). Otherwise infer the usual
        # way — first phase-2 / first sample / initial plan succeeded
        # ⇔ debate_loops == 1 ∧ success.
        if "first_pass_success" in r:
            first_pass = bool(r["first_pass_success"])
        else:
            first_pass = success and loops == 1

        tdir = find_task_dir(src_dir, idx)
        gen_steps = final_generated_steps(tdir) if tdir else None
        g = gt_steps.get(idx)
        delta = (gen_steps - g) if (gen_steps is not None and g is not None) else None
        tokens = task_token_cost(tdir, count_fn, args.image_tokens) if tdir else None

        rows.append({
            "idx": idx, "task_id": r.get("task_id"),
            "success": success, "loops": loops, "conv_rounds": conv_rounds,
            "first_pass": first_pass, "gen_steps": gen_steps, "gt_steps": g,
            "delta_steps": delta, "tokens": tokens,
        })

    # ── Per-task table ──
    if args.verbose:
        hdr = ("idx", "ok", "loops", "conv", "fp", "gen", "gt", "Δstep", "tokens")
        print(f"\n{hdr[0]:>5} {hdr[1]:>3} {hdr[2]:>5} {hdr[3]:>4} {hdr[4]:>3} "
              f"{hdr[5]:>4} {hdr[6]:>4} {hdr[7]:>6} {hdr[8]:>9}  task_id")
        for x in rows:
            ok_s    = "Y" if x["success"] else "N"
            fp_s    = "Y" if x["first_pass"] else "N"
            delta_s = "" if x["delta_steps"] is None else f"{x['delta_steps']:+d}"
            print(f"{x['idx']:>5} {ok_s:>3} "
                  f"{str(x['loops']):>5} {str(x['conv_rounds']):>4} {fp_s:>3} "
                  f"{str(x['gen_steps']):>4} {str(x['gt_steps']):>4} "
                  f"{delta_s:>6} {str(x['tokens']):>9}  {x['task_id']}")

    # ── Aggregates ──
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None, len(vals)

    n_total       = len(rows)
    # delta_steps is only meaningful for plans that actually succeeded — a
    # failed plan may have been cut short mid-execution, so its length is
    # not comparable to the ground-truth plan length.
    delta_mean, delta_n   = mean([x["delta_steps"] for x in rows if x["success"]])
    conv_mean,  conv_n    = mean([x["conv_rounds"] for x in rows])
    loops_mean, loops_n   = mean([x["loops"]       for x in rows])
    token_mean, token_n   = mean([x["tokens"]      for x in rows])
    n_first_pass          = sum(1 for x in rows if x["first_pass"])
    first_pass_rate       = n_first_pass / n_total

    def fmt(v, nd=3):
        return "  N/A" if v is None else f"{v:.{nd}f}"

    label = (leaf_dirs[0].name if len(leaf_dirs) == 1
             else f"{len(leaf_dirs)} runs merged")
    print("\n" + "=" * 64)
    print(f"AGGREGATE METRICS  —  {label}")
    print("=" * 64)
    print(f"Tasks evaluated:  {n_total}"
          + (f"   (+{n_err} error/skip records ignored)" if n_err else ""))
    print(f"Token estimator:  {tok_label};  image={args.image_tokens} tok/img")
    print("-" * 64)
    print(f"  delta_steps           {fmt(delta_mean):>10}   "
          f"(gen - gt, SUCCESSFUL tasks only, n={delta_n})")
    print(f"  Convergence Rounds    {fmt(conv_mean):>10}   "
          f"(rounds of loop 1, n={conv_n})")
    print(f"  First-Pass Success    {fmt(first_pass_rate):>10}   "
          f"({n_first_pass}/{n_total})")
    print(f"  Debate Loops          {fmt(loops_mean):>10}   "
          f"(Phase-2 invocations, n={loops_n})")
    print(f"  Error Reduction Rate  {'  N/A':>10}   "
          f"(per-loop error counts not recorded — see module docstring)")
    print(f"  Token Cost            {fmt(token_mean, 0):>10}   "
          f"(est. input+output tokens/task, n={token_n})")
    print("=" * 64)


if __name__ == "__main__":
    main()
