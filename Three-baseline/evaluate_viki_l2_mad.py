"""
evaluate_viki_l2_mad  (Three-baseline / 3-embodiment variant)
=============================================================
**Multi-Agents Debate (MAD)** baseline — Liang et al., EMNLP 2024
"Encouraging Divergent Thinking in Large Language Models through
Multi-Agent Debate" (github.com/Skytliang/Multi-Agents-Debate).

3-robot (R1+R2+R3) version of the sibling `../evaluate_viki_l2_mad.py`.
The debate MECHANICS are identical — three roles (Affirmative, Negative,
Moderator) plus a fallback Judge, one-shot execution, no retry. The
only changes vs the 2-robot version are:

  - Robot block is rendered DYNAMICALLY from the task's `robots` dict,
    so plans and prompts include R3 (not just R1/R2).
  - JSON schema examples show 3 robots to prime the model correctly.
  - Task filter is inherited from Three-baseline/single_agent_common.py
    (`find_task_indices(n_robots=3, required_ids=("R1","R2","R3"))`).

Round-by-round protocol
-----------------------
  Round 1
    • AFFIRMATIVE  sees task + scene image → proposes plan_aff
    • NEGATIVE    sees plan_aff → MUST DISAGREE → proposes alternative plan_neg
    • MODERATOR   sees both plans → returns JSON verdict
                  {preference: "Yes"/"No", supported_side, reason}
  Round 2..max
    • AFFIRMATIVE sees NEG's latest → new plan (defend or concede)
    • NEGATIVE   sees AFF's latest → new plan
    • MODERATOR  re-evaluates → verdict
  Stop condition
    • Moderator's `preference == "Yes"` → pick that side's LATEST plan; done
    • Max rounds reached without preference → fall back to JUDGE
      (step 1: list candidates, step 2: pick winner)
    • Winner's LATEST plan is executed in the simulator (ONE-SHOT, no retry).

Records produced match evaluate_viki_l2_localmodel.py's schema so
aggregate_metrics.py works unchanged:
  - debate_loops           = 1 (single execution)
  - debate_rounds_per_loop = [n_debate_rounds]

Example
-------
  python evaluate_viki_l2_mad.py --model-path /path/to/Qwen2.5-VL-32B-Instruct \\
      --max-rounds 3 --limit 20 --trust-remote-code --tensor-parallel-size 2
"""

import argparse
import json
import re
from typing import Optional

from multi_agent_debate import (
    SimulatorInterface, DebateRole, parse_plan_from_response,
    ROBOT_DESCRIPTION, ACTION_DESCRIPTION, AGENT_AVAIL_ACTIONS,
)
from evaluate_viki_l2_localmodel import LocalVLMInterface
from single_agent_common import (
    add_common_args, run_evaluation,
)


# ─── MAD prompts, adapted from config4all.json for VIKI-L2 plans ──────
# 3-embodiment note: the robot block is now built DYNAMICALLY from the
# task's robots dict (any number of robots), so R3 is no longer dropped.

_MAD_ROBOT_BLOCK_TEMPLATE = """\
## Available robots and their action primitives
{robot_lines}

## VIKI-Bench action primitives and rules
Each action is a JSON list `[action_type, target_object_or_location]`. \
**Each robot can only perform ONE action per time step.** Every step must include an \
entry for EVERY robot (use `["Wait"]` for idle).
Action primitives and descriptions:
{action_descriptions}
Use exact object and location names from the task. Do not invent new entity names. Choose \
the primitive that advances the current object state (Move → Reach → Grasp → Move → Place \
is the canonical pick-and-place sequence; Interact activates appliances).
  - If the robot is at/near the object but has not reached it, use Reach on the object.
  - If the robot has reached the object and is not carrying it, use Grasp on the object.
  - If the robot is carrying an object at the target area, use Place on the target location.
  - Robot panda cannot move, if it is already at the object area, it can directly Reach or Grasp the object.
  - Robot unitree_go2 and anymal_c can move to the cardboardbox and push it to the locations of other robots like panda.
  - In most tasks, cardboardbox plays the role of a carrier, anymal_c/unitree_go2 can push it to other robots' locations, panda can take things out of or put things in a cardboardbox. 
  - Note that the robot should not always wait throughout all steps."""


DEBATER_META_PROMPT = """\
You are a debater. Hello and welcome to the debate about the correct joint plan for a \
multi-robot task in VIKI-Bench. There are TWO debaters (Affirmative and Negative). It's \
not necessary to fully agree with each other's perspectives — the objective is to find \
the CORRECT plan through argument.

You must first analyze the image to understand the scene, then analyze the task, then \
produce or defend your plan. Your reasoning must strictly adhere to the visual content \
and the task description — no assumptions, hypotheses, or guesses.

{robot_block}

## Required output format
Every response you give MUST include a plan in this exact JSON schema at the end:
```json
{{
  "reasoning": "step-by-step chain of thought and rebuttal...",
  "steps": [
    {{"step": 1, "actions": {{"R1": ["Move", "pumpkin"], "R2": ["Move", "apple"], "R3": ["Move", "box"]}}}},
    {{"step": 2, "actions": {{"R1": ["Reach", "pumpkin"], "R2": ["Reach", "apple"], "R3": ["Reach", "box"]}}}}
  ]
}}
```
Rules:
- `step` starts at 1 and increments sequentially.
- `actions` maps robot id ("R1", "R2", "R3") to `[action_type, target_or_location]`.
- Only use action primitives in that robot's "Available actions" list above.
- Every step MUST include an entry for every robot (use `["Wait"]` for idle).
"""


MODERATOR_META_PROMPT = """\
You are a moderator. There will be two debaters (Affirmative and Negative) presenting \
their joint plans for a multi-robot task in VIKI-Bench and discussing their perspectives. \
At the end of each round you evaluate the plans and decide whether there is a clear \
preference for one side. If yes, the debate ends. If no, it continues to the next round.

You have access to the same scene image, robot info and action rules as the debaters, \
and can independently judge plan feasibility.

{robot_block}
"""


JUDGE_META_PROMPT = """\
You are the final judge. Two debaters (Affirmative and Negative) argued about the correct \
joint plan for a multi-robot task in VIKI-Bench, but the moderator did not reach a decision \
within the round limit. Your job is to pick the better plan and end the debate.

You have access to the same scene image, robot info and action rules as the debaters.

{robot_block}
"""


# ─── Turn-level user prompts ──────────────────────────────────────────

_TASK_INTRO = """\
## Task
Look at the scene image. Here is the task description:
{task_description}

## Initial world state (from the symbolic simulator)
{world_state}
"""


AFFIRMATIVE_PROMPT = _TASK_INTRO + """
## Your task (Affirmative)
Propose a complete joint plan that controls ALL robots to accomplish the task. Explain \
your reasoning, then output the plan in the required JSON format.
"""


NEGATIVE_PROMPT_R1 = _TASK_INTRO + """
## The Affirmative side just proposed
{aff_ans}

## Your task (Negative)
You DISAGREE with the Affirmative's plan above. Propose YOUR OWN alternative joint plan \
and explain concretely WHY the Affirmative's plan is wrong or worse than yours (e.g. a \
broken pre-condition, a coordination conflict, a missing setup step). Output your plan \
in the required JSON format.
"""


DEBATE_PROMPT = """\
## Your opponent just said
{oppo_ans}

## Your task
Do you agree with your opponent's perspective? Provide your reasons and either DEFEND your \
previous plan (with a refined version) or CONCEDE and adopt an updated plan. Either way, \
end your reply with your (possibly revised) plan in the required JSON format.
"""


MODERATOR_PROMPT = _TASK_INTRO + """
## The current debate — Round {round_num}

### Affirmative side arguing
{aff_ans}

### Negative side arguing
{neg_ans}

## Your task
Evaluate both sides' plans and determine if there is a clear preference for one of them. \
If YES, briefly justify the winning side and the debate will conclude. If NOT, output \
"preference": "No" and the debate will continue to the next round.

Respond with strict JSON only (no other text):
```json
{{
  "preference": "Yes" or "No",
  "supported_side": "Affirmative" or "Negative" or "",
  "reason": "why you prefer that side, or why no clear preference yet"
}}
```
"""


JUDGE_PROMPT_1 = _TASK_INTRO + """
## Affirmative side's final position
{aff_ans}

## Negative side's final position
{neg_ans}

## Your task
Concisely summarize the answer candidates from both sides — the KEY DIFFERENCES between \
the two plans without your own reasons yet.
"""


JUDGE_PROMPT_2 = """\
Now, given the two candidates you just summarized, decide which side has the more correct \
plan for the task. Output ONLY strict JSON:
```json
{{"winning_side": "Affirmative" or "Negative", "reason": "brief"}}
```
"""


# ─── Render + parse helpers ───────────────────────────────────────────

def _render_robot_block(robots: dict) -> str:
    """Build the shared robot/action block used by all four roles.
    Enumerates ALL robots in the task (R1, R2, R3, …), not just R1/R2."""
    robot_lines = []
    relevant = set()
    for rid, rname in robots.items():
        if not rname:
            continue
        relevant.update(AGENT_AVAIL_ACTIONS.get(rname, []))
        robot_lines.append(
            f"- {rid} ({rname}): {ROBOT_DESCRIPTION.get(rname, '')}\n"
            f"  Available actions: {AGENT_AVAIL_ACTIONS.get(rname, [])}"
        )
    action_block = "\n".join(f"- {ACTION_DESCRIPTION[a]}"
                             for a in ACTION_DESCRIPTION if a in relevant)
    return _MAD_ROBOT_BLOCK_TEMPLATE.format(
        robot_lines="\n".join(robot_lines),
        action_descriptions=action_block,
    )


def _render_system(template: str, robots: dict) -> str:
    return template.format(robot_block=_render_robot_block(robots))


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_blob(text: str) -> Optional[str]:
    """Return the first JSON-looking blob: fenced ```json ...``` if
    present, else the first {...} span. None if nothing looks like JSON."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    return text[s:e]


def _parse_moderator_verdict(response: str) -> dict:
    """{'preference': 'Yes'/'No', 'supported_side': 'Aff'/'Neg'/'',
        'reason': str}. Defaults to 'No preference' on any parse failure —
    this errs toward continuing the debate, which matches the paper's
    conservative behavior on ambiguous moderator output."""
    blob = _extract_json_blob(response)
    if blob is None:
        return {"preference": "No", "supported_side": "", "reason": "parse-failed"}
    try:
        d = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return {"preference": "No", "supported_side": "", "reason": "parse-failed"}
    if not isinstance(d, dict):
        return {"preference": "No", "supported_side": "", "reason": "parse-failed"}
    return {
        "preference":     str(d.get("preference",     "No")).strip(),
        "supported_side": str(d.get("supported_side", "")).strip(),
        "reason":         str(d.get("reason",         "")).strip(),
    }


def _parse_judge_verdict(response: str) -> str:
    """Return 'Affirmative' or 'Negative' from the judge's final pick.
    Defaults to 'Affirmative' on parse failure (arbitrary but stable)."""
    blob = _extract_json_blob(response)
    if blob is None:
        return "Affirmative"
    try:
        d = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return "Affirmative"
    if not isinstance(d, dict):
        return "Affirmative"
    side = str(d.get("winning_side", "")).strip().lower()
    return "Negative" if side.startswith("neg") else "Affirmative"


# ─── Per-task strategy ────────────────────────────────────────────────

def run_one_task(task, llm, sampling_params, args, logger, stats):
    robots = task["scene_config"]["robots"]
    sim = SimulatorInterface(scene_seed=0)

    # 3 roles share one LLM instance; each gets its own system prompt.
    aff = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    neg = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM2_R2_ADVOCATE, logger=logger, stats=stats,
    )
    # For the moderator + judge we re-use one of the DebateRole enums for
    # log-file naming; the semantic role is set via system prompt.
    mod = LocalVLMInterface(
        llm=llm, sampling_params=sampling_params, model_name=args.model_path,
        role=DebateRole.VLM1_R1_ADVOCATE, logger=logger, stats=stats,
    )
    aff.set_system_prompt(_render_system(DEBATER_META_PROMPT,   robots))
    neg.set_system_prompt(_render_system(DEBATER_META_PROMPT,   robots))
    mod.set_system_prompt(_render_system(MODERATOR_META_PROMPT, robots))

    try:
        world_state = sim.get_initial_world_state(task["scene_config"])
    except Exception as e:
        print(f"  [WARN] could not render initial world state: {e}")
        world_state = ""
    img = task["image_path"]
    intro_kwargs = dict(
        task_description=task["task_description"],
        world_state=world_state or "(unavailable)",
    )

    # ── Round 1: opening ──
    print("\n--- MAD round 1 (opening) ---")
    aff_response = aff.query(
        AFFIRMATIVE_PROMPT.format(**intro_kwargs),
        image_path=img,
    )
    neg_response = neg.query(
        NEGATIVE_PROMPT_R1.format(aff_ans=aff_response, **intro_kwargs),
        image_path=img,
    )
    aff_plan = parse_plan_from_response(aff_response)
    neg_plan = parse_plan_from_response(neg_response)

    # Robustness: if one side didn't produce a parseable plan on turn 1,
    # seed with the other side's plan so we always have SOMETHING to
    # execute at the end. Neither parseable = we fail cleanly below.
    if aff_plan is None and neg_plan is not None:
        aff_plan = neg_plan
    if neg_plan is None and aff_plan is not None:
        neg_plan = aff_plan

    winner_side: Optional[str] = None
    total_rounds = 0

    # ── Debate loop ──
    for round_idx in range(1, args.max_rounds + 1):
        total_rounds = round_idx

        # Moderator evaluates the current round's positions.
        mod_response = mod.query(
            MODERATOR_PROMPT.format(
                round_num=round_idx,
                aff_ans=aff_response,
                neg_ans=neg_response,
                **intro_kwargs,
            ),
            image_path=img,
        )
        verdict = _parse_moderator_verdict(mod_response)
        pref_yes = verdict["preference"].lower().startswith("y")
        side     = verdict["supported_side"].lower()
        print(f"  [MOD round {round_idx}] preference={verdict['preference']}  "
              f"side={verdict['supported_side'] or '—'}")

        if pref_yes and side.startswith("aff"):
            winner_side = "Affirmative"
            break
        if pref_yes and side.startswith("neg"):
            winner_side = "Negative"
            break

        # No preference. Continue unless we've exhausted rounds.
        if round_idx == args.max_rounds:
            break

        # Rebuttal: aff sees neg's LATEST, then neg sees aff's UPDATED
        # response (sequential — matches upstream MAD).
        print(f"\n--- MAD round {round_idx + 1} (rebuttal) ---")
        aff_response = aff.query(
            DEBATE_PROMPT.format(oppo_ans=neg_response),
            image_path=img,
        )
        neg_response = neg.query(
            DEBATE_PROMPT.format(oppo_ans=aff_response),
            image_path=img,
        )
        p_aff = parse_plan_from_response(aff_response)
        p_neg = parse_plan_from_response(neg_response)
        if p_aff is not None:
            aff_plan = p_aff
        if p_neg is not None:
            neg_plan = p_neg

    # ── Judge fallback ──
    if winner_side is None:
        print("  [JUDGE] moderator did not converge; invoking final judge")
        judge = LocalVLMInterface(
            llm=llm, sampling_params=sampling_params, model_name=args.model_path,
            role=DebateRole.VLM2_R2_ADVOCATE, logger=logger, stats=stats,
        )
        judge.set_system_prompt(_render_system(JUDGE_META_PROMPT, robots))
        summary = judge.query(
            JUDGE_PROMPT_1.format(
                aff_ans=aff_response, neg_ans=neg_response, **intro_kwargs,
            ),
            image_path=img,
        )
        # Inline the summary into step 2's prompt so this call is stateless
        # (matches the rest of the codebase's stateless VLM contract).
        pick_response = judge.query(
            f"You previously summarized:\n{summary}\n\n" + JUDGE_PROMPT_2,
            image_path=img,
        )
        winner_side = _parse_judge_verdict(pick_response)
        print(f"  [JUDGE] picked {winner_side}")

    # ── Execute the winning side's LATEST plan ──
    winning_plan = aff_plan if winner_side == "Affirmative" else neg_plan
    if winning_plan is None:
        return {
            "success":                False,
            "debate_loops":           1,
            "debate_rounds_per_loop": [total_rounds],
            "last_failure":           "no parseable plan from either side",
            "mad_winner_side":        winner_side,
            "mad_rounds_run":         total_rounds,
        }

    try:
        exec_result = sim.execute_plan(winning_plan, task["scene_config"])
    except Exception as e:
        return {
            "success":                False,
            "debate_loops":           1,
            "debate_rounds_per_loop": [total_rounds],
            "last_failure":           f"simulator crash: {e}",
            "mad_winner_side":        winner_side,
            "mad_rounds_run":         total_rounds,
        }

    success = bool(exec_result.get("success"))
    last_failure = None if success else exec_result.get("failure_reason")
    print(f"  [SIM] winner={winner_side}: "
          f"{'SUCCESS' if success else 'FAIL — ' + str(last_failure).splitlines()[0]}")

    return {
        "success":                success,
        "debate_loops":           1,               # MAD is one-shot
        "debate_rounds_per_loop": [total_rounds],  # rounds until moderator/judge picked
        "last_failure":           last_failure,
        # MAD-specific fields for downstream analysis
        "mad_winner_side":        winner_side,
        "mad_rounds_run":         total_rounds,
        "mad_used_judge_fallback": winner_side is not None
                                    and total_rounds == args.max_rounds,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Agents Debate (MAD, Liang et al. EMNLP 2024) on "
                    "VIKI-L2 3-robot (R1+R2+R3) tasks")
    add_common_args(p)
    p.add_argument("--max-rounds", type=int, default=3,
                   help="max debate rounds before invoking the fallback "
                        "judge (default 3, matches the paper's config).")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(args, strategy_label="mad", run_one_task=run_one_task)


if __name__ == "__main__":
    main()
