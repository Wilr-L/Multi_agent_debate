"""
viki_loader
===========
Load VIKI-Bench L2 task-planning samples from the parquet files shipped in
`VIKI_data/viki/VIKI-L*/test.parquet`, normalizing the ground-truth schema
(numpy arrays, stringified Python literals) into plain Python objects that
`MultiAgentDebateEngine` and `SimulatorInterface` consume directly.
"""

import ast
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─── Type / value normalization ────────────────────────────────────────

def _denumpy(obj):
    """
    Recursively convert numpy arrays/scalars to native Python; opportunistically
    `ast.literal_eval` any string that looks like a Python list/dict literal —
    several VIKI fields (e.g. `goal_constraints`) are stored as the *string*
    representation of a Python object inside the parquet.
    """
    if isinstance(obj, np.ndarray):
        return [_denumpy(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _denumpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_denumpy(x) for x in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("[", "{")):
            try:
                return _denumpy(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return obj
        return obj
    return obj


def _filter_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _save_image_bytes(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Write inline image bytes to a temp file and return its absolute path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        f.write(image_bytes)
    finally:
        f.close()
    return f.name


# ─── Public API ────────────────────────────────────────────────────────

def load_viki_task(parquet_path, idx: int) -> dict:
    """
    Read row `idx` from a VIKI-L2 test parquet and assemble it into the shape
    `MultiAgentDebateEngine.run()` wants:

        {
            "task_description":  str,
            "image_path":        str,   # path to a temp .jpg
            "scene_config": {
                "robots":               {"R1": "<type>", "R2": "<type>"},
                "init_pos":             {"<asset>_<i>": [<pos>, ...]},
                "goal_constraints":     [...],
                "temporal_constraints": [...],
            },
            "task_id":     str,
            "task_name":   str,
            "idle_robots": list,
        }
    """
    df = pd.read_parquet(str(parquet_path))
    row = df.iloc[idx]
    gt = _denumpy(row["reward_model"]["ground_truth"])

    # Task description: the parquet's user message wraps the task as
    # "<image>actual task text"; strip the placeholder.
    user_msg = row["prompt"][-1]["content"]
    task_description = user_msg.replace("<image>", "").strip()
    if not task_description:
        task_description = gt.get("description", "")

    # Image: bytes embedded in parquet → write to a temp .jpg the VLM client
    # can read like any other local path.
    image_path = _save_image_bytes(row["images"][0]["bytes"])

    scene_config = {
        "robots":              _filter_none(gt.get("robots", {})),
        "init_pos":            _filter_none(gt.get("init_pos", {})),
        "goal_constraints":    gt.get("goal_constraints", []) or [],
        "temporal_constraints": gt.get("temporal_constraints", []) or [],
    }

    return {
        "task_description": task_description,
        "image_path":       image_path,
        "scene_config":     scene_config,
        "task_id":          gt.get("task_id", ""),
        "task_name":        gt.get("task_name", ""),
        "idle_robots":      gt.get("idle_robots", []),
    }


def find_task_indices(
    parquet_path,
    n_robots: int = 2,
    required_ids: Optional[tuple[str, ...]] = ("R1", "R2"),
    limit: Optional[int] = None,
) -> list[int]:
    """
    Return row indices where exactly `n_robots` are active AND every id in
    `required_ids` is non-null. Useful for filtering the 878 two-robot tasks
    down to the R1+R2 subset our two-VLM debate engine expects.
    """
    df = pd.read_parquet(str(parquet_path))
    out = []
    for i, robots in enumerate(df["reward_model"].map(lambda r: r["ground_truth"]["robots"])):
        active = sum(1 for v in robots.values() if v is not None)
        if active != n_robots:
            continue
        if required_ids and any(robots.get(rid) is None for rid in required_ids):
            continue
        out.append(i)
        if limit and len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    # Quick CLI sanity check.
    pq = Path(__file__).parent / "VIKI_data" / "viki" / "VIKI-L2" / "test.parquet"
    indices = find_task_indices(pq, n_robots=2, required_ids=("R1", "R2"), limit=5)
    print("first 5 R1+R2 task indices:", indices)
    task = load_viki_task(pq, indices[0])
    print()
    print("idx       :", indices[0])
    print("task_id   :", task["task_id"])
    print("task_name :", task["task_name"])
    print("robots    :", task["scene_config"]["robots"])
    print("init_pos  :", task["scene_config"]["init_pos"])
    print("goal_cstr :", task["scene_config"]["goal_constraints"])
    print("description:", task["task_description"])
    print("image     :", task["image_path"])
