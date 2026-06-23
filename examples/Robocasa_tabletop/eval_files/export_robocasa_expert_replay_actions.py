"""Export raw RoboCasa GR1 expert actions for simulator replay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(
    "/home/zhangfeihong/playground/Data/RoboCasa-GR1/"
    "PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot"
)


def _env_task_name(env_name: str) -> str:
    task = env_name.split("/", 1)[-1]
    return re.sub(r"_GR1.*$", "", task)


def _dataset_path(env_name: str) -> Path:
    return DATA_ROOT / f"gr1_unified.{_env_task_name(env_name)}"


def _episode_rows(dataset_path: Path) -> list[dict]:
    rows = []
    with (dataset_path / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _trajectory_seed(row: dict) -> int:
    match = re.search(r"-(\d+)$", str(row.get("trajectory_id", "")))
    if match:
        return int(match.group(1))
    return int(row["episode_index"])


def _to_gr1_29(action44: np.ndarray) -> np.ndarray:
    action44 = np.asarray(action44, dtype=np.float32)
    if action44.ndim != 2 or action44.shape[1] != 44:
        raise ValueError(f"Expected raw action [T, 44], got {action44.shape}.")
    return np.concatenate(
        [
            action44[:, 0:7],    # left_arm
            action44[:, 22:29],  # right_arm
            action44[:, 7:13],   # left_hand
            action44[:, 29:35],  # right_hand
            action44[:, 41:44],  # waist
        ],
        axis=1,
    ).astype(np.float32)


def _load_episode_actions(dataset_path: Path, episode_index: int) -> np.ndarray:
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        # The current local release uses v2-style one parquet per episode under
        # chunk-000, but keep a glob fallback for robustness.
        matches = sorted((dataset_path / "data").glob(f"**/episode_{episode_index:06d}.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not find parquet for episode {episode_index}: {matches}")
        parquet_path = matches[0]
    df = pd.read_parquet(parquet_path)
    action44 = np.stack(df["action"].to_numpy()).astype(np.float32)
    return _to_gr1_29(action44)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start_episode", type=int, default=0)
    args = parser.parse_args()

    dataset_path = _dataset_path(args.env_name)
    rows = _episode_rows(dataset_path)[args.start_episode : args.start_episode + args.episodes]
    records = []
    for row in rows:
        episode_index = int(row["episode_index"])
        expert = _load_episode_actions(dataset_path, episode_index)
        records.append(
            {
                "env_name": args.env_name,
                "dataset_name": dataset_path.name,
                "trajectory_id": episode_index,
                "trajectory_length": int(len(expert)),
                "seed": _trajectory_seed(row),
                "expert": expert,
                "mean_abs_delta": 0.0,
                "max_abs_delta": 0.0,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, records=np.asarray(records, dtype=object))
    print(json.dumps([{k: v for k, v in r.items() if k != "expert"} for r in records], indent=2))
    print(f"Wrote {len(records)} expert replay records to {args.output}")


if __name__ == "__main__":
    main()
