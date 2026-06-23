"""Action-only RoboTwin 2.0 raw zip dataset for VAR Stage 1 training."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from starVLA.utils.action_spec import ActionSpec


ROBOTWIN_TASKS_50 = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


ACTION_KEYS = ["action.left_joints", "action.left_gripper", "action.right_joints", "action.right_gripper"]
ACTION_KEY_DIMS = {
    "action.left_joints": 6,
    "action.left_gripper": 1,
    "action.right_joints": 6,
    "action.right_gripper": 1,
}
STATE_KEYS = ["state.left_joints", "state.left_gripper", "state.right_joints", "state.right_gripper"]
STATE_KEY_DIMS = {
    "state.left_joints": 6,
    "state.left_gripper": 1,
    "state.right_joints": 6,
    "state.right_gripper": 1,
}
GRIPPER_DIMS = [6, 13]
JOINT_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


@dataclass(frozen=True)
class _EpisodeRecord:
    task_name: str
    split: str
    zip_path: Path
    member_name: str
    actions_raw: np.ndarray


def _as_list(value: Any, *, default: list[str]) -> list[str]:
    if value in (None, "all"):
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _member_episode_index(member_name: str) -> int:
    stem = Path(member_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _read_action_member(zip_file: zipfile.ZipFile, member_name: str, action_key: str) -> np.ndarray:
    payload = zip_file.read(member_name)
    with h5py.File(io.BytesIO(payload), "r") as handle:
        if action_key not in handle:
            raise KeyError(f"Action key {action_key!r} not found in {member_name}.")
        actions = np.asarray(handle[action_key], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected RoboTwin action shape [T, 14], got {actions.shape} in {member_name}.")
    return actions


class RoboTwinRawStage1ActionDataset(Dataset):
    """Preload action chunks from official RoboTwin 2.0 zip files.

    The official release stores per-task zip files under
    ``dataset/<task>/<embodiment>_<split>_<episodes>.zip``.  Stage 1 only needs
    the 14D expert actions, so this dataset reads ``/joint_action/vector`` once
    at construction time and trains from in-memory normalized action windows.
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        *,
        splits: list[str] | tuple[str, ...] = ("clean",),
        embodiment: str = "aloha-agilex",
        task_names: list[str] | tuple[str, ...] | str | None = None,
        action_key: str = "/joint_action/vector",
        horizon: int = 50,
        action_dim: int = 14,
        max_episodes_per_zip: int | None = None,
        binary_threshold: float = 0.49,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.dataset_dir = self.data_root_dir / "dataset"
        self.splits = [str(item).lower() for item in splits]
        self.embodiment = str(embodiment)
        self.task_names = _as_list(task_names, default=ROBOTWIN_TASKS_50)
        self.action_key = str(action_key)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.max_episodes_per_zip = None if max_episodes_per_zip is None else int(max_episodes_per_zip)
        self.binary_threshold = float(binary_threshold)

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}.")
        if self.action_dim != 14:
            raise ValueError(f"RoboTwin raw Stage 1 currently expects action_dim=14, got {self.action_dim}.")
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"RoboTwin dataset directory not found: {self.dataset_dir}")

        self.episodes = self._load_episodes()
        self.raw_min, self.raw_max = self._compute_minmax()
        self._windows = self._build_windows()
        self.action_spec = ActionSpec(
            action_dim=14,
            horizon=self.horizon,
            action_keys=list(ACTION_KEYS),
            state_keys=list(STATE_KEYS),
            action_key_dims=dict(ACTION_KEY_DIMS),
            state_key_dims=dict(STATE_KEY_DIMS),
            dim_groups={"gripper": list(GRIPPER_DIMS)},
            normalization_modes={
                "action.left_joints": "min_max",
                "action.right_joints": "min_max",
                "action.left_gripper": "binary",
                "action.right_gripper": "binary",
            },
            source="robotwin2_raw_zip",
            metadata={
                "data_root_dir": str(self.data_root_dir),
                "splits": list(self.splits),
                "embodiment": self.embodiment,
                "action_key": self.action_key,
                "task_count": len(self.task_names),
                "episode_count": len(self.episodes),
            },
        )

    def _zip_for(self, task_name: str, split: str) -> Path:
        task_dir = self.dataset_dir / task_name
        matches = sorted(task_dir.glob(f"{self.embodiment}_{split}_*.zip"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {self.embodiment}_{split}_*.zip under {task_dir}, got {len(matches)}."
            )
        return matches[0]

    def _load_episodes(self) -> list[_EpisodeRecord]:
        episodes: list[_EpisodeRecord] = []
        total_zips = len(self.task_names) * len(self.splits)
        loaded_zips = 0
        for task_name in self.task_names:
            for split in self.splits:
                zip_path = self._zip_for(task_name, split)
                loaded_zips += 1
                print(
                    f"[robotwin_raw] loading {loaded_zips}/{total_zips}: "
                    f"task={task_name} split={split} zip={zip_path.name}",
                    flush=True,
                )
                with zipfile.ZipFile(zip_path) as archive:
                    members = sorted(
                        [name for name in archive.namelist() if name.endswith(".hdf5") and "/data/" in name],
                        key=_member_episode_index,
                    )
                    if self.max_episodes_per_zip is not None:
                        members = members[: self.max_episodes_per_zip]
                    if not members:
                        raise RuntimeError(f"No HDF5 episodes found in {zip_path}.")
                    for member_name in members:
                        actions_raw = _read_action_member(archive, member_name, self.action_key)
                        episodes.append(
                            _EpisodeRecord(
                                task_name=task_name,
                                split=split,
                                zip_path=zip_path,
                                member_name=member_name,
                                actions_raw=actions_raw,
                            )
                        )
                print(f"[robotwin_raw] loaded episodes so far: {len(episodes)}", flush=True)
        if not episodes:
            raise RuntimeError("No RoboTwin episodes loaded.")
        print(f"[robotwin_raw] finished loading {len(episodes)} episodes", flush=True)
        return episodes

    def _compute_minmax(self) -> tuple[np.ndarray, np.ndarray]:
        raw_min = np.full((self.action_dim,), np.inf, dtype=np.float32)
        raw_max = np.full((self.action_dim,), -np.inf, dtype=np.float32)
        for episode in self.episodes:
            raw_min = np.minimum(raw_min, episode.actions_raw.min(axis=0))
            raw_max = np.maximum(raw_max, episode.actions_raw.max(axis=0))
        if not np.isfinite(raw_min).all() or not np.isfinite(raw_max).all():
            raise RuntimeError("Failed to compute finite RoboTwin action min/max statistics.")
        return raw_min, raw_max

    def _build_windows(self) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            max_start = int(episode.actions_raw.shape[0]) - self.horizon + 1
            for base_index in range(max(0, max_start)):
                windows.append((episode_index, base_index))
        if not windows:
            raise RuntimeError(f"No complete RoboTwin action windows found for horizon={self.horizon}.")
        print(f"[robotwin_raw] built {len(windows)} complete action windows", flush=True)
        return windows

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(actions, dtype=np.float32)
        denom = self.raw_max - self.raw_min
        valid = denom != 0
        normalized[:, valid] = 2.0 * (actions[:, valid] - self.raw_min[valid]) / denom[valid] - 1.0
        normalized[:, ~valid] = 0.0
        normalized[:, GRIPPER_DIMS] = (actions[:, GRIPPER_DIMS] > self.binary_threshold).astype(np.float32)
        return normalized

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, base_index = self._windows[index]
        episode = self.episodes[episode_index]
        raw = episode.actions_raw[base_index : base_index + self.horizon].astype(np.float32, copy=False)
        actions = self._normalize(raw)
        return {
            "actions": torch.from_numpy(actions),
            "actions_raw": torch.from_numpy(raw.copy()),
            "metadata": {
                "task_name": episode.task_name,
                "split": episode.split,
                "zip_path": str(episode.zip_path),
                "member_name": episode.member_name,
                "episode_index": int(episode_index),
                "base_index": int(base_index),
                "window_mode": "full",
            },
        }
