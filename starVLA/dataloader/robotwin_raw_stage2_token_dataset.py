"""RoboTwin 2.0 raw-zip Stage 2 dataset for VAR action-token policy training."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.robotwin_raw_stage1_action_dataset import (
    GRIPPER_DIMS,
    JOINT_DIMS,
    RoboTwinRawStage1ActionDataset,
)
from starVLA.model.modules.action_tokenizer import Stage1Artifact, load_frozen_var_action_tokenizer


CAMERA_KEYS = (
    "observation/head_camera/rgb",
    "observation/left_camera/rgb",
    "observation/right_camera/rgb",
)

# Convert raw RoboTwin vector order L,LG,R,RG to StarVLA RoboTwin DataConfig
# order L,R,LG,RG, which is what PolicyNormProcessor and the eval client expect.
ROBOTWIN_RAW_TO_STARVLA_ACTION_ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def robotwin_task_to_instruction(task_name: str) -> str:
    return task_name.replace("_", " ")


def _decode_rgb(payload: Any) -> Image.Image:
    if isinstance(payload, np.ndarray):
        payload = payload.item()
    if isinstance(payload, np.void):
        payload = bytes(payload)
    if not isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload)
    return Image.open(io.BytesIO(payload)).convert("RGB")


class RoboTwinRawStage2TokenDataset(Dataset):
    """Expose RoboTwin raw-zip observations with frozen Stage 1 token labels.

    Stage 1 was trained directly from official RoboTwin zip files, not from the
    LeRobot conversion.  This dataset keeps Stage 2 labels on that same action
    normalization and tokenizer contract while reading RGB observations from the
    raw HDF5 members.
    """

    def __init__(
        self,
        stage1_cfg: Any,
        *,
        stage1_artifact: Stage1Artifact | None = None,
        stage1_artifact_path: str | Path | None = None,
        token_cache_path: str | Path | None = None,
        validate_cache_online: bool = False,
        mode: str = "train",
        device: str | torch.device = "cpu",
        max_samples: int | None = None,
        sample_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if mode != "train":
            raise ValueError(f"RoboTwin raw Stage 2 currently supports mode='train' only, got {mode!r}.")
        if stage1_artifact is None:
            if stage1_artifact_path is None:
                raise ValueError("Either stage1_artifact or stage1_artifact_path must be provided.")
            stage1_artifact = load_frozen_var_action_tokenizer(stage1_artifact_path, device=device)

        self.stage1_cfg = stage1_cfg
        self.stage1_artifact = stage1_artifact
        self.tokenizer = stage1_artifact.tokenizer
        self.token_cache = self._load_token_cache(token_cache_path) if token_cache_path is not None else None
        self.validate_cache_online = bool(validate_cache_online)
        self.sample_indices = None if sample_indices is None else [int(index) for index in sample_indices]
        self.max_samples = None if max_samples is None else max(0, int(max_samples))

        data_cfg = stage1_cfg.data
        self.stage1_dataset = RoboTwinRawStage1ActionDataset(
            data_cfg.data_root_dir,
            splits=list(data_cfg.get("splits", ["clean"])),
            embodiment=str(data_cfg.get("embodiment", "aloha-agilex")),
            task_names=data_cfg.get("task_names", "all"),
            action_key=str(data_cfg.get("raw_action_key", "/joint_action/vector")),
            horizon=int(data_cfg.get("expected_action_horizon", 50)),
            action_dim=int(data_cfg.get("expected_action_dim", 14)),
            max_episodes_per_zip=data_cfg.get("max_episodes_per_zip", None),
            binary_threshold=float(data_cfg.get("binary_threshold", 0.49)),
        )
        self.action_spec = self.stage1_dataset.action_spec
        self._validate_stage1_contract()
        self._length = len(self.stage1_dataset)
        if self.token_cache is not None:
            self._validate_token_cache()
        self._apply_subset()

    @property
    def token_dim(self) -> int:
        return int(self.tokenizer.token_dim)

    def __len__(self) -> int:
        return int(self._length)

    def _apply_subset(self) -> None:
        if self.sample_indices is not None:
            if any(index < 0 or index >= self._length for index in self.sample_indices):
                raise ValueError(f"sample_indices must be within [0, {self._length}).")
            self._length = len(self.sample_indices)
        if self.max_samples is not None:
            self._length = min(self._length, self.max_samples)

    def _source_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Stage2 index {index} outside [0, {len(self)}).")
        if self.sample_indices is None:
            return int(index)
        return int(self.sample_indices[int(index)])

    def _validate_stage1_contract(self) -> None:
        dataset_spec = self.stage1_dataset.action_spec
        artifact_spec = self.stage1_artifact.action_spec
        if dataset_spec.action_dim != artifact_spec.action_dim:
            raise ValueError(f"Action dim mismatch: dataset={dataset_spec.action_dim}, artifact={artifact_spec.action_dim}")
        if dataset_spec.horizon != artifact_spec.horizon:
            raise ValueError(f"Action horizon mismatch: dataset={dataset_spec.horizon}, artifact={artifact_spec.horizon}")
        if dataset_spec.action_keys != artifact_spec.action_keys:
            raise ValueError(f"Action key mismatch: dataset={dataset_spec.action_keys}, artifact={artifact_spec.action_keys}")

    def _load_token_cache(self, path: str | Path) -> dict[str, Any]:
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Stage 2 token cache not found: {cache_path}")
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache, dict):
            raise ValueError(f"Expected token cache to be a dict, got {type(cache).__name__}.")
        cache["path"] = str(cache_path)
        return cache

    def _validate_token_cache(self) -> None:
        assert self.token_cache is not None
        metadata = dict(self.token_cache.get("metadata", {}))
        if metadata.get("stage1_artifact_id") != self.stage1_artifact.artifact_id:
            raise ValueError(
                "Stage 2 token cache artifact mismatch: "
                f"cache={metadata.get('stage1_artifact_id')!r}, current={self.stage1_artifact.artifact_id!r}"
            )
        if metadata.get("stage1_checkpoint_sha256") is not None and metadata["stage1_checkpoint_sha256"] != self.stage1_artifact.checkpoint_sha256:
            raise ValueError(
                "Stage 2 token cache checkpoint hash mismatch: "
                f"cache={metadata['stage1_checkpoint_sha256']}, current={self.stage1_artifact.checkpoint_sha256}"
            )
        if int(metadata.get("token_dim", -1)) != self.token_dim:
            raise ValueError(f"Stage 2 token cache token_dim mismatch: cache={metadata.get('token_dim')}, current={self.token_dim}")
        tokens = self.token_cache.get("tokens")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"Stage 2 token cache tokens must have shape [N, {self.token_dim}], got {getattr(tokens, 'shape', None)}.")
        if tokens.shape[0] > len(self.stage1_dataset):
            raise ValueError(f"Stage 2 token cache is longer than source dataset: cache={tokens.shape[0]}, source={len(self.stage1_dataset)}")
        source_dataset_len = metadata.get("source_dataset_len")
        if source_dataset_len is not None and int(source_dataset_len) != len(self.stage1_dataset):
            raise ValueError(
                "Stage 2 token cache source_dataset_len mismatch: "
                f"cache={source_dataset_len}, current={len(self.stage1_dataset)}"
            )
        self._length = int(tokens.shape[0])

    @torch.no_grad()
    def _encode_actions(self, actions: np.ndarray | torch.Tensor) -> torch.LongTensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.to(device=next(self.tokenizer.parameters()).device, dtype=torch.float32)
        tokens = self.tokenizer.encode(actions.unsqueeze(0))[0].detach().cpu().long()
        if tokens.numel() != self.token_dim:
            raise ValueError(f"Expected {self.token_dim} Stage 1 tokens, got {tokens.numel()}.")
        return tokens

    def _read_images(self, zip_path: Path, member_name: str, base_index: int) -> list[Image.Image]:
        with zipfile.ZipFile(zip_path) as archive:
            payload = archive.read(member_name)
        with h5py.File(io.BytesIO(payload), "r") as handle:
            return [_decode_rgb(handle[key][base_index]) for key in CAMERA_KEYS]

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self._source_index(int(index))
        episode_index, base_index = self.stage1_dataset._windows[source_index]
        episode = self.stage1_dataset.episodes[episode_index]
        raw = episode.actions_raw[base_index : base_index + self.action_spec.horizon].astype(np.float32, copy=False)
        actions = self.stage1_dataset._normalize(raw)

        if self.token_cache is not None:
            action_tokens = self.token_cache["tokens"][source_index].long()
            if self.validate_cache_online:
                encoded_tokens = self._encode_actions(actions)
                if not torch.equal(action_tokens, encoded_tokens):
                    raise ValueError(f"Stage 2 token cache mismatch at index {source_index}.")
        else:
            action_tokens = self._encode_actions(actions)

        metadata = {
            "task_name": episode.task_name,
            "split": episode.split,
            "zip_path": str(episode.zip_path),
            "member_name": episode.member_name,
            "episode_index": int(episode_index),
            "base_index": int(base_index),
            "source_index": int(source_index),
            "window_mode": "full",
            "stage1_artifact_id": self.stage1_artifact.artifact_id,
        }
        return {
            "image": self._read_images(episode.zip_path, episode.member_name, int(base_index)),
            "lang": robotwin_task_to_instruction(episode.task_name),
            "action": actions.astype(np.float32),
            "action_tokens": action_tokens,
            "stage1_artifact_id": self.stage1_artifact.artifact_id,
            "metadata": metadata,
        }

    def dataset_statistics(self) -> dict[str, Any]:
        order = ROBOTWIN_RAW_TO_STARVLA_ACTION_ORDER
        action_min = self.stage1_dataset.raw_min[order].astype(np.float32)
        action_max = self.stage1_dataset.raw_max[order].astype(np.float32)
        action_mean = np.zeros_like(action_min, dtype=np.float32)
        action_std = np.ones_like(action_min, dtype=np.float32)
        state_min = action_min.copy()
        state_max = action_max.copy()
        return {
            "robotwin50": {
                "action": {
                    "min": action_min.tolist(),
                    "max": action_max.tolist(),
                    "mean": action_mean.tolist(),
                    "std": action_std.tolist(),
                    "q01": action_min.tolist(),
                    "q99": action_max.tolist(),
                    "mask": np.ones_like(action_min, dtype=bool).tolist(),
                },
                "state": {
                    "min": state_min.tolist(),
                    "max": state_max.tolist(),
                    "mean": action_mean.tolist(),
                    "std": action_std.tolist(),
                    "q01": state_min.tolist(),
                    "q99": state_max.tolist(),
                    "mask": np.ones_like(state_min, dtype=bool).tolist(),
                },
                "num_trajectories": len(self.stage1_dataset.episodes),
                "num_transitions": len(self.stage1_dataset),
            }
        }
