"""Export RoboCasa expert, FAST, and Stage-1-decoded actions for simulator replay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import load_frozen_var_action_tokenizer
from starVLA.training.train_var_stage1 import load_starvla_base_config


def _env_task_name(env_name: str) -> str:
    task = env_name.split("/", 1)[-1]
    return re.sub(r"_GR1.*$", "", task)


def _env_name_for_dataset(dataset: Any) -> str:
    dataset_name = str(dataset.dataset_name)
    task_name = dataset_name.split(".", 1)[-1]
    return f"gr1_unified/{task_name}_GR1ArmsAndWaistFourierHands_Env"


def _dataset_for_env(stage1_dataset: VARStage1ActionDataset, env_name: str) -> Any:
    target = "gr1_unified." + _env_task_name(env_name)
    matches = [dataset for dataset in stage1_dataset.source_dataset.datasets if str(dataset.dataset_name) == target]
    if len(matches) != 1:
        names = [str(dataset.dataset_name) for dataset in stage1_dataset.source_dataset.datasets]
        raise ValueError(f"Expected one dataset named {target}, got {len(matches)} from {names}.")
    return matches[0]


def _concat_action_dict(data: dict[str, Any], action_keys: list[str]) -> np.ndarray:
    values = []
    for key in action_keys:
        value = data[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(values, axis=1)


def _split_action_array(actions: np.ndarray, action_keys: list[str], action_dims: dict[str, int]) -> dict[str, torch.Tensor]:
    data: dict[str, torch.Tensor] = {}
    start = 0
    for key in action_keys:
        dim = int(action_dims[key])
        data[key] = torch.as_tensor(actions[:, start : start + dim], dtype=torch.float32)
        start += dim
    if start != actions.shape[1]:
        raise ValueError(f"Split consumed {start} dims from action array with shape {actions.shape}.")
    return data


def _unnormalize_actions(
    *,
    dataset: Any,
    normalized_actions: np.ndarray,
) -> np.ndarray:
    action_keys = dataset.modality_keys["action"]
    action_dims: dict[str, int] = {}
    for key in action_keys:
        meta_key = key.split(".", 1)[-1]
        if key in dataset.metadata.modalities.action:
            action_dims[key] = int(dataset.metadata.modalities.action[key].shape[0])
        elif meta_key in dataset.metadata.modalities.action:
            action_dims[key] = int(dataset.metadata.modalities.action[meta_key].shape[0])
        else:
            raise KeyError(f"Could not find action metadata for {key!r} or {meta_key!r}.")
    split = _split_action_array(normalized_actions, action_keys, action_dims)
    raw = dataset.transforms.unapply(split)
    return _concat_action_dict(raw, action_keys).astype(np.float32)


def _trajectory_seed(dataset: Any, trajectory_id: int) -> int:
    try:
        with (dataset.dataset_path / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if int(item["episode_index"]) == int(trajectory_id):
                    match = re.search(r"-(\d+)$", str(item.get("trajectory_id", "")))
                    if match:
                        return int(match.group(1))
    except FileNotFoundError:
        pass
    return int(trajectory_id)


def _get_actions(
    *,
    stage1_dataset: VARStage1ActionDataset,
    dataset: Any,
    trajectory_id: int,
    trajectory_length: int,
    model: torch.nn.Module,
    device: torch.device,
    fast_tokenizer: Any | None = None,
) -> dict[str, np.ndarray]:
    horizon = int(stage1_dataset.action_spec.horizon)
    expert_norm_chunks: list[np.ndarray] = []
    expert_raw_chunks: list[np.ndarray] = []
    decoded_norm_chunks: list[np.ndarray] = []
    decoded_raw_chunks: list[np.ndarray] = []
    fast_norm_chunks: list[np.ndarray] = []
    fast_raw_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for base_index in range(0, int(trajectory_length), horizon):
            raw_data = stage1_dataset._get_action_only_data(dataset, int(trajectory_id), int(base_index))
            raw = _concat_action_dict(raw_data, dataset.modality_keys["action"])
            transformed = dataset.transforms(dict(raw_data))
            normalized = _concat_action_dict(transformed, dataset.modality_keys["action"])
            normalized_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
            recon_norm = model(normalized_tensor)["recon"][0].detach().cpu().numpy()
            recon_raw = _unnormalize_actions(dataset=dataset, normalized_actions=recon_norm)
            expert_norm_chunks.append(normalized)
            expert_raw_chunks.append(raw)
            decoded_norm_chunks.append(recon_norm)
            decoded_raw_chunks.append(recon_raw)
            if fast_tokenizer is not None:
                tokens = fast_tokenizer(normalized[None].astype(np.float32))
                fast_norm = np.asarray(
                    fast_tokenizer.decode(tokens, time_horizon=horizon, action_dim=stage1_dataset.action_spec.action_dim)
                )[0].astype(np.float32)
                fast_raw = _unnormalize_actions(dataset=dataset, normalized_actions=fast_norm)
                fast_norm_chunks.append(fast_norm)
                fast_raw_chunks.append(fast_raw)

    expert_norm = np.concatenate(expert_norm_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
    expert = np.concatenate(expert_raw_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
    decoded_norm = np.concatenate(decoded_norm_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
    decoded = np.concatenate(decoded_raw_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
    result = {
        "expert": expert,
        "decoded": decoded,
        "expert_norm": expert_norm,
        "decoded_norm": decoded_norm,
    }
    if fast_tokenizer is not None:
        result["fast_norm"] = np.concatenate(fast_norm_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
        result["fast"] = np.concatenate(fast_raw_chunks, axis=0)[: int(trajectory_length)].astype(np.float32)
    return result


def _select_records(
    *,
    stage1_dataset: VARStage1ActionDataset,
    env_name: str | None,
    episodes: int,
    start_episode: int,
    all_tasks: bool,
    samples_per_task: int | None,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if all_tasks:
        if samples_per_task is None:
            raise ValueError("--all_tasks requires --samples_per_task.")
        rng = np.random.default_rng(sample_seed)
        selected: list[dict[str, Any]] = []
        for dataset in stage1_dataset.source_dataset.datasets:
            trajectory_pairs = list(zip(dataset.trajectory_ids, dataset.trajectory_lengths, strict=True))
            if len(trajectory_pairs) < samples_per_task:
                raise ValueError(f"{dataset.dataset_name} has only {len(trajectory_pairs)} trajectories.")
            indices = sorted(rng.choice(len(trajectory_pairs), size=samples_per_task, replace=False).tolist())
            for index in indices:
                trajectory_id, trajectory_length = trajectory_pairs[index]
                selected.append(
                    {
                        "env_name": _env_name_for_dataset(dataset),
                        "dataset": dataset,
                        "trajectory_id": int(trajectory_id),
                        "trajectory_length": int(trajectory_length),
                        "sample_index": int(index),
                    }
                )
        return selected

    if env_name is None:
        raise ValueError("--env_name is required unless --all_tasks is set.")
    dataset = _dataset_for_env(stage1_dataset, env_name)
    selected = []
    trajectory_pairs = list(zip(dataset.trajectory_ids, dataset.trajectory_lengths, strict=True))[
        start_episode : start_episode + episodes
    ]
    for local_index, (trajectory_id, trajectory_length) in enumerate(trajectory_pairs, start=start_episode):
        selected.append(
            {
                "env_name": env_name,
                "dataset": dataset,
                "trajectory_id": int(trajectory_id),
                "trajectory_length": int(trajectory_length),
                "sample_index": int(local_index),
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--samples_per_task", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=20260611)
    parser.add_argument("--sample_manifest", type=Path, default=None)
    parser.add_argument("--include_fast", action="store_true")
    parser.add_argument("--fast_tokenizer_name", type=str, default="playground/Pretrained_models/fast")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    artifact = load_frozen_var_action_tokenizer(args.checkpoint, device=device)
    train_cfg = OmegaConf.create(artifact.checkpoint["stage1_config"])
    base_cfg = load_starvla_base_config(train_cfg)
    stage1_dataset = VARStage1ActionDataset(
        base_cfg,
        mode="train",
        balance_dataset_weights=bool(train_cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(train_cfg.data.get("balance_trajectory_weights", False)),
        seed=int(train_cfg.experiment.get("seed", 42)),
        return_raw_actions=False,
        window_mode=str(train_cfg.data.get("window_mode", "full")),
    )
    fast_tokenizer = None
    if args.include_fast:
        from transformers import AutoProcessor

        fast_tokenizer = AutoProcessor.from_pretrained(args.fast_tokenizer_name, trust_remote_code=True)

    selected = _select_records(
        stage1_dataset=stage1_dataset,
        env_name=args.env_name,
        episodes=args.episodes,
        start_episode=args.start_episode,
        all_tasks=bool(args.all_tasks),
        samples_per_task=args.samples_per_task,
        sample_seed=int(args.sample_seed),
    )

    records = []
    manifest = []
    for item in selected:
        dataset = item["dataset"]
        actions = _get_actions(
            stage1_dataset=stage1_dataset,
            dataset=dataset,
            trajectory_id=int(item["trajectory_id"]),
            trajectory_length=int(item["trajectory_length"]),
            model=artifact.tokenizer,
            device=device,
            fast_tokenizer=fast_tokenizer,
        )
        record = {
            "env_name": str(item["env_name"]),
            "dataset_name": str(dataset.dataset_name),
            "trajectory_id": int(item["trajectory_id"]),
            "trajectory_length": int(item["trajectory_length"]),
            "seed": _trajectory_seed(dataset, int(item["trajectory_id"])),
            "sample_index": int(item["sample_index"]),
            **actions,
            "mean_abs_delta": float(np.abs(actions["decoded"] - actions["expert"]).mean()),
            "max_abs_delta": float(np.abs(actions["decoded"] - actions["expert"]).max()),
            "mean_abs_delta_norm": float(np.abs(actions["decoded_norm"] - actions["expert_norm"]).mean()),
            "max_abs_delta_norm": float(np.abs(actions["decoded_norm"] - actions["expert_norm"]).max()),
        }
        if "fast" in actions:
            record.update(
                {
                    "mean_abs_delta_fast": float(np.abs(actions["fast"] - actions["expert"]).mean()),
                    "max_abs_delta_fast": float(np.abs(actions["fast"] - actions["expert"]).max()),
                    "mean_abs_delta_fast_norm": float(np.abs(actions["fast_norm"] - actions["expert_norm"]).mean()),
                    "max_abs_delta_fast_norm": float(np.abs(actions["fast_norm"] - actions["expert_norm"]).max()),
                }
            )
        records.append(record)
        manifest.append(
            {
                "env_name": record["env_name"],
                "dataset_name": record["dataset_name"],
                "trajectory_id": record["trajectory_id"],
                "trajectory_length": record["trajectory_length"],
                "seed": record["seed"],
                "sample_index": record["sample_index"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, records=np.asarray(records, dtype=object))
    if args.sample_manifest is not None:
        args.sample_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.sample_manifest.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_seed": int(args.sample_seed),
                    "samples_per_task": args.samples_per_task,
                    "records": manifest,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    array_keys = {"expert", "decoded", "fast", "expert_norm", "decoded_norm", "fast_norm"}
    print(json.dumps([{k: v for k, v in r.items() if k not in array_keys} for r in records], indent=2))
    print(f"Wrote {len(records)} replay records to {args.output}")


if __name__ == "__main__":
    main()
