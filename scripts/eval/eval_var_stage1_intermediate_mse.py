#!/usr/bin/env python3
"""Evaluate scale-aligned intermediate trajectory MSE for one Stage1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Mapping
import warnings


REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_TEXT = str(REPO_ROOT)
if _REPO_ROOT_TEXT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_TEXT)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer.stage1_artifact import (
    load_frozen_var_action_tokenizer,
)
from starVLA.training.train_var_stage1 import (
    collate_action_batch,
    load_starvla_base_config,
)
from starVLA.utils.var_stage1_metrics import (
    METRIC_VERSION,
    finalize_intermediate_mse_stats,
    resolve_intermediate_metric_dims,
    update_intermediate_mse_stats,
)


SCHEMA_VERSION = "stage1_intermediate_eval.v1"
BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "libero": {
        "horizon": 8,
        "action_dim": 7,
        "scales": [1, 2, 4, 8],
        "excluded_groups": ["gripper"],
    },
    "robocasa": {
        "horizon": 16,
        "action_dim": 29,
        "scales": [1, 2, 4, 8, 16],
        "excluded_groups": [],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device_text!r} requested, but CUDA is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible."
            )
    return device


def _validate_model_output_tensor(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if tuple(value.shape) != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: actual={tuple(value.shape)}, expected={expected_shape}."
        )
    if value.device != expected_device:
        raise RuntimeError(
            f"{name} device mismatch: actual={value.device}, expected={expected_device}."
        )
    if not torch.is_floating_point(value):
        raise RuntimeError(f"{name} must have a floating dtype, got {value.dtype}.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN or infinite values.")
    return value


def _as_plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return {str(key): item for key, item in value.items()}


def _infer_method(train_cfg: Any) -> str:
    intermediate = train_cfg.loss.get("intermediate", None)
    if intermediate is not None:
        mode = str(intermediate.get("mode", "none"))
    else:
        mode = "mtr" if float(train_cfg.loss.get("scale_weight", 0.0)) > 0.0 else "none"
    return "multiscale_base" if mode == "none" else mode


def _prepare_training_config(
    checkpoint: Mapping[str, Any],
    *,
    data_root: Path | None,
) -> tuple[Any, str, str, str]:
    if "stage1_config" not in checkpoint:
        raise ValueError("Checkpoint has no stage1_config; it is not a formal Stage1 training artifact.")
    train_cfg = OmegaConf.create(checkpoint["stage1_config"])

    original_data_root = str(train_cfg.data.data_root_dir)
    if not original_data_root.strip():
        raise ValueError("Checkpoint stage1_config has an empty data.data_root_dir.")
    effective_data_root_path = (
        Path(data_root).expanduser() if data_root is not None else Path(original_data_root).expanduser()
    )
    if not effective_data_root_path.is_absolute():
        effective_data_root_path = REPO_ROOT / effective_data_root_path
    effective_data_root_path = effective_data_root_path.resolve()
    if not effective_data_root_path.is_dir():
        raise FileNotFoundError(
            "Stage1 dataset root does not exist: "
            f"configured={original_data_root!r}, resolved={effective_data_root_path}"
        )
    effective_data_root = str(effective_data_root_path)
    train_cfg.data.data_root_dir = effective_data_root

    original_base_config = str(train_cfg.data.starvla_config_yaml)
    base_config_path = Path(original_base_config).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = REPO_ROOT / base_config_path
    base_config_path = base_config_path.resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(
            "StarVLA base config referenced by checkpoint does not exist: "
            f"{original_base_config} (resolved={base_config_path})"
        )
    train_cfg.data.starvla_config_yaml = str(base_config_path)
    return train_cfg, original_data_root, effective_data_root, original_base_config


def _validate_checkpoint_and_dataset(
    *,
    benchmark: str,
    tokenizer: Any,
    artifact_action_spec: Any,
    dataset_action_spec: Any,
) -> tuple[list[int], list[int]]:
    spec = BENCHMARK_SPECS[benchmark]
    expected_horizon = int(spec["horizon"])
    expected_action_dim = int(spec["action_dim"])
    expected_scales = [int(scale) for scale in spec["scales"]]

    actual_scales = [int(scale) for scale in tokenizer.scales]
    checks = {
        "tokenizer action_dim": (int(tokenizer.action_dim), expected_action_dim),
        "tokenizer horizon": (int(tokenizer.seq_len), expected_horizon),
        "checkpoint ActionSpec action_dim": (int(artifact_action_spec.action_dim), expected_action_dim),
        "checkpoint ActionSpec horizon": (int(artifact_action_spec.horizon), expected_horizon),
        "dataset ActionSpec action_dim": (int(dataset_action_spec.action_dim), expected_action_dim),
        "dataset ActionSpec horizon": (int(dataset_action_spec.horizon), expected_horizon),
    }
    mismatches = [
        f"{name}: actual={actual}, expected={expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if actual_scales != expected_scales:
        mismatches.append(f"tokenizer scales: actual={actual_scales}, expected={expected_scales}")
    if mismatches:
        raise ValueError(f"{benchmark} checkpoint/dataset contract mismatch: " + "; ".join(mismatches))

    checkpoint_groups = _as_plain_mapping(artifact_action_spec.dim_groups, name="checkpoint dim_groups")
    dataset_groups = _as_plain_mapping(dataset_action_spec.dim_groups, name="dataset dim_groups")
    checkpoint_groups = {
        name: [int(index) for index in indices]
        for name, indices in checkpoint_groups.items()
    }
    dataset_groups = {
        name: [int(index) for index in indices]
        for name, indices in dataset_groups.items()
    }
    if checkpoint_groups != dataset_groups:
        raise ValueError(
            "Checkpoint and dataset ActionSpec dim_groups differ: "
            f"checkpoint={checkpoint_groups}, dataset={dataset_groups}."
        )
    model_groups = {
        str(name): [int(index) for index in indices]
        for name, indices in tokenizer.dim_groups.items()
    }
    if model_groups != checkpoint_groups:
        raise ValueError(
            "Tokenizer and checkpoint ActionSpec dim_groups differ: "
            f"tokenizer={model_groups}, checkpoint={checkpoint_groups}."
        )

    included_dims = resolve_intermediate_metric_dims(
        benchmark=benchmark,
        action_dim=expected_action_dim,
        dim_groups=checkpoint_groups,
    )
    return actual_scales, included_dims


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    benchmark: str,
    output_path: Path,
    device_text: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 8,
    max_batches: int = 0,
    data_root: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the formal Stage1 intermediate-trajectory evaluator."""

    benchmark = str(benchmark).lower()
    if benchmark not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {sorted(BENCHMARK_SPECS)}.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if checkpoint_path == output_path:
        raise ValueError("Output path must not be the checkpoint path.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage1 checkpoint does not exist: {checkpoint_path}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if max_batches < 0:
        raise ValueError(f"max_batches must be non-negative, got {max_batches}.")

    device = _resolve_device(device_text)
    checkpoint_sha256_before = _sha256(checkpoint_path)
    artifact = load_frozen_var_action_tokenizer(checkpoint_path, device=device)
    tokenizer = artifact.tokenizer
    if str(tokenizer.quantization_mode) == "none":
        raise ValueError(
            "Pure-AE checkpoint cannot be evaluated: intermediate scale reconstructions "
            "require a vq or product_vq quantization path."
        )

    train_cfg, original_data_root, effective_data_root, original_base_config = _prepare_training_config(
        artifact.checkpoint,
        data_root=data_root,
    )
    window_mode = str(train_cfg.data.get("window_mode", "full"))
    if window_mode != "full":
        raise ValueError(
            "Formal intermediate trajectory MSE requires data.window_mode='full'; "
            f"checkpoint config has {window_mode!r}."
        )
    base_cfg = load_starvla_base_config(train_cfg)
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        dataset = VARStage1ActionDataset(
            base_cfg,
            mode="train",
            balance_dataset_weights=bool(train_cfg.data.get("balance_dataset_weights", False)),
            balance_trajectory_weights=bool(train_cfg.data.get("balance_trajectory_weights", False)),
            seed=int(train_cfg.experiment.get("seed", 42)),
            return_raw_actions=False,
            window_mode=window_mode,
        )
        captured_warnings = [str(record.message) for record in warning_records]
    for message in captured_warnings:
        print(f"Dataset warning: {message}", file=sys.stderr)
    missing_trajectory_warnings = [
        message
        for message in captured_warnings
        if "Skipped missing trajectory parquet files" in message
    ]
    if missing_trajectory_warnings:
        raise RuntimeError(
            "Dataset construction skipped missing trajectory parquet files; refusing to "
            "produce a partial formal result: " + " | ".join(missing_trajectory_warnings)
        )

    all_scales, included_dims = _validate_checkpoint_and_dataset(
        benchmark=benchmark,
        tokenizer=tokenizer,
        artifact_action_spec=artifact.action_spec,
        dataset_action_spec=dataset.action_spec,
    )
    intermediate_scales = all_scales[:-1]
    if any(scale >= int(tokenizer.seq_len) for scale in intermediate_scales):
        raise ValueError(
            f"Intermediate scales must be below horizon {tokenizer.seq_len}, got {intermediate_scales}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_action_batch,
        persistent_workers=num_workers > 0,
    )

    intermediate_stats: dict[int, dict[str, float | int]] = {}
    final_recon_sse = 0.0
    final_recon_count = 0
    max_abs_final_scale_vs_recon = 0.0
    num_samples = 0
    num_batches = 0

    progress = tqdm(loader, desc=f"{benchmark} intermediate MSE", disable=not show_progress)
    with torch.inference_mode():
        for batch in progress:
            actions = batch["actions"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            expected_action_tail = (int(tokenizer.seq_len), int(tokenizer.action_dim))
            if actions.ndim != 3 or tuple(actions.shape[1:]) != expected_action_tail:
                raise RuntimeError(
                    "Action batch shape mismatch: "
                    f"actual={tuple(actions.shape)}, expected=[B, {expected_action_tail[0]}, "
                    f"{expected_action_tail[1]}]."
                )
            if not bool(torch.isfinite(actions).all().item()):
                raise RuntimeError("Action batch contains NaN or infinite values.")
            output = tokenizer(actions, return_scale_recons=True)
            if not isinstance(output, Mapping):
                raise RuntimeError(
                    f"Tokenizer output must be a mapping, got {type(output).__name__}."
                )
            returned_scales = [int(scale) for scale in output.get("scale_recon_scales", [])]
            scale_recons = output.get("scale_recons")
            if returned_scales != all_scales:
                raise RuntimeError(
                    f"Tokenizer scale output order mismatch: returned={returned_scales}, expected={all_scales}."
                )
            if not isinstance(scale_recons, list) or len(scale_recons) != len(all_scales):
                raise RuntimeError(
                    "Tokenizer must return exactly one cumulative reconstruction per scale; "
                    f"got type={type(scale_recons).__name__}, len="
                    f"{len(scale_recons) if isinstance(scale_recons, list) else 'n/a'}, "
                    f"expected={len(all_scales)}."
                )
            expected_output_shape = tuple(actions.shape)
            recon = _validate_model_output_tensor(
                output.get("recon"),
                name="tokenizer recon",
                expected_shape=expected_output_shape,
                expected_device=actions.device,
            )
            final_scale_recon = _validate_model_output_tensor(
                scale_recons[-1],
                name=f"scale {all_scales[-1]} cumulative reconstruction",
                expected_shape=expected_output_shape,
                expected_device=actions.device,
            )
            final_delta = (final_scale_recon.float() - recon.float()).abs()
            batch_max_delta = float(final_delta.max().item())
            max_abs_final_scale_vs_recon = max(max_abs_final_scale_vs_recon, batch_max_delta)
            if not torch.allclose(final_scale_recon, recon):
                raise RuntimeError(
                    "Final cumulative scale reconstruction differs from tokenizer recon: "
                    f"max_abs_delta={batch_max_delta}."
                )

            update_intermediate_mse_stats(
                intermediate_stats,
                actions=actions,
                scale_recons=scale_recons[:-1],
                scales=intermediate_scales,
                included_dims=included_dims,
            )
            final_diff = recon.float() - actions.float()
            final_recon_sse += float(final_diff.square().sum(dtype=torch.float64).item())
            final_recon_count += int(final_diff.numel())
            num_samples += int(actions.shape[0])
            num_batches += 1
            if intermediate_stats:
                current_mse = sum(
                    float(value["sse"]) / int(value["count"])
                    for value in intermediate_stats.values()
                ) / len(intermediate_stats)
                progress.set_postfix(mse=f"{current_mse:.7g}", samples=num_samples)
            if max_batches > 0 and num_batches >= max_batches:
                break

    if num_samples == 0 or final_recon_count == 0:
        raise RuntimeError("Evaluator processed no samples.")
    if max_batches == 0 and num_samples != len(dataset):
        raise RuntimeError(
            f"Full evaluation processed {num_samples} samples, but dataset_len={len(dataset)}."
        )

    intermediate_traj_mse, per_scale_mse = finalize_intermediate_mse_stats(intermediate_stats)
    final_recon_mse = final_recon_sse / final_recon_count
    numeric_values = [intermediate_traj_mse, final_recon_mse, *per_scale_mse.values()]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_values):
        raise RuntimeError(f"Evaluator produced an invalid metric value: {numeric_values}")

    checkpoint_sha256_after = _sha256(checkpoint_path)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise RuntimeError(
            "Checkpoint changed during evaluation: "
            f"before={checkpoint_sha256_before}, after={checkpoint_sha256_after}."
        )

    stage1_config = _as_plain_mapping(artifact.checkpoint["stage1_config"], name="stage1_config")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "benchmark": benchmark,
        "method": _infer_method(train_cfg),
        "seed": int(train_cfg.experiment.get("seed", 42)),
        "split": "train_full_windows",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256_before,
            "epoch": int(artifact.checkpoint.get("epoch", -1)),
        },
        "action_space": "normalized",
        "horizon": int(tokenizer.seq_len),
        "action_dim": int(tokenizer.action_dim),
        "all_scales": all_scales,
        "evaluated_scales": intermediate_scales,
        "included_dims": included_dims,
        "excluded_groups": list(BENCHMARK_SPECS[benchmark]["excluded_groups"]),
        "target": {
            "downsample": "adaptive_avg_pool1d",
            "upsample": "linear",
            "align_corners": False,
        },
        "num_samples": num_samples,
        "num_batches": num_batches,
        "dataset_len": len(dataset),
        "per_scale_mse": {
            str(scale): float(per_scale_mse[scale])
            for scale in intermediate_scales
        },
        "intermediate_traj_mse": float(intermediate_traj_mse),
        "sanity": {
            "final_recon_mse_all_dims": float(final_recon_mse),
            "max_abs_final_scale_vs_recon": float(max_abs_final_scale_vs_recon),
        },
        "data": {
            "original_data_root": original_data_root,
            "effective_data_root": effective_data_root,
            "data_root_overridden": data_root is not None,
            "data_mix": str(train_cfg.data.get("data_mix", "")),
            "window_mode": window_mode,
            "original_starvla_config_yaml": original_base_config,
            "warnings": captured_warnings,
        },
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_batches": max_batches,
        },
        "stage1_training": {
            "experiment_name": str(stage1_config.get("experiment", {}).get("name", "")),
            "intermediate_supervision": stage1_config.get("loss", {}).get("intermediate"),
        },
    }
    _atomic_write_json(output_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scale-aligned intermediate trajectory MSE for one Stage1 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=512)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--max-batches", "--max_batches", dest="max_batches", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        benchmark=args.benchmark,
        output_path=output_path,
        device_text=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        data_root=args.data_root,
        show_progress=not args.no_progress,
    )
    print(json.dumps({
        "benchmark": report["benchmark"],
        "method": report["method"],
        "num_samples": report["num_samples"],
        "intermediate_traj_mse": report["intermediate_traj_mse"],
        "per_scale_mse": report["per_scale_mse"],
        "sanity": report["sanity"],
    }, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
