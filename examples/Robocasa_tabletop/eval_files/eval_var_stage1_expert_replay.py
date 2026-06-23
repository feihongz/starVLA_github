"""Expert-action replay diagnostics for RoboCasa VAR Stage 1 tokenizers.

This is the RoboCasa analogue of the LIBERO Stage 1 oracle replay check, but it
does not claim simulator success. The current GR1 LeRobot expert release stores
actions and robot observations, but not the MuJoCo XML / flattened simulator
state needed to reset RoboCasa to each expert episode. This script therefore
measures the action sequence that would be replayed after Stage 1
encode/decode, both in normalized training space and in raw GR1 action space.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import load_frozen_var_action_tokenizer
from starVLA.training.train_var_stage1 import collate_action_batch, load_starvla_base_config


def _raw_from_normalized(dataset: Any, action: torch.Tensor) -> torch.Tensor:
    """Invert StarVLA action transforms for normalized action tensors ending in D."""

    data: dict[str, torch.Tensor] = {}
    start = 0
    for key in dataset.modality_keys["action"]:
        _, subkey = key.split(".", 1)
        dim = int(dataset.metadata.modalities.action[subkey].shape[0])
        data[key] = action.detach().cpu()[..., start : start + dim]
        start += dim
    if start != int(action.shape[-1]):
        raise ValueError(f"Split action dims sum to {start}, but action dim is {action.shape[-1]}.")
    split = dataset.transforms.unapply(data)
    raw_parts = [torch.as_tensor(split[key]) for key in dataset.modality_keys["action"]]
    return torch.cat(raw_parts, dim=-1).to(dtype=torch.float32)


def _new_group() -> dict[str, Any]:
    return {
        "samples": 0,
        "elements": 0,
        "norm_sse": 0.0,
        "norm_abs": 0.0,
        "raw_sse": 0.0,
        "raw_abs": 0.0,
        "raw_max_abs": 0.0,
        "vel_sse": 0.0,
        "vel_elements": 0,
    }


def _update_group(group: dict[str, Any], norm_err: torch.Tensor, raw_err: torch.Tensor) -> None:
    group["samples"] += int(norm_err.shape[0])
    group["elements"] += int(norm_err.numel())
    group["norm_sse"] += float(norm_err.pow(2).sum())
    group["norm_abs"] += float(norm_err.abs().sum())
    group["raw_sse"] += float(raw_err.pow(2).sum())
    group["raw_abs"] += float(raw_err.abs().sum())
    group["raw_max_abs"] = max(float(group["raw_max_abs"]), float(raw_err.abs().max()))
    if norm_err.shape[1] > 1:
        group["vel_sse"] += float((norm_err[:, 1:] - norm_err[:, :-1]).pow(2).sum())
        group["vel_elements"] += int((norm_err[:, 1:] - norm_err[:, :-1]).numel())


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    elements = max(int(group["elements"]), 1)
    vel_elements = max(int(group["vel_elements"]), 1)
    norm_mse = float(group["norm_sse"]) / elements
    raw_mse = float(group["raw_sse"]) / elements
    return {
        "samples": int(group["samples"]),
        "norm_mse": norm_mse,
        "norm_rmse": norm_mse**0.5,
        "norm_mae": float(group["norm_abs"]) / elements,
        "norm_vel_mse": float(group["vel_sse"]) / vel_elements,
        "raw_mse": raw_mse,
        "raw_rmse": raw_mse**0.5,
        "raw_mae": float(group["raw_abs"]) / elements,
        "raw_max_abs": float(group["raw_max_abs"]),
    }


def evaluate(
    checkpoint_path: Path,
    output_path: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    max_samples: int,
    samples_per_dataset: int,
) -> dict[str, Any]:
    stage1 = load_frozen_var_action_tokenizer(checkpoint_path, device=device)
    model = stage1.tokenizer
    train_cfg = OmegaConf.create(stage1.checkpoint["stage1_config"])
    base_cfg = load_starvla_base_config(train_cfg)
    dataset = VARStage1ActionDataset(
        base_cfg,
        mode="train",
        balance_dataset_weights=bool(train_cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(train_cfg.data.get("balance_trajectory_weights", False)),
        seed=int(train_cfg.experiment.get("seed", 42)),
        return_raw_actions=True,
        window_mode=str(train_cfg.data.get("window_mode", "full")),
    )

    eval_dataset = dataset
    if samples_per_dataset > 0:
        by_source_dataset: dict[int, list[int]] = defaultdict(list)
        for source_index, window in enumerate(dataset._full_windows):
            dataset_index = int(window[0])
            by_source_dataset[dataset_index].append(source_index)
        selected_indices: list[int] = []
        for dataset_index in sorted(by_source_dataset):
            indices = by_source_dataset[dataset_index]
            if len(indices) <= samples_per_dataset:
                selected_indices.extend(indices)
            else:
                positions = torch.linspace(0, len(indices) - 1, steps=samples_per_dataset).round().long().tolist()
                selected_indices.extend(indices[pos] for pos in positions)
        eval_dataset = Subset(dataset, selected_indices)

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=collate_action_batch,
        persistent_workers=num_workers > 0,
    )

    overall = _new_group()
    by_dataset: dict[str, dict[str, Any]] = defaultdict(_new_group)
    dim_raw_sse = torch.zeros(model.action_dim, dtype=torch.float64)
    dim_raw_abs = torch.zeros(model.action_dim, dtype=torch.float64)
    dim_count = 0
    code_counts = torch.zeros(model.codebook_size, dtype=torch.long)
    scale_code_counts = [torch.zeros(model.codebook_size, dtype=torch.long) for _ in model.scales]

    processed = 0
    batches = 0
    progress = tqdm(loader, desc="robocasa stage1 expert replay")
    with torch.no_grad():
        for batch in progress:
            actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
            raw_actions = batch["actions_raw"].to(dtype=torch.float32)
            out = model(actions)
            recon = out["recon"]
            recon_cpu = recon.detach().cpu().to(dtype=torch.float32)

            raw_recon_items: list[torch.Tensor | None] = [None] * int(recon_cpu.shape[0])
            batch_indices_by_dataset: dict[int, list[int]] = defaultdict(list)
            for item_index, metadata in enumerate(batch["metadata"]):
                batch_indices_by_dataset[int(metadata["dataset_index"])].append(item_index)
            for dataset_index, batch_indices in batch_indices_by_dataset.items():
                source_dataset = dataset.source_dataset.datasets[dataset_index]
                index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
                raw_group = _raw_from_normalized(source_dataset, recon_cpu[index_tensor])
                for local_offset, item_index in enumerate(batch_indices):
                    raw_recon_items[item_index] = raw_group[local_offset]
            raw_recon = torch.stack([item for item in raw_recon_items if item is not None], dim=0)

            norm_err = (recon_cpu - actions.detach().cpu().to(dtype=torch.float32))
            raw_err = raw_recon - raw_actions
            _update_group(overall, norm_err, raw_err)
            for item_index, metadata in enumerate(batch["metadata"]):
                name = str(metadata["dataset_name"])
                _update_group(
                    by_dataset[name],
                    norm_err[item_index : item_index + 1],
                    raw_err[item_index : item_index + 1],
                )

            dim_raw_sse += raw_err.pow(2).sum(dim=(0, 1)).double()
            dim_raw_abs += raw_err.abs().sum(dim=(0, 1)).double()
            dim_count += int(raw_err.shape[0] * raw_err.shape[1])

            flat_tokens = out["flat_token_ids"].detach().reshape(-1).cpu()
            code_counts += torch.bincount(flat_tokens, minlength=model.codebook_size)
            for idx, token_ids in enumerate(out["token_ids"]):
                scale_code_counts[idx] += torch.bincount(
                    token_ids.detach().reshape(-1).cpu(),
                    minlength=model.codebook_size,
                )

            processed += int(actions.shape[0])
            batches += 1
            progress.set_postfix(samples=processed, raw_mae=f"{overall['raw_abs'] / max(overall['elements'], 1):.5f}")
            if max_samples > 0 and processed >= max_samples:
                break
            if max_batches > 0 and batches >= max_batches:
                break

    used_codes = int((code_counts > 0).sum().item())
    total_tokens = int(code_counts.sum().item())
    probs = code_counts[code_counts > 0].double() / max(total_tokens, 1)
    perplexity = float(torch.exp(-(probs * probs.log()).sum()).item()) if probs.numel() else 0.0

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(stage1.checkpoint.get("epoch", -1)),
        "dataset_len": len(dataset),
        "eval_dataset_len": len(eval_dataset),
        "num_samples": int(processed),
        "num_batches": int(batches),
        "note": (
            "This is an expert-action encode/decode replay diagnostic. The current RoboCasa "
            "LeRobot expert data does not include MuJoCo XML or flattened simulator states, "
            "so simulator success replay cannot be computed from this dataset alone."
        ),
        "action_spec": dataset.action_spec.to_dict(),
        "metrics": {
            "overall": _finalize_group(overall),
            "by_dataset": {name: _finalize_group(group) for name, group in sorted(by_dataset.items())},
            "per_dim_raw_mse": (dim_raw_sse / max(dim_count, 1)).tolist(),
            "per_dim_raw_mae": (dim_raw_abs / max(dim_count, 1)).tolist(),
        },
        "codebook": {
            "codebook_size": model.codebook_size,
            "used_codes": used_codes,
            "usage_ratio": used_codes / float(model.codebook_size),
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "scale_usage": [
                {
                    "scale": int(scale),
                    "used_codes": int((counts > 0).sum().item()),
                    "usage_ratio": int((counts > 0).sum().item()) / float(model.codebook_size),
                    "total_tokens": int(counts.sum().item()),
                }
                for scale, counts in zip(model.scales, scale_code_counts, strict=True)
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboCasa VAR Stage 1 expert-action replay diagnostics.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "playground/Checkpoints/"
            "var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/"
            "latest.ckpt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "playground/Checkpoints/"
            "var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/"
            "expert_replay_eval.json"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=0,
        help="If >0, evaluate an evenly spaced subset of this many windows per RoboCasa dataset.",
    )
    args = parser.parse_args()

    report = evaluate(
        args.checkpoint,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        samples_per_dataset=args.samples_per_dataset,
    )
    print(json.dumps(report["metrics"]["overall"], indent=2))
    print(json.dumps(report["codebook"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
