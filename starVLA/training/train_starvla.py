# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def barrier_if_distributed() -> None:
    if is_dist_initialized():
        dist.barrier()


def build_accelerator(cfg) -> Accelerator:
    gradient_accumulation_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)
    return accelerator


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    barrier_if_distributed()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.metric_ema = {}

    def prepare_training(self):
        rank = get_rank()
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            if isinstance(self.config, AccessTrackedConfig):
                wandb_config = OmegaConf.to_container(self.config.unwrap(), resolve=True)
            else:
                wandb_config = OmegaConf.to_container(self.config, resolve=True)

            run = wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
                config=wandb_config,
            )
            run.summary["output_dir"] = self.config.output_dir
            run.summary["framework"] = self.config.framework.name
            if hasattr(self.config.trainer, "pretrained_checkpoint"):
                run.summary["pretrained_checkpoint"] = self.config.trainer.pretrained_checkpoint
            if hasattr(self.config.framework, "stage3"):
                run.summary["stage3/var_ce_weight"] = self.config.framework.stage3.var_ce_weight
                run.summary["stage3/flow_loss_weight"] = self.config.framework.stage3.flow_loss_weight

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _checkpoint_score_file(self) -> Path:
        return Path(self.config.output_dir) / "checkpoint_scores.json"

    def _load_checkpoint_scores(self) -> dict:
        score_file = self._checkpoint_score_file()
        if not score_file.exists():
            return {}
        try:
            with open(score_file, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse checkpoint score file: {score_file}")
            return {}

    def _write_checkpoint_scores(self, scores: dict) -> None:
        score_file = self._checkpoint_score_file()
        tmp_file = score_file.with_suffix(score_file.suffix + ".tmp")
        with open(tmp_file, "w") as f:
            json.dump(scores, f, indent=2, sort_keys=True)
        os.replace(tmp_file, score_file)

    def _list_step_checkpoints(self) -> list[tuple[int, Path]]:
        pattern = re.compile(r"^steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoints = []
        for path in checkpoint_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort(key=lambda item: item[0])
        return checkpoints

    def _record_checkpoint_score(self, checkpoint_file: Path, metrics: dict | None) -> None:
        scores = self._load_checkpoint_scores()
        record = {"step": self.completed_steps, "path": checkpoint_file.name, "metrics": {}}
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    record["metrics"][key] = float(value)
        scores[checkpoint_file.name] = record

        existing_names = {path.name for _, path in self._list_step_checkpoints()}
        scores = {name: value for name, value in scores.items() if name in existing_names}
        self._write_checkpoint_scores(scores)

    def _prune_checkpoints(self) -> None:
        policy = getattr(self.config.trainer, "checkpoint_retention_policy", "all")
        if policy in (None, "all", "none"):
            return
        if policy != "latest_and_best":
            raise ValueError(f"Unsupported checkpoint_retention_policy `{policy}`.")

        keep_latest = int(getattr(self.config.trainer, "checkpoint_keep_latest", 1))
        keep_best = int(getattr(self.config.trainer, "checkpoint_keep_best", 2))
        metric_name = str(getattr(self.config.trainer, "checkpoint_metric", "mse_score"))
        metric_mode = str(getattr(self.config.trainer, "checkpoint_metric_mode", "min"))
        if metric_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint_metric_mode `{metric_mode}`. Expected `min` or `max`.")

        checkpoints = self._list_step_checkpoints()
        if not checkpoints:
            return

        scores = self._load_checkpoint_scores()
        keep_names = set()
        if keep_latest > 0:
            keep_names.update(path.name for _, path in checkpoints[-keep_latest:])

        scored = []
        for step, path in checkpoints:
            metric = scores.get(path.name, {}).get("metrics", {}).get(metric_name)
            if metric is None:
                continue
            scored.append((float(metric), step, path))
        scored.sort(key=lambda item: item[0], reverse=(metric_mode == "max"))
        if keep_best > 0:
            keep_names.update(path.name for _, _, path in scored[:keep_best])

        deleted = []
        for _, path in checkpoints:
            if path.name in keep_names:
                continue
            path.unlink()
            deleted.append(path.name)

        if deleted:
            remaining_names = {path.name for _, path in self._list_step_checkpoints()}
            scores = {name: value for name, value in scores.items() if name in remaining_names}
            self._write_checkpoint_scores(scores)
            logger.info(
                "Pruned %d checkpoints with policy=%s, metric=%s, kept=%s",
                len(deleted),
                policy,
                metric_name,
                sorted(keep_names),
            )

    def _metric_is_better(self, candidate: float, best: float, metric_mode: str) -> bool:
        if metric_mode == "min":
            return candidate < best
        if metric_mode == "max":
            return candidate > best
        raise ValueError(f"Unsupported checkpoint_metric_mode `{metric_mode}`. Expected `min` or `max`.")

    def _should_save_metric_checkpoint(self, metrics: dict | None) -> bool:
        should_save = False
        if self.accelerator.is_main_process and metrics:
            metric_name = str(getattr(self.config.trainer, "checkpoint_metric", "mse_score"))
            metric_mode = str(getattr(self.config.trainer, "checkpoint_metric_mode", "min"))
            metric = metrics.get(metric_name)
            if isinstance(metric, (int, float)) and np.isfinite(metric):
                scored_metrics = []
                scores = self._load_checkpoint_scores()
                for record in scores.values():
                    saved_metric = record.get("metrics", {}).get(metric_name)
                    if isinstance(saved_metric, (int, float)) and np.isfinite(saved_metric):
                        scored_metrics.append(float(saved_metric))

                should_save = not scored_metrics or self._metric_is_better(
                    float(metric),
                    min(scored_metrics) if metric_mode == "min" else max(scored_metrics),
                    metric_mode,
                )

        if is_dist_initialized():
            decision = [should_save]
            dist.broadcast_object_list(decision, src=0)
            should_save = bool(decision[0])
        return should_save

    def _save_checkpoint(self, metrics: dict | None = None):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                checkpoint_file = Path(checkpoint_path + "_model.safetensors")
                save_file(state_dict, str(checkpoint_file))
            elif save_format == "pt":
                checkpoint_file = Path(checkpoint_path + "_pytorch_model.pt")
                torch.save(state_dict, checkpoint_file)
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            self._record_checkpoint_score(checkpoint_file, metrics)
            self._prune_checkpoints()

            summary_data = {"steps": self.completed_steps}
            if metrics and "mse_score" in metrics:
                summary_data["mse_score"] = metrics["mse_score"]
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        ema_beta = float(getattr(self.config.trainer, "metric_ema_beta", 0.98))
        if 0.0 <= ema_beta < 1.0:
            for key, value in list(metrics.items()):
                if isinstance(value, (int, float)):
                    prev = self.metric_ema.get(key)
                    ema = float(value) if prev is None else ema_beta * prev + (1.0 - ema_beta) * float(value)
                    self.metric_ema[key] = ema
                    metrics[f"{key}_ema"] = ema

        if self.completed_steps % self.config.trainer.logging_frequency == 0 and get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            did_optimizer_step = bool(step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients))

            if did_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if not did_optimizer_step:
                continue

            did_eval = self.completed_steps % self.config.trainer.eval_interval == 0
            if did_eval:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            save_on_interval = self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0
            save_on_best_metric = did_eval and self._should_save_metric_checkpoint(step_metrics)
            if save_on_interval or save_on_best_metric:
                self._save_checkpoint(step_metrics)

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        barrier_if_distributed()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        if self._uses_deepspeed_gradient_accumulation():
            return self._train_step_deepspeed_accumulation(batch_vla)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        metrics = {"action_dit_loss": action_loss.item()}
        for key, value in output_dict.items():
            if key == "action_loss" or not isinstance(value, torch.Tensor):
                continue
            if value.numel() == 1:
                metrics[key] = float(value.detach().float().cpu().item())
        metrics["_optimizer_step"] = bool(self.accelerator.sync_gradients)
        return metrics

    def _uses_deepspeed_gradient_accumulation(self) -> bool:
        return (
            self.accelerator.gradient_accumulation_steps > 1
            and callable(getattr(self.model, "is_gradient_accumulation_boundary", None))
            and callable(getattr(self.model, "step", None))
        )

    def _train_step_deepspeed_accumulation(self, batch_vla):
        """Run one DeepSpeed micro-step and let ZeRO handle accumulation internally."""
        is_boundary = bool(self.model.is_gradient_accumulation_boundary())

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            total_loss = action_loss

        self.model.backward(total_loss)
        self.model.step()

        if is_boundary:
            self.lr_scheduler.step()

        metrics = {"action_dit_loss": action_loss.item(), "_optimizer_step": is_boundary}
        for key, value in output_dict.items():
            if key == "action_loss" or not isinstance(value, torch.Tensor):
                continue
            if value.numel() == 1:
                metrics[key] = float(value.detach().float().cpu().item())
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    accelerator = build_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
