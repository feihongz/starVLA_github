"""Replay RoboTwin expert or Stage-1-decoded actions in the RoboTwin simulator."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import traceback
import types
import zipfile
from pathlib import Path
from shutil import which
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from starVLA.dataloader.robotwin_raw_stage1_action_dataset import (
    GRIPPER_DIMS,
    ROBOTWIN_TASKS_50,
)
from starVLA.model.modules.action_tokenizer import load_frozen_var_action_tokenizer


def _add_robotwin_path(robotwin_path: Path) -> None:
    sys.path.insert(0, str(robotwin_path))
    sys.path.insert(0, str(robotwin_path / "description" / "utils"))


def _patch_warp_torch_compat() -> None:
    """Adapt newer warp-lang top-level torch helpers to older cuRobo calls."""
    try:
        import warp as wp
    except ImportError:
        return
    if hasattr(wp, "torch"):
        return
    helper_names = (
        "device_from_torch",
        "device_to_torch",
        "dtype_from_torch",
        "dtype_to_torch",
        "from_torch",
        "stream_from_torch",
        "stream_to_torch",
        "to_torch",
    )
    helpers = {name: getattr(wp, name) for name in helper_names if hasattr(wp, name)}
    if helpers:
        wp.torch = types.SimpleNamespace(**helpers)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


def _resolve_env_args(
    robotwin_path: Path,
    task_name: str,
    task_config: str,
    *,
    eval_video_save_dir: Path | None = None,
) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    args = _load_yaml(robotwin_path / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "stage1_replay"
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = eval_video_save_dir is not None
    if eval_video_save_dir is not None:
        args["eval_video_save_dir"] = eval_video_save_dir

    embodiment_config = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

    def embodiment_file(embodiment_type: str) -> str:
        robot_file = embodiment_config[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_type!r}")
        return str(robot_file)

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = _load_yaml(Path(args["right_robot_file"]) / "config.yml")
    return args


def _head_video_size(robotwin_path: Path, env_args: dict[str, Any]) -> tuple[int, int]:
    camera_config = _load_yaml(robotwin_path / "task_config" / "_camera_config.yml")
    camera_type = env_args["camera"]["head_camera_type"]
    if camera_type not in camera_config:
        raise KeyError(f"Unknown head camera type {camera_type!r}")
    config = camera_config[camera_type]
    return int(config["w"]), int(config["h"])


def _task_env(robotwin_path: Path, task_name: str) -> Any:
    _add_robotwin_path(robotwin_path)
    _patch_warp_torch_compat()
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()


def _zip_for(data_root: Path, task_name: str, embodiment: str, split: str) -> Path:
    task_dir = data_root / "dataset" / task_name
    matches = sorted(task_dir.glob(f"{embodiment}_{split}_*.zip"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {embodiment}_{split}_*.zip under {task_dir}, got {len(matches)}")
    return matches[0]


def _zip_prefix(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith("/seed.txt"):
                return name.rsplit("/", 1)[0]
    raise FileNotFoundError(f"No seed.txt found in {zip_path}")


def _read_seed_list(zip_path: Path, prefix: str) -> list[int]:
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(f"{prefix}/seed.txt").decode("utf-8")
    return [int(item) for item in text.split()]


def _read_actions(zip_path: Path, prefix: str, episode_index: int, action_key: str) -> np.ndarray:
    member = f"{prefix}/data/episode{episode_index}.hdf5"
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(member)
    with h5py.File(io.BytesIO(payload), "r") as handle:
        actions = np.asarray(handle[action_key], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected [T, 14] actions in {member}, got {actions.shape}")
    return actions


def _compute_norm_stats(
    *,
    stage1_cfg: Any,
    data_root: Path,
    cache_path: Path | None,
    task_names_override: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["raw_min"].astype(np.float32), cached["raw_max"].astype(np.float32)

    data_cfg = stage1_cfg.data
    task_names = list(task_names_override) if task_names_override is not None else list(data_cfg.get("task_names", ROBOTWIN_TASKS_50))
    splits = [str(item).lower() for item in data_cfg.get("splits", ["clean"])]
    embodiment = str(data_cfg.get("embodiment", "aloha-agilex"))
    action_key = str(data_cfg.get("raw_action_key", "/joint_action/vector"))

    raw_min = np.full((14,), np.inf, dtype=np.float32)
    raw_max = np.full((14,), -np.inf, dtype=np.float32)
    for task_name in task_names:
        for split in splits:
            zip_path = _zip_for(data_root, str(task_name), embodiment, split)
            prefix = _zip_prefix(zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                members = sorted(name for name in archive.namelist() if name.endswith(".hdf5") and "/data/" in name)
                max_eps = data_cfg.get("max_episodes_per_zip", None)
                if max_eps is not None:
                    members = members[: int(max_eps)]
                for member in members:
                    with h5py.File(io.BytesIO(archive.read(member)), "r") as handle:
                        actions = np.asarray(handle[action_key], dtype=np.float32)
                    raw_min = np.minimum(raw_min, actions.min(axis=0))
                    raw_max = np.maximum(raw_max, actions.max(axis=0))
            print(f"[norm] scanned task={task_name} split={split} zip={zip_path.name}", flush=True)

    if not np.isfinite(raw_min).all() or not np.isfinite(raw_max).all():
        raise RuntimeError("Failed to compute finite RoboTwin normalization stats.")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, raw_min=raw_min, raw_max=raw_max)
        print(f"[norm] wrote {cache_path}", flush=True)
    return raw_min, raw_max


def _normalize(actions: np.ndarray, raw_min: np.ndarray, raw_max: np.ndarray, binary_threshold: float) -> np.ndarray:
    normalized = np.zeros_like(actions, dtype=np.float32)
    denom = raw_max - raw_min
    valid = denom != 0
    normalized[:, valid] = 2.0 * (actions[:, valid] - raw_min[valid]) / denom[valid] - 1.0
    normalized[:, ~valid] = 0.0
    normalized[:, GRIPPER_DIMS] = (actions[:, GRIPPER_DIMS] > binary_threshold).astype(np.float32)
    return normalized


def _unnormalize(actions: np.ndarray, raw_min: np.ndarray, raw_max: np.ndarray, binary_threshold: float) -> np.ndarray:
    raw = (actions + 1.0) * 0.5 * (raw_max - raw_min) + raw_min
    raw[:, GRIPPER_DIMS] = (actions[:, GRIPPER_DIMS] > binary_threshold).astype(np.float32)
    return raw.astype(np.float32)


def _decode_actions(
    actions_raw: np.ndarray,
    *,
    tokenizer: torch.nn.Module,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    horizon: int,
    binary_threshold: float,
    device: torch.device,
) -> np.ndarray:
    decoded_chunks: list[np.ndarray] = []
    tokenizer.eval()
    with torch.no_grad():
        for start in range(0, len(actions_raw), horizon):
            chunk = actions_raw[start : start + horizon]
            valid_len = len(chunk)
            if valid_len < horizon:
                pad = np.repeat(chunk[-1:], horizon - valid_len, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            norm = _normalize(chunk, raw_min, raw_max, binary_threshold)
            tensor = torch.as_tensor(norm, dtype=torch.float32, device=device).unsqueeze(0)
            recon = tokenizer(tensor)["recon"][0].detach().cpu().numpy()
            decoded = _unnormalize(recon, raw_min, raw_max, binary_threshold)[:valid_len]
            decoded_chunks.append(decoded)
    return np.concatenate(decoded_chunks, axis=0)


def _decode_fast_actions(
    actions_raw: np.ndarray,
    *,
    fast_tokenizer: Any,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    horizon: int,
    action_dim: int,
    binary_threshold: float,
) -> np.ndarray:
    decoded_chunks: list[np.ndarray] = []
    for start in range(0, len(actions_raw), horizon):
        chunk = actions_raw[start : start + horizon]
        valid_len = len(chunk)
        if valid_len < horizon:
            pad = np.repeat(chunk[-1:], horizon - valid_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        norm = _normalize(chunk, raw_min, raw_max, binary_threshold)
        tokens = fast_tokenizer(norm[None].astype(np.float32))
        decoded = fast_tokenizer.decode(tokens, time_horizon=horizon, action_dim=action_dim)
        decoded = np.asarray(decoded, dtype=np.float32)
        if decoded.shape != (1, horizon, action_dim):
            raise ValueError(f"FAST decoded shape {decoded.shape} != {(1, horizon, action_dim)}")
        raw = _unnormalize(decoded[0], raw_min, raw_max, binary_threshold)[:valid_len]
        decoded_chunks.append(raw)
    return np.concatenate(decoded_chunks, axis=0)


def _start_video_writer(env: Any, video_path: Path, width: int, height: int) -> subprocess.Popen[Any]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise FileNotFoundError("ffmpeg is not on PATH and imageio_ffmpeg is not installed.") from exc
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg = subprocess.Popen(
        [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    env._set_eval_video_ffmpeg(ffmpeg)
    return ffmpeg


def _run_episode(
    env: Any,
    actions: np.ndarray,
    max_steps: int | None = None,
    *,
    video_path: Path | None = None,
    video_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    steps = 0
    limit = min(len(actions), int(env.step_lim if max_steps is None else max_steps))
    ffmpeg = None
    if video_path is not None:
        if video_size is None:
            raise ValueError("video_size is required when video_path is set.")
        ffmpeg = _start_video_writer(env, video_path, *video_size)
    try:
        for idx in range(limit):
            env.get_obs()
            env.take_action(actions[idx])
            steps += 1
            if bool(getattr(env, "eval_success", False)):
                break
        success = bool(getattr(env, "eval_success", False) or env.check_success())
    finally:
        if ffmpeg is not None:
            env._del_eval_video_ffmpeg()
    result = {"success": success, "steps": int(steps), "num_actions": int(len(actions)), "step_lim": int(env.step_lim)}
    if video_path is not None:
        result["video_path"] = str(video_path)
    return result


def _read_completed_records(log_path: Path | None) -> set[tuple[str, int, str]]:
    if log_path is None or not log_path.exists():
        return set()

    completed: set[tuple[str, int, str]] = set()
    text = log_path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            record, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        pos = start + end
        if not isinstance(record, dict) or record.get("counted", True) is False:
            continue
        if {"task", "episode", "mode", "success"} <= set(record):
            completed.add((str(record["task"]), int(record["episode"]), str(record["mode"])))
    return completed


def _mark_counted_result(report: dict[str, Any], task_name: str, mode: str, result: dict[str, Any], modes: list[str]) -> None:
    report["summary"][mode]["episodes"] += 1
    report["summary"][mode]["successes"] += int(result["success"])
    task_bucket = report["per_task"].setdefault(
        task_name,
        {m: {"successes": 0, "episodes": 0, "success_rate": 0.0} for m in modes},
    )
    task_bucket[mode]["episodes"] += 1
    task_bucket[mode]["successes"] += int(result["success"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    robotwin_path = Path(args.robotwin_path).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_yaml = Path(args.config_yaml).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    norm_stats = Path(args.norm_stats).expanduser().resolve() if args.norm_stats else None
    video_root = Path(args.video_out_dir).expanduser().resolve() if args.save_videos else None
    skip_completed_log = Path(args.skip_completed_log).expanduser().resolve() if args.skip_completed_log else None
    completed_records = _read_completed_records(skip_completed_log)
    _add_robotwin_path(robotwin_path)
    _patch_warp_torch_compat()
    os.chdir(robotwin_path)

    stage1_cfg = OmegaConf.load(config_yaml)
    action_key = str(stage1_cfg.data.get("raw_action_key", "/joint_action/vector"))
    embodiment = str(stage1_cfg.data.get("embodiment", "aloha-agilex"))
    binary_threshold = float(stage1_cfg.data.get("binary_threshold", 0.49))
    split = args.split or ("randomized" if args.task_config == "demo_randomized" else "clean")

    task_names = list(ROBOTWIN_TASKS_50 if args.tasks == "all" else [item.strip() for item in args.tasks.split(",") if item.strip()])
    if args.max_tasks > 0:
        task_names = task_names[: args.max_tasks]

    if args.mode == "both":
        modes = ["expert", "decoded"]
    elif args.mode == "all":
        modes = ["expert", "fast", "decoded"]
    else:
        modes = [args.mode]
    artifact = None
    fast_tokenizer = None
    raw_min = raw_max = None
    horizon = None
    device = torch.device(args.device)
    if "decoded" in modes or "fast" in modes:
        if device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        if "decoded" in modes:
            artifact = load_frozen_var_action_tokenizer(checkpoint_path, device=device)
            horizon = int(artifact.action_spec.horizon)
        else:
            # FAST still needs the Stage1 config horizon for chunking.
            horizon = int(stage1_cfg.model.get("action_horizon", stage1_cfg.data.get("action_horizon", 50)))
        if "fast" in modes:
            from transformers import AutoProcessor

            fast_tokenizer = AutoProcessor.from_pretrained(args.fast_tokenizer_name, trust_remote_code=True)
        norm_cache = norm_stats if norm_stats else checkpoint_path.parent / "robotwin_raw_norm_stats.npz"
        norm_task_names = None
        if args.norm_tasks:
            norm_task_names = task_names if args.norm_tasks == "selected" else [item.strip() for item in args.norm_tasks.split(",") if item.strip()]
            if norm_stats is None:
                safe = "selected" if args.norm_tasks == "selected" else "_".join(norm_task_names)
                norm_cache = checkpoint_path.parent / f"robotwin_raw_norm_stats_{safe}.npz"
        raw_min, raw_max = _compute_norm_stats(
            stage1_cfg=stage1_cfg,
            data_root=data_root,
            cache_path=norm_cache,
            task_names_override=norm_task_names,
        )

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "config_yaml": str(config_yaml),
        "task_config": str(args.task_config),
        "split": split,
        "modes": modes,
        "tasks": task_names,
        "skip_completed_log": str(skip_completed_log) if skip_completed_log is not None else None,
        "skipped_completed": len(completed_records),
        "failed_uncounted": [],
        "episodes": [],
        "summary": {mode: {"successes": 0, "episodes": 0, "success_rate": 0.0} for mode in modes},
        "per_task": {},
    }

    for task_name in task_names:
        zip_path = _zip_for(data_root, task_name, embodiment, split)
        prefix = _zip_prefix(zip_path)
        seeds = _read_seed_list(zip_path, prefix)
        start = int(args.start_episode)
        end = min(len(seeds), start + int(args.num_episodes_per_task))

        task_video_root = video_root / task_name if video_root is not None else None
        env_args = _resolve_env_args(
            robotwin_path,
            task_name,
            args.task_config,
            eval_video_save_dir=task_video_root,
        )
        video_size = _head_video_size(robotwin_path, env_args) if video_root is not None else None
        for episode_index in range(start, end):
            seed = int(seeds[episode_index])
            expert_actions = _read_actions(zip_path, prefix, episode_index, action_key)
            decoded_actions = None
            fast_actions = None
            if "fast" in modes:
                if fast_tokenizer is None or raw_min is None or raw_max is None or horizon is None:
                    raise RuntimeError("FAST replay requested before tokenizer/stat initialization.")
                fast_actions = _decode_fast_actions(
                    expert_actions,
                    fast_tokenizer=fast_tokenizer,
                    raw_min=raw_min,
                    raw_max=raw_max,
                    horizon=horizon,
                    action_dim=14,
                    binary_threshold=binary_threshold,
                )
            if "decoded" in modes:
                if artifact is None or raw_min is None or raw_max is None or horizon is None:
                    raise RuntimeError("Decoded replay requested before tokenizer/stat initialization.")
                decoded_actions = _decode_actions(
                    expert_actions,
                    tokenizer=artifact.tokenizer,
                    raw_min=raw_min,
                    raw_max=raw_max,
                    horizon=horizon,
                    binary_threshold=binary_threshold,
                    device=device,
                )

            episode_record: dict[str, Any] = {
                "task_name": task_name,
                "episode_index": int(episode_index),
                "seed": seed,
                "zip_path": str(zip_path),
                "results": {},
            }
            if decoded_actions is not None:
                episode_record["mean_abs_delta"] = float(np.abs(decoded_actions - expert_actions).mean())
                episode_record["max_abs_delta"] = float(np.abs(decoded_actions - expert_actions).max())
            if fast_actions is not None:
                episode_record["mean_abs_delta_fast"] = float(np.abs(fast_actions - expert_actions).mean())
                episode_record["max_abs_delta_fast"] = float(np.abs(fast_actions - expert_actions).max())

            for mode in modes:
                if (task_name, int(episode_index), mode) in completed_records:
                    continue

                env = _task_env(robotwin_path, task_name)
                try:
                    try:
                        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **env_args)
                        if mode == "expert":
                            actions = expert_actions
                        elif mode == "fast":
                            actions = fast_actions
                        else:
                            actions = decoded_actions
                        if actions is None:
                            raise RuntimeError(f"{mode} actions were not prepared.")
                        video_path = None
                        if video_root is not None:
                            video_path = video_root / mode / task_name / f"episode{episode_index:03d}_seed{seed}.mp4"
                        result = _run_episode(
                            env,
                            actions,
                            max_steps=args.max_steps if args.max_steps > 0 else None,
                            video_path=video_path,
                            video_size=video_size,
                        )
                    except Exception as exc:
                        result = {
                            "skipped": True,
                            "counted": False,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                        report["failed_uncounted"].append(
                            {
                                "task": task_name,
                                "episode": int(episode_index),
                                "mode": mode,
                                "error": result["error"],
                            }
                        )
                finally:
                    env.close_env()
                episode_record["results"][mode] = result
                if result.get("counted", True) is not False:
                    _mark_counted_result(report, task_name, mode, result, modes)
                print(json.dumps({"task": task_name, "episode": episode_index, "mode": mode, **result}), flush=True)

            if episode_record["results"]:
                report["episodes"].append(episode_record)
            for mode in modes:
                item = report["summary"][mode]
                item["success_rate"] = item["successes"] / max(item["episodes"], 1)

    for task_bucket in report["per_task"].values():
        for mode in modes:
            item = task_bucket[mode]
            item["success_rate"] = item["successes"] / max(item["episodes"], 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin Stage 1 tokenizer by simulator replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robotwin_path", type=Path, default=Path(os.environ.get("ROBOTWIN_PATH", "/home/zhangfeihong/RoboTwin")))
    parser.add_argument("--data_root", type=Path, default=Path("playground/Datasets/RoboTwin"))
    parser.add_argument("--task_config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    parser.add_argument("--split", choices=["clean", "randomized"], default=None)
    parser.add_argument("--mode", choices=["expert", "fast", "decoded", "both", "all"], default="both")
    parser.add_argument("--tasks", type=str, default="all", help="'all' or comma-separated task names.")
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--num_episodes_per_task", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--norm_stats", type=Path, default=None)
    parser.add_argument(
        "--norm_tasks",
        type=str,
        default="",
        help="Optional smoke-test override for norm stats: 'selected' or comma-separated task names. Default uses the Stage1 training task set.",
    )
    parser.add_argument("--fast_tokenizer_name", type=str, default="physical-intelligence/fast")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--video_out_dir", type=Path, default=Path("playground/Checkpoints/robotwin_stage1_replay_videos"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip_completed_log",
        type=Path,
        default=None,
        help="Optional previous chunk log. Counted task/episode/mode records in this log are skipped on restart.",
    )
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote replay report to {args.output}")


if __name__ == "__main__":
    main()
