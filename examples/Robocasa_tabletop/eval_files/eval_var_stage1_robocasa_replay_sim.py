"""Replay exported expert / FAST / Stage-1-decoded actions in the RoboCasa eval env."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio_ffmpeg
import numpy as np

import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper


@dataclass
class VideoConfig:
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))


class OpenCVVideoRecorder:
    def __init__(self, fps: int) -> None:
        self.fps = fps
        self.writer = None
        self.file_path: Path | None = None
        self.is_mp4 = False
        self.stderr = None

    def is_ready(self) -> bool:
        return self.writer is not None

    def start(self, file_path: Path, frame: np.ndarray) -> None:
        import cv2

        if self.is_ready():
            self.stop()
        h, w = frame.shape[:2]
        self.is_mp4 = file_path.suffix.lower() == ".mp4"
        if self.is_mp4:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{w}x{h}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                str(file_path),
            ]
            self.writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(str(file_path), fourcc, float(self.fps), (w, h))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Failed to open video writer for {file_path}")
        self.file_path = file_path

    def write_frame(self, frame: np.ndarray) -> None:
        import cv2

        if not self.is_ready():
            raise RuntimeError("Must call start() before write_frame().")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 video frame, got {frame.dtype}.")
        if self.is_mp4:
            if self.writer.stdin is None:
                raise RuntimeError("Video writer stdin is closed.")
            self.writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop(self) -> None:
        if self.writer is not None:
            if self.is_mp4:
                if self.writer.stdin is not None:
                    self.writer.stdin.close()
                stderr = self.writer.stderr.read().decode("utf-8", errors="replace") if self.writer.stderr else ""
                returncode = self.writer.wait(timeout=120)
                if returncode != 0:
                    raise RuntimeError(f"ffmpeg failed with code {returncode}:\n{stderr[-2000:]}")
            else:
                self.writer.release()
        self.writer = None
        self.file_path = None
        self.is_mp4 = False


class OpenCVVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, video_dir: Path, fps: int, steps_per_render: int) -> None:
        super().__init__(env)
        video_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = video_dir
        self.steps_per_render = steps_per_render
        self.recorder = OpenCVVideoRecorder(fps=fps)
        self.file_path: Path | None = None
        self.step_count = 0
        self.is_success = False
        self.finalized_path: Path | None = None

    def reset(self, **kwargs):
        self._finalize()
        self.step_count = 0
        self.is_success = False
        self.finalized_path = None
        self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
        return super().reset(**kwargs)

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        info = result[-1]
        self.is_success = self.is_success or _info_success(info)
        if self.file_path is not None and self.step_count % self.steps_per_render == 0:
            frame = self.env.render()
            if not self.recorder.is_ready():
                self.recorder.start(self.file_path, frame)
            self.recorder.write_frame(frame)
        return result

    def render(self, *args, **kwargs):
        self._finalize()
        return self.finalized_path

    def close(self):
        self._finalize()
        return super().close()

    def _finalize(self) -> None:
        self.recorder.stop()
        if self.file_path is None or not self.file_path.exists():
            return
        new_path = self.video_dir / f"{self.file_path.stem}_success{int(self.is_success)}.mp4"
        if new_path != self.file_path:
            os.rename(self.file_path, new_path)
        self.finalized_path = new_path
        self.file_path = None


def _split_gr1_action_chunk(chunk: np.ndarray) -> dict[str, np.ndarray]:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 29:
        raise ValueError(f"Expected action chunk [T, 29], got {chunk.shape}.")
    return {
        "action.left_arm": chunk[:, :7],
        "action.right_arm": chunk[:, 7:14],
        "action.left_hand": chunk[:, 14:20],
        "action.right_hand": chunk[:, 20:26],
        "action.waist": chunk[:, 26:29],
    }


def _task_name_from_env(env_name: str) -> str:
    task = env_name.split("/", 1)[-1]
    return re.sub(r"_GR1.*$", "", task)


def _hdf5_path_for_env(env_name: str, hdf5_root: Path) -> Path:
    return hdf5_root / f"{_task_name_from_env(env_name)}.hdf5"


def _unwrap_groot_env(env: gym.Env) -> Any:
    current = env
    while hasattr(current, "env"):
        current = current.env
        if isinstance(current, GrootRoboCasaEnv):
            return current
    if isinstance(current, GrootRoboCasaEnv):
        return current
    raise TypeError(f"Could not find GrootRoboCasaEnv inside wrapper stack rooted at {type(env)}.")


def _make_ik_indicator_invisible(str_xml: str) -> str:
    import xml.etree.ElementTree as ET

    raw_xml = ET.fromstring(str_xml)
    for site in raw_xml.findall(".//site"):
        name = site.get("name", "")
        if "pinch_spheres" in name:
            site.set("rgba", "0 0 0 0")
    return ET.tostring(raw_xml).decode("utf-8")


def _reset_inner_env_to_hdf5_state(env: gym.Env, *, hdf5_path: Path, demo_id: int) -> None:
    import h5py

    groot_env = _unwrap_groot_env(env)
    raw_env = groot_env.env
    demo_key = f"demo_{demo_id}"
    with h5py.File(hdf5_path, "r") as handle:
        if demo_key not in handle["data"]:
            raise KeyError(f"{demo_key} not found in {hdf5_path}.")
        group = handle["data"][demo_key]
        state = np.asarray(group["states"][0])
        model = _make_ik_indicator_invisible(group.attrs["model_file"])
        ep_meta = group.attrs.get("ep_meta", None)

    if ep_meta is not None:
        ep_meta_dict = json.loads(ep_meta)
    else:
        ep_meta_dict = {}
    if hasattr(raw_env, "set_attrs_from_ep_meta"):
        raw_env.set_attrs_from_ep_meta(ep_meta_dict)
    elif hasattr(raw_env, "set_ep_meta"):
        raw_env.set_ep_meta(ep_meta_dict)

    raw_env.reset()
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(model)
    else:
        xml = raw_env.edit_model_xml(model)
    raw_env.reset_from_xml_string(xml)
    raw_env.sim.reset()
    raw_env.sim.set_state_from_flattened(state)
    raw_env.sim.forward()
    if hasattr(raw_env, "update_sites"):
        raw_env.update_sites()
    if hasattr(raw_env, "update_state"):
        raw_env.update_state()


def _pad_chunk(chunk: np.ndarray, n_action_steps: int) -> np.ndarray:
    if len(chunk) >= n_action_steps:
        return chunk[:n_action_steps]
    if len(chunk) == 0:
        raise ValueError("Cannot pad an empty action chunk.")
    pad = np.repeat(chunk[-1:], n_action_steps - len(chunk), axis=0)
    return np.concatenate([chunk, pad], axis=0)


def _info_success(info: dict[str, Any]) -> bool:
    if "success" not in info:
        return False
    value = info["success"]
    if isinstance(value, (list, tuple)):
        return any(bool(np.asarray(item).any()) for item in value)
    return bool(np.asarray(value).any())


def _wrapper_success(env: gym.Env) -> bool:
    current: Any = env
    while True:
        if bool(getattr(current, "is_success", False)):
            return True
        if not hasattr(current, "env"):
            return False
        current = current.env


def _make_env(env_name: str, *, video_dir: Path | None, n_action_steps: int, max_episode_steps: int) -> gym.Env:
    env = gym.make(env_name, enable_render=True)
    if video_dir is not None:
        video_config = VideoConfig()
        env = OpenCVVideoRecordingWrapper(
            env,
            video_dir=video_dir,
            fps=video_config.fps,
            steps_per_render=video_config.steps_per_render,
        )
    multistep_config = MultiStepConfig()
    env = MultiStepWrapper(
        env,
        video_delta_indices=multistep_config.video_delta_indices,
        state_delta_indices=multistep_config.state_delta_indices,
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
    )
    return env


def _run_one(
    *,
    env: gym.Env,
    actions: np.ndarray,
    seed: int | None,
    n_action_steps: int,
    max_episode_steps: int,
    hdf5_path: Path | None,
) -> dict[str, Any]:
    env.reset(seed=seed)
    if hdf5_path is not None:
        if seed is None:
            raise ValueError("HDF5 reset requires a demo id; use --seed_strategy record for LeRobot trajectory ids.")
        _reset_inner_env_to_hdf5_state(env, hdf5_path=hdf5_path, demo_id=int(seed))
    success = False
    chunks_executed = 0
    primitive_steps = 0
    terminated = False
    truncated = False

    for start in range(0, min(len(actions), max_episode_steps), n_action_steps):
        chunk = _pad_chunk(actions[start : start + n_action_steps], n_action_steps)
        _, _, terminated, truncated, info = env.step(_split_gr1_action_chunk(chunk))
        chunks_executed += 1
        primitive_steps += int(min(n_action_steps, max(0, len(actions) - start)))
        success = success or _info_success(info) or _wrapper_success(env)
        if success or terminated or truncated:
            break

    return {
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "chunks_executed": int(chunks_executed),
        "primitive_steps": int(primitive_steps),
        "num_actions": int(len(actions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions_npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["expert", "fast", "decoded", "both", "all"], default="both")
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--start_record", type=int, default=0)
    parser.add_argument("--end_record", type=int, default=None)
    parser.add_argument("--seed_strategy", choices=["record", "index", "none"], default="record")
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument(
        "--hdf5_root",
        type=Path,
        default=None,
        help="If set, reset each episode to HDF5 demo_<seed> model XML and initial MuJoCo state before replay.",
    )
    args = parser.parse_args()

    all_records = list(np.load(args.actions_npz, allow_pickle=True)["records"])
    end_record = len(all_records) if args.end_record is None else min(int(args.end_record), len(all_records))
    records = all_records[int(args.start_record) : end_record]
    if args.mode == "both":
        modes = ["expert", "decoded"]
    elif args.mode == "all":
        modes = ["expert", "fast", "decoded"]
    else:
        modes = [args.mode]
    results: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(args.output.suffix + ".jsonl")

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for offset, record in enumerate(records):
            record_index = int(args.start_record) + offset
            record = dict(record)
            env_name = str(record["env_name"])
            task_name = _task_name_from_env(env_name)
            for mode in modes:
                if mode not in record:
                    raise KeyError(f"Record {record_index} does not contain mode {mode!r}.")
                video_dir = None
                if args.video_dir is not None:
                    video_dir = args.video_dir / mode / task_name / f"record_{record_index:03d}_demo{int(record['seed']):05d}"
                env = _make_env(
                    env_name,
                    video_dir=video_dir,
                    n_action_steps=args.n_action_steps,
                    max_episode_steps=args.max_episode_steps,
                )
                try:
                    if args.seed_strategy == "record":
                        seed = int(record["seed"])
                    elif args.seed_strategy == "index":
                        seed = record_index
                    else:
                        seed = None
                    sim = _run_one(
                        env=env,
                        actions=np.asarray(record[mode], dtype=np.float32),
                        seed=seed,
                        n_action_steps=args.n_action_steps,
                        max_episode_steps=args.max_episode_steps,
                        hdf5_path=_hdf5_path_for_env(env_name, args.hdf5_root) if args.hdf5_root is not None else None,
                    )
                    if video_dir is not None:
                        video_path = env.render()
                        sim["video_path"] = str(video_path) if video_path is not None else None
                finally:
                    env.close()
                if mode == "decoded":
                    mean_delta = float(record.get("mean_abs_delta", 0.0))
                    max_delta = float(record.get("max_abs_delta", 0.0))
                    mean_delta_norm = float(record.get("mean_abs_delta_norm", 0.0))
                    max_delta_norm = float(record.get("max_abs_delta_norm", 0.0))
                elif mode == "fast":
                    mean_delta = float(record.get("mean_abs_delta_fast", 0.0))
                    max_delta = float(record.get("max_abs_delta_fast", 0.0))
                    mean_delta_norm = float(record.get("mean_abs_delta_fast_norm", 0.0))
                    max_delta_norm = float(record.get("max_abs_delta_fast_norm", 0.0))
                else:
                    mean_delta = max_delta = mean_delta_norm = max_delta_norm = 0.0
                results.append(
                    {
                        "mode": mode,
                        "env_name": env_name,
                        "dataset_name": str(record["dataset_name"]),
                        "trajectory_id": int(record["trajectory_id"]),
                        "trajectory_length": int(record.get("trajectory_length", len(record[mode]))),
                        "sample_index": int(record.get("sample_index", record_index)),
                        "seed": seed,
                        "mean_abs_delta": mean_delta,
                        "max_abs_delta": max_delta,
                        "mean_abs_delta_norm": mean_delta_norm,
                        "max_abs_delta_norm": max_delta_norm,
                        **sim,
                    }
                )
                print(json.dumps(results[-1], ensure_ascii=False))
                jsonl_handle.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
                jsonl_handle.flush()

    summary: dict[str, Any] = {"actions_npz": str(args.actions_npz), "results": results, "summary": {}}
    for mode in modes:
        subset = [item for item in results if item["mode"] == mode]
        summary["summary"][mode] = {
            "episodes": len(subset),
            "successes": int(sum(bool(item["success"]) for item in subset)),
            "success_rate": float(np.mean([bool(item["success"]) for item in subset])) if subset else 0.0,
        }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote replay report to {args.output}")


if __name__ == "__main__":
    main()
