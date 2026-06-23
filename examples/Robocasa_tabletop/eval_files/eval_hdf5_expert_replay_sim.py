"""Replay raw RoboCasa HDF5 expert actions with the original simulator controller."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import robocasa  # noqa: F401
import robosuite
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, reset_to


def _make_env(dataset_path: Path) -> Any:
    env_meta = get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    if isinstance(env_kwargs.get("render_camera"), list):
        env_kwargs["render_camera"] = env_kwargs["render_camera"][0]
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs)


def _check_success(env: Any) -> bool:
    for name in ("_check_success", "check_success"):
        if hasattr(env, name):
            return bool(getattr(env, name)())
    return False


class H264MP4Writer:
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
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
            str(path),
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Video writer stdin is closed.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        returncode = self.process.wait(timeout=120)
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {returncode}:\n{stderr[-2000:]}")


def _open_video_writer(path: Path, fps: float, width: int, height: int) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        return H264MP4Writer(path, fps, width, height)
    suffix = path.suffix.lower()
    codecs = ("MJPG",) if suffix == ".avi" else ("avc1", "H264", "mp4v")
    for codec in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo", type=str, required=True, help="HDF5 demo key, e.g. demo_611")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--camera", type=str, default="egoview")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=800)
    parser.add_argument("--video_skip", type=int, default=2)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.demo}"]
        states = group["states"][()]
        actions = group["actions"][()]
        initial_state = {
            "states": states[0],
            "model": group.attrs["model_file"],
            "ep_meta": group.attrs.get("ep_meta", None),
        }

    env = _make_env(args.dataset)
    reset_to(env, initial_state)

    writer = _open_video_writer(args.video, args.fps, args.render_width, args.render_height)
    success = False
    frames = 0
    errors: list[float] = []
    try:
        for i, action in enumerate(actions):
            env.step(action)
            success = success or _check_success(env)
            if i < len(states) - 1:
                state_playback = np.asarray(env.sim.get_state().flatten())
                if state_playback.shape == states[i + 1].shape:
                    errors.append(float(np.linalg.norm(states[i + 1] - state_playback)))
            if i % args.video_skip == 0:
                frame = env.sim.render(
                    height=args.render_height,
                    width=args.render_width,
                    camera_name=args.camera,
                )[::-1]
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames += 1
    finally:
        writer.release()
        env.close()

    report = {
        "dataset": str(args.dataset),
        "demo": args.demo,
        "video": str(args.video),
        "success": bool(success),
        "num_actions": int(len(actions)),
        "frames": int(frames),
        "state_error": {
            "first": errors[0] if errors else None,
            "median": float(np.median(errors)) if errors else None,
            "last": errors[-1] if errors else None,
            "max": max(errors) if errors else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
