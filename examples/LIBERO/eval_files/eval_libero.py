import atexit
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import time

import imageio
import numpy as np
from PIL import Image
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_start: int = 0  # First task id to evaluate.
    task_count: int = -1  # Number of tasks to evaluate from task_start. -1 = all remaining tasks.
    trial_start: int = 0  # First initial-state / episode index to evaluate for each task.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_videos: bool = True
    save_only_success_videos: bool = False
    max_success_videos_per_task: int = -1
    image_views: str = "primary+wrist"  # primary+wrist matches the LIBERO report eval; also supports auto | primary.
    policy_image_size: int = 0  # 0 keeps render size; QwenVAR stage2 training used 224x224 PIL images.
    validate_inputs: bool = True
    min_image_mean: float = 2.0
    min_image_std: float = 1.0
    strict_trial_count: bool = True

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""
    eval_fingerprint: str = ""  # Manager-issued contract fingerprint recorded in every chunk log.

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True
    constrain_to_action_tokens: bool | None = None
    max_new_tokens: int | None = None

    job_name: str = "test"


def _resolve_reset_context_start(trial_start: int) -> int | None:
    """Resolve the optional reset-only replay prefix for an isolated trial."""
    raw_value = os.environ.get("RESET_CONTEXT_START")
    if raw_value is None or raw_value == "":
        return None
    if not raw_value.isdigit():
        raise ValueError(
            "RESET_CONTEXT_START must be a non-negative integer, "
            f"got {raw_value!r}"
        )
    reset_context_start = int(raw_value)
    if reset_context_start > trial_start:
        raise ValueError(
            "RESET_CONTEXT_START must be <= trial_start, "
            f"got {reset_context_start} > {trial_start}"
        )
    return reset_context_start


def _prime_reset_context(
    env,
    initial_states,
    *,
    reset_context_start: int | None,
    trial_start: int,
    task_id: int,
) -> None:
    """Replay only prior resets from the trial's canonical multi-trial chunk."""
    if reset_context_start is None:
        return
    for episode_idx in range(reset_context_start, trial_start):
        logging.info(
            "RESET_CONTEXT_PRIME task=%s episode=%s target=%s",
            task_id,
            episode_idx,
            trial_start,
        )
        env.reset()
        env.set_init_state(initial_states[episode_idx])


RESET_RNG_CONTEXT_VERSION = 1


def _numpy_rng_state_payload(state: tuple) -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = state
    return {
        "algorithm": str(algorithm),
        "keys": [int(value) for value in np.asarray(keys, dtype=np.uint32)],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_state_from_payload(payload: dict) -> tuple:
    if not isinstance(payload, dict) or set(payload) != {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("invalid NumPy RNG state payload keys")
    if payload["algorithm"] != "MT19937":
        raise ValueError(f"unsupported NumPy RNG algorithm: {payload['algorithm']!r}")
    keys = payload["keys"]
    if (
        not isinstance(keys, list)
        or len(keys) != 624
        or any(isinstance(value, bool) or not isinstance(value, int) for value in keys)
        or any(value < 0 or value > np.iinfo(np.uint32).max for value in keys)
    ):
        raise ValueError("invalid MT19937 key array")
    position = payload["position"]
    has_gauss = payload["has_gauss"]
    cached_gaussian = payload["cached_gaussian"]
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= 624:
        raise ValueError("invalid MT19937 position")
    if isinstance(has_gauss, bool) or has_gauss not in (0, 1):
        raise ValueError("invalid MT19937 has_gauss flag")
    if isinstance(cached_gaussian, bool) or not isinstance(cached_gaussian, (int, float)):
        raise ValueError("invalid MT19937 cached Gaussian")
    if not math.isfinite(float(cached_gaussian)):
        raise ValueError("non-finite MT19937 cached Gaussian")
    return (
        "MT19937",
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        float(cached_gaussian),
    )


def _rng_payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_reset_rng_context(
    path: pathlib.Path,
    *,
    suite: str,
    task_id: int,
    seed: int,
    nominal_chunk: int,
    ordinal: int,
    eval_fingerprint: str,
) -> tuple[tuple, str]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reset RNG context {path}: {exc}") from exc
    expected = {
        "version": RESET_RNG_CONTEXT_VERSION,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "nominal_chunk": nominal_chunk,
        "ordinal": ordinal,
        "eval_fingerprint": eval_fingerprint,
    }
    if not isinstance(document, dict):
        raise ValueError(f"reset RNG context is not an object: {path}")
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(
                f"reset RNG context {key} mismatch in {path}: "
                f"expected {value!r}, got {document.get(key)!r}"
            )
    state_payload = document.get("numpy_state")
    state_sha256 = document.get("numpy_state_sha256")
    if not isinstance(state_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", state_sha256):
        raise ValueError(f"invalid reset RNG state digest in {path}")
    if _rng_payload_sha256(state_payload) != state_sha256:
        raise ValueError(f"reset RNG state digest mismatch in {path}")
    return _numpy_rng_state_from_payload(state_payload), state_sha256


def _write_reset_rng_context(
    path: pathlib.Path,
    *,
    state: tuple,
    suite: str,
    task_id: int,
    seed: int,
    nominal_chunk: int,
    ordinal: int,
    eval_fingerprint: str,
) -> None:
    state_payload = _numpy_rng_state_payload(state)
    document = {
        "version": RESET_RNG_CONTEXT_VERSION,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "nominal_chunk": nominal_chunk,
        "ordinal": ordinal,
        "eval_fingerprint": eval_fingerprint,
        "numpy_state": state_payload,
        "numpy_state_sha256": _rng_payload_sha256(state_payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def materialize_reset_rng_context(
    output_path: str,
    input_path: str,
    suite: str,
    task_id: int,
    seed: int,
    nominal_chunk: int,
    ordinal: int,
    eval_fingerprint: str,
) -> None:
    """Build one reset-RNG ordinal; callers use one fresh process per transition."""
    destination = pathlib.Path(output_path)
    if ordinal < 0 or ordinal >= nominal_chunk:
        raise ValueError(f"ordinal {ordinal} is outside nominal chunk {nominal_chunk}")
    if destination.exists():
        _load_reset_rng_context(
            destination,
            suite=suite,
            task_id=task_id,
            seed=seed,
            nominal_chunk=nominal_chunk,
            ordinal=ordinal,
            eval_fingerprint=eval_fingerprint,
        )
        return

    if ordinal == 0:
        if input_path:
            raise ValueError("ordinal 0 must not have an input RNG context")
        np.random.seed(seed)
        state = np.random.get_state()
    else:
        if not input_path:
            raise ValueError(f"ordinal {ordinal} requires an input RNG context")
        prior_state, _ = _load_reset_rng_context(
            pathlib.Path(input_path),
            suite=suite,
            task_id=task_id,
            seed=seed,
            nominal_chunk=nominal_chunk,
            ordinal=ordinal - 1,
            eval_fingerprint=eval_fingerprint,
        )
        task_suite = benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(task_id)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        try:
            np.random.set_state(prior_state)
            env.reset()
            state = np.random.get_state()
        finally:
            env.close()

    _write_reset_rng_context(
        destination,
        state=state,
        suite=suite,
        task_id=task_id,
        seed=seed,
        nominal_chunk=nominal_chunk,
        ordinal=ordinal,
        eval_fingerprint=eval_fingerprint,
    )


def _resolve_reset_rng_context(
    *,
    suite: str,
    task_id: int,
    trial_start: int,
    seed: int,
    eval_fingerprint: str,
    reset_context_start: int | None,
) -> dict | None:
    raw_path = os.environ.get("RESET_RNG_CONTEXT_PATH", "")
    if not raw_path:
        return None
    if reset_context_start is None:
        raise ValueError("RESET_RNG_CONTEXT_PATH requires RESET_CONTEXT_START")
    raw_chunk = os.environ.get("RESET_CONTEXT_CHUNK_SIZE", "")
    if not raw_chunk.isdigit() or int(raw_chunk) <= 0:
        raise ValueError("RESET_CONTEXT_CHUNK_SIZE must be a positive integer")
    nominal_chunk = int(raw_chunk)
    ordinal = trial_start - reset_context_start
    if ordinal < 0 or ordinal >= nominal_chunk:
        raise ValueError(
            f"reset RNG ordinal {ordinal} is outside nominal chunk {nominal_chunk}"
        )
    path = pathlib.Path(raw_path).expanduser().resolve(strict=True)
    state, state_sha256 = _load_reset_rng_context(
        path,
        suite=suite,
        task_id=task_id,
        seed=seed,
        nominal_chunk=nominal_chunk,
        ordinal=ordinal,
        eval_fingerprint=eval_fingerprint,
    )
    return {
        "state": state,
        "path": str(path),
        "state_sha256": state_sha256,
        "ordinal": ordinal,
        "nominal_chunk": nominal_chunk,
    }


def _select_policy_images(args: Args, client_model: ModelClient, primary_img: np.ndarray, wrist_img: np.ndarray) -> list:
    if args.image_views == "primary":
        return [primary_img]
    if args.image_views == "primary+wrist":
        return [primary_img, wrist_img]
    if args.image_views == "wrist+primary":
        return [wrist_img, primary_img]
    if args.image_views != "auto":
        raise ValueError(
            f"Unsupported image_views={args.image_views!r}; "
            "use auto, primary, primary+wrist, or wrist+primary"
        )

    # Align eval input with the checkpoint's training observation config.
    # LIBERO report checkpoints advertise obs: [image_0], so they should receive
    # the front agentview only. Multi-view checkpoints can opt in through config.
    obs_keys = set(getattr(client_model, "vla_obs", []) or [])
    if any(key in obs_keys for key in ("wrist_image", "image_1", "video.wrist_image")):
        return [primary_img, wrist_img]
    return [primary_img]


def _resize_policy_images(images: list, size: int) -> list:
    if size <= 0:
        return images
    resized = []
    for image in images:
        arr = np.asarray(image)
        if arr.shape[:2] == (size, size):
            resized.append(np.ascontiguousarray(arr))
            continue
        resized.append(np.asarray(Image.fromarray(arr).resize((size, size))).copy())
    return resized


def _orient_libero_image(image: np.ndarray) -> np.ndarray:
    orientation = os.getenv("LIBERO_IMAGE_ORIENTATION", "rot180")
    if orientation == "raw":
        return np.ascontiguousarray(image)
    if orientation == "flipud":
        return np.ascontiguousarray(image[::-1])
    if orientation == "fliplr":
        return np.ascontiguousarray(image[:, ::-1])
    if orientation == "rot180":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(
        f"Unsupported LIBERO_IMAGE_ORIENTATION={orientation!r}; "
        "use raw, flipud, fliplr, or rot180"
    )


def _image_summary(image: np.ndarray) -> dict:
    arr = np.asarray(image)
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _validate_policy_images(images: list, args: Args, *, task_id: int, episode_idx: int, step: int) -> None:
    if not args.validate_inputs:
        return
    for view_idx, image in enumerate(images):
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(
                f"Invalid policy image shape at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: shape={arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid policy image dtype at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: dtype={arr.dtype}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite policy image at task={task_id} episode={episode_idx} step={step} view={view_idx}"
            )
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        max_value = float(np.max(arr))
        if mean < args.min_image_mean or std < args.min_image_std or max_value <= args.min_image_mean:
            raise RuntimeError(
                f"Degenerate policy image at task={task_id} episode={episode_idx} step={step} "
                f"view={view_idx}: mean={mean:.3f}, std={std:.3f}, max={max_value:.3f}"
            )


def _build_libero_state(obs: dict, expected_dim: int | None) -> np.ndarray:
    """Build the checkpoint's training-time LIBERO proprio vector.

    Legacy checkpoints used one gripper coordinate (7-D total); current
    ProductVQ checkpoints advertise both gripper joint positions (8-D total).
    """
    pose = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()
            ).astype(np.float32).reshape(-1),
        )
    )
    if pose.shape != (6,):
        raise ValueError(
            f"Unexpected LIBERO end-effector pose shape {pose.shape}; expected (6,)"
        )
    target_dim = 7 if expected_dim is None else int(expected_dim)
    gripper_count = target_dim - pose.size
    gripper_q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_count < 0 or gripper_count > gripper_q.size:
        raise ValueError(
            f"Cannot construct {target_dim}-D LIBERO state: pose has {pose.size} values, "
            f"observation has {gripper_q.size} gripper values"
        )
    return np.concatenate((pose, gripper_q[:gripper_count])).astype(np.float32)


def _validate_state(
    state: np.ndarray,
    *,
    expected_dim: int,
    task_id: int,
    episode_idx: int,
    step: int,
) -> None:
    if state.shape != (expected_dim,):
        raise ValueError(
            f"Unexpected LIBERO state shape {state.shape}; expected ({expected_dim},)"
        )
    if not np.isfinite(state).all():
        raise RuntimeError(
            f"Non-finite LIBERO state at task={task_id} episode={episode_idx} step={step}: {state}"
        )


def _video_path_for_episode(
    args: Args,
    *,
    task_id: int,
    task_description: str,
    episode_idx: int,
    success: bool,
) -> pathlib.Path | None:
    if not args.save_videos:
        return None
    if args.save_only_success_videos and not success:
        return None

    root = pathlib.Path(args.video_out_path)
    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")

    if success and args.max_success_videos_per_task >= 0:
        task_dir = root / f"task_{task_id:02d}" / "success"
        task_dir.mkdir(parents=True, exist_ok=True)
        if len(list(task_dir.glob("*.mp4"))) >= args.max_success_videos_per_task:
            return None
        return task_dir / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"

    root.mkdir(parents=True, exist_ok=True)
    return root / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"


def _build_eval_run_metadata(
    args: Args,
    client_model: ModelClient,
    *,
    effective_task_count: int,
    reset_context_start: int | None,
    reset_rng_context: dict | None,
) -> dict:
    server_meta = getattr(client_model, "_server_metadata", None)
    if not isinstance(server_meta, dict):
        raise RuntimeError("ModelClient did not expose policy-server handshake metadata")

    checkpoint_value = args.pretrained_path or server_meta.get("ckpt_path")
    if not checkpoint_value:
        raise ValueError("pretrained_path and server checkpoint metadata are both empty")
    checkpoint_path = pathlib.Path(str(checkpoint_value)).expanduser().resolve(strict=True)
    checkpoint_stat = checkpoint_path.stat()

    raw_server_checkpoint = server_meta.get("ckpt_path")
    if not raw_server_checkpoint:
        raise RuntimeError("Policy server did not advertise ckpt_path")
    server_checkpoint_path = pathlib.Path(str(raw_server_checkpoint)).expanduser().resolve(strict=True)
    if checkpoint_path != server_checkpoint_path:
        raise RuntimeError(
            f"Checkpoint identity changed after handshake: eval={checkpoint_path}, "
            f"server={server_checkpoint_path}"
        )

    expected_state_dim = server_meta.get("expected_state_dim")
    return {
        "contract_version": 2,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_base": checkpoint_path.stem,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "suite": args.task_suite_name,
        "task_start": args.task_start,
        "task_count": effective_task_count,
        "trial_start": args.trial_start,
        "reset_context_priming_version": (
            1
            if reset_context_start is not None and reset_rng_context is None
            else None
        ),
        "reset_context_start": reset_context_start,
        "reset_context_count": (
            0
            if reset_context_start is None
            else (
                args.trial_start - reset_context_start
                if reset_rng_context is None
                else None
            )
        ),
        "reset_rng_context_version": (
            RESET_RNG_CONTEXT_VERSION if reset_rng_context is not None else None
        ),
        "reset_rng_context_ordinal": (
            None if reset_rng_context is None else reset_rng_context["ordinal"]
        ),
        "reset_rng_context_state_sha256": (
            None if reset_rng_context is None else reset_rng_context["state_sha256"]
        ),
        "num_trials": args.num_trials_per_task,
        "seed": args.seed,
        "unnorm_key": args.unnorm_key,
        "image_views": args.image_views,
        "policy_image_size": args.policy_image_size,
        "eval_fingerprint": args.eval_fingerprint,
        "server_protocol_version": int(server_meta.get("protocol_version", 0)),
        "server_action_chunk_size": int(server_meta.get("action_chunk_size", 0)),
        "server_action_dim": int(server_meta.get("action_dim", 0)),
        "server_state_dim": int(server_meta.get("state_dim", 0)),
        "server_expected_state_dim": (
            None if expected_state_dim is None else int(expected_state_dim)
        ),
    }


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        constrain_to_action_tokens=args.constrain_to_action_tokens,
        max_new_tokens=args.max_new_tokens,
        expected_ckpt_path=args.pretrained_path or None,
    )
    # Each chunk is a standalone CLI process. Close promptly on success and
    # retain an atexit fallback for every exception/termination path. This
    # prevents stale websocket sessions from accumulating across retries.
    atexit.register(client_model.close)

    if args.task_start < 0 or args.task_start >= num_tasks_in_suite:
        raise ValueError(f"task_start must be in [0, {num_tasks_in_suite}), got {args.task_start}")
    if args.trial_start < 0:
        raise ValueError(f"trial_start must be >= 0, got {args.trial_start}")
    reset_context_start = _resolve_reset_context_start(args.trial_start)

    # Optional smoke-test caps (still useful for quick verification with -1 = full run).
    remaining_tasks = num_tasks_in_suite - args.task_start
    task_limit = remaining_tasks if args.task_count <= 0 else min(args.task_count, remaining_tasks)
    if args.max_tasks > 0:
        task_limit = min(task_limit, args.max_tasks)
    task_ids = list(range(args.task_start, args.task_start + task_limit))
    logging.info(
        f"Evaluating {len(task_ids)} of {num_tasks_in_suite} tasks "
        f"(task_start={args.task_start}, task_count={args.task_count}, max_tasks={args.max_tasks})"
    )

    reset_rng_context = _resolve_reset_rng_context(
        suite=args.task_suite_name,
        task_id=args.task_start,
        trial_start=args.trial_start,
        seed=args.seed,
        eval_fingerprint=args.eval_fingerprint,
        reset_context_start=reset_context_start,
    )
    if reset_rng_context is not None and (
        len(task_ids) != 1 or args.num_trials_per_task != 1
    ):
        raise ValueError("reset RNG context is allowed only for one task and one trial")

    run_metadata = _build_eval_run_metadata(
        args,
        client_model,
        effective_task_count=len(task_ids),
        reset_context_start=reset_context_start,
        reset_rng_context=reset_rng_context,
    )
    logging.info("EVAL_RUN_META_JSON %s", json.dumps(run_metadata, sort_keys=True, separators=(",", ":")))

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            trial_end = min(args.trial_start + args.num_trials_per_task, len(initial_states))
            if args.strict_trial_count and trial_end - args.trial_start != args.num_trials_per_task:
                raise RuntimeError(
                    f"Requested {args.num_trials_per_task} trials from trial_start={args.trial_start}, "
                    f"but task={task_id} only has {len(initial_states)} initial states"
                )
            if reset_context_start is not None and reset_rng_context is None:
                logging.info(
                    "Priming reset context for task=%s over initial states [%s, %s)",
                    task_id,
                    reset_context_start,
                    args.trial_start,
                )
            _prime_reset_context(
                env,
                initial_states,
                reset_context_start=(
                    reset_context_start if reset_rng_context is None else None
                ),
                trial_start=args.trial_start,
                task_id=task_id,
            )
            logged_policy_input = False
            for episode_idx in tqdm.tqdm(range(args.trial_start, trial_end)):
                logging.info(f"\nTask id: {task_id}")
                logging.info(f"Task: {task_description}")

                # Reset environment
                client_model.reset(task_description=task_description)  # Reset the client connection
                if reset_rng_context is not None:
                    logging.info(
                        "RESTORE_RESET_RNG_CONTEXT task=%s episode=%s ordinal=%s state_sha256=%s",
                        task_id,
                        episode_idx,
                        reset_rng_context["ordinal"],
                        reset_rng_context["state_sha256"],
                    )
                    np.random.set_state(reset_rng_context["state"])
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                full_actions = []

                logging.info(f"Starting episode {episode_idx + 1}...")
                step = 0

                # full_actions = np.load("./debug/action.npy")

                while t < max_steps + args.num_steps_wait:
                    # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = _orient_libero_image(obs["agentview_image"])
                    wrist_img = _orient_libero_image(obs["robot0_eye_in_hand_image"])

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    state = _build_libero_state(obs, client_model.expected_state_dim)
                    if args.validate_inputs:
                        _validate_state(
                            state,
                            expected_dim=(
                                state.size
                                if client_model.expected_state_dim is None
                                else client_model.expected_state_dim
                            ),
                            task_id=task_id,
                            episode_idx=episode_idx,
                            step=step,
                        )

                    observation = {  #
                        "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                        "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }

                    # align key with model API --> two images provided here --> check training
                    policy_images = _select_policy_images(
                        args,
                        client_model,
                        observation["observation.primary"][0],
                        observation["observation.wrist_image"][0],
                    )
                    policy_images = _resize_policy_images(policy_images, args.policy_image_size)
                    _validate_policy_images(
                        policy_images,
                        args,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        step=step,
                    )
                    example_dict = {
                        "image": policy_images,
                        "lang": observation["instruction"][0],
                        "state": observation["observation.state"],
                    }
                    if not logged_policy_input:
                        logging.info(
                            "Policy input check: image_views=%s, policy_image_size=%s, image_summaries=%s, num_images=%d, state_shape=%s, server_vla_obs=%s",
                            args.image_views,
                            args.policy_image_size,
                            [_image_summary(image) for image in example_dict["image"]],
                            len(example_dict["image"]),
                            tuple(example_dict["state"].shape),
                            getattr(client_model, "vla_obs", []),
                        )
                        logged_policy_input = True

                    start_time = time.time()

                    response = client_model.step(example=example_dict, step=step)

                    end_time = time.time()
                    # print(f"time: {end_time - start_time}")

                    # #
                    raw_action = response["raw_action"]

                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        logging.warning(
                            f"Unexpected action sizes: "
                            f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                            f"Falling back to LIBERO_DUMMY_ACTION."
                        )
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )
                    else:
                        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                    if not np.isfinite(delta_action).all():
                        raise RuntimeError(
                            f"Non-finite action at task={task_id} episode={episode_idx} step={step}: "
                            f"{delta_action}"
                        )

                    full_actions.append(delta_action)
                    if step == 0:
                        logging.info(
                            "First env action: finite=%s, min=%.6f, max=%.6f, values=%s",
                            bool(np.isfinite(delta_action).all()),
                            float(np.nanmin(delta_action)),
                            float(np.nanmax(delta_action)),
                            np.array2string(delta_action, precision=6, suppress_small=False),
                        )

                    # __import__("ipdb").set_trace()
                    # see ../robosuite/controllers/controller_factory.py
                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode.
                video_path = _video_path_for_episode(
                    args,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    success=bool(done),
                )
                if video_path is not None:
                    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
                    logging.info(f"Saved replay video: {video_path}")

                full_actions = np.stack(full_actions)
                # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

                # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        finally:
            env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    client_model.close()
    atexit.unregister(client_model.close)
    logging.info("EVAL_CHUNK_OK")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        start_debugpy_once()
    tyro.cli(eval_libero)
