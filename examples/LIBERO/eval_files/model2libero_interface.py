# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

from collections import deque
import os
import logging
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


POLICY_PROTOCOL_VERSION = 2
LIBERO_STATE_KEYS = (
    "state.x",
    "state.y",
    "state.z",
    "state.roll",
    "state.pitch",
    "state.yaw",
    "state.pad",
    "state.gripper",
)
LIBERO_ACTION_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)


def _require_int(metadata: dict, key: str, *, minimum: int) -> int:
    value = metadata.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{key} must be an int >= {minimum} (bool is invalid), got {value!r}"
        )
    return value


def _require_bool(metadata: dict, key: str) -> bool:
    value = metadata.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool, got {value!r}")
    return value


def _require_string_list(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if (
        type(value) is not list
        or (not allow_empty and not value)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{label} must be a {'possibly empty ' if allow_empty else ''}"
            f"list of unique, non-empty strings, got {value!r}"
        )
    return list(value)


def _validate_key_dims(
    keys: list[str], raw_dims: object, total_dim: int, label: str
) -> dict[str, int]:
    if type(raw_dims) is not dict or list(raw_dims) != keys:
        raise ValueError(
            f"{label}_key_dims must be a dict ordered exactly like {label}_keys; "
            f"keys={keys!r}, dims={raw_dims!r}"
        )
    if any(
        type(key) is not str
        or type(value) is not int
        or value <= 0
        for key, value in raw_dims.items()
    ):
        raise ValueError(
            f"{label}_key_dims values must be positive ints (bool is invalid), "
            f"got {raw_dims!r}"
        )
    flattened_dim = sum(raw_dims[key] for key in keys)
    if flattened_dim != total_dim:
        raise ValueError(
            f"Inconsistent {label} dimension metadata: dims sum to {flattened_dim}, "
            f"server advertises {total_dim}"
        )
    return dict(raw_dims)


def _require_checkpoint_identity(value: object) -> dict[str, int]:
    if type(value) is not dict or set(value) != {"size", "mtime_ns"}:
        raise ValueError(
            "checkpoint_identity must be a dict containing exactly size and "
            f"mtime_ns, got {value!r}"
        )
    if any(type(value[key]) is not int or value[key] < 0 for key in value):
        raise ValueError(
            "checkpoint_identity values must be non-negative ints "
            f"(bool is invalid), got {value!r}"
        )
    return {"size": value["size"], "mtime_ns": value["mtime_ns"]}


def _stat_identity(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _as_float32_array(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(
            f"{label} must have a real floating dtype, got {array.dtype}"
        )
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _connect_and_validate_server(
    host: str,
    port: int,
    *,
    expected_ckpt_path: Optional[str],
    unnorm_key: Optional[str],
) -> tuple[WebsocketClientPolicy, dict, dict]:
    client = WebsocketClientPolicy(host, port)
    try:
        meta = client.get_server_metadata()
        if type(meta) is not dict:
            raise RuntimeError(
                f"Policy-server metadata must be a dict, got {type(meta).__name__}"
            )
        if meta.get("env") != "starvla_policy_server":
            raise RuntimeError(f"Unexpected policy-server identity: {meta.get('env')!r}")
        protocol_version = _require_int(meta, "protocol_version", minimum=1)
        if protocol_version != POLICY_PROTOCOL_VERSION:
            raise RuntimeError(
                f"Unsupported policy-server protocol_version={protocol_version}; "
                f"expected {POLICY_PROTOCOL_VERSION}. Restart the policy server with current code."
            )
        action_chunk_size = _require_int(meta, "action_chunk_size", minimum=1)

        advertised_ckpt = meta.get("ckpt_path")
        if type(advertised_ckpt) is not str or not advertised_ckpt:
            raise RuntimeError(
                f"Policy server ckpt_path must be a non-empty string, got {advertised_ckpt!r}"
            )
        advertised_real = os.path.realpath(advertised_ckpt)
        if expected_ckpt_path:
            expected_real = os.path.realpath(expected_ckpt_path)
            if expected_real != advertised_real:
                raise RuntimeError(
                    "Connected to the wrong policy server: "
                    f"expected checkpoint {expected_real}, server advertises {advertised_real}"
                )
        else:
            expected_real = advertised_real

        checkpoint_identity = _require_checkpoint_identity(
            meta.get("checkpoint_identity")
        )
        current_identity = _stat_identity(expected_real)
        if checkpoint_identity != current_identity:
            raise RuntimeError(
                "Checkpoint file changed after the policy server loaded it: "
                f"server={checkpoint_identity}, current={current_identity}"
            )

        available_keys = _require_string_list(
            meta.get("available_unnorm_keys"), "available_unnorm_keys"
        )
        default_unnorm_key = meta.get("default_unnorm_key")
        if (
            type(default_unnorm_key) is not str
            or not default_unnorm_key
            or default_unnorm_key not in available_keys
        ):
            raise ValueError(
                "default_unnorm_key must be a non-empty advertised key, "
                f"got default={default_unnorm_key!r}, available={available_keys!r}"
            )
        if unnorm_key is not None:
            if type(unnorm_key) is not str or not unnorm_key:
                raise ValueError(
                    f"unnorm_key must be null or a non-empty string, got {unnorm_key!r}"
                )
            if unnorm_key != default_unnorm_key:
                raise ValueError(
                    f"unnorm_key mismatch: requested {unnorm_key!r}, "
                    f"server default is {default_unnorm_key!r}"
                )

        action_keys = _require_string_list(meta.get("action_keys"), "action_keys")
        action_dim = _require_int(meta, "action_dim", minimum=1)
        _validate_key_dims(action_keys, meta.get("action_key_dims"), action_dim, "action")
        if tuple(action_keys) != LIBERO_ACTION_KEYS or action_dim != len(LIBERO_ACTION_KEYS):
            raise ValueError(
                "LIBERO action contract mismatch: "
                f"keys={action_keys}, action_dim={action_dim}, expected={list(LIBERO_ACTION_KEYS)}"
            )

        state_keys = _require_string_list(
            meta.get("state_keys"), "state_keys", allow_empty=True
        )
        state_dim = _require_int(meta, "state_dim", minimum=0)
        _validate_key_dims(state_keys, meta.get("state_key_dims"), state_dim, "state")
        proprio_enabled = _require_bool(meta, "proprio_state_enabled")
        raw_expected_state_dim = meta.get("expected_state_dim")
        if raw_expected_state_dim is not None and (
            type(raw_expected_state_dim) is not int or raw_expected_state_dim <= 0
        ):
            raise ValueError(
                "expected_state_dim must be null or a positive int "
                f"(bool is invalid), got {raw_expected_state_dim!r}"
            )
        expected_state_dim = raw_expected_state_dim
        if proprio_enabled:
            if (
                expected_state_dim != len(LIBERO_STATE_KEYS)
                or state_dim != expected_state_dim
                or tuple(state_keys) != LIBERO_STATE_KEYS
            ):
                raise ValueError(
                    "LIBERO state contract mismatch: "
                    f"keys={state_keys}, state_dim={state_dim}, "
                    f"expected_state_dim={expected_state_dim}, expected={list(LIBERO_STATE_KEYS)}"
                )
        elif expected_state_dim is not None:
            raise ValueError(
                "expected_state_dim must be null when proprio_state_enabled is false"
            )

        vla_obs = _require_string_list(
            meta.get("vla_obs"), "vla_obs", allow_empty=True
        )

        parsed = {
            "action_chunk_size": action_chunk_size,
            "vla_obs": vla_obs,
            "state_keys": state_keys,
            "action_keys": action_keys,
            "proprio_state_enabled": proprio_enabled,
            "expected_state_dim": expected_state_dim,
            "expected_action_dim": action_dim,
            "checkpoint_identity": checkpoint_identity,
            "default_unnorm_key": default_unnorm_key,
        }
        return client, meta, parsed
    except BaseException:
        client.close()
        raise


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        constrain_to_action_tokens: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 10095,
        expected_ckpt_path: Optional[str] = None,
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client, self._server_metadata, parsed = _connect_and_validate_server(
            host,
            port,
            expected_ckpt_path=expected_ckpt_path,
            unnorm_key=unnorm_key,
        )
        meta = self._server_metadata
        self.action_chunk_size = parsed["action_chunk_size"]
        self.vla_obs = parsed["vla_obs"]
        self.state_keys = parsed["state_keys"]
        self.action_keys = parsed["action_keys"]
        self.proprio_state_enabled = parsed["proprio_state_enabled"]
        self.expected_state_dim = parsed["expected_state_dim"]
        self.expected_action_dim = parsed["expected_action_dim"]

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.constrain_to_action_tokens = constrain_to_action_tokens
        self.max_new_tokens = max_new_tokens
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ModelClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Refresh chunk if needed.
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            if type(example) is not dict:
                raise TypeError(
                    f"Policy example must be a dict, got {type(example).__name__}"
                )
            request_example = dict(example)
            if self.proprio_state_enabled:
                if "state" not in request_example:
                    raise ValueError(
                        "Policy example is missing required proprio state"
                    )
                state = _as_float32_array(
                    request_example["state"], label="LIBERO proprio state"
                )
                if (
                    state.ndim not in (1, 2)
                    or state.shape[-1] != self.expected_state_dim
                    or (state.ndim == 2 and state.shape[0] <= 0)
                ):
                    raise ValueError(
                        f"Invalid LIBERO state shape {state.shape}; expected "
                        f"({self.expected_state_dim},) or "
                        f"(T, {self.expected_state_dim}) with T > 0"
                    )
                request_example["state"] = state

            vla_input = {
                "examples": [request_example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            if self.constrain_to_action_tokens is not None:
                vla_input["constrain_to_action_tokens"] = self.constrain_to_action_tokens
            if self.max_new_tokens is not None:
                vla_input["max_new_tokens"] = self.max_new_tokens
            response = self.client.predict_action(vla_input)
            if response.get("ok") is False or response.get("status") == "error":
                error = response.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                code = error.get("code", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
                error_type = error.get("type", "UnknownError") if isinstance(error, dict) else "UnknownError"
                raise RuntimeError(
                    f"Policy server inference failed [{code}/{error_type}]: {message or response}"
                )
            data = response.get("data")
            if not isinstance(data, dict) or "actions" not in data:
                raise KeyError(
                    f"Key 'actions' not found in response data: "
                    f"keys={list(data.keys()) if isinstance(data, dict) else None}, "
                    f"full response={response}"
                )
            actions_batch = data["actions"]  # (B, T, D), unnormalized server-side
            actions_array = _as_float32_array(
                actions_batch, label="Policy server actions"
            )
            if actions_array.ndim != 3 or actions_array.shape[0] != 1:
                raise ValueError(
                    f"Invalid action batch shape {actions_array.shape}; expected (1, T, D)"
                )
            if actions_array.shape[1] != self.action_chunk_size:
                raise ValueError(
                    f"Action horizon mismatch: got T={actions_array.shape[1]}, "
                    f"server metadata says {self.action_chunk_size}"
                )
            if actions_array.shape[2] != self.expected_action_dim:
                raise ValueError(
                    f"Action dim mismatch: got D={actions_array.shape[2]}, "
                    f"expected {self.expected_action_dim}"
                )
            diagnostics = data.get("generation_diagnostics")
            action_tokens = data.get("action_tokens")
            normalized_actions = data.get("normalized_actions")
            if diagnostics is not None or action_tokens is not None or normalized_actions is not None:
                token_preview = None
                if action_tokens is not None:
                    first_tokens = action_tokens[0] if isinstance(action_tokens, (list, tuple)) else np.asarray(action_tokens, dtype=object)[0]
                    if first_tokens is not None:
                        token_preview = np.asarray(first_tokens).reshape(-1)[:16].tolist()
                norm_summary = None
                if normalized_actions is not None:
                    normalized_array = _as_float32_array(
                        normalized_actions,
                        label="Policy server normalized actions",
                    )
                    if normalized_array.shape != actions_array.shape:
                        raise ValueError(
                            "Normalized action shape mismatch: "
                            f"got {normalized_array.shape}, "
                            f"expected {actions_array.shape}"
                        )
                    norm_arr = normalized_array[0]
                    norm_summary = {
                        "shape": tuple(norm_arr.shape),
                        "min": float(np.nanmin(norm_arr)),
                        "max": float(np.nanmax(norm_arr)),
                        "first": np.array2string(norm_arr[0], precision=6, suppress_small=False),
                    }
                action_arr = actions_array[0]
                action_summary = {
                    "shape": tuple(action_arr.shape),
                    "min": float(np.nanmin(action_arr)),
                    "max": float(np.nanmax(action_arr)),
                    "first": np.array2string(action_arr[0], precision=6, suppress_small=False),
                }
                logging.info(
                    "Policy generation diagnostics: diagnostics=%s, first_action_tokens=%s, normalized=%s, unnormalized=%s",
                    diagnostics,
                    token_preview,
                    norm_summary,
                    action_summary,
                )
            self.raw_actions = actions_array[0]  # (T, D)

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
