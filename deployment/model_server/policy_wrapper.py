# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def _as_float32_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(
            f"{label} must have a real floating dtype, got {array.dtype}"
        )
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = os.path.realpath(str(ckpt_path))
        if not os.path.isfile(self._ckpt_path):
            raise FileNotFoundError(
                f"PolicyServerWrapper: checkpoint does not exist or is not a file: {self._ckpt_path}"
            )
        checkpoint_stat_before_load = os.stat(self._ckpt_path)
        checkpoint_identity_before_load = {
            "size": int(checkpoint_stat_before_load.st_size),
            "mtime_ns": int(checkpoint_stat_before_load.st_mtime_ns),
        }

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework
        checkpoint_stat_after_load = os.stat(self._ckpt_path)
        self._checkpoint_identity = {
            "size": int(checkpoint_stat_after_load.st_size),
            "mtime_ns": int(checkpoint_stat_after_load.st_mtime_ns),
        }
        if self._checkpoint_identity != checkpoint_identity_before_load:
            raise RuntimeError(
                "Checkpoint changed while the policy server was loading it: "
                f"before={checkpoint_identity_before_load}, "
                f"after={self._checkpoint_identity}"
            )

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        framework_cfg = model_cfg.get("framework", {})
        proprio_cfg = framework_cfg.get("proprio_state", {})
        if not isinstance(proprio_cfg, dict):
            proprio_cfg = {}
        self._proprio_state_enabled = bool(
            getattr(framework, "use_proprio_state", False)
            or proprio_cfg.get("enabled", False)
        )
        configured_state_dim = getattr(framework, "proprio_state_dim", None)
        if configured_state_dim is None:
            configured_state_dim = proprio_cfg.get("state_dim")
        self._expected_state_dim = (
            int(configured_state_dim)
            if self._proprio_state_enabled and configured_state_dim is not None
            else None
        )
        if self._expected_state_dim is not None and self._expected_state_dim <= 0:
            raise ValueError(f"Invalid proprio state_dim={self._expected_state_dim}")

        # action_chunk_size = future_action_window_size + 1 for legacy configs.
        action_model_cfg = framework_cfg.get("action_model", {})

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        elif hasattr(framework, "action_horizon"):
            self._action_chunk_size = int(framework.action_horizon)
        elif hasattr(framework, "stage1_tokenizer") and hasattr(framework.stage1_tokenizer, "seq_len"):
            self._action_chunk_size = int(framework.stage1_tokenizer.seq_len)
        else:
            raise ValueError(
                f"PolicyServerWrapper: could not infer action_chunk_size for {self._ckpt_path}; "
                "expected framework.action_model.action_horizon, future_action_window_size, "
                "framework.action_horizon, or framework.stage1_tokenizer.seq_len."
            )
        if self._action_chunk_size <= 0:
            raise ValueError(
                f"PolicyServerWrapper: invalid action_chunk_size={self._action_chunk_size}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())
        if (
            not self._available_unnorm_keys
            or any(
                type(key) is not str or not key
                for key in self._available_unnorm_keys
            )
            or len(set(self._available_unnorm_keys))
            != len(self._available_unnorm_keys)
        ):
            raise ValueError(
                "PolicyServerWrapper: available unnorm keys must be unique, "
                f"non-empty strings; got {self._available_unnorm_keys!r}"
            )

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            if self._proprio_state_enabled and self._expected_state_dim is None:
                self._expected_state_dim = default_proc.state_dim
            # PolicyNormProcessor.state_dim is the flattened raw state
            # dimension represented by dataset statistics. The framework
            # dimension is the post-transform model input dimension, so they
            # need not be equal (GR1 is 29 raw dimensions and 58 sin/cos
            # dimensions). Request payloads are validated against the model
            # dimension in predict_action below.
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s, "
                "raw_state_dim=%s, expected_model_state_dim=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
                default_proc.state_dim,
                self._expected_state_dim,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "protocol_version": 2,
            "ckpt_path": self._ckpt_path,
            "checkpoint_identity": dict(self._checkpoint_identity),
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "proprio_state_enabled": self._proprio_state_enabled,
            "expected_state_dim": self._expected_state_dim,
            "vla_obs": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("obs", []),
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
            base["action_key_dims"] = proc.action_key_dims
            base["state_key_dims"] = proc.state_key_dims
            base["action_dim"] = proc.action_dim
            base["state_dim"] = proc.state_dim
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        if type(examples) is not list or not examples:
            raise TypeError("predict_action: examples must be a non-empty list")
        validated_examples: List[dict] = []
        for index, example in enumerate(examples):
            if type(example) is not dict:
                raise TypeError(
                    f"predict_action: example {index} must be a dict, "
                    f"got {type(example).__name__}"
                )
            validated_examples.append(dict(example))

        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)
        expected_state_dim = self._expected_state_dim
        if self._proprio_state_enabled:
            if expected_state_dim is None:
                expected_state_dim = proc.state_dim
            for index, example in enumerate(validated_examples):
                if "state" not in example:
                    raise ValueError(
                        f"Missing proprio state for example {index}; expected final dim {expected_state_dim}"
                    )
                state = _as_float32_array(
                    example["state"], label=f"Proprio state for example {index}"
                )
                if (
                    state.ndim not in (1, 2)
                    or state.shape[-1] != expected_state_dim
                    or (state.ndim == 2 and state.shape[0] <= 0)
                ):
                    raise ValueError(
                        f"State dim mismatch for example {index}: "
                        f"got shape {state.shape}, expected ({expected_state_dim},) "
                        f"or (T, {expected_state_dim}) with T > 0"
                    )
                example["state"] = state

        out = self._framework.predict_action(examples=validated_examples, **kwargs)
        if "normalized_actions" not in out:
            raise KeyError("Framework output is missing normalized_actions")
        normalized = _as_float32_array(
            out["normalized_actions"], label="Framework normalized actions"
        )
        expected_shape = (len(examples), self._action_chunk_size, proc.action_dim)
        if normalized.shape != expected_shape:
            raise ValueError(
                "Framework normalized action shape mismatch: "
                f"got {normalized.shape}, expected {expected_shape}"
            )
        normalized_for_unnorm = normalized
        if os.getenv("CLIP_NORMALIZED_ACTIONS", "0") == "1":
            normalized_for_unnorm = normalized.copy()
            normalized_for_unnorm[..., :6] = np.clip(normalized_for_unnorm[..., :6], -1.0, 1.0)
            if normalized_for_unnorm.shape[-1] > 6:
                normalized_for_unnorm[..., 6:] = np.clip(normalized_for_unnorm[..., 6:], 0.0, 1.0)
        if (
            os.getenv("FAST_BINARY_GRIPPER_BEFORE_UNNORM", "0") == "1"
            and normalized_for_unnorm.shape[-1] > 6
        ):
            if normalized_for_unnorm is normalized:
                normalized_for_unnorm = normalized.copy()
            normalized_for_unnorm[..., 6] = np.where(normalized_for_unnorm[..., 6] < 0.5, 0.0, 1.0)

        unnorm = _as_float32_array(
            np.stack(
                [
                    proc.unapply_actions(normalized_for_unnorm[b])
                    for b in range(normalized_for_unnorm.shape[0])
                ],
                axis=0,
            ),
            label="Unnormalized actions",
        )
        if unnorm.shape != expected_shape:
            raise ValueError(
                "Action unnormalization shape mismatch: "
                f"got {unnorm.shape}, expected {expected_shape}"
            )
        result: Dict[str, Any] = {"actions": unnorm, "normalized_actions": normalized_for_unnorm}
        for key in ("action_tokens", "generation_diagnostics"):
            if key in out:
                result[key] = out[key]
        return result
