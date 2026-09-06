#!/usr/bin/env python3
"""Validate that a live policy server is the one requested by an eval worker."""

from __future__ import annotations

import argparse
import json
import os

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _require_int(metadata: dict, key: str, *, minimum: int) -> int:
    value = metadata.get(key)
    if type(value) is not int or value < minimum:
        raise RuntimeError(
            f"{key} must be an int >= {minimum} (bool is invalid), got {value!r}"
        )
    return value


def _require_bool(metadata: dict, key: str) -> bool:
    value = metadata.get(key)
    if type(value) is not bool:
        raise RuntimeError(f"{key} must be a bool, got {value!r}")
    return value


def _require_string_list(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if (
        type(value) is not list
        or (not allow_empty and not value)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeError(
            f"{label} must be a {'possibly empty ' if allow_empty else ''}"
            f"list of unique, non-empty strings, got {value!r}"
        )
    return list(value)


def _require_key_dims(
    keys: list[str], raw_dims: object, total_dim: int, label: str
) -> dict[str, int]:
    if type(raw_dims) is not dict or list(raw_dims) != keys:
        raise RuntimeError(
            f"{label}_key_dims must be a dict ordered exactly like {label}_keys; "
            f"keys={keys!r}, dims={raw_dims!r}"
        )
    if any(
        type(key) is not str
        or type(value) is not int
        or value <= 0
        for key, value in raw_dims.items()
    ):
        raise RuntimeError(
            f"{label}_key_dims values must be positive ints (bool is invalid), "
            f"got {raw_dims!r}"
        )
    if sum(raw_dims[key] for key in keys) != total_dim:
        raise RuntimeError(
            f"{label}_key_dims do not sum to {label}_dim={total_dim}: {raw_dims!r}"
        )
    return dict(raw_dims)


def _require_checkpoint_identity(value: object) -> dict[str, int]:
    if type(value) is not dict:
        raise RuntimeError(f"checkpoint_identity must be a dict, got {value!r}")
    if set(value) != {"size", "mtime_ns"}:
        raise RuntimeError(
            "checkpoint_identity must contain exactly size and mtime_ns, "
            f"got {value!r}"
        )
    if any(type(value[key]) is not int or value[key] < 0 for key in value):
        raise RuntimeError(
            "checkpoint_identity values must be non-negative ints "
            f"(bool is invalid), got {value!r}"
        )
    return {"size": value["size"], "mtime_ns": value["mtime_ns"]}


def _stat_identity(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-ckpt", required=True)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    client = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        connect_timeout=args.timeout,
        handshake_timeout=args.timeout,
        request_timeout=args.timeout,
    )
    try:
        metadata = client.get_server_metadata()
        if metadata.get("env") != "starvla_policy_server":
            raise RuntimeError(
                f"unexpected server identity: {metadata.get('env')!r}"
            )
        protocol_version = _require_int(metadata, "protocol_version", minimum=1)
        if protocol_version != 2:
            raise RuntimeError(
                f"unsupported protocol_version={protocol_version}; expected 2. "
                "Restart the policy server with current code."
            )

        expected = os.path.realpath(args.expected_ckpt)
        advertised_raw = metadata.get("ckpt_path")
        if type(advertised_raw) is not str or not advertised_raw:
            raise RuntimeError(
                f"ckpt_path must be a non-empty string, got {advertised_raw!r}"
            )
        advertised = os.path.realpath(advertised_raw)
        if expected != advertised:
            raise RuntimeError(
                f"checkpoint mismatch: expected {expected}, server advertises {advertised}"
            )

        checkpoint_identity = _require_checkpoint_identity(
            metadata.get("checkpoint_identity")
        )
        current_identity = _stat_identity(expected)
        if checkpoint_identity != current_identity:
            raise RuntimeError(
                "checkpoint file changed after the server loaded it: "
                f"server={checkpoint_identity}, current={current_identity}"
            )

        chunk_size = _require_int(metadata, "action_chunk_size", minimum=1)

        expected_action_keys = [
            "action.x", "action.y", "action.z", "action.roll",
            "action.pitch", "action.yaw", "action.gripper",
        ]
        action_keys = _require_string_list(metadata.get("action_keys"), "action_keys")
        action_dim = _require_int(metadata, "action_dim", minimum=1)
        action_key_dims = _require_key_dims(
            action_keys, metadata.get("action_key_dims"), action_dim, "action"
        )
        if (
            action_keys != expected_action_keys
            or action_key_dims != {key: 1 for key in expected_action_keys}
            or action_dim != 7
        ):
            raise RuntimeError(
                "LIBERO action metadata mismatch: "
                f"keys={action_keys}, dims={action_key_dims}, action_dim={action_dim}"
            )

        available_keys = _require_string_list(
            metadata.get("available_unnorm_keys"), "available_unnorm_keys"
        )
        default_unnorm_key = metadata.get("default_unnorm_key")
        if (
            type(default_unnorm_key) is not str
            or not default_unnorm_key
            or default_unnorm_key not in available_keys
        ):
            raise RuntimeError(
                "default_unnorm_key must be a non-empty advertised key, "
                f"got default={default_unnorm_key!r}, available={available_keys!r}"
            )
        if args.unnorm_key is not None:
            if not args.unnorm_key or args.unnorm_key != default_unnorm_key:
                raise RuntimeError(
                    f"unnorm_key mismatch: requested {args.unnorm_key!r}, "
                    f"server default is {default_unnorm_key!r}"
                )

        proprio_enabled = _require_bool(metadata, "proprio_state_enabled")
        advertised_state_dim = _require_int(metadata, "state_dim", minimum=0)
        state_keys = _require_string_list(
            metadata.get("state_keys"), "state_keys", allow_empty=True
        )
        state_key_dims = _require_key_dims(
            state_keys,
            metadata.get("state_key_dims"),
            advertised_state_dim,
            "state",
        )
        state_dim = metadata.get("expected_state_dim")
        if state_dim is not None and (type(state_dim) is not int or state_dim <= 0):
            raise RuntimeError(
                "expected_state_dim must be null or a positive int "
                f"(bool is invalid), got {state_dim!r}"
            )
        if proprio_enabled:
            expected_state_keys = [
                "state.x", "state.y", "state.z", "state.roll",
                "state.pitch", "state.yaw", "state.pad", "state.gripper",
            ]
            if (
                state_dim is None
                or state_dim != 8
                or advertised_state_dim != 8
                or state_keys != expected_state_keys
                or state_key_dims != {key: 1 for key in expected_state_keys}
            ):
                raise RuntimeError(
                    "LIBERO state metadata mismatch: "
                    f"expected_state_dim={state_dim}, state_dim={advertised_state_dim}, "
                    f"state_keys={state_keys}, state_key_dims={state_key_dims}"
                )
        elif state_dim is not None:
            raise RuntimeError(
                "expected_state_dim must be null when proprio_state_enabled is false"
            )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "ckpt_path": advertised,
                    "checkpoint_identity": checkpoint_identity,
                    "action_chunk_size": chunk_size,
                    "expected_state_dim": metadata.get("expected_state_dim"),
                    "available_unnorm_keys": available_keys,
                },
                sort_keys=True,
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
