#!/usr/bin/env python3
"""Strictly validate chunked LIBERO eval logs and summarize exact coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_VERSION = 2
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CONTRACT_CHUNK_KEYS = {
    "libero_spatial": "spatial_chunk_trials",
    "libero_object": "object_chunk_trials",
    "libero_goal": "goal_chunk_trials",
    "libero_10": "libero10_chunk_trials",
}
CHUNK_NAME_RE = re.compile(
    r"^(?P<checkpoint>.+)_stage2_chunked_t(?P<task>[0-9]+)"
    r"_r(?P<start>[0-9]+)_n(?P<count>[0-9]+)[.]log$"
)
# Older launchers placed policy-server diagnostics beside result chunks using
# names such as ``..._n5.server.log``. They are auxiliary logs, not malformed
# result chunks. Keep all other chunk-like filenames strict so typos still
# fail validation instead of being silently ignored.
AUXILIARY_CHUNK_LOG_SUFFIXES = (".server.log",)
TASK_ID_RE = re.compile(r"Task id:[ ]*([0-9]+)")
EPISODE_RE = re.compile(r"Starting episode[ ]+([0-9]+)")
SUCCESS_RE = re.compile(r"Success:[ ]*(True|False)")
TOTAL_EPISODES_RE = re.compile(r"Total episodes:[ ]*([0-9]+)[ ]*$", re.MULTILINE)
FLOAT_TOKEN = r"[+-]?(?:(?:[0-9]+(?:[.][0-9]*)?)|(?:[.][0-9]+))(?:[eE][+-]?[0-9]+)?"
TOTAL_RATE_RE = re.compile(
    rf"(?<!Current )Total success rate:[ ]*({FLOAT_TOKEN})[ ]*$",
    re.MULTILINE,
)
META_MARKER = "EVAL_RUN_META_JSON "
REQUIRED_META_KEYS = frozenset(
    {
        "contract_version",
        "checkpoint_path",
        "checkpoint_base",
        "checkpoint_size",
        "checkpoint_mtime_ns",
        "suite",
        "task_start",
        "task_count",
        "trial_start",
        "num_trials",
        "seed",
        "unnorm_key",
        "image_views",
        "policy_image_size",
        "eval_fingerprint",
        "server_protocol_version",
        "server_action_chunk_size",
        "server_action_dim",
        "server_state_dim",
        "server_expected_state_dim",
    }
)


class ValidationError(RuntimeError):
    pass


class ScheduleValidationError(ValidationError):
    """A structurally valid result log that does not match the eval schedule."""


@dataclass(frozen=True)
class Chunk:
    checkpoint: str
    suite: str
    task: int
    start: int
    count: int
    results: dict[int, bool]
    metadata: dict[str, Any]
    path: Path

    @property
    def eval_fingerprint(self) -> str:
        return str(self.metadata["eval_fingerprint"])


@dataclass(frozen=True)
class EvalSchedule:
    trials_per_task: int
    chunk_sizes: dict[str, int]


def _load_eval_schedule(
    contract_path: Path,
    expected_eval_fingerprint: str | None,
) -> tuple[EvalSchedule, str]:
    try:
        payload = contract_path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read eval contract {contract_path}: {exc}") from exc
    fingerprint = hashlib.sha256(payload).hexdigest()
    if (
        expected_eval_fingerprint is not None
        and fingerprint != expected_eval_fingerprint
    ):
        raise ValidationError(
            f"eval contract fingerprint mismatch: expected {expected_eval_fingerprint}, "
            f"got {fingerprint} from {contract_path}"
        )

    values: dict[str, str] = {}
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValidationError(f"eval contract is not UTF-8: {contract_path}") from exc
    for line_number, line in enumerate(lines, 1):
        key, separator, value = line.partition("=")
        if not separator:
            raise ValidationError(
                f"{contract_path}:{line_number}: malformed contract line"
            )
        if key in values:
            raise ValidationError(
                f"{contract_path}:{line_number}: duplicate contract key {key!r}"
            )
        values[key] = value

    def positive_int(key: str) -> int:
        raw_value = values.get(key)
        if raw_value is None or not raw_value.isdigit() or int(raw_value) <= 0:
            raise ValidationError(
                f"{contract_path}: contract {key} must be a positive integer"
            )
        return int(raw_value)

    if values.get("contract_version") != str(CONTRACT_VERSION):
        raise ValidationError(
            f"{contract_path}: unsupported contract_version="
            f"{values.get('contract_version')!r}"
        )
    return (
        EvalSchedule(
            trials_per_task=positive_int("trials_per_task"),
            chunk_sizes={
                suite: positive_int(contract_key)
                for suite, contract_key in CONTRACT_CHUNK_KEYS.items()
            },
        ),
        fingerprint,
    )


def _validate_chunk_schedule(chunk: Chunk, schedule: EvalSchedule) -> None:
    nominal_chunk = schedule.chunk_sizes[chunk.suite]
    if chunk.start >= schedule.trials_per_task or (
        chunk.start + chunk.count > schedule.trials_per_task
    ):
        raise ScheduleValidationError(
            f"{chunk.path}: chunk lies outside contract trials_per_task="
            f"{schedule.trials_per_task}"
        )
    canonical_start = (chunk.start // nominal_chunk) * nominal_chunk
    reset_context_start = chunk.metadata.get("reset_context_start")
    if reset_context_start is None:
        expected_count = min(
            nominal_chunk,
            schedule.trials_per_task - canonical_start,
        )
        if chunk.start != canonical_start or chunk.count != expected_count:
            raise ScheduleValidationError(
                f"{chunk.path}: unprimed chunk does not match canonical "
                f"schedule start={canonical_start}, count={expected_count}"
            )
    elif reset_context_start != canonical_start:
        raise ScheduleValidationError(
            f"{chunk.path}: reset_context_start={reset_context_start}, "
            f"expected canonical start {canonical_start}"
        )


def _parse_name(path: Path) -> tuple[str, int, int, int]:
    match = CHUNK_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValidationError(f"invalid chunk filename: {path}")
    checkpoint = match.group("checkpoint")
    task = int(match.group("task"))
    start = int(match.group("start"))
    count = int(match.group("count"))
    if task not in range(10):
        raise ValidationError(f"task id outside 0..9 in {path.name}")
    if count <= 0:
        raise ValidationError(f"non-positive chunk size in {path.name}")
    return checkpoint, task, start, count


def _require_int(
    metadata: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"metadata {key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValidationError(f"metadata {key} must be >= {minimum}, got {value}")
    return value


def _require_string(metadata: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValidationError(f"metadata {key} must be a string, got {value!r}")
    return value


def _parse_metadata(text: str, path: Path) -> dict[str, Any]:
    payloads = [
        line.split(META_MARKER, 1)[1].strip()
        for line in text.splitlines()
        if META_MARKER in line
    ]
    if len(payloads) != 1:
        raise ValidationError(
            f"{path}: expected exactly one {META_MARKER.strip()} line, got {len(payloads)}"
        )
    try:
        metadata = json.loads(payloads[0])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValidationError(f"{path}: invalid eval metadata JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValidationError(f"{path}: eval metadata must be a JSON object")
    missing = sorted(REQUIRED_META_KEYS - set(metadata))
    if missing:
        raise ValidationError(f"{path}: eval metadata is missing keys: {missing}")
    return metadata


def _validate_metadata(
    metadata: dict[str, Any],
    path: Path,
    *,
    filename_checkpoint: str,
    filename_task: int,
    filename_start: int,
    filename_count: int,
    expected_eval_fingerprint: str | None,
) -> str:
    contract_version = _require_int(metadata, "contract_version")
    if contract_version != CONTRACT_VERSION:
        raise ValidationError(
            f"{path}: metadata contract_version={contract_version}, expected {CONTRACT_VERSION}"
        )

    checkpoint_text = _require_string(metadata, "checkpoint_path")
    checkpoint_path = Path(checkpoint_text)
    if not checkpoint_path.is_absolute():
        raise ValidationError(f"{path}: checkpoint_path is not absolute: {checkpoint_text}")
    try:
        canonical_checkpoint = checkpoint_path.resolve(strict=True)
        checkpoint_stat = canonical_checkpoint.stat()
    except (OSError, RuntimeError) as exc:
        raise ValidationError(f"{path}: checkpoint_path cannot be resolved/stat'ed: {exc}") from exc
    if str(canonical_checkpoint) != checkpoint_text:
        raise ValidationError(
            f"{path}: checkpoint_path is not canonical: {checkpoint_text} != {canonical_checkpoint}"
        )

    checkpoint_base = _require_string(metadata, "checkpoint_base")
    if checkpoint_base != canonical_checkpoint.stem:
        raise ValidationError(
            f"{path}: checkpoint_base={checkpoint_base!r} does not match canonical file "
            f"base={canonical_checkpoint.stem!r}"
        )
    if checkpoint_base != filename_checkpoint:
        raise ValidationError(
            f"{path}: metadata checkpoint={checkpoint_base!r}, filename checkpoint={filename_checkpoint!r}"
        )
    if _require_int(metadata, "checkpoint_size", minimum=1) != checkpoint_stat.st_size:
        raise ValidationError(f"{path}: checkpoint size changed since evaluation")
    if _require_int(metadata, "checkpoint_mtime_ns", minimum=0) != checkpoint_stat.st_mtime_ns:
        raise ValidationError(f"{path}: checkpoint mtime_ns changed since evaluation")

    suite = _require_string(metadata, "suite")
    path_suite = path.parent.name
    if suite not in SUITES:
        raise ValidationError(f"{path}: unsupported metadata suite={suite!r}")
    if path_suite != suite:
        raise ValidationError(
            f"{path}: metadata suite={suite!r}, parent directory suite={path_suite!r}"
        )
    if _require_int(metadata, "task_start", minimum=0) != filename_task:
        raise ValidationError(f"{path}: metadata task_start does not match filename task")
    if _require_int(metadata, "task_count", minimum=1) != 1:
        raise ValidationError(f"{path}: chunk log metadata task_count must be exactly 1")
    trial_start = _require_int(metadata, "trial_start", minimum=0)
    if trial_start != filename_start:
        raise ValidationError(f"{path}: metadata trial_start does not match filename start")
    num_trials = _require_int(metadata, "num_trials", minimum=1)
    if num_trials != filename_count:
        raise ValidationError(f"{path}: metadata num_trials does not match filename count")

    reset_context_start = metadata.get("reset_context_start")
    reset_context_version = metadata.get("reset_context_priming_version")
    reset_context_count = metadata.get("reset_context_count")
    reset_rng_version = metadata.get("reset_rng_context_version")
    reset_rng_ordinal = metadata.get("reset_rng_context_ordinal")
    reset_rng_sha256 = metadata.get("reset_rng_context_state_sha256")
    if reset_context_start is None:
        if reset_context_version is not None:
            raise ValidationError(
                f"{path}: reset_context_priming_version requires reset_context_start"
            )
        if reset_context_count is not None:
            if (
                isinstance(reset_context_count, bool)
                or not isinstance(reset_context_count, int)
                or reset_context_count != 0
            ):
                raise ValidationError(
                    f"{path}: reset_context_count must be 0 when reset context is disabled"
                )
        if any(
            value is not None
            for value in (reset_rng_version, reset_rng_ordinal, reset_rng_sha256)
        ):
            raise ValidationError(
                f"{path}: reset RNG context metadata requires reset_context_start"
            )
    else:
        if (
            isinstance(reset_context_start, bool)
            or not isinstance(reset_context_start, int)
            or not 0 <= reset_context_start <= trial_start
        ):
            raise ValidationError(
                f"{path}: invalid reset_context_start={reset_context_start!r}"
            )
        if num_trials != 1:
            raise ValidationError(
                f"{path}: reset-context recovery is allowed only for a single-trial chunk"
            )
        expected_reset_count = trial_start - reset_context_start
        if isinstance(reset_context_version, bool) or reset_context_version not in (None, 1):
            raise ValidationError(
                f"{path}: unsupported reset_context_priming_version={reset_context_version!r}"
            )
        if isinstance(reset_rng_version, bool) or reset_rng_version not in (None, 1):
            raise ValidationError(
                f"{path}: unsupported reset_rng_context_version={reset_rng_version!r}"
            )
        uses_legacy_priming = reset_context_version == 1
        uses_rng_restore = reset_rng_version == 1
        if uses_legacy_priming == uses_rng_restore:
            raise ValidationError(
                f"{path}: reset context must use exactly one supported recovery strategy"
            )
        if uses_legacy_priming:
            if (
                isinstance(reset_context_count, bool)
                or not isinstance(reset_context_count, int)
                or reset_context_count != expected_reset_count
            ):
                raise ValidationError(
                    f"{path}: reset_context_count={reset_context_count!r}, "
                    f"expected {expected_reset_count}"
                )
            if any(value is not None for value in (reset_rng_ordinal, reset_rng_sha256)):
                raise ValidationError(
                    f"{path}: legacy reset priming cannot include reset RNG metadata"
                )
        else:
            if reset_context_version is not None or reset_context_count not in (None, 0):
                raise ValidationError(
                    f"{path}: reset RNG recovery cannot include priming metadata"
                )
            if (
                isinstance(reset_rng_ordinal, bool)
                or not isinstance(reset_rng_ordinal, int)
                or reset_rng_ordinal != expected_reset_count
            ):
                raise ValidationError(
                    f"{path}: reset_rng_context_ordinal={reset_rng_ordinal!r}, "
                    f"expected {expected_reset_count}"
                )
            if not isinstance(reset_rng_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", reset_rng_sha256
            ) is None:
                raise ValidationError(
                    f"{path}: invalid reset_rng_context_state_sha256"
                )

    _require_int(metadata, "seed", minimum=0)
    unnorm_key = metadata.get("unnorm_key")
    if unnorm_key is not None and (not isinstance(unnorm_key, str) or not unnorm_key):
        raise ValidationError(f"{path}: metadata unnorm_key must be null or a non-empty string")
    image_views = _require_string(metadata, "image_views")
    if image_views not in {"auto", "primary", "primary+wrist", "wrist+primary"}:
        raise ValidationError(f"{path}: invalid metadata image_views={image_views!r}")
    _require_int(metadata, "policy_image_size", minimum=0)

    eval_fingerprint = _require_string(metadata, "eval_fingerprint", allow_empty=True)
    if expected_eval_fingerprint is not None and eval_fingerprint != expected_eval_fingerprint:
        raise ValidationError(
            f"{path}: eval_fingerprint={eval_fingerprint!r}, expected={expected_eval_fingerprint!r}"
        )

    protocol_version = _require_int(metadata, "server_protocol_version", minimum=1)
    if protocol_version != 2:
        raise ValidationError(f"{path}: unsupported server protocol version {protocol_version}")
    _require_int(metadata, "server_action_chunk_size", minimum=1)
    action_dim = _require_int(metadata, "server_action_dim", minimum=1)
    if action_dim != 7:
        raise ValidationError(
            f"{path}: LIBERO server_action_dim must be exactly 7, got {action_dim}"
        )
    state_dim = _require_int(metadata, "server_state_dim", minimum=1)
    if state_dim != 8:
        raise ValidationError(
            f"{path}: LIBERO server_state_dim must be exactly 8, got {state_dim}"
        )
    expected_state_dim = metadata.get("server_expected_state_dim")
    if expected_state_dim is not None:
        if isinstance(expected_state_dim, bool) or not isinstance(expected_state_dim, int):
            raise ValidationError(
                f"{path}: server_expected_state_dim must be null or an integer"
            )
        if expected_state_dim != 8 or expected_state_dim != state_dim:
            raise ValidationError(
                f"{path}: expected state dim {expected_state_dim} is inconsistent with state dim {state_dim}"
            )
    return suite


def _validate_chunk_impl(
    path: Path,
    expected_checkpoint: str | None,
    expected_eval_fingerprint: str | None,
    expected_schedule: EvalSchedule | None,
) -> Chunk:
    checkpoint, expected_task, start, count = _parse_name(path)
    if expected_checkpoint is not None and checkpoint != expected_checkpoint:
        raise ValidationError(
            f"checkpoint filename mismatch in {path}: expected {expected_checkpoint}, got {checkpoint}"
        )

    text = path.read_text(errors="replace")
    metadata = _parse_metadata(text, path)
    suite = _validate_metadata(
        metadata,
        path,
        filename_checkpoint=checkpoint,
        filename_task=expected_task,
        filename_start=start,
        filename_count=count,
        expected_eval_fingerprint=expected_eval_fingerprint,
    )

    if text.count("EVAL_CHUNK_OK") != 1:
        raise ValidationError(f"{path}: expected exactly one EVAL_CHUNK_OK marker")
    total_episodes = [int(match.group(1)) for match in TOTAL_EPISODES_RE.finditer(text)]
    if total_episodes != [count]:
        raise ValidationError(
            f"{path}: Total episodes values {total_episodes}, expected exactly [{count}]"
        )

    total_rates: list[float] = []
    for match in TOTAL_RATE_RE.finditer(text):
        try:
            rate = float(match.group(1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"{path}: invalid total success rate: {exc}") from exc
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValidationError(f"{path}: total success rate outside [0, 1]: {rate!r}")
        total_rates.append(rate)
    if len(total_rates) != 1:
        raise ValidationError(
            f"{path}: expected exactly one finite final Total success rate, got {total_rates}"
        )

    current_task: int | None = None
    current_episode: int | None = None
    results: dict[int, bool] = {}
    for line in text.splitlines():
        if match := TASK_ID_RE.search(line):
            current_task = int(match.group(1))
        if match := EPISODE_RE.search(line):
            current_episode = int(match.group(1)) - 1
        if match := SUCCESS_RE.search(line):
            if current_task is None or current_episode is None:
                raise ValidationError(f"{path}: Success line lacks a preceding task/episode")
            if current_task != expected_task:
                raise ValidationError(
                    f"{path}: filename says task {expected_task}, log says task {current_task}"
                )
            if current_episode in results:
                raise ValidationError(f"{path}: duplicate result for episode {current_episode}")
            results[current_episode] = match.group(1) == "True"
            current_episode = None

    expected_episodes = set(range(start, start + count))
    if set(results) != expected_episodes:
        missing = sorted(expected_episodes - set(results))
        extra = sorted(set(results) - expected_episodes)
        raise ValidationError(
            f"{path}: episode coverage mismatch; missing={missing}, extra={extra}"
        )
    measured_rate = sum(results.values()) / count
    if abs(measured_rate - total_rates[0]) > 1e-6:
        raise ValidationError(
            f"{path}: success rate mismatch; results={measured_rate}, logged={total_rates[0]}"
        )
    chunk = Chunk(checkpoint, suite, expected_task, start, count, results, metadata, path)
    if expected_schedule is not None:
        _validate_chunk_schedule(chunk, expected_schedule)
    return chunk


def validate_chunk(
    path: Path,
    expected_checkpoint: str | None = None,
    expected_eval_fingerprint: str | None = None,
    expected_schedule: EvalSchedule | None = None,
) -> Chunk:
    try:
        return _validate_chunk_impl(
            path,
            expected_checkpoint,
            expected_eval_fingerprint,
            expected_schedule,
        )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"{path}: unexpected validation failure: {type(exc).__name__}: {exc}"
        ) from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _invalid_manifest(
    path: Path | None,
    *,
    checkpoint: str | None,
    eval_fingerprint: str | None,
    expected_trials: int | None,
    errors: list[str],
    accepted_logs: list[str] | None = None,
) -> None:
    if path is None:
        return
    _atomic_json(
        path,
        {
            "contract_version": CONTRACT_VERSION,
            "status": "invalid",
            "checkpoint": checkpoint,
            "eval_fingerprint": eval_fingerprint,
            "expected_trials_per_task": expected_trials,
            "accepted_logs": accepted_logs or [],
            "errors": errors,
        },
    )


def summarize(
    log_root: Path,
    checkpoint: str | None,
    expected_trials: int | None,
    require_complete: bool,
    manifest_out: Path | None,
    expected_eval_fingerprint: str | None,
    expected_schedule: EvalSchedule | None,
) -> int:
    coverage: dict[tuple[str, int, int], tuple[bool, Path]] = {}
    errors: list[str] = []
    accepted_logs: list[str] = []
    rejected_schedule_logs: list[str] = []
    observed_fingerprint: str | None = None

    try:
        for suite in SUITES:
            suite_dir = log_root / suite
            for path in sorted(suite_dir.glob("*_stage2_chunked_t*_r*_n*.log")):
                if path.name.endswith(AUXILIARY_CHUNK_LOG_SUFFIXES):
                    continue
                try:
                    filename_checkpoint, _, _, _ = _parse_name(path)
                except ValidationError as exc:
                    errors.append(str(exc))
                    continue
                if checkpoint is not None and filename_checkpoint != checkpoint:
                    continue
                try:
                    text = path.read_text(errors="replace")
                except OSError as exc:
                    errors.append(f"{path}: cannot read log: {exc}")
                    continue
                if "EVAL_CHUNK_OK" not in text:
                    continue
                try:
                    chunk = validate_chunk(
                        path,
                        checkpoint,
                        expected_eval_fingerprint,
                        expected_schedule,
                    )
                except ScheduleValidationError as exc:
                    rejected_schedule_logs.append(str(path))
                    print(f"REJECTED_SCHEDULE_LOG: {exc}", file=sys.stderr)
                    continue
                except ValidationError as exc:
                    errors.append(str(exc))
                    continue
                if observed_fingerprint is None:
                    observed_fingerprint = chunk.eval_fingerprint
                elif chunk.eval_fingerprint != observed_fingerprint:
                    errors.append(
                        f"mixed eval fingerprints: {observed_fingerprint!r} and "
                        f"{chunk.eval_fingerprint!r} in {path}"
                    )
                    continue
                accepted_logs.append(str(path))
                for episode, success in chunk.results.items():
                    key = (chunk.suite, chunk.task, episode)
                    if key in coverage:
                        previous = coverage[key][1]
                        errors.append(f"duplicate coverage for {key}: {previous} and {path}")
                    else:
                        coverage[key] = (success, path)
    except Exception as exc:
        errors.append(f"unexpected summary failure: {type(exc).__name__}: {exc}")

    manifest_fingerprint = (
        expected_eval_fingerprint
        if expected_eval_fingerprint is not None
        else observed_fingerprint
    )
    expected_keys: set[tuple[str, int, int]] | None = None
    missing: list[tuple[str, int, int]] = []
    if expected_trials is not None:
        if expected_trials <= 0:
            errors.append(f"expected_trials_per_task must be positive, got {expected_trials}")
        else:
            expected_keys = {
                (suite, task, episode)
                for suite in SUITES
                for task in range(10)
                for episode in range(expected_trials)
            }
            missing = sorted(expected_keys - set(coverage))
            extra = sorted(set(coverage) - expected_keys)
            if extra:
                errors.append(f"found {len(extra)} out-of-range results; first={extra[:10]}")
            if require_complete and missing:
                errors.append(f"missing {len(missing)} results; first={missing[:10]}")
    elif require_complete:
        errors.append("--require-complete needs --expected-trials-per-task")

    if errors:
        _invalid_manifest(
            manifest_out,
            checkpoint=checkpoint,
            eval_fingerprint=manifest_fingerprint,
            expected_trials=expected_trials,
            errors=errors,
            accepted_logs=accepted_logs,
        )
        for error in errors:
            print(f"VALIDATION_ERROR: {error}", file=sys.stderr)
        return 2

    task_rates: dict[str, dict[int, float]] = {}
    for suite in SUITES:
        suite_rates: dict[int, float] = {}
        for task in range(10):
            trials = {
                episode: value
                for (covered_suite, covered_task, episode), (value, _) in coverage.items()
                if covered_suite == suite and covered_task == task
            }
            if not trials:
                continue
            rate = sum(trials.values()) / len(trials)
            suite_rates[task] = rate
            print(f"{suite} task {task}: {len(trials)} trials, success={rate * 100:.2f}%")
        task_rates[suite] = suite_rates
        if suite_rates:
            print(
                f"{suite}: {len(suite_rates)} tasks, "
                f"task-mean={sum(suite_rates.values()) / len(suite_rates) * 100:.2f}%"
            )
        else:
            print(f"{suite}: 0 completed tasks")

    rates = [rate for suite_rates in task_rates.values() for rate in suite_rates.values()]
    complete = expected_keys is not None and set(coverage) == expected_keys
    if complete and len(rates) == 40:
        print(f"overall_40_task_mean: 40 tasks, {sum(rates) / len(rates) * 100:.2f}%")
    elif rates:
        print(f"partial_task_mean: {len(rates)} tasks, {sum(rates) / len(rates) * 100:.2f}%")
    else:
        print("partial_task_mean: no completed tasks")

    if manifest_out is not None:
        _atomic_json(
            manifest_out,
            {
                "contract_version": CONTRACT_VERSION,
                "status": "complete" if complete else "partial",
                "checkpoint": checkpoint,
                "eval_fingerprint": manifest_fingerprint,
                "expected_trials_per_task": expected_trials,
                "covered_results": len(coverage),
                "accepted_logs": accepted_logs,
                "rejected_schedule_logs": rejected_schedule_logs,
                "task_rates": {
                    suite: {str(task): rate for task, rate in rates_by_task.items()}
                    for suite, rates_by_task in task_rates.items()
                },
            },
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", nargs="?", type=Path)
    parser.add_argument("--checkpoint-base")
    parser.add_argument("--eval-fingerprint")
    parser.add_argument("--expected-trials-per-task", type=int)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--validate-log", type=Path)
    parser.add_argument("--eval-contract", type=Path)
    args = parser.parse_args()

    expected_schedule: EvalSchedule | None = None
    if args.eval_contract is not None:
        try:
            expected_schedule, contract_fingerprint = _load_eval_schedule(
                args.eval_contract,
                args.eval_fingerprint,
            )
        except ValidationError as exc:
            print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2)
        if args.eval_fingerprint is None:
            args.eval_fingerprint = contract_fingerprint

    if args.validate_log is not None:
        try:
            chunk = validate_chunk(
                args.validate_log,
                args.checkpoint_base,
                args.eval_fingerprint,
                expected_schedule,
            )
        except ValidationError as exc:
            error = str(exc)
            _invalid_manifest(
                args.manifest_out,
                checkpoint=args.checkpoint_base,
                eval_fingerprint=args.eval_fingerprint,
                expected_trials=args.expected_trials_per_task,
                errors=[error],
            )
            print(f"VALIDATION_ERROR: {error}", file=sys.stderr)
            raise SystemExit(2)
        print(
            f"EVAL_LOG_VALID checkpoint={chunk.checkpoint} suite={chunk.suite} "
            f"task={chunk.task} trials={chunk.start}..{chunk.start + chunk.count - 1} "
            f"eval_fingerprint={chunk.eval_fingerprint}"
        )
        return
    if args.log_root is None:
        parser.error("log_root is required unless --validate-log is used")
    raise SystemExit(
        summarize(
            args.log_root,
            args.checkpoint_base,
            args.expected_trials_per_task,
            args.require_complete,
            args.manifest_out,
            args.eval_fingerprint,
            expected_schedule,
        )
    )


if __name__ == "__main__":
    main()
