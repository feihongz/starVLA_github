from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "LIBERO"
    / "eval_files"
    / "run_stage2_eval_chunked_persistent.sh"
)
EVAL_SCRIPT = SCRIPT.parent / "eval_libero.py"
VALIDATOR = SCRIPT.parent / "validate_and_summarize_libero.py"


def _load_validator_namespace() -> dict:
    return runpy.run_path(str(VALIDATOR))


def _load_eval_helpers(*names: str):
    tree = ast.parse(EVAL_SCRIPT.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "RESET_RNG_CONTEXT_VERSION": 1,
        "hashlib": hashlib,
        "json": json,
        "logging": logging,
        "math": math,
        "np": np,
        "os": os,
        "pathlib": pathlib,
        "re": re,
    }
    exec(compile(module, str(EVAL_SCRIPT), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _bash_function(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\) \{{.*?^\}}",
        source,
    )
    assert match is not None
    return match.group(0)


def _fatal_failure_status(tmp_path: Path, log_text: str) -> int:
    source = SCRIPT.read_text()
    function = _bash_function(source, "fatal_failure")
    log_path = tmp_path / "chunk.log"
    log_path.write_text(log_text)
    command = f"set -u\n{function}\nfatal_failure \"$1\""
    return subprocess.run(
        ["bash", "-c", command, "bash", str(log_path)],
        check=False,
    ).returncode


def test_normal_run_metadata_after_sigabrt_remains_retryable(tmp_path: Path) -> None:
    log_text = (
        'EVAL_RUN_META_JSON {"contract_version":2,"suite":"libero_10"}\n'
        "Starting episode 27...\n"
    )

    assert _fatal_failure_status(tmp_path, log_text) != 0


@pytest.mark.parametrize(
    "error_text",
    [
        "EVAL_CONTRACT_ERROR checkpoint mismatch",
        "METADATA_HANDSHAKE_ERROR",
        "LIBERO action contract mismatch: expected 7, got 8",
        "CUDA out of memory",
    ],
)
def test_explicit_deterministic_errors_remain_fatal(
    tmp_path: Path,
    error_text: str,
) -> None:
    assert _fatal_failure_status(tmp_path, error_text + "\n") == 0


def test_client_and_validation_statuses_are_preserved_in_retry_log() -> None:
    source = SCRIPT.read_text()

    assert "client_status=$?" in source
    assert "validation_status=$?" in source
    assert "client_status=${client_status}" in source
    assert "validation_status=${validation_status}" in source

def _coverage_status(
    tmp_path: Path,
    *,
    candidate_text: str,
    requested_start: int,
    requested_count: int,
    candidate_start: int = 25,
    candidate_count: int = 5,
    expected_reset_context: str = "",
    expected_context_span: str = "5",
) -> subprocess.CompletedProcess[str]:
    source = SCRIPT.read_text()
    function = _bash_function(source, "chunk_covered_by_existing_log")
    suite_dir = tmp_path / "logs" / "libero_10"
    suite_dir.mkdir(parents=True)
    candidate = (
        suite_dir
        / f"ckpt_stage2_chunked_t3_r{candidate_start}_n{candidate_count}.log"
    )
    candidate.write_text(candidate_text)
    command = (
        "set -u\n"
        'LOG_ROOT="$1"\n'
        "CKPT_BASE=ckpt\n"
        'validate_chunk() { grep -qx VALID "$1" || return 1; '
        '[[ -z "${2:-}" ]] || grep -Eq "\\\"reset_context_start\\\":${2}([,}])" "$1"; }\n'
        f"{function}\n"
        "chunk_covered_by_existing_log libero_10 3 \"$2\" \"$3\" \"$4\" \"$5\""
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(tmp_path / "logs"),
            str(requested_start),
            str(requested_count),
            expected_reset_context,
            expected_context_span,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def test_valid_wider_chunk_strictly_covers_requested_single_trial(
    tmp_path: Path,
) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=27,
        requested_count=1,
        expected_reset_context="25",
    )

    assert result.returncode == 0
    assert result.stdout.rstrip().endswith("_t3_r25_n5.log")


def test_invalid_wider_chunk_does_not_cover_requested_single_trial(
    tmp_path: Path,
) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="INVALID\n",
        requested_start=27,
        requested_count=1,
        expected_reset_context="25",
    )

    assert result.returncode != 0
    assert result.stdout == ""


def test_unprimed_nonboundary_single_trial_is_not_reused(tmp_path: Path) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=26,
        requested_count=1,
        candidate_start=26,
        candidate_count=1,
        expected_reset_context="25",
    )

    assert result.returncode != 0


def test_primed_nonboundary_single_trial_is_reused(tmp_path: Path) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text='EVAL_RUN_META_JSON {"reset_context_start":25}\nVALID\n',
        requested_start=26,
        requested_count=1,
        candidate_start=26,
        candidate_count=1,
        expected_reset_context="25",
    )

    assert result.returncode == 0


def test_canonical_wider_chunk_is_reused_within_reset_boundary(tmp_path: Path) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=29,
        requested_count=1,
        candidate_start=25,
        candidate_count=5,
        expected_reset_context="25",
        expected_context_span="5",
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("candidate_start", "candidate_count"),
    [(26, 2), (20, 10)],
)
def test_unprimed_noncanonical_wider_chunk_is_not_reused(
    tmp_path: Path,
    candidate_start: int,
    candidate_count: int,
) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=26,
        requested_count=1,
        candidate_start=candidate_start,
        candidate_count=candidate_count,
        expected_reset_context="25",
        expected_context_span="5",
    )

    assert result.returncode != 0


def test_wider_chunk_cannot_cross_next_reset_boundary(tmp_path: Path) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=26,
        requested_count=1,
        candidate_start=25,
        candidate_count=10,
        expected_reset_context="25",
        expected_context_span="5",
    )

    assert result.returncode != 0


def test_normal_schedule_does_not_reuse_cross_boundary_chunk(tmp_path: Path) -> None:
    result = _coverage_status(
        tmp_path,
        candidate_text="VALID\n",
        requested_start=5,
        requested_count=5,
        candidate_start=0,
        candidate_count=10,
        expected_reset_context="5",
        expected_context_span="5",
    )

    assert result.returncode != 0


def test_reset_context_environment_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    (resolve,) = _load_eval_helpers("_resolve_reset_context_start")

    monkeypatch.delenv("RESET_CONTEXT_START", raising=False)
    assert resolve(25) is None
    monkeypatch.setenv("RESET_CONTEXT_START", "")
    assert resolve(25) is None
    monkeypatch.setenv("RESET_CONTEXT_START", "25")
    assert resolve(26) == 25
    for invalid in ("-1", "abc", "27"):
        monkeypatch.setenv("RESET_CONTEXT_START", invalid)
        with pytest.raises(ValueError):
            resolve(26)


def test_reset_context_priming_replays_only_prior_resets() -> None:
    (prime,) = _load_eval_helpers("_prime_reset_context")
    events: list[tuple[str, object | None]] = []

    class FakeEnv:
        def reset(self) -> None:
            events.append(("reset", None))

        def set_init_state(self, state: object) -> None:
            events.append(("set", state))

    prime(
        FakeEnv(),
        list(range(50)),
        reset_context_start=25,
        trial_start=27,
        task_id=3,
    )

    assert events == [("reset", None), ("set", 25), ("reset", None), ("set", 26)]


@pytest.mark.parametrize(
    ("trial", "expected"),
    [(25, "25"), (26, "25"), (29, "25"), (30, "30")],
)
def test_canonical_reset_context_respects_original_chunk_boundaries(
    trial: int,
    expected: str,
) -> None:
    function = _bash_function(SCRIPT.read_text(), "canonical_reset_context_start")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"{function}\ncanonical_reset_context_start \"$1\" 0 5",
            "bash",
            str(trial),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("client_status", "chunk", "should_fallback"),
    [(134, 5, True), (139, 5, True), (134, 1, False), (1, 5, False)],
)
def test_adaptive_fallback_condition(
    client_status: int,
    chunk: int,
    should_fallback: bool,
) -> None:
    source = SCRIPT.read_text()
    match = re.search(
        r"if \(\( (\(client_status == 134 \|\| client_status == 139\) && chunk > 1) \)\); then",
        source,
    )
    assert match is not None
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"client_status=$1; chunk=$2; (( {match.group(1)} ))",
            "bash",
            str(client_status),
            str(chunk),
        ],
        check=False,
    )

    assert (result.returncode == 0) is should_fallback


def test_adaptive_fallback_is_persistent_and_true_fatal_has_priority() -> None:
    source = SCRIPT.read_text()
    fatal_index = source.index('if fatal_failure "${log_path}"')
    fallback_index = source.index(
        "if (( (client_status == 134 || client_status == 139) && chunk > 1 ))"
    )

    assert fatal_index < fallback_index
    assert ".single_trial_mode" in source
    assert 'FALLBACK_MODE_ROOT="${OUTPUT_BASE}/fallback_modes"' in source
    assert 'single_trial_marker="${FALLBACK_MODE_ROOT}/' in source
    assert 'single_trial_marker="${WORKER_ROOT}/' not in source
    assert 'load_single_trial_marker_partition "${single_trial_marker}"' in source
    assert "trial_start >= marker_fallback_start" in source
    assert 'reset_context_start="$(canonical_reset_context_start' in source
    assert 'RESET_CONTEXT_START="${reset_context_start}"' in source
    assert 'RESET_CONTEXT_CHUNK_SIZE="${reset_context_span}"' in source
    assert 'RESET_RNG_CONTEXT_PATH="${reset_rng_context_path}"' in source
    assert "materialize_reset_rng_context" in source
    assert "PYTHONFAULTHANDLER=1" in source
    assert 'validate_chunk "${log_path}" "${reset_context_start}"' in source
    assert "continue 2" in source


def test_validator_enforces_reset_context_metadata_contract() -> None:
    source = VALIDATOR.read_text()
    assert 'metadata.get("reset_context_start")' in source
    assert "reset-context recovery is allowed only for a single-trial chunk" in source
    assert "reset_context_count" in source
    assert "reset_rng_context_ordinal" in source
    assert "reset context must use exactly one supported recovery strategy" in source


def test_reset_rng_context_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    (
        state_payload,
        state_from_payload,
        payload_sha256,
        load_context,
        write_context,
    ) = _load_eval_helpers(
        "_numpy_rng_state_payload",
        "_numpy_rng_state_from_payload",
        "_rng_payload_sha256",
        "_load_reset_rng_context",
        "_write_reset_rng_context",
    )
    np.random.seed(7)
    state = np.random.get_state()
    path = tmp_path / "ordinal1.json"
    write_context(
        path,
        state=state,
        suite="libero_10",
        task_id=3,
        seed=7,
        nominal_chunk=5,
        ordinal=1,
        eval_fingerprint="a" * 64,
    )

    restored, digest = load_context(
        path,
        suite="libero_10",
        task_id=3,
        seed=7,
        nominal_chunk=5,
        ordinal=1,
        eval_fingerprint="a" * 64,
    )
    assert restored[0] == state[0]
    np.testing.assert_array_equal(restored[1], state[1])
    assert restored[2:] == state[2:]
    assert digest == payload_sha256(state_payload(state))
    np.testing.assert_array_equal(state_from_payload(state_payload(state))[1], state[1])

    document = json.loads(path.read_text())
    document["numpy_state"]["keys"][0] ^= 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_context(
            path,
            suite="libero_10",
            task_id=3,
            seed=7,
            nominal_chunk=5,
            ordinal=1,
            eval_fingerprint="a" * 64,
        )


def test_contract_schedule_accepts_canonical_and_primed_chunks() -> None:
    namespace = _load_validator_namespace()
    Chunk = namespace["Chunk"]
    EvalSchedule = namespace["EvalSchedule"]
    validate_schedule = namespace["_validate_chunk_schedule"]
    schedule = EvalSchedule(
        trials_per_task=50,
        chunk_sizes={
            "libero_spatial": 5,
            "libero_object": 1,
            "libero_goal": 1,
            "libero_10": 5,
        },
    )

    validate_schedule(
        Chunk("ckpt", "libero_10", 3, 25, 5, {}, {}, Path("r25_n5.log")),
        schedule,
    )
    validate_schedule(
        Chunk(
            "ckpt",
            "libero_10",
            3,
            27,
            1,
            {},
            {"reset_context_start": 25},
            Path("r27_n1.log"),
        ),
        schedule,
    )
    validate_schedule(
        Chunk("ckpt", "libero_object", 0, 17, 1, {}, {}, Path("object.log")),
        schedule,
    )


@pytest.mark.parametrize(
    ("start", "count", "reset_context"),
    [(25, 10, None), (30, 1, 25), (26, 1, None)],
)
def test_contract_schedule_rejects_cross_boundary_or_unprimed_chunks(
    start: int,
    count: int,
    reset_context: int | None,
) -> None:
    namespace = _load_validator_namespace()
    Chunk = namespace["Chunk"]
    EvalSchedule = namespace["EvalSchedule"]
    ScheduleValidationError = namespace["ScheduleValidationError"]
    validate_schedule = namespace["_validate_chunk_schedule"]
    schedule = EvalSchedule(
        trials_per_task=50,
        chunk_sizes={suite: (1 if suite in {"libero_object", "libero_goal"} else 5)
                     for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")},
    )
    metadata = {} if reset_context is None else {"reset_context_start": reset_context}
    chunk = Chunk("ckpt", "libero_10", 3, start, count, {}, metadata, Path("bad.log"))

    with pytest.raises(ScheduleValidationError):
        validate_schedule(chunk, schedule)


def test_eval_contract_digest_and_schedule_are_verified(tmp_path: Path) -> None:
    namespace = _load_validator_namespace()
    ValidationError = namespace["ValidationError"]
    load_schedule = namespace["_load_eval_schedule"]
    contract = tmp_path / "eval_contract.txt"
    contract.write_text(
        "contract_version=2\n"
        "trials_per_task=50\n"
        "spatial_chunk_trials=5\n"
        "object_chunk_trials=1\n"
        "goal_chunk_trials=1\n"
        "libero10_chunk_trials=5\n"
    )

    schedule, fingerprint = load_schedule(contract, None)
    assert schedule.trials_per_task == 50
    assert schedule.chunk_sizes["libero_10"] == 5
    assert load_schedule(contract, fingerprint)[0] == schedule
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        load_schedule(contract, "0" * 64)


def test_contract_schedule_validation_is_wired_into_worker_and_manager() -> None:
    worker = SCRIPT.read_text()
    manager = (SCRIPT.parent / "run_jike_stage2_all_checkpoints_8gpu_v2.sh").read_text()
    validator = VALIDATOR.read_text()

    assert '--eval-contract "${EVAL_CONTRACT_PATH}"' in worker
    assert '--eval-contract "${checkpoint_root}/eval_contract.txt"' in manager
    assert "except ScheduleValidationError as exc:" in validator
    assert '"rejected_schedule_logs": rejected_schedule_logs' in validator


def test_v2_contract_allows_only_audited_worker_resume_patch() -> None:
    manager = (
        SCRIPT.parent / "run_jike_stage2_all_checkpoints_8gpu_v2.sh"
    ).read_text()

    assert (
        "run_stage2_eval_chunked_persistent.sh="
        "7d00de96b68f7fa7a9b08ca93ccaf1f1709dafb56f2e9dd710989baf7c9ced79"
        in manager
    )
    assert (
        "eval_libero.py="
        "9265d85bec017d01fda042942aa50b03b35af84959353722d637a46862c2ac16"
        in manager
    )
    assert "compatibility=eval-orchestration-resilience-v4" in manager
    assert "contract_resume_compatibility_v4.txt" in manager
