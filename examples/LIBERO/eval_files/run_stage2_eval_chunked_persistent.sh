#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <stage2_checkpoint.pt> [gpu_id] [port]" >&2
  exit 2
fi

CKPT="$1"
GPU_ID="${2:-0}"
PORT="${3:-18620}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LOCAL_RUNNER="${SCRIPT_DIR}/run_local_eval_once.sh"
VALIDATOR="${SCRIPT_DIR}/validate_and_summarize_libero.py"

if [[ -z "${LIBERO_HOME:-}" ]]; then
  if [[ -d "/root/feihong/LIBERO/libero" ]]; then
    LIBERO_HOME="/root/feihong/LIBERO"
  else
    LIBERO_HOME="${STARVLA_DIR}/third_party/LIBERO"
  fi
fi
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_HOME}/.venv/bin/python}"

if [[ -n "${TASK_SUITES_OVERRIDE:-}" ]]; then
  read -r -a TASK_SUITES <<< "${TASK_SUITES_OVERRIDE}"
else
  TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-5}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS:-2}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-1200}"
POLICY_REQUEST_TIMEOUT_SECONDS="${POLICY_REQUEST_TIMEOUT_SECONDS:-600}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS:-0}"
MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK:--1}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_stage2}"
EVAL_FINGERPRINT="${EVAL_FINGERPRINT:-}"
SEED="${SEED:-7}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TASK_START_VALUE="${TASK_START:-0}"
TASK_COUNT_VALUE="${TASK_COUNT:--1}"
TRIAL_START_VALUE="${TRIAL_START:-0}"
WORKER_ID="${WORKER_ID:-gpu${GPU_ID}_port${PORT}}"

die() {
  echo "[stage2_eval] ERROR: $*" >&2
  exit 1
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'"
}
load_single_trial_marker_partition() {
  local marker_path="$1" expected_suite="$2" expected_task="$3"
  local key value
  local schedule_start="" nominal_chunk="" fallback_start=""
  local legacy_schedule="" legacy_chunk="" legacy_fallback=""
  local marker_suite="" marker_task=""

  while IFS='=' read -r key value || [[ -n "${key}${value}" ]]; do
    case "${key}" in
      schedule_trial_start) schedule_start="${value}" ;;
      nominal_chunk_trials) nominal_chunk="${value}" ;;
      fallback_start) fallback_start="${value}" ;;
      canonical_trial_start) legacy_schedule="${value}" ;;
      canonical_chunk_trials) legacy_chunk="${value}" ;;
      failed_trial_start) legacy_fallback="${value}" ;;
      suite) marker_suite="${value}" ;;
      task) marker_task="${value}" ;;
    esac
  done <"${marker_path}"

  [[ -z "${schedule_start}" || -z "${legacy_schedule}" || "${schedule_start}" == "${legacy_schedule}" ]] || return 1
  [[ -z "${nominal_chunk}" || -z "${legacy_chunk}" || "${nominal_chunk}" == "${legacy_chunk}" ]] || return 1
  [[ -z "${fallback_start}" || -z "${legacy_fallback}" || "${fallback_start}" == "${legacy_fallback}" ]] || return 1
  marker_partition_start="${schedule_start:-${legacy_schedule:-${TRIAL_START_VALUE}}}"
  marker_partition_chunk="${nominal_chunk:-${legacy_chunk:-${CHUNK_TRIALS}}}"
  marker_fallback_start="${fallback_start:-${legacy_fallback}}"

  [[ "${marker_partition_start}" =~ ^[0-9]+$ ]] || return 1
  [[ "${marker_partition_chunk}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${marker_fallback_start}" =~ ^[0-9]+$ ]] || return 1
  [[ -z "${marker_suite}" || "${marker_suite}" == "${expected_suite}" ]] || return 1
  [[ -z "${marker_task}" || "${marker_task}" == "${expected_task}" ]] || return 1
  (( marker_fallback_start >= marker_partition_start )) || return 1
  (( (marker_fallback_start - marker_partition_start) % marker_partition_chunk == 0 )) || return 1
  (( marker_fallback_start < TRIALS_PER_TASK )) || return 1
}

canonical_reset_context_start() {
  local target="$1" origin="$2" chunk_size="$3"
  [[ "${target}" =~ ^[0-9]+$ && "${origin}" =~ ^[0-9]+$ && "${chunk_size}" =~ ^[1-9][0-9]*$ ]] || return 1
  (( target >= origin )) || return 1
  printf '%d\n' "$((origin + ((target - origin) / chunk_size) * chunk_size))"
}


NUMERIC_PAIRS=(
  "GPU_ID:${GPU_ID}"
  "PORT:${PORT}"
  "TRIALS_PER_TASK:${TRIALS_PER_TASK}"
  "CHUNK_TRIALS:${CHUNK_TRIALS}"
  "MAX_RETRIES:${MAX_RETRIES}"
  "MAX_SERVER_RESTARTS:${MAX_SERVER_RESTARTS}"
  "CHUNK_TIMEOUT_SECONDS:${CHUNK_TIMEOUT_SECONDS}"
  "SERVER_STARTUP_TIMEOUT_SECONDS:${SERVER_STARTUP_TIMEOUT_SECONDS}"
  "SEED:${SEED}"
  "TASK_START:${TASK_START_VALUE}"
  "TRIAL_START:${TRIAL_START_VALUE}"
)
for pair in "${NUMERIC_PAIRS[@]}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
[[ "${TASK_COUNT_VALUE}" == "-1" || "${TASK_COUNT_VALUE}" =~ ^[1-9][0-9]*$ ]] || die "TASK_COUNT must be -1 or positive, got ${TASK_COUNT_VALUE}"
(( PORT > 0 && PORT <= 65535 )) || die "invalid port: ${PORT}"
(( TRIALS_PER_TASK > 0 && CHUNK_TRIALS > 0 && MAX_RETRIES > 0 && MAX_SERVER_RESTARTS >= 0 )) || die "trial/chunk/retry values must be valid"
(( CHUNK_TIMEOUT_SECONDS > 0 && SERVER_STARTUP_TIMEOUT_SECONDS > 0 )) || die "timeouts must be positive"
(( TASK_START_VALUE < 10 )) || die "TASK_START must be below 10"
(( TRIAL_START_VALUE < TRIALS_PER_TASK )) || die "TRIAL_START must be below TRIALS_PER_TASK"
if [[ "${TASK_COUNT_VALUE}" != "-1" ]]; then
  (( TASK_START_VALUE + TASK_COUNT_VALUE <= 10 )) || die "TASK_START + TASK_COUNT must not exceed 10"
fi
[[ "${SKIP_COMPLETED}" == "0" || "${SKIP_COMPLETED}" == "1" ]] || die "SKIP_COMPLETED must be 0 or 1"
[[ -z "${EVAL_FINGERPRINT}" || "${EVAL_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ]] || die "EVAL_FINGERPRINT must be empty or a lowercase SHA-256"
[[ -n "${EVAL_OUTPUT_ROOT}" && "${EVAL_OUTPUT_ROOT}" != /* ]] || die "EVAL_OUTPUT_ROOT must be a non-empty relative path"
[[ "${WORKER_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "WORKER_ID contains unsafe characters: ${WORKER_ID}"
[[ "${WORKER_ID}" != "." && "${WORKER_ID}" != ".." ]] || die "WORKER_ID must not be ${WORKER_ID}"
[[ -x "${LOCAL_RUNNER}" ]] || die "missing executable local runner: ${LOCAL_RUNNER}"
[[ -f "${VALIDATOR}" ]] || die "missing strict validator: ${VALIDATOR}"
[[ -x "${LIBERO_PYTHON}" ]] || die "LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}"
for command_name in realpath setsid timeout; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done
[[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
CKPT="$(readlink -f "${CKPT}")"
checkpoint_parent="$(dirname "${CKPT}")"
if [[ -n "${MODEL_ROOT:-}" ]]; then
  MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd)"
elif [[ "$(basename "${checkpoint_parent}")" == "checkpoints" ]]; then
  MODEL_ROOT="$(cd "${checkpoint_parent}/.." && pwd)"
else
  die "set MODEL_ROOT when checkpoint is not under checkpoints/"
fi
[[ "${CKPT}" == "${MODEL_ROOT}/checkpoints/"* ]] || die "checkpoint is outside MODEL_ROOT/checkpoints"
OUTPUT_BASE="$(realpath -m "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}")"
[[ "${OUTPUT_BASE}" == "${MODEL_ROOT}/"* ]] || die "EVAL_OUTPUT_ROOT escapes MODEL_ROOT: ${EVAL_OUTPUT_ROOT}"
EVAL_CONTRACT_PATH="${EVAL_CONTRACT_PATH:-${OUTPUT_BASE}/eval_contract.txt}"
if [[ -n "${EVAL_FINGERPRINT}" && ! -f "${EVAL_CONTRACT_PATH}" ]]; then
  die "eval fingerprint requires contract file: ${EVAL_CONTRACT_PATH}"
fi

for suite in "${TASK_SUITES[@]}"; do
  case "${suite}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) die "unsupported task suite: ${suite}" ;;
  esac
done
(( ${#TASK_SUITES[@]} > 0 )) || die "no task suites selected"

export STARVLA_DIR LIBERO_HOME LIBERO_CONFIG_PATH LIBERO_PYTHON MODEL_ROOT
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export POLICY_REQUEST_TIMEOUT_SECONDS
unset DEBUG
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
unset TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_ERROR_FILE

CKPT_BASE="$(basename "${CKPT}" .pt)"
LOG_ROOT="${OUTPUT_BASE}/logs"
WORKER_ROOT="${OUTPUT_BASE}/workers/${WORKER_ID}"
FALLBACK_MODE_ROOT="${OUTPUT_BASE}/fallback_modes"
RESET_RNG_CONTEXT_ROOT="${OUTPUT_BASE}/reset_rng_contexts/${EVAL_FINGERPRINT:-unfingerprinted}"
SERVER_LOG_PATH="${WORKER_ROOT}/${CKPT_BASE}.server.log"
PROBE_LOG_PATH="${WORKER_ROOT}/${CKPT_BASE}.probe.log"
PROGRESS_PATH="${WORKER_ROOT}/${CKPT_BASE}.progress.txt"
PARTIAL_SUMMARY_PATH="${WORKER_ROOT}/${CKPT_BASE}.partial_summary.txt"
mkdir -p "${LOG_ROOT}" "${WORKER_ROOT}" "${FALLBACK_MODE_ROOT}" "${RESET_RNG_CONTEXT_ROOT}"

SERVER_PID=""
ACTIVE_CLIENT_PID=""
SERVER_START_COUNT=0

process_group_alive() {
  kill -0 -- "-$1" >/dev/null 2>&1
}

stop_active_client() {
  local pid="${ACTIVE_CLIENT_PID}"
  [[ -n "${pid}" ]] || return 0
  if process_group_alive "${pid}"; then
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 40); do
      process_group_alive "${pid}" || break
      sleep 0.25
    done
    process_group_alive "${pid}" && kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
  fi
  wait "${pid}" >/dev/null 2>&1 || true
  ACTIVE_CLIENT_PID=""
}

run_client_command() {
  local status
  setsid timeout --kill-after=60s "${CHUNK_TIMEOUT_SECONDS}s" "$@" &
  ACTIVE_CLIENT_PID=$!
  if wait "${ACTIVE_CLIENT_PID}"; then
    status=0
  else
    status=$?
  fi
  # A failed pipeline can leave descendants after its leader exits. Always
  # tear down the dedicated process group before retrying or returning.
  stop_active_client
  return "${status}"
}

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 40); do
      kill -0 "${SERVER_PID}" >/dev/null 2>&1 || break
      sleep 0.25
    done
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
  SERVER_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  stop_active_client
  stop_server
  exit "${status}"
}

handle_signal() {
  local status="$1"
  trap - EXIT INT TERM HUP
  stop_active_client
  stop_server
  exit "${status}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP

server_probe() {
  env POLICY_CONNECT_TIMEOUT_SECONDS=5 POLICY_HANDSHAKE_TIMEOUT_SECONDS=5 "${LIBERO_PYTHON}" "${SCRIPT_DIR}/check_policy_server.py" --host 127.0.0.1 --port "${PORT}" --expected-ckpt "${CKPT}" --unnorm-key "${UNNORM_KEY}" --timeout 5
}

wait_for_server() {
  local start_seconds="${SECONDS}"
  while (( SECONDS - start_seconds < SERVER_STARTUP_TIMEOUT_SECONDS )); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "[stage2_eval] policy server exited during startup" >&2
      tail -120 "${SERVER_LOG_PATH}" >&2 || true
      return 1
    fi
    if server_probe >"${PROBE_LOG_PATH}" 2>&1; then
      cat "${PROBE_LOG_PATH}"
      return 0
    fi
    sleep 2
  done
  echo "[stage2_eval] policy server readiness timed out after ${SERVER_STARTUP_TIMEOUT_SECONDS}s" >&2
  tail -120 "${SERVER_LOG_PATH}" >&2 || true
  tail -40 "${PROBE_LOG_PATH}" >&2 || true
  return 1
}

start_server() {
  while true; do
    stop_server
    if (( SERVER_START_COUNT > MAX_SERVER_RESTARTS )); then
      die "policy server exceeded ${MAX_SERVER_RESTARTS} allowed restarts"
    fi
    SERVER_START_COUNT=$((SERVER_START_COUNT + 1))
    : >"${SERVER_LOG_PATH}"
    echo "[stage2_eval] loading policy server start=${SERVER_START_COUNT} gpu=${GPU_ID} port=${PORT} ckpt=${CKPT_BASE}"
    env POLICY_SERVER_MODE=server-only SKIP_EVAL_PREFLIGHT=1 EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" USE_BF16="${USE_BF16:-1}" "${LOCAL_RUNNER}" "${CKPT}" "${TASK_SUITES[0]}" "${GPU_ID}" "${PORT}" >"${SERVER_LOG_PATH}" 2>&1 &
    SERVER_PID=$!
    if wait_for_server; then
      return 0
    fi
    stop_server
    if (( SERVER_START_COUNT > MAX_SERVER_RESTARTS )); then
      die "policy server failed readiness/identity validation after ${SERVER_START_COUNT} starts"
    fi
    echo "[stage2_eval] retrying policy server startup within restart budget" >&2
  done
}

validate_chunk() {
  local log_path="$1"
  local expected_reset_context="${2:-}"
  local -a validator_args=(--validate-log "${log_path}" --checkpoint-base "${CKPT_BASE}")
  if [[ -n "${EVAL_FINGERPRINT}" ]]; then
    validator_args+=(--eval-fingerprint "${EVAL_FINGERPRINT}" --eval-contract "${EVAL_CONTRACT_PATH}")
  fi
  if ! "${LIBERO_PYTHON}" "${VALIDATOR}" "${validator_args[@]}" >/dev/null 2>&1; then
    return 1
  fi
  if [[ -n "${expected_reset_context}" ]] \
    && ! grep -Eq "\"reset_context_start\":${expected_reset_context}([,}])" "${log_path}"; then
    return 1
  fi
}

chunk_covered_by_existing_log() {
  local suite="$1" task_id="$2" requested_start="$3" requested_count="$4"
  local expected_reset_context="${5:-}"
  local expected_context_span="${6:-}"
  local suite_dir="${LOG_ROOT}/${suite}"
  local prefix="${CKPT_BASE}_stage2_chunked_t${task_id}_r"
  local requested_end=$((requested_start + requested_count))
  local candidate candidate_base suffix candidate_start candidate_count candidate_end candidate_reset_context
  local -a candidates=()

  [[ -d "${suite_dir}" ]] || return 1
  shopt -s nullglob
  candidates=("${suite_dir}/${prefix}"*"_n"*.log)
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    candidate_base="$(basename "${candidate}")"
    [[ "${candidate_base}" == "${prefix}"* ]] || continue
    suffix="${candidate_base:${#prefix}}"
    [[ "${suffix}" =~ ^([0-9]+)_n([0-9]+)[.]log$ ]] || continue
    candidate_start="${BASH_REMATCH[1]}"
    candidate_count="${BASH_REMATCH[2]}"
    candidate_end=$((candidate_start + candidate_count))
    candidate_reset_context=""
    if [[ -n "${expected_reset_context}" ]]; then
      if (( candidate_start == expected_reset_context )); then
        # A canonical wider chunk may cover isolated fallback trials, but it
        # must not cross the next process/reset boundary in the schedule.
        (( candidate_end <= expected_reset_context + expected_context_span )) || continue
      else
        # A non-canonical candidate is reusable only when its validated
        # metadata proves that it replayed the exact reset context.
        candidate_reset_context="${expected_reset_context}"
      fi
    fi
    if (( candidate_start <= requested_start && candidate_end >= requested_end )) \
      && validate_chunk "${candidate}" "${candidate_reset_context}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

fatal_failure() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  # EVAL_RUN_META_JSON is normal startup output, not a deterministic failure.
  # Match only explicit contract/metadata error markers so signal exits such as
  # SIGABRT (status 134) remain retryable while the policy server is healthy.
  grep -Eqi 'ModuleNotFoundError|ImportError|No module named|State dim mismatch|checkpoint mismatch|fingerprint mismatch|wrong policy server|Unexpected policy-server identity|invalid .*metadata|metadata mismatch|LIBERO (action|state) contract mismatch|Action (horizon|dim) mismatch|Invalid action batch|non-finite (action|actions|proprio)|CUDA out of memory|No space left on device|missing (LIBERO config|checkpoint companion)|expected scalar type|METADATA_HANDSHAKE_ERROR|EVAL_(CONTRACT|RUN_META)_(ERROR|INVALID|MISMATCH)' "${log_path}"
}

ensure_reset_rng_context() {
  local suite="$1" task_id="$2" context_start="$3" target="$4" nominal_chunk="$5"
  local ordinal current input step attempt status
  ordinal=$((target - context_start))
  (( ordinal >= 0 && ordinal < nominal_chunk )) || return 1

  for ((step = 0; step <= ordinal; step++)); do
    current="${RESET_RNG_CONTEXT_ROOT}/${suite}_task${task_id}_seed${SEED}_chunk${nominal_chunk}_ordinal${step}.json"
    input=""
    if (( step > 0 )); then
      input="${RESET_RNG_CONTEXT_ROOT}/${suite}_task${task_id}_seed${SEED}_chunk${nominal_chunk}_ordinal$((step - 1)).json"
    fi
    # Context files are written atomically into a fingerprint-scoped cache.
    # The target is validated again by eval_libero before any reset, while a
    # missing successor strictly validates its predecessor during generation.
    # Avoid importing LIBERO solely to re-check every cached prefix ordinal.
    if [[ -f "${current}" ]]; then
      continue
    fi

    attempt=1
    status=1
    while (( attempt <= MAX_RETRIES )); do
      echo "[stage2_eval] prepare reset RNG context suite=${suite} task=${task_id} ordinal=${step}/${ordinal} attempt=${attempt}/${MAX_RETRIES}"
      if timeout --kill-after=10s "${SERVER_STARTUP_TIMEOUT_SECONDS}s" env \
        PYTHONFAULTHANDLER=1 \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        MUJOCO_GL="${MUJOCO_GL:-egl}" \
        PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
        MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}" \
        "${LIBERO_PYTHON}" -c \
        'import sys; from examples.LIBERO.eval_files.eval_libero import materialize_reset_rng_context; materialize_reset_rng_context(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])' \
        "${current}" "${input}" "${suite}" "${task_id}" "${SEED}" "${nominal_chunk}" "${step}" "${EVAL_FINGERPRINT}"; then
        status=0
        break
      else
        status=$?
      fi
      echo "[stage2_eval] reset RNG context attempt failed suite=${suite} task=${task_id} ordinal=${step} status=${status}" >&2
      attempt=$((attempt + 1))
    done
    (( status == 0 )) || die "cannot prepare reset RNG context suite=${suite} task=${task_id} ordinal=${step}"
  done
  reset_rng_context_path="${current}"
}

write_progress() {
  local -a validator_args=("${LOG_ROOT}" --checkpoint-base "${CKPT_BASE}" --expected-trials-per-task "${TRIALS_PER_TASK}")
  if [[ -n "${EVAL_FINGERPRINT}" ]]; then
    validator_args+=(--eval-fingerprint "${EVAL_FINGERPRINT}" --eval-contract "${EVAL_CONTRACT_PATH}")
  fi
  {
    echo "========== stage2 progress worker=${WORKER_ID} =========="
    date --iso-8601=seconds
    "${LIBERO_PYTHON}" "${VALIDATOR}" "${validator_args[@]}"
  } | tee "${PROGRESS_PATH}"
}

if [[ "${SKIP_WORKER_PREFLIGHT:-0}" != "1" ]]; then
  echo "[stage2_eval] preflight worker=${WORKER_ID} ckpt=${CKPT_BASE}"
  env POLICY_SERVER_MODE=preflight SKIP_EVAL_PREFLIGHT=0 EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" "${LOCAL_RUNNER}" "${CKPT}" "${TASK_SUITES[0]}" "${GPU_ID}" "${PORT}"
fi

for suite in "${TASK_SUITES[@]}"; do
  for task_id in {0..9}; do
    if (( task_id < TASK_START_VALUE )); then
      continue
    fi
    if [[ "${TASK_COUNT_VALUE}" != "-1" ]] && (( task_id >= TASK_START_VALUE + TASK_COUNT_VALUE )); then
      continue
    fi

    single_trial_marker="${FALLBACK_MODE_ROOT}/${CKPT_BASE}.${suite}.task${task_id}.single_trial_mode"
    trial_start="${TRIAL_START_VALUE}"
    while (( trial_start < TRIALS_PER_TASK )); do
      remaining=$((TRIALS_PER_TASK - trial_start))
      reset_context_start=""
      reset_rng_context_path=""
      reset_context_span="${CHUNK_TRIALS}"
      if [[ -f "${single_trial_marker}" ]]; then
        if ! load_single_trial_marker_partition "${single_trial_marker}" "${suite}" "${task_id}"; then
          die "invalid single-trial fallback marker: ${single_trial_marker}"
        fi
        reset_context_span="${marker_partition_chunk}"
        if (( trial_start >= marker_fallback_start )); then
          chunk=1
          if ! reset_context_start="$(canonical_reset_context_start "${trial_start}" "${marker_partition_start}" "${marker_partition_chunk}")"; then
            die "cannot resolve reset context for suite=${suite} task=${task_id} trial=${trial_start}"
          fi
        else
          chunk="${marker_partition_chunk}"
        fi
      else
        chunk="${CHUNK_TRIALS}"
      fi
      if (( remaining < chunk )); then
        chunk="${remaining}"
      fi
      log_suffix="_stage2_chunked_t${task_id}_r${trial_start}_n${chunk}"
      log_path="${LOG_ROOT}/${suite}/${CKPT_BASE}${log_suffix}.log"
      covered_by=""
      coverage_reset_context="${reset_context_start:-${trial_start}}"
      if [[ "${SKIP_COMPLETED}" == "1" ]] && covered_by="$(chunk_covered_by_existing_log "${suite}" "${task_id}" "${trial_start}" "${chunk}" "${coverage_reset_context}" "${reset_context_span}")"; then
        echo "[stage2_eval] skip covered chunk suite=${suite} task=${task_id} start=${trial_start} count=${chunk} source=${covered_by}"
        trial_start=$((trial_start + chunk))
        continue
      fi
      if [[ -n "${reset_context_start}" ]]; then
        ensure_reset_rng_context "${suite}" "${task_id}" "${reset_context_start}" "${trial_start}" "${reset_context_span}"
      fi

      attempt=1
      chunk_ok=0
      while (( attempt <= MAX_RETRIES )); do
        echo "========== stage2 suite=${suite} task=${task_id} trials=${trial_start}..$((trial_start + chunk - 1)) attempt=${attempt}/${MAX_RETRIES} =========="
        if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" >/dev/null 2>&1 || ! server_probe >/dev/null 2>&1; then
          echo "[stage2_eval] server is unavailable before chunk; restarting"
          start_server
        fi

        CLIENT_COMMAND=(
          env
          PYTHONFAULTHANDLER=1
          POLICY_SERVER_MODE=external
          SKIP_EVAL_PREFLIGHT=1
          TASK_START="${task_id}"
          TASK_COUNT=1
          TRIAL_START="${trial_start}"
          RESET_CONTEXT_START="${reset_context_start}"
          RESET_CONTEXT_CHUNK_SIZE="${reset_context_span}"
          RESET_RNG_CONTEXT_PATH="${reset_rng_context_path}"
          NUM_TRIALS_PER_TASK="${chunk}"
          MAX_TASKS=-1
          UNNORM_KEY="${UNNORM_KEY}"
          SAVE_VIDEOS="${SAVE_VIDEOS}"
          SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS}"
          MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK}"
          IMAGE_VIEWS="${IMAGE_VIEWS}"
          POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}"
          CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS}"
          MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
          CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS}"
          VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
          STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
          MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
          MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"
          EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}"
          EVAL_FINGERPRINT="${EVAL_FINGERPRINT}"
          LOG_SUFFIX="${log_suffix}"
          "${LOCAL_RUNNER}"
          "${CKPT}"
          "${suite}"
          "${GPU_ID}"
          "${PORT}"
        )
        client_status=0
        if run_client_command "${CLIENT_COMMAND[@]}"; then
          client_status=0
        else
          client_status=$?
        fi
        validation_status=0
        if validate_chunk "${log_path}" "${reset_context_start}"; then
          validation_status=0
        else
          validation_status=$?
        fi
        if (( client_status == 0 && validation_status == 0 )); then
          chunk_ok=1
          break
        fi
        echo "[stage2_eval] chunk attempt failed suite=${suite} task=${task_id} start=${trial_start} count=${chunk} attempt=${attempt}/${MAX_RETRIES} client_status=${client_status} validation_status=${validation_status}" >&2
        if fatal_failure "${log_path}"; then
          tail -120 "${log_path}" >&2 || true
          die "deterministic chunk failure; refusing to retry suite=${suite} task=${task_id} start=${trial_start}"
        fi
        if (( (client_status == 134 || client_status == 139) && chunk > 1 )); then
          marker_tmp="${single_trial_marker}.tmp.$$"
          {
            echo "marker_version=2"
            echo "enabled_at=$(date --iso-8601=seconds)"
            echo "reason=client_status_${client_status}"
            echo "suite=${suite}"
            echo "task=${task_id}"
            echo "schedule_trial_start=${TRIAL_START_VALUE}"
            echo "nominal_chunk_trials=${CHUNK_TRIALS}"
            echo "fallback_start=${trial_start}"
            echo "failed_trial_start=${trial_start}"
            echo "failed_chunk=${chunk}"
          } >"${marker_tmp}"
          mv -f "${marker_tmp}" "${single_trial_marker}"
          echo "[stage2_eval] client exited status=${client_status}; switching suite=${suite} task=${task_id} to reset-context-preserving single-trial chunks from trial=${trial_start} marker=${single_trial_marker}" >&2
          # Re-enter the trial loop without advancing trial_start. Its next
          # iteration observes the marker, uses n1, and can still reuse a
          # previously validated wider chunk through the strict coverage check.
          continue 2
        fi
        if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1 || ! server_probe >/dev/null 2>&1; then
          echo "[stage2_eval] server failed during chunk; restarting within budget"
          start_server
        fi
        attempt=$((attempt + 1))
        if (( attempt <= MAX_RETRIES )); then
          sleep $((attempt * 2))
        fi
      done
      (( chunk_ok == 1 )) || die "chunk failed after ${MAX_RETRIES} attempts suite=${suite} task=${task_id} start=${trial_start}"
      trial_start=$((trial_start + chunk))
    done
    write_progress
  done
done

write_progress | tee "${PARTIAL_SUMMARY_PATH}"
echo "[stage2_eval] worker complete id=${WORKER_ID} partial_summary=${PARTIAL_SUMMARY_PATH}"
stop_server
trap - EXIT INT TERM HUP
exit 0
