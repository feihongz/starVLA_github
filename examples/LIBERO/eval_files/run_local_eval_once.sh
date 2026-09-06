#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint.pt> <task_suite> [gpu_id] [port]"
  echo "Example: $0 playground/Checkpoints/run/checkpoints/steps_32000_pytorch_model.pt libero_spatial 2 18080"
  exit 2
fi

CKPT="$1"
TASK_SUITE="$2"
GPU_ID="${3:-2}"
PORT="${4:-18080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_DIR="${STARVLA_DIR:-${DEFAULT_STARVLA_DIR}}"
if [[ -z "${LIBERO_HOME:-}" ]]; then
  if [[ -d "/root/feihong/LIBERO/libero" ]]; then
    LIBERO_HOME="/root/feihong/LIBERO"
  else
    LIBERO_HOME="${STARVLA_DIR}/third_party/LIBERO"
  fi
fi
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
if [[ -x "/root/feihong/starVLA/.venv/bin/python" ]]; then
  DEFAULT_STARVLA_PYTHON="/root/feihong/starVLA/.venv/bin/python"
else
  DEFAULT_STARVLA_PYTHON="python"
fi
if [[ -x "${LIBERO_HOME}/.venv/bin/python" ]]; then
  DEFAULT_LIBERO_PYTHON="${LIBERO_HOME}/.venv/bin/python"
else
  DEFAULT_LIBERO_PYTHON="python"
fi
STARVLA_PYTHON="${STARVLA_PYTHON:-${DEFAULT_STARVLA_PYTHON}}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${DEFAULT_LIBERO_PYTHON}}"
POLICY_SERVER_MODE="${POLICY_SERVER_MODE:-managed}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-1200}"
POLICY_CONNECT_TIMEOUT_SECONDS="${POLICY_CONNECT_TIMEOUT_SECONDS:-30}"
POLICY_HANDSHAKE_TIMEOUT_SECONDS="${POLICY_HANDSHAKE_TIMEOUT_SECONDS:-30}"
POLICY_REQUEST_TIMEOUT_SECONDS="${POLICY_REQUEST_TIMEOUT_SECONDS:-600}"
SKIP_EVAL_PREFLIGHT="${SKIP_EVAL_PREFLIGHT:-0}"
EVAL_FINGERPRINT="${EVAL_FINGERPRINT:-}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
TASK_START="${TASK_START:-0}"
TASK_COUNT="${TASK_COUNT:--1}"
TRIAL_START="${TRIAL_START:-0}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
USE_BF16="${USE_BF16:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS:-0}"
MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK:--1}"
LOG_SUFFIX="${LOG_SUFFIX:-}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_smoke}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-0}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
SEED="${SEED:-7}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"

die() {
  echo "[eval] ERROR: $*" >&2
  exit 2
}

require_uint() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'"
}

require_bool() {
  local name="$1" value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || die "${name} must be 0 or 1, got '${value}'"
}

resolve_executable() {
  local candidate="$1"
  local executable_dir
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    executable_dir="$(cd "$(dirname "${candidate}")" && pwd -P)" || return 1
    printf '%s/%s\n' "${executable_dir}" "$(basename "${candidate}")"
  else
    candidate="$(command -v "${candidate}")" || return 1
    printf '%s\n' "${candidate}"
  fi
}

case "${POLICY_SERVER_MODE}" in
  managed|server-only|external|preflight) ;;
  *) die "POLICY_SERVER_MODE must be managed, server-only, external, or preflight" ;;
esac
case "${TASK_SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) die "unsupported task suite: ${TASK_SUITE}" ;;
esac
for pair in \
  "GPU_ID:${GPU_ID}" "PORT:${PORT}" "SERVER_STARTUP_TIMEOUT_SECONDS:${SERVER_STARTUP_TIMEOUT_SECONDS}" \
  "NUM_TRIALS_PER_TASK:${NUM_TRIALS_PER_TASK}" "TASK_START:${TASK_START}" "TRIAL_START:${TRIAL_START}" \
  "POLICY_IMAGE_SIZE:${POLICY_IMAGE_SIZE}" "SEED:${SEED}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "SKIP_EVAL_PREFLIGHT:${SKIP_EVAL_PREFLIGHT}" "USE_BF16:${USE_BF16}" "SAVE_VIDEOS:${SAVE_VIDEOS}" \
  "SAVE_ONLY_SUCCESS_VIDEOS:${SAVE_ONLY_SUCCESS_VIDEOS}" \
  "CONSTRAIN_TO_ACTION_TOKENS:${CONSTRAIN_TO_ACTION_TOKENS}" \
  "CLIP_NORMALIZED_ACTIONS:${CLIP_NORMALIZED_ACTIONS}" "VALIDATE_INPUTS:${VALIDATE_INPUTS}" \
  "STRICT_TRIAL_COUNT:${STRICT_TRIAL_COUNT}"; do
  require_bool "${pair%%:*}" "${pair#*:}"
done
[[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT <= 65535 )) || die "invalid port: ${PORT}"
[[ "${SERVER_STARTUP_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] && (( SERVER_STARTUP_TIMEOUT_SECONDS > 0 )) || die "invalid SERVER_STARTUP_TIMEOUT_SECONDS=${SERVER_STARTUP_TIMEOUT_SECONDS}"
(( NUM_TRIALS_PER_TASK > 0 )) || die "NUM_TRIALS_PER_TASK must be positive"
(( TASK_START < 10 )) || die "TASK_START must be below 10"
[[ "${MAX_TASKS}" == "-1" || "${MAX_TASKS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_TASKS must be -1 or positive"
[[ "${TASK_COUNT}" == "-1" || "${TASK_COUNT}" =~ ^[1-9][0-9]*$ ]] || die "TASK_COUNT must be -1 or positive"
if [[ "${TASK_COUNT}" != "-1" ]]; then
  (( TASK_START + TASK_COUNT <= 10 )) || die "TASK_START + TASK_COUNT must not exceed 10"
fi
[[ "${MAX_SUCCESS_VIDEOS_PER_TASK}" == "-1" || "${MAX_SUCCESS_VIDEOS_PER_TASK}" =~ ^[0-9]+$ ]] || die "invalid MAX_SUCCESS_VIDEOS_PER_TASK=${MAX_SUCCESS_VIDEOS_PER_TASK}"
[[ -z "${MAX_NEW_TOKENS}" || "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "invalid MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
[[ -z "${EVAL_FINGERPRINT}" || "${EVAL_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ]] || die "EVAL_FINGERPRINT must be empty or a lowercase SHA-256"
[[ "${LOG_SUFFIX}" =~ ^[A-Za-z0-9_.-]*$ ]] || die "LOG_SUFFIX contains unsafe characters"
case "${IMAGE_VIEWS}" in
  auto|primary|primary+wrist|wrist+primary) ;;
  *) die "invalid IMAGE_VIEWS=${IMAGE_VIEWS}" ;;
esac
STARVLA_PYTHON="$(resolve_executable "${STARVLA_PYTHON}")" || die "STARVLA_PYTHON is not executable: ${STARVLA_PYTHON}"
LIBERO_PYTHON="$(resolve_executable "${LIBERO_PYTHON}")" || die "LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}"
[[ -f "${CKPT}" ]] || die "checkpoint does not exist: ${CKPT}"
CKPT="$(readlink -f "${CKPT}")"
checkpoint_parent="$(dirname "${CKPT}")"
if [[ -n "${MODEL_ROOT:-}" ]]; then
  [[ -d "${MODEL_ROOT}" ]] || die "MODEL_ROOT does not exist: ${MODEL_ROOT}"
  MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd)"
elif [[ "$(basename "${checkpoint_parent}")" == "checkpoints" ]]; then
  MODEL_ROOT="$(cd "${checkpoint_parent}/.." && pwd)"
else
  die "cannot infer MODEL_ROOT because checkpoint is not under a checkpoints/ directory; set MODEL_ROOT explicitly"
fi
[[ "${CKPT}" == "${MODEL_ROOT}/checkpoints/"* ]] || die "checkpoint must be inside ${MODEL_ROOT}/checkpoints: ${CKPT}"
for asset in config.full.yaml dataset_statistics.json; do
  [[ -f "${MODEL_ROOT}/${asset}" ]] || die "missing checkpoint companion asset: ${MODEL_ROOT}/${asset}"
done
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || die "missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml"
[[ -d "${LIBERO_HOME}/libero" ]] || die "invalid LIBERO_HOME: ${LIBERO_HOME}"
[[ -n "${EVAL_OUTPUT_ROOT}" && "${EVAL_OUTPUT_ROOT}" != /* ]] || die "EVAL_OUTPUT_ROOT must be a non-empty relative path"
OUTPUT_BASE="$(realpath -m "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}")"
[[ "${OUTPUT_BASE}" == "${MODEL_ROOT}/"* ]] || die "EVAL_OUTPUT_ROOT escapes MODEL_ROOT: ${EVAL_OUTPUT_ROOT}"

export LIBERO_CONFIG_PATH
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CLIP_NORMALIZED_ACTIONS
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export POLICY_CONNECT_TIMEOUT_SECONDS
export POLICY_HANDSHAKE_TIMEOUT_SECONDS
export POLICY_REQUEST_TIMEOUT_SECONDS
unset DEBUG
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
unset TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_ERROR_FILE

cd "${STARVLA_DIR}"

CKPT_BASENAME="$(basename "${CKPT}" .pt)"
RUN_NAME="$(basename "${MODEL_ROOT}")"
VIDEO_OUT_PATH="${OUTPUT_BASE}/videos/${TASK_SUITE}/${CKPT_BASENAME}"
LOG_DIR="${OUTPUT_BASE}/logs/${TASK_SUITE}"
LOG_STEM="${CKPT_BASENAME}${LOG_SUFFIX}"
LOG_PATH="${LOG_DIR}/${LOG_STEM}.log"
SERVER_LOG_PATH="${LOG_DIR}/${LOG_STEM}.server.log"

mkdir -p "${VIDEO_OUT_PATH}" "${LOG_DIR}"

run_preflight() {
  echo "[eval] preflight server_python=${STARVLA_PYTHON}"
  env -u DEBUG "${STARVLA_PYTHON}" -c 'import sys; from deployment.model_server.server_policy import build_argparser; from starVLA.model.framework.share_tools import read_mode_config; cfg, stats = read_mode_config(sys.argv[1]); assert isinstance(cfg, dict); assert isinstance(stats, dict)' "${CKPT}" </dev/null
  echo "[eval] preflight libero_python=${LIBERO_PYTHON}"
  env -u DEBUG "${LIBERO_PYTHON}" -c 'from libero.libero.envs import OffScreenRenderEnv; from examples.LIBERO.eval_files.model2libero_interface import ModelClient; import examples.LIBERO.eval_files.eval_libero' </dev/null
}

if [[ "${SKIP_EVAL_PREFLIGHT}" != "1" ]]; then
  run_preflight
fi
if [[ "${POLICY_SERVER_MODE}" == "preflight" ]]; then
  echo "[eval] preflight OK ckpt=${CKPT}"
  exit 0
fi

SERVER_ARGS=(
  deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --host 127.0.0.1
  --port "${PORT}"
  --unnorm_key "${UNNORM_KEY}"
  --idle_timeout -1
)
if [[ "${USE_BF16}" == "1" ]]; then
  SERVER_ARGS+=(--use_bf16)
fi
EVAL_EXTRA_ARGS=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.save-videos)
else
  EVAL_EXTRA_ARGS+=(--args.no-save-videos)
fi
if [[ "${SAVE_ONLY_SUCCESS_VIDEOS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.save-only-success-videos)
fi
if [[ "${MAX_SUCCESS_VIDEOS_PER_TASK}" != "-1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.max-success-videos-per-task "${MAX_SUCCESS_VIDEOS_PER_TASK}")
fi
if [[ "${CONSTRAIN_TO_ACTION_TOKENS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.constrain-to-action-tokens)
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  EVAL_EXTRA_ARGS+=(--args.max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${POLICY_IMAGE_SIZE}" != "0" ]]; then
  EVAL_EXTRA_ARGS+=(--args.policy-image-size "${POLICY_IMAGE_SIZE}")
fi
if [[ "${VALIDATE_INPUTS}" != "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.no-validate-inputs)
fi
if [[ "${STRICT_TRIAL_COUNT}" != "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.no-strict-trial-count)
fi
EVAL_EXTRA_ARGS+=(--args.min-image-mean "${MIN_IMAGE_MEAN}")
EVAL_EXTRA_ARGS+=(--args.min-image-std "${MIN_IMAGE_STD}")

echo "[eval] run=${RUN_NAME}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] suite=${TASK_SUITE} trials_per_task=${NUM_TRIALS_PER_TASK} max_tasks=${MAX_TASKS} task_start=${TASK_START} task_count=${TASK_COUNT} trial_start=${TRIAL_START} seed=${SEED} image_views=${IMAGE_VIEWS} policy_image_size=${POLICY_IMAGE_SIZE} constrain_to_action_tokens=${CONSTRAIN_TO_ACTION_TOKENS} max_new_tokens=${MAX_NEW_TOKENS} clip_normalized_actions=${CLIP_NORMALIZED_ACTIONS} validate_inputs=${VALIDATE_INPUTS} strict_trial_count=${STRICT_TRIAL_COUNT} save_only_success_videos=${SAVE_ONLY_SUCCESS_VIDEOS} max_success_videos_per_task=${MAX_SUCCESS_VIDEOS_PER_TASK} output_root=${EVAL_OUTPUT_ROOT}"
echo "[eval] gpu=${GPU_ID} port=${PORT} mujoco_egl_device_id=${MUJOCO_EGL_DEVICE_ID}"
echo "[eval] videos=${VIDEO_OUT_PATH}"
echo "[eval] log=${LOG_PATH}"

port_is_free() {
  "${LIBERO_PYTHON}" -c 'import socket, sys; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); s.close()' "${PORT}" </dev/null
}

if [[ "${POLICY_SERVER_MODE}" == "server-only" ]]; then
  port_is_free || die "port ${PORT} is already occupied; refusing to start or connect to an unknown server"
  echo "[eval] starting persistent policy server gpu=${GPU_ID} port=${PORT} ckpt=${CKPT}"
  exec env -u DEBUG CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" "${SERVER_ARGS[@]}"
fi

SERVER_PID=""

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
  stop_server
  exit "${status}"
}

handle_signal() {
  local status="$1"
  trap - EXIT INT TERM HUP
  stop_server
  exit "${status}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP

wait_for_server() {
  local start_seconds="${SECONDS}"
  local probe_log="${SERVER_LOG_PATH}.probe"
  while (( SECONDS - start_seconds < SERVER_STARTUP_TIMEOUT_SECONDS )); do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "[eval] policy server exited before readiness; server log:" >&2
      tail -120 "${SERVER_LOG_PATH}" >&2 || true
      return 1
    fi
    if env -u DEBUG POLICY_CONNECT_TIMEOUT_SECONDS=5 POLICY_HANDSHAKE_TIMEOUT_SECONDS=5 "${LIBERO_PYTHON}" "${SCRIPT_DIR}/check_policy_server.py" --host 127.0.0.1 --port "${PORT}" --expected-ckpt "${CKPT}" --unnorm-key "${UNNORM_KEY}" --timeout 5 >"${probe_log}" 2>&1; then
      cat "${probe_log}"
      return 0
    fi
    sleep 2
  done
  echo "[eval] policy server did not pass websocket readiness/identity checks in ${SERVER_STARTUP_TIMEOUT_SECONDS}s" >&2
  tail -120 "${SERVER_LOG_PATH}" >&2 || true
  tail -40 "${SERVER_LOG_PATH}.probe" >&2 || true
  return 1
}

if [[ "${POLICY_SERVER_MODE}" == "managed" ]]; then
  port_is_free || die "port ${PORT} is already occupied; refusing to connect to a stale server"
  env -u DEBUG CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" "${SERVER_ARGS[@]}" >"${SERVER_LOG_PATH}" 2>&1 &
  SERVER_PID=$!
  echo "[eval] waiting for policy server pid=${SERVER_PID}"
  wait_for_server
fi

"${LIBERO_PYTHON}" examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.task-start "${TASK_START}" \
  --args.task-count "${TASK_COUNT}" \
  --args.trial-start "${TRIAL_START}" \
  --args.seed "${SEED}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  --args.image-views "${IMAGE_VIEWS}" \
  --args.eval-fingerprint "${EVAL_FINGERPRINT}" \
  "${EVAL_EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[eval] completed ${TASK_SUITE}; result log=${LOG_PATH}"
