#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DEFAULT_RUN_DIR="${STARVLA_DIR}/Checkpoints/qwen_var_productvq_g16_s1248_mtr_from_scratch_e99_nextscale_50k_gbs128_jike8h100"

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_dir]

Strictly evaluate every archived Stage2 checkpoint with eight independent
single-GPU LIBERO workers. The original launcher transparently executes v2.

Important overrides:
  EVAL_GPUS="0 1 2 3 4 5 6 7"
  EVAL_PORT_BASE=19100
  TRIALS_PER_TASK=50 EVAL_SEED=7
  EVAL_OUTPUT_ROOT=eval_all_checkpoints_8gpu_seed7
  MIN_STEP=0 MAX_STEP=999999999 CHECKPOINT_ORDER=asc
  MAX_RETRIES=3 MAX_SERVER_RESTARTS=2
  REQUIRE_H100=1 REQUIRE_IDLE_GPUS=1
  MAX_GPU_MEMORY_USED_MIB=1024 MAX_GPU_UTILIZATION=10
  DRY_RUN=1  # performs every preflight, but starts no evaluator

Default run_dir:
  ${DEFAULT_RUN_DIR}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

die() {
  echo "[all_ckpt_eval] ERROR: $*" >&2
  exit 1
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
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    # Preserve the venv entrypoint symlink: resolving it to /usr/bin/python
    # disables pyvenv.cfg discovery and silently drops the venv packages.
    printf '%s/%s\n' "$(cd "$(dirname "${candidate}")" && pwd)" "$(basename "${candidate}")"
  else
    command -v "${candidate}"
  fi
}

RUN_DIR="${1:-${RUN_DIR:-${DEFAULT_RUN_DIR}}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3 4 5 6 7}"
EVAL_PORT_BASE="${EVAL_PORT_BASE:-19100}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
EVAL_SEED="${EVAL_SEED:-7}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_all_checkpoints_8gpu_seed${EVAL_SEED}}"
MIN_STEP="${MIN_STEP:-0}"
MAX_STEP="${MAX_STEP:-999999999}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-asc}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
REQUIRE_H100="${REQUIRE_H100:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
CHECKPOINT_STABILITY_SECONDS="${CHECKPOINT_STABILITY_SECONDS:-2}"
DRY_RUN="${DRY_RUN:-0}"

SPATIAL_CHUNK_TRIALS="${SPATIAL_CHUNK_TRIALS:-5}"
OBJECT_CHUNK_TRIALS="${OBJECT_CHUNK_TRIALS:-1}"
GOAL_CHUNK_TRIALS="${GOAL_CHUNK_TRIALS:-1}"
LIBERO10_CHUNK_TRIALS="${LIBERO10_CHUNK_TRIALS:-5}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS:-2}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-1200}"
POLICY_REQUEST_TIMEOUT_SECONDS="${POLICY_REQUEST_TIMEOUT_SECONDS:-600}"
PREFLIGHT_TIMEOUT_SECONDS="${PREFLIGHT_TIMEOUT_SECONDS:-300}"
WORKER_TERM_TIMEOUT_SECONDS="${WORKER_TERM_TIMEOUT_SECONDS:-30}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS:-0}"
MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK:--1}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"
USE_BF16="${EVAL_USE_BF16:-${USE_BF16:-1}}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-8}"

[[ -d "${RUN_DIR}" ]] || die "run directory does not exist: ${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
elif [[ "${CHECKPOINT_DIR}" != /* ]]; then
  CHECKPOINT_DIR="${RUN_DIR}/${CHECKPOINT_DIR}"
fi
[[ -d "${CHECKPOINT_DIR}" ]] || die "checkpoint directory does not exist: ${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
[[ "${CHECKPOINT_DIR}" == "${RUN_DIR}/checkpoints" || "${CHECKPOINT_DIR}" == "${RUN_DIR}/checkpoints/"* ]] || \
  die "CHECKPOINT_DIR must stay inside ${RUN_DIR}/checkpoints: ${CHECKPOINT_DIR}"

[[ -n "${EVAL_OUTPUT_ROOT}" && "${EVAL_OUTPUT_ROOT}" != /* ]] || die "EVAL_OUTPUT_ROOT must be a non-empty relative path"
OUTPUT_BASE="$(realpath -m "${RUN_DIR}/${EVAL_OUTPUT_ROOT}")"
[[ "${OUTPUT_BASE}" == "${RUN_DIR}/"* ]] || die "EVAL_OUTPUT_ROOT escapes RUN_DIR: ${EVAL_OUTPUT_ROOT}"

for pair in \
  "EVAL_PORT_BASE:${EVAL_PORT_BASE}" "TRIALS_PER_TASK:${TRIALS_PER_TASK}" \
  "EVAL_SEED:${EVAL_SEED}" "MIN_STEP:${MIN_STEP}" "MAX_STEP:${MAX_STEP}" \
  "SPATIAL_CHUNK_TRIALS:${SPATIAL_CHUNK_TRIALS}" "OBJECT_CHUNK_TRIALS:${OBJECT_CHUNK_TRIALS}" \
  "GOAL_CHUNK_TRIALS:${GOAL_CHUNK_TRIALS}" "LIBERO10_CHUNK_TRIALS:${LIBERO10_CHUNK_TRIALS}" \
  "MAX_RETRIES:${MAX_RETRIES}" "MAX_SERVER_RESTARTS:${MAX_SERVER_RESTARTS}" \
  "CHUNK_TIMEOUT_SECONDS:${CHUNK_TIMEOUT_SECONDS}" "SERVER_STARTUP_TIMEOUT_SECONDS:${SERVER_STARTUP_TIMEOUT_SECONDS}" \
  "POLICY_REQUEST_TIMEOUT_SECONDS:${POLICY_REQUEST_TIMEOUT_SECONDS}" "PREFLIGHT_TIMEOUT_SECONDS:${PREFLIGHT_TIMEOUT_SECONDS}" \
  "WORKER_TERM_TIMEOUT_SECONDS:${WORKER_TERM_TIMEOUT_SECONDS}" \
  "MAX_GPU_MEMORY_USED_MIB:${MAX_GPU_MEMORY_USED_MIB}" "MAX_GPU_UTILIZATION:${MAX_GPU_UTILIZATION}" \
  "CHECKPOINT_STABILITY_SECONDS:${CHECKPOINT_STABILITY_SECONDS}" "EVAL_CPU_THREADS:${EVAL_CPU_THREADS}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "SKIP_COMPLETED:${SKIP_COMPLETED}" "REQUIRE_H100:${REQUIRE_H100}" \
  "REQUIRE_IDLE_GPUS:${REQUIRE_IDLE_GPUS}" "DRY_RUN:${DRY_RUN}" \
  "SAVE_VIDEOS:${SAVE_VIDEOS}" "SAVE_ONLY_SUCCESS_VIDEOS:${SAVE_ONLY_SUCCESS_VIDEOS}" \
  "CONSTRAIN_TO_ACTION_TOKENS:${CONSTRAIN_TO_ACTION_TOKENS}" \
  "CLIP_NORMALIZED_ACTIONS:${CLIP_NORMALIZED_ACTIONS}" "VALIDATE_INPUTS:${VALIDATE_INPUTS}" \
  "STRICT_TRIAL_COUNT:${STRICT_TRIAL_COUNT}" "USE_BF16:${USE_BF16}"; do
  require_bool "${pair%%:*}" "${pair#*:}"
done
(( TRIALS_PER_TASK > 0 )) || die "TRIALS_PER_TASK must be positive"
(( EVAL_PORT_BASE > 0 && EVAL_PORT_BASE + 7 <= 65535 )) || die "eight ports must fit in 1..65535"
(( MIN_STEP <= MAX_STEP )) || die "MIN_STEP must not exceed MAX_STEP"
(( MAX_RETRIES > 0 && CHUNK_TIMEOUT_SECONDS > 0 && SERVER_STARTUP_TIMEOUT_SECONDS > 0 )) || die "retry/timeouts must be positive"
(( POLICY_REQUEST_TIMEOUT_SECONDS > 0 && PREFLIGHT_TIMEOUT_SECONDS > 0 && WORKER_TERM_TIMEOUT_SECONDS > 0 )) || die "preflight/request/termination timeouts must be positive"
(( CHECKPOINT_STABILITY_SECONDS > 0 )) || die "CHECKPOINT_STABILITY_SECONDS must be positive"
for chunk in "${SPATIAL_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}"; do
  (( chunk > 0 )) || die "chunk sizes must be positive"
done
[[ "${CHECKPOINT_ORDER}" == "asc" || "${CHECKPOINT_ORDER}" == "desc" ]] || die "CHECKPOINT_ORDER must be asc or desc"
[[ "${MAX_SUCCESS_VIDEOS_PER_TASK}" == "-1" || "${MAX_SUCCESS_VIDEOS_PER_TASK}" =~ ^[0-9]+$ ]] || die "invalid MAX_SUCCESS_VIDEOS_PER_TASK"

if [[ -d "${STARVLA_DIR}/third_party/LIBERO/libero" ]]; then
  DEFAULT_LIBERO_HOME="${STARVLA_DIR}/third_party/LIBERO"
else
  DEFAULT_LIBERO_HOME="/root/feihong/LIBERO"
fi
LIBERO_HOME="${LIBERO_HOME:-${DEFAULT_LIBERO_HOME}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
STARVLA_PYTHON="$(resolve_executable "${STARVLA_PYTHON:-/root/feihong/starVLA/.venv/bin/python}")" || die "STARVLA_PYTHON is not executable"
LIBERO_PYTHON="$(resolve_executable "${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}")" || die "LIBERO_PYTHON is not executable"

[[ -d "${LIBERO_HOME}/libero" ]] || die "invalid LIBERO_HOME: ${LIBERO_HOME}"
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || die "missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml"
[[ -x "${SCRIPT_DIR}/run_stage2_eval_chunked.sh" ]] || die "missing executable chunk runner"
[[ -x "${SCRIPT_DIR}/run_stage2_eval_chunked_persistent.sh" ]] || die "missing executable persistent chunk runner"
[[ -x "${SCRIPT_DIR}/run_local_eval_once.sh" ]] || die "missing executable local eval runner"
VALIDATOR="${SCRIPT_DIR}/validate_and_summarize_libero.py"
[[ -f "${VALIDATOR}" ]] || die "missing strict validator: ${VALIDATOR}"
for command_name in flock nvidia-smi realpath setsid sha256sum stat timeout; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

gpu_spec="${EVAL_GPUS//,/ }"
read -r -a GPUS <<< "${gpu_spec}"
[[ "${#GPUS[@]}" -eq 8 ]] || die "EVAL_GPUS must contain exactly eight IDs; got ${#GPUS[@]}"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  require_uint "GPU ID" "${gpu}"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU ID: ${gpu}"
  SEEN_GPUS["${gpu}"]=1
done

declare -a CHECKPOINTS=()
while IFS= read -r checkpoint; do
  checkpoint_name="$(basename "${checkpoint}")"
  if [[ "${checkpoint_name}" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]]; then
    step=$((10#${BASH_REMATCH[1]}))
    if (( step >= MIN_STEP && step <= MAX_STEP )); then
      CHECKPOINTS+=("$(readlink -f "${checkpoint}")")
    fi
  fi
done < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -print | sort -V)
[[ "${#CHECKPOINTS[@]}" -gt 0 ]] || die "no checkpoints found for steps ${MIN_STEP}..${MAX_STEP}"
if [[ "${CHECKPOINT_ORDER}" == "desc" ]]; then
  declare -a reversed=()
  for ((idx=${#CHECKPOINTS[@]} - 1; idx >= 0; idx--)); do
    reversed+=("${CHECKPOINTS[idx]}")
  done
  CHECKPOINTS=("${reversed[@]}")
fi

JOB_LABELS=(spatial_t0_9 object_t0_4 object_t5_9 goal_t0_4 goal_t5_9 libero10_t0_2 libero10_t3_5 libero10_t6_9)
JOB_SUITES=(libero_spatial libero_object libero_object libero_goal libero_goal libero_10 libero_10 libero_10)
JOB_TASK_STARTS=(0 0 5 0 5 0 3 6)
JOB_TASK_COUNTS=(10 5 5 5 5 3 3 4)
JOB_CHUNKS=("${SPATIAL_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}")

DISTRIBUTED_ENV_UNSET_ARGS=(
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE
  -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE
  -u MASTER_ADDR -u MASTER_PORT -u TORCHELASTIC_RUN_ID
  -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS
  -u TORCHELASTIC_ERROR_FILE -u DEBUG
)

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

CONTRACT_CODE_FILES=(
  "${SCRIPT_DIR}/run_jike_stage2_all_checkpoints_8gpu.sh"
  "${SCRIPT_DIR}/run_jike_stage2_all_checkpoints_8gpu_v2.sh"
  "${SCRIPT_DIR}/run_stage2_eval_chunked.sh"
  "${SCRIPT_DIR}/run_stage2_eval_chunked_persistent.sh"
  "${SCRIPT_DIR}/run_local_eval_once.sh"
  "${SCRIPT_DIR}/validate_and_summarize_libero.py"
  "${SCRIPT_DIR}/check_policy_server.py"
  "${SCRIPT_DIR}/eval_libero.py"
  "${SCRIPT_DIR}/model2libero_interface.py"
  "${STARVLA_DIR}/deployment/model_server/policy_norm_processor.py"
  "${STARVLA_DIR}/deployment/model_server/policy_wrapper.py"
  "${STARVLA_DIR}/deployment/model_server/server_policy.py"
  "${STARVLA_DIR}/deployment/model_server/tools/websocket_policy_client.py"
  "${STARVLA_DIR}/deployment/model_server/tools/websocket_policy_server.py"
  "${STARVLA_DIR}/starVLA/model/framework/VLM4A/QwenVARParallel.py"
  "${STARVLA_DIR}/starVLA/model/framework/VLM4A/QwenVARScaleParallel.py"
)

# These exact pre-fix hashes identify the only existing contract that may be
# resumed across the audited eval-orchestration resilience repairs. The
# comparison below still requires every model, environment, argument, and
# every non-orchestration code hash to remain byte-for-byte identical.
SAFE_RESUME_PATCH_BASELINES=(
  "examples/LIBERO/eval_files/run_jike_stage2_all_checkpoints_8gpu_v2.sh=4c1a8392f08688d6d6ef6bda2f756d865ca059f7d1cbfb6fffe075e4694a110f"
  "examples/LIBERO/eval_files/validate_and_summarize_libero.py=b867092618ae6d5f228442cf61191a6559e53a0458abfbb2437b4d710ce7565a"
  "examples/LIBERO/eval_files/run_local_eval_once.sh=8f2902ef0d0ec3647348fa56a27e1ab0a151390ab43d26423e02d8023dcd6576"
  "examples/LIBERO/eval_files/run_stage2_eval_chunked_persistent.sh=7d00de96b68f7fa7a9b08ca93ccaf1f1709dafb56f2e9dd710989baf7c9ced79"
  "examples/LIBERO/eval_files/eval_libero.py=9265d85bec017d01fda042942aa50b03b35af84959353722d637a46862c2ac16"
)

mkdir -p "${OUTPUT_BASE}"
MANAGER_LOCK_PATH="${OUTPUT_BASE}/.all_checkpoints_8gpu.manager.lock"
exec {MANAGER_LOCK_FD}>"${MANAGER_LOCK_PATH}"
flock -n "${MANAGER_LOCK_FD}" || die "another manager holds ${MANAGER_LOCK_PATH}"

DRIVER_LOG="${OUTPUT_BASE}/all_checkpoints_8gpu.log"
RESULTS_TSV="${OUTPUT_BASE}/all_checkpoint_results.tsv"

log() {
  local line
  line="[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
  echo "${line}"
  echo "${line}" >> "${DRIVER_LOG}"
}

declare -a ACTIVE_PIDS=()
declare -A PID_LABEL=()

process_group_alive() {
  kill -0 -- "-$1" >/dev/null 2>&1
}

stop_active_workers() {
  local pid deadline any_alive
  [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]] || return 0
  for pid in "${ACTIVE_PIDS[@]}"; do
    process_group_alive "${pid}" && kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
  done
  deadline=$((SECONDS + WORKER_TERM_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${ACTIVE_PIDS[@]}"; do
      if process_group_alive "${pid}"; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 0.25
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    process_group_alive "${pid}" && kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  ACTIVE_PIDS=()
  PID_LABEL=()
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    log "stopping ${#ACTIVE_PIDS[@]} active worker process groups"
    stop_active_workers
  fi
  if [[ "${status}" -ne 0 ]]; then
    log "exiting status=${status}; valid completed chunks remain resumable"
  fi
  exit "${status}"
}

handle_signal() {
  local status="$1"
  trap - EXIT INT TERM HUP
  log "received signal; stopping active worker process groups"
  stop_active_workers
  exit "${status}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP

print_configuration() {
  echo "[all_ckpt_eval] manager=v2"
  echo "[all_ckpt_eval] run_dir=${RUN_DIR}"
  echo "[all_ckpt_eval] checkpoint_dir=${CHECKPOINT_DIR}"
  echo "[all_ckpt_eval] output_base=${OUTPUT_BASE}"
  echo "[all_ckpt_eval] server_python=${STARVLA_PYTHON}"
  echo "[all_ckpt_eval] libero_python=${LIBERO_PYTHON}"
  echo "[all_ckpt_eval] checkpoints=${#CHECKPOINTS[@]} order=${CHECKPOINT_ORDER} steps=${MIN_STEP}..${MAX_STEP}"
  echo "[all_ckpt_eval] gpus=${GPUS[*]} ports=${EVAL_PORT_BASE}..$((EVAL_PORT_BASE + 7))"
  echo "[all_ckpt_eval] trials=${TRIALS_PER_TASK} seed=${EVAL_SEED} max_retries=${MAX_RETRIES} max_server_restarts=${MAX_SERVER_RESTARTS}"
  for idx in "${!JOB_LABELS[@]}"; do
    echo "  worker=${idx} gpu=${GPUS[idx]} port=$((EVAL_PORT_BASE + idx)) suite=${JOB_SUITES[idx]} tasks=${JOB_TASK_STARTS[idx]}..$((JOB_TASK_STARTS[idx] + JOB_TASK_COUNTS[idx] - 1)) chunk=${JOB_CHUNKS[idx]}"
  done
  echo "[all_ckpt_eval] checkpoint order:"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  $(basename "${checkpoint}")"
  done
}

preflight_contract_inputs() {
  local code_file
  for code_file in "${CONTRACT_CODE_FILES[@]}"; do
    [[ -f "${code_file}" && -r "${code_file}" ]] || die "contract input is missing or unreadable: ${code_file}"
  done
  echo "[all_ckpt_eval] contract code inputs OK: ${#CONTRACT_CODE_FILES[@]} files"
}

preflight_imports() {
  local sample_checkpoint="${CHECKPOINTS[0]}"
  echo "[all_ckpt_eval] preflight server imports/config"
  timeout --kill-after=10s "${PREFLIGHT_TIMEOUT_SECONDS}s" env "${DISTRIBUTED_ENV_UNSET_ARGS[@]}" \
    PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}" \
    "${STARVLA_PYTHON}" -c \
    'import sys; from deployment.model_server.server_policy import build_argparser; from starVLA.model.framework.share_tools import read_mode_config; cfg, stats = read_mode_config(sys.argv[1]); assert isinstance(cfg, dict); assert isinstance(stats, dict)' \
    "${sample_checkpoint}" </dev/null
  echo "[all_ckpt_eval] preflight LIBERO/robosuite/client imports"
  timeout --kill-after=10s "${PREFLIGHT_TIMEOUT_SECONDS}s" env "${DISTRIBUTED_ENV_UNSET_ARGS[@]}" \
    PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    "${LIBERO_PYTHON}" -c \
    'from libero.libero.envs import OffScreenRenderEnv; import robosuite, mujoco; from examples.LIBERO.eval_files.model2libero_interface import ModelClient; import examples.LIBERO.eval_files.eval_libero' \
    </dev/null
}

preflight_checkpoint_assets_and_stability() {
  local asset checkpoint signature
  for asset in config.full.yaml dataset_statistics.json; do
    [[ -s "${RUN_DIR}/${asset}" ]] || die "missing/empty checkpoint companion asset: ${RUN_DIR}/${asset}"
  done
  declare -A before=()
  for checkpoint in "${CHECKPOINTS[@]}"; do
    [[ "${checkpoint}" == "${RUN_DIR}/checkpoints/"* ]] || die "checkpoint escapes RUN_DIR/checkpoints: ${checkpoint}"
    [[ -s "${checkpoint}" ]] || die "checkpoint is missing or empty: ${checkpoint}"
    signature="$(stat -Lc '%s:%Y' "${checkpoint}")" || die "cannot stat checkpoint: ${checkpoint}"
    before["${checkpoint}"]="${signature}"
  done
  echo "[all_ckpt_eval] checking checkpoint stability for ${CHECKPOINT_STABILITY_SECONDS}s"
  sleep "${CHECKPOINT_STABILITY_SECONDS}"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    signature="$(stat -Lc '%s:%Y' "${checkpoint}")" || die "cannot restat checkpoint: ${checkpoint}"
    [[ "${signature}" == "${before[${checkpoint}]}" ]] || die "checkpoint changed during preflight: ${checkpoint}"
  done
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  REPLY="${value}"
}

preflight_gpus() {
  local gpu info gpu_name memory_used utilization
  for gpu in "${GPUS[@]}"; do
    info="$(nvidia-smi --id="${gpu}" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || die "GPU ${gpu} is not visible"
    [[ "${info}" != *$'\n'* ]] || die "GPU ${gpu} query returned multiple rows"
    IFS=',' read -r gpu_name memory_used utilization <<< "${info}"
    trim_whitespace "${gpu_name}"; gpu_name="${REPLY}"
    trim_whitespace "${memory_used}"; memory_used="${REPLY}"
    trim_whitespace "${utilization}"; utilization="${REPLY}"
    require_uint "GPU ${gpu} memory.used" "${memory_used}"
    require_uint "GPU ${gpu} utilization" "${utilization}"
    if [[ "${REQUIRE_H100}" == "1" && "${gpu_name}" != *H100* ]]; then
      die "GPU ${gpu} is '${gpu_name}', not an H100"
    fi
    if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]] && (( memory_used > MAX_GPU_MEMORY_USED_MIB || utilization > MAX_GPU_UTILIZATION )); then
      die "GPU ${gpu} is busy: memory_used=${memory_used}MiB utilization=${utilization}%"
    fi
    echo "[all_ckpt_eval] gpu=${gpu} name=${gpu_name} memory_used=${memory_used}MiB utilization=${utilization}%"
  done
}

preflight_ports() {
  local idx port
  local -a ports=()
  for idx in "${!GPUS[@]}"; do
    ports+=("$((EVAL_PORT_BASE + idx))")
  done
  for port in "${ports[@]}"; do
    "${LIBERO_PYTHON}" -c \
      'import socket, sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); s.close()' \
      "${port}" </dev/null || die "eval port ${port} has an active listener or non-reusable owner"
  done
  echo "[all_ckpt_eval] ports are free: ${ports[*]}"
}

print_configuration
preflight_contract_inputs
preflight_imports
preflight_checkpoint_assets_and_stability
preflight_gpus
preflight_ports
SERVER_ENV_FINGERPRINT="$(timeout --kill-after=10s "${PREFLIGHT_TIMEOUT_SECONDS}s" "${STARVLA_PYTHON}" -c 'import platform; from importlib.metadata import version; names=("torch", "transformers", "accelerate", "deepspeed", "websockets"); print("python=" + platform.python_version() + ";" + ";".join(name + "=" + version(name) for name in names))')"
LIBERO_ENV_FINGERPRINT="$(timeout --kill-after=10s "${PREFLIGHT_TIMEOUT_SECONDS}s" "${LIBERO_PYTHON}" -c 'import platform; from importlib.metadata import version; names=("libero", "robosuite", "mujoco", "numpy", "scipy", "torch", "websockets"); print("python=" + platform.python_version() + ";" + ";".join(name + "=" + version(name) for name in names))')"
echo "[all_ckpt_eval] server_env=${SERVER_ENV_FINGERPRINT}"
echo "[all_ckpt_eval] libero_env=${LIBERO_ENV_FINGERPRINT}"
echo "[all_ckpt_eval] full preflight OK"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[all_ckpt_eval] DRY_RUN=1; no evaluator was started"
  ACTIVE_PIDS=()
  trap - EXIT INT TERM HUP
  exit 0
fi

safe_resume_contract_patch() {
  local old_contract="$1" new_contract="$2" marker_path="$3"
  "${LIBERO_PYTHON}" -c '
import hashlib
import os
import sys
from pathlib import Path

old_path = Path(sys.argv[1])
new_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
baseline = dict(item.split("=", 1) for item in sys.argv[4:])
allowed = {"sha256[" + path + "]": digest for path, digest in baseline.items()}

def read_contract(path):
    lines = path.read_text().splitlines()
    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return lines, values

old_lines, old_values = read_contract(old_path)
new_lines, new_values = read_contract(new_path)
if any(old_values.get(key) != digest for key, digest in allowed.items()):
    raise SystemExit(1)

def semantic_lines(lines):
    return [
        line
        for line in lines
        if line.partition("=")[0] not in allowed
    ]

if semantic_lines(old_lines) != semantic_lines(new_lines):
    raise SystemExit(1)
if any(key not in new_values for key in allowed):
    raise SystemExit(1)

old_digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
snapshot = "".join(f"{key}={new_values[key]}\n" for key in sorted(allowed))
payload = (
    "compatibility=eval-orchestration-resilience-v4\n"
    f"old_contract_sha256={old_digest}\n"
    + snapshot
)
if marker_path.exists():
    if marker_path.read_text() != payload:
        raise SystemExit(1)
else:
    temporary = marker_path.with_name(marker_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, marker_path)
' "${old_contract}" "${new_contract}" "${marker_path}" "${SAFE_RESUME_PATCH_BASELINES[@]}"
}

prepare_contract() {
  local checkpoint="$1" checkpoint_base="$2" checkpoint_root="$3"
  local contract_path="${checkpoint_root}/eval_contract.txt"
  local fingerprint_path="${checkpoint_root}/eval_contract.sha256"
  local compatibility_marker="${checkpoint_root}/contract_resume_compatibility_v4.txt"
  local temporary="${contract_path}.tmp.$$"
  local code_file digest relative old_success=0
  mkdir -p "${checkpoint_root}"
  {
    echo "contract_version=2"
    echo "checkpoint=${checkpoint}"
    echo "checkpoint_size=$(stat -Lc '%s' "${checkpoint}")"
    echo "checkpoint_mtime=$(stat -Lc '%Y' "${checkpoint}")"
    echo "checkpoint_ctime=$(stat -Lc '%Z' "${checkpoint}")"
    echo "checkpoint_inode=$(stat -Lc '%i' "${checkpoint}")"
    echo "run_dir=${RUN_DIR}"
    echo "checkpoint_base=${checkpoint_base}"
    echo "trials_per_task=${TRIALS_PER_TASK}"
    echo "seed=${EVAL_SEED}"
    echo "image_views=${IMAGE_VIEWS}"
    echo "policy_image_size=${POLICY_IMAGE_SIZE}"
    echo "unnorm_key=${UNNORM_KEY}"
    echo "use_bf16=${USE_BF16}"
    echo "constrain_to_action_tokens=${CONSTRAIN_TO_ACTION_TOKENS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "clip_normalized_actions=${CLIP_NORMALIZED_ACTIONS}"
    echo "validate_inputs=${VALIDATE_INPUTS}"
    echo "strict_trial_count=${STRICT_TRIAL_COUNT}"
    echo "min_image_mean=${MIN_IMAGE_MEAN}"
    echo "min_image_std=${MIN_IMAGE_STD}"
    echo "spatial_chunk_trials=${SPATIAL_CHUNK_TRIALS}"
    echo "object_chunk_trials=${OBJECT_CHUNK_TRIALS}"
    echo "goal_chunk_trials=${GOAL_CHUNK_TRIALS}"
    echo "libero10_chunk_trials=${LIBERO10_CHUNK_TRIALS}"
    echo "save_videos=${SAVE_VIDEOS}"
    echo "save_only_success_videos=${SAVE_ONLY_SUCCESS_VIDEOS}"
    echo "max_success_videos_per_task=${MAX_SUCCESS_VIDEOS_PER_TASK}"
    echo "starvla_python=${STARVLA_PYTHON}"
    echo "libero_python=${LIBERO_PYTHON}"
    echo "server_env=${SERVER_ENV_FINGERPRINT}"
    echo "libero_env=${LIBERO_ENV_FINGERPRINT}"
    echo "eval_cpu_threads=${EVAL_CPU_THREADS}"
    echo "max_retries=${MAX_RETRIES}"
    echo "libero_home=${LIBERO_HOME}"
    echo "libero_config_path=${LIBERO_CONFIG_PATH}"
    for code_file in "${CONTRACT_CODE_FILES[@]}" "${RUN_DIR}/config.full.yaml" "${RUN_DIR}/dataset_statistics.json" "${LIBERO_CONFIG_PATH}/config.yaml"; do
      [[ -f "${code_file}" ]] || die "contract input is missing: ${code_file}"
      digest="$(sha256sum "${code_file}")"
      digest="${digest%% *}"
      relative="${code_file#${STARVLA_DIR}/}"
      echo "sha256[${relative}]=${digest}"
    done
  } > "${temporary}"

  if [[ -f "${contract_path}" ]] && cmp -s "${temporary}" "${contract_path}"; then
    :
  else
    if [[ -d "${checkpoint_root}/logs" ]] && grep -Rql --include='*.log' 'EVAL_CHUNK_OK' "${checkpoint_root}/logs"; then
      old_success=1
    fi
    if (( old_success == 1 )); then
      if safe_resume_contract_patch "${contract_path}" "${temporary}" "${compatibility_marker}"; then
        echo "[all_ckpt_eval] retaining prior eval fingerprint for audited orchestration-only resume patch: ${checkpoint_base}" >&2
      else
        die "eval contract changed beyond the audited orchestration-only resume patch for ${checkpoint_base}; choose a new EVAL_OUTPUT_ROOT"
      fi
    else
      mv "${temporary}" "${contract_path}"
    fi
  fi
  [[ ! -f "${temporary}" ]] || rm -f "${temporary}"
  digest="$(sha256sum "${contract_path}")"
  digest="${digest%% *}"
  printf '%s\n' "${digest}" > "${fingerprint_path}.tmp.$$"
  mv "${fingerprint_path}.tmp.$$" "${fingerprint_path}"
  echo "${digest}"
}

strict_checkpoint_complete() {
  local checkpoint_base="$1" log_root="$2" fingerprint="$3"
  local checkpoint_root="$(dirname "${log_root}")"
  "${LIBERO_PYTHON}" "${VALIDATOR}" "${log_root}" \
    --checkpoint-base "${checkpoint_base}" \
    --eval-fingerprint "${fingerprint}" \
    --eval-contract "${checkpoint_root}/eval_contract.txt" \
    --expected-trials-per-task "${TRIALS_PER_TASK}" \
    --require-complete >/dev/null 2>&1
}

finalize_checkpoint() {
  local checkpoint_base="$1" checkpoint_root="$2" fingerprint="$3"
  local log_root="${checkpoint_root}/logs"
  local summary_path="${log_root}/libero_40task_summary.txt"
  local manifest_path="${checkpoint_root}/complete_manifest.json"
  local temporary="${summary_path}.tmp.$$"
  mkdir -p "${log_root}"
  if ! "${LIBERO_PYTHON}" "${VALIDATOR}" "${log_root}" \
      --checkpoint-base "${checkpoint_base}" \
      --eval-fingerprint "${fingerprint}" \
      --eval-contract "${checkpoint_root}/eval_contract.txt" \
      --expected-trials-per-task "${TRIALS_PER_TASK}" \
      --require-complete --manifest-out "${manifest_path}" > "${temporary}"; then
    return 1
  fi
  mv "${temporary}" "${summary_path}"
}

rebuild_results_table() {
  local temporary="${RESULTS_TSV}.tmp.$$"
  local checkpoint checkpoint_base step checkpoint_root log_root summary_text overall fingerprint
  printf 'step\tcheckpoint\toverall_40_task_mean\tsummary\tmanifest\n' > "${temporary}"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_base="$(basename "${checkpoint}" .pt)"
    checkpoint_root="${OUTPUT_BASE}/${checkpoint_base}"
    log_root="${checkpoint_root}/logs"
    [[ -f "${checkpoint_root}/eval_contract.sha256" && -f "${checkpoint_root}/complete_manifest.json" ]] || continue
    fingerprint="$(<"${checkpoint_root}/eval_contract.sha256")"
    if ! summary_text="$("${LIBERO_PYTHON}" "${VALIDATOR}" "${log_root}" \
        --checkpoint-base "${checkpoint_base}" \
        --eval-fingerprint "${fingerprint}" \
        --eval-contract "${checkpoint_root}/eval_contract.txt" \
        --expected-trials-per-task "${TRIALS_PER_TASK}" --require-complete 2>/dev/null)"; then
      continue
    fi
    overall="$(printf '%s\n' "${summary_text}" | awk -F', ' '/^overall_40_task_mean:/ {print $2}')"
    [[ -n "${overall}" ]] || continue
    step="${checkpoint_base#steps_}"
    step="${step%_pytorch_model}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${step}" "${checkpoint_base}" "${overall}" \
      "${log_root}/libero_40task_summary.txt" "${checkpoint_root}/complete_manifest.json" >> "${temporary}"
  done
  mv "${temporary}" "${RESULTS_TSV}"
}

remove_active_pid() {
  local finished="$1" pid
  local -a remaining=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    [[ "${pid}" == "${finished}" ]] || remaining+=("${pid}")
  done
  ACTIVE_PIDS=("${remaining[@]}")
  unset 'PID_LABEL['"${finished}"']'
}

cd "${STARVLA_DIR}"
log "starting ${#CHECKPOINTS[@]} checkpoints with eight persistent single-GPU workers"

for checkpoint_idx in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[checkpoint_idx]}"
  checkpoint_base="$(basename "${checkpoint}" .pt)"
  checkpoint_root="${OUTPUT_BASE}/${checkpoint_base}"
  log_root="${checkpoint_root}/logs"
  mkdir -p "${log_root}"
  contract_fingerprint="$(prepare_contract "${checkpoint}" "${checkpoint_base}" "${checkpoint_root}")"
  log "contract ${checkpoint_base} sha256=${contract_fingerprint}"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && strict_checkpoint_complete "${checkpoint_base}" "${log_root}" "${contract_fingerprint}"; then
    finalize_checkpoint "${checkpoint_base}" "${checkpoint_root}" "${contract_fingerprint}" || die "strict finalization failed for ${checkpoint_base}"
    rebuild_results_table
    log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] skip strictly completed ${checkpoint_base}"
    continue
  fi

  log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] evaluating ${checkpoint_base}"
  ACTIVE_PIDS=()
  PID_LABEL=()
  for job_idx in "${!JOB_LABELS[@]}"; do
    label="${JOB_LABELS[job_idx]}"
    suite="${JOB_SUITES[job_idx]}"
    gpu="${GPUS[job_idx]}"
    port=$((EVAL_PORT_BASE + job_idx))
    task_start="${JOB_TASK_STARTS[job_idx]}"
    task_count="${JOB_TASK_COUNTS[job_idx]}"
    chunk_trials="${JOB_CHUNKS[job_idx]}"
    launch_log="${checkpoint_root}/launch_${job_idx}_${label}.log"
    log "launch ${checkpoint_base} worker=${label} gpu=${gpu} port=${port} tasks=${task_start}..$((task_start + task_count - 1))"
    setsid env "${DISTRIBUTED_ENV_UNSET_ARGS[@]}" \
      PATH="$(dirname "${STARVLA_PYTHON}"):$(dirname "${LIBERO_PYTHON}"):${PATH}" \
      PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS="${EVAL_CPU_THREADS}" MKL_NUM_THREADS="${EVAL_CPU_THREADS}" \
      STARVLA_DIR="${STARVLA_DIR}" STARVLA_PYTHON="${STARVLA_PYTHON}" LIBERO_PYTHON="${LIBERO_PYTHON}" \
      LIBERO_HOME="${LIBERO_HOME}" LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
      MODEL_ROOT="${RUN_DIR}" RUN_DIR="${RUN_DIR}" WORKER_ID="${job_idx}_${label}" SKIP_WORKER_PREFLIGHT=1 \
      TASK_SUITES_OVERRIDE="${suite}" TASK_START="${task_start}" TASK_COUNT="${task_count}" \
      EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${checkpoint_base}" \
      EVAL_FINGERPRINT="${contract_fingerprint}" SKIP_COMPLETED="${SKIP_COMPLETED}" \
      TRIALS_PER_TASK="${TRIALS_PER_TASK}" CHUNK_TRIALS="${chunk_trials}" \
      MAX_RETRIES="${MAX_RETRIES}" MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS}" \
      CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS}" SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS}" \
      POLICY_REQUEST_TIMEOUT_SECONDS="${POLICY_REQUEST_TIMEOUT_SECONDS}" UNNORM_KEY="${UNNORM_KEY}" \
      SAVE_VIDEOS="${SAVE_VIDEOS}" SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS}" \
      MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK}" IMAGE_VIEWS="${IMAGE_VIEWS}" \
      POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}" CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS}" \
      VALIDATE_INPUTS="${VALIDATE_INPUTS}" STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT}" \
      MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN}" MIN_IMAGE_STD="${MIN_IMAGE_STD}" \
      USE_BF16="${USE_BF16}" SEED="${EVAL_SEED}" \
      bash "${SCRIPT_DIR}/run_stage2_eval_chunked.sh" "${checkpoint}" "${gpu}" "${port}" \
      > "${launch_log}" 2>&1 &
    worker_pid=$!
    ACTIVE_PIDS+=("${worker_pid}")
    PID_LABEL["${worker_pid}"]="${label}"
  done

  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    finished_pid=""
    if wait -n -p finished_pid "${ACTIVE_PIDS[@]}"; then
      worker_status=0
    else
      worker_status=$?
    fi
    [[ -n "${finished_pid}" ]] || die "wait -n did not report a finished worker"
    finished_label="${PID_LABEL[${finished_pid}]:-pid_${finished_pid}}"
    if (( worker_status != 0 )); then
      log "worker failed ${checkpoint_base} worker=${finished_label} status=${worker_status}; cancelling peers"
      # Keep the failed leader's PGID in ACTIVE_PIDS. wait(1) has reaped the
      # leader, but a descendant can still be alive in that process group.
      stop_active_workers
      die "worker ${finished_label} failed for ${checkpoint_base}; rerun the same command to resume"
    fi
    remove_active_pid "${finished_pid}"
    log "worker completed ${checkpoint_base} worker=${finished_label} remaining=${#ACTIVE_PIDS[@]}"
  done

  finalize_checkpoint "${checkpoint_base}" "${checkpoint_root}" "${contract_fingerprint}" || \
    die "workers exited but strict 40 x ${TRIALS_PER_TASK} validation failed for ${checkpoint_base}"
  rebuild_results_table
  log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] completed ${checkpoint_base}; manifest=${checkpoint_root}/complete_manifest.json"
done

rebuild_results_table
log "all checkpoints completed; results=${RESULTS_TSV}"
ACTIVE_PIDS=()
trap - EXIT INT TERM HUP
exit 0
