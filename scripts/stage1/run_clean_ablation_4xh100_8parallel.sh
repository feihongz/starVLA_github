#!/usr/bin/env bash
set -Eeuo pipefail

# Run the controlled Stage1 2-benchmark x 4-method matrix on four H100s.
# Each GPU hosts exactly one LIBERO and one RoboCasa process during a phase.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

readonly SEED=42
readonly EPOCHS=50
readonly ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
readonly METHODS=(
  multiscale_base
  full_target_time
  mint_paper_dct
  mtr
)
# Cross the benchmark mappings so no GPU receives the same method twice.
readonly LIBERO_GPUS=(0 1 2 3)
readonly ROBOCASA_GPUS=(3 2 1 0)

DRY_RUN="${DRY_RUN:-0}"
RESUME_QUEUE="${RESUME_QUEUE:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_GPU_PREFLIGHT="${SKIP_GPU_PREFLIGHT:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-2}"
LIBERO_INTERMEDIATE_WEIGHT="${LIBERO_INTERMEDIATE_WEIGHT:-0.02}"
ROBOCASA_INTERMEDIATE_WEIGHT="${ROBOCASA_INTERMEDIATE_WEIGHT:-0.1}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv310/bin/python}"
CHECKPOINT_ROOT="${STAGE1_ABLATION_CHECKPOINT_ROOT:-${REPO_ROOT}/playground/Checkpoints/stage1_clean_supervision_ablation}"
LOG_ROOT="${STAGE1_LAUNCHER_LOG_ROOT:-${CHECKPOINT_ROOT}/launcher_logs}"
LOCK_PATH="${STAGE1_LAUNCHER_LOCK_PATH:-${CHECKPOINT_ROOT}.4xh100.lock}"
RUN_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')_pid$$"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA}"
LIBERO_INIT_CHECKPOINT="${LIBERO_STAGE1_INIT_CHECKPOINT:-${REPO_ROOT}/playground/Checkpoints/var_stage1_pi05_libero_q99_pure_ae_e32/best_recon.ckpt}"

DEFAULT_ROBOCASA_DATA_ROOT="${REPO_ROOT}/playground/Datasets/RoboCasa-GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot"
LEGACY_ROBOCASA_DATA_ROOT="/root/feihong/starVLA/playground/Datasets/RoboCasa-GR1/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot"
if [[ ! -d "${DEFAULT_ROBOCASA_DATA_ROOT}" && -d "${LEGACY_ROBOCASA_DATA_ROOT}" ]]; then
  DEFAULT_ROBOCASA_DATA_ROOT="${LEGACY_ROBOCASA_DATA_ROOT}"
fi
ROBOCASA_DATA_ROOT="${ROBOCASA_DATA_ROOT:-${DEFAULT_ROBOCASA_DATA_ROOT}}"

ROBOCASA_INIT_IS_EXTERNAL=0
if [[ -n "${ROBOCASA_STAGE1_INIT_CHECKPOINT:-}" ]]; then
  ROBOCASA_INIT_IS_EXTERNAL=1
fi
ROBOCASA_INIT_CHECKPOINT="${ROBOCASA_STAGE1_INIT_CHECKPOINT:-${REPO_ROOT}/playground/Checkpoints/var_stage1_robocasa_gr1_pure_ae_e64/best_recon.ckpt}"
ROBOCASA_PURE_AE_OUTPUT_DIR="${ROBOCASA_PURE_AE_OUTPUT_DIR:-$(dirname "${ROBOCASA_INIT_CHECKPOINT}")}"

readonly LIBERO_WRAPPER="${REPO_ROOT}/examples/LIBERO/train_files/run_stage1_clean_supervision_ablation.sh"
readonly ROBOCASA_WRAPPER="${REPO_ROOT}/examples/Robocasa_tabletop/train_files/run_stage1_clean_supervision_ablation.sh"
readonly ROBOCASA_PURE_AE_CONFIG="${REPO_ROOT}/examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_pure_ae_e64.yaml"
readonly INSPECTOR="${REPO_ROOT}/scripts/stage1/inspect_clean_ablation_queue.py"
readonly ROBOCASA_REGISTRY="${REPO_ROOT}/examples/Robocasa_tabletop/train_files/data_registry/data_config.py"

declare -A JOB_ACTIONS=()
declare -a GPU_TOKENS=()
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a ACTIVE_LOGS=()

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] [stage1-4xh100] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

validate_boolean() {
  local name="$1"
  local value="$2"
  [[ "${value}" == 0 || "${value}" == 1 ]] || die "${name} must be 0 or 1, got ${value@Q}."
}

validate_positive_number() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$ ]] || die "${name} must be a positive finite number, got ${value@Q}."
  awk -v value="${value}" 'BEGIN { exit !(value > 0) }' || die "${name} must be > 0, got ${value@Q}."
}

resolve_python() {
  if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
    PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
  else
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")" || die "Python not found: ${PYTHON_BIN}"
  fi
}

initialize_gpu_tokens() {
  local -a requested=()
  local token
  if [[ -n "${ORIGINAL_CUDA_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -r -a requested <<<"${ORIGINAL_CUDA_VISIBLE_DEVICES}"
    for token in "${requested[@]}"; do
      token="${token//[[:space:]]/}"
      [[ -n "${token}" ]] && GPU_TOKENS+=("${token}")
    done
    if (( ${#GPU_TOKENS[@]} != 4 )); then
      if [[ "${DRY_RUN}" == 1 ]]; then
        log "DRY_RUN: inherited CUDA_VISIBLE_DEVICES does not expose exactly four tokens; previewing logical tokens 0,1,2,3."
        GPU_TOKENS=(0 1 2 3)
      else
        die "CUDA_VISIBLE_DEVICES must expose exactly four tokens, got ${ORIGINAL_CUDA_VISIBLE_DEVICES@Q}."
      fi
    fi
  else
    GPU_TOKENS=(0 1 2 3)
  fi
  [[ "${GPU_TOKENS[0]}" != "${GPU_TOKENS[1]}" &&
     "${GPU_TOKENS[0]}" != "${GPU_TOKENS[2]}" &&
     "${GPU_TOKENS[0]}" != "${GPU_TOKENS[3]}" &&
     "${GPU_TOKENS[1]}" != "${GPU_TOKENS[2]}" &&
     "${GPU_TOKENS[1]}" != "${GPU_TOKENS[3]}" &&
     "${GPU_TOKENS[2]}" != "${GPU_TOKENS[3]}" ]] || die "CUDA_VISIBLE_DEVICES contains duplicate tokens."
}

validate_gpu_layout() {
  initialize_gpu_tokens
  [[ "${DRY_RUN}" == 1 || "${SKIP_GPU_PREFLIGHT}" == 1 ]] && return 0
  local report
  report="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count()); [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')"
  local -a lines=()
  mapfile -t lines <<<"${report}"
  [[ "${lines[0]:-}" =~ ^[0-9]+$ ]] || die "Could not determine visible CUDA device count."
  (( lines[0] == 4 )) || die "This launcher requires exactly four visible GPUs, but found ${lines[0]}."
  local slot
  for slot in 0 1 2 3; do
    [[ "${lines[$((slot + 1))]:-}" == *H100* ]] || die "CUDA slot ${slot} is not an H100: ${lines[$((slot + 1))]:-unknown}."
    log "GPU slot ${slot}: token=${GPU_TOKENS[${slot}]}, model=${lines[$((slot + 1))]}."
  done
}

static_preflight() {
  validate_boolean DRY_RUN "${DRY_RUN}"
  validate_boolean RESUME_QUEUE "${RESUME_QUEUE}"
  validate_boolean SKIP_SMOKE "${SKIP_SMOKE}"
  validate_boolean SKIP_GPU_PREFLIGHT "${SKIP_GPU_PREFLIGHT}"
  [[ "${SKIP_GPU_PREFLIGHT}" == 0 || "${DRY_RUN}" == 1 ]] || die "SKIP_GPU_PREFLIGHT=1 is allowed only with DRY_RUN=1."
  [[ "${LAUNCH_STAGGER_SECONDS}" =~ ^[0-9]+$ ]] || die "LAUNCH_STAGGER_SECONDS must be a non-negative integer."
  validate_positive_number LIBERO_INTERMEDIATE_WEIGHT "${LIBERO_INTERMEDIATE_WEIGHT}"
  validate_positive_number ROBOCASA_INTERMEDIATE_WEIGHT "${ROBOCASA_INTERMEDIATE_WEIGHT}"
  resolve_python
  command -v setsid >/dev/null || die "Required command is missing: setsid"
  command -v tee >/dev/null || die "Required command is missing: tee"
  command -v flock >/dev/null || die "Required command is missing: flock"
  for path in "${LIBERO_WRAPPER}" "${ROBOCASA_WRAPPER}" "${ROBOCASA_PURE_AE_CONFIG}" "${INSPECTOR}" "${ROBOCASA_REGISTRY}"; do
    [[ -f "${path}" ]] || die "Required file is missing: ${path}"
  done
  [[ -d "${LIBERO_DATA_ROOT}" ]] || die "LIBERO data root is missing: ${LIBERO_DATA_ROOT}"
  [[ -f "${LIBERO_INIT_CHECKPOINT}" ]] || die "LIBERO e32 pure-AE checkpoint is missing: ${LIBERO_INIT_CHECKPOINT}"
  [[ -d "${ROBOCASA_DATA_ROOT}" ]] || die "RoboCasa 24-task short-name data root is missing: ${ROBOCASA_DATA_ROOT}"
  [[ "$("${PYTHON_BIN}" "${INSPECTOR}" robocasa-data-root --data-root "${ROBOCASA_DATA_ROOT}" --registry "${ROBOCASA_REGISTRY}")" == ready ]] || die "RoboCasa 24-task data-root validation failed."
  if [[ "${ROBOCASA_INIT_IS_EXTERNAL}" == 1 ]]; then
    [[ -f "${ROBOCASA_INIT_CHECKPOINT}" ]] || die "Explicit RoboCasa init checkpoint is missing: ${ROBOCASA_INIT_CHECKPOINT}"
  else
    [[ "${ROBOCASA_INIT_CHECKPOINT}" == "${ROBOCASA_PURE_AE_OUTPUT_DIR}/best_recon.ckpt" ]] || die "Generated RoboCasa init must be ROBOCASA_PURE_AE_OUTPUT_DIR/best_recon.ckpt."
  fi
  if [[ "${DRY_RUN}" != 1 ]]; then
    [[ "$("${PYTHON_BIN}" "${INSPECTOR}" init-checkpoint --benchmark libero --path "${LIBERO_INIT_CHECKPOINT}")" == ready ]] || die "LIBERO pure-AE validation failed."
    if [[ "${ROBOCASA_INIT_IS_EXTERNAL}" == 1 ]]; then
      [[ "$("${PYTHON_BIN}" "${INSPECTOR}" init-checkpoint --benchmark robocasa --path "${ROBOCASA_INIT_CHECKPOINT}")" == ready ]] || die "Explicit RoboCasa pure-AE validation failed."
    fi
  fi
  validate_gpu_layout
}

acquire_global_lock() {
  [[ "${DRY_RUN}" == 1 ]] && return 0
  mkdir -p "$(dirname "${LOCK_PATH}")"
  exec 9>>"${LOCK_PATH}"
  flock -n 9 || die "Another Stage1 4xH100 launcher or inherited training child holds ${LOCK_PATH}."
  export STAGE1_LAUNCH_LOCK_FD=9
  printf '%s pid=%s run=%s\n' "$(timestamp)" "$$" "${RUN_STAMP}" >&9
}

terminate_active_groups() {
  local signal="${1:-TERM}"
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    kill "-${signal}" -- "-${pid}" 2>/dev/null || true
  done
}

cleanup_active_groups() {
  terminate_active_groups TERM
  local attempt pid any_alive
  for attempt in {1..30}; do
    any_alive=0
    for pid in "${ACTIVE_PIDS[@]:-}"; do
      [[ -n "${pid}" ]] || continue
      if kill -0 -- "-${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done
  terminate_active_groups KILL
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    wait "${pid}" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_LOGS=()
}

cleanup_on_exit() {
  local rc="$1"
  trap - EXIT INT TERM
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    log "Launcher is exiting with active jobs; forwarding TERM, waiting up to 30s, then using KILL."
    cleanup_active_groups
  fi
  exit "${rc}"
}

on_signal() {
  log "Received a termination signal."
  exit 130
}
trap 'cleanup_on_exit $?' EXIT
trap on_signal INT TERM

inspect_pure_ae() {
  local command=("${PYTHON_BIN}" "${INSPECTOR}")
  [[ "${RESUME_QUEUE}" == 1 ]] && command+=(--resume-enabled)
  command+=(
    pure-ae
    --config "${ROBOCASA_PURE_AE_CONFIG}"
    --output-dir "${ROBOCASA_PURE_AE_OUTPUT_DIR}"
    --data-root "${ROBOCASA_DATA_ROOT}"
    --init-checkpoint "${ROBOCASA_INIT_CHECKPOINT}"
  )
  "${command[@]}"
}

run_single_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  log "command[${label}]: $(quote_command "$@")"
  if [[ "${DRY_RUN}" == 1 ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${log_file}")"
  setsid "$@" >"${log_file}" 2>&1 &
  local pid=$!
  ACTIVE_PIDS=("${pid}")
  ACTIVE_LABELS=("${label}")
  ACTIVE_LOGS=("${log_file}")
  log "START ${label}: pid=${pid}, log=${log_file}"
  local rc=0
  if wait "${pid}"; then
    log "DONE ${label}: exit=0"
  else
    rc=$?
    log "FAILED ${label}: exit=${rc}, log=${log_file}"
  fi
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_LOGS=()
  (( rc == 0 )) || return "${rc}"
}

ensure_robocasa_pure_ae() {
  if [[ "${ROBOCASA_INIT_IS_EXTERNAL}" == 1 ]]; then
    log "Using explicit shared RoboCasa initialization: ${ROBOCASA_INIT_CHECKPOINT}"
    return 0
  fi
  local action
  action="$(inspect_pure_ae)"
  if [[ "${action}" == ready ]]; then
    log "Verified the shared RoboCasa e64 pure-AE is complete (30/30)."
    return 0
  fi
  [[ "${action}" == fresh || "${action}" == resume ]] || die "Unexpected RoboCasa pure-AE action: ${action@Q}"
  local command=(
    env "CUDA_VISIBLE_DEVICES=${GPU_TOKENS[0]}" "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}"
    "${PYTHON_BIN}" starVLA/training/train_var_stage1.py
    --config_yaml "${ROBOCASA_PURE_AE_CONFIG}"
    --override "experiment.output_dir=${ROBOCASA_PURE_AE_OUTPUT_DIR}"
    --override "data.data_root_dir=${ROBOCASA_DATA_ROOT}"
  )
  if [[ "${action}" == resume ]]; then
    command+=(--override "train.resume_checkpoint=${ROBOCASA_PURE_AE_OUTPUT_DIR}/latest.ckpt")
  fi
  log "The common RoboCasa e64 pure-AE prerequisite is ${action}; it runs alone on GPU 0 before the 8-way matrix."
  run_single_logged "robocasa_pure_ae_e64_${action}" "${LOG_ROOT}/prerequisite/robocasa_pure_ae_e64_${RUN_STAMP}.log" "${command[@]}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    return 0
  fi
  local verify_command=(
    "${PYTHON_BIN}" "${INSPECTOR}" --resume-enabled pure-ae
    --config "${ROBOCASA_PURE_AE_CONFIG}"
    --output-dir "${ROBOCASA_PURE_AE_OUTPUT_DIR}"
    --data-root "${ROBOCASA_DATA_ROOT}"
    --init-checkpoint "${ROBOCASA_INIT_CHECKPOINT}"
  )
  [[ "$("${verify_command[@]}")" == ready ]] || die "RoboCasa pure-AE process exited zero but did not produce a verified 30-epoch artifact."
}

validate_stats_caches() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "DRY_RUN: statistics-cache validation is deferred until the pure-AE prerequisite exists."
    return 0
  fi
  [[ "$("${PYTHON_BIN}" "${INSPECTOR}" stats-caches --benchmark libero --data-root "${LIBERO_DATA_ROOT}")" == ready ]] || die "LIBERO statistics-cache validation failed."
  [[ "$("${PYTHON_BIN}" "${INSPECTOR}" stats-caches --benchmark robocasa --data-root "${ROBOCASA_DATA_ROOT}" --registry "${ROBOCASA_REGISTRY}")" == ready ]] || die "RoboCasa statistics-cache validation failed."
  log "Validated parseable abs-mode stats_gr00t caches for LIBERO 4/4 and RoboCasa 24/24."
}

job_key() {
  printf '%s:%s:%s' "$1" "$2" "$3"
}

job_gpu() {
  local benchmark="$1"
  local index="$2"
  local slot
  slot="$(job_slot "${benchmark}" "${index}")"
  printf '%s\n' "${GPU_TOKENS[${slot}]}"
}

job_slot() {
  local benchmark="$1"
  local index="$2"
  if [[ "${benchmark}" == libero ]]; then
    printf '%s\n' "${LIBERO_GPUS[${index}]}"
  else
    printf '%s\n' "${ROBOCASA_GPUS[${index}]}"
  fi
}

job_weight() {
  if [[ "$1" == libero ]]; then
    printf '%s\n' "${LIBERO_INTERMEDIATE_WEIGHT}"
  else
    printf '%s\n' "${ROBOCASA_INTERMEDIATE_WEIGHT}"
  fi
}

job_data_root() {
  if [[ "$1" == libero ]]; then
    printf '%s\n' "${LIBERO_DATA_ROOT}"
  else
    printf '%s\n' "${ROBOCASA_DATA_ROOT}"
  fi
}

job_init_checkpoint() {
  if [[ "$1" == libero ]]; then
    printf '%s\n' "${LIBERO_INIT_CHECKPOINT}"
  else
    printf '%s\n' "${ROBOCASA_INIT_CHECKPOINT}"
  fi
}

job_wrapper() {
  if [[ "$1" == libero ]]; then
    printf '%s\n' "${LIBERO_WRAPPER}"
  else
    printf '%s\n' "${ROBOCASA_WRAPPER}"
  fi
}

planned_job_output() {
  local benchmark="$1"
  local mode="$2"
  local method="$3"
  if [[ "${mode}" == smoke ]]; then
    printf '%s\n' "${CHECKPOINT_ROOT}/${benchmark}/smoke/${method}/seed_${SEED}"
  else
    printf '%s\n' "${CHECKPOINT_ROOT}/${benchmark}/${method}/seed_${SEED}"
  fi
}

inspect_job() {
  local benchmark="$1"
  local mode="$2"
  local method="$3"
  local force_resume_inspection="${4:-0}"
  if [[ "${DRY_RUN}" == 1 && "${benchmark}" == robocasa && ! -f "${ROBOCASA_INIT_CHECKPOINT}" ]]; then
    local pending_output
    pending_output="$(planned_job_output "${benchmark}" "${mode}" "${method}")"
    if [[ -L "${pending_output}" || ( -d "${pending_output}" && -n "$(find "${pending_output}" -mindepth 1 -maxdepth 1 -print -quit)" ) || ( -e "${pending_output}" && ! -d "${pending_output}" ) ]]; then
      die "DRY_RUN cannot safely infer a RoboCasa action before pure-AE exists because output is non-empty/foreign: ${pending_output}"
    fi
    printf 'fresh\n'
    return 0
  fi
  local command=("${PYTHON_BIN}" "${INSPECTOR}")
  if [[ "${RESUME_QUEUE}" == 1 || "${force_resume_inspection}" == 1 ]]; then
    command+=(--resume-enabled)
  fi
  command+=(
    ablation
    --benchmark "${benchmark}"
    --method "${method}"
    --mode "${mode}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --data-root "$(job_data_root "${benchmark}")"
    --init-checkpoint "$(job_init_checkpoint "${benchmark}")"
    --python-bin "${PYTHON_BIN}"
    --intermediate-weight "$(job_weight "${benchmark}")"
  )
  "${command[@]}"
}

plan_phase() {
  local mode="$1"
  local benchmark method action index
  log "Joint preflight for phase=${mode}: all 8 jobs are checked before any process starts."
  for index in "${!METHODS[@]}"; do
    method="${METHODS[${index}]}"
    for benchmark in libero robocasa; do
      action="$(inspect_job "${benchmark}" "${mode}" "${method}")"
      [[ "${action}" == fresh || "${action}" == resume || "${action}" == skip ]] || die "Unexpected action for ${benchmark}/${mode}/${method}: ${action@Q}"
      JOB_ACTIONS["$(job_key "${mode}" "${benchmark}" "${method}")"]="${action}"
      log "PLAN phase=${mode} benchmark=${benchmark} method=${method} slot=$(job_slot "${benchmark}" "${index}") cuda_token=$(job_gpu "${benchmark}" "${index}") action=${action}"
    done
  done
}

launch_job() {
  local benchmark="$1"
  local mode="$2"
  local method="$3"
  local gpu_token="$4"
  local gpu_slot="$5"
  local action="$6"
  [[ "${action}" != skip ]] || {
    log "SKIP verified completed job: ${benchmark}/${mode}/${method}"
    return 0
  }
  local command=(
    bash "$(job_wrapper "${benchmark}")"
    --methods "${method}"
    --seeds "${SEED}"
    --gpus "${gpu_token}"
    --mode "${mode}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --data-root "$(job_data_root "${benchmark}")"
    --init-checkpoint "$(job_init_checkpoint "${benchmark}")"
    --python-bin "${PYTHON_BIN}"
    --intermediate-weight "$(job_weight "${benchmark}")"
  )
  [[ "${mode}" == train ]] && command+=(--epochs "${EPOCHS}")
  [[ "${action}" == resume ]] && command+=(--resume)
  [[ "${DRY_RUN}" == 1 ]] && command+=(--dry-run)
  local label="${benchmark}_${mode}_${method}_gpu${gpu_slot}"
  local log_file="${LOG_ROOT}/${mode}/${label}_${RUN_STAMP}.log"
  log "command[${label}]: $(quote_command "${command[@]}")"
  if [[ "${DRY_RUN}" == 1 ]]; then
    return 0
  fi
  setsid "${command[@]}" >"${log_file}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${label}")
  ACTIVE_LOGS+=("${log_file}")
  log "START ${label}: pid=${ACTIVE_PIDS[-1]}, log=${log_file}"
}

wait_for_phase() {
  local mode="$1"
  local failed=0
  local index rc
  for index in "${!ACTIVE_PIDS[@]}"; do
    rc=0
    if wait "${ACTIVE_PIDS[${index}]}"; then
      log "DONE ${ACTIVE_LABELS[${index}]}: exit=0"
    else
      rc=$?
      failed=1
      log "FAILED ${ACTIVE_LABELS[${index}]}: exit=${rc}, log=${ACTIVE_LOGS[${index}]}"
    fi
    ACTIVE_PIDS[${index}]=""
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_LOGS=()
  (( failed == 0 )) || die "At least one ${mode} job failed; all peer jobs were allowed to finish. Inspect the per-job logs."
}

verify_phase_complete() {
  local mode="$1"
  local benchmark method action index
  for index in "${!METHODS[@]}"; do
    method="${METHODS[${index}]}"
    for benchmark in libero robocasa; do
      action="$(inspect_job "${benchmark}" "${mode}" "${method}" 1)"
      [[ "${action}" == skip ]] || die "Post-run verification did not find a complete job: ${benchmark}/${mode}/${method} (action=${action})."
    done
  done
  log "Verified all 8 ${mode} jobs have succeeded manifests, exact configs/init hashes, contiguous history, and valid final checkpoints."
}

run_phase() {
  local mode="$1"
  plan_phase "${mode}"
  if [[ "${DRY_RUN}" != 1 ]]; then
    mkdir -p "${LOG_ROOT}/${mode}"
  fi
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_LOGS=()
  local index method benchmark action gpu_token gpu_slot launched=0
  for index in "${!METHODS[@]}"; do
    method="${METHODS[${index}]}"
    for benchmark in libero robocasa; do
      action="${JOB_ACTIONS["$(job_key "${mode}" "${benchmark}" "${method}")"]}"
      gpu_slot="$(job_slot "${benchmark}" "${index}")"
      gpu_token="$(job_gpu "${benchmark}" "${index}")"
      launch_job "${benchmark}" "${mode}" "${method}" "${gpu_token}" "${gpu_slot}" "${action}"
      if [[ "${action}" != skip ]]; then
        launched=$((launched + 1))
        if [[ "${DRY_RUN}" != 1 && "${LAUNCH_STAGGER_SECONDS}" != 0 && "${launched}" != 8 ]]; then
          sleep "${LAUNCH_STAGGER_SECONDS}"
        fi
      fi
    done
  done
  log "Phase=${mode}: launched ${launched} processes; fresh runs use exactly two processes on each GPU."
  [[ "${DRY_RUN}" == 1 ]] && return 0
  wait_for_phase "${mode}"
  verify_phase_complete "${mode}"
}

main() {
  static_preflight
  acquire_global_lock
  if [[ "${DRY_RUN}" != 1 ]]; then
    mkdir -p "${LOG_ROOT}"
    local master_log="${LOG_ROOT}/launcher_${RUN_STAMP}.log"
    exec > >(tee -a "${master_log}") 2>&1
    log "Master log: ${master_log}"
  fi
  log "Mapping: GPU0=L-Base+R-MTR, GPU1=L-Full+R-DCT, GPU2=L-DCT+R-Full, GPU3=L-MTR+R-Base."
  log "Formal contract: 8 concurrent processes, seed=${SEED}, epochs=${EPOCHS}, batch=256/process, workers=8/process."
  ensure_robocasa_pure_ae
  validate_stats_caches
  if [[ "${SKIP_SMOKE}" == 0 ]]; then
    run_phase smoke
  else
    log "SKIP_SMOKE=1: skipping the recommended 8-process smoke phase."
  fi
  run_phase train
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "DRY_RUN complete: no training/output/log directories were created."
  else
    log "SUCCESS: all 8 formal Stage1 trainings completed and passed artifact verification."
  fi
}

main "$@"
