#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the established Jike launch command stable while the hardened manager
# lives in a separately reviewable implementation.
exec "${SCRIPT_DIR}/run_jike_stage2_all_checkpoints_8gpu_v2.sh" "$@"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DEFAULT_RUN_DIR="${STARVLA_DIR}/Checkpoints/qwen_var_productvq_g16_s1248_mtr_from_scratch_e99_nextscale_50k_gbs128_jike8h100"

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_dir]

Evaluate every steps_*_pytorch_model.pt checkpoint in numeric step order.
For each checkpoint, 8 independent LIBERO workers use 8 H100 GPUs in
parallel; the next checkpoint starts only after all 40 tasks are complete.

Default run_dir:
  ${DEFAULT_RUN_DIR}

Useful environment overrides:
  EVAL_GPUS="0 1 2 3 4 5 6 7"  GPU IDs (exactly eight; commas also work)
  EVAL_PORT_BASE=19100             First of eight worker ports
  TRIALS_PER_TASK=50               Rollouts per LIBERO task
  EVAL_SEED=7                      Evaluation seed
  EVAL_OUTPUT_ROOT=...             Output directory relative to run_dir
  MIN_STEP=0 MAX_STEP=999999999    Optional inclusive checkpoint filter
  CHECKPOINT_ORDER=asc             asc or desc
  SKIP_COMPLETED=1                 Resume from completed chunk logs
  REQUIRE_H100=1                   Reject requested GPUs that are not H100s
  DRY_RUN=1                        Validate/list work without starting eval

Example:
  nohup bash $0 > eval_all_ckpts_8gpu.nohup.log 2>&1 &
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
DRY_RUN="${DRY_RUN:-0}"

# Keep the established Stage2 evaluation contract. Object/goal use one-trial
# chunks because those suites have historically aborted mid-chunk; spatial/10
# use five-trial chunks to reduce model-server restart overhead.
SPATIAL_CHUNK_TRIALS="${SPATIAL_CHUNK_TRIALS:-5}"
OBJECT_CHUNK_TRIALS="${OBJECT_CHUNK_TRIALS:-1}"
GOAL_CHUNK_TRIALS="${GOAL_CHUNK_TRIALS:-1}"
LIBERO10_CHUNK_TRIALS="${LIBERO10_CHUNK_TRIALS:-5}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
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
USE_BF16="${EVAL_USE_BF16:-${USE_BF16:-1}}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-8}"

die() {
  echo "[all_ckpt_eval] ERROR: $*" >&2
  exit 1
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'"
}

resolve_executable() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    printf '%s\n' "${candidate}"
  else
    command -v "${candidate}"
  fi
}

[[ -d "${RUN_DIR}" ]] || die "run directory does not exist: ${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
elif [[ "${CHECKPOINT_DIR}" != /* ]]; then
  CHECKPOINT_DIR="${STARVLA_DIR}/${CHECKPOINT_DIR}"
fi
[[ -d "${CHECKPOINT_DIR}" ]] || die "checkpoint directory does not exist: ${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
[[ -x "${SCRIPT_DIR}/run_stage2_eval_chunked.sh" ]] || die "missing executable: ${SCRIPT_DIR}/run_stage2_eval_chunked.sh"
[[ -f "${SCRIPT_DIR}/summarize_libero_success.py" ]] || die "missing summarizer: ${SCRIPT_DIR}/summarize_libero_success.py"
[[ "${EVAL_OUTPUT_ROOT}" != /* ]] || die "EVAL_OUTPUT_ROOT must be relative to run_dir"

require_uint EVAL_PORT_BASE "${EVAL_PORT_BASE}"
require_uint TRIALS_PER_TASK "${TRIALS_PER_TASK}"
require_uint EVAL_SEED "${EVAL_SEED}"
require_uint MIN_STEP "${MIN_STEP}"
require_uint MAX_STEP "${MAX_STEP}"
require_uint SPATIAL_CHUNK_TRIALS "${SPATIAL_CHUNK_TRIALS}"
require_uint OBJECT_CHUNK_TRIALS "${OBJECT_CHUNK_TRIALS}"
require_uint GOAL_CHUNK_TRIALS "${GOAL_CHUNK_TRIALS}"
require_uint LIBERO10_CHUNK_TRIALS "${LIBERO10_CHUNK_TRIALS}"
require_uint MAX_RETRIES "${MAX_RETRIES}"
require_uint CHUNK_TIMEOUT_SECONDS "${CHUNK_TIMEOUT_SECONDS}"
require_uint EVAL_CPU_THREADS "${EVAL_CPU_THREADS}"
(( TRIALS_PER_TASK > 0 )) || die "TRIALS_PER_TASK must be > 0"
(( EVAL_PORT_BASE > 0 && EVAL_PORT_BASE + 7 <= 65535 )) || die "eight ports starting at EVAL_PORT_BASE must fit in 1..65535"
(( MIN_STEP <= MAX_STEP )) || die "MIN_STEP must be <= MAX_STEP"
for value in "${SPATIAL_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}" "${MAX_RETRIES}" "${CHUNK_TIMEOUT_SECONDS}" "${EVAL_CPU_THREADS}"; do
  (( value > 0 )) || die "chunk sizes, MAX_RETRIES, and CHUNK_TIMEOUT_SECONDS must be > 0"
done
[[ "${CHECKPOINT_ORDER}" == "asc" || "${CHECKPOINT_ORDER}" == "desc" ]] || die "CHECKPOINT_ORDER must be asc or desc"
[[ "${SKIP_COMPLETED}" == "0" || "${SKIP_COMPLETED}" == "1" ]] || die "SKIP_COMPLETED must be 0 or 1"
[[ "${REQUIRE_H100}" == "0" || "${REQUIRE_H100}" == "1" ]] || die "REQUIRE_H100 must be 0 or 1"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"

DEFAULT_PYTHON="/root/feihong/starVLA/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
STARVLA_PYTHON="$(resolve_executable "${STARVLA_PYTHON:-${DEFAULT_PYTHON}}")" || die "STARVLA_PYTHON is not executable"
LIBERO_PYTHON="$(resolve_executable "${LIBERO_PYTHON:-${DEFAULT_PYTHON}}")" || die "LIBERO_PYTHON is not executable"
PYTHON_PATH="$(dirname "${STARVLA_PYTHON}"):$(dirname "${LIBERO_PYTHON}"):${PATH}"
if [[ -d "${STARVLA_DIR}/third_party/LIBERO/libero" ]]; then
  DEFAULT_LIBERO_HOME="${STARVLA_DIR}/third_party/LIBERO"
else
  DEFAULT_LIBERO_HOME="/root/feihong/LIBERO"
fi
LIBERO_HOME="${LIBERO_HOME:-${DEFAULT_LIBERO_HOME}}"
[[ -d "${LIBERO_HOME}/libero" ]] || die "LIBERO_HOME is invalid: ${LIBERO_HOME} (expected ${LIBERO_HOME}/libero)"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || die "missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml"

gpu_spec="${EVAL_GPUS//,/ }"
read -r -a GPUS <<< "${gpu_spec}"
[[ "${#GPUS[@]}" -eq 8 ]] || die "EVAL_GPUS must contain exactly 8 GPU IDs; got ${#GPUS[@]}"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  require_uint "GPU ID" "${gpu}"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU ID in EVAL_GPUS: ${gpu}"
  SEEN_GPUS["${gpu}"]=1
done

declare -a CHECKPOINTS=()
while IFS= read -r checkpoint; do
  checkpoint_name="$(basename "${checkpoint}")"
  if [[ "${checkpoint_name}" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]]; then
    step=$((10#${BASH_REMATCH[1]}))
    if (( step >= MIN_STEP && step <= MAX_STEP )); then
      CHECKPOINTS+=("${checkpoint}")
    fi
  fi
done < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -print | sort -V)

[[ "${#CHECKPOINTS[@]}" -gt 0 ]] || die "no archived checkpoints found in ${CHECKPOINT_DIR} for steps ${MIN_STEP}..${MAX_STEP}"
if [[ "${CHECKPOINT_ORDER}" == "desc" ]]; then
  declare -a REVERSED_CHECKPOINTS=()
  for ((idx=${#CHECKPOINTS[@]} - 1; idx >= 0; idx--)); do
    REVERSED_CHECKPOINTS+=("${CHECKPOINTS[idx]}")
  done
  CHECKPOINTS=("${REVERSED_CHECKPOINTS[@]}")
fi

# Balance approximate simulator work, not just task count. libero_10 permits up
# to 520 steps/episode, versus spatial/object/goal at 220/280/300, so it gets
# three shards. Every task range is disjoint and the union is all 40 tasks.
JOB_LABELS=(spatial_t0_9 object_t0_4 object_t5_9 goal_t0_4 goal_t5_9 libero10_t0_2 libero10_t3_5 libero10_t6_9)
JOB_SUITES=(libero_spatial libero_object libero_object libero_goal libero_goal libero_10 libero_10 libero_10)
JOB_TASK_STARTS=(0 0 5 0 5 0 3 6)
JOB_TASK_COUNTS=(10 5 5 5 5 3 3 4)
JOB_CHUNKS=("${SPATIAL_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${OBJECT_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${GOAL_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}" "${LIBERO10_CHUNK_TRIALS}")
SUITES=(libero_spatial libero_object libero_goal libero_10)

# Jike injects torchrun-style variables into the job shell. These eval workers
# are eight independent single-GPU services, not an eight-rank process group.
# Leaving WORLD_SIZE/RANK set makes every service initialize torch.distributed
# and contend for the same injected MASTER_PORT before loading the model.
DISTRIBUTED_ENV_UNSET_ARGS=(
  -u WORLD_SIZE
  -u RANK
  -u LOCAL_RANK
  -u LOCAL_WORLD_SIZE
  -u GROUP_RANK
  -u ROLE_RANK
  -u ROLE_WORLD_SIZE
  -u MASTER_ADDR
  -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID
  -u TORCHELASTIC_RESTART_COUNT
  -u TORCHELASTIC_MAX_RESTARTS
  -u TORCHELASTIC_ERROR_FILE
)

print_configuration() {
  echo "[all_ckpt_eval] run_dir=${RUN_DIR}"
  echo "[all_ckpt_eval] checkpoint_dir=${CHECKPOINT_DIR}"
  echo "[all_ckpt_eval] checkpoints=${#CHECKPOINTS[@]} order=${CHECKPOINT_ORDER} steps=${MIN_STEP}..${MAX_STEP}"
  echo "[all_ckpt_eval] gpus=${GPUS[*]} ports=${EVAL_PORT_BASE}..$((EVAL_PORT_BASE + 7))"
  echo "[all_ckpt_eval] trials_per_task=${TRIALS_PER_TASK} seed=${EVAL_SEED} bf16=${USE_BF16}"
  echo "[all_ckpt_eval] libero_home=${LIBERO_HOME}"
  echo "[all_ckpt_eval] libero_config_path=${LIBERO_CONFIG_PATH}"
  echo "[all_ckpt_eval] output_root=${EVAL_OUTPUT_ROOT}"
  echo "[all_ckpt_eval] workers:"
  for idx in "${!JOB_LABELS[@]}"; do
    echo "  worker=${idx} gpu=${GPUS[idx]} port=$((EVAL_PORT_BASE + idx)) suite=${JOB_SUITES[idx]} tasks=${JOB_TASK_STARTS[idx]}..$((JOB_TASK_STARTS[idx] + JOB_TASK_COUNTS[idx] - 1)) chunk=${JOB_CHUNKS[idx]}"
  done
  echo "[all_ckpt_eval] checkpoint order:"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  $(basename "${checkpoint}")"
  done
}

print_configuration
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[all_ckpt_eval] DRY_RUN=1; no evaluation was started."
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found"
if ! PATH="${PYTHON_PATH}" PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
  "${LIBERO_PYTHON}" -c 'import libero' >/dev/null 2>&1; then
  die "LIBERO_PYTHON cannot import libero with LIBERO_HOME=${LIBERO_HOME}"
fi
for gpu in "${GPUS[@]}"; do
  gpu_name="$(nvidia-smi --id="${gpu}" --query-gpu=name --format=csv,noheader 2>/dev/null)" || die "GPU ${gpu} is not visible"
  gpu_name="${gpu_name//$'\n'/ }"
  if [[ "${REQUIRE_H100}" == "1" && "${gpu_name}" != *H100* ]]; then
    die "GPU ${gpu} is '${gpu_name}', not an H100 (set REQUIRE_H100=0 to override)"
  fi
  echo "[all_ckpt_eval] gpu=${gpu} name=${gpu_name}"
done

mkdir -p "${RUN_DIR}/${EVAL_OUTPUT_ROOT}"
DRIVER_LOG="${RUN_DIR}/${EVAL_OUTPUT_ROOT}/all_checkpoints_8gpu.log"
RESULTS_TSV="${RUN_DIR}/${EVAL_OUTPUT_ROOT}/all_checkpoint_results.tsv"

log() {
  local line
  printf -v line '[%(%Y-%m-%dT%H:%M:%SZ)T] %s' -1 "$*"
  echo "${line}"
  echo "${line}" >> "${DRIVER_LOG}"
}

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    log "stopping ${#ACTIVE_PIDS[@]} active evaluation workers"
    for pid in "${ACTIVE_PIDS[@]}"; do
      kill "${pid}" >/dev/null 2>&1
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
      wait "${pid}" >/dev/null 2>&1
    done
  fi
  if [[ "${status}" -ne 0 ]]; then
    log "exiting with status=${status}; completed chunks are preserved for resume"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

shopt -s nullglob
chunk_completed() {
  local log_path="$1"
  local expected_episodes="$2"
  [[ -f "${log_path}" ]] || return 1
  grep -q 'EVAL_CHUNK_OK' "${log_path}" || return 1
  grep -q 'Total success rate:' "${log_path}" || return 1
  grep -q "Total episodes: ${expected_episodes}" "${log_path}" || return 1
}

checkpoint_completed() {
  local log_root="$1"
  local checkpoint_base="$2"
  local suite log_path filename task_id trial_start trial_count trial
  local -a logs=()
  local -A covered=()

  for suite in "${SUITES[@]}"; do
    logs=("${log_root}/${suite}/${checkpoint_base}"_stage2_chunked_t*_r*_n*.log)
    for log_path in "${logs[@]}"; do
      filename="$(basename "${log_path}")"
      if [[ "${filename}" =~ _stage2_chunked_t([0-9]+)_r([0-9]+)_n([0-9]+)\.log$ ]]; then
        task_id=$((10#${BASH_REMATCH[1]}))
        trial_start=$((10#${BASH_REMATCH[2]}))
        trial_count=$((10#${BASH_REMATCH[3]}))
        (( task_id >= 0 && task_id < 10 && trial_count > 0 )) || continue
        chunk_completed "${log_path}" "${trial_count}" || continue
        for ((trial=trial_start; trial<trial_start + trial_count; trial++)); do
          if (( trial >= 0 && trial < TRIALS_PER_TASK )); then
            covered["${suite}:${task_id}:${trial}"]=1
          fi
        done
      fi
    done
  done

  for suite in "${SUITES[@]}"; do
    for ((task_id=0; task_id<10; task_id++)); do
      for ((trial=0; trial<TRIALS_PER_TASK; trial++)); do
        [[ -n "${covered[${suite}:${task_id}:${trial}]:-}" ]] || return 1
      done
    done
  done
  return 0
}

write_summary() {
  local log_root="$1"
  local summary_path="$2"
  PATH="${PYTHON_PATH}" "${LIBERO_PYTHON}" "${SCRIPT_DIR}/summarize_libero_success.py" \
    "${log_root}" --chunked --require-ok-marker | tee "${summary_path}"
}

rebuild_results_table() {
  local tmp_path="${RESULTS_TSV}.tmp.$$"
  local checkpoint checkpoint_base step summary_path overall_line overall_value
  printf 'step\tcheckpoint\toverall_40_task_mean\tsummary\n' > "${tmp_path}"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_base="$(basename "${checkpoint}" .pt)"
    step="${checkpoint_base#steps_}"
    step="${step%_pytorch_model}"
    summary_path="${RUN_DIR}/${EVAL_OUTPUT_ROOT}/${checkpoint_base}/logs/libero_40task_summary.txt"
    [[ -f "${summary_path}" ]] || continue
    overall_line="$(grep '^overall_40_task_mean:' "${summary_path}" || true)"
    overall_value="${overall_line##*, }"
    printf '%s\t%s\t%s\t%s\n' "${step}" "${checkpoint_base}" "${overall_value}" "${summary_path}" >> "${tmp_path}"
  done
  mv "${tmp_path}" "${RESULTS_TSV}"
}

cd "${STARVLA_DIR}"
log "starting ${#CHECKPOINTS[@]} checkpoints with 8 GPUs"

for checkpoint_idx in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[checkpoint_idx]}"
  checkpoint_base="$(basename "${checkpoint}" .pt)"
  checkpoint_output_root="${EVAL_OUTPUT_ROOT}/${checkpoint_base}"
  log_root="${RUN_DIR}/${checkpoint_output_root}/logs"
  summary_path="${log_root}/libero_40task_summary.txt"
  mkdir -p "${log_root}"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && checkpoint_completed "${log_root}" "${checkpoint_base}"; then
    log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] skip completed ${checkpoint_base}"
    write_summary "${log_root}" "${summary_path}"
    rebuild_results_table
    continue
  fi

  log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] evaluating ${checkpoint_base}"
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  for job_idx in "${!JOB_LABELS[@]}"; do
    label="${JOB_LABELS[job_idx]}"
    suite="${JOB_SUITES[job_idx]}"
    gpu="${GPUS[job_idx]}"
    port=$((EVAL_PORT_BASE + job_idx))
    task_start="${JOB_TASK_STARTS[job_idx]}"
    task_count="${JOB_TASK_COUNTS[job_idx]}"
    chunk_trials="${JOB_CHUNKS[job_idx]}"
    launch_log="${RUN_DIR}/${checkpoint_output_root}/launch_${job_idx}_${label}.log"

    log "launch ${checkpoint_base} worker=${label} gpu=${gpu} port=${port} tasks=${task_start}..$((task_start + task_count - 1))"
    env "${DISTRIBUTED_ENV_UNSET_ARGS[@]}" \
      PATH="${PYTHON_PATH}" \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS="${EVAL_CPU_THREADS}" \
      MKL_NUM_THREADS="${EVAL_CPU_THREADS}" \
      STARVLA_PYTHON="${STARVLA_PYTHON}" \
      LIBERO_PYTHON="${LIBERO_PYTHON}" \
      LIBERO_HOME="${LIBERO_HOME}" \
      LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
      TASK_SUITES_OVERRIDE="${suite}" \
      TASK_START="${task_start}" \
      TASK_COUNT="${task_count}" \
      EVAL_OUTPUT_ROOT="${checkpoint_output_root}" \
      TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
      CHUNK_TRIALS="${chunk_trials}" \
      MAX_RETRIES="${MAX_RETRIES}" \
      CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS}" \
      UNNORM_KEY="${UNNORM_KEY}" \
      SAVE_VIDEOS="${SAVE_VIDEOS}" \
      SAVE_ONLY_SUCCESS_VIDEOS="${SAVE_ONLY_SUCCESS_VIDEOS}" \
      MAX_SUCCESS_VIDEOS_PER_TASK="${MAX_SUCCESS_VIDEOS_PER_TASK}" \
      IMAGE_VIEWS="${IMAGE_VIEWS}" \
      POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}" \
      CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS}" \
      VALIDATE_INPUTS="${VALIDATE_INPUTS}" \
      STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT}" \
      USE_BF16="${USE_BF16}" \
      SEED="${EVAL_SEED}" \
      bash "${SCRIPT_DIR}/run_stage2_eval_chunked.sh" "${checkpoint}" "${gpu}" "${port}" \
      > "${launch_log}" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_LABELS+=("${label}")
  done

  worker_failed=0
  for job_idx in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[job_idx]}"
    label="${ACTIVE_LABELS[job_idx]}"
    if wait "${pid}"; then
      log "worker completed ${checkpoint_base} worker=${label}"
    else
      status=$?
      log "worker failed ${checkpoint_base} worker=${label} status=${status}"
      worker_failed=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()

  [[ "${worker_failed}" -eq 0 ]] || die "at least one worker failed for ${checkpoint_base}; rerun the same command to resume"
  checkpoint_completed "${log_root}" "${checkpoint_base}" || die "workers exited but ${checkpoint_base} does not contain all 40 x ${TRIALS_PER_TASK} completed trials"

  write_summary "${log_root}" "${summary_path}"
  rebuild_results_table
  log "[$((checkpoint_idx + 1))/${#CHECKPOINTS[@]}] completed ${checkpoint_base}; summary=${summary_path}"
done

rebuild_results_table
log "all checkpoints completed; results=${RESULTS_TSV}"
ACTIVE_PIDS=()
trap - EXIT INT TERM
exit 0
