#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the five retained checkpoints from the RoboCasa MTR-e80 Stage2 run.
# The evaluation contract intentionally matches the verified 55.08% baseline:
# 24 tasks x 50 episodes, one 50-episode chunk per task, FP32, no video,
# n_envs=1, horizon=720, and n_action_steps=12.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DEFAULT_RUN_DIR="${REPO_DIR}/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_mtr_stage1_e80_100k_lr1e4_warmup5000_gbs512_jike8h100"

usage() {
  cat <<EOF
Usage: $(basename "$0") [run_dir]

Evaluate the retained RoboCasa Stage2 checkpoints on four H100 GPUs. For each
checkpoint, four independent single-GPU workers evaluate six of the 24 tasks;
the next checkpoint begins only after the current checkpoint is complete.

Default checkpoint order (86k first for the direct 55.08% baseline comparison):
  86000 80000 82000 90000 100000

Useful overrides:
  EVAL_GPUS="0 1 2 3"       exactly four GPU indices (commas also accepted)
  CHECKPOINT_STEPS="..."    explicit retained checkpoint steps
  BASE_PORT=22000            worker ports span BASE_PORT..BASE_PORT+323
  MAX_PASSES=20              passes over missing/failed task chunks
  CHUNK_MAX_RETRIES=3        retries per task within one pass
  STATUS_INTERVAL_SECONDS=120
  REQUIRE_H100=1 REQUIRE_IDLE_GPUS=1
  MAX_GPU_MEMORY_USED_MIB=2048 MAX_GPU_UTILIZATION=20
  DRY_RUN=1                  validate and print the plan without launching

The run is resumable. A rerun skips only complete 50-episode chunks whose
COMPLETE.json records the exact same checkpoint path.

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
  echo "[robocasa-4gpu-eval] ERROR: $*" >&2
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
    # Keep the venv entrypoint symlink intact so pyvenv.cfg is discovered.
    printf '%s/%s\n' "$(cd "$(dirname "${candidate}")" && pwd)" "$(basename "${candidate}")"
  else
    command -v "${candidate}"
  fi
}

RUN_DIR="${1:-${RUN_DIR:-${DEFAULT_RUN_DIR}}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-86000 80000 82000 90000 100000}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-22000}"
MAX_PASSES="${MAX_PASSES:-20}"
CHUNK_MAX_RETRIES="${CHUNK_MAX_RETRIES:-3}"
WORKER_STAGGER_SECONDS="${WORKER_STAGGER_SECONDS:-8}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-120}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SERVER_IDLE_TIMEOUT="${SERVER_IDLE_TIMEOUT:-1800}"
SIM_TIMEOUT="${SIM_TIMEOUT:-10800}"
REQUIRE_H100="${REQUIRE_H100:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-2048}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
CHECKPOINT_STABILITY_SECONDS="${CHECKPOINT_STABILITY_SECONDS:-2}"
DRY_RUN="${DRY_RUN:-0}"

# These are deliberately fixed to the verified full-benchmark protocol.
TASKS_PRESET="gr1_24"
TRIALS_PER_TASK=50
CHUNK_EPISODES=50
N_ENVS=1
MAX_EPISODE_STEPS=720
N_ACTION_STEPS=12
OUTPUT_SUFFIX="gr1_24_50eps_chunk50_robust"

[[ -d "${REPO_DIR}" ]] || die "repository does not exist: ${REPO_DIR}"
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
[[ -d "${RUN_DIR}" ]] || die "run directory does not exist: ${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
[[ -d "${CHECKPOINT_DIR}" ]] || die "checkpoint directory does not exist: ${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
[[ "${CHECKPOINT_DIR}" == "${RUN_DIR}/checkpoints" ]] || \
  die "CHECKPOINT_DIR must be ${RUN_DIR}/checkpoints, got ${CHECKPOINT_DIR}"

CHUNK_RUNNER="${SCRIPT_DIR}/run_robocasa_stage2_eval_chunked.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_robocasa_success.py"
[[ -f "${CHUNK_RUNNER}" ]] || die "missing chunk runner: ${CHUNK_RUNNER}"
[[ -f "${SUMMARIZER}" ]] || die "missing summarizer: ${SUMMARIZER}"

STARVLA_PYTHON="$(resolve_executable "${STARVLA_PYTHON:-/root/feihong/starVLA/.venv/bin/python}")" || \
  die "STARVLA_PYTHON is not executable"
ROBOCASA_PYTHON="$(resolve_executable "${ROBOCASA_PYTHON:-/root/feihong/starVLA/.venv-robocasa-eval/bin/python}")" || \
  die "ROBOCASA_PYTHON is not executable"

for command_name in flock nvidia-smi setsid stat tee; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

for pair in \
  "BASE_PORT:${BASE_PORT}" "MAX_PASSES:${MAX_PASSES}" \
  "CHUNK_MAX_RETRIES:${CHUNK_MAX_RETRIES}" "WORKER_STAGGER_SECONDS:${WORKER_STAGGER_SECONDS}" \
  "STATUS_INTERVAL_SECONDS:${STATUS_INTERVAL_SECONDS}" "SERVER_READY_TIMEOUT:${SERVER_READY_TIMEOUT}" \
  "SERVER_IDLE_TIMEOUT:${SERVER_IDLE_TIMEOUT}" "SIM_TIMEOUT:${SIM_TIMEOUT}" \
  "MAX_GPU_MEMORY_USED_MIB:${MAX_GPU_MEMORY_USED_MIB}" \
  "MAX_GPU_UTILIZATION:${MAX_GPU_UTILIZATION}" \
  "CHECKPOINT_STABILITY_SECONDS:${CHECKPOINT_STABILITY_SECONDS}"; do
  require_uint "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "REQUIRE_H100:${REQUIRE_H100}" "REQUIRE_IDLE_GPUS:${REQUIRE_IDLE_GPUS}" \
  "DRY_RUN:${DRY_RUN}"; do
  require_bool "${pair%%:*}" "${pair#*:}"
done
(( BASE_PORT > 0 && BASE_PORT + 323 <= 65535 )) || die "BASE_PORT must leave room through BASE_PORT+323"
(( MAX_PASSES > 0 && CHUNK_MAX_RETRIES > 0 )) || die "retry counts must be positive"
(( STATUS_INTERVAL_SECONDS > 0 && SERVER_READY_TIMEOUT > 0 )) || die "status/startup timeouts must be positive"
(( SERVER_IDLE_TIMEOUT > 0 && SIM_TIMEOUT > 0 )) || die "server/simulator timeouts must be positive"
(( CHECKPOINT_STABILITY_SECONDS > 0 )) || die "CHECKPOINT_STABILITY_SECONDS must be positive"

gpu_spec="${EVAL_GPUS//,/ }"
read -r -a GPUS <<< "${gpu_spec}"
WORKER_COUNT="${#GPUS[@]}"
(( WORKER_COUNT == 4 )) || die "EVAL_GPUS must contain exactly four IDs; got ${WORKER_COUNT}: ${GPUS[*]}"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  require_uint "GPU ID" "${gpu}"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU ID: ${gpu}"
  SEEN_GPUS["${gpu}"]=1
done

checkpoint_step_spec="${CHECKPOINT_STEPS//,/ }"
read -r -a REQUESTED_STEPS <<< "${checkpoint_step_spec}"
(( ${#REQUESTED_STEPS[@]} > 0 )) || die "CHECKPOINT_STEPS is empty"
declare -A SEEN_STEPS=()
declare -a CHECKPOINTS=()
declare -a NORMALIZED_STEPS=()
for step_text in "${REQUESTED_STEPS[@]}"; do
  require_uint "checkpoint step" "${step_text}"
  step=$((10#${step_text}))
  (( step > 0 )) || die "checkpoint steps must be positive"
  [[ -z "${SEEN_STEPS[${step}]:-}" ]] || die "duplicate checkpoint step: ${step}"
  SEEN_STEPS["${step}"]=1
  checkpoint="${CHECKPOINT_DIR}/steps_${step}_pytorch_model.pt"
  [[ -f "${checkpoint}" ]] || die "missing requested checkpoint: ${checkpoint}"
  [[ -s "${checkpoint}" ]] || die "requested checkpoint is empty: ${checkpoint}"
  CHECKPOINTS+=("$(readlink -f "${checkpoint}")")
  NORMALIZED_STEPS+=("${step}")
done

LOG_ROOT="${RUN_DIR}/robocasa_eval_queue_logs/mtr_e80_4xh100_gr1_24_50eps"
MASTER_LOG="${MASTER_LOG:-${LOG_ROOT}/all_checkpoints_4xh100.log}"
RESULTS_TSV="${LOG_ROOT}/all_checkpoint_results.tsv"
LOCK_PATH="${LOG_ROOT}/.all_checkpoints_4xh100.lock"
ROBOCASA_MJCF_TMPDIR="${ROBOCASA_MJCF_TMPDIR:-/tmp/robocasa_mjcf_tmp_mtr_e80_4xh100}"

print_configuration() {
  echo "[robocasa-4gpu-eval] repo=${REPO_DIR}"
  echo "[robocasa-4gpu-eval] run_dir=${RUN_DIR}"
  echo "[robocasa-4gpu-eval] starvla_python=${STARVLA_PYTHON}"
  echo "[robocasa-4gpu-eval] robocasa_python=${ROBOCASA_PYTHON}"
  echo "[robocasa-4gpu-eval] gpus=${GPUS[*]} workers=${WORKER_COUNT} base_port=${BASE_PORT}"
  echo "[robocasa-4gpu-eval] protocol=${TASKS_PRESET} x ${TRIALS_PER_TASK} episodes, chunk${CHUNK_EPISODES}, FP32, no-video, n_envs=${N_ENVS}, horizon=${MAX_EPISODE_STEPS}, action_steps=${N_ACTION_STEPS}"
  echo "[robocasa-4gpu-eval] checkpoint_count=${#CHECKPOINTS[@]} order=${NORMALIZED_STEPS[*]}"
  echo "[robocasa-4gpu-eval] scheduling=one checkpoint at a time; four GPUs evaluate six tasks each"
  echo "[robocasa-4gpu-eval] output_pattern=${RUN_DIR}/robocasa_eval/steps_<step>_pytorch_model_${OUTPUT_SUFFIX}"
  echo "[robocasa-4gpu-eval] master_log=${MASTER_LOG}"
  local index
  for index in "${!CHECKPOINTS[@]}"; do
    echo "  [$((index + 1))/${#CHECKPOINTS[@]}] ${CHECKPOINTS[index]}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  print_configuration
  echo "[robocasa-4gpu-eval] DRY_RUN=1; no output directories or evaluation processes were created."
  exit 0
fi

mkdir -p "${LOG_ROOT}" "${ROBOCASA_MJCF_TMPDIR}"
exec 9>"${LOCK_PATH}"
flock -n 9 || die "another 4-GPU sweep already holds ${LOCK_PATH}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] [robocasa-4gpu-eval] $*"
}

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
MONITOR_PID=""

stop_progress_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup_workers() {
  local pid
  stop_progress_monitor
  for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    [[ -n "${pid}" ]] && kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    [[ -n "${pid}" ]] && kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  cleanup_workers
  log "SWEEP_EXIT status=${status}; completed chunks remain resumable"
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# JiKe may inject torchrun variables into the job shell. These workers are
# independent single-GPU processes and must not initialize a process group.
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK NODE_RANK GROUP_RANK \
  ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID \
  TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_ERROR_FILE || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HOME="${HF_HOME:-/root/feihong/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/feihong/.cache/torch}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

DISTRIBUTED_ENV_UNSET_ARGS=(
  -u WORLD_SIZE -u LOCAL_WORLD_SIZE -u RANK -u LOCAL_RANK -u NODE_RANK
  -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT
  -u TORCHELASTIC_MAX_RESTARTS -u TORCHELASTIC_ERROR_FILE
)

validate_gpus() {
  local gpu info name memory_used utilization
  for gpu in "${GPUS[@]}"; do
    info="$(nvidia-smi --id="${gpu}" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || \
      die "GPU ${gpu} is not visible"
    IFS=',' read -r name memory_used utilization <<< "${info}"
    name="${name#${name%%[![:space:]]*}}"
    name="${name%${name##*[![:space:]]}}"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    require_uint "GPU ${gpu} memory.used" "${memory_used}"
    require_uint "GPU ${gpu} utilization" "${utilization}"
    if [[ "${REQUIRE_H100}" == "1" && "${name}" != *H100* ]]; then
      die "GPU ${gpu} is '${name}', not an H100 (set REQUIRE_H100=0 to override)"
    fi
    if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]] && \
       (( memory_used > MAX_GPU_MEMORY_USED_MIB || utilization > MAX_GPU_UTILIZATION )); then
      die "GPU ${gpu} is not idle: memory=${memory_used}MiB utilization=${utilization}%"
    fi
    log "gpu=${gpu} name=${name} memory_used=${memory_used}MiB utilization=${utilization}%"
  done
}

validate_checkpoint_stability() {
  local checkpoint before after
  for checkpoint in "${CHECKPOINTS[@]}"; do
    before="$(stat -c '%s:%Y' "${checkpoint}")"
    sleep "${CHECKPOINT_STABILITY_SECONDS}"
    after="$(stat -c '%s:%Y' "${checkpoint}")"
    [[ "${before}" == "${after}" ]] || die "checkpoint is still changing: ${checkpoint}"
    log "checkpoint_stable path=${checkpoint} bytes=${before%%:*}"
  done
}

summarize_checkpoint() {
  local output_root="$1" require_complete="${2:-0}"
  local summary_path="${output_root}/summary.txt"
  local extra=()
  [[ -d "${output_root}" ]] || return 1
  if [[ "${require_complete}" == "1" ]]; then
    summary_path="${output_root}/summary.require_complete.txt"
    extra+=(--require-complete)
  fi
  PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" "${STARVLA_PYTHON}" \
    "${SUMMARIZER}" \
    "${output_root}" \
    --tasks-preset "${TASKS_PRESET}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --chunk-episodes "${CHUNK_EPISODES}" \
    --expected-episodes-per-chunk "${CHUNK_EPISODES}" \
    "${extra[@]}" > "${summary_path}" 2>&1
}

assert_no_foreign_results() {
  local output_root="$1" checkpoint="$2"
  [[ -d "${output_root}" ]] || return 0
  "${STARVLA_PYTHON}" -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = Path(sys.argv[2]).resolve()
foreign = []
for path in sorted(root.glob("*/r*_n*/COMPLETE.json")):
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        foreign.append(f"{path}: unreadable ({exc})")
        continue
    recorded = item.get("ckpt")
    if not recorded or Path(recorded).resolve() != expected:
        foreign.append(f"{path}: ckpt={recorded!r}")
if foreign:
    print("Refusing to mix results from another checkpoint:", file=sys.stderr)
    print("\n".join(foreign[:20]), file=sys.stderr)
raise SystemExit(1 if foreign else 0)
' "${output_root}" "${checkpoint}"
}

checkpoint_complete() {
  local output_root="$1" checkpoint="$2"
  summarize_checkpoint "${output_root}" 1 || return 1
  "${STARVLA_PYTHON}" -c '
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
root = Path(sys.argv[2])
expected = Path(sys.argv[3]).resolve()
with summary_path.open(encoding="utf-8") as handle:
    data = json.load(handle)
tasks = data.get("tasks") or []
complete_files = sorted(root.glob("*/r000_n050/COMPLETE.json"))
identity_ok = len(complete_files) == 24
for path in complete_files:
    with path.open(encoding="utf-8") as handle:
        item = json.load(handle)
    recorded = item.get("ckpt")
    if not recorded or Path(recorded).resolve() != expected:
        identity_ok = False
        break
complete = (
    data.get("total_episodes") == 1200
    and len(tasks) == 24
    and not data.get("incomplete_chunks")
    and identity_ok
    and all(task.get("complete_chunks") == 1 and task.get("episodes") == 50 for task in tasks)
)
raise SystemExit(0 if complete else 1)
' "${output_root}/summary.json" "${output_root}" "${checkpoint}"
}

print_checkpoint_progress() {
  local output_root="$1" checkpoint="$2" stem="$3" pass="$4"
  "${STARVLA_PYTHON}" -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = Path(sys.argv[2]).resolve()
tasks = episodes = successes = 0
for path in sorted(root.glob("*/r000_n050/COMPLETE.json")):
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    recorded = item.get("ckpt")
    rows = item.get("episodes") or []
    if item.get("status") != "complete" or len(rows) != 50:
        continue
    if not recorded or Path(recorded).resolve() != expected:
        continue
    tasks += 1
    episodes += len(rows)
    successes += sum(row.get("success") is True for row in rows)
rate = 100.0 * successes / episodes if episodes else 0.0
print(f"tasks={tasks}/24 episodes={episodes}/1200 successes={successes} partial_success={rate:.2f}%")
' "${output_root}" "${checkpoint}" | while IFS= read -r status; do
    log "PROGRESS ckpt=${stem} pass=${pass} ${status}"
  done
}

ensure_eval_ports_free() {
  "${STARVLA_PYTHON}" -c '
import socket
import sys

base = int(sys.argv[1])
workers = int(sys.argv[2])
ports = [base + (task % workers) * 100 + task for task in range(24)]
busy = []
for port in ports:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.05):
            busy.append(port)
    except OSError:
        pass
if busy:
    print("Eval ports already have listeners: " + ", ".join(map(str, busy)), file=sys.stderr)
raise SystemExit(1 if busy else 0)
' "${BASE_PORT}" "${WORKER_COUNT}"
}

start_progress_monitor() {
  local output_root="$1" checkpoint="$2" stem="$3" pass="$4"
  (
    trap 'exit 0' INT TERM HUP
    while true; do
      sleep "${STATUS_INTERVAL_SECONDS}"
      print_checkpoint_progress "${output_root}" "${checkpoint}" "${stem}" "${pass}" || true
    done
  ) &
  MONITOR_PID=$!
}

run_checkpoint_pass() {
  local checkpoint="$1" output_root="$2" stem="$3" pass="$4"
  local worker gpu worker_port log_path pid rc worker_tmp
  local pass_failed=0
  local worker_log_root="${LOG_ROOT}/worker_logs/${stem}/pass_$(printf '%02d' "${pass}")"

  mkdir -p "${worker_log_root}"
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()

  for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu="${GPUS[worker]}"
    worker_port=$((BASE_PORT + worker * 100))
    log_path="${worker_log_root}/worker_${worker}_gpu_${gpu}.log"
    worker_tmp="${ROBOCASA_MJCF_TMPDIR}/gpu_${gpu}"
    mkdir -p "${worker_tmp}"
    log "LAUNCH ckpt=${stem} pass=${pass} worker=${worker}/${WORKER_COUNT} gpu=${gpu} port_base=${worker_port} log=${log_path}"

    setsid env \
      "${DISTRIBUTED_ENV_UNSET_ARGS[@]}" \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
      ROBOCASA_MJCF_TMPDIR="${worker_tmp}" \
      "${STARVLA_PYTHON}" -u \
      "${CHUNK_RUNNER}" \
      "${checkpoint}" \
      --output-root "${output_root}" \
      --repo-root "${REPO_DIR}" \
      --starvla-python "${STARVLA_PYTHON}" \
      --robocasa-python "${ROBOCASA_PYTHON}" \
      --tasks-preset "${TASKS_PRESET}" \
      --trials-per-task "${TRIALS_PER_TASK}" \
      --chunk-episodes "${CHUNK_EPISODES}" \
      --worker-count "${WORKER_COUNT}" \
      --worker-index "${worker}" \
      --gpu "${gpu}" \
      --base-port "${BASE_PORT}" \
      --max-retries "${CHUNK_MAX_RETRIES}" \
      --server-ready-timeout "${SERVER_READY_TIMEOUT}" \
      --server-idle-timeout "${SERVER_IDLE_TIMEOUT}" \
      --sim-timeout "${SIM_TIMEOUT}" \
      --n-envs "${N_ENVS}" \
      --max-episode-steps "${MAX_EPISODE_STEPS}" \
      --n-action-steps "${N_ACTION_STEPS}" \
      --no-video > "${log_path}" 2>&1 &
    pid=$!
    ACTIVE_PIDS+=("${pid}")
    ACTIVE_LABELS+=("worker=${worker} gpu=${gpu}")

    if (( worker + 1 < WORKER_COUNT && WORKER_STAGGER_SECONDS > 0 )); then
      sleep "${WORKER_STAGGER_SECONDS}"
    fi
  done

  print_checkpoint_progress "${output_root}" "${checkpoint}" "${stem}" "${pass}" || true
  start_progress_monitor "${output_root}" "${checkpoint}" "${stem}" "${pass}"

  for worker in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[worker]}"
    if wait "${pid}"; then
      log "WORKER_OK ckpt=${stem} pass=${pass} ${ACTIVE_LABELS[worker]}"
    else
      rc=$?
      pass_failed=1
      log "WORKER_FAILED rc=${rc} ckpt=${stem} pass=${pass} ${ACTIVE_LABELS[worker]}"
    fi
    # A worker should clean up its policy server; terminate any orphan in the
    # process group before a port or GPU is reused.
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done

  stop_progress_monitor
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  print_checkpoint_progress "${output_root}" "${checkpoint}" "${stem}" "${pass}" || true
  return "${pass_failed}"
}

write_results_table() {
  "${STARVLA_PYTHON}" - "${RESULTS_TSV}" "${OUTPUT_SUFFIX}" "${CHECKPOINTS[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

target = Path(sys.argv[1])
suffix = sys.argv[2]
checkpoints = [Path(item) for item in sys.argv[3:]]
lines = ["step\ttasks\tepisodes\tsuccesses\tsuccess_rate\tcheckpoint\tsummary"]
for checkpoint in checkpoints:
    match = re.fullmatch(r"steps_(\d+)_pytorch_model\.pt", checkpoint.name)
    step = match.group(1) if match else checkpoint.stem
    output = checkpoint.parent.parent / "robocasa_eval" / f"{checkpoint.stem}_{suffix}"
    summary_path = output / "summary.json"
    tasks = episodes = successes = 0
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            tasks = sum(task.get("complete_chunks") == 1 and task.get("episodes") == 50 for task in summary.get("tasks", []))
            episodes = int(summary.get("total_episodes", 0))
            successes = int(summary.get("total_successes", 0))
        except Exception:
            pass
    rate = successes / episodes if episodes else 0.0
    lines.append(
        f"{step}\t{tasks}/24\t{episodes}/1200\t{successes}\t{rate:.6f}\t{checkpoint}\t{summary_path}"
    )
tmp = target.with_suffix(target.suffix + ".tmp")
tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
tmp.replace(target)
print("\n".join(lines))
PY
}

log "SWEEP_START"
print_configuration
validate_gpus
validate_checkpoint_stability

for index in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[index]}"
  step="${NORMALIZED_STEPS[index]}"
  stem="$(basename "${checkpoint}" .pt)"
  output_root="${RUN_DIR}/robocasa_eval/${stem}_${OUTPUT_SUFFIX}"
  mkdir -p "${output_root}"

  assert_no_foreign_results "${output_root}" "${checkpoint}"
  if checkpoint_complete "${output_root}" "${checkpoint}"; then
    log "SKIP_COMPLETE step=${step} checkpoint=${checkpoint} summary=${output_root}/summary.txt"
    continue
  fi

  log "CHECKPOINT_START index=$((index + 1))/${#CHECKPOINTS[@]} step=${step} checkpoint=${checkpoint}"
  completed=0
  for pass in $(seq 1 "${MAX_PASSES}"); do
    ensure_eval_ports_free || die "required policy-server ports are busy before step=${step} pass=${pass}"
    run_checkpoint_pass "${checkpoint}" "${output_root}" "${stem}" "${pass}" || true
    summarize_checkpoint "${output_root}" 0 || true
    if checkpoint_complete "${output_root}" "${checkpoint}"; then
      completed=1
      log "CHECKPOINT_DONE step=${step} checkpoint=${checkpoint} summary=${output_root}/summary.txt"
      grep '^overall:' "${output_root}/summary.txt" || true
      write_results_table
      break
    fi
    log "CHECKPOINT_INCOMPLETE step=${step} pass=${pass}/${MAX_PASSES}; retrying only missing tasks"
  done

  (( completed == 1 )) || die "checkpoint remained incomplete after ${MAX_PASSES} passes: ${checkpoint}"
done

write_results_table
log "ALL_DONE results=${RESULTS_TSV}"
