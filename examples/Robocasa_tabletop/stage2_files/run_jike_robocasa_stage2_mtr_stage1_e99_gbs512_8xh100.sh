#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/root/feihong/starVLA_github}"
cd "${REPO_DIR}"

RUN_ID="qwen_var_productvq_g16_s124816_robocasa_mtr_stage1_e99_100k_lr1e4_warmup5000_gbs512_jike8h100"
CONFIG_YAML="examples/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_mtr_stage1_e99_100k_lr1e4_warmup5000_gbs512_jike8h100.yaml"
STAGE1_CONFIG="examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_abs_productvq_g16_s1_2_4_8_16_e256_closebalanced_mtr_from_scratch_e100.yaml"
STAGE1_DIR="${REPO_DIR}/Checkpoints/var_stage1_robocasa_gr1_abs_productvq_g16_s124816_e256_closebalanced_mtr_from_scratch_e100"
STAGE1_ARTIFACT="${STAGE1_DIR}/epoch_099.ckpt"
TOKEN_CACHE="${STAGE1_DIR}/stage2_token_cache_epoch099.pt"
BASE_MODEL="${BASE_MODEL:-/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-/root/tianyi/LDA-1B/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
RUN_DIR="${REPO_DIR}/Checkpoints/${RUN_ID}"
# Keep the same log filename as the verified e80 Stage2 run.
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_bucket100m.log}"

CURRENT_PHASE="bootstrap"
ERR_REPORTED=0

mark_phase() {
  CURRENT_PHASE="$1"
  printf '[%s] [stage2-launcher] phase=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CURRENT_PHASE}"
}

report_launcher_error() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  ERR_REPORTED=1
  trap - ERR
  printf '[stage2-launcher] ERROR phase=%s status=%s line=%s command=%q log=%s\n' \
    "${CURRENT_PHASE}" "${exit_code}" "${line_number}" "${failed_command}" "${LOG_FILE}" >&2
  exit "${exit_code}"
}

report_launcher_exit() {
  local exit_code=$?
  if (( exit_code != 0 && ERR_REPORTED == 0 )); then
    printf '[stage2-launcher] EXIT phase=%s status=%s log=%s\n' \
      "${CURRENT_PHASE}" "${exit_code}" "${LOG_FILE}" >&2
  fi
}

trap 'report_launcher_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap report_launcher_exit EXIT

echo "[stage2-launcher] entered=$(date -u +%Y-%m-%dT%H:%M:%SZ) repo=${REPO_DIR}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[stage2-launcher] full_log=${LOG_FILE}"

EXPECTED_STAGE1_SHA256="1194f76588f94be17e5f86cea1df8ba372b06ee9bc5241f9a7a8f205ff6a38ec"
EXPECTED_TOKEN_CACHE_BYTES="23665715710"

mark_phase required_files
for required_file in \
  "${CONFIG_YAML}" \
  "${STAGE1_CONFIG}" \
  "${STAGE1_ARTIFACT}" \
  "${TOKEN_CACHE}" \
  "${BASE_MODEL}/config.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

mark_phase data_root
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing RoboCasa data root: ${DATA_ROOT}" >&2
  exit 1
fi

mark_phase artifact_integrity
actual_stage1_sha256="$(sha256sum "${STAGE1_ARTIFACT}" | awk '{print $1}')"
if [[ "${actual_stage1_sha256}" != "${EXPECTED_STAGE1_SHA256}" ]]; then
  echo "Stage1 checkpoint SHA256 mismatch." >&2
  echo "  expected: ${EXPECTED_STAGE1_SHA256}" >&2
  echo "  actual:   ${actual_stage1_sha256}" >&2
  exit 1
fi

actual_token_cache_bytes="$(stat -c '%s' "${TOKEN_CACHE}")"
if [[ "${actual_token_cache_bytes}" != "${EXPECTED_TOKEN_CACHE_BYTES}" ]]; then
  echo "Stage2 token cache size mismatch; refusing to use a partial or different cache." >&2
  echo "  expected: ${EXPECTED_TOKEN_CACHE_BYTES}" >&2
  echo "  actual:   ${actual_token_cache_bytes}" >&2
  exit 1
fi

# Use the same Python environment as the verified 57.50% RoboCasa Stage2 run.
mark_phase python_dependencies
PYTHON_BIN="${PYTHON_BIN:-/root/feihong/starVLA/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[stage2-launcher] dependency preflight begin"
"${PYTHON_BIN}" -c \
  'import accelerate, deepspeed, flash_attn, qwen_vl_utils, torch, transformers; print(f"Stage2 dependency preflight OK: torch={torch.__version__}, transformers={transformers.__version__}, accelerate={accelerate.__version__}, deepspeed={deepspeed.__version__}")'
echo "[stage2-launcher] dependency preflight complete"

mark_phase distributed_contract
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
if [[ "${NUM_PROCESSES}" -ne 8 ]]; then
  echo "This launcher reproduces the 57.50% global-batch contract and requires NUM_PROCESSES=8; got ${NUM_PROCESSES}." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
if [[ "${MIXED_PRECISION}" != "bf16" ]]; then
  echo "This launcher reproduces the 57.50% bf16 contract; got MIXED_PRECISION=${MIXED_PRECISION}." >&2
  exit 1
fi
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29568}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PATH="/root/feihong/starVLA/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Conservative distributed defaults for long 8xH100 jobs on JiKe.
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DEEPSPEED_REDUCE_BUCKET_SIZE="${DEEPSPEED_REDUCE_BUCKET_SIZE:-100000000}"
export DEEPSPEED_ALLGATHER_BUCKET_SIZE="${DEEPSPEED_ALLGATHER_BUCKET_SIZE:-100000000}"

export HF_HOME="${HF_HOME:-/root/feihong/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/feihong/.cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${REPO_DIR}/wandb}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}" "${RUN_DIR}"

mark_phase cuda_visibility
available_gpus="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${available_gpus}" -lt "${NUM_PROCESSES}" ]]; then
  echo "Need ${NUM_PROCESSES} visible CUDA devices, but only ${available_gpus} detected." >&2
  exit 1
fi
echo "[stage2-launcher] visible_gpus=${available_gpus}"

mark_phase launch_summary
cat <<EOF
[jike_robocasa_stage2_mtr_e99]
  baseline=57.50%@steps_86000 (690/1200 episodes)
  run_id=${RUN_ID}
  config=${CONFIG_YAML}
  python_bin=${PYTHON_BIN}
  base_model=${BASE_MODEL}
  run_dir=${RUN_DIR}
  stage1_config=${STAGE1_CONFIG}
  stage1_artifact=${STAGE1_ARTIFACT}
  stage1_sha256=${actual_stage1_sha256}
  token_cache=${TOKEN_CACHE}
  token_cache_bytes=${actual_token_cache_bytes}
  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
  num_processes=${NUM_PROCESSES}
  mixed_precision=${MIXED_PRECISION}
  per_device_batch_size=32
  gradient_accumulation_steps=2
  global_batch_size=$((NUM_PROCESSES * 32 * 2))
  max_train_steps=100000
  warmup_steps=5000
  save_interval=2000
  eval_interval=2000
  base_lr=1e-4
  qwen_vl_interface_lr=1e-5
  deepspeed_reduce_bucket_size=${DEEPSPEED_REDUCE_BUCKET_SIZE}
  deepspeed_allgather_bucket_size=${DEEPSPEED_ALLGATHER_BUCKET_SIZE}
  distributed_timeout_seconds=${TORCH_DISTRIBUTED_TIMEOUT_SECONDS}
  main_process_port=${MAIN_PROCESS_PORT}
  wandb_mode=${WANDB_MODE}
  log_file=${LOG_FILE}
EOF

mark_phase accelerate_launch
"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}"

mark_phase complete
