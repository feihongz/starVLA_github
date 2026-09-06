#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/feihong/starVLA_github}"
cd "${REPO_DIR}"

RUN_ID="qwen_var_productvq_g16_s1248_mtr_from_scratch_e99_nextscale_65k_gbs64_jike8h100"
CONFIG_YAML="examples/LIBERO/stage2_files/train_qwen_var_productvq_g16_s1248_mtr_from_scratch_e99_nextscale_65k_gbs64_jike8h100.yaml"
STAGE1_CONFIG="examples/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_productvq_weighted_tasks_mtr_from_scratch_e100.yaml"
STAGE1_ARTIFACT="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_productvq_weighted_tasks_mtr_from_scratch_e100/best_recon.ckpt"
TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_productvq_weighted_tasks_mtr_from_scratch_e100/stage2_token_cache_full_best_recon_epoch099.pt"
BASE_MODEL="/root/public/models/Qwen/Qwen3-VL-4B-Instruct"
RUN_DIR="${REPO_DIR}/Checkpoints/${RUN_ID}"
EXPECTED_STAGE1_SHA256="0902d30458d7da10f4871aa799f520cbaa2ce692732e3eec1891072a00ab74bd"

for required_path in \
  "${CONFIG_YAML}" \
  "${STAGE1_CONFIG}" \
  "${STAGE1_ARTIFACT}" \
  "${TOKEN_CACHE}" \
  "${BASE_MODEL}/config.json"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

actual_stage1_sha256="$(sha256sum "${STAGE1_ARTIFACT}" | awk '{print $1}')"
if [[ "${actual_stage1_sha256}" != "${EXPECTED_STAGE1_SHA256}" ]]; then
  echo "Stage1 checkpoint SHA256 mismatch." >&2
  echo "  expected: ${EXPECTED_STAGE1_SHA256}" >&2
  echo "  actual:   ${actual_stage1_sha256}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/root/feihong/starVLA/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29566}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PATH="/root/feihong/starVLA/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
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

available_gpus="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${available_gpus}" -lt "${NUM_PROCESSES}" ]]; then
  echo "Need ${NUM_PROCESSES} visible CUDA devices, but only ${available_gpus} detected." >&2
  exit 1
fi

LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

cat <<EOF
[jike_libero_stage2_mtr_from_scratch]
  run_id=${RUN_ID}
  config=${CONFIG_YAML}
  run_dir=${RUN_DIR}
  stage1_artifact=${STAGE1_ARTIFACT}
  stage1_sha256=${actual_stage1_sha256}
  token_cache=${TOKEN_CACHE}
  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
  num_processes=${NUM_PROCESSES}
  mixed_precision=${MIXED_PRECISION}
  per_device_batch_size=8
  gradient_accumulation_steps=1
  global_batch_size=$((NUM_PROCESSES * 8))
  max_train_steps=65000
  checkpoint_retained_range_at_completion=40000-65000
  main_process_port=${MAIN_PROCESS_PORT}
  wandb_mode=${WANDB_MODE}
  log_file=${LOG_FILE}
EOF

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  2>&1 | tee -a "${LOG_FILE}"
