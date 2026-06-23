#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_robotwin_abs_clean_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/stage2_token_cache_full.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/Robotwin/stage2_files/build_productvq_g16_s12348163250_abs_clean_token_cache.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s12348163250_robotwin_abs_clean_100k_4gpu_bs8_accum8_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
import h5py  # noqa: F401
PY

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29562}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/stage2_files/train_qwen_var_productvq_g16_s12348163250_abs_clean_100k_4gpu_bs8_accum8.yaml \
  2>&1 | tee -a "${LOG_FILE}"
