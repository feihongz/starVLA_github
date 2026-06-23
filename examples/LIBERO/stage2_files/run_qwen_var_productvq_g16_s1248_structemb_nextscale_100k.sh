#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/stage2_token_cache_full.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/LIBERO/stage2_files/build_productvq_g16_s1248_structemb_full_token_cache.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29542}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/stage2_files/train_qwen_var_productvq_g16_s1248_structemb_nextscale_100k.yaml \
  2>&1 | tee -a "${LOG_FILE}"
