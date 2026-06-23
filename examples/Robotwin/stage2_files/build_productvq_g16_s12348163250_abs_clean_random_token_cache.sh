#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${CACHE_OUTPUT:-playground/Checkpoints/var_stage1_robotwin_abs_clean_random_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/stage2_token_cache_full.pt}"

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/Robotwin/train_files/train_var_stage1_robotwin_abs_clean_random_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50.yaml \
  --stage1_artifact playground/Checkpoints/var_stage1_robotwin_abs_clean_random_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/best_recon.ckpt \
  --output "${OUTPUT}" \
  --mode train \
  --device "${CACHE_DEVICE:-cpu}" \
  --batch_size "${CACHE_BATCH_SIZE:-512}" \
  --num_workers "${CACHE_NUM_WORKERS:-0}" \
  --max_batches "${CACHE_MAX_BATCHES:-0}"
