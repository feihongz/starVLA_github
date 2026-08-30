#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv310/bin/python}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/LIBERO/train_files/train_var_stage1_pi05_libero_q99_pure_ae_e32_longweighted_v2_from_scratch.yaml

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_productvq_longweighted_mtr_v2_from_scratch_e100.yaml
