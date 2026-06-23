#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/robotwin/bin/python}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/zhangfeihong/RoboTwin}"
STARVLA_SITE_PACKAGES="${STARVLA_SITE_PACKAGES:-/home/zhangfeihong/miniconda3/envs/starVLA/lib/python3.10/site-packages}"
GPU_ID="${GPU_ID:-0}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SPLIT="${SPLIT:-clean}"
MODE="${MODE:-both}"
TASKS="${TASKS:-all}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_EPISODES_PER_TASK="${NUM_EPISODES_PER_TASK:-5}"
START_EPISODE="${START_EPISODE:-0}"
NORM_TASKS="${NORM_TASKS:-}"
FAST_TOKENIZER_NAME="${FAST_TOKENIZER_NAME:-physical-intelligence/fast}"
SKIP_COMPLETED_LOG="${SKIP_COMPLETED_LOG:-}"

CHECKPOINT="${CHECKPOINT:-playground/Checkpoints/var_stage1_robotwin_abs_clean_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/best_recon.ckpt}"
CONFIG_YAML="${CONFIG_YAML:-examples/Robotwin/train_files/train_var_stage1_robotwin_abs_clean_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50_resume100.yaml}"
OUT_DIR="${OUT_DIR:-playground/Checkpoints/var_stage1_robotwin_abs_clean_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/replay}"
OUTPUT="${OUTPUT:-${OUT_DIR}/robotwin_stage1_replay_${TASK_CONFIG}_${SPLIT}_${MODE}_${NUM_EPISODES_PER_TASK}eps.json}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-${OUT_DIR}/videos_${TASK_CONFIG}_${SPLIT}_${MODE}_${NUM_EPISODES_PER_TASK}eps}"

mkdir -p "${OUT_DIR}"

VIDEO_ARGS=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
  VIDEO_ARGS=(--save_videos --video_out_dir "${VIDEO_OUT_DIR}")
fi
NORM_ARGS=()
if [[ -n "${NORM_TASKS}" ]]; then
  NORM_ARGS=(--norm_tasks "${NORM_TASKS}")
fi
SKIP_ARGS=()
if [[ -n "${SKIP_COMPLETED_LOG}" ]]; then
  SKIP_ARGS=(--skip_completed_log "${SKIP_COMPLETED_LOG}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
PYTHONPATH="/home/zhangfeihong/starVLA:${ROBOTWIN_PATH}:${ROBOTWIN_PATH}/description/utils:${STARVLA_SITE_PACKAGES}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" examples/Robotwin/eval_files/eval_var_stage1_robotwin_replay.py \
  --checkpoint "${CHECKPOINT}" \
  --config_yaml "${CONFIG_YAML}" \
  --output "${OUTPUT}" \
  --robotwin_path "${ROBOTWIN_PATH}" \
  --task_config "${TASK_CONFIG}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --tasks "${TASKS}" \
  --max_tasks "${MAX_TASKS}" \
  --start_episode "${START_EPISODE}" \
  --num_episodes_per_task "${NUM_EPISODES_PER_TASK}" \
  "${NORM_ARGS[@]}" \
  "${SKIP_ARGS[@]}" \
  --fast_tokenizer_name "${FAST_TOKENIZER_NAME}" \
  "${VIDEO_ARGS[@]}" \
  --device cuda
