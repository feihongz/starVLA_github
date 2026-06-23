#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

SEEDS=(${SEEDS:-11 17 23 31})
SUITES=(libero_spatial libero_object libero_goal libero_10)
GPUS=(${GPUS:-8 9 6 7})

NAME="g16_s1_2_4_8_structemb"
CHECKPOINT="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/best_recon.ckpt"
OUTROOT="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/replay_multiseed"

run_one_suite() {
  local seed="$1"
  local suite="$2"
  local gpu="$3"

  local json_out="${OUTROOT}/oracle_replay_${suite}_seed${seed}_recon.json"
  local log_out="${OUTROOT}/replay_${suite}_seed${seed}.log"

  echo "RUN:${NAME}:${suite}:seed${seed}:gpu${gpu}" > "${log_out}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  NO_ALBUMENTATIONS_UPDATE=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONWARNINGS=ignore \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPATH=/home/zhangfeihong/starVLA:/home/zhangfeihong/LIBERO:${PYTHONPATH:-} \
  /home/zhangfeihong/miniconda3/envs/starVLA/bin/python scripts/eval/run_var_stage1_replay_cleanenv.py \
    --checkpoint "${CHECKPOINT}" \
    --output "${json_out}" \
    --task_suite_name "${suite}" \
    --mode recon \
    --max_tasks 10 \
    --num_episodes_per_task 5 \
    --num_steps_wait 0 \
    --init_state_strategy hdf5_action \
    --gripper_mode open01 \
    --seed "${seed}" \
    --device cuda \
    >> "${log_out}" 2>&1
  echo "DONE:0" >> "${log_out}"
}

mkdir -p "${OUTROOT}"
echo "START $(date)"
echo "SEEDS ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  echo "SEED_START ${seed} $(date)"
  for idx in "${!SUITES[@]}"; do
    run_one_suite "${seed}" "${SUITES[$idx]}" "${GPUS[$idx]}" &
  done
  wait
  echo "SEED_DONE ${seed} $(date)"
done

echo "ALL_DONE $(date)"
