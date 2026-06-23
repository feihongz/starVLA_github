#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

SEEDS=(${SEEDS:-11 17 23 31})
SUITES=(libero_spatial libero_object libero_goal libero_10)
GPUS=(8 9 6 7)

run_one_suite() {
  local name="$1"
  local checkpoint="$2"
  local outroot="$3"
  local seed="$4"
  local suite="$5"
  local gpu="$6"

  local json_out="${outroot}/oracle_replay_${suite}_seed${seed}_recon.json"
  local log_out="${outroot}/replay_${suite}_seed${seed}.log"

  echo "RUN:${name}:${suite}:seed${seed}:gpu${gpu}" > "${log_out}"
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
    --checkpoint "${checkpoint}" \
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

run_checkpoint() {
  local name="$1"
  local checkpoint="$2"
  local outroot="$3"
  mkdir -p "${outroot}"

  for seed in "${SEEDS[@]}"; do
    echo "[$(date)] ${name}: starting seed ${seed}"
    for idx in "${!SUITES[@]}"; do
      run_one_suite "${name}" "${checkpoint}" "${outroot}" "${seed}" "${SUITES[$idx]}" "${GPUS[$idx]}" &
    done
    wait
    echo "[$(date)] ${name}: finished seed ${seed}"
  done
}

run_checkpoint \
  "g16_s4_8" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s4_8/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s4_8/replay_multiseed"

run_checkpoint \
  "g8_s4_8" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s4_8/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s4_8/replay_multiseed"

run_checkpoint \
  "g8_s8_cb1024" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s8_cb1024/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s8_cb1024/replay_multiseed"

run_checkpoint \
  "g8_baseline" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8/replay_multiseed"

echo "[$(date)] all requested stage1 tokenizer multiseed evals completed"
