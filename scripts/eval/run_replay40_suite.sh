#!/usr/bin/env bash
set -euo pipefail

name="$1"
checkpoint="$2"
out_dir="$3"
suite="$4"
gpu="$5"

mkdir -p "$out_dir"

json_out="${out_dir}/oracle_replay_${suite}_10tasks_5eps_hdf5_init_wait0_fast_cleanenv_40tasks_gpu${gpu}.json"
log_out="${out_dir}/replay40_${suite}_gpu${gpu}.log"

cd /home/zhangfeihong/starVLA

echo "RUN:${name}:${suite}:gpu${gpu}" > "$log_out"
CUDA_VISIBLE_DEVICES="$gpu" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONWARNINGS=ignore \
NUMBA_CACHE_DIR=/tmp/numba_cache \
MPLCONFIGDIR=/tmp/mplconfig \
PYTHONPATH=/home/zhangfeihong/starVLA:/home/zhangfeihong/LIBERO:${PYTHONPATH:-} \
/home/zhangfeihong/miniconda3/envs/starVLA/bin/python scripts/eval/run_var_stage1_replay_cleanenv.py \
  --checkpoint "$checkpoint" \
  --output "$json_out" \
  --task_suite_name "$suite" \
  --mode all \
  --max_tasks 10 \
  --num_episodes_per_task 5 \
  --num_steps_wait 0 \
  --init_state_strategy hdf5_action \
  --gripper_mode open01 \
  --device cuda \
  --fast_tokenizer_name /home/zhangfeihong/.cache/huggingface/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4 \
  >> "$log_out" 2>&1
echo "DONE:0" >> "$log_out"
