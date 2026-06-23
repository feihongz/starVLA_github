#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"
GPU_LIST=(${GPU_LIST:-0 1})
MAX_RETRIES="${MAX_RETRIES:-2}"
SUITES=(${SUITES:-libero_spatial libero_object libero_goal libero_10})
TASK_IDS=(${TASK_IDS:-0 1 2 3 4 5 6 7 8 9})

run_chunk() {
  local name="$1"
  local checkpoint="$2"
  local outroot="$3"
  local suite="$4"
  local task_id="$5"
  local gpu="$6"

  local chunk_dir="${outroot}/chunks/${suite}"
  local json_out="${chunk_dir}/task${task_id}_5eps_recon_gpu${gpu}.json"
  local log_out="${chunk_dir}/task${task_id}_5eps_recon_gpu${gpu}.log"
  mkdir -p "${chunk_dir}"

  if [[ -f "${json_out}" ]]; then
    echo "[skip] ${name} ${suite} task=${task_id} already has ${json_out}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[run] ${name} ${suite} task=${task_id} gpu=${gpu} attempt=${attempt}" | tee "${log_out}"
    if CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      NO_ALBUMENTATIONS_UPDATE=1 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONWARNINGS=ignore \
      NUMBA_CACHE_DIR=/tmp/numba_cache \
      MPLCONFIGDIR=/tmp/mplconfig \
      PYTHONPATH=/home/zhangfeihong/starVLA:/home/zhangfeihong/LIBERO:${PYTHONPATH:-} \
      "${PYTHON_BIN}" scripts/eval/run_var_stage1_replay_cleanenv.py \
        --checkpoint "${checkpoint}" \
        --output "${json_out}" \
        --task_suite_name "${suite}" \
        --mode recon \
        --task_ids "${task_id}" \
        --max_tasks 0 \
        --num_episodes_per_task 5 \
        --num_steps_wait 0 \
        --init_state_strategy hdf5_action \
        --gripper_mode open01 \
        --device cuda \
        >> "${log_out}" 2>&1; then
      echo "[done] ${name} ${suite} task=${task_id}" | tee -a "${log_out}"
      return 0
    fi
    echo "[retry] ${name} ${suite} task=${task_id} failed attempt=${attempt}" | tee -a "${log_out}"
    attempt=$((attempt + 1))
    sleep 5
  done

  echo "[failed] ${name} ${suite} task=${task_id}" | tee -a "${log_out}"
  return 1
}

aggregate() {
  local name="$1"
  local outroot="$2"
  "${PYTHON_BIN}" - "${name}" "${outroot}" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
outroot = Path(sys.argv[2])
summary = {
    "name": name,
    "outroot": str(outroot),
    "suites": {},
    "total": {"successes": 0, "episodes": 0, "success_rate": 0.0},
    "missing": [],
}

for suite_dir in sorted((outroot / "chunks").glob("libero_*")):
    suite = suite_dir.name
    suite_summary = {"successes": 0, "episodes": 0, "success_rate": 0.0, "chunks": []}
    for path in sorted(suite_dir.glob("task*_5eps_recon_gpu*.json")):
        report = json.loads(path.read_text())
        item = report["summary"]["recon"]
        chunk = {
            "path": str(path),
            "successes": int(item["successes"]),
            "episodes": int(item["episodes"]),
            "success_rate": float(item["success_rate"]),
        }
        suite_summary["chunks"].append(chunk)
        suite_summary["successes"] += chunk["successes"]
        suite_summary["episodes"] += chunk["episodes"]
    suite_summary["success_rate"] = (
        suite_summary["successes"] / suite_summary["episodes"]
        if suite_summary["episodes"]
        else 0.0
    )
    summary["suites"][suite] = suite_summary
    summary["total"]["successes"] += suite_summary["successes"]
    summary["total"]["episodes"] += suite_summary["episodes"]

for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    for task_id in range(10):
        matches = list((outroot / "chunks" / suite).glob(f"task{task_id}_5eps_recon_gpu*.json"))
        if not matches:
            summary["missing"].append({"suite": suite, "task_id": task_id})

if summary["total"]["episodes"]:
    summary["total"]["success_rate"] = summary["total"]["successes"] / summary["total"]["episodes"]

outroot.mkdir(parents=True, exist_ok=True)
(outroot / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["total"], indent=2))
print(f"Wrote {outroot / 'summary.json'}")
PY
}

run_checkpoint() {
  local name="$1"
  local checkpoint="$2"
  local outroot="$3"
  mkdir -p "${outroot}"

  local failures=0
  for suite in "${SUITES[@]}"; do
    echo "[suite] ${name} ${suite}"
    local slot=0
    local pids=()
    for task_id in "${TASK_IDS[@]}"; do
      local gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
      run_chunk "${name}" "${checkpoint}" "${outroot}" "${suite}" "${task_id}" "${gpu}" &
      pids+=("$!")
      slot=$((slot + 1))
      if (( ${#pids[@]} >= ${#GPU_LIST[@]} )); then
        for pid in "${pids[@]}"; do
          if ! wait "${pid}"; then
            failures=$((failures + 1))
          fi
        done
        pids=()
      fi
    done
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failures=$((failures + 1))
      fi
    done
    aggregate "${name}" "${outroot}"
  done

  aggregate "${name}" "${outroot}"
  if (( failures > 0 )); then
    echo "[warn] ${name} had ${failures} failed chunks after retries"
  fi
}

run_checkpoint \
  "s1248_baseline" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/replay40_chunked_recon"

run_checkpoint \
  "s1248_replayfocus" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb_replayfocus/best_recon.ckpt" \
  "playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb_replayfocus/replay40_chunked_recon"

echo "[all done] $(date)"
