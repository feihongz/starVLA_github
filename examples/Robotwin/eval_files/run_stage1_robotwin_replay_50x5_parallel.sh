#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

KIND="${KIND:-clean}"  # clean or clean_random
GPUS=(${GPUS:-0 1 2 3})
NUM_EPISODES_PER_TASK="${NUM_EPISODES_PER_TASK:-5}"
MODE="${MODE:-all}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
TASK_CONFIG="${TASK_CONFIG:-}"
SPLIT="${SPLIT:-}"
NORM_TASKS="${NORM_TASKS:-}"
MAX_RESTARTS="${MAX_RESTARTS:-2}"
RESUME_FROM_LOG="${RESUME_FROM_LOG:-0}"
LOG_APPEND="${LOG_APPEND:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TASKS_ALL=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

if [[ "${KIND}" == "clean" ]]; then
  RUN_SCRIPT="examples/Robotwin/eval_files/run_stage1_robotwin_abs_clean_replay.sh"
  TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
  SPLIT="${SPLIT:-clean}"
  OUT_ROOT="${OUT_ROOT:-playground/Checkpoints/var_stage1_robotwin_abs_clean_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/replay_50x5}"
elif [[ "${KIND}" == "clean_random" ]]; then
  RUN_SCRIPT="examples/Robotwin/eval_files/run_stage1_robotwin_abs_clean_random_replay.sh"
  TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
  SPLIT="${SPLIT:-randomized}"
  OUT_ROOT="${OUT_ROOT:-playground/Checkpoints/var_stage1_robotwin_abs_clean_random_e64_aeinit_productvq_g16_s1_2_4_8_16_32_50/replay_50x5_${TASK_CONFIG}_${SPLIT}}"
else
  echo "Unsupported KIND=${KIND}; expected clean or clean_random" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/chunks" "${OUT_ROOT}/logs" "${OUT_ROOT}/videos"

pids=()
for slot in "${!GPUS[@]}"; do
  gpu="${GPUS[$slot]}"
  chunk_tasks=()
  for idx in "${!TASKS_ALL[@]}"; do
    if (( idx % ${#GPUS[@]} == slot )); then
      chunk_tasks+=("${TASKS_ALL[$idx]}")
    fi
  done
  task_csv="$(IFS=,; echo "${chunk_tasks[*]}")"
  chunk_output="${OUT_ROOT}/chunks/chunk${slot}_gpu${gpu}.json"
  if [[ "${RESUME_FROM_LOG}" == "1" ]]; then
    chunk_output="${OUT_ROOT}/chunks/chunk${slot}_gpu${gpu}_retry_${RUN_ID}.json"
  fi
  chunk_video_dir="${OUT_ROOT}/videos/chunk${slot}_gpu${gpu}"
  chunk_log="${OUT_ROOT}/logs/chunk${slot}_gpu${gpu}.log"
  echo "[launch] slot=${slot} gpu=${gpu} tasks=${#chunk_tasks[@]} output=${chunk_output}"
  if [[ "${LOG_APPEND}" == "1" ]]; then
    touch "${chunk_log}"
  else
    : > "${chunk_log}"
  fi
  (
    attempt=0
    while true; do
      attempt=$((attempt + 1))
      echo "[attempt] slot=${slot} gpu=${gpu} attempt=${attempt} resume_from_log=${RESUME_FROM_LOG} $(date '+%F %T')"
      skip_log=""
      if [[ "${RESUME_FROM_LOG}" == "1" ]]; then
        skip_log="${chunk_log}"
      fi
      if GPU_ID="${gpu}" \
        TASK_CONFIG="${TASK_CONFIG}" \
        SPLIT="${SPLIT}" \
        MODE="${MODE}" \
        TASKS="${task_csv}" \
        NUM_EPISODES_PER_TASK="${NUM_EPISODES_PER_TASK}" \
        SAVE_VIDEOS="${SAVE_VIDEOS}" \
        VIDEO_OUT_DIR="${chunk_video_dir}" \
        OUTPUT="${chunk_output}" \
        NORM_TASKS="${NORM_TASKS}" \
        SKIP_COMPLETED_LOG="${skip_log}" \
        bash "${RUN_SCRIPT}"; then
        exit 0
      fi
      if (( attempt > MAX_RESTARTS )); then
        echo "[failed] slot=${slot} gpu=${gpu} exhausted MAX_RESTARTS=${MAX_RESTARTS}"
        exit 1
      fi
      echo "[restart] slot=${slot} gpu=${gpu} next_attempt=$((attempt + 1)) $(date '+%F %T')"
      sleep 10
    done
  ) >> "${chunk_log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

/home/zhangfeihong/miniconda3/envs/robotwin/bin/python - "${OUT_ROOT}" "${RESUME_FROM_LOG}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
resume_from_log = sys.argv[2] == "1"
summary = {"chunks": [], "summary": {}, "per_task": {}}
if resume_from_log:
    records = {}
    decoder = json.JSONDecoder()
    for path in sorted((root / "logs").glob("chunk*_gpu*.log")):
        summary["chunks"].append(str(path))
        text = path.read_text(errors="replace")
        pos = 0
        while True:
            start = text.find("{", pos)
            if start < 0:
                break
            try:
                record, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                pos = start + 1
                continue
            pos = start + end
            if not isinstance(record, dict) or record.get("counted", True) is False:
                continue
            if not ({"task", "episode", "mode", "success"} <= set(record)):
                continue
            key = (str(record["task"]), int(record["episode"]), str(record["mode"]))
            records[key] = record
    for (task, _episode, mode), record in records.items():
        bucket = summary["summary"].setdefault(mode, {"successes": 0, "episodes": 0, "success_rate": 0.0})
        bucket["successes"] += int(bool(record["success"]))
        bucket["episodes"] += 1
        task_bucket = summary["per_task"].setdefault(task, {})
        per_mode = task_bucket.setdefault(mode, {"successes": 0, "episodes": 0, "success_rate": 0.0})
        per_mode["successes"] += int(bool(record["success"]))
        per_mode["episodes"] += 1
else:
    for path in sorted((root / "chunks").glob("chunk*_gpu*.json")):
        report = json.loads(path.read_text())
        summary["chunks"].append(str(path))
        for mode, item in report.get("summary", {}).items():
            bucket = summary["summary"].setdefault(mode, {"successes": 0, "episodes": 0, "success_rate": 0.0})
            bucket["successes"] += int(item.get("successes", 0))
            bucket["episodes"] += int(item.get("episodes", 0))
        for task, modes in report.get("per_task", {}).items():
            task_bucket = summary["per_task"].setdefault(task, {})
            for mode, item in modes.items():
                bucket = task_bucket.setdefault(mode, {"successes": 0, "episodes": 0, "success_rate": 0.0})
                bucket["successes"] += int(item.get("successes", 0))
                bucket["episodes"] += int(item.get("episodes", 0))

for modes in [summary["summary"], *summary["per_task"].values()]:
    for item in modes.values():
        item["success_rate"] = item["successes"] / max(item["episodes"], 1)

out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))
print(f"Wrote {out}")
PY

exit "${status}"
