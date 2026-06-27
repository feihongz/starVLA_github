# RoboCasa robust stage2 eval

This directory contains a resumable RoboCasa stage2 evaluation chain modeled after the LIBERO robust eval flow.

## Chain

```text
run_robocasa_stage2_parallel_eval_supervisor.sh
  -> run_robocasa_stage2_eval_chunked.py
    -> run_robocasa_ckpt_eval.py
      -> simulation_env.py
        -> deployment/model_server/server_policy.py
```

Shared task presets are defined in:

```text
robocasa_eval_tasks.py
```

Result aggregation is handled by:

```text
summarize_robocasa_success.py
```

## Robustness behavior

- Evaluates by task and episode chunks.
- Completed chunks are skipped on resume.
- Each chunk writes its own status directory:
  - `RUNNING.json`
  - `COMPLETE.json`
  - `INVALID.json`
  - `CHUNK_RUNNING.json`
  - `ROBOCASA_CHUNK_OK.json`
  - `CHUNK_FAILED.json`
- The supervisor checks expected tasks and expected chunks, not just whether tmux workers exited.
- Missing or incomplete chunks cause the supervisor to relaunch workers.
- `run_robocasa_ckpt_eval.py` starts and tears down the policy server for each chunk.
- Server startup has a ready timeout.
- Simulation has a timeout.
- Interrupted evals are marked invalid through signal handlers.
- Optional video QA checks sampled mp4 frames and rejects empty/black videos.
- `summarize_robocasa_success.py --require-complete` exits nonzero if any expected chunk is missing or incomplete.

## Task presets

Two presets are available:

- `gr1_5`: the 5-task set used by earlier 82k small eval runs.
- `gr1_24`: the 24-task list from the older batch eval script.

You can also provide a custom task list with `TASKS_FILE=/path/to/tasks.txt`, one RoboCasa env name per line.

## Usage

From the repo root:

```bash
cd /home/zhangfeihong/starVLA_github

EVAL_GPUS="0 1" \
TASKS_PRESET=gr1_5 \
TRIALS_PER_TASK=10 \
CHUNK_EPISODES=1 \
EVAL_USE_BF16=0 \
SAVE_VIDEOS=0 \
bash examples/Robocasa_tabletop/eval_files/run_robocasa_stage2_parallel_eval_supervisor.sh \
  playground/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache/checkpoints/steps_82000_pytorch_model.pt
```

For the 24-task benchmark:

```bash
EVAL_GPUS="0 1 2 3" \
TASKS_PRESET=gr1_24 \
TRIALS_PER_TASK=50 \
CHUNK_EPISODES=1 \
EVAL_USE_BF16=0 \
SAVE_VIDEOS=0 \
bash examples/Robocasa_tabletop/eval_files/run_robocasa_stage2_parallel_eval_supervisor.sh \
  playground/Checkpoints/<run>/checkpoints/<ckpt>.pt
```

## Useful environment variables

- `EVAL_GPUS`: worker GPU ids. Default `0 1`.
- `WORKER_COUNT`: number of workers. Default is the number of GPUs.
- `TASKS_PRESET`: `gr1_5` or `gr1_24`. Default `gr1_5`.
- `TASKS_FILE`: optional custom task file. Overrides `TASKS_PRESET`.
- `TRIALS_PER_TASK`: total episodes per task. Default `10`.
- `CHUNK_EPISODES`: episodes per chunk. Default `1`.
- `BASE_PORT`: base policy server port. Default `6700`.
- `MAX_RETRIES`: per-chunk retry limit. Default `100000`.
- `CHECK_INTERVAL_SECONDS`: supervisor polling interval. Default `60`.
- `MAX_EPISODE_STEPS`: simulator max episode steps. Default `720`.
- `N_ACTION_STEPS`: action chunk length passed to RoboCasa sim. Default `12`.
- `SERVER_READY_TIMEOUT`: policy server ready timeout. Default `900`.
- `SERVER_IDLE_TIMEOUT`: policy server idle timeout. Default `1800`.
- `SIM_TIMEOUT`: simulation timeout per chunk. Default `3600`.
- `EVAL_USE_BF16`: `1` enables bf16 server inference, `0` uses fp32. Default `0`.
- `SAVE_VIDEOS`: `1` saves videos and enables video QA; `0` disables videos. Default `0`.
- `ACTION_STATS_EVERY`: log unnormalized RoboCasa action stats every N sim steps. Default `0`.
- `NORM_ACTION_STATS_EVERY`: log normalized/unnormalized server action stats every N predictions. Default `0`.

## Outputs

For checkpoint:

```text
playground/Checkpoints/<run>/checkpoints/<ckpt>.pt
```

the default eval output is:

```text
playground/Checkpoints/<run>/robocasa_eval/<ckpt>_<TASKS_PRESET>_<TRIALS_PER_TASK>eps_chunk<CHUNK_EPISODES>_robust/
```

Important files:

- `summary.txt`: current human-readable summary.
- `summary.json`: machine-readable aggregate summary.
- `logs/robocasa_parallel_eval_supervisor.log`: supervisor log.
- `logs/worker_<id>.log`: worker logs.
- `<task_slug>/rXXX_nYYY/COMPLETE.json`: successful chunk result.
- `<task_slug>/rXXX_nYYY/INVALID.json`: failed chunk result from `run_robocasa_ckpt_eval.py`.

## Manual summary

To summarize an existing output root:

```bash
python examples/Robocasa_tabletop/eval_files/summarize_robocasa_success.py \
  playground/Checkpoints/<run>/robocasa_eval/<output_root> \
  --tasks-preset gr1_5 \
  --trials-per-task 10 \
  --chunk-episodes 1 \
  --expected-episodes-per-chunk 1 \
  --require-complete
```
