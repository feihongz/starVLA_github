# LIBERO robust stage2 eval

This directory contains the resumable LIBERO stage2 evaluation chain used for the 40-task, 4-suite benchmark.

## Chain

```text
run_stage2_100k_4suite_parallel_eval_supervisor.sh
  -> run_stage2_eval_chunked.sh
    -> run_local_eval_once.sh
      -> eval_libero.py
        -> deployment/model_server/server_policy.py
```

`run_stage2_100k_4suite_parallel_eval_supervisor.sh` is the top-level launcher. It assigns one tmux session, GPU, and port to each LIBERO suite:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

## Robustness behavior

- Evaluates by task/trial chunks instead of one monolithic run.
- Completed chunks are skipped on resume.
- A chunk is considered valid only if its log contains:
  - `EVAL_CHUNK_OK`
  - `Total success rate:`
  - `Total episodes: <expected_count>`
- Each chunk is wrapped by a timeout.
- Failed chunks are retried up to `MAX_RETRIES`.
- The supervisor periodically checks progress and relaunches missing tmux sessions.
- `eval_libero.py` validates policy input images by shape, dtype, finite values, mean, std, and max value, which catches black/empty policy inputs.
- `summarize_libero_success.py --chunked --require-ok-marker` only aggregates completed chunks.

## Known validated 99k eval

The LIBERO stage2 checkpoint below has a recorded 40-task mean success rate of `98.05%`:

```text
playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_99000_pytorch_model.pt
```

That run used:

- `EVAL_SEED=7`
- `EVAL_USE_BF16=0`
- output root `eval_stage2_99k_40task_4suite_seed7_0616_fp32`
- GPUs `6 0 1 8`
- ports `19010 19011 19012 19013`

## Usage

From the repo root:

```bash
cd /home/zhangfeihong/starVLA_github

EVAL_OUTPUT_ROOT=eval_stage2_99k_40task_4suite_seed7_0616_fp32 \
EVAL_SEED=7 \
EVAL_USE_BF16=0 \
EVAL_GPUS="6 0 1 8" \
EVAL_PORTS="19010 19011 19012 19013" \
SESSION_SUFFIX="_99k_seed7_0616fp32" \
bash examples/LIBERO/eval_files/run_stage2_100k_4suite_parallel_eval_supervisor.sh \
  playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_99000_pytorch_model.pt
```

For a new checkpoint, replace the checkpoint path and choose a new `EVAL_OUTPUT_ROOT`.

## Useful environment variables

- `TRIALS_PER_TASK`: default `50`.
- `CHECK_INTERVAL_SECONDS`: supervisor polling interval, default `60`.
- `MAX_RETRIES`: per-chunk retry limit, default `100000`.
- `CHUNK_TIMEOUT_SECONDS`: per-chunk timeout, default `1800`.
- `EVAL_GPUS`: four GPU ids, one per suite.
- `EVAL_PORTS`: four server ports, one per suite.
- `EVAL_USE_BF16`: `1` enables bf16 server inference, `0` uses fp32.
- `OBJECT_GOAL_CHUNK_TRIALS`: chunk size for object/goal suites, default `1`.
- `SPATIAL_CHUNK_TRIALS`: chunk size for spatial suite, default `1`.
- `LIBERO_10_CHUNK_TRIALS`: chunk size for libero_10, default `5`.

## Outputs

For checkpoint:

```text
playground/Checkpoints/<run>/checkpoints/<ckpt>.pt
```

the eval output is written under:

```text
playground/Checkpoints/<run>/<EVAL_OUTPUT_ROOT>/
```

Important files:

- `logs/libero_40task_progress.txt`: periodically updated partial summary.
- `logs/libero_40task_summary.txt`: final summary.
- `logs/<suite>/*_chunked_t*_r*_n*.log`: chunk logs.
- `videos/`: optional videos when `SAVE_VIDEOS=1`.
