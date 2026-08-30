# Clean Stage1 ablation launcher (4x H100)

The recommended entry point for the controlled LIBERO/RoboCasa four-method
Stage1 experiment is:

```bash
cd /root/feihong/starVLA_github
bash scripts/stage1/run_clean_ablation_4xh100_8parallel.sh
```

The launcher requires four visible GPUs with local indices `0,1,2,3`. It
starts eight independent Stage1 processes in each phase, with one LIBERO and
one RoboCasa process on every H100:

| GPU | LIBERO | RoboCasa |
| --- | --- | --- |
| 0 | Multi-scale Base | MTR |
| 1 | Full-target Time | Paper-DCT |
| 2 | Paper-DCT | Full-target Time |
| 3 | MTR | Multi-scale Base |

All formal jobs use seed 42, 50 epochs, batch size 256 per process, and eight
data workers per process. The auxiliary weight is 0.02 for LIBERO and 0.1 for
RoboCasa; Base remains exactly zero. Every method within a benchmark shares
the same canonical config, dataset, seed, and frozen pure-AE initialization.

The end-to-end order is:

1. verify the fixed LIBERO e32 pure-AE artifact;
2. verify or train the common 30-epoch RoboCasa e64 pure-AE on GPU 0;
3. run all eight two-batch smoke jobs concurrently;
4. jointly preflight all eight formal jobs;
5. run all eight 50-epoch jobs concurrently;
6. verify every manifest, config/init hash, contiguous history, and final
   checkpoint.

The RoboCasa pure-AE prerequisite must finish before the four RoboCasa methods
start. This also initializes the RoboCasa dataset/statistics path before eight
workers begin reading it concurrently. A complete existing pure-AE is reused
only after artifact validation. An explicit external initialization can be
provided with `ROBOCASA_STAGE1_INIT_CHECKPOINT`.

## Preview

This prints the pure-AE, smoke, and formal commands without creating files:

```bash
cd /root/feihong/starVLA_github
DRY_RUN=1 bash scripts/stage1/run_clean_ablation_4xh100_8parallel.sh
```

## Resume

After an interrupted run, use:

```bash
cd /root/feihong/starVLA_github
RESUME_QUEUE=1 bash scripts/stage1/run_clean_ablation_4xh100_8parallel.sh
```

Completed jobs are cryptographically checked and skipped. Incomplete owned
jobs resume only from their own valid `latest.ckpt`; foreign, corrupt, or
configuration-mismatched directories fail closed before any peer starts.

## Outputs and logs

The default output root is:

```text
playground/Checkpoints/stage1_clean_supervision_ablation/
```

Per-job and master logs are stored below its `launcher_logs/` directory.
Important overrides are:

- `LIBERO_DATA_ROOT`
- `LIBERO_STAGE1_INIT_CHECKPOINT`
- `ROBOCASA_DATA_ROOT`
- `ROBOCASA_STAGE1_INIT_CHECKPOINT`
- `ROBOCASA_PURE_AE_OUTPUT_DIR`
- `STAGE1_ABLATION_CHECKPOINT_ROOT`
- `STAGE1_LAUNCHER_LOG_ROOT`
- `LIBERO_INTERMEDIATE_WEIGHT`
- `ROBOCASA_INTERMEDIATE_WEIGHT`
- `PYTHON_BIN`

The four-method comparison should not change either auxiliary weight after any
job has started. Resume validation rejects such a mixed matrix.

Eight processes create 64 data-loader workers in total. If an allocated node
cannot sustain that CPU or storage load, change `num_workers` once in both
canonical YAMLs before starting any run; never change it for only one method.
