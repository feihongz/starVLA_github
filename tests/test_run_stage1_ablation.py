from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts.stage1 import run_stage1_ablation as runner


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "dataset"
    data_root.mkdir(parents=True, exist_ok=True)
    init_checkpoint = tmp_path / "ae_init.ckpt"
    if not init_checkpoint.exists():
        init_checkpoint.write_bytes(b"fixed-ae-checkpoint")
    return data_root, init_checkpoint


def _jobs(
    tmp_path: Path,
    *,
    benchmark: str = "libero",
    methods: tuple[str, ...] = ("multiscale_base",),
    mode: str = "train",
    resume: bool = False,
    epochs: int | None = None,
) -> list[runner.RunJob]:
    data_root, init_checkpoint = _artifacts(tmp_path)
    return runner.build_jobs(
        benchmark=benchmark,
        methods=methods,
        seeds=(42,),
        gpus=("0", "1"),
        mode=mode,
        resume=resume,
        checkpoint_root=str(tmp_path / "checkpoints"),
        data_root=str(data_root),
        init_checkpoint=str(init_checkpoint),
        python_bin=sys.executable,
        epochs=epochs,
        environ={},
    )


def _write_latest_checkpoint(job: runner.RunJob, *, epoch: int) -> None:
    torch.save({"epoch": epoch}, job.output_dir / "latest.ckpt")


def test_method_mapping_changes_only_native_method_fields_and_smoke_is_bounded(
    tmp_path: Path,
) -> None:
    jobs = _jobs(
        tmp_path,
        methods=runner.METHODS,
        mode="smoke",
        epochs=999,
    )
    assert [job.gpu for job in jobs] == ["0", "1", "0", "1"]
    assert all("/libero/smoke/" in str(job.output_dir) for job in jobs)

    canonical = OmegaConf.load(
        runner.REPO_ROOT / runner.BENCHMARKS["libero"]["config"]
    )
    expected_group_weights = OmegaConf.to_container(
        canonical.loss.intermediate.group_weights, resolve=True
    )
    expected_scale_weights = canonical.loss.intermediate.scale_weights
    by_method = {job.method: job.config for job in jobs}

    assert by_method["multiscale_base"].loss.intermediate.mode == "none"
    assert by_method["multiscale_base"].loss.intermediate.weight == 0.0
    for method in ("full_target_time", "mint_paper_dct", "mtr"):
        cfg = by_method[method]
        assert cfg.loss.intermediate.mode == method
        assert cfg.loss.intermediate.weight == pytest.approx(0.02)
        assert cfg.loss.intermediate.scale_weights == expected_scale_weights
        assert OmegaConf.to_container(
            cfg.loss.intermediate.group_weights, resolve=True
        ) == expected_group_weights
    assert OmegaConf.to_container(
        by_method["mint_paper_dct"].loss.intermediate.spectral, resolve=True
    ) == {"formulation": "raw_dct_ii_mse", "normalization": "2_over_h"}
    assert OmegaConf.to_container(
        by_method["mtr"].loss.intermediate.spectral, resolve=True
    ) == {}

    for cfg in by_method.values():
        assert cfg.train.epochs == 1
        assert cfg.train.max_batches_per_epoch == 2
        assert cfg.train.batch_size <= 8
        assert cfg.train.num_workers == 0
        assert cfg.train.save_every_epochs == 1
        # Avoid the otherwise hidden 64-batch codebook-init pass.
        assert cfg.train.init_codebook_from_data_batches == 2


def test_robocasa_uses_shared_point_one_weight_and_canonical_empty_groups(
    tmp_path: Path,
) -> None:
    jobs = _jobs(
        tmp_path,
        benchmark="robocasa",
        methods=("full_target_time", "mint_paper_dct", "mtr"),
    )
    for job in jobs:
        assert job.config.loss.intermediate.weight == pytest.approx(0.1)
        assert OmegaConf.to_container(
            job.config.loss.intermediate.group_weights, resolve=True
        ) == {}


def test_pilot_weight_is_uniform_across_auxiliary_methods(tmp_path: Path) -> None:
    data_root, init_checkpoint = _artifacts(tmp_path)
    jobs = runner.build_jobs(
        benchmark="libero",
        methods=runner.METHODS,
        seeds=(42,),
        gpus=("0",),
        mode="train",
        resume=False,
        checkpoint_root=str(tmp_path / "checkpoints"),
        data_root=str(data_root),
        init_checkpoint=str(init_checkpoint),
        python_bin=sys.executable,
        intermediate_weight=0.05,
        environ={},
    )
    weights = {job.method: float(job.config.loss.intermediate.weight) for job in jobs}
    assert weights == {
        "multiscale_base": 0.0,
        "full_target_time": 0.05,
        "mint_paper_dct": 0.05,
        "mtr": 0.05,
    }
    for invalid in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(runner.RunnerError, match="finite and > 0"):
            runner.build_jobs(
                benchmark="libero",
                methods=("mtr",),
                seeds=(42,),
                gpus=("0",),
                mode="train",
                resume=False,
                intermediate_weight=invalid,
                environ={},
            )


def test_runtime_path_precedence_cli_then_benchmark_env_then_yaml(
    tmp_path: Path,
) -> None:
    cli_data, cli_init = _artifacts(tmp_path / "cli")
    env_data, env_init = _artifacts(tmp_path / "env")
    environment = {
        "LIBERO_DATA_ROOT": str(env_data),
        "LIBERO_STAGE1_INIT_CHECKPOINT": str(env_init),
        "STAGE1_CHECKPOINT_ROOT": str(tmp_path / "env_checkpoints"),
        "PYTHON_BIN": "env-python",
    }
    cli_job = runner.build_jobs(
        benchmark="libero",
        methods=("mtr",),
        seeds=(7,),
        gpus=("3",),
        mode="train",
        resume=False,
        checkpoint_root=str(tmp_path / "cli_checkpoints"),
        data_root=str(cli_data),
        init_checkpoint=str(cli_init),
        python_bin="cli-python",
        environ=environment,
    )[0]
    assert cli_job.config.data.data_root_dir == str(cli_data)
    assert cli_job.config.train.init_checkpoint == str(cli_init)
    assert cli_job.python_bin == "cli-python"
    assert cli_job.output_dir == (
        tmp_path / "cli_checkpoints/libero/mtr/seed_7"
    )

    env_job = runner.build_jobs(
        benchmark="libero",
        methods=("mtr",),
        seeds=(7,),
        gpus=("3",),
        mode="train",
        resume=False,
        environ=environment,
    )[0]
    assert env_job.config.data.data_root_dir == str(env_data)
    assert env_job.config.train.init_checkpoint == str(env_init)
    assert env_job.python_bin == "env-python"
    assert env_job.output_dir == (
        tmp_path / "env_checkpoints/libero/mtr/seed_7"
    )

    alias_environment = {
        "LIBERO_DATA_ROOT": str(env_data),
        "LIBERO_STAGE1_AE_CKPT": str(env_init),
    }
    alias_job = runner.build_jobs(
        benchmark="libero",
        methods=("mtr",),
        seeds=(7,),
        gpus=("3",),
        mode="train",
        resume=False,
        checkpoint_root=str(tmp_path / "alias_checkpoints"),
        environ=alias_environment,
    )[0]
    assert alias_job.config.train.init_checkpoint == str(env_init)


def test_dry_run_allows_missing_artifacts_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "never_created"
    result = runner.main(
        [
            "--benchmark",
            "libero",
            "--methods",
            "mtr",
            "--gpus",
            "0",
            "--mode",
            "smoke",
            "--dry-run",
            "--checkpoint-root",
            str(output_root),
            "--data-root",
            str(tmp_path / "missing_data"),
            "--init-checkpoint",
            str(tmp_path / "missing.ckpt"),
        ]
    )
    assert result == 0
    assert not output_root.exists()
    output = capsys.readouterr().out
    assert "loss:" in output
    assert "command:" in output
    assert "warning: data root directory not found" in output
    assert "warning: init checkpoint not found" in output


def test_preflight_checks_everything_before_any_output_write(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    missing_init = tmp_path / "missing.ckpt"
    jobs = runner.build_jobs(
        benchmark="libero",
        methods=("multiscale_base", "mtr"),
        seeds=(42,),
        gpus=("0",),
        mode="train",
        resume=False,
        checkpoint_root=str(tmp_path / "outputs"),
        data_root=str(data_root),
        init_checkpoint=str(missing_init),
        python_bin=sys.executable,
        environ={},
    )
    with pytest.raises(runner.RunnerError, match="preflight failed"):
        runner.preflight_jobs(jobs, resume=False)
    assert all(not job.output_dir.exists() for job in jobs)


def test_nonempty_output_is_refused_without_resume(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path)
    jobs[0].output_dir.mkdir(parents=True)
    (jobs[0].output_dir / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(runner.RunnerError, match="nonempty"):
        runner.preflight_jobs(jobs, resume=False)


def test_manifest_and_strict_own_latest_resume_contract(tmp_path: Path) -> None:
    initial = _jobs(tmp_path, epochs=3)
    runner.preflight_jobs(initial, resume=False)
    runner.prepare_jobs(initial, runner_invocation=("runner", "initial"))
    job = initial[0]
    _write_latest_checkpoint(job, epoch=1)

    manifest = json.loads(job.manifest_path.read_text())
    assert manifest["resolved_config_sha256"] == job.config_sha256
    assert manifest["init_checkpoint_sha256"] == runner.sha256_file(
        job.init_checkpoint
    )
    assert manifest["dataset"]["data_mix"] == "libero_all_q99"
    assert manifest["status"] == "planned"

    resumed = _jobs(tmp_path, resume=True, epochs=4)
    runner.preflight_jobs(resumed, resume=True)
    assert resumed[0].prior_manifest is not None
    assert resumed[0].prior_checkpoint_epoch == 1
    assert resumed[0].config.train.resume_checkpoint == str(
        Path(resumed[0].config.experiment.output_dir) / "latest.ckpt"
    )

    same_epochs = _jobs(tmp_path, resume=True, epochs=3)
    runner.preflight_jobs(same_epochs, resume=True)
    assert same_epochs[0].prior_checkpoint_epoch == 1

    decreased = _jobs(tmp_path, resume=True, epochs=2)
    with pytest.raises(runner.RunnerError, match="must not decrease"):
        runner.preflight_jobs(decreased, resume=True)

    _write_latest_checkpoint(job, epoch=2)
    completed = _jobs(tmp_path, resume=True, epochs=3)
    with pytest.raises(runner.RunnerError, match="already reached the target"):
        runner.preflight_jobs(completed, resume=True)

    _write_latest_checkpoint(job, epoch=1)

    prior_cfg = OmegaConf.load(job.materialized_config_path)
    prior_cfg.train.learning_rate = 0.123
    OmegaConf.save(prior_cfg, job.materialized_config_path)
    tampered = _jobs(tmp_path, resume=True, epochs=4)
    with pytest.raises(runner.RunnerError, match="config hash"):
        runner.preflight_jobs(tampered, resume=True)


def test_skipped_runner_shell_can_be_reused_but_training_artifacts_cannot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial = _jobs(
        tmp_path,
        methods=("multiscale_base", "mtr"),
    )
    for job in initial:
        job.gpu = "0"
    runner.preflight_jobs(initial, resume=False)
    runner.prepare_jobs(initial, runner_invocation=("runner", "initial"))

    monkeypatch.setattr(runner, "execute_job", lambda _job: 7)
    assert runner.run_jobs(initial) == 1
    skipped = initial[1]
    assert json.loads(skipped.manifest_path.read_text())["status"] == "skipped"
    assert {path.name for path in skipped.output_dir.iterdir()} == {
        runner.MANIFEST_NAME,
        runner.MATERIALIZED_CONFIG_NAME,
    }

    retry = _jobs(tmp_path, methods=("mtr",))
    runner.preflight_jobs(retry, resume=False)
    assert retry[0].reclaim_skipped_shell is True
    runner.prepare_jobs(retry, runner_invocation=("runner", "retry"))
    assert json.loads(retry[0].manifest_path.read_text())["status"] == "planned"

    _write_latest_checkpoint(retry[0], epoch=0)
    unsafe_retry = _jobs(tmp_path, methods=("mtr",))
    with pytest.raises(runner.RunnerError, match="nonempty"):
        runner.preflight_jobs(unsafe_retry, resume=False)


def test_worker_failure_produces_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jobs = _jobs(
        tmp_path,
        methods=("multiscale_base", "mtr"),
    )
    calls: list[str] = []

    def fake_execute(job: runner.RunJob) -> int:
        calls.append(job.method)
        return 7 if job.method == "mtr" else 0

    monkeypatch.setattr(runner, "execute_job", fake_execute)
    assert runner.run_jobs(jobs) == 1
    assert sorted(calls) == ["mtr", "multiscale_base"]


def test_execute_job_prepends_repo_to_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _jobs(tmp_path)[0]
    captured: dict[str, object] = {}
    monkeypatch.setenv("PYTHONPATH", "existing-python-path")
    monkeypatch.setattr(runner, "_update_manifest", lambda *_args, **_kwargs: None)

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.execute_job(job) == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(runner.REPO_ROOT),
        "existing-python-path",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["cwd"] == runner.REPO_ROOT


def test_execute_job_passes_inherited_launcher_lock_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _jobs(tmp_path)[0]
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "_update_manifest", lambda *_args, **_kwargs: None)

    def fake_run(command, *, cwd, env, check, pass_fds):
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            check=check,
            pass_fds=pass_fds,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    lock_path = tmp_path / "launcher.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        monkeypatch.setenv("STAGE1_LAUNCH_LOCK_FD", str(lock.fileno()))
        assert runner.execute_job(job) == 0
        assert captured["pass_fds"] == (lock.fileno(),)


@pytest.mark.parametrize(
    ("value", "message"),
    [("", "at least one"), ("0,0", "duplicates"), ("0,gpu1", "GPU/MIG UUIDs")],
)
def test_gpu_parser_rejects_invalid_values(value: str, message: str) -> None:
    with pytest.raises(runner.RunnerError, match=message):
        runner.parse_gpus(value)


def test_gpu_parser_accepts_scheduler_uuid_tokens() -> None:
    assert runner.parse_gpus("GPU-a1b2,MIG-GPU-c3d4/1/2") == [
        "GPU-a1b2",
        "MIG-GPU-c3d4/1/2",
    ]
