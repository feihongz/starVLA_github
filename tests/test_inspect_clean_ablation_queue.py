from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scripts.stage1 import inspect_clean_ablation_queue as inspector
from scripts.stage1 import run_stage1_ablation as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PURE_AE_CONFIG = (
    REPO_ROOT
    / "examples/Robocasa_tabletop/train_files/"
    "train_var_stage1_robocasa_gr1_pure_ae_e64.yaml"
)


def _history(length: int) -> list[dict[str, int]]:
    return [{"epoch": epoch} for epoch in range(length)]


def _write_checkpoint(path: Path, *, epoch: int) -> None:
    history = _history(epoch + 1)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_config": {},
            "action_spec": {},
            "history": history,
            "stage1_config": {},
        },
        path,
    )


def _write_pure_checkpoint(path: Path, *, epoch: int, output_dir: Path) -> None:
    config = OmegaConf.to_container(
        OmegaConf.load(output_dir / "config.yaml"),
        resolve=True,
    )
    assert isinstance(config, dict)
    model_config = dict(inspector.EXPECTED_INIT_ARCHITECTURE["robocasa"])
    action_spec = {
        "action_dim": 29,
        "horizon": 16,
        "token_order": "scale_major",
    }
    history = _history(epoch + 1)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {},
            "model_config": model_config,
            "action_spec": action_spec,
            "history": history,
            "stage1_config": config,
        },
        path,
    )


def _prepare_pure_ae_shell(output_dir: Path, data_root: Path, *, epoch: int) -> None:
    output_dir.mkdir(parents=True)
    cfg = OmegaConf.load(PURE_AE_CONFIG)
    cfg.experiment.output_dir = str(output_dir)
    cfg.data.data_root_dir = str(data_root)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    (output_dir / "action_spec.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "starvla_base_config.yaml").write_text("{}\n", encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps(_history(epoch + 1)),
        encoding="utf-8",
    )
    _write_pure_checkpoint(
        output_dir / "latest.ckpt",
        epoch=epoch,
        output_dir=output_dir,
    )
    _write_pure_checkpoint(
        output_dir / "best_recon.ckpt",
        epoch=epoch,
        output_dir=output_dir,
    )


def test_pure_ae_complete_is_reused_but_partial_requires_resume(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    complete = tmp_path / "complete"
    _prepare_pure_ae_shell(complete, data_root, epoch=49)
    _write_pure_checkpoint(
        complete / "final.ckpt",
        epoch=49,
        output_dir=complete,
    )
    assert (
        inspector.inspect_pure_ae(
            config_path=PURE_AE_CONFIG,
            output_dir=complete,
            data_root=data_root,
            init_checkpoint=complete / "best_recon.ckpt",
            resume_enabled=False,
        )
        == "ready"
    )

    partial = tmp_path / "partial"
    _prepare_pure_ae_shell(partial, data_root, epoch=3)
    with pytest.raises(inspector.QueueStateError, match="set RESUME_QUEUE=1"):
        inspector.inspect_pure_ae(
            config_path=PURE_AE_CONFIG,
            output_dir=partial,
            data_root=data_root,
            init_checkpoint=partial / "best_recon.ckpt",
            resume_enabled=False,
        )
    assert (
        inspector.inspect_pure_ae(
            config_path=PURE_AE_CONFIG,
            output_dir=partial,
            data_root=data_root,
            init_checkpoint=partial / "best_recon.ckpt",
            resume_enabled=True,
        )
        == "resume"
    )


def _build_job(
    tmp_path: Path,
    *,
    init_checkpoint: Path,
    intermediate_weight: float = 0.02,
    resume: bool = False,
) -> runner.RunJob:
    data_root = tmp_path / "libero_data"
    data_root.mkdir(exist_ok=True)
    return runner.build_jobs(
        benchmark="libero",
        methods=("mtr",),
        seeds=(42,),
        gpus=("0",),
        mode="train",
        resume=resume,
        checkpoint_root=str(tmp_path / "outputs"),
        data_root=str(data_root),
        init_checkpoint=str(init_checkpoint),
        python_bin=sys.executable,
        epochs=2,
        intermediate_weight=intermediate_weight,
        environ={},
    )[0]


def _prepare_owned_job(job: runner.RunJob) -> None:
    runner.preflight_jobs([job], resume=False)
    runner.prepare_jobs([job], runner_invocation=["test"])


def test_ablation_inspector_verifies_complete_identity_and_hashes(
    tmp_path: Path,
) -> None:
    init_checkpoint = tmp_path / "init.ckpt"
    init_checkpoint.write_bytes(b"fixed init")
    job = _build_job(tmp_path, init_checkpoint=init_checkpoint)
    _prepare_owned_job(job)
    history = _history(2)
    (job.output_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    _write_checkpoint(job.output_dir / "latest.ckpt", epoch=1)
    _write_checkpoint(job.output_dir / "final.ckpt", epoch=1)
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "succeeded", "exit_code": 0})
    job.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert inspector.inspect_ablation(job, resume_enabled=True) == "skip"

    changed = _build_job(
        tmp_path,
        init_checkpoint=init_checkpoint,
        intermediate_weight=0.03,
    )
    with pytest.raises(inspector.QueueStateError, match="config hash mismatch"):
        inspector.inspect_ablation(changed, resume_enabled=True)


def test_ablation_inspector_preflights_fresh_and_validates_resume(
    tmp_path: Path,
) -> None:
    missing_init = tmp_path / "missing.ckpt"
    missing_job = _build_job(tmp_path, init_checkpoint=missing_init)
    with pytest.raises(runner.RunnerError, match="init checkpoint not found"):
        inspector.inspect_ablation(missing_job, resume_enabled=False)

    init_checkpoint = tmp_path / "init.ckpt"
    init_checkpoint.write_bytes(b"fixed init")
    job = _build_job(tmp_path, init_checkpoint=init_checkpoint)
    _prepare_owned_job(job)
    history = _history(1)
    (job.output_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    _write_checkpoint(job.output_dir / "latest.ckpt", epoch=0)
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "failed", "exit_code": 1})
    job.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert inspector.inspect_ablation(job, resume_enabled=True) == "resume"

    resume_job = _build_job(
        tmp_path,
        init_checkpoint=init_checkpoint,
        resume=True,
    )
    runner.preflight_jobs([resume_job], resume=True)
    runner.prepare_jobs([resume_job], runner_invocation=["test", "--resume"])
    completed_history = _history(2)
    (resume_job.output_dir / "history.json").write_text(
        json.dumps(completed_history),
        encoding="utf-8",
    )
    _write_checkpoint(resume_job.output_dir / "latest.ckpt", epoch=1)
    _write_checkpoint(resume_job.output_dir / "final.ckpt", epoch=1)
    completed_manifest = json.loads(
        resume_job.manifest_path.read_text(encoding="utf-8")
    )
    completed_manifest.update({"status": "succeeded", "exit_code": 0})
    resume_job.manifest_path.write_text(
        json.dumps(completed_manifest),
        encoding="utf-8",
    )
    assert inspector.inspect_ablation(job, resume_enabled=True) == "skip"


def test_statistics_cache_validation_is_strict_and_read_only(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "libero"
    payload = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": {"action": {"mean": [0.0]}},
    }
    for dataset in inspector.LIBERO_DATASETS:
        meta = data_root / dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "stats_gr00t.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    assert (
        inspector.inspect_stats_caches(
            benchmark="libero",
            data_root=data_root,
        )
        == "ready"
    )

    corrupt = data_root / inspector.LIBERO_DATASETS[0] / "meta/stats_gr00t.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(inspector.QueueStateError, match="1/4 datasets"):
        inspector.inspect_stats_caches(
            benchmark="libero",
            data_root=data_root,
        )
