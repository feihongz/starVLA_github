"""Read-only recovery-state checks for the controlled Stage1 launchers."""

from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

try:
    from scripts.stage1 import run_stage1_ablation as runner
except ModuleNotFoundError:
    import run_stage1_ablation as runner  # type: ignore[no-redef]


class QueueStateError(RuntimeError):
    pass


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise QueueStateError(f"cannot read checkpoint {path}: {exc}") from exc
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "action_spec",
        "history",
        "stage1_config",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        missing = sorted(required.difference(payload if isinstance(payload, Mapping) else {}))
        raise QueueStateError(f"checkpoint {path} is not a complete Stage1 artifact; missing={missing}")
    return payload


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    path: Path,
    *,
    target_epochs: int,
    final: bool,
) -> int:
    epoch = payload["epoch"]
    history = payload["history"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < target_epochs:
        raise QueueStateError(f"checkpoint {path} has invalid epoch {epoch!r}")
    if not isinstance(history, list) or len(history) != epoch + 1:
        raise QueueStateError(
            f"checkpoint {path} history length must be epoch+1 ({epoch + 1}), got "
            f"{len(history) if isinstance(history, list) else type(history).__name__}"
        )
    for expected_epoch, record in enumerate(history):
        if not isinstance(record, Mapping) or record.get("epoch") != expected_epoch:
            raise QueueStateError(
                f"checkpoint {path} has non-contiguous history at index {expected_epoch}"
            )
    if final and epoch != target_epochs - 1:
        raise QueueStateError(
            f"final checkpoint {path} must be epoch {target_epochs - 1}, got {epoch}"
        )
    return epoch


def _load_checkpoint(path: Path, *, target_epochs: int, final: bool) -> int:
    return _validate_checkpoint_payload(
        _read_checkpoint(path),
        path,
        target_epochs=target_epochs,
        final=final,
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QueueStateError(f"cannot read JSON {path}: {exc}") from exc


def _validate_history_json(output_dir: Path, expected_length: int) -> None:
    history = _read_json(output_dir / "history.json")
    if not isinstance(history, list) or len(history) != expected_length:
        raise QueueStateError(
            f"{output_dir / 'history.json'} must contain {expected_length} epochs"
        )
    if any(
        not isinstance(record, Mapping) or record.get("epoch") != index
        for index, record in enumerate(history)
    ):
        raise QueueStateError(f"{output_dir / 'history.json'} is not contiguous")


def _plain(cfg: Any) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if not isinstance(value, dict):
        raise QueueStateError("expected a mapping config")
    return value


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise QueueStateError(f"saved pure-AE config is missing {dotted}")
        current = current[key]
    return current


PURE_AE_CRITICAL_KEYS = (
    "experiment.name",
    "experiment.output_dir",
    "experiment.seed",
    "data.data_root_dir",
    "data.data_mix",
    "data.expected_action_horizon",
    "data.expected_action_dim",
    "model.embed_dim",
    "model.codebook_size",
    "model.scales",
    "model.decoder_head_type",
    "model.quantization_mode",
    "model.use_time_embedding",
    "model.use_action_type_embedding",
    "loss.recon_weight",
    "loss.vel_weight",
    "loss.vq_weight",
    "train.epochs",
    "train.batch_size",
    "train.learning_rate",
)

EXPECTED_INIT_ARCHITECTURE: dict[str, dict[str, Any]] = {
    "libero": {
        "action_dim": 7,
        "seq_len": 8,
        "embed_dim": 32,
        "codebook_size": 512,
        "scales": [1, 2, 4, 8],
        "quantization_mode": "none",
        "decoder_head_type": "plain",
    },
    "robocasa": {
        "action_dim": 29,
        "seq_len": 16,
        "embed_dim": 64,
        "codebook_size": 512,
        "scales": [1, 2, 4, 8, 16],
        "quantization_mode": "none",
        "decoder_head_type": "plain",
    },
}

LIBERO_DATASETS = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)


def inspect_init_checkpoint(path: Path, *, benchmark: str) -> str:
    if benchmark not in EXPECTED_INIT_ARCHITECTURE:
        raise QueueStateError(f"unsupported init benchmark: {benchmark}")
    payload = _read_checkpoint(path)
    _validate_checkpoint_payload(
        payload,
        path,
        target_epochs=1_000_000,
        final=False,
    )
    model_config = payload["model_config"]
    action_spec = payload["action_spec"]
    model_state = payload["model_state_dict"]
    if not isinstance(model_config, Mapping) or not isinstance(action_spec, Mapping):
        raise QueueStateError(f"checkpoint {path} has invalid model_config/action_spec")
    if not isinstance(model_state, Mapping) or not model_state:
        raise QueueStateError(f"checkpoint {path} has an empty model_state_dict")
    for key, expected in EXPECTED_INIT_ARCHITECTURE[benchmark].items():
        if model_config.get(key) != expected:
            raise QueueStateError(
                f"checkpoint {path} model_config.{key}={model_config.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key, expected in (
        ("action_dim", EXPECTED_INIT_ARCHITECTURE[benchmark]["action_dim"]),
        ("horizon", EXPECTED_INIT_ARCHITECTURE[benchmark]["seq_len"]),
        ("token_order", "scale_major"),
    ):
        if action_spec.get(key) != expected:
            raise QueueStateError(
                f"checkpoint {path} action_spec.{key}={action_spec.get(key)!r}, "
                f"expected {expected!r}"
            )
    return "ready"


def _validate_checkpoint_config(path: Path, expected: Mapping[str, Any]) -> None:
    checkpoint_config = _plain(_read_checkpoint(path)["stage1_config"])
    for key in PURE_AE_CRITICAL_KEYS:
        if _nested(checkpoint_config, key) != _nested(expected, key):
            raise QueueStateError(
                f"checkpoint {path} config mismatch at {key}: "
                f"{_nested(checkpoint_config, key)!r} != {_nested(expected, key)!r}"
            )


def _robocasa_datasets(registry_path: Path) -> list[str]:
    try:
        module = ast.parse(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QueueStateError(f"cannot read RoboCasa registry {registry_path}: {exc}") from exc
    datasets: list[str] | None = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "LOCAL_GR1_UNIFIED_1000_DATASETS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                datasets = value
            break
    if datasets is None or len(datasets) != 24 or len(set(datasets)) != 24:
        raise QueueStateError("RoboCasa registry must contain 24 unique local datasets")
    return datasets


def inspect_robocasa_data_root(*, data_root: Path, registry_path: Path) -> str:
    if not data_root.is_dir():
        raise QueueStateError(f"RoboCasa data root is missing: {data_root}")
    datasets = _robocasa_datasets(registry_path)
    missing = [name for name in datasets if not (data_root / name).is_dir()]
    if missing:
        raise QueueStateError(
            f"RoboCasa data root is missing {len(missing)}/24 datasets: {missing}"
        )
    return "ready"


def inspect_stats_caches(
    *,
    benchmark: str,
    data_root: Path,
    registry_path: Path | None = None,
) -> str:
    if benchmark == "libero":
        datasets = list(LIBERO_DATASETS)
    elif benchmark == "robocasa":
        if registry_path is None:
            raise QueueStateError("RoboCasa statistics validation requires a registry")
        datasets = _robocasa_datasets(registry_path)
    else:
        raise QueueStateError(f"unsupported statistics benchmark: {benchmark}")
    failures: list[str] = []
    for dataset in datasets:
        stats_path = data_root / dataset / "meta/stats_gr00t.json"
        tmp_path = stats_path.with_suffix(".tmp")
        if tmp_path.exists():
            failures.append(f"{dataset}: stale concurrent-writer file {tmp_path.name}")
            continue
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{dataset}: cannot parse {stats_path} ({exc})")
            continue
        if (
            not isinstance(payload, Mapping)
            or payload.get("__format_version") != 2
            or payload.get("__cache_config") != {"mode": "abs"}
            or not isinstance(payload.get("statistics"), Mapping)
            or not payload["statistics"]
        ):
            failures.append(f"{dataset}: incompatible stats_gr00t cache")
    if failures:
        raise QueueStateError(
            f"{benchmark} statistics cache validation failed for "
            f"{len(failures)}/{len(datasets)} datasets: {failures}"
        )
    return "ready"


def _validate_pure_ae_identity(
    config_path: Path,
    *,
    output_dir: Path,
    data_root: Path,
) -> None:
    try:
        saved = _plain(OmegaConf.load(output_dir / "config.yaml"))
        canonical = _plain(OmegaConf.load(config_path))
    except Exception as exc:
        if isinstance(exc, QueueStateError):
            raise
        raise QueueStateError(f"cannot read pure-AE config: {exc}") from exc
    canonical["experiment"]["output_dir"] = str(output_dir)
    canonical["data"]["data_root_dir"] = str(data_root)
    for key in PURE_AE_CRITICAL_KEYS:
        if _nested(saved, key) != _nested(canonical, key):
            raise QueueStateError(
                f"saved pure-AE config mismatch at {key}: "
                f"{_nested(saved, key)!r} != {_nested(canonical, key)!r}"
            )


def inspect_pure_ae(
    *,
    config_path: Path,
    output_dir: Path,
    data_root: Path,
    init_checkpoint: Path,
    resume_enabled: bool,
) -> str:
    if output_dir.is_symlink():
        raise QueueStateError(f"pure-AE output must not be a symlink: {output_dir}")
    if not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())):
        return "fresh"
    if not output_dir.is_dir():
        raise QueueStateError(f"pure-AE output is not a directory: {output_dir}")
    if init_checkpoint != output_dir / "best_recon.ckpt":
        raise QueueStateError("pure-AE init must be OUTPUT_DIR/best_recon.ckpt")
    _validate_pure_ae_identity(config_path, output_dir=output_dir, data_root=data_root)
    saved_config = _plain(OmegaConf.load(output_dir / "config.yaml"))
    for required in ("action_spec.json", "starvla_base_config.yaml", "history.json"):
        if not (output_dir / required).is_file():
            raise QueueStateError(f"pure-AE artifact is missing: {output_dir / required}")
    if not init_checkpoint.is_file():
        raise QueueStateError(f"pure-AE best checkpoint is missing: {init_checkpoint}")
    inspect_init_checkpoint(init_checkpoint, benchmark="robocasa")
    _validate_checkpoint_config(init_checkpoint, saved_config)
    final = output_dir / "final.ckpt"
    if final.exists():
        if not final.is_file():
            raise QueueStateError(f"pure-AE final checkpoint is not a file: {final}")
        _load_checkpoint(final, target_epochs=30, final=True)
        _validate_checkpoint_config(final, saved_config)
        _validate_history_json(output_dir, 30)
        return "ready"
    if not resume_enabled:
        raise QueueStateError(
            f"pure-AE output is partial; set RESUME_QUEUE=1 to recover: {output_dir}"
        )
    latest = output_dir / "latest.ckpt"
    if not latest.is_file():
        raise QueueStateError(f"partial pure-AE output has no latest.ckpt: {output_dir}")
    epoch = _load_checkpoint(latest, target_epochs=30, final=False)
    _validate_checkpoint_config(latest, saved_config)
    if epoch >= 29:
        raise QueueStateError("pure-AE latest reached epoch 29 but final.ckpt is missing")
    _validate_history_json(output_dir, epoch + 1)
    return "resume"


def _validate_completed_ablation(job: runner.RunJob) -> None:
    try:
        manifest = _read_json(job.manifest_path)
        prior_cfg = OmegaConf.load(job.materialized_config_path)
    except Exception as exc:
        if isinstance(exc, QueueStateError):
            raise
        raise QueueStateError(f"cannot inspect completed {job.label}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise QueueStateError(f"invalid manifest for {job.label}")
    for key, expected in (
        ("benchmark", job.benchmark),
        ("method", job.method),
        ("seed", job.seed),
        ("mode", job.mode),
        ("status", "succeeded"),
        ("exit_code", 0),
    ):
        if manifest.get(key) != expected:
            raise QueueStateError(
                f"completed manifest mismatch for {job.label}: {key}={manifest.get(key)!r}"
            )
    actual_hash = runner.config_sha256(prior_cfg)
    if runner._manifest_config_hash(manifest) != actual_hash:
        raise QueueStateError(f"completed manifest/config hash mismatch for {job.label}")
    prior_compatible_hash = runner.config_sha256(
        runner._resume_compatible_config(prior_cfg)
    )
    expected_compatible_hash = runner.config_sha256(
        runner._resume_compatible_config(job.config)
    )
    if prior_compatible_hash != expected_compatible_hash:
        raise QueueStateError(f"completed config hash mismatch for {job.label}")
    if not job.init_checkpoint.is_file():
        raise QueueStateError(f"completed job init is missing: {job.init_checkpoint}")
    init_hash = runner.sha256_file(job.init_checkpoint)
    if runner._manifest_init_hash(manifest) != init_hash:
        raise QueueStateError(f"completed init hash mismatch for {job.label}")
    final = job.output_dir / "final.ckpt"
    if not final.is_file():
        raise QueueStateError(f"succeeded job has no final.ckpt: {job.label}")
    epochs = int(job.config.train.epochs)
    _load_checkpoint(final, target_epochs=epochs, final=True)
    _validate_history_json(job.output_dir, epochs)


def inspect_ablation(job: runner.RunJob, *, resume_enabled: bool) -> str:
    if job.output_dir.is_symlink():
        raise QueueStateError(f"job output must not be a symlink: {job.output_dir}")
    if not job.output_dir.exists() or (
        job.output_dir.is_dir() and not any(job.output_dir.iterdir())
    ):
        runner.preflight_jobs([job], resume=False)
        return "fresh"
    if not job.output_dir.is_dir():
        raise QueueStateError(f"job output is not a directory: {job.output_dir}")
    if not resume_enabled:
        raise QueueStateError(
            f"job output is non-empty; set RESUME_QUEUE=1 to inspect recovery: {job.output_dir}"
        )
    if runner._is_reclaimable_skipped_shell(job):
        return "fresh"
    if not job.manifest_path.is_file() or not job.materialized_config_path.is_file():
        raise QueueStateError(f"foreign/incomplete runner ownership artifacts: {job.output_dir}")
    manifest = _read_json(job.manifest_path)
    if isinstance(manifest, Mapping) and manifest.get("status") == "succeeded":
        _validate_completed_ablation(job)
        return "skip"
    latest = job.output_dir / "latest.ckpt"
    if not latest.is_file():
        raise QueueStateError(f"owned incomplete job has no latest.ckpt: {job.output_dir}")
    resume_job = runner.build_jobs(
        benchmark=job.benchmark,
        methods=(job.method,),
        seeds=(job.seed,),
        gpus=(job.gpu,),
        mode=job.mode,
        resume=True,
        checkpoint_root=str(job.output_dir.parents[2] if job.mode == "smoke" else job.output_dir.parents[1]),
        data_root=str(job.data_root),
        init_checkpoint=str(job.init_checkpoint),
        python_bin=job.python_bin,
        epochs=int(job.config.train.epochs),
        intermediate_weight=(
            float(job.config.loss.intermediate.weight)
            if job.method != "multiscale_base"
            else float(runner.BENCHMARKS[job.benchmark]["weight"])
        ),
        environ={},
    )[0]
    try:
        runner.preflight_jobs([resume_job], resume=True)
    except runner.RunnerError as exc:
        raise QueueStateError(str(exc)) from exc
    epoch = _load_checkpoint(
        latest,
        target_epochs=int(job.config.train.epochs),
        final=False,
    )
    _validate_history_json(job.output_dir, epoch + 1)
    return "resume"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    pure = subparsers.add_parser("pure-ae")
    pure.add_argument("--config", required=True)
    pure.add_argument("--output-dir", required=True)
    pure.add_argument("--data-root", required=True)
    pure.add_argument("--init-checkpoint", required=True)
    init = subparsers.add_parser("init-checkpoint")
    init.add_argument("--benchmark", required=True, choices=tuple(runner.BENCHMARKS))
    init.add_argument("--path", required=True)
    data = subparsers.add_parser("robocasa-data-root")
    data.add_argument("--data-root", required=True)
    data.add_argument("--registry", required=True)
    stats = subparsers.add_parser("stats-caches")
    stats.add_argument("--benchmark", required=True, choices=tuple(runner.BENCHMARKS))
    stats.add_argument("--data-root", required=True)
    stats.add_argument("--registry")
    ablation = subparsers.add_parser("ablation")
    ablation.add_argument("--benchmark", required=True, choices=tuple(runner.BENCHMARKS))
    ablation.add_argument("--method", required=True, choices=runner.METHODS)
    ablation.add_argument("--mode", required=True, choices=("smoke", "train"))
    ablation.add_argument("--checkpoint-root", required=True)
    ablation.add_argument("--data-root", required=True)
    ablation.add_argument("--init-checkpoint", required=True)
    ablation.add_argument("--python-bin", required=True)
    ablation.add_argument("--intermediate-weight", required=True, type=float)
    parser.add_argument("--resume-enabled", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.kind == "pure-ae":
            action = inspect_pure_ae(
                config_path=Path(args.config).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                data_root=Path(args.data_root).resolve(),
                init_checkpoint=Path(args.init_checkpoint).resolve(),
                resume_enabled=args.resume_enabled,
            )
        elif args.kind == "init-checkpoint":
            action = inspect_init_checkpoint(
                Path(args.path).resolve(),
                benchmark=args.benchmark,
            )
        elif args.kind == "robocasa-data-root":
            action = inspect_robocasa_data_root(
                data_root=Path(args.data_root).resolve(),
                registry_path=Path(args.registry).resolve(),
            )
        elif args.kind == "stats-caches":
            action = inspect_stats_caches(
                benchmark=args.benchmark,
                data_root=Path(args.data_root).resolve(),
                registry_path=(
                    Path(args.registry).resolve()
                    if args.registry is not None
                    else None
                ),
            )
        else:
            job = runner.build_jobs(
                benchmark=args.benchmark,
                methods=(args.method,),
                seeds=(42,),
                gpus=("0",),
                mode=args.mode,
                resume=False,
                checkpoint_root=args.checkpoint_root,
                data_root=args.data_root,
                init_checkpoint=args.init_checkpoint,
                python_bin=args.python_bin,
                epochs=50 if args.mode == "train" else None,
                intermediate_weight=args.intermediate_weight,
                environ={},
            )[0]
            action = inspect_ablation(job, resume_enabled=args.resume_enabled)
    except (QueueStateError, runner.RunnerError) as exc:
        print(f"queue-state error: {exc}", file=sys.stderr)
        return 2
    print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
