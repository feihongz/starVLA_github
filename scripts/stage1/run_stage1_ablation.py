"""Launch controlled VAR Stage1 intermediate-supervision ablations.

The launcher deliberately materializes one complete configuration per run.  It
does not import the trainer (and therefore does not initialize PyTorch/CUDA), so
all jobs can be validated before any output directory is created or process is
started.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

try:  # Works when imported as a repository module.
    from scripts.stage1.validate_ablation_configs import (
        AblationConfigError,
        materialize_ablation_config,
        materialize_ablation_configs,
    )
except ModuleNotFoundError:  # Works when invoked directly by file path.
    from validate_ablation_configs import (  # type: ignore[no-redef]
        AblationConfigError,
        materialize_ablation_config,
        materialize_ablation_configs,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ENTRYPOINT = Path("starVLA/training/train_var_stage1.py")
MANIFEST_NAME = "run_manifest.json"
MATERIALIZED_CONFIG_NAME = "materialized_config.yaml"
SCHEMA_VERSION = "stage1_ablation_run.v1"

METHODS = (
    "multiscale_base",
    "full_target_time",
    "mint_paper_dct",
    "mtr",
)

GPU_TOKEN_PATTERN = re.compile(
    r"^(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9_./-]+)$"
)

BENCHMARKS: dict[str, dict[str, Any]] = {
    "libero": {
        "config": Path(
            "examples/LIBERO/train_files/"
            "train_var_stage1_libero_clean_supervision_ablation.yaml"
        ),
        "weight": 0.02,
        "data_env": (
            "LIBERO_DATA_ROOT",
            "STAGE1_LIBERO_DATA_ROOT",
            "LIBERO_STAGE1_DATA_ROOT",
        ),
        "init_env": (
            "LIBERO_STAGE1_INIT_CHECKPOINT",
            "LIBERO_STAGE1_AE_CKPT",
            "STAGE1_LIBERO_INIT_CHECKPOINT",
            "LIBERO_INIT_CHECKPOINT",
        ),
    },
    "robocasa": {
        "config": Path(
            "examples/Robocasa_tabletop/train_files/"
            "train_var_stage1_robocasa_clean_supervision_ablation.yaml"
        ),
        "weight": 0.1,
        "data_env": (
            "ROBOCASA_DATA_ROOT",
            "STAGE1_ROBOCASA_DATA_ROOT",
            "ROBOCASA_STAGE1_DATA_ROOT",
        ),
        "init_env": (
            "ROBOCASA_STAGE1_INIT_CHECKPOINT",
            "ROBOCASA_STAGE1_AE_CKPT",
            "STAGE1_ROBOCASA_INIT_CHECKPOINT",
            "ROBOCASA_INIT_CHECKPOINT",
        ),
    },
}


class RunnerError(RuntimeError):
    """A user-facing launcher validation error."""


@dataclass
class RunJob:
    benchmark: str
    method: str
    seed: int
    gpu: str
    mode: str
    canonical_config: Path
    config: DictConfig
    output_dir: Path
    python_bin: str
    command: list[str]
    init_checkpoint: Path
    data_root: Path
    config_sha256: str
    init_sha256: str | None = None
    prior_manifest: dict[str, Any] | None = None
    prior_epochs: int | None = None
    prior_checkpoint_epoch: int | None = None
    reclaim_skipped_shell: bool = False
    artifact_warnings: list[str] = field(default_factory=list)

    @property
    def materialized_config_path(self) -> Path:
        return self.output_dir / MATERIALIZED_CONFIG_NAME

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_NAME

    @property
    def label(self) -> str:
        return f"{self.benchmark}/{self.method}/seed_{self.seed}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        value = OmegaConf.to_container(cfg, resolve=True)
    else:
        value = copy.deepcopy(dict(cfg))
    if not isinstance(value, dict):
        raise RunnerError("Expected the Stage1 YAML root to be a mapping.")
    return value


def config_sha256(cfg: DictConfig | Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _first_env(names: Sequence[str], environ: Mapping[str, str]) -> str | None:
    for name in names:
        value = environ.get(name)
        if value:
            return value
    return None


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise RunnerError("--gpus must contain at least one GPU identifier.")
    if len(gpus) != len(set(gpus)):
        raise RunnerError(f"--gpus contains duplicates: {value!r}.")
    for gpu in gpus:
        if GPU_TOKEN_PATTERN.fullmatch(gpu) is None:
            raise RunnerError(
                "GPU identifiers must be non-negative integer indices or CUDA "
                f"GPU/MIG UUIDs, got {gpu!r}."
            )
    return gpus


def method_intermediate(
    method: str,
    benchmark: str,
    intermediate_weight: float | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise RunnerError(f"Unknown method {method!r}; expected one of {METHODS}.")
    weight = (
        float(BENCHMARKS[benchmark]["weight"])
        if intermediate_weight is None
        else float(intermediate_weight)
    )
    if not math.isfinite(weight) or weight <= 0:
        raise RunnerError(f"Intermediate weight must be finite and > 0, got {weight}.")
    if method == "multiscale_base":
        return {"mode": "none", "weight": 0.0, "spectral": {}}
    if method == "mint_paper_dct":
        return {
            "mode": "mint_paper_dct",
            "weight": weight,
            "spectral": {
                "formulation": "raw_dct_ii_mse",
                "normalization": "2_over_h",
            },
        }
    return {"mode": method, "weight": weight, "spectral": {}}


def _resolve_python_bin(cli_value: str | None, environ: Mapping[str, str]) -> str:
    return cli_value or environ.get("PYTHON_BIN") or sys.executable


def _resolve_executable(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.is_absolute() or len(path.parts) > 1:
        resolved = _repo_path(path)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
        return None
    return shutil.which(value)


def _benchmark_output_root(
    benchmark: str,
    canonical_cfg: DictConfig,
    cli_checkpoint_root: str | None,
    environ: Mapping[str, str],
) -> Path:
    explicit = cli_checkpoint_root or environ.get("STAGE1_CHECKPOINT_ROOT")
    if explicit:
        root = Path(explicit).expanduser()
        # STAGE1_CHECKPOINT_ROOT is shared by both benchmarks.  Avoid adding the
        # benchmark twice when the caller already supplied a benchmark root.
        return root if root.name.lower() == benchmark else root / benchmark
    canonical_output = Path(str(canonical_cfg.experiment.output_dir)).expanduser()
    return canonical_output.parent


def _run_output_dir(root: Path, mode: str, method: str, seed: int) -> Path:
    # Smoke artifacts are isolated so the required smoke pass cannot make the
    # subsequent formal run look like an accidental overwrite.
    if mode == "smoke":
        root = root / "smoke"
    return root / method / f"seed_{seed}"


def _resolve_runtime_value(
    cli_value: str | None,
    env_names: Sequence[str],
    yaml_value: Any,
    environ: Mapping[str, str],
) -> str:
    value = cli_value or _first_env(env_names, environ) or str(yaml_value)
    if not value:
        raise RunnerError("Resolved runtime path is empty.")
    return value


def materialize_method_config(
    canonical_cfg: DictConfig,
    *,
    benchmark: str,
    method: str,
    seed: int,
    mode: str,
    output_dir: Path,
    data_root: str,
    init_checkpoint: str,
    epochs: int | None,
    resume: bool,
    intermediate_weight: float | None = None,
) -> DictConfig:
    """Return a detached resolved config for one controlled run."""

    suffix = "_smoke" if mode == "smoke" else ""
    try:
        value = materialize_ablation_config(
            canonical_cfg,
            method,
            benchmark=benchmark,
            intermediate_weight=(
                float(BENCHMARKS[benchmark]["weight"])
                if intermediate_weight is None
                else float(intermediate_weight)
            ),
            experiment_name=f"var_stage1_{benchmark}_{method}_seed{seed}{suffix}",
            output_dir=str(output_dir),
        )
    except AblationConfigError as exc:
        raise RunnerError(str(exc)) from exc
    cfg = OmegaConf.create(value)
    # Keep an explicit empty mapping in materialized YAMLs for the three
    # time-domain/non-spectral rows.
    if "spectral" not in cfg.loss.intermediate:
        cfg.loss.intermediate.spectral = OmegaConf.create({})

    cfg.experiment.seed = int(seed)
    cfg.data.data_root_dir = str(data_root)
    cfg.train.init_checkpoint = str(init_checkpoint)
    if epochs is not None:
        if int(epochs) <= 0:
            raise RunnerError(f"--epochs must be positive, got {epochs}.")
        cfg.train.epochs = int(epochs)

    if mode == "smoke":
        cfg.train.epochs = 1
        cfg.train.max_batches_per_epoch = 2
        cfg.train.batch_size = min(int(cfg.train.batch_size), 8)
        cfg.train.num_workers = 0
        cfg.train.save_every_epochs = 1
        # Codebook initialization consumes its own loader batches before the
        # epoch loop.  Cap it as well so a 2-batch smoke really stays small.
        cfg.train.init_codebook_from_data_batches = min(
            int(cfg.train.get("init_codebook_from_data_batches", 0)), 2
        )

    if resume:
        cfg.train.resume_checkpoint = str(output_dir / "latest.ckpt")
    elif "resume_checkpoint" in cfg.train:
        del cfg.train["resume_checkpoint"]
    return cfg


def build_jobs(
    *,
    benchmark: str,
    methods: Sequence[str],
    seeds: Sequence[int],
    gpus: Sequence[str],
    mode: str,
    resume: bool,
    checkpoint_root: str | None = None,
    data_root: str | None = None,
    init_checkpoint: str | None = None,
    python_bin: str | None = None,
    epochs: int | None = None,
    intermediate_weight: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[RunJob]:
    environ = os.environ if environ is None else environ
    if benchmark not in BENCHMARKS:
        raise RunnerError(f"Unsupported benchmark {benchmark!r}.")
    if mode not in {"smoke", "train"}:
        raise RunnerError(f"Unsupported mode {mode!r}.")
    if not methods or not seeds or not gpus:
        raise RunnerError("At least one method, seed, and GPU are required.")
    if len(methods) != len(set(methods)):
        raise RunnerError("--methods must not contain duplicates.")
    if len(seeds) != len(set(seeds)):
        raise RunnerError("--seeds must not contain duplicates.")
    if any(int(seed) < 0 for seed in seeds):
        raise RunnerError("Seeds must be non-negative integers.")

    spec = BENCHMARKS[benchmark]
    canonical_path = _repo_path(spec["config"])
    if not canonical_path.is_file():
        raise RunnerError(f"Canonical config not found: {canonical_path}")
    canonical_cfg = OmegaConf.load(canonical_path)
    resolved_weight = (
        float(spec["weight"])
        if intermediate_weight is None
        else float(intermediate_weight)
    )
    if not math.isfinite(resolved_weight) or resolved_weight <= 0:
        raise RunnerError(
            f"--intermediate-weight must be finite and > 0, got {resolved_weight}."
        )
    try:
        # Validate the complete four-row controlled contract even when the CLI
        # asks to run only a subset of methods.
        materialize_ablation_configs(
            canonical_cfg,
            benchmark=benchmark,
            intermediate_weight=resolved_weight,
        )
    except AblationConfigError as exc:
        raise RunnerError(f"Invalid canonical ablation config: {exc}") from exc
    root = _benchmark_output_root(
        benchmark, canonical_cfg, checkpoint_root, environ
    )
    resolved_data_root = _resolve_runtime_value(
        data_root,
        spec["data_env"],
        canonical_cfg.data.data_root_dir,
        environ,
    )
    resolved_init = _resolve_runtime_value(
        init_checkpoint,
        spec["init_env"],
        canonical_cfg.train.init_checkpoint,
        environ,
    )
    resolved_python = _resolve_python_bin(python_bin, environ)

    jobs: list[RunJob] = []
    index = 0
    for seed in seeds:
        for method in methods:
            if method not in METHODS:
                raise RunnerError(
                    f"Unknown method {method!r}; expected one of {METHODS}."
                )
            output_dir = _run_output_dir(root, mode, method, int(seed))
            cfg = materialize_method_config(
                canonical_cfg,
                benchmark=benchmark,
                method=method,
                seed=int(seed),
                mode=mode,
                output_dir=output_dir,
                data_root=resolved_data_root,
                init_checkpoint=resolved_init,
                epochs=epochs,
                resume=resume,
                intermediate_weight=resolved_weight,
            )
            materialized = output_dir / MATERIALIZED_CONFIG_NAME
            command = [
                resolved_python,
                str(TRAIN_ENTRYPOINT),
                "--config_yaml",
                str(materialized),
            ]
            jobs.append(
                RunJob(
                    benchmark=benchmark,
                    method=method,
                    seed=int(seed),
                    gpu=str(gpus[index % len(gpus)]),
                    mode=mode,
                    canonical_config=canonical_path,
                    config=cfg,
                    output_dir=_repo_path(output_dir),
                    python_bin=resolved_python,
                    command=command,
                    init_checkpoint=_repo_path(resolved_init),
                    data_root=_repo_path(resolved_data_root),
                    config_sha256=config_sha256(cfg),
                )
            )
            index += 1
    return jobs


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _resume_compatible_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    value = _plain_config(cfg)
    train = value.get("train")
    if isinstance(train, dict):
        train.pop("epochs", None)
        train.pop("resume_checkpoint", None)
    return value


def _manifest_config_hash(manifest: Mapping[str, Any]) -> str | None:
    nested = manifest.get("resolved_config")
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return str(nested["sha256"])
    value = manifest.get("resolved_config_sha256")
    return str(value) if value else None


def _manifest_init_hash(manifest: Mapping[str, Any]) -> str | None:
    nested = manifest.get("init_checkpoint")
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return str(nested["sha256"])
    value = manifest.get("init_checkpoint_sha256")
    return str(value) if value else None


def _checkpoint_epoch(path: Path) -> int:
    """Read the zero-based epoch from one of this runner's checkpoints.

    PyTorch is imported lazily so planning and fresh-run preflight retain the
    launcher's lightweight path. Resume inspects the checkpoint itself instead
    of inferring progress from the prior configured target.
    """

    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RunnerError(f"Cannot read resume checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping) or "epoch" not in checkpoint:
        raise RunnerError(
            f"Cannot resume from {path}: checkpoint does not contain an epoch."
        )
    raw_epoch = checkpoint["epoch"]
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise RunnerError(
            f"Cannot resume from {path}: invalid checkpoint epoch {raw_epoch!r}."
        )
    return raw_epoch


def validate_resume(job: RunJob) -> dict[str, Any]:
    latest = job.output_dir / "latest.ckpt"
    manifest_path = job.manifest_path
    prior_config_path = job.materialized_config_path
    for path in (latest, manifest_path, prior_config_path):
        if not path.is_file():
            raise RunnerError(
                f"Cannot resume {job.label}: required own-run artifact is missing: {path}"
            )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"Cannot resume {job.label}: invalid prior manifest: {exc}"
        ) from exc
    for field_name, expected in (
        ("benchmark", job.benchmark),
        ("method", job.method),
        ("seed", job.seed),
        ("mode", job.mode),
    ):
        if manifest.get(field_name) != expected:
            raise RunnerError(
                f"Cannot resume {job.label}: manifest {field_name}="
                f"{manifest.get(field_name)!r}, expected {expected!r}."
            )

    prior_cfg = OmegaConf.load(prior_config_path)
    actual_prior_hash = config_sha256(prior_cfg)
    recorded_prior_hash = _manifest_config_hash(manifest)
    if not recorded_prior_hash or recorded_prior_hash != actual_prior_hash:
        raise RunnerError(
            f"Cannot resume {job.label}: prior materialized config hash does not "
            "match run_manifest.json."
        )

    if job.init_sha256 is None:
        raise RunnerError(f"Cannot resume {job.label}: init checkpoint was not hashed.")
    recorded_init_hash = _manifest_init_hash(manifest)
    if not recorded_init_hash or recorded_init_hash != job.init_sha256:
        raise RunnerError(
            f"Cannot resume {job.label}: init checkpoint SHA256 changed."
        )

    prior_epochs = int(prior_cfg.train.epochs)
    current_epochs = int(job.config.train.epochs)
    if current_epochs < prior_epochs:
        raise RunnerError(
            f"Cannot resume {job.label}: train.epochs must not decrease "
            f"({prior_epochs} -> {current_epochs})."
        )
    checkpoint_epoch = _checkpoint_epoch(latest)
    completed_epochs = checkpoint_epoch + 1
    if completed_epochs > prior_epochs:
        raise RunnerError(
            f"Cannot resume {job.label}: latest.ckpt reached epoch "
            f"{checkpoint_epoch}, beyond the prior configured target of "
            f"{prior_epochs} epochs."
        )
    if completed_epochs >= current_epochs:
        raise RunnerError(
            f"Cannot resume {job.label}: latest.ckpt already reached the target "
            f"({completed_epochs}/{current_epochs} epochs)."
        )
    if config_sha256(_resume_compatible_config(prior_cfg)) != config_sha256(
        _resume_compatible_config(job.config)
    ):
        raise RunnerError(
            f"Cannot resume {job.label}: configuration changed beyond the allowed "
            "non-decreasing train.epochs target and own latest.ckpt resume "
            "field."
        )
    job.prior_epochs = prior_epochs
    job.prior_checkpoint_epoch = checkpoint_epoch
    return manifest


def _is_reclaimable_skipped_shell(job: RunJob) -> bool:
    """Return true only for an untouched runner-created skipped job shell."""

    if not job.output_dir.is_dir() or job.output_dir.is_symlink():
        return False
    try:
        entries = {path.name: path for path in job.output_dir.iterdir()}
    except OSError:
        return False
    expected_names = {MANIFEST_NAME, MATERIALIZED_CONFIG_NAME}
    if set(entries) != expected_names:
        return False
    if any(path.is_symlink() or not path.is_file() for path in entries.values()):
        return False
    try:
        manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
        prior_cfg = OmegaConf.load(job.materialized_config_path)
    except Exception:
        # Fail closed for malformed JSON/YAML or any filesystem race.
        return False
    if not isinstance(manifest, Mapping) or manifest.get("status") != "skipped":
        return False
    for field_name, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("benchmark", job.benchmark),
        ("method", job.method),
        ("seed", job.seed),
        ("mode", job.mode),
    ):
        if manifest.get(field_name) != expected:
            return False
    recorded_hash = _manifest_config_hash(manifest)
    return bool(recorded_hash) and recorded_hash == config_sha256(prior_cfg)


def _artifact_issues(job: RunJob, *, resume: bool) -> list[str]:
    issues: list[str] = []
    if _resolve_executable(job.python_bin) is None:
        issues.append(f"python executable not found/executable: {job.python_bin}")
    if not job.data_root.is_dir():
        issues.append(f"data root directory not found: {job.data_root}")
    if not job.init_checkpoint.is_file():
        issues.append(f"init checkpoint not found: {job.init_checkpoint}")
    starvla_yaml = _repo_path(str(job.config.data.starvla_config_yaml))
    if not starvla_yaml.is_file():
        issues.append(f"dataset config not found: {starvla_yaml}")
    if resume:
        for name in ("latest.ckpt", MATERIALIZED_CONFIG_NAME, MANIFEST_NAME):
            path = job.output_dir / name
            if not path.is_file():
                issues.append(f"resume artifact not found: {path}")
    elif job.output_dir.exists():
        if not job.output_dir.is_dir():
            issues.append(f"output path exists and is not a directory: {job.output_dir}")
        elif any(job.output_dir.iterdir()) and not _is_reclaimable_skipped_shell(job):
            issues.append(f"output directory is nonempty: {job.output_dir}")
    return issues


def preflight_jobs(jobs: Sequence[RunJob], *, resume: bool) -> None:
    """Validate all jobs without creating or modifying any output."""

    if len({job.output_dir for job in jobs}) != len(jobs):
        raise RunnerError("Multiple jobs resolved to the same output directory.")
    failures: list[str] = []
    init_hash_cache: dict[Path, str] = {}
    for job in jobs:
        issues = _artifact_issues(job, resume=resume)
        if issues:
            failures.extend(f"{job.label}: {issue}" for issue in issues)
            continue
        resolved_python = _resolve_executable(job.python_bin)
        if resolved_python is None:  # Defensive: _artifact_issues already checked it.
            failures.append(f"{job.label}: python executable could not be resolved")
            continue
        job.python_bin = resolved_python
        job.command[0] = resolved_python
        if job.init_checkpoint not in init_hash_cache:
            init_hash_cache[job.init_checkpoint] = sha256_file(job.init_checkpoint)
        job.init_sha256 = init_hash_cache[job.init_checkpoint]
        if resume:
            try:
                job.prior_manifest = validate_resume(job)
            except RunnerError as exc:
                failures.append(str(exc))
        else:
            job.reclaim_skipped_shell = _is_reclaimable_skipped_shell(job)
    if failures:
        formatted = "\n  - ".join(failures)
        raise RunnerError(f"Stage1 ablation preflight failed:\n  - {formatted}")


def _dataset_identifier(job: RunJob) -> dict[str, Any]:
    data = job.config.data
    keys = (
        "data_mix",
        "starvla_config_yaml",
        "expected_action_horizon",
        "expected_action_dim",
        "action_mode",
        "window_mode",
    )
    identifier = {key: data.get(key) for key in keys if data.get(key) is not None}
    identifier["data_root_dir"] = str(data.data_root_dir)
    return identifier


def make_manifest(
    job: RunJob,
    *,
    runner_invocation: Sequence[str],
    git_commit: str,
) -> dict[str, Any]:
    created = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": job.benchmark,
        "method": job.method,
        "seed": job.seed,
        "mode": job.mode,
        "gpu": job.gpu,
        "status": "planned",
        "git_commit": git_commit,
        "canonical_config": str(job.canonical_config),
        "canonical_config_sha256": sha256_file(job.canonical_config),
        "resolved_config": {
            "path": MATERIALIZED_CONFIG_NAME,
            "sha256": job.config_sha256,
        },
        "resolved_config_sha256": job.config_sha256,
        "init_checkpoint": {
            "path": str(job.config.train.init_checkpoint),
            "sha256": job.init_sha256,
        },
        "init_checkpoint_sha256": job.init_sha256,
        "dataset": _dataset_identifier(job),
        "command": job.command,
        "command_shell": shlex.join(job.command),
        "runner_invocation": list(runner_invocation),
        "timestamps": {
            "created_at": created,
            "started_at": None,
            "finished_at": None,
        },
        "exit_code": None,
    }
    if job.prior_manifest is not None:
        if job.prior_epochs is None or job.prior_checkpoint_epoch is None:
            raise RunnerError(
                f"Validated resume metadata is incomplete for {job.label}."
            )
        manifest["resume"] = {
            "checkpoint": "latest.ckpt",
            "prior_config_sha256": _manifest_config_hash(job.prior_manifest),
            "prior_epochs": job.prior_epochs,
            "prior_checkpoint_epoch": job.prior_checkpoint_epoch,
        }
    return manifest


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_jobs(
    jobs: Sequence[RunJob],
    *,
    runner_invocation: Sequence[str],
) -> None:
    """Create configs/manifests after the entire job matrix passed preflight."""

    commit = _git_commit()
    for job in jobs:
        if job.reclaim_skipped_shell and not _is_reclaimable_skipped_shell(job):
            raise RunnerError(
                f"Refusing to reuse changed skipped-job shell for {job.label}: "
                f"{job.output_dir}"
            )
        job.output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(job.config, job.materialized_config_path, resolve=True)
        # Check the bytes we wrote still represent exactly the preflighted config.
        written_hash = config_sha256(OmegaConf.load(job.materialized_config_path))
        if written_hash != job.config_sha256:
            raise RunnerError(
                f"Materialized config hash changed unexpectedly for {job.label}."
            )
        _write_json(
            job.manifest_path,
            make_manifest(
                job,
                runner_invocation=runner_invocation,
                git_commit=commit,
            ),
        )


def _update_manifest(job: RunJob, **updates: Any) -> None:
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
    timestamps = manifest.setdefault("timestamps", {})
    for key, value in updates.items():
        if key in {"started_at", "finished_at"}:
            timestamps[key] = value
        else:
            manifest[key] = value
    _write_json(job.manifest_path, manifest)


def execute_job(job: RunJob) -> int:
    _update_manifest(job, status="running", started_at=utc_now())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env["PYTHONUNBUFFERED"] = "1"
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not prior_pythonpath
        else str(REPO_ROOT) + os.pathsep + prior_pythonpath
    )
    command = list(job.command)
    resolved_python = _resolve_executable(job.python_bin)
    if resolved_python:
        command[0] = resolved_python
    run_kwargs: dict[str, Any] = {}
    inherited_lock_fd = env.get("STAGE1_LAUNCH_LOCK_FD")
    if inherited_lock_fd:
        try:
            lock_fd = int(inherited_lock_fd)
            os.fstat(lock_fd)
        except (OSError, TypeError, ValueError) as exc:
            _update_manifest(
                job,
                status="failed",
                finished_at=utc_now(),
                exit_code=126,
                error=f"invalid inherited Stage1 launcher lock fd: {exc}",
            )
            return 126
        run_kwargs["pass_fds"] = (lock_fd,)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            **run_kwargs,
        )
        return_code = int(completed.returncode)
        error = None
    except OSError as exc:
        return_code = 127
        error = str(exc)
    updates: dict[str, Any] = {
        "status": "succeeded" if return_code == 0 else "failed",
        "finished_at": utc_now(),
        "exit_code": return_code,
    }
    if error is not None:
        updates["error"] = error
    _update_manifest(job, **updates)
    return return_code


def run_jobs(jobs: Sequence[RunJob]) -> int:
    """Run one serial queue per GPU; return nonzero if any worker fails."""

    queues: dict[str, list[RunJob]] = {}
    for job in jobs:
        queues.setdefault(job.gpu, []).append(job)

    def worker(queue: Sequence[RunJob]) -> bool:
        for index, job in enumerate(queue):
            print(f"[stage1-ablation] starting {job.label} on GPU {job.gpu}", flush=True)
            return_code = execute_job(job)
            if return_code != 0:
                for skipped in queue[index + 1 :]:
                    _update_manifest(
                        skipped,
                        status="skipped",
                        finished_at=utc_now(),
                        exit_code=None,
                        error=f"previous job on GPU {job.gpu} failed",
                    )
                return False
        return True

    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        outcomes = list(executor.map(worker, queues.values()))
    return 0 if all(outcomes) else 1


def print_dry_run(jobs: Sequence[RunJob], *, resume: bool) -> None:
    for job in jobs:
        warnings = _artifact_issues(job, resume=resume)
        print(f"=== {job.label} | gpu={job.gpu} | mode={job.mode} ===")
        print(OmegaConf.to_yaml(job.config, resolve=True).rstrip())
        print(f"command: {shlex.join(job.command)}")
        for warning in warnings:
            print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch controlled VAR Stage1 four-method ablations."
    )
    parser.add_argument("--benchmark", required=True, choices=tuple(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU indices; each GPU runs a serial queue.",
    )
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--data-root")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--python-bin")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--intermediate-weight",
        type=float,
        help=(
            "Benchmark-uniform auxiliary weight for Full-target, Paper-DCT, "
            "and MTR (Base remains zero)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        gpus = parse_gpus(args.gpus)
        jobs = build_jobs(
            benchmark=args.benchmark,
            methods=args.methods,
            seeds=args.seeds,
            gpus=gpus,
            mode=args.mode,
            resume=args.resume,
            checkpoint_root=args.checkpoint_root,
            data_root=args.data_root,
            init_checkpoint=args.init_checkpoint,
            python_bin=args.python_bin,
            epochs=args.epochs,
            intermediate_weight=args.intermediate_weight,
        )
        if args.dry_run:
            print_dry_run(jobs, resume=args.resume)
            return 0
        preflight_jobs(jobs, resume=args.resume)
        invocation = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
        prepare_jobs(jobs, runner_invocation=invocation)
        return run_jobs(jobs)
    except RunnerError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
