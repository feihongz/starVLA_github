#!/usr/bin/env python3
"""Evaluate the complete Stage1 clean-supervision ablation checkpoint matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "scripts/eval/eval_var_stage1_intermediate_mse.py"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "playground/Checkpoints/stage1_clean_supervision_ablation"
METHODS = (
    "multiscale_base",
    "full_target_time",
    "mint_paper_dct",
    "mtr",
)
BENCHMARKS = ("libero", "robocasa")
METRIC_VERSION = "scale_aligned_down_up_v1"
RESULT_FILENAME = "intermediate_traj_mse_eval.json"
MANIFEST_SCHEMA_VERSION = "stage1_ablation_run.v1"
REPORT_SCHEMA_VERSION = "stage1_intermediate_eval.v1"
BENCHMARK_REPORT_CONTRACTS: dict[str, dict[str, Any]] = {
    "libero": {
        "horizon": 8,
        "action_dim": 7,
        "all_scales": [1, 2, 4, 8],
        "evaluated_scales": [1, 2, 4],
        "included_dims": list(range(6)),
        "excluded_groups": ["gripper"],
    },
    "robocasa": {
        "horizon": 16,
        "action_dim": 29,
        "all_scales": [1, 2, 4, 8, 16],
        "evaluated_scales": [1, 2, 4, 8],
        "included_dims": list(range(29)),
        "excluded_groups": [],
    },
}
TARGET_CONTRACT = {
    "downsample": "adaptive_avg_pool1d",
    "upsample": "linear",
    "align_corners": False,
}


@dataclass(frozen=True)
class EvalTask:
    benchmark: str
    method: str
    seed: int
    run_dir: Path
    checkpoint: Path
    manifest: Path
    output: Path
    log: Path
    # None means the evaluator must restore the original path from the checkpoint.
    # A value is present only for an explicit benchmark-specific CLI override.
    data_root: Path | None

    @property
    def label(self) -> str:
        return f"{self.benchmark}/{self.method}/seed_{self.seed}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    buffer = io.StringIO()
    json.dump(payload, buffer, indent=2, ensure_ascii=False, allow_nan=False)
    buffer.write("\n")
    _atomic_write_text(path, buffer.getvalue())


def _parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated GPU list: {value!r}.") from exc
    if not gpus or any(gpu < 0 for gpu in gpus) or len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError(
            f"GPU list must contain unique non-negative indices, got {gpus}."
        )
    return gpus


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<log unavailable>"
    return "".join(lines[-line_count:])


def _build_tasks(args: argparse.Namespace) -> list[EvalTask]:
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    overrides = {
        "libero": args.libero_data_root,
        "robocasa": args.robocasa_data_root,
    }
    tasks: list[EvalTask] = []
    errors: list[str] = []
    for benchmark in args.benchmarks:
        for method in args.methods:
            for seed in args.seeds:
                run_dir = checkpoint_root / benchmark / method / f"seed_{seed}"
                checkpoint = run_dir / args.checkpoint_name
                manifest_path = run_dir / "run_manifest.json"
                output = run_dir / RESULT_FILENAME
                log = run_dir / "intermediate_traj_mse_eval.log"
                if not checkpoint.is_file():
                    errors.append(f"missing checkpoint: {checkpoint}")
                if not manifest_path.is_file():
                    errors.append(f"missing run manifest: {manifest_path}")
                    continue
                try:
                    manifest = _read_json(manifest_path)
                except Exception as exc:
                    errors.append(f"invalid run manifest {manifest_path}: {exc}")
                    continue
                expected = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "benchmark": benchmark,
                    "method": method,
                    "seed": int(seed),
                    "mode": "train",
                    "status": "succeeded",
                    "exit_code": 0,
                }
                for key, expected_value in expected.items():
                    if manifest.get(key) != expected_value:
                        errors.append(
                            f"manifest mismatch {manifest_path}: {key}="
                            f"{manifest.get(key)!r}, expected={expected_value!r}"
                        )
                dataset_info = manifest.get("dataset")
                if not isinstance(dataset_info, Mapping):
                    errors.append(f"manifest has no dataset object: {manifest_path}")
                    continue

                benchmark_contract = BENCHMARK_REPORT_CONTRACTS[benchmark]
                dataset_expected = {
                    "expected_action_horizon": benchmark_contract["horizon"],
                    "expected_action_dim": benchmark_contract["action_dim"],
                    "window_mode": "full",
                }
                for key, expected_value in dataset_expected.items():
                    if dataset_info.get(key) != expected_value:
                        errors.append(
                            f"dataset contract mismatch {manifest_path}: {key}="
                            f"{dataset_info.get(key)!r}, expected={expected_value!r}"
                        )

                configured_root_text = dataset_info.get("data_root_dir")
                if not isinstance(configured_root_text, str) or not configured_root_text.strip():
                    errors.append(f"manifest has no usable dataset data_root_dir: {manifest_path}")
                    continue
                configured_root = Path(configured_root_text).expanduser()
                if not configured_root.is_absolute():
                    configured_root = REPO_ROOT / configured_root
                configured_root = configured_root.resolve()
                override = overrides[benchmark]
                data_root = override.expanduser().resolve() if override is not None else None
                preflight_data_root = data_root if data_root is not None else configured_root
                if not preflight_data_root.is_dir():
                    source = "explicit override" if data_root is not None else "run manifest"
                    errors.append(
                        f"missing data root for {benchmark} ({source}): {preflight_data_root}"
                    )
                if output.exists() and not args.overwrite:
                    errors.append(
                        f"output already exists: {output}; pass --overwrite to replace formal results"
                    )
                tasks.append(
                    EvalTask(
                        benchmark=benchmark,
                        method=method,
                        seed=int(seed),
                        run_dir=run_dir,
                        checkpoint=checkpoint,
                        manifest=manifest_path,
                        output=output,
                        log=log,
                        data_root=data_root,
                    )
                )
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(f"Preflight failed:\n  - {joined}")
    if not tasks:
        raise RuntimeError("Preflight produced no evaluation tasks.")
    return tasks


def _command_for_task(task: EvalTask, gpu: int, args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python_bin),
        str(EVALUATOR),
        "--checkpoint",
        str(task.checkpoint),
        "--benchmark",
        task.benchmark,
        "--output",
        str(task.output),
        "--device",
        f"cuda:{gpu}",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-batches",
        str(args.max_batches),
        "--no-progress",
    ]
    if task.data_root is not None:
        command.extend(["--data-root", str(task.data_root)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def _run_gpu_queue(gpu: int, tasks: Sequence[EvalTask], args: argparse.Namespace) -> list[Path]:
    outputs: list[Path] = []
    for queue_index, task in enumerate(tasks, start=1):
        command = _command_for_task(task, gpu, args)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] LAUNCH gpu={gpu} "
            f"queue={queue_index}/{len(tasks)} task={task.label}",
            flush=True,
        )
        task.log.parent.mkdir(parents=True, exist_ok=True)
        with task.log.open("w", encoding="utf-8") as log_handle:
            log_handle.write("command=" + " ".join(command) + "\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Evaluation failed rc={completed.returncode}, gpu={gpu}, task={task.label}, "
                f"log={task.log}\n{_tail(task.log)}"
            )
        if not task.output.is_file():
            raise RuntimeError(
                f"Evaluator returned success without creating {task.output} for {task.label}."
            )
        outputs.append(task.output)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] DONE gpu={gpu} task={task.label} "
            f"output={task.output}",
            flush=True,
        )
    return outputs


def _validate_reports(
    tasks: Sequence[EvalTask],
    *,
    expected_max_batches: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    per_benchmark_contracts: dict[str, tuple[Any, ...]] = {}
    for task in tasks:
        report = _read_json(task.output)
        expected_fields = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "metric_version": METRIC_VERSION,
            "benchmark": task.benchmark,
            "method": task.method,
            "seed": task.seed,
        }
        for key, expected in expected_fields.items():
            if report.get(key) != expected:
                raise ValueError(
                    f"Invalid result {task.output}: {key}={report.get(key)!r}, expected={expected!r}."
                )

        benchmark_contract = BENCHMARK_REPORT_CONTRACTS[task.benchmark]
        fixed_contract = {
            "split": "train_full_windows",
            "action_space": "normalized",
            "horizon": benchmark_contract["horizon"],
            "action_dim": benchmark_contract["action_dim"],
            "all_scales": benchmark_contract["all_scales"],
            "evaluated_scales": benchmark_contract["evaluated_scales"],
            "included_dims": benchmark_contract["included_dims"],
            "excluded_groups": benchmark_contract["excluded_groups"],
            "target": TARGET_CONTRACT,
        }
        for key, expected in fixed_contract.items():
            if report.get(key) != expected:
                raise ValueError(
                    f"Invalid result contract {task.output}: {key}={report.get(key)!r}, "
                    f"expected={expected!r}."
                )

        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Invalid result {task.output}: checkpoint must be an object.")
        reported_checkpoint = Path(str(checkpoint.get("path", ""))).expanduser()
        if reported_checkpoint.resolve() != task.checkpoint.resolve():
            raise ValueError(
                f"Invalid result {task.output}: checkpoint path {reported_checkpoint} does not "
                f"match requested checkpoint {task.checkpoint}."
            )
        checkpoint_sha256 = checkpoint.get("sha256")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_sha256.lower())
        ):
            raise ValueError(f"Invalid checkpoint SHA256 in {task.output}: {checkpoint_sha256!r}.")
        checkpoint_epoch = checkpoint.get("epoch")
        if (
            isinstance(checkpoint_epoch, bool)
            or not isinstance(checkpoint_epoch, int)
            or checkpoint_epoch < 0
        ):
            raise ValueError(f"Invalid checkpoint epoch in {task.output}: {checkpoint_epoch!r}.")

        runtime = report.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("max_batches") != expected_max_batches:
            actual_max_batches = runtime.get("max_batches") if isinstance(runtime, Mapping) else None
            raise ValueError(
                f"Invalid runtime contract {task.output}: max_batches={actual_max_batches!r}, "
                f"expected={expected_max_batches}."
            )

        data = report.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"Invalid result {task.output}: data must be an object.")
        expected_override = task.data_root is not None
        if data.get("data_root_overridden") is not expected_override:
            raise ValueError(
                f"Invalid data-root provenance {task.output}: data_root_overridden="
                f"{data.get('data_root_overridden')!r}, expected={expected_override!r}."
            )
        if task.data_root is not None:
            effective_data_root = Path(str(data.get("effective_data_root", ""))).expanduser()
            if effective_data_root.resolve() != task.data_root.resolve():
                raise ValueError(
                    f"Invalid data-root override {task.output}: effective={effective_data_root}, "
                    f"expected={task.data_root}."
                )

        num_samples = report.get("num_samples")
        dataset_len = report.get("dataset_len")
        num_batches = report.get("num_batches")
        for name, value in (
            ("num_samples", num_samples),
            ("dataset_len", dataset_len),
            ("num_batches", num_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid {name} in {task.output}: {value!r}.")
        if expected_max_batches == 0 and num_samples != dataset_len:
            raise ValueError(
                f"Incomplete full-dataset result {task.output}: num_samples={num_samples}, "
                f"dataset_len={dataset_len}."
            )

        per_scale_mse = report.get("per_scale_mse")
        expected_scale_keys = [str(scale) for scale in benchmark_contract["evaluated_scales"]]
        if not isinstance(per_scale_mse, Mapping) or list(per_scale_mse) != expected_scale_keys:
            actual_scale_keys = list(per_scale_mse) if isinstance(per_scale_mse, Mapping) else None
            raise ValueError(
                f"Invalid per-scale contract {task.output}: keys={actual_scale_keys!r}, "
                f"expected={expected_scale_keys!r}."
            )
        sanity = report.get("sanity")
        if not isinstance(sanity, Mapping):
            raise ValueError(f"Invalid result {task.output}: sanity must be an object.")
        numeric_values = {
            "intermediate_traj_mse": report.get("intermediate_traj_mse"),
            "final_recon_mse_all_dims": sanity.get("final_recon_mse_all_dims"),
            "max_abs_final_scale_vs_recon": sanity.get("max_abs_final_scale_vs_recon"),
            **{f"per_scale_mse[{scale}]": per_scale_mse[scale] for scale in expected_scale_keys},
        }
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid numeric field {name} in {task.output}: {value!r}.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Non-finite or negative field {name} in {task.output}: {value!r}.")
        per_scale_average = sum(float(per_scale_mse[key]) for key in expected_scale_keys) / len(
            expected_scale_keys
        )
        if not math.isclose(
            float(report["intermediate_traj_mse"]),
            per_scale_average,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Invalid aggregate metric {task.output}: intermediate_traj_mse="
                f"{report['intermediate_traj_mse']!r}, per-scale average={per_scale_average!r}."
            )

        contract = (
            num_samples,
            dataset_len,
            tuple(report["all_scales"]),
            tuple(report["evaluated_scales"]),
            tuple(report["included_dims"]),
            json.dumps(report["target"], sort_keys=True),
            str(data.get("data_mix", "")),
            str(data.get("window_mode", "")),
            str(Path(str(data.get("effective_data_root", ""))).expanduser().resolve()),
        )
        previous = per_benchmark_contracts.setdefault(task.benchmark, contract)
        if previous != contract:
            raise ValueError(
                f"Within-benchmark evaluation contract differs for {task.label}: "
                f"actual={contract}, expected={previous}."
            )
        reports.append(report)
    return reports


def _summary_csv(reports: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "benchmark",
        "method",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "num_samples",
        "dataset_len",
        "evaluated_scales",
        "included_dims",
        "intermediate_traj_mse",
        "final_recon_mse_all_dims",
        "per_scale_mse",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for report in reports:
        writer.writerow(
            {
                "benchmark": report["benchmark"],
                "method": report["method"],
                "seed": report["seed"],
                "checkpoint": report["checkpoint"]["path"],
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "checkpoint_epoch": report["checkpoint"]["epoch"],
                "num_samples": report["num_samples"],
                "dataset_len": report["dataset_len"],
                "evaluated_scales": json.dumps(report["evaluated_scales"], separators=(",", ":")),
                "included_dims": json.dumps(report["included_dims"], separators=(",", ":")),
                "intermediate_traj_mse": format(float(report["intermediate_traj_mse"]), ".17g"),
                "final_recon_mse_all_dims": format(
                    float(report["sanity"]["final_recon_mse_all_dims"]), ".17g"
                ),
                "per_scale_mse": json.dumps(report["per_scale_mse"], separators=(",", ":")),
            }
        )
    return buffer.getvalue()


def _format_metric(value: float) -> str:
    return f"{float(value):.6e}"


def _summary_markdown(reports: Sequence[Mapping[str, Any]]) -> str:
    lookup = {
        (str(report["benchmark"]), str(report["method"])): report
        for report in reports
    }
    display_names = {
        "multiscale_base": "Multi-scale Base",
        "full_target_time": "Full-target Time",
        "mint_paper_dct": "MINT-style",
        "mtr": "Ours (MTR)",
    }
    lines = [
        "# Stage1 Scale-Aligned Intermediate Trajectory MSE",
        "",
        f"- Metric version: `{METRIC_VERSION}`",
        "- Action space: normalized",
        "- Checkpoint selection: pre-existing `best_balanced.ckpt` only",
        "- Seed: 42 (single seed; values are not mean ± std)",
        "- Final tokenizer scale is excluded",
        "- LIBERO excludes binary gripper; RoboCasa includes all 29 continuous dimensions",
        "",
        "## Paper table",
        "",
        "| Method | Intermediate Traj. MSE ↓ (LIBERO / RoboCasa) |",
        "|---|---:|",
    ]
    for method in METHODS:
        libero = lookup.get(("libero", method))
        robocasa = lookup.get(("robocasa", method))
        if libero is None or robocasa is None:
            continue
        lines.append(
            f"| {display_names[method]} | "
            f"{_format_metric(libero['intermediate_traj_mse'])} / "
            f"{_format_metric(robocasa['intermediate_traj_mse'])} |"
        )
    lines.extend(
        [
            "",
            "## Full results",
            "",
            "| Benchmark | Method | Seed | Samples | Scales | Intermediate MSE ↓ | Final recon MSE (sanity) | Per-scale MSE |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for report in reports:
        per_scale = ", ".join(
            f"s{scale}={_format_metric(value)}"
            for scale, value in report["per_scale_mse"].items()
        )
        scales = ",".join(str(scale) for scale in report["evaluated_scales"])
        lines.append(
            f"| {report['benchmark']} | {report['method']} | {report['seed']} | "
            f"{report['num_samples']} | {scales} | "
            f"{_format_metric(report['intermediate_traj_mse'])} | "
            f"{_format_metric(report['sanity']['final_recon_mse_all_dims'])} | {per_scale} |"
        )
    lines.extend(
        [
            "",
            "The metric uses exact whole-dataset elementwise SSE/count at each scale,",
            "followed by an equal average across intermediate scales. LIBERO and RoboCasa",
            "are reported separately and must not be averaged together.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_summaries(root: Path, reports: Sequence[dict[str, Any]]) -> list[Path]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary_json = root / "intermediate_traj_mse_summary.json"
    summary_csv = root / "intermediate_traj_mse_summary.csv"
    summary_md = root / "intermediate_traj_mse_summary.md"
    payload = {
        "schema_version": "stage1_intermediate_eval_summary.v1",
        "metric_version": METRIC_VERSION,
        "generated_at": generated_at,
        "single_seed": True,
        "runs": list(reports),
    }
    _atomic_write_json(summary_json, payload)
    _atomic_write_text(summary_csv, _summary_csv(reports))
    _atomic_write_text(summary_md, _summary_markdown(reports))
    return [summary_json, summary_csv, summary_md]


def _assign_queues(tasks: Sequence[EvalTask], gpus: Sequence[int]) -> dict[int, list[EvalTask]]:
    queues = {gpu: [] for gpu in gpus}
    for index, task in enumerate(tasks):
        queues[gpus[index % len(gpus)]].append(task)
    return queues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--checkpoint-name", default="best_balanced.ckpt")
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1,2,3"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    # Keep the invoked interpreter path intact: resolving a venv symlink can turn
    # `.venv310/bin/python` into the base `/usr/bin/python*` interpreter.
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--libero-data-root", type=Path)
    parser.add_argument("--robocasa-data-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_batches < 0:
        raise ValueError(
            "batch-size must be positive and num-workers/max-batches must be non-negative."
        )
    if len(args.methods) != len(set(args.methods)) or len(args.benchmarks) != len(set(args.benchmarks)):
        raise ValueError("benchmarks and methods must not contain duplicates.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must not contain duplicates.")
    args.python_bin = args.python_bin.expanduser().absolute()
    if not args.python_bin.is_file() or not os.access(args.python_bin, os.X_OK):
        raise FileNotFoundError(f"Python executable does not exist: {args.python_bin}")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Single-checkpoint evaluator does not exist: {EVALUATOR}")

    tasks = _build_tasks(args)
    queues = _assign_queues(tasks, args.gpus)
    print(f"Preflight passed for {len(tasks)} formal checkpoint(s).", flush=True)
    for gpu, gpu_tasks in queues.items():
        for task in gpu_tasks:
            command = _command_for_task(task, gpu, args)
            print(f"PLAN gpu={gpu} task={task.label} command={' '.join(command)}", flush=True)
    if args.dry_run:
        print("Dry run complete; no result files were written.", flush=True)
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = {
            pool.submit(_run_gpu_queue, gpu, gpu_tasks, args): gpu
            for gpu, gpu_tasks in queues.items()
            if gpu_tasks
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"GPU queue {gpu}: {exc}")
    if failures:
        raise RuntimeError("One or more GPU queues failed:\n\n" + "\n\n".join(failures))

    reports = _validate_reports(tasks, expected_max_batches=args.max_batches)
    summary_paths = _write_summaries(args.checkpoint_root.expanduser().resolve(), reports)
    print("SUCCESS: all formal evaluations completed and passed cross-run validation.", flush=True)
    for path in summary_paths:
        print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
