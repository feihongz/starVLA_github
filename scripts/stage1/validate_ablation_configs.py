"""Validate controlled Stage1 intermediate-supervision ablations.

This module is intentionally independent from the training entry point so the
ablation runner can materialize and validate configurations before importing
PyTorch or touching a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


METHODS = (
    "multiscale_base",
    "full_target_time",
    "mint_paper_dct",
    "mtr",
)
METHOD_MODES = {
    "multiscale_base": "none",
    "full_target_time": "full_target_time",
    "mint_paper_dct": "mint_paper_dct",
    "mtr": "mtr",
}
PAPER_DCT_SPECTRAL = {
    "formulation": "raw_dct_ii_mse",
    "normalization": "2_over_h",
}
DEFAULT_INTERMEDIATE_WEIGHTS = {"libero": 0.02, "robocasa": 0.1}
DEFAULT_CANONICAL_PATHS = {
    "libero": Path(
        "examples/LIBERO/train_files/"
        "train_var_stage1_libero_clean_supervision_ablation.yaml"
    ),
    "robocasa": Path(
        "examples/Robocasa_tabletop/train_files/"
        "train_var_stage1_robocasa_clean_supervision_ablation.yaml"
    ),
}
BENCHMARK_CONTRACTS = {
    "libero": {
        ("data", "expected_action_horizon"): 8,
        ("data", "expected_action_dim"): 7,
        ("model", "embed_dim"): 32,
        ("model", "codebook_size"): 512,
        ("model", "quantization_mode"): "product_vq",
        ("model", "product_codebook_groups"): 16,
        ("model", "scales"): [1, 2, 4, 8],
        ("model", "decoder_head_type"): "plain",
        ("model", "use_time_embedding"): True,
        ("model", "use_action_type_embedding"): True,
    },
    "robocasa": {
        ("data", "expected_action_horizon"): 16,
        ("data", "expected_action_dim"): 29,
        ("data", "action_mode"): "abs",
        ("model", "embed_dim"): 64,
        ("model", "codebook_size"): 512,
        ("model", "quantization_mode"): "product_vq",
        ("model", "product_codebook_groups"): 16,
        ("model", "scales"): [1, 2, 4, 8, 16],
        ("model", "decoder_head_type"): "plain",
        ("model", "use_time_embedding"): True,
        ("model", "use_action_type_embedding"): False,
    },
}

_ALLOWED_DIFFERENCES = {
    ("experiment", "name"),
    ("experiment", "output_dir"),
    ("loss", "intermediate", "mode"),
    ("loss", "intermediate", "weight"),
    ("loss", "intermediate", "spectral"),
}
_MACHINE_HOME_RE = re.compile(r"(?:^|[,:=])\s*/(?:root|home)(?:/|$)")
_HOME_REFERENCE_RE = re.compile(
    r"(?:\$HOME|\$\{HOME\}|\$\{oc\.env:HOME(?:,[^}]*)?\})"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_OPTIONAL_WEIGHTING_BLOCKS = (
    "sample_weighting", "trajectory_weighting", "trajectory_phase_weighting",
    "time_weighting", "task_balance_weighting", "adaptive_task_weighting",
)


class AblationConfigError(ValueError):
    """The four-method clean-ablation contract was violated."""


@dataclass(frozen=True)
class AblationValidation:
    benchmark: str
    methods: tuple[str, ...]
    intermediate_weight: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["methods"] = list(self.methods)
        return result


def _benchmark(value: str) -> str:
    result = str(value).strip().lower()
    if result not in DEFAULT_CANONICAL_PATHS:
        raise AblationConfigError(
            f"Unknown benchmark {value!r}; expected libero or robocasa."
        )
    return result


def _plain(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=False)
    elif isinstance(config, Mapping):
        value = copy.deepcopy(dict(config))
    else:
        raise TypeError(
            f"Expected a mapping or DictConfig, got {type(config).__name__}."
        )
    if not isinstance(value, dict):
        raise AblationConfigError("The YAML root must be a mapping.")
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config without resolving environment interpolations."""

    config_path = Path(path)
    if not config_path.is_file():
        raise AblationConfigError(f"Configuration does not exist: {config_path}")
    return _plain(OmegaConf.load(config_path))


def _mapping(parent: Mapping[str, Any], key: str, location: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise AblationConfigError(f"{location}.{key} must be a mapping.")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise AblationConfigError(f"{location} must be a finite number, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AblationConfigError(f"{location} must be a finite number.") from exc
    if not math.isfinite(result):
        raise AblationConfigError(f"{location} must be finite, got {value!r}.")
    return result


def expected_group_weights(benchmark: str) -> dict[str, float]:
    """Return the locked action mask for the clean ablation.

    RoboCasa deliberately remains ungrouped: weighted MSE therefore covers all
    29 dimensions equally under the repository's current ActionSpec semantics.
    """

    if _benchmark(benchmark) == "libero":
        return {"position": 1.0, "rotation": 1.0, "gripper": 0.0}
    return {}


def _method_value(base: Any, method: str, location: str) -> str:
    if not isinstance(base, str) or not base:
        raise AblationConfigError(f"{location} must be a non-empty string.")
    return base.replace("{method}", method) if "{method}" in base else f"{base}_{method}"


def materialize_ablation_config(
    canonical_config: Mapping[str, Any] | DictConfig,
    method: str,
    *,
    benchmark: str,
    intermediate_weight: float | None = None,
    experiment_name: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create one resolved method config from a canonical benchmark config."""

    benchmark = _benchmark(benchmark)
    if method not in METHOD_MODES:
        raise AblationConfigError(
            f"Unknown method {method!r}; expected one of {list(METHODS)!r}."
        )
    result = _plain(canonical_config)
    experiment = result.setdefault("experiment", {})
    loss = result.setdefault("loss", {})
    if not isinstance(experiment, dict) or not isinstance(loss, dict):
        raise AblationConfigError("experiment and loss must be mappings.")
    intermediate = loss.setdefault("intermediate", {})
    if not isinstance(intermediate, dict):
        raise AblationConfigError("loss.intermediate must be a mapping.")

    if intermediate_weight is None:
        configured = _number(
            intermediate.get("weight", 0.0), "loss.intermediate.weight"
        )
        shared_weight = (
            configured if configured > 0 else DEFAULT_INTERMEDIATE_WEIGHTS[benchmark]
        )
    else:
        shared_weight = _number(intermediate_weight, "intermediate_weight")
    if shared_weight <= 0:
        raise AblationConfigError("The shared non-Base weight must be > 0.")

    experiment["name"] = experiment_name or _method_value(
        experiment.get("name"), method, "experiment.name"
    )
    experiment["output_dir"] = output_dir or _method_value(
        experiment.get("output_dir"), method, "experiment.output_dir"
    )
    intermediate["mode"] = METHOD_MODES[method]
    intermediate["weight"] = 0.0 if method == "multiscale_base" else shared_weight
    intermediate["include_final"] = intermediate.get("include_final", False)
    intermediate["scale_weights"] = intermediate.get("scale_weights", "uniform")
    intermediate["group_weights"] = copy.deepcopy(
        intermediate.get("group_weights", expected_group_weights(benchmark))
    )
    intermediate.pop("spectral", None)
    if method == "mint_paper_dct":
        intermediate["spectral"] = copy.deepcopy(PAPER_DCT_SPECTRAL)
    return result


def materialize_ablation_configs(
    canonical_config: Mapping[str, Any] | DictConfig,
    *,
    benchmark: str,
    intermediate_weight: float | None = None,
    require_portable_paths: bool = True,
) -> dict[str, dict[str, Any]]:
    """Materialize and validate all four clean-ablation configs."""

    configs = {
        method: materialize_ablation_config(
            canonical_config,
            method,
            benchmark=benchmark,
            intermediate_weight=intermediate_weight,
        )
        for method in METHODS
    }
    validate_ablation_configs(
        configs,
        benchmark=benchmark,
        require_portable_paths=require_portable_paths,
    )
    return configs


def _strings(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _strings(child, (*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _strings(child, (*path, str(index)))
    elif isinstance(value, str):
        yield path, value


def _check_paths(config: Mapping[str, Any], method: str) -> None:
    invalid = []
    for path, value in _strings(config):
        if (
            value.startswith("/")
            or value.startswith("~")
            or value[:2] == chr(92) * 2
            or _WINDOWS_ABSOLUTE_RE.match(value)
            or _MACHINE_HOME_RE.search(value)
            or _HOME_REFERENCE_RE.search(value)
        ):
            invalid.append(f"{'.'.join(path)}={value!r}")
    if invalid:
        raise AblationConfigError(
            f"{method} contains non-portable machine-specific path(s): "
            + ", ".join(invalid[:8])
        )


def _remove(config: dict[str, Any], path: tuple[str, ...]) -> None:
    value: Any = config
    for key in path[:-1]:
        if not isinstance(value, dict) or key not in value:
            return
        value = value[key]
    if isinstance(value, dict):
        value.pop(path[-1], None)


def controlled_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config with the difference allowlist removed."""

    result = copy.deepcopy(dict(config))
    for path in _ALLOWED_DIFFERENCES:
        _remove(result, path)
    return result


def _diff_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = []
        for key in sorted(set(left) | set(right), key=str):
            child_path = (*path, str(key))
            if key not in left or key not in right:
                result.append(".".join(child_path))
            else:
                result.extend(_diff_paths(left[key], right[key], child_path))
        return result
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return [".".join(path)]
        result = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            result.extend(_diff_paths(a, b, (*path, str(index))))
        return result
    return [] if left == right else [".".join(path)]


def _check_groups(value: Any, benchmark: str, method: str) -> None:
    if not isinstance(value, Mapping):
        raise AblationConfigError(
            f"{method} loss.intermediate.group_weights must be a mapping."
        )
    parsed = {
        str(key): _number(
            weight, f"{method}.loss.intermediate.group_weights.{key}"
        )
        for key, weight in value.items()
    }
    expected = expected_group_weights(benchmark)
    if parsed != expected:
        if benchmark == "robocasa":
            raise AblationConfigError(
                f"{method} RoboCasa group_weights must be {{}} so all 29 "
                f"dimensions are supervised equally; got {parsed!r}."
            )
        raise AblationConfigError(
            f"{method} LIBERO group_weights must be "
            "{'position': 1.0, 'rotation': 1.0, 'gripper': 0.0}; "
            f"got {parsed!r}."
        )


def _value_at(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _check_benchmark_contract(
    config: Mapping[str, Any], benchmark: str, method: str
) -> None:
    for path, expected in BENCHMARK_CONTRACTS[benchmark].items():
        actual = _value_at(config, path)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            location = ".".join(path)
            raise AblationConfigError(
                f"{method} {benchmark} {location} must be {expected!r}, "
                f"got {actual!r}."
            )


def _check_clean_weighting(config: Mapping[str, Any], method: str) -> None:
    data = _mapping(config, "data", method)
    for key in ("balance_dataset_weights", "balance_trajectory_weights"):
        if data.get(key) is not False:
            raise AblationConfigError(f"{method} data.{key} must be false.")

    loss = _mapping(config, "loss", method)
    legacy = sorted(
        key
        for key in loss
        if key in {"scale_weight", "scale_loss_weights"}
        or str(key).endswith("_scale_weight")
    )
    if legacy:
        raise AblationConfigError(
            f"{method} uses forbidden legacy intermediate keys: {legacy!r}."
        )
    for key in _OPTIONAL_WEIGHTING_BLOCKS:
        if key not in loss:
            continue
        block = loss[key]
        if not isinstance(block, Mapping) or block.get("enabled") is not False:
            raise AblationConfigError(
                f"{method} loss.{key} must be absent or explicitly enabled=false."
            )


def validate_ablation_configs(
    configs: Mapping[str, Mapping[str, Any] | DictConfig],
    *,
    benchmark: str,
    require_portable_paths: bool = True,
) -> AblationValidation:
    """Validate four resolved configs, raising on any uncontrolled drift."""

    benchmark = _benchmark(benchmark)
    missing = [method for method in METHODS if method not in configs]
    extra = sorted(set(configs) - set(METHODS))
    if missing or extra:
        raise AblationConfigError(
            f"Expected exactly {list(METHODS)!r}; missing={missing!r}, extra={extra!r}."
        )
    plain = {method: _plain(configs[method]) for method in METHODS}
    names: dict[str, str] = {}
    outputs: dict[str, str] = {}
    aux_weights: dict[str, float] = {}

    for method, config in plain.items():
        for section in ("experiment", "data", "model", "loss", "train"):
            _mapping(config, section, method)
        if require_portable_paths:
            _check_paths(config, method)
        _check_benchmark_contract(config, benchmark, method)
        _check_clean_weighting(config, method)
        experiment = _mapping(config, "experiment", method)
        name = experiment.get("name")
        output = experiment.get("output_dir")
        if not isinstance(name, str) or not name:
            raise AblationConfigError(f"{method} experiment.name must be non-empty.")
        if not isinstance(output, str) or not output:
            raise AblationConfigError(f"{method} experiment.output_dir must be non-empty.")
        names[method], outputs[method] = name, output

        loss = _mapping(config, "loss", method)
        intermediate = _mapping(loss, "intermediate", f"{method}.loss")
        expected_mode = METHOD_MODES[method]
        if intermediate.get("mode") != expected_mode:
            raise AblationConfigError(
                f"{method} loss.intermediate.mode must be {expected_mode!r}."
            )
        weight = _number(
            intermediate.get("weight"), f"{method}.loss.intermediate.weight"
        )
        if method == "multiscale_base":
            if weight != 0:
                raise AblationConfigError("multiscale_base weight must be 0.0.")
        else:
            if weight <= 0:
                raise AblationConfigError(f"{method} weight must be > 0.")
            aux_weights[method] = weight
        if intermediate.get("include_final", False) is not False:
            raise AblationConfigError(f"{method} include_final must be false.")
        if intermediate.get("scale_weights") != "uniform":
            raise AblationConfigError(
                f"{method} loss.intermediate.scale_weights must be 'uniform'."
            )
        _check_groups(intermediate.get("group_weights"), benchmark, method)

        spectral = intermediate.get("spectral")
        if method == "mint_paper_dct":
            if spectral != PAPER_DCT_SPECTRAL:
                raise AblationConfigError(
                    f"mint_paper_dct spectral must be {PAPER_DCT_SPECTRAL!r}."
                )
        elif spectral not in (None, {}):
            raise AblationConfigError(
                f"{method} must not have non-empty spectral settings."
            )

    if len(set(names.values())) != len(METHODS):
        raise AblationConfigError(f"experiment.name must be unique: {names!r}.")
    if len(set(outputs.values())) != len(METHODS):
        raise AblationConfigError(f"experiment.output_dir must be unique: {outputs!r}.")
    if len(set(aux_weights.values())) != 1:
        raise AblationConfigError(
            f"All non-Base methods must share one weight, got {aux_weights!r}."
        )

    base = controlled_projection(plain["multiscale_base"])
    for method in METHODS[1:]:
        differences = _diff_paths(base, controlled_projection(plain[method]))
        if differences:
            suffix = " ..." if len(differences) > 20 else ""
            raise AblationConfigError(
                f"{method} differs outside the allowlist: "
                + ", ".join(differences[:20])
                + suffix
            )

    return AblationValidation(
        benchmark=benchmark,
        methods=METHODS,
        intermediate_weight=next(iter(aux_weights.values())),
    )


def load_and_validate_ablation_configs(
    config_paths: Mapping[str, str | os.PathLike[str]],
    *,
    benchmark: str,
    require_portable_paths: bool = True,
) -> tuple[dict[str, dict[str, Any]], AblationValidation]:
    configs = {method: load_config(path) for method, path in config_paths.items()}
    report = validate_ablation_configs(
        configs,
        benchmark=benchmark,
        require_portable_paths=require_portable_paths,
    )
    return configs, report


def _config_args(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise AblationConfigError(f"Invalid --config {value!r}; use METHOD=PATH.")
        method, raw_path = (part.strip() for part in value.split("=", 1))
        if method not in METHOD_MODES or not raw_path:
            raise AblationConfigError(f"Invalid --config assignment {value!r}.")
        if method in result:
            raise AblationConfigError(f"Duplicate --config for {method!r}.")
        result[method] = Path(raw_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(DEFAULT_CANONICAL_PATHS))
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--config", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--intermediate-weight", type=float)
    parser.add_argument("--print-configs", action="store_true")
    parser.add_argument(
        "--allow-runtime-paths",
        action="store_true",
        help="Allow absolute/runtime paths; canonical validation is strict by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.config:
            if args.canonical is not None or args.intermediate_weight is not None:
                raise AblationConfigError(
                    "--config cannot be combined with --canonical or --intermediate-weight."
                )
            configs, report = load_and_validate_ablation_configs(
                _config_args(args.config),
                benchmark=args.benchmark,
                require_portable_paths=not args.allow_runtime_paths,
            )
        else:
            canonical_path = args.canonical
            if canonical_path is None:
                repo_root = Path(__file__).resolve().parents[2]
                canonical_path = repo_root / DEFAULT_CANONICAL_PATHS[args.benchmark]
            configs = materialize_ablation_configs(
                load_config(canonical_path),
                benchmark=args.benchmark,
                intermediate_weight=args.intermediate_weight,
                require_portable_paths=not args.allow_runtime_paths,
            )
            report = validate_ablation_configs(
                configs,
                benchmark=args.benchmark,
                require_portable_paths=not args.allow_runtime_paths,
            )
    except (AblationConfigError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {"ok": True, **report.to_dict()}
    if args.print_configs:
        payload["configs"] = configs
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
