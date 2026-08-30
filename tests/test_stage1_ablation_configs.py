from __future__ import annotations

import copy
import json

import pytest
from omegaconf import OmegaConf

from scripts.stage1.validate_ablation_configs import (
    METHODS,
    PAPER_DCT_SPECTRAL,
    AblationConfigError,
    main,
    materialize_ablation_configs,
    validate_ablation_configs,
)


def canonical_config(benchmark: str):
    robocasa = benchmark == "robocasa"
    return {
        "experiment": {
            "name": f"{benchmark}_clean_ablation",
            "output_dir": f"playground/Checkpoints/{benchmark}_clean_ablation",
            "seed": 42,
        },
        "data": {
            "data_root_dir": f"playground/Datasets/{benchmark}",
            "data_mix": f"{benchmark}_all",
            "expected_action_horizon": 16 if robocasa else 8,
            "expected_action_dim": 29 if robocasa else 7,
            "balance_dataset_weights": False,
            "balance_trajectory_weights": False,
            **({"action_mode": "abs"} if robocasa else {}),
        },
        "model": {
            "embed_dim": 64 if robocasa else 32,
            "codebook_size": 512,
            "quantization_mode": "product_vq",
            "product_codebook_groups": 16,
            "scales": [1, 2, 4, 8, 16] if robocasa else [1, 2, 4, 8],
            "decoder_head_type": "plain",
            "use_time_embedding": True,
            "use_action_type_embedding": False if robocasa else True,
        },
        "loss": {
            "recon_weight": 1.0,
            "intermediate": {
                "mode": "mtr",
                "weight": 0.1 if robocasa else 0.02,
                "include_final": False,
                "scale_weights": "uniform",
                "group_weights": (
                    {}
                    if robocasa
                    else {"position": 1.0, "rotation": 1.0, "gripper": 0.0}
                ),
            },
        },
        "train": {
            "epochs": 50,
            "batch_size": 256,
            "learning_rate": 5.0e-5,
            "init_checkpoint": f"playground/Checkpoints/{benchmark}_ae/best_recon.ckpt",
        },
    }


@pytest.mark.parametrize(
    ("benchmark", "expected_weight"),
    [("libero", 0.02), ("robocasa", 0.1)],
)
def test_materializes_and_validates_the_locked_four_methods(
    benchmark, expected_weight
):
    canonical = canonical_config(benchmark)
    original = copy.deepcopy(canonical)

    configs = materialize_ablation_configs(canonical, benchmark=benchmark)
    report = validate_ablation_configs(configs, benchmark=benchmark)

    assert tuple(configs) == METHODS
    assert canonical == original
    assert report.intermediate_weight == pytest.approx(expected_weight)
    assert configs["multiscale_base"]["loss"]["intermediate"]["mode"] == "none"
    assert configs["multiscale_base"]["loss"]["intermediate"]["weight"] == 0.0
    for method in METHODS[1:]:
        assert configs[method]["loss"]["intermediate"]["weight"] == pytest.approx(
            expected_weight
        )
    assert (
        configs["mint_paper_dct"]["loss"]["intermediate"]["spectral"]
        == PAPER_DCT_SPECTRAL
    )
    assert "spectral" not in configs["mtr"]["loss"]["intermediate"]


def test_libero_locks_position_rotation_and_excludes_gripper():
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    expected = {"position": 1.0, "rotation": 1.0, "gripper": 0.0}
    assert all(
        config["loss"]["intermediate"]["group_weights"] == expected
        for config in configs.values()
    )


def test_robocasa_preserves_all_29_dim_semantics_and_plain_decoder():
    configs = materialize_ablation_configs(
        canonical_config("robocasa"), benchmark="robocasa"
    )
    assert all(
        config["loss"]["intermediate"]["group_weights"] == {}
        for config in configs.values()
    )

    bad_groups = copy.deepcopy(configs)
    bad_groups["mtr"]["loss"]["intermediate"]["group_weights"] = {"hand": 0.0}
    with pytest.raises(AblationConfigError, match="all 29 dimensions"):
        validate_ablation_configs(bad_groups, benchmark="robocasa")

    bad_embedding = copy.deepcopy(configs)
    bad_embedding["mtr"]["model"]["use_action_type_embedding"] = True
    with pytest.raises(AblationConfigError, match="use_action_type_embedding"):
        validate_ablation_configs(bad_embedding, benchmark="robocasa")

    bad_decoder = copy.deepcopy(configs)
    bad_decoder["mtr"]["model"]["decoder_head_type"] = "structured"
    with pytest.raises(AblationConfigError, match="decoder_head_type"):
        validate_ablation_configs(bad_decoder, benchmark="robocasa")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("train", "batch_size"), 128),
        (("model", "embed_dim"), 128),
        (("data", "data_mix"), "different_mix"),
        (("experiment", "seed"), 7),
    ],
)
def test_rejects_any_difference_outside_the_allowlist(path, value):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    configs["mtr"][path[0]][path[1]] = value

    with pytest.raises(AblationConfigError, match=rf"{path[0]}\.{path[1]}"):
        validate_ablation_configs(configs, benchmark="libero")


def test_rejects_method_specific_auxiliary_weight():
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    configs["mtr"]["loss"]["intermediate"]["weight"] = 0.2
    with pytest.raises(AblationConfigError, match="share one weight"):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize(
    "absolute_path",
    ["/root/private/data", "/mnt/author/checkpoint.ckpt", "C:\\Users\\author\\data"],
)
def test_rejects_machine_specific_absolute_paths(absolute_path):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    for config in configs.values():
        config["data"]["data_root_dir"] = absolute_path
    with pytest.raises(AblationConfigError, match="non-portable"):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize(
    "nonportable_path",
    [
        "\\\\server\\share",
        "~/data",
        "~alice/data",
        "$HOME/data",
        "${HOME}/data",
        "${oc.env:HOME}/data",
        "${oc.env:HOME,/tmp/fallback}/data",
    ],
)
def test_rejects_home_expansion_tilde_and_unc_paths(nonportable_path):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    for config in configs.values():
        config["data"]["data_root_dir"] = nonportable_path
    with pytest.raises(AblationConfigError, match="non-portable"):
        validate_ablation_configs(configs, benchmark="libero")


def test_runtime_configs_can_explicitly_allow_absolute_paths():
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    for config in configs.values():
        config["data"]["data_root_dir"] = "/runtime-mounted/libero"
    report = validate_ablation_configs(
        configs, benchmark="libero", require_portable_paths=False
    )
    assert report.benchmark == "libero"


def test_rejects_wrong_dct_contract_and_final_scale_supervision():
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    configs["mint_paper_dct"]["loss"]["intermediate"]["spectral"] = {
        "formulation": "ortho_dct_ii_mse",
        "normalization": "none",
    }
    with pytest.raises(AblationConfigError, match="spectral"):
        validate_ablation_configs(configs, benchmark="libero")

    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    configs["mtr"]["loss"]["intermediate"]["include_final"] = True
    with pytest.raises(AblationConfigError, match="include_final"):
        validate_ablation_configs(configs, benchmark="libero")


def test_cli_materializes_a_canonical_yaml(tmp_path, capsys):
    path = tmp_path / "canonical.yaml"
    OmegaConf.save(OmegaConf.create(canonical_config("libero")), path)

    assert main(["--benchmark", "libero", "--canonical", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "benchmark": "libero",
        "intermediate_weight": 0.02,
        "methods": list(METHODS),
        "ok": True,
    }


def _set_all(configs, path, value):
    for config in configs.values():
        parent = config
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = copy.deepcopy(value)


@pytest.mark.parametrize(
    ("benchmark", "path", "value"),
    [
        ("libero", ("data", "expected_action_horizon"), 16),
        ("libero", ("data", "expected_action_dim"), 29),
        ("libero", ("model", "embed_dim"), 64),
        ("libero", ("model", "codebook_size"), 1024),
        ("libero", ("model", "quantization_mode"), "vq"),
        ("libero", ("model", "product_codebook_groups"), 8),
        ("libero", ("model", "scales"), [1, 2, 4, 16]),
        ("libero", ("model", "use_time_embedding"), False),
        ("libero", ("model", "use_action_type_embedding"), False),
        ("robocasa", ("data", "expected_action_horizon"), 8),
        ("robocasa", ("data", "expected_action_dim"), 28),
        ("robocasa", ("data", "action_mode"), "delta"),
        ("robocasa", ("model", "embed_dim"), 256),
        ("robocasa", ("model", "scales"), [1, 2, 4, 8]),
        ("robocasa", ("model", "use_action_type_embedding"), True),
    ],
)
def test_rejects_benchmark_contract_drift_shared_by_all_methods(
    benchmark, path, value
):
    configs = materialize_ablation_configs(
        canonical_config(benchmark), benchmark=benchmark
    )
    _set_all(configs, path, value)

    with pytest.raises(AblationConfigError, match="must be"):
        validate_ablation_configs(configs, benchmark=benchmark)


def test_rejects_nonuniform_scale_weights_shared_by_all_methods():
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    _set_all(configs, ("loss", "intermediate", "scale_weights"), {1: 1.0})
    with pytest.raises(AblationConfigError, match="scale_weights.*uniform"):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scale_weight", 0.1),
        ("scale_loss_weights", {1: 1.0}),
        ("position_scale_weight", 1.0),
    ],
)
def test_rejects_legacy_intermediate_aliases_shared_by_all_methods(key, value):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    _set_all(configs, ("loss", key), value)
    with pytest.raises(AblationConfigError, match="forbidden legacy"):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize(
    "key",
    [
        "sample_weighting",
        "trajectory_weighting",
        "trajectory_phase_weighting",
        "time_weighting",
        "task_balance_weighting",
        "adaptive_task_weighting",
    ],
)
def test_rejects_extra_weighting_shared_by_all_methods(key):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    _set_all(configs, ("loss", key), {"enabled": True})
    with pytest.raises(AblationConfigError, match=key):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize("key", ["balance_dataset_weights", "balance_trajectory_weights"])
def test_rejects_data_balancing_shared_by_all_methods(key):
    configs = materialize_ablation_configs(canonical_config("libero"), benchmark="libero")
    _set_all(configs, ("data", key), True)
    with pytest.raises(AblationConfigError, match=key):
        validate_ablation_configs(configs, benchmark="libero")


@pytest.mark.parametrize(("benchmark", "weight"), [("libero", 0.02), ("robocasa", 0.1)])
def test_default_committed_canonical_cli(benchmark, weight, capsys):
    assert main(["--benchmark", benchmark]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["benchmark"] == benchmark
    assert payload["intermediate_weight"] == pytest.approx(weight)
