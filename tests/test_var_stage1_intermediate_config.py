from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from starVLA.training.intermediate_supervision import (
    ResolvedIntermediateSupervision,
    resolve_intermediate_supervision,
)
from starVLA.training.train_var_stage1 import merge_config_overrides


LIBERO_GROUPS = {
    "position": [0, 1, 2],
    "rotation": [3, 4, 5],
    "gripper": [6],
}
LIBERO_INTERMEDIATE_SCALES = [1, 2, 4]
ROBOCASA_LEGACY_GROUPS: dict[str, list[int]] = {}
ROBOCASA_INTERMEDIATE_SCALES = [1, 2, 4, 8]


def resolve(
    loss_cfg,
    *,
    dim_groups=LIBERO_GROUPS,
    available_scales=LIBERO_INTERMEDIATE_SCALES,
):
    return resolve_intermediate_supervision(
        loss_cfg,
        dim_groups=dim_groups,
        available_scales=available_scales,
    )


@pytest.mark.parametrize(
    "loss_cfg",
    [{}, {"recon_weight": 1.0}, {"scale_weight": 0.0}],
)
def test_legacy_missing_or_zero_scale_weight_resolves_to_base(loss_cfg):
    resolved = resolve(loss_cfg)

    assert isinstance(resolved, ResolvedIntermediateSupervision)
    assert resolved.mode == "none"
    assert resolved.weight == 0.0
    assert resolved.include_final is False
    assert resolved.scale_weights == "uniform"
    assert resolved.group_weights == {}
    assert resolved.spectral == {}
    assert resolved.resolved_from == "legacy"
    assert resolved.enabled is False


def test_disabled_legacy_config_ignores_stale_invalid_scale_loss_weights():
    resolved = resolve(
        {
            "scale_weight": 0.0,
            # This field is inert when legacy MTR itself is disabled. Old Base
            # configs must not become invalid merely because it is present.
            "scale_loss_weights": {8: -1.0},
        }
    )

    assert resolved.mode == "none"
    assert resolved.scale_weights == "uniform"


def test_legacy_positive_scale_weight_resolves_to_mtr_with_libero_gripper_mask():
    resolved = resolve({"scale_weight": 0.1})

    assert resolved.mode == "mtr"
    assert resolved.weight == pytest.approx(0.1)
    assert resolved.group_weights == {"gripper": 0.0}
    assert resolved.enabled is True
    assert resolved.resolved_from == "legacy"


def test_legacy_robocasa_empty_groups_preserve_all_dimensional_mse_behavior():
    resolved = resolve(
        {
            "scale_weight": 0.1,
            "position_scale_weight": 1.0,
            "rotation_scale_weight": 1.0,
            "gripper_scale_weight": 0.0,
        },
        dim_groups=ROBOCASA_LEGACY_GROUPS,
        available_scales=ROBOCASA_INTERMEDIATE_SCALES,
    )

    assert resolved.mode == "mtr"
    assert resolved.group_weights == {}


@pytest.mark.parametrize(
    ("mode", "weight", "expected_spectral"),
    [
        ("none", 0.0, {}),
        ("full_target_time", 0.05, {}),
        (
            "mint_paper_dct",
            0.05,
            {
                "formulation": "raw_dct_ii_mse",
                "normalization": "2_over_h",
            },
        ),
        ("mtr", 0.05, {}),
    ],
)
def test_all_native_modes_resolve_to_canonical_configuration(
    mode,
    weight,
    expected_spectral,
):
    resolved = resolve(
        {
            "intermediate": {
                "mode": mode,
                "weight": weight,
                "include_final": False,
                "scale_weights": "uniform",
                "group_weights": {},
            }
        }
    )

    assert resolved.mode == mode
    assert resolved.weight == pytest.approx(weight)
    assert resolved.scale_weights == "uniform"
    assert resolved.group_weights == {}
    assert resolved.spectral == expected_spectral
    assert resolved.resolved_from == "native"
    assert resolved.enabled is (mode != "none")


@pytest.mark.parametrize(
    ("mode", "weight"),
    [
        ("none", 0.1),
        ("mtr", 0.0),
        ("full_target_time", 0.0),
        ("mint_paper_dct", 0.0),
        ("mtr", -0.1),
        ("mtr", math.inf),
        ("mtr", math.nan),
    ],
)
def test_native_mode_and_weight_must_be_consistent(mode, weight):
    with pytest.raises(ValueError, match="mode|weight|finite|non-negative"):
        resolve({"intermediate": {"mode": mode, "weight": weight}})


def test_unknown_native_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown intermediate supervision mode"):
        resolve({"intermediate": {"mode": "not_a_method", "weight": 0.1}})


def test_final_scale_cannot_be_included_in_auxiliary_supervision():
    with pytest.raises(ValueError, match="include_final|final reconstruction"):
        resolve(
            {
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "include_final": True,
                }
            }
        )


@pytest.mark.parametrize(
    ("scale_weights", "message"),
    [
        ({8: 1.0}, "unknown intermediate scales"),
        ({1: -0.1}, "non-negative"),
        ({1: math.inf}, "finite"),
        ({1: 0.0, 2: 0.0, 4: 0.0}, "at least one"),
        ("linear_schedule", "mapping"),
    ],
)
def test_invalid_native_scale_weights_are_rejected(scale_weights, message):
    with pytest.raises(ValueError, match=f"(?i){message}"):
        resolve(
            {
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "scale_weights": scale_weights,
                }
            }
        )


def test_enabled_intermediate_supervision_requires_an_intermediate_scale():
    with pytest.raises(ValueError, match="at least one scale"):
        resolve(
            {"intermediate": {"mode": "mtr", "weight": 0.1}},
            available_scales=[],
        )


@pytest.mark.parametrize(
    ("group_weights", "message"),
    [
        ({"not_a_group": 1.0}, "unknown action groups"),
        ({"gripper": -0.1}, "non-negative"),
        ({"position": math.nan}, "finite"),
    ],
)
def test_invalid_native_group_weights_are_rejected(group_weights, message):
    with pytest.raises(ValueError, match=message):
        resolve(
            {
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "group_weights": group_weights,
                }
            }
        )


def test_paper_dct_accepts_only_the_locked_raw_spectral_formulation():
    resolved = resolve(
        {
            "intermediate": {
                "mode": "mint_paper_dct",
                "weight": 0.1,
                "spectral": {
                    "formulation": "raw_dct_ii_mse",
                    "normalization": "2_over_h",
                },
            }
        }
    )

    assert resolved.spectral == {
        "formulation": "raw_dct_ii_mse",
        "normalization": "2_over_h",
    }


@pytest.mark.parametrize(
    "spectral",
    [
        {"formulation": "orthonormal_dct_ii_mse", "normalization": "none"},
        {"formulation": "raw_dct_ii_mse", "normalization": "ortho"},
    ],
)
def test_paper_dct_rejects_non_paper_spectral_variants(spectral):
    with pytest.raises(ValueError, match="raw_dct_ii_mse|2_over_h"):
        resolve(
            {
                "intermediate": {
                    "mode": "mint_paper_dct",
                    "weight": 0.1,
                    "spectral": spectral,
                }
            }
        )


def test_non_dct_mode_rejects_spectral_configuration():
    with pytest.raises(ValueError, match="spectral.*only valid"):
        resolve(
            {
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "spectral": {
                        "formulation": "raw_dct_ii_mse",
                        "normalization": "2_over_h",
                    },
                }
            }
        )


def test_paper_dct_rejects_time_weighting():
    with pytest.raises(ValueError, match="time_weighting|frequency"):
        resolve(
            {
                "time_weighting": {"enabled": True},
                "intermediate": {
                    "mode": "mint_paper_dct",
                    "weight": 0.1,
                },
            }
        )


def test_paper_dct_allows_explicitly_disabled_time_weighting():
    resolved = resolve(
        {
            "time_weighting": {"enabled": False},
            "intermediate": {
                "mode": "mint_paper_dct",
                "weight": 0.1,
            },
        }
    )

    assert resolved.mode == "mint_paper_dct"


def test_equivalent_native_and_legacy_mtr_fields_are_accepted():
    resolved = resolve(
        {
            "scale_weight": 0.1,
            "scale_loss_weights": {1: 0.25, 2: 0.5, 4: 1.0},
            "position_scale_weight": 0.75,
            "rotation_scale_weight": 1.0,
            "gripper_scale_weight": 0.0,
            "intermediate": {
                "mode": "mtr",
                "weight": 0.1,
                "scale_weights": {1: 0.25, 2: 0.5, 4: 1.0},
                "group_weights": {
                    "position": 0.75,
                    "rotation": 1.0,
                    "gripper": 0.0,
                },
            },
        }
    )

    assert resolved.mode == "mtr"
    assert resolved.scale_weights == {1: 0.25, 2: 0.5, 4: 1.0}
    assert resolved.group_weights["gripper"] == 0.0


@pytest.mark.parametrize(
    "legacy_loss_cfg",
    [
        {"recon_weight": 1.0, "scale_weight": 0.0},
        {
            "recon_weight": 1.0,
            "scale_weight": 0.1,
            "scale_loss_weights": {1: 0.25, 2: 0.5, 4: 1.0},
            "position_scale_weight": 1.0,
            "rotation_scale_weight": 1.0,
            "gripper_scale_weight": 0.0,
        },
    ],
)
def test_legacy_config_roundtrips_after_trainer_saves_resolved_native_block(
    legacy_loss_cfg,
):
    # The trainer preserves legacy aliases but adds the resolved canonical
    # block before saving config.yaml/checkpoints. Reloading that artifact must
    # accept the synonymous old and new fields without changing semantics.
    first = resolve(legacy_loss_cfg)
    saved_loss_cfg = OmegaConf.create(legacy_loss_cfg)
    saved_loss_cfg.intermediate = OmegaConf.create(first.to_dict())
    reloaded_loss_cfg = OmegaConf.create(OmegaConf.to_yaml(saved_loss_cfg))

    second = resolve(reloaded_loss_cfg)

    assert second.mode == first.mode
    assert second.weight == first.weight
    assert second.include_final == first.include_final
    assert second.scale_weights == first.scale_weights
    assert second.group_weights == first.group_weights
    assert second.spectral == first.spectral
    assert first.resolved_from == "legacy"
    assert second.resolved_from == "native"


@pytest.mark.parametrize(
    ("loss_cfg", "message"),
    [
        (
            {
                "scale_weight": 0.1,
                "intermediate": {
                    "mode": "full_target_time",
                    "weight": 0.1,
                },
            },
            "Conflicting native and legacy",
        ),
        (
            {
                "scale_weight": 0.1,
                "scale_loss_weights": {1: 0.5},
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "scale_weights": {1: 0.75},
                    "group_weights": {"gripper": 0.0},
                },
            },
            "scale_weights",
        ),
        (
            {
                "scale_weight": 0.1,
                "gripper_scale_weight": 0.0,
                "intermediate": {
                    "mode": "mtr",
                    "weight": 0.1,
                    "group_weights": {"gripper": 1.0},
                },
            },
            "group weight",
        ),
    ],
)
def test_conflicting_native_and_legacy_fields_are_rejected(loss_cfg, message):
    with pytest.raises(ValueError, match=message):
        resolve(loss_cfg)


def test_resolved_config_serializes_canonical_scale_keys():
    resolved = resolve(
        {
            "intermediate": {
                "mode": "mtr",
                "weight": 0.1,
                "scale_weights": {1: 0.5, 2: 1.0},
                "group_weights": {"gripper": 0.0},
            }
        }
    )

    assert resolved.to_dict() == {
        "mode": "mtr",
        "weight": 0.1,
        "include_final": False,
        "scale_weights": {"1": 0.5, "2": 1.0},
        "group_weights": {"gripper": 0.0},
        "spectral": {},
        "resolved_from": "native",
    }


def test_repeatable_dotlist_overrides_are_merged_in_cli_order():
    cfg = OmegaConf.create(
        {
            "name": "base",
            "train": {"epochs": 100, "seed": 42},
            "loss": {"intermediate": {"mode": "none", "weight": 0.0}},
        }
    )

    merged = merge_config_overrides(
        cfg,
        [
            "name=mtr_smoke",
            "train.epochs=2",
            "loss.intermediate.mode=mtr",
            "loss.intermediate.weight=0.05",
            "train.epochs=3",
        ],
    )

    assert merged.name == "mtr_smoke"
    assert merged.train.epochs == 3
    assert merged.train.seed == 42
    assert merged.loss.intermediate.mode == "mtr"
    assert merged.loss.intermediate.weight == pytest.approx(0.05)
    assert cfg.name == "base"
    assert cfg.train.epochs == 100
    assert cfg.loss.intermediate.mode == "none"


def test_no_cli_overrides_returns_the_original_config():
    cfg = OmegaConf.create({"train": {"epochs": 10}})

    assert merge_config_overrides(cfg, []) is cfg
    assert merge_config_overrides(cfg, None) is cfg
