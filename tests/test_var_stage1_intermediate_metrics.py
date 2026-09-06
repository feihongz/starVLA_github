from __future__ import annotations

import math

import pytest
import torch

from starVLA.training.intermediate_supervision import build_temporal_scale_target
from starVLA.utils import var_stage1_metrics as metrics
from starVLA.utils.var_stage1_metrics import (
    METRIC_VERSION,
    finalize_intermediate_mse_stats,
    resolve_intermediate_metric_dims,
    update_intermediate_mse_stats,
)


LIBERO_GROUPS = {
    "position": [0, 1, 2],
    "rotation": [3, 4, 5],
    "gripper": [6],
}


def _update(
    actions: torch.Tensor,
    recons: list[torch.Tensor],
    scales: list[int],
    dims: list[int],
) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = {}
    update_intermediate_mse_stats(
        stats,
        actions=actions,
        scale_recons=recons,
        scales=scales,
        included_dims=dims,
    )
    return stats


def test_metric_version_is_locked() -> None:
    assert METRIC_VERSION == "scale_aligned_down_up_v1"


def test_resolve_libero_dims_excludes_the_explicit_binary_gripper() -> None:
    assert resolve_intermediate_metric_dims(
        benchmark="libero",
        action_dim=7,
        dim_groups=LIBERO_GROUPS,
    ) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("benchmark", "action_dim", "dim_groups", "message"),
    [
        ("LIBERO", 7, LIBERO_GROUPS, "Unsupported benchmark"),
        ("unknown", 7, LIBERO_GROUPS, "Unsupported benchmark"),
        ("libero", 6, LIBERO_GROUPS, "action_dim=7"),
        ("libero", 7, {}, "explicitly define"),
        ("libero", 7, {"gripper": [5]}, "exactly \\[6\\]"),
        ("libero", 7, {"gripper": [6, 6]}, "unique"),
        ("robocasa", 28, {}, "action_dim=29"),
    ],
)
def test_resolve_dims_fails_closed(
    benchmark: str,
    action_dim: int,
    dim_groups: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_intermediate_metric_dims(
            benchmark=benchmark,
            action_dim=action_dim,
            dim_groups=dim_groups,
        )


def test_robocasa_includes_all_29_continuous_dimensions() -> None:
    dims = resolve_intermediate_metric_dims(
        benchmark="robocasa",
        action_dim=29,
        dim_groups={},
    )
    assert dims == list(range(29))

    actions = torch.zeros(1, 16, 29)
    recon = build_temporal_scale_target(actions, 1)
    recon[..., 28] = 2.0
    stats = _update(actions, [recon], [1], dims)
    overall, per_scale = finalize_intermediate_mse_stats(stats)

    assert stats[1]["count"] == 1 * 16 * 29
    assert per_scale[1] == pytest.approx(64.0 / (1 * 16 * 29))
    assert overall == per_scale[1]


def test_zero_error_and_scale_one_target_is_the_repeated_time_mean() -> None:
    actions = torch.tensor(
        [[[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]]
    )
    expected = actions.mean(dim=1, keepdim=True).expand_as(actions)
    target = build_temporal_scale_target(actions, 1)
    assert torch.equal(target, expected)

    stats = _update(actions, [target], [1], [0, 1])
    overall, per_scale = finalize_intermediate_mse_stats(stats)
    assert stats == {1: {"sse": 0.0, "count": 8}}
    assert per_scale == {1: 0.0}
    assert overall == 0.0


def test_single_element_error_has_the_exact_analytic_contribution() -> None:
    actions = torch.zeros(1, 4, 2)
    recon = build_temporal_scale_target(actions, 1)
    recon[0, 0, 0] = 2.0

    stats = _update(actions, [recon], [1], [0, 1])
    overall, per_scale = finalize_intermediate_mse_stats(stats)

    assert stats == {1: {"sse": 4.0, "count": 8}}
    assert per_scale == {1: 0.5}
    assert overall == 0.5


def test_sse_uses_a_float64_reduction() -> None:
    actions = torch.zeros(1, 7, 1, dtype=torch.float32)
    recon = torch.full_like(actions, 0.1)
    expected_sse = float(recon.square().sum(dtype=torch.float64).item())

    stats = _update(actions, [recon], [1], [0])

    assert stats[1]["sse"] == expected_sse
    assert isinstance(stats[1]["sse"], float)
    assert stats[1]["count"] == recon.numel()


def test_libero_gripper_error_is_excluded_but_arm_error_is_counted() -> None:
    actions = torch.zeros(1, 8, 7)
    dims = resolve_intermediate_metric_dims(
        benchmark="libero", action_dim=7, dim_groups=LIBERO_GROUPS
    )
    gripper_only = build_temporal_scale_target(actions, 1)
    gripper_only[..., 6] = 100.0

    gripper_stats = _update(actions, [gripper_only], [1], dims)
    assert finalize_intermediate_mse_stats(gripper_stats)[0] == 0.0

    arm_error = gripper_only.clone()
    arm_error[..., 0] = 1.0
    arm_stats = _update(actions, [arm_error], [1], dims)
    assert finalize_intermediate_mse_stats(arm_stats)[0] == pytest.approx(1.0 / 6.0)


def test_each_scale_uses_the_shared_canonical_target_builder(monkeypatch) -> None:
    actions = torch.zeros(2, 8, 3)
    calls: list[int] = []
    canonical_builder = metrics.build_temporal_scale_target

    def counted_builder(value: torch.Tensor, scale: int) -> torch.Tensor:
        calls.append(scale)
        return canonical_builder(value, scale)

    monkeypatch.setattr(metrics, "build_temporal_scale_target", counted_builder)
    recons = [canonical_builder(actions, scale) for scale in (1, 2, 4)]
    stats = _update(actions, recons, [1, 2, 4], [0, 1, 2])

    assert calls == [1, 2, 4]
    assert finalize_intermediate_mse_stats(stats)[0] == 0.0


def test_per_scale_mse_is_equal_weighted_after_elementwise_aggregation() -> None:
    actions = torch.zeros(1, 4, 1)
    scale_one = torch.ones_like(actions)
    scale_two = torch.full_like(actions, 3.0)

    overall, per_scale = finalize_intermediate_mse_stats(
        _update(actions, [scale_one, scale_two], [1, 2], [0])
    )

    assert per_scale == {1: 1.0, 2: 9.0}
    assert overall == 5.0


def test_non_divisible_batching_matches_one_shot_elementwise_aggregation() -> None:
    actions = torch.zeros(5, 4, 3)
    scale_one = torch.zeros_like(actions)
    scale_two = torch.zeros_like(actions)
    for sample in range(5):
        scale_one[sample, :, 0] = float(sample)
        scale_two[sample, :, 1] = float(sample + 1)

    full_stats = _update(actions, [scale_one, scale_two], [1, 2], [0, 1])

    chunked_stats: dict[int, dict[str, float | int]] = {}
    for start, end in ((0, 2), (2, 4), (4, 5)):
        update_intermediate_mse_stats(
            chunked_stats,
            actions=actions[start:end],
            scale_recons=[scale_one[start:end], scale_two[start:end]],
            scales=[1, 2],
            included_dims=[0, 1],
        )

    assert chunked_stats == full_stats
    assert finalize_intermediate_mse_stats(chunked_stats) == finalize_intermediate_mse_stats(
        full_stats
    )


def test_caller_must_explicitly_exclude_the_final_scale() -> None:
    actions = torch.zeros(1, 8, 2)
    all_scales = [1, 2, 4, 8]
    all_recons = [build_temporal_scale_target(actions, scale) for scale in all_scales]

    with pytest.raises(ValueError, match="caller must explicitly exclude the final scale"):
        _update(actions, all_recons, all_scales, [0, 1])

    stats = _update(actions, all_recons[:-1], all_scales[:-1], [0, 1])
    assert list(finalize_intermediate_mse_stats(stats)[1]) == [1, 2, 4]


@pytest.mark.parametrize(
    ("dims", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "must be unique"),
        ([-1], "out-of-range"),
        ([2], "out-of-range"),
    ],
)
def test_invalid_included_dims_are_rejected(dims: list[int], message: str) -> None:
    actions = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match=message):
        _update(actions, [actions.clone()], [1], dims)


@pytest.mark.parametrize(
    ("scales", "recon_count", "message"),
    [
        ([], 0, "At least one"),
        ([1, 1], 2, "must be unique"),
        ([0], 1, "strictly below"),
        ([4], 1, "explicitly exclude"),
        ([5], 1, "strictly below"),
        ([1, 2], 1, "equal lengths"),
    ],
)
def test_invalid_scales_are_rejected(
    scales: list[int], recon_count: int, message: str
) -> None:
    actions = torch.zeros(1, 4, 2)
    recons = [actions.clone() for _ in range(recon_count)]
    with pytest.raises(ValueError, match=message):
        _update(actions, recons, scales, [0, 1])


def test_bad_tensor_shape_dtype_and_values_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \[B, H, D\]"):
        _update(torch.zeros(4, 2), [torch.zeros(4, 2)], [1], [0])
    with pytest.raises(TypeError, match="floating dtype"):
        _update(
            torch.zeros(1, 4, 2, dtype=torch.int64),
            [torch.zeros(1, 4, 2, dtype=torch.float32)],
            [1],
            [0],
        )

    actions = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="does not match"):
        _update(actions, [torch.zeros(1, 4, 3)], [1], [0])
    non_finite = actions.clone()
    non_finite[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        _update(actions, [non_finite], [1], [0])


def test_failed_update_does_not_partially_mutate_stats() -> None:
    actions = torch.zeros(1, 4, 2)
    stats = {
        1: {"sse": 2.0, "count": 8},
        2: {"sse": 3.0, "count": 8},
    }
    before = {scale: dict(bucket) for scale, bucket in stats.items()}
    bad_second_recon = torch.zeros(1, 4, 3)

    with pytest.raises(ValueError, match="does not match"):
        update_intermediate_mse_stats(
            stats,
            actions=actions,
            scale_recons=[actions.clone(), bad_second_recon],
            scales=[1, 2],
            included_dims=[0, 1],
        )
    assert stats == before


def test_updates_cannot_mix_different_scale_sets() -> None:
    actions = torch.zeros(1, 4, 2)
    stats = _update(actions, [actions.clone()], [1], [0, 1])
    with pytest.raises(ValueError, match="same intermediate scales"):
        update_intermediate_mse_stats(
            stats,
            actions=actions,
            scale_recons=[actions.clone()],
            scales=[2],
            included_dims=[0, 1],
        )


@pytest.mark.parametrize(
    ("stats", "exception", "message"),
    [
        ({}, ValueError, "empty"),
        ({1: {"sse": 0.0, "count": 0}}, ValueError, "positive"),
        ({1: {"sse": -1.0, "count": 1}}, ValueError, "non-negative"),
        ({1: {"sse": math.inf, "count": 1}}, ValueError, "finite"),
        ({0: {"sse": 0.0, "count": 1}}, ValueError, "positive"),
    ],
)
def test_finalize_rejects_invalid_stats(stats, exception, message) -> None:
    with pytest.raises(exception, match=message):
        finalize_intermediate_mse_stats(stats)
