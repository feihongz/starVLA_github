"""Offline metrics for intermediate VAR Stage 1 trajectory reconstructions.

The metric in this module deliberately has one target definition for every
training method.  It compares each *cumulative* intermediate reconstruction
with the canonical temporally downsampled-and-upsampled action trajectory and
aggregates element-wise squared error over the complete dataset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Integral, Real

import torch

from starVLA.training.intermediate_supervision import build_temporal_scale_target


METRIC_VERSION = "scale_aligned_down_up_v1"


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _validate_dims(included_dims: Sequence[int], *, action_dim: int) -> list[int]:
    if isinstance(included_dims, (str, bytes)) or not isinstance(included_dims, Sequence):
        raise TypeError("included_dims must be a sequence of integer indices.")

    dims: list[int] = []
    for index, dim in enumerate(included_dims):
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            raise TypeError(
                f"included_dims[{index}] must be an integer, got {type(dim).__name__}."
            )
        dims.append(int(dim))

    if not dims:
        raise ValueError("included_dims must not be empty.")
    if len(set(dims)) != len(dims):
        raise ValueError(f"included_dims must be unique, got {dims}.")
    out_of_range = [dim for dim in dims if dim < 0 or dim >= action_dim]
    if out_of_range:
        raise ValueError(
            f"included_dims must lie in [0, {action_dim}), got out-of-range values "
            f"{out_of_range}."
        )
    return dims


def resolve_intermediate_metric_dims(
    *,
    benchmark: str,
    action_dim: int,
    dim_groups: dict[str, list[int]],
) -> list[int]:
    """Resolve the canonical action dimensions for the intermediate metric.

    The benchmark contract is intentionally fail-closed.  LIBERO must use its
    canonical seven-dimensional action convention with the binary gripper at
    index 6.  RoboCasa must use all 29 continuous action dimensions.
    """

    if not isinstance(benchmark, str):
        raise TypeError(f"benchmark must be a string, got {type(benchmark).__name__}.")
    resolved_action_dim = _require_positive_int(action_dim, name="action_dim")
    if not isinstance(dim_groups, Mapping):
        raise TypeError(f"dim_groups must be a mapping, got {type(dim_groups).__name__}.")

    if benchmark == "libero":
        if resolved_action_dim != 7:
            raise ValueError(
                "LIBERO intermediate trajectory MSE requires action_dim=7, "
                f"got {resolved_action_dim}."
            )
        if "gripper" not in dim_groups:
            raise ValueError(
                "LIBERO dim_groups must explicitly define binary gripper index [6]."
            )
        gripper = _validate_dims(dim_groups["gripper"], action_dim=resolved_action_dim)
        if gripper != [6]:
            raise ValueError(
                "LIBERO dim_groups['gripper'] must be exactly [6], "
                f"got {gripper}."
            )
        return _validate_dims(list(range(6)), action_dim=resolved_action_dim)

    if benchmark == "robocasa":
        if resolved_action_dim != 29:
            raise ValueError(
                "RoboCasa intermediate trajectory MSE requires action_dim=29, "
                f"got {resolved_action_dim}."
            )
        return _validate_dims(list(range(29)), action_dim=resolved_action_dim)

    raise ValueError(
        f"Unsupported benchmark {benchmark!r}; expected exactly 'libero' or 'robocasa'."
    )


def _validate_trajectory(tensor: object, *, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, H, D], got {tuple(tensor.shape)}.")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be non-empty, got {tuple(tensor.shape)}.")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a real floating dtype, got {tensor.dtype}.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _validate_scales(scales: Sequence[int], *, horizon: int) -> list[int]:
    if isinstance(scales, (str, bytes)) or not isinstance(scales, Sequence):
        raise TypeError("scales must be a sequence of integer scale values.")

    resolved: list[int] = []
    for index, scale in enumerate(scales):
        if isinstance(scale, bool) or not isinstance(scale, Integral):
            raise TypeError(f"scales[{index}] must be an integer, got {type(scale).__name__}.")
        resolved.append(int(scale))

    if not resolved:
        raise ValueError("At least one intermediate scale is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Intermediate scales must be unique, got {resolved}.")
    invalid = [scale for scale in resolved if scale <= 0 or scale >= horizon]
    if invalid:
        raise ValueError(
            "Only intermediate scales strictly below the action horizon are allowed; "
            f"horizon={horizon}, invalid scales={invalid}. The caller must explicitly "
            "exclude the final scale."
        )
    return resolved


def _validate_stats_bucket(
    bucket: object,
    *,
    scale: int,
    require_positive_count: bool,
) -> tuple[float, int]:
    if not isinstance(bucket, Mapping):
        raise TypeError(f"stats[{scale}] must be a mapping.")
    if "sse" not in bucket or "count" not in bucket:
        raise ValueError(f"stats[{scale}] must contain both 'sse' and 'count'.")

    sse_value = bucket["sse"]
    count_value = bucket["count"]
    if isinstance(sse_value, bool) or not isinstance(sse_value, Real):
        raise TypeError(f"stats[{scale}]['sse'] must be a real number.")
    sse = float(sse_value)
    if not math.isfinite(sse) or sse < 0.0:
        raise ValueError(f"stats[{scale}]['sse'] must be finite and non-negative, got {sse}.")

    if isinstance(count_value, bool) or not isinstance(count_value, Integral):
        raise TypeError(f"stats[{scale}]['count'] must be an integer.")
    count = int(count_value)
    minimum = 1 if require_positive_count else 0
    if count < minimum:
        qualifier = "positive" if require_positive_count else "non-negative"
        raise ValueError(f"stats[{scale}]['count'] must be {qualifier}, got {count}.")
    return sse, count


def _validate_existing_stats(
    stats: MutableMapping[int, MutableMapping[str, float | int]],
    *,
    scales: Sequence[int],
) -> None:
    if not isinstance(stats, MutableMapping):
        raise TypeError(f"stats must be a mutable mapping, got {type(stats).__name__}.")
    if not stats:
        return

    for scale in stats:
        if isinstance(scale, bool) or not isinstance(scale, Integral):
            raise TypeError(f"stats scale keys must be integers, got {scale!r}.")
    if set(int(scale) for scale in stats) != set(scales):
        raise ValueError(
            "Every update must use the same intermediate scales; "
            f"stats has {sorted(int(scale) for scale in stats)}, update has {sorted(scales)}."
        )
    for scale in scales:
        _validate_stats_bucket(stats[scale], scale=scale, require_positive_count=False)


def update_intermediate_mse_stats(
    stats: dict[int, dict[str, float | int]],
    *,
    actions: torch.Tensor,
    scale_recons: list[torch.Tensor],
    scales: list[int],
    included_dims: list[int],
) -> None:
    """Accumulate float64 SSE and element counts for one evaluation batch.

    ``scale_recons`` must contain cumulative decoded trajectories for only the
    intermediate scales.  Passing the final scale is an error rather than an
    implicit exclusion so the evaluator's slicing decision remains explicit.
    """

    actions = _validate_trajectory(actions, name="actions")
    if isinstance(scale_recons, (str, bytes)) or not isinstance(scale_recons, Sequence):
        raise TypeError("scale_recons must be a sequence of torch.Tensor values.")
    if len(scale_recons) != len(scales):
        raise ValueError(
            f"scale_recons and scales must have equal lengths, got "
            f"{len(scale_recons)} and {len(scales)}."
        )

    resolved_scales = _validate_scales(scales, horizon=int(actions.shape[1]))
    dims = _validate_dims(included_dims, action_dim=int(actions.shape[2]))
    _validate_existing_stats(stats, scales=resolved_scales)

    validated_recons: list[torch.Tensor] = []
    for index, recon in enumerate(scale_recons):
        recon = _validate_trajectory(recon, name=f"scale_recons[{index}]")
        if recon.shape != actions.shape:
            raise ValueError(
                f"scale_recons[{index}] shape {tuple(recon.shape)} does not match "
                f"actions shape {tuple(actions.shape)}."
            )
        if recon.device != actions.device:
            raise ValueError(
                f"scale_recons[{index}] is on {recon.device}, but actions is on "
                f"{actions.device}."
            )
        validated_recons.append(recon)

    # Compute every contribution before mutating stats. A bad later scale must
    # not leave a partially updated aggregate behind.
    contributions: list[tuple[int, float, int]] = []
    for scale, scale_recon in zip(resolved_scales, validated_recons, strict=True):
        target = build_temporal_scale_target(actions, scale)
        diff = scale_recon[..., dims].float() - target[..., dims].float()
        sse = float(diff.square().sum(dtype=torch.float64).item())
        if not math.isfinite(sse) or sse < 0.0:
            raise ValueError(f"Non-finite intermediate SSE at scale {scale}: {sse}.")
        contributions.append((scale, sse, int(diff.numel())))

    updates: list[tuple[int, float, int]] = []
    for scale, sse, count in contributions:
        bucket = stats.get(scale, {"sse": 0.0, "count": 0})
        previous_sse, previous_count = _validate_stats_bucket(
            bucket, scale=scale, require_positive_count=False
        )
        updated_sse = previous_sse + sse
        updated_count = previous_count + count
        if not math.isfinite(updated_sse):
            raise ValueError(f"Accumulated intermediate SSE overflowed at scale {scale}.")
        updates.append((scale, updated_sse, updated_count))

    for scale, updated_sse, updated_count in updates:
        if scale not in stats:
            stats[scale] = {"sse": 0.0, "count": 0}
        stats[scale]["sse"] = updated_sse
        stats[scale]["count"] = updated_count


def finalize_intermediate_mse_stats(
    stats: dict[int, dict[str, float | int]],
) -> tuple[float, dict[int, float]]:
    """Return the equal-scale mean and the element-wise MSE for each scale."""

    if not isinstance(stats, Mapping):
        raise TypeError(f"stats must be a mapping, got {type(stats).__name__}.")
    if not stats:
        raise ValueError("Cannot finalize empty intermediate MSE statistics.")

    per_scale_mse: dict[int, float] = {}
    normalized_items: list[tuple[int, object]] = []
    for scale, bucket in stats.items():
        if isinstance(scale, bool) or not isinstance(scale, Integral):
            raise TypeError(f"stats scale keys must be positive integers, got {scale!r}.")
        resolved_scale = int(scale)
        if resolved_scale <= 0:
            raise ValueError(f"stats scale keys must be positive, got {resolved_scale}.")
        normalized_items.append((resolved_scale, bucket))

    if len({scale for scale, _ in normalized_items}) != len(normalized_items):
        raise ValueError("stats contains duplicate scale keys after integer normalization.")

    for scale, bucket in sorted(normalized_items):
        sse, count = _validate_stats_bucket(
            bucket, scale=scale, require_positive_count=True
        )
        mse = sse / count
        if not math.isfinite(mse) or mse < 0.0:
            raise ValueError(f"Invalid intermediate MSE at scale {scale}: {mse}.")
        per_scale_mse[scale] = mse

    overall = math.fsum(per_scale_mse.values()) / len(per_scale_mse)
    if not math.isfinite(overall) or overall < 0.0:
        raise ValueError(f"Invalid aggregate intermediate trajectory MSE: {overall}.")
    return overall, per_scale_mse
