"""Intermediate trajectory supervision losses for VAR Stage 1.

The functions in this module are deliberately independent of trainer and model
state. They define the four controlled intermediate-supervision modes while
keeping the existing MTR helper import-compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


INTERMEDIATE_SUPERVISION_MODES = frozenset(
    {
        "none",
        "full_target_time",
        "mint_paper_dct",
        "mtr",
    }
)


@dataclass(frozen=True)
class ResolvedIntermediateSupervision:
    """Validated canonical configuration for intermediate supervision."""

    mode: str
    weight: float
    include_final: bool
    scale_weights: str | dict[int, float]
    group_weights: dict[str, float]
    spectral: dict[str, str]
    resolved_from: str

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.weight > 0.0

    def to_dict(self) -> dict[str, object]:
        scale_weights: str | dict[str, float]
        if isinstance(self.scale_weights, str):
            scale_weights = self.scale_weights
        else:
            scale_weights = {
                str(scale): float(weight)
                for scale, weight in self.scale_weights.items()
            }
        return {
            "mode": self.mode,
            "weight": self.weight,
            "include_final": self.include_final,
            "scale_weights": scale_weights,
            "group_weights": dict(self.group_weights),
            "spectral": dict(self.spectral),
            "resolved_from": self.resolved_from,
        }


def resolve_intermediate_supervision(
    loss_cfg: Mapping[str, object],
    *,
    dim_groups: Mapping[str, Sequence[int]],
    available_scales: Sequence[int],
) -> ResolvedIntermediateSupervision:
    """Resolve native or legacy Stage 1 intermediate-loss configuration."""

    native_value = loss_cfg.get("intermediate", None)
    if native_value is None:
        weight = float(loss_cfg.get("scale_weight", 0.0))
        _validate_non_negative_finite(weight, name="loss.scale_weight")
        mode = "mtr" if weight > 0.0 else "none"
        if mode != "none" and not available_scales:
            raise ValueError(
                "Intermediate supervision requires at least one scale below the horizon."
            )
        if mode == "none":
            scale_weights: str | dict[int, float] = "uniform"
        else:
            scale_weights = _normalize_config_scale_weights(
                loss_cfg.get("scale_loss_weights", "uniform"),
                available_scales=available_scales,
            )
        group_weights = _legacy_intermediate_group_weights(
            loss_cfg,
            dim_groups=dim_groups,
            enabled=mode != "none",
        )
        return ResolvedIntermediateSupervision(
            mode=mode,
            weight=weight,
            include_final=False,
            scale_weights=scale_weights,
            group_weights=group_weights,
            spectral={},
            resolved_from="legacy",
        )

    native = _require_mapping(native_value, name="loss.intermediate")
    mode = str(native.get("mode", "none"))
    if mode not in INTERMEDIATE_SUPERVISION_MODES:
        raise ValueError(
            f"Unknown intermediate supervision mode {mode!r}. "
            f"Expected one of {sorted(INTERMEDIATE_SUPERVISION_MODES)}."
        )
    weight = float(native.get("weight", 0.0))
    _validate_non_negative_finite(weight, name="loss.intermediate.weight")
    if (mode == "none") != (weight == 0.0):
        raise ValueError(
            "loss.intermediate requires mode='none' exactly when weight=0; "
            f"got mode={mode!r}, weight={weight}."
        )
    include_final = bool(native.get("include_final", False))
    if include_final:
        raise ValueError(
            "loss.intermediate.include_final must be false because final "
            "reconstruction is already supervised by recon_loss."
        )
    if mode != "none" and not available_scales:
        raise ValueError("Intermediate supervision requires at least one scale below the horizon.")

    scale_weights = _normalize_config_scale_weights(
        native.get("scale_weights", "uniform"),
        available_scales=available_scales,
    )
    group_weights = _normalize_config_group_weights(
        native.get("group_weights", {}),
        dim_groups=dim_groups,
    )
    spectral = _normalize_spectral_config(native.get("spectral", None), mode=mode)

    time_weighting = loss_cfg.get("time_weighting", None)
    if mode == "mint_paper_dct" and isinstance(time_weighting, Mapping):
        if bool(time_weighting.get("enabled", False)):
            raise ValueError(
                "mint_paper_dct cannot be combined with time_weighting because "
                "frequency bins are not time steps."
            )

    _validate_native_legacy_compatibility(
        loss_cfg,
        mode=mode,
        weight=weight,
        scale_weights=scale_weights,
        group_weights=group_weights,
        dim_groups=dim_groups,
        available_scales=available_scales,
    )
    return ResolvedIntermediateSupervision(
        mode=mode,
        weight=weight,
        include_final=include_final,
        scale_weights=scale_weights,
        group_weights=group_weights,
        spectral=spectral,
        resolved_from="native",
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _validate_non_negative_finite(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")


def _normalize_config_scale_weights(
    value: object,
    *,
    available_scales: Sequence[int],
) -> str | dict[int, float]:
    if value is None or value == "uniform":
        return "uniform"
    mapping = _require_mapping(value, name="scale_weights")
    normalized = {int(scale): float(weight) for scale, weight in mapping.items()}
    unknown = sorted(set(normalized) - {int(scale) for scale in available_scales})
    if unknown:
        raise ValueError(f"scale_weights contains unknown intermediate scales: {unknown}.")
    for scale, weight in normalized.items():
        _validate_non_negative_finite(weight, name=f"scale_weights[{scale}]")
    effective = [
        normalized.get(int(scale), 1.0)
        for scale in available_scales
    ]
    if effective and sum(effective) <= 0.0:
        raise ValueError("At least one intermediate scale weight must be positive.")
    return normalized or "uniform"


def _normalize_config_group_weights(
    value: object,
    *,
    dim_groups: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    mapping = _require_mapping(value, name="group_weights")
    normalized = {str(group): float(weight) for group, weight in mapping.items()}
    unknown = sorted(set(normalized) - set(dim_groups))
    if unknown:
        raise ValueError(
            f"group_weights contains unknown action groups {unknown}; "
            f"available groups are {sorted(dim_groups)}."
        )
    for group, weight in normalized.items():
        _validate_non_negative_finite(weight, name=f"group_weights[{group!r}]")
    return normalized


def _legacy_scale_group_weights(loss_cfg: Mapping[str, object]) -> dict[str, float]:
    suffix = "_scale_weight"
    return {
        str(key)[: -len(suffix)]: float(value)
        for key, value in loss_cfg.items()
        if str(key).endswith(suffix)
    }


def _legacy_intermediate_group_weights(
    loss_cfg: Mapping[str, object],
    *,
    dim_groups: Mapping[str, Sequence[int]],
    enabled: bool,
) -> dict[str, float]:
    if not enabled:
        return {}
    explicit = _legacy_scale_group_weights(loss_cfg)
    result: dict[str, float] = {}
    for group in dim_groups:
        if group in explicit:
            value = explicit[group]
        elif group == "gripper":
            value = 0.0
        else:
            continue
        _validate_non_negative_finite(value, name=f"loss.{group}_scale_weight")
        result[str(group)] = value
    return result


def _normalize_spectral_config(value: object, *, mode: str) -> dict[str, str]:
    if mode != "mint_paper_dct":
        if value is not None and dict(_require_mapping(value, name="spectral")):
            raise ValueError("loss.intermediate.spectral is only valid for mint_paper_dct.")
        return {}
    mapping = {} if value is None else dict(_require_mapping(value, name="spectral"))
    formulation = str(mapping.get("formulation", "raw_dct_ii_mse"))
    normalization = str(mapping.get("normalization", "2_over_h"))
    if formulation != "raw_dct_ii_mse" or normalization != "2_over_h":
        raise ValueError(
            "mint_paper_dct requires spectral.formulation='raw_dct_ii_mse' "
            "and spectral.normalization='2_over_h'."
        )
    return {
        "formulation": formulation,
        "normalization": normalization,
    }


def _effective_scale_weights(
    value: str | dict[int, float],
    scales: Sequence[int],
) -> list[float]:
    if isinstance(value, str):
        return [1.0 for _ in scales]
    return [float(value.get(int(scale), 1.0)) for scale in scales]


def _validate_native_legacy_compatibility(
    loss_cfg: Mapping[str, object],
    *,
    mode: str,
    weight: float,
    scale_weights: str | dict[int, float],
    group_weights: Mapping[str, float],
    dim_groups: Mapping[str, Sequence[int]],
    available_scales: Sequence[int],
) -> None:
    if "scale_weight" in loss_cfg:
        legacy_weight = float(loss_cfg["scale_weight"])
        legacy_mode = "mtr" if legacy_weight > 0.0 else "none"
        if not math.isclose(legacy_weight, weight) or legacy_mode != mode:
            raise ValueError(
                "Conflicting native and legacy intermediate supervision: "
                f"native mode/weight={mode}/{weight}, "
                f"legacy mode/scale_weight={legacy_mode}/{legacy_weight}."
            )
    if "scale_loss_weights" in loss_cfg:
        legacy_scale_weights = _normalize_config_scale_weights(
            loss_cfg["scale_loss_weights"],
            available_scales=available_scales,
        )
        if _effective_scale_weights(legacy_scale_weights, available_scales) != (
            _effective_scale_weights(scale_weights, available_scales)
        ):
            raise ValueError("Conflicting native scale_weights and legacy scale_loss_weights.")

    legacy_explicit = _legacy_scale_group_weights(loss_cfg)
    legacy_semantics_active = bool(legacy_explicit) or (
        "scale_weight" in loss_cfg and float(loss_cfg["scale_weight"]) > 0.0
    )
    if legacy_semantics_active and mode != "none":
        for group in dim_groups:
            legacy_value = legacy_explicit.get(
                group,
                0.0 if group == "gripper" else 1.0,
            )
            native_value = float(group_weights.get(group, 1.0))
            if not math.isclose(legacy_value, native_value):
                raise ValueError(
                    "Conflicting native and legacy group weight for "
                    f"{group!r}: native={native_value}, legacy={legacy_value}."
                )

_DCT_BASIS_CACHE: dict[tuple[int, str, int | None, str], torch.Tensor] = {}


def build_temporal_scale_target(actions: torch.Tensor, scale: int) -> torch.Tensor:
    """Build a scale-matched Down-Up target in normalized action space."""

    _validate_trajectory(actions, name="actions")
    horizon = actions.shape[1]
    scale = int(scale)
    if scale <= 0 or scale > horizon:
        raise ValueError(f"Scale must be within [1, {horizon}], got {scale}.")

    action_channels = actions.transpose(1, 2)
    coarse = F.adaptive_avg_pool1d(action_channels, output_size=scale)
    target = F.interpolate(
        coarse,
        size=horizon,
        mode="linear",
        align_corners=False,
    )
    return target.transpose(1, 2).contiguous()


def dct_ii(actions: torch.Tensor, *, normalization: str = "none") -> torch.Tensor:
    """Apply DCT-II along time for an action tensor shaped [B, H, D].

    The raw transform is the paper-form transform used by the MINT-inspired
    loss. The orthonormal variant is exposed to lock the Parseval boundary in
    tests. Spectral arithmetic is always FP32.
    """

    _validate_trajectory(actions, name="actions")
    if normalization not in {"none", "ortho"}:
        raise ValueError(
            "DCT normalization must be 'none' or 'ortho', "
            f"got {normalization!r}."
        )
    basis = _dct_basis(
        actions.shape[1],
        device=actions.device,
        normalization=normalization,
    )
    return torch.einsum("kh,bhd->bkd", basis, actions.float())


def weighted_dim_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dim_groups: Mapping[str, Sequence[int]],
    gripper_weight: float = 1.0,
    group_weights: Mapping[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    weight_normalization: str = "mean",
) -> torch.Tensor:
    """MSE with optional action-group, sample, and time-axis weights.

    gripper_weight remains as a compatibility shortcut for existing MTR
    configs. New configs should use group_weights.
    """

    _validate_matching_trajectories(pred, target)
    _validate_dim_groups(dim_groups, action_dim=pred.shape[-1])
    resolved_group_weights = {
        str(group): float(weight)
        for group, weight in (group_weights or {}).items()
    }
    if gripper_weight != 1.0 and "gripper" in dim_groups:
        resolved_group_weights.setdefault("gripper", float(gripper_weight))

    if (
        not resolved_group_weights
        and sample_weights is None
        and time_weights is None
    ):
        return F.mse_loss(pred, target)

    if weight_normalization not in {"mean", "none"}:
        raise ValueError(
            "weight_normalization must be 'mean' or 'none', "
            f"got {weight_normalization!r}."
        )

    dim_weights = torch.ones(
        pred.shape[-1],
        dtype=pred.dtype,
        device=pred.device,
    )
    for group_name, group_weight in resolved_group_weights.items():
        if group_name not in dim_groups:
            raise ValueError(
                f"Unknown action dim group {group_name!r}. "
                f"Available groups: {sorted(dim_groups)}"
            )
        if not math.isfinite(group_weight) or group_weight < 0.0:
            raise ValueError(
                "Action group weight must be finite and non-negative, "
                f"got group={group_name!r}, weight={group_weight}."
            )
        indices = [int(index) for index in dim_groups[group_name]]
        if any(index < 0 or index >= pred.shape[-1] for index in indices):
            raise ValueError(
                f"Action dim group {group_name!r} contains an index outside "
                f"[0, {pred.shape[-1]}): {indices}."
            )
        dim_weights[indices] = group_weight

    weight = dim_weights.view(1, 1, -1)
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != pred.shape[0]:
            raise ValueError(
                f"Expected sample_weights with shape [{pred.shape[0]}], "
                f"got {tuple(sample_weights.shape)}."
            )
        weight = weight * sample_weights.to(
            device=pred.device,
            dtype=pred.dtype,
        ).view(-1, 1, 1)
    if time_weights is not None:
        if time_weights.ndim != 1 or time_weights.shape[0] != pred.shape[1]:
            raise ValueError(
                f"Expected time_weights with shape [{pred.shape[1]}], "
                f"got {tuple(time_weights.shape)}."
            )
        weight = weight * time_weights.to(
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, -1, 1)

    error = (pred - target).pow(2)
    if weight_normalization == "none":
        denominator = torch.as_tensor(
            error.numel(),
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        denominator = weight.expand_as(error).sum()
    return (error * weight).sum() / denominator.clamp_min(1e-6)


def normalized_raw_dct_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dim_groups: Mapping[str, Sequence[int]],
    gripper_weight: float = 1.0,
    group_weights: Mapping[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    weight_normalization: str = "mean",
) -> torch.Tensor:
    """Compute paper-form raw DCT-II MSE normalized by 2 / H."""

    _validate_matching_trajectories(pred, target)
    if time_weights is not None:
        raise ValueError(
            "time_weights cannot be applied to mint_paper_dct because its "
            "second axis contains frequency bins, not time steps."
        )
    horizon = pred.shape[1]
    spectral_mse = weighted_dim_mse(
        dct_ii(pred, normalization="none"),
        dct_ii(target, normalization="none"),
        dim_groups=dim_groups,
        gripper_weight=gripper_weight,
        group_weights=group_weights,
        sample_weights=sample_weights,
        weight_normalization=weight_normalization,
    )
    return (2.0 / float(horizon)) * spectral_mse


def compute_intermediate_loss(
    actions: torch.Tensor,
    scale_recons: Sequence[torch.Tensor],
    scales: Sequence[int],
    *,
    mode: str,
    dim_groups: Mapping[str, Sequence[int]],
    gripper_weight: float = 1.0,
    group_weights: Mapping[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    weight_normalization: str = "mean",
    scale_weights: Mapping[int, float] | str | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Compute one of the four Stage 1 intermediate-supervision modes.

    Only scales below the final horizon are accepted. In none mode callers
    must pass empty sequences, making accidental Base decoder work visible.
    """

    _validate_trajectory(actions, name="actions")
    if mode not in INTERMEDIATE_SUPERVISION_MODES:
        raise ValueError(
            f"Unknown intermediate supervision mode {mode!r}. "
            f"Expected one of {sorted(INTERMEDIATE_SUPERVISION_MODES)}."
        )
    if len(scale_recons) != len(scales):
        raise ValueError(
            f"Expected one reconstruction per scale, got {len(scale_recons)} "
            f"reconstructions for scales={list(scales)}."
        )

    if mode == "none":
        if len(scale_recons) > 0 or len(scales) > 0:
            raise ValueError(
                "mode='none' expects no scale reconstructions; the Base method "
                "must not request intermediate decoder outputs."
            )
        return torch.zeros((), device=actions.device, dtype=actions.dtype), {}

    resolved_scales = [int(scale) for scale in scales]
    if resolved_scales != sorted(set(resolved_scales)):
        raise ValueError(
            "Intermediate scales must be unique and strictly increasing, "
            f"got {resolved_scales}."
        )
    horizon = actions.shape[1]
    invalid_scales = [
        scale
        for scale in resolved_scales
        if scale <= 0 or scale >= horizon
    ]
    if invalid_scales:
        raise ValueError(
            f"Intermediate scales must be within [1, {horizon}) and exclude "
            f"the final horizon, got {invalid_scales}."
        )
    if mode == "mint_paper_dct" and time_weights is not None:
        raise ValueError(
            "time_weights are invalid for mint_paper_dct because frequency "
            "bins must not be treated as time steps."
        )

    per_scale_losses: dict[int, torch.Tensor] = {}
    for scale, scale_recon in zip(
        resolved_scales,
        scale_recons,
        strict=True,
    ):
        _validate_matching_trajectories(scale_recon, actions)
        if mode == "full_target_time":
            current_loss = weighted_dim_mse(
                scale_recon,
                actions,
                dim_groups=dim_groups,
                gripper_weight=gripper_weight,
                group_weights=group_weights,
                sample_weights=sample_weights,
                time_weights=time_weights,
                weight_normalization=weight_normalization,
            )
        elif mode == "mint_paper_dct":
            current_loss = normalized_raw_dct_mse(
                scale_recon,
                actions,
                dim_groups=dim_groups,
                gripper_weight=gripper_weight,
                group_weights=group_weights,
                sample_weights=sample_weights,
                weight_normalization=weight_normalization,
            )
        else:
            current_loss = weighted_dim_mse(
                scale_recon,
                build_temporal_scale_target(actions, scale),
                dim_groups=dim_groups,
                gripper_weight=gripper_weight,
                group_weights=group_weights,
                sample_weights=sample_weights,
                time_weights=time_weights,
                weight_normalization=weight_normalization,
            )
        per_scale_losses[scale] = current_loss

    if not per_scale_losses:
        return torch.zeros((), device=actions.device, dtype=actions.dtype), {}

    resolved_weights = _resolve_scale_weights(
        scale_weights,
        available_scales=resolved_scales,
    )
    losses = torch.stack(list(per_scale_losses.values()))
    weight_tensor = losses.new_tensor(
        [resolved_weights[scale] for scale in resolved_scales]
    )
    return (losses * weight_tensor).sum() / weight_tensor.sum(), per_scale_losses


def compute_mtr_loss(
    actions: torch.Tensor,
    scale_recons: Sequence[torch.Tensor],
    scales: Sequence[int],
    *,
    dim_groups: Mapping[str, Sequence[int]],
    gripper_weight: float,
    group_weights: Mapping[str, float] | None = None,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    weight_normalization: str = "mean",
    scale_loss_weights: Mapping[int, float] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Backward-compatible wrapper for the original MTR helper."""

    return compute_intermediate_loss(
        actions,
        scale_recons,
        scales,
        mode="mtr",
        dim_groups=dim_groups,
        gripper_weight=gripper_weight,
        group_weights=group_weights,
        sample_weights=sample_weights,
        time_weights=time_weights,
        weight_normalization=weight_normalization,
        scale_weights=scale_loss_weights,
    )


def _dct_basis(
    horizon: int,
    *,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    device = torch.device(device)
    cache_key = (horizon, device.type, device.index, normalization)
    cached = _DCT_BASIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    time = torch.arange(
        horizon,
        device=device,
        dtype=torch.float64,
    ) + 0.5
    frequency = torch.arange(
        horizon,
        device=device,
        dtype=torch.float64,
    ).unsqueeze(1)
    basis = torch.cos(
        (math.pi / float(horizon)) * frequency * time.unsqueeze(0)
    )
    if normalization == "ortho":
        basis[0] *= math.sqrt(1.0 / float(horizon))
        if horizon > 1:
            basis[1:] *= math.sqrt(2.0 / float(horizon))
    basis = basis.float()
    _DCT_BASIS_CACHE[cache_key] = basis
    return basis


def _resolve_scale_weights(
    scale_weights: Mapping[int, float] | str | None,
    *,
    available_scales: Sequence[int],
) -> dict[int, float]:
    if scale_weights is None or scale_weights == "uniform":
        configured: dict[int, float] = {}
    elif isinstance(scale_weights, Mapping):
        configured = {
            int(scale): float(weight)
            for scale, weight in scale_weights.items()
        }
    else:
        raise ValueError(
            "scale_weights must be a mapping, 'uniform', or None, "
            f"got {scale_weights!r}."
        )

    unknown_scales = sorted(set(configured) - set(available_scales))
    if unknown_scales:
        raise ValueError(
            "scale_weights contains unknown intermediate scales: "
            f"{unknown_scales}."
        )

    resolved: dict[int, float] = {}
    for scale in available_scales:
        weight = configured.get(scale, 1.0)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "Intermediate scale weight must be finite and non-negative, "
                f"got scale={scale}, weight={weight}."
            )
        resolved[scale] = weight
    if sum(resolved.values()) <= 0.0:
        raise ValueError(
            "At least one intermediate scale weight must be positive."
        )
    return resolved


def _validate_trajectory(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(value).__name__}."
        )
    if value.ndim != 3:
        raise ValueError(
            f"Expected {name} with shape [B, H, D], "
            f"got {tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise TypeError(
            f"{name} must be floating point, got dtype={value.dtype}."
        )


def _validate_dim_groups(
    dim_groups: Mapping[str, Sequence[int]],
    *,
    action_dim: int,
) -> None:
    seen_indices: dict[int, str] = {}
    for group_name, group_indices in dim_groups.items():
        indices = [int(index) for index in group_indices]
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Action dim group {group_name!r} contains duplicate "
                f"indices: {indices}."
            )
        for index in indices:
            if index < 0 or index >= action_dim:
                raise ValueError(
                    f"Action dim group {group_name!r} contains index {index} "
                    f"outside [0, {action_dim})."
                )
            previous_group = seen_indices.get(index)
            if previous_group is not None:
                raise ValueError(
                    f"Action dim index {index} appears in both "
                    f"{previous_group!r} and {group_name!r}; overlapping "
                    "groups are not supported by intermediate loss masks."
                )
            seen_indices[index] = str(group_name)


def _validate_matching_trajectories(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _validate_trajectory(pred, name="pred")
    _validate_trajectory(target, name="target")
    if pred.shape != target.shape:
        raise ValueError(
            "Expected pred and target to have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}."
        )
