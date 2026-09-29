"""Permutation-invariant neural calibration for FuXi precipitation ensembles.

The public model consumes physical weekly precipitation members with shape
``[batch, member, lead, latitude, longitude]``.  It predicts a location shift
and a positive spread multiplier in ``log1p`` rainfall space, then applies the
same exchangeable transformation to every member::

    u_m = log1p(x_m)
    mu = mean_m(u_m)
    u'_m = clamp_min(mu + delta_mu + s * (u_m - mu), 0)

where ``s = exp(log_s)``.  Both output heads are zero initialized, so a newly
constructed model has ``delta_mu == 0`` and ``s == 1`` and returns the input
members exactly.  The shared member encoder and symmetric mean pooling make
the learned parameters invariant to member order; applying those parameters
to each member makes the corrected ensemble permutation equivariant.

This module intentionally contains only reusable model, loss, and metric
primitives.  Data splitting, normalization, fitting, and artifact publication
belong to the experiment driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


CalibrationMode = Literal["location_spread", "location_only", "summary_only"]


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest GroupNorm group count that divides ``channels``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_members(members: torch.Tensor) -> None:
    if members.ndim != 5:
        raise ValueError(
            "members must have shape [batch, member, lead, height, width], "
            f"got {tuple(members.shape)}"
        )
    if not members.is_floating_point():
        raise TypeError("members must be a floating-point tensor")
    if min(members.shape) < 1:
        raise ValueError("all member tensor dimensions must be non-empty")
    if not bool(torch.isfinite(members).all()):
        raise ValueError("members must be finite before model evaluation")
    if bool(torch.any(members < 0.0)):
        raise ValueError("physical precipitation members must be nonnegative")


def _validate_parameter_field(
    field: torch.Tensor,
    members: torch.Tensor,
    name: str,
) -> None:
    expected = (members.shape[0], members.shape[2], *members.shape[-2:])
    if field.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(field.shape)}")
    if field.device != members.device:
        raise ValueError(f"{name} and members must be on the same device")
    if field.dtype != members.dtype:
        raise ValueError(f"{name} and members must have the same dtype")
    if not bool(torch.isfinite(field).all()):
        raise ValueError(f"{name} must be finite")


def random_member_indices(
    member_count: int,
    sample_size: int | None,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select member indices without replacement.

    ``None`` or a sample size equal to ``member_count`` selects every member in
    archive order without consuming random numbers.  A sampled index vector is
    sorted so checkpointed/debug output has a stable order even though the set
    itself was selected randomly.
    """

    if member_count < 1:
        raise ValueError("member_count must be positive")
    if sample_size is None:
        sample_size = member_count
    if not 1 <= sample_size <= member_count:
        raise ValueError(
            f"sample_size must be in [1, {member_count}], got {sample_size}"
        )
    target_device = torch.device("cpu" if device is None else device)
    if sample_size == member_count:
        return torch.arange(member_count, device=target_device)
    return torch.sort(
        torch.randperm(
            member_count,
            device=target_device,
            generator=generator,
        )[:sample_size]
    ).values


def subsample_ensemble_members(
    members: torch.Tensor,
    sample_size: int | None,
    *,
    generator: torch.Generator | None = None,
    member_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a common member subset for every case in a mini-batch.

    Keeping one index vector for the complete batch preserves each selected
    member's full six-week spatial trajectory.  The operation must therefore
    happen on the member axis, never independently per grid cell.
    """

    _validate_members(members)
    total = members.shape[1]
    if member_indices is not None and sample_size is not None:
        raise ValueError("pass either member_indices or sample_size, not both")
    if member_indices is None:
        indices = random_member_indices(
            total,
            sample_size,
            device=members.device,
            generator=generator,
        )
    else:
        indices = torch.as_tensor(member_indices, device=members.device)
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("member_indices must be a non-empty one-dimensional tensor")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("member_indices must contain integers")
        indices = indices.to(dtype=torch.long)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= total)):
            raise ValueError("member_indices contain an out-of-range member")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("member_indices must not contain duplicates")
    return torch.index_select(members, dim=1, index=indices), indices


def apply_location_spread_correction(
    members: torch.Tensor,
    delta_log_location: torch.Tensor,
    *,
    log_spread: torch.Tensor | None = None,
    spread_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply exchangeable log-space location/spread parameters to members.

    Supply exactly one of ``log_spread`` or ``spread_factor``.  Omitting both
    applies a location-only correction (spread factor one).  The physical
    residual formulation below is deliberate: when ``delta_log_location`` and
    ``log_spread`` are exactly zero, the two ``expm1`` terms are bit-identical,
    their difference is exactly zero, and the raw physical members are
    returned exactly rather than merely within a floating-point tolerance.
    """

    _validate_members(members)
    _validate_parameter_field(delta_log_location, members, "delta_log_location")
    if log_spread is not None and spread_factor is not None:
        raise ValueError("pass either log_spread or spread_factor, not both")
    if log_spread is not None:
        _validate_parameter_field(log_spread, members, "log_spread")
        spread = torch.exp(log_spread)
    elif spread_factor is not None:
        _validate_parameter_field(spread_factor, members, "spread_factor")
        if bool(torch.any(spread_factor <= 0.0)):
            raise ValueError("spread_factor must be strictly positive")
        spread = spread_factor
    else:
        spread = torch.ones_like(delta_log_location)

    log_members = torch.log1p(members)
    # Sorting makes the floating-point reduction canonical as well as
    # mathematically symmetric under member permutations.
    canonical = torch.sort(log_members, dim=1).values
    log_location = canonical.mean(dim=1)
    deviations = log_members - log_location[:, None]
    # This residual form is algebraically identical to
    # ``mu + delta + spread * (u - mu)`` but preserves the exact identity at
    # delta=0 and spread=1: no subtraction/readdition of ``mu`` touches ``u``.
    corrected_log = log_members + delta_log_location[:, None]
    corrected_log = corrected_log + (spread[:, None] - 1.0) * deviations
    corrected_log = torch.clamp_min(corrected_log, 0.0)

    physical_from_log = torch.expm1(log_members)
    corrected_from_log = torch.expm1(corrected_log)
    corrected = members + (corrected_from_log - physical_from_log)
    corrected = torch.clamp_min(corrected, 0.0)
    if not bool(torch.isfinite(corrected).all()):
        raise FloatingPointError("location/spread correction produced non-finite rainfall")
    return corrected


class _PointwiseMemberEncoder(nn.Module):
    """Encode each member/grid/lead point with one shared MLP."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(
        self,
        canonical_log_members: torch.Tensor,
        log_location: torch.Tensor,
    ) -> torch.Tensor:
        batch, member_count, leads, height, width = canonical_log_members.shape
        deviations = canonical_log_members - log_location[:, None]
        features = torch.stack((canonical_log_members, deviations), dim=2)
        encoded = self.network(
            features.reshape(batch * member_count, 2, leads, height, width)
        )
        return encoded.reshape(
            batch, member_count, -1, leads, height, width
        ).mean(dim=1)


class _CompactLeadSpatialBackbone(nn.Module):
    """A compact residual Conv3D backbone over lead, latitude, and longitude."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = _group_count(hidden_channels)
        self.input = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_norm = nn.GroupNorm(groups, hidden_channels)
        self.block = nn.Sequential(
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
        )
        self.activation = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input_norm(self.input(features)))
        return self.activation(hidden + self.block(hidden))


@dataclass(frozen=True)
class EnsembleCalibrationParameters:
    """Predicted fields needed to reconstruct a corrected ensemble."""

    delta_log_location: torch.Tensor
    log_spread: torch.Tensor
    spread_factor: torch.Tensor
    selected_member_indices: torch.Tensor


@dataclass(frozen=True)
class EnsembleCalibrationOutput:
    """Corrected members and their explicit calibration parameters."""

    corrected_members: torch.Tensor
    delta_log_location: torch.Tensor
    log_spread: torch.Tensor
    spread_factor: torch.Tensor
    selected_member_indices: torch.Tensor


class EnsembleLocationSpreadCalibrator(nn.Module):
    """Permutation-invariant neural location/spread ensemble calibrator.

    Modes
    -----
    ``location_spread``
        Learned member encoder plus explicit log-location/log-spread summaries
        and context; predicts both location and spread corrections.
    ``location_only``
        Same learned set representation, but spread is fixed to exactly one.
    ``summary_only``
        Removes the learned member encoder.  Only explicit ensemble
        log-location/RMS log-spread summaries and context reach the backbone;
        both correction parameters are predicted.
    """

    MODES = ("location_spread", "location_only", "summary_only")

    def __init__(
        self,
        context_channels: int,
        member_hidden_channels: int = 8,
        backbone_channels: int = 24,
        mode: CalibrationMode = "location_spread",
        dropout: float = 0.0,
        max_abs_log_spread: float = 2.0,
    ) -> None:
        super().__init__()
        if context_channels < 0:
            raise ValueError("context_channels must be nonnegative")
        if member_hidden_channels < 1:
            raise ValueError("member_hidden_channels must be positive")
        if backbone_channels < 1:
            raise ValueError("backbone_channels must be positive")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if max_abs_log_spread <= 0.0:
            raise ValueError("max_abs_log_spread must be positive")

        self.context_channels = int(context_channels)
        self.member_hidden_channels = int(member_hidden_channels)
        self.backbone_channels = int(backbone_channels)
        self.mode: CalibrationMode = mode
        self.max_abs_log_spread = float(max_abs_log_spread)

        if mode == "summary_only":
            self.member_encoder: _PointwiseMemberEncoder | None = None
            set_channels = 0
        else:
            self.member_encoder = _PointwiseMemberEncoder(member_hidden_channels)
            set_channels = member_hidden_channels
        # Explicit summaries are log-location and RMS member deviation in log
        # space.  The context is assumed to have been normalized by the driver.
        backbone_inputs = set_channels + 2 + context_channels
        self.backbone = _CompactLeadSpatialBackbone(
            backbone_inputs,
            backbone_channels,
            dropout,
        )
        output_channels = 1 if mode == "location_only" else 2
        self.parameter_head = nn.Conv3d(
            backbone_channels,
            output_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)

    def _validate_context(
        self,
        members: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        expected = (
            members.shape[0],
            members.shape[2],
            self.context_channels,
            *members.shape[-2:],
        )
        if self.context_channels == 0:
            if context is not None and context.shape != expected:
                raise ValueError(
                    f"zero-channel context must have shape {expected} when supplied"
                )
            return context
        if context is None:
            raise ValueError(
                f"context with {self.context_channels} channels is required"
            )
        if context.shape != expected:
            raise ValueError(f"context must have shape {expected}, got {tuple(context.shape)}")
        if context.device != members.device:
            raise ValueError("context and members must be on the same device")
        if context.dtype != members.dtype:
            raise ValueError("context and members must have the same dtype")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context must be finite after preprocessing")
        return context

    def _predict_selected(
        self,
        selected: torch.Tensor,
        context: torch.Tensor | None,
        indices: torch.Tensor,
    ) -> EnsembleCalibrationParameters:
        log_members = torch.log1p(selected)
        canonical = torch.sort(log_members, dim=1).values
        log_location = canonical.mean(dim=1)
        deviations = canonical - log_location[:, None]
        rms_spread = torch.sqrt(torch.mean(deviations.square(), dim=1))

        features = [log_location[:, None], rms_spread[:, None]]
        if self.member_encoder is not None:
            features.insert(0, self.member_encoder(canonical, log_location))
        if context is not None and self.context_channels:
            features.append(context.permute(0, 2, 1, 3, 4))
        combined = torch.cat(features, dim=1)
        raw = self.parameter_head(self.backbone(combined)).to(dtype=selected.dtype)
        delta = raw[:, 0]
        if self.mode == "location_only":
            log_spread = torch.zeros_like(delta)
        else:
            log_spread = self.max_abs_log_spread * torch.tanh(raw[:, 1])
        return EnsembleCalibrationParameters(
            delta_log_location=delta,
            log_spread=log_spread,
            spread_factor=torch.exp(log_spread),
            selected_member_indices=indices,
        )

    def predict_parameters(
        self,
        members: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        member_indices: torch.Tensor | None = None,
        member_subsample_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EnsembleCalibrationParameters:
        """Predict parameter fields, optionally from a random member subset."""

        _validate_members(members)
        context = self._validate_context(members, context)
        selected, indices = subsample_ensemble_members(
            members,
            member_subsample_size,
            generator=generator,
            member_indices=member_indices,
        )
        return self._predict_selected(selected, context, indices)

    def forward(
        self,
        members: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        member_indices: torch.Tensor | None = None,
        member_subsample_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EnsembleCalibrationOutput:
        """Return corrected physical members and explicit parameter fields."""

        _validate_members(members)
        context = self._validate_context(members, context)
        selected, indices = subsample_ensemble_members(
            members,
            member_subsample_size,
            generator=generator,
            member_indices=member_indices,
        )
        parameters = self._predict_selected(selected, context, indices)
        corrected = apply_location_spread_correction(
            selected,
            parameters.delta_log_location,
            log_spread=parameters.log_spread,
        )
        return EnsembleCalibrationOutput(
            corrected_members=corrected,
            delta_log_location=parameters.delta_log_location,
            log_spread=parameters.log_spread,
            spread_factor=parameters.spread_factor,
            selected_member_indices=indices,
        )


def ensemble_crps(
    members: torch.Tensor,
    target: torch.Tensor,
    *,
    member_dim: int = 1,
) -> torch.Tensor:
    """Return empirical ensemble CRPS without forming an ``M x M`` tensor.

    For sorted members ``x_(i)``, the pairwise term is evaluated in
    ``O(M log M)`` time and ``O(M)`` memory using

    ``sum_i (2 i - M + 1) x_(i) / M^2`` (zero-based ``i``).
    """

    if members.ndim < 2:
        raise ValueError("members must have at least two dimensions")
    member_dim = member_dim % members.ndim
    if members.shape[member_dim] < 1:
        raise ValueError("member dimension must be non-empty")
    expected = tuple(
        size for index, size in enumerate(members.shape) if index != member_dim
    )
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target.shape)}")
    if target.device != members.device or target.dtype != members.dtype:
        raise ValueError("target and members must share device and dtype")

    ordered = torch.sort(members, dim=member_dim).values
    member_count = members.shape[member_dim]
    target_with_member = target.unsqueeze(member_dim)
    reliability = torch.mean(torch.abs(members - target_with_member), dim=member_dim)
    coefficient = (
        2.0 * torch.arange(member_count, dtype=members.dtype, device=members.device)
        - member_count
        + 1.0
    )
    coefficient_shape = [1] * members.ndim
    coefficient_shape[member_dim] = member_count
    pairwise = torch.sum(
        ordered * coefficient.reshape(coefficient_shape), dim=member_dim
    ) / float(member_count * member_count)
    return reliability - pairwise


def _broadcast_weights(
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if weights.device != target.device:
        raise ValueError("weights and target must be on the same device")
    if not weights.is_floating_point():
        weights = weights.to(dtype=target.dtype)
    elif weights.dtype != target.dtype:
        weights = weights.to(dtype=target.dtype)
    try:
        return torch.broadcast_to(weights, target.shape)
    except RuntimeError as error:
        raise ValueError(
            f"weights with shape {tuple(weights.shape)} do not broadcast to "
            f"target shape {tuple(target.shape)}"
        ) from error


def weighted_ensemble_crps(
    members: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    member_dim: int = 1,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Area/mask-weighted CRPS suitable as the probabilistic training loss."""

    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be 'mean' or 'none'")
    member_dim = member_dim % members.ndim
    expected = tuple(
        size for index, size in enumerate(members.shape) if index != member_dim
    )
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target.shape)}")
    broadcast_weights = _broadcast_weights(weights, target)
    valid = (
        torch.isfinite(target)
        & torch.isfinite(broadcast_weights)
        & (broadcast_weights > 0.0)
    )
    if valid_mask is not None:
        try:
            valid = valid & torch.broadcast_to(
                valid_mask.to(device=target.device, dtype=torch.bool), target.shape
            )
        except RuntimeError as error:
            raise ValueError("valid_mask does not broadcast to target shape") from error

    finite_members = torch.isfinite(members).all(dim=member_dim)
    if bool(torch.any(valid & ~finite_members)):
        raise FloatingPointError(
            "ensemble contains non-finite members on positive-weight target support"
        )

    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_members = torch.where(
        valid.unsqueeze(member_dim),
        members,
        torch.zeros_like(members),
    )
    pointwise = ensemble_crps(safe_members, safe_target, member_dim=member_dim)
    if reduction == "none":
        return torch.where(valid, pointwise, torch.full_like(pointwise, torch.nan))
    effective_weights = torch.where(valid, broadcast_weights, torch.zeros_like(broadcast_weights))
    denominator = effective_weights.sum()
    if not bool(denominator > 0.0):
        raise ValueError("weights contain no positive valid support")
    return torch.sum(pointwise * effective_weights) / denominator


def weighted_ensemble_crps_loss(
    members: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Training-oriented alias for member-axis-one weighted CRPS."""

    return weighted_ensemble_crps(
        members,
        target,
        spatial_weights,
        valid_mask=valid_mask,
        member_dim=1,
        reduction="mean",
    )


def _spatial_weight_contract(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must match and include height/width")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("prediction and target must share device and dtype")
    broadcast_weights = _broadcast_weights(weights, target)
    valid = (
        torch.isfinite(target)
        & torch.isfinite(broadcast_weights)
        & (broadcast_weights > 0.0)
    )
    if bool(torch.any(valid & ~torch.isfinite(prediction))):
        raise FloatingPointError(
            "prediction is non-finite on positive-weight target support"
        )
    effective = torch.where(valid, broadcast_weights, torch.zeros_like(broadcast_weights))
    denominator = effective.sum(dim=(-2, -1))
    if bool(torch.any(denominator <= 0.0)):
        raise ValueError("at least one case/lead has no positive valid spatial support")
    return valid, effective, denominator


def weighted_spatial_deterministic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    climatology: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return RMSE, MAE, bias, and anomaly correlation per case/lead.

    Only the final latitude/longitude axes are reduced.  For the standard
    ``[batch, lead, height, width]`` fields, each returned score therefore has
    shape ``[batch, lead]`` and can be aggregated weekwise without losing the
    initialization block structure.
    """

    valid, effective, denominator = _spatial_weight_contract(
        prediction, target, weights
    )
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = safe_prediction - safe_target
    rmse = torch.sqrt(torch.sum(effective * error.square(), dim=(-2, -1)) / denominator)
    mae = torch.sum(effective * torch.abs(error), dim=(-2, -1)) / denominator
    bias = torch.sum(effective * error, dim=(-2, -1)) / denominator

    if climatology is None:
        predicted_anomaly = safe_prediction
        target_anomaly = safe_target
    else:
        if climatology.shape != target.shape:
            raise ValueError("climatology must match target shape")
        if climatology.device != target.device or climatology.dtype != target.dtype:
            raise ValueError("climatology and target must share device and dtype")
        climo_valid = torch.isfinite(climatology)
        if bool(torch.any(valid & ~climo_valid)):
            raise FloatingPointError(
                "climatology is non-finite on positive-weight target support"
            )
        valid = valid & climo_valid
        effective = torch.where(valid, effective, torch.zeros_like(effective))
        denominator = effective.sum(dim=(-2, -1))
        if bool(torch.any(denominator <= 0.0)):
            raise ValueError("climatology leaves a case/lead without valid support")
        safe_climo = torch.where(valid, climatology, torch.zeros_like(climatology))
        predicted_anomaly = (
            torch.where(valid, prediction, torch.zeros_like(prediction))
            - safe_climo
        )
        target_anomaly = torch.where(valid, target, torch.zeros_like(target)) - safe_climo

    prediction_mean = torch.sum(
        effective * predicted_anomaly, dim=(-2, -1)
    ) / denominator
    target_mean = torch.sum(effective * target_anomaly, dim=(-2, -1)) / denominator
    prediction_centered = predicted_anomaly - prediction_mean[..., None, None]
    target_centered = target_anomaly - target_mean[..., None, None]
    covariance = torch.sum(
        effective * prediction_centered * target_centered, dim=(-2, -1)
    )
    prediction_variance = torch.sum(
        effective * prediction_centered.square(), dim=(-2, -1)
    )
    target_variance = torch.sum(
        effective * target_centered.square(), dim=(-2, -1)
    )
    acc_denominator = torch.sqrt(prediction_variance * target_variance)
    acc = torch.where(
        acc_denominator > 0.0,
        covariance / torch.clamp_min(acc_denominator, torch.finfo(target.dtype).tiny),
        torch.full_like(covariance, torch.nan),
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "acc": acc,
        "valid_weight": denominator,
    }


def weighted_spatial_brier_score(
    members: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
    *,
    member_dim: int = 1,
) -> torch.Tensor:
    """Return ensemble-probability Brier score per case/lead."""

    if threshold < 0.0:
        raise ValueError("precipitation threshold must be nonnegative")
    member_dim = member_dim % members.ndim
    expected = tuple(
        size for index, size in enumerate(members.shape) if index != member_dim
    )
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target.shape)}")
    finite_members = torch.isfinite(members).all(dim=member_dim)
    probability = torch.mean((members >= threshold).to(target.dtype), dim=member_dim)
    observation = (target >= threshold).to(target.dtype)
    valid, effective, denominator = _spatial_weight_contract(
        probability, target, weights
    )
    if bool(torch.any(valid & ~finite_members)):
        raise FloatingPointError(
            "ensemble contains non-finite members on Brier-score support"
        )
    squared = torch.where(valid, (probability - observation).square(), 0.0)
    return torch.sum(effective * squared, dim=(-2, -1)) / denominator


def weighted_spatial_interval_metrics(
    members: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    coverage: float = 0.9,
    *,
    member_dim: int = 1,
) -> dict[str, torch.Tensor]:
    """Return central-interval empirical coverage and width per case/lead."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    member_dim = member_dim % members.ndim
    expected = tuple(
        size for index, size in enumerate(members.shape) if index != member_dim
    )
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target.shape)}")
    alpha = (1.0 - coverage) / 2.0
    lower = torch.quantile(members, alpha, dim=member_dim)
    upper = torch.quantile(members, 1.0 - alpha, dim=member_dim)
    valid, effective, denominator = _spatial_weight_contract(lower, target, weights)
    if bool(torch.any(valid & ~torch.isfinite(upper))):
        raise FloatingPointError(
            "ensemble interval is non-finite on positive-weight target support"
        )
    inside = ((target >= lower) & (target <= upper)).to(target.dtype)
    width = upper - lower
    return {
        "coverage": torch.sum(effective * inside, dim=(-2, -1)) / denominator,
        "width": torch.sum(effective * width, dim=(-2, -1)) / denominator,
        "lower": lower,
        "upper": upper,
    }


def ensemble_rank_histogram(
    members: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    member_dim: int = 1,
    randomize_ties: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return an ``M + 1`` bin observation-rank histogram.

    Exact ties (especially dry zero rainfall) are randomized uniformly across
    their admissible ranks by default.  Optional broadcastable weights produce
    weighted floating-point bin totals; otherwise integer counts are returned.
    """

    member_dim = member_dim % members.ndim
    expected = tuple(
        size for index, size in enumerate(members.shape) if index != member_dim
    )
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target.shape)}")
    finite_members = torch.isfinite(members).all(dim=member_dim)
    finite = torch.isfinite(target)
    expanded_target = target.unsqueeze(member_dim)
    below = torch.sum(members < expanded_target, dim=member_dim)
    tied = torch.sum(members == expanded_target, dim=member_dim)
    if randomize_ties:
        uniform = torch.rand(
            target.shape,
            dtype=members.dtype,
            device=members.device,
            generator=generator,
        )
        tie_offset = torch.floor(uniform * (tied + 1).to(uniform.dtype)).to(torch.long)
    else:
        tie_offset = torch.div(tied, 2, rounding_mode="floor")
    ranks = (below + tie_offset).to(torch.long)
    bin_count = members.shape[member_dim] + 1
    valid_ranks = ranks[finite]
    if weights is None:
        if bool(torch.any(finite & ~finite_members)):
            raise FloatingPointError(
                "ensemble contains non-finite members where rank target is finite"
            )
        finite = finite & finite_members
        return torch.bincount(valid_ranks, minlength=bin_count)
    broadcast_weights = _broadcast_weights(weights, target)
    finite = finite & torch.isfinite(broadcast_weights) & (broadcast_weights > 0.0)
    if bool(torch.any(finite & ~finite_members)):
        raise FloatingPointError(
            "ensemble contains non-finite members on weighted rank support"
        )
    histogram = torch.zeros(bin_count, dtype=target.dtype, device=target.device)
    return histogram.scatter_add(0, ranks[finite], broadcast_weights[finite])


__all__ = [
    "CalibrationMode",
    "EnsembleCalibrationOutput",
    "EnsembleCalibrationParameters",
    "EnsembleLocationSpreadCalibrator",
    "apply_location_spread_correction",
    "ensemble_crps",
    "ensemble_rank_histogram",
    "random_member_indices",
    "subsample_ensemble_members",
    "weighted_ensemble_crps",
    "weighted_ensemble_crps_loss",
    "weighted_spatial_brier_score",
    "weighted_spatial_deterministic_metrics",
    "weighted_spatial_interval_metrics",
]
