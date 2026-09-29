#!/usr/bin/env python3
"""All-season deterministic--probabilistic hybrid-loss ablation.

This V2 experiment changes only the training loss of the frozen 51-member
``location_spread`` architecture.  Loss profiles are selected on 2018--2019
before the reused 2020--2021 development metrics are calculated.  The 2025
control is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import fuxi_allseason_ensemble_calibration as base
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_hybrid_loss_ablation_v1"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_hybrid_loss_ablation"
)
PLAN_PATH = PROJECT_ROOT / "plan/HYBRID_LOSS_ABLATION_20260822.md"
EXPECTED_CACHE_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
)


@dataclass(frozen=True)
class LossProfile:
    name: str
    alpha_mse: float
    label: str
    selectable: bool = True

    def __post_init__(self) -> None:
        if not self.name or not 0.0 <= self.alpha_mse <= 1.0:
            raise ValueError("loss profile needs a name and alpha in [0, 1]")


LOSS_PROFILES = (
    LossProfile("crps_only", 0.00, "CRPS only"),
    LossProfile("hybrid_010", 0.10, "Hybrid · α=0.10"),
    LossProfile("hybrid_025", 0.25, "Hybrid · α=0.25"),
    LossProfile("hybrid_050", 0.50, "Hybrid · α=0.50"),
    LossProfile("mse_only", 1.00, "MSE only", selectable=False),
)
PROFILE_BY_NAME = {profile.name: profile for profile in LOSS_PROFILES}
PROFILE_NAMES = tuple(PROFILE_BY_NAME)
PROFILE_COLORS = {
    "crps_only": "#009E73",
    "hybrid_010": "#56B4E9",
    "hybrid_025": "#0072B2",
    "hybrid_050": "#CC79A7",
    "mse_only": "#D55E00",
}
PROFILE_MARKERS = {
    "crps_only": "P",
    "hybrid_010": "^",
    "hybrid_025": "D",
    "hybrid_050": "s",
    "mse_only": "X",
}

# The V1 numerical evaluators deliberately use global method registries.
# Register V2 loss arms locally in this process without changing V1 defaults.
for _profile in LOSS_PROFILES:
    base.METHOD_LABELS[_profile.name] = _profile.label
    base.PLOT_METHOD_LABELS[_profile.name] = _profile.label
    base.METHOD_COLORS[_profile.name] = PROFILE_COLORS[_profile.name]
    base.METHOD_MARKERS[_profile.name] = PROFILE_MARKERS[_profile.name]


class HybridAblationError(RuntimeError):
    """Raised when the frozen V2 experimental contract is violated."""


@dataclass(frozen=True)
class HybridLossComponents:
    total: torch.Tensor
    crps: torch.Tensor
    mse: torch.Tensor
    scaled_mse: torch.Tensor


@dataclass(frozen=True)
class HybridTrainingRun:
    profile: str
    alpha_mse: float
    seed: int
    best_epoch: int
    stopped_epoch: int
    stopping_reason: str
    best_validation_objective: float
    best_validation_crps: float
    best_validation_mse: float
    best_validation_rmse: float
    best_validation_coverage_error: float
    elapsed_seconds: float
    parameter_count: int
    checkpoint: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def weighted_ensemble_mean_mse(
    members: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Area-weighted MSE of the physical ensemble mean."""

    if members.ndim != truth.ndim + 1:
        raise ValueError("members must add one member axis to truth")
    expected = (members.shape[0], *members.shape[2:])
    if tuple(truth.shape) != expected:
        raise ValueError(f"truth must have shape {expected}, got {tuple(truth.shape)}")
    if members.device != truth.device or weights.device != truth.device:
        raise ValueError("members, truth, and weights must share a device")
    forecast = members.mean(dim=1)
    try:
        broadcast_weights = torch.broadcast_to(
            weights.to(dtype=truth.dtype), truth.shape
        )
    except RuntimeError as error:
        raise ValueError("weights do not broadcast to truth") from error
    valid = (
        torch.isfinite(truth)
        & torch.isfinite(broadcast_weights)
        & (broadcast_weights > 0.0)
    )
    if bool(torch.any(valid & ~torch.isfinite(forecast))):
        raise FloatingPointError("ensemble mean is non-finite on scoring support")
    effective = torch.where(valid, broadcast_weights, torch.zeros_like(broadcast_weights))
    denominator = effective.sum()
    if not bool(denominator > 0.0):
        raise ValueError("weights contain no positive valid support")
    safe_error = torch.where(valid, forecast - truth, torch.zeros_like(truth))
    return torch.sum(safe_error.square() * effective) / denominator


def weighted_mean_case_lead_rmse(
    members: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean case/lead spatial RMSE, matching the reported evaluation metric."""

    if members.ndim != truth.ndim + 1:
        raise ValueError("members must add one member axis to truth")
    if tuple(truth.shape) != (members.shape[0], *members.shape[2:]):
        raise ValueError("truth shape does not match members")
    if members.device != truth.device or weights.device != truth.device:
        raise ValueError("members, truth, and weights must share a device")
    forecast = members.mean(dim=1)
    try:
        broadcast_weights = torch.broadcast_to(
            weights.to(dtype=truth.dtype), truth.shape
        )
    except RuntimeError as error:
        raise ValueError("weights do not broadcast to truth") from error
    valid = (
        torch.isfinite(truth)
        & torch.isfinite(broadcast_weights)
        & (broadcast_weights > 0.0)
    )
    if bool(torch.any(valid & ~torch.isfinite(forecast))):
        raise FloatingPointError("ensemble mean is non-finite on scoring support")
    effective = torch.where(valid, broadcast_weights, torch.zeros_like(broadcast_weights))
    safe_error = torch.where(valid, forecast - truth, torch.zeros_like(truth))
    numerator = torch.sum(safe_error.square() * effective, dim=(-2, -1))
    denominator = torch.sum(effective, dim=(-2, -1))
    if not bool(torch.all(denominator > 0.0)):
        raise ValueError("one or more case/lead fields have no positive valid support")
    return torch.sqrt(numerator / denominator).mean()


def hybrid_loss(
    members: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
    profile: LossProfile,
    *,
    raw_train_crps: float,
    raw_train_mse: float,
) -> HybridLossComponents:
    """Return the frozen CRPS--ensemble-mean-MSE scalarization."""

    if raw_train_crps <= 0.0 or raw_train_mse <= 0.0:
        raise ValueError("training-only raw CRPS and MSE scales must be positive")
    crps = base._crps_loss(members, truth, weights)
    mse = weighted_ensemble_mean_mse(members, truth, weights)
    scaled_mse = mse * (float(raw_train_crps) / float(raw_train_mse))
    alpha = profile.alpha_mse
    if alpha == 0.0:
        total = crps
    elif alpha == 1.0:
        total = scaled_mse
    else:
        total = (1.0 - alpha) * crps + alpha * scaled_mse
    return HybridLossComponents(total=total, crps=crps, mse=mse, scaled_mse=scaled_mse)


def raw_training_references(
    members: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 8,
) -> tuple[float, float]:
    """Compute C0/E0 from raw FuXi using only effective training cases."""

    selected = np.asarray(indices, dtype=np.int64)
    crps_sum = 0.0
    mse_sum = 0.0
    value_count = 0
    for start in range(0, len(selected), chunk_size):
        chunk_indices = selected[start : start + chunk_size]
        ensemble = np.asarray(members[chunk_indices], dtype=np.float32)
        target = np.asarray(truth[chunk_indices], dtype=np.float32)
        crps = base._weighted_field_mean(
            base.numpy_ensemble_crps(ensemble, target), weights
        )
        error = ensemble.mean(axis=1, dtype=np.float64) - target
        mse = base._weighted_field_mean(error**2, weights)
        if not np.isfinite(crps).all() or not np.isfinite(mse).all():
            raise HybridAblationError("training-only raw reference is non-finite")
        crps_sum += float(crps.sum(dtype=np.float64))
        mse_sum += float(mse.sum(dtype=np.float64))
        value_count += int(crps.size)
    if value_count != len(selected) * 6:
        raise HybridAblationError("training reference did not cover all cases and leads")
    result = crps_sum / value_count, mse_sum / value_count
    if min(result) <= 0.0:
        raise HybridAblationError("training-only raw reference scales must be positive")
    return result


@torch.no_grad()
def validation_components(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    weights: torch.Tensor,
    device: torch.device,
    profile: LossProfile,
    *,
    raw_train_crps: float,
    raw_train_mse: float,
    use_amp: bool,
) -> dict[str, float]:
    """Evaluate full-member validation objective and calibration guards."""

    model.eval()
    sums = {"objective": 0.0, "crps": 0.0, "mse": 0.0, "scaled_mse": 0.0}
    reported_rmse_sum = 0.0
    coverage_numerators = {coverage: 0.0 for coverage in base.COVERAGES}
    coverage_denominator = 0.0
    case_count = 0
    for members, context, truth in loader:
        members = members.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            corrected, _, _ = base._call_model(
                model, members, context, member_subsample=None
            )
            components = hybrid_loss(
                corrected,
                truth,
                weights,
                profile,
                raw_train_crps=raw_train_crps,
                raw_train_mse=raw_train_mse,
            )
        batch_count = int(members.shape[0])
        case_count += batch_count
        for name, value in (
            ("objective", components.total),
            ("crps", components.crps),
            ("mse", components.mse),
            ("scaled_mse", components.scaled_mse),
        ):
            sums[name] += float(value.detach().cpu()) * batch_count
        corrected_float = corrected.detach().float()
        truth_float = truth.detach().float()
        reported_rmse_sum += float(
            weighted_mean_case_lead_rmse(
                corrected_float,
                truth_float,
                weights.float(),
            ).cpu()
        ) * batch_count

        broadcast_weights = torch.broadcast_to(
            weights.float(), truth_float.shape
        )
        valid = (
            torch.isfinite(truth_float)
            & torch.isfinite(broadcast_weights)
            & (broadcast_weights > 0.0)
        )
        effective = torch.where(
            valid, broadcast_weights, torch.zeros_like(broadcast_weights)
        )
        coverage_denominator += float(effective.sum().cpu())
        for coverage in base.COVERAGES:
            alpha = (1.0 - coverage) / 2.0
            lower = torch.quantile(corrected_float, alpha, dim=1)
            upper = torch.quantile(corrected_float, 1.0 - alpha, dim=1)
            covered = valid & (truth_float >= lower) & (truth_float <= upper)
            coverage_numerators[coverage] += float(
                torch.sum(effective * covered.to(effective.dtype)).cpu()
            )
    if case_count == 0 or coverage_denominator <= 0.0:
        raise HybridAblationError("validation loader is empty or unscorable")
    result = {name: value / case_count for name, value in sums.items()}
    # The loss uses pooled MSE, but the paper reports the arithmetic mean of
    # case/lead spatial RMSE values. This is identical to evaluate_ensemble()
    # followed by summarize_metrics(), and is therefore the selection metric.
    result["rmse"] = reported_rmse_sum / case_count
    errors = []
    for coverage in base.COVERAGES:
        name = f"coverage_{int(round(100.0 * coverage))}"
        result[name] = coverage_numerators[coverage] / coverage_denominator
        errors.append(abs(result[name] - coverage))
    result["coverage_error"] = float(np.mean(errors))
    return result


def train_one_profile(
    profile: LossProfile,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: base.ContextBundle,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    weights: np.ndarray,
    run_directory: Path,
    *,
    raw_train_crps: float,
    raw_train_mse: float,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    member_subsample: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[pd.DataFrame, HybridTrainingRun]:
    """Train the fixed location--spread architecture for one loss arm."""

    base.set_deterministic_seed(seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best.pt"
    model = EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loader = base.make_loader(
        base.EnsembleCaseDataset(members, truth, context, train_indices),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    validation_loader = base.make_loader(
        base.EnsembleCaseDataset(members, truth, context, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    spatial_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
        min_lr=1.0e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    history: list[dict[str, Any]] = []
    start_time = time.monotonic()

    initial = validation_components(
        model,
        validation_loader,
        spatial_weights,
        device,
        profile,
        raw_train_crps=raw_train_crps,
        raw_train_mse=raw_train_mse,
        use_amp=use_amp,
    )
    history.append(
        {
            "profile": profile.name,
            "alpha_mse": profile.alpha_mse,
            "seed": seed,
            "epoch": 0,
            "train_objective": np.nan,
            "train_crps": np.nan,
            "train_mse": np.nan,
            **{f"validation_{name}": value for name, value in initial.items()},
            "learning_rate": learning_rate,
            "is_best": True,
        }
    )
    best = initial
    best_epoch = 0
    stale_epochs = 0
    stopping_reason = "max_epochs"

    def save_checkpoint(epoch: int, validation: Mapping[str, float]) -> None:
        torch.save(
            {
                "experiment": EXPERIMENT,
                "profile": profile.name,
                "alpha_mse": profile.alpha_mse,
                "seed": seed,
                "epoch": epoch,
                "raw_train_crps": raw_train_crps,
                "raw_train_mse": raw_train_mse,
                "validation": dict(validation),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    save_checkpoint(0, initial)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sums = {"objective": 0.0, "crps": 0.0, "mse": 0.0}
        train_count = 0
        for batch_members, batch_context, batch_truth in train_loader:
            batch_members = batch_members.to(device, non_blocking=True)
            batch_context = batch_context.to(device, non_blocking=True)
            batch_truth = batch_truth.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp and device.type == "cuda",
            ):
                corrected, _, _ = base._call_model(
                    model,
                    batch_members,
                    batch_context,
                    member_subsample=member_subsample,
                )
                components = hybrid_loss(
                    corrected,
                    batch_truth,
                    spatial_weights,
                    profile,
                    raw_train_crps=raw_train_crps,
                    raw_train_mse=raw_train_mse,
                )
            if not torch.isfinite(components.total):
                raise FloatingPointError(
                    f"non-finite hybrid loss for {profile.name}, seed {seed}, epoch {epoch}"
                )
            scaler.scale(components.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch_members.shape[0])
            train_count += count
            train_sums["objective"] += float(components.total.detach().cpu()) * count
            train_sums["crps"] += float(components.crps.detach().cpu()) * count
            train_sums["mse"] += float(components.mse.detach().cpu()) * count
        train_values = {name: value / train_count for name, value in train_sums.items()}
        current = validation_components(
            model,
            validation_loader,
            spatial_weights,
            device,
            profile,
            raw_train_crps=raw_train_crps,
            raw_train_mse=raw_train_mse,
            use_amp=use_amp,
        )
        scheduler.step(current["objective"])
        improved = current["objective"] < best["objective"] - 1.0e-6
        if improved:
            best = current
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(epoch, current)
        else:
            stale_epochs += 1
        history.append(
            {
                "profile": profile.name,
                "alpha_mse": profile.alpha_mse,
                "seed": seed,
                "epoch": epoch,
                "train_objective": train_values["objective"],
                "train_crps": train_values["crps"],
                "train_mse": train_values["mse"],
                **{f"validation_{name}": value for name, value in current.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"[{profile.name} seed={seed}] epoch={epoch:03d} "
            f"train_L={train_values['objective']:.6f} "
            f"val_L={current['objective']:.6f} val_CRPS={current['crps']:.6f} "
            f"val_RMSE={current['rmse']:.6f} best={best['objective']:.6f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= patience:
            stopping_reason = "early_stopping_patience"
            break

    record = HybridTrainingRun(
        profile=profile.name,
        alpha_mse=profile.alpha_mse,
        seed=seed,
        best_epoch=best_epoch,
        stopped_epoch=int(history[-1]["epoch"]),
        stopping_reason=stopping_reason,
        best_validation_objective=float(best["objective"]),
        best_validation_crps=float(best["crps"]),
        best_validation_mse=float(best["mse"]),
        best_validation_rmse=float(best["rmse"]),
        best_validation_coverage_error=float(best["coverage_error"]),
        elapsed_seconds=float(time.monotonic() - start_time),
        parameter_count=parameter_count,
        checkpoint=str(checkpoint_path),
    )
    del model
    return pd.DataFrame(history), record


def select_loss_profile(
    validation_summary: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = base.SEEDS,
) -> dict[str, Any]:
    """Apply the frozen validation-only non-inferiority and Pareto rule."""

    required = {
        "split",
        "profile",
        "alpha_mse",
        "seed",
        "crps",
        "rmse",
        "coverage_error",
    }
    missing = sorted(required - set(validation_summary.columns))
    if missing:
        raise ValueError(f"validation summary lacks columns: {missing}")
    if set(validation_summary.profile) != set(PROFILE_NAMES):
        raise ValueError("validation summary must contain every frozen loss profile")
    if set(validation_summary.split) != {"validation"}:
        raise ValueError("loss-profile selection may use validation rows only")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    if not expected_seed_set:
        raise ValueError("selection requires at least one optimization seed")
    for profile, rows in validation_summary.groupby("profile"):
        observed = {int(seed) for seed in rows.seed}
        if observed != expected_seed_set or len(rows) != len(expected_seed_set):
            raise ValueError(
                f"profile {profile!r} has seed rows {sorted(observed)}; "
                f"expected exactly {sorted(expected_seed_set)}"
            )
        expected_alpha = PROFILE_BY_NAME[str(profile)].alpha_mse
        if not np.allclose(
            rows.alpha_mse.to_numpy(dtype=np.float64),
            expected_alpha,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"profile {profile!r} has inconsistent alpha values")
    grouped = (
        validation_summary.groupby(["profile", "alpha_mse"], as_index=False)[
            ["crps", "rmse", "coverage_error"]
        ]
        .mean()
        .sort_values("alpha_mse")
    )
    control = grouped.loc[grouped.profile == "crps_only"].iloc[0]
    records = []
    for row in grouped.itertuples(index=False):
        profile = PROFILE_BY_NAME[row.profile]
        crps_ratio = float(row.crps / control.crps)
        coverage_delta = float(row.coverage_error - control.coverage_error)
        eligible = bool(
            profile.selectable
            and profile.name != "crps_only"
            and crps_ratio <= 1.005
            and coverage_delta <= 0.01
        )
        joint_candidate = bool(profile.name == "crps_only" or eligible)
        records.append(
            {
                "profile": profile.name,
                "alpha_mse": profile.alpha_mse,
                "mean_validation_crps": float(row.crps),
                "mean_validation_rmse": float(row.rmse),
                "mean_validation_coverage_error": float(row.coverage_error),
                "crps_ratio_vs_control": crps_ratio,
                "coverage_error_delta_vs_control": coverage_delta,
                "eligible_hybrid": eligible,
                "eligible_joint_candidate": joint_candidate,
            }
        )
    eligible_hybrids = [row for row in records if row["eligible_hybrid"]]
    candidates = [row for row in records if row["eligible_joint_candidate"]]
    minimum = min(row["mean_validation_rmse"] for row in candidates)
    tied = [
        row
        for row in candidates
        if row["mean_validation_rmse"] <= minimum * 1.0025
    ]
    winner = min(tied, key=lambda row: row["alpha_mse"])
    selected = str(winner["profile"])
    if selected != "crps_only":
        reason = (
            "lowest validation RMSE among the CRPS control and guard-eligible hybrids; "
            "smaller alpha wins within 0.25%"
        )
    elif not eligible_hybrids:
        reason = "no hybrid passed both validation non-inferiority guards"
    else:
        reason = (
            "CRPS-only had the lowest validation RMSE, or was within the 0.25% "
            "smaller-alpha tie, among all guard-eligible candidates"
        )
    return {
        "status": "frozen_before_development_evaluation",
        "selected_profile": selected,
        "selected_alpha_mse": PROFILE_BY_NAME[selected].alpha_mse,
        "reason": reason,
        "test_metrics_consulted": False,
        "rules": {
            "crps_noninferiority_ratio_max": 1.005,
            "coverage_error_delta_max": 0.01,
            "rmse_tie_relative_tolerance": 0.0025,
            "mse_only_selectable": False,
            "crps_control_competes_in_rmse_selection": True,
            "selected_hybrid_may_not_be_validation_rmse_dominated_by_control": True,
        },
        "validation_profiles": records,
    }


def load_checkpoint_model(
    checkpoint_path: Path,
    profile: LossProfile,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    """Restore one trusted, locally produced best checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if (
        checkpoint.get("experiment") != EXPERIMENT
        or checkpoint.get("profile") != profile.name
        or int(checkpoint.get("seed", -1)) != seed
        or not math.isclose(
            float(checkpoint.get("alpha_mse", np.nan)),
            profile.alpha_mse,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise HybridAblationError(
            f"checkpoint identity mismatch for {profile.name}, seed {seed}"
        )
    model = EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def source_snapshot(output: Path) -> dict[str, str]:
    """Copy the exact executable contract into the result before training."""

    sources = {
        "src/fuxi_allseason_hybrid_loss_ablation.py": Path(__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "slurm/run_fuxi_allseason_hybrid_loss_ablation.sbatch": PROJECT_ROOT
        / "slurm/run_fuxi_allseason_hybrid_loss_ablation.sbatch",
        "plan/HYBRID_LOSS_ABLATION_20260822.md": PLAN_PATH,
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"frozen source is missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative not in {"manifest.json", "failure.json"}:
            checksums[relative] = sha256_file(path)
    return checksums


def plot_training_histories(
    history: pd.DataFrame,
    output: Path,
    *,
    smoke: bool,
) -> None:
    """Plot the actual optimized train/validation L_alpha histories."""

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), constrained_layout=True)
    seed_styles = ("-", "--", ":")
    for axis, profile in zip(axes.ravel(), LOSS_PROFILES):
        selected = history.loc[history.profile == profile.name]
        for seed_index, (seed, rows) in enumerate(selected.groupby("seed")):
            rows = rows.sort_values("epoch")
            trained = rows.loc[rows.epoch > 0]
            style = seed_styles[seed_index % len(seed_styles)]
            axis.plot(
                trained.epoch,
                trained.train_objective,
                color="0.62",
                linestyle=style,
                linewidth=0.95,
                label=f"train · seed {seed}",
            )
            axis.plot(
                rows.epoch,
                rows.validation_objective,
                color=PROFILE_COLORS[profile.name],
                linestyle=style,
                linewidth=1.25,
                label=f"validation · seed {seed}",
            )
            best = rows.loc[rows.is_best.astype(bool)].iloc[-1]
            axis.scatter(
                [best.epoch],
                [best.validation_objective],
                s=15,
                color=PROFILE_COLORS[profile.name],
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )
        axis.set_title(profile.label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(r"$L_\alpha$ (CRPS-scaled units)")
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=4.9, loc="best", ncol=2)
    axes.ravel()[-1].axis("off")
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + r"Hybrid-loss ablation: optimized $L_\alpha$ histories",
        fontsize=10.2,
        fontweight="semibold",
    )
    base._save_figure(figure, output)


def plot_validation_metric_histories(
    history: pd.DataFrame,
    output: Path,
    *,
    smoke: bool,
) -> None:
    """Show the guarded CRPS and reported-RMSE validation trajectories."""

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), constrained_layout=True)
    seed_styles = ("-", "--", ":")
    for axis, profile in zip(axes.ravel(), LOSS_PROFILES):
        selected = history.loc[history.profile == profile.name]
        secondary = axis.twinx()
        for seed_index, (seed, rows) in enumerate(selected.groupby("seed")):
            rows = rows.sort_values("epoch")
            style = seed_styles[seed_index % len(seed_styles)]
            axis.plot(
                rows.epoch,
                rows.validation_crps,
                color=PROFILE_COLORS[profile.name],
                linestyle=style,
                linewidth=1.25,
                label=f"CRPS · seed {seed}",
            )
            secondary.plot(
                rows.epoch,
                rows.validation_rmse,
                color="0.38",
                linestyle=style,
                linewidth=0.9,
                alpha=0.8,
            )
        axis.set_title(profile.label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation CRPS")
        secondary.set_ylabel("Validation RMSE", color="0.38")
        axis.grid(True, alpha=0.2)
        axis.spines[["top"]].set_visible(False)
        secondary.spines[["top"]].set_visible(False)
        axis.legend(frameon=False, fontsize=5.4, loc="best")
    axes.ravel()[-1].axis("off")
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "Hybrid-loss ablation: guarded validation metrics",
        fontsize=10.2,
        fontweight="semibold",
    )
    base._save_figure(figure, output)


def plot_validation_tradeoff(
    validation_profiles: pd.DataFrame,
    selected_profile: str,
    output: Path,
    *,
    smoke: bool,
) -> None:
    """Visualize the pre-test deterministic--probabilistic selection surface."""

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.17,
        top=0.69,
        wspace=0.30,
    )
    for profile in LOSS_PROFILES:
        row = validation_profiles.loc[
            validation_profiles.profile == profile.name
        ].iloc[0]
        size = 86 if profile.name == selected_profile else 48
        edge = "black" if profile.name == selected_profile else "white"
        axes[0].scatter(
            row.mean_validation_crps,
            row.mean_validation_rmse,
            s=size,
            color=PROFILE_COLORS[profile.name],
            marker=PROFILE_MARKERS[profile.name],
            edgecolor=edge,
            linewidth=1.0,
            label=profile.label,
            zorder=4,
        )
        axes[1].scatter(
            profile.alpha_mse,
            row.mean_validation_coverage_error,
            s=size,
            color=PROFILE_COLORS[profile.name],
            marker=PROFILE_MARKERS[profile.name],
            edgecolor=edge,
            linewidth=1.0,
            zorder=4,
        )
    axes[0].set_xlabel("Validation CRPS (lower is better)")
    axes[0].set_ylabel("Ensemble-mean RMSE (lower is better)")
    axes[0].set_title("(a) Accuracy trade-off")
    axes[1].set_xlabel("Deterministic loss weight α")
    axes[1].set_ylabel("Mean |coverage − nominal|")
    axes[1].set_title("(b) Probabilistic guard")
    axes[1].axhline(
        validation_profiles.loc[
            validation_profiles.profile == "crps_only",
            "mean_validation_coverage_error",
        ].iloc[0]
        + 0.01,
        color="0.45",
        linestyle=":",
        linewidth=0.9,
    )
    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        fontsize=6.5,
    )
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "Selection uses 2018–2019 validation only",
        y=0.98,
        fontsize=10.0,
        fontweight="semibold",
    )
    base._save_figure(figure, output)


def plot_development_pareto(
    pooled: pd.DataFrame,
    selected_profile: str,
    output: Path,
    *,
    smoke: bool,
) -> None:
    """Descriptive post-selection point/probabilistic performance plot."""

    figure, axis = plt.subplots(figsize=(4.6, 3.35), constrained_layout=True)
    order = ("raw_fuxi", *PROFILE_NAMES)
    for method in order:
        row = pooled.loc[pooled.method == method].iloc[0]
        selected = method == selected_profile
        axis.scatter(
            row.crps,
            row.rmse,
            s=95 if selected else 52,
            color=base.METHOD_COLORS[method],
            marker=base.METHOD_MARKERS[method],
            edgecolor="black" if selected else "white",
            linewidth=1.0,
            label=base.PLOT_METHOD_LABELS[method],
            zorder=4,
        )
    axis.set_xlabel("Development CRPS (lower is better)")
    axis.set_ylabel("Development ensemble-mean RMSE (lower is better)")
    axis.grid(True, alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=6.4, ncol=2)
    prefix = "SMOKE — " if smoke else ""
    axis.set_title(
        prefix + "2020–2021 deterministic–probabilistic Pareto view",
        fontweight="semibold",
    )
    base._save_figure(figure, output)


def build_readme(
    pooled: pd.DataFrame,
    selection: Mapping[str, Any],
    raw_bootstrap: pd.DataFrame,
    control_bootstrap: pd.DataFrame,
    *,
    smoke: bool,
    test_count: int,
) -> str:
    status = (
        "NON-SCIENTIFIC GPU SMOKE TEST"
        if smoke
        else "POST-HOC 2020–2021 DEVELOPMENT ABLATION (not an untouched final test)"
    )
    selected = str(selection["selected_profile"])
    lines = [
        "# FuXi deterministic–probabilistic hybrid-loss ablation",
        "",
        f"Status: **{status}**",
        "",
        "This one-factor experiment keeps the same India-domain, 51-member FuXi input, "
        "IMD target, all-season splits, location–spread architecture, optimizer, and seeds. "
        "Only the training loss weight changes. The same corrected ensemble is used "
        "probabilistically (members, probabilities, quantiles) and deterministically "
        "(its physical ensemble mean); there is no second forecast to blend.",
        "",
        "The selected arm was frozen from 2018–2019 validation before any 2020–2021 "
        f"metric was calculated: **`{selected}` (α={selection['selected_alpha_mse']:.2f})**. "
        f"Reason: {selection['reason']}.",
        "",
        f"Reporting covers {test_count} reused 2020–2021 development initializations. "
        "The 2025 control remains sealed and unopened.",
        "",
        "## Pooled development metrics",
        "",
        "| Method | CRPS | CRPSS vs raw | RMSE | RMSE skill | MAE | Bias | ACC | Spread/error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ("raw_fuxi", *PROFILE_NAMES)
    for method in order:
        row = pooled.loc[pooled.method == method].iloc[0]
        lines.append(
            f"| {row.method_label} | {row.crps:.4f} | {row.crpss_vs_raw:+.3f} | "
            f"{row.rmse:.4f} | {row.rmse_skill_pct_vs_raw:+.2f}% | "
            f"{row.mae:.4f} | {row.bias:+.4f} | {row.acc:.4f} | "
            f"{row.spread_skill_ratio:.3f} |"
        )
    if not smoke:
        raw_rows = raw_bootstrap.loc[
            (raw_bootstrap.method == selected)
            & (raw_bootstrap.lead_scope == "W1-W6")
        ]
        lines.extend(["", "## Selected-arm paired uncertainty", ""])
        comparisons: list[tuple[str, pd.DataFrame]] = [("versus raw FuXi", raw_rows)]
        if selected != "crps_only":
            control_rows = control_bootstrap.loc[
                (control_bootstrap.method == selected)
                & (control_bootstrap.lead_scope == "W1-W6")
            ]
            comparisons.append(("versus CRPS only", control_rows))
        for heading, rows in comparisons:
            lines.append(f"**{heading}:**")
            lines.append("")
            for row in rows.itertuples(index=False):
                lines.append(
                    f"- `{row.effect_name}`: {row.effect:+.3f} "
                    f"(95% block-bootstrap CI {row.ci_lower:+.3f} to {row.ci_upper:+.3f})."
                )
            lines.append("")
        if selected == "crps_only":
            lines.extend(
                [
                    "**versus CRPS only:** not applicable; the validation rule retained the control.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Artifact map",
            "",
            "- `selection.json`: immutable validation-only choice and guard outcomes.",
            "- `history/training_history.csv`: CRPS, MSE, RMSE, coverage, and hybrid-objective histories.",
            "- `metrics/validation_checkpoint_metrics_by_seed.csv`: pre-test checkpoint evidence.",
            "- `metrics/pooled_metrics.csv` and `weekwise_metrics.csv`: headline scores.",
            "- `metrics/seed_*`: optimization-replicate diagnostics, not extra weather samples.",
            "- `metrics/paired_block_bootstrap*.csv`: paired dependence-aware uncertainty.",
            "- `metrics/matrices/` and `seasonal_matrices/`: paper-ready method × week tables.",
            "- `figures/training_loss_curves.*`: the actual optimized train/validation `L_alpha` curves.",
            "- `figures/validation_metric_curves.*`: validation CRPS and reported-RMSE trajectories.",
            "- `figures/`: validation trade-off plus deterministic and probabilistic diagnostics.",
            "- `manifest.json`: frozen contract, provenance, checksums, and run settings.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    """Train, select, and evaluate the frozen five-arm loss ablation."""

    started_at = time.monotonic()
    snapshot_checksums = source_snapshot(output)
    profiles = base._parse_names(args.profiles, PROFILE_NAMES, "loss profiles")
    seeds = base._parse_seeds(args.seeds)
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading verified member cache {args.cache}...", flush=True)
    cache = base.load_member_cache(Path(args.cache), allow_partial=args.smoke)
    cache_provenance = base.cache_provenance(cache)
    if cache_provenance.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise HybridAblationError("member cache source fingerprint is not canonical")
    if not args.smoke and cache_provenance.get("data_sha256") != EXPECTED_CACHE_SHA256:
        raise HybridAblationError("full member cache SHA-256 is not canonical")

    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(indices) for name, indices in splits.as_dict().items()}
    if args.smoke and cache.members.shape[0] == 64:
        expected_smoke = {"train": 32, "validation": 16, "test": 16, "embargo": 0}
        if split_counts != expected_smoke:
            raise HybridAblationError(
                f"stratified smoke split {split_counts}, expected {expected_smoke}"
            )
    train_indices = splits.train
    validation_indices = splits.validation
    test_indices = splits.test
    if args.smoke:
        train_indices = base.select_evenly(train_indices, 32)
        validation_indices = base.select_evenly(validation_indices, 16)
        test_indices = base.select_evenly(test_indices, 16)
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise HybridAblationError("train, validation, and development splits must be nonempty")
    print(
        f"Effective cases: train={len(train_indices)}, "
        f"validation={len(validation_indices)}, test={len(test_indices)}, "
        f"embargo={len(splits.embargo)}",
        flush=True,
    )
    print("Loading IMD 2002–2022 and fitting train-only context...", flush=True)
    observations = base.load_imd_observations(cache)
    context = base.build_context_bundle(cache, observations, train_indices)
    evaluation_directory = output / "evaluation"
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    normalized_weights = observations.weights / observations.weights.sum(dtype=np.float64)
    scoring_support_path = evaluation_directory / "scoring_support.npz"
    np.savez_compressed(
        scoring_support_path,
        latitude=cache.latitude.astype(np.float64),
        longitude=cache.longitude.astype(np.float64),
        observation_fraction=observations.observation_fraction.astype(np.float32),
        support_mask=observations.weights > 0.0,
        scoring_weight_km2_fraction=observations.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights.astype(np.float64),
    )

    print("Computing raw C0/E0 on the effective training split only...", flush=True)
    raw_train_crps, raw_train_mse = raw_training_references(
        cache.members,
        observations.weekly_truth,
        train_indices,
        observations.weights,
        chunk_size=args.evaluation_batch_size,
    )
    write_json(
        output / "training_reference.json",
        {
            "scope": "effective_training_split_only",
            "case_count": len(train_indices),
            "raw_train_crps": raw_train_crps,
            "raw_train_mse": raw_train_mse,
            "crps_per_mse_scale": raw_train_crps / raw_train_mse,
        },
    )
    print(
        f"Training references: C0={raw_train_crps:.6f}, "
        f"E0={raw_train_mse:.6f}, C0/E0={raw_train_crps/raw_train_mse:.6f}",
        flush=True,
    )

    histories: list[pd.DataFrame] = []
    training_runs: list[HybridTrainingRun] = []
    validation_rows: list[dict[str, Any]] = []
    models_directory = output / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    for profile_name in profiles:
        profile = PROFILE_BY_NAME[profile_name]
        for seed in seeds:
            print(f"Training {profile.name}, seed {seed}...", flush=True)
            run_directory = models_directory / profile.name / f"seed_{seed}"
            history, record = train_one_profile(
                profile,
                seed,
                cache.members,
                observations.weekly_truth,
                context,
                train_indices,
                validation_indices,
                observations.weights,
                run_directory,
                raw_train_crps=raw_train_crps,
                raw_train_mse=raw_train_mse,
                device=device,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                member_subsample=args.member_subsample,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            histories.append(history)
            training_runs.append(record)
            checkpoint = torch.load(
                Path(record.checkpoint), map_location="cpu", weights_only=False
            )
            validation = checkpoint["validation"]
            validation_rows.append(
                {
                    "split": "validation",
                    "profile": profile.name,
                    "alpha_mse": profile.alpha_mse,
                    "seed": seed,
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    **{name: float(value) for name, value in validation.items()},
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    history_frame = pd.concat(histories, ignore_index=True)
    validation_by_seed = pd.DataFrame(validation_rows).sort_values(
        ["alpha_mse", "seed"]
    )
    metrics_directory = output / "metrics"
    history_directory = output / "history"
    figures_directory = output / "figures"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    history_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(history_directory / "training_history.csv", index=False)
    validation_by_seed.to_csv(
        metrics_directory / "validation_checkpoint_metrics_by_seed.csv", index=False
    )

    # Scientific firewall: this file is durably written before test members/truth
    # are sliced or any 2020--2021 forecast metric is calculated.
    selection = select_loss_profile(validation_by_seed, expected_seeds=seeds)
    selection["written_utc"] = utc_now()
    selection["development_evaluation_started"] = False
    selection_path = output / "selection.json"
    write_json(selection_path, selection)
    selection_frame = pd.DataFrame(selection["validation_profiles"])
    selection_frame.to_csv(
        metrics_directory / "validation_profile_summary.csv", index=False
    )
    selected_profile = str(selection["selected_profile"])
    print(
        f"VALIDATION SELECTION LOCKED: {selected_profile}; {selection['reason']}",
        flush=True,
    )

    development_evaluation_started_utc = utc_now()
    test_initializations = cache.initializations[test_indices]
    test_truth = observations.weekly_truth[test_indices]
    test_climatology = observations.weekly_climatology[test_indices]
    test_members = base.materialize_cases(cache.members, test_indices)
    method_order = ("raw_fuxi", *profiles)
    metric_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    reliability_frames: list[pd.DataFrame] = []
    seed_metric_frames: list[pd.DataFrame] = []
    seed_rank_frames: list[pd.DataFrame] = []
    seed_reliability_frames: list[pd.DataFrame] = []

    print("Evaluating raw FuXi on 2020–2021 development data...", flush=True)
    raw_metrics, raw_ranks = base.evaluate_ensemble(
        "raw_fuxi",
        test_members,
        test_truth,
        test_climatology,
        test_initializations,
        observations.weights,
        chunk_size=args.evaluation_batch_size,
    )
    metric_frames.append(raw_metrics)
    rank_frames.append(raw_ranks)
    reliability_frames.append(
        base.reliability_bins(
            "raw_fuxi",
            test_members,
            test_truth,
            observations.weights,
            chunk_size=args.evaluation_batch_size,
        )
    )

    for profile_name in profiles:
        profile = PROFILE_BY_NAME[profile_name]
        profile_metrics: list[pd.DataFrame] = []
        profile_ranks: list[pd.DataFrame] = []
        profile_bins: list[pd.DataFrame] = []
        for seed in seeds:
            print(f"Evaluating {profile.name}, seed {seed}...", flush=True)
            run_directory = models_directory / profile.name / f"seed_{seed}"
            model = load_checkpoint_model(
                run_directory / "best.pt", profile, seed, device
            )
            delta, log_scale = base.predict_adjustments(
                model,
                cache.members,
                observations.weekly_truth,
                context,
                test_indices,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            spread_factor = np.exp(np.clip(log_scale, -2.0, 2.0)).astype(np.float32)
            np.savez_compressed(
                run_directory / "test_adjustments.npz",
                initializations=test_initializations,
                delta_log_location=delta,
                log_spread=log_scale,
                spread_factor=spread_factor,
                profile=profile.name,
                alpha_mse=np.float64(profile.alpha_mse),
                seed=np.int64(seed),
            )
            corrected = base.apply_affine_log_calibration(
                test_members, delta, spread_factor
            )
            seed_metrics, seed_ranks = base.evaluate_ensemble(
                profile.name,
                corrected,
                test_truth,
                test_climatology,
                test_initializations,
                observations.weights,
                chunk_size=args.evaluation_batch_size,
                seed_label=seed,
            )
            seed_ranks.insert(1, "seed", seed)
            seed_bins = base.reliability_bins(
                profile.name,
                corrected,
                test_truth,
                observations.weights,
                chunk_size=args.evaluation_batch_size,
            )
            seed_bins.insert(2, "seed", seed)
            seed_metric_frames.append(seed_metrics)
            seed_rank_frames.append(seed_ranks)
            seed_reliability_frames.append(seed_bins)
            profile_metrics.append(seed_metrics)
            profile_ranks.append(seed_ranks)
            profile_bins.append(seed_bins)
            del corrected, spread_factor, delta, log_scale, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        metric_frames.append(
            base.mean_seed_case_metrics(pd.concat(profile_metrics), seeds)
        )
        rank_frames.append(base.mean_seed_rank_histograms(pd.concat(profile_ranks)))
        reliability_frames.append(
            base.mean_seed_reliability_bins(pd.concat(profile_bins))
        )

    case_metrics = pd.concat(metric_frames, ignore_index=True)
    rank_histograms = pd.concat(rank_frames, ignore_index=True)
    reliability_bins = pd.concat(reliability_frames, ignore_index=True)
    seed_case_metrics = pd.concat(seed_metric_frames, ignore_index=True)
    seed_rank_histograms = pd.concat(seed_rank_frames, ignore_index=True)
    seed_reliability_bins = pd.concat(seed_reliability_frames, ignore_index=True)
    expected_metric_rows = len(method_order) * len(test_indices) * 6
    if len(case_metrics) != expected_metric_rows:
        raise HybridAblationError(
            f"expected {expected_metric_rows} case-metric rows, found {len(case_metrics)}"
        )

    weekwise, pooled, seasonal, threshold_reliability = base.summarize_metrics(
        case_metrics
    )
    seed_weekwise = base.summarize_seed_metrics(
        seed_case_metrics,
        weekwise.loc[weekwise.method == "raw_fuxi"],
    )
    seed_variability = base.summarize_seed_variability(seed_weekwise)
    seasonal = base.add_seasonal_raw_comparisons(seasonal)
    bootstrap_samples = min(args.bootstrap_samples, 100) if args.smoke else args.bootstrap_samples
    raw_bootstrap = base.paired_block_bootstrap(
        case_metrics,
        test_initializations,
        method_order,
        n_resamples=bootstrap_samples,
        baseline="raw_fuxi",
    )
    control_bootstrap = base.paired_block_bootstrap(
        case_metrics,
        test_initializations,
        profiles,
        n_resamples=bootstrap_samples,
        baseline="crps_only",
        seed=314159,
    )

    case_metrics.to_csv(metrics_directory / "case_metrics.csv", index=False)
    seed_case_metrics.to_csv(metrics_directory / "seed_case_metrics.csv", index=False)
    weekwise.to_csv(metrics_directory / "weekwise_metrics.csv", index=False)
    seed_weekwise.to_csv(metrics_directory / "seed_weekwise_metrics.csv", index=False)
    seed_variability.to_csv(
        metrics_directory / "seed_variability_by_week.csv", index=False
    )
    pooled.to_csv(metrics_directory / "pooled_metrics.csv", index=False)
    seasonal.to_csv(metrics_directory / "seasonal_weekwise_metrics.csv", index=False)
    threshold_reliability.to_csv(
        metrics_directory / "threshold_reliability_by_week.csv", index=False
    )
    reliability_bins.to_csv(metrics_directory / "reliability_bins.csv", index=False)
    seed_reliability_bins.to_csv(
        metrics_directory / "seed_reliability_bins.csv", index=False
    )
    rank_histograms.to_csv(metrics_directory / "rank_histograms.csv", index=False)
    seed_rank_histograms.to_csv(
        metrics_directory / "seed_rank_histograms.csv", index=False
    )
    raw_bootstrap.to_csv(
        metrics_directory / "paired_block_bootstrap_vs_raw.csv", index=False
    )
    control_bootstrap.to_csv(
        metrics_directory / "paired_block_bootstrap_vs_crps_only.csv", index=False
    )
    base.write_metric_matrices(weekwise, metrics_directory / "matrices")
    for season in ("DJF", "MAM", "JJA", "SON"):
        base.write_metric_matrices(
            seasonal.loc[seasonal.season == season].drop(columns="season"),
            metrics_directory / "seasonal_matrices" / season,
        )

    plot_training_histories(
        history_frame,
        figures_directory / "training_loss_curves",
        smoke=args.smoke,
    )
    plot_validation_metric_histories(
        history_frame,
        figures_directory / "validation_metric_curves",
        smoke=args.smoke,
    )
    plot_validation_tradeoff(
        selection_frame,
        selected_profile,
        figures_directory / "validation_tradeoff",
        smoke=args.smoke,
    )
    original_primary = base.PRIMARY_CONFIGURATION
    base.PRIMARY_CONFIGURATION = selected_profile
    try:
        base.plot_weekwise_metrics(
            weekwise,
            figures_directory / "weekwise_metrics",
            method_order,
            smoke=args.smoke,
            bootstrap=raw_bootstrap,
            seed_variability=seed_variability,
        )
        base.plot_skill_heatmaps(
            weekwise,
            figures_directory / "weekwise_loss_heatmaps",
            method_order,
            smoke=args.smoke,
        )
        base.plot_rank_histograms(
            rank_histograms,
            figures_directory / "rank_histograms_raw_vs_selected",
            selected_profile,
            smoke=args.smoke,
        )
        base.plot_reliability(
            reliability_bins,
            figures_directory / "reliability_diagrams",
            method_order,
            smoke=args.smoke,
        )
        base.plot_probabilistic_diagnostics(
            weekwise,
            seasonal,
            figures_directory / "probabilistic_diagnostics",
            method_order,
            smoke=args.smoke,
        )
    finally:
        base.PRIMARY_CONFIGURATION = original_primary
    plot_development_pareto(
        pooled,
        selected_profile,
        figures_directory / "development_pareto",
        smoke=args.smoke,
    )

    (output / "README.md").write_text(
        build_readme(
            pooled,
            selection,
            raw_bootstrap,
            control_bootstrap,
            smoke=args.smoke,
            test_count=len(test_indices),
        ),
        encoding="utf-8",
    )
    run_records: list[dict[str, Any]] = []
    for record in training_runs:
        values = asdict(record)
        checkpoint_path = Path(record.checkpoint)
        values["checkpoint_sha256"] = sha256_file(checkpoint_path)
        values["checkpoint"] = str(checkpoint_path.relative_to(output))
        run_records.append(values)

    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "post-hoc 2020-2021 development ablation; not an untouched final test"
        ),
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "profiles": list(profiles),
        "seeds": list(seeds),
        "selection": selection,
        "selection_artifact": "selection.json",
        "selection_sha256": sha256_file(selection_path),
        "development_evaluation_started_utc": development_evaluation_started_utc,
        "contract": {
            "forecast": "FuXi native global reforecast; 51 members retained",
            "prediction_and_scoring_region": "39N-0N, 60E-99E, 27x27 India box",
            "target": "IMD weekly mean precipitation, mm day-1",
            "train_years": list(base.TRAIN_YEARS),
            "validation_years": list(base.VALIDATION_YEARS),
            "development_years": list(base.TEST_YEARS),
            "sealed_unopened_years": list(base.SEALED_YEARS),
            "outcome_window_days": 42,
            "statistical_unit": "initialization with all 51 members and six leads grouped",
            "architecture": "fixed permutation-invariant location_spread calibrator",
            "single_ensemble_dual_use": {
                "probabilistic": "empirical 51-member distribution",
                "deterministic": "physical mean of those same 51 calibrated members",
                "separate_model_blend": False,
            },
            "loss": "L_alpha=(1-alpha)*CRPS + alpha*(C0/E0)*ensemble_mean_MSE",
            "memberwise_mse_used": False,
            "test_reuse_warning": "2020-2021 has prior development exposure and is not independent",
            "sealed_2025_target_opened": False,
        },
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test_development": len(test_indices),
        },
        "loss_profiles": [asdict(PROFILE_BY_NAME[name]) for name in profiles],
        "training_reference": {
            "scope": "effective training split only",
            "raw_train_crps": raw_train_crps,
            "raw_train_mse": raw_train_mse,
            "crps_per_mse_scale": raw_train_crps / raw_train_mse,
        },
        "training": {
            "batch_size": args.batch_size,
            "member_subsample": args.member_subsample,
            "full_members_for_validation_and_evaluation": 51,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "automatic_mixed_precision": not args.no_amp,
            "device": str(device),
            "checkpoint_metric": "profile-specific full-51-member validation L_alpha",
            "runs": run_records,
        },
        "seed_aggregation": {
            "role": "optimization variability only; never weather samples",
            "headline": "arithmetic mean of per-seed scores for each initialization and lead",
            "parameter_averaging": False,
            "prediction_averaging": False,
        },
        "evaluation": {
            "methods": list(method_order),
            "thresholds_mm_day": list(base.THRESHOLDS_MM_DAY),
            "central_coverages": list(base.COVERAGES),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_block_length_initializations": 13,
            "support_cells": int(np.count_nonzero(observations.weights > 0.0)),
            "scoring_support_artifact": "evaluation/scoring_support.npz",
            "scoring_support_sha256": sha256_file(scoring_support_path),
            "area_weighting": "India cell area x IMD observation fraction",
            "seasons": ["DJF", "MAM", "JJA", "SON"],
        },
        "cache": cache_provenance,
        "observation_stores": list(observations.source_stores),
        "normalization": {
            "fit_indices": train_indices.tolist(),
            "climatology_log1p_mean_by_lead": context.climatology_mean_by_lead.tolist(),
            "climatology_log1p_std_by_lead": context.climatology_std_by_lead.tolist(),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot_checksums,
    }
    manifest["artifact_sha256"] = output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen all-season CRPS--MSE hybrid-loss ablation."
    )
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--profiles", default=",".join(PROFILE_NAMES))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--member-subsample", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds is None:
        args.seeds = "42" if args.smoke else "42,43,44"
    if args.max_epochs is None:
        args.max_epochs = 2 if args.smoke else 100
    if args.patience is None:
        args.patience = 1 if args.smoke else 15
    profiles = base._parse_names(args.profiles, PROFILE_NAMES, "loss profiles")
    seeds = base._parse_seeds(args.seeds)
    if profiles != PROFILE_NAMES:
        raise ValueError(f"canonical loss profiles must be {PROFILE_NAMES} in order")
    expected_seeds = (42,) if args.smoke else base.SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"canonical {'smoke' if args.smoke else 'full'} seeds must be {expected_seeds}"
        )
    positive = (
        "max_epochs",
        "patience",
        "batch_size",
        "member_subsample",
        "evaluation_batch_size",
        "bootstrap_samples",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.member_subsample > 51:
        raise ValueError("--member-subsample cannot exceed 51")
    fixed = {
        "max_epochs": (args.max_epochs, 2 if args.smoke else 100),
        "patience": (args.patience, 1 if args.smoke else 15),
        "batch_size": (args.batch_size, 8),
        "member_subsample": (args.member_subsample, 16),
        "bootstrap_samples": (args.bootstrap_samples, 2000),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical run settings differ: {mismatch}")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --learning-rate 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --weight-decay 0.0001")
    if args.no_amp:
        raise ValueError("canonical GPU run requires automatic mixed precision")
    if args.device == "cpu":
        raise ValueError("canonical run requires CUDA")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested_output = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {requested_output}")
    staging = requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
        os.replace(staging, requested_output)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "requested_output": str(requested_output),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} run at {requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
