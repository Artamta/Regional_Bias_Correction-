#!/usr/bin/env python3
"""One-shot all-season physical-context screen for the neural calibrator.

The accepted seven-channel location--spread adapter is compared with a
same-width all-zero control and one frozen candidate carrying ten weekly FuXi
physical fields.  Every physical field is standardized independently by lead
week using 2002--2017 training rows and positive IMD area weights only.  Model
core, member transform, CRPS training loss, optimizer, and seed recipe remain
unchanged.

Arm selection uses purged 2018--2019 validation cases only.  After the arm is
locked, its equal-weight seed-CDF pool is benchmarked on the same validation
cases against the immutable accepted base-neural pool, Combined PBC, and
Persistence++.  The driver has no development or sealed-test scoring path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_operational_categorical_comparison as operational_pbc
import fuxi_allseason_pbc_baseline as pbc_driver
import fuxi_allseason_persistence_augmented as persistence_reference
import fuxi_allseason_physical_context_cache as physical_cache
import fuxi_lead_gated_cdf_hybrid_validation as frozen_benchmark
import fuxi_pbc_core as pbc
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator
from fuxi_persistence_context import BASE_CONTEXT_CHANNEL_NAMES
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_physical_context_v1"
SCORING_CONTRACT_VERSION = "normalized_informative_positive_cut_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv2/fuxi_allseason_physical_context"

BASE_ARM = "base_42k"
ZERO_PHYSICAL_ARM = "zero_physical_49k"
PHYSICAL_ARM = "physical_context_49k"
ARMS = (BASE_ARM, ZERO_PHYSICAL_ARM, PHYSICAL_ARM)

PHYSICAL_CONTEXT_FEATURE_NAMES = (
    "t2m_mean",
    "tcwv_mean",
    "q850_mean",
    "u850_mean",
    "v850_mean",
    "q850_u850_flux_mean",
    "q850_v850_flux_mean",
    "z500_mean",
    "msl_mean",
    "olr_mean",
)
PHYSICAL_CONTEXT_CHANNEL_NAMES = (
    *BASE_CONTEXT_CHANNEL_NAMES,
    *PHYSICAL_CONTEXT_FEATURE_NAMES,
)
BASE_CONTEXT_CHANNEL_COUNT = len(BASE_CONTEXT_CHANNEL_NAMES)
PHYSICAL_FEATURE_COUNT = len(PHYSICAL_CONTEXT_FEATURE_NAMES)
PHYSICAL_CONTEXT_CHANNEL_COUNT = len(PHYSICAL_CONTEXT_CHANNEL_NAMES)
LEAD_COUNT = 6
GRID_SHAPE = (27, 27)

EXPECTED_PARAMETER_COUNTS = {
    BASE_ARM: 42_434,
    ZERO_PHYSICAL_ARM: 48_914,
    PHYSICAL_ARM: 48_914,
}
ARM_LABELS = {
    BASE_ARM: "Exact base · 42.4k",
    ZERO_PHYSICAL_ARM: "Zero physical context · 48.9k",
    PHYSICAL_ARM: "Ten-field physical context · 48.9k",
}
ARM_ROLES = {
    BASE_ARM: "exact_architecture_control",
    ZERO_PHYSICAL_ARM: "same_width_zero_input_control_nonselectable",
    PHYSICAL_ARM: "only_promotion_eligible_candidate",
}

QUINTILE_LEVELS = np.asarray(np.arange(0.2, 1.0, 0.2), dtype=np.float32)
CALENDAR_WINDOW_DAYS = 31
MINIMUM_CALENDAR_SAMPLES = 8
CONTROL_REPRODUCTION_RELATIVE_TOLERANCE = 0.0025
MAX_CRPS_WORSENING_FRACTION = 0.005
MIN_MATCHED_SEED_IMPROVEMENTS = 2

SELECTED_POOL_METHOD = "selected_physical_experiment_seed_cdf_pool"
BASE_POOL_METHOD = frozen_benchmark.NEURAL_POOL_METHOD
PERSISTENCE_METHOD = frozen_benchmark.PERSISTENCE_METHOD
COMBINED_PBC_METHOD = frozen_benchmark.COMBINED_METHOD
BENCHMARK_METHODS = (
    SELECTED_POOL_METHOD,
    BASE_POOL_METHOD,
    PERSISTENCE_METHOD,
    COMBINED_PBC_METHOD,
)
BENCHMARK_BASELINES = (
    BASE_POOL_METHOD,
    PERSISTENCE_METHOD,
    COMBINED_PBC_METHOD,
)


class PhysicalContextExperimentError(RuntimeError):
    """Raised when the frozen physical-context experiment contract moves."""


@dataclass(frozen=True)
class PhysicalContextBundle:
    """Raw physical fields plus training-only per-lead normalization state."""

    base_context: base.ContextBundle
    raw_fields: np.ndarray
    mean_by_lead_feature: np.ndarray
    std_by_lead_feature: np.ndarray
    normalization_fit_indices: np.ndarray
    feature_names: tuple[str, ...] = PHYSICAL_CONTEXT_FEATURE_NAMES

    def __post_init__(self) -> None:
        if not isinstance(self.base_context, base.ContextBundle):
            raise PhysicalContextExperimentError(
                "base_context must be the accepted ContextBundle"
            )
        raw = np.asarray(self.raw_fields)
        expected = (
            int(self.base_context.normalized_climatology.shape[0]),
            LEAD_COUNT,
            PHYSICAL_FEATURE_COUNT,
            *GRID_SHAPE,
        )
        if raw.shape != expected or not np.issubdtype(raw.dtype, np.floating):
            raise PhysicalContextExperimentError(
                f"physical fields must have floating shape {expected}, got {raw.shape}"
            )
        if tuple(self.feature_names) != PHYSICAL_CONTEXT_FEATURE_NAMES:
            raise PhysicalContextExperimentError(
                "physical feature names/order differ from the one-shot contract"
            )
        means = np.asarray(self.mean_by_lead_feature)
        stds = np.asarray(self.std_by_lead_feature)
        expected_stats = (LEAD_COUNT, PHYSICAL_FEATURE_COUNT)
        if (
            means.shape != expected_stats
            or stds.shape != expected_stats
            or not np.isfinite(means).all()
            or not np.isfinite(stds).all()
            or np.any(stds <= 0.0)
        ):
            raise PhysicalContextExperimentError(
                "physical normalization statistics are invalid"
            )
        selected = _validate_indices(
            self.normalization_fit_indices, expected[0], "normalization fit indices"
        )
        if not np.isfinite(raw[selected]).all():
            raise PhysicalContextExperimentError(
                "training physical fields contain non-finite values"
            )

    @property
    def case_count(self) -> int:
        return int(self.raw_fields.shape[0])


@dataclass(frozen=True)
class PhysicalTrainingRun:
    arm: str
    seed: int
    context_channels: int
    parameter_count: int
    initial_state_sha256: str
    training_rng_state_sha256: str
    best_epoch: int
    stopped_epoch: int
    stopping_reason: str
    best_validation_crps: float
    elapsed_seconds: float
    checkpoint: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _validate_indices(values: Any, case_count: int, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise PhysicalContextExperimentError(
            f"{label} must be a non-empty one-dimensional integer array"
        )
    selected = raw.astype(np.int64, copy=False)
    if (
        np.any(selected < 0)
        or np.any(selected >= case_count)
        or np.unique(selected).size != selected.size
        or (selected.size > 1 and np.any(np.diff(selected) <= 0))
    ):
        raise PhysicalContextExperimentError(
            f"{label} must be unique, increasing, and in range"
        )
    return selected.copy()


def _validate_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown physical-context arm {arm!r}; expected {ARMS}")
    return arm


def context_channels_for_arm(arm: str) -> int:
    return (
        BASE_CONTEXT_CHANNEL_COUNT
        if _validate_arm(arm) == BASE_ARM
        else PHYSICAL_CONTEXT_CHANNEL_COUNT
    )


def context_channel_names_for_arm(arm: str) -> tuple[str, ...]:
    return (
        tuple(BASE_CONTEXT_CHANNEL_NAMES)
        if _validate_arm(arm) == BASE_ARM
        else PHYSICAL_CONTEXT_CHANNEL_NAMES
    )


def build_physical_context_bundle(
    base_context: base.ContextBundle,
    raw_fields: np.ndarray,
    train_indices: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: Sequence[str] = PHYSICAL_CONTEXT_FEATURE_NAMES,
) -> PhysicalContextBundle:
    """Fit independent feature/lead standardization using training rows only."""

    if not isinstance(base_context, base.ContextBundle):
        raise PhysicalContextExperimentError("base_context has the wrong type")
    raw = np.asarray(raw_fields)
    case_count = int(base_context.normalized_climatology.shape[0])
    expected = (case_count, LEAD_COUNT, PHYSICAL_FEATURE_COUNT, *GRID_SHAPE)
    if raw.shape != expected or not np.issubdtype(raw.dtype, np.floating):
        raise PhysicalContextExperimentError(
            f"raw physical cache has wrong geometry: {raw.shape} != {expected}"
        )
    if tuple(feature_names) != PHYSICAL_CONTEXT_FEATURE_NAMES:
        raise PhysicalContextExperimentError(
            "raw physical cache feature order differs from the frozen order"
        )
    selected = _validate_indices(train_indices, case_count, "training indices")
    spatial_weights = np.asarray(weights, dtype=np.float64)
    support = np.asarray(base_context.support, dtype=bool)
    if (
        spatial_weights.shape != GRID_SHAPE
        or not np.isfinite(spatial_weights).all()
        or np.any(spatial_weights < 0.0)
        or not np.array_equal(spatial_weights > 0.0, support)
    ):
        raise PhysicalContextExperimentError(
            "normalization weights must be finite and match scoring support"
        )
    if not np.isfinite(raw[selected]).all():
        raise PhysicalContextExperimentError(
            "raw training physical fields contain non-finite values"
        )

    means = np.empty((LEAD_COUNT, PHYSICAL_FEATURE_COUNT), dtype=np.float32)
    stds = np.empty_like(means)
    for feature in range(PHYSICAL_FEATURE_COUNT):
        feature_mean, feature_std = base.weighted_lead_moments(
            raw[:, :, feature], selected, spatial_weights
        )
        means[:, feature] = np.asarray(feature_mean, dtype=np.float32)
        stds[:, feature] = np.asarray(feature_std, dtype=np.float32)
    return PhysicalContextBundle(
        base_context=base_context,
        raw_fields=raw_fields,
        mean_by_lead_feature=means,
        std_by_lead_feature=stds,
        normalization_fit_indices=selected,
        feature_names=tuple(feature_names),
    )


def context_for_case(
    bundle: PhysicalContextBundle, case_index: int, arm: str
) -> np.ndarray:
    selected_arm = _validate_arm(arm)
    if not isinstance(bundle, PhysicalContextBundle):
        raise PhysicalContextExperimentError(
            "context must be a PhysicalContextBundle"
        )
    if isinstance(case_index, (bool, np.bool_)) or not isinstance(
        case_index, (int, np.integer)
    ):
        raise IndexError("case index must be an integer")
    index = int(case_index)
    if index < 0 or index >= bundle.case_count:
        raise IndexError("case index is out of range")
    base_fields = base.context_for_case(bundle.base_context, index)
    if selected_arm == BASE_ARM:
        return base_fields
    if selected_arm == ZERO_PHYSICAL_ARM:
        physical = np.zeros(
            (LEAD_COUNT, PHYSICAL_FEATURE_COUNT, *GRID_SHAPE), dtype=np.float32
        )
    else:
        raw = np.asarray(bundle.raw_fields[index], dtype=np.float32)
        physical = (
            raw - bundle.mean_by_lead_feature[:, :, None, None]
        ) / bundle.std_by_lead_feature[:, :, None, None]
        physical = np.asarray(physical, dtype=np.float32)
        if not np.isfinite(physical).all():
            raise PhysicalContextExperimentError(
                "normalized physical context contains non-finite values"
            )
    result = np.concatenate((base_fields, physical), axis=1).astype(
        np.float32, copy=False
    )
    expected = (LEAD_COUNT, PHYSICAL_CONTEXT_CHANNEL_COUNT, *GRID_SHAPE)
    if result.shape != expected or not np.isfinite(result).all():
        raise PhysicalContextExperimentError(
            "expanded physical context has invalid geometry or values"
        )
    return np.ascontiguousarray(result)


class PhysicalCaseDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        members: np.ndarray,
        truth: np.ndarray,
        context: PhysicalContextBundle,
        indices: Sequence[int] | np.ndarray,
        arm: str,
    ) -> None:
        self.arm = _validate_arm(arm)
        self.context = context
        self.indices = _validate_indices(indices, context.case_count, "dataset indices")
        if np.asarray(members).shape[:1] != (context.case_count,) or np.asarray(
            members
        ).shape[2:] != (LEAD_COUNT, *GRID_SHAPE):
            raise PhysicalContextExperimentError("member fields are not aligned")
        if np.asarray(truth).shape != (context.case_count, LEAD_COUNT, *GRID_SHAPE):
            raise PhysicalContextExperimentError("truth fields are not aligned")
        self.members = members
        self.truth = truth

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = int(item)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("dataset item is out of range")
        case = int(self.indices[position])
        return (
            torch.from_numpy(np.array(self.members[case], dtype=np.float32, copy=True)),
            torch.from_numpy(context_for_case(self.context, case, self.arm)),
            torch.from_numpy(np.array(self.truth[case], dtype=np.float32, copy=True)),
        )


def validate_physical_cache_alignment(
    fields: np.ndarray,
    metadata: Mapping[str, Any],
    member_cache: base.MemberCache,
) -> dict[str, Any]:
    """Bind cache feature order and coordinates to the accepted member cache."""

    expected_shape = (
        len(member_cache.initializations),
        LEAD_COUNT,
        PHYSICAL_FEATURE_COUNT,
        *GRID_SHAPE,
    )
    if np.asarray(fields).shape != expected_shape:
        raise PhysicalContextExperimentError(
            f"physical cache shape moved: {np.asarray(fields).shape} != {expected_shape}"
        )
    if tuple(metadata.get("feature_names", ())) != PHYSICAL_CONTEXT_FEATURE_NAMES:
        raise PhysicalContextExperimentError("physical cache feature order moved")
    if tuple(metadata.get("dims", ())) != (
        "init",
        "lead_week",
        "feature",
        "lat",
        "lon",
    ):
        raise PhysicalContextExperimentError("physical cache dimension order moved")
    if str(metadata.get("normalization", "")) not in {
        "none",
        "none_raw_native_values",
    }:
        raise PhysicalContextExperimentError(
            "physical cache must contain raw unnormalized fields"
        )
    starts = np.asarray(metadata.get("initializations", ()), dtype="datetime64[D]")
    latitude = np.asarray(metadata.get("latitude", ()), dtype=np.float64)
    longitude = np.asarray(metadata.get("longitude", ()), dtype=np.float64)
    if not np.array_equal(starts, member_cache.initializations.astype("datetime64[D]")):
        raise PhysicalContextExperimentError(
            "physical cache initializations do not align exactly"
        )
    if not np.array_equal(latitude, member_cache.latitude) or not np.array_equal(
        longitude, member_cache.longitude
    ):
        raise PhysicalContextExperimentError("physical cache grid does not align")
    if any("2025" in str(value) for value in starts):
        raise PhysicalContextExperimentError("sealed 2025 entered physical cache")
    return {
        "shape": list(expected_shape),
        "feature_names": list(PHYSICAL_CONTEXT_FEATURE_NAMES),
        "output_dims": list(metadata["dims"]),
        "normalization_in_cache": str(metadata["normalization"]),
        "source_fingerprint": _json_safe(metadata.get("source_fingerprint")),
        "cache_path": str(metadata.get("data_file", "")),
        "cache_sha256": metadata.get("data_sha256"),
        "feature_contract_sha256": metadata.get("feature_contract_sha256"),
        "alignment_exact": True,
    }


def _new_base_model() -> EnsembleLocationSpreadCalibrator:
    return EnsembleLocationSpreadCalibrator(
        context_channels=BASE_CONTEXT_CHANNEL_COUNT,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
        max_abs_log_spread=2.0,
    )


def _expand_from_base(
    control: EnsembleLocationSpreadCalibrator,
) -> EnsembleLocationSpreadCalibrator:
    """Transplant the exact 42k state and zero the ten new input channels."""

    expanded = EnsembleLocationSpreadCalibrator(
        context_channels=PHYSICAL_CONTEXT_CHANNEL_COUNT,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
        max_abs_log_spread=2.0,
    )
    source = control.state_dict()
    destination = expanded.state_dict()
    widened_name = "backbone.input.weight"
    for name, target in destination.items():
        original = source.get(name)
        if original is None:
            raise PhysicalContextExperimentError(
                f"cannot transplant missing base tensor {name!r}"
            )
        if name == widened_name:
            if (
                target.ndim != 5
                or original.ndim != 5
                or target.shape[0] != original.shape[0]
                or target.shape[1] != original.shape[1] + PHYSICAL_FEATURE_COUNT
                or target.shape[2:] != original.shape[2:]
            ):
                raise PhysicalContextExperimentError(
                    "unexpected widened input-convolution geometry"
                )
            target.zero_()
            target[:, : original.shape[1]].copy_(original)
        elif target.shape == original.shape:
            target.copy_(original)
        else:
            raise PhysicalContextExperimentError(
                f"cannot transplant base tensor {name!r}"
            )
    expanded.load_state_dict(destination, strict=True)
    widened = expanded.state_dict()[widened_name]
    base_width = source[widened_name].shape[1]
    if not torch.equal(
        widened[:, :base_width], source[widened_name]
    ) or bool(torch.count_nonzero(widened[:, base_width:])):
        raise PhysicalContextExperimentError(
            "widened input state is not an exact base/zero transplant"
        )
    return expanded


def build_seeded_model(arm: str, seed: int) -> EnsembleLocationSpreadCalibrator:
    selected_arm = _validate_arm(arm)
    base.set_deterministic_seed(int(seed))
    control = _new_base_model()
    model = control if selected_arm == BASE_ARM else _expand_from_base(control)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNTS[selected_arm]:
        raise PhysicalContextExperimentError(
            f"{selected_arm} has {count:,} parameters; expected "
            f"{EXPECTED_PARAMETER_COUNTS[selected_arm]:,}"
        )
    return model


def validation_crps(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    weights: torch.Tensor,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
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
                loss = base._crps_loss(corrected, truth, weights)
            batch = int(members.shape[0])
            total += float(loss.detach().cpu()) * batch
            count += batch
    if count == 0:
        raise PhysicalContextExperimentError("validation loader is empty")
    return total / count


def train_one_arm(
    arm: str,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: PhysicalContextBundle,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    weights: np.ndarray,
    run_directory: Path,
    *,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    member_subsample: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[pd.DataFrame, PhysicalTrainingRun]:
    """Train one arm with the accepted finite-ensemble CRPS recipe."""

    model = build_seeded_model(arm, seed).to(device)
    initial_state = persistence_reference.model_state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = run_directory / "checkpoints/best.pt"
    train_data = PhysicalCaseDataset(members, truth, context, train_indices, arm)
    validation_data = PhysicalCaseDataset(
        members, truth, context, validation_indices, arm
    )
    train_loader = base.make_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    validation_loader = base.make_loader(
        validation_data,
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
    initial_validation = validation_crps(
        model, validation_loader, spatial_weights, device, use_amp=use_amp
    )
    history: list[dict[str, Any]] = [
        {
            "arm": arm,
            "seed": seed,
            "epoch": 0,
            "train_crps": np.nan,
            "validation_crps": initial_validation,
            "learning_rate": learning_rate,
            "is_best": True,
        }
    ]
    started = time.monotonic()

    # Widened construction consumes extra random draws.  Reset here so every
    # arm starts dropout/member subsampling from one same-seed stream.
    base.set_deterministic_seed(int(seed))
    training_rng_receipt = persistence_reference.training_rng_state_sha256(device)

    def save_checkpoint(epoch: int, score: float) -> None:
        _save_checkpoint_atomic(
            checkpoint_path,
            {
                "experiment": EXPERIMENT,
                "arm": arm,
                "seed": int(seed),
                "epoch": int(epoch),
                "validation_crps": float(score),
                "context_channels": context_channels_for_arm(arm),
                "parameter_count": parameter_count,
                "initial_state_sha256": initial_state,
                "training_rng_state_sha256": training_rng_receipt,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
        )

    best_loss = float(initial_validation)
    best_epoch = 0
    stale_epochs = 0
    stopping_reason = "max_epochs"
    save_checkpoint(0, best_loss)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
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
                loss = base._crps_loss(corrected, batch_truth, spatial_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite CRPS for {arm}, seed {seed}, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(batch_members.shape[0])
            train_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count
        current_validation = validation_crps(
            model, validation_loader, spatial_weights, device, use_amp=use_amp
        )
        scheduler.step(current_validation)
        improved = current_validation < best_loss - 1.0e-6
        if improved:
            best_loss = float(current_validation)
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(epoch, best_loss)
        else:
            stale_epochs += 1
        history.append(
            {
                "arm": arm,
                "seed": seed,
                "epoch": epoch,
                "train_crps": train_sum / train_count,
                "validation_crps": current_validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"[{arm} seed={seed}] epoch={epoch:03d} "
            f"train_CRPS={train_sum / train_count:.6f} "
            f"val_CRPS={current_validation:.6f} best={best_loss:.6f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= patience:
            stopping_reason = "early_stopping_patience"
            break
    record = PhysicalTrainingRun(
        arm=arm,
        seed=int(seed),
        context_channels=context_channels_for_arm(arm),
        parameter_count=parameter_count,
        initial_state_sha256=initial_state,
        training_rng_state_sha256=training_rng_receipt,
        best_epoch=best_epoch,
        stopped_epoch=int(history[-1]["epoch"]),
        stopping_reason=stopping_reason,
        best_validation_crps=best_loss,
        elapsed_seconds=float(time.monotonic() - started),
        checkpoint=str(checkpoint_path),
    )
    del model
    return pd.DataFrame(history), record


def load_checkpoint_model(
    checkpoint_path: Path, arm: str, seed: int, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if (
        checkpoint.get("experiment") != EXPERIMENT
        or checkpoint.get("arm") != arm
        or int(checkpoint.get("seed", -1)) != int(seed)
        or int(checkpoint.get("context_channels", -1))
        != context_channels_for_arm(arm)
        or int(checkpoint.get("parameter_count", -1))
        != EXPECTED_PARAMETER_COUNTS[arm]
    ):
        raise PhysicalContextExperimentError(
            f"checkpoint identity mismatch for {arm}, seed {seed}"
        )
    model = build_seeded_model(arm, seed).to(device)
    if checkpoint.get(
        "initial_state_sha256"
    ) != persistence_reference.model_state_sha256(model):
        raise PhysicalContextExperimentError("checkpoint initial state moved")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(
        bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    ):
        raise PhysicalContextExperimentError("checkpoint contains non-finite tensors")
    model.eval()
    return model


@torch.no_grad()
def predict_adjustments(
    model: torch.nn.Module,
    arm: str,
    members: np.ndarray,
    truth: np.ndarray,
    context: PhysicalContextBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    dataset = PhysicalCaseDataset(members, truth, context, selected, arm)
    loader = base.make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    expected = (len(selected), LEAD_COUNT, *GRID_SHAPE)
    delta = np.empty(expected, dtype=np.float32)
    log_spread = np.empty(expected, dtype=np.float32)
    cursor = 0
    for batch_members, batch_context, _ in loader:
        batch_members = batch_members.to(device, non_blocking=True)
        batch_context = batch_context.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            _, batch_delta, batch_scale = base._call_model(
                model, batch_members, batch_context, member_subsample=None
            )
        count = int(batch_members.shape[0])
        delta[cursor : cursor + count] = batch_delta.float().cpu().numpy()
        log_spread[cursor : cursor + count] = np.log(
            batch_scale.float().cpu().numpy()
        ).astype(np.float32)
        cursor += count
    if cursor != len(selected) or not np.isfinite(delta).all() or not np.isfinite(
        log_spread
    ).all():
        raise PhysicalContextExperimentError("predicted adjustments are invalid")
    return delta, log_spread


def _validate_selection_metrics(
    validation_metrics: pd.DataFrame,
    expected_seeds: Sequence[int],
    expected_years: Sequence[int],
    expected_initializations: Sequence[Any] | None,
) -> pd.DataFrame:
    required = {
        "split",
        "arm",
        "seed",
        "init",
        "year",
        "lead_week",
        "crps",
        "quintile_rps",
        "score_contract",
    }
    missing = sorted(required - set(validation_metrics.columns))
    if missing:
        raise ValueError(f"validation metrics lack columns: {missing}")
    metrics = validation_metrics.copy()
    if set(metrics.split) != {"validation"}:
        raise ValueError("physical selection may use validation rows only")
    if set(metrics.arm) != set(ARMS):
        raise ValueError("validation metrics must contain all three frozen arms")
    if set(metrics.score_contract) != {SCORING_CONTRACT_VERSION}:
        raise ValueError("validation metrics use the wrong score contract")
    numerical = metrics[["crps", "quintile_rps"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numerical).all() or np.any(numerical < 0.0):
        raise ValueError("validation scores must be finite and nonnegative")
    for column in ("seed", "year", "lead_week"):
        values = pd.to_numeric(metrics[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
            raise ValueError(f"validation {column} values must be integers")
        metrics[column] = values.astype(np.int64)
    if set(metrics.lead_week) != set(range(1, LEAD_COUNT + 1)):
        raise ValueError("validation metrics must contain exact leads 1..6")
    parsed = pd.to_datetime(metrics.init, errors="raise", utc=True)
    metrics["init"] = parsed.dt.strftime("%Y-%m-%d")
    if not np.array_equal(parsed.dt.year.to_numpy(), metrics.year.to_numpy()):
        raise ValueError("initialization years disagree with year column")

    seeds = {int(value) for value in expected_seeds}
    years = {int(value) for value in expected_years}
    expected_dates: set[str] | None = None
    if expected_initializations is not None:
        dates = pd.to_datetime(
            pd.Series(np.asarray(expected_initializations)), errors="raise", utc=True
        ).dt.strftime("%Y-%m-%d")
        expected_dates = set(dates)
        if len(expected_dates) != len(dates):
            raise ValueError("expected validation initialization inventory is not unique")
    reference_keys: pd.DataFrame | None = None
    for arm, rows in metrics.groupby("arm"):
        if set(rows.seed) != seeds or set(rows.year) != years:
            raise ValueError(f"arm {arm!r} has wrong seed/year inventory")
        keys = rows[["seed", "init", "year", "lead_week"]].sort_values(
            ["seed", "init", "year", "lead_week"]
        ).reset_index(drop=True)
        if keys.duplicated().any():
            raise ValueError(f"arm {arm!r} contains duplicate rows")
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise ValueError("physical arms do not share identical validation cases")
        for seed in seeds:
            seed_rows = rows.loc[rows.seed == seed]
            lead_counts = seed_rows.groupby("init").lead_week.nunique()
            if not bool((lead_counts == LEAD_COUNT).all()):
                raise ValueError(f"arm {arm!r}, seed {seed} lacks exact leads")
            if expected_dates is not None and set(seed_rows.init) != expected_dates:
                raise ValueError(
                    f"arm {arm!r}, seed {seed} has wrong initialization inventory"
                )
    return metrics


def _skill_fraction(candidate: float, reference: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("selection scores must be finite and references positive")
    return 1.0 - candidate / reference


def select_physical_arm(
    validation_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = base.SEEDS,
    expected_years: Sequence[int] = base.VALIDATION_YEARS,
    expected_initializations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen RPS-led validation-only physical-context selector."""

    metrics = _validate_selection_metrics(
        validation_metrics,
        expected_seeds,
        expected_years,
        expected_initializations,
    )
    seeds = sorted({int(value) for value in expected_seeds})
    years = sorted({int(value) for value in expected_years})
    pooled = metrics.groupby("arm")[["crps", "quintile_rps"]].mean()
    late = metrics.loc[metrics.lead_week >= 2].groupby("arm").quintile_rps.mean()
    by_year = metrics.groupby(["arm", "year"]).quintile_rps.mean()
    by_seed = metrics.groupby(["arm", "seed"]).quintile_rps.mean()

    base_crps = float(pooled.loc[BASE_ARM, "crps"])
    base_rps = float(pooled.loc[BASE_ARM, "quintile_rps"])
    zero_crps = float(pooled.loc[ZERO_PHYSICAL_ARM, "crps"])
    zero_rps = float(pooled.loc[ZERO_PHYSICAL_ARM, "quintile_rps"])
    reproduction = {
        "crps_relative_difference": (zero_crps - base_crps) / base_crps,
        "quintile_rps_relative_difference": (zero_rps - base_rps) / base_rps,
    }
    reproduction["crps_within_tolerance"] = bool(
        abs(reproduction["crps_relative_difference"])
        <= CONTROL_REPRODUCTION_RELATIVE_TOLERANCE + 1.0e-12
    )
    reproduction["quintile_rps_within_tolerance"] = bool(
        abs(reproduction["quintile_rps_relative_difference"])
        <= CONTROL_REPRODUCTION_RELATIVE_TOLERANCE + 1.0e-12
    )
    reproduction["passes"] = bool(
        reproduction["crps_within_tolerance"]
        and reproduction["quintile_rps_within_tolerance"]
    )

    candidate_crps = float(pooled.loc[PHYSICAL_ARM, "crps"])
    candidate_rps = float(pooled.loc[PHYSICAL_ARM, "quintile_rps"])
    candidate_late = float(late.loc[PHYSICAL_ARM])
    seed_required = min(MIN_MATCHED_SEED_IMPROVEMENTS, len(seeds))
    comparisons: dict[str, Any] = {}
    for reference_arm in (ZERO_PHYSICAL_ARM, BASE_ARM):
        reference_crps = float(pooled.loc[reference_arm, "crps"])
        reference_rps = float(pooled.loc[reference_arm, "quintile_rps"])
        reference_late = float(late.loc[reference_arm])
        year_rows: list[dict[str, Any]] = []
        every_year_improves = True
        for year in years:
            candidate_year = float(by_year.loc[(PHYSICAL_ARM, year)])
            reference_year = float(by_year.loc[(reference_arm, year)])
            improves = bool(candidate_year < reference_year - 1.0e-12)
            every_year_improves &= improves
            year_rows.append(
                {
                    "year": year,
                    "candidate_quintile_rps": candidate_year,
                    "reference_quintile_rps": reference_year,
                    "rps_reduction_fraction": _skill_fraction(
                        candidate_year, reference_year
                    ),
                    "improves": improves,
                }
            )
        seed_rows: list[dict[str, Any]] = []
        improved_seeds = 0
        for seed in seeds:
            candidate_seed = float(by_seed.loc[(PHYSICAL_ARM, seed)])
            reference_seed = float(by_seed.loc[(reference_arm, seed)])
            improves = bool(candidate_seed < reference_seed - 1.0e-12)
            improved_seeds += int(improves)
            seed_rows.append(
                {
                    "seed": seed,
                    "candidate_quintile_rps": candidate_seed,
                    "reference_quintile_rps": reference_seed,
                    "rps_reduction_fraction": _skill_fraction(
                        candidate_seed, reference_seed
                    ),
                    "improves": improves,
                }
            )
        record = {
            "reference_arm": reference_arm,
            "candidate_mean_validation_crps": candidate_crps,
            "reference_mean_validation_crps": reference_crps,
            "candidate_mean_validation_quintile_rps": candidate_rps,
            "reference_mean_validation_quintile_rps": reference_rps,
            "candidate_mean_w2_w6_quintile_rps": candidate_late,
            "reference_mean_w2_w6_quintile_rps": reference_late,
            "pooled_rps_reduction_fraction": _skill_fraction(
                candidate_rps, reference_rps
            ),
            "w2_w6_rps_reduction_fraction": _skill_fraction(
                candidate_late, reference_late
            ),
            "pooled_rps_improvement_guard": bool(
                candidate_rps < reference_rps - 1.0e-12
            ),
            "w2_w6_rps_improvement_guard": bool(
                candidate_late < reference_late - 1.0e-12
            ),
            "pooled_crps_max_0p5pct_worsening_guard": bool(
                candidate_crps
                <= reference_crps * (1.0 + MAX_CRPS_WORSENING_FRACTION)
                + 1.0e-12
            ),
            "both_years_rps_improvement_guard": bool(every_year_improves),
            "matched_seed_rps_improvement_passes": improved_seeds,
            "matched_seed_rps_improvement_required": seed_required,
            "matched_seed_rps_guard": bool(improved_seeds >= seed_required),
            "validation_by_year": year_rows,
            "validation_by_seed": seed_rows,
        }
        record["passes"] = bool(
            record["pooled_rps_improvement_guard"]
            and record["w2_w6_rps_improvement_guard"]
            and record["pooled_crps_max_0p5pct_worsening_guard"]
            and record["both_years_rps_improvement_guard"]
            and record["matched_seed_rps_guard"]
        )
        comparisons[reference_arm] = record

    promoted = bool(
        reproduction["passes"]
        and comparisons[ZERO_PHYSICAL_ARM]["passes"]
        and comparisons[BASE_ARM]["passes"]
    )
    selected = PHYSICAL_ARM if promoted else BASE_ARM
    return {
        "status": "validation_selection_locked",
        "selected_arm": selected,
        "candidate_promoted": promoted,
        "reason": (
            "physical context passed every frozen validation guard"
            if promoted
            else "physical context failed at least one frozen validation guard; exact base retained"
        ),
        "test_metrics_consulted": False,
        "zero_physical_selectable": False,
        "post_lock_pbc_benchmark_used_for_selection": False,
        "rules": {
            "primary_metric": SCORING_CONTRACT_VERSION,
            "pooled_rps_must_strictly_improve_vs_base_and_zero": True,
            "w2_w6_rps_must_strictly_improve_vs_base_and_zero": True,
            "maximum_pooled_crps_worsening_fraction": MAX_CRPS_WORSENING_FRACTION,
            "both_validation_years_must_strictly_improve": True,
            "minimum_matched_seed_improvements": seed_required,
            "zero_vs_base_relative_crps_rps_tolerance": (
                CONTROL_REPRODUCTION_RELATIVE_TOLERANCE
            ),
            "parameters_or_predictions_averaged_across_seeds_for_selection": False,
        },
        "zero_physical_control_reproduction": reproduction,
        "candidate_comparisons": comparisons,
        "validation_arm_means": {
            arm: {
                "mean_validation_crps": float(pooled.loc[arm, "crps"]),
                "mean_validation_quintile_rps": float(
                    pooled.loc[arm, "quintile_rps"]
                ),
                "mean_w2_w6_validation_quintile_rps": float(late.loc[arm]),
                "role": ARM_ROLES[arm],
                "parameter_count": EXPECTED_PARAMETER_COUNTS[arm],
            }
            for arm in ARMS
        },
    }


def validation_summary_frames(
    validation_metrics: pd.DataFrame, selection: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {"arm": arm, **values}
            for arm, values in selection["validation_arm_means"].items()
        ]
    )
    by_year = (
        validation_metrics.groupby(["arm", "year"], as_index=False)[
            ["crps", "quintile_rps"]
        ]
        .mean()
        .rename(
            columns={
                "crps": "mean_validation_crps",
                "quintile_rps": "mean_validation_quintile_rps",
            }
        )
    )
    by_seed = (
        validation_metrics.groupby(["arm", "seed"], as_index=False)[
            ["crps", "quintile_rps"]
        ]
        .mean()
        .rename(
            columns={
                "crps": "mean_validation_crps",
                "quintile_rps": "mean_validation_quintile_rps",
            }
        )
    )
    by_lead = (
        validation_metrics.groupby(["arm", "lead_week"], as_index=False)[
            ["crps", "quintile_rps"]
        ]
        .mean()
        .rename(
            columns={
                "crps": "mean_validation_crps",
                "quintile_rps": "mean_validation_quintile_rps",
            }
        )
    )
    return summary, by_year, by_seed, by_lead


def _load_generated_adjustment(
    path: Path,
    arm: str,
    seed: int,
    initializations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        starts = np.asarray(archive["initializations"], dtype="datetime64[D]")
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        stored_arm = str(np.asarray(archive["arm"]).item())
        stored_seed = int(np.asarray(archive["seed"]).item())
    expected = (len(initializations), LEAD_COUNT, *GRID_SHAPE)
    if (
        not np.array_equal(starts, initializations)
        or delta.shape != expected
        or spread.shape != expected
        or stored_arm != arm
        or stored_seed != seed
        or not np.isfinite(delta).all()
        or not np.isfinite(spread).all()
        or np.any(spread <= 0.0)
    ):
        raise PhysicalContextExperimentError(
            f"generated adjustment receipt moved for {arm}, seed {seed}"
        )
    return delta, spread


def reconstruct_generated_seed_pool(
    members: np.ndarray,
    validation_indices: np.ndarray,
    validation_initializations: np.ndarray,
    thresholds: np.ndarray,
    support: np.ndarray,
    adjustment_paths: Mapping[int, Path],
    arm: str,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct and equally pool the locked arm's three validation CDFs."""

    selected = np.asarray(validation_indices, dtype=np.int64)
    expected_cdf = (
        len(selected),
        LEAD_COUNT,
        len(QUINTILE_LEVELS),
        *GRID_SHAPE,
    )
    if thresholds.shape != expected_cdf:
        raise PhysicalContextExperimentError("benchmark thresholds have wrong shape")
    if set(adjustment_paths) != set(base.SEEDS):
        raise PhysicalContextExperimentError(
            "full benchmark requires generated adjustments for seeds 42--44"
        )
    accumulator = np.zeros(expected_cdf, dtype=np.float64)
    validity: dict[str, Any] = {}
    for seed in base.SEEDS:
        delta, spread = _load_generated_adjustment(
            Path(adjustment_paths[seed]),
            arm,
            seed,
            validation_initializations,
        )
        seed_cdf = np.empty(expected_cdf, dtype=np.float32)
        for start in range(0, len(selected), chunk_size):
            stop = min(start + chunk_size, len(selected))
            raw = np.asarray(members[selected[start:stop]], dtype=np.float32)
            corrected = base.apply_affine_log_calibration(
                raw, delta[start:stop], spread[start:stop]
            )
            seed_cdf[start:stop] = pbc.ensemble_cdf(
                corrected, thresholds[start:stop], chunk_size=stop - start
            )
            del raw, corrected
        validity[f"{arm}_seed_{seed}"] = frozen_benchmark.cdf_validity(
            seed_cdf, thresholds, support, f"{arm} seed {seed}"
        )
        accumulator += seed_cdf.astype(np.float64)
        del seed_cdf, delta, spread
        gc.collect()
    pooled = (accumulator / float(len(base.SEEDS))).astype(np.float32)
    validity[SELECTED_POOL_METHOD] = frozen_benchmark.cdf_validity(
        pooled, thresholds, support, SELECTED_POOL_METHOD
    )
    return pooled, validity


def benchmark_case_scores(
    methods: Mapping[str, np.ndarray],
    observed: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    initializations: np.ndarray,
    *,
    selected_arm: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    for method in BENCHMARK_METHODS:
        if method not in methods:
            raise PhysicalContextExperimentError(
                f"post-lock benchmark lacks method {method!r}"
            )
        scores = pbc.weighted_spatial_mean(
            pbc.ranked_probability_score(methods[method], observed, thresholds),
            weights,
        )
        if (
            scores.shape != (len(initializations), LEAD_COUNT)
            or not np.isfinite(scores).all()
            or np.any(scores < 0.0)
        ):
            raise PhysicalContextExperimentError(
                f"invalid post-lock RPS for {method}"
            )
        for case, initialization in enumerate(initializations):
            for lead in range(LEAD_COUNT):
                records.append(
                    {
                        "split": "validation_reused_post_lock",
                        "score_contract": SCORING_CONTRACT_VERSION,
                        "selected_arm": selected_arm,
                        "method": method,
                        "initialization": np.datetime_as_string(
                            initialization, unit="D"
                        ),
                        "year": int(years[case]),
                        "lead_week": lead + 1,
                        "rps": float(scores[case, lead]),
                    }
                )
    result = pd.DataFrame.from_records(records)
    expected_rows = len(BENCHMARK_METHODS) * len(initializations) * LEAD_COUNT
    if len(result) != expected_rows:
        raise PhysicalContextExperimentError("post-lock score table is incomplete")
    return result


def summarize_post_lock_benchmark(
    case_scores: pd.DataFrame,
    initializations: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = (
        case_scores.groupby("method", as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values("method")
    )
    weekwise = (
        case_scores.groupby(["method", "lead_week"], as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values(["method", "lead_week"])
    )
    yearwise = (
        case_scores.groupby(["method", "year"], as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values(["method", "year"])
    )
    per_initialization = case_scores.groupby(
        ["initialization", "method"], as_index=False
    ).agg(rps=("rps", "mean"))
    labels = [np.datetime_as_string(value, unit="D") for value in initializations]
    pivot = per_initialization.pivot(
        index="initialization", columns="method", values="rps"
    ).reindex(index=labels, columns=BENCHMARK_METHODS)
    if pivot.isna().any().any():
        raise PhysicalContextExperimentError(
            "paired post-lock initialization scores are incomplete"
        )
    draws = frozen_benchmark._bootstrap_draws(initializations)
    values = pivot.to_numpy(dtype=np.float64)
    indices = {method: index for index, method in enumerate(BENCHMARK_METHODS)}
    candidate = values[:, indices[SELECTED_POOL_METHOD]]
    records: list[dict[str, Any]] = []
    for baseline_method in BENCHMARK_BASELINES:
        reference = values[:, indices[baseline_method]]
        effects = np.asarray(
            [
                1.0
                - np.mean(candidate[draw], dtype=np.float64)
                / np.mean(reference[draw], dtype=np.float64)
                for draw in draws
            ],
            dtype=np.float64,
        )
        records.append(
            {
                "method": SELECTED_POOL_METHOD,
                "baseline": baseline_method,
                "method_rps": float(candidate.mean()),
                "baseline_rps": float(reference.mean()),
                "rps_reduction_fraction": float(
                    1.0 - candidate.mean() / reference.mean()
                ),
                "ci_lower_95": float(np.quantile(effects, 0.025)),
                "ci_upper_95": float(np.quantile(effects, 0.975)),
                "bootstrap_probability_improvement": float(
                    np.mean(effects > 0.0)
                ),
                "bootstrap_draws": frozen_benchmark.BOOTSTRAP_DRAWS,
                "bootstrap_seed": frozen_benchmark.BOOTSTRAP_SEED,
                "block_length_initializations": (
                    frozen_benchmark.BOOTSTRAP_BLOCK_LENGTH
                ),
                "year_clusters": 2,
            }
        )
    return pooled, weekwise, yearwise, pd.DataFrame.from_records(records)


def post_lock_superiority_decision(
    bootstrap: pd.DataFrame,
    yearwise: pd.DataFrame,
    *,
    candidate_promoted: bool,
    selected_arm: str,
    cdf_checks_pass: bool,
) -> dict[str, Any]:
    expected = {(SELECTED_POOL_METHOD, name) for name in BENCHMARK_BASELINES}
    actual = set(zip(bootstrap.method, bootstrap.baseline, strict=True))
    if actual != expected or len(bootstrap) != len(expected):
        raise ValueError("post-lock benchmark contrast set moved")
    indexed = yearwise.set_index(["method", "year"])
    pooled_gate = bool((bootstrap.rps_reduction_fraction > 0.0).all())
    interval_gate = bool((bootstrap.ci_lower_95 > 0.0).all())
    year_rows: list[dict[str, Any]] = []
    years_gate = True
    for baseline_method in BENCHMARK_BASELINES:
        for year in base.VALIDATION_YEARS:
            candidate = float(indexed.loc[(SELECTED_POOL_METHOD, year), "rps"])
            reference = float(indexed.loc[(baseline_method, year), "rps"])
            improves = bool(candidate < reference - 1.0e-12)
            years_gate &= improves
            year_rows.append(
                {
                    "year": int(year),
                    "baseline": baseline_method,
                    "selected_pool_rps": candidate,
                    "baseline_rps": reference,
                    "rps_reduction_fraction": _skill_fraction(
                        candidate, reference
                    ),
                    "improves": improves,
                }
            )
    physical_claim = bool(
        candidate_promoted
        and selected_arm == PHYSICAL_ARM
        and pooled_gate
        and years_gate
        and interval_gate
        and cdf_checks_pass
    )
    return {
        "status": "physical_candidate_superiority_supported" if physical_claim else "not_supported",
        "physical_candidate_superiority_supported": physical_claim,
        "selected_arm": selected_arm,
        "post_lock_only": True,
        "used_for_arm_selection": False,
        "gates": {
            "physical_candidate_passed_predeclared_selection": bool(
                candidate_promoted and selected_arm == PHYSICAL_ARM
            ),
            "pooled_rps_lower_than_all_three_references": pooled_gate,
            "rps_lower_in_both_years_than_all_three_references": bool(years_gate),
            "paired_95_percent_intervals_above_zero_vs_all_three": interval_gate,
            "all_cdf_checks_pass": bool(cdf_checks_pass),
        },
        "pooled_comparisons": bootstrap.to_dict(orient="records"),
        "yearwise_comparisons": year_rows,
        "scientific_status": (
            "reused 2018-2019 validation benchmark; not independent superiority"
        ),
        "independent_superiority_claim_permitted": False,
        "sealed_2025_target_opened": False,
    }


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_physical_context.py": Path(__file__).resolve(),
        "src/fuxi_allseason_physical_context_cache.py": Path(
            physical_cache.__file__
        ).resolve(),
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "src/fuxi_physical_feature_cache.py": PROJECT_ROOT
        / "src/fuxi_physical_feature_cache.py",
        "src/fuxi_allseason_ensemble_calibration.py": Path(base.__file__).resolve(),
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_persistence_context.py": PROJECT_ROOT
        / "src/fuxi_persistence_context.py",
        "src/fuxi_allseason_persistence_augmented.py": Path(
            persistence_reference.__file__
        ).resolve(),
        "src/fuxi_lead_gated_cdf_hybrid_validation.py": Path(
            frozen_benchmark.__file__
        ).resolve(),
        "src/fuxi_allseason_operational_categorical_comparison.py": Path(
            operational_pbc.__file__
        ).resolve(),
        "src/fuxi_allseason_pbc_baseline.py": Path(pbc_driver.__file__).resolve(),
        "src/fuxi_pbc_core.py": Path(pbc.__file__).resolve(),
        "tests/test_allseason_physical_context.py": PROJECT_ROOT
        / "tests/test_allseason_physical_context.py",
        "tests/test_allseason_physical_context_cache.py": PROJECT_ROOT
        / "tests/test_allseason_physical_context_cache.py",
        "plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md": PROJECT_ROOT
        / "plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md",
        "slurm/build_fuxi_allseason_physical_context_cache.sbatch": PROJECT_ROOT
        / "slurm/build_fuxi_allseason_physical_context_cache.sbatch",
        "slurm/run_allseason_physical_context.sbatch": PROJECT_ROOT
        / "slurm/run_allseason_physical_context.sbatch",
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "failure.json"}
    }


def _load_physical_cache(
    path: Path, expected_source_fingerprint: str | None
) -> tuple[np.ndarray, Mapping[str, Any]]:
    fields, metadata = physical_cache.load_physical_context_cache(
        output=path,
        verify=True,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    if not isinstance(metadata, Mapping):
        raise PhysicalContextExperimentError("physical cache metadata is not a mapping")
    return fields, metadata


def _post_lock_benchmark(
    output: Path,
    selection: Mapping[str, Any],
    cache: base.MemberCache,
    splits: base.SplitIndices,
    observations: base.ObservationBundle,
    adjustment_paths: Mapping[tuple[str, int], Path],
    *,
    cdf_chunk_size: int,
) -> Mapping[str, Any]:
    """Benchmark the already-locked arm without feeding scores to selection."""

    frozen_inputs = frozen_benchmark.validate_frozen_inputs()
    validation = np.asarray(splits.validation, dtype=np.int64)
    starts = np.asarray(cache.initializations[validation], dtype="datetime64[D]")
    if len(starts) != frozen_benchmark.EXPECTED_CASES:
        raise PhysicalContextExperimentError("full validation inventory moved")

    daily_dates, daily_values = pbc_driver.load_daily_imd_for_lags(
        cache, observations.source_stores
    )
    lags = pbc.build_daily_issue_time_lags(
        cache.initializations, daily_dates, daily_values, lag_weeks=(1, 2)
    )
    del daily_dates, daily_values
    if not np.all(lags.available[validation]):
        raise PhysicalContextExperimentError("validation persistence lags are incomplete")

    latitude, longitude, support, weights, support_digest = (
        operational_pbc.load_pbc_static_support(
            frozen_benchmark.PBC_ROOT, frozen_inputs["pbc_manifest"]
        )
    )
    if (
        support_digest != frozen_benchmark.PBC_SUPPORT_SHA256
        or not np.array_equal(latitude, cache.latitude)
        or not np.array_equal(longitude, cache.longitude)
        or not np.array_equal(weights, observations.weights)
        or not np.array_equal(support, observations.weights > 0.0)
    ):
        raise PhysicalContextExperimentError("PBC/neural scoring support moved")
    frozen_pbc = operational_pbc.load_frozen_pbc(
        frozen_benchmark.PBC_ROOT,
        frozen_inputs["pbc_manifest"],
        support,
        cache.initializations[splits.train],
    )
    family = frozen_pbc.quintile
    thresholds = pbc.calendar_fields(family.model, starts, LEAD_COUNT)
    empirical = pbc.calendar_fields(
        family.model, starts, LEAD_COUNT, empirical=True
    )
    raw_cdf = pbc.ensemble_cdf(
        cache.members,
        thresholds,
        chunk_size=cdf_chunk_size,
        case_indices=validation,
    )
    truth = np.asarray(observations.weekly_truth[validation], dtype=np.float32)
    observed = pbc.observation_cdf(truth, thresholds)
    lag_cdf = pbc.lag_observation_cdf(lags, family.model, validation)
    pbc_methods = operational_pbc.apply_frozen_pbc(
        raw_cdf,
        family,
        starts,
        thresholds,
        empirical,
        lag_cdf,
    )
    selected_arm = str(selection["selected_arm"])
    selected_pool, validity = reconstruct_generated_seed_pool(
        cache.members,
        validation,
        starts,
        thresholds,
        support,
        {
            seed: adjustment_paths[(selected_arm, seed)]
            for seed in base.SEEDS
        },
        selected_arm,
        chunk_size=cdf_chunk_size,
    )
    base_pool, base_validity = frozen_benchmark.reconstruct_neural_pool(
        cache.members,
        validation,
        starts,
        thresholds,
        support,
        chunk_size=cdf_chunk_size,
    )
    validity.update(base_validity)
    methods = {
        SELECTED_POOL_METHOD: selected_pool,
        BASE_POOL_METHOD: base_pool,
        PERSISTENCE_METHOD: pbc_methods[PERSISTENCE_METHOD],
        COMBINED_PBC_METHOD: pbc_methods[COMBINED_PBC_METHOD],
    }
    validity["observed_cdf"] = frozen_benchmark.cdf_validity(
        observed, thresholds, support, "observed"
    )
    for method, values in methods.items():
        validity[method] = frozen_benchmark.cdf_validity(
            values, thresholds, support, method
        )
    cdf_checks_pass = all(
        all(value for key, value in evidence.items() if key != "shape")
        for evidence in validity.values()
    )
    scores = benchmark_case_scores(
        methods,
        observed,
        thresholds,
        weights,
        starts,
        selected_arm=selected_arm,
    )
    pooled, weekwise, yearwise, bootstrap = summarize_post_lock_benchmark(
        scores, starts
    )
    decision = post_lock_superiority_decision(
        bootstrap,
        yearwise,
        candidate_promoted=bool(selection["candidate_promoted"]),
        selected_arm=selected_arm,
        cdf_checks_pass=cdf_checks_pass,
    )
    metrics = output / "metrics"
    evaluation = output / "evaluation"
    scores.to_csv(metrics / "post_lock_validation_case_rps.csv", index=False)
    pooled.to_csv(metrics / "post_lock_validation_pooled_rps.csv", index=False)
    weekwise.to_csv(metrics / "post_lock_validation_weekwise_rps.csv", index=False)
    yearwise.to_csv(metrics / "post_lock_validation_yearwise_rps.csv", index=False)
    bootstrap.to_csv(metrics / "post_lock_paired_bootstrap.csv", index=False)
    write_json(evaluation / "post_lock_cdf_validity.json", validity)
    write_json(output / "post_lock_superiority.json", decision)
    return {
        "status": "complete",
        "selection_contamination": False,
        "selected_arm": selected_arm,
        "superiority": decision,
        "accepted_pbc_manifest": str(frozen_benchmark.PBC_MANIFEST_PATH),
        "accepted_pbc_manifest_sha256": frozen_benchmark.PBC_MANIFEST_SHA256,
        "accepted_base_manifest": str(frozen_benchmark.NEURAL_MANIFEST_PATH),
        "accepted_base_manifest_sha256": frozen_benchmark.NEURAL_MANIFEST_SHA256,
    }


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    started = time.monotonic()
    snapshot = source_snapshot(output)
    arms = base._parse_names(args.arms, ARMS, "physical-context arms")
    seeds = base._parse_seeds(args.seeds)
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)

    cache = base.load_member_cache(Path(args.cache), allow_partial=False)
    provenance = base.cache_provenance(cache)
    if (
        provenance.get("source_fingerprint")
        != persistence_reference.EXPECTED_SOURCE_FINGERPRINT
        or provenance.get("data_sha256")
        != persistence_reference.EXPECTED_CACHE_SHA256
    ):
        raise PhysicalContextExperimentError("canonical member cache is required")
    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(values) for name, values in splits.as_dict().items()}
    if split_counts != base.EXPECTED_COUNTS:
        raise PhysicalContextExperimentError("canonical archive split moved")
    train_indices = np.asarray(splits.train, dtype=np.int64)
    validation_indices = np.asarray(splits.validation, dtype=np.int64)
    if args.smoke:
        train_indices = base.select_evenly(train_indices, 32)
        validation_indices = base.select_evenly(validation_indices, 16)

    observations = base.load_imd_observations(cache)
    if any("2025" in str(path) for path in observations.source_stores):
        raise PhysicalContextExperimentError("observation loader opened sealed 2025")
    fields, physical_metadata = _load_physical_cache(
        Path(args.physical_cache), args.expected_physical_source_fingerprint
    )
    physical_provenance = validate_physical_cache_alignment(
        fields, physical_metadata, cache
    )
    base_context = base.build_context_bundle(cache, observations, train_indices)
    context = build_physical_context_bundle(
        base_context,
        fields,
        train_indices,
        observations.weights,
        feature_names=tuple(physical_metadata["feature_names"]),
    )

    evaluation = output / "evaluation"
    metrics = output / "metrics"
    history_directory = output / "history"
    models = output / "models"
    for directory in (evaluation, metrics, history_directory, models):
        directory.mkdir(parents=True, exist_ok=True)
    normalization_path = evaluation / "physical_context_normalization.npz"
    np.savez_compressed(
        normalization_path,
        feature_names=np.asarray(PHYSICAL_CONTEXT_FEATURE_NAMES),
        mean_by_lead_feature=context.mean_by_lead_feature,
        std_by_lead_feature=context.std_by_lead_feature,
        normalization_fit_indices=context.normalization_fit_indices,
        normalization=np.asarray(
            "independent mean/std by feature and lead; training rows and positive IMD weights only"
        ),
    )
    write_json(evaluation / "physical_cache_provenance.json", physical_provenance)

    support = observations.weights > 0.0
    quantiles = pbc.fit_calendar_quantiles(
        observations.weekly_truth,
        cache.initializations,
        train_indices,
        QUINTILE_LEVELS,
        support,
        window_radius_days=(CALENDAR_WINDOW_DAYS - 1) // 2,
        minimum_samples=MINIMUM_CALENDAR_SAMPLES,
    )
    validation_starts = cache.initializations[validation_indices]
    thresholds = pbc.calendar_fields(quantiles, validation_starts, LEAD_COUNT)
    validation_truth = np.asarray(
        observations.weekly_truth[validation_indices], dtype=np.float32
    )
    observed_cdf = pbc.observation_cdf(validation_truth, thresholds)

    histories: list[pd.DataFrame] = []
    runs: list[PhysicalTrainingRun] = []
    validation_frames: list[pd.DataFrame] = []
    adjustment_paths: dict[tuple[str, int], Path] = {}
    initial_receipts: dict[tuple[str, int], str] = {}
    rng_receipts: dict[tuple[str, int], str] = {}
    for arm in arms:
        for seed in seeds:
            print(f"Training {arm}, seed {seed}...", flush=True)
            run_directory = models / arm / f"seed_{seed}"
            history, record = train_one_arm(
                arm,
                seed,
                cache.members,
                observations.weekly_truth,
                context,
                train_indices,
                validation_indices,
                observations.weights,
                run_directory,
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
            runs.append(record)
            initial_receipts[(arm, seed)] = record.initial_state_sha256
            rng_receipts[(arm, seed)] = record.training_rng_state_sha256
            model = load_checkpoint_model(Path(record.checkpoint), arm, seed, device)
            delta, log_spread = predict_adjustments(
                model,
                arm,
                cache.members,
                observations.weekly_truth,
                context,
                validation_indices,
                device=device,
                batch_size=args.evaluation_batch_size,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            adjustment_path = run_directory / "validation_adjustments.npz"
            np.savez_compressed(
                adjustment_path,
                initializations=validation_starts,
                delta_log_location=delta,
                log_spread=log_spread,
                spread_factor=np.exp(np.clip(log_spread, -2.0, 2.0)).astype(
                    np.float32
                ),
                arm=np.asarray(arm),
                seed=np.int64(seed),
            )
            adjustment_paths[(arm, seed)] = adjustment_path
            validation_frames.append(
                persistence_reference.validation_case_metrics(
                    arm,
                    seed,
                    cache.members,
                    observations.weekly_truth,
                    cache.initializations,
                    validation_indices,
                    delta,
                    log_spread,
                    observations.weights,
                    thresholds,
                    observed_cdf,
                    chunk_size=args.cdf_chunk_size,
                )
            )
            del model, delta, log_spread
            torch.cuda.empty_cache()
    for seed in seeds:
        if initial_receipts[(ZERO_PHYSICAL_ARM, seed)] != initial_receipts[
            (PHYSICAL_ARM, seed)
        ]:
            raise PhysicalContextExperimentError(
                f"wide arms do not share initial state for seed {seed}"
            )
        if len({rng_receipts[(arm, seed)] for arm in ARMS}) != 1:
            raise PhysicalContextExperimentError(
                f"arms do not share training RNG for seed {seed}"
            )

    history_frame = pd.concat(histories, ignore_index=True)
    validation_frame = pd.concat(validation_frames, ignore_index=True)
    selection = select_physical_arm(
        validation_frame,
        expected_seeds=seeds,
        expected_initializations=validation_starts,
    )
    selection["written_utc"] = utc_now()
    selection["scientific_selection"] = not args.smoke
    summary, by_year, by_seed, by_lead = validation_summary_frames(
        validation_frame, selection
    )
    history_frame.to_csv(history_directory / "training_history.csv", index=False)
    validation_frame.to_csv(metrics / "validation_case_metrics.csv", index=False)
    summary.to_csv(metrics / "validation_summary.csv", index=False)
    by_year.to_csv(metrics / "validation_by_year.csv", index=False)
    by_seed.to_csv(metrics / "validation_by_seed.csv", index=False)
    by_lead.to_csv(metrics / "validation_by_lead.csv", index=False)
    write_json(output / "selection.json", selection)

    post_lock: Mapping[str, Any]
    if args.smoke:
        post_lock = {
            "status": "skipped_non_scientific_smoke",
            "selection_contamination": False,
        }
    else:
        post_lock = _post_lock_benchmark(
            output,
            selection,
            cache,
            splits,
            observations,
            adjustment_paths,
            cdf_chunk_size=args.cdf_chunk_size,
        )

    run_records = []
    for record in runs:
        values = asdict(record)
        checkpoint = Path(record.checkpoint)
        adjustment = checkpoint.parent.parent / "validation_adjustments.npz"
        values.update(
            {
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_adjustment": str(adjustment.relative_to(output)),
                "validation_adjustment_sha256": sha256_file(adjustment),
            }
        )
        run_records.append(values)
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started),
        "output_path": str(Path(args.output).resolve()),
        "contract": {
            "forecast": "FuXi native weekly TP plus ten frozen physical fields; all seasons; 51 members",
            "target": "IMD weekly mean precipitation, mm day-1",
            "train_years": list(base.TRAIN_YEARS),
            "validation_years": list(base.VALIDATION_YEARS),
            "selection_data": "2018-2019 validation only",
            "development_years_not_scored": list(base.TEST_YEARS),
            "test_metrics_consulted": False,
            "sealed_2025_target_opened": False,
            "single_changed_factor": "ten train-normalized physical context channels",
            "added_channel_order": list(PHYSICAL_CONTEXT_FEATURE_NAMES),
            "normalization": "independent feature/lead mean/std from training rows and positive IMD weights",
            "core_member_transform_loss_optimizer_unchanged": True,
            "post_lock_pbc_benchmark_used_for_selection": False,
        },
        "arms": [
            {
                "name": arm,
                "role": ARM_ROLES[arm],
                "context_channels": context_channels_for_arm(arm),
                "parameter_count": EXPECTED_PARAMETER_COUNTS[arm],
                "selectable": arm != ZERO_PHYSICAL_ARM,
            }
            for arm in arms
        ],
        "seeds": list(seeds),
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train_indices),
            "validation": len(validation_indices),
        },
        "selection": selection,
        "post_lock_benchmark": post_lock,
        "training": {
            "batch_size": args.batch_size,
            "member_subsample": args.member_subsample,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "automatic_mixed_precision": not args.no_amp,
            "runs": run_records,
        },
        "physical_cache": physical_provenance,
        "normalization_artifact": str(normalization_path.relative_to(output)),
        "normalization_artifact_sha256": sha256_file(normalization_path),
        "member_cache": provenance,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": str(getattr(base.matplotlib, "__version__", "unknown")),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot,
    }
    manifest["artifact_sha256"] = output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-shot all-season neural physical-context screen."
    )
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument(
        "--physical-cache", type=Path, default=physical_cache.DEFAULT_CACHE
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--member-subsample", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--cdf-chunk-size", type=int, default=8)
    parser.add_argument("--expected-physical-source-fingerprint", default=None)
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
    arms = base._parse_names(args.arms, ARMS, "physical-context arms")
    seeds = base._parse_seeds(args.seeds)
    if arms != ARMS:
        raise ValueError(f"canonical arms must be {ARMS} in order")
    expected_seeds = (42,) if args.smoke else base.SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"canonical {'smoke' if args.smoke else 'full'} seeds must be {expected_seeds}"
        )
    fixed = {
        "max_epochs": (args.max_epochs, 2 if args.smoke else 100),
        "patience": (args.patience, 1 if args.smoke else 15),
        "batch_size": (args.batch_size, 8),
        "member_subsample": (args.member_subsample, 16),
        "evaluation_batch_size": (args.evaluation_batch_size, 8),
        "cdf_chunk_size": (args.cdf_chunk_size, 8),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical run settings differ: {mismatch}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --learning-rate 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --weight-decay 0.0001")
    if args.no_amp or args.device == "cpu":
        raise ValueError("canonical run requires CUDA automatic mixed precision")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested = (default_output() if args.output is None else Path(args.output)).resolve()
    args.output = requested
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {requested}")
    staging = requested.parent / f".{requested.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
        os.replace(staging, requested)
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
                "requested_output": str(requested),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} physical-context run at {requested}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
