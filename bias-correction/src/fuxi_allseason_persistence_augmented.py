#!/usr/bin/env python3
"""Validation-only persistence-feature ablation for the all-season adapter.

The experiment preserves the accepted seven-channel ``base_42k`` model and
compares it with a parameter-matched eleven-channel zero-input control and an
eleven-channel model carrying two completed, issuance-time IMD rainfall lags
plus their availability masks.  Models are fitted on purged 2002--2017 cases;
checkpoint and arm selection use purged 2018--2019 validation cases only.
This driver deliberately does not score the reused 2020--2021 development
period and has no route to the sealed 2025 target.
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

import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_pbc_baseline as pbc_driver
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator
from fuxi_pbc_core import (
    build_daily_issue_time_lags,
    calendar_fields,
    ensemble_cdf,
    fit_calendar_quantiles,
    is_valid_cdf_for_thresholds,
    observation_cdf,
    ranked_probability_score,
    weighted_spatial_mean,
)
from fuxi_persistence_context import (
    ARMS,
    BASE_ARM,
    LAG_CONTEXT_CHANNEL_NAMES,
    PERSISTENCE_LAG_ARM,
    ZERO_LAG_ARM,
    PersistenceCaseDataset,
    PersistenceContextBundle,
    build_persistence_context_bundle,
    context_channels_for_arm,
)
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_persistence_augmented_v1"
SCORING_CONTRACT_VERSION = "normalized_informative_positive_cut_v2"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_persistence_augmented"
)
PLAN_PATH = PROJECT_ROOT / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
SLURM_PATH = PROJECT_ROOT / "slurm/run_allseason_persistence_augmented.sbatch"
HELPER_PATH = PROJECT_ROOT / "src/fuxi_persistence_context.py"
EXPECTED_CACHE_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
)

QUINTILE_LEVELS = np.asarray(
    np.arange(0.2, 1.0, 0.2, dtype=np.float64), dtype=np.float32
)
CALENDAR_WINDOW_DAYS = 31
MINIMUM_CALENDAR_SAMPLES = 8
MIN_RPS_SKILL_PCT = 0.5
CONTROL_REPRODUCTION_RELATIVE_TOLERANCE = 0.0025
MIN_MATCHED_SEED_IMPROVEMENTS = 2
EXPECTED_PARAMETER_COUNTS = {
    BASE_ARM: 42_434,
    ZERO_LAG_ARM: 45_026,
    PERSISTENCE_LAG_ARM: 45_026,
}
ARM_LABELS = {
    BASE_ARM: "Exact base · 42.4k",
    ZERO_LAG_ARM: "Zero-lag control · 45.0k",
    PERSISTENCE_LAG_ARM: "Persistence lag-1/lag-2 · 45.0k",
}
ARM_ROLES = {
    BASE_ARM: "exact_architecture_control",
    ZERO_LAG_ARM: "parameter_input_shape_control_nonselectable",
    PERSISTENCE_LAG_ARM: "only_promotion_eligible_candidate",
}
ARM_COLORS = {
    BASE_ARM: "#009E73",
    ZERO_LAG_ARM: "#7F7F7F",
    PERSISTENCE_LAG_ARM: "#0072B2",
}


class PersistenceExperimentError(RuntimeError):
    """Raised when the frozen persistence experiment contract is violated."""


@dataclass(frozen=True)
class PersistenceTrainingRun:
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
    """Atomically replace JSON inside an experiment-owned staging directory."""

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


def model_state_sha256(model: torch.nn.Module) -> str:
    """Content-bind a model state independently of checkpoint metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(values.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(memoryview(values.view(np.uint8)))
    return digest.hexdigest()


def training_rng_state_sha256(device: torch.device) -> str:
    """Bind the reset stochastic-training stream without advancing it."""

    digest = hashlib.sha256()
    cpu_state = torch.get_rng_state().cpu().contiguous().numpy()
    digest.update(b"torch_cpu\0")
    digest.update(memoryview(cpu_state.view(np.uint8)))
    if device.type == "cuda":
        digest.update(b"torch_cuda\0")
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            values = state.cpu().contiguous().numpy()
            digest.update(str(index).encode("ascii") + b"\0")
            digest.update(memoryview(values.view(np.uint8)))
    return digest.hexdigest()


def _new_base_model() -> EnsembleLocationSpreadCalibrator:
    return EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
        max_abs_log_spread=2.0,
    )


def _expand_from_base(
    control: EnsembleLocationSpreadCalibrator,
) -> EnsembleLocationSpreadCalibrator:
    """Copy the 42k state into an 11-channel model and zero new input weights."""

    expanded = EnsembleLocationSpreadCalibrator(
        context_channels=11,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
        max_abs_log_spread=2.0,
    )
    source = control.state_dict()
    destination = expanded.state_dict()
    changed_shape = "backbone.input.weight"
    for name, target in destination.items():
        if name == changed_shape:
            original = source[name]
            if (
                target.ndim != 5
                or original.ndim != 5
                or target.shape[0] != original.shape[0]
                or target.shape[1] != original.shape[1] + 4
                or target.shape[2:] != original.shape[2:]
            ):
                raise PersistenceExperimentError(
                    "unexpected widened input-convolution geometry"
                )
            target.zero_()
            target[:, : original.shape[1]].copy_(original)
        else:
            original = source.get(name)
            if original is None or original.shape != target.shape:
                raise PersistenceExperimentError(
                    f"cannot transplant base state tensor {name!r}"
                )
            target.copy_(original)
    expanded.load_state_dict(destination, strict=True)
    widened = expanded.state_dict()[changed_shape]
    original_width = source[changed_shape].shape[1]
    if not torch.equal(widened[:, :original_width], source[changed_shape]) or bool(
        torch.count_nonzero(widened[:, original_width:])
    ):
        raise PersistenceExperimentError("widened input state is not exact/zero")
    return expanded


def build_seeded_model(arm: str, seed: int) -> EnsembleLocationSpreadCalibrator:
    """Build one frozen arm with same-seed common tensors copied exactly."""

    if arm not in ARMS:
        raise ValueError(f"unknown persistence arm {arm!r}")
    base.set_deterministic_seed(int(seed))
    control = _new_base_model()
    model = control if arm == BASE_ARM else _expand_from_base(control)
    count = sum(parameter.numel() for parameter in model.parameters())
    expected = EXPECTED_PARAMETER_COUNTS[arm]
    if count != expected:
        raise PersistenceExperimentError(
            f"{arm} has {count:,} parameters; frozen contract expects {expected:,}"
        )
    return model


@torch.no_grad()
def validation_crps(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    weights: torch.Tensor,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.eval()
    numerator = 0.0
    count = 0
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
        batch_count = int(members.shape[0])
        numerator += float(loss.detach().cpu()) * batch_count
        count += batch_count
    if count == 0:
        raise PersistenceExperimentError("validation loader is empty")
    return numerator / count


def train_one_arm(
    arm: str,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: PersistenceContextBundle,
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
) -> tuple[pd.DataFrame, PersistenceTrainingRun]:
    """Train one arm with the unchanged finite-ensemble CRPS recipe."""

    model = build_seeded_model(arm, seed).to(device)
    initial_state = model_state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = run_directory / "checkpoints" / "best.pt"
    train_data = PersistenceCaseDataset(
        members, truth, context, train_indices, arm
    )
    validation_data = PersistenceCaseDataset(
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
    started = time.monotonic()
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

    # Constructing the widened 11-channel model consumes more random draws than
    # constructing the exact 7-channel base. Reset only after all deterministic
    # setup/evaluation so dropout and member subsampling begin from the same
    # same-seed stream in every arm. The DataLoader has its own same-seed
    # generator, so this reset does not disturb case-order matching.
    base.set_deterministic_seed(int(seed))
    training_rng_receipt = training_rng_state_sha256(device)
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
                    f"non-finite training CRPS for {arm}, seed {seed}, epoch {epoch}"
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
    record = PersistenceTrainingRun(
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
    checkpoint_path: Path,
    arm: str,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    expected_count = EXPECTED_PARAMETER_COUNTS[arm]
    if (
        checkpoint.get("experiment") != EXPERIMENT
        or checkpoint.get("arm") != arm
        or int(checkpoint.get("seed", -1)) != int(seed)
        or int(checkpoint.get("context_channels", -1))
        != context_channels_for_arm(arm)
        or int(checkpoint.get("parameter_count", -1)) != expected_count
    ):
        raise PersistenceExperimentError(
            f"checkpoint identity mismatch for {arm}, seed {seed}"
        )
    model = build_seeded_model(arm, seed).to(device)
    expected_initial = model_state_sha256(model)
    if checkpoint.get("initial_state_sha256") != expected_initial:
        raise PersistenceExperimentError(
            f"checkpoint initial-state receipt mismatch for {arm}, seed {seed}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise PersistenceExperimentError("checkpoint contains non-finite tensors")
    model.eval()
    return model


@torch.no_grad()
def predict_adjustments(
    model: torch.nn.Module,
    arm: str,
    members: np.ndarray,
    truth: np.ndarray,
    context: PersistenceContextBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    dataset = PersistenceCaseDataset(members, truth, context, selected, arm)
    loader = base.make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    expected = (len(selected), 6, 27, 27)
    delta = np.empty(expected, dtype=np.float32)
    log_spread = np.empty(expected, dtype=np.float32)
    cursor = 0
    model.eval()
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
        # The core's third output is spread factor. Persist log spread so the
        # reconstruction contract matches accepted neural artifacts.
        factor = batch_scale.float().cpu().numpy()
        log_spread[cursor : cursor + count] = np.log(factor).astype(np.float32)
        cursor += count
    if cursor != len(selected) or not np.isfinite(delta).all() or not np.isfinite(
        log_spread
    ).all():
        raise PersistenceExperimentError("invalid predicted adjustment fields")
    return delta, log_spread


def validation_case_metrics(
    arm: str,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    validation_indices: np.ndarray,
    delta_log_location: np.ndarray,
    log_spread: np.ndarray,
    weights: np.ndarray,
    quintile_thresholds: np.ndarray,
    quintile_observed_cdf: np.ndarray,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """Score continuous CRPS and common-support quintile RPS on validation."""

    selected = np.asarray(validation_indices, dtype=np.int64)
    starts = np.asarray(initializations, dtype="datetime64[D]")[selected]
    years = pd.DatetimeIndex(starts).year.to_numpy()
    if set(years) != set(base.VALIDATION_YEARS):
        raise PersistenceExperimentError(
            "validation scorer must contain both frozen validation years only"
        )
    expected_adjustment = (len(selected), 6, 27, 27)
    expected_cdf = (len(selected), 6, len(QUINTILE_LEVELS), 27, 27)
    if (
        delta_log_location.shape != expected_adjustment
        or log_spread.shape != expected_adjustment
        or quintile_thresholds.shape != expected_cdf
        or quintile_observed_cdf.shape != expected_cdf
    ):
        raise ValueError("validation adjustments or categorical arrays have wrong shape")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), chunk_size):
        stop = min(start + chunk_size, len(selected))
        source = selected[start:stop]
        raw = np.asarray(members[source], dtype=np.float32)
        target = np.asarray(truth[source], dtype=np.float32)
        spread = np.exp(np.clip(log_spread[start:stop], -2.0, 2.0)).astype(
            np.float32
        )
        corrected = base.apply_affine_log_calibration(
            raw, delta_log_location[start:stop], spread
        )
        crps = base._weighted_field_mean(
            base.numpy_ensemble_crps(corrected, target), weights
        )
        thresholds = quintile_thresholds[start:stop]
        observed = quintile_observed_cdf[start:stop]
        forecast_cdf = ensemble_cdf(
            corrected, thresholds, chunk_size=max(1, stop - start)
        )
        geometry_equal = bool(
            forecast_cdf.shape == observed.shape == thresholds.shape
        )
        defined = np.isfinite(thresholds)
        forecast_finite = bool(np.isfinite(forecast_cdf[defined]).all())
        observed_finite = bool(np.isfinite(observed[defined]).all())
        forecast_valid = bool(
            geometry_equal
            and is_valid_cdf_for_thresholds(forecast_cdf, thresholds, axis=2)
        )
        observed_valid = bool(
            geometry_equal
            and is_valid_cdf_for_thresholds(observed, thresholds, axis=2)
        )
        if not (
            geometry_equal
            and forecast_finite
            and observed_finite
            and forecast_valid
            and observed_valid
        ):
            raise PersistenceExperimentError(
                "validation categorical CDF violates finite/equality geometry"
            )
        rps = weighted_spatial_mean(
            ranked_probability_score(forecast_cdf, observed, thresholds), weights
        )
        if crps.shape != (stop - start, 6) or rps.shape != (stop - start, 6):
            raise PersistenceExperimentError("unexpected validation score shape")
        if not np.isfinite(crps).all() or not np.isfinite(rps).all():
            raise PersistenceExperimentError("validation scores are non-finite")
        for local in range(stop - start):
            position = start + local
            for lead in range(6):
                rows.append(
                    {
                        "split": "validation",
                        "arm": arm,
                        "seed": int(seed),
                        "init": np.datetime_as_string(starts[position], unit="D"),
                        "year": int(years[position]),
                        "lead_week": lead + 1,
                        "crps": float(crps[local, lead]),
                        "quintile_rps": float(rps[local, lead]),
                        "score_contract": SCORING_CONTRACT_VERSION,
                        "cdf_shape_matches_thresholds": geometry_equal,
                        "forecast_cdf_finite_where_threshold_defined": (
                            forecast_finite
                        ),
                        "observed_cdf_finite_where_threshold_defined": (
                            observed_finite
                        ),
                        "forecast_cdf_valid_for_thresholds": forecast_valid,
                        "observed_cdf_valid_for_thresholds": observed_valid,
                    }
                )
        del raw, target, spread, corrected, crps, forecast_cdf, rps
    result = pd.DataFrame(rows)
    if len(result) != len(selected) * 6:
        raise PersistenceExperimentError(
            "validation scoring did not retain every case and lead"
        )
    return result


def _skill_pct(candidate: float, reference: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("selection scores must be finite and references positive")
    return 100.0 * (reference - candidate) / reference


def select_persistence_arm(
    validation_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = base.SEEDS,
    expected_years: Sequence[int] = base.VALIDATION_YEARS,
    expected_initializations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen RPS-led, validation-only three-arm selector."""

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
    validation_metrics = validation_metrics.copy()
    if set(validation_metrics.split) != {"validation"}:
        raise ValueError("persistence selection may use validation rows only")
    if set(validation_metrics.arm) != set(ARMS):
        raise ValueError("validation metrics must contain all three frozen arms")
    if set(validation_metrics.score_contract) != {SCORING_CONTRACT_VERSION}:
        raise ValueError("validation metrics use the wrong categorical score contract")
    numeric = validation_metrics[["crps", "quintile_rps"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
        raise ValueError("validation scores must be finite and nonnegative")
    for column in ("seed", "year", "lead_week"):
        converted = pd.to_numeric(validation_metrics[column], errors="raise")
        if not np.isfinite(converted.to_numpy(dtype=np.float64)).all() or not np.equal(
            converted.to_numpy(dtype=np.float64),
            converted.to_numpy(dtype=np.int64),
        ).all():
            raise ValueError(f"validation {column} values must be finite integers")
        validation_metrics[column] = converted.astype(np.int64)
    required_leads = set(range(1, 7))
    if set(validation_metrics.lead_week) != required_leads:
        raise ValueError("validation metrics must contain exact leads 1..6")
    parsed_initializations = pd.to_datetime(
        validation_metrics.init, errors="raise", utc=True
    )
    if not np.array_equal(
        parsed_initializations.dt.year.to_numpy(dtype=np.int64),
        validation_metrics.year.to_numpy(dtype=np.int64),
    ):
        raise ValueError("validation initialization years disagree with year column")
    validation_metrics["init"] = parsed_initializations.dt.strftime("%Y-%m-%d")

    expected_seed_set = {int(seed) for seed in expected_seeds}
    expected_year_set = {int(year) for year in expected_years}
    if not expected_seed_set or not expected_year_set:
        raise ValueError("selection requires seeds and validation years")
    expected_initialization_set: set[str] | None = None
    if expected_initializations is not None:
        expected_values = np.asarray(expected_initializations)
        if expected_values.ndim != 1 or expected_values.size == 0:
            raise ValueError(
                "expected validation initializations must be a non-empty 1-D inventory"
            )
        expected_dates = pd.to_datetime(
            pd.Series(expected_values), errors="raise", utc=True
        ).dt.strftime("%Y-%m-%d")
        expected_initialization_set = set(expected_dates.tolist())
        if len(expected_initialization_set) != len(expected_dates):
            raise ValueError("expected validation initialization inventory is not unique")
        if set(pd.to_datetime(expected_dates).dt.year) != expected_year_set:
            raise ValueError(
                "expected validation initialization inventory has wrong years"
            )
    keys = ["seed", "init", "year", "lead_week"]
    reference_keys: pd.DataFrame | None = None
    for arm, rows in validation_metrics.groupby("arm"):
        if {int(value) for value in rows.seed} != expected_seed_set:
            raise ValueError(f"arm {arm!r} has the wrong seed set")
        if {int(value) for value in rows.year} != expected_year_set:
            raise ValueError(f"arm {arm!r} has the wrong validation years")
        arm_keys = rows[keys].sort_values(keys).reset_index(drop=True)
        if arm_keys.duplicated().any():
            raise ValueError(f"arm {arm!r} has duplicate validation rows")
        grouped_leads = rows.groupby(
            ["seed", "init", "year"], sort=False
        )["lead_week"]
        incomplete = [
            key
            for key, values in grouped_leads
            if set(values.tolist()) != required_leads or len(values) != 6
        ]
        if incomplete:
            raise ValueError(
                f"arm {arm!r} lacks exact leads 1..6 for "
                f"{len(incomplete)} seed/initialization rows"
            )
        seed_case_reference: pd.DataFrame | None = None
        for seed in sorted(expected_seed_set):
            seed_cases = (
                rows.loc[rows.seed == seed, ["init", "year"]]
                .drop_duplicates()
                .sort_values(["init", "year"])
                .reset_index(drop=True)
            )
            if expected_initialization_set is not None and set(
                seed_cases.init
            ) != expected_initialization_set:
                raise ValueError(
                    f"arm {arm!r}, seed {seed} does not match the expected "
                    "validation initialization inventory"
                )
            if seed_case_reference is None:
                seed_case_reference = seed_cases
            elif not seed_cases.equals(seed_case_reference):
                raise ValueError(
                    f"arm {arm!r} does not share one validation case inventory "
                    "across seeds"
                )
        if reference_keys is None:
            reference_keys = arm_keys
        elif not arm_keys.equals(reference_keys):
            raise ValueError("persistence arms do not share identical validation cases")

    pooled = validation_metrics.groupby("arm")[["crps", "quintile_rps"]].mean()
    late = validation_metrics.loc[validation_metrics.lead_week >= 2].groupby("arm")[
        "quintile_rps"
    ].mean()
    by_year = validation_metrics.groupby(["arm", "year"])["quintile_rps"].mean()
    by_seed = validation_metrics.groupby(["arm", "seed"])["quintile_rps"].mean()

    base_crps = float(pooled.loc[BASE_ARM, "crps"])
    base_rps = float(pooled.loc[BASE_ARM, "quintile_rps"])
    zero_crps = float(pooled.loc[ZERO_LAG_ARM, "crps"])
    zero_rps = float(pooled.loc[ZERO_LAG_ARM, "quintile_rps"])
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

    candidate_crps = float(pooled.loc[PERSISTENCE_LAG_ARM, "crps"])
    candidate_rps = float(pooled.loc[PERSISTENCE_LAG_ARM, "quintile_rps"])
    candidate_late_rps = float(late.loc[PERSISTENCE_LAG_ARM])
    seed_required = min(MIN_MATCHED_SEED_IMPROVEMENTS, len(expected_seed_set))
    comparisons: dict[str, Any] = {}
    for reference_arm in (ZERO_LAG_ARM, BASE_ARM):
        reference_crps = float(pooled.loc[reference_arm, "crps"])
        reference_rps = float(pooled.loc[reference_arm, "quintile_rps"])
        reference_late_rps = float(late.loc[reference_arm])
        year_rows = []
        all_years_noninferior = True
        for year in sorted(expected_year_set):
            candidate_year = float(by_year.loc[(PERSISTENCE_LAG_ARM, year)])
            reference_year = float(by_year.loc[(reference_arm, year)])
            passes = bool(candidate_year <= reference_year + 1.0e-12)
            all_years_noninferior = all_years_noninferior and passes
            year_rows.append(
                {
                    "year": int(year),
                    "candidate_quintile_rps": candidate_year,
                    "reference_quintile_rps": reference_year,
                    "rps_skill_pct": _skill_pct(candidate_year, reference_year),
                    "noninferior": passes,
                }
            )
        seed_rows = []
        improved_seeds = 0
        for seed in sorted(expected_seed_set):
            candidate_seed = float(by_seed.loc[(PERSISTENCE_LAG_ARM, seed)])
            reference_seed = float(by_seed.loc[(reference_arm, seed)])
            improves = bool(candidate_seed < reference_seed - 1.0e-12)
            improved_seeds += int(improves)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "candidate_quintile_rps": candidate_seed,
                    "reference_quintile_rps": reference_seed,
                    "rps_skill_pct": _skill_pct(candidate_seed, reference_seed),
                    "improves": improves,
                }
            )
        pooled_skill = _skill_pct(candidate_rps, reference_rps)
        late_skill = _skill_pct(candidate_late_rps, reference_late_rps)
        record = {
            "reference_arm": reference_arm,
            "candidate_mean_validation_crps": candidate_crps,
            "reference_mean_validation_crps": reference_crps,
            "candidate_mean_validation_quintile_rps": candidate_rps,
            "reference_mean_validation_quintile_rps": reference_rps,
            "candidate_mean_w2_w6_quintile_rps": candidate_late_rps,
            "reference_mean_w2_w6_quintile_rps": reference_late_rps,
            "pooled_rps_skill_pct": pooled_skill,
            "w2_w6_rps_skill_pct": late_skill,
            "pooled_rps_minimum_improvement_guard": bool(
                pooled_skill >= MIN_RPS_SKILL_PCT - 1.0e-12
            ),
            "w2_w6_rps_minimum_improvement_guard": bool(
                late_skill >= MIN_RPS_SKILL_PCT - 1.0e-12
            ),
            "pooled_crps_noninferiority_guard": bool(
                candidate_crps <= reference_crps + 1.0e-12
            ),
            "all_years_rps_noninferiority_guard": bool(all_years_noninferior),
            "matched_seed_rps_improvement_passes": int(improved_seeds),
            "matched_seed_rps_improvement_required": int(seed_required),
            "matched_seed_rps_guard": bool(improved_seeds >= seed_required),
            "validation_by_year": year_rows,
            "validation_by_seed": seed_rows,
        }
        record["passes"] = bool(
            record["pooled_rps_minimum_improvement_guard"]
            and record["w2_w6_rps_minimum_improvement_guard"]
            and record["pooled_crps_noninferiority_guard"]
            and record["all_years_rps_noninferiority_guard"]
            and record["matched_seed_rps_guard"]
        )
        comparisons[reference_arm] = record

    promoted = bool(
        reproduction["passes"]
        and comparisons[ZERO_LAG_ARM]["passes"]
        and comparisons[BASE_ARM]["passes"]
    )
    selected = PERSISTENCE_LAG_ARM if promoted else BASE_ARM
    if promoted:
        reason = (
            "persistence candidate passed every frozen pooled/W2-W6 RPS, CRPS, "
            "year, matched-seed, and control-reproduction guard"
        )
    elif not reproduction["passes"]:
        reason = (
            "zero-lag 45k control did not reproduce exact base within the frozen "
            "0.25% CRPS/RPS tolerance; exact base retained"
        )
    else:
        reason = (
            "persistence candidate failed at least one frozen guard against base "
            "or the parameter-matched zero-lag control; exact base retained"
        )
    return {
        "status": "validation_selection_locked",
        "selected_arm": selected,
        "candidate_promoted": promoted,
        "reason": reason,
        "test_metrics_consulted": False,
        "zero_lag_selectable": False,
        "rules": {
            "primary_metric": SCORING_CONTRACT_VERSION,
            "minimum_pooled_rps_skill_pct_vs_each_control": MIN_RPS_SKILL_PCT,
            "minimum_w2_w6_rps_skill_pct_vs_each_control": MIN_RPS_SKILL_PCT,
            "pooled_crps_ratio_max_vs_each_control": 1.0,
            "yearwise_rps_ratio_max_vs_each_control": 1.0,
            "minimum_matched_seed_rps_improvements_vs_each_control": seed_required,
            "zero_vs_base_absolute_relative_crps_rps_tolerance": (
                CONTROL_REPRODUCTION_RELATIVE_TOLERANCE
            ),
            "required_validation_years": sorted(expected_year_set),
            "required_validation_initialization_count": (
                None
                if expected_initialization_set is None
                else len(expected_initialization_set)
            ),
            "parameters_or_predictions_averaged_across_seeds": False,
        },
        "zero_lag_control_reproduction": reproduction,
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
    validation_metrics: pd.DataFrame,
    selection: Mapping[str, Any],
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


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_persistence_augmented.py": Path(__file__).resolve(),
        "src/fuxi_persistence_context.py": HELPER_PATH,
        "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_pbc_baseline.py": PROJECT_ROOT
        / "src/fuxi_allseason_pbc_baseline.py",
        "src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "slurm/run_allseason_persistence_augmented.sbatch": SLURM_PATH,
        "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": PLAN_PATH,
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"frozen persistence source is missing: {source}")
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


def plot_training_history(history: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    for axis, arm in zip(axes, ARMS, strict=True):
        selected = history.loc[history.arm == arm]
        for seed, rows in selected.groupby("seed"):
            rows = rows.sort_values("epoch")
            trained = rows.loc[rows.epoch > 0]
            axis.plot(trained.epoch, trained.train_crps, color="0.65", linewidth=0.9)
            axis.plot(
                rows.epoch,
                rows.validation_crps,
                color=ARM_COLORS[arm],
                linewidth=1.1,
                label=f"seed {seed}",
            )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("Area-weighted ensemble CRPS")
    figure.suptitle(
        "Persistence-feature training"
        + (" · smoke, non-scientific" if smoke else " · validation only")
    )
    base._save_figure(figure, output)


def plot_validation_by_lead(
    by_lead: pd.DataFrame, output: Path, *, smoke: bool
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for arm in ARMS:
        rows = by_lead.loc[by_lead.arm == arm].sort_values("lead_week")
        axes[0].plot(
            rows.lead_week,
            rows.mean_validation_crps,
            marker="o",
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
        axes[1].plot(
            rows.lead_week,
            rows.mean_validation_quintile_rps,
            marker="o",
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    axes[0].set_ylabel("CRPS")
    axes[1].set_ylabel("Normalized informative-cut RPS")
    for axis in axes:
        axis.set_xlabel("Lead week")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=6)
    figure.suptitle(
        "Validation-only persistence feature ablation"
        + (" · smoke" if smoke else "")
    )
    base._save_figure(figure, output)


def build_readme(selection: Mapping[str, Any], *, smoke: bool) -> str:
    status = "NON-SCIENTIFIC PLUMBING SMOKE" if smoke else "VALIDATION-ONLY SCREEN"
    return "\n".join(
        [
            "# FuXi all-season persistence-feature ablation",
            "",
            f"**Status:** {status}",
            "",
            f"Validation selected `{selection['selected_arm']}`. {selection['reason']}",
            "",
            "The exact seven-channel 42k model is compared with a nonselectable 45k "
            "zero-input control and a matched 45k candidate carrying two completed, "
            "strictly pre-issuance IMD rainfall lags plus availability masks.",
            "",
            "Selection uses 2018–2019 only. This driver computes no 2020–2021 "
            "development score and cannot access the sealed 2025 target. A separate "
            "frozen evaluator is required for any PBC comparison.",
            "",
            "Main artifacts: `selection.json`, validation case/summary tables, "
            "per-seed checkpoints and adjustment fields, lag/context provenance, "
            "training-only categorical thresholds, and the hash-bound `manifest.json`.",
            "",
        ]
    )


def _validate_digest(value: str | None, option: str) -> None:
    if value is not None and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{option} must be 64 lowercase hex characters")


def assert_validation_temporal_contract(
    initializations: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    lags: Any,
) -> dict[str, Any]:
    """Prove train/validation timing without indexing development cases."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    ends = starts + np.timedelta64(41, "D")
    if np.any(ends[train] >= np.datetime64("2018-01-01")):
        raise PersistenceExperimentError(
            "training outcome crosses the validation boundary"
        )
    if np.any(ends[validation] >= np.datetime64("2020-01-01")):
        raise PersistenceExperimentError(
            "validation outcome crosses the development boundary"
        )
    if set(pd.DatetimeIndex(starts[validation]).year) != set(
        base.VALIDATION_YEARS
    ):
        raise PersistenceExperimentError(
            "validation initializations do not cover the frozen years"
        )
    if np.any(pd.DatetimeIndex(starts).year == min(base.SEALED_YEARS)):
        raise PersistenceExperimentError("sealed 2025 appears in forecast cache")
    active = np.concatenate((train, validation))
    issue = np.broadcast_to(starts[active, None], lags.window_end[active].shape)
    available = np.asarray(lags.source_indices[active]) >= 0
    if np.any(lags.window_end[active][available] >= issue[available]):
        raise PersistenceExperimentError(
            "persistence lag reaches or follows forecast issuance"
        )
    if not np.all(np.asarray(lags.source_indices[validation]) >= 0):
        raise PersistenceExperimentError(
            "validation issues lack one or more exact persistence lag windows"
        )
    return {
        "training_latest_outcome_end": np.datetime_as_string(
            ends[train].max(), unit="D"
        ),
        "validation_latest_outcome_end": np.datetime_as_string(
            ends[validation].max(), unit="D"
        ),
        "lag_timing_checked_scope": "train_and_validation_only",
        "development_indices_accessed": False,
    }


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    started_at = time.monotonic()
    snapshot = source_snapshot(output)
    arms = base._parse_names(args.arms, ARMS, "persistence arms")
    seeds = base._parse_seeds(args.seeds)
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    cache = base.load_member_cache(Path(args.cache), allow_partial=False)
    provenance = base.cache_provenance(cache)
    if provenance.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise PersistenceExperimentError("member cache source fingerprint is not canonical")
    if provenance.get("data_sha256") != EXPECTED_CACHE_SHA256:
        raise PersistenceExperimentError("canonical full-cache SHA-256 is required")
    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(value) for name, value in splits.as_dict().items()}
    if split_counts != base.EXPECTED_COUNTS:
        raise PersistenceExperimentError("canonical full archive split is required")
    train_indices = splits.train
    validation_indices = splits.validation
    if args.smoke:
        train_indices = base.select_evenly(train_indices, 32)
        validation_indices = base.select_evenly(validation_indices, 16)
    if set(pd.DatetimeIndex(cache.initializations[validation_indices]).year) != set(
        base.VALIDATION_YEARS
    ):
        raise PersistenceExperimentError("selected validation cases lack both years")
    print(
        f"Effective cases: train={len(train_indices)}, validation={len(validation_indices)}; "
        "development/test will not be scored",
        flush=True,
    )

    observations = base.load_imd_observations(cache)
    if any("2025" in str(path) for path in observations.source_stores):
        raise PersistenceExperimentError("observation loader opened a sealed store")
    observation_provenance = pbc_driver.array_bundle_provenance(
        {
            "weekly_truth": observations.weekly_truth,
            "observation_fraction": observations.observation_fraction,
            "weights": observations.weights,
        },
        bundle_name="imd_weekly_truth_fraction_and_scoring_weights",
    )
    expected_observation = args.expected_observation_bundle_sha256
    if expected_observation is not None and (
        observation_provenance["sha256"] != expected_observation
    ):
        raise PersistenceExperimentError("observation bundle differs from passed smoke")
    daily_dates, daily_values = pbc_driver.load_daily_imd_for_lags(
        cache, observations.source_stores
    )
    lags = build_daily_issue_time_lags(
        cache.initializations, daily_dates, daily_values, lag_weeks=(1, 2)
    )
    del daily_dates, daily_values
    lag_provenance = dict(pbc_driver.issue_time_lag_provenance(lags))
    support = observations.weights > 0.0
    available_values = np.asarray(lags.values)[lags.source_indices >= 0][:, support]
    if not np.isfinite(available_values).all() or np.any(available_values < 0.0):
        raise PersistenceExperimentError("lag values are invalid on scoring support")
    lag_provenance["available_supported_values_finite_nonnegative"] = True
    expected_lag = args.expected_persistence_lag_sha256
    if expected_lag is not None and lag_provenance["sha256"] != expected_lag:
        raise PersistenceExperimentError("lag bundle differs from passed smoke")
    temporal_evidence = assert_validation_temporal_contract(
        cache.initializations, splits.train, splits.validation, lags
    )
    temporal_evidence["lag_windows"] = (
        "issue-7 through issue-1 and issue-14 through issue-8; every end < issue"
    )
    base_context = base.build_context_bundle(cache, observations, train_indices)
    context = build_persistence_context_bundle(
        base_context,
        cache.initializations,
        lags,
        train_indices,
        observations.weights,
    )
    context_provenance = pbc_driver.array_bundle_provenance(
        {
            "normalized_lag_log1p": context.normalized_lag_log1p,
            "lag_available": context.lag_available,
            "lag_log1p_mean_by_lag": context.lag_log1p_mean_by_lag,
            "lag_log1p_std_by_lag": context.lag_log1p_std_by_lag,
            "normalization_fit_indices": context.normalization_fit_indices,
        },
        bundle_name="train_normalized_persistence_context",
    )

    evaluation = output / "evaluation"
    metrics_directory = output / "metrics"
    history_directory = output / "history"
    models_directory = output / "models"
    figures_directory = output / "figures"
    for directory in (
        evaluation,
        metrics_directory,
        history_directory,
        models_directory,
        figures_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    normalized_weights = observations.weights / observations.weights.sum(
        dtype=np.float64
    )
    scoring_support_path = evaluation / "scoring_support.npz"
    np.savez_compressed(
        scoring_support_path,
        latitude=cache.latitude.astype(np.float64),
        longitude=cache.longitude.astype(np.float64),
        observation_fraction=observations.observation_fraction.astype(np.float32),
        support_mask=support,
        scoring_weight_km2_fraction=observations.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights.astype(np.float64),
    )
    np.savez_compressed(
        evaluation / "persistence_normalization.npz",
        lag_log1p_mean_by_lag=context.lag_log1p_mean_by_lag,
        lag_log1p_std_by_lag=context.lag_log1p_std_by_lag,
        normalization_fit_indices=context.normalization_fit_indices,
        channel_names=np.asarray(LAG_CONTEXT_CHANNEL_NAMES),
    )
    write_json(evaluation / "persistence_lag_provenance.json", lag_provenance)
    write_json(evaluation / "observation_bundle_provenance.json", observation_provenance)
    write_json(evaluation / "persistence_context_provenance.json", context_provenance)

    quantiles = fit_calendar_quantiles(
        observations.weekly_truth,
        cache.initializations,
        train_indices,
        QUINTILE_LEVELS,
        support,
        window_radius_days=(CALENDAR_WINDOW_DAYS - 1) // 2,
        minimum_samples=MINIMUM_CALENDAR_SAMPLES,
    )
    validation_starts = cache.initializations[validation_indices]
    validation_thresholds = calendar_fields(quantiles, validation_starts, 6)
    validation_truth = np.asarray(
        observations.weekly_truth[validation_indices], dtype=np.float32
    )
    validation_observed = observation_cdf(validation_truth, validation_thresholds)
    threshold_path = evaluation / "validation_quintile_fit.npz"
    np.savez_compressed(
        threshold_path,
        levels=quantiles.levels,
        thresholds=quantiles.thresholds,
        empirical_cdf=quantiles.empirical_cdf,
        support=quantiles.support,
        fit_indices=quantiles.fit_indices,
        sample_count_by_day=quantiles.sample_count_by_day,
        validation_initializations=validation_starts,
        validation_thresholds=validation_thresholds,
    )

    histories: list[pd.DataFrame] = []
    runs: list[PersistenceTrainingRun] = []
    validation_frames: list[pd.DataFrame] = []
    initial_receipts: dict[tuple[str, int], str] = {}
    training_rng_receipts: dict[tuple[str, int], str] = {}
    for arm in arms:
        for seed in seeds:
            print(f"Training {arm}, seed {seed}...", flush=True)
            run_directory = models_directory / arm / f"seed_{seed}"
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
            training_rng_receipts[(arm, seed)] = record.training_rng_state_sha256
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
            validation_frames.append(
                validation_case_metrics(
                    arm,
                    seed,
                    cache.members,
                    observations.weekly_truth,
                    cache.initializations,
                    validation_indices,
                    delta,
                    log_spread,
                    observations.weights,
                    validation_thresholds,
                    validation_observed,
                    chunk_size=args.cdf_chunk_size,
                )
            )
            del model, delta, log_spread
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for seed in seeds:
        if initial_receipts[(ZERO_LAG_ARM, seed)] != initial_receipts[
            (PERSISTENCE_LAG_ARM, seed)
        ]:
            raise PersistenceExperimentError(
                f"45k arms do not share exact initial state for seed {seed}"
            )
        same_seed_rng = {
            training_rng_receipts[(arm, seed)] for arm in ARMS
        }
        if len(same_seed_rng) != 1:
            raise PersistenceExperimentError(
                f"arms do not share one stochastic training stream for seed {seed}"
            )

    history_frame = pd.concat(histories, ignore_index=True)
    validation_frame = pd.concat(validation_frames, ignore_index=True)
    cdf_evidence_columns = (
        "cdf_shape_matches_thresholds",
        "forecast_cdf_finite_where_threshold_defined",
        "observed_cdf_finite_where_threshold_defined",
        "forecast_cdf_valid_for_thresholds",
        "observed_cdf_valid_for_thresholds",
    )
    categorical_validity_evidence = {
        "threshold_and_observed_shapes_equal": bool(
            validation_thresholds.shape == validation_observed.shape
        ),
        "thresholds_finite_on_scoring_support": bool(
            np.isfinite(validation_thresholds[..., support]).all()
        ),
        "observed_cdf_finite_on_scoring_support": bool(
            np.isfinite(validation_observed[..., support]).all()
        ),
        "observed_cdf_valid_for_thresholds": bool(
            is_valid_cdf_for_thresholds(
                validation_observed, validation_thresholds, axis=2
            )
        ),
        "every_scoring_chunk_shape_matched_thresholds": bool(
            validation_frame["cdf_shape_matches_thresholds"].all()
        ),
        "every_forecast_cdf_finite_where_threshold_defined": bool(
            validation_frame[
                "forecast_cdf_finite_where_threshold_defined"
            ].all()
        ),
        "every_observed_cdf_finite_where_threshold_defined": bool(
            validation_frame[
                "observed_cdf_finite_where_threshold_defined"
            ].all()
        ),
        "every_forecast_cdf_valid_for_thresholds": bool(
            validation_frame["forecast_cdf_valid_for_thresholds"].all()
        ),
        "every_observed_cdf_valid_for_thresholds": bool(
            validation_frame["observed_cdf_valid_for_thresholds"].all()
        ),
        "every_validation_score_finite": bool(
            np.isfinite(
                validation_frame[["crps", "quintile_rps"]].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "evidence_columns": list(cdf_evidence_columns),
        "validation_metric_rows": int(len(validation_frame)),
    }
    if not all(
        value
        for key, value in categorical_validity_evidence.items()
        if key not in {"evidence_columns", "validation_metric_rows"}
    ):
        raise PersistenceExperimentError(
            "categorical validation evidence is not uniformly finite/valid"
        )
    selection = select_persistence_arm(
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
    validation_frame.to_csv(
        metrics_directory / "validation_case_metrics.csv", index=False
    )
    summary.to_csv(metrics_directory / "validation_summary.csv", index=False)
    by_year.to_csv(metrics_directory / "validation_by_year.csv", index=False)
    by_seed.to_csv(metrics_directory / "validation_by_seed.csv", index=False)
    by_lead.to_csv(metrics_directory / "validation_by_lead.csv", index=False)
    write_json(output / "selection.json", selection)
    plot_training_history(
        history_frame, figures_directory / "training_loss_curves", smoke=args.smoke
    )
    plot_validation_by_lead(
        by_lead, figures_directory / "validation_by_lead", smoke=args.smoke
    )
    (output / "README.md").write_text(
        build_readme(selection, smoke=args.smoke), encoding="utf-8"
    )

    run_records = []
    for record in runs:
        values = asdict(record)
        checkpoint = Path(record.checkpoint)
        values["checkpoint_sha256"] = sha256_file(checkpoint)
        values["checkpoint"] = str(checkpoint.relative_to(output))
        adjustment = checkpoint.parent.parent / "validation_adjustments.npz"
        values["validation_adjustment"] = str(adjustment.relative_to(output))
        values["validation_adjustment_sha256"] = sha256_file(adjustment)
        run_records.append(values)
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "validation-only persistence-feature screen; no development or sealed-test metrics"
        ),
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "forecast": "FuXi native weekly TP; all seasons; 51 members",
            "target": "IMD weekly mean precipitation, mm day-1",
            "train_years": list(base.TRAIN_YEARS),
            "validation_years": list(base.VALIDATION_YEARS),
            "selection_data": "2018-2019 validation only",
            "test_metrics_consulted": False,
            "development_years_not_scored": list(base.TEST_YEARS),
            "sealed_unopened_years": list(base.SEALED_YEARS),
            "sealed_2025_target_opened": False,
            "single_changed_factor": "two issue-time IMD lag fields plus two availability masks",
            "lag_windows": "issue-7..issue-1 and issue-14..issue-8 inclusive",
            "lag_transform": "log1p then per-lag train-only area-weighted scalar standardization",
            "context_order_added": list(LAG_CONTEXT_CHANNEL_NAMES),
            "primary_training_loss": "area-weighted empirical finite-ensemble CRPS",
            "checkpoint_metric": "full-51-member validation CRPS",
            "categorical_score_contract": SCORING_CONTRACT_VERSION,
            "statistical_unit": "initialization with all members and all six leads grouped",
        },
        "arms": [
            {
                "name": arm,
                "label": ARM_LABELS[arm],
                "role": ARM_ROLES[arm],
                "context_channels": context_channels_for_arm(arm),
                "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[arm],
                "selectable": arm != ZERO_LAG_ARM,
            }
            for arm in arms
        ],
        "seeds": list(seeds),
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train_indices),
            "validation": len(validation_indices),
        },
        "retained_initializations": {
            "train": [
                np.datetime_as_string(value, unit="D")
                for value in cache.initializations[train_indices]
            ],
            "validation": [
                np.datetime_as_string(value, unit="D")
                for value in cache.initializations[validation_indices]
            ],
        },
        "selection": selection,
        "training": {
            "batch_size": args.batch_size,
            "member_subsample": args.member_subsample,
            "full_members_for_validation": 51,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "automatic_mixed_precision": not args.no_amp,
            "device": str(device),
            "optimizer": "torch.optim.AdamW",
            "scheduler": "ReduceLROnPlateau(factor=0.5,min_lr=1e-6)",
            "gradient_clip_max_norm": 5.0,
            "same_seed_45k_initial_states_exact": True,
            "same_seed_stochastic_training_streams_exact_across_all_arms": True,
            "training_rng_receipts_by_seed": {
                str(seed): training_rng_receipts[(BASE_ARM, seed)]
                for seed in seeds
            },
            "runs": run_records,
        },
        "categorical_validation": {
            "levels": QUINTILE_LEVELS.tolist(),
            "calendar_window_days": CALENDAR_WINDOW_DAYS,
            "minimum_calendar_samples": MINIMUM_CALENDAR_SAMPLES,
            "threshold_fit_indices": train_indices.tolist(),
            "threshold_fit_artifact": str(threshold_path.relative_to(output)),
            "threshold_fit_sha256": sha256_file(threshold_path),
            "score_contract": SCORING_CONTRACT_VERSION,
            "finite_and_cdf_validity_evidence": categorical_validity_evidence,
        },
        "evaluation": {
            "scope": "validation only",
            "scoring_support_artifact": str(scoring_support_path.relative_to(output)),
            "scoring_support_sha256": sha256_file(scoring_support_path),
            "support_cells": int(np.count_nonzero(support)),
        },
        "temporal_evidence": temporal_evidence,
        "persistence_lag_provenance": lag_provenance,
        "observation_bundle_provenance": observation_provenance,
        "persistence_context_provenance": context_provenance,
        "cache": provenance,
        "observation_stores": list(observations.source_stores),
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
        description="Run the validation-only all-season persistence-feature ablation."
    )
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-persistence-lag-sha256", default=None)
    parser.add_argument("--expected-observation-bundle-sha256", default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds is None:
        args.seeds = "42" if args.smoke else "42,43,44"
    if args.max_epochs is None:
        args.max_epochs = 2 if args.smoke else 100
    if args.patience is None:
        args.patience = 1 if args.smoke else 15
    arms = base._parse_names(args.arms, ARMS, "persistence arms")
    seeds = base._parse_seeds(args.seeds)
    if arms != ARMS:
        raise ValueError(f"canonical persistence arms must be {ARMS} in order")
    expected_seeds = (42,) if args.smoke else base.SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"canonical {'smoke' if args.smoke else 'full'} seeds must be {expected_seeds}"
        )
    for name in (
        "max_epochs",
        "patience",
        "batch_size",
        "member_subsample",
        "evaluation_batch_size",
        "cdf_chunk_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
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
    if args.member_subsample > 51:
        raise ValueError("--member-subsample cannot exceed 51")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --learning-rate 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --weight-decay 0.0001")
    if args.no_amp:
        raise ValueError("canonical GPU run requires automatic mixed precision")
    if args.device == "cpu":
        raise ValueError("canonical run requires CUDA")
    _validate_digest(
        args.expected_persistence_lag_sha256,
        "--expected-persistence-lag-sha256",
    )
    _validate_digest(
        args.expected_observation_bundle_sha256,
        "--expected-observation-bundle-sha256",
    )
    if args.smoke and (
        args.expected_persistence_lag_sha256 is not None
        or args.expected_observation_bundle_sha256 is not None
    ):
        raise ValueError("smoke mode may not accept smoke-to-full provenance hashes")
    if not args.smoke and (
        args.expected_persistence_lag_sha256 is None
        or args.expected_observation_bundle_sha256 is None
    ):
        raise ValueError("full mode requires both completed-smoke provenance hashes")


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
        f"PASS: completed {'smoke' if args.smoke else 'full'} persistence run at "
        f"{requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
