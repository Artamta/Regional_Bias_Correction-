#!/usr/bin/env python3
"""Validation-only 2x2 anchor/loss ablation for FuXi-to-IMD correction.

The experiment changes only two scientifically motivated factors around the
proven ``width24_batch16`` compact architecture:

* the established shrinkage-10 lead/month/grid log anchor versus that same
  anchor with a train-only scalar physical-mean recentering per lead/month;
* the established 0.75/0.20/0.05 objective versus a bias-aware
  0.55/0.20/0.25 objective with a train-only physical bias scale.

Training uses 2002--2017, checkpoint selection and all comparisons use only
2018--2019, and no residual or metric is produced for 2020 onward.  Since the
two loss definitions have different numerical scales, ranking deliberately
uses physical IMD verification metrics only, never objective values.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

for path in (SOURCE_ROOT, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_compact_validation_sweep as compact  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    LogBiasCorrection,
    apply_log_bias_correction,
    verification_midpoint_months,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


base = compact.base
common = compact.common
RESULTS_ROOT = HERE / "results" / "fuxi_imd_bias_aware_validation_sweep"
TRAIN_YEARS = compact.TRAIN_YEARS
VALIDATION_YEARS = compact.VALIDATION_YEARS
QUARANTINED_YEARS = compact.QUARANTINED_YEARS
SEEDS = compact.SEEDS
LEAD_WEIGHTS = compact.LEAD_WEIGHTS
REFERENCE_CONFIGURATION = "log_anchor_current_loss"
MODEL_SPEC = compact.CANDIDATE_BY_NAME["width24_batch16"]

CURRENT_LOSS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
BIAS_AWARE_LOSS = {"smooth_l1": 0.55, "acc": 0.20, "bias": 0.25}


@dataclass(frozen=True)
class BiasCandidate:
    name: str
    label: str
    anchor_kind: str
    loss_kind: str
    loss_coefficients: Mapping[str, float]
    heavy_rain_threshold_mm_day: float | None = None
    heavy_rain_multiplier: float = 1.0

    @property
    def uses_bias_scale(self) -> bool:
        return self.loss_kind == "bias_aware"


CANDIDATES = (
    BiasCandidate(
        REFERENCE_CONFIGURATION,
        "Log anchor + current loss",
        "log_anchor",
        "current",
        CURRENT_LOSS,
    ),
    BiasCandidate(
        "log_anchor_bias_aware_loss",
        "Log anchor + bias-aware loss",
        "log_anchor",
        "bias_aware",
        BIAS_AWARE_LOSS,
    ),
    BiasCandidate(
        "recentered_anchor_current_loss",
        "Physical-recentered anchor + current loss",
        "physical_recentered",
        "current",
        CURRENT_LOSS,
    ),
    BiasCandidate(
        "recentered_anchor_bias_aware_loss",
        "Physical-recentered anchor + bias-aware loss",
        "physical_recentered",
        "bias_aware",
        BIAS_AWARE_LOSS,
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


@dataclass
class AnchorBundle:
    name: str
    bias_baseline: np.ndarray
    target: np.ndarray
    target_scale: np.ndarray


@dataclass
class PreparedBiasExperiment:
    shared: compact.PreparedData
    anchors: Mapping[str, AnchorBundle]
    bias_scale: np.ndarray


@dataclass(frozen=True)
class RecenterFit:
    correction: LogBiasCorrection
    scalar_by_lead_month: np.ndarray
    diagnostics: tuple[Mapping[str, Any], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_candidates(names: str | None) -> tuple[BiasCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(value.strip() for value in names.split(",") if value.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def validate_quarantined_splits(
    initializations: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> np.ndarray:
    """Validate the blocked-year contract and return quarantined indices."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    years = compact._calendar_years(dates)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1:
        raise ValueError("split indices must be one-dimensional")
    if len(np.unique(train)) != len(train) or len(np.unique(validation)) != len(
        validation
    ):
        raise ValueError("split indices must be unique")
    if np.intersect1d(train, validation).size:
        raise ValueError("train and validation indices overlap")
    if np.any(train < 0) or np.any(validation < 0):
        raise ValueError("split indices must be nonnegative")
    if np.any(train >= len(dates)) or np.any(validation >= len(dates)):
        raise ValueError("split index is out of bounds")
    if not np.all(np.isin(years[train], TRAIN_YEARS)):
        raise ValueError("train indices contain a year outside 2002--2017")
    if not np.all(np.isin(years[validation], VALIDATION_YEARS)):
        raise ValueError("validation indices contain a year outside 2018--2019")
    selected = np.concatenate((train, validation))
    if np.any(np.isin(years[selected], QUARANTINED_YEARS)):
        raise ValueError("selected indices contain a quarantined 2020+ year")
    return np.flatnonzero(np.isin(years, QUARANTINED_YEARS)).astype(np.int64)


def _weighted_reconstructed_mean(
    base_log_rain: np.ndarray,
    spatial_weights: np.ndarray,
    scalar: float,
) -> float:
    log_rain = np.maximum(0.0, np.asarray(base_log_rain, dtype=np.float64) + scalar)
    log_rain = np.minimum(log_rain, 60.0)
    rainfall = np.expm1(log_rain)
    weights = np.asarray(spatial_weights, dtype=np.float64)
    denominator = float(weights.sum(dtype=np.float64)) * rainfall.shape[0]
    return float(np.sum(rainfall * weights[None], dtype=np.float64) / denominator)


def solve_physical_mean_recenter_scalar(
    base_log_rain: np.ndarray,
    spatial_weights: np.ndarray,
    target_mean: float,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-10,
    maximum_iterations: int = 120,
) -> tuple[float, float]:
    """Solve a monotone nonnegative reconstruction for one scalar offset."""

    values = np.asarray(base_log_rain, dtype=np.float64)
    weights = np.asarray(spatial_weights, dtype=np.float64)
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("base_log_rain must be [case, cell] with matching weights")
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("base_log_rain must contain finite cases")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0:
        raise ValueError("spatial_weights must be finite with positive support")
    if not np.isfinite(target_mean) or target_mean < 0.0:
        raise ValueError("target_mean must be finite and nonnegative")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")

    lower = float(-np.max(values) - 1.0)
    lower_mean = _weighted_reconstructed_mean(values, weights, lower)
    tolerance = absolute_tolerance + relative_tolerance * max(target_mean, 1.0)
    if target_mean <= tolerance:
        return lower, lower_mean

    upper = max(0.0, lower + 1.0)
    upper_mean = _weighted_reconstructed_mean(values, weights, upper)
    step = 1.0
    expansions = 0
    while upper_mean < target_mean and upper < 60.0:
        upper += step
        step *= 2.0
        upper_mean = _weighted_reconstructed_mean(values, weights, upper)
        expansions += 1
        if expansions > 32:
            break
    if upper_mean < target_mean:
        raise RuntimeError("failed to bracket physical-mean recenter scalar")

    for _ in range(maximum_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_mean = _weighted_reconstructed_mean(values, weights, midpoint)
        if abs(midpoint_mean - target_mean) <= tolerance:
            return midpoint, midpoint_mean
        if midpoint_mean < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    midpoint = 0.5 * (lower + upper)
    midpoint_mean = _weighted_reconstructed_mean(values, weights, midpoint)
    if abs(midpoint_mean - target_mean) > 10.0 * tolerance:
        raise RuntimeError("physical-mean recenter bisection did not converge")
    return midpoint, midpoint_mean


def fit_physical_recentered_log_correction(
    fuxi_mean: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    fit_indices: np.ndarray,
    area_weights: np.ndarray,
    lead_month_residual: np.ndarray,
    *,
    split_name: str,
    shrinkage: float = 10.0,
    allowed_years: Sequence[int] | None = None,
) -> RecenterFit:
    """Fit train-only lead/month scalars on top of a fixed log correction."""

    if split_name != "train":
        raise ValueError("physical recentering must be fitted on the train split")
    raw = np.asarray(fuxi_mean, dtype=np.float64)
    observations = np.asarray(truth, dtype=np.float64)
    dates = np.asarray(initializations, dtype="datetime64[D]")
    indices = np.asarray(fit_indices, dtype=np.int64)
    weights = np.asarray(area_weights, dtype=np.float64)
    delta = np.asarray(lead_month_residual, dtype=np.float64)
    if raw.shape != observations.shape or raw.ndim != 4:
        raise ValueError("fuxi_mean and truth must have shape [case, lead, lat, lon]")
    if dates.shape != (raw.shape[0],):
        raise ValueError("initializations do not match the case dimension")
    if weights.shape != raw.shape[-2:]:
        raise ValueError("area_weights do not match the spatial grid")
    expected_delta = (raw.shape[1], 12, *raw.shape[-2:])
    if delta.shape != expected_delta:
        raise ValueError(f"lead_month_residual must have shape {expected_delta}")
    if indices.ndim != 1 or indices.size < 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("fit_indices must be a nonempty unique vector")
    if np.any(indices < 0) or np.any(indices >= len(raw)):
        raise ValueError("fit_indices contain an out-of-bounds case")
    years = compact._calendar_years(dates[indices])
    permitted_years = TRAIN_YEARS if allowed_years is None else tuple(allowed_years)
    if not permitted_years or not np.all(np.isin(years, permitted_years)):
        raise ValueError("physical recentering received non-training years")
    support = weights > 0.0
    if not np.any(support) or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("area_weights must be finite and positive on support")
    if not np.isfinite(raw[indices][..., support]).all() or np.any(
        raw[indices][..., support] < 0.0
    ):
        raise ValueError("training FuXi rainfall must be finite and nonnegative")
    if not np.isfinite(observations[indices][..., support]).all() or np.any(
        observations[indices][..., support] < 0.0
    ):
        raise ValueError("training IMD rainfall must be finite and nonnegative")

    months = verification_midpoint_months(dates, raw.shape[1])
    scalars = np.zeros((raw.shape[1], 12), dtype=np.float64)
    diagnostics: list[Mapping[str, Any]] = []
    spatial_weights = weights[support]
    weight_sum = float(spatial_weights.sum(dtype=np.float64))
    for lead in range(raw.shape[1]):
        for month in range(1, 13):
            selected = indices[months[indices, lead] == month]
            if selected.size == 0:
                diagnostics.append(
                    {
                        "lead": lead + 1,
                        "month": month,
                        "training_cases": 0,
                        "scalar": 0.0,
                        "target_imd_mean": np.nan,
                        "uncentered_anchor_mean": np.nan,
                        "recentered_anchor_mean": np.nan,
                        "absolute_closure_error": np.nan,
                    }
                )
                continue
            month_delta = delta[lead, month - 1][support]
            if not np.isfinite(month_delta).all():
                raise ValueError("log correction is nonfinite on weighted support")
            base_log = (
                np.log1p(np.maximum(raw[selected, lead][:, support], 0.0))
                + month_delta[None]
            )
            target_mean = float(
                np.sum(
                    observations[selected, lead][:, support]
                    * spatial_weights[None],
                    dtype=np.float64,
                )
                / (selected.size * weight_sum)
            )
            uncentered = _weighted_reconstructed_mean(base_log, spatial_weights, 0.0)
            scalar, achieved = solve_physical_mean_recenter_scalar(
                base_log, spatial_weights, target_mean
            )
            scalars[lead, month - 1] = scalar
            diagnostics.append(
                {
                    "lead": lead + 1,
                    "month": month,
                    "training_cases": int(selected.size),
                    "scalar": float(scalar),
                    "target_imd_mean": target_mean,
                    "uncentered_anchor_mean": uncentered,
                    "recentered_anchor_mean": achieved,
                    "absolute_closure_error": abs(achieved - target_mean),
                }
            )

    recentered = delta.copy()
    for lead in range(raw.shape[1]):
        for month in range(12):
            finite = np.isfinite(recentered[lead, month])
            recentered[lead, month, finite] += scalars[lead, month]
    correction = LogBiasCorrection(recentered.astype(np.float32), float(shrinkage))
    return RecenterFit(
        correction=correction,
        scalar_by_lead_month=scalars.astype(np.float32),
        diagnostics=tuple(diagnostics),
    )


def fit_training_bias_scale(
    truth: np.ndarray,
    area_weights: np.ndarray,
    fit_indices: np.ndarray,
    initializations: np.ndarray,
    *,
    split_name: str,
    allowed_years: Sequence[int] | None = None,
) -> np.ndarray:
    """Return train-only area/case weighted mean IMD rainfall by lead."""

    if split_name != "train":
        raise ValueError("bias scale must be fitted on the train split")
    values = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    indices = np.asarray(fit_indices, dtype=np.int64)
    years = compact._calendar_years(
        np.asarray(initializations, dtype="datetime64[D]")[indices]
    )
    permitted_years = TRAIN_YEARS if allowed_years is None else tuple(allowed_years)
    if not permitted_years or not np.all(np.isin(years, permitted_years)):
        raise ValueError("bias scale received non-training years")
    support = weights > 0.0
    spatial_weights = weights[support]
    denominator = len(indices) * float(spatial_weights.sum(dtype=np.float64))
    means = np.asarray(
        [
            np.sum(
                values[indices, lead][:, support] * spatial_weights[None],
                dtype=np.float64,
            )
            / denominator
            for lead in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    if not np.isfinite(means).all() or np.any(means <= 0.0):
        raise ValueError("train-only IMD lead means must be finite and positive")
    return means.astype(np.float32)


def _make_anchor_bundle(
    name: str,
    correction: LogBiasCorrection,
    shared: compact.PreparedData,
) -> AnchorBundle:
    selected = np.sort(
        np.concatenate((shared.train_indices, shared.validation_indices)).astype(
            np.int64
        )
    )
    baseline = np.zeros_like(shared.raw_fuxi, dtype=np.float32)
    baseline[selected] = apply_log_bias_correction(
        shared.raw_fuxi[selected], shared.initializations[selected], correction
    )
    scale = fit_anchored_target_scale(
        shared.truth[shared.train_indices],
        baseline[shared.train_indices],
        shared.weights,
        split_name="train",
        valid_mask=shared.valid_mask[shared.train_indices],
    )
    target = np.zeros_like(shared.truth, dtype=np.float32)
    target[selected] = standardize_anchored_target(
        shared.truth[selected],
        baseline[selected],
        scale,
        valid_mask=shared.valid_mask[selected],
    )
    quarantined = validate_quarantined_splits(
        shared.initializations, shared.train_indices, shared.validation_indices
    )
    if np.count_nonzero(baseline[quarantined]) or np.count_nonzero(target[quarantined]):
        raise base.DataContractError("anchor created values in quarantined years")
    return AnchorBundle(name, baseline, target, np.asarray(scale, dtype=np.float32))


def prepare_data() -> tuple[
    PreparedBiasExperiment,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    # Supplying only the proven standard model prevents member/physical caches
    # from being loaded during this focused ablation.
    shared, normalization, preparation = compact.prepare_data((MODEL_SPEC,))
    quarantined = validate_quarantined_splits(
        shared.initializations, shared.train_indices, shared.validation_indices
    )
    if len(shared.train_indices) != 560 or len(shared.validation_indices) != 70:
        raise base.DataContractError("unexpected train/validation case counts")
    current_correction = LogBiasCorrection(
        np.asarray(preparation["anchor"]["lead_month_residual"], dtype=np.float32),
        float(preparation["anchor"]["shrinkage"]),
    )
    recentered = fit_physical_recentered_log_correction(
        shared.raw_fuxi,
        shared.truth,
        shared.initializations,
        shared.train_indices,
        shared.weights,
        current_correction.lead_month_residual,
        split_name="train",
        shrinkage=current_correction.shrinkage,
    )
    anchors = {
        "log_anchor": _make_anchor_bundle("log_anchor", current_correction, shared),
        "physical_recentered": _make_anchor_bundle(
            "physical_recentered", recentered.correction, shared
        ),
    }
    bias_scale = fit_training_bias_scale(
        shared.truth,
        shared.weights,
        shared.train_indices,
        shared.initializations,
        split_name="train",
    )
    metadata = dict(preparation["metadata"])
    metadata.update(
        {
            "model_architecture": MODEL_SPEC.name,
            "quarantined_case_count_in_loaded_archive": int(len(quarantined)),
            "quarantined_anchor_values_nonzero": False,
            "anchor_target_scales_refitted_independently": True,
            "bias_scale_definition": (
                "train-only area/case-weighted mean IMD rainfall per lead"
            ),
        }
    )
    artifacts = {
        "current_correction": current_correction,
        "recentered_fit": recentered,
        "recenter_diagnostics": pd.DataFrame(recentered.diagnostics),
    }
    return PreparedBiasExperiment(shared, anchors, bias_scale), normalization, {
        "metadata": metadata,
        **artifacts,
    }


def _observation_data(shared: compact.PreparedData) -> Any:
    return base.ObservationData(
        weekly_truth=shared.truth,
        weekly_climatology=shared.climatology,
        observation_fraction=(shared.weights > 0.0).astype(np.float32),
        source_stores=(),
    )


def _worker(
    worker_index: int,
    device: str,
    tasks: Sequence[tuple[BiasCandidate, int]],
    prepared: PreparedBiasExperiment,
    output: Path,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> None:
    try:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        torch.set_num_threads(max(1, allocated // 2))
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        shared = prepared.shared
        observations = _observation_data(shared)
        support = shared.weights > 0.0
        datasets: dict[str, tuple[Any, Any]] = {}
        for anchor_name, anchor in prepared.anchors.items():
            datasets[anchor_name] = (
                common.make_dataset(
                    shared.train_indices,
                    shared.features,
                    anchor.target,
                    anchor.bias_baseline,
                    observations,
                    support,
                ),
                common.make_dataset(
                    shared.validation_indices,
                    shared.features,
                    anchor.target,
                    anchor.bias_baseline,
                    observations,
                    support,
                ),
            )
        for candidate, seed in tasks:
            print(
                f"[{device}] {candidate.name}, seed={seed}, "
                f"anchor={candidate.anchor_kind}, loss={candidate.loss_kind}",
                flush=True,
            )
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            (run_directory / "logs").mkdir(parents=True, exist_ok=False)
            (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
            set_deterministic_seed(seed)
            model = compact.build_model(
                MODEL_SPEC,
                shared.features.shape[2],
                shared.mean_to_anomaly_ratio,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            train_data, validation_data = datasets[candidate.anchor_kind]
            anchor = prepared.anchors[candidate.anchor_kind]
            result = train_anchored_model(
                model,
                train_data,
                validation_data,
                shared.weights,
                anchor.target_scale,
                LEAD_WEIGHTS,
                candidate.loss_coefficients,
                run_directory,
                seed=seed,
                device=device,
                batch_size=MODEL_SPEC.batch_size,
                max_epochs=2 if smoke else max_epochs,
                patience=1 if smoke else patience,
                learning_rate=MODEL_SPEC.learning_rate,
                weight_decay=MODEL_SPEC.weight_decay,
                smooth_l1_beta=1.0,
                bias_scale=(prepared.bias_scale if candidate.uses_bias_scale else None),
                heavy_rain_threshold_mm_day=(
                    candidate.heavy_rain_threshold_mm_day
                ),
                heavy_rain_multiplier=candidate.heavy_rain_multiplier,
                num_workers=0,
                use_amp=True,
            )
            residual = predict(
                model,
                shared.features[shared.validation_indices],
                device=device,
                batch_size=32,
                use_amp=True,
            )
            expected_shape = (len(shared.validation_indices), 6, 27, 27)
            if residual.shape != expected_shape or not np.isfinite(residual).all():
                raise ValueError(f"unexpected validation residual {residual.shape}")
            np.save(run_directory / "validation_residual.npy", residual)
            checkpoint = run_directory / "checkpoints" / "best.pt"
            record = {
                "status": "complete",
                "configuration": candidate.name,
                "candidate": asdict(candidate),
                "seed": int(seed),
                "device": device,
                "worker_index": int(worker_index),
                "parameter_count": int(parameter_count),
                "best_epoch_zero_based": int(result.best_epoch),
                "best_epoch_display": int(result.best_epoch + 1),
                "best_validation_objective": float(result.best_validation_loss),
                "elapsed_seconds": float(result.elapsed_seconds),
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": compact.sha256_file(checkpoint),
                "history": str(
                    (run_directory / "logs" / "training_history.csv").relative_to(
                        output
                    )
                ),
                "validation_residual": str(
                    (run_directory / "validation_residual.npy").relative_to(output)
                ),
                "objective_values_comparable_across_configurations": False,
            }
            (run_directory / "run_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
    except Exception:
        (output / f"worker_{worker_index}_failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


def run_parallel(
    candidates: Sequence[BiasCandidate],
    seeds: Sequence[int],
    prepared: PreparedBiasExperiment,
    output: Path,
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    workers: int,
) -> None:
    tasks = [(candidate, int(seed)) for candidate in candidates for seed in seeds]
    if not tasks or workers < 1:
        raise ValueError("sweep must contain tasks and at least one worker")
    if workers == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _worker(0, device, tasks, prepared, output, max_epochs, patience, smoke)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
        raise RuntimeError(
            f"requested {workers} GPU workers but only "
            f"{torch.cuda.device_count()} are visible"
        )
    groups = [tasks[index::workers] for index in range(workers)]
    context = mp.get_context("spawn")
    processes = []
    for index, group in enumerate(groups):
        process = context.Process(
            target=_worker,
            args=(
                index,
                f"cuda:{index}",
                group,
                prepared,
                output,
                max_epochs,
                patience,
                smoke,
            ),
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append((process.pid, process.exitcode))
    if failures:
        raise RuntimeError(f"one or more sweep workers failed: {failures}")


def _prediction_from_residual(
    residual: np.ndarray,
    candidate: BiasCandidate,
    prepared: PreparedBiasExperiment,
) -> np.ndarray:
    shared = prepared.shared
    indices = shared.validation_indices
    anchor = prepared.anchors[candidate.anchor_kind]
    return reconstruct_anchored_precipitation(
        anchor.bias_baseline[indices],
        residual,
        anchor.target_scale,
        valid_mask=shared.valid_mask[indices],
    )


def _case_metrics(
    prediction: np.ndarray,
    prepared: PreparedBiasExperiment,
    *,
    predictor: str,
) -> pd.DataFrame:
    shared = prepared.shared
    indices = shared.validation_indices
    truth = shared.truth[indices]
    climatology = shared.climatology[indices]
    return compute_case_metrics(
        truth,
        prediction,
        truth - climatology,
        prediction - climatology,
        shared.weights,
        predictor=predictor,
        case_ids=shared.initializations[indices],
        leads=np.arange(1, 7),
        valid_mask=shared.valid_mask[indices],
    )


def intensity_strata_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    area_weights: np.ndarray,
    *,
    configuration: str,
) -> pd.DataFrame:
    """Physical error diagnostics stratified by verifying IMD intensity."""

    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 4:
        raise ValueError("prediction and truth must be [case, lead, lat, lon]")
    if weights.shape != observed.shape[-2:]:
        raise ValueError("area_weights do not match prediction")
    strata = (
        ("dry_lt1", 0.0, 1.0),
        ("light_1_5", 1.0, 5.0),
        ("moderate_5_10", 5.0, 10.0),
        ("heavy_ge10", 10.0, np.inf),
    )
    rows = []
    spatial = weights[None, None]
    for lead_index in (None, *range(observed.shape[1])):
        obs = observed if lead_index is None else observed[:, lead_index : lead_index + 1]
        pred = predicted if lead_index is None else predicted[:, lead_index : lead_index + 1]
        broadcast_weights = np.broadcast_to(spatial, obs.shape)
        for label, lower, upper in strata:
            mask = (
                (weights[None, None] > 0.0)
                & np.isfinite(obs)
                & np.isfinite(pred)
                & (obs >= lower)
                & (obs < upper)
            )
            selected_weights = np.where(mask, broadcast_weights, 0.0)
            denominator = float(selected_weights.sum(dtype=np.float64))
            if denominator <= 0.0:
                rmse = mae = bias = truth_mean = prediction_mean = np.nan
            else:
                error = np.where(mask, pred - obs, 0.0)
                safe_obs = np.where(mask, obs, 0.0)
                safe_pred = np.where(mask, pred, 0.0)
                rmse = float(
                    np.sqrt(
                        np.sum(selected_weights * error**2, dtype=np.float64)
                        / denominator
                    )
                )
                mae = float(
                    np.sum(selected_weights * np.abs(error), dtype=np.float64)
                    / denominator
                )
                bias = float(
                    np.sum(selected_weights * error, dtype=np.float64) / denominator
                )
                truth_mean = float(
                    np.sum(selected_weights * safe_obs, dtype=np.float64) / denominator
                )
                prediction_mean = float(
                    np.sum(selected_weights * safe_pred, dtype=np.float64) / denominator
                )
            rows.append(
                {
                    "configuration": configuration,
                    "lead": "ALL_WEEKS" if lead_index is None else f"W{lead_index + 1}",
                    "stratum": label,
                    "lower_mm_day": lower,
                    "upper_mm_day": upper,
                    "cell_case_count": int(np.count_nonzero(mask)),
                    "weight_sum": denominator,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "truth_mean": truth_mean,
                    "prediction_mean": prediction_mean,
                }
            )
    return pd.DataFrame(rows)


def aggregate_results(
    output: Path,
    candidates: Sequence[BiasCandidate],
    seeds: Sequence[int],
    prepared: PreparedBiasExperiment,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[Mapping[str, Any]] = []
    histories = []
    case_frames = []
    intensity_frames = []
    shared = prepared.shared
    validation_truth = shared.truth[shared.validation_indices]
    for candidate in candidates:
        residuals = []
        parameter_counts = set()
        for seed in seeds:
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            record = json.loads(
                (run_directory / "run_record.json").read_text(encoding="utf-8")
            )
            if record.get("status") != "complete":
                raise ValueError(f"incomplete run: {run_directory}")
            records.append(record)
            parameter_counts.add(int(record["parameter_count"]))
            history = pd.read_csv(run_directory / "logs" / "training_history.csv")
            history.insert(0, "loss_kind", candidate.loss_kind)
            history.insert(0, "seed", int(seed))
            history.insert(0, "configuration", candidate.name)
            histories.append(history)
            residual = np.load(run_directory / "validation_residual.npy")
            residuals.append(residual)
            prediction = _prediction_from_residual(residual, candidate, prepared)
            frame = _case_metrics(
                prediction, prepared, predictor=f"{candidate.name}_seed_{seed}"
            )
            frame.insert(0, "member", f"seed_{seed}")
            frame.insert(0, "configuration", candidate.name)
            case_frames.append(frame)
        if len(parameter_counts) != 1:
            raise ValueError(f"parameter count varied for {candidate.name}")
        ensemble = np.mean(residuals, axis=0, dtype=np.float64).astype(np.float32)
        np.save(
            output / "models" / candidate.name / "validation_residual_ensemble.npy",
            ensemble,
        )
        prediction = _prediction_from_residual(ensemble, candidate, prepared)
        frame = _case_metrics(
            prediction, prepared, predictor=f"{candidate.name}_ensemble"
        )
        frame.insert(0, "member", "ensemble")
        frame.insert(0, "configuration", candidate.name)
        case_frames.append(frame)
        intensity_frames.append(
            intensity_strata_metrics(
                prediction,
                validation_truth,
                shared.weights,
                configuration=candidate.name,
            )
        )

    baseline_predictions = {
        "raw_fuxi": shared.raw_fuxi[shared.validation_indices],
        "log_anchor": prepared.anchors["log_anchor"].bias_baseline[
            shared.validation_indices
        ],
        "physical_recentered_anchor": prepared.anchors[
            "physical_recentered"
        ].bias_baseline[shared.validation_indices],
    }
    for name, prediction in baseline_predictions.items():
        frame = _case_metrics(prediction, prepared, predictor=name)
        frame.insert(0, "member", "deterministic")
        frame.insert(0, "configuration", name)
        case_frames.append(frame)
        intensity_frames.append(
            intensity_strata_metrics(
                prediction,
                validation_truth,
                shared.weights,
                configuration=name,
            )
        )

    records_frame = pd.json_normalize(records, sep="_")
    history_frame = pd.concat(histories, ignore_index=True)
    case_frame = pd.concat(case_frames, ignore_index=True)
    intensity_frame = pd.concat(intensity_frames, ignore_index=True)
    records_frame.to_csv(output / "metrics" / "run_records.csv", index=False)
    history_frame.to_csv(
        output / "metrics" / "training_history_tidy.csv", index=False
    )
    case_frame.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    intensity_frame.to_csv(
        output / "metrics" / "validation_intensity_strata.csv", index=False
    )
    matrix = compact.summarize_case_metrics(case_frame)
    matrix.to_csv(
        output / "metrics" / "validation_year_lead_matrix.csv", index=False
    )
    return records_frame, history_frame, case_frame, intensity_frame


def _absolute_bias_reduction_percent(candidate: float, reference: float) -> float:
    reference_abs = abs(float(reference))
    candidate_abs = abs(float(candidate))
    if reference_abs <= 1.0e-12:
        return 0.0 if candidate_abs <= 1.0e-12 else -np.inf
    return 100.0 * (reference_abs - candidate_abs) / reference_abs


def build_seed_physical_guards(
    case_metrics: pd.DataFrame,
    candidates: Sequence[BiasCandidate],
    *,
    reference_configuration: str = REFERENCE_CONFIGURATION,
) -> pd.DataFrame:
    """Compare each seed with the matching reference seed in physical units."""

    candidate_names = {candidate.name for candidate in candidates}
    members = case_metrics.member.astype(str)
    seeded = case_metrics.loc[
        members.str.startswith("seed_")
        & case_metrics.configuration.isin(candidate_names)
    ].copy()
    reference = seeded.loc[
        seeded.configuration.eq(reference_configuration)
    ]
    if reference.empty:
        raise ValueError("seed guards require matching reference-seed metrics")
    reference_members = tuple(sorted(reference.member.unique()))
    rows = []
    for candidate in candidates:
        candidate_frame = seeded.loc[seeded.configuration.eq(candidate.name)]
        candidate_members = tuple(sorted(candidate_frame.member.unique()))
        if candidate_members != reference_members:
            raise ValueError(
                f"seed members differ for {candidate.name}: "
                f"{candidate_members} vs {reference_members}"
            )
        for member in reference_members:
            candidate_seed = candidate_frame.loc[candidate_frame.member.eq(member)]
            reference_seed = reference.loc[reference.member.eq(member)]
            values = {
                f"candidate_{metric}": float(candidate_seed[metric].mean())
                for metric in ("rmse", "bias", "acc")
            }
            values.update(
                {
                    f"reference_{metric}": float(reference_seed[metric].mean())
                    for metric in ("rmse", "bias", "acc")
                }
            )
            bias_improved = abs(values["candidate_bias"]) < abs(
                values["reference_bias"]
            )
            rmse_guard = values["candidate_rmse"] <= values["reference_rmse"] * 1.01
            acc_guard = values["candidate_acc"] >= values["reference_acc"] - 0.01
            rows.append(
                {
                    "configuration": candidate.name,
                    "member": member,
                    **values,
                    "absolute_bias_improved": bool(bias_improved),
                    "rmse_within_one_percent": bool(rmse_guard),
                    "acc_drop_within_0p01": bool(acc_guard),
                    "seed_passes_all_guards": bool(
                        bias_improved and rmse_guard and acc_guard
                    ),
                }
            )
    return pd.DataFrame(rows)


def candidate_vs_own_anchor(
    case_metrics: pd.DataFrame,
    candidates: Sequence[BiasCandidate],
) -> pd.DataFrame:
    """Return transparent physical comparisons against each candidate's anchor."""

    ensemble = case_metrics.loc[case_metrics.member.eq("ensemble")].copy()
    deterministic = case_metrics.loc[case_metrics.member.eq("deterministic")].copy()
    ensemble["year"] = pd.DatetimeIndex(ensemble.case_id).year
    deterministic["year"] = pd.DatetimeIndex(deterministic.case_id).year
    rows = []
    for candidate in candidates:
        anchor_configuration = (
            "log_anchor"
            if candidate.anchor_kind == "log_anchor"
            else "physical_recentered_anchor"
        )
        candidate_frame = ensemble.loc[
            ensemble.configuration.eq(candidate.name)
        ]
        anchor_frame = deterministic.loc[
            deterministic.configuration.eq(anchor_configuration)
        ]
        if candidate_frame.empty or anchor_frame.empty:
            raise ValueError(f"missing own-anchor comparison for {candidate.name}")
        scopes: list[tuple[str, pd.Series, pd.Series]] = [
            (
                "ALL_WEEKS",
                pd.Series(True, index=candidate_frame.index),
                pd.Series(True, index=anchor_frame.index),
            )
        ]
        scopes.extend(
            (
                str(year),
                candidate_frame.year.eq(year),
                anchor_frame.year.eq(year),
            )
            for year in VALIDATION_YEARS
        )
        scopes.extend(
            (
                f"W{lead}",
                candidate_frame.lead.eq(lead),
                anchor_frame.lead.eq(lead),
            )
            for lead in range(1, 7)
        )
        for scope, candidate_mask, anchor_mask in scopes:
            candidate_scope = candidate_frame.loc[candidate_mask]
            anchor_scope = anchor_frame.loc[anchor_mask]
            if candidate_scope.empty or anchor_scope.empty:
                raise ValueError(
                    f"empty own-anchor scope {scope} for {candidate.name}"
                )
            for metric in ("rmse", "mae", "bias", "acc"):
                candidate_mean = float(candidate_scope[metric].mean())
                anchor_mean = float(anchor_scope[metric].mean())
                rows.append(
                    {
                        "configuration": candidate.name,
                        "anchor_configuration": anchor_configuration,
                        "scope": scope,
                        "metric": metric,
                        "candidate_mean": candidate_mean,
                        "anchor_mean": anchor_mean,
                        "delta_candidate_minus_anchor": candidate_mean - anchor_mean,
                        "absolute_bias_delta_candidate_minus_anchor": (
                            abs(candidate_mean) - abs(anchor_mean)
                            if metric == "bias"
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_physical_ranking(
    records: pd.DataFrame,
    case_metrics: pd.DataFrame,
    candidates: Sequence[BiasCandidate],
    *,
    reference_configuration: str = REFERENCE_CONFIGURATION,
    seed_guards: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank heterogeneous-loss candidates using physical metrics only."""

    names = {candidate.name for candidate in candidates}
    if reference_configuration not in names:
        raise ValueError("reference configuration must be present")
    ensemble = case_metrics.loc[
        case_metrics.member.eq("ensemble")
        & case_metrics.configuration.isin(names)
    ].copy()
    ensemble["year"] = pd.DatetimeIndex(ensemble.case_id).year
    raw = case_metrics.loc[
        case_metrics.configuration.eq("raw_fuxi")
        & case_metrics.member.eq("deterministic")
    ].copy()
    if ensemble.empty or raw.empty:
        raise ValueError("ranking requires candidate ensembles and raw FuXi")
    raw_summary = {
        metric: float(raw[metric].mean()) for metric in ("rmse", "bias", "acc")
    }
    if seed_guards is None:
        seed_guards = build_seed_physical_guards(
            case_metrics,
            candidates,
            reference_configuration=reference_configuration,
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = ensemble.loc[ensemble.configuration.eq(candidate.name)]
        if metrics.empty:
            raise ValueError(f"missing ensemble metrics for {candidate.name}")
        run_rows = records.loc[records.configuration.eq(candidate.name)]
        row: dict[str, Any] = {
            "configuration": candidate.name,
            "label": candidate.label,
            "anchor_kind": candidate.anchor_kind,
            "loss_kind": candidate.loss_kind,
            "parameter_count": int(run_rows.parameter_count.iloc[0]),
            "mean_best_validation_objective": float(
                run_rows.best_validation_objective.mean()
            ),
            "mean_runtime_seconds": float(run_rows.elapsed_seconds.mean()),
            "objective_values_comparable_across_configurations": False,
        }
        candidate_seed_guards = seed_guards.loc[
            seed_guards.configuration.eq(candidate.name)
        ]
        seed_total = int(len(candidate_seed_guards))
        if seed_total < 1:
            raise ValueError(f"missing seed guards for {candidate.name}")
        seed_required = 1 if seed_total == 1 else int(np.ceil(2.0 * seed_total / 3.0))
        row.update(
            {
                "seed_guard_passes": int(
                    candidate_seed_guards.seed_passes_all_guards.sum()
                ),
                "seed_guard_required": seed_required,
                "seed_guard_total": seed_total,
            }
        )
        for metric in ("rmse", "mae", "bias", "acc"):
            row[f"pooled_{metric}"] = float(metrics[metric].mean())
            for year in VALIDATION_YEARS:
                year_values = metrics.loc[metrics.year.eq(year), metric]
                if year_values.empty:
                    raise ValueError(f"missing {year} metrics for {candidate.name}")
                row[f"{year}_{metric}"] = float(year_values.mean())
            for lead in range(1, 7):
                lead_values = metrics.loc[metrics.lead.eq(lead), metric]
                if lead_values.empty:
                    raise ValueError(f"missing W{lead} metrics for {candidate.name}")
                row[f"W{lead}_{metric}"] = float(lead_values.mean())
        rows.append(row)
    ranking = pd.DataFrame(rows)
    reference = ranking.loc[
        ranking.configuration.eq(reference_configuration)
    ].iloc[0]

    ranking["pooled_abs_bias"] = ranking.pooled_bias.abs()
    ranking["pooled_abs_bias_reduction_pct"] = [
        _absolute_bias_reduction_percent(value, reference.pooled_bias)
        for value in ranking.pooled_bias
    ]
    ranking["bias_improves_both_years"] = [
        all(
            abs(float(row[f"{year}_bias"]))
            < abs(float(reference[f"{year}_bias"]))
            for year in VALIDATION_YEARS
        )
        for _, row in ranking.iterrows()
    ]
    ranking["bias_leads_improved"] = [
        sum(
            abs(float(row[f"W{lead}_bias"]))
            < abs(float(reference[f"W{lead}_bias"]))
            for lead in range(1, 7)
        )
        for _, row in ranking.iterrows()
    ]
    ranking["mean_abs_lead_bias"] = [
        float(np.mean([abs(float(row[f"W{lead}_bias"])) for lead in range(1, 7)]))
        for _, row in ranking.iterrows()
    ]
    reference_mean_abs_lead_bias = float(
        np.mean([abs(float(reference[f"W{lead}_bias"])) for lead in range(1, 7)])
    )
    ranking["mean_abs_lead_bias_reduction_pct"] = [
        _absolute_bias_reduction_percent(value, reference_mean_abs_lead_bias)
        for value in ranking.mean_abs_lead_bias
    ]
    ranking["all_lead_abs_bias_not_worse"] = [
        all(
            abs(float(row[f"W{lead}_bias"]))
            <= abs(float(reference[f"W{lead}_bias"])) + 1.0e-12
            for lead in range(1, 7)
        )
        for _, row in ranking.iterrows()
    ]
    ranking["pooled_rmse_guard"] = ranking.pooled_rmse <= float(
        reference.pooled_rmse
    ) * 1.005
    ranking["year_rmse_guard"] = [
        all(
            float(row[f"{year}_rmse"])
            <= float(reference[f"{year}_rmse"]) * 1.01
            for year in VALIDATION_YEARS
        )
        for _, row in ranking.iterrows()
    ]
    ranking["lead_rmse_guard"] = [
        all(
            float(row[f"W{lead}_rmse"])
            <= float(reference[f"W{lead}_rmse"]) * 1.015
            for lead in range(1, 7)
        )
        for _, row in ranking.iterrows()
    ]
    ranking["pooled_acc_guard"] = ranking.pooled_acc >= float(
        reference.pooled_acc
    ) - 0.005
    ranking["lead_acc_guard"] = [
        all(
            float(row[f"W{lead}_acc"])
            >= float(reference[f"W{lead}_acc"]) - 0.015
            for lead in range(1, 7)
        )
        for _, row in ranking.iterrows()
    ]
    ranking["pooled_mae_guard"] = ranking.pooled_mae <= float(
        reference.pooled_mae
    ) * 1.005
    ranking["beats_raw_pooled_rmse_and_acc"] = (
        (ranking.pooled_rmse < raw_summary["rmse"])
        & (ranking.pooled_acc > raw_summary["acc"])
    )
    ranking["bias_beats_raw_guard"] = ranking.pooled_bias.abs() <= abs(
        raw_summary["bias"]
    ) + 1.0e-12
    ranking["seed_robustness_guard"] = (
        ranking.seed_guard_passes >= ranking.seed_guard_required
    )
    ranking["qualifies"] = (
        (ranking.pooled_abs_bias_reduction_pct >= 20.0)
        & (ranking.mean_abs_lead_bias_reduction_pct >= 20.0)
        & ranking.bias_improves_both_years
        & (ranking.bias_leads_improved >= 4)
        & ranking.all_lead_abs_bias_not_worse
        & ranking.pooled_rmse_guard
        & ranking.year_rmse_guard
        & ranking.lead_rmse_guard
        & ranking.pooled_acc_guard
        & ranking.lead_acc_guard
        & ranking.pooled_mae_guard
        & ranking.beats_raw_pooled_rmse_and_acc
        & ranking.bias_beats_raw_guard
        & ranking.seed_robustness_guard
    )
    ranking["reference_configuration"] = reference_configuration
    ranking["raw_pooled_rmse"] = raw_summary["rmse"]
    ranking["raw_pooled_bias"] = raw_summary["bias"]
    ranking["raw_pooled_abs_bias"] = abs(raw_summary["bias"])
    ranking["raw_pooled_acc"] = raw_summary["acc"]
    ranking = ranking.sort_values(
        ["qualifies", "pooled_rmse", "pooled_abs_bias"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def select_configuration(
    ranking: pd.DataFrame,
    *,
    reference_configuration: str = REFERENCE_CONFIGURATION,
) -> tuple[str, pd.Series]:
    """Select only a qualified candidate; otherwise retain the control."""

    reference = ranking.loc[
        ranking.configuration.eq(reference_configuration)
    ]
    if len(reference) != 1:
        raise ValueError("ranking must contain exactly one reference row")
    qualified = ranking.loc[
        ranking.qualifies & ~ranking.configuration.eq(reference_configuration)
    ].sort_values(["pooled_rmse", "pooled_abs_bias"])
    if qualified.empty:
        return "no_candidate_qualified_reference_retained", reference.iloc[0]
    return "qualified_candidate_selected", qualified.iloc[0]


def paired_physical_deltas(
    case_metrics: pd.DataFrame,
    candidates: Sequence[BiasCandidate],
    reference_configuration: str = REFERENCE_CONFIGURATION,
) -> pd.DataFrame:
    ensemble = case_metrics.loc[
        case_metrics.member.eq("ensemble")
        & case_metrics.configuration.isin([candidate.name for candidate in candidates])
    ].copy()
    reference = ensemble.loc[
        ensemble.configuration.eq(reference_configuration)
    ].set_index(["case_id", "lead"])
    rows = []
    for name, frame in ensemble.groupby("configuration"):
        candidate = frame.set_index(["case_id", "lead"])
        common_index = candidate.index.intersection(reference.index)
        for metric in ("rmse", "mae", "bias", "acc"):
            candidate_values = candidate.loc[common_index, metric].to_numpy(float)
            reference_values = reference.loc[common_index, metric].to_numpy(float)
            rows.append(
                {
                    "configuration": name,
                    "reference_configuration": reference_configuration,
                    "metric": metric,
                    "candidate_mean": float(candidate_values.mean()),
                    "reference_mean": float(reference_values.mean()),
                    "mean_paired_delta_candidate_minus_reference": float(
                        np.mean(candidate_values - reference_values)
                    ),
                    "paired_case_leads": int(len(common_index)),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    output: Path,
    ranking: pd.DataFrame,
    history: pd.DataFrame,
    case_metrics: pd.DataFrame,
) -> None:
    candidate_names = ranking.configuration.tolist()
    selected = case_metrics.loc[
        (
            case_metrics.configuration.isin(candidate_names)
            & case_metrics.member.eq("ensemble")
        )
        | (
            case_metrics.configuration.isin(
                ("raw_fuxi", "log_anchor", "physical_recentered_anchor")
            )
            & case_metrics.member.eq("deterministic")
        )
    ]
    summary = selected.groupby(["configuration", "lead"], as_index=False).agg(
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        bias=("bias", "mean"),
        acc=("acc", "mean"),
    )
    labels = {
        "raw_fuxi": "Raw FuXi",
        "log_anchor": "Log anchor",
        "physical_recentered_anchor": "Recentered anchor",
        **dict(zip(ranking.configuration, ranking.label)),
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)
    for name, group in summary.groupby("configuration"):
        style = "--" if name.endswith("anchor") or name == "raw_fuxi" else "-"
        for axis, metric in zip(axes.flat, ("rmse", "mae", "acc", "bias")):
            axis.plot(
                group.lead,
                group[metric],
                marker="o",
                linewidth=1.5,
                linestyle=style,
                label=labels[name],
            )
    axes[0, 0].set_ylabel("RMSE (mm day$^{-1}$)")
    axes[0, 1].set_ylabel("MAE (mm day$^{-1}$)")
    axes[1, 0].set_ylabel("Spatial ACC")
    axes[1, 1].set_ylabel("Bias (mm day$^{-1}$)")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    for axis in axes.flat:
        axis.set_xticks(range(1, 7))
        axis.set_xlabel("Lead week")
        axis.grid(alpha=0.22)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.suptitle("FuXi–IMD bias-aware ablation · blocked 2018–2019 validation")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.png", dpi=240)
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=False)
    for axis, candidate in zip(axes.flat, CANDIDATES):
        subset = history.loc[history.configuration.eq(candidate.name)]
        for seed, frame in subset.groupby("seed"):
            epoch = frame.epoch.to_numpy() + 1
            axis.plot(epoch, frame.train_loss, "--", alpha=0.65, label=f"train {seed}")
            axis.plot(epoch, frame.validation_loss, alpha=0.85, label=f"val {seed}")
        axis.set_title(candidate.label, fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"{candidate.loss_kind} objective")
        axis.grid(alpha=0.22)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Training curves (objective scales are not compared across losses)")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output / "figures" / "02_training_validation_curves.png", dpi=240)
    figure.savefig(output / "figures" / "02_training_validation_curves.pdf")
    plt.close(figure)


def output_files(output: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def copy_sources(output: Path) -> None:
    sources = (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
        HERE / "slurm" / "run_imd_bias_aware_validation_sweep.sbatch",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "baselines.py",
        NEURAL_SRC / "fuxi_adapter" / "validation_sweep_models.py",
    )
    for source in sources:
        if source.exists():
            shutil.copy2(source, output / "code" / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", help="comma-separated configuration names")
    parser.add_argument("--seeds", help="comma-separated integer seeds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--reference-configuration", default=REFERENCE_CONFIGURATION
    )
    args = parser.parse_args()
    try:
        candidates = selected_candidates(args.configs)
        seeds = compact.selected_seeds(args.seeds, smoke=args.smoke)
    except ValueError as exc:
        parser.error(str(exc))
    if args.reference_configuration not in {candidate.name for candidate in candidates}:
        parser.error("reference configuration must be included in --configs")
    output = (
        args.output.resolve()
        if args.output
        else (
            RESULTS_ROOT
            / f"{'smoke' if args.smoke else 'full'}_"
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        ).resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "metrics", "figures", "code"):
        (output / name).mkdir()
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "status": "running",
        "created_utc": utc_now(),
        "purpose": "validation-only 2x2 bias anchor and loss ablation",
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "quarantined_years": list(QUARANTINED_YEARS),
        "test_predictions_created": False,
        "ranking_uses_physical_metrics_only": True,
        "objective_values_comparable_across_configurations": False,
        "model": asdict(MODEL_SPEC),
        "candidates": [asdict(candidate) for candidate in candidates],
        "seeds": list(seeds),
        "reference_configuration": args.reference_configuration,
        "smoke": bool(args.smoke),
        "command": sys.argv,
        "predeclared_guards": {
            "minimum_pooled_absolute_bias_reduction_pct": 20.0,
            "minimum_mean_absolute_lead_bias_reduction_pct": 20.0,
            "bias_improves_each_validation_year": True,
            "minimum_leads_with_absolute_bias_improvement": 4,
            "each_matching_lead_absolute_bias_not_worse": True,
            "maximum_pooled_rmse_regression_pct": 0.5,
            "maximum_each_year_rmse_regression_pct": 1.0,
            "maximum_each_lead_rmse_regression_pct": 1.5,
            "maximum_pooled_acc_drop": 0.005,
            "maximum_each_lead_acc_drop": 0.015,
            "maximum_pooled_mae_regression_pct": 0.5,
            "must_beat_raw_fuxi_pooled_rmse_and_acc": True,
            "pooled_absolute_bias_must_not_exceed_raw_fuxi": True,
            "seed_robustness": (
                "at least 2/3 matching seeds improve absolute bias with RMSE "
                "within 1% and ACC drop within 0.01; smoke requires 1/1"
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    try:
        prepared, normalization, preparation = prepare_data()
        (output / "normalization.json").write_text(
            json.dumps(compact._json_safe(normalization), indent=2) + "\n",
            encoding="utf-8",
        )
        current = preparation["current_correction"]
        recentered = preparation["recentered_fit"]
        np.savez_compressed(
            output / "models" / "anchor_parameters.npz",
            current_lead_month_residual=current.lead_month_residual,
            current_shrinkage=np.float32(current.shrinkage),
            recentered_lead_month_residual=recentered.correction.lead_month_residual,
            recenter_scalar_by_lead_month=recentered.scalar_by_lead_month,
            log_anchor_target_scale=prepared.anchors["log_anchor"].target_scale,
            recentered_anchor_target_scale=prepared.anchors[
                "physical_recentered"
            ].target_scale,
            bias_aware_training_scale=prepared.bias_scale,
        )
        preparation["recenter_diagnostics"].to_csv(
            output / "metrics" / "anchor_training_recenter_diagnostics.csv",
            index=False,
        )
        manifest.update(preparation["metadata"])
        manifest["bias_aware_training_scale_by_lead"] = [
            float(value) for value in prepared.bias_scale
        ]
        workers = args.workers
        if workers <= 0:
            workers = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        manifest["workers"] = int(workers)
        (output / "manifest.json").write_text(
            json.dumps(compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        run_parallel(
            candidates,
            seeds,
            prepared,
            output,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            workers=workers,
        )
        records, history, case_metrics, _ = aggregate_results(
            output, candidates, seeds, prepared
        )
        seed_guards = build_seed_physical_guards(
            case_metrics,
            candidates,
            reference_configuration=args.reference_configuration,
        )
        seed_guards.to_csv(
            output / "metrics" / "seed_physical_guards.csv", index=False
        )
        ranking = build_physical_ranking(
            records,
            case_metrics,
            candidates,
            reference_configuration=args.reference_configuration,
            seed_guards=seed_guards,
        )
        ranking.to_csv(output / "metrics" / "ranked_configurations.csv", index=False)
        candidate_vs_own_anchor(case_metrics, candidates).to_csv(
            output / "metrics" / "candidate_vs_own_anchor.csv", index=False
        )
        paired_physical_deltas(
            case_metrics, candidates, args.reference_configuration
        ).to_csv(
            output
            / "metrics"
            / f"paired_physical_deltas_vs_{args.reference_configuration}.csv",
            index=False,
        )
        save_plots(output, ranking, history, case_metrics)
        copy_sources(output)
        top_ranked = ranking.iloc[0]
        selection_status, selected = select_configuration(
            ranking,
            reference_configuration=args.reference_configuration,
        )
        selection_record = {
            "selection_status": selection_status,
            "selected_configuration": str(selected.configuration),
            "selected_label": str(selected.label),
            "selected_qualifies": bool(selected.qualifies),
            "reference_configuration": args.reference_configuration,
            "top_ranked_configuration": str(top_ranked.configuration),
        }
        (output / "metrics" / "selection.json").write_text(
            json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
        )
        readme = [
            "# FuXi–IMD bias-aware validation ablation",
            "",
            "Train: 2002–2017. Blocked validation: 2018–2019. "
            "No prediction or metric was created for 2020 onward.",
            "",
            "The four runs form an exact 2×2 anchor/loss ablation around the "
            "same width-24, batch-16 architecture.",
            "Training objective values are not compared across loss definitions; "
            "ranking uses physical IMD RMSE, MAE, ACC, and bias only.",
            "",
            f"Selection status: **{selection_status}**.",
            f"Selected configuration: **{selected.label}** "
            f"(`{selected.configuration}`).",
            f"Selected row passed every predeclared guard: "
            f"**{bool(selected.qualifies)}**.",
            (
                "No experimental candidate passed every guard, so the reference "
                "control is explicitly retained; the numerically first ranking row "
                "is not silently selected."
                if selection_status == "no_candidate_qualified_reference_retained"
                else "The selected experimental candidate passed every predeclared guard."
            ),
            f"Top numerical rank for audit: **{top_ranked.label}** "
            f"(`{top_ranked.configuration}`).",
            f"Selected pooled absolute-bias reduction vs control: "
            f"**{selected.pooled_abs_bias_reduction_pct:.2f}%**.",
            "",
            "Inspect `metrics/ranked_configurations.csv`, "
            "`metrics/validation_year_lead_matrix.csv`, "
            "`metrics/seed_physical_guards.csv`, "
            "`metrics/candidate_vs_own_anchor.csv`, "
            "`metrics/validation_intensity_strata.csv`, and the saved residuals.",
        ]
        (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "selection_status": selection_status,
                "selected_configuration": str(selected.configuration),
                "selected_qualifies": bool(selected.qualifies),
                "selected_pooled_rmse": float(selected.pooled_rmse),
                "selected_pooled_acc": float(selected.pooled_acc),
                "selected_pooled_bias": float(selected.pooled_bias),
                "selected_pooled_abs_bias_reduction_pct": float(
                    selected.pooled_abs_bias_reduction_pct
                ),
                "top_ranked_configuration": str(top_ranked.configuration),
                "test_predictions_created": False,
                "software": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda_visible_devices": torch.cuda.device_count(),
                },
            }
        )
        manifest["artifacts"] = {
            str(path.relative_to(output)): compact.sha256_file(path)
            for path in output_files(output)
        }
        (output / "manifest.json").write_text(
            json.dumps(compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        print(ranking.to_string(index=False), flush=True)
        print(f"PASS: bias-aware validation sweep complete: {output}", flush=True)
    except Exception:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "failure": traceback.format_exc(),
            }
        )
        (output / "manifest.json").write_text(
            json.dumps(compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
