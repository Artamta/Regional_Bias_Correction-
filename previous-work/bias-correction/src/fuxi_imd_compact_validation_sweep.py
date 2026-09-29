#!/usr/bin/env python3
"""Parallel validation-only sweep for compact FuXi-to-IMD residual adapters.

The sweep keeps the data split, target, anchor, loss, and verification contract
of the frozen full-context experiment fixed.  It varies only compact model
capacity and optimization regularization, plus one training-only smooth-noise
augmentation.  It never creates predictions for 2020 onward.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imd_attention_climatology as experiment  # noqa: E402
import fuxi_imerg_a100_big_temporal as engine  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402
from fuxi_adapter.validation_sweep_models import (  # noqa: E402
    CompactLeadReliabilityTemporalUNet,
    FixedCapacityPhysicalTemporalUNet,
    FixedClimatologyFactorized3DUNet,
    SixHeadTemporalAttentionUNet,
    SmoothNoiseTemporalAdapter,
)


base = engine.base
common = engine.common
RESULTS_ROOT = HERE / "results" / "fuxi_imd_compact_validation_sweep"
SEEDS = (42, 43, 44)
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
LEAD_WEIGHTS = (1.0 / 6.0,) * 6
TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
QUARANTINED_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
STANDARD_BACKBONE_CHANNELS = 11
MEMBER_BACKBONE_CHANNELS = 16
MEMBER_MODEL_KINDS = frozenset(
    {
        "temporal_member",
        "temporal_leadrank2",
        "temporal_spread_gate",
        "temporal_leadrank2_spread_gate",
    }
)
PHYSICAL_MODEL_KIND = "temporal_physical"
PHYSICAL_CACHE_FIELD_NAMES = (
    "tcwv_mean",
    "tcwv_spread",
    "q850_mean",
    "u850_mean",
    "v850_mean",
    "z500_mean",
    "msl_mean",
    "olr_mean",
    "q850_u850_flux_mean",
    "q850_v850_flux_mean",
)
# Keep a fixed nine-slot projection across all ablations.  TCWV spread is
# cached for diagnostics/future work but deliberately excluded from this first
# ensemble-mean screen.
PHYSICAL_PREDICTOR_NAMES = (
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
TCWV_PREDICTORS = ("tcwv_mean",)
MOISTURE_CIRCULATION_PREDICTORS = PHYSICAL_PREDICTOR_NAMES[:6]
FULL_PHYSICAL_PREDICTORS = PHYSICAL_PREDICTOR_NAMES


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    label: str
    model_kind: str = "temporal"
    base_channels: int = 16
    dropout: float = 0.30
    batch_size: int = 32
    learning_rate: float = 2.0e-4
    weight_decay: float = 2.0e-3
    noise_std: float = 0.0
    noise_probability: float = 0.0
    t2m_noise_std: float = 0.0
    coarse_noise_size: int = 5
    backbone_channels: int = STANDARD_BACKBONE_CHANNELS
    lead_rank: int = 0
    use_spread_gate: bool = False
    use_member_summaries: bool = False
    physical_predictors: tuple[str, ...] = ()


CANDIDATES = (
    SweepCandidate("control", "Current compact control"),
    SweepCandidate("width12", "Narrow temporal · width 12", base_channels=12),
    SweepCandidate("width20", "Medium temporal · width 20", base_channels=20),
    SweepCandidate("width24", "Medium temporal · width 24", base_channels=24),
    SweepCandidate("drop020", "Lower dropout · 0.20", dropout=0.20),
    SweepCandidate("drop040", "Strong dropout · 0.40", dropout=0.40),
    SweepCandidate("wd0005", "Lower weight decay · 5e-4", weight_decay=5.0e-4),
    SweepCandidate("wd0050", "Strong weight decay · 5e-3", weight_decay=5.0e-3),
    SweepCandidate("lr0001", "Lower learning rate · 1e-4", learning_rate=1.0e-4),
    SweepCandidate("batch16", "Smaller batch · 16", batch_size=16),
    SweepCandidate(
        "smooth_noise003",
        "Smooth FuXi input noise · 0.03",
        noise_std=0.03,
        noise_probability=0.50,
        t2m_noise_std=0.02,
    ),
    SweepCandidate(
        "factor3d24",
        "Factorized 3-D residual U-Net · width 24",
        model_kind="factor3d",
        base_channels=24,
        dropout=0.30,
    ),
    SweepCandidate(
        "factor3d16",
        "Factorized 3-D residual U-Net · width 16",
        model_kind="factor3d",
        base_channels=16,
        dropout=0.30,
    ),
    SweepCandidate(
        "factor3d24_batch16",
        "Factorized 3-D · width 24 · batch 16",
        model_kind="factor3d",
        base_channels=24,
        dropout=0.30,
        batch_size=16,
    ),
    SweepCandidate(
        "factor3d32",
        "Factorized 3-D residual U-Net · width 32",
        model_kind="factor3d",
        base_channels=32,
        dropout=0.30,
    ),
    SweepCandidate(
        "width24_batch16",
        "Temporal · width 24 · batch 16",
        base_channels=24,
        batch_size=16,
    ),
    SweepCandidate(
        "physical_control",
        "Physical-variable ablation · unchanged compact control",
        base_channels=24,
        batch_size=16,
    ),
    SweepCandidate(
        "physical_tcwv",
        "Physical-variable ablation · + TCWV",
        model_kind=PHYSICAL_MODEL_KIND,
        base_channels=24,
        batch_size=16,
        physical_predictors=TCWV_PREDICTORS,
    ),
    SweepCandidate(
        "physical_moisture_circulation",
        "Physical-variable ablation · moisture and circulation",
        model_kind=PHYSICAL_MODEL_KIND,
        base_channels=24,
        batch_size=16,
        physical_predictors=MOISTURE_CIRCULATION_PREDICTORS,
    ),
    SweepCandidate(
        "physical_full_compact",
        "Physical-variable ablation · full compact bank",
        model_kind=PHYSICAL_MODEL_KIND,
        base_channels=24,
        batch_size=16,
        physical_predictors=FULL_PHYSICAL_PREDICTORS,
    ),
    SweepCandidate(
        "width24_batch16_members",
        "Temporal · width 24 · batch 16 · member summaries",
        model_kind="temporal_member",
        base_channels=24,
        batch_size=16,
        backbone_channels=MEMBER_BACKBONE_CHANNELS,
        use_member_summaries=True,
    ),
    SweepCandidate(
        "member_leadrank2",
        "Member summaries · rank-2 lead adaptation",
        model_kind="temporal_leadrank2",
        base_channels=24,
        batch_size=16,
        backbone_channels=MEMBER_BACKBONE_CHANNELS,
        lead_rank=2,
        use_member_summaries=True,
    ),
    SweepCandidate(
        "member_spreadgate",
        "Member summaries · spread reliability gate",
        model_kind="temporal_spread_gate",
        base_channels=24,
        batch_size=16,
        backbone_channels=MEMBER_BACKBONE_CHANNELS,
        use_spread_gate=True,
        use_member_summaries=True,
    ),
    SweepCandidate(
        "member_leadrank2_spreadgate",
        "Member summaries · rank-2 lead adaptation · spread gate",
        model_kind="temporal_leadrank2_spread_gate",
        base_channels=24,
        batch_size=16,
        backbone_channels=MEMBER_BACKBONE_CHANNELS,
        lead_rank=2,
        use_spread_gate=True,
        use_member_summaries=True,
    ),
    SweepCandidate(
        "width24_batch16_6heads",
        "Temporal · width 24 · batch 16 · six lead heads",
        model_kind="temporal_6head",
        base_channels=24,
        batch_size=16,
    ),
    SweepCandidate(
        "width32_batch16",
        "Temporal · width 32 · batch 16",
        base_channels=32,
        batch_size=16,
    ),
    SweepCandidate(
        "width24_batch16_noise",
        "Temporal · width 24 · batch 16 · smooth noise",
        base_channels=24,
        batch_size=16,
        noise_std=0.03,
        noise_probability=0.50,
        t2m_noise_std=0.02,
    ),
    SweepCandidate(
        "strong_combo",
        "Strong dropout + decay",
        dropout=0.40,
        weight_decay=5.0e-3,
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


@dataclass
class PreparedData:
    features: np.ndarray
    target: np.ndarray
    target_scale: np.ndarray
    raw_fuxi: np.ndarray
    bias_baseline: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    valid_mask: np.ndarray
    weights: np.ndarray
    initializations: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    mean_to_anomaly_ratio: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_candidates(names: str | None) -> tuple[SweepCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(value.strip() for value in names.split(",") if value.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def selected_seeds(values: str | None, *, smoke: bool) -> tuple[int, ...]:
    """Resolve explicit screening seeds separately from smoke-test behavior."""

    if values is None:
        return (SEEDS[0],) if smoke else SEEDS
    raw_values = tuple(value.strip() for value in values.split(",") if value.strip())
    if not raw_values:
        raise ValueError("--seeds must contain at least one integer")
    try:
        seeds = tuple(int(value) for value in raw_values)
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds values must be nonnegative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds values must be unique")
    return seeds


def needs_member_summaries(candidates: Sequence[SweepCandidate]) -> bool:
    """Return whether at least one selected model consumes member summaries."""

    required = False
    for candidate in candidates:
        member_model = candidate.model_kind in MEMBER_MODEL_KINDS
        if member_model != candidate.use_member_summaries:
            raise ValueError(
                f"{candidate.name} has inconsistent model_kind="
                f"{candidate.model_kind!r} and use_member_summaries="
                f"{candidate.use_member_summaries}"
            )
        if candidate.use_member_summaries:
            if candidate.backbone_channels != MEMBER_BACKBONE_CHANNELS:
                raise ValueError(
                    f"{candidate.name} requests member summaries but has "
                    f"backbone_channels={candidate.backbone_channels}; expected "
                    f"{MEMBER_BACKBONE_CHANNELS}"
                )
            required = True
        elif candidate.backbone_channels != STANDARD_BACKBONE_CHANNELS:
            raise ValueError(
                f"{candidate.name} does not request member summaries but has "
                f"backbone_channels={candidate.backbone_channels}; expected "
                f"{STANDARD_BACKBONE_CHANNELS}"
            )
    return required


def needs_physical_predictors(candidates: Sequence[SweepCandidate]) -> bool:
    """Validate candidate contracts and report whether the cache is needed."""

    required = False
    allowed = set(PHYSICAL_PREDICTOR_NAMES)
    for candidate in candidates:
        physical_model = candidate.model_kind == PHYSICAL_MODEL_KIND
        requested = tuple(candidate.physical_predictors)
        if physical_model != bool(requested):
            raise ValueError(
                f"{candidate.name} has inconsistent model_kind="
                f"{candidate.model_kind!r} and physical_predictors={requested!r}"
            )
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(
                f"{candidate.name} requests unknown physical predictors: {unknown}"
            )
        if len(set(requested)) != len(requested):
            raise ValueError(
                f"{candidate.name} physical predictor names must be unique"
            )
        canonical = tuple(name for name in PHYSICAL_PREDICTOR_NAMES if name in requested)
        if requested != canonical:
            raise ValueError(
                f"{candidate.name} physical predictors must follow the fixed "
                f"projection order {PHYSICAL_PREDICTOR_NAMES}"
            )
        required |= physical_model
    return required


def _calendar_years(values: np.ndarray) -> np.ndarray:
    dates = np.asarray(values, dtype="datetime64[D]")
    return dates.astype("datetime64[Y]").astype(np.int64) + 1970


def append_physical_predictors(
    features: np.ndarray,
    normalization: Mapping[str, Any],
    predictors: Any,
    forecast: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Append leakage-safe physical fields from a 2002--2019-only cache.

    The cache is required to contain exactly the training and blocked-
    validation initializations, but may store them in any order.  Each raw
    field is aligned by initialization into the full forecast index.  Missing
    quarantined rows are represented as NaN while fitting and become exact
    zeros after normalization; no 2020+ physical input is requested.
    """

    initializations = np.asarray(forecast.initializations, dtype="datetime64[D]")
    cached_initializations = np.asarray(
        predictors.initializations, dtype="datetime64[D]"
    )
    selected_indices = np.sort(
        np.concatenate((train_indices, validation_indices)).astype(np.int64)
    )
    expected_initializations = initializations[selected_indices]
    if len(np.unique(cached_initializations)) != len(cached_initializations):
        raise base.DataContractError(
            "physical-predictor cache contains duplicate initializations"
        )
    if np.any(_calendar_years(cached_initializations) >= 2020):
        raise base.DataContractError(
            "physical-predictor cache must not contain quarantined 2020+ years"
        )
    if not np.array_equal(
        np.sort(cached_initializations), np.sort(expected_initializations)
    ):
        raise base.DataContractError(
            "physical-predictor cache must contain exactly the 2002-2019 "
            "training and validation initializations"
        )
    positions = np.searchsorted(initializations, cached_initializations)
    if (
        np.any(positions >= len(initializations))
        or not np.array_equal(initializations[positions], cached_initializations)
    ):
        raise base.DataContractError(
            "physical-predictor cache has an initialization absent from FuXi"
        )
    if not np.array_equal(np.sort(positions), selected_indices):
        raise base.DataContractError(
            "physical-predictor alignment differs from train and validation splits"
        )
    if not np.array_equal(
        np.asarray(predictors.latitude), np.asarray(forecast.latitude)
    ) or not np.array_equal(
        np.asarray(predictors.longitude), np.asarray(forecast.longitude)
    ):
        raise base.DataContractError(
            "physical-predictor latitude/longitude differs from FuXi"
        )

    fields = predictors.feature_fields
    missing = sorted(set(PHYSICAL_CACHE_FIELD_NAMES) - set(fields))
    if missing:
        raise base.DataContractError(
            f"physical-predictor cache is missing fields: {missing}"
        )
    expected_field_shape = (len(cached_initializations), 6, 27, 27)
    normalized_channels: list[np.ndarray] = []
    statistics: dict[str, Any] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    split_cache_masks = {
        "train": np.isin(positions, train_indices),
        "validation": np.isin(positions, validation_indices),
    }
    quarantined_indices = np.setdiff1d(
        np.arange(len(initializations), dtype=np.int64), selected_indices
    )
    for name in PHYSICAL_PREDICTOR_NAMES:
        raw = np.asarray(fields[name], dtype=np.float32)
        if raw.shape != expected_field_shape or not np.isfinite(raw).all():
            raise base.DataContractError(
                f"physical predictor {name} has invalid shape or values: {raw.shape}"
            )
        aligned = np.full(forecast.ensemble_mean.shape, np.nan, dtype=np.float32)
        aligned[positions] = raw
        normalized, field_stats = common.normalize_feature(
            aligned,
            train_indices,
            weights,
            preserve_full_domain=True,
        )
        if not np.isfinite(normalized[selected_indices]).all():
            raise base.DataContractError(
                f"normalized physical predictor {name} is incomplete in train/validation"
            )
        if np.any(normalized[quarantined_indices] != 0.0):
            raise base.DataContractError(
                f"physical predictor {name} is nonzero in quarantined years"
            )
        normalized_channels.append(normalized[:, :, None])
        statistics[name] = field_stats
        for split_name, cache_mask in split_cache_masks.items():
            values = raw[cache_mask].astype(np.float64, copy=False)
            diagnostic_rows.append(
                {
                    "split": split_name,
                    "feature": name,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    augmented = np.concatenate((features, *normalized_channels), axis=2).astype(
        np.float32
    )
    expected_shape = (
        len(initializations),
        6,
        features.shape[2] + len(PHYSICAL_PREDICTOR_NAMES),
        27,
        27,
    )
    if augmented.shape != expected_shape or not np.isfinite(augmented).all():
        raise base.DataContractError(
            f"unexpected augmented physical feature shape: {augmented.shape}"
        )
    updated = dict(normalization)
    updated.update(statistics)
    updated["input_channels"] = [
        *normalization["input_channels"],
        *PHYSICAL_PREDICTOR_NAMES,
    ]
    spatial_context = dict(normalization["spatial_context"])
    spatial_context["full_domain_channels"] = [
        *spatial_context["full_domain_channels"],
        *PHYSICAL_PREDICTOR_NAMES,
    ]
    updated["spatial_context"] = spatial_context
    updated["fuxi_physical_predictors"] = {
        "cache_scope": "train and validation initializations only (2002-2019)",
        "cache_initialization_count": int(len(cached_initializations)),
        "cache_latest_initialization": str(cached_initializations.max()),
        "cache_fields": list(PHYSICAL_CACHE_FIELD_NAMES),
        "model_feature_order": list(PHYSICAL_PREDICTOR_NAMES),
        "unused_cache_fields": ["tcwv_spread"],
        "normalization_fit": "training cases and positive target weights only",
        "normalization": "independent mean/std by lead week",
        "quarantined_rows": "not cached; exact zero after normalization",
        "source_fingerprint": _json_safe(predictors.source_fingerprint),
        "cache_path": (
            None if predictors.cache_path is None else str(predictors.cache_path)
        ),
        "cache_sha256": predictors.cache_sha256,
    }
    diagnostics = {
        "provenance": updated["fuxi_physical_predictors"],
        "statistics": diagnostic_rows,
    }
    return augmented, updated, diagnostics


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def member_summary_diagnostics(
    summaries: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> Mapping[str, Any]:
    """Describe raw member summaries without inspecting quarantined years."""

    names = tuple(common.MEMBER_SUMMARY_FEATURE_NAMES)
    fields = summaries.feature_fields
    if len(names) != 5 or tuple(fields) != names:
        raise base.DataContractError(
            "member-summary feature names and fields must contain five entries"
        )
    rows: list[dict[str, Any]] = []
    for split_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        for name in names:
            field = fields[name]
            values = np.asarray(field, dtype=np.float32)[indices]
            if values.ndim != 4 or values.shape[1:] != (6, 27, 27):
                raise base.DataContractError(
                    f"unexpected member-summary shape for {name}: {values.shape}"
                )
            for lead_index in (None, *range(6)):
                selected = values if lead_index is None else values[:, lead_index]
                finite = selected[np.isfinite(selected)].astype(np.float64, copy=False)
                if finite.size != selected.size:
                    raise base.DataContractError(
                        f"member summary {name} contains non-finite values in "
                        f"the {split_name} split"
                    )
                quantiles = np.quantile(finite, (0.01, 0.50, 0.99))
                rows.append(
                    {
                        "split": split_name,
                        "feature": name,
                        "lead": "all" if lead_index is None else f"W{lead_index + 1}",
                        "count": int(finite.size),
                        "mean": float(finite.mean()),
                        "std": float(finite.std(ddof=0)),
                        "minimum": float(finite.min()),
                        "p01": float(quantiles[0]),
                        "median": float(quantiles[1]),
                        "p99": float(quantiles[2]),
                        "maximum": float(finite.max()),
                    }
                )
    return {
        "provenance": {
            "raw_unnormalized": True,
            "statistics_scope": "train and validation only; full 27x27 regional grid",
            "feature_names": list(names),
            "definitions": _json_safe(common.MEMBER_SUMMARY_DEFINITIONS),
            "feature_channel_indices": list(
                range(STANDARD_BACKBONE_CHANNELS, MEMBER_BACKBONE_CHANNELS)
            ),
            "source_fingerprint": _json_safe(summaries.source_fingerprint),
            "cache_path": (
                None if summaries.cache_path is None else str(summaries.cache_path)
            ),
            "cache_sha256": summaries.cache_sha256,
        },
        "statistics": rows,
    }


def prepare_data(
    candidates: Sequence[SweepCandidate] = (),
) -> tuple[PreparedData, Mapping[str, Any], Mapping[str, Any]]:
    experiment.set_experiment_scope(
        all_weeks=True,
        large_model=False,
        regularized_large=False,
        full_fuxi_context=True,
    )
    experiment.configure_contract()
    print("Loading FuXi 2002-2021 once for the complete sweep...", flush=True)
    forecast = base.load_fuxi()
    print("Loading aligned IMD and training-only climatology...", flush=True)
    observations, climatology_daily, _, source_stores = experiment.load_imd(forecast)
    weights = experiment.load_imd_weights(forecast, observations.observation_fraction)
    support = weights > 0.0
    splits = base.split_indices(forecast.initializations)
    counts = {name: int(len(indices)) for name, indices in splits.items()}
    if counts != {"train": 560, "validation": 70, "test": 70}:
        raise base.DataContractError(f"unexpected split counts: {counts}")
    use_member_summaries = needs_member_summaries(candidates)
    use_physical_predictors = needs_physical_predictors(candidates)
    summaries = None
    if use_member_summaries:
        print(
            "Loading cached 51-member FuXi summary predictors for selected models...",
            flush=True,
        )
        summaries = common.load_fuxi_member_summaries(forecast)
        t2m_weekly = summaries.t2m_weekly_mean
        features, normalization, centre_difference = (
            experiment.build_climatology_features(
                forecast,
                observations,
                climatology_daily,
                weights,
                splits["train"],
                t2m_weekly,
                member_summaries=summaries,
                preserve_fuxi_context=True,
            )
        )
    else:
        # Preserve the established loader and exact 29-channel tensor whenever
        # the selected sweep contains no member-aware candidate.
        t2m_weekly = common.load_t2m_weekly(forecast)
        features, normalization, centre_difference = (
            experiment.build_climatology_features(
                forecast,
                observations,
                climatology_daily,
                weights,
                splits["train"],
                t2m_weekly,
                preserve_fuxi_context=True,
            )
        )
    physical_diagnostics = None
    if use_physical_predictors:
        print(
            "Loading 2002-2019 cached FuXi physical predictors for selected models...",
            flush=True,
        )
        cache_module = importlib.import_module("fuxi_physical_feature_cache")
        physical_predictors = cache_module.load_fuxi_physical_predictors(forecast)
        features, normalization, physical_diagnostics = append_physical_predictors(
            features,
            normalization,
            physical_predictors,
            forecast,
            np.asarray(splits["train"], dtype=np.int64),
            np.asarray(splits["validation"], dtype=np.int64),
            weights,
        )
    correction = fit_log_bias_correction(
        forecast.ensemble_mean[splits["train"]],
        observations.weekly_truth[splits["train"]],
        forecast.initializations[splits["train"]],
        support,
        shrinkage=10.0,
    )
    bias_baseline = apply_log_bias_correction(
        forecast.ensemble_mean, forecast.initializations, correction
    )
    valid_mask = np.broadcast_to(support[None, None], bias_baseline.shape).copy()
    target_scale = fit_anchored_target_scale(
        observations.weekly_truth[splits["train"]],
        bias_baseline[splits["train"]],
        weights,
        split_name="train",
        valid_mask=valid_mask[splits["train"]],
    )
    target = standardize_anchored_target(
        observations.weekly_truth,
        bias_baseline,
        target_scale,
        valid_mask=valid_mask,
    )
    mean_std = np.asarray(
        normalization["log_fuxi_mean"]["std_by_lead"], dtype=np.float32
    )
    anomaly_std = np.asarray(
        normalization["explicit_log_fuxi_anomaly"]["std_by_lead"],
        dtype=np.float32,
    )
    ratio = mean_std / anomaly_std
    if ratio.shape != (6,) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("invalid mean-to-anomaly noise normalization ratio")
    prepared = PreparedData(
        features=np.asarray(features, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        target_scale=np.asarray(target_scale, dtype=np.float32),
        raw_fuxi=np.asarray(forecast.ensemble_mean, dtype=np.float32),
        bias_baseline=np.asarray(bias_baseline, dtype=np.float32),
        truth=np.asarray(observations.weekly_truth, dtype=np.float32),
        climatology=np.asarray(observations.weekly_climatology, dtype=np.float32),
        valid_mask=valid_mask,
        weights=np.asarray(weights, dtype=np.float64),
        initializations=np.asarray(forecast.initializations, dtype="datetime64[D]"),
        train_indices=np.asarray(splits["train"], dtype=np.int64),
        validation_indices=np.asarray(splits["validation"], dtype=np.int64),
        mean_to_anomaly_ratio=ratio,
    )
    metadata = {
        "split_counts": counts,
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "quarantined_years": list(QUARANTINED_YEARS),
        "support_cells": int(support.sum()),
        "source_stores": list(source_stores),
        "climatology_centre_max_abs_difference": float(centre_difference),
        "feature_shape": list(features.shape),
        "target_shape": list(target.shape),
        "member_summaries_loaded": bool(use_member_summaries),
        "physical_predictors_loaded": bool(use_physical_predictors),
        "effective_feature_channels_by_candidate": {
            candidate.name: int(candidate.backbone_channels) for candidate in candidates
        },
        "physical_predictors_by_candidate": {
            candidate.name: list(candidate.physical_predictors)
            for candidate in candidates
        },
    }
    anchor = {
        "lead_month_residual": correction.lead_month_residual,
        "shrinkage": np.float32(correction.shrinkage),
        "target_scale": target_scale,
    }
    diagnostics = (
        None
        if summaries is None
        else member_summary_diagnostics(
            summaries,
            prepared.train_indices,
            prepared.validation_indices,
        )
    )
    return (
        prepared,
        normalization,
        {
            "metadata": metadata,
            "anchor": anchor,
            "member_diagnostics": diagnostics,
            "physical_diagnostics": physical_diagnostics,
        },
    )


def build_model(
    candidate: SweepCandidate,
    input_channels: int,
    mean_to_anomaly_ratio: np.ndarray,
) -> torch.nn.Module:
    if candidate.model_kind == "temporal":
        return SmoothNoiseTemporalAdapter(
            input_channels=input_channels,
            backbone_channels=STANDARD_BACKBONE_CHANNELS,
            base_channels=candidate.base_channels,
            dropout=candidate.dropout,
            noise_std=candidate.noise_std,
            noise_probability=candidate.noise_probability,
            mean_to_anomaly_ratio=mean_to_anomaly_ratio,
            t2m_noise_std=candidate.t2m_noise_std,
            coarse_size=candidate.coarse_noise_size,
        )
    if candidate.model_kind == "factor3d":
        return FixedClimatologyFactorized3DUNet(
            input_channels=input_channels,
            backbone_channels=STANDARD_BACKBONE_CHANNELS,
            base_channels=candidate.base_channels,
            dropout=candidate.dropout,
        )
    if candidate.model_kind == "temporal_6head":
        return SixHeadTemporalAttentionUNet(
            input_channels=input_channels,
            backbone_channels=STANDARD_BACKBONE_CHANNELS,
            base_channels=candidate.base_channels,
            dropout=candidate.dropout,
        )
    if candidate.model_kind == PHYSICAL_MODEL_KIND:
        if not candidate.physical_predictors:
            raise ValueError(
                f"{candidate.name} uses the physical model without predictors"
            )
        if input_channels < STANDARD_BACKBONE_CHANNELS + len(
            PHYSICAL_PREDICTOR_NAMES
        ):
            raise ValueError(
                f"{candidate.name} requires the appended physical predictor bank"
            )
        physical_start = input_channels - len(PHYSICAL_PREDICTOR_NAMES)
        active_slots = tuple(
            PHYSICAL_PREDICTOR_NAMES.index(name)
            for name in candidate.physical_predictors
        )
        return FixedCapacityPhysicalTemporalUNet(
            input_channels=input_channels,
            physical_channel_indices=tuple(
                range(physical_start, input_channels)
            ),
            active_physical_slots=active_slots,
            base_channels=candidate.base_channels,
            dropout=candidate.dropout,
        )
    if candidate.model_kind in MEMBER_MODEL_KINDS:
        if not candidate.use_member_summaries:
            raise ValueError(
                f"{candidate.name} uses a member-aware model kind without "
                "requesting member summaries"
            )
        if input_channels < candidate.backbone_channels:
            raise ValueError(
                f"{candidate.name} needs {candidate.backbone_channels} effective "
                f"channels but the feature tensor has only {input_channels}"
            )
        return CompactLeadReliabilityTemporalUNet(
            input_channels=input_channels,
            backbone_channels=candidate.backbone_channels,
            base_channels=candidate.base_channels,
            dropout=candidate.dropout,
            lead_rank=candidate.lead_rank,
            use_spread_gate=candidate.use_spread_gate,
        )
    raise ValueError(f"unknown model kind: {candidate.model_kind}")


def _worker(
    worker_index: int,
    device: str,
    tasks: Sequence[tuple[SweepCandidate, int]],
    prepared: PreparedData,
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
        train_data = common.make_dataset(
            prepared.train_indices,
            prepared.features,
            prepared.target,
            prepared.bias_baseline,
            base.ObservationData(
                weekly_truth=prepared.truth,
                weekly_climatology=prepared.climatology,
                observation_fraction=(prepared.weights > 0).astype(np.float32),
                source_stores=(),
            ),
            prepared.weights > 0,
        )
        validation_data = common.make_dataset(
            prepared.validation_indices,
            prepared.features,
            prepared.target,
            prepared.bias_baseline,
            base.ObservationData(
                weekly_truth=prepared.truth,
                weekly_climatology=prepared.climatology,
                observation_fraction=(prepared.weights > 0).astype(np.float32),
                source_stores=(),
            ),
            prepared.weights > 0,
        )
        for candidate, seed in tasks:
            print(
                f"[{device}] {candidate.name}, seed={seed}, "
                f"width={candidate.base_channels}, dropout={candidate.dropout}",
                flush=True,
            )
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            (run_directory / "logs").mkdir(parents=True, exist_ok=False)
            (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
            set_deterministic_seed(seed)
            model = build_model(
                candidate,
                prepared.features.shape[2],
                prepared.mean_to_anomaly_ratio,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result = train_anchored_model(
                model,
                train_data,
                validation_data,
                prepared.weights,
                prepared.target_scale,
                LEAD_WEIGHTS,
                LOSS_COEFFICIENTS,
                run_directory,
                seed=seed,
                device=device,
                batch_size=candidate.batch_size,
                max_epochs=2 if smoke else max_epochs,
                patience=1 if smoke else patience,
                learning_rate=candidate.learning_rate,
                weight_decay=candidate.weight_decay,
                smooth_l1_beta=1.0,
                num_workers=0,
                use_amp=True,
            )
            residual = predict(
                model,
                prepared.features[prepared.validation_indices],
                device=device,
                batch_size=32,
                use_amp=True,
            )
            if residual.shape != (len(prepared.validation_indices), 6, 27, 27):
                raise ValueError(
                    f"unexpected validation residual shape {residual.shape}"
                )
            np.save(run_directory / "validation_residual.npy", residual)
            checkpoint = run_directory / "checkpoints" / "best.pt"
            record = {
                "status": "complete",
                "candidate": asdict(candidate),
                "seed": int(seed),
                "device": device,
                "worker_index": int(worker_index),
                "parameter_count": int(parameter_count),
                "best_epoch_zero_based": int(result.best_epoch),
                "best_epoch_display": int(result.best_epoch + 1),
                "best_validation_loss": float(result.best_validation_loss),
                "elapsed_seconds": float(result.elapsed_seconds),
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "history": str(
                    (run_directory / "logs" / "training_history.csv").relative_to(
                        output
                    )
                ),
                "validation_residual": str(
                    (run_directory / "validation_residual.npy").relative_to(output)
                ),
            }
            (run_directory / "run_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
    except Exception:
        failure = output / f"worker_{worker_index}_failure.txt"
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run_parallel(
    candidates: Sequence[SweepCandidate],
    seeds: Sequence[int],
    prepared: PreparedData,
    output: Path,
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    workers: int,
) -> None:
    tasks = [(candidate, int(seed)) for candidate in candidates for seed in seeds]
    if not tasks:
        raise ValueError("sweep contains no tasks")
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _worker(0, device, tasks, prepared, output, max_epochs, patience, smoke)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
        raise RuntimeError(
            f"requested {workers} GPU workers but only {torch.cuda.device_count()} visible"
        )
    task_groups = [tasks[index::workers] for index in range(workers)]
    context = mp.get_context("spawn")
    processes = []
    for index, group in enumerate(task_groups):
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


def load_run_records(
    output: Path, candidates: Sequence[SweepCandidate], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    records = []
    for candidate in candidates:
        counts = set()
        for seed in seeds:
            path = (
                output / "models" / candidate.name / f"seed_{seed}" / "run_record.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                raise ValueError(f"incomplete run record: {path}")
            counts.add(int(record["parameter_count"]))
            records.append(record)
        if len(counts) != 1:
            raise ValueError(f"parameter count changed across seeds: {candidate.name}")
    return records


def residual_metrics(
    residual: np.ndarray,
    prepared: PreparedData,
    *,
    predictor: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    indices = prepared.validation_indices
    valid = prepared.valid_mask[indices]
    prediction = reconstruct_anchored_precipitation(
        prepared.bias_baseline[indices],
        residual,
        prepared.target_scale,
        valid_mask=valid,
    )
    climatology = prepared.climatology[indices]
    truth = prepared.truth[indices]
    case_metrics = compute_case_metrics(
        truth,
        prediction,
        truth - climatology,
        prediction - climatology,
        prepared.weights,
        predictor=predictor,
        case_ids=prepared.initializations[indices],
        leads=np.arange(1, 7),
        valid_mask=valid,
    )
    observations = base.ObservationData(
        weekly_truth=prepared.truth,
        weekly_climatology=prepared.climatology,
        observation_fraction=(prepared.weights > 0).astype(np.float32),
        source_stores=(),
    )
    score = dict(
        engine.composite_score(
            residual,
            indices,
            prepared.target,
            prepared.bias_baseline,
            observations,
            prepared.target_scale,
            prepared.weights,
            lead_weights=LEAD_WEIGHTS,
            loss_coefficients=LOSS_COEFFICIENTS,
        )
    )
    validation_years = pd.DatetimeIndex(prepared.initializations[indices]).year
    for year in VALIDATION_YEARS:
        year_mask = validation_years == year
        year_score = engine.composite_score(
            residual[year_mask],
            indices[year_mask],
            prepared.target,
            prepared.bias_baseline,
            observations,
            prepared.target_scale,
            prepared.weights,
            lead_weights=LEAD_WEIGHTS,
            loss_coefficients=LOSS_COEFFICIENTS,
        )
        score.update(
            {f"{year}_{name}": float(value) for name, value in year_score.items()}
        )
    for lead_index in range(6):
        lead_weights = tuple(1.0 if index == lead_index else 0.0 for index in range(6))
        lead_score = engine.composite_score(
            residual,
            indices,
            prepared.target,
            prepared.bias_baseline,
            observations,
            prepared.target_scale,
            prepared.weights,
            lead_weights=lead_weights,
            loss_coefficients=LOSS_COEFFICIENTS,
        )
        score.update(
            {
                f"W{lead_index + 1}_{name}": float(value)
                for name, value in lead_score.items()
            }
        )
    return score, case_metrics


def baseline_case_metrics(prepared: PreparedData, method: str) -> pd.DataFrame:
    indices = prepared.validation_indices
    if method == "raw_fuxi":
        prediction = prepared.raw_fuxi[indices]
    elif method == "log_bias":
        prediction = prepared.bias_baseline[indices]
    else:
        raise ValueError(method)
    truth = prepared.truth[indices]
    climatology = prepared.climatology[indices]
    return compute_case_metrics(
        truth,
        prediction,
        truth - climatology,
        prediction - climatology,
        prepared.weights,
        predictor=method,
        case_ids=prepared.initializations[indices],
        leads=np.arange(1, 7),
        valid_mask=prepared.valid_mask[indices],
    )


def summarize_case_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["year"] = pd.DatetimeIndex(result.case_id).year
    rows = []
    group_columns = ["configuration", "member", "year", "lead"]
    for keys, group in result.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                metric: float(group[metric].mean())
                for metric in ("rmse", "mae", "bias", "acc")
            }
        )
        row["cases"] = int(group.case_id.nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def aggregate_results(
    output: Path,
    candidates: Sequence[SweepCandidate],
    seeds: Sequence[int],
    prepared: PreparedData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = load_run_records(output, candidates, seeds)
    record_frame = pd.json_normalize(records, sep="_")
    record_frame.to_csv(output / "metrics" / "run_records.csv", index=False)
    histories = []
    all_case_metrics = []
    score_rows = []
    for candidate in candidates:
        residuals = []
        for seed in seeds:
            run_dir = output / "models" / candidate.name / f"seed_{seed}"
            history = pd.read_csv(run_dir / "logs" / "training_history.csv")
            history.insert(0, "seed", int(seed))
            history.insert(0, "configuration", candidate.name)
            histories.append(history)
            residual = np.load(run_dir / "validation_residual.npy")
            residuals.append(residual)
            score, case_metrics = residual_metrics(
                residual, prepared, predictor=f"{candidate.name}_seed_{seed}"
            )
            case_metrics.insert(0, "member", f"seed_{seed}")
            case_metrics.insert(0, "configuration", candidate.name)
            all_case_metrics.append(case_metrics)
            score_rows.append(
                {
                    "configuration": candidate.name,
                    "member": f"seed_{seed}",
                    **score,
                }
            )
        ensemble = np.mean(residuals, axis=0, dtype=np.float64).astype(np.float32)
        np.save(
            output / "models" / candidate.name / "validation_residual_ensemble.npy",
            ensemble,
        )
        score, case_metrics = residual_metrics(
            ensemble, prepared, predictor=f"{candidate.name}_ensemble"
        )
        case_metrics.insert(0, "member", "ensemble")
        case_metrics.insert(0, "configuration", candidate.name)
        all_case_metrics.append(case_metrics)
        score_rows.append(
            {"configuration": candidate.name, "member": "ensemble", **score}
        )
    for baseline_name in ("raw_fuxi", "log_bias"):
        baseline = baseline_case_metrics(prepared, baseline_name)
        baseline.insert(0, "member", "deterministic")
        baseline.insert(0, "configuration", baseline_name)
        all_case_metrics.append(baseline)
    history_frame = pd.concat(histories, ignore_index=True)
    history_frame.to_csv(output / "metrics" / "training_history_tidy.csv", index=False)
    case_frame = pd.concat(all_case_metrics, ignore_index=True)
    case_frame.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    score_frame = pd.DataFrame(score_rows)
    score_frame.to_csv(
        output / "metrics" / "validation_composite_scores.csv", index=False
    )
    matrix = summarize_case_metrics(case_frame)
    matrix.to_csv(output / "metrics" / "validation_year_lead_matrix.csv", index=False)
    return record_frame, history_frame, case_frame, score_frame


def build_ranking(
    records: pd.DataFrame,
    case_metrics: pd.DataFrame,
    composite_scores: pd.DataFrame,
    candidates: Sequence[SweepCandidate],
    *,
    reference_configuration: str = "control",
    minimum_loss_improvement: float = 0.0025,
) -> pd.DataFrame:
    candidate_names = {candidate.name for candidate in candidates}
    if reference_configuration not in candidate_names:
        raise ValueError(
            f"reference configuration {reference_configuration!r} is not in the sweep"
        )
    if minimum_loss_improvement < 0.0:
        raise ValueError("minimum_loss_improvement must be nonnegative")
    ensemble = case_metrics.loc[case_metrics.member.eq("ensemble")].copy()
    ensemble["year"] = pd.DatetimeIndex(ensemble.case_id).year
    rows = []
    for candidate in candidates:
        name = candidate.name
        seed_records = records.loc[records.candidate_name.eq(name)]
        metrics = ensemble.loc[ensemble.configuration.eq(name)]
        ensemble_score = composite_scores.loc[
            composite_scores.configuration.eq(name)
            & composite_scores.member.eq("ensemble")
        ].iloc[0]
        row: dict[str, Any] = {
            "configuration": name,
            "label": candidate.label,
            "model_kind": candidate.model_kind,
            "base_channels": candidate.base_channels,
            "dropout": candidate.dropout,
            "learning_rate": candidate.learning_rate,
            "weight_decay": candidate.weight_decay,
            "batch_size": candidate.batch_size,
            "noise_std": candidate.noise_std,
            "backbone_channels": candidate.backbone_channels,
            "lead_rank": candidate.lead_rank,
            "use_spread_gate": candidate.use_spread_gate,
            "use_member_summaries": candidate.use_member_summaries,
            "physical_predictors": ",".join(candidate.physical_predictors),
            "parameter_count": int(seed_records.parameter_count.iloc[0]),
            "mean_best_validation_loss": float(
                seed_records.best_validation_loss.mean()
            ),
            "std_best_validation_loss": float(
                seed_records.best_validation_loss.std(ddof=1)
            ),
            "ensemble_validation_loss": float(ensemble_score.composite_loss),
            "mean_runtime_seconds": float(seed_records.elapsed_seconds.mean()),
        }
        for year in VALIDATION_YEARS:
            row[f"{year}_composite_loss"] = float(
                ensemble_score[f"{year}_composite_loss"]
            )
        for lead in range(1, 7):
            row[f"W{lead}_composite_loss"] = float(
                ensemble_score[f"W{lead}_composite_loss"]
            )
        for metric in ("rmse", "mae", "bias", "acc"):
            row[f"pooled_{metric}"] = float(metrics[metric].mean())
            for year in VALIDATION_YEARS:
                row[f"{year}_{metric}"] = float(
                    metrics.loc[metrics.year.eq(year), metric].mean()
                )
            for lead in range(1, 7):
                row[f"W{lead}_{metric}"] = float(
                    metrics.loc[metrics.lead.eq(lead), metric].mean()
                )
        rows.append(row)
    ranking = pd.DataFrame(rows)
    reference = ranking.loc[ranking.configuration.eq(reference_configuration)].iloc[0]
    reference_seed = records.loc[
        records.candidate_name.eq(reference_configuration),
        ["seed", "best_validation_loss"],
    ].set_index("seed")
    beats = []
    for name in ranking.configuration:
        candidate_seed = records.loc[
            records.candidate_name.eq(name), ["seed", "best_validation_loss"]
        ].set_index("seed")
        paired = candidate_seed.join(
            reference_seed, lsuffix="_candidate", rsuffix="_reference"
        )
        beats.append(
            int(
                np.count_nonzero(
                    paired.best_validation_loss_candidate
                    < paired.best_validation_loss_reference
                )
            )
        )
    ranking["reference_configuration"] = reference_configuration
    ranking["seeds_beating_reference"] = beats
    ranking["validation_loss_delta_vs_reference"] = (
        ranking.mean_best_validation_loss - float(reference.mean_best_validation_loss)
    )
    ranking["ensemble_loss_delta_vs_reference"] = (
        ranking.ensemble_validation_loss - float(reference.ensemble_validation_loss)
    )
    ranking["pooled_rmse_skill_vs_reference_pct"] = (
        100.0
        * (float(reference.pooled_rmse) - ranking.pooled_rmse)
        / float(reference.pooled_rmse)
    )
    ranking["pooled_acc_delta_vs_reference"] = ranking.pooled_acc - float(
        reference.pooled_acc
    )
    ranking["pooled_abs_bias_delta_vs_reference"] = ranking.pooled_bias.abs() - abs(
        float(reference.pooled_bias)
    )
    ranking["improves_rmse_both_years"] = (
        ranking["2018_rmse"] < float(reference["2018_rmse"])
    ) & (ranking["2019_rmse"] < float(reference["2019_rmse"]))
    improving_leads = []
    maximum_rmse_regression = []
    minimum_acc_delta = []
    for _, row in ranking.iterrows():
        improving_leads.append(
            sum(
                row[f"W{lead}_composite_loss"]
                < float(reference[f"W{lead}_composite_loss"])
                for lead in range(1, 7)
            )
        )
        maximum_rmse_regression.append(
            max(
                100.0
                * (row[f"W{lead}_rmse"] - float(reference[f"W{lead}_rmse"]))
                / float(reference[f"W{lead}_rmse"])
                for lead in range(1, 7)
            )
        )
        minimum_acc_delta.append(
            min(
                row[f"W{lead}_acc"] - float(reference[f"W{lead}_acc"])
                for lead in range(1, 7)
            )
        )
    ranking["leads_improving_composite"] = improving_leads
    ranking["max_lead_rmse_regression_pct"] = maximum_rmse_regression
    ranking["min_lead_acc_delta"] = minimum_acc_delta
    ranking["qualifies"] = (
        (
            ranking.mean_best_validation_loss
            <= float(reference.mean_best_validation_loss) - minimum_loss_improvement
        )
        & (
            ranking.ensemble_validation_loss
            <= float(reference.ensemble_validation_loss) - minimum_loss_improvement
        )
        & (ranking.seeds_beating_reference >= 2)
        & ranking.improves_rmse_both_years
        & (ranking.pooled_acc_delta_vs_reference >= -0.002)
        & (ranking.pooled_abs_bias_delta_vs_reference <= 0.05)
        & (ranking.leads_improving_composite >= 4)
        & (ranking.max_lead_rmse_regression_pct <= 0.5)
        & (ranking.min_lead_acc_delta >= -0.01)
    )
    ranking = ranking.sort_values(
        ["qualifies", "mean_best_validation_loss"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def paired_deltas(
    case_metrics: pd.DataFrame,
    reference_configuration: str = "control",
) -> pd.DataFrame:
    ensemble = case_metrics.loc[case_metrics.member.eq("ensemble")].copy()
    reference = ensemble.loc[
        ensemble.configuration.eq(reference_configuration)
    ].set_index(["case_id", "lead"])
    if reference.empty:
        raise ValueError(
            f"reference configuration {reference_configuration!r} has no ensemble metrics"
        )
    rows = []
    for name, frame in ensemble.groupby("configuration"):
        candidate = frame.set_index(["case_id", "lead"])
        common_index = candidate.index.intersection(reference.index)
        for lead_scope, mask in (
            ("W1-W6", np.ones(len(common_index), dtype=bool)),
            *[
                (
                    f"W{week}",
                    np.asarray(common_index.get_level_values("lead") == week),
                )
                for week in range(1, 7)
            ],
        ):
            selected_index = common_index[mask]
            for metric in ("rmse", "mae", "bias", "acc"):
                candidate_values = candidate.loc[selected_index, metric].to_numpy(float)
                reference_values = reference.loc[selected_index, metric].to_numpy(float)
                if metric in ("rmse", "mae"):
                    effect = (
                        100.0
                        * (reference_values.mean() - candidate_values.mean())
                        / reference_values.mean()
                    )
                    units = "percent reduction"
                else:
                    effect = candidate_values.mean() - reference_values.mean()
                    units = "difference"
                rows.append(
                    {
                        "configuration": name,
                        "reference_configuration": reference_configuration,
                        "lead_scope": lead_scope,
                        "metric": metric,
                        "candidate_mean": float(candidate_values.mean()),
                        "reference_mean": float(reference_values.mean()),
                        "effect_vs_reference": float(effect),
                        "effect_units": units,
                        "paired_case_leads": int(len(selected_index)),
                    }
                )
    return pd.DataFrame(rows)


def save_plots(
    output: Path,
    ranking: pd.DataFrame,
    history: pd.DataFrame,
    case_metrics: pd.DataFrame,
    reference_configuration: str = "control",
) -> None:
    ordered = ranking.sort_values("mean_best_validation_loss", ascending=False)
    figure_height = max(3.2, 0.65 * len(ordered) + 1.8)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    positions = np.arange(len(ordered))
    colors = np.where(
        ordered.configuration.eq(reference_configuration),
        "#555555",
        np.where(ordered.qualifies, "#009E73", "#0072B2"),
    )
    for position, (_, row), color in zip(positions, ordered.iterrows(), colors):
        axis.errorbar(
            row.mean_best_validation_loss,
            position,
            xerr=(
                0.0
                if pd.isna(row.std_best_validation_loss)
                else row.std_best_validation_loss
            ),
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
        )
    axis.set_yticks(positions, labels=ordered.label)
    reference_loss = float(
        ranking.loc[
            ranking.configuration.eq(reference_configuration),
            "mean_best_validation_loss",
        ].iloc[0]
    )
    axis.axvline(reference_loss, color="#D55E00", linestyle="--", linewidth=1.5)
    minimum = float(
        (
            ordered.mean_best_validation_loss
            - ordered.std_best_validation_loss.fillna(0.0)
        ).min()
    )
    maximum = float(
        (
            ordered.mean_best_validation_loss
            + ordered.std_best_validation_loss.fillna(0.0)
        ).max()
    )
    margin = max(0.001, 0.12 * (maximum - minimum))
    axis.set_xlim(minimum - margin, maximum + margin)
    axis.set_xlabel(
        "Mean best validation composite loss across seeds (lower is better)"
    )
    axis.set_title("Compact FuXi–IMD validation sweep · 2018–2019")
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    figure.savefig(output / "figures" / "01_validation_loss_ranking.png", dpi=240)
    figure.savefig(output / "figures" / "01_validation_loss_ranking.pdf")
    plt.close(figure)

    top_names = ranking.head(min(5, len(ranking))).configuration.tolist()
    figure, axes = plt.subplots(
        1, len(top_names), figsize=(4.1 * len(top_names), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, top_names):
        subset = history.loc[history.configuration.eq(name)]
        for seed, seed_frame in subset.groupby("seed"):
            display_epoch = seed_frame.epoch.to_numpy() + 1
            axis.plot(
                display_epoch,
                seed_frame.train_loss,
                linestyle="--",
                linewidth=1.1,
                alpha=0.65,
                label=f"train {seed}",
            )
            axis.plot(
                display_epoch,
                seed_frame.validation_loss,
                linewidth=1.5,
                alpha=0.85,
                label=f"val {seed}",
            )
        label = ranking.loc[ranking.configuration.eq(name), "label"].iloc[0]
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Composite objective")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Training and clean blocked-validation curves", fontweight="bold", y=0.995
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    figure.savefig(output / "figures" / "02_top_training_curves.png", dpi=240)
    figure.savefig(output / "figures" / "02_top_training_curves.pdf")
    plt.close(figure)

    top_model_names = ranking.head(min(3, len(ranking))).configuration.tolist()
    selected = case_metrics.loc[
        (
            case_metrics.configuration.isin(top_model_names)
            & case_metrics.member.eq("ensemble")
        )
        | (
            case_metrics.configuration.isin(("raw_fuxi", "log_bias"))
            & case_metrics.member.eq("deterministic")
        )
    ].copy()
    lead_summary = selected.groupby(["configuration", "lead"], as_index=False).agg(
        rmse=("rmse", "mean"), acc=("acc", "mean"), bias=("bias", "mean")
    )
    label_lookup = {
        "raw_fuxi": "Raw FuXi",
        "log_bias": "Log-bias",
        **dict(zip(ranking.configuration, ranking.label)),
    }
    color_lookup = {
        "raw_fuxi": "#555555",
        "log_bias": "#CC79A7",
        **{
            name: color
            for name, color in zip(top_model_names, ("#0072B2", "#009E73", "#D55E00"))
        },
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.9))
    for name, group in lead_summary.groupby("configuration"):
        for axis, metric in zip(axes, ("rmse", "acc", "bias")):
            axis.plot(
                group.lead,
                group[metric],
                marker="o",
                linewidth=1.8,
                label=label_lookup[name],
                color=color_lookup[name],
            )
    axes[0].set_ylabel("RMSE (mm day$^{-1}$; lower is better)")
    axes[1].set_ylabel("Spatial ACC (higher is better)")
    axes[2].set_ylabel("Bias (mm day$^{-1}$; zero is best)")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    for axis in axes:
        axis.set_xlabel("Lead week")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    figure.suptitle(
        "Blocked validation skill by lead · 2018–2019",
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    figure.savefig(output / "figures" / "03_validation_metrics_by_lead.png", dpi=240)
    figure.savefig(output / "figures" / "03_validation_metrics_by_lead.pdf")
    plt.close(figure)


def copy_sources(output: Path) -> None:
    sources = (
        Path(__file__),
        NEURAL_SRC / "fuxi_adapter" / "validation_sweep_models.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        SOURCE_ROOT / "fuxi_imd_attention_climatology.py",
        SOURCE_ROOT / "fuxi_physical_feature_cache.py",
    )
    for source in sources:
        if source.exists():
            shutil.copy2(source, output / "code" / source.name)


def output_files(output: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def replot_existing(output: Path) -> None:
    """Refresh figures and hashes without retraining an already complete sweep."""

    output = output.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"cannot replot an incomplete sweep: {output}")
    metrics = output / "metrics"
    save_plots(
        output,
        pd.read_csv(metrics / "ranked_configurations.csv"),
        pd.read_csv(metrics / "training_history_tidy.csv"),
        pd.read_csv(metrics / "validation_case_metrics.csv"),
        reference_configuration=manifest.get("reference_configuration", "control"),
    )
    copy_sources(output)
    manifest["plots_refreshed_utc"] = utc_now()
    manifest["artifacts"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in output_files(output)
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", help="comma-separated configuration names")
    parser.add_argument(
        "--seeds",
        help=(
            "comma-separated integer seeds; for example, --seeds 42 runs a "
            "full-duration one-seed screening experiment (unlike --smoke)"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--reference-configuration", default="control")
    parser.add_argument("--minimum-loss-improvement", type=float, default=0.0025)
    args = parser.parse_args()
    candidates = selected_candidates(args.configs)
    if args.reference_configuration not in {candidate.name for candidate in candidates}:
        parser.error("--reference-configuration must be included in --configs")
    if args.minimum_loss_improvement < 0.0:
        parser.error("--minimum-loss-improvement must be nonnegative")
    try:
        seeds = selected_seeds(args.seeds, smoke=args.smoke)
    except ValueError as exc:
        parser.error(str(exc))
    output = (
        args.output.resolve()
        if args.output
        else (
            RESULTS_ROOT / f"full_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        ).resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "metrics", "figures", "code"):
        (output / name).mkdir()
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "status": "running",
        "created_utc": utc_now(),
        "purpose": "validation-only compact hyperparameter and architecture sweep",
        "test_predictions_created": False,
        "loss_coefficients": LOSS_COEFFICIENTS,
        "lead_weights": list(LEAD_WEIGHTS),
        "candidates": [asdict(candidate) for candidate in candidates],
        "seeds": list(seeds),
        "reference_configuration": args.reference_configuration,
        "minimum_loss_improvement": args.minimum_loss_improvement,
        "smoke": bool(args.smoke),
        "training_mode": (
            "smoke"
            if args.smoke
            else ("full_duration_screening" if len(seeds) == 1 else "full")
        ),
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    try:
        prepared, normalization, preparation = prepare_data(candidates)
        (output / "normalization.json").write_text(
            json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
        )
        member_diagnostics = preparation.get("member_diagnostics")
        if member_diagnostics is not None:
            (output / "metrics" / "member_summary_diagnostics.json").write_text(
                json.dumps(member_diagnostics, indent=2) + "\n", encoding="utf-8"
            )
            pd.DataFrame(member_diagnostics["statistics"]).to_csv(
                output / "metrics" / "member_summary_diagnostics.csv", index=False
            )
        physical_diagnostics = preparation.get("physical_diagnostics")
        if physical_diagnostics is not None:
            (output / "metrics" / "physical_predictor_diagnostics.json").write_text(
                json.dumps(physical_diagnostics, indent=2) + "\n", encoding="utf-8"
            )
            pd.DataFrame(physical_diagnostics["statistics"]).to_csv(
                output / "metrics" / "physical_predictor_diagnostics.csv",
                index=False,
            )
        np.savez_compressed(
            output / "models" / "log_bias_anchor.npz", **preparation["anchor"]
        )
        manifest.update(preparation["metadata"])
        worker_count = args.workers
        if worker_count <= 0:
            worker_count = (
                min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            )
        manifest["workers"] = int(worker_count)
        run_parallel(
            candidates,
            seeds,
            prepared,
            output,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            workers=worker_count,
        )
        records, history, case_metrics, composite_scores = aggregate_results(
            output, candidates, seeds, prepared
        )
        ranking = build_ranking(
            records,
            case_metrics,
            composite_scores,
            candidates,
            reference_configuration=args.reference_configuration,
            minimum_loss_improvement=args.minimum_loss_improvement,
        )
        ranking.to_csv(output / "metrics" / "ranked_configurations.csv", index=False)
        paired = paired_deltas(case_metrics, args.reference_configuration)
        paired.to_csv(
            output / "metrics" / f"paired_deltas_vs_{args.reference_configuration}.csv",
            index=False,
        )
        save_plots(
            output,
            ranking,
            history,
            case_metrics,
            reference_configuration=args.reference_configuration,
        )
        copy_sources(output)
        winner = ranking.iloc[0]
        readme = [
            "# Compact FuXi–IMD validation sweep",
            "",
            "Validation-only experiment: 2002–2017 train and 2018–2019 validation.",
            "No prediction or metric was created for 2020 onward.",
            "",
            f"Reference configuration: `{args.reference_configuration}`.",
            f"Top ranked configuration: **{winner.label}** (`{winner.configuration}`).",
            f"Mean best seed validation loss: **{winner.mean_best_validation_loss:.6f}**.",
            f"Qualified under all predeclared guards: **{bool(winner.qualifies)}**.",
            "",
            "See `metrics/ranked_configurations.csv`, `metrics/validation_year_lead_matrix.csv`, and the figures.",
        ]
        (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "top_configuration": str(winner.configuration),
                "top_mean_best_validation_loss": float(
                    winner.mean_best_validation_loss
                ),
                "top_qualifies": bool(winner.qualifies),
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
            str(path.relative_to(output)): sha256_file(path)
            for path in output_files(output)
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(ranking.to_string(index=False), flush=True)
        print(f"PASS: compact validation sweep complete: {output}", flush=True)
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
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
