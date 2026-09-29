#!/usr/bin/env python
"""Compare normal and forecast-conditioned attention climatology using IMD.

Both candidates use the same scope-configurable temporal adapter, log-bias
anchor, loss, seeds, and split by default.  ``--training-anchor raw_fuxi``
provides a one-factor no-log-bias ablation: the identical neural models learn
log-rainfall residuals around raw FuXi while the fitted log-bias forecast is
retained only as a reporting baseline. The attention candidate can choose a convex
combination of nine training-only IMD calendar-climatology maps. The fixed
31-day IMD climatology remains the common anomaly reference; absolute-field
PCC is reported separately without a climatology.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imerg_a100_big_temporal as engine  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.metrics import (  # noqa: E402
    compute_case_metrics,
    weighted_spatial_acc,
)

base = engine.base
common = engine.common
diagnostics = engine.diagnostics

TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
TEST_YEARS = (2020, 2021)
OFFSETS_DAYS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)
ACTIVE_LEADS = (2, 3, 4, 5)
ACTIVE_WEEKS = (3, 4, 5, 6)
ACTIVE_SCOPE = "W3-W6"
INACTIVE_LEAD_COUNT = 2
LEAD_WEIGHTS = (0.0, 0.0, 0.25, 0.25, 0.25, 0.25)
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
TRAINING_ANCHORS = ("log_bias", "raw_fuxi")
VALIDATION_SCORE_COLUMN = "w3_w6_case_mean_rmse"
RESULTS_ROOT = HERE / "results" / "fuxi_imd_attention_climatology_w3w6"
IMD_DAILY = (
    base.BENCHMARK_ROOT / "observations/ground_truth_v1/daily/imd/tp/india_1p5_27x27_v1"
)

NORMAL = engine.Candidate(
    "normal_climo_model",
    "Normal-climatology temporal (W3–W6)",
    "fixed_climatology_week36",
    32,
    2.0e-4,
    2.0e-3,
    0.30,
    "#0072B2",
)
ATTENTION = engine.Candidate(
    "attention_climo_model",
    "Attention-climatology temporal (W3–W6)",
    "attention_climatology_week36",
    32,
    2.0e-4,
    2.0e-3,
    0.30,
    "#009E73",
)
CANDIDATES = (NORMAL, ATTENTION)
METHOD_ORDER = (
    "raw_fuxi",
    "log_bias",
    NORMAL.name,
    ATTENTION.name,
    "selected_model",
)
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    NORMAL.name: NORMAL.label,
    ATTENTION.name: ATTENTION.label,
    "selected_model": "Validation-selected",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#CC79A7",
    NORMAL.name: NORMAL.color,
    ATTENTION.name: ATTENTION.color,
    "selected_model": "#D55E00",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    NORMAL.name: "^",
    ATTENTION.name: "P",
    "selected_model": "*",
}


def set_experiment_scope(
    *,
    all_weeks: bool,
    large_model: bool = False,
    regularized_large: bool = False,
    full_fuxi_context: bool = False,
) -> None:
    """Select W3--W6 or all-week training without mixing result archives."""
    global ACTIVE_LEADS, ACTIVE_WEEKS, ACTIVE_SCOPE, INACTIVE_LEAD_COUNT
    global LEAD_WEIGHTS, VALIDATION_SCORE_COLUMN, RESULTS_ROOT
    global NORMAL, ATTENTION, CANDIDATES, METHOD_LABELS
    if regularized_large:
        large_model = True
    if large_model and not all_weeks:
        raise ValueError("the large-model experiment requires all-week training")
    if full_fuxi_context and (not all_weeks or large_model):
        raise ValueError(
            "full FuXi context is a compact all-week experiment; use --all-weeks "
            "without a large-model flag"
        )
    if not all_weeks:
        return
    ACTIVE_LEADS = (0, 1, 2, 3, 4, 5)
    ACTIVE_WEEKS = (1, 2, 3, 4, 5, 6)
    ACTIVE_SCOPE = "W1-W6"
    INACTIVE_LEAD_COUNT = 0
    LEAD_WEIGHTS = (1.0 / 6.0,) * 6
    VALIDATION_SCORE_COLUMN = "w1_w6_case_mean_rmse"
    RESULTS_ROOT = HERE / "results" / "fuxi_imd_attention_climatology_allweeks"
    NORMAL = engine.Candidate(
        "normal_climo_model",
        "Normal-climatology temporal (W1–W6)",
        "fixed_climatology_allweeks",
        32,
        2.0e-4,
        2.0e-3,
        0.30,
        "#0072B2",
    )
    ATTENTION = engine.Candidate(
        "attention_climo_model",
        "Attention-climatology temporal (W1–W6)",
        "attention_climatology_allweeks",
        32,
        2.0e-4,
        2.0e-3,
        0.30,
        "#009E73",
    )
    CANDIDATES = (NORMAL, ATTENTION)
    METHOD_LABELS = {
        "raw_fuxi": "Raw FuXi",
        "log_bias": "Log-bias",
        NORMAL.name: NORMAL.label,
        ATTENTION.name: ATTENTION.label,
        "selected_model": "Validation-selected",
    }
    if large_model:
        RESULTS_ROOT = HERE / "results" / "fuxi_imd_attention_climatology_big_allweeks"
        NORMAL = engine.Candidate(
            "normal_climo_model",
            "Large normal-climatology temporal (W1–W6)",
            "fixed_climatology_big_allweeks",
            32,
            1.5e-4,
            1.0e-3,
            0.25,
            "#0072B2",
        )
        ATTENTION = engine.Candidate(
            "attention_climo_model",
            "Large attention-climatology temporal (W1–W6)",
            "attention_climatology_big_allweeks",
            32,
            1.5e-4,
            1.0e-3,
            0.25,
            "#009E73",
        )
        CANDIDATES = (NORMAL, ATTENTION)
        METHOD_LABELS = {
            "raw_fuxi": "Raw FuXi",
            "log_bias": "Log-bias",
            NORMAL.name: NORMAL.label,
            ATTENTION.name: ATTENTION.label,
            "selected_model": "Validation-selected",
        }
    if regularized_large:
        RESULTS_ROOT = (
            HERE / "results" / "fuxi_imd_attention_climatology_big_allweeks_regularized"
        )
        NORMAL = engine.Candidate(
            "normal_climo_model",
            "Regularized large normal-climatology temporal (W1–W6)",
            "fixed_climatology_big_allweeks_regularized",
            32,
            1.5e-4,
            3.0e-3,
            0.30,
            "#0072B2",
        )
        ATTENTION = engine.Candidate(
            "attention_climo_model",
            "Regularized large attention-climatology temporal (W1–W6)",
            "attention_climatology_big_allweeks_regularized",
            32,
            1.5e-4,
            3.0e-3,
            0.30,
            "#009E73",
        )
        CANDIDATES = (NORMAL, ATTENTION)
        METHOD_LABELS = {
            "raw_fuxi": "Raw FuXi",
            "log_bias": "Log-bias",
            NORMAL.name: NORMAL.label,
            ATTENTION.name: ATTENTION.label,
            "selected_model": "Validation-selected",
        }
    if full_fuxi_context:
        RESULTS_ROOT = HERE / "results" / "fuxi_imd_full_context_compact_allweeks"
        NORMAL = engine.Candidate(
            "normal_climo_model",
            "Full-context normal-climatology temporal (W1–W6)",
            "fixed_climatology_allweeks",
            32,
            2.0e-4,
            2.0e-3,
            0.30,
            "#0072B2",
        )
        ATTENTION = engine.Candidate(
            "attention_climo_model",
            "Full-context attention-climatology temporal (W1–W6)",
            "attention_climatology_allweeks",
            32,
            2.0e-4,
            2.0e-3,
            0.30,
            "#009E73",
        )
        CANDIDATES = (NORMAL, ATTENTION)
        METHOD_LABELS = {
            "raw_fuxi": "Raw FuXi",
            "log_bias": "Log-bias",
            NORMAL.name: NORMAL.label,
            ATTENTION.name: ATTENTION.label,
            "selected_model": "Validation-selected",
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_training_baseline(
    anchor_kind: str,
    raw_fuxi: np.ndarray,
    log_bias: np.ndarray,
) -> np.ndarray:
    """Return the reconstruction baseline for the declared one-factor ablation."""
    if anchor_kind not in TRAINING_ANCHORS:
        raise ValueError(
            f"unknown training anchor {anchor_kind!r}; expected {TRAINING_ANCHORS}"
        )
    raw = np.asarray(raw_fuxi, dtype=np.float32)
    corrected = np.asarray(log_bias, dtype=np.float32)
    if raw.shape != corrected.shape or raw.ndim != 4:
        raise base.DataContractError(
            "raw FuXi and log-bias baselines must be matching four-dimensional fields"
        )
    selected = corrected if anchor_kind == "log_bias" else raw
    if not np.isfinite(selected).all() or np.any(selected < 0.0):
        raise base.DataContractError(
            f"{anchor_kind} training baseline must be finite and nonnegative"
        )
    return selected.copy()


def configure_contract() -> None:
    base.TRAIN_YEARS = TRAIN_YEARS
    base.VALIDATION_YEARS = VALIDATION_YEARS
    base.TEST_YEARS = TEST_YEARS
    base.ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS
    engine.CANDIDATES = CANDIDATES
    engine.CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
    engine.METHOD_ORDER = METHOD_ORDER
    engine.METHOD_LABELS = METHOD_LABELS
    engine.METHOD_COLORS = METHOD_COLORS
    engine.METHOD_MARKERS = METHOD_MARKERS
    engine.ACTIVE_LEADS = ACTIVE_LEADS
    engine.LEAD_WEIGHTS = LEAD_WEIGHTS
    engine.LOSS_COEFFICIENTS = LOSS_COEFFICIENTS
    diagnostics.METHOD_ORDER = METHOD_ORDER
    diagnostics.METHOD_LABELS = METHOD_LABELS
    diagnostics.METHOD_COLORS = METHOD_COLORS
    diagnostics.METHOD_MARKERS = METHOD_MARKERS
    diagnostics.PLOT_METHODS = METHOD_ORDER[:-1]
    diagnostics.LEAD_SCOPES = {
        **diagnostics.LEAD_SCOPES,
        ACTIVE_SCOPE: ACTIVE_WEEKS,
    }
    diagnostics.ALL_COMPARISONS = (
        ("log_bias", "raw_fuxi"),
        (NORMAL.name, "raw_fuxi"),
        (ATTENTION.name, "raw_fuxi"),
        ("selected_model", "raw_fuxi"),
        (NORMAL.name, "log_bias"),
        (ATTENTION.name, "log_bias"),
        (ATTENTION.name, NORMAL.name),
        ("selected_model", "log_bias"),
        ("selected_model", NORMAL.name),
    )


def load_imd(
    forecast: base.ForecastData,
) -> tuple[base.ObservationData, np.ndarray, np.ndarray, tuple[str, ...]]:
    values_by_year = []
    dates_by_year = []
    source_stores = []
    observation_fraction = None
    for year in base.ALL_YEARS:
        store = IMD_DAILY / f"{year}.zarr"
        if not (store / ".zmetadata").is_file():
            raise FileNotFoundError(store)
        with xr.open_zarr(store, consolidated=True) as dataset:
            if (
                dataset.attrs.get("source") != "imd"
                or dataset.attrs.get("units") != "mm day-1"
            ):
                raise base.DataContractError(f"unexpected IMD metadata: {store}")
            if not np.array_equal(dataset.latitude.values, forecast.latitude):
                raise base.DataContractError(f"IMD latitude differs: {store}")
            if not np.array_equal(dataset.longitude.values, forecast.longitude):
                raise base.DataContractError(f"IMD longitude differs: {store}")
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            fraction = base.collapse_fraction(dataset, (27, 27))
            if observation_fraction is None:
                observation_fraction = fraction
            elif not np.allclose(
                observation_fraction, fraction, rtol=0.0, atol=1.0e-7, equal_nan=True
            ):
                raise base.DataContractError("IMD support differs between years")
            if values.shape != (dates.size, 27, 27):
                raise base.DataContractError(f"unexpected IMD shape: {store}")
            values_by_year.append(values)
            dates_by_year.append(dates)
            source_stores.append(str(store))
    assert observation_fraction is not None
    dates = np.concatenate(dates_by_year)
    values = np.concatenate(values_by_year)
    order = np.argsort(dates)
    dates = dates[order]
    values = values[order]
    requested = forecast.valid_dates.reshape(-1)
    positions = np.searchsorted(dates, requested)
    if np.any(positions >= dates.size) or not np.array_equal(
        dates[positions], requested
    ):
        raise base.DataContractError("one or more IMD verification dates are missing")
    daily = values[positions].reshape(*forecast.valid_dates.shape, 27, 27)
    weekly_truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    support = observation_fraction > 0.0
    climatology_daily = base.build_training_climatology(dates, values, support)
    climatology_positions = base.calendar_positions(forecast.valid_dates)
    weekly_climatology = np.mean(
        climatology_daily[climatology_positions], axis=2, dtype=np.float64
    ).astype(np.float32)
    observations = base.ObservationData(
        weekly_truth=weekly_truth,
        weekly_climatology=weekly_climatology,
        observation_fraction=observation_fraction,
        source_stores=tuple(source_stores),
    )
    return observations, climatology_daily, dates, tuple(source_stores)


def load_imd_weights(
    forecast: base.ForecastData, observation_fraction: np.ndarray
) -> np.ndarray:
    with xr.open_zarr(base.SPATIAL_STORE, consolidated=True) as dataset:
        if not np.array_equal(dataset.latitude.values, forecast.latitude):
            raise base.DataContractError("area-weight latitude differs")
        if not np.array_equal(dataset.longitude.values, forecast.longitude):
            raise base.DataContractError("area-weight longitude differs")
        india_area = np.asarray(dataset.india_area_weight_km2.load(), dtype=np.float64)
    weights = india_area * np.asarray(observation_fraction, dtype=np.float64)
    weights[~np.isfinite(weights) | (weights <= 0.0)] = 0.0
    if np.count_nonzero(weights) != 171:
        raise base.DataContractError(
            f"expected 171 supported IMD cells, found {np.count_nonzero(weights)}"
        )
    return weights


def build_climatology_features(
    forecast: base.ForecastData,
    observations: base.ObservationData,
    climatology_daily: np.ndarray,
    weights: np.ndarray,
    train_indices: np.ndarray,
    t2m_weekly: np.ndarray,
    *,
    member_summaries: common.FuxiMemberSummaries | None = None,
    preserve_fuxi_context: bool = False,
) -> tuple[np.ndarray, Mapping[str, object], float]:
    standard, normalization = common.make_features(
        forecast,
        observations,
        weights,
        train_indices,
        t2m_weekly,
        preserve_fuxi_context=preserve_fuxi_context,
    )
    support = weights > 0.0
    member_channels: list[np.ndarray] = []
    member_statistics: dict[str, Mapping[str, object]] = {}
    member_names: tuple[str, ...] = ()
    if member_summaries is not None:
        common.validate_fuxi_member_summaries(member_summaries, forecast)
        if not np.array_equal(member_summaries.t2m_weekly_mean, t2m_weekly):
            raise base.DataContractError(
                "member-summary T2M mean differs from the supplied T2M predictor"
            )
        member_names = tuple(common.MEMBER_SUMMARY_FEATURE_NAMES)
        if tuple(member_summaries.feature_fields) != member_names:
            raise base.DataContractError("member-summary feature order differs")
        for name in member_names:
            normalized, statistics = common.normalize_feature(
                member_summaries.feature_fields[name],
                train_indices,
                weights,
                preserve_full_domain=preserve_fuxi_context,
            )
            member_channels.append(normalized[:, :, None])
            member_statistics[name] = statistics
    candidates = []
    for offset in OFFSETS_DAYS:
        shifted = forecast.valid_dates + np.timedelta64(offset, "D")
        positions = base.calendar_positions(shifted)
        weekly = np.mean(climatology_daily[positions], axis=2, dtype=np.float64)
        candidates.append(weekly.astype(np.float32))
    bank = np.stack(candidates, axis=2)
    centre_difference = float(
        np.nanmax(
            np.abs(bank[:, :, OFFSETS_DAYS.index(0)] - observations.weekly_climatology)
        )
    )
    if centre_difference > 2.0e-6:
        raise base.DataContractError(
            f"zero-offset climatology differs by {centre_difference:.6g}"
        )

    climo_stats = normalization["log_imerg_climatology"]
    climo_mean = np.asarray(climo_stats["mean_by_lead"], dtype=np.float32)
    climo_std = np.asarray(climo_stats["std_by_lead"], dtype=np.float32)
    log_bank = np.log1p(bank).astype(np.float32)
    normalized_bank = (log_bank - climo_mean[None, :, None, None, None]) / climo_std[
        None, :, None, None, None
    ]
    anomaly_stats = normalization["explicit_log_fuxi_anomaly"]
    anomaly_mean = np.asarray(anomaly_stats["mean_by_lead"], dtype=np.float32)
    anomaly_std = np.asarray(anomaly_stats["std_by_lead"], dtype=np.float32)
    raw_anomaly = np.log1p(forecast.ensemble_mean)[:, :, None] - log_bank
    normalized_anomaly = (
        raw_anomaly - anomaly_mean[None, :, None, None, None]
    ) / anomaly_std[None, :, None, None, None]
    valid = support[None, None, None]
    normalized_bank = np.where(valid, normalized_bank, 0.0).astype(np.float32)
    normalized_anomaly = np.where(valid, normalized_anomaly, 0.0).astype(np.float32)
    features = np.concatenate(
        (standard, *member_channels, normalized_bank, normalized_anomaly), axis=2
    ).astype(np.float32)
    expected_channels = 29 + len(member_names)
    if features.shape != (
        len(forecast.initializations),
        6,
        expected_channels,
        27,
        27,
    ):
        raise base.DataContractError(
            f"unexpected attentive feature shape: {features.shape}"
        )
    normalization = dict(normalization)
    normalization["log_imd_climatology"] = normalization.pop("log_imerg_climatology")
    channels = list(normalization["input_channels"])
    channels[2] = "log_imd_calendar_climatology"
    channels[9] = "explicit_log_fuxi_minus_imd_climatology"
    normalization["input_channels"] = [
        *channels,
        *member_names,
        *[f"imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
        *[f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
    ]
    if member_summaries is not None:
        normalization.update(member_statistics)
        normalization["fuxi_member_summaries"] = {
            "source": "native FuXi-S2S reforecast shards",
            "member_count": 51,
            "feature_order": list(member_names),
            "definitions": {
                name: common.MEMBER_SUMMARY_DEFINITIONS[name] for name in member_names
            },
            "tp_source_units": "mm h-1",
            "tp_feature_units": "mm day-1 before log1p or thresholding",
            "t2m_units": "K",
            "weekly_reduction": (
                "complete seven-day mean per member before ensemble summary"
            ),
            "quantile_method": "numpy method=linear",
            "probability_thresholds_mm_day": [1.0, 10.0],
            "ensemble_std_ddof": 0,
            "normalization_fit": ("training cases and positive target weights only"),
            "cache_representation": "raw unnormalized float32 summaries",
            "source_fingerprint": member_summaries.source_fingerprint,
            "cache_path": member_summaries.cache_path,
            "cache_sha256": member_summaries.cache_sha256,
        }
    normalization["climatology_attention"] = {
        "source": "IMD training years 2002-2017 only",
        "offsets_days": list(OFFSETS_DAYS),
        "candidate_axis": "convex softmax attention",
        "verification_reference": "fixed zero-offset 31-day climatology",
    }
    context = dict(normalization["spatial_context"])
    context["support_limited_channels"] = [
        "log_imd_calendar_climatology",
        "explicit_log_fuxi_minus_imd_climatology",
        *[f"imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
        *[f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
    ]
    context["domain"] = "27x27 grid: 0-39N, 60-99E"
    context["target_support_cells"] = int(support.sum())
    if member_names and preserve_fuxi_context:
        context["full_domain_channels"] = [
            *context["full_domain_channels"],
            *member_names,
        ]
    normalization["spatial_context"] = context
    if preserve_fuxi_context:
        climatology_start = 11 + len(member_names)
        support_limited = [2, 9, *range(climatology_start, expected_channels)]
        if any(
            np.any(features[:, :, index, ~support] != 0.0) for index in support_limited
        ):
            raise base.DataContractError(
                "one or more IMD-derived channels are nonzero outside IMD support"
            )
    return features, normalization, centre_difference


def add_attention_comparison(
    selection: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    weights: np.ndarray,
    initializations: np.ndarray,
) -> pd.DataFrame:
    result = selection.copy()
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    normal_rmse = engine.case_rmse(predictions[NORMAL.name], truth, weights)[
        :, ACTIVE_LEADS
    ]
    attention_rmse = engine.case_rmse(predictions[ATTENTION.name], truth, weights)[
        :, ACTIVE_LEADS
    ]
    skill_columns = []
    for year in sorted(np.unique(years)):
        mask = years == year
        normal_score = float(normal_rmse[mask].mean())
        attention_score = float(attention_rmse[mask].mean())
        skill = 100.0 * (normal_score - attention_score) / normal_score
        column = f"attention_rmse_skill_vs_normal_{year}_pct"
        result[column] = np.nan
        result.loc[result.candidate.eq(ATTENTION.name), column] = skill
        skill_columns.append(column)
    result["attention_beats_normal_every_year"] = False
    attention_row = result.candidate.eq(ATTENTION.name)
    result.loc[attention_row, "attention_beats_normal_every_year"] = bool(
        result.loc[attention_row, skill_columns].iloc[0].gt(0.0).all()
    )
    pooled_skill = (
        100.0 * (normal_rmse.mean() - attention_rmse.mean()) / normal_rmse.mean()
    )
    result["attention_rmse_skill_vs_normal_pooled_pct"] = np.nan
    result.loc[attention_row, "attention_rmse_skill_vs_normal_pooled_pct"] = (
        pooled_skill
    )
    return result


def choose_model(
    selection: pd.DataFrame,
    *,
    baseline_method: str = "log_bias",
    baseline_label: str = "log-bias",
    matched_normal_only: bool = False,
) -> tuple[str, float, str]:
    normal = selection.loc[selection.candidate.eq(NORMAL.name)].iloc[0]
    attention = selection.loc[selection.candidate.eq(ATTENTION.name)].iloc[0]
    if matched_normal_only:
        if bool(normal.improves_every_validation_year):
            return (
                NORMAL.name,
                float(normal.alpha),
                "Predeclared matched-architecture ablation: the frozen normal-"
                f"climatology architecture improved over {baseline_label} in both "
                "validation years. Attention is reported only as a secondary screen.",
            )
        return (
            baseline_method,
            0.0,
            "Predeclared matched-architecture ablation: the frozen normal-"
            f"climatology architecture did not improve over {baseline_label} in "
            "both validation years. Attention is reported only as a secondary screen.",
        )
    if bool(attention.improves_every_validation_year) and bool(
        attention.attention_beats_normal_every_year
    ):
        return (
            ATTENTION.name,
            float(attention.alpha),
            f"Attention improved over both {baseline_label} and the "
            "normal-climatology model in each validation year.",
        )
    if bool(normal.improves_every_validation_year):
        return (
            NORMAL.name,
            float(normal.alpha),
            "Attention did not beat the matched normal-climatology model in both validation years.",
        )
    if bool(attention.improves_every_validation_year):
        return (
            ATTENTION.name,
            float(attention.alpha),
            f"Normal climatology failed the {baseline_label} guard; attention "
            f"improved over {baseline_label} in both years.",
        )
    return (
        baseline_method,
        0.0,
        f"Neither learned candidate improved over {baseline_label} in both validation years.",
    )


def plot_validation(scan: pd.DataFrame, selection: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for candidate in CANDIDATES:
        values = scan.loc[scan.candidate.eq(candidate.name)].sort_values("alpha")
        chosen = selection.loc[selection.candidate.eq(candidate.name)].iloc[0]
        axes[0].plot(
            values.alpha,
            values[VALIDATION_SCORE_COLUMN],
            color=candidate.color,
            label=candidate.label,
        )
        axes[0].scatter(
            chosen.alpha,
            chosen[VALIDATION_SCORE_COLUMN],
            color=candidate.color,
            edgecolor="black",
            zorder=4,
        )
    axes[0].set_xlabel("Residual gate α")
    axes[0].set_ylabel(f"Validation {ACTIVE_SCOPE} RMSE (mm day$^{{-1}}$)")
    axes[0].set_title("Matched model validation", fontweight="semibold")
    axes[0].legend(frameon=False)

    attention = selection.loc[selection.candidate.eq(ATTENTION.name)].iloc[0]
    years = sorted(
        int(column.removeprefix("attention_rmse_skill_vs_normal_").removesuffix("_pct"))
        for column in selection
        if column.startswith("attention_rmse_skill_vs_normal_")
        and column.endswith("_pct")
        and "pooled" not in column
    )
    skills = [
        float(getattr(attention, f"attention_rmse_skill_vs_normal_{year}_pct"))
        for year in years
    ]
    axes[1].bar([str(year) for year in years], skills, color=("#0072B2", "#E69F00"))
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Attention RMSE skill vs normal model (%)")
    axes[1].set_title("Attention-specific validation value", fontweight="semibold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Forecast-conditioned IMD climatology\n2002–2017 train · 2018–2019 validation",
        y=1.03,
        fontweight="semibold",
    )
    figure.tight_layout()
    diagnostics.save_figure(figure, output)


def plot_imd_skill_by_lead(
    summary: pd.DataFrame,
    output: Path,
    *,
    smoke: bool,
    support_cells: int,
) -> None:
    """Plot lead-wise scores with the correct IMD verification labels."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    panels = (
        ("acc", "Spatial ACC\n(common 2002–2017 IMD climatology)"),
        ("pcc", "Spatial PCC of absolute rainfall\n(no climatology)"),
        ("rmse", "RMSE (mm day$^{-1}$)"),
        ("bias", "Bias (mm day$^{-1}$)"),
    )
    plot_methods = ("raw_fuxi", "log_bias", NORMAL.name, ATTENTION.name)
    handles = {}
    for panel_index, (axis, (metric, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        for method in plot_methods:
            selected = summary.loc[summary.method.eq(method)].sort_values("lead_week")
            (line,) = axis.plot(
                selected.lead_week,
                selected[metric],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle="--" if method == "raw_fuxi" else "-",
                linewidth=1.8,
                markersize=5.5,
                markerfacecolor=(
                    "white" if method == "raw_fuxi" else METHOD_COLORS[method]
                ),
            )
            handles[method] = line
        if metric == "bias":
            axis.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(1, 7), [f"W{lead}" for lead in range(1, 7)])
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.01,
            0.98,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            va="top",
            fontweight="semibold",
        )
    figure.legend(
        [handles[method] for method in plot_methods],
        [METHOD_LABELS[method] for method in plot_methods],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
    )
    count = int(summary.case_count.max())
    context = (
        "SMOKE CHECK"
        if smoke
        else f"IMD verification; reused 2020–2021 test (n={count})"
    )
    figure.suptitle(
        "FuXi-S2S weekly rainfall post-processing over India\n" + context,
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.015,
        f"Area-weighted case scores on {support_cells} IMD-support cells; "
        "ACC uses one fixed training-only IMD reference; PCC uses absolute fields",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.1,
        top=0.76,
        wspace=0.25,
        hspace=0.32,
    )
    diagnostics.save_figure(figure, output)


def attention_parameter_table(
    output: Path, training: Mapping[str, Mapping[str, object]]
) -> pd.DataFrame:
    rows = []
    for record in training[ATTENTION.name]["runs"]:
        checkpoint = torch.load(output / str(record["checkpoint"]), map_location="cpu")
        state = checkpoint["model_state_dict"]
        gate = torch.sigmoid(state["conditioner.lead_gate_logit"]).numpy()
        gate[:INACTIVE_LEAD_COUNT] = 0.0
        temperature = torch.nn.functional.softplus(
            state["conditioner.log_temperature"]
        ).numpy()
        for lead, value in enumerate(gate, start=1):
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "kind": "lead_gate",
                    "index": lead,
                    "value": float(value),
                }
            )
        for head, value in enumerate(temperature, start=1):
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "kind": "head_temperature",
                    "index": head,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def plot_spatial(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    forecast: base.ForecastData,
    weights: np.ndarray,
    output: Path,
) -> pd.DataFrame:
    support = weights > 0.0
    pairs = (
        (ATTENTION.name, NORMAL.name, "Attention vs normal model"),
        (ATTENTION.name, "log_bias", "Attention model vs log-bias"),
    )
    fields = []
    rows = []
    for method, baseline, label in pairs:
        model_error = predictions[method][:, ACTIVE_LEADS] - truth[:, ACTIVE_LEADS]
        base_error = predictions[baseline][:, ACTIVE_LEADS] - truth[:, ACTIVE_LEADS]
        model_rmse = np.sqrt(np.mean(model_error.astype(np.float64) ** 2, axis=(0, 1)))
        base_rmse = np.sqrt(np.mean(base_error.astype(np.float64) ** 2, axis=(0, 1)))
        reduction = base_rmse - model_rmse
        reduction[~support] = np.nan
        fields.append(reduction)
        improved = support & (reduction > 0.0)
        rows.append(
            {
                "comparison": label,
                "area_fraction_improved_pct": float(
                    100.0 * weights[improved].sum() / weights[support].sum()
                ),
                "area_weighted_mean_rmse_reduction_mm_day": float(
                    np.sum(weights[support] * reduction[support])
                    / weights[support].sum()
                ),
            }
        )
    values = np.stack(fields)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(0.1, float(np.ceil(np.percentile(finite, 98) / 0.1) * 0.1))
    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1, 2, figsize=(8.8, 4.3), subplot_kw={"projection": projection}
    )
    image = None
    for index, (axis, row) in enumerate(zip(axes, rows)):
        image = axis.pcolormesh(
            forecast.longitude,
            forecast.latitude,
            np.ma.masked_invalid(values[index]),
            transform=projection,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
        )
        axis.coastlines(resolution="50m", linewidth=0.65)
        axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
        axis.set_extent([59, 99, -1, 38], crs=projection)
        axis.set_title(row["comparison"], fontsize=10, fontweight="semibold")
        axis.text(
            0.98,
            0.03,
            f"{row['area_fraction_improved_pct']:.1f}% area improved",
            transform=axis.transAxes,
            ha="right",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    assert image is not None
    color_axis = figure.add_axes([0.27, 0.13, 0.46, 0.04])
    colorbar = figure.colorbar(
        image, cax=color_axis, orientation="horizontal", extend="both"
    )
    colorbar.set_label(f"Local {ACTIVE_SCOPE} RMSE reduction (mm day$^{{-1}}$)")
    figure.suptitle(
        "Exploratory IMD spatial comparison · 2020–2021", fontweight="semibold"
    )
    figure.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.25, wspace=0.08)
    diagnostics.save_figure(figure, output)
    return pd.DataFrame(rows)


def evaluate_predictions(
    truth: np.ndarray,
    climatology: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    initializations: np.ndarray,
    weights: np.ndarray,
    *,
    split: str,
) -> pd.DataFrame:
    frames = []
    case_ids = [np.datetime_as_string(value, unit="D") for value in initializations]
    support_count = int(np.count_nonzero(weights > 0.0))
    for method in METHOD_ORDER:
        frame = compute_case_metrics(
            truth,
            predictions[method],
            truth - climatology,
            predictions[method] - climatology,
            weights,
            predictor=method,
            case_ids=case_ids,
            leads=np.arange(1, 7),
            valid_mask=weights > 0.0,
        ).rename(
            columns={"predictor": "method", "case_id": "init", "lead": "lead_week"}
        )
        frame.insert(0, "split", split)
        frame.insert(2, "year", pd.DatetimeIndex(frame.init).year)
        frame["spatial_acc_common_imd"] = frame["acc"]
        frame["mse"] = frame["rmse"] ** 2
        frame["climatology_mse"] = (
            engine.case_rmse(climatology, truth, weights).reshape(-1) ** 2
        )
        frame["pcc"] = [
            weighted_spatial_acc(
                truth[case_index, lead_index],
                predictions[method][case_index, lead_index],
                weights,
                weights > 0.0,
            )
            for case_index in range(len(initializations))
            for lead_index in range(6)
        ]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(initializations) * 6 * len(METHOD_ORDER)
    if len(result) != expected or not np.all(result.valid_cells == support_count):
        raise base.DataContractError("unexpected IMD metric table shape or support")
    return result


def summarize_by_lead(case_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add climatology-free PCC to the standard lead-wise summary."""
    summary = diagnostics.summarize_by_lead(case_metrics)
    extra = case_metrics.groupby(["method", "lead_week"], as_index=False).agg(
        pcc=("pcc", "mean"),
        mean_mse=("mse", "mean"),
        climatology_mean_mse=("climatology_mse", "mean"),
    )
    summary = summary.merge(extra, on=["method", "lead_week"], validate="one_to_one")
    summary["mse_skill_vs_imd_climatology"] = 1.0 - (
        summary.mean_mse / summary.climatology_mean_mse
    )
    raw = summary.loc[summary.method.eq("raw_fuxi")].set_index("lead_week")
    summary["delta_pcc_vs_raw"] = [
        float(row.pcc - raw.loc[row.lead_week, "pcc"])
        for row in summary.itertuples(index=False)
    ]
    return summary


def paired_intervals_with_pcc(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    *,
    smoke: bool,
) -> pd.DataFrame:
    """Append paired-init PCC differences to the standard bootstrap table."""
    intervals = diagnostics.paired_intervals(case_metrics, initializations, smoke=smoke)
    n_resamples = 50 if smoke else diagnostics.BOOTSTRAP_SAMPLES
    block_length = 2 if smoke else diagnostics.BOOTSTRAP_BLOCK_LENGTH
    sampled = base._two_stage_block_indices(
        initializations, n_resamples, block_length, seed=42
    )
    case_order = [np.datetime_as_string(value, unit="D") for value in initializations]
    rows = []
    for method, baseline_method in diagnostics.ALL_COMPARISONS:
        for scope, leads in diagnostics.LEAD_SCOPES.items():
            selected = case_metrics.loc[
                case_metrics.lead_week.isin(leads)
                & case_metrics.method.isin((method, baseline_method))
            ]
            pivot = selected.pivot_table(
                index="init", columns="method", values="pcc", aggfunc="mean"
            ).reindex(case_order)
            model = pivot[method].to_numpy(dtype=np.float64)
            baseline_values = pivot[baseline_method].to_numpy(dtype=np.float64)
            if not np.isfinite(model).all() or not np.isfinite(baseline_values).all():
                raise base.DataContractError(
                    "paired PCC bootstrap contains missing values"
                )
            effect = float(model.mean() - baseline_values.mean())
            draws = model[sampled].mean(axis=1) - baseline_values[sampled].mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "baseline": baseline_method,
                    "lead_scope": scope,
                    "metric": "pcc",
                    "paired_case_count": len(case_order),
                    "model_mean": float(model.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "effect": effect,
                    "ci_lower": float(np.percentile(draws, 2.5)),
                    "ci_upper": float(np.percentile(draws, 97.5)),
                    "block_length": block_length,
                    "n_resamples": n_resamples,
                    "seed": 42,
                }
            )
    return pd.concat((intervals, pd.DataFrame(rows)), ignore_index=True)


def active_headline(case_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize the predeclared W3--W6 correction scope."""
    selected = case_metrics.loc[case_metrics.lead_week.isin(ACTIVE_WEEKS)]
    table = selected.groupby("method", as_index=False).agg(
        cases=("init", "nunique"),
        acc=("acc", "mean"),
        pcc=("pcc", "mean"),
        rmse_mm_day=("rmse", "mean"),
        mae_mm_day=("mae", "mean"),
        bias_mm_day=("bias", "mean"),
        mean_mse=("mse", "mean"),
        climatology_mean_mse=("climatology_mse", "mean"),
    )
    raw = table.loc[table.method.eq("raw_fuxi")].iloc[0]
    anchor = table.loc[table.method.eq("log_bias")].iloc[0]
    table["rmse_skill_vs_raw_pct"] = (
        100.0 * (raw.rmse_mm_day - table.rmse_mm_day) / raw.rmse_mm_day
    )
    table["rmse_skill_vs_log_bias_pct"] = (
        100.0 * (anchor.rmse_mm_day - table.rmse_mm_day) / anchor.rmse_mm_day
    )
    table["mse_skill_vs_imd_climatology"] = 1.0 - (
        table.mean_mse / table.climatology_mean_mse
    )
    table["method_label"] = table.method.map(METHOD_LABELS)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    table["_order"] = table.method.map(order)
    return table.sort_values("_order").drop(columns="_order")


def write_prediction_store(
    path: Path,
    forecast: base.ForecastData,
    observations: base.ObservationData,
    indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    selection: Mapping[str, object],
    *,
    smoke: bool,
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack([predictions[method] for method in METHOD_ORDER]).astype(
                    np.float32
                ),
            ),
            "truth_imd": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_truth[indices].astype(np.float32),
            ),
            "fixed_imd_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_climatology[indices].astype(np.float32),
            ),
            "area_weight_km2": (("latitude", "longitude"), weights.astype(np.float64)),
        },
        coords={
            "method": list(METHOD_ORDER),
            "init": forecast.initializations[indices].astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": "FuXi-to-IMD forecast-conditioned climatology experiment",
            "train_years": "2002-2017",
            "validation_years": "2018-2019",
            "test_years": "2020-2021",
            "test_status": "exploratory; reused test period",
            "selected_model": selection["selected_model"],
            "selected_alpha": float(selection["selected_alpha"]),
            "selection_scope": "validation_only",
            "training_anchor": str(selection["training_anchor"]),
            "active_weeks": list(ACTIVE_WEEKS),
            "inactive_weeks_exact_training_anchor": list(
                range(1, INACTIVE_LEAD_COUNT + 1)
            ),
            "lead_weights": list(LEAD_WEIGHTS),
            "verification_climatology": "fixed training-only 31-day IMD climatology",
            "acc_definition": "common-reference spatial ACC using fixed 2002-2017 IMD climatology",
            "pcc_definition": "spatial PCC of absolute weekly rainfall; no climatology",
            "attention_bank_offsets_days": list(OFFSETS_DAYS),
            "full_fuxi_context": bool(
                selection.get("spatial_context", {}).get("enabled", False)
            ),
            "units": "mm day-1",
            "smoke": smoke,
        },
    )
    chunk_cases = min(35, len(indices))
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={
            "prediction": {"chunks": (1, chunk_cases, 1, 27, 27)},
            "truth_imd": {"chunks": (chunk_cases, 1, 27, 27)},
            "fixed_imd_climatology": {"chunks": (chunk_cases, 1, 27, 27)},
            "area_weight_km2": {"chunks": (27, 27)},
        },
    )


def write_results(
    output: Path,
    training: Mapping[str, Mapping[str, object]],
    validation: pd.DataFrame,
    selection: Mapping[str, object],
    late: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    smoke: bool,
) -> None:
    attention = validation.loc[validation.candidate.eq(ATTENTION.name)].iloc[0]
    training_anchor = str(selection["training_anchor"])
    training_anchor_label = (
        "training-only fitted log-bias"
        if training_anchor == "log_bias"
        else "raw FuXi (no fitted log-bias correction)"
    )
    skill_2018 = float(
        getattr(attention, "attention_rmse_skill_vs_normal_2018_pct", np.nan)
    )
    skill_2019 = float(
        getattr(attention, "attention_rmse_skill_vs_normal_2019_pct", np.nan)
    )
    lines = [
        "# FuXi–IMD attention-climatology experiment",
        "",
        (
            "> Smoke check only; do not interpret."
            if smoke
            else "> Exploratory 2020–2021 test; selection was frozen on 2018–2019."
        ),
        "",
        "## Design",
        "",
        "- Train 2002–2017; validate 2018–2019; exploratory test 2020–2021",
        "- Target and verification: IMD daily rainfall aggregated to FuXi weeks",
        f"- Neural reconstruction baseline: {training_anchor_label}",
        "- The fitted 2002–2017 log-bias forecast remains a reporting baseline",
        "- Control: fixed training-only 31-day IMD climatology",
        "- Attention: convex choice over nine training-only calendar climatologies (±28 days)",
        "- Both models use the same regularized temporal backbone and three fixed seeds",
        (
            "- FuXi TP mean, ensemble spread, and T2M retain the full 27×27 regional context; IMD targets and loss remain on 171 India cells"
            if bool(selection.get("spatial_context", {}).get("enabled", False))
            else "- Dynamic predictors are restricted to the 171-cell IMD support"
        ),
        (
            "- The learned residual is active at all six weeks"
            if INACTIVE_LEAD_COUNT == 0
            else f"- W1–W{INACTIVE_LEAD_COUNT} remain exactly equal to the "
            f"{training_anchor_label} baseline; "
            f"the learned residual is active at {ACTIVE_SCOPE}"
        ),
        "- ACC uses one fixed training-only IMD climatology for every method",
        "- PCC is also reported from absolute fields and needs no climatology",
        "",
        "## Capacity and validation loss",
        "",
        "| Candidate | Parameters | Mean best validation loss |",
        "|---|---:|---:|",
    ]
    for candidate in CANDIDATES:
        metadata = training[candidate.name]
        mean_loss = np.mean([run["best_validation_loss"] for run in metadata["runs"]])
        lines.append(
            f"| {candidate.label} | {metadata['parameter_count']:,} | {mean_loss:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation decision",
            "",
            f"Selected **{METHOD_LABELS[str(selection['selected_model'])]}**, "
            f"α={float(selection['selected_alpha']):.3f}.",
            "",
            str(selection["selection_reason"]),
            "",
            f"Attention RMSE skill versus the normal model: "
            f"2018 {skill_2018:+.2f}%, 2019 {skill_2019:+.2f}%.",
            "",
            f"## Exploratory test — {ACTIVE_SCOPE}",
            "",
            "| Method | Common-IMD ACC | PCC | RMSE | MAE | Bias | MSESS vs IMD climo | RMSE skill vs log-bias |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in late.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.acc:.3f} | {row.pcc:.3f} | "
            f"{row.rmse_mm_day:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:+.3f} | "
            f"{row.mse_skill_vs_imd_climatology:+.3f} | "
            f"{row.rmse_skill_vs_log_bias_pct:+.2f}% |"
        )
    direct = intervals.loc[
        intervals.method.eq(ATTENTION.name)
        & intervals.baseline.eq(NORMAL.name)
        & intervals.lead_scope.eq(ACTIVE_SCOPE)
    ].set_index("metric")
    lines.extend(
        [
            "",
            "## Attention-specific paired comparison",
            "",
            f"- Δ common-IMD ACC: {direct.loc['acc', 'effect']:+.3f} "
            f"[{direct.loc['acc', 'ci_lower']:+.3f}, {direct.loc['acc', 'ci_upper']:+.3f}]",
            f"- Δ PCC (no climatology): {direct.loc['pcc', 'effect']:+.3f} "
            f"[{direct.loc['pcc', 'ci_lower']:+.3f}, {direct.loc['pcc', 'ci_upper']:+.3f}]",
            f"- RMSE reduction: {direct.loc['rmse', 'effect']:+.2f}% "
            f"[{direct.loc['rmse', 'ci_lower']:+.2f}%, {direct.loc['rmse', 'ci_upper']:+.2f}%]",
            f"- MAE reduction: {direct.loc['mae', 'effect']:+.2f}% "
            f"[{direct.loc['mae', 'ci_lower']:+.2f}%, {direct.loc['mae', 'ci_upper']:+.2f}%]",
            "",
            "Do not claim attention-specific improvement unless the paired interval and both validation years support it.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="short integration check")
    parser.add_argument(
        "--training-anchor",
        choices=TRAINING_ANCHORS,
        default="log_bias",
        help=(
            "reconstruction baseline used to define the neural target; raw_fuxi "
            "is the no-fitted-log-bias ablation"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="fresh immutable output directory (must not already exist)",
    )
    parser.add_argument(
        "--all-weeks",
        action="store_true",
        help="learn corrections for W1-W6 and use a separate result archive",
    )
    parser.add_argument(
        "--large-model",
        action="store_true",
        help="use the 2.54M-parameter multi-scale temporal backbone for W1-W6",
    )
    parser.add_argument(
        "--regularized-large",
        action="store_true",
        help="use stronger weight decay and dropout in the 2.54M W1-W6 model",
    )
    parser.add_argument(
        "--full-fuxi-context",
        action="store_true",
        help=(
            "retain FuXi TP mean/spread and T2M over the full 27x27 domain; "
            "requires --all-weeks and the compact model"
        ),
    )
    args = parser.parse_args()
    set_experiment_scope(
        all_weeks=args.all_weeks or args.large_model or args.regularized_large,
        large_model=args.large_model or args.regularized_large,
        regularized_large=args.regularized_large,
        full_fuxi_context=args.full_fuxi_context,
    )
    configure_contract()
    started = time.monotonic()
    prefix = "smoke" if args.smoke else "full"
    output = (
        args.output.resolve()
        if args.output is not None
        else RESULTS_ROOT / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    for directory in (
        output,
        output / "models",
        output / "metrics",
        output / "figures",
        output / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    print("Loading FuXi 2002–2021...", flush=True)
    forecast = base.load_fuxi()
    print(
        "Loading aligned IMD and fitting the 2002–2017 calendar climatology...",
        flush=True,
    )
    observations, climatology_daily, _, source_stores = load_imd(forecast)
    weights = load_imd_weights(forecast, observations.observation_fraction)
    support = weights > 0.0
    splits = base.split_indices(forecast.initializations)
    counts = {name: len(indices) for name, indices in splits.items()}
    if counts != {"train": 560, "validation": 70, "test": 70}:
        raise base.DataContractError(f"unexpected split counts: {counts}")

    print(
        "Building normal and attention-climatology predictors"
        + (" with full-domain FuXi context..." if args.full_fuxi_context else "..."),
        flush=True,
    )
    t2m_weekly = common.load_t2m_weekly(forecast)
    features, normalization, centre_difference = build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        splits["train"],
        t2m_weekly,
        preserve_fuxi_context=args.full_fuxi_context,
    )
    (output / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )

    correction = fit_log_bias_correction(
        forecast.ensemble_mean[splits["train"]],
        observations.weekly_truth[splits["train"]],
        forecast.initializations[splits["train"]],
        support,
        shrinkage=10.0,
    )
    log_bias_baseline = apply_log_bias_correction(
        forecast.ensemble_mean, forecast.initializations, correction
    )
    training_baseline = select_training_baseline(
        args.training_anchor,
        forecast.ensemble_mean,
        log_bias_baseline,
    )
    valid = np.broadcast_to(support[None, None], training_baseline.shape)
    target_scale = fit_anchored_target_scale(
        observations.weekly_truth[splits["train"]],
        training_baseline[splits["train"]],
        weights,
        split_name="train",
        valid_mask=valid[splits["train"]],
    )
    target = standardize_anchored_target(
        observations.weekly_truth,
        training_baseline,
        target_scale,
        valid_mask=valid,
    )
    np.savez_compressed(
        output / "models" / "log_bias_anchor.npz",
        lead_month_residual=correction.lead_month_residual,
        shrinkage=np.float32(correction.shrinkage),
        target_scale=(
            target_scale if args.training_anchor == "log_bias" else np.array([])
        ),
    )
    np.savez_compressed(
        output / "models" / "training_anchor_contract.npz",
        anchor_kind=np.asarray(args.training_anchor),
        target_scale=target_scale,
        fitted_target_years=np.asarray(TRAIN_YEARS, dtype=np.int16),
    )

    validation_indices = (
        splits["validation"][:16] if args.smoke else splits["validation"]
    )
    training = {}
    validation_residuals = {}
    for candidate in CANDIDATES:
        print(f"Training {candidate.label}...", flush=True)
        residual, metadata = engine.train_candidate(
            candidate,
            features,
            target,
            target_scale,
            training_baseline,
            observations,
            weights,
            splits,
            output,
            smoke=args.smoke,
            lead_weights=LEAD_WEIGHTS,
            inactive_lead_count=INACTIVE_LEAD_COUNT,
            loss_coefficients=LOSS_COEFFICIENTS,
        )
        validation_residuals[candidate.name] = residual
        training[candidate.name] = metadata

    print("Freezing validation-only selection...", flush=True)
    alpha_scan, validation_selection = engine.scan_validation_alpha(
        validation_residuals,
        validation_indices,
        target,
        training_baseline,
        observations,
        target_scale,
        weights,
        active_leads=ACTIVE_LEADS,
        lead_weights=LEAD_WEIGHTS,
        loss_coefficients=LOSS_COEFFICIENTS,
        score_column=VALIDATION_SCORE_COLUMN,
    )
    validation_selection = engine.add_year_validation_scores(
        validation_selection,
        validation_residuals,
        validation_indices,
        training_baseline,
        observations,
        target_scale,
        weights,
        forecast.initializations,
        active_leads=ACTIVE_LEADS,
        baseline_name=args.training_anchor,
    )
    gated_validation = {}
    for candidate in CANDIDATES:
        alpha = float(
            validation_selection.loc[
                validation_selection.candidate.eq(candidate.name), "alpha"
            ].iloc[0]
        )
        gated_validation[candidate.name] = common.reconstruct(
            training_baseline[validation_indices],
            (alpha * validation_residuals[candidate.name]).astype(np.float32),
            target_scale,
            support,
        )
    validation_selection = add_attention_comparison(
        validation_selection,
        gated_validation,
        observations.weekly_truth[validation_indices],
        weights,
        forecast.initializations[validation_indices],
    )
    training_anchor_label = (
        "log-bias" if args.training_anchor == "log_bias" else "raw FuXi"
    )
    selected_model, selected_alpha, selection_reason = choose_model(
        validation_selection,
        baseline_method=args.training_anchor,
        baseline_label=training_anchor_label,
        matched_normal_only=args.training_anchor == "raw_fuxi",
    )
    frozen_utc = utc_now()
    selection_record = {
        "status": "frozen",
        "smoke": args.smoke,
        "selection_scope": "validation_only",
        "training_anchor": args.training_anchor,
        "uses_fitted_log_bias_in_neural_training": args.training_anchor == "log_bias",
        "log_bias_role": (
            "training_reconstruction_baseline"
            if args.training_anchor == "log_bias"
            else "reporting_only"
        ),
        "selection_policy": (
            "matched_normal_architecture_primary; attention_secondary_only"
            if args.training_anchor == "raw_fuxi"
            else "canonical_normal_vs_attention_guards"
        ),
        "observation_source": "IMD",
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "test_years_quarantined_during_selection": list(TEST_YEARS),
        "candidate_set": [candidate.name for candidate in CANDIDATES],
        "attention_bank_offsets_days": list(OFFSETS_DAYS),
        "spatial_context": normalization["spatial_context"],
        "active_weeks": list(ACTIVE_WEEKS),
        "inactive_weeks_exact_training_anchor": list(range(1, INACTIVE_LEAD_COUNT + 1)),
        "lead_weights": list(LEAD_WEIGHTS),
        "primary_validation_metric": f"equal-case {ACTIVE_SCOPE} area-weighted RMSE",
        "selected_model": selected_model,
        "selected_alpha": selected_alpha,
        "selection_reason": selection_reason,
        "frozen_utc": frozen_utc,
        "test_predictions_created": False,
        "checkpoint_sha256": {
            candidate.name: [
                record["checkpoint_sha256"]
                for record in training[candidate.name]["runs"]
            ]
            for candidate in CANDIDATES
        },
    }
    selection_path = output / "selection.json"
    selection_path.write_text(
        json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
    )
    selection_hash = engine.sha256_file(selection_path)
    alpha_scan.to_csv(output / "metrics" / "validation_alpha_scan.csv", index=False)
    validation_selection.to_csv(
        output / "metrics" / "validation_selection.csv", index=False
    )
    parameter_table = attention_parameter_table(output, training)
    parameter_table.to_csv(output / "metrics" / "attention_parameters.csv", index=False)

    raw_validation = forecast.ensemble_mean[validation_indices].copy()
    raw_validation[..., ~support] = np.nan
    log_validation = log_bias_baseline[validation_indices].copy()
    log_validation[..., ~support] = np.nan
    validation_baselines = {
        "raw_fuxi": raw_validation,
        "log_bias": log_validation,
    }
    selected_validation = (
        validation_baselines[selected_model]
        if selected_model in validation_baselines
        else gated_validation[selected_model]
    )
    validation_predictions = {
        "raw_fuxi": raw_validation,
        "log_bias": log_validation,
        **gated_validation,
        "selected_model": selected_validation,
    }
    validation_metrics = evaluate_predictions(
        observations.weekly_truth[validation_indices],
        observations.weekly_climatology[validation_indices],
        validation_predictions,
        forecast.initializations[validation_indices],
        weights,
        split="validation",
    )
    validation_metrics.to_csv(
        output / "metrics" / "validation_case_metrics.csv", index=False
    )
    tidy = engine.tidy_training_history(output, training)
    tidy.to_csv(output / "metrics" / "training_history_tidy.csv", index=False)
    engine.plot_training_components(
        tidy, training, output / "figures" / "00_training_components"
    )
    plot_validation(
        alpha_scan, validation_selection, output / "figures" / "01_validation_selection"
    )

    test_started_utc = utc_now()
    if engine.sha256_file(selection_path) != selection_hash:
        raise RuntimeError("selection changed before test prediction")
    print(
        f"Selection frozen: {selected_model}, alpha={selected_alpha:.3f}. "
        "Creating exploratory 2020–2021 predictions...",
        flush=True,
    )
    test_indices = splits["test"][:8] if args.smoke else splits["test"]
    test_predictions = {}
    for candidate in CANDIDATES:
        residual = engine.predict_candidate(
            candidate,
            training[candidate.name],
            features,
            test_indices,
            output,
            inactive_lead_count=INACTIVE_LEAD_COUNT,
        )
        alpha = float(
            validation_selection.loc[
                validation_selection.candidate.eq(candidate.name), "alpha"
            ].iloc[0]
        )
        test_predictions[candidate.name] = common.reconstruct(
            training_baseline[test_indices],
            (alpha * residual).astype(np.float32),
            target_scale,
            support,
        )
    raw_test = forecast.ensemble_mean[test_indices].copy()
    raw_test[..., ~support] = np.nan
    log_test = log_bias_baseline[test_indices].copy()
    log_test[..., ~support] = np.nan
    test_baselines = {
        "raw_fuxi": raw_test,
        "log_bias": log_test,
    }
    selected_test = (
        test_baselines[selected_model]
        if selected_model in test_baselines
        else test_predictions[selected_model]
    )
    predictions = {
        "raw_fuxi": raw_test,
        "log_bias": log_test,
        **test_predictions,
        "selected_model": selected_test,
    }
    inactive_baseline = test_baselines[args.training_anchor]
    for method in METHOD_ORDER[2:]:
        if not np.array_equal(
            predictions[method][:, :INACTIVE_LEAD_COUNT],
            inactive_baseline[:, :INACTIVE_LEAD_COUNT],
            equal_nan=True,
        ):
            raise base.DataContractError(f"{method} violates inactive-week identity")

    truth = observations.weekly_truth[test_indices]
    climatology = observations.weekly_climatology[test_indices]
    initializations = forecast.initializations[test_indices]
    case_metrics = evaluate_predictions(
        truth, climatology, predictions, initializations, weights, split="test"
    )
    summary = summarize_by_lead(case_metrics)
    intervals = paired_intervals_with_pcc(
        case_metrics, initializations, smoke=args.smoke
    )
    headline = diagnostics.headline_table(case_metrics, intervals)
    late = active_headline(case_metrics)
    case_metrics.to_csv(output / "metrics" / "case_metrics.csv", index=False)
    summary.to_csv(output / "metrics" / "summary_by_lead.csv", index=False)
    intervals.to_csv(output / "metrics" / "paired_skill.csv", index=False)
    headline.to_csv(output / "metrics" / "headline_metrics.csv", index=False)
    late.to_csv(output / "metrics" / "active_headline_metrics.csv", index=False)
    plot_imd_skill_by_lead(
        summary,
        output / "figures" / "02_test_skill_by_lead",
        smoke=args.smoke,
        support_cells=int(support.sum()),
    )
    spatial = plot_spatial(
        predictions,
        truth,
        forecast,
        weights,
        output / "figures" / "03_test_spatial_improvement",
    )
    spatial.to_csv(output / "metrics" / "active_spatial_summary.csv", index=False)

    prediction_store = output / "predictions.zarr"
    write_prediction_store(
        prediction_store,
        forecast,
        observations,
        test_indices,
        predictions,
        weights,
        selection_record,
        smoke=args.smoke,
    )
    with xr.open_zarr(prediction_store, consolidated=True) as stored:
        for method in METHOD_ORDER:
            rebuilt = np.asarray(stored.prediction.sel({"method": method}).load())
            if not np.array_equal(rebuilt, predictions[method], equal_nan=True):
                raise base.DataContractError(f"prediction round-trip failed: {method}")

    write_results(
        output,
        training,
        validation_selection,
        selection_record,
        late,
        intervals,
        smoke=args.smoke,
    )
    sources = (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imd_no_log_bias_validation.py",
        SOURCE_ROOT / "fuxi_imerg_a100_big_temporal.py",
        SOURCE_ROOT / "fuxi_imerg_full_archive_latelead.py",
        SOURCE_ROOT / "fuxi_imerg_experiment.py",
        NEURAL_SRC / "fuxi_adapter" / "models.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "baselines.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
    )
    code_hashes = {}
    for source in sources:
        destination = output / "code" / source.name
        shutil.copy2(source, destination)
        code_hashes[source.name] = engine.sha256_file(destination)

    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "smoke": args.smoke,
        "created_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "scientific_status": "exploratory; reused 2020-2021 test",
        "experiment_role": (
            "matched no-fitted-log-bias neural ablation"
            if args.training_anchor == "raw_fuxi"
            else "canonical anchored climatology experiment"
        ),
        "training_anchor": args.training_anchor,
        "uses_fitted_log_bias_in_neural_training": args.training_anchor == "log_bias",
        "log_bias_role": selection_record["log_bias_role"],
        "observation_source": "IMD",
        "split_years": {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": list(TEST_YEARS),
        },
        "split_counts": counts,
        "test_count_used": len(test_indices),
        "support_cells": int(support.sum()),
        "active_leads": list(ACTIVE_WEEKS),
        "inactive_lead_identity_verified": True,
        "lead_weights": list(LEAD_WEIGHTS),
        "primary_validation_metric": f"equal-case {ACTIVE_SCOPE} area-weighted RMSE",
        "metric_definitions": {
            "acc": "area-weighted spatial correlation after subtracting the common fixed 2002-2017 IMD climatology from forecast and observation",
            "pcc": "area-weighted spatial correlation of absolute weekly forecast and observation; no climatology",
            "mse_skill_vs_imd_climatology": "1 - mean forecast MSE / mean fixed-IMD-climatology MSE",
        },
        "training": training,
        "selected_model": selected_model,
        "selected_alpha": selected_alpha,
        "selection_sha256": selection_hash,
        "selection_frozen_utc": frozen_utc,
        "test_evaluation_started_utc": test_started_utc,
        "climatology": {
            "normal": "2002-2017 equal-year centered 31-day IMD calendar climatology",
            "attention_offsets_days": list(OFFSETS_DAYS),
            "zero_offset_max_abs_difference": centre_difference,
            "verification_reference_fixed": True,
        },
        "source_stores": list(source_stores),
        "features": normalization["input_channels"],
        "spatial_context": normalization["spatial_context"],
        "quarantined_final_initialization_years": [2025],
        "prediction_store_roundtrip_verified": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code_sha256": code_hashes,
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "artifacts": {},
    }
    for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
        if prediction_store in artifact.parents:
            continue
        manifest["artifacts"][str(artifact.relative_to(output))] = engine.sha256_file(
            artifact
        )
    manifest["artifacts"]["predictions.zarr"] = engine.sha256_tree(prediction_store)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print("\n" + late.to_string(index=False), flush=True)
    identity = (
        "all weeks active"
        if INACTIVE_LEAD_COUNT == 0
        else f"W1-W{INACTIVE_LEAD_COUNT} identity"
    )
    print(
        f"\nPASS: IMD support, validation freeze, {identity}, and Zarr round-trip",
        flush=True,
    )
    print(f"Completed in {elapsed / 60.0:.1f} minutes", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
