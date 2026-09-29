#!/usr/bin/env python3
"""Build a small exploratory global FuXi/IMERG probability cache.

This is deliberately not the official AI Weather Quest ERA5 preparation
path.  It uses IMERG Final V07B as an exploratory precipitation target and a
fixed 2002--2016 IMERG climatology.  Whole FuXi initialization cases are
selected in blocked calendar years, reduced to D19--25 and D26--32, and
written as one NPZ compatible with :mod:`train`, :mod:`predict`, and
:mod:`evaluate`.

The builder reads the real archives only when executed.  Importing it is
side-effect free, which keeps its scientific contracts unit-testable without
opening the multi-terabyte FuXi store.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from config import EXPERIMENT


FUXI_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
IMERG_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "imerg_final_v07b_global_2002_2021/"
    "imerg_final_v07b_daily_fuxi_1p5_2002_2024.zarr"
)
DEFAULT_OUTPUT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/"
    "fuxi_imerg_exploratory_smoke_2017_2020.npz"
)

N_MEMBER = 51
N_LATITUDE = 121
N_LONGITUDE = 240
N_LEAD = 2
N_CATEGORY = 5
N_FEATURE = 18
LEAD_WINDOWS = ((19, 25), (26, 32))
MEMBER_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
BOUNDARY_QUANTILES = (0.20, 0.40, 0.60, 0.80)
CLIMATOLOGY_YEARS = tuple(range(2002, 2017))
CALENDAR_OFFSETS = (-4, -2, 0, 2, 4)
DEFAULT_SEASON_MONTHS = (6, 7, 8, 9)
DEFAULT_CASES_PER_YEAR = 4
DEFAULT_MIN_WEEKLY_COVERAGE = 0.80
DEFAULT_MIN_CLIMATOLOGY_SAMPLES = 60
FEATURE_NAMES = tuple(EXPERIMENT.feature_names[:-1]) + ("imerg_support_fraction",)
EXPLORATORY_NOTICE = (
    "Exploratory FuXi/IMERG pretraining cache; not official ERA5 data, not "
    "official AI Weather Quest validation, and not submission evidence."
)
FUXI_PERMISSION_NOTICE = (
    "FuXi-derived data and weights require written author permission before "
    "competition use."
)


@dataclass(frozen=True)
class SelectedCases:
    """Deterministically selected whole initialization cases."""

    source_indices: np.ndarray
    init_dates: pd.DatetimeIndex
    splits: np.ndarray


@dataclass(frozen=True)
class ClimatologyBounds:
    """Fixed-calendar quintile boundaries and their cellwise support."""

    bounds: np.ndarray
    valid_sample_count: np.ndarray
    support: np.ndarray


def _canonical_ints(
    values: Sequence[int], *, name: str, minimum: int, maximum: int
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    if any(value < minimum or value > maximum for value in result):
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def validate_blocked_years(
    train_years: Sequence[int],
    validation_years: Sequence[int],
    test_years: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return valid, mutually exclusive whole-case year blocks."""

    train = _canonical_ints(train_years, name="train_years", minimum=1, maximum=9999)
    validation = _canonical_ints(
        validation_years, name="validation_years", minimum=1, maximum=9999
    )
    test = _canonical_ints(test_years, name="test_years", minimum=1, maximum=9999)
    memberships: dict[int, list[str]] = {}
    for label, years in (("train", train), ("validation", validation), ("test", test)):
        for year in years:
            memberships.setdefault(year, []).append(label)
    overlap = {year: labels for year, labels in memberships.items() if len(labels) > 1}
    if overlap:
        raise ValueError(f"split years overlap: {overlap}")
    if max(train) >= min(validation) or max(validation) >= min(test):
        raise ValueError(
            "year blocks must be chronological: train before validation before test"
        )
    return train, validation, test


def select_blocked_cases(
    initialization_dates: Sequence[Any],
    *,
    train_years: Sequence[int] = (2017, 2018),
    validation_years: Sequence[int] = (2019,),
    test_years: Sequence[int] = (2020,),
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
    cases_per_year: int = DEFAULT_CASES_PER_YEAR,
) -> SelectedCases:
    """Select the same deterministic seasonal case count in every year.

    Candidates are divided into equally sized ordinal bins within each year
    and the center case of each bin is selected.  No grid cell from a case is
    assigned to another split.
    """

    train, validation, test = validate_blocked_years(
        train_years, validation_years, test_years
    )
    months = _canonical_ints(
        season_months, name="season_months", minimum=1, maximum=12
    )
    count = int(cases_per_year)
    if count < 1:
        raise ValueError("cases_per_year must be positive")
    dates = pd.DatetimeIndex(initialization_dates).normalize()
    if dates.ndim != 1 or len(dates) == 0:
        raise ValueError("initialization_dates must be a non-empty one-dimensional index")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(
            "initialization_dates must be finite, unique, and strictly increasing"
        )

    records: list[tuple[pd.Timestamp, int, str]] = []
    for label, years in (("train", train), ("validation", validation), ("test", test)):
        for year in years:
            candidates = np.flatnonzero(
                (dates.year == year) & np.isin(dates.month, np.asarray(months))
            )
            if len(candidates) < count:
                raise ValueError(
                    f"year {year} has {len(candidates)} seasonal cases; need {count}"
                )
            # Centers of equal ordinal bins avoid selecting only the start of
            # the season while remaining invariant to random seeds.
            positions = np.floor(
                (np.arange(count, dtype=np.float64) + 0.5) * len(candidates) / count
            ).astype(np.int64)
            chosen = candidates[positions]
            if len(np.unique(chosen)) != count:
                raise RuntimeError("deterministic seasonal selection produced duplicates")
            records.extend((dates[index], int(index), label) for index in chosen)

    records.sort(key=lambda item: item[0])
    return SelectedCases(
        source_indices=np.asarray([item[1] for item in records], dtype=np.int64),
        init_dates=pd.DatetimeIndex([item[0] for item in records]),
        splits=np.asarray([item[2] for item in records], dtype="U10"),
    )


def aggregate_fuxi_tp(native_tp: np.ndarray) -> np.ndarray:
    """Return D19--25 and D26--32 member accumulations in millimetres.

    ``native_tp`` is the already selected 14-day FuXi slice with shape
    ``[member,14,latitude,longitude]``.  FuXi TP is treated as a 24-hour mean
    rate: negative round-off is clipped, values are multiplied by 24 to
    ``mm day-1``, and seven days are summed.
    """

    values = np.asarray(native_tp, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != N_MEMBER or values.shape[1] != 14:
        raise ValueError(
            "native_tp must have shape [51 members,14 days,latitude,longitude]"
        )
    if not np.isfinite(values).all():
        raise ValueError("FuXi TP contains non-finite values")
    daily = np.clip(values, 0.0, None) * np.float32(24.0)
    weekly = daily.reshape(N_MEMBER, N_LEAD, 7, *values.shape[-2:]).sum(axis=2)
    result = np.moveaxis(weekly, 1, 0).astype(np.float32, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise RuntimeError("invalid FuXi weekly accumulations")
    return result


def _exact_time_positions(
    available: pd.DatetimeIndex, requested: pd.DatetimeIndex
) -> np.ndarray:
    positions = available.get_indexer(requested)
    if np.any(positions < 0):
        missing = requested[np.flatnonzero(positions < 0)]
        raise KeyError(
            f"IMERG is missing {len(missing)} required daily dates; first is "
            f"{missing[0].date()}"
        )
    return positions.astype(np.int64, copy=False)


def imerg_weekly_sums(
    imerg: xr.Dataset, week_starts: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage-aware seven-day IMERG sums and coverage fractions.

    A daily value is weighted by its represented target-cell area.  The
    weighted daily mean is multiplied by seven so a complete week is exactly
    the sum of its seven daily ``mm day-1`` fields.  Cells with no represented
    area are returned as NaN with zero coverage.
    """

    if "tp" not in imerg or "observation_fraction" not in imerg:
        raise ValueError("IMERG must contain tp and observation_fraction")
    starts = pd.DatetimeIndex(week_starts).normalize()
    if len(starts) == 0 or starts.hasnans:
        raise ValueError("week_starts must contain finite dates")
    daily_dates = pd.DatetimeIndex(
        np.asarray(
            [start + pd.Timedelta(days=day) for start in starts for day in range(7)],
            dtype="datetime64[ns]",
        )
    )
    unique_dates, inverse = np.unique(daily_dates.values, return_inverse=True)
    available = pd.DatetimeIndex(imerg.time.values).normalize()
    positions = _exact_time_positions(available, pd.DatetimeIndex(unique_dates))
    tp_unique = np.asarray(imerg["tp"].isel(time=positions).values, dtype=np.float32)
    fraction_unique = np.asarray(
        imerg["observation_fraction"].isel(time=positions).values, dtype=np.float32
    )
    tp = tp_unique[inverse].reshape(len(starts), 7, *tp_unique.shape[-2:])
    fraction = fraction_unique[inverse].reshape(
        len(starts), 7, *fraction_unique.shape[-2:]
    )
    finite_fraction = np.isfinite(fraction)
    if np.any(fraction[finite_fraction] < 0.0) or np.any(fraction[finite_fraction] > 1.0):
        raise ValueError("IMERG observation_fraction must lie in [0,1]")
    valid = np.isfinite(tp) & finite_fraction & (fraction > 0.0)
    if np.any(tp[valid] < 0.0):
        raise ValueError("IMERG precipitation must be non-negative")
    numerator = np.where(valid, tp * fraction, 0.0).sum(axis=1, dtype=np.float64)
    denominator = np.where(valid, fraction, 0.0).sum(axis=1, dtype=np.float64)
    weekly = np.divide(
        numerator * 7.0,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    coverage = denominator / 7.0
    return weekly.astype(np.float32), coverage.astype(np.float32)


def _replace_calendar_year(value: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return value.replace(year=int(year))
    except ValueError:
        # Fixed handling for 29 February; the default JJAS selection never
        # reaches this branch, but the CLI accepts other seasons explicitly.
        return value.replace(year=int(year), day=28)


def fixed_imerg_climatology(
    imerg: xr.Dataset,
    valid_start: Any,
    *,
    min_weekly_coverage: float = DEFAULT_MIN_WEEKLY_COVERAGE,
    min_samples: int = DEFAULT_MIN_CLIMATOLOGY_SAMPLES,
) -> ClimatologyBounds:
    """Build four fixed 2002--2016 calendar-plus-offset boundaries."""

    coverage_threshold = float(min_weekly_coverage)
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("min_weekly_coverage must lie in (0,1]")
    sample_minimum = int(min_samples)
    total_samples = len(CLIMATOLOGY_YEARS) * len(CALENDAR_OFFSETS)
    if sample_minimum < 1 or sample_minimum > total_samples:
        raise ValueError(f"min_samples must lie in [1,{total_samples}]")
    target = pd.Timestamp(valid_start).normalize()
    starts = [
        _replace_calendar_year(target, year) + pd.Timedelta(days=offset)
        for year in CLIMATOLOGY_YEARS
        for offset in CALENDAR_OFFSETS
    ]
    samples, coverage = imerg_weekly_sums(imerg, starts)
    valid = np.isfinite(samples) & (coverage >= coverage_threshold)
    masked = np.where(valid, samples, np.nan)
    counts = valid.sum(axis=0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bounds = np.nanquantile(
            masked, BOUNDARY_QUANTILES, axis=0
        ).astype(np.float32)
    support = (counts >= sample_minimum) & np.isfinite(bounds).all(axis=0)
    bounds[:, ~support] = np.nan
    return ClimatologyBounds(bounds=bounds, valid_sample_count=counts, support=support)


def positive_jeffreys_anchor(
    weekly_members: np.ndarray,
    bounds: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Return positive five-category probabilities, uniform off support."""

    members = np.asarray(weekly_members, dtype=np.float32)
    thresholds = np.asarray(bounds, dtype=np.float32)
    mask = np.asarray(support, dtype=bool)
    if members.ndim != 3 or thresholds.shape != (4, *members.shape[-2:]):
        raise ValueError("members/bounds shapes do not match")
    if mask.shape != members.shape[-2:]:
        raise ValueError("support does not match the spatial grid")
    if not np.isfinite(members).all():
        raise ValueError("weekly FuXi members must be finite")
    valid = mask & np.isfinite(thresholds).all(axis=0)
    if np.any(np.diff(thresholds[:, valid], axis=0) < 0.0):
        raise ValueError("climatology bounds must be non-decreasing")
    result = np.full((N_CATEGORY, *members.shape[-2:]), 0.2, dtype=np.float32)
    if np.any(valid):
        categories = np.sum(
            members[:, None, :, :] >= thresholds[None, :, :, :], axis=1
        )
        counts = np.stack(
            [(categories == category).sum(axis=0) for category in range(N_CATEGORY)]
        )
        probabilities = (counts + 0.5) / (members.shape[0] + 2.5)
        result[:, valid] = probabilities[:, valid]
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("Jeffreys anchor must be finite and strictly positive")
    np.testing.assert_allclose(result.sum(axis=0), 1.0, rtol=1.0e-6, atol=1.0e-7)
    return result


def imerg_category_target(
    weekly_truth: np.ndarray,
    bounds: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Categorize IMERG; equality moves upward and invalid cells become -1."""

    truth = np.asarray(weekly_truth, dtype=np.float32)
    thresholds = np.asarray(bounds, dtype=np.float32)
    mask = np.asarray(support, dtype=bool)
    if thresholds.shape != (4, *truth.shape) or mask.shape != truth.shape:
        raise ValueError("truth, bounds, and support shapes do not match")
    valid = mask & np.isfinite(truth) & np.isfinite(thresholds).all(axis=0)
    valid &= np.all(np.diff(thresholds, axis=0) >= 0.0, axis=0)
    # Match the precipitation arid-cell exclusion used by the official-style
    # path: four identical bounds do not define five meaningful categories.
    valid &= np.ptp(thresholds, axis=0) > 0.0
    target = np.sum(truth[None, :, :] >= thresholds, axis=0).astype(np.int8)
    target[~valid] = -1
    return target


def training_normalization(
    raw_quantiles: np.ndarray,
    support: np.ndarray,
    splits: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit lead/quantile statistics on supported training cells only."""

    fields = np.asarray(raw_quantiles, dtype=np.float32)
    valid_support = np.asarray(support, dtype=bool)
    labels = np.asarray(splits).astype(str)
    if fields.ndim != 5 or fields.shape[1:3] != (N_LEAD, 5):
        raise ValueError("raw_quantiles must be [case,2,5,latitude,longitude]")
    if valid_support.shape != fields.shape[:2] + fields.shape[-2:]:
        raise ValueError("support must be [case,2,latitude,longitude]")
    if labels.shape != (fields.shape[0],):
        raise ValueError("splits must contain one label per case")
    train_indices = np.flatnonzero(labels == "train")
    if len(train_indices) == 0:
        raise ValueError("normalization requires at least one training case")
    mean = np.empty((N_LEAD, 5), dtype=np.float32)
    std = np.empty_like(mean)
    for lead in range(N_LEAD):
        for quantile in range(5):
            values = np.concatenate(
                [
                    fields[index, lead, quantile][valid_support[index, lead]]
                    for index in train_indices
                ]
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                raise ValueError(
                    f"normalization channel lead={lead} quantile={quantile} is empty"
                )
            mean[lead, quantile] = np.mean(values, dtype=np.float64)
            variance = np.var(values, dtype=np.float64)
            std[lead, quantile] = np.sqrt(max(float(variance), 1.0e-6))
    return mean, std


def build_compatible_features(
    anchors: np.ndarray,
    raw_quantiles: np.ndarray,
    init_dates: Sequence[Any],
    latitude: np.ndarray,
    longitude: np.ndarray,
    static_support_fraction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Build the existing 18-channel precipitation feature contract."""

    p0 = np.asarray(anchors, dtype=np.float32)
    quantiles = np.asarray(raw_quantiles, dtype=np.float32)
    dates = pd.DatetimeIndex(init_dates).normalize()
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    static_support = np.asarray(static_support_fraction, dtype=np.float32)
    if p0.ndim != 5 or p0.shape[1:3] != (N_LEAD, N_CATEGORY):
        raise ValueError("anchors must be [case,2,5,latitude,longitude]")
    if quantiles.shape != p0.shape:
        raise ValueError("raw_quantiles must match anchors")
    if len(dates) != p0.shape[0] or static_support.shape != p0.shape[-2:]:
        raise ValueError("dates or static support do not match anchors")
    if mean.shape != (2, 5) or std.shape != (2, 5) or np.any(std <= 0.0):
        raise ValueError("normalization mean/std must have shape [2,5] and positive std")
    features = np.empty(
        (p0.shape[0], N_LEAD, N_FEATURE, *p0.shape[-2:]), dtype=np.float32
    )
    lat_radians = np.deg2rad(lat)[:, None]
    lon_radians = np.deg2rad(lon)[None, :]
    for case, init_date in enumerate(dates):
        features[case, :, :5] = np.log(np.maximum(p0[case], 1.0e-8))
        features[case, :, 5:10] = (
            quantiles[case] - mean[:, :, None, None]
        ) / std[:, :, None, None]
        features[case, :, 10] = np.sin(lat_radians)
        features[case, :, 11] = np.cos(lat_radians)
        features[case, :, 12] = np.sin(lon_radians)
        features[case, :, 13] = np.cos(lon_radians)
        for lead, (start_day, _) in enumerate(LEAD_WINDOWS):
            valid_start = init_date + pd.Timedelta(days=start_day - 1)
            phase = 2.0 * np.pi * (valid_start.dayofyear - 1) / 365.2425
            features[case, lead, 14] = np.sin(phase)
            features[case, lead, 15] = np.cos(phase)
            features[case, lead, 16] = -1.0 if lead == 0 else 1.0
        features[case, :, 17] = static_support
    if not np.isfinite(features).all():
        raise RuntimeError("compatible features contain non-finite values")
    return features


def assemble_npz_payload(
    *,
    init_dates: Sequence[Any],
    splits: Sequence[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
    anchors: np.ndarray,
    raw_quantiles: np.ndarray,
    targets: np.ndarray,
    weekly_truth: np.ndarray,
    weekly_coverage: np.ndarray,
    support: np.ndarray,
    thresholds: np.ndarray,
    climatology_sample_count: np.ndarray,
    provenance: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Assemble a validated, trainer-compatible NPZ payload."""

    dates = pd.DatetimeIndex(init_dates).normalize()
    labels = np.asarray(splits).astype("U10")
    p0 = np.asarray(anchors, dtype=np.float32)
    raw = np.asarray(raw_quantiles, dtype=np.float32)
    truth = np.asarray(weekly_truth, dtype=np.float32)
    coverage = np.asarray(weekly_coverage, dtype=np.float32)
    valid = np.asarray(support, dtype=bool)
    category = np.asarray(targets, dtype=np.int8)
    expected_field = (len(dates), N_LEAD, *p0.shape[-2:])
    if p0.shape != (len(dates), N_LEAD, N_CATEGORY, *p0.shape[-2:]):
        raise ValueError("anchors have an invalid shape")
    if raw.shape != p0.shape:
        raise ValueError("raw_quantiles must match anchors")
    for name, value in (
        ("targets", category),
        ("weekly_truth", truth),
        ("weekly_coverage", coverage),
        ("support", valid),
    ):
        if value.shape != expected_field:
            raise ValueError(f"{name} must have shape {expected_field}")
    if labels.shape != (len(dates),):
        raise ValueError("splits must contain one label per date")
    if not set(labels).issubset({"train", "validation", "test"}):
        raise ValueError("unexpected split label")
    if thresholds.shape != (len(dates), N_LEAD, 4, *p0.shape[-2:]):
        raise ValueError("thresholds have an invalid shape")
    if climatology_sample_count.shape != expected_field:
        raise ValueError("climatology_sample_count has an invalid shape")
    if not np.isfinite(p0).all() or np.any(p0 <= 0.0):
        raise ValueError("anchors must be finite and strictly positive")
    np.testing.assert_allclose(p0.sum(axis=2), 1.0, rtol=1.0e-6, atol=1.0e-7)
    if np.any((category < -1) | (category > 4)):
        raise ValueError("targets must contain -1 or 0..4")
    if np.any(category[~valid] != -1):
        raise ValueError("unsupported targets must be encoded as -1")

    mean, std = training_normalization(raw, valid, labels)
    train_indices = np.flatnonzero(labels == "train")
    static_support = valid[train_indices].mean(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    features = build_compatible_features(
        p0,
        raw,
        dates,
        latitude,
        longitude,
        static_support,
        mean,
        std,
    )
    init_yyyymmdd = np.asarray(
        [int(date.strftime("%Y%m%d")) for date in dates], dtype=np.int32
    )
    valid_starts = np.asarray(
        [
            [
                int((date + pd.Timedelta(days=start - 1)).strftime("%Y%m%d"))
                for start, _ in LEAD_WINDOWS
            ]
            for date in dates
        ],
        dtype=np.int32,
    )
    complete_provenance = dict(provenance)
    complete_provenance.update(
        {
            "schema": "fuxi_imerg_exploratory_probability_npz_v1",
            "purpose": "exploratory_pretraining",
            "official_ai_weather_quest_validation": False,
            "official_era5_target": False,
            "notice": EXPLORATORY_NOTICE,
            "fuxi_permission": FUXI_PERMISSION_NOTICE,
            "normalization": "lead/quantile mean and std from supported training cells only",
            "spatial_support_fraction": (
                "fraction of training case-leads with IMERG/climatology support; "
                "not a geographic land-sea mask"
            ),
            "feature_names": list(FEATURE_NAMES),
            "selected_initializations": [str(date.date()) for date in dates],
            "split_labels": labels.tolist(),
        }
    )
    contract_document = {
        "provenance": complete_provenance,
        "feature_names": list(FEATURE_NAMES),
        "tp_quantile_mean": mean.tolist(),
        "tp_quantile_std": std.tolist(),
        "init_dates": init_yyyymmdd.tolist(),
        "split": labels.tolist(),
    }
    contract_json = json.dumps(
        contract_document, sort_keys=True, separators=(",", ":"), default=str
    )
    contract_sha256 = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return {
        # These names are the direct trainer/predictor/evaluator contract.
        "features": features.astype(np.float16),
        "p0": p0.astype(np.float16),
        "target": category,
        "init_dates": init_yyyymmdd,
        "latitude": np.asarray(latitude, dtype=np.float64),
        "longitude": np.asarray(longitude, dtype=np.float64),
        "spatial_support_fraction": static_support,
        # The remaining arrays preserve the exploratory scientific provenance.
        "split": labels,
        "valid_start_yyyymmdd": valid_starts,
        "raw_tp_quantiles": raw.astype(np.float16),
        "thresholds": np.asarray(thresholds, dtype=np.float32),
        "imerg_weekly_sum": truth,
        "imerg_weekly_coverage": coverage.astype(np.float16),
        "imerg_support_mask": valid,
        "climatology_sample_count": np.asarray(
            climatology_sample_count, dtype=np.uint8
        ),
        "tp_quantile_mean": mean,
        "tp_quantile_std": std,
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
        "cache_contract_sha256": np.asarray(contract_sha256),
        "provenance_json": np.asarray(
            json.dumps(complete_provenance, sort_keys=True, default=str)
        ),
    }


def write_npz_atomic(
    output: Path, payload: Mapping[str, np.ndarray], *, overwrite: bool = False
) -> Path:
    """Atomically write one compressed NPZ without silently replacing a run."""

    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output must end in .npz")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **payload)
        with np.load(temporary_path, allow_pickle=False) as archive:
            required = {
                "features",
                "p0",
                "target",
                "init_dates",
                "latitude",
                "longitude",
                "spatial_support_fraction",
                "cache_contract_sha256",
                "provenance_json",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"temporary NPZ is missing {sorted(missing)}")
            if archive["features"].shape[0] != archive["init_dates"].shape[0]:
                raise RuntimeError("temporary NPZ case count is inconsistent")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _validate_sources(fuxi: xr.Dataset, imerg: xr.Dataset) -> None:
    if fuxi.attrs.get("status") != "complete":
        raise RuntimeError("FuXi source is not marked complete")
    expected = {
        "init": 2080,
        "member": N_MEMBER,
        "lead_day": 42,
        "channel": 26,
        "lat": N_LATITUDE,
        "lon": N_LONGITUDE,
    }
    for name, size in expected.items():
        if fuxi.sizes.get(name) != size:
            raise ValueError(f"FuXi {name}={fuxi.sizes.get(name)}, expected {size}")
    if "forecast" not in fuxi or "tp" not in {str(value) for value in fuxi.channel.values}:
        raise ValueError("FuXi source has no forecast/tp channel")
    if not np.array_equal(fuxi.lead_day.values, np.arange(1, 43)):
        raise ValueError("FuXi lead_day must be 1..42")
    if "init_complete" in fuxi and not bool(fuxi.init_complete.all()):
        raise RuntimeError("FuXi contains incomplete initialization records")
    if imerg.attrs.get("status") != "complete":
        raise RuntimeError("IMERG source is not marked complete")
    if imerg.sizes.get("lat") != N_LATITUDE or imerg.sizes.get("lon") != N_LONGITUDE:
        raise ValueError("IMERG must use the native 121x240 FuXi grid")
    if not np.allclose(fuxi.lat.values, imerg.lat.values, atol=1.0e-6):
        raise ValueError("FuXi and IMERG latitudes differ")
    if not np.allclose(fuxi.lon.values, imerg.lon.values, atol=1.0e-6):
        raise ValueError("FuXi and IMERG longitudes differ")
    if str(imerg["tp"].attrs.get("units", "")).lower() != "mm day-1":
        raise ValueError("IMERG tp must explicitly use mm day-1")


def build_cache(
    *,
    fuxi_store: Path = FUXI_STORE,
    imerg_store: Path = IMERG_STORE,
    output: Path = DEFAULT_OUTPUT,
    train_years: Sequence[int] = (2017, 2018),
    validation_years: Sequence[int] = (2019,),
    test_years: Sequence[int] = (2020,),
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
    cases_per_year: int = DEFAULT_CASES_PER_YEAR,
    min_weekly_coverage: float = DEFAULT_MIN_WEEKLY_COVERAGE,
    min_climatology_samples: int = DEFAULT_MIN_CLIMATOLOGY_SAMPLES,
    overwrite: bool = False,
) -> Path:
    """Read real archives, reduce selected cases, and write one compatible NPZ."""

    fuxi = xr.open_zarr(Path(fuxi_store), consolidated=True, chunks=None)
    imerg = xr.open_zarr(Path(imerg_store), consolidated=True, chunks=None)
    _validate_sources(fuxi, imerg)
    selected = select_blocked_cases(
        fuxi.init.values,
        train_years=train_years,
        validation_years=validation_years,
        test_years=test_years,
        season_months=season_months,
        cases_per_year=cases_per_year,
    )
    height, width = N_LATITUDE, N_LONGITUDE
    case_count = len(selected.init_dates)
    p0 = np.empty((case_count, N_LEAD, 5, height, width), dtype=np.float32)
    raw_quantiles = np.empty_like(p0)
    targets = np.full((case_count, N_LEAD, height, width), -1, dtype=np.int8)
    truth = np.full((case_count, N_LEAD, height, width), np.nan, dtype=np.float32)
    coverage = np.zeros_like(truth)
    support = np.zeros_like(truth, dtype=bool)
    thresholds = np.full(
        (case_count, N_LEAD, 4, height, width), np.nan, dtype=np.float32
    )
    sample_count = np.zeros((case_count, N_LEAD, height, width), dtype=np.uint8)
    climatology_cache: dict[tuple[int, int], ClimatologyBounds] = {}

    for case, (source_index, init_date) in enumerate(
        zip(selected.source_indices, selected.init_dates)
    ):
        print(
            f"case {case + 1}/{case_count} {selected.splits[case]} "
            f"{init_date.date()}",
            flush=True,
        )
        native = np.asarray(
            fuxi.forecast.isel(init=int(source_index))
            .sel(channel="tp", lead_day=slice(19, 32))
            .transpose("member", "lead_day", "lat", "lon")
            .values,
            dtype=np.float32,
        )
        weekly_members = aggregate_fuxi_tp(native)
        member_quantiles = np.quantile(
            np.log1p(weekly_members), MEMBER_QUANTILES, axis=1
        ).astype(np.float32)
        raw_quantiles[case] = np.moveaxis(member_quantiles, 0, 1)

        for lead, (start_day, _) in enumerate(LEAD_WINDOWS):
            valid_start = init_date + pd.Timedelta(days=start_day - 1)
            weekly_truth, weekly_coverage = imerg_weekly_sums(imerg, [valid_start])
            key = (valid_start.month, valid_start.day)
            if key not in climatology_cache:
                climatology_cache[key] = fixed_imerg_climatology(
                    imerg,
                    valid_start,
                    min_weekly_coverage=min_weekly_coverage,
                    min_samples=min_climatology_samples,
                )
            climatology = climatology_cache[key]
            lead_support = (
                climatology.support
                & np.isfinite(weekly_truth[0])
                & (weekly_coverage[0] >= float(min_weekly_coverage))
            )
            truth[case, lead] = weekly_truth[0]
            coverage[case, lead] = weekly_coverage[0]
            support[case, lead] = lead_support
            thresholds[case, lead] = climatology.bounds
            sample_count[case, lead] = climatology.valid_sample_count
            p0[case, lead] = positive_jeffreys_anchor(
                weekly_members[lead], climatology.bounds, lead_support
            )
            targets[case, lead] = imerg_category_target(
                weekly_truth[0], climatology.bounds, lead_support
            )
            if not np.any(targets[case, lead] >= 0):
                raise RuntimeError(
                    f"{init_date.date()} lead {lead + 1} has no valid IMERG targets"
                )

    train, validation, test = validate_blocked_years(
        train_years, validation_years, test_years
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fuxi_source": str(Path(fuxi_store).resolve()),
        "fuxi_source_archive_manifest_sha256": fuxi.attrs.get(
            "archive_manifest_sha256", ""
        ),
        "fuxi_source_status": fuxi.attrs.get("status", ""),
        "fuxi_full_data_verification": fuxi.attrs.get("full_data_verification"),
        "imerg_source": str(Path(imerg_store).resolve()),
        "imerg_product": imerg.attrs.get("product", ""),
        "imerg_revision": imerg.attrs.get("revision", ""),
        "imerg_doi": imerg.attrs.get("doi", ""),
        "split_years": {
            "train": list(train),
            "validation": list(validation),
            "test": list(test),
        },
        "selection": {
            "season_months": [int(value) for value in season_months],
            "cases_per_year": int(cases_per_year),
            "policy": "center of equal ordinal bins within each year and season",
            "unit": "whole FuXi initialization case",
        },
        "lead_windows": [list(window) for window in LEAD_WINDOWS],
        "fuxi_tp_conversion": (
            "clip native values at zero, multiply by 24 to mm/day, then sum 7 days"
        ),
        "imerg_target_aggregation": (
            "coverage-weighted daily mean multiplied by 7; stored as mm/week"
        ),
        "climatology": {
            "years": list(CLIMATOLOGY_YEARS),
            "calendar_offsets_days": list(CALENDAR_OFFSETS),
            "sample_count": len(CLIMATOLOGY_YEARS) * len(CALENDAR_OFFSETS),
            "boundary_quantiles": list(BOUNDARY_QUANTILES),
            "min_weekly_coverage": float(min_weekly_coverage),
            "min_valid_samples": int(min_climatology_samples),
            "fixed_before_train_validation_test": True,
        },
        "anchor": "Jeffreys (count+0.5)/(51+2.5); unsupported cells uniform",
        "target_boundary_equality": "upper_category",
        "grid": "native FuXi global 1.5 degree 121x240",
    }
    payload = assemble_npz_payload(
        init_dates=selected.init_dates,
        splits=selected.splits,
        latitude=np.asarray(fuxi.lat.values, dtype=np.float64),
        longitude=np.asarray(fuxi.lon.values, dtype=np.float64),
        anchors=p0,
        raw_quantiles=raw_quantiles,
        targets=targets,
        weekly_truth=truth,
        weekly_coverage=coverage,
        support=support,
        thresholds=thresholds,
        climatology_sample_count=sample_count,
        provenance=provenance,
    )
    destination = write_npz_atomic(output, payload, overwrite=overwrite)
    print(EXPLORATORY_NOTICE)
    print(f"wrote {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuxi-store", type=Path, default=FUXI_STORE)
    parser.add_argument("--imerg-store", type=Path, default=IMERG_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-years", type=int, nargs="+", default=[2017, 2018])
    parser.add_argument("--validation-years", type=int, nargs="+", default=[2019])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2020])
    parser.add_argument(
        "--season-months", type=int, nargs="+", default=list(DEFAULT_SEASON_MONTHS)
    )
    parser.add_argument("--cases-per-year", type=int, default=DEFAULT_CASES_PER_YEAR)
    parser.add_argument(
        "--min-weekly-coverage", type=float, default=DEFAULT_MIN_WEEKLY_COVERAGE
    )
    parser.add_argument(
        "--min-climatology-samples",
        type=int,
        default=DEFAULT_MIN_CLIMATOLOGY_SAMPLES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_cache(
        fuxi_store=args.fuxi_store,
        imerg_store=args.imerg_store,
        output=args.output,
        train_years=args.train_years,
        validation_years=args.validation_years,
        test_years=args.test_years,
        season_months=args.season_months,
        cases_per_year=args.cases_per_year,
        min_weekly_coverage=args.min_weekly_coverage,
        min_climatology_samples=args.min_climatology_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
