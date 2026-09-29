"""Frozen scientific contract for the 2026-08-13 precipitation Quest run.

This module contains only deterministic array/date rules.  It deliberately has
no observation-availability dependent feature logic: observation coverage may
invalidate a target, but can never modify a predictor, climatology sample, or
FuXi anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


ISSUE_DATE = "20260813"
GRID_SHAPE = (121, 240)
N_MEMBERS = 51
N_CATEGORIES = 5
CONTEXT_WINDOWS = tuple((1 + 7 * i, 7 + 7 * i) for i in range(6))
TARGET_WINDOWS = ((19, 25), (26, 32))
CLIMATOLOGY_OFFSETS = (-4, -2, 0, 2, 4)
BOUNDARY_QUANTILES = (0.20, 0.40, 0.60, 0.80)
MEMBER_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
PHYSICAL_VARIABLES = ("tcwv", "q850", "q500", "q250", "t2m")
CONTEXT_DYNAMIC_NAMES = (
    "tp_q10",
    "tp_q25",
    "tp_q50",
    "tp_q75",
    "tp_q90",
    "tcwv_mean",
    "tcwv_std",
    "q850_mean",
    "q850_std",
    "q500_mean",
    "q500_std",
    "q250_mean",
    "q250_std",
    "t2m_mean",
    "t2m_std",
)
STATIC_NAMES = (
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "sin_valid_doy",
    "cos_valid_doy",
    "token_position",
    "land_fraction",
)


@dataclass(frozen=True)
class Split:
    train_years: tuple[int, ...] = tuple(range(2002, 2019))
    validation_years: tuple[int, ...] = (2019, 2020)
    test_years: tuple[int, ...] = (2021,)

    def label(self, year: int) -> str:
        if year in self.train_years:
            return "train"
        if year in self.validation_years:
            return "validation"
        if year in self.test_years:
            return "test"
        raise ValueError(f"year {year} is outside the frozen 2002-2021 split")


SPLIT = Split()


def canonical_contract() -> dict[str, object]:
    """Return the hashable run contract without machine-specific paths."""

    return {
        "schema": "ai-wq-precip-20260813-split17-2-1-v2",
        "issue_date": ISSUE_DATE,
        "grid_shape": list(GRID_SHAPE),
        "members": N_MEMBERS,
        "context_windows": [list(item) for item in CONTEXT_WINDOWS],
        "target_windows": [list(item) for item in TARGET_WINDOWS],
        "climatology_years": 20,
        "climatology_offsets_days": list(CLIMATOLOGY_OFFSETS),
        "boundary_quantiles": list(BOUNDARY_QUANTILES),
        "member_quantiles": list(MEMBER_QUANTILES),
        "train_years": list(SPLIT.train_years),
        "validation_years": list(SPLIT.validation_years),
        "test_years": list(SPLIT.test_years),
        "context_dynamic_names": list(CONTEXT_DYNAMIC_NAMES),
        "static_names": list(STATIC_NAMES),
        "era5_week": "28 six-hour accumulations S+06h through S+7d00h; m to mm",
        "equality_rule": "boundary equality enters upper category",
        "arid_mask": "target invalid where q80 precipitation boundary equals zero",
        "p0": "(unsmoothed_member_count+0.5)/53.5",
    }


def contract_sha256() -> str:
    encoded = json.dumps(
        canonical_contract(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def era5_week_times(valid_start: object) -> pd.DatetimeIndex:
    """Return exactly S+06h, ..., S+7d00h at six-hour spacing."""

    start = _timestamp(valid_start).normalize()
    return pd.date_range(
        start + pd.Timedelta(hours=6),
        start + pd.Timedelta(days=7),
        freq="6h",
    )


def aggregate_era5_six_hour_week(
    precipitation: xr.DataArray,
    valid_start: object,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Sum a WeatherBench2 ERA5 week and convert accumulated metres to mm."""

    if time_dim not in precipitation.dims:
        raise ValueError(f"precipitation has no {time_dim!r} dimension")
    expected = era5_week_times(valid_start)
    available = pd.DatetimeIndex(precipitation[time_dim].values)
    positions = available.get_indexer(expected)
    if np.any(positions < 0):
        missing = expected[np.flatnonzero(positions < 0)[0]]
        raise KeyError(f"missing ERA5 six-hour accumulation at {missing}")
    selected = precipitation.isel({time_dim: positions})
    if selected.sizes[time_dim] != 28:
        raise RuntimeError("an ERA5 precipitation week must contain 28 accumulations")
    values = np.asarray(selected.values)
    if not np.isfinite(values).all():
        raise ValueError("ERA5 precipitation week contains non-finite values")
    units = str(precipitation.attrs.get("units", "")).strip().lower()
    if units not in {"m", "metre", "metres", "meter", "meters"}:
        raise ValueError(f"expected six-hour ERA5 precipitation in metres, got {units!r}")
    result = selected.sum(time_dim, dtype=np.float64) * 1000.0
    result = result.astype(np.float32)
    result.attrs.update(units="mm", accumulation_count=28)
    return result


def _shift_year(value: pd.Timestamp, years: int) -> pd.Timestamp:
    try:
        return value.replace(year=value.year + years)
    except ValueError:  # 29 February follows the package's 28 February convention.
        return value.replace(year=value.year + years, day=28)


def climatology_sample_starts(valid_start: object) -> tuple[pd.Timestamp, ...]:
    start = _timestamp(valid_start).normalize()
    samples = tuple(
        _shift_year(start, -years) + pd.Timedelta(days=offset)
        for years in range(20, 0, -1)
        for offset in CLIMATOLOGY_OFFSETS
    )
    if len(samples) != 100 or any(item >= start for item in samples):
        raise RuntimeError("climatology must be exactly 100 prior-only samples")
    return samples


def climatology_boundaries(samples: np.ndarray) -> np.ndarray:
    """Compute q20/q40/q60/q80 from exactly 100 weekly fields."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim < 1 or values.shape[0] != 100:
        raise ValueError("climatology samples must have leading size 100")
    if not np.isfinite(values).all():
        raise ValueError("climatology samples must be finite")
    return np.quantile(values, BOUNDARY_QUANTILES, axis=0).astype(np.float32)


def categories(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Categorize values; equality with any boundary goes to the upper bin."""

    value = np.asarray(values)
    bounds = np.asarray(boundaries)
    if bounds.ndim < 2 or bounds.shape[0] != 4:
        raise ValueError("boundaries must have leading size four")
    spatial_shape = bounds.shape[1:]
    if value.shape[-len(spatial_shape) :] != spatial_shape:
        raise ValueError("values and four boundaries have incompatible shapes")
    expanded = np.expand_dims(value, axis=value.ndim - len(spatial_shape))
    category_axis = expanded.ndim - len(spatial_shape) - 1
    return np.sum(expanded >= bounds, axis=category_axis).astype(np.int8)


def observation_categories(
    observation: np.ndarray, boundaries: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observation = np.asarray(observation, dtype=np.float32)
    boundaries = np.asarray(boundaries, dtype=np.float32)
    if boundaries.shape != (4, *observation.shape):
        raise ValueError("boundaries must be [4, ...] and match observation")
    valid = np.isfinite(observation) & np.isfinite(boundaries).all(axis=0)
    # Match Quest scoring: persistently arid cells are those whose upper
    # climatological-quintile boundary is zero. Other tied boundaries remain
    # valid and use the declared upper-category equality rule.
    valid &= boundaries[3] > 0.0
    result = np.sum(observation[None] >= boundaries, axis=0).astype(np.int8)
    result[~valid] = -1
    return result, valid


def member_category_counts(members: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    members = np.asarray(members, dtype=np.float32)
    boundaries = np.asarray(boundaries, dtype=np.float32)
    if members.ndim < 2 or members.shape[0] != N_MEMBERS:
        raise ValueError(f"members must have leading size {N_MEMBERS}")
    if boundaries.shape != (4, *members.shape[1:]):
        raise ValueError("boundaries must be [4, ...] and match member fields")
    if not np.isfinite(members).all() or not np.isfinite(boundaries).all():
        raise ValueError("member counts require finite members and boundaries")
    member_category = np.sum(members[:, None] >= boundaries[None], axis=1)
    counts = np.stack(
        [(member_category == category).sum(axis=0) for category in range(5)], axis=0
    ).astype(np.uint8)
    if not np.all(counts.sum(axis=0) == N_MEMBERS):
        raise RuntimeError("member category counts do not sum to 51")
    return counts


def jeffreys_probabilities(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.shape[-3] != N_CATEGORIES:
        raise ValueError("category axis must have size five at position -3")
    if np.any(counts < 0) or not np.all(counts.sum(axis=-3) == N_MEMBERS):
        raise ValueError("unsmoothed counts must sum to 51")
    result = (counts.astype(np.float32) + 0.5) / 53.5
    if not np.all(result > 0.0) or not np.allclose(result.sum(axis=-3), 1.0):
        raise RuntimeError("invalid Jeffreys-smoothed probabilities")
    return result


def weekly_fuxi_fields(
    daily: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one 51-member, 42-day FuXi case to context and target summaries.

    FuXi ``tp`` is a 24-hour mean rate in mm h-1.  It is clipped at zero,
    multiplied by 24, and summed across each seven-day window.  State fields
    are averaged across the seven days before their ensemble mean/std is taken.
    """

    required = ("tp", *PHYSICAL_VARIABLES)
    missing = [name for name in required if name not in daily]
    if missing:
        raise KeyError(f"missing FuXi fields: {missing}")
    arrays = {name: np.asarray(daily[name], dtype=np.float32) for name in required}
    reference_shape = (N_MEMBERS, 42, *GRID_SHAPE)
    if any(value.shape != reference_shape for value in arrays.values()):
        shapes = {name: value.shape for name, value in arrays.items()}
        raise ValueError(f"FuXi fields must all have shape {reference_shape}; got {shapes}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("FuXi context contains non-finite values")

    tp_daily_mm = np.clip(arrays["tp"], 0.0, None) * np.float32(24.0)
    context = np.empty((6, 15, *GRID_SHAPE), dtype=np.float32)
    for week, (start_day, end_day) in enumerate(CONTEXT_WINDOWS):
        tp_week = tp_daily_mm[:, start_day - 1 : end_day].sum(axis=1)
        context[week, :5] = np.quantile(
            np.log1p(tp_week), MEMBER_QUANTILES, axis=0
        ).astype(np.float32)
        channel = 5
        for variable in PHYSICAL_VARIABLES:
            weekly_member = arrays[variable][:, start_day - 1 : end_day].mean(axis=1)
            context[week, channel] = weekly_member.mean(axis=0)
            context[week, channel + 1] = weekly_member.std(axis=0)
            channel += 2

    target_members = np.empty((2, N_MEMBERS, *GRID_SHAPE), dtype=np.float32)
    for lead, (start_day, end_day) in enumerate(TARGET_WINDOWS):
        target_members[lead] = tp_daily_mm[:, start_day - 1 : end_day].sum(axis=1)
    return context, target_members


def fit_train_land_normalization(
    fields: Iterable[np.ndarray], land_fraction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit [6,15] mean/std using only provided training fields over land."""

    land = np.asarray(land_fraction, dtype=np.float32) >= 0.5
    if land.shape != GRID_SHAPE or not land.any():
        raise ValueError("land fraction must define at least one land grid cell")
    total = np.zeros((6, 15), dtype=np.float64)
    square = np.zeros_like(total)
    count = np.zeros((6, 15), dtype=np.int64)
    seen = 0
    for value in fields:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (6, 15, *GRID_SHAPE):
            raise ValueError("normalization fields must be [6,15,121,240]")
        selected = array[:, :, land]
        finite = np.isfinite(selected)
        total += np.where(finite, selected, 0.0).sum(axis=-1)
        square += np.where(finite, selected * selected, 0.0).sum(axis=-1)
        count += finite.sum(axis=-1)
        seen += 1
    if seen == 0 or np.any(count == 0):
        raise ValueError("normalization requires non-empty training-only fields")
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def static_maps(
    initialization: object,
    windows: Sequence[tuple[int, int]],
    land_fraction: np.ndarray,
) -> np.ndarray:
    """Build the eight coordinate/time/static channels for arbitrary tokens."""

    initialization = _timestamp(initialization).normalize()
    land = np.asarray(land_fraction, dtype=np.float32)
    if land.shape != GRID_SHAPE or not np.isfinite(land).all():
        raise ValueError("land_fraction must be a finite 121x240 field")
    latitude = np.deg2rad(np.linspace(90.0, -90.0, GRID_SHAPE[0]))[:, None]
    longitude = np.deg2rad(
        np.arange(GRID_SHAPE[1], dtype=np.float64) * (360.0 / GRID_SHAPE[1])
    )[None, :]
    output = np.empty((len(windows), 8, *GRID_SHAPE), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, len(windows), dtype=np.float32)
    for token, ((start_day, end_day), position) in enumerate(zip(windows, positions)):
        midpoint = initialization + pd.Timedelta(days=(start_day + end_day - 2) / 2)
        angle = 2.0 * np.pi * (midpoint.dayofyear - 1) / 365.2425
        output[token, 0] = np.sin(latitude)
        output[token, 1] = np.cos(latitude)
        output[token, 2] = np.sin(longitude)
        output[token, 3] = np.cos(longitude)
        output[token, 4] = np.sin(angle)
        output[token, 5] = np.cos(angle)
        output[token, 6] = position
        output[token, 7] = land
    return output


def build_model_inputs(
    context_dynamic: np.ndarray,
    target_quantiles: np.ndarray,
    p0: np.ndarray,
    initialization: object,
    land_fraction: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    target_normalization_mean: np.ndarray | None = None,
    target_normalization_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the frozen ``context_x`` and ``target_x`` model interfaces."""

    context = np.asarray(context_dynamic, dtype=np.float32)
    quantiles = np.asarray(target_quantiles, dtype=np.float32)
    anchor = np.asarray(p0, dtype=np.float32)
    mean = np.asarray(normalization_mean, dtype=np.float32)
    std = np.asarray(normalization_std, dtype=np.float32)
    if context.shape != (6, 15, *GRID_SHAPE):
        raise ValueError("context_dynamic must be [6,15,121,240]")
    if quantiles.shape != (2, 5, *GRID_SHAPE):
        raise ValueError("target_quantiles must be [2,5,121,240]")
    if anchor.shape != (2, 5, *GRID_SHAPE) or np.any(anchor <= 0.0):
        raise ValueError("p0 must be positive [2,5,121,240]")
    if mean.shape != (6, 15) or std.shape != (6, 15) or np.any(std <= 0.0):
        raise ValueError("normalization must have shape [6,15] with positive std")
    target_mean = (
        np.zeros((2, 5), dtype=np.float32)
        if target_normalization_mean is None
        else np.asarray(target_normalization_mean, dtype=np.float32)
    )
    target_std = (
        np.ones((2, 5), dtype=np.float32)
        if target_normalization_std is None
        else np.asarray(target_normalization_std, dtype=np.float32)
    )
    if target_mean.shape != (2, 5) or target_std.shape != (2, 5) or np.any(target_std <= 0):
        raise ValueError("target normalization must be [2,5] with positive std")

    normalized = (context - mean[:, :, None, None]) / std[:, :, None, None]
    context_x = np.concatenate(
        (normalized, static_maps(initialization, CONTEXT_WINDOWS, land_fraction)), axis=1
    )
    target_x = np.concatenate(
        (
            np.log(np.maximum(anchor, 1.0e-8)),
            (quantiles - target_mean[:, :, None, None])
            / target_std[:, :, None, None],
            static_maps(initialization, TARGET_WINDOWS, land_fraction),
        ),
        axis=1,
    )
    if context_x.shape != (6, 23, *GRID_SHAPE) or target_x.shape != (
        2,
        18,
        *GRID_SHAPE,
    ):
        raise RuntimeError("constructed model inputs violate the frozen interface")
    return context_x.astype(np.float32), target_x.astype(np.float32)


def validate_archive_dates(values: Sequence[object]) -> None:
    dates = pd.DatetimeIndex(values).normalize()
    if len(dates) != 2080 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("FuXi archive must contain 2,080 unique increasing dates")
    counts = dates.year.value_counts().sort_index()
    if tuple(counts.index) != tuple(range(2002, 2022)) or not np.all(counts == 104):
        raise ValueError("FuXi archive must contain 104 cases in each year 2002-2021")


__all__ = [
    "BOUNDARY_QUANTILES",
    "CLIMATOLOGY_OFFSETS",
    "CONTEXT_DYNAMIC_NAMES",
    "CONTEXT_WINDOWS",
    "GRID_SHAPE",
    "ISSUE_DATE",
    "MEMBER_QUANTILES",
    "N_MEMBERS",
    "PHYSICAL_VARIABLES",
    "SPLIT",
    "STATIC_NAMES",
    "TARGET_WINDOWS",
    "aggregate_era5_six_hour_week",
    "build_model_inputs",
    "canonical_contract",
    "climatology_boundaries",
    "climatology_sample_starts",
    "contract_sha256",
    "era5_week_times",
    "fit_train_land_normalization",
    "jeffreys_probabilities",
    "member_category_counts",
    "observation_categories",
    "static_maps",
    "validate_archive_dates",
    "weekly_fuxi_fields",
]
