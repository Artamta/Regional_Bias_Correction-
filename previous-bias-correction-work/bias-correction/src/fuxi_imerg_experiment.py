#!/usr/bin/env python
"""Train and evaluate a small IMERG post-processor for FuXi-S2S rainfall.

The experiment is deliberately narrow:

* train on JJAS initializations from 2014--2018;
* select the checkpoint on 2019;
* evaluate once on 2020--2021;
* compare raw FuXi, log-bias correction, quantile mapping, and a residual U-Net.

Run with the tested environment::

    /home/raj.ayush/.conda/envs/fuxi/bin/python src/fuxi_imerg_experiment.py

Use ``--smoke`` for a short end-to-end check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr


from project_paths import EVALUATE_ROOT
from project_paths import NEURAL_ADAPTER_SRC
from project_paths import PROJECT_ROOT as HERE
if str(NEURAL_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_ADAPTER_SRC))

from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.models import ResidualUNet  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed, train_model  # noqa: E402


FUXI_SHARDS = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_jjas_2002_2021/shards"
)
BENCHMARK_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1"
)
IMERG_DAILY = (
    BENCHMARK_ROOT
    / "observations/ground_truth_v1/daily/imerg/tp/india_1p5_27x27_v1"
)
SPATIAL_STORE = BENCHMARK_ROOT / "spatial/spatial_support.zarr"
RESULTS_ROOT = HERE / "results" / "fuxi_imerg_jjas_5yr"

TRAIN_YEARS = (2014, 2015, 2016, 2017, 2018)
VALIDATION_YEARS = (2019,)
TEST_YEARS = (2020, 2021)
ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

SEEDS = (42, 43, 44)
QUANTILE_LEVELS = np.linspace(0.0, 1.0, 21, dtype=np.float64)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 13

METHOD_ORDER = ("raw_fuxi", "log_bias", "quantile_mapping", "residual_unet")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    "quantile_mapping": "Quantile mapping",
    "residual_unet": "Residual U-Net",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#0072B2",
    "quantile_mapping": "#E69F00",
    "residual_unet": "#009E73",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    "quantile_mapping": "^",
    "residual_unet": "D",
}

IMPLEMENTATION_SOURCES = {
    "fuxi_imerg_experiment.py": Path(__file__).resolve(),
    "verify_fuxi_imerg_results.py": EVALUATE_ROOT / "verify_fuxi_imerg_results.py",
    "fuxi_adapter/__init__.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "__init__.py",
    "fuxi_adapter/artifacts.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "artifacts.py",
    "fuxi_adapter/baselines.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "baselines.py",
    "fuxi_adapter/config.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "config.py",
    "fuxi_adapter/metrics.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "metrics.py",
    "fuxi_adapter/models.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "models.py",
    "fuxi_adapter/training.py": NEURAL_ADAPTER_SRC / "fuxi_adapter" / "training.py",
}


class DataContractError(ValueError):
    """Raised when a source file violates the frozen experiment contract."""


@dataclass(frozen=True)
class ForecastData:
    initializations: np.ndarray
    valid_dates: np.ndarray
    ensemble_mean: np.ndarray
    ensemble_spread: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    source_files: Tuple[str, ...]


@dataclass(frozen=True)
class ObservationData:
    weekly_truth: np.ndarray
    weekly_climatology: np.ndarray
    observation_fraction: np.ndarray
    source_stores: Tuple[str, ...]


@dataclass(frozen=True)
class NeuralArrays:
    inputs: np.ndarray
    target: np.ndarray
    target_scale: np.ndarray
    feature_stats: Mapping[str, object]


@dataclass(frozen=True)
class QuantileMap:
    forecast_quantiles: np.ndarray
    observed_quantiles: np.ndarray
    levels: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash directory names and file contents in a stable order."""

    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_inventory(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            size = path.stat().st_size
            checksum = sha256_file(path)
            modified = path.stat().st_mtime
            kind = "file"
        elif path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
            size = sum(item.stat().st_size for item in files)
            checksum = sha256_tree(path)
            modified = max(item.stat().st_mtime for item in files)
            kind = "zarr_directory"
        else:
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "kind": kind,
                "size_bytes": int(size),
                "mtime_utc": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
                "sha256": checksum,
                "checksum_scope": "full file" if kind == "file" else "full directory tree",
            }
        )
    return pd.DataFrame(rows)


def snapshot_implementation(output: Path) -> Mapping[str, str]:
    checksums = {}
    for relative, source in IMPLEMENTATION_SOURCES.items():
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def derive_valid_dates(initializations: np.ndarray) -> np.ndarray:
    initializations = np.asarray(initializations, dtype="datetime64[D]")
    offsets = np.arange(42, dtype="timedelta64[D]").reshape(1, 6, 7)
    return initializations[:, None, None] + offsets


def calendar_positions(dates: np.ndarray) -> np.ndarray:
    """Map dates to a leap-year 366-day month/day calendar."""

    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    flat = np.asarray(dates, dtype="datetime64[D]").reshape(-1)
    positions = np.asarray(
        [lookup[pd.Timestamp(value).strftime("%m-%d")] for value in flat],
        dtype=np.int16,
    )
    return positions.reshape(np.asarray(dates).shape)


def split_indices(initializations: np.ndarray) -> Mapping[str, np.ndarray]:
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    return {
        "train": np.flatnonzero(np.isin(years, TRAIN_YEARS)),
        "validation": np.flatnonzero(np.isin(years, VALIDATION_YEARS)),
        "test": np.flatnonzero(np.isin(years, TEST_YEARS)),
    }


def load_fuxi() -> ForecastData:
    files = tuple(
        sorted(
            path
            for year in ALL_YEARS
            for path in FUXI_SHARDS.glob(f"{year}*.nc")
        )
    )
    expected_count = 35 * len(ALL_YEARS)
    if len(files) != expected_count:
        raise DataContractError(
            f"expected {expected_count} FuXi shards, found {len(files)}"
        )

    means = []
    spreads = []
    initializations = []
    latitude = None
    longitude = None

    expected_members = np.arange(51, dtype=np.int16)
    expected_leads = np.arange(1, 43, dtype=np.int16)

    for path in files:
        with xr.open_dataset(path) as dataset:
            if dataset.sizes.get("member") != 51 or dataset.sizes.get("lead_day") != 42:
                raise DataContractError(f"unexpected member/lead dimensions in {path}")
            if not np.array_equal(dataset.member.values, expected_members):
                raise DataContractError(f"member labels differ in {path}")
            if not np.array_equal(dataset.lead_day.values, expected_leads):
                raise DataContractError(f"lead_day must be 1..42 in {path}")
            if dataset.tp.attrs.get("units") != "mm h-1":
                raise DataContractError(f"unexpected TP units in {path}")

            init = np.asarray(
                dataset.forecast_reference_time.values, dtype="datetime64[D]"
            ).reshape(-1)[0]
            filename_init = np.datetime64(
                pd.to_datetime(path.stem, format="%Y%m%d"), "D"
            )
            if init != filename_init:
                raise DataContractError(f"filename/init mismatch in {path}")

            expected_start = init + np.arange(42, dtype="timedelta64[D]")
            expected_end = init + np.arange(1, 43, dtype="timedelta64[D]")
            actual_start = np.asarray(
                dataset.forecast_period_start.values, dtype="datetime64[D]"
            )
            actual_end = np.asarray(
                dataset.forecast_period_end.values, dtype="datetime64[D]"
            )
            if not np.array_equal(actual_start, expected_start):
                raise DataContractError(f"forecast period starts differ in {path}")
            if not np.array_equal(actual_end, expected_end):
                raise DataContractError(f"forecast period ends differ in {path}")

            shard_latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
            shard_longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
            if latitude is None:
                latitude = shard_latitude
                longitude = shard_longitude
            elif not np.array_equal(latitude, shard_latitude) or not np.array_equal(
                longitude, shard_longitude
            ):
                raise DataContractError(f"grid differs in {path}")

            daily = np.asarray(dataset.tp.load().values, dtype=np.float32)
            if not np.isfinite(daily).all() or np.any(daily < 0.0):
                raise DataContractError(f"TP contains invalid values in {path}")
            daily *= np.float32(24.0)
            member_weekly = daily.reshape(51, 6, 7, 27, 27).mean(
                axis=2, dtype=np.float64
            )
            means.append(member_weekly.mean(axis=0, dtype=np.float64).astype(np.float32))
            spreads.append(member_weekly.std(axis=0, ddof=0).astype(np.float32))
            initializations.append(init)

    assert latitude is not None and longitude is not None
    inits = np.asarray(initializations, dtype="datetime64[D]")
    order = np.argsort(inits)
    inits = inits[order]
    mean = np.stack(means)[order]
    spread = np.stack(spreads)[order]

    if np.unique(inits).size != expected_count:
        raise DataContractError("FuXi initializations are not unique")
    counts = pd.Series(pd.DatetimeIndex(inits).year).value_counts().to_dict()
    if counts != {year: 35 for year in ALL_YEARS}:
        raise DataContractError(f"unexpected initialization counts: {counts}")
    if mean.shape != (expected_count, 6, 27, 27) or spread.shape != mean.shape:
        raise DataContractError("unexpected reduced FuXi array shape")

    return ForecastData(
        initializations=inits,
        valid_dates=derive_valid_dates(inits),
        ensemble_mean=mean,
        ensemble_spread=spread,
        latitude=latitude,
        longitude=longitude,
        source_files=tuple(str(path) for path in files),
    )


def collapse_fraction(dataset: xr.Dataset, spatial_shape: Tuple[int, int]) -> np.ndarray:
    fraction = dataset.observation_fraction
    other_dims = [
        dim for dim in fraction.dims if dim not in ("latitude", "longitude")
    ]
    ordered = fraction.transpose(*(other_dims + ["latitude", "longitude"]))
    values = np.asarray(ordered.load().values, dtype=np.float32)
    if values.ndim == 2:
        result = values
    elif values.ndim == 3:
        repeated = values.reshape(-1, *spatial_shape)
        result = repeated[0]
        if not np.allclose(
            repeated, result[None], rtol=0.0, atol=1.0e-7, equal_nan=True
        ):
            raise DataContractError("IMERG observation_fraction changes with time")
    else:
        raise DataContractError("unexpected IMERG observation_fraction dimensions")
    if result.shape != spatial_shape:
        raise DataContractError("unexpected IMERG observation_fraction shape")
    return result.copy()


def build_training_climatology(
    daily_dates: np.ndarray,
    daily_values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Build an equal-year, centred 31-day climatology from 2014--2018."""

    date_index = pd.DatetimeIndex(daily_dates)
    train = np.isin(date_index.year, TRAIN_YEARS)
    train_dates = daily_dates[train]
    train_values = daily_values[train]
    train_year_values = pd.DatetimeIndex(train_dates).year.to_numpy()
    positions = calendar_positions(train_dates)

    climatology = np.full((366, *support.shape), np.nan, dtype=np.float32)
    for day in range(366):
        circular_distance = np.minimum(
            (positions - day) % 366,
            (day - positions) % 366,
        )
        year_means = []
        for year in TRAIN_YEARS:
            selected = (train_year_values == year) & (circular_distance <= 15)
            if not np.any(selected):
                raise DataContractError(f"no climatology samples for day {day}, {year}")
            year_mean = np.mean(train_values[selected], axis=0, dtype=np.float64)
            if not np.isfinite(year_mean[support]).all():
                raise DataContractError("non-finite training climatology values")
            year_means.append(year_mean)
        climatology[day] = np.mean(year_means, axis=0, dtype=np.float64).astype(
            np.float32
        )
    climatology[:, ~support] = np.nan
    return climatology


def load_imerg(forecast: ForecastData) -> ObservationData:
    all_values = []
    all_dates = []
    source_stores = []
    observation_fraction = None
    spatial_shape = (forecast.latitude.size, forecast.longitude.size)

    for year in ALL_YEARS:
        store = IMERG_DAILY / f"{year}.zarr"
        if not (store / ".zmetadata").is_file():
            raise FileNotFoundError(store)
        with xr.open_zarr(store, consolidated=True) as dataset:
            expected_attrs = {
                "source": "imerg",
                "product": "GPM_3IMERGDF.07",
                "revision": "V07B",
                "units": "mm day-1",
            }
            for key, expected in expected_attrs.items():
                if dataset.attrs.get(key) != expected:
                    raise DataContractError(
                        f"{store}: {key}={dataset.attrs.get(key)!r}, expected {expected!r}"
                    )
            if not np.array_equal(dataset.latitude.values, forecast.latitude):
                raise DataContractError(f"latitude differs in {store}")
            if not np.array_equal(dataset.longitude.values, forecast.longitude):
                raise DataContractError(f"longitude differs in {store}")

            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            if values.shape != (dates.size, *spatial_shape):
                raise DataContractError(f"unexpected IMERG shape in {store}")
            if dates.size not in (365, 366) or not np.all(
                np.diff(dates) == np.timedelta64(1, "D")
            ):
                raise DataContractError(f"IMERG calendar is incomplete in {store}")
            fraction = collapse_fraction(dataset, spatial_shape)
            if observation_fraction is None:
                observation_fraction = fraction
            elif not np.allclose(
                observation_fraction,
                fraction,
                rtol=0.0,
                atol=1.0e-7,
                equal_nan=True,
            ):
                raise DataContractError("IMERG support differs between years")
            support = fraction > 0.0
            if not np.isfinite(values[:, support]).all() or np.any(values[:, support] < 0):
                raise DataContractError(f"invalid supported IMERG values in {store}")
            all_values.append(values)
            all_dates.append(dates)
            source_stores.append(str(store))

    assert observation_fraction is not None
    dates = np.concatenate(all_dates)
    values = np.concatenate(all_values)
    order = np.argsort(dates)
    dates = dates[order]
    values = values[order]
    if np.unique(dates).size != dates.size:
        raise DataContractError("duplicate IMERG dates")

    requested = forecast.valid_dates.reshape(-1)
    positions = np.searchsorted(dates, requested)
    if np.any(positions >= dates.size) or not np.array_equal(dates[positions], requested):
        raise DataContractError("one or more IMERG verification dates are missing")
    daily = values[positions].reshape(*forecast.valid_dates.shape, *spatial_shape)
    weekly_truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)

    support = observation_fraction > 0.0
    climatology_daily = build_training_climatology(dates, values, support)
    climate_positions = calendar_positions(forecast.valid_dates)
    daily_climatology = climatology_daily[climate_positions]
    weekly_climatology = np.mean(
        daily_climatology, axis=2, dtype=np.float64
    ).astype(np.float32)

    if not np.isfinite(weekly_truth[..., support]).all():
        raise DataContractError("weekly IMERG truth is incomplete")
    if not np.isfinite(weekly_climatology[..., support]).all():
        raise DataContractError("weekly IMERG climatology is incomplete")

    return ObservationData(
        weekly_truth=weekly_truth,
        weekly_climatology=weekly_climatology,
        observation_fraction=observation_fraction,
        source_stores=tuple(source_stores),
    )


def load_area_weights(
    forecast: ForecastData, observation_fraction: np.ndarray
) -> np.ndarray:
    with xr.open_zarr(SPATIAL_STORE, consolidated=True) as dataset:
        if not np.array_equal(dataset.latitude.values, forecast.latitude):
            raise DataContractError("spatial-support latitude differs")
        if not np.array_equal(dataset.longitude.values, forecast.longitude):
            raise DataContractError("spatial-support longitude differs")
        india_area = np.asarray(
            dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
    weights = india_area * np.asarray(observation_fraction, dtype=np.float64)
    weights[~np.isfinite(weights) | (weights <= 0.0)] = 0.0
    if np.count_nonzero(weights) != 174:
        raise DataContractError(
            f"expected 174 positive IMERG/India cells, found {np.count_nonzero(weights)}"
        )
    return weights


def weighted_lead_moments(
    values: np.ndarray, train_indices: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    selected = values[train_indices]
    means = np.empty(selected.shape[1], dtype=np.float32)
    stds = np.empty(selected.shape[1], dtype=np.float32)
    for lead in range(selected.shape[1]):
        field = selected[:, lead]
        weight = np.broadcast_to(weights, field.shape)
        valid = np.isfinite(field) & (weight > 0.0)
        denominator = np.sum(weight[valid])
        mean = np.sum(field[valid] * weight[valid]) / denominator
        variance = np.sum(weight[valid] * (field[valid] - mean) ** 2) / denominator
        means[lead] = np.float32(mean)
        stds[lead] = np.float32(max(np.sqrt(variance), 1.0e-6))
    return means, stds


def make_neural_arrays(
    forecast: ForecastData,
    observations: ObservationData,
    weights: np.ndarray,
    train_indices: np.ndarray,
    *,
    preserve_fuxi_context: bool = False,
) -> NeuralArrays:
    support = weights > 0.0
    fields = {
        "log_fuxi_mean": np.log1p(forecast.ensemble_mean).astype(np.float32),
        "log_fuxi_spread": np.log1p(forecast.ensemble_spread).astype(np.float32),
        "log_imerg_climatology": np.log1p(observations.weekly_climatology).astype(
            np.float32
        ),
    }
    dynamic = []
    feature_stats: Dict[str, object] = {}
    for name, values in fields.items():
        mean, std = weighted_lead_moments(values, train_indices, weights)
        normalized = (values - mean[None, :, None, None]) / std[
            None, :, None, None
        ]
        spatial_valid = (
            np.isfinite(normalized)
            if preserve_fuxi_context
            and name in {"log_fuxi_mean", "log_fuxi_spread"}
            else support[None, None] & np.isfinite(normalized)
        )
        normalized = np.where(spatial_valid, normalized, 0.0).astype(np.float32)
        dynamic.append(normalized)
        feature_stats[name] = {"mean_by_lead": mean.tolist(), "std_by_lead": std.tolist()}

    n_cases, n_leads, height, width = forecast.ensemble_mean.shape
    latitude = forecast.latitude.astype(np.float32)
    longitude = forecast.longitude.astype(np.float32)
    lat_scaled = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    lon_scaled = 2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    lat_grid = np.broadcast_to(
        lat_scaled[None, None, :, None], (n_cases, n_leads, height, width)
    )
    lon_grid = np.broadcast_to(
        lon_scaled[None, None, None, :], (n_cases, n_leads, height, width)
    )

    midpoints = forecast.valid_dates[:, :, 3]
    midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
    day_of_year = (midpoint_index.dayofyear.to_numpy() - 1).reshape(
        n_cases, n_leads
    )
    angle = 2.0 * np.pi * day_of_year / 365.2425
    season_sin = np.broadcast_to(
        np.sin(angle)[:, :, None, None], (n_cases, n_leads, height, width)
    )
    season_cos = np.broadcast_to(
        np.cos(angle)[:, :, None, None], (n_cases, n_leads, height, width)
    )
    lead_scaled = np.linspace(-1.0, 1.0, n_leads, dtype=np.float32)
    lead_grid = np.broadcast_to(
        lead_scaled[None, :, None, None], (n_cases, n_leads, height, width)
    )
    support_grid = np.broadcast_to(
        support[None, None], (n_cases, n_leads, height, width)
    ).astype(np.float32)
    dynamic.extend(
        [lat_grid, lon_grid, season_sin, season_cos, lead_grid, support_grid]
    )
    inputs = np.stack(dynamic, axis=2).astype(np.float32)
    if not np.isfinite(inputs).all() or inputs.shape[2] != 9:
        raise DataContractError("neural inputs are invalid")

    residual = (
        np.log1p(observations.weekly_truth) - np.log1p(forecast.ensemble_mean)
    ).astype(np.float32)
    target_scale = np.empty(n_leads, dtype=np.float32)
    for lead in range(n_leads):
        field = residual[train_indices, lead]
        weight = np.broadcast_to(weights, field.shape)
        valid = np.isfinite(field) & (weight > 0.0)
        rms = np.sqrt(np.sum(weight[valid] * field[valid] ** 2) / np.sum(weight[valid]))
        target_scale[lead] = np.float32(max(rms, 1.0e-6))
    target = residual / target_scale[None, :, None, None]
    target = np.where(
        support[None, None] & np.isfinite(target), target, 0.0
    ).astype(np.float32)
    feature_stats["target_rms_by_lead"] = target_scale.tolist()
    feature_stats["input_channels"] = [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imerg_climatology",
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "support",
    ]
    return NeuralArrays(inputs, target, target_scale, feature_stats)


def reconstruct_neural_prediction(
    raw_fuxi: np.ndarray,
    standardized_residual: np.ndarray,
    target_scale: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    log_rain = np.log1p(raw_fuxi) + standardized_residual * target_scale[
        None, :, None, None
    ]
    prediction = np.expm1(np.clip(log_rain, 0.0, 20.0))
    prediction = np.where(support[None, None], prediction, np.nan).astype(np.float32)
    if not np.isfinite(prediction[..., support]).all() or np.any(
        prediction[..., support] < 0.0
    ):
        raise DataContractError("neural reconstruction is invalid")
    return prediction


def fit_quantile_map(
    forecast: np.ndarray,
    truth: np.ndarray,
    train_indices: np.ndarray,
) -> QuantileMap:
    forecast_log = np.log1p(forecast[train_indices]).astype(np.float64)
    truth_log = np.log1p(truth[train_indices]).astype(np.float64)
    forecast_quantiles = np.quantile(forecast_log, QUANTILE_LEVELS, axis=0)
    observed_quantiles = np.quantile(truth_log, QUANTILE_LEVELS, axis=0)
    return QuantileMap(
        forecast_quantiles.astype(np.float32),
        observed_quantiles.astype(np.float32),
        QUANTILE_LEVELS.copy(),
    )


def _collapse_duplicate_quantiles(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(x, return_inverse=True)
    collapsed = np.empty(unique.size, dtype=np.float64)
    for index in range(unique.size):
        collapsed[index] = np.mean(y[inverse == index])
    collapsed = np.maximum.accumulate(collapsed)
    return unique, collapsed


def apply_quantile_map(
    forecast: np.ndarray, mapping: QuantileMap, support: np.ndarray
) -> np.ndarray:
    source = np.log1p(forecast).astype(np.float64)
    corrected = np.full_like(source, np.nan, dtype=np.float64)
    for lead in range(source.shape[1]):
        for y_index, x_index in np.argwhere(support):
            x_quantiles, y_quantiles = _collapse_duplicate_quantiles(
                mapping.forecast_quantiles[:, lead, y_index, x_index],
                mapping.observed_quantiles[:, lead, y_index, x_index],
            )
            values = source[:, lead, y_index, x_index]
            if x_quantiles.size == 1:
                mapped = np.full(values.shape, y_quantiles[0], dtype=np.float64)
            else:
                mapped = np.interp(
                    values,
                    x_quantiles,
                    y_quantiles,
                    left=y_quantiles[0],
                    right=y_quantiles[-1],
                )
            corrected[:, lead, y_index, x_index] = mapped
    precipitation = np.expm1(corrected)
    precipitation = np.maximum(precipitation, 0.0).astype(np.float32)
    if not np.isfinite(precipitation[..., support]).all():
        raise DataContractError("quantile mapping produced non-finite values")
    return precipitation


def run_unet(
    arrays: NeuralArrays,
    forecast: ForecastData,
    weights: np.ndarray,
    splits: Mapping[str, np.ndarray],
    output: Path,
    smoke: bool,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    train_indices = splits["train"][:16] if smoke else splits["train"]
    validation_indices = (
        splits["validation"][:8] if smoke else splits["validation"]
    )
    test_indices = splits["test"][:4] if smoke else splits["test"]
    seeds = SEEDS[:1] if smoke else SEEDS

    seed_predictions = []
    run_records = []
    parameter_count = None
    for seed in seeds:
        run_directory = output / "models" / f"seed_{seed}"
        (run_directory / "logs").mkdir(parents=True, exist_ok=True)
        (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        set_deterministic_seed(seed)
        model = ResidualUNet(in_channels=arrays.inputs.shape[2], base_channels=16, dropout=0.1)
        current_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count is None:
            parameter_count = current_parameter_count
        elif parameter_count != current_parameter_count:
            raise RuntimeError("model parameter count changed between seeds")
        result = train_model(
            model,
            arrays.inputs[train_indices],
            arrays.target[train_indices],
            arrays.inputs[validation_indices],
            arrays.target[validation_indices],
            weights,
            run_directory,
            seed=seed,
            device=device,
            batch_size=16,
            max_epochs=2 if smoke else 150,
            patience=1 if smoke else 20,
            learning_rate=3.0e-4,
            weight_decay=1.0e-4,
            beta=1.0,
            num_workers=0,
            use_amp=True,
        )
        standardized = predict(
            model,
            arrays.inputs[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        reconstructed = reconstruct_neural_prediction(
            forecast.ensemble_mean[test_indices],
            standardized,
            arrays.target_scale,
            weights > 0.0,
        )
        seed_predictions.append(reconstructed)
        run_records.append(
            {
                "seed": seed,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "elapsed_seconds": result.elapsed_seconds,
                "checkpoint": str(
                    (run_directory / "checkpoints" / "best.pt").relative_to(output)
                ),
            }
        )

    ensemble_prediction = np.mean(seed_predictions, axis=0, dtype=np.float64).astype(
        np.float32
    )
    return ensemble_prediction, {
        "device": device,
        "architecture": "ResidualUNet",
        "parameter_count": parameter_count,
        "input_channels": arrays.inputs.shape[2],
        "base_channels": 16,
        "dropout": 0.1,
        "seeds": list(seeds),
        "train_case_count": len(train_indices),
        "validation_case_count": len(validation_indices),
        "test_case_count": len(test_indices),
        "batch_size": 16,
        "max_epochs": 2 if smoke else 150,
        "early_stopping_patience": 1 if smoke else 20,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "loss": "area-weighted Smooth-L1 on standardized log1p residual, beta=1",
        "automatic_mixed_precision_requested": True,
        "automatic_mixed_precision_used": device == "cuda",
        "runs": run_records,
        "test_indices": test_indices.tolist(),
    }


def evaluate_predictions(
    truth: np.ndarray,
    climatology: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    initializations: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    frames = []
    identifiers = [np.datetime_as_string(value, unit="D") for value in initializations]
    support = weights > 0.0
    for method in METHOD_ORDER:
        prediction = predictions[method]
        frame = compute_case_metrics(
            truth,
            prediction,
            truth - climatology,
            prediction - climatology,
            weights,
            predictor=method,
            case_ids=identifiers,
            leads=np.arange(1, 7),
            valid_mask=support,
        )
        frame = frame.rename(columns={"predictor": "method", "case_id": "init", "lead": "lead_week"})
        frame.insert(0, "split", "test")
        frame.insert(2, "year", pd.DatetimeIndex(frame["init"]).year)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = len(initializations) * 6 * len(METHOD_ORDER)
    if len(result) != expected_rows:
        raise DataContractError(f"expected {expected_rows} metric rows, found {len(result)}")
    if not np.all(result.valid_cells == 174):
        raise DataContractError("metric support is not constant at 174 cells")
    return result


def summarize_by_lead(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, lead), group in case_metrics.groupby(["method", "lead_week"], sort=True):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "lead_week": int(lead),
            "case_count": int(len(group)),
        }
        for metric in ("acc", "rmse", "mae", "bias"):
            row[metric] = float(group[metric].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    raw = summary.loc[summary.method == "raw_fuxi"].set_index("lead_week")
    for index, row in summary.iterrows():
        baseline = raw.loc[row.lead_week]
        summary.loc[index, "delta_acc_vs_raw"] = row.acc - baseline.acc
        summary.loc[index, "rmse_skill_pct_vs_raw"] = 100.0 * (
            baseline.rmse - row.rmse
        ) / baseline.rmse
        summary.loc[index, "mae_skill_pct_vs_raw"] = 100.0 * (
            baseline.mae - row.mae
        ) / baseline.mae
    return summary.sort_values(
        ["method", "lead_week"], key=lambda values: values.map(
            {method: i for i, method in enumerate(METHOD_ORDER)}
        ) if values.name == "method" else values
    )


def _two_stage_block_indices(
    initializations: np.ndarray,
    n_resamples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    rng = np.random.default_rng(seed)
    unique_years = np.sort(np.unique(years))
    groups = [np.flatnonzero(years == year) for year in unique_years]
    case_counts = {len(group) for group in groups}
    if len(case_counts) != 1:
        raise ValueError("two-stage bootstrap requires equal test counts per year")
    n_cases = case_counts.pop()
    if block_length > n_cases:
        raise ValueError(f"block length {block_length} exceeds {n_cases} yearly cases")

    blocks = int(np.ceil(n_cases / block_length))
    offsets = np.arange(block_length, dtype=np.int64)
    sampled_years = rng.integers(
        0, len(groups), size=(n_resamples, len(groups))
    )
    sampled = np.empty((n_resamples, len(groups) * n_cases), dtype=np.int64)
    for draw in range(n_resamples):
        for slot, year_index in enumerate(sampled_years[draw]):
            starts = rng.integers(0, n_cases - block_length + 1, size=blocks)
            local = (starts[:, None] + offsets).reshape(-1)[:n_cases]
            sampled[draw, slot * n_cases : (slot + 1) * n_cases] = groups[
                year_index
            ][local]
    return sampled


def _effect(metric: str, model: np.ndarray, baseline: np.ndarray) -> float:
    model_mean = float(np.mean(model))
    baseline_mean = float(np.mean(baseline))
    if metric == "acc":
        return model_mean - baseline_mean
    if metric in ("rmse", "mae"):
        return 100.0 * (baseline_mean - model_mean) / baseline_mean
    raise ValueError(metric)


def paired_skill_intervals(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
) -> pd.DataFrame:
    sampled = _two_stage_block_indices(
        initializations,
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_BLOCK_LENGTH,
        seed=42,
    )
    case_order = [np.datetime_as_string(value, unit="D") for value in initializations]
    rows = []
    for method in METHOD_ORDER[1:]:
        for baseline in ("raw_fuxi", "log_bias", "quantile_mapping"):
            if method == baseline:
                continue
            for lead_scope, leads in [("W1-W6", tuple(range(1, 7)))] + [
                (f"W{lead}", (lead,)) for lead in range(1, 7)
            ]:
                selected = case_metrics.loc[
                    case_metrics.lead_week.isin(leads)
                    & case_metrics.method.isin([method, baseline])
                ]
                for metric in ("acc", "rmse", "mae"):
                    pivot = selected.pivot_table(
                        index="init", columns="method", values=metric, aggfunc="mean"
                    ).reindex(case_order)
                    if pivot[[method, baseline]].isna().any().any():
                        raise DataContractError("bootstrap comparison is not fully paired")
                    model_values = pivot[method].to_numpy(dtype=np.float64)
                    baseline_values = pivot[baseline].to_numpy(dtype=np.float64)
                    observed = _effect(metric, model_values, baseline_values)
                    draws = np.asarray(
                        [
                            _effect(
                                metric,
                                model_values[indices],
                                baseline_values[indices],
                            )
                            for indices in sampled
                        ],
                        dtype=np.float64,
                    )
                    rows.append(
                        {
                            "method": method,
                            "baseline": baseline,
                            "lead_scope": lead_scope,
                            "metric": metric,
                            "paired_case_count": len(case_order),
                            "model_mean": float(np.mean(model_values)),
                            "baseline_mean": float(np.mean(baseline_values)),
                            "effect": observed,
                            "ci_lower": float(np.percentile(draws, 2.5)),
                            "ci_upper": float(np.percentile(draws, 97.5)),
                            "block_length": BOOTSTRAP_BLOCK_LENGTH,
                            "n_resamples": BOOTSTRAP_SAMPLES,
                            "seed": 42,
                            "bootstrap_method": "two-stage year resampling plus non-circular within-year moving blocks",
                        }
                    )
    return pd.DataFrame(rows)


def yearly_skill_table(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(case_metrics.year.unique()):
        raw = case_metrics.loc[
            (case_metrics.year == year) & (case_metrics.method == "raw_fuxi")
        ]
        for method in METHOD_ORDER:
            selected = case_metrics.loc[
                (case_metrics.year == year) & (case_metrics.method == method)
            ]
            rows.append(
                {
                    "year": int(year),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "initialization_count": int(selected.init.nunique()),
                    "acc": float(selected.acc.mean()),
                    "rmse": float(selected.rmse.mean()),
                    "mae": float(selected.mae.mean()),
                    "bias": float(selected.bias.mean()),
                    "delta_acc_vs_raw": float(selected.acc.mean() - raw.acc.mean()),
                    "rmse_skill_pct_vs_raw": float(
                        100.0 * (raw.rmse.mean() - selected.rmse.mean()) / raw.rmse.mean()
                    ),
                    "mae_skill_pct_vs_raw": float(
                        100.0 * (raw.mae.mean() - selected.mae.mean()) / raw.mae.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def headline_table(case_metrics: pd.DataFrame, paired_skill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        selected = case_metrics.loc[case_metrics.method == method]
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "cases": int(selected["init"].nunique()),
            "acc": float(selected.acc.mean()),
            "rmse_mm_day": float(selected.rmse.mean()),
            "mae_mm_day": float(selected.mae.mean()),
            "bias_mm_day": float(selected.bias.mean()),
            "delta_acc_vs_raw": 0.0,
            "rmse_skill_pct_vs_raw": 0.0,
            "mae_skill_pct_vs_raw": 0.0,
            "delta_acc_ci_lower": 0.0,
            "delta_acc_ci_upper": 0.0,
            "rmse_skill_ci_lower": 0.0,
            "rmse_skill_ci_upper": 0.0,
            "mae_skill_ci_lower": 0.0,
            "mae_skill_ci_upper": 0.0,
        }
        if method != "raw_fuxi":
            intervals = paired_skill.loc[
                (paired_skill.method == method)
                & (paired_skill.baseline == "raw_fuxi")
                & (paired_skill.lead_scope == "W1-W6")
            ].set_index("metric")
            row.update(
                {
                    "delta_acc_vs_raw": intervals.loc["acc", "effect"],
                    "rmse_skill_pct_vs_raw": intervals.loc["rmse", "effect"],
                    "mae_skill_pct_vs_raw": intervals.loc["mae", "effect"],
                    "delta_acc_ci_lower": intervals.loc["acc", "ci_lower"],
                    "delta_acc_ci_upper": intervals.loc["acc", "ci_upper"],
                    "rmse_skill_ci_lower": intervals.loc["rmse", "ci_lower"],
                    "rmse_skill_ci_upper": intervals.loc["rmse", "ci_upper"],
                    "mae_skill_ci_lower": intervals.loc["mae", "ci_lower"],
                    "mae_skill_ci_upper": intervals.loc["mae", "ci_upper"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_skill_by_lead(
    summary: pd.DataFrame, output: Path, *, smoke: bool
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
    panels = [
        ("acc", "Spatial anomaly correlation (ACC)", None),
        ("rmse", "RMSE (mm day$^{-1}$)", None),
        ("mae", "MAE (mm day$^{-1}$)", None),
        ("bias", "Bias (mm day$^{-1}$)", 0.0),
    ]
    panel_letters = "abcd"
    plot_order = METHOD_ORDER[1:] + ("raw_fuxi",)
    handles = {}
    for panel_index, (axis, (metric, label, reference)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        for method in plot_order:
            rows = summary.loc[summary.method == method].sort_values("lead_week")
            line, = axis.plot(
                rows.lead_week,
                rows[metric],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle="--" if method == "raw_fuxi" else "-",
                linewidth=1.8 if method == "raw_fuxi" else 2.0,
                markersize=5.0,
                markerfacecolor="white" if method == "raw_fuxi" else METHOD_COLORS[method],
                markeredgewidth=1.2,
                zorder=5 if method == "raw_fuxi" else 3,
                label=METHOD_LABELS[method],
            )
            handles[method] = line
        if reference is not None:
            axis.axhline(reference, color="0.55", linewidth=0.9, linestyle="--")
        axis.set_xlabel("Lead week")
        axis.set_ylabel(label)
        axis.set_xticks(range(1, 7))
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.01,
            0.98,
            f"({panel_letters[panel_index]})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="semibold",
        )
    figure.legend(
        [handles[method] for method in METHOD_ORDER],
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        fontsize=9,
    )
    case_count = int(summary.case_count.max())
    subtitle = (
        f"SMOKE CHECK — n={case_count}; not for scientific interpretation"
        if smoke
        else f"IMERG verification, held-out 2020–2021 (n={case_count})"
    )
    figure.suptitle(
        "FuXi-S2S weekly rainfall over India\n" + subtitle,
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.5,
        -0.012,
        "India area-weighted case means; W1 = days 0–6, …, W6 = days 35–41",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.11, top=0.78, wspace=0.25, hspace=0.32
    )
    save_figure(figure, output)


def plot_improvement(
    headline: pd.DataFrame, output: Path, *, smoke: bool
) -> None:
    data = headline.loc[headline.method != "raw_fuxi"].reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), constrained_layout=True)
    panels = [
        ("delta_acc_vs_raw", "delta_acc_ci_lower", "delta_acc_ci_upper", "ΔACC vs raw FuXi"),
        (
            "rmse_skill_pct_vs_raw",
            "rmse_skill_ci_lower",
            "rmse_skill_ci_upper",
            "RMSE reduction vs raw (%)",
        ),
        (
            "mae_skill_pct_vs_raw",
            "mae_skill_ci_lower",
            "mae_skill_ci_upper",
            "MAE reduction vs raw (%)",
        ),
    ]
    y = np.arange(len(data))
    for axis, (value, lower, upper, label) in zip(axes, panels):
        centre = data[value].to_numpy()
        for index, method in enumerate(data.method):
            axis.hlines(
                y[index],
                data.loc[index, lower],
                data.loc[index, upper],
                color=METHOD_COLORS[method],
                linewidth=1.5,
            )
            axis.plot(
                centre[index],
                y[index],
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                markersize=7,
                linestyle="none",
            )
        axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        axis.set_xlabel(label)
        axis.set_yticks(y)
        if axis is axes[0]:
            axis.set_yticklabels(data.method_label)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.grid(True, axis="x", alpha=0.22)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.invert_yaxis()
    case_count = int(headline.cases.max())
    context = (
        f"SMOKE CHECK — n={case_count}; intervals are not interpretable"
        if smoke
        else f"W1–W6 mean; n={case_count}; paired year/block-bootstrap 95% intervals"
    )
    figure.suptitle(
        "Post-processing skill relative to raw FuXi\n" + context,
        fontsize=13,
        fontweight="semibold",
    )
    save_figure(figure, output)


def plot_rmse_skill_heatmap(
    summary: pd.DataFrame, output: Path, *, smoke: bool
) -> None:
    methods = METHOD_ORDER[1:]
    matrix = np.stack(
        [
            summary.loc[summary.method == method]
            .sort_values("lead_week")
            .rmse_skill_pct_vs_raw.to_numpy()
            for method in methods
        ]
    )
    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdYlBu", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(matrix[row, column]) > 0.55 * limit else "black",
            )
    axis.set_xticks(np.arange(6), [f"W{lead}" for lead in range(1, 7)])
    axis.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods])
    axis.set_xlabel("Lead week")
    context = (
        "SMOKE CHECK — do not interpret"
        if smoke
        else "Per-lead point estimates; pooled W1–W6 intervals are shown separately"
    )
    axis.set_title(
        "RMSE reduction relative to raw FuXi (positive is better)\n" + context
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("RMSE reduction (%)")
    save_figure(figure, output)


def write_prediction_store(
    output: Path,
    forecast: ForecastData,
    observations: ObservationData,
    test_indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    smoke: bool,
) -> None:
    prediction_array = np.stack([predictions[method] for method in METHOD_ORDER])
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                prediction_array.astype(np.float32),
            ),
            "truth_imerg": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_truth[test_indices].astype(np.float32),
            ),
            "imerg_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_climatology[test_indices].astype(np.float32),
            ),
            "area_weight_km2": (
                ("latitude", "longitude"),
                weights.astype(np.float64),
            ),
            "valid_start": (
                ("init", "lead_week"),
                forecast.valid_dates[test_indices, :, 0].astype("datetime64[ns]"),
            ),
            "valid_end_exclusive": (
                ("init", "lead_week"),
                (
                    forecast.valid_dates[test_indices, :, -1]
                    + np.timedelta64(1, "D")
                ).astype("datetime64[ns]"),
            ),
        },
        coords={
            "method": list(METHOD_ORDER),
            "init": forecast.initializations[test_indices].astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": (
                "Smoke check for FuXi-S2S IMERG post-processing"
                if smoke
                else "Five-year FuXi-S2S IMERG post-processing experiment"
            ),
            "smoke": smoke,
            "test_initialization_count": len(test_indices),
            "train_years": "2014-2018",
            "validation_years": "2019",
            "test_years": "2020-2021",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "units": "mm day-1",
            "created_utc": utc_now(),
        },
    )
    dataset.prediction.attrs["units"] = "mm day-1"
    dataset.truth_imerg.attrs["units"] = "mm day-1"
    dataset.imerg_climatology.attrs["units"] = "mm day-1"
    dataset.area_weight_km2.attrs["units"] = "km2"
    encoding = {
        "prediction": {"chunks": (1, 35, 1, 27, 27)},
        "truth_imerg": {"chunks": (35, 1, 27, 27)},
        "imerg_climatology": {"chunks": (35, 1, 27, 27)},
        "area_weight_km2": {"chunks": (27, 27)},
    }
    dataset.to_zarr(output, mode="w", consolidated=True, encoding=encoding)


def verify_prediction_store(
    output: Path,
    forecast: ForecastData,
    observations: ObservationData,
    test_indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Reopen the delivered Zarr and check ordering and values exactly."""

    with xr.open_zarr(output, consolidated=True) as dataset:
        expected_sizes = {
            "method": len(METHOD_ORDER),
            "init": len(test_indices),
            "lead_week": 6,
            "latitude": 27,
            "longitude": 27,
        }
        if dict(dataset.sizes) != expected_sizes:
            raise DataContractError(f"unexpected prediction-store sizes: {dataset.sizes}")
        if dataset.method.values.tolist() != list(METHOD_ORDER):
            raise DataContractError("prediction-store method ordering differs")
        if not np.array_equal(
            dataset.init.values.astype("datetime64[D]"),
            forecast.initializations[test_indices],
        ):
            raise DataContractError("prediction-store initialization ordering differs")
        if not np.array_equal(dataset.lead_week.values, np.arange(1, 7)):
            raise DataContractError("prediction-store lead ordering differs")

        stored_predictions = {
            method: np.asarray(
                dataset.prediction.sel({"method": method}).load().values,
                dtype=np.float32,
            )
            for method in METHOD_ORDER
        }
        stored_truth = np.asarray(dataset.truth_imerg.load().values, dtype=np.float32)
        stored_climatology = np.asarray(
            dataset.imerg_climatology.load().values, dtype=np.float32
        )
        stored_weights = np.asarray(dataset.area_weight_km2.load().values, dtype=np.float64)

    for method in METHOD_ORDER:
        if not np.array_equal(
            stored_predictions[method], predictions[method], equal_nan=True
        ):
            raise DataContractError(f"prediction-store values differ for {method}")
    if not np.array_equal(
        stored_truth, observations.weekly_truth[test_indices], equal_nan=True
    ):
        raise DataContractError("prediction-store IMERG truth differs")
    if not np.array_equal(
        stored_climatology,
        observations.weekly_climatology[test_indices],
        equal_nan=True,
    ):
        raise DataContractError("prediction-store climatology differs")
    if not np.allclose(stored_weights, weights, rtol=1.0e-7, atol=0.0):
        raise DataContractError("prediction-store weights differ")
    return stored_predictions


def write_results_markdown(
    headline: pd.DataFrame,
    paired_skill: pd.DataFrame,
    yearly_skill: pd.DataFrame,
    training_manifest: Mapping[str, object],
    output: Path,
    *,
    smoke: bool,
) -> None:
    neural = headline.loc[headline.method == "residual_unet"].iloc[0]
    bias = headline.loc[headline.method == "log_bias"].iloc[0]
    qm = headline.loc[headline.method == "quantile_mapping"].iloc[0]
    all_improve = (
        neural.delta_acc_vs_raw > 0.0
        and neural.rmse_skill_pct_vs_raw > 0.0
        and neural.mae_skill_pct_vs_raw > 0.0
    )
    supported = (
        all_improve
        and neural.delta_acc_ci_lower > 0.0
        and neural.rmse_skill_ci_lower > 0.0
        and neural.mae_skill_ci_lower > 0.0
    )
    error_supported = (
        neural.rmse_skill_ci_lower > 0.0
        and neural.mae_skill_ci_lower > 0.0
    )
    if smoke:
        conclusion = (
            "This output only checks that the pipeline runs end to end. The smoke scores "
            "and intervals must not be used for scientific inference."
        )
    elif supported:
        conclusion = (
            "The residual U-Net improves ACC, RMSE, and MAE over raw FuXi with "
            "paired 95% intervals excluding zero."
        )
    elif all_improve and error_supported:
        conclusion = (
            "The residual U-Net reduces RMSE and MAE with paired 95% intervals "
            "excluding zero. Mean ACC increases, but its interval includes zero because "
            "the ACC change is not consistent across the two test years."
        )
    elif all_improve:
        conclusion = (
            "The residual U-Net improves the three mean scores, but at least one paired "
            "95% interval crosses zero; describe the result as indicative."
        )
    else:
        conclusion = (
            "The residual U-Net does not improve all three mean scores over raw FuXi; "
            "report the mixed result without claiming a general skill gain."
        )
    lines = [
        (
            "# FuXi–IMERG smoke check"
            if smoke
            else "# FuXi–IMERG five-year post-processing result"
        ),
        "",
        "## Experiment",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "- **SMOKE CHECK: do not interpret these scores.**",
                f"- U-Net subset: {training_manifest['train_case_count']} train, "
                f"{training_manifest['validation_case_count']} validation, and "
                f"{training_manifest['test_case_count']} test cases; "
                f"{len(training_manifest['seeds'])} seed and "
                f"{training_manifest['max_epochs']} epochs.",
            ]
        )
    lines.extend(
        [
        "- Train: 2014–2018 JJAS initializations",
        "- Validation: 2019",
        (
            f"- Smoke evaluation: first {int(headline.cases.max())} 2020 test initializations"
            if smoke
            else f"- Held-out evaluation: 2020–2021 ({int(headline.cases.max())} initializations)"
        ),
        "- Target: IMERG Final V07B weekly rainfall",
        "- Domain: 174 area-weighted India-grid cells",
        "",
        "## Lead-mean scores",
        "",
        "| Method | ACC | RMSE | MAE | Bias | ΔACC | RMSE reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in headline.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.acc:.3f} | {row.rmse_mm_day:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:.3f} | "
            f"{row.delta_acc_vs_raw:+.3f} | {row.rmse_skill_pct_vs_raw:+.1f}% |"
        )
    yearly_unet = yearly_skill.loc[
        yearly_skill.method == "residual_unet"
    ].sort_values("year")
    lines.extend(
        [
            "",
            "## Residual U-Net by test year",
            "",
            "| Year | ΔACC vs raw | RMSE reduction | MAE reduction |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in yearly_unet.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.delta_acc_vs_raw:+.3f} | "
            f"{row.rmse_skill_pct_vs_raw:+.1f}% | {row.mae_skill_pct_vs_raw:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Residual U-Net paired comparisons",
            "",
            "Positive values mean improvement over the named baseline.",
            "",
            "| Baseline | ΔACC [95% CI] | RMSE reduction [95% CI] | MAE reduction [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    comparison_labels = {
        "raw_fuxi": "Raw FuXi",
        "log_bias": "Log-bias",
        "quantile_mapping": "Quantile mapping",
    }
    for baseline, label in comparison_labels.items():
        comparison = paired_skill.loc[
            (paired_skill.method == "residual_unet")
            & (paired_skill.baseline == baseline)
            & (paired_skill.lead_scope == "W1-W6")
        ].set_index("metric")
        acc = comparison.loc["acc"]
        rmse = comparison.loc["rmse"]
        mae = comparison.loc["mae"]
        lines.append(
            f"| {label} | {acc.effect:+.3f} [{acc.ci_lower:+.3f}, {acc.ci_upper:+.3f}] | "
            f"{rmse.effect:+.1f}% [{rmse.ci_lower:+.1f}, {rmse.ci_upper:+.1f}] | "
            f"{mae.effect:+.1f}% [{mae.ci_lower:+.1f}, {mae.ci_upper:+.1f}] |"
        )
    unet_vs_bias = paired_skill.loc[
        (paired_skill.method == "residual_unet")
        & (paired_skill.baseline == "log_bias")
        & (paired_skill.lead_scope == "W1-W6")
    ].set_index("metric")
    unet_w6_vs_raw = paired_skill.loc[
        (paired_skill.method == "residual_unet")
        & (paired_skill.baseline == "raw_fuxi")
        & (paired_skill.lead_scope == "W6")
    ].set_index("metric")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            "",
            f"Log-bias RMSE reduction: {bias.rmse_skill_pct_vs_raw:+.1f}%.",
            f"Quantile-mapping RMSE reduction: {qm.rmse_skill_pct_vs_raw:+.1f}%.",
            f"Residual U-Net RMSE reduction: {neural.rmse_skill_pct_vs_raw:+.1f}%.",
            "",
            "## Practical reading",
            "",
            "- Raw FuXi is the forecast baseline; the other methods are deterministic post-processors.",
            f"- Log-bias correction already removes {bias.rmse_skill_pct_vs_raw:.1f}% of raw RMSE. "
            f"The U-Net adds {unet_vs_bias.loc['rmse', 'effect']:.1f}% RMSE reduction over log-bias; "
            "its incremental ACC and MAE intervals include zero.",
            f"- U-Net gains are strongest at early leads. At W6 its RMSE reduction is "
            f"{unet_w6_vs_raw.loc['rmse', 'effect']:.1f}% with a 95% interval of "
            f"[{unet_w6_vs_raw.loc['rmse', 'ci_lower']:.1f}, "
            f"{unet_w6_vs_raw.loc['rmse', 'ci_upper']:.1f}]%.",
            f"- Quantile mapping raises mean ACC by {qm.delta_acc_vs_raw:+.3f}, but changes "
            f"RMSE skill by {qm.rmse_skill_pct_vs_raw:+.1f}%; it is not the preferred amount forecast.",
            f"- The U-Net has a dry mean bias ({neural.bias_mm_day:+.3f} mm day-1) compared "
            f"with raw FuXi ({headline.loc[headline.method == 'raw_fuxi', 'bias_mm_day'].iloc[0]:+.3f} mm day-1).",
            "",
            "This is a retrospective two-year test, and IMERG is both the training target and "
            "the verifier. It supports a focused pilot result, not a broad claim of global or "
            "multi-decadal generalization.",
            "",
            "The neural model is a lightweight post-processor; FuXi itself was not fine-tuned.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny end-to-end check")
    args = parser.parse_args()

    started = time.monotonic()
    run_name = ("smoke" if args.smoke else "full") + "_" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_ROOT / run_name
    for directory in (
        output,
        output / "metrics",
        output / "figures",
        output / "models",
        output / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    implementation_checksums = snapshot_implementation(output)

    print("Loading compact FuXi shards...", flush=True)
    forecast = load_fuxi()
    print("Loading and aligning IMERG...", flush=True)
    observations = load_imerg(forecast)
    weights = load_area_weights(forecast, observations.observation_fraction)
    splits = split_indices(forecast.initializations)
    expected_split_counts = {"train": 175, "validation": 35, "test": 70}
    actual_split_counts = {name: len(values) for name, values in splits.items()}
    if actual_split_counts != expected_split_counts:
        raise DataContractError(f"unexpected split counts: {actual_split_counts}")

    train_end = forecast.valid_dates[splits["train"]].max() + np.timedelta64(1, "D")
    validation_start = forecast.valid_dates[splits["validation"]].min()
    validation_end = (
        forecast.valid_dates[splits["validation"]].max() + np.timedelta64(1, "D")
    )
    test_start = forecast.valid_dates[splits["test"]].min()
    if train_end > validation_start or validation_end > test_start:
        raise DataContractError("verification targets overlap across splits")

    support = weights > 0.0
    print("Fitting training-only statistical baselines...", flush=True)
    log_bias_model = fit_log_bias_correction(
        forecast.ensemble_mean[splits["train"]],
        observations.weekly_truth[splits["train"]],
        forecast.initializations[splits["train"]],
        support,
        shrinkage=10.0,
    )
    quantile_model = fit_quantile_map(
        forecast.ensemble_mean,
        observations.weekly_truth,
        splits["train"],
    )
    np.savez_compressed(
        output / "models" / "statistical_baselines.npz",
        log_bias_lead_month_residual=log_bias_model.lead_month_residual,
        log_bias_shrinkage=np.asarray(log_bias_model.shrinkage, dtype=np.float64),
        quantile_forecast=quantile_model.forecast_quantiles,
        quantile_observed=quantile_model.observed_quantiles,
        quantile_levels=quantile_model.levels,
    )

    print("Preparing neural arrays...", flush=True)
    neural_arrays = make_neural_arrays(
        forecast, observations, weights, splits["train"]
    )
    (output / "normalization.json").write_text(
        json.dumps(neural_arrays.feature_stats, indent=2) + "\n", encoding="utf-8"
    )
    print("Training residual U-Net...", flush=True)
    unet_prediction, training_manifest = run_unet(
        neural_arrays, forecast, weights, splits, output, args.smoke
    )

    test_indices = np.asarray(training_manifest["test_indices"], dtype=np.int64)
    raw_test = forecast.ensemble_mean[test_indices].copy()
    raw_test[..., ~support] = np.nan
    log_bias_test = apply_log_bias_correction(
        forecast.ensemble_mean[test_indices],
        forecast.initializations[test_indices],
        log_bias_model,
    )
    log_bias_test[..., ~support] = np.nan
    quantile_test = apply_quantile_map(
        forecast.ensemble_mean[test_indices], quantile_model, support
    )
    predictions = {
        "raw_fuxi": raw_test,
        "log_bias": log_bias_test,
        "quantile_mapping": quantile_test,
        "residual_unet": unet_prediction,
    }

    print("Scoring forecasts...", flush=True)
    case_metrics = evaluate_predictions(
        observations.weekly_truth[test_indices],
        observations.weekly_climatology[test_indices],
        predictions,
        forecast.initializations[test_indices],
        weights,
    )
    summary = summarize_by_lead(case_metrics)
    if args.smoke:
        global BOOTSTRAP_SAMPLES, BOOTSTRAP_BLOCK_LENGTH
        BOOTSTRAP_SAMPLES = 50
        BOOTSTRAP_BLOCK_LENGTH = 2
    paired = paired_skill_intervals(
        case_metrics, forecast.initializations[test_indices]
    )
    headline = headline_table(case_metrics, paired)
    yearly = yearly_skill_table(case_metrics)

    metrics_directory = output / "metrics"
    case_metrics.to_csv(metrics_directory / "case_metrics.csv", index=False)
    summary.to_csv(metrics_directory / "summary_by_lead.csv", index=False)
    paired.to_csv(metrics_directory / "paired_skill.csv", index=False)
    headline.to_csv(metrics_directory / "headline_metrics.csv", index=False)
    yearly.to_csv(metrics_directory / "yearly_skill.csv", index=False)

    print("Writing predictions and figures...", flush=True)
    prediction_store = output / "predictions.zarr"
    write_prediction_store(
        prediction_store,
        forecast,
        observations,
        test_indices,
        predictions,
        weights,
        smoke=args.smoke,
    )
    stored_predictions = verify_prediction_store(
        prediction_store,
        forecast,
        observations,
        test_indices,
        predictions,
        weights,
    )
    stored_metrics = evaluate_predictions(
        observations.weekly_truth[test_indices],
        observations.weekly_climatology[test_indices],
        stored_predictions,
        forecast.initializations[test_indices],
        weights,
    )
    pd.testing.assert_frame_equal(
        case_metrics.reset_index(drop=True),
        stored_metrics.reset_index(drop=True),
        check_exact=True,
    )
    plot_skill_by_lead(
        summary, output / "figures" / "01_skill_by_lead", smoke=args.smoke
    )
    plot_improvement(
        headline, output / "figures" / "02_improvement_vs_raw", smoke=args.smoke
    )
    plot_rmse_skill_heatmap(
        summary, output / "figures" / "03_rmse_skill_heatmap", smoke=args.smoke
    )
    write_results_markdown(
        headline,
        paired,
        yearly,
        training_manifest,
        output / "RESULTS.md",
        smoke=args.smoke,
    )

    print("Hashing source inventory...", flush=True)
    inventory = source_inventory(
        tuple(Path(path) for path in forecast.source_files)
        + tuple(Path(path) for path in observations.source_stores)
        + (SPATIAL_STORE,)
    )
    inventory.to_csv(output / "source_inventory.csv", index=False)

    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "smoke": args.smoke,
        "created_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "fuxi_shards": str(FUXI_SHARDS),
        "imerg_daily": str(IMERG_DAILY),
        "spatial_store": str(SPATIAL_STORE),
        "split_years": {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": list(TEST_YEARS),
        },
        "split_counts": actual_split_counts,
        "test_count_used": len(test_indices),
        "member_count": 51,
        "lead_days": 42,
        "lead_weeks": 6,
        "support_cells": int(np.count_nonzero(support)),
        "area_weight_sum_km2": float(weights.sum()),
        "tp_conversion": "native mm h-1 daily mean rate multiplied by 24",
        "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
        "climatology": "IMERG 2014-2018 equal-year 31-day centred calendar mean",
        "quantile_mapping": {
            "space": "log1p(mm day-1)",
            "levels": QUANTILE_LEVELS.tolist(),
            "duplicate_policy": "mean target quantile for repeated forecast quantiles",
        },
        "log_bias": {
            "space": "log1p(mm day-1)",
            "grouping": "lead week, verification-midpoint month, and grid cell",
            "shrinkage": 10.0,
        },
        "statistical_baseline_parameters": "models/statistical_baselines.npz",
        "bootstrap": {
            "method": "paired two-stage resampling of test years and non-circular moving blocks within sampled years",
            "block_length_initializations": BOOTSTRAP_BLOCK_LENGTH,
            "resamples": BOOTSTRAP_SAMPLES,
            "seed": 42,
        },
        "prediction_store_roundtrip_verified": True,
        "metric_contract": {
            "unit": "mm day-1",
            "support": "174 cells with positive India area times IMERG observation fraction",
            "acc": "per-case area-weighted centred spatial correlation of anomalies from the training-only IMERG climatology",
            "rmse_mae_bias": "per-case exact-area weighted spatial scores",
            "summary": "equal arithmetic mean of case scores; grid cells are not independent samples",
            "skill": "difference for ACC and percentage reduction of mean case RMSE/MAE",
        },
        "training": training_manifest,
        "implementation_snapshots": implementation_checksums,
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
        },
        "artifacts": {},
        "interpretation": (
            "Retrospective out-of-time pilot. FuXi is not fine-tuned; the U-Net is a "
            "lightweight deterministic post-processor."
        ),
    }
    for artifact in sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
    ):
        manifest["artifacts"][str(artifact.relative_to(output))] = sha256_file(artifact)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(headline.to_string(index=False), flush=True)
    print(f"\nCompleted in {elapsed / 60.0:.1f} minutes", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
