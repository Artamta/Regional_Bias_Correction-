#!/usr/bin/env python3
"""All-season probabilistic calibration of the FuXi-S2S rainfall ensemble.

This experiment keeps the 51 FuXi members as an ensemble.  It trains on
2002--2017, selects checkpoints on 2018--2019, and reports a retrospective
2020--2021 development test.  Starts whose 42-day outcome window crosses a
train/validation boundary are purged.  The untouched 2025 control is never
opened by this program.

The fixed model is a permutation-invariant location-and-spread calibrator
trained with area-weighted ensemble CRPS.  A newly fitted training-only moment
calibration is evaluated against raw FuXi using the same cases, members,
leads, grid, climatology, and support.  This corrected v2 experiment never
loads the invalid zero-offset v1 checkpoints or fitted products.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
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
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from project_paths import PROJECT_ROOT as HERE

import fuxi_allseason_member_cache as canonical_member_cache
from fuxi_ensemble_calibration_core import (
    EnsembleLocationSpreadCalibrator,
    weighted_ensemble_crps,
)


EXPERIMENT = "fuxi_allseason_ensemble_calibration_v2_aligned"
CONTRACT_REVISION = "imd_end_labelled_init_plus_1_through_42_v2"
TARGET_DAY_OFFSETS = tuple(range(1, 43))
NATIVE_SOURCE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
BENCHMARK_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1"
)
IMD_DAILY = (
    BENCHMARK_ROOT
    / "observations/ground_truth_v1/daily/imd/tp/india_1p5_27x27_v1"
)
SPATIAL_STORE = BENCHMARK_ROOT / "spatial/spatial_support.zarr"
DEFAULT_CACHE = (
    HERE / "cache" / "fuxi_tp_members_weekly_2002_2021_allseason_v1.npy"
)
DEFAULT_OUTPUT_ROOT = HERE / "resultsv3" / "fuxi_allseason_ensemble_calibration"

TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
TEST_YEARS = (2020, 2021)
OBSERVATION_YEARS = tuple(range(2002, 2023))
SEALED_YEARS = (2025,)
SEEDS = (42, 43, 44)
PRIMARY_CONFIGURATION = "location_spread"
CONFIGURATIONS = (PRIMARY_CONFIGURATION,)
THRESHOLDS_MM_DAY = (1.0, 5.0, 10.0, 20.0)
COVERAGES = (0.50, 0.80, 0.90)
EXPECTED_SHAPE = (2080, 51, 6, 27, 27)
EXPECTED_COUNTS = {
    "train": 1652,
    "validation": 196,
    "test": 208,
    "embargo": 24,
}
SUPPORT_CELLS = 171

METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "moment_calibration": "Train-only moment calibration",
    "summary_only": "Summary-only neural calibration",
    "location_only": "Set neural location-only",
    "location_spread": "Set neural location + spread",
}
PLOT_METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "moment_calibration": "Moment",
    "summary_only": "Summary neural",
    "location_only": "Location-only",
    "location_spread": "Location + spread",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "moment_calibration": "#E69F00",
    "summary_only": "#56B4E9",
    "location_only": "#0072B2",
    "location_spread": "#009E73",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "moment_calibration": "s",
    "summary_only": "^",
    "location_only": "D",
    "location_spread": "P",
}

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
    }
)


class DataContractError(ValueError):
    """Raised when data violate the frozen all-season experiment contract."""


@dataclass(frozen=True)
class MemberCache:
    members: np.ndarray
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    member_labels: np.ndarray
    cache_root: Path
    members_path: Path
    metadata_path: Path
    manifest_path: Path | None


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    embargo: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "embargo": self.embargo,
        }


@dataclass(frozen=True)
class ObservationBundle:
    weekly_truth: np.ndarray
    weekly_climatology: np.ndarray
    observation_fraction: np.ndarray
    weights: np.ndarray
    source_stores: tuple[str, ...]


@dataclass(frozen=True)
class ContextBundle:
    normalized_climatology: np.ndarray
    climatology_mean_by_lead: np.ndarray
    climatology_std_by_lead: np.ndarray
    latitude_scaled: np.ndarray
    longitude_scaled: np.ndarray
    season_sin: np.ndarray
    season_cos: np.ndarray
    lead_scaled: np.ndarray
    support: np.ndarray


@dataclass(frozen=True)
class MomentFit:
    delta_log_location: np.ndarray
    spread_factor: np.ndarray
    shrinkage: float


@dataclass(frozen=True)
class TrainingRun:
    configuration: str
    seed: int
    best_epoch: int
    stopped_epoch: int
    stopping_reason: str
    best_validation_crps: float
    elapsed_seconds: float
    parameter_count: int
    checkpoint: str


def select_deployable_run(
    runs: Sequence[TrainingRun],
    *,
    configuration: str = PRIMARY_CONFIGURATION,
) -> TrainingRun:
    """Select one forecast using validation CRPS only; exact ties use seed."""

    candidates = [run for run in runs if run.configuration == configuration]
    if not candidates:
        raise DataContractError(
            f"no training runs are available for deployable {configuration!r}"
        )
    seeds = [run.seed for run in candidates]
    if len(seeds) != len(set(seeds)):
        raise DataContractError("deployable candidates contain duplicate seeds")
    if any(not math.isfinite(run.best_validation_crps) for run in candidates):
        raise DataContractError("deployable candidate validation CRPS is non-finite")
    return min(candidates, key=lambda run: (run.best_validation_crps, run.seed))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def derive_valid_dates(initializations: np.ndarray) -> np.ndarray:
    """Return IMD end-date labels aligned to FuXi lead days 1--42.

    The native FuXi cache stores lead_day=1,...,42.  IMD daily records are
    labelled by the day on which rainfall ends, so W1 must verify on
    initialization+1 through initialization+7 (not initialization+0..+6).
    """

    starts = np.asarray(initializations, dtype="datetime64[D]")
    offsets = np.asarray(TARGET_DAY_OFFSETS, dtype="timedelta64[D]").reshape(
        1, 6, 7
    )
    return starts[:, None, None] + offsets


def calendar_positions(dates: np.ndarray) -> np.ndarray:
    """Map month/day to a stable 366-position leap-year calendar."""

    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    values = np.asarray(dates, dtype="datetime64[D]")
    result = np.asarray(
        [lookup[pd.Timestamp(value).strftime("%m-%d")] for value in values.reshape(-1)],
        dtype=np.int16,
    )
    return result.reshape(values.shape)


def make_split_indices(initializations: np.ndarray) -> SplitIndices:
    """Apply initialization-year splits and purge crossing outcome windows."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    if starts.ndim != 1 or starts.size == 0:
        raise DataContractError("initializations must be a non-empty 1-D array")
    if np.unique(starts).size != starts.size or np.any(np.diff(starts) <= np.timedelta64(0, "D")):
        raise DataContractError("initializations must be unique and strictly increasing")
    years = pd.DatetimeIndex(starts).year.to_numpy()
    ends = starts + np.timedelta64(42, "D")
    train_candidate = np.isin(years, TRAIN_YEARS)
    validation_candidate = np.isin(years, VALIDATION_YEARS)
    test_candidate = np.isin(years, TEST_YEARS)
    train = np.flatnonzero(train_candidate & (ends < np.datetime64("2018-01-01")))
    validation = np.flatnonzero(
        validation_candidate & (ends < np.datetime64("2020-01-01"))
    )
    test = np.flatnonzero(test_candidate)
    retained = np.zeros(starts.size, dtype=bool)
    retained[np.concatenate((train, validation, test))] = True
    in_scope = train_candidate | validation_candidate | test_candidate
    embargo = np.flatnonzero(in_scope & ~retained)
    result = SplitIndices(
        train.astype(np.int64),
        validation.astype(np.int64),
        test.astype(np.int64),
        embargo.astype(np.int64),
    )
    if starts.size == EXPECTED_SHAPE[0]:
        counts = {name: len(indices) for name, indices in result.as_dict().items()}
        if counts != EXPECTED_COUNTS:
            raise DataContractError(
                f"unexpected all-season split counts {counts}; expected {EXPECTED_COUNTS}"
            )
    return result


def _first_existing(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def load_member_cache(path: Path, *, allow_partial: bool = False) -> MemberCache:
    """Load the finalized cache without copying the member array into RAM."""

    requested = Path(path).resolve()
    root = requested.parent if requested.suffix == ".npy" else requested
    if requested.is_file() and requested.suffix == ".npy":
        members_path = requested
    else:
        members_path = _first_existing(
            root,
            (
                "weekly_tp_members_mm_day.npy",
                "tp_weekly_members_mm_day.npy",
                "members_weekly_mm_day.npy",
                "members.npy",
            ),
        )
    if members_path is None:
        raise FileNotFoundError(f"no finalized member .npy found under {root}")
    canonical_stem = members_path.with_suffix("")
    metadata_json = canonical_stem.with_suffix(".metadata.json")
    manifest_json = canonical_stem.with_suffix(".manifest.json")
    if not metadata_json.is_file():
        raise DataContractError(
            f"canonical experiment requires {metadata_json.name}; legacy metadata are rejected"
        )
    if not manifest_json.is_file():
        raise DataContractError(
            f"canonical experiment requires {manifest_json.name}; legacy manifests are rejected"
        )
    metadata_path = metadata_json
    manifest_path = manifest_json
    try:
        members, metadata = canonical_member_cache.load_member_cache(
            members_path,
            verify=True,
        )
    except (canonical_member_cache.MemberCacheContractError, OSError, ValueError) as error:
        raise DataContractError(f"canonical cache verification failed: {error}") from error
    required_keys = {
        "initializations",
        "source_init_indices",
        "latitude",
        "longitude",
        "member_labels",
        "shape",
        "dims",
        "output_units",
    }
    missing = sorted(required_keys - set(metadata))
    if missing:
        raise DataContractError(f"cache metadata is missing keys: {missing}")
    if tuple(metadata["dims"]) != (
        "init",
        "member",
        "lead_week",
        "lat",
        "lon",
    ):
        raise DataContractError(f"unexpected cache dims: {metadata['dims']}")
    if metadata["output_units"] != "mm day-1":
        raise DataContractError(f"unexpected cache units: {metadata['output_units']}")
    if tuple(metadata["shape"]) != tuple(members.shape):
        raise DataContractError("metadata shape differs from the NPY shape")
    initializations = np.asarray(metadata["initializations"], dtype="datetime64[D]")
    latitude = np.asarray(metadata["latitude"], dtype=np.float64)
    longitude = np.asarray(metadata["longitude"], dtype=np.float64)
    member_labels = np.asarray(metadata["member_labels"])
    source_indices = np.asarray(metadata["source_init_indices"], dtype=np.int64)
    if source_indices.shape != initializations.shape or np.unique(source_indices).size != source_indices.size:
        raise DataContractError("cache source_init_indices are invalid")
    if not allow_partial and str(metadata.get("scope")) != "full_archive":
        raise DataContractError("a full run requires cache scope='full_archive'")
    expected = EXPECTED_SHAPE
    if not allow_partial and tuple(members.shape) != expected:
        raise DataContractError(f"member cache shape {members.shape}, expected {expected}")
    if members.ndim != 5 or members.shape[1:] != expected[1:]:
        raise DataContractError(
            f"member cache must be [case,51,6,27,27], found {members.shape}"
        )
    if initializations.shape != (members.shape[0],):
        raise DataContractError("cache initialization count does not match member data")
    if latitude.shape != (27,) or longitude.shape != (27,):
        raise DataContractError("cache must contain the 27x27 India grid")
    if member_labels.shape != (51,) or np.unique(member_labels).size != 51:
        raise DataContractError("cache must contain 51 unique member labels")
    if np.unique(initializations).size != initializations.size:
        raise DataContractError("cache initializations are not unique")
    if np.any(np.diff(initializations) <= np.timedelta64(0, "D")):
        raise DataContractError("cache initializations are not sorted")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, Mapping):
        status = str(manifest_payload.get("status", ""))
        if status and status not in {"complete", "complete_smoke", "verified"}:
            raise DataContractError(f"cache manifest is not complete: status={status!r}")
    return MemberCache(
        members=members,
        initializations=initializations,
        latitude=latitude,
        longitude=longitude,
        member_labels=member_labels,
        cache_root=root,
        members_path=members_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )


def collapse_fraction(dataset: xr.Dataset, spatial_shape: tuple[int, int]) -> np.ndarray:
    fraction = dataset.observation_fraction
    other_dims = [dim for dim in fraction.dims if dim not in ("latitude", "longitude")]
    values = np.asarray(
        fraction.transpose(*(other_dims + ["latitude", "longitude"])).load().values,
        dtype=np.float32,
    )
    if values.ndim == 2:
        result = values
    elif values.ndim == 3:
        flattened = values.reshape(-1, *spatial_shape)
        result = flattened[0]
        if not np.allclose(flattened, result[None], rtol=0.0, atol=1.0e-7, equal_nan=True):
            raise DataContractError("IMD observation_fraction changes with time")
    else:
        raise DataContractError("unexpected IMD observation_fraction dimensions")
    if result.shape != spatial_shape:
        raise DataContractError("unexpected IMD observation_fraction shape")
    return result.copy()


def build_training_climatology(
    daily_dates: np.ndarray,
    daily_values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Equal-year centred 31-day IMD climatology using 2002--2017 only."""

    dates = np.asarray(daily_dates, dtype="datetime64[D]")
    values = np.asarray(daily_values, dtype=np.float32)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    train = np.isin(years, TRAIN_YEARS)
    train_dates = dates[train]
    train_values = values[train]
    train_years = years[train]
    positions = calendar_positions(train_dates)
    climatology = np.full((366, *support.shape), np.nan, dtype=np.float32)
    for day in range(366):
        circular_distance = np.minimum((positions - day) % 366, (day - positions) % 366)
        equal_year_means = []
        for year in TRAIN_YEARS:
            selected = (train_years == year) & (circular_distance <= 15)
            if not np.any(selected):
                raise DataContractError(f"no climatology values for day {day}, year {year}")
            mean = np.mean(train_values[selected], axis=0, dtype=np.float64)
            if not np.isfinite(mean[support]).all():
                raise DataContractError("non-finite supported training climatology")
            equal_year_means.append(mean)
        climatology[day] = np.mean(equal_year_means, axis=0, dtype=np.float64).astype(
            np.float32
        )
    climatology[:, ~support] = np.nan
    return climatology


def load_imd_observations(cache: MemberCache) -> ObservationBundle:
    """Load only 2002--2022 IMD; 2025 is deliberately inaccessible here."""

    if max(OBSERVATION_YEARS) >= min(SEALED_YEARS):
        raise RuntimeError("observation-year contract would open a sealed year")
    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    source_stores: list[str] = []
    observation_fraction: np.ndarray | None = None
    for year in OBSERVATION_YEARS:
        store = IMD_DAILY / f"{year}.zarr"
        if not (store / ".zmetadata").is_file():
            raise FileNotFoundError(store)
        with xr.open_zarr(store, consolidated=True) as dataset:
            if dataset.attrs.get("source") != "imd" or dataset.attrs.get("units") != "mm day-1":
                raise DataContractError(f"unexpected IMD metadata in {store}")
            if not np.array_equal(dataset.latitude.values, cache.latitude):
                raise DataContractError(f"IMD latitude differs in {store}")
            if not np.array_equal(dataset.longitude.values, cache.longitude):
                raise DataContractError(f"IMD longitude differs in {store}")
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            if values.shape != (dates.size, 27, 27):
                raise DataContractError(f"unexpected IMD observation shape in {store}")
            if dates.size not in (365, 366) or not np.all(np.diff(dates) == np.timedelta64(1, "D")):
                raise DataContractError(f"incomplete IMD daily calendar in {store}")
            fraction = collapse_fraction(dataset, (27, 27))
            if observation_fraction is None:
                observation_fraction = fraction
            elif not np.allclose(
                observation_fraction, fraction, rtol=0.0, atol=1.0e-7, equal_nan=True
            ):
                raise DataContractError("IMD support differs between years")
            support = fraction > 0.0
            if not np.isfinite(values[:, support]).all() or np.any(values[:, support] < 0.0):
                raise DataContractError(f"invalid supported IMD values in {store}")
            all_dates.append(dates)
            all_values.append(values)
            source_stores.append(str(store))
    assert observation_fraction is not None
    dates = np.concatenate(all_dates)
    values = np.concatenate(all_values)
    order = np.argsort(dates)
    dates = dates[order]
    values = values[order]
    if np.unique(dates).size != dates.size:
        raise DataContractError("duplicate IMD dates")
    valid_dates = derive_valid_dates(cache.initializations)
    requested = valid_dates.reshape(-1)
    positions = np.searchsorted(dates, requested)
    if np.any(positions >= dates.size) or not np.array_equal(dates[positions], requested):
        raise DataContractError("one or more FuXi verification dates are absent from IMD")
    daily = values[positions].reshape(*valid_dates.shape, 27, 27)
    weekly_truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    support = observation_fraction > 0.0
    daily_climatology = build_training_climatology(dates, values, support)
    climatology_positions = calendar_positions(valid_dates)
    weekly_climatology = np.mean(
        daily_climatology[climatology_positions], axis=2, dtype=np.float64
    ).astype(np.float32)
    with xr.open_zarr(SPATIAL_STORE, consolidated=True) as spatial:
        if not np.array_equal(spatial.latitude.values, cache.latitude):
            raise DataContractError("area-weight latitude differs")
        if not np.array_equal(spatial.longitude.values, cache.longitude):
            raise DataContractError("area-weight longitude differs")
        area = np.asarray(spatial.india_area_weight_km2.load().values, dtype=np.float64)
    weights = area * np.asarray(observation_fraction, dtype=np.float64)
    weights[~np.isfinite(weights) | (weights <= 0.0)] = 0.0
    if np.count_nonzero(weights) != SUPPORT_CELLS:
        raise DataContractError(
            f"expected {SUPPORT_CELLS} supported IMD cells, found {np.count_nonzero(weights)}"
        )
    if not np.isfinite(weekly_truth[..., support]).all() or np.any(weekly_truth[..., support] < 0):
        raise DataContractError("weekly IMD truth is invalid")
    if not np.isfinite(weekly_climatology[..., support]).all():
        raise DataContractError("weekly training climatology is invalid")
    return ObservationBundle(
        weekly_truth=weekly_truth,
        weekly_climatology=weekly_climatology,
        observation_fraction=observation_fraction,
        weights=weights,
        source_stores=tuple(source_stores),
    )


def weighted_lead_moments(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(values[indices], dtype=np.float64)
    means = np.empty(selected.shape[1], dtype=np.float32)
    stds = np.empty(selected.shape[1], dtype=np.float32)
    for lead in range(selected.shape[1]):
        field = selected[:, lead]
        expanded_weights = np.broadcast_to(weights, field.shape)
        valid = np.isfinite(field) & (expanded_weights > 0.0)
        denominator = expanded_weights[valid].sum(dtype=np.float64)
        mean = np.sum(expanded_weights[valid] * field[valid], dtype=np.float64) / denominator
        variance = (
            np.sum(expanded_weights[valid] * (field[valid] - mean) ** 2, dtype=np.float64)
            / denominator
        )
        means[lead] = np.float32(mean)
        stds[lead] = np.float32(max(math.sqrt(variance), 1.0e-6))
    return means, stds


def build_context_bundle(
    cache: MemberCache,
    observations: ObservationBundle,
    train_indices: np.ndarray,
) -> ContextBundle:
    support = observations.weights > 0.0
    log_climatology = np.log1p(observations.weekly_climatology).astype(np.float32)
    means, stds = weighted_lead_moments(log_climatology, train_indices, observations.weights)
    normalized = (log_climatology - means[None, :, None, None]) / stds[
        None, :, None, None
    ]
    normalized = np.where(support[None, None] & np.isfinite(normalized), normalized, 0.0)
    latitude = cache.latitude.astype(np.float32)
    longitude = cache.longitude.astype(np.float32)
    lat_scaled = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    lon_scaled = 2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    midpoints = derive_valid_dates(cache.initializations)[:, :, 3]
    midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
    day_of_year = (midpoint_index.dayofyear.to_numpy() - 1).reshape(len(midpoints), 6)
    angle = 2.0 * np.pi * day_of_year / 365.2425
    return ContextBundle(
        normalized_climatology=normalized.astype(np.float32),
        climatology_mean_by_lead=means,
        climatology_std_by_lead=stds,
        latitude_scaled=lat_scaled,
        longitude_scaled=lon_scaled,
        season_sin=np.sin(angle).astype(np.float32),
        season_cos=np.cos(angle).astype(np.float32),
        lead_scaled=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        support=support,
    )


def context_for_case(bundle: ContextBundle, case_index: int) -> np.ndarray:
    """Return seven shared channels in [lead, channel, latitude, longitude]."""

    height, width = bundle.support.shape
    fields = [
        bundle.normalized_climatology[case_index],
        np.broadcast_to(bundle.season_sin[case_index, :, None, None], (6, height, width)),
        np.broadcast_to(bundle.season_cos[case_index, :, None, None], (6, height, width)),
        np.broadcast_to(bundle.latitude_scaled[None, :, None], (6, height, width)),
        np.broadcast_to(bundle.longitude_scaled[None, None, :], (6, height, width)),
        np.broadcast_to(bundle.lead_scaled[:, None, None], (6, height, width)),
        np.broadcast_to(bundle.support[None], (6, height, width)),
    ]
    result = np.stack(fields, axis=1).astype(np.float32)
    if result.shape != (6, 7, 27, 27) or not np.isfinite(result).all():
        raise DataContractError("invalid context tensor")
    return result


class EnsembleCaseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        members: np.ndarray,
        truth: np.ndarray,
        context: ContextBundle,
        indices: np.ndarray,
    ) -> None:
        self.members = members
        self.truth = truth
        self.context = context
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        members = np.array(self.members[index], dtype=np.float32, copy=True)
        truth = np.array(self.truth[index], dtype=np.float32, copy=True)
        context = context_for_case(self.context, index)
        return torch.from_numpy(members), torch.from_numpy(context), torch.from_numpy(truth)


def select_evenly(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.size <= count:
        return values.copy()
    positions = np.linspace(0, values.size - 1, count, dtype=np.int64)
    return values[positions]


def verification_midpoint_months(initializations: np.ndarray) -> np.ndarray:
    midpoints = derive_valid_dates(initializations)[:, :, 3]
    return pd.DatetimeIndex(midpoints.reshape(-1)).month.to_numpy().reshape(-1, 6)


def fit_moment_calibration(
    members: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    train_indices: np.ndarray,
    weights: np.ndarray,
    *,
    shrinkage: float = 10.0,
) -> MomentFit:
    """Fit train-only lead/month log-location fields and scalar spread factors."""

    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    months = verification_midpoint_months(initializations)
    support = weights > 0.0
    delta = np.zeros((6, 12, 27, 27), dtype=np.float32)
    scale = np.ones((6, 12), dtype=np.float32)
    weight_sum = float(weights.sum(dtype=np.float64))
    for lead in range(6):
        for month in range(1, 13):
            selected = train_indices[months[train_indices, lead] == month]
            if train_indices.size >= 100 and selected.size < 10:
                raise DataContractError(f"too few train cases for W{lead + 1}, month {month}")
            if selected.size == 0:
                # Only reachable for a deliberately small non-scientific smoke cache.
                selected = train_indices
            raw = np.asarray(members[selected, :, lead], dtype=np.float32)
            raw_log = np.log1p(raw).astype(np.float32)
            mean_log = raw_log.mean(axis=1, dtype=np.float64)
            target_log = np.log1p(truth[selected, lead]).astype(np.float64)
            residual = target_log - mean_log
            global_delta = float(
                np.sum(
                    residual[:, support] * weights[support][None],
                    dtype=np.float64,
                )
                / (selected.size * weight_sum)
            )
            fitted_delta = np.zeros((27, 27), dtype=np.float64)
            cell_mean = np.mean(residual[:, support], axis=0, dtype=np.float64)
            fitted_delta[support] = (
                selected.size * cell_mean + shrinkage * global_delta
            ) / (selected.size + shrinkage)
            delta[lead, month - 1] = fitted_delta.astype(np.float32)
            centred = raw_log.astype(np.float64) - mean_log[:, None]
            variance = np.mean(centred**2, axis=1, dtype=np.float64)
            corrected_error = target_log - (mean_log + fitted_delta[None])
            numerator = float(
                np.sum(
                    corrected_error[:, support] ** 2 * weights[support][None],
                    dtype=np.float64,
                )
                / (selected.size * weight_sum)
            )
            denominator = float(
                np.sum(
                    variance[:, support] * weights[support][None],
                    dtype=np.float64,
                )
                / (selected.size * weight_sum)
            )
            ratio = math.sqrt(max(numerator, 1.0e-8) / max(denominator, 1.0e-8))
            scale[lead, month - 1] = np.float32(np.clip(ratio, 0.25, 4.0))
    return MomentFit(delta, scale, float(shrinkage))


def apply_affine_log_calibration(
    members: np.ndarray,
    delta_log_location: np.ndarray,
    spread_factor: np.ndarray,
) -> np.ndarray:
    """Apply a rank-preserving affine transform in log1p space."""

    raw = np.asarray(members, dtype=np.float32)
    delta = np.asarray(delta_log_location, dtype=np.float32)
    scale = np.asarray(spread_factor, dtype=np.float32)
    if raw.ndim != 5 or delta.shape != (raw.shape[0], 6, 27, 27):
        raise ValueError("members/delta have incompatible shapes")
    if scale.shape not in {(raw.shape[0], 6), (raw.shape[0], 6, 27, 27)}:
        raise ValueError("spread_factor must be [case,lead] or [case,lead,y,x]")
    log_members = np.log1p(raw).astype(np.float32)
    centre = log_members.mean(axis=1, dtype=np.float64).astype(np.float32)
    expanded_scale = scale[:, None]
    if scale.ndim == 2:
        expanded_scale = expanded_scale[..., None, None]
    corrected_log = centre[:, None] + delta[:, None] + expanded_scale * (
        log_members - centre[:, None]
    )
    corrected = np.expm1(np.clip(corrected_log, 0.0, 20.0)).astype(np.float32)
    if not np.isfinite(corrected).all() or np.any(corrected < 0.0):
        raise DataContractError("calibration produced invalid ensemble values")
    return corrected


def apply_moment_fit(
    members: np.ndarray,
    initializations: np.ndarray,
    fit: MomentFit,
) -> np.ndarray:
    months = verification_midpoint_months(initializations)
    n_cases = len(initializations)
    delta = np.empty((n_cases, 6, 27, 27), dtype=np.float32)
    scale = np.empty((n_cases, 6), dtype=np.float32)
    for lead in range(6):
        delta[:, lead] = fit.delta_log_location[lead, months[:, lead] - 1]
        scale[:, lead] = fit.spread_factor[lead, months[:, lead] - 1]
    return apply_affine_log_calibration(members, delta, scale)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_output_fields(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the small core API and fail loudly if its contract changes."""

    if isinstance(output, Mapping):
        corrected = output["corrected_members"]
        delta = output["delta_log_location"]
        scale = output["spread_factor"]
    else:
        corrected = output.corrected_members
        delta = output.delta_log_location
        scale = output.spread_factor
    return corrected, delta, scale


def _call_model(
    model: torch.nn.Module,
    members: torch.Tensor,
    context: torch.Tensor,
    *,
    member_subsample: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = {"context": context}
    if member_subsample is not None and member_subsample < members.shape[1]:
        kwargs["member_subsample_size"] = member_subsample
    output = model(members, **kwargs)
    return _model_output_fields(output)


def _crps_loss(
    corrected_members: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Call the core loss while tolerating an optional reduction keyword."""

    try:
        result = weighted_ensemble_crps(corrected_members, truth, weights)
    except TypeError:
        result = weighted_ensemble_crps(
            corrected_members, truth, weights, reduction="mean"
        )
    if result.ndim:
        result = result.mean()
    return result


def make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def validation_crps(
    model: torch.nn.Module,
    loader: DataLoader[Any],
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
            corrected, _, _ = _call_model(
                model, members, context, member_subsample=None
            )
            loss = _crps_loss(corrected, truth, weights)
        batch_count = int(members.shape[0])
        numerator += float(loss.detach().cpu()) * batch_count
        count += batch_count
    if count == 0:
        raise RuntimeError("validation loader is empty")
    return numerator / count


def train_one_model(
    configuration: str,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: ContextBundle,
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
) -> tuple[torch.nn.Module, pd.DataFrame, TrainingRun]:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {configuration!r}")
    set_deterministic_seed(seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best.pt"
    model = EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=8,
        backbone_channels=24,
        mode=configuration,
        dropout=0.05,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_data = EnsembleCaseDataset(members, truth, context, train_indices)
    validation_data = EnsembleCaseDataset(members, truth, context, validation_indices)
    train_loader = make_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    validation_loader = make_loader(
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
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3), min_lr=1.0e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    history_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    initial_validation = validation_crps(
        model, validation_loader, spatial_weights, device, use_amp=use_amp
    )
    history_rows.append(
        {
            "configuration": configuration,
            "seed": seed,
            "epoch": 0,
            "train_crps": np.nan,
            "validation_crps": initial_validation,
            "learning_rate": learning_rate,
            "is_best": True,
        }
    )
    best_loss = initial_validation
    best_epoch = 0
    stale_epochs = 0
    stopping_reason = "max_epochs"
    torch.save(
        {
            "experiment": EXPERIMENT,
            "configuration": configuration,
            "seed": seed,
            "epoch": 0,
            "validation_crps": initial_validation,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
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
                corrected, _, _ = _call_model(
                    model,
                    batch_members,
                    batch_context,
                    member_subsample=member_subsample,
                )
                loss = _crps_loss(corrected, batch_truth, spatial_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training CRPS for {configuration}, seed {seed}, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(batch_members.shape[0])
            train_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count
        train_loss = train_sum / train_count
        current_validation = validation_crps(
            model, validation_loader, spatial_weights, device, use_amp=use_amp
        )
        scheduler.step(current_validation)
        improved = current_validation < best_loss - 1.0e-6
        if improved:
            best_loss = current_validation
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "configuration": configuration,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_crps": current_validation,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        history_rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "epoch": epoch,
                "train_crps": train_loss,
                "validation_crps": current_validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"[{configuration} seed={seed}] epoch={epoch:03d} "
            f"train_CRPS={train_loss:.6f} val_CRPS={current_validation:.6f} "
            f"best={best_loss:.6f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= patience:
            stopping_reason = "early_stopping_patience"
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    run = TrainingRun(
        configuration=configuration,
        seed=seed,
        best_epoch=best_epoch,
        stopped_epoch=int(history_rows[-1]["epoch"]),
        stopping_reason=stopping_reason,
        best_validation_crps=float(best_loss),
        elapsed_seconds=float(time.monotonic() - start_time),
        parameter_count=parameter_count,
        checkpoint=str(checkpoint_path),
    )
    return model, pd.DataFrame(history_rows), run


@torch.no_grad()
def predict_adjustments(
    model: torch.nn.Module,
    members: np.ndarray,
    truth: np.ndarray,
    context: ContextBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = EnsembleCaseDataset(members, truth, context, indices)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    model.eval()
    deltas: list[np.ndarray] = []
    log_scales: list[np.ndarray] = []
    for batch_members, batch_context, _ in loader:
        batch_members = batch_members.to(device, non_blocking=True)
        batch_context = batch_context.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            _, delta, scale = _call_model(
                model, batch_members, batch_context, member_subsample=None
            )
        delta_array = delta.detach().float().cpu().numpy()
        scale_array = scale.detach().float().cpu().numpy()
        if delta_array.ndim == 5 and delta_array.shape[1] == 1:
            delta_array = delta_array[:, 0]
        if scale_array.ndim == 5 and scale_array.shape[1] == 1:
            scale_array = scale_array[:, 0]
        deltas.append(delta_array.astype(np.float32))
        log_scales.append(np.log(np.clip(scale_array, 1.0e-4, 1.0e4)).astype(np.float32))
    delta_result = np.concatenate(deltas, axis=0)
    log_scale_result = np.concatenate(log_scales, axis=0)
    if delta_result.shape != (len(indices), 6, 27, 27):
        raise DataContractError(f"unexpected predicted delta shape {delta_result.shape}")
    if log_scale_result.shape == (len(indices), 6):
        log_scale_result = np.broadcast_to(
            log_scale_result[..., None, None], delta_result.shape
        ).copy()
    if log_scale_result.shape != delta_result.shape:
        raise DataContractError(
            f"unexpected predicted log-spread shape {log_scale_result.shape}"
        )
    return delta_result, log_scale_result


def _weighted_field_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    support = np.asarray(weights, dtype=np.float64) > 0.0
    selected = np.asarray(values)[..., support]
    selected_weights = np.asarray(weights, dtype=np.float64)[support]
    if not np.isfinite(selected).all():
        raise DataContractError("metric field is non-finite on positive-weight support")
    return np.sum(
        selected * selected_weights,
        axis=-1,
        dtype=np.float64,
    ) / float(selected_weights.sum(dtype=np.float64))


def numpy_ensemble_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Finite-ensemble CRPS at every case/lead/grid point."""

    values = np.asarray(members, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    member_count = values.shape[1]
    first = np.mean(np.abs(values - target[:, None]), axis=1, dtype=np.float64)
    ordered = np.sort(values, axis=1)
    coefficients = (2.0 * np.arange(member_count) - member_count + 1.0).reshape(
        1, member_count, 1, 1, 1
    )
    dispersion = np.sum(ordered * coefficients, axis=1, dtype=np.float64) / (
        member_count**2
    )
    return first - dispersion


def _centred_weighted_acc(
    truth_anomaly: np.ndarray,
    prediction_anomaly: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    support = np.asarray(weights, dtype=np.float64) > 0.0
    selected_weights = np.asarray(weights, dtype=np.float64)[support]
    truth_selected = np.asarray(truth_anomaly, dtype=np.float64)[..., support]
    prediction_selected = np.asarray(prediction_anomaly, dtype=np.float64)[..., support]
    if not np.isfinite(truth_selected).all() or not np.isfinite(prediction_selected).all():
        raise DataContractError("ACC anomaly is non-finite on positive-weight support")
    denominator_weight = float(selected_weights.sum(dtype=np.float64))
    truth_mean = np.sum(
        truth_selected * selected_weights, axis=-1, dtype=np.float64
    ) / denominator_weight
    prediction_mean = np.sum(
        prediction_selected * selected_weights, axis=-1, dtype=np.float64
    ) / denominator_weight
    truth_centred = truth_selected - truth_mean[..., None]
    prediction_centred = prediction_selected - prediction_mean[..., None]
    covariance = np.sum(
        truth_centred * prediction_centred * selected_weights,
        axis=-1,
        dtype=np.float64,
    )
    truth_variance = np.sum(
        truth_centred**2 * selected_weights, axis=-1, dtype=np.float64
    )
    prediction_variance = np.sum(
        prediction_centred**2 * selected_weights, axis=-1, dtype=np.float64
    )
    denominator = np.sqrt(truth_variance * prediction_variance)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = covariance / denominator
    result[denominator <= np.finfo(np.float64).eps * denominator_weight] = np.nan
    return np.clip(result, -1.0, 1.0)


def season_labels(initializations: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(initializations).month.to_numpy()
    return np.asarray(
        [
            "DJF"
            if month in (12, 1, 2)
            else "MAM"
            if month in (3, 4, 5)
            else "JJA"
            if month in (6, 7, 8)
            else "SON"
            for month in months
        ],
        dtype=object,
    )


def evaluate_ensemble(
    method: str,
    members: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    initializations: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 8,
    rank_seed: int = 42,
    seed_label: str | int = "not_applicable",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce one statistically grouped row per initialization and lead."""

    n_cases, member_count, n_leads, height, width = members.shape
    if truth.shape != (n_cases, n_leads, height, width):
        raise DataContractError("truth shape does not match ensemble")
    if climatology.shape != truth.shape:
        raise DataContractError("climatology shape does not match truth")
    rng = np.random.default_rng(rank_seed)
    rank_counts = np.zeros((n_leads, member_count + 1), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    labels = season_labels(initializations)
    init_strings = [np.datetime_as_string(value, unit="D") for value in initializations]
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    support = weights > 0.0
    for start in range(0, n_cases, chunk_size):
        stop = min(start + chunk_size, n_cases)
        ensemble = np.asarray(members[start:stop], dtype=np.float32)
        target = np.asarray(truth[start:stop], dtype=np.float32)
        climate = np.asarray(climatology[start:stop], dtype=np.float32)
        if not np.isfinite(ensemble).all() or np.any(ensemble < 0.0):
            raise DataContractError(f"{method} ensemble contains invalid values")
        if not np.isfinite(target[..., support]).all() or np.any(target[..., support] < 0.0):
            raise DataContractError("truth is invalid on positive-weight support")
        if not np.isfinite(climate[..., support]).all():
            raise DataContractError("climatology is invalid on positive-weight support")
        mean = ensemble.mean(axis=1, dtype=np.float64)
        error = mean - target
        crps = _weighted_field_mean(numpy_ensemble_crps(ensemble, target), weights)
        mean_squared_error = _weighted_field_mean(error**2, weights)
        rmse = np.sqrt(mean_squared_error)
        mae = _weighted_field_mean(np.abs(error), weights)
        bias = _weighted_field_mean(error, weights)
        acc = _centred_weighted_acc(target - climate, mean - climate, weights)
        ensemble_variance = _weighted_field_mean(
            ensemble.var(axis=1, ddof=0, dtype=np.float64), weights
        )
        spread = np.sqrt(ensemble_variance)
        extras: dict[str, np.ndarray] = {}
        for threshold in THRESHOLDS_MM_DAY:
            probability = np.mean(ensemble >= threshold, axis=1, dtype=np.float64)
            event = (target >= threshold).astype(np.float64)
            suffix = f"{threshold:g}"
            extras[f"brier_{suffix}"] = _weighted_field_mean(
                (probability - event) ** 2, weights
            )
            extras[f"forecast_probability_{suffix}"] = _weighted_field_mean(
                probability, weights
            )
            extras[f"observed_frequency_{suffix}"] = _weighted_field_mean(event, weights)
        for coverage in COVERAGES:
            alpha = (1.0 - coverage) / 2.0
            lower, upper = np.quantile(
                ensemble, (alpha, 1.0 - alpha), axis=1, method="linear"
            )
            covered = ((target >= lower) & (target <= upper)).astype(np.float64)
            suffix = f"{int(round(100 * coverage))}"
            extras[f"coverage_{suffix}"] = _weighted_field_mean(covered, weights)
            extras[f"width_{suffix}"] = _weighted_field_mean(upper - lower, weights)
        for lead in range(n_leads):
            member_values = ensemble[:, :, lead, support]
            target_values = target[:, lead, support]
            lower = np.sum(member_values < target_values[:, None], axis=1)
            ties = np.sum(member_values == target_values[:, None], axis=1)
            tie_offsets = np.asarray(
                [rng.integers(0, int(tie) + 1) for tie in ties.reshape(-1)],
                dtype=np.int64,
            ).reshape(ties.shape)
            ranks = (lower + tie_offsets).reshape(-1)
            rank_weights = np.broadcast_to(
                weights[support][None], lower.shape
            ).reshape(-1)
            rank_counts[lead] += np.bincount(
                ranks,
                weights=rank_weights,
                minlength=member_count + 1,
            )
        for local in range(stop - start):
            global_index = start + local
            for lead in range(n_leads):
                row: dict[str, Any] = {
                    "split": "test_development",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed_label,
                    "init": init_strings[global_index],
                    "year": int(years[global_index]),
                    "season": str(labels[global_index]),
                    "lead_week": lead + 1,
                    "member_count": member_count,
                    "support_cells": int(np.count_nonzero(support)),
                    "crps": float(crps[local, lead]),
                    "rmse": float(rmse[local, lead]),
                    "mae": float(mae[local, lead]),
                    "bias": float(bias[local, lead]),
                    "absolute_bias": float(abs(bias[local, lead])),
                    "acc": float(acc[local, lead]),
                    "ensemble_spread": float(spread[local, lead]),
                    "ensemble_variance": float(ensemble_variance[local, lead]),
                    "mean_squared_error": float(mean_squared_error[local, lead]),
                    "spread_skill_ratio": (
                        float(spread[local, lead] / rmse[local, lead])
                        if rmse[local, lead] > 0.0
                        else np.nan
                    ),
                }
                for name, values in extras.items():
                    row[name] = float(values[local, lead])
                rows.append(row)
    rank_rows = [
        {
            "method": method,
            "lead_week": lead + 1,
            "rank": rank,
            "count": float(rank_counts[lead, rank]),
            "weighting": "india_area_weight_km2_x_observation_fraction",
        }
        for lead in range(n_leads)
        for rank in range(member_count + 1)
    ]
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def reliability_bins(
    method: str,
    members: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    *,
    bin_count: int = 10,
    chunk_size: int = 8,
) -> pd.DataFrame:
    """Bin member event probabilities without treating cells as samples in CIs."""

    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    n_cases, _, n_leads, _, _ = members.shape
    support = weights > 0.0
    weight_values = weights[support].astype(np.float64)
    weight_sum = np.zeros((n_leads, len(THRESHOLDS_MM_DAY), bin_count), dtype=np.float64)
    probability_sum = np.zeros_like(weight_sum)
    event_sum = np.zeros_like(weight_sum)
    sample_count = np.zeros_like(weight_sum, dtype=np.int64)
    for start in range(0, n_cases, chunk_size):
        stop = min(start + chunk_size, n_cases)
        ensemble = np.asarray(members[start:stop], dtype=np.float32)
        target = np.asarray(truth[start:stop], dtype=np.float32)
        for threshold_index, threshold in enumerate(THRESHOLDS_MM_DAY):
            probability = np.mean(ensemble >= threshold, axis=1, dtype=np.float64)
            event = target >= threshold
            for lead in range(n_leads):
                lead_probability = probability[:, lead, support]
                lead_event = event[:, lead, support].astype(np.float64)
                bins = np.minimum((lead_probability * bin_count).astype(np.int64), bin_count - 1)
                expanded_weights = np.broadcast_to(weight_values[None], lead_probability.shape)
                for bin_index in range(bin_count):
                    selected = bins == bin_index
                    if not np.any(selected):
                        continue
                    selected_weights = expanded_weights[selected]
                    weight_sum[lead, threshold_index, bin_index] += selected_weights.sum()
                    probability_sum[lead, threshold_index, bin_index] += np.sum(
                        selected_weights * lead_probability[selected], dtype=np.float64
                    )
                    event_sum[lead, threshold_index, bin_index] += np.sum(
                        selected_weights * lead_event[selected], dtype=np.float64
                    )
                    sample_count[lead, threshold_index, bin_index] += int(np.count_nonzero(selected))
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for lead in range(n_leads):
        for threshold_index, threshold in enumerate(THRESHOLDS_MM_DAY):
            for bin_index in range(bin_count):
                denominator = weight_sum[lead, threshold_index, bin_index]
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "lead_week": lead + 1,
                        "threshold_mm_day": threshold,
                        "probability_bin": bin_index,
                        "bin_lower": edges[bin_index],
                        "bin_upper": edges[bin_index + 1],
                        "cell_case_count": int(sample_count[lead, threshold_index, bin_index]),
                        "area_weight_sum": float(denominator),
                        "forecast_probability_weighted_sum": float(
                            probability_sum[lead, threshold_index, bin_index]
                        ),
                        "observed_event_weighted_sum": float(
                            event_sum[lead, threshold_index, bin_index]
                        ),
                        "mean_forecast_probability": (
                            float(probability_sum[lead, threshold_index, bin_index] / denominator)
                            if denominator > 0.0
                            else np.nan
                        ),
                        "observed_frequency": (
                            float(event_sum[lead, threshold_index, bin_index] / denominator)
                            if denominator > 0.0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def add_raw_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    raw = result.loc[result.method == "raw_fuxi"].set_index("lead_week")
    for index, row in result.iterrows():
        reference = raw.loc[int(row.lead_week)]
        result.loc[index, "crpss_vs_raw"] = _safe_skill(row.crps, reference.crps)
        result.loc[index, "crps_skill_pct_vs_raw"] = 100.0 * _safe_skill(
            row.crps, reference.crps
        )
        result.loc[index, "rmse_skill_pct_vs_raw"] = 100.0 * _safe_skill(
            row.rmse, reference.rmse
        )
        result.loc[index, "mae_skill_pct_vs_raw"] = 100.0 * _safe_skill(
            row.mae, reference.mae
        )
        result.loc[index, "delta_acc_vs_raw"] = row.acc - reference.acc
        result.loc[index, "delta_bias_vs_raw"] = row.bias - reference.bias
        for threshold in THRESHOLDS_MM_DAY:
            column = f"brier_{threshold:g}"
            result.loc[index, f"{column}_skill_vs_raw"] = _safe_skill(
                row[column], reference[column]
            )
    return result


def _safe_skill(value: float, reference: float) -> float:
    """Return 1-value/reference, defining identical perfect scores as zero skill."""

    if reference > np.finfo(np.float64).eps:
        return float(1.0 - value / reference)
    if abs(value) <= np.finfo(np.float64).eps:
        return 0.0
    return float("nan")


def recompute_grouped_spread_skill(frame: pd.DataFrame) -> pd.DataFrame:
    """Use pooled second moments, never a mean of unstable case-wise ratios."""

    result = frame.copy()
    result["ensemble_spread"] = np.sqrt(
        np.maximum(result.ensemble_variance.to_numpy(dtype=np.float64), 0.0)
    )
    denominator = result.mean_squared_error.to_numpy(dtype=np.float64)
    numerator = result.ensemble_variance.to_numpy(dtype=np.float64)
    ratio = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(
        np.maximum(numerator, 0.0),
        denominator,
        out=ratio,
        where=denominator > 0.0,
    )
    result["spread_skill_ratio"] = np.sqrt(ratio)
    return result


def summarize_metrics(
    case_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [
        column
        for column in case_metrics.columns
        if column
        not in {
            "split",
            "method",
            "method_label",
            "seed",
            "init",
            "year",
            "season",
            "lead_week",
            "member_count",
            "support_cells",
        }
    ]
    weekwise = (
        case_metrics.groupby(
            ["method", "method_label", "seed", "lead_week"], as_index=False
        )[numeric]
        .mean()
        .sort_values(["method", "seed", "lead_week"])
    )
    weekwise = recompute_grouped_spread_skill(weekwise)
    counts = case_metrics.groupby(
        ["method", "seed", "lead_week"], as_index=False
    ).agg(
        n_initializations=("init", "nunique"),
        n_valid_cells=("support_cells", "first"),
    )
    weekwise = weekwise.merge(
        counts, on=["method", "seed", "lead_week"], validate="one_to_one"
    )
    weekwise.insert(0, "split", "test_development")
    weekwise["week"] = weekwise.lead_week
    weekwise["case_count"] = weekwise.n_initializations
    weekwise = add_raw_comparisons(weekwise)
    pooled = (
        case_metrics.groupby(["method", "method_label", "seed"], as_index=False)[numeric]
        .mean()
        .sort_values("method")
    )
    pooled = recompute_grouped_spread_skill(pooled)
    raw_pooled = pooled.loc[pooled.method == "raw_fuxi"].iloc[0]
    pooled["crpss_vs_raw"] = [
        _safe_skill(value, raw_pooled.crps) for value in pooled.crps
    ]
    pooled["crps_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, raw_pooled.crps) for value in pooled.crps
    ]
    pooled["rmse_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, raw_pooled.rmse) for value in pooled.rmse
    ]
    pooled["mae_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, raw_pooled.mae) for value in pooled.mae
    ]
    pooled["delta_acc_vs_raw"] = pooled.acc - raw_pooled.acc
    pooled["delta_bias_vs_raw"] = pooled.bias - raw_pooled.bias
    pooled["case_count"] = case_metrics.groupby("method")["init"].nunique().reindex(
        pooled.method
    ).to_numpy()
    seasonal = (
        case_metrics.groupby(
            ["method", "method_label", "seed", "season", "lead_week"], as_index=False
        )[numeric]
        .mean()
        .sort_values(["season", "method", "lead_week"])
    )
    seasonal = recompute_grouped_spread_skill(seasonal)
    reliability_columns = [
        "split",
        "method",
        "method_label",
        "seed",
        "lead_week",
        "n_initializations",
        "n_valid_cells",
        *[
            name
            for threshold in THRESHOLDS_MM_DAY
            for name in (
                f"brier_{threshold:g}",
                f"forecast_probability_{threshold:g}",
                f"observed_frequency_{threshold:g}",
            )
        ],
    ]
    reliability = weekwise[reliability_columns].copy()
    return weekwise, pooled, seasonal, reliability


def write_metric_matrices(weekwise: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    excluded = {"split", "method", "method_label", "seed", "lead_week", "week"}
    written: list[str] = []
    for metric in [column for column in weekwise.columns if column not in excluded]:
        if not pd.api.types.is_numeric_dtype(weekwise[metric]):
            continue
        matrix = weekwise.pivot(index="method", columns="lead_week", values=metric)
        matrix = matrix.rename(columns={lead: f"W{int(lead)}" for lead in matrix.columns})
        matrix.insert(0, "method_label", [METHOD_LABELS[name] for name in matrix.index])
        path = output / f"{metric}_by_method_week.csv"
        matrix.to_csv(path)
        written.append(str(path))
    return written


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_training_loss(history: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    configurations = [name for name in CONFIGURATIONS if name in set(history.configuration)]
    figure, axes = plt.subplots(
        1,
        len(configurations),
        figsize=(7.2, 2.85),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    seed_styles = ("-", "--", ":", "-.")
    for axis_index, (axis, configuration) in enumerate(zip(axes[0], configurations)):
        selected = history.loc[history.configuration == configuration]
        for seed_index, (seed, rows) in enumerate(selected.groupby("seed")):
            rows = rows.sort_values("epoch")
            train = rows.loc[rows.epoch > 0]
            line_style = seed_styles[seed_index % len(seed_styles)]
            axis.plot(
                train.epoch,
                train.train_crps,
                color="0.62",
                linestyle=line_style,
                linewidth=1.0,
            )
            axis.plot(
                rows.epoch,
                rows.validation_crps,
                color=METHOD_COLORS[configuration],
                linewidth=1.5,
                linestyle=line_style,
                label=f"Seed {seed}",
            )
            best = rows.loc[rows.is_best.astype(bool)].iloc[-1]
            axis.scatter(
                [best.epoch],
                [best.validation_crps],
                color=METHOD_COLORS[configuration],
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
        axis.set_title(METHOD_LABELS[configuration])
        axis.set_xlabel("Epoch")
        if axis_index == 0:
            axis.set_ylabel("Ensemble CRPS (mm day$^{-1}$)")
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            frameon=False,
            fontsize=6.2,
            title="colored=val · gray=train",
            title_fontsize=5.8,
            handlelength=2.2,
        )
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "All-season FuXi ensemble calibration: training histories",
        fontsize=10.0,
        fontweight="semibold",
    )
    _save_figure(figure, output)


def plot_weekwise_metrics(
    weekwise: pd.DataFrame,
    output: Path,
    method_order: Sequence[str],
    *,
    smoke: bool,
    bootstrap: pd.DataFrame | None = None,
    seed_variability: pd.DataFrame | None = None,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), constrained_layout=False)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.12,
        top=0.78,
        hspace=0.42,
        wspace=0.38,
    )
    panels = (
        ("crps", "CRPS (mm day$^{-1}$)", None),
        ("crpss_vs_raw", "CRPSS versus raw FuXi", 0.0),
        ("rmse", "Ensemble-mean RMSE (mm day$^{-1}$)", None),
        ("mae", "Ensemble-mean MAE (mm day$^{-1}$)", None),
        ("bias", "Ensemble-mean bias (mm day$^{-1}$)", 0.0),
        ("acc", "Spatial anomaly correlation", None),
    )
    for panel_index, (axis, (metric, ylabel, reference)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        for method in method_order:
            rows = weekwise.loc[weekwise.method == method].sort_values("lead_week")
            axis.plot(
                rows.lead_week,
                rows[metric],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle="--" if method == "raw_fuxi" else "-",
                linewidth=1.15,
                markersize=3.0,
                label=PLOT_METHOD_LABELS[method],
            )
        if metric == "crpss_vs_raw" and bootstrap is not None:
            uncertainty = bootstrap.loc[
                (bootstrap.method == PRIMARY_CONFIGURATION)
                & (bootstrap.metric == "crps")
                & bootstrap.lead_scope.str.fullmatch(r"W[1-6]")
            ].copy()
            if len(uncertainty) == 6:
                uncertainty["lead_week"] = uncertainty.lead_scope.str[1:].astype(int)
                uncertainty = uncertainty.sort_values("lead_week")
                axis.fill_between(
                    uncertainty.lead_week,
                    uncertainty.ci_lower / 100.0,
                    uncertainty.ci_upper / 100.0,
                    color=METHOD_COLORS[PRIMARY_CONFIGURATION],
                    alpha=0.16,
                    linewidth=0,
                    zorder=0,
                )
        elif seed_variability is not None and f"{metric}_min" in seed_variability:
            variability = seed_variability.loc[
                seed_variability.method == PRIMARY_CONFIGURATION
            ].sort_values("lead_week")
            if len(variability) == 6:
                axis.fill_between(
                    variability.lead_week,
                    variability[f"{metric}_min"],
                    variability[f"{metric}_max"],
                    color=METHOD_COLORS[PRIMARY_CONFIGURATION],
                    alpha=0.12,
                    linewidth=0,
                    zorder=0,
                )
        if reference is not None:
            axis.axhline(reference, color="0.5", linewidth=0.8, linestyle=":")
        axis.set_xticks(range(1, 7))
        axis.set_xlabel("Lead week", labelpad=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.0,
            1.03,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            va="bottom",
            fontweight="semibold",
            clip_on=False,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        columnspacing=1.0,
        handlelength=1.8,
    )
    title = "SMOKE — " if smoke else ""
    figure.suptitle(
        title + "2020–2021 all-season development test: calibrated ensemble vs raw FuXi",
        fontsize=10.2,
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Shading: primary three-seed range; CRPSS panel: 95% paired block-bootstrap CI.",
        ha="center",
        fontsize=6.8,
    )
    _save_figure(figure, output)


def plot_skill_heatmaps(
    weekwise: pd.DataFrame,
    output: Path,
    method_order: Sequence[str],
    *,
    smoke: bool,
) -> None:
    calibrated = [method for method in method_order if method != "raw_fuxi"]
    panels = (
        ("crps_skill_pct_vs_raw", "CRPS skill vs raw (%)", "RdYlBu", True),
        ("rmse_skill_pct_vs_raw", "RMSE reduction vs raw (%)", "RdYlBu", True),
        ("delta_acc_vs_raw", "ACC difference vs raw", "RdYlBu", True),
        ("bias", "Mean bias (mm day$^{-1}$)", "RdBu", True),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)
    for axis, (metric, title, cmap, symmetric) in zip(axes.ravel(), panels):
        matrix = np.stack(
            [
                weekwise.loc[weekwise.method == method]
                .sort_values("lead_week")[metric]
                .to_numpy(dtype=np.float64)
                for method in calibrated
            ]
        )
        limit = max(float(np.nanmax(np.abs(matrix))), 1.0e-6)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=-limit if symmetric else None,
            vmax=limit if symmetric else None,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                display = f"{value:+.1f}" if "pct" in metric else f"{value:+.2f}"
                axis.text(
                    column,
                    row,
                    display,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) > 0.55 * limit else "black",
                )
        axis.set_xticks(np.arange(6), [f"W{lead}" for lead in range(1, 7)])
        axis.set_yticks(
            np.arange(len(calibrated)),
            [PLOT_METHOD_LABELS[m] for m in calibrated],
        )
        axis.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(calibrated), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.72)
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "Weekwise ablation metrics (skill and ACC vs raw; bias absolute)",
        fontweight="semibold",
    )
    _save_figure(figure, output)


def plot_rank_histograms(
    ranks: pd.DataFrame,
    output: Path,
    primary_method: str,
    *,
    smoke: bool,
) -> None:
    methods = ["raw_fuxi", primary_method]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.75),
        sharey=True,
        constrained_layout=True,
    )
    for axis, method in zip(axes, methods):
        pooled = ranks.loc[ranks.method == method].groupby("rank", as_index=False)["count"].sum()
        total = pooled["count"].sum()
        frequency = pooled["count"] / total
        axis.bar(
            pooled["rank"],
            frequency,
            width=0.9,
            color=METHOD_COLORS[method],
            alpha=0.82,
        )
        axis.axhline(1.0 / len(pooled), color="black", linestyle="--", linewidth=1.0)
        edge_mass = float(frequency.iloc[0] + frequency.iloc[-1])
        axis.text(
            0.97,
            0.93,
            f"edge mass = {edge_mass:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
        )
        axis.set_title(PLOT_METHOD_LABELS[method])
        axis.set_xlabel("Verification rank (0–51)")
        axis.set_ylabel("Relative frequency")
        axis.spines[["top", "right"]].set_visible(False)
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(prefix + "Pooled ensemble rank histograms", fontweight="semibold")
    _save_figure(figure, output)


def plot_reliability(
    reliability: pd.DataFrame,
    output: Path,
    method_order: Sequence[str],
    *,
    smoke: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.12,
        top=0.79,
        hspace=0.38,
        wspace=0.28,
    )
    for axis, threshold in zip(axes.ravel(), THRESHOLDS_MM_DAY):
        selected_threshold = reliability.loc[
            reliability.threshold_mm_day == threshold
        ]
        for method in method_order:
            selected = selected_threshold.loc[selected_threshold.method == method]
            pooled = selected.groupby("probability_bin", as_index=False)[
                [
                    "area_weight_sum",
                    "forecast_probability_weighted_sum",
                    "observed_event_weighted_sum",
                ]
            ].sum()
            valid = pooled.area_weight_sum > 0.0
            x = (
                pooled.loc[valid, "forecast_probability_weighted_sum"]
                / pooled.loc[valid, "area_weight_sum"]
            )
            y = (
                pooled.loc[valid, "observed_event_weighted_sum"]
                / pooled.loc[valid, "area_weight_sum"]
            )
            axis.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                linewidth=1.1,
                label=PLOT_METHOD_LABELS[method],
            )
            support = pooled.loc[valid, "area_weight_sum"].to_numpy(dtype=np.float64)
            support_scale = support / max(float(support.max()), 1.0e-12)
            axis.scatter(
                x,
                y,
                s=7.0 + 24.0 * np.sqrt(support_scale),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidths=0.25,
                edgecolors="white",
                zorder=3,
            )
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.9)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Forecast probability")
        axis.set_ylabel("Observed frequency")
        axis.set_title(f"Rainfall ≥ {threshold:g} mm day$^{{-1}}$")
        axis.grid(True, alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        columnspacing=1.0,
        handlelength=1.8,
    )
    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "All-season ensemble reliability (W1–W6 pooled)",
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Marker area is proportional to probability-bin support within each method and threshold.",
        ha="center",
        fontsize=6.8,
    )
    _save_figure(figure, output)


def plot_probabilistic_diagnostics(
    weekwise: pd.DataFrame,
    seasonal: pd.DataFrame,
    output: Path,
    method_order: Sequence[str],
    *,
    smoke: bool,
) -> None:
    """Plot spread, coverage, threshold skill, and seasonal robustness."""

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.12,
        top=0.82,
        hspace=0.42,
        wspace=0.32,
    )

    axis = axes[0, 0]
    for method in method_order:
        rows = weekwise.loc[weekwise.method == method].sort_values("lead_week")
        axis.plot(
            rows.lead_week,
            rows.spread_skill_ratio,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="--" if method == "raw_fuxi" else "-",
            linewidth=1.15,
            markersize=3.0,
            label=PLOT_METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    axis.set_ylabel("RMS spread / pooled RMS error")
    axis.set_title("(a) Spread–error consistency")
    axis.legend(frameon=False, fontsize=6.2, ncol=2, columnspacing=0.8)

    axis = axes[0, 1]
    coverage_colors = {50: "#0072B2", 80: "#E69F00", 90: "#CC79A7"}
    for coverage, color in coverage_colors.items():
        column = f"coverage_{coverage}"
        for method, line_style in (("raw_fuxi", "--"), (PRIMARY_CONFIGURATION, "-")):
            rows = weekwise.loc[weekwise.method == method].sort_values("lead_week")
            axis.plot(
                rows.lead_week,
                rows[column],
                color=color,
                linestyle=line_style,
                linewidth=1.2,
                marker="o" if method == PRIMARY_CONFIGURATION else None,
                markersize=2.7,
                label=f"{coverage}% · {'primary' if method == PRIMARY_CONFIGURATION else 'raw'}",
            )
        axis.axhline(coverage / 100.0, color=color, alpha=0.25, linewidth=0.7)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Empirical central coverage")
    axis.set_title("(b) Interval coverage")
    axis.legend(frameon=False, fontsize=5.9, ncol=2, columnspacing=0.7)

    axis = axes[1, 0]
    primary = weekwise.loc[
        weekwise.method == PRIMARY_CONFIGURATION
    ].sort_values("lead_week")
    threshold_colors = {
        1: "#56B4E9",
        5: "#009E73",
        10: "#E69F00",
        20: "#D55E00",
    }
    for threshold, color in threshold_colors.items():
        axis.plot(
            primary.lead_week,
            100.0 * primary[f"brier_{threshold}_skill_vs_raw"],
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2.7,
            label=f"≥{threshold} mm day$^{{-1}}$",
        )
    axis.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
    axis.set_ylabel("Brier skill vs raw (%)")
    axis.set_title("(c) Primary threshold-probability skill")
    axis.legend(frameon=False, fontsize=6.2, ncol=2, columnspacing=0.8)

    axis = axes[1, 1]
    seasons = ("DJF", "MAM", "JJA", "SON")
    bar_methods = tuple(
        method
        for method in ("summary_only", PRIMARY_CONFIGURATION)
        if method in set(seasonal.method)
    )
    positions = np.arange(len(seasons), dtype=np.float64)
    width = 0.36
    for method_index, method in enumerate(bar_methods):
        skills = []
        for season in seasons:
            rows = seasonal.loc[
                (seasonal.season == season) & (seasonal.method == method)
            ]
            raw_rows = seasonal.loc[
                (seasonal.season == season) & (seasonal.method == "raw_fuxi")
            ]
            skills.append(
                100.0
                * (raw_rows.crps.mean() - rows.crps.mean())
                / raw_rows.crps.mean()
            )
        offset = (method_index - (len(bar_methods) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            skills,
            width=width,
            color=METHOD_COLORS[method],
            label=PLOT_METHOD_LABELS[method],
        )
    axis.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
    axis.set_xticks(positions, seasons)
    axis.set_ylabel("Pooled CRPS skill vs raw (%)")
    axis.set_title("(d) Seasonal robustness")
    axis.legend(frameon=False, fontsize=6.2)

    for axis in axes.ravel():
        axis.set_xlabel("Lead week" if axis is not axes[1, 1] else "Season")
        if axis is not axes[1, 1]:
            axis.set_xticks(range(1, 7))
        axis.grid(True, alpha=0.18, axis="y")
        axis.spines[["top", "right"]].set_visible(False)

    prefix = "SMOKE — " if smoke else ""
    figure.suptitle(
        prefix + "All-season probabilistic calibration diagnostics",
        fontsize=10.2,
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Coverage and Brier panels are descriptive; no interval-level bootstrap is claimed.",
        ha="center",
        fontsize=6.8,
    )
    _save_figure(figure, output)


def _block_bootstrap_indices(
    initializations: np.ndarray,
    *,
    n_resamples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Two-stage year/circular-block resampling of initialization units."""

    years = pd.DatetimeIndex(initializations).year.to_numpy()
    groups = [np.flatnonzero(years == year) for year in np.sort(np.unique(years))]
    rng = np.random.default_rng(seed)
    if len({len(group) for group in groups}) != 1:
        # Small smoke caches are diagnostic only; retain grouped case sampling.
        return rng.integers(0, len(initializations), size=(n_resamples, len(initializations)))
    cases_per_year = len(groups[0])
    effective_block = min(block_length, cases_per_year)
    block_count = int(math.ceil(cases_per_year / effective_block))
    offsets = np.arange(effective_block, dtype=np.int64)
    draws = np.empty((n_resamples, len(initializations)), dtype=np.int64)
    for draw in range(n_resamples):
        sampled_years = rng.integers(0, len(groups), size=len(groups))
        cursor = 0
        for year_slot in sampled_years:
            group = groups[int(year_slot)]
            starts = rng.integers(0, cases_per_year, size=block_count)
            local = ((starts[:, None] + offsets[None]) % cases_per_year).reshape(-1)
            chosen = group[local[:cases_per_year]]
            draws[draw, cursor : cursor + cases_per_year] = chosen
            cursor += cases_per_year
    return draws


def paired_block_bootstrap(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    methods: Sequence[str],
    *,
    n_resamples: int,
    block_length: int = 13,
    seed: int = 42,
    baseline: str = "raw_fuxi",
) -> pd.DataFrame:
    draws = _block_bootstrap_indices(
        initializations,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
    )
    case_order = [np.datetime_as_string(value, unit="D") for value in initializations]
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == baseline:
            continue
        for lead_scope, leads in [("W1-W6", tuple(range(1, 7)))] + [
            (f"W{lead}", (lead,)) for lead in range(1, 7)
        ]:
            selected = case_metrics.loc[
                case_metrics.lead_week.isin(leads)
                & case_metrics.method.isin((baseline, method))
            ]
            for metric in ("crps", "rmse", "mae", "acc", "bias"):
                pivot = selected.pivot_table(
                    index="init", columns="method", values=metric, aggfunc="mean"
                ).reindex(case_order)
                if pivot[[baseline, method]].isna().any().any():
                    raise DataContractError("bootstrap comparison is not completely paired")
                reference = pivot[baseline].to_numpy(dtype=np.float64)
                calibrated = pivot[method].to_numpy(dtype=np.float64)
                if metric in {"crps", "rmse", "mae"}:
                    effect_name = f"{metric}_skill_pct_vs_{baseline}"
                    observed = (
                        100.0
                        * (reference.mean() - calibrated.mean())
                        / reference.mean()
                    )
                    sampled_reference = reference[draws].mean(axis=1)
                    sampled_calibrated = calibrated[draws].mean(axis=1)
                    effects = (
                        100.0
                        * (sampled_reference - sampled_calibrated)
                        / sampled_reference
                    )
                else:
                    effect_name = f"delta_{metric}_vs_{baseline}"
                    observed = calibrated.mean() - reference.mean()
                    effects = (
                        calibrated[draws].mean(axis=1)
                        - reference[draws].mean(axis=1)
                    )
                rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "lead_scope": lead_scope,
                        "metric": metric,
                        "effect_name": effect_name,
                        "effect": float(observed),
                        "ci_lower": float(np.percentile(effects, 2.5)),
                        "ci_upper": float(np.percentile(effects, 97.5)),
                        "paired_initializations": len(initializations),
                        "n_resamples": n_resamples,
                        "block_length_initializations": block_length,
                        "seed": seed,
                        "resampling_unit": "initialization; all members and six leads remain grouped",
                        "bootstrap": "year resampling plus circular moving blocks within year",
                    }
                )
    return pd.DataFrame(rows)


def materialize_cases(members: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Use a zero-copy slice when possible, otherwise materialize selected cases."""

    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("cannot materialize an empty case selection")
    if np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
        return members[int(selected[0]) : int(selected[-1]) + 1]
    return np.asarray(members[selected], dtype=np.float32)


def add_seasonal_raw_comparisons(seasonal: pd.DataFrame) -> pd.DataFrame:
    raw = seasonal.loc[seasonal.method == "raw_fuxi"][
        ["season", "lead_week", "crps", "rmse", "mae", "acc", "bias"]
    ].rename(
        columns={
            "crps": "raw_crps",
            "rmse": "raw_rmse",
            "mae": "raw_mae",
            "acc": "raw_acc",
            "bias": "raw_bias",
        }
    )
    result = seasonal.merge(raw, on=["season", "lead_week"], validate="many_to_one")
    result["crps_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.crps, result.raw_crps, strict=True)
    ]
    result["rmse_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.rmse, result.raw_rmse, strict=True)
    ]
    result["mae_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.mae, result.raw_mae, strict=True)
    ]
    result["delta_acc_vs_raw"] = result.acc - result.raw_acc
    result["delta_bias_vs_raw"] = result.bias - result.raw_bias
    return result.drop(columns=["raw_crps", "raw_rmse", "raw_mae", "raw_acc", "raw_bias"])


def summarize_seed_metrics(
    seed_case_metrics: pd.DataFrame,
    raw_weekwise: pd.DataFrame,
) -> pd.DataFrame:
    excluded = {
        "split",
        "method",
        "method_label",
        "seed",
        "init",
        "year",
        "season",
        "lead_week",
        "member_count",
        "support_cells",
    }
    numeric = [
        column for column in seed_case_metrics.columns if column not in excluded
    ]
    result = seed_case_metrics.groupby(
        ["method", "method_label", "seed", "lead_week"], as_index=False
    )[numeric].mean()
    result = recompute_grouped_spread_skill(result)
    counts = seed_case_metrics.groupby(
        ["method", "seed", "lead_week"], as_index=False
    ).agg(
        n_initializations=("init", "nunique"),
        n_valid_cells=("support_cells", "first"),
    )
    result = result.merge(
        counts, on=["method", "seed", "lead_week"], validate="one_to_one"
    )
    result.insert(0, "split", "test_development")
    result["week"] = result.lead_week
    raw = raw_weekwise[
        ["lead_week", "crps", "rmse", "mae", "acc", "bias"]
    ].rename(
        columns={
            "crps": "raw_crps",
            "rmse": "raw_rmse",
            "mae": "raw_mae",
            "acc": "raw_acc",
            "bias": "raw_bias",
        }
    )
    result = result.merge(raw, on="lead_week", validate="many_to_one")
    result["crpss_vs_raw"] = [
        _safe_skill(value, reference)
        for value, reference in zip(result.crps, result.raw_crps, strict=True)
    ]
    result["crps_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.crps, result.raw_crps, strict=True)
    ]
    result["rmse_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.rmse, result.raw_rmse, strict=True)
    ]
    result["mae_skill_pct_vs_raw"] = [
        100.0 * _safe_skill(value, reference)
        for value, reference in zip(result.mae, result.raw_mae, strict=True)
    ]
    result["delta_acc_vs_raw"] = result.acc - result.raw_acc
    result["delta_bias_vs_raw"] = result.bias - result.raw_bias
    return result.drop(columns=["raw_crps", "raw_rmse", "raw_mae", "raw_acc", "raw_bias"])


def summarize_seed_variability(seed_weekwise: pd.DataFrame) -> pd.DataFrame:
    """Report optimization variability without treating seeds as weather samples."""

    keys = [
        "split",
        "method",
        "method_label",
        "lead_week",
        "week",
        "n_initializations",
        "n_valid_cells",
    ]
    excluded = {*keys, "seed"}
    metrics = [
        column
        for column in seed_weekwise.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(seed_weekwise[column])
    ]
    grouped = seed_weekwise.groupby(keys, sort=True)
    means = grouped[metrics].mean().add_suffix("_mean")
    standard_deviations = grouped[metrics].std(ddof=0).add_suffix("_std")
    minima = grouped[metrics].min().add_suffix("_min")
    maxima = grouped[metrics].max().add_suffix("_max")
    result = pd.concat((means, standard_deviations, minima, maxima), axis=1).reset_index()
    result["seed_count"] = grouped.seed.nunique().to_numpy()
    result["aggregation"] = "mean/std/min/max across optimization seeds; not weather samples"
    return result


def mean_seed_case_metrics(
    seed_metrics: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Average scores across fits without constructing an undeclared model ensemble."""

    keys = [
        "split",
        "method",
        "method_label",
        "init",
        "year",
        "season",
        "lead_week",
        "member_count",
        "support_cells",
    ]
    expected_seeds = {int(seed) for seed in seeds}
    observed_seeds = {int(seed) for seed in seed_metrics.seed.unique()}
    if observed_seeds != expected_seeds:
        raise DataContractError(
            f"seed metrics contain {sorted(observed_seeds)}, expected {sorted(expected_seeds)}"
        )
    group_counts = seed_metrics.groupby(keys).agg(
        rows=("seed", "size"), unique_seeds=("seed", "nunique")
    )
    if not (
        (group_counts.rows == len(expected_seeds))
        & (group_counts.unique_seeds == len(expected_seeds))
    ).all():
        raise DataContractError("seed metric rows are missing or duplicated for one or more cases")
    numeric = [column for column in seed_metrics.columns if column not in {*keys, "seed"}]
    result = seed_metrics.groupby(keys, as_index=False)[numeric].mean()
    result.insert(
        3,
        "seed",
        "mean_of_seed_metrics_" + "_".join(str(seed) for seed in seeds),
    )
    return recompute_grouped_spread_skill(result)


def mean_seed_rank_histograms(seed_ranks: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "lead_week", "rank", "weighting"]
    return seed_ranks.groupby(keys, as_index=False)["count"].mean()


def mean_seed_reliability_bins(seed_bins: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method",
        "method_label",
        "lead_week",
        "threshold_mm_day",
        "probability_bin",
        "bin_lower",
        "bin_upper",
    ]
    averaged = seed_bins.groupby(keys, as_index=False)[
        [
            "cell_case_count",
            "area_weight_sum",
            "forecast_probability_weighted_sum",
            "observed_event_weighted_sum",
        ]
    ].mean()
    denominator = averaged.area_weight_sum.to_numpy(dtype=np.float64)
    probability = np.full(denominator.shape, np.nan, dtype=np.float64)
    observed = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(
        averaged.forecast_probability_weighted_sum,
        denominator,
        out=probability,
        where=denominator > 0.0,
    )
    np.divide(
        averaged.observed_event_weighted_sum,
        denominator,
        out=observed,
        where=denominator > 0.0,
    )
    averaged["mean_forecast_probability"] = probability
    averaged["observed_frequency"] = observed
    return averaged


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_ensemble_calibration.py": Path(__file__).resolve(),
        "src/fuxi_ensemble_calibration_core.py": HERE
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": HERE / "src/fuxi_allseason_member_cache.py",
        "slurm/run_fuxi_allseason_ensemble_calibration.sbatch": HERE
        / "slurm/run_fuxi_allseason_ensemble_calibration.sbatch",
        "slurm/build_fuxi_allseason_member_cache.sbatch": HERE
        / "slurm/build_fuxi_allseason_member_cache.sbatch",
        "plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md": HERE
        / "plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md",
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            continue
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        relative = str(path.relative_to(output))
        if relative in {"manifest.json", "failure.json"}:
            continue
        checksums[relative] = sha256_file(path)
    return checksums


def cache_provenance(cache: MemberCache) -> dict[str, Any]:
    metadata: Mapping[str, Any] = {}
    if cache.metadata_path.suffix == ".json":
        metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    data_sha = metadata.get("data_sha256")
    if not data_sha:
        checksum_path = cache.members_path.with_suffix("").with_suffix(".sha256")
        if checksum_path.is_file():
            data_sha = checksum_path.read_text(encoding="utf-8").split()[0]
    return {
        "data_file": str(cache.members_path),
        "metadata_file": str(cache.metadata_path),
        "manifest_file": None if cache.manifest_path is None else str(cache.manifest_path),
        "data_sha256": data_sha,
        "metadata_sha256": sha256_file(cache.metadata_path),
        "manifest_sha256": (
            None if cache.manifest_path is None else sha256_file(cache.manifest_path)
        ),
        "source_store": metadata.get("source_store", str(NATIVE_SOURCE)),
        "source_fingerprint": metadata.get("source_fingerprint"),
        "scope": metadata.get("scope"),
        "source_init_indices": metadata.get("source_init_indices"),
    }


def build_readme(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    smoke: bool,
    test_count: int,
    deployable_seed: int,
) -> str:
    status = (
        "NON-SCIENTIFIC GPU SMOKE TEST"
        if smoke
        else "RETROSPECTIVE 2020–2021 DEVELOPMENT TEST (not an untouched final test)"
    )
    lines = [
        "# FuXi all-season ensemble calibration — corrected alignment v2",
        "",
        f"Status: **{status}**",
        "",
        "This run keeps all 51 FuXi members and all seasons. One initialization is the "
        "statistical unit: its 51 member trajectories and six lead weeks are never split "
        "across partitions or bootstrap draws.",
        "",
        "Training uses 2002–2017; checkpoint selection uses 2018–2019; reporting uses "
        f"{test_count} 2020–2021 development initializations. Starts whose 42-day "
        "verification window crosses the 2018 or 2020 boundary are purged. The 2025 "
        "control is not opened.",
        "",
        "FuXi native lead days 1–42 pair with end-labelled IMD days initialization+1 "
        "through initialization+42. No v1 zero-offset checkpoint or fitted product is loaded.",
        "",
        "The fixed `location_spread` architecture is trained by area-weighted ensemble CRPS. "
        f"Seed {deployable_seed} is the single deployable forecast selected using 2018–2019 "
        "validation CRPS only. Other seeds are retained as sensitivity evidence; parameters, "
        "predictions, and headline scores are not averaged across seeds. Raw FuXi and a newly "
        "fitted training-only moment calibration are the baselines.",
        "",
        "## Pooled headline metrics",
        "",
        "| Method | CRPS | CRPSS vs raw | RMSE | MAE | Bias | ACC | RMS spread / pooled RMS error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.crps:.4f} | {row.crpss_vs_raw:+.3f} | "
            f"{row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.4f} | {row.acc:.4f} | "
            f"{row.spread_skill_ratio:.3f} |"
        )
    if not smoke and not bootstrap.empty:
        primary = bootstrap.loc[
            (bootstrap.method == PRIMARY_CONFIGURATION)
            & (bootstrap.lead_scope == "W1-W6")
        ]
        lines.extend(
            [
                "",
                "## Primary paired uncertainty",
                "",
            ]
        )
        for row in primary.itertuples(index=False):
            lines.append(
                f"- `{row.effect_name}`: {row.effect:+.3f} "
                f"(95% block-bootstrap CI {row.ci_lower:+.3f} to {row.ci_upper:+.3f})."
            )
    lines.extend(
        [
            "",
            "## Artifact map",
            "",
            "- `history/training_history.csv`: epoch-zero, train, and validation CRPS curves.",
            "- `metrics/case_metrics.csv`: one row per method, initialization, and lead.",
            "- `metrics/weekwise_metrics.csv`: all weekwise deterministic and probabilistic metrics.",
            "- `metrics/seed_weekwise_metrics.csv`: per-seed neural metrics and raw-relative skill.",
            "- `metrics/seed_variability_by_week.csv`: mean/std/min/max across optimization seeds.",
            "- `metrics/seed_rank_histograms.csv` and `seed_reliability_bins.csv`: per-seed diagnostics.",
            "- `metrics/matrices/`: one method × W1–W6 CSV for every reported metric.",
            "- `metrics/seasonal_weekwise_metrics.csv`: DJF/MAM/JJA/SON robustness table.",
            "- `metrics/paired_block_bootstrap.csv`: dependence-aware paired uncertainty.",
            "- `models/deployable_selection.json`: validation-only seed selection and checkpoint hashes.",
            "- `figures/`: loss curves, weekwise comparison, skill heatmaps, and ranks.",
            "- `evaluation/scoring_support.npz`: exact support mask and scoring weights.",
            "- `manifest.json`: frozen contract, provenance, run settings, and artifact hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_names(value: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(names) - set(allowed))
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(f"invalid {label}: names={names}, unknown={unknown}")
    return names


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique comma-separated list")
    return seeds


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    return device


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    # Freeze the executable sources before any long data load or optimization.
    # The final manifest hashes this startup snapshot, not a potentially edited
    # working tree observed hours later.
    snapshot_checksums = source_snapshot(output)
    configurations = _parse_names(args.configs, CONFIGURATIONS, "configurations")
    seeds = _parse_seeds(args.seeds)
    if args.smoke:
        seeds = seeds[:1]
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading member cache {args.cache}...", flush=True)
    cache = load_member_cache(Path(args.cache), allow_partial=args.smoke)
    splits = make_split_indices(cache.initializations)
    split_counts = {name: len(indices) for name, indices in splits.as_dict().items()}
    if args.smoke and cache.members.shape[0] == 64:
        expected_smoke = {"train": 32, "validation": 16, "test": 16, "embargo": 0}
        if split_counts != expected_smoke:
            raise DataContractError(
                f"stratified smoke-cache splits {split_counts}, expected {expected_smoke}"
            )
    train_indices = splits.train
    validation_indices = splits.validation
    test_indices = splits.test
    if args.smoke:
        train_indices = select_evenly(train_indices, 32)
        validation_indices = select_evenly(validation_indices, 16)
        test_indices = select_evenly(test_indices, 16)
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise DataContractError("train, validation, and test must all be non-empty")
    print(
        f"Effective cases: train={len(train_indices)}, validation={len(validation_indices)}, "
        f"test={len(test_indices)}, embargoed={len(splits.embargo)}",
        flush=True,
    )
    print("Loading IMD 2002–2022 and fitting the training-only climatology...", flush=True)
    observations = load_imd_observations(cache)
    scoring_directory = output / "evaluation"
    scoring_directory.mkdir(parents=True, exist_ok=True)
    scoring_support_path = scoring_directory / "scoring_support.npz"
    normalized_weights = observations.weights / observations.weights.sum(dtype=np.float64)
    np.savez_compressed(
        scoring_support_path,
        latitude=cache.latitude.astype(np.float64),
        longitude=cache.longitude.astype(np.float64),
        observation_fraction=observations.observation_fraction.astype(np.float32),
        support_mask=(observations.weights > 0.0),
        scoring_weight_km2_fraction=observations.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights.astype(np.float64),
    )
    scoring_support_sha256 = sha256_file(scoring_support_path)
    context = build_context_bundle(cache, observations, train_indices)
    test_initializations = cache.initializations[test_indices]
    test_truth = observations.weekly_truth[test_indices]
    test_climatology = observations.weekly_climatology[test_indices]
    test_members = materialize_cases(cache.members, test_indices)
    if test_members.shape[0] != len(test_indices):
        raise DataContractError("test member selection is not aligned")
    method_order = ("raw_fuxi", "moment_calibration", *configurations)
    metric_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    seed_metric_frames: list[pd.DataFrame] = []
    seed_rank_frames: list[pd.DataFrame] = []
    seed_reliability_bin_frames: list[pd.DataFrame] = []
    reliability_bin_frames: list[pd.DataFrame] = []
    print("Evaluating raw FuXi...", flush=True)
    raw_metrics, raw_ranks = evaluate_ensemble(
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
    reliability_bin_frames.append(
        reliability_bins(
            "raw_fuxi",
            test_members,
            test_truth,
            observations.weights,
            chunk_size=args.evaluation_batch_size,
        )
    )
    print("Fitting and evaluating the train-only moment calibration...", flush=True)
    moment_fit = fit_moment_calibration(
        cache.members,
        observations.weekly_truth,
        cache.initializations,
        train_indices,
        observations.weights,
        shrinkage=10.0,
    )
    moment_members = apply_moment_fit(test_members, test_initializations, moment_fit)
    moment_metrics, moment_ranks = evaluate_ensemble(
        "moment_calibration",
        moment_members,
        test_truth,
        test_climatology,
        test_initializations,
        observations.weights,
        chunk_size=args.evaluation_batch_size,
    )
    metric_frames.append(moment_metrics)
    rank_frames.append(moment_ranks)
    reliability_bin_frames.append(
        reliability_bins(
            "moment_calibration",
            moment_members,
            test_truth,
            observations.weights,
            chunk_size=args.evaluation_batch_size,
        )
    )
    del moment_members
    (output / "models").mkdir(parents=True, exist_ok=True)
    histories: list[pd.DataFrame] = []
    training_runs: list[TrainingRun] = []
    deployable_run: TrainingRun | None = None
    max_epochs = min(args.max_epochs, 2) if args.smoke else args.max_epochs
    patience = min(args.patience, 1) if args.smoke else args.patience
    for configuration in configurations:
        configuration_seed_metrics: dict[int, pd.DataFrame] = {}
        configuration_seed_ranks: dict[int, pd.DataFrame] = {}
        configuration_seed_bins: dict[int, pd.DataFrame] = {}
        configuration_runs: list[TrainingRun] = []
        for seed in seeds:
            print(f"Training {configuration}, seed {seed}...", flush=True)
            run_directory = output / "models" / configuration / f"seed_{seed}"
            model, history, record = train_one_model(
                configuration,
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
                max_epochs=max_epochs,
                patience=patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                member_subsample=args.member_subsample,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            histories.append(history)
            training_runs.append(record)
            configuration_runs.append(record)
            delta, log_scale = predict_adjustments(
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
            seed_scale = np.exp(np.clip(log_scale, -2.0, 2.0)).astype(np.float32)
            np.savez_compressed(
                run_directory / "test_adjustments.npz",
                initializations=test_initializations,
                delta_log_location=delta,
                log_spread=log_scale,
                spread_factor=seed_scale,
                seed=np.int64(seed),
            )
            seed_corrected = apply_affine_log_calibration(
                test_members, delta, seed_scale
            )
            seed_metrics, seed_ranks = evaluate_ensemble(
                configuration,
                seed_corrected,
                test_truth,
                test_climatology,
                test_initializations,
                observations.weights,
                chunk_size=args.evaluation_batch_size,
                seed_label=seed,
            )
            seed_ranks.insert(1, "seed", seed)
            seed_bins = reliability_bins(
                configuration,
                seed_corrected,
                test_truth,
                observations.weights,
                chunk_size=args.evaluation_batch_size,
            )
            seed_bins.insert(2, "seed", seed)
            seed_metric_frames.append(seed_metrics)
            seed_rank_frames.append(seed_ranks)
            seed_reliability_bin_frames.append(seed_bins)
            configuration_seed_metrics[seed] = seed_metrics
            configuration_seed_ranks[seed] = seed_ranks
            configuration_seed_bins[seed] = seed_bins
            del seed_corrected, seed_scale, seed_metrics, seed_ranks, seed_bins
            del model, delta, log_scale
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selected = select_deployable_run(
            configuration_runs, configuration=configuration
        )
        if configuration == PRIMARY_CONFIGURATION:
            deployable_run = selected
        metrics = configuration_seed_metrics[selected.seed].copy()
        ranks = configuration_seed_ranks[selected.seed].drop(columns="seed").copy()
        bins = configuration_seed_bins[selected.seed].drop(columns="seed").copy()
        metric_frames.append(metrics)
        rank_frames.append(ranks)
        reliability_bin_frames.append(
            bins
        )
        del configuration_seed_metrics, configuration_seed_ranks
        del configuration_seed_bins, configuration_runs
    if deployable_run is None:
        raise DataContractError("no deployable primary checkpoint was selected")
    history_frame = pd.concat(histories, ignore_index=True)
    case_metrics = pd.concat(metric_frames, ignore_index=True)
    rank_histograms = pd.concat(rank_frames, ignore_index=True)
    seed_case_metrics = pd.concat(seed_metric_frames, ignore_index=True)
    seed_rank_histograms = pd.concat(seed_rank_frames, ignore_index=True)
    seed_reliability_bins = pd.concat(seed_reliability_bin_frames, ignore_index=True)
    reliability_bin_frame = pd.concat(reliability_bin_frames, ignore_index=True)
    expected_rows = len(method_order) * len(test_indices) * 6
    if len(case_metrics) != expected_rows:
        raise DataContractError(f"expected {expected_rows} metric rows, found {len(case_metrics)}")
    weekwise, pooled, seasonal, reliability = summarize_metrics(case_metrics)
    seed_weekwise = summarize_seed_metrics(
        seed_case_metrics,
        weekwise.loc[weekwise.method == "raw_fuxi"],
    )
    seed_variability = summarize_seed_variability(seed_weekwise)
    seasonal = add_seasonal_raw_comparisons(seasonal)
    bootstrap_samples = min(args.bootstrap_samples, 100) if args.smoke else args.bootstrap_samples
    bootstrap = paired_block_bootstrap(
        case_metrics,
        test_initializations,
        method_order,
        n_resamples=bootstrap_samples,
    )
    if {PRIMARY_CONFIGURATION, "summary_only"}.issubset(method_order):
        primary_vs_summary_bootstrap = paired_block_bootstrap(
            case_metrics,
            test_initializations,
            (PRIMARY_CONFIGURATION,),
            n_resamples=bootstrap_samples,
            baseline="summary_only",
        )
    else:
        primary_vs_summary_bootstrap = pd.DataFrame(columns=bootstrap.columns)
    history_directory = output / "history"
    metrics_directory = output / "metrics"
    figures_directory = output / "figures"
    history_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(history_directory / "training_history.csv", index=False)
    case_metrics.to_csv(metrics_directory / "case_metrics.csv", index=False)
    seed_case_metrics.to_csv(metrics_directory / "seed_case_metrics.csv", index=False)
    weekwise.to_csv(metrics_directory / "weekwise_metrics.csv", index=False)
    seed_weekwise.to_csv(metrics_directory / "seed_weekwise_metrics.csv", index=False)
    seed_variability.to_csv(metrics_directory / "seed_variability_by_week.csv", index=False)
    pooled.to_csv(metrics_directory / "pooled_metrics.csv", index=False)
    seasonal.to_csv(metrics_directory / "seasonal_weekwise_metrics.csv", index=False)
    reliability.to_csv(metrics_directory / "threshold_reliability_by_week.csv", index=False)
    reliability_bin_frame.to_csv(metrics_directory / "reliability_bins.csv", index=False)
    seed_reliability_bins.to_csv(
        metrics_directory / "seed_reliability_bins.csv", index=False
    )
    rank_histograms.to_csv(metrics_directory / "rank_histograms.csv", index=False)
    seed_rank_histograms.to_csv(
        metrics_directory / "seed_rank_histograms.csv", index=False
    )
    bootstrap.to_csv(metrics_directory / "paired_block_bootstrap.csv", index=False)
    primary_vs_summary_bootstrap.to_csv(
        metrics_directory / "primary_vs_summary_block_bootstrap.csv",
        index=False,
    )
    np.savez_compressed(
        output / "models" / "moment_calibration_fit.npz",
        delta_log_location=moment_fit.delta_log_location,
        spread_factor=moment_fit.spread_factor,
        shrinkage=np.float32(moment_fit.shrinkage),
    )
    write_metric_matrices(weekwise, metrics_directory / "matrices")
    for season in ("DJF", "MAM", "JJA", "SON"):
        write_metric_matrices(
            seasonal.loc[seasonal.season == season].drop(columns="season"),
            metrics_directory / "seasonal_matrices" / season,
        )
    plot_training_loss(
        history_frame, figures_directory / "training_loss_curves", smoke=args.smoke
    )
    plot_weekwise_metrics(
        weekwise,
        figures_directory / "weekwise_metrics",
        method_order,
        smoke=args.smoke,
        bootstrap=bootstrap,
        seed_variability=seed_variability,
    )
    plot_skill_heatmaps(
        weekwise,
        figures_directory / "weekwise_ablation_heatmaps",
        method_order,
        smoke=args.smoke,
    )
    primary = PRIMARY_CONFIGURATION if PRIMARY_CONFIGURATION in configurations else configurations[-1]
    plot_rank_histograms(
        rank_histograms,
        figures_directory / "rank_histograms_raw_vs_primary",
        primary,
        smoke=args.smoke,
    )
    plot_reliability(
        reliability_bin_frame,
        figures_directory / "reliability_diagrams",
        method_order,
        smoke=args.smoke,
    )
    plot_probabilistic_diagnostics(
        weekwise,
        seasonal,
        figures_directory / "probabilistic_diagnostics",
        method_order,
        smoke=args.smoke,
    )
    (output / "README.md").write_text(
        build_readme(
            pooled,
            bootstrap,
            smoke=args.smoke,
            test_count=len(test_indices),
            deployable_seed=deployable_run.seed,
        ),
        encoding="utf-8",
    )
    run_records = []
    for record in training_runs:
        values = asdict(record)
        values["checkpoint_sha256"] = sha256_file(Path(record.checkpoint))
        values["checkpoint"] = str(Path(record.checkpoint).relative_to(output))
        run_records.append(values)
    selected_checkpoint = Path(deployable_run.checkpoint)
    deployment_selection = {
        "configuration": PRIMARY_CONFIGURATION,
        "criterion": "minimum 2018-2019 validation CRPS; exact ties use lowest seed",
        "selected_seed": deployable_run.seed,
        "best_validation_crps": deployable_run.best_validation_crps,
        "checkpoint": str(selected_checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "test_metrics_used_for_selection": False,
        "prediction_averaging": False,
        "parameter_averaging": False,
        "candidates": [
            {
                "seed": record.seed,
                "best_validation_crps": record.best_validation_crps,
                "checkpoint": str(Path(record.checkpoint).relative_to(output)),
                "checkpoint_sha256": sha256_file(Path(record.checkpoint)),
            }
            for record in sorted(
                (
                    item
                    for item in training_runs
                    if item.configuration == PRIMARY_CONFIGURATION
                ),
                key=lambda item: item.seed,
            )
        ],
    }
    write_json(output / "models" / "deployable_selection.json", deployment_selection)
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "retrospective 2020-2021 development test; not an untouched final test"
        ),
        "created_utc": utc_now(),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "contract": {
            "revision": CONTRACT_REVISION,
            "forecast": "FuXi native reforecast weekly TP; all seasons; 51 members",
            "region": "39N-0N, 60E-99E, 27x27 India box",
            "target": "IMD weekly mean precipitation, mm day-1",
            "weekly_alignment": "FuXi lead_day 1..42; W1 init+1..+7 through W6 init+36..+42",
            "target_day_offsets": list(TARGET_DAY_OFFSETS),
            "initialization_day_included_in_target": False,
            "legacy_v1_zero_offset_checkpoint_loaded": False,
            "train_years": list(TRAIN_YEARS),
            "validation_years": list(VALIDATION_YEARS),
            "test_years": list(TEST_YEARS),
            "sealed_unopened_years": list(SEALED_YEARS),
            "outcome_window_days": 42,
            "embargo": "purge train/validation starts whose 42-day outcome crosses the next boundary",
            "statistical_unit": "initialization with all members and all six lead weeks grouped",
            "primary_configuration": PRIMARY_CONFIGURATION,
            "primary_loss": "area-weighted empirical ensemble CRPS",
            "acc_reference": "2002-2017 equal-year centred 31-day IMD daily climatology, weekly aggregated",
            "test_reuse_warning": "2020-2021 has prior development exposure and is not independent",
            "all_reported_ensemble_metrics_use_member_count": 51,
            "sealed_2025_target_opened": False,
        },
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
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
            "test": [
                np.datetime_as_string(value, unit="D")
                for value in cache.initializations[test_indices]
            ],
            "embargo": [
                np.datetime_as_string(value, unit="D")
                for value in cache.initializations[splits.embargo]
            ],
        },
        "configurations": list(configurations),
        "seeds": list(seeds),
        "seed_aggregation": {
            "role": "optimization variability only; seeds are not ensemble members or independent weather samples",
            "headline_case_metrics": "single validation-selected deployable checkpoint",
            "spread_skill_summary": "single validation-selected deployable checkpoint",
            "rank_histograms": "single validation-selected deployable checkpoint",
            "reliability_bins": "single validation-selected deployable checkpoint",
            "parameter_averaging": False,
            "prediction_averaging": False,
            "per_seed_artifacts": [
                "metrics/seed_case_metrics.csv",
                "metrics/seed_weekwise_metrics.csv",
                "metrics/seed_variability_by_week.csv",
                "metrics/seed_rank_histograms.csv",
                "metrics/seed_reliability_bins.csv",
            ],
        },
        "deployment_selection": deployment_selection,
        "methods": list(method_order),
        "training": {
            "batch_size": args.batch_size,
            "member_subsample": args.member_subsample,
            "full_members_for_validation_and_evaluation": 51,
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "automatic_mixed_precision": not args.no_amp and device.type == "cuda",
            "device": str(device),
            "objective": "area-weighted empirical finite-ensemble CRPS",
            "model": {
                "class": "EnsembleLocationSpreadCalibrator",
                "context_channels": 7,
                "member_hidden_channels": 8,
                "backbone_channels": 24,
                "dropout": 0.05,
                "max_abs_log_spread": 2.0,
                "permutation_invariant_member_axis": True,
            },
            "optimizer": {
                "class": "torch.optim.AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience": max(2, patience // 3),
                "minimum_learning_rate": 1.0e-6,
            },
            "early_stopping": {
                "metric": "full-51-member validation area-weighted CRPS",
                "patience": patience,
                "minimum_improvement": 1.0e-6,
                "restore_best_checkpoint": True,
            },
            "gradient_clip_max_norm": 5.0,
            "runs": run_records,
        },
        "evaluation": {
            "thresholds_mm_day": list(THRESHOLDS_MM_DAY),
            "central_coverages": list(COVERAGES),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_block_length_initializations": 13,
            "support_cells": int(np.count_nonzero(observations.weights > 0)),
            "scoring_support_artifact": "evaluation/scoring_support.npz",
            "scoring_support_sha256": scoring_support_sha256,
            "support_mask_definition": "positive india_area_weight_km2 times IMD observation_fraction",
            "spatial_area_source": str(SPATIAL_STORE),
            "scoring_weight_sum": float(observations.weights.sum(dtype=np.float64)),
            "area_weighting": "india_area_weight_km2 x IMD observation_fraction; normalized by its sum in every spatial score",
            "seasons": ["DJF", "MAM", "JJA", "SON"],
        },
        "cache": cache_provenance(cache),
        "observation_stores": list(observations.source_stores),
        "normalization": {
            "climatology_log1p_mean_by_lead": context.climatology_mean_by_lead.tolist(),
            "climatology_log1p_std_by_lead": context.climatology_std_by_lead.tolist(),
            "fit_indices": train_indices.tolist(),
        },
        "moment_calibration": {
            "equation": "u'_m = u_bar + delta_lead_month_grid + s_lead_month*(u_m-u_bar)",
            "location_shrinkage": moment_fit.shrinkage,
            "spread_clip": [0.25, 4.0],
            "fit_scope": "effective training split only",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "xarray": xr.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "zarr": importlib.metadata.version("zarr"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
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
        description="Train/evaluate the all-season probabilistic FuXi ensemble calibrator."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--seeds",
        default=None,
        help="default: 42 for --smoke, otherwise 42,43,44",
    )
    parser.add_argument("--configs", default=",".join(CONFIGURATIONS))
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="default: 2 for --smoke, otherwise 100"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="default: 1 for --smoke, otherwise 15"
    )
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
    positive_integer_names = (
        "max_epochs",
        "patience",
        "batch_size",
        "member_subsample",
        "evaluation_batch_size",
        "bootstrap_samples",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.member_subsample > 51:
        raise ValueError("--member-subsample cannot exceed 51")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    configurations = _parse_names(args.configs, CONFIGURATIONS, "configurations")
    seeds = _parse_seeds(args.seeds)
    if configurations != CONFIGURATIONS:
        raise ValueError(
            f"canonical {EXPERIMENT} requires configurations {CONFIGURATIONS} in order"
        )
    expected_seeds = (42,) if args.smoke else SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"canonical {'smoke' if args.smoke else 'full'} run requires seeds {expected_seeds}"
        )
    expected_epochs = 2 if args.smoke else 100
    expected_patience = 1 if args.smoke else 15
    fixed = {
        "max_epochs": (args.max_epochs, expected_epochs),
        "patience": (args.patience, expected_patience),
        "batch_size": (args.batch_size, 8),
        "member_subsample": (args.member_subsample, 16),
        "bootstrap_samples": (args.bootstrap_samples, 2000),
    }
    mismatched = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatched:
        raise ValueError(f"canonical run settings differ: {mismatched}")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --learning-rate 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --weight-decay 0.0001")
    if args.no_amp:
        raise ValueError("canonical GPU run requires automatic mixed precision")
    if args.device == "cpu":
        raise ValueError("canonical run requires a CUDA device")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested_output = (default_output() if args.output is None else Path(args.output)).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output; choose a fresh path: {requested_output}"
        )
    staging = requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
        os.replace(staging, requested_output)
    except Exception as error:
        failure = {
            "experiment": EXPERIMENT,
            "status": "failed",
            "started_utc": started,
            "failed_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "requested_output": str(requested_output),
        }
        write_json(staging / "failure.json", failure)
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(f"PASS: completed {args.smoke and 'smoke' or 'full'} run at {requested_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
