#!/usr/bin/env python
"""Build a leakage-safe weekly FuXi physical-predictor cache.

The source archive is globally chunked, so extracting the requested variables
touches every channel chunk group.  This module deliberately builds one
initialization per atomic part file and exposes a strided task interface for a
Slurm array.  A separate finalization pass validates all 630 JJAS 2002--2019
parts before atomically publishing the compact NPZ used by model screening.

No normalization, climatology, target data, 2020--2021 forecasts, or 2025 data
are included in this cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


from project_paths import PROJECT_ROOT as HERE
DEFAULT_SOURCE_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
DEFAULT_PHYSICAL_CACHE = HERE / "cache" / "fuxi_physical_weekly_2002_2019_jjas_v1.npz"
DEFAULT_PARTS_DIR = HERE / "cache" / "fuxi_physical_weekly_2002_2019_jjas_v1.parts"

CACHE_SCHEMA_NAME = "fuxi-physical-weekly-jjas"
CACHE_SCHEMA_VERSION = 1
SCREENING_YEARS = tuple(range(2002, 2020))
SCREENING_MONTHS = (6, 7, 8, 9)
EXPECTED_INITIALIZATIONS_PER_YEAR = 35
EXPECTED_INITIALIZATION_COUNT = len(SCREENING_YEARS) * EXPECTED_INITIALIZATIONS_PER_YEAR
MEMBER_COUNT = 51
LEAD_DAY_COUNT = 42
LEAD_WEEK_COUNT = 6
DAYS_PER_WEEK = 7
SOURCE_CHANNEL_CHUNK = 4

SOURCE_CHANNEL_NAMES = (
    "z850",
    "z500",
    "z250",
    "t850",
    "t500",
    "t250",
    "u850",
    "u500",
    "u250",
    "v850",
    "v500",
    "v250",
    "q850",
    "q500",
    "q250",
    "t2m",
    "d2m",
    "sst",
    "ttr",
    "10u",
    "10v",
    "100u",
    "100v",
    "msl",
    "tcwv",
    "tp",
)
RAW_CHANNELS = ("tcwv", "q850", "u850", "v850", "z500", "msl", "ttr")
PHYSICAL_FEATURE_NAMES = (
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
PHYSICAL_FEATURE_DEFINITIONS = {
    "tcwv_mean": (
        "Mean across 51 member-wise seven-day means of FuXi TCWV after "
        "clipping native values to >=0"
    ),
    "tcwv_spread": (
        "Population standard deviation across 51 member-wise seven-day "
        "means of FuXi TCWV after clipping native values to >=0"
    ),
    "q850_mean": (
        "Mean across 51 member-wise seven-day means of FuXi q850 after "
        "clipping native values to >=0"
    ),
    "u850_mean": ("Mean across 51 member-wise seven-day means of native FuXi u850"),
    "v850_mean": ("Mean across 51 member-wise seven-day means of native FuXi v850"),
    "z500_mean": ("Mean across 51 member-wise seven-day means of native FuXi z500"),
    "msl_mean": ("Mean across 51 member-wise seven-day means of native FuXi MSL"),
    "olr_mean": (
        "Positive outgoing longwave radiation: negative of the mean across "
        "51 member-wise seven-day means of native FuXi TTR"
    ),
    "q850_u850_flux_mean": (
        "Mean across seven days and 51 members of the daily per-member "
        "FuXi product max(q850,0)*u850"
    ),
    "q850_v850_flux_mean": (
        "Mean across seven days and 51 members of the daily per-member "
        "FuXi product max(q850,0)*v850"
    ),
}
PHYSICAL_FEATURE_UNITS = {
    "tcwv_mean": "kg m-2",
    "tcwv_spread": "kg m-2",
    "q850_mean": "kg kg-1",
    "u850_mean": "m s-1",
    "v850_mean": "m s-1",
    "z500_mean": "m2 s-2",
    "msl_mean": "Pa",
    "olr_mean": "W m-2",
    "q850_u850_flux_mean": "kg kg-1 m s-1",
    "q850_v850_flux_mean": "kg kg-1 m s-1",
}
PHYSICAL_TRANSFORMS = {
    "tcwv": "maximum(native_tcwv, 0) before weekly statistics",
    "q850": (
        "maximum(native_q850, 0) before weekly mean and daily moisture-flux " "products"
    ),
    "ttr": "olr=-native_ttr",
    "other_fields": "native float32 values; no unit conversion",
}

EXPECTED_LATITUDE = np.arange(39.0, -0.01, -1.5, dtype=np.float64)
EXPECTED_LONGITUDE = np.arange(60.0, 99.01, 1.5, dtype=np.float64)
EXPECTED_GRID_SHAPE = (27, 27)


class PhysicalCacheContractError(ValueError):
    """Raised when the source, an atomic part, or the final cache is unsafe."""


@dataclass(frozen=True)
class SourceContract:
    """Validated source metadata and the exact screening subset."""

    source_store: str
    source_fingerprint: str
    all_initializations: np.ndarray
    selected_source_indices: np.ndarray
    selected_initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    latitude_slice: slice
    longitude_slice: slice
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class FuxiPhysicalPredictors:
    """Raw weekly physical predictors on the 27x27 FuXi India context grid."""

    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    tcwv_mean: np.ndarray
    tcwv_spread: np.ndarray
    q850_mean: np.ndarray
    u850_mean: np.ndarray
    v850_mean: np.ndarray
    z500_mean: np.ndarray
    msl_mean: np.ndarray
    olr_mean: np.ndarray
    q850_u850_flux_mean: np.ndarray
    q850_v850_flux_mean: np.ndarray
    source_fingerprint: str
    source_store: str
    cache_path: str | None = None
    cache_sha256: str | None = None

    @property
    def feature_fields(self) -> Mapping[str, np.ndarray]:
        """Return predictors in the frozen model-channel order."""

        return {name: getattr(self, name) for name in PHYSICAL_FEATURE_NAMES}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum for one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an NPZ in the target directory and publish it with os.replace."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".temporary",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _datetime_days(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")
    if not np.issubdtype(values.dtype, np.integer):
        raise PhysicalCacheContractError(
            f"unsupported initialization coordinate dtype: {values.dtype}"
        )
    return values.astype(np.int64).astype("datetime64[ns]").astype("datetime64[D]")


def _date_parts(initializations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stamps = np.datetime_as_string(
        np.asarray(initializations, dtype="datetime64[D]"), unit="D"
    )
    years = np.asarray([int(value[:4]) for value in stamps], dtype=np.int16)
    months = np.asarray([int(value[5:7]) for value in stamps], dtype=np.int8)
    return years, months


def _select_screening_initializations(
    all_initializations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and validate exactly 35 June--September inits in every year."""

    all_initializations = np.asarray(all_initializations, dtype="datetime64[D]")
    if all_initializations.ndim != 1:
        raise PhysicalCacheContractError("source initialization coordinate is not 1-D")
    if np.unique(all_initializations).size != len(all_initializations):
        raise PhysicalCacheContractError("source initializations are not unique")
    if np.any(all_initializations[1:] <= all_initializations[:-1]):
        raise PhysicalCacheContractError("source initializations are not increasing")
    years, months = _date_parts(all_initializations)
    selected = np.isin(years, SCREENING_YEARS) & np.isin(months, SCREENING_MONTHS)
    indices = np.flatnonzero(selected).astype(np.int64)
    initializations = all_initializations[indices]
    selected_years, _ = _date_parts(initializations)
    counts = {
        year: int(np.count_nonzero(selected_years == year)) for year in SCREENING_YEARS
    }
    expected_counts = {
        year: EXPECTED_INITIALIZATIONS_PER_YEAR for year in SCREENING_YEARS
    }
    if counts != expected_counts or len(indices) != EXPECTED_INITIALIZATION_COUNT:
        raise PhysicalCacheContractError(
            "expected exactly 35 JJAS initializations per year in 2002--2019; "
            f"found {counts}"
        )
    if np.any(selected_years >= 2020):
        raise PhysicalCacheContractError("screening selection contains 2020+ data")
    return indices, initializations


def _contiguous_slice(indices: np.ndarray, name: str) -> slice:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise PhysicalCacheContractError(f"empty {name} selection")
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise PhysicalCacheContractError(f"{name} selection is not contiguous")
    return slice(int(indices[0]), int(indices[-1] + 1))


def _source_fingerprint(
    source_store: Path,
    root_attrs: Mapping[str, Any],
    forecast: Any,
    initializations: np.ndarray,
    selected_indices: np.ndarray,
    channel_names: Sequence[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> str:
    """Fingerprint stable source identity and the complete extraction contract."""

    payload = {
        "source_store": str(source_store.resolve()),
        "schema_version": str(root_attrs.get("schema_version", "")),
        "status": str(root_attrs.get("status", "")),
        "archive_manifest_sha256": str(root_attrs.get("archive_manifest_sha256", "")),
        "archive_records_sha256": str(root_attrs.get("archive_records_sha256", "")),
        "completed_utc": str(root_attrs.get("completed_utc", "")),
        "forecast_shape": list(map(int, forecast.shape)),
        "forecast_chunks": list(map(int, forecast.chunks)),
        "forecast_dtype": np.dtype(forecast.dtype).str,
        "selected_source_indices": selected_indices.tolist(),
        "selected_initializations": np.datetime_as_string(
            initializations, unit="D"
        ).tolist(),
        "channel_names": list(channel_names),
        "latitude": np.asarray(latitude, dtype=np.float64).tolist(),
        "longitude": np.asarray(longitude, dtype=np.float64).tolist(),
        "cache_schema_name": CACHE_SCHEMA_NAME,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "feature_names": list(PHYSICAL_FEATURE_NAMES),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_source(
    source_store: Path = DEFAULT_SOURCE_STORE,
) -> tuple[Any, SourceContract]:
    """Open the global Zarr and validate all metadata needed before field reads."""

    source_store = Path(source_store).resolve()
    if not source_store.is_dir():
        raise FileNotFoundError(source_store)
    zarr = importlib.import_module("zarr")
    group = zarr.open_consolidated(str(source_store), mode="r")
    root_attrs = dict(group.attrs)
    if root_attrs.get("status") != "complete":
        raise PhysicalCacheContractError(
            f"source Zarr status is {root_attrs.get('status')!r}, not 'complete'"
        )
    if str(root_attrs.get("schema_version")) != "1.0":
        raise PhysicalCacheContractError("unsupported source Zarr schema")

    required = {
        "forecast",
        "init",
        "init_complete",
        "member",
        "lead_day",
        "channel",
        "lat",
        "lon",
    }
    missing = required.difference(group.array_keys())
    if missing:
        raise PhysicalCacheContractError(
            f"source Zarr is missing arrays: {sorted(missing)}"
        )

    forecast = group["forecast"]
    initializations = _datetime_days(group["init"][:])
    members = np.asarray(group["member"][:], dtype=np.int16)
    lead_days = np.asarray(group["lead_day"][:], dtype=np.int16)
    channel_names = tuple(np.asarray(group["channel"][:]).astype(str).tolist())
    global_latitude = np.asarray(group["lat"][:], dtype=np.float64)
    global_longitude = np.asarray(group["lon"][:], dtype=np.float64)

    expected_shape = (
        len(initializations),
        MEMBER_COUNT,
        LEAD_DAY_COUNT,
        len(SOURCE_CHANNEL_NAMES),
        len(global_latitude),
        len(global_longitude),
    )
    expected_chunks = (1, MEMBER_COUNT, DAYS_PER_WEEK, 4, 121, 240)
    if tuple(forecast.shape) != expected_shape:
        raise PhysicalCacheContractError(
            f"unexpected forecast shape: {forecast.shape}; expected {expected_shape}"
        )
    if tuple(forecast.chunks) != expected_chunks:
        raise PhysicalCacheContractError(
            f"unexpected forecast chunks: {forecast.chunks}; expected {expected_chunks}"
        )
    if np.dtype(forecast.dtype) != np.dtype(np.float32):
        raise PhysicalCacheContractError("source forecast is not float32")
    if not np.array_equal(members, np.arange(MEMBER_COUNT, dtype=np.int16)):
        raise PhysicalCacheContractError("member labels are not 0..50")
    if not np.array_equal(lead_days, np.arange(1, LEAD_DAY_COUNT + 1, dtype=np.int16)):
        raise PhysicalCacheContractError("lead-day labels are not 1..42")
    if channel_names != SOURCE_CHANNEL_NAMES:
        raise PhysicalCacheContractError("source channel order has changed")
    if not np.array_equal(
        global_latitude, np.linspace(90.0, -90.0, 121, dtype=np.float64)
    ):
        raise PhysicalCacheContractError("source latitude grid has changed")
    if not np.array_equal(
        global_longitude, np.arange(0.0, 360.0, 1.5, dtype=np.float64)
    ):
        raise PhysicalCacheContractError("source longitude grid has changed")

    latitude_indices = np.flatnonzero(
        (global_latitude <= 39.0) & (global_latitude >= 0.0)
    )
    longitude_indices = np.flatnonzero(
        (global_longitude >= 60.0) & (global_longitude <= 99.0)
    )
    latitude = global_latitude[latitude_indices]
    longitude = global_longitude[longitude_indices]
    if not np.array_equal(latitude, EXPECTED_LATITUDE) or not np.array_equal(
        longitude, EXPECTED_LONGITUDE
    ):
        raise PhysicalCacheContractError(
            "source does not contain the exact 0--39N, 60--99E 1.5-degree grid"
        )
    latitude_slice = _contiguous_slice(latitude_indices, "latitude")
    longitude_slice = _contiguous_slice(longitude_indices, "longitude")
    selected_indices, selected_initializations = _select_screening_initializations(
        initializations
    )
    fingerprint = _source_fingerprint(
        source_store,
        root_attrs,
        forecast,
        selected_initializations,
        selected_indices,
        channel_names,
        latitude,
        longitude,
    )
    contract = SourceContract(
        source_store=str(source_store),
        source_fingerprint=fingerprint,
        all_initializations=initializations,
        selected_source_indices=selected_indices,
        selected_initializations=selected_initializations,
        latitude=latitude,
        longitude=longitude,
        latitude_slice=latitude_slice,
        longitude_slice=longitude_slice,
        channel_names=channel_names,
    )
    return group, contract


def _member_weekly(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 4 or values.shape[:2] != (MEMBER_COUNT, DAYS_PER_WEEK):
        raise PhysicalCacheContractError(
            f"unexpected member/day block shape: {values.shape}"
        )
    if not np.isfinite(values).all():
        raise PhysicalCacheContractError("source block contains non-finite values")
    return values.mean(axis=1, dtype=np.float64)


def _weekly_ensemble_mean(values: np.ndarray) -> np.ndarray:
    return _member_weekly(values).mean(axis=0, dtype=np.float64).astype(np.float32)


def _daily_product_ensemble_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise PhysicalCacheContractError("moisture-flux operands have different shapes")
    if first.ndim != 4 or first.shape[:2] != (MEMBER_COUNT, DAYS_PER_WEEK):
        raise PhysicalCacheContractError(
            f"unexpected moisture-flux operand shape: {first.shape}"
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise PhysicalCacheContractError(
            "moisture-flux operands contain non-finite values"
        )
    daily_product = first.astype(np.float64) * second.astype(np.float64)
    return daily_product.mean(axis=1).mean(axis=0).astype(np.float32)


def _summarize_initialization(
    forecast: Any,
    source_index: int,
    channel_names: Sequence[str],
    latitude_slice: slice,
    longitude_slice: slice,
) -> Mapping[str, np.ndarray]:
    """Read exactly 42 source chunks and reduce one initialization.

    Each lead-week/channel-group chunk is read once.  Only u850 and v850 are
    retained until q850 is read so daily per-member moisture flux can be
    formed without rereading their original chunks.
    """

    channel_names = tuple(channel_names)
    channel_indices = {name: channel_names.index(name) for name in RAW_CHANNELS}
    result_by_feature: dict[str, list[np.ndarray]] = {
        name: [] for name in PHYSICAL_FEATURE_NAMES
    }
    for lead_week in range(LEAD_WEEK_COUNT):
        lead_slice = slice(lead_week * DAYS_PER_WEEK, (lead_week + 1) * DAYS_PER_WEEK)
        retained_u850: np.ndarray | None = None
        retained_v850: np.ndarray | None = None
        produced: set[str] = set()
        for group_start in range(0, len(channel_names), SOURCE_CHANNEL_CHUNK):
            group_stop = min(group_start + SOURCE_CHANNEL_CHUNK, len(channel_names))
            block = np.asarray(
                forecast[
                    int(source_index),
                    slice(None),
                    lead_slice,
                    slice(group_start, group_stop),
                    latitude_slice,
                    longitude_slice,
                ],
                dtype=np.float32,
            )
            if block.ndim != 5 or block.shape[:3] != (
                MEMBER_COUNT,
                DAYS_PER_WEEK,
                group_stop - group_start,
            ):
                raise PhysicalCacheContractError(
                    f"unexpected source chunk result shape: {block.shape}"
                )
            if not np.isfinite(block).all():
                raise PhysicalCacheContractError(
                    f"non-finite source values at init {source_index}, "
                    f"week {lead_week + 1}, channel group {group_start}"
                )

            def field(name: str) -> np.ndarray:
                return block[:, :, channel_indices[name] - group_start]

            names_in_group = {
                name
                for name, index in channel_indices.items()
                if group_start <= index < group_stop
            }
            if "z500" in names_in_group:
                result_by_feature["z500_mean"].append(
                    _weekly_ensemble_mean(field("z500"))
                )
                produced.add("z500_mean")
            if "u850" in names_in_group:
                retained_u850 = field("u850").copy()
                result_by_feature["u850_mean"].append(
                    _weekly_ensemble_mean(retained_u850)
                )
                produced.add("u850_mean")
            if "v850" in names_in_group:
                retained_v850 = field("v850").copy()
                result_by_feature["v850_mean"].append(
                    _weekly_ensemble_mean(retained_v850)
                )
                produced.add("v850_mean")
            if "q850" in names_in_group:
                q850 = np.maximum(field("q850"), np.float32(0.0))
                if retained_u850 is None or retained_v850 is None:
                    raise PhysicalCacheContractError(
                        "u850/v850 must be read before q850 for moisture flux"
                    )
                result_by_feature["q850_mean"].append(_weekly_ensemble_mean(q850))
                result_by_feature["q850_u850_flux_mean"].append(
                    _daily_product_ensemble_mean(q850, retained_u850)
                )
                result_by_feature["q850_v850_flux_mean"].append(
                    _daily_product_ensemble_mean(q850, retained_v850)
                )
                produced.update(
                    {
                        "q850_mean",
                        "q850_u850_flux_mean",
                        "q850_v850_flux_mean",
                    }
                )
                retained_u850 = None
                retained_v850 = None
            if "ttr" in names_in_group:
                result_by_feature["olr_mean"].append(
                    -_weekly_ensemble_mean(field("ttr"))
                )
                produced.add("olr_mean")
            if "msl" in names_in_group:
                result_by_feature["msl_mean"].append(
                    _weekly_ensemble_mean(field("msl"))
                )
                produced.add("msl_mean")
            if "tcwv" in names_in_group:
                tcwv = np.maximum(field("tcwv"), np.float32(0.0))
                member_weekly = _member_weekly(tcwv)
                result_by_feature["tcwv_mean"].append(
                    member_weekly.mean(axis=0, dtype=np.float64).astype(np.float32)
                )
                result_by_feature["tcwv_spread"].append(
                    member_weekly.std(axis=0, ddof=0, dtype=np.float64).astype(
                        np.float32
                    )
                )
                produced.update({"tcwv_mean", "tcwv_spread"})
        if produced != set(PHYSICAL_FEATURE_NAMES):
            missing = set(PHYSICAL_FEATURE_NAMES).difference(produced)
            raise PhysicalCacheContractError(
                f"week {lead_week + 1} did not produce features: {sorted(missing)}"
            )

    result = {
        name: np.asarray(np.stack(values), dtype=np.float32)
        for name, values in result_by_feature.items()
    }
    shapes = {values.shape for values in result.values()}
    if len(shapes) != 1:
        raise PhysicalCacheContractError(f"physical feature shapes differ: {shapes}")
    shape = next(iter(shapes))
    if (
        shape[0] != LEAD_WEEK_COUNT
        or not np.isfinite(np.stack(tuple(result.values()))).all()
    ):
        raise PhysicalCacheContractError("invalid reduced physical features")
    if np.any(result["tcwv_spread"] < 0.0):
        raise PhysicalCacheContractError("TCWV ensemble spread is negative")
    if np.any(result["olr_mean"] < 0.0):
        raise PhysicalCacheContractError(
            "negative OLR after applying the required OLR=-TTR conversion"
        )
    return result


def _validate_predictors(
    predictors: FuxiPhysicalPredictors,
    expected_initializations: np.ndarray | None = None,
) -> None:
    initializations = np.asarray(predictors.initializations, dtype="datetime64[D]")
    if initializations.ndim != 1 or not len(initializations):
        raise PhysicalCacheContractError("physical-cache initializations are empty")
    if np.unique(initializations).size != len(initializations):
        raise PhysicalCacheContractError(
            "physical-cache initializations are not unique"
        )
    if np.any(initializations[1:] <= initializations[:-1]):
        raise PhysicalCacheContractError(
            "physical-cache initializations are not sorted"
        )
    years, months = _date_parts(initializations)
    if not np.all(np.isin(years, SCREENING_YEARS)) or not np.all(
        np.isin(months, SCREENING_MONTHS)
    ):
        raise PhysicalCacheContractError(
            "physical cache contains a non-JJAS or post-2019 initialization"
        )
    if expected_initializations is not None and not np.array_equal(
        initializations,
        np.asarray(expected_initializations, dtype="datetime64[D]"),
    ):
        raise PhysicalCacheContractError(
            "physical-cache initializations differ from the expected screening subset"
        )
    if not np.array_equal(
        np.asarray(predictors.latitude, dtype=np.float64), EXPECTED_LATITUDE
    ) or not np.array_equal(
        np.asarray(predictors.longitude, dtype=np.float64), EXPECTED_LONGITUDE
    ):
        raise PhysicalCacheContractError("physical-cache grid is not exact")
    expected_shape = (len(initializations), LEAD_WEEK_COUNT, *EXPECTED_GRID_SHAPE)
    for name, values in predictors.feature_fields.items():
        if values.shape != expected_shape or values.dtype != np.float32:
            raise PhysicalCacheContractError(
                f"{name} has shape/dtype {values.shape}/{values.dtype}; "
                f"expected {expected_shape}/float32"
            )
        if not np.isfinite(values).all():
            raise PhysicalCacheContractError(f"{name} contains non-finite values")
    if np.any(predictors.tcwv_spread < 0.0):
        raise PhysicalCacheContractError("TCWV ensemble spread is negative")
    if np.any(predictors.olr_mean < 0.0):
        raise PhysicalCacheContractError("positive-OLR cache contains negatives")
    if not predictors.source_fingerprint:
        raise PhysicalCacheContractError("physical cache has no source fingerprint")


def validate_fuxi_physical_predictors(
    predictors: FuxiPhysicalPredictors, forecast: Any
) -> None:
    """Validate a 2002--2019 cache against a possibly 2002--2021 forecast."""

    forecast_initializations = np.asarray(
        forecast.initializations, dtype="datetime64[D]"
    )
    forecast_years, forecast_months = _date_parts(forecast_initializations)
    relevant = np.isin(forecast_years, SCREENING_YEARS) & np.isin(
        forecast_months, SCREENING_MONTHS
    )
    expected = forecast_initializations[relevant]
    _validate_predictors(predictors, expected_initializations=expected)
    if not np.array_equal(
        np.asarray(forecast.latitude, dtype=np.float64), predictors.latitude
    ) or not np.array_equal(
        np.asarray(forecast.longitude, dtype=np.float64), predictors.longitude
    ):
        raise PhysicalCacheContractError(
            "physical-cache grid differs from the FuXi rainfall forecast grid"
        )


def _part_path(parts_dir: Path, initialization: np.datetime64) -> Path:
    stamp = np.datetime_as_string(initialization, unit="D").replace("-", "")
    return Path(parts_dir) / f"{stamp}.npz"


def _part_payload(
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
    feature_fields: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(contract.source_store),
        "source_fingerprint": np.asarray(contract.source_fingerprint),
        "source_init_index": np.asarray(source_index, dtype=np.int32),
        "initialization": np.asarray(initialization, dtype="datetime64[D]"),
        "latitude": np.asarray(contract.latitude, dtype=np.float64),
        "longitude": np.asarray(contract.longitude, dtype=np.float64),
        "normalization": np.asarray("none_raw_native_values"),
        "physical_transforms_json": np.asarray(
            json.dumps(PHYSICAL_TRANSFORMS, sort_keys=True)
        ),
        **feature_fields,
    }


def _load_part(
    part_path: Path,
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
) -> Mapping[str, np.ndarray]:
    required = {
        "schema_name",
        "schema_version",
        "source_store",
        "source_fingerprint",
        "source_init_index",
        "initialization",
        "latitude",
        "longitude",
        "normalization",
        "physical_transforms_json",
        *PHYSICAL_FEATURE_NAMES,
    }
    try:
        with np.load(part_path, allow_pickle=False) as part:
            missing = required.difference(part.files)
            if missing:
                raise PhysicalCacheContractError(
                    f"part {part_path} is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(part[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise PhysicalCacheContractError(f"part {part_path} schema differs")
            if (
                str(scalar("source_store")) != contract.source_store
                or str(scalar("source_fingerprint")) != contract.source_fingerprint
            ):
                raise PhysicalCacheContractError(f"part {part_path} source is stale")
            if (
                int(scalar("source_init_index")) != int(source_index)
                or np.asarray(part["initialization"], dtype="datetime64[D]").item()
                != np.asarray(initialization, dtype="datetime64[D]").item()
            ):
                raise PhysicalCacheContractError(f"part {part_path} init differs")
            if str(scalar("normalization")) != "none_raw_native_values":
                raise PhysicalCacheContractError(f"part {part_path} was normalized")
            if json.loads(str(scalar("physical_transforms_json"))) != (
                PHYSICAL_TRANSFORMS
            ):
                raise PhysicalCacheContractError(
                    f"part {part_path} physical transforms differ"
                )
            if not np.array_equal(
                part["latitude"], contract.latitude
            ) or not np.array_equal(part["longitude"], contract.longitude):
                raise PhysicalCacheContractError(f"part {part_path} grid differs")
            fields = {
                name: np.asarray(part[name], dtype=np.float32).copy()
                for name in PHYSICAL_FEATURE_NAMES
            }
    except (OSError, ValueError) as error:
        if isinstance(error, PhysicalCacheContractError):
            raise
        raise PhysicalCacheContractError(
            f"cannot read physical-cache part {part_path}: {error}"
        ) from error
    for name, values in fields.items():
        if values.shape != (LEAD_WEEK_COUNT, *EXPECTED_GRID_SHAPE):
            raise PhysicalCacheContractError(
                f"part {part_path} has invalid {name} shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise PhysicalCacheContractError(
                f"part {part_path} has non-finite {name} values"
            )
    if np.any(fields["tcwv_spread"] < 0.0) or np.any(fields["olr_mean"] < 0.0):
        raise PhysicalCacheContractError(f"part {part_path} violates physical bounds")
    return fields


def build_part(
    group: Any,
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
    parts_dir: Path = DEFAULT_PARTS_DIR,
) -> tuple[Path, bool]:
    """Build one validated atomic part, or safely reuse an existing part."""

    part_path = _part_path(parts_dir, initialization).resolve()
    if part_path.is_file():
        try:
            _load_part(part_path, contract, source_index, initialization)
            return part_path, False
        except PhysicalCacheContractError as error:
            print(f"rebuilding invalid part {part_path}: {error}", flush=True)
    if not bool(np.asarray(group["init_complete"][int(source_index)]).item()):
        raise PhysicalCacheContractError(
            f"source initialization {initialization} is not marked complete"
        )
    fields = _summarize_initialization(
        group["forecast"],
        source_index,
        contract.channel_names,
        contract.latitude_slice,
        contract.longitude_slice,
    )
    _atomic_npz(
        part_path,
        _part_payload(contract, source_index, initialization, fields),
    )
    _load_part(part_path, contract, source_index, initialization)
    return part_path, True


def _cache_payload(predictors: FuxiPhysicalPredictors) -> Mapping[str, Any]:
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(predictors.source_store),
        "source_fingerprint": np.asarray(predictors.source_fingerprint),
        "initializations": np.asarray(
            predictors.initializations, dtype="datetime64[D]"
        ),
        "latitude": np.asarray(predictors.latitude, dtype=np.float64),
        "longitude": np.asarray(predictors.longitude, dtype=np.float64),
        "feature_names": np.asarray(PHYSICAL_FEATURE_NAMES),
        "feature_definitions_json": np.asarray(
            json.dumps(PHYSICAL_FEATURE_DEFINITIONS, sort_keys=True)
        ),
        "feature_units_json": np.asarray(
            json.dumps(PHYSICAL_FEATURE_UNITS, sort_keys=True)
        ),
        "normalization": np.asarray("none_raw_native_values"),
        "physical_transforms_json": np.asarray(
            json.dumps(PHYSICAL_TRANSFORMS, sort_keys=True)
        ),
        "selection": np.asarray("JJAS initializations, 2002-2019 inclusive"),
        "weekly_reduction": np.asarray(
            "seven-day mean per member, then 51-member ensemble statistics"
        ),
        "moisture_flux_reduction": np.asarray(
            "daily per-member q850*wind products, then seven-day and " "51-member means"
        ),
        "olr_conversion": np.asarray("olr=-ttr"),
        **predictors.feature_fields,
    }


def _write_final_cache(path: Path, predictors: FuxiPhysicalPredictors) -> str:
    _validate_predictors(predictors)
    _atomic_npz(path, _cache_payload(predictors))
    return sha256_file(path)


def _load_final_cache(
    cache_path: Path,
    *,
    expected_initializations: np.ndarray | None = None,
    expected_source_fingerprint: str | None = None,
) -> FuxiPhysicalPredictors:
    cache_path = Path(cache_path).resolve()
    required = {
        "schema_name",
        "schema_version",
        "source_store",
        "source_fingerprint",
        "initializations",
        "latitude",
        "longitude",
        "feature_names",
        "feature_definitions_json",
        "feature_units_json",
        "normalization",
        "physical_transforms_json",
        "selection",
        "weekly_reduction",
        "moisture_flux_reduction",
        "olr_conversion",
        *PHYSICAL_FEATURE_NAMES,
    }
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            missing = required.difference(cached.files)
            if missing:
                raise PhysicalCacheContractError(
                    f"physical cache is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(cached[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise PhysicalCacheContractError("physical-cache schema differs")
            if tuple(np.asarray(cached["feature_names"]).astype(str)) != (
                PHYSICAL_FEATURE_NAMES
            ):
                raise PhysicalCacheContractError(
                    "physical-cache feature order has changed"
                )
            if str(scalar("normalization")) != "none_raw_native_values":
                raise PhysicalCacheContractError("physical cache is not raw")
            if json.loads(str(scalar("physical_transforms_json"))) != (
                PHYSICAL_TRANSFORMS
            ):
                raise PhysicalCacheContractError(
                    "physical-cache transforms have changed"
                )
            if str(scalar("olr_conversion")) != "olr=-ttr":
                raise PhysicalCacheContractError("physical cache has wrong OLR sign")
            source_fingerprint = str(scalar("source_fingerprint"))
            if (
                expected_source_fingerprint is not None
                and source_fingerprint != expected_source_fingerprint
            ):
                raise PhysicalCacheContractError(
                    "physical-cache source fingerprint is stale"
                )
            values = {
                name: np.asarray(cached[name], dtype=np.float32).copy()
                for name in PHYSICAL_FEATURE_NAMES
            }
            predictors = FuxiPhysicalPredictors(
                initializations=np.asarray(
                    cached["initializations"], dtype="datetime64[D]"
                ).copy(),
                latitude=np.asarray(cached["latitude"], dtype=np.float64).copy(),
                longitude=np.asarray(cached["longitude"], dtype=np.float64).copy(),
                source_fingerprint=source_fingerprint,
                source_store=str(scalar("source_store")),
                cache_path=str(cache_path),
                cache_sha256=sha256_file(cache_path),
                **values,
            )
    except (OSError, ValueError) as error:
        if isinstance(error, PhysicalCacheContractError):
            raise
        raise PhysicalCacheContractError(
            f"cannot read physical cache {cache_path}: {error}"
        ) from error
    _validate_predictors(predictors, expected_initializations=expected_initializations)
    return predictors


def load_fuxi_physical_predictors(
    forecast: Any,
    cache_path: Path = DEFAULT_PHYSICAL_CACHE,
    *,
    expected_source_fingerprint: str | None = None,
) -> FuxiPhysicalPredictors:
    """Load raw 2002--2019 predictors and align them to a full FuXi forecast.

    The returned cache is intentionally a subset of a 2002--2021 rainfall
    archive.  Consumers must align by initialization date and must never fill
    held-out 2020--2021 rows with physical predictors during screening.
    """

    predictors = _load_final_cache(
        cache_path,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    validate_fuxi_physical_predictors(predictors, forecast)
    return predictors


def finalize_cache(
    contract: SourceContract,
    parts_dir: Path = DEFAULT_PARTS_DIR,
    output: Path = DEFAULT_PHYSICAL_CACHE,
) -> tuple[Path, str]:
    """Validate all 630 parts and atomically publish the raw final cache."""

    fields: dict[str, list[np.ndarray]] = {name: [] for name in PHYSICAL_FEATURE_NAMES}
    missing: list[str] = []
    for source_index, initialization in zip(
        contract.selected_source_indices,
        contract.selected_initializations,
        strict=True,
    ):
        part_path = _part_path(parts_dir, initialization)
        if not part_path.is_file():
            missing.append(np.datetime_as_string(initialization, unit="D"))
            continue
        part_fields = _load_part(part_path, contract, int(source_index), initialization)
        for name in PHYSICAL_FEATURE_NAMES:
            fields[name].append(part_fields[name])
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise PhysicalCacheContractError(
            f"cannot finalize: {len(missing)} of "
            f"{EXPECTED_INITIALIZATION_COUNT} parts are missing ({preview}{suffix})"
        )
    values = {
        name: np.asarray(np.stack(feature_parts), dtype=np.float32)
        for name, feature_parts in fields.items()
    }
    predictors = FuxiPhysicalPredictors(
        initializations=contract.selected_initializations.copy(),
        latitude=contract.latitude.copy(),
        longitude=contract.longitude.copy(),
        source_fingerprint=contract.source_fingerprint,
        source_store=contract.source_store,
        **values,
    )
    _validate_predictors(
        predictors, expected_initializations=contract.selected_initializations
    )
    output = Path(output).resolve()
    checksum = _write_final_cache(output, predictors)
    _load_final_cache(
        output,
        expected_initializations=contract.selected_initializations,
        expected_source_fingerprint=contract.source_fingerprint,
    )
    return output, checksum


def _selected_task_records(
    contract: SourceContract,
    *,
    task_index: int | None,
    task_count: int | None,
    initialization: str | None,
    max_initializations: int | None,
) -> list[tuple[int, np.datetime64]]:
    records = list(
        zip(
            contract.selected_source_indices.tolist(),
            contract.selected_initializations.tolist(),
            strict=True,
        )
    )
    if initialization is not None:
        if task_index is not None or task_count is not None:
            raise PhysicalCacheContractError(
                "--init cannot be combined with --task-index/--task-count"
            )
        requested = np.datetime64(initialization, "D")
        records = [record for record in records if record[1] == requested]
        if len(records) != 1:
            raise PhysicalCacheContractError(
                f"{initialization} is not a 2002--2019 JJAS FuXi initialization"
            )
    else:
        if task_index is None or task_count is None:
            raise PhysicalCacheContractError(
                "build requires either --init or both --task-index and --task-count"
            )
        if task_count < 2:
            raise PhysicalCacheContractError(
                "refusing a serial 630-init build; use at least two array tasks"
            )
        if task_index < 0 or task_index >= task_count:
            raise PhysicalCacheContractError(
                f"task index {task_index} is outside [0, {task_count})"
            )
        records = records[task_index::task_count]
    if max_initializations is not None:
        if max_initializations < 1:
            raise PhysicalCacheContractError("--max-inits must be positive")
        records = records[:max_initializations]
    return [(int(index), np.datetime64(init, "D")) for index, init in records]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory", help="validate source metadata and report work")

    build = subparsers.add_parser(
        "build", help="build one initialization or one strided array-task shard"
    )
    build.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    build.add_argument("--task-index", type=int)
    build.add_argument("--task-count", type=int)
    build.add_argument("--init")
    build.add_argument("--max-inits", type=int)

    finalize = subparsers.add_parser(
        "finalize", help="validate all parts and publish the final cache"
    )
    finalize.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    finalize.add_argument("--output", type=Path, default=DEFAULT_PHYSICAL_CACHE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    group, contract = inspect_source(args.source_store)
    if args.command == "inventory":
        print(
            json.dumps(
                {
                    "source_store": contract.source_store,
                    "source_fingerprint": contract.source_fingerprint,
                    "selected_initialization_count": len(
                        contract.selected_initializations
                    ),
                    "first_initialization": np.datetime_as_string(
                        contract.selected_initializations[0], unit="D"
                    ),
                    "last_initialization": np.datetime_as_string(
                        contract.selected_initializations[-1], unit="D"
                    ),
                    "latitude_count": len(contract.latitude),
                    "longitude_count": len(contract.longitude),
                    "forecast_data_years_read": "2002-2019 only during build",
                    "recommended_array_tasks": 64,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.command == "build":
        records = _selected_task_records(
            contract,
            task_index=args.task_index,
            task_count=args.task_count,
            initialization=args.init,
            max_initializations=args.max_inits,
        )
        for ordinal, (source_index, initialization) in enumerate(records, start=1):
            part_path, created = build_part(
                group,
                contract,
                source_index,
                initialization,
                args.parts_dir,
            )
            action = "built" if created else "reused"
            print(
                f"[{ordinal}/{len(records)}] {action} "
                f"{np.datetime_as_string(initialization, unit='D')}: {part_path}",
                flush=True,
            )
        return
    output, checksum = finalize_cache(contract, args.parts_dir, args.output)
    print(f"published {output}", flush=True)
    print(f"sha256={checksum}", flush=True)


if __name__ == "__main__":
    main()
