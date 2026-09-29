#!/usr/bin/env python
"""Build a year-round weekly FuXi precipitation-member cache.

The source is the complete native 2002--2021 FuXi-S2S Zarr archive.  One
initialization remains one statistical sample: all 51 exchangeable members,
all six seven-day leads, and the 27x27 India context grid are retained.  The
only physical conversion is the documented native TP conversion from
``mm h-1`` to ``mm day-1`` before taking each member's seven-day mean.

The build is deliberately two-stage.  Strided workers atomically publish one
compressed NPZ part per initialization; a separate finalizer validates every
part and streams it into an atomically published float32 NPY memmap.  JSON
metadata, a part manifest, and a SHA-256 completion file make the final cache
auditable without loading it into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from project_paths import PROJECT_ROOT


DEFAULT_SOURCE_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "cache" / "fuxi_tp_members_weekly_2002_2021_allseason_v1.npy"
)
DEFAULT_PARTS_DIR = (
    PROJECT_ROOT
    / "cache"
    / "fuxi_tp_members_weekly_2002_2021_allseason_v1.parts"
)

CACHE_SCHEMA_NAME = "fuxi-tp-members-weekly-allseason"
CACHE_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = "1.0"

START_YEAR = 2002
END_YEAR = 2021
YEARS = tuple(range(START_YEAR, END_YEAR + 1))
EXPECTED_INITIALIZATIONS_PER_YEAR = 104
INITIALIZATION_COUNT = len(YEARS) * EXPECTED_INITIALIZATIONS_PER_YEAR
MEMBER_COUNT = 51
LEAD_DAY_COUNT = 42
LEAD_WEEK_COUNT = 6
DAYS_PER_WEEK = 7
CHANNEL_COUNT = 26
GLOBAL_LATITUDE_COUNT = 121
GLOBAL_LONGITUDE_COUNT = 240
EXPECTED_SOURCE_SHAPE = (
    INITIALIZATION_COUNT,
    MEMBER_COUNT,
    LEAD_DAY_COUNT,
    CHANNEL_COUNT,
    GLOBAL_LATITUDE_COUNT,
    GLOBAL_LONGITUDE_COUNT,
)
EXPECTED_SOURCE_CHUNKS = (1, MEMBER_COUNT, DAYS_PER_WEEK, 4, 121, 240)

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
TP_CHANNEL_INDEX = SOURCE_CHANNEL_NAMES.index("tp")

EXPECTED_LATITUDE = np.arange(39.0, -0.01, -1.5, dtype=np.float64)
EXPECTED_LONGITUDE = np.arange(60.0, 99.01, 1.5, dtype=np.float64)
GRID_SHAPE = (len(EXPECTED_LATITUDE), len(EXPECTED_LONGITUDE))
MEMBER_FIELD_SHAPE = (MEMBER_COUNT, LEAD_WEEK_COUNT, *GRID_SHAPE)
OUTPUT_DIMS = ("init", "member", "lead_week", "lat", "lon")
INPUT_UNITS = "mm h-1"
OUTPUT_UNITS = "mm day-1"
CONVERSION_DESCRIPTION = (
    "multiply native daily-mean TP rate by 24, then take a seven-day "
    "arithmetic mean independently for every member"
)
SMOKE_SPLIT_COUNTS = {
    "train_2002_2017": 32,
    "validation_2018_2019": 16,
    "test_2020_2021": 16,
}
SMOKE_INITIALIZATION_COUNT = sum(SMOKE_SPLIT_COUNTS.values())


class MemberCacheContractError(ValueError):
    """Raised when source, part, or final-cache evidence violates the contract."""


@dataclass(frozen=True)
class SourceContract:
    """Validated source identity and exact all-season extraction coordinates."""

    source_store: str
    source_fingerprint: str
    source_indices: np.ndarray
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    latitude_slice: slice
    longitude_slice: slice
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class CacheArtifacts:
    """Paths and checksums published by :func:`finalize_cache`."""

    data: Path
    metadata: Path
    manifest: Path
    checksums: Path
    data_sha256: str
    metadata_sha256: str
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order value bytes."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.", suffix=".temporary", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _datetime_days(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")
    if not np.issubdtype(values.dtype, np.integer):
        raise MemberCacheContractError(
            f"unsupported initialization coordinate dtype: {values.dtype}"
        )
    return values.astype(np.int64).astype("datetime64[ns]").astype("datetime64[D]")


def _initialization_years(initializations: np.ndarray) -> np.ndarray:
    stamps = np.datetime_as_string(
        np.asarray(initializations, dtype="datetime64[D]"), unit="D"
    )
    return np.asarray([int(stamp[:4]) for stamp in stamps], dtype=np.int16)


def _validate_allseason_initializations(initializations: np.ndarray) -> np.ndarray:
    """Require exactly 104 unique, ordered initializations in every source year."""

    values = np.asarray(initializations, dtype="datetime64[D]")
    if values.shape != (INITIALIZATION_COUNT,):
        raise MemberCacheContractError(
            f"expected {INITIALIZATION_COUNT} source initializations; found {values.shape}"
        )
    if np.isnat(values).any():
        raise MemberCacheContractError("source initializations contain NaT")
    if np.unique(values).size != INITIALIZATION_COUNT:
        raise MemberCacheContractError("source initializations are not unique")
    if np.any(values[1:] <= values[:-1]):
        raise MemberCacheContractError(
            "source initializations are not strictly increasing"
        )
    years = _initialization_years(values)
    counts = {year: int(np.count_nonzero(years == year)) for year in YEARS}
    expected = {year: EXPECTED_INITIALIZATIONS_PER_YEAR for year in YEARS}
    if counts != expected or not np.all(np.isin(years, YEARS)):
        raise MemberCacheContractError(
            "expected exactly 104 initializations in each year 2002--2021; "
            f"found {counts}"
        )
    return values


def _contiguous_slice(indices: np.ndarray, name: str) -> slice:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise MemberCacheContractError(f"empty {name} selection")
    expected = np.arange(indices[0], indices[-1] + 1, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise MemberCacheContractError(f"{name} selection is not contiguous")
    return slice(int(indices[0]), int(indices[-1] + 1))


def _source_fingerprint(
    source_store: Path,
    root_attrs: Mapping[str, Any],
    forecast: Any,
    initializations: np.ndarray,
    channel_names: Sequence[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> str:
    payload = {
        "source_store": str(source_store.resolve()),
        "source_schema_version": str(root_attrs.get("schema_version", "")),
        "source_status": str(root_attrs.get("status", "")),
        "archive_manifest_sha256": str(root_attrs.get("archive_manifest_sha256", "")),
        "archive_records_sha256": str(root_attrs.get("archive_records_sha256", "")),
        "completed_utc": str(root_attrs.get("completed_utc", "")),
        "forecast_shape": list(map(int, forecast.shape)),
        "forecast_chunks": list(map(int, forecast.chunks)),
        "forecast_dtype": np.dtype(forecast.dtype).str,
        "initializations": np.datetime_as_string(initializations, unit="D").tolist(),
        "channel_names": list(channel_names),
        "latitude": np.asarray(latitude, dtype=np.float64).tolist(),
        "longitude": np.asarray(longitude, dtype=np.float64).tolist(),
        "cache_schema_name": CACHE_SCHEMA_NAME,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "output_dims": list(OUTPUT_DIMS),
        "conversion": CONVERSION_DESCRIPTION,
    }
    return _canonical_sha256(payload)


def inspect_source(
    source_store: Path = DEFAULT_SOURCE_STORE,
) -> tuple[Any, SourceContract]:
    """Open and strictly validate the complete native FuXi Zarr metadata."""

    source_store = Path(source_store).resolve()
    if not source_store.is_dir():
        raise FileNotFoundError(source_store)
    zarr = importlib.import_module("zarr")
    group = zarr.open_consolidated(str(source_store), mode="r")
    attrs = dict(group.attrs)
    if attrs.get("status") != "complete":
        raise MemberCacheContractError(
            f"source Zarr status is {attrs.get('status')!r}, not 'complete'"
        )
    if str(attrs.get("schema_version")) != SOURCE_SCHEMA_VERSION:
        raise MemberCacheContractError("unsupported source Zarr schema")

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
    missing = required.difference(set(group.array_keys()))
    if missing:
        raise MemberCacheContractError(
            f"source Zarr is missing arrays: {sorted(missing)}"
        )

    forecast = group["forecast"]
    if tuple(forecast.shape) != EXPECTED_SOURCE_SHAPE:
        raise MemberCacheContractError(
            f"unexpected forecast shape {forecast.shape}; expected {EXPECTED_SOURCE_SHAPE}"
        )
    if tuple(forecast.chunks) != EXPECTED_SOURCE_CHUNKS:
        raise MemberCacheContractError(
            f"unexpected forecast chunks {forecast.chunks}; expected {EXPECTED_SOURCE_CHUNKS}"
        )
    if np.dtype(forecast.dtype) != np.dtype(np.float32):
        raise MemberCacheContractError("source forecast is not float32")

    initializations = _validate_allseason_initializations(
        _datetime_days(group["init"][:])
    )
    members = np.asarray(group["member"][:], dtype=np.int16)
    lead_days = np.asarray(group["lead_day"][:], dtype=np.int16)
    channel_names = tuple(np.asarray(group["channel"][:]).astype(str).tolist())
    complete = np.asarray(group["init_complete"][:])
    if complete.shape != (INITIALIZATION_COUNT,) or complete.dtype != np.bool_:
        raise MemberCacheContractError("init_complete has an invalid shape or dtype")
    if not np.all(complete):
        raise MemberCacheContractError(
            f"{int(np.count_nonzero(~complete))} source initializations are incomplete"
        )
    if not np.array_equal(members, np.arange(MEMBER_COUNT, dtype=np.int16)):
        raise MemberCacheContractError("member labels are not 0..50")
    if not np.array_equal(
        lead_days, np.arange(1, LEAD_DAY_COUNT + 1, dtype=np.int16)
    ):
        raise MemberCacheContractError("lead-day labels are not 1..42")
    if channel_names != SOURCE_CHANNEL_NAMES:
        raise MemberCacheContractError("source channel order has changed")

    global_latitude = np.asarray(group["lat"][:], dtype=np.float64)
    global_longitude = np.asarray(group["lon"][:], dtype=np.float64)
    if not np.array_equal(
        global_latitude, np.linspace(90.0, -90.0, 121, dtype=np.float64)
    ):
        raise MemberCacheContractError("source latitude grid has changed")
    if not np.array_equal(
        global_longitude, np.arange(0.0, 360.0, 1.5, dtype=np.float64)
    ):
        raise MemberCacheContractError("source longitude grid has changed")
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
        raise MemberCacheContractError(
            "source does not contain the exact 0--39N, 60--99E 1.5-degree grid"
        )

    fingerprint = _source_fingerprint(
        source_store,
        attrs,
        forecast,
        initializations,
        channel_names,
        latitude,
        longitude,
    )
    return group, SourceContract(
        source_store=str(source_store),
        source_fingerprint=fingerprint,
        source_indices=np.arange(INITIALIZATION_COUNT, dtype=np.int64),
        initializations=initializations,
        latitude=latitude,
        longitude=longitude,
        latitude_slice=_contiguous_slice(latitude_indices, "latitude"),
        longitude_slice=_contiguous_slice(longitude_indices, "longitude"),
        channel_names=channel_names,
    )


def _weekly_tp_members(tp_hourly: np.ndarray) -> np.ndarray:
    """Convert native TP to member-wise weekly mean rainfall in mm/day."""

    values = np.asarray(tp_hourly)
    if values.dtype != np.float32:
        raise MemberCacheContractError(
            f"source TP block is {values.dtype}, expected float32"
        )
    if values.ndim != 4 or values.shape[:2] != (MEMBER_COUNT, LEAD_DAY_COUNT):
        raise MemberCacheContractError(
            f"unexpected TP member/day block shape: {values.shape}"
        )
    if not np.isfinite(values).all():
        raise MemberCacheContractError("source TP block contains non-finite values")
    if np.any(values < 0.0):
        raise MemberCacheContractError("source TP block contains negative rainfall")
    height, width = values.shape[-2:]
    weekly = (
        (values * np.float32(24.0))
        .reshape(MEMBER_COUNT, LEAD_WEEK_COUNT, DAYS_PER_WEEK, height, width)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )
    if not np.isfinite(weekly).all() or np.any(weekly < 0.0):
        raise MemberCacheContractError("weekly TP conversion produced invalid values")
    return weekly


def _extract_initialization(
    forecast: Any,
    source_index: int,
    latitude_slice: slice,
    longitude_slice: slice,
) -> np.ndarray:
    """Read TP once for one initialization and preserve every member."""

    raw = np.asarray(
        forecast[
            int(source_index),
            slice(None),
            slice(None),
            TP_CHANNEL_INDEX,
            latitude_slice,
            longitude_slice,
        ]
    )
    weekly = _weekly_tp_members(raw)
    if weekly.shape != MEMBER_FIELD_SHAPE:
        raise MemberCacheContractError(
            f"weekly India TP shape {weekly.shape}; expected {MEMBER_FIELD_SHAPE}"
        )
    return weekly


def _part_path(parts_dir: Path, initialization: np.datetime64) -> Path:
    stamp = np.datetime_as_string(initialization, unit="D").replace("-", "")
    return Path(parts_dir) / f"{stamp}.npz"


def _part_payload(
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
    weekly_tp: np.ndarray,
) -> Mapping[str, Any]:
    values = np.asarray(weekly_tp)
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(contract.source_store),
        "source_fingerprint": np.asarray(contract.source_fingerprint),
        "source_init_index": np.asarray(source_index, dtype=np.int32),
        "initialization": np.asarray(initialization, dtype="datetime64[D]"),
        "latitude": np.asarray(contract.latitude, dtype=np.float64),
        "longitude": np.asarray(contract.longitude, dtype=np.float64),
        "input_units": np.asarray(INPUT_UNITS),
        "output_units": np.asarray(OUTPUT_UNITS),
        "conversion": np.asarray(CONVERSION_DESCRIPTION),
        "tp_members_weekly_sha256": np.asarray(_array_sha256(values)),
        "tp_members_weekly": values,
    }


def _load_part(
    path: Path,
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
) -> np.ndarray:
    required = {
        "schema_name",
        "schema_version",
        "source_store",
        "source_fingerprint",
        "source_init_index",
        "initialization",
        "latitude",
        "longitude",
        "input_units",
        "output_units",
        "conversion",
        "tp_members_weekly_sha256",
        "tp_members_weekly",
    }
    try:
        with np.load(path, allow_pickle=False) as part:
            missing = required.difference(part.files)
            if missing:
                raise MemberCacheContractError(
                    f"part {path} is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(part[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise MemberCacheContractError(f"part {path} schema differs")
            if (
                str(scalar("source_store")) != contract.source_store
                or str(scalar("source_fingerprint")) != contract.source_fingerprint
            ):
                raise MemberCacheContractError(f"part {path} source is stale")
            stored_initialization = np.asarray(
                part["initialization"], dtype="datetime64[D]"
            ).item()
            expected_initialization = np.asarray(
                initialization, dtype="datetime64[D]"
            ).item()
            if (
                int(scalar("source_init_index")) != int(source_index)
                or stored_initialization != expected_initialization
            ):
                raise MemberCacheContractError(f"part {path} initialization differs")
            if (
                str(scalar("input_units")) != INPUT_UNITS
                or str(scalar("output_units")) != OUTPUT_UNITS
                or str(scalar("conversion")) != CONVERSION_DESCRIPTION
            ):
                raise MemberCacheContractError(f"part {path} conversion differs")
            if not np.array_equal(part["latitude"], contract.latitude) or not np.array_equal(
                part["longitude"], contract.longitude
            ):
                raise MemberCacheContractError(f"part {path} grid differs")
            stored = part["tp_members_weekly"]
            if stored.dtype != np.float32:
                raise MemberCacheContractError(f"part {path} is not float32")
            values = np.asarray(stored).copy()
            expected_hash = str(scalar("tp_members_weekly_sha256"))
    except MemberCacheContractError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise MemberCacheContractError(f"cannot read cache part {path}: {error}") from error
    if values.shape != MEMBER_FIELD_SHAPE:
        raise MemberCacheContractError(
            f"part {path} shape {values.shape}; expected {MEMBER_FIELD_SHAPE}"
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise MemberCacheContractError(f"part {path} contains invalid rainfall")
    if _array_sha256(values) != expected_hash:
        raise MemberCacheContractError(f"part {path} data checksum differs")
    return values


def build_part(
    group: Any,
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
    parts_dir: Path = DEFAULT_PARTS_DIR,
) -> tuple[Path, bool]:
    """Build one atomic part or safely reuse an already valid part."""

    path = _part_path(parts_dir, initialization).resolve()
    if path.is_file():
        try:
            _load_part(path, contract, source_index, initialization)
            return path, False
        except MemberCacheContractError as error:
            print(f"rebuilding invalid part {path}: {error}", flush=True)
    if not bool(np.asarray(group["init_complete"][int(source_index)]).item()):
        raise MemberCacheContractError(
            f"source initialization {initialization} is not marked complete"
        )
    values = _extract_initialization(
        group["forecast"],
        source_index,
        contract.latitude_slice,
        contract.longitude_slice,
    )
    _atomic_npz(path, _part_payload(contract, source_index, initialization, values))
    _load_part(path, contract, source_index, initialization)
    return path, True


def _effective_limit(count: int, *, max_initializations: int | None) -> int:
    """Validate an optional chronological-prefix diagnostic limit."""

    requested = max_initializations
    if requested is None:
        return count
    if requested < 1:
        raise MemberCacheContractError("--max-inits must be positive")
    if requested > count:
        raise MemberCacheContractError(
            f"requested {requested} initializations but the source has {count}"
        )
    return int(requested)


def _evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Choose a deterministic, endpoint-inclusive subset without replacement."""

    indices = np.asarray(indices, dtype=np.int64)
    if count < 1 or count > len(indices):
        raise MemberCacheContractError(
            f"cannot choose {count} smoke cases from {len(indices)} candidates"
        )
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    selected = indices[positions]
    if np.unique(selected).size != count:
        raise MemberCacheContractError("smoke-case spacing produced duplicate indices")
    return selected


def _smoke_source_indices(contract: SourceContract) -> np.ndarray:
    """Select 32/16/16 real cases across the train/validation/test eras."""

    years = _initialization_years(contract.initializations)
    forecast_ends = contract.initializations + np.timedelta64(41, "D")
    groups = (
        (
            np.flatnonzero(
                (years >= 2002)
                & (years <= 2017)
                & (forecast_ends < np.datetime64("2018-01-01", "D"))
            ),
            SMOKE_SPLIT_COUNTS["train_2002_2017"],
        ),
        (
            np.flatnonzero(
                (years >= 2018)
                & (years <= 2019)
                & (forecast_ends < np.datetime64("2020-01-01", "D"))
            ),
            SMOKE_SPLIT_COUNTS["validation_2018_2019"],
        ),
        (
            np.flatnonzero((years >= 2020) & (years <= 2021)),
            SMOKE_SPLIT_COUNTS["test_2020_2021"],
        ),
    )
    selected = np.sort(
        np.concatenate(
            [_evenly_spaced_indices(indices, count) for indices, count in groups]
        )
    )
    if selected.shape != (SMOKE_INITIALIZATION_COUNT,):
        raise MemberCacheContractError("smoke selection has the wrong case count")
    return selected


def _effective_scope_positions(
    contract: SourceContract, *, max_initializations: int | None, smoke: bool
) -> np.ndarray:
    if smoke and max_initializations is not None:
        raise MemberCacheContractError("--smoke cannot be combined with --max-inits")
    if smoke:
        return _smoke_source_indices(contract)
    limit = _effective_limit(
        len(contract.initializations), max_initializations=max_initializations
    )
    return np.arange(limit, dtype=np.int64)


def _scope_records(
    contract: SourceContract,
    *,
    max_initializations: int | None = None,
    smoke: bool = False,
) -> list[tuple[int, np.datetime64]]:
    """Bound the global prefix before any worker striding."""

    all_records = [
        (int(index), np.datetime64(initialization, "D"))
        for index, initialization in zip(
            contract.source_indices, contract.initializations, strict=True
        )
    ]
    positions = _effective_scope_positions(
        contract, max_initializations=max_initializations, smoke=smoke
    )
    return [all_records[int(position)] for position in positions]


def _selected_task_records(
    contract: SourceContract,
    *,
    task_index: int | None,
    task_count: int | None,
    initialization: str | None,
    max_initializations: int | None,
    smoke: bool,
) -> list[tuple[int, np.datetime64]]:
    records = _scope_records(
        contract, max_initializations=max_initializations, smoke=smoke
    )
    if initialization is not None:
        if task_index is not None or task_count is not None or smoke or max_initializations:
            raise MemberCacheContractError(
                "--init cannot be combined with task striding, --smoke, or --max-inits"
            )
        requested = np.datetime64(initialization, "D")
        matches = [record for record in records if record[1] == requested]
        if len(matches) != 1:
            raise MemberCacheContractError(
                f"{initialization} is not a 2002--2021 FuXi initialization"
            )
        return matches
    if task_index is None or task_count is None:
        raise MemberCacheContractError(
            "build requires either --init or both --task-index and --task-count"
        )
    if task_count < 1:
        raise MemberCacheContractError("--task-count must be positive")
    if task_index < 0 or task_index >= task_count:
        raise MemberCacheContractError(
            f"task index {task_index} is outside [0, {task_count})"
        )
    if task_count == 1 and len(records) == len(contract.initializations):
        raise MemberCacheContractError(
            "refusing a serial full-archive build; use at least two array tasks"
        )
    return records[task_index::task_count]


def _sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    output = Path(output).resolve()
    if output.suffix != ".npy":
        raise MemberCacheContractError("final cache output must end in .npy")
    base = output.with_suffix("")
    return (
        base.with_name(base.name + ".metadata.json"),
        base.with_name(base.name + ".manifest.json"),
        base.with_name(base.name + ".sha256"),
    )


def _write_atomic_memmap(
    output: Path,
    shape: tuple[int, ...],
    rows: Sequence[tuple[Path, SourceContract, int, np.datetime64, str]],
) -> str:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.{os.getpid()}.",
        suffix=".temporary",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        target = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=shape
        )
        for output_index, (part, contract, source_index, initialization, checksum) in enumerate(
            rows
        ):
            if sha256_file(part) != checksum:
                raise MemberCacheContractError(
                    f"part {part} changed during finalization"
                )
            target[output_index] = _load_part(
                part, contract, source_index, initialization
            )
        target.flush()
        del target
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        checksum = sha256_file(temporary)
        os.replace(temporary, output)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)


def finalize_cache(
    contract: SourceContract,
    parts_dir: Path = DEFAULT_PARTS_DIR,
    output: Path = DEFAULT_CACHE,
    *,
    max_initializations: int | None = None,
    smoke: bool = False,
) -> CacheArtifacts:
    """Validate the requested scope and publish a memory-mappable final cache."""

    scope = _scope_records(
        contract, max_initializations=max_initializations, smoke=smoke
    )
    rows: list[tuple[Path, SourceContract, int, np.datetime64, str]] = []
    manifest_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for output_index, (source_index, initialization) in enumerate(scope):
        part = _part_path(parts_dir, initialization).resolve()
        if not part.is_file():
            missing.append(np.datetime_as_string(initialization, unit="D"))
            continue
        values = _load_part(part, contract, source_index, initialization)
        part_checksum = sha256_file(part)
        rows.append((part, contract, source_index, initialization, part_checksum))
        manifest_records.append(
            {
                "output_index": output_index,
                "source_init_index": source_index,
                "initialization": np.datetime_as_string(initialization, unit="D"),
                "part_path": str(part),
                "part_sha256": part_checksum,
                "tp_members_weekly_sha256": _array_sha256(values),
            }
        )
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise MemberCacheContractError(
            f"cannot finalize: {len(missing)} of {len(scope)} parts are missing "
            f"({preview}{suffix})"
        )

    output = Path(output).resolve()
    metadata_path, manifest_path, checksums_path = _sidecar_paths(output)
    shape = (len(scope), *MEMBER_FIELD_SHAPE)
    data_checksum = _write_atomic_memmap(output, shape, rows)
    initializations = np.asarray([record[1] for record in scope], dtype="datetime64[D]")
    source_indices = np.asarray([record[0] for record in scope], dtype=np.int64)
    is_full_archive = len(scope) == INITIALIZATION_COUNT
    if smoke:
        scope_name = "stratified_32_train_16_validation_16_test_non_scientific"
    elif is_full_archive:
        scope_name = "full_archive"
    else:
        scope_name = "bounded_prefix_non_scientific"
    metadata: dict[str, Any] = {
        "schema_name": CACHE_SCHEMA_NAME,
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_store": contract.source_store,
        "source_fingerprint": contract.source_fingerprint,
        "selection": (
            "all-season source initializations from 2002-2021; scope states "
            "whether the artifact contains the full archive or a diagnostic subset"
        ),
        "scope": scope_name,
        "full_archive": is_full_archive,
        "initialization_count": len(scope),
        "initializations": np.datetime_as_string(initializations, unit="D").tolist(),
        "source_init_indices": source_indices.tolist(),
        "shape": list(shape),
        "dtype": np.dtype(np.float32).str,
        "dims": list(OUTPUT_DIMS),
        "member_labels": list(range(MEMBER_COUNT)),
        "lead_week_labels": list(range(1, LEAD_WEEK_COUNT + 1)),
        "latitude": contract.latitude.tolist(),
        "longitude": contract.longitude.tolist(),
        "input_variable": "tp",
        "input_units": INPUT_UNITS,
        "output_units": OUTPUT_UNITS,
        "conversion": CONVERSION_DESCRIPTION,
        "normalization": "none",
        "ensemble_aggregation": "none; all 51 members retained",
        "data_file": output.name,
        "data_sha256": data_checksum,
        "manifest_file": manifest_path.name,
        "checksums_file": checksums_path.name,
    }
    manifest: dict[str, Any] = {
        "schema_name": CACHE_SCHEMA_NAME,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": contract.source_fingerprint,
        "initialization_count": len(scope),
        "records": manifest_records,
    }
    _atomic_json(metadata_path, metadata)
    _atomic_json(manifest_path, manifest)
    metadata_checksum = sha256_file(metadata_path)
    manifest_checksum = sha256_file(manifest_path)
    checksum_text = (
        f"{data_checksum}  {output.name}\n"
        f"{metadata_checksum}  {metadata_path.name}\n"
        f"{manifest_checksum}  {manifest_path.name}\n"
    )
    _atomic_text(checksums_path, checksum_text)
    artifacts = CacheArtifacts(
        data=output,
        metadata=metadata_path,
        manifest=manifest_path,
        checksums=checksums_path,
        data_sha256=data_checksum,
        metadata_sha256=metadata_checksum,
        manifest_sha256=manifest_checksum,
    )
    verify_cache(output)
    return artifacts


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemberCacheContractError(f"cannot read JSON sidecar {path}: {error}") from error
    if not isinstance(value, dict):
        raise MemberCacheContractError(f"JSON sidecar {path} is not an object")
    return value


def _read_checksums(path: Path) -> Mapping[str, str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MemberCacheContractError(f"cannot read checksum file {path}: {error}") from error
    result: dict[str, str] = {}
    for line in lines:
        pieces = line.split(None, 1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise MemberCacheContractError(f"invalid checksum line in {path}: {line!r}")
        checksum, name = pieces[0].lower(), pieces[1].strip()
        if any(character not in "0123456789abcdef" for character in checksum):
            raise MemberCacheContractError(f"invalid SHA-256 in {path}: {checksum!r}")
        if name in result:
            raise MemberCacheContractError(f"duplicate checksum entry for {name}")
        result[name] = checksum
    return result


def verify_cache(
    output: Path = DEFAULT_CACHE,
    *,
    metadata_path: Path | None = None,
    manifest_path: Path | None = None,
    checksums_path: Path | None = None,
    expected_source_fingerprint: str | None = None,
) -> Mapping[str, Any]:
    """Verify sidecars, whole-file hashes, schema, alignment, and data values."""

    output = Path(output).resolve()
    default_metadata, default_manifest, default_checksums = _sidecar_paths(output)
    metadata_path = Path(metadata_path or default_metadata).resolve()
    manifest_path = Path(manifest_path or default_manifest).resolve()
    checksums_path = Path(checksums_path or default_checksums).resolve()
    required_paths = (output, metadata_path, manifest_path, checksums_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise MemberCacheContractError(f"cache artifacts are missing: {missing}")

    checksums = _read_checksums(checksums_path)
    expected_names = {output.name, metadata_path.name, manifest_path.name}
    if set(checksums) != expected_names:
        raise MemberCacheContractError(
            f"checksum entries {sorted(checksums)} differ from {sorted(expected_names)}"
        )
    actual_checksums = {
        output.name: sha256_file(output),
        metadata_path.name: sha256_file(metadata_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    if checksums != actual_checksums:
        raise MemberCacheContractError("one or more final cache checksums differ")

    metadata = _load_json(metadata_path)
    manifest = _load_json(manifest_path)
    for payload, name in ((metadata, "metadata"), (manifest, "manifest")):
        if (
            payload.get("schema_name") != CACHE_SCHEMA_NAME
            or int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION
        ):
            raise MemberCacheContractError(f"{name} schema differs")
    source_fingerprint = str(metadata.get("source_fingerprint", ""))
    if (
        not source_fingerprint
        or manifest.get("source_fingerprint") != source_fingerprint
    ):
        raise MemberCacheContractError("metadata/manifest source fingerprints differ")
    if (
        expected_source_fingerprint is not None
        and source_fingerprint != expected_source_fingerprint
    ):
        raise MemberCacheContractError("final cache source fingerprint is stale")
    if metadata.get("data_sha256") != actual_checksums[output.name]:
        raise MemberCacheContractError("metadata data checksum differs")
    if metadata.get("dims") != list(OUTPUT_DIMS):
        raise MemberCacheContractError("final cache dimension order differs")
    if (
        metadata.get("input_variable") != "tp"
        or metadata.get("input_units") != INPUT_UNITS
        or metadata.get("output_units") != OUTPUT_UNITS
        or metadata.get("conversion") != CONVERSION_DESCRIPTION
    ):
        raise MemberCacheContractError("final cache TP conversion differs")
    if (
        metadata.get("normalization") != "none"
        or metadata.get("ensemble_aggregation")
        != "none; all 51 members retained"
    ):
        raise MemberCacheContractError("final cache was normalized or ensemble-aggregated")
    if (
        metadata.get("member_labels") != list(range(MEMBER_COUNT))
        or metadata.get("lead_week_labels")
        != list(range(1, LEAD_WEEK_COUNT + 1))
    ):
        raise MemberCacheContractError("final cache member/lead labels differ")
    if (
        metadata.get("latitude") != EXPECTED_LATITUDE.tolist()
        or metadata.get("longitude") != EXPECTED_LONGITUDE.tolist()
    ):
        raise MemberCacheContractError("final cache grid differs")
    if (
        metadata.get("data_file") != output.name
        or metadata.get("manifest_file") != manifest_path.name
        or metadata.get("checksums_file") != checksums_path.name
    ):
        raise MemberCacheContractError("final cache sidecar names differ")

    try:
        data = np.load(output, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise MemberCacheContractError(f"cannot memory-map final cache: {error}") from error
    count = int(metadata.get("initialization_count", -1))
    expected_shape = (count, *MEMBER_FIELD_SHAPE)
    if tuple(metadata.get("shape", ())) != expected_shape or data.shape != expected_shape:
        raise MemberCacheContractError(
            f"final cache shape differs: metadata={metadata.get('shape')}, data={data.shape}"
        )
    if data.dtype != np.float32 or metadata.get("dtype") != np.dtype(np.float32).str:
        raise MemberCacheContractError("final cache is not float32")

    initializations = np.asarray(
        metadata.get("initializations", []), dtype="datetime64[D]"
    )
    source_indices = np.asarray(
        metadata.get("source_init_indices", []), dtype=np.int64
    )
    if initializations.shape != (count,) or source_indices.shape != (count,):
        raise MemberCacheContractError("final cache initialization metadata differs")
    if count and (
        np.isnat(initializations).any()
        or np.unique(initializations).size != count
        or np.any(initializations[1:] <= initializations[:-1])
    ):
        raise MemberCacheContractError("final cache initializations are invalid")
    if count and (
        np.unique(source_indices).size != count
        or np.any(source_indices[1:] <= source_indices[:-1])
        or source_indices[0] < 0
        or source_indices[-1] >= INITIALIZATION_COUNT
    ):
        raise MemberCacheContractError("final cache source indices are invalid")
    scope_name = metadata.get("scope")
    if bool(metadata.get("full_archive")):
        if scope_name != "full_archive" or not np.array_equal(
            source_indices, np.arange(INITIALIZATION_COUNT, dtype=np.int64)
        ):
            raise MemberCacheContractError("full cache scope/source indices differ")
        _validate_allseason_initializations(initializations)
    elif count >= INITIALIZATION_COUNT:
        raise MemberCacheContractError("non-full cache has an invalid scope size")
    elif scope_name == "bounded_prefix_non_scientific":
        if not np.array_equal(source_indices, np.arange(count, dtype=np.int64)):
            raise MemberCacheContractError("bounded cache is not a source prefix")
    elif scope_name == "stratified_32_train_16_validation_16_test_non_scientific":
        years = _initialization_years(initializations)
        ends = initializations + np.timedelta64(41, "D")
        train = years <= 2017
        validation = (years >= 2018) & (years <= 2019)
        test = years >= 2020
        if (
            count != SMOKE_INITIALIZATION_COUNT
            or int(np.count_nonzero(train)) != SMOKE_SPLIT_COUNTS["train_2002_2017"]
            or int(np.count_nonzero(validation))
            != SMOKE_SPLIT_COUNTS["validation_2018_2019"]
            or int(np.count_nonzero(test)) != SMOKE_SPLIT_COUNTS["test_2020_2021"]
            or np.any(ends[train] >= np.datetime64("2018-01-01", "D"))
            or np.any(ends[validation] >= np.datetime64("2020-01-01", "D"))
        ):
            raise MemberCacheContractError("stratified smoke scope differs")
    else:
        raise MemberCacheContractError(f"unsupported non-full cache scope: {scope_name!r}")

    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != count
        or manifest.get("initialization_count") != count
    ):
        raise MemberCacheContractError("final manifest record count differs")
    for index, (record, initialization) in enumerate(
        zip(records, initializations, strict=True)
    ):
        if not isinstance(record, dict):
            raise MemberCacheContractError("final manifest contains a non-object record")
        if (
            record.get("output_index") != index
            or record.get("source_init_index") != int(source_indices[index])
            or record.get("initialization")
            != np.datetime_as_string(initialization, unit="D")
        ):
            raise MemberCacheContractError(
                f"final manifest alignment differs at output index {index}"
            )
        for hash_name in ("part_sha256", "tp_members_weekly_sha256"):
            checksum = record.get(hash_name)
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise MemberCacheContractError(
                    f"final manifest {hash_name} differs at output index {index}"
                )

    for start in range(0, count, 8):
        block = np.asarray(data[start : start + 8])
        if not np.isfinite(block).all() or np.any(block < 0.0):
            raise MemberCacheContractError(
                f"final cache contains invalid rainfall near row {start}"
            )
    return {
        "status": "verified",
        "data": str(output),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "source_fingerprint": source_fingerprint,
        "data_sha256": actual_checksums[output.name],
        "full_archive": bool(metadata.get("full_archive")),
    }


def load_member_cache(
    output: Path = DEFAULT_CACHE,
    *,
    verify: bool = True,
    expected_source_fingerprint: str | None = None,
) -> tuple[np.memmap, Mapping[str, Any]]:
    """Return the read-only memmap and its alignment metadata."""

    output = Path(output).resolve()
    metadata_path, _, _ = _sidecar_paths(output)
    if verify:
        verify_cache(
            output, expected_source_fingerprint=expected_source_fingerprint
        )
    metadata = _load_json(metadata_path)
    data = np.load(output, mmap_mode="r", allow_pickle=False)
    return data, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="validate source metadata")
    inventory.add_argument("--max-inits", type=int)
    inventory.add_argument("--smoke", action="store_true")

    build = commands.add_parser(
        "build", help="build one initialization or a strided worker shard"
    )
    build.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    build.add_argument("--task-index", type=int)
    build.add_argument("--task-count", type=int)
    build.add_argument("--init")
    build.add_argument("--max-inits", type=int)
    build.add_argument("--smoke", action="store_true")

    finalize = commands.add_parser(
        "finalize", help="validate parts and publish NPY plus audited sidecars"
    )
    finalize.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    finalize.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    finalize.add_argument("--max-inits", type=int)
    finalize.add_argument("--smoke", action="store_true")

    verify = commands.add_parser("verify", help="verify the final cache")
    verify.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    verify.add_argument("--metadata", type=Path)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--checksums", type=Path)
    verify.add_argument("--check-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "verify":
        expected_fingerprint = None
        if args.check_source:
            _, contract = inspect_source(args.source_store)
            expected_fingerprint = contract.source_fingerprint
        report = verify_cache(
            args.output,
            metadata_path=args.metadata,
            manifest_path=args.manifest,
            checksums_path=args.checksums,
            expected_source_fingerprint=expected_fingerprint,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return

    group, contract = inspect_source(args.source_store)
    if args.command == "inventory":
        scope = _scope_records(
            contract, max_initializations=args.max_inits, smoke=args.smoke
        )
        print(
            json.dumps(
                {
                    "source_store": contract.source_store,
                    "source_fingerprint": contract.source_fingerprint,
                    "source_initialization_count": len(contract.initializations),
                    "selected_initialization_count": len(scope),
                    "first_initialization": np.datetime_as_string(scope[0][1], unit="D"),
                    "last_initialization": np.datetime_as_string(scope[-1][1], unit="D"),
                    "members": MEMBER_COUNT,
                    "lead_weeks": LEAD_WEEK_COUNT,
                    "grid_shape": list(GRID_SHAPE),
                    "expected_output_shape": [len(scope), *MEMBER_FIELD_SHAPE],
                    "recommended_array_tasks": 260,
                    "recommended_max_concurrent": 6,
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
            smoke=args.smoke,
        )
        for ordinal, (source_index, initialization) in enumerate(records, start=1):
            path, created = build_part(
                group, contract, source_index, initialization, args.parts_dir
            )
            action = "built" if created else "reused"
            print(
                f"[{ordinal}/{len(records)}] {action} "
                f"{np.datetime_as_string(initialization, unit='D')}: {path}",
                flush=True,
            )
        return
    artifacts = finalize_cache(
        contract,
        args.parts_dir,
        args.output,
        max_initializations=args.max_inits,
        smoke=args.smoke,
    )
    print(f"published {artifacts.data}", flush=True)
    print(f"sha256={artifacts.data_sha256}", flush=True)
    print(f"completion={artifacts.checksums}", flush=True)


if __name__ == "__main__":
    main()
