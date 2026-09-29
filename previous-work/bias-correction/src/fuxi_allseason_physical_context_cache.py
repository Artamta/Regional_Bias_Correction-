#!/usr/bin/env python
"""Build an audited all-season FuXi weekly physical-context cache.

The cache contains predictor-only FuXi fields for every 2002--2021 source
initialization.  Each field is reduced to the arithmetic mean across the 51
members and seven lead days for each of six lead weeks.  Moisture-flux fields
instead average the daily, per-member product, exactly matching the accepted
``fuxi_physical_feature_cache`` convention.

The build follows the all-season precipitation-member cache contract: array
workers atomically publish one compressed part per initialization, and a
separate finalizer validates every requested part before streaming a float32
NPY memmap plus metadata, manifest, and SHA-256 completion sidecars.  The
cache is intentionally raw and unnormalized; downstream models must fit any
normalization using training rows only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fuxi_allseason_member_cache as member_cache
from fuxi_physical_feature_cache import (
    PhysicalCacheContractError,
    _daily_product_ensemble_mean as _accepted_daily_product_ensemble_mean,
    _weekly_ensemble_mean as _accepted_weekly_ensemble_mean,
)
from project_paths import PROJECT_ROOT


DEFAULT_SOURCE_STORE = member_cache.DEFAULT_SOURCE_STORE
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "cache"
    / "fuxi_physical_context_weekly_2002_2021_allseason_v1.npy"
)
DEFAULT_PARTS_DIR = (
    PROJECT_ROOT
    / "cache"
    / "fuxi_physical_context_weekly_2002_2021_allseason_v1.parts"
)

CACHE_SCHEMA_NAME = "fuxi-physical-context-weekly-allseason"
CACHE_SCHEMA_VERSION = 1

START_YEAR = member_cache.START_YEAR
END_YEAR = member_cache.END_YEAR
YEARS = member_cache.YEARS
EXPECTED_INITIALIZATIONS_PER_YEAR = (
    member_cache.EXPECTED_INITIALIZATIONS_PER_YEAR
)
INITIALIZATION_COUNT = member_cache.INITIALIZATION_COUNT
MEMBER_COUNT = member_cache.MEMBER_COUNT
LEAD_DAY_COUNT = member_cache.LEAD_DAY_COUNT
LEAD_WEEK_COUNT = member_cache.LEAD_WEEK_COUNT
DAYS_PER_WEEK = member_cache.DAYS_PER_WEEK
SOURCE_CHANNEL_CHUNK = 4
SOURCE_CHANNEL_NAMES = member_cache.SOURCE_CHANNEL_NAMES
EXPECTED_LATITUDE = member_cache.EXPECTED_LATITUDE
EXPECTED_LONGITUDE = member_cache.EXPECTED_LONGITUDE
GRID_SHAPE = member_cache.GRID_SHAPE
SMOKE_SPLIT_COUNTS = {
    "train_2002_2017": 1,
    "validation_2018_2019": 1,
    "test_2020_2021": 1,
}
SMOKE_INITIALIZATION_COUNT = sum(SMOKE_SPLIT_COUNTS.values())

# This order is a frozen modeling contract.  It is not alphabetical and must
# not be inferred from dictionaries or source-channel order.
PHYSICAL_CONTEXT_FEATURE_NAMES = (
    "t2m_mean",
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
FEATURE_NAMES = PHYSICAL_CONTEXT_FEATURE_NAMES
FEATURE_COUNT = len(PHYSICAL_CONTEXT_FEATURE_NAMES)
RAW_CHANNEL_NAMES = (
    "z500",
    "u850",
    "v850",
    "q850",
    "t2m",
    "ttr",
    "msl",
    "tcwv",
)
PHYSICAL_CONTEXT_FEATURE_UNITS = {
    "t2m_mean": "K",
    "tcwv_mean": "kg m-2",
    "q850_mean": "kg kg-1",
    "u850_mean": "m s-1",
    "v850_mean": "m s-1",
    "q850_u850_flux_mean": "kg kg-1 m s-1",
    "q850_v850_flux_mean": "kg kg-1 m s-1",
    "z500_mean": "m2 s-2",
    "msl_mean": "Pa",
    "olr_mean": "W m-2",
}
PHYSICAL_CONTEXT_FEATURE_DEFINITIONS = {
    "t2m_mean": (
        "Mean across 51 member-wise seven-day means of native FuXi t2m"
    ),
    "tcwv_mean": (
        "Mean across 51 member-wise seven-day means of FuXi TCWV after "
        "clipping native values to >=0"
    ),
    "q850_mean": (
        "Mean across 51 member-wise seven-day means of FuXi q850 after "
        "clipping native values to >=0"
    ),
    "u850_mean": (
        "Mean across 51 member-wise seven-day means of native FuXi u850"
    ),
    "v850_mean": (
        "Mean across 51 member-wise seven-day means of native FuXi v850"
    ),
    "q850_u850_flux_mean": (
        "Mean across seven days and 51 members of the daily per-member "
        "FuXi product max(q850,0)*u850"
    ),
    "q850_v850_flux_mean": (
        "Mean across seven days and 51 members of the daily per-member "
        "FuXi product max(q850,0)*v850"
    ),
    "z500_mean": (
        "Mean across 51 member-wise seven-day means of native FuXi z500"
    ),
    "msl_mean": (
        "Mean across 51 member-wise seven-day means of native FuXi MSL"
    ),
    "olr_mean": (
        "Positive outgoing longwave radiation: negative of the mean across "
        "51 member-wise seven-day means of native FuXi TTR"
    ),
}
PHYSICAL_CONTEXT_TRANSFORMS = {
    "tcwv": "maximum(native_tcwv, 0) before weekly averaging",
    "q850": (
        "maximum(native_q850, 0) before weekly averaging and daily "
        "moisture-flux products"
    ),
    "ttr": "olr=-native_ttr",
    "other_fields": "native float32 values; no unit conversion",
}
TEMPORAL_AGGREGATION = (
    "arithmetic mean over each member's seven daily values, followed by an "
    "arithmetic mean across all 51 members"
)
ENSEMBLE_AGGREGATION = (
    "arithmetic mean across all 51 members; no member subsampling"
)
FLUX_AGGREGATION = (
    "arithmetic mean of daily per-member q850*wind products over seven days "
    "and all 51 members"
)
OUTPUT_DIMS = ("init", "lead_week", "feature", "lat", "lon")
CONTEXT_FIELD_SHAPE = (LEAD_WEEK_COUNT, FEATURE_COUNT, *GRID_SHAPE)

FEATURE_CONTRACT = {
    "feature_names": list(PHYSICAL_CONTEXT_FEATURE_NAMES),
    "feature_units": dict(PHYSICAL_CONTEXT_FEATURE_UNITS),
    "feature_definitions": dict(PHYSICAL_CONTEXT_FEATURE_DEFINITIONS),
    "transforms": dict(PHYSICAL_CONTEXT_TRANSFORMS),
    "temporal_aggregation": TEMPORAL_AGGREGATION,
    "ensemble_aggregation": ENSEMBLE_AGGREGATION,
    "flux_aggregation": FLUX_AGGREGATION,
    "normalization": "none",
    "output_dims": list(OUTPUT_DIMS),
}
FEATURE_CONTRACT_SHA256 = member_cache._canonical_sha256(FEATURE_CONTRACT)

# Reuse the fully validated all-season source, scope, and artifact contracts.
PhysicalContextCacheContractError = member_cache.MemberCacheContractError
SourceContract = member_cache.SourceContract
CacheArtifacts = member_cache.CacheArtifacts
sha256_file = member_cache.sha256_file
_array_sha256 = member_cache._array_sha256
_atomic_text = member_cache._atomic_text
_atomic_json = member_cache._atomic_json
_atomic_npz = member_cache._atomic_npz
_load_json = member_cache._load_json
_read_checksums = member_cache._read_checksums
_initialization_years = member_cache._initialization_years
_validate_allseason_initializations = (
    member_cache._validate_allseason_initializations
)


def _smoke_source_indices(contract: SourceContract) -> np.ndarray:
    """Choose one deterministic, split-safe real case from each era."""

    years = _initialization_years(contract.initializations)
    forecast_ends = contract.initializations + np.timedelta64(41, "D")
    groups = (
        np.flatnonzero(
            (years >= 2002)
            & (years <= 2017)
            & (forecast_ends < np.datetime64("2018-01-01", "D"))
        ),
        np.flatnonzero(
            (years >= 2018)
            & (years <= 2019)
            & (forecast_ends < np.datetime64("2020-01-01", "D"))
        ),
        np.flatnonzero((years >= 2020) & (years <= 2021)),
    )
    if any(len(indices) == 0 for indices in groups):
        raise PhysicalContextCacheContractError(
            "cannot form the three-era physical-cache smoke selection"
        )
    selected = np.asarray(
        [indices[len(indices) // 2] for indices in groups], dtype=np.int64
    )
    if selected.shape != (SMOKE_INITIALIZATION_COUNT,) or np.any(
        selected[1:] <= selected[:-1]
    ):
        raise PhysicalContextCacheContractError(
            "physical-cache smoke selection is invalid"
        )
    return selected


def _scope_records(
    contract: SourceContract,
    *,
    max_initializations: int | None = None,
    smoke: bool = False,
) -> list[tuple[int, np.datetime64]]:
    """Select the full archive, a diagnostic prefix, or three smoke cases."""

    if smoke and max_initializations is not None:
        raise PhysicalContextCacheContractError(
            "--smoke cannot be combined with --max-inits"
        )
    if smoke:
        positions = _smoke_source_indices(contract)
    else:
        count = len(contract.initializations)
        if max_initializations is None:
            limit = count
        else:
            if max_initializations < 1:
                raise PhysicalContextCacheContractError(
                    "--max-inits must be positive"
                )
            if max_initializations > count:
                raise PhysicalContextCacheContractError(
                    f"requested {max_initializations} initializations but the "
                    f"source has {count}"
                )
            limit = int(max_initializations)
        positions = np.arange(limit, dtype=np.int64)
    return [
        (
            int(contract.source_indices[int(position)]),
            np.datetime64(contract.initializations[int(position)], "D"),
        )
        for position in positions
    ]


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
        if (
            task_index is not None
            or task_count is not None
            or smoke
            or max_initializations is not None
        ):
            raise PhysicalContextCacheContractError(
                "--init cannot be combined with task striding, --smoke, or "
                "--max-inits"
            )
        requested = np.datetime64(initialization, "D")
        matches = [record for record in records if record[1] == requested]
        if len(matches) != 1:
            raise PhysicalContextCacheContractError(
                f"{initialization} is not a 2002--2021 FuXi initialization"
            )
        return matches
    if task_index is None or task_count is None:
        raise PhysicalContextCacheContractError(
            "build requires either --init or both --task-index and --task-count"
        )
    if task_count < 1:
        raise PhysicalContextCacheContractError(
            "--task-count must be positive"
        )
    if task_index < 0 or task_index >= task_count:
        raise PhysicalContextCacheContractError(
            f"task index {task_index} is outside [0, {task_count})"
        )
    if task_count == 1 and len(records) == len(contract.initializations):
        raise PhysicalContextCacheContractError(
            "refusing a serial full-archive build; use at least two array tasks"
        )
    return records[task_index::task_count]


def _source_fingerprint(group: Any, contract: SourceContract) -> str:
    """Bind the native source identity to this exact physical-field contract."""

    forecast = group["forecast"]
    attrs = dict(group.attrs)
    payload = {
        "source_store": contract.source_store,
        "source_schema_version": str(attrs.get("schema_version", "")),
        "source_status": str(attrs.get("status", "")),
        "archive_manifest_sha256": str(
            attrs.get("archive_manifest_sha256", "")
        ),
        "archive_records_sha256": str(
            attrs.get("archive_records_sha256", "")
        ),
        "completed_utc": str(attrs.get("completed_utc", "")),
        "forecast_shape": list(map(int, forecast.shape)),
        "forecast_chunks": list(map(int, forecast.chunks)),
        "forecast_dtype": np.dtype(forecast.dtype).str,
        "initializations": np.datetime_as_string(
            contract.initializations, unit="D"
        ).tolist(),
        "channel_names": list(contract.channel_names),
        "latitude": contract.latitude.tolist(),
        "longitude": contract.longitude.tolist(),
        "cache_schema_name": CACHE_SCHEMA_NAME,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
    }
    return member_cache._canonical_sha256(payload)


def inspect_source(
    source_store: Path = DEFAULT_SOURCE_STORE,
) -> tuple[Any, SourceContract]:
    """Validate the complete native archive and return this cache contract."""

    group, base_contract = member_cache.inspect_source(source_store)
    contract = replace(
        base_contract,
        source_fingerprint=_source_fingerprint(group, base_contract),
    )
    return group, contract


def _weekly_ensemble_mean(values: np.ndarray) -> np.ndarray:
    """Apply the accepted member-week then ensemble-mean reduction."""

    try:
        return _accepted_weekly_ensemble_mean(values)
    except PhysicalCacheContractError as error:
        raise PhysicalContextCacheContractError(str(error)) from error


def _daily_product_ensemble_mean(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Apply the accepted daily product then member/week mean reduction."""

    try:
        return _accepted_daily_product_ensemble_mean(first, second)
    except PhysicalCacheContractError as error:
        raise PhysicalContextCacheContractError(str(error)) from error


def _feature_index(name: str) -> int:
    return PHYSICAL_CONTEXT_FEATURE_NAMES.index(name)


def _validate_context_values(
    values: np.ndarray, *, expected_shape: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise PhysicalContextCacheContractError(
            f"physical context is {array.dtype}, expected float32"
        )
    if array.shape != expected_shape:
        raise PhysicalContextCacheContractError(
            f"physical-context shape {array.shape}; expected {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise PhysicalContextCacheContractError(
            "physical context contains non-finite values"
        )
    feature_axis = len(expected_shape) - 3
    tcwv = np.take(array, _feature_index("tcwv_mean"), axis=feature_axis)
    q850 = np.take(array, _feature_index("q850_mean"), axis=feature_axis)
    olr = np.take(array, _feature_index("olr_mean"), axis=feature_axis)
    if np.any(tcwv < 0.0):
        raise PhysicalContextCacheContractError("TCWV mean is negative")
    if np.any(q850 < 0.0):
        raise PhysicalContextCacheContractError("q850 mean is negative")
    if np.any(olr < 0.0):
        raise PhysicalContextCacheContractError(
            "OLR is negative after the required OLR=-TTR transform"
        )
    return array


def _summarize_initialization(
    forecast: Any,
    source_index: int,
    channel_names: Sequence[str],
    latitude_slice: slice,
    longitude_slice: slice,
) -> np.ndarray:
    """Read 42 native chunks once and reduce one initialization.

    The global archive stores four adjacent channels per chunk.  Reading all
    seven channel groups for each lead week avoids rereading chunks while
    retaining only u850 and v850 until q850 arrives for daily moisture fluxes.
    """

    channel_names = tuple(channel_names)
    if channel_names != SOURCE_CHANNEL_NAMES:
        raise PhysicalContextCacheContractError("source channel order has changed")
    channel_indices = {
        name: channel_names.index(name) for name in RAW_CHANNEL_NAMES
    }
    by_feature: dict[str, list[np.ndarray]] = {
        name: [] for name in PHYSICAL_CONTEXT_FEATURE_NAMES
    }
    for lead_week in range(LEAD_WEEK_COUNT):
        lead_slice = slice(
            lead_week * DAYS_PER_WEEK,
            (lead_week + 1) * DAYS_PER_WEEK,
        )
        retained_u850: np.ndarray | None = None
        retained_v850: np.ndarray | None = None
        produced: set[str] = set()
        for group_start in range(0, len(channel_names), SOURCE_CHANNEL_CHUNK):
            group_stop = min(
                group_start + SOURCE_CHANNEL_CHUNK, len(channel_names)
            )
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
            expected_block_shape = (
                MEMBER_COUNT,
                DAYS_PER_WEEK,
                group_stop - group_start,
                *GRID_SHAPE,
            )
            if block.shape != expected_block_shape:
                raise PhysicalContextCacheContractError(
                    f"unexpected source chunk result shape {block.shape}; "
                    f"expected {expected_block_shape}"
                )
            if not np.isfinite(block).all():
                raise PhysicalContextCacheContractError(
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
                by_feature["z500_mean"].append(
                    _weekly_ensemble_mean(field("z500"))
                )
                produced.add("z500_mean")
            if "u850" in names_in_group:
                retained_u850 = field("u850").copy()
                by_feature["u850_mean"].append(
                    _weekly_ensemble_mean(retained_u850)
                )
                produced.add("u850_mean")
            if "v850" in names_in_group:
                retained_v850 = field("v850").copy()
                by_feature["v850_mean"].append(
                    _weekly_ensemble_mean(retained_v850)
                )
                produced.add("v850_mean")
            if "q850" in names_in_group:
                q850 = np.maximum(field("q850"), np.float32(0.0))
                if retained_u850 is None or retained_v850 is None:
                    raise PhysicalContextCacheContractError(
                        "u850/v850 must be read before q850 for moisture flux"
                    )
                by_feature["q850_mean"].append(_weekly_ensemble_mean(q850))
                by_feature["q850_u850_flux_mean"].append(
                    _daily_product_ensemble_mean(q850, retained_u850)
                )
                by_feature["q850_v850_flux_mean"].append(
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
            if "t2m" in names_in_group:
                by_feature["t2m_mean"].append(
                    _weekly_ensemble_mean(field("t2m"))
                )
                produced.add("t2m_mean")
            if "ttr" in names_in_group:
                by_feature["olr_mean"].append(
                    -_weekly_ensemble_mean(field("ttr"))
                )
                produced.add("olr_mean")
            if "msl" in names_in_group:
                by_feature["msl_mean"].append(
                    _weekly_ensemble_mean(field("msl"))
                )
                produced.add("msl_mean")
            if "tcwv" in names_in_group:
                tcwv = np.maximum(field("tcwv"), np.float32(0.0))
                by_feature["tcwv_mean"].append(
                    _weekly_ensemble_mean(tcwv)
                )
                produced.add("tcwv_mean")
        expected_features = set(PHYSICAL_CONTEXT_FEATURE_NAMES)
        if produced != expected_features:
            missing = sorted(expected_features.difference(produced))
            extra = sorted(produced.difference(expected_features))
            raise PhysicalContextCacheContractError(
                f"week {lead_week + 1} feature mismatch: "
                f"missing={missing}, extra={extra}"
            )

    stacked = np.stack(
        [np.stack(by_feature[name]) for name in PHYSICAL_CONTEXT_FEATURE_NAMES],
        axis=1,
    ).astype(np.float32, copy=False)
    return _validate_context_values(
        stacked, expected_shape=CONTEXT_FIELD_SHAPE
    )


def _part_path(parts_dir: Path, initialization: np.datetime64) -> Path:
    stamp = np.datetime_as_string(initialization, unit="D").replace("-", "")
    return Path(parts_dir) / f"{stamp}.npz"


def _part_payload(
    contract: SourceContract,
    source_index: int,
    initialization: np.datetime64,
    physical_context: np.ndarray,
) -> Mapping[str, Any]:
    values = _validate_context_values(
        physical_context, expected_shape=CONTEXT_FIELD_SHAPE
    )
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(contract.source_store),
        "source_fingerprint": np.asarray(contract.source_fingerprint),
        "source_init_index": np.asarray(source_index, dtype=np.int32),
        "initialization": np.asarray(initialization, dtype="datetime64[D]"),
        "latitude": np.asarray(contract.latitude, dtype=np.float64),
        "longitude": np.asarray(contract.longitude, dtype=np.float64),
        "feature_names": np.asarray(PHYSICAL_CONTEXT_FEATURE_NAMES),
        "feature_contract_sha256": np.asarray(FEATURE_CONTRACT_SHA256),
        "normalization": np.asarray("none"),
        "physical_context_weekly_sha256": np.asarray(_array_sha256(values)),
        "physical_context_weekly": values,
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
        "feature_names",
        "feature_contract_sha256",
        "normalization",
        "physical_context_weekly_sha256",
        "physical_context_weekly",
    }
    try:
        with np.load(path, allow_pickle=False) as part:
            missing = required.difference(part.files)
            if missing:
                raise PhysicalContextCacheContractError(
                    f"part {path} is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(part[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise PhysicalContextCacheContractError(
                    f"part {path} schema differs"
                )
            if (
                str(scalar("source_store")) != contract.source_store
                or str(scalar("source_fingerprint"))
                != contract.source_fingerprint
            ):
                raise PhysicalContextCacheContractError(
                    f"part {path} source is stale"
                )
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
                raise PhysicalContextCacheContractError(
                    f"part {path} initialization differs"
                )
            feature_names = tuple(
                np.asarray(part["feature_names"]).astype(str).tolist()
            )
            if (
                feature_names != PHYSICAL_CONTEXT_FEATURE_NAMES
                or str(scalar("feature_contract_sha256"))
                != FEATURE_CONTRACT_SHA256
                or str(scalar("normalization")) != "none"
            ):
                raise PhysicalContextCacheContractError(
                    f"part {path} feature contract differs"
                )
            if not np.array_equal(
                part["latitude"], contract.latitude
            ) or not np.array_equal(part["longitude"], contract.longitude):
                raise PhysicalContextCacheContractError(
                    f"part {path} grid differs"
                )
            stored = part["physical_context_weekly"]
            if stored.dtype != np.float32:
                raise PhysicalContextCacheContractError(
                    f"part {path} is not float32"
                )
            values = np.asarray(stored).copy()
            expected_hash = str(scalar("physical_context_weekly_sha256"))
    except PhysicalContextCacheContractError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise PhysicalContextCacheContractError(
            f"cannot read cache part {path}: {error}"
        ) from error
    _validate_context_values(values, expected_shape=CONTEXT_FIELD_SHAPE)
    if _array_sha256(values) != expected_hash:
        raise PhysicalContextCacheContractError(
            f"part {path} data checksum differs"
        )
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
        except PhysicalContextCacheContractError as error:
            print(f"rebuilding invalid part {path}: {error}", flush=True)
    if not bool(np.asarray(group["init_complete"][int(source_index)]).item()):
        raise PhysicalContextCacheContractError(
            f"source initialization {initialization} is not marked complete"
        )
    values = _summarize_initialization(
        group["forecast"],
        source_index,
        contract.channel_names,
        contract.latitude_slice,
        contract.longitude_slice,
    )
    _atomic_npz(
        path,
        _part_payload(contract, source_index, initialization, values),
    )
    _load_part(path, contract, source_index, initialization)
    return path, True


def _sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    return member_cache._sidecar_paths(output)


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
        for output_index, (
            part,
            contract,
            source_index,
            initialization,
            checksum,
        ) in enumerate(rows):
            if sha256_file(part) != checksum:
                raise PhysicalContextCacheContractError(
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
    """Validate the requested scope and publish an audited NPY memmap."""

    scope = _scope_records(
        contract, max_initializations=max_initializations, smoke=smoke
    )
    rows: list[
        tuple[Path, SourceContract, int, np.datetime64, str]
    ] = []
    manifest_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for output_index, (source_index, initialization) in enumerate(scope):
        part = _part_path(parts_dir, initialization).resolve()
        if not part.is_file():
            missing.append(np.datetime_as_string(initialization, unit="D"))
            continue
        values = _load_part(part, contract, source_index, initialization)
        part_checksum = sha256_file(part)
        rows.append(
            (part, contract, source_index, initialization, part_checksum)
        )
        manifest_records.append(
            {
                "output_index": output_index,
                "source_init_index": source_index,
                "initialization": np.datetime_as_string(
                    initialization, unit="D"
                ),
                "part_path": str(part),
                "part_sha256": part_checksum,
                "physical_context_weekly_sha256": _array_sha256(values),
            }
        )
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise PhysicalContextCacheContractError(
            f"cannot finalize: {len(missing)} of {len(scope)} parts are "
            f"missing ({preview}{suffix})"
        )

    output = Path(output).resolve()
    metadata_path, manifest_path, checksums_path = _sidecar_paths(output)
    shape = (len(scope), *CONTEXT_FIELD_SHAPE)
    data_checksum = _write_atomic_memmap(output, shape, rows)
    initializations = np.asarray(
        [record[1] for record in scope], dtype="datetime64[D]"
    )
    source_indices = np.asarray(
        [record[0] for record in scope], dtype=np.int64
    )
    is_full_archive = len(scope) == INITIALIZATION_COUNT
    if smoke:
        scope_name = "three_era_train_validation_test_non_scientific"
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
            "whether the artifact contains the full archive or a diagnostic "
            "subset"
        ),
        "scope": scope_name,
        "full_archive": is_full_archive,
        "initialization_count": len(scope),
        "initializations": np.datetime_as_string(
            initializations, unit="D"
        ).tolist(),
        "source_init_indices": source_indices.tolist(),
        "shape": list(shape),
        "dtype": np.dtype(np.float32).str,
        "dims": list(OUTPUT_DIMS),
        "lead_week_labels": list(range(1, LEAD_WEEK_COUNT + 1)),
        "latitude": contract.latitude.tolist(),
        "longitude": contract.longitude.tolist(),
        "source_variables": list(RAW_CHANNEL_NAMES),
        "feature_names": list(PHYSICAL_CONTEXT_FEATURE_NAMES),
        "feature_units": dict(PHYSICAL_CONTEXT_FEATURE_UNITS),
        "feature_definitions": dict(PHYSICAL_CONTEXT_FEATURE_DEFINITIONS),
        "transforms": dict(PHYSICAL_CONTEXT_TRANSFORMS),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "temporal_aggregation": TEMPORAL_AGGREGATION,
        "ensemble_aggregation": ENSEMBLE_AGGREGATION,
        "flux_aggregation": FLUX_AGGREGATION,
        "normalization": "none",
        "data_file": output.name,
        "data_sha256": data_checksum,
        "manifest_file": manifest_path.name,
        "checksums_file": checksums_path.name,
    }
    manifest: dict[str, Any] = {
        "schema_name": CACHE_SCHEMA_NAME,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": contract.source_fingerprint,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
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


def _validate_scope_metadata(
    metadata: Mapping[str, Any],
    initializations: np.ndarray,
    source_indices: np.ndarray,
) -> None:
    count = len(initializations)
    scope_name = metadata.get("scope")
    if bool(metadata.get("full_archive")):
        if scope_name != "full_archive" or not np.array_equal(
            source_indices, np.arange(INITIALIZATION_COUNT, dtype=np.int64)
        ):
            raise PhysicalContextCacheContractError(
                "full cache scope/source indices differ"
            )
        _validate_allseason_initializations(initializations)
        return
    if count >= INITIALIZATION_COUNT:
        raise PhysicalContextCacheContractError(
            "non-full cache has an invalid scope size"
        )
    if scope_name == "bounded_prefix_non_scientific":
        if not np.array_equal(source_indices, np.arange(count, dtype=np.int64)):
            raise PhysicalContextCacheContractError(
                "bounded cache is not a source prefix"
            )
        return
    if scope_name == "three_era_train_validation_test_non_scientific":
        years = _initialization_years(initializations)
        forecast_ends = initializations + np.timedelta64(41, "D")
        train = years <= 2017
        validation = (years >= 2018) & (years <= 2019)
        test = years >= 2020
        if (
            count != SMOKE_INITIALIZATION_COUNT
            or int(np.count_nonzero(train))
            != SMOKE_SPLIT_COUNTS["train_2002_2017"]
            or int(np.count_nonzero(validation))
            != SMOKE_SPLIT_COUNTS["validation_2018_2019"]
            or int(np.count_nonzero(test))
            != SMOKE_SPLIT_COUNTS["test_2020_2021"]
            or np.any(forecast_ends[train] >= np.datetime64("2018-01-01", "D"))
            or np.any(
                forecast_ends[validation] >= np.datetime64("2020-01-01", "D")
            )
        ):
            raise PhysicalContextCacheContractError(
                "stratified smoke scope differs"
            )
        return
    raise PhysicalContextCacheContractError(
        f"unsupported non-full cache scope: {scope_name!r}"
    )


def verify_cache(
    output: Path = DEFAULT_CACHE,
    *,
    metadata_path: Path | None = None,
    manifest_path: Path | None = None,
    checksums_path: Path | None = None,
    expected_source_fingerprint: str | None = None,
) -> Mapping[str, Any]:
    """Verify hashes, schema, alignment, field order, and physical values."""

    output = Path(output).resolve()
    default_metadata, default_manifest, default_checksums = _sidecar_paths(output)
    metadata_path = Path(metadata_path or default_metadata).resolve()
    manifest_path = Path(manifest_path or default_manifest).resolve()
    checksums_path = Path(checksums_path or default_checksums).resolve()
    required_paths = (output, metadata_path, manifest_path, checksums_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise PhysicalContextCacheContractError(
            f"cache artifacts are missing: {missing}"
        )

    checksums = _read_checksums(checksums_path)
    expected_names = {output.name, metadata_path.name, manifest_path.name}
    if set(checksums) != expected_names:
        raise PhysicalContextCacheContractError(
            f"checksum entries {sorted(checksums)} differ from "
            f"{sorted(expected_names)}"
        )
    actual_checksums = {
        output.name: sha256_file(output),
        metadata_path.name: sha256_file(metadata_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    if checksums != actual_checksums:
        raise PhysicalContextCacheContractError(
            "one or more final cache checksums differ"
        )

    metadata = _load_json(metadata_path)
    manifest = _load_json(manifest_path)
    for payload, name in ((metadata, "metadata"), (manifest, "manifest")):
        if (
            payload.get("schema_name") != CACHE_SCHEMA_NAME
            or int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION
        ):
            raise PhysicalContextCacheContractError(f"{name} schema differs")
    source_fingerprint = str(metadata.get("source_fingerprint", ""))
    if (
        not source_fingerprint
        or manifest.get("source_fingerprint") != source_fingerprint
    ):
        raise PhysicalContextCacheContractError(
            "metadata/manifest source fingerprints differ"
        )
    if (
        expected_source_fingerprint is not None
        and source_fingerprint != expected_source_fingerprint
    ):
        raise PhysicalContextCacheContractError(
            "final cache source fingerprint is stale"
        )
    if metadata.get("data_sha256") != actual_checksums[output.name]:
        raise PhysicalContextCacheContractError(
            "metadata data checksum differs"
        )
    if metadata.get("dims") != list(OUTPUT_DIMS):
        raise PhysicalContextCacheContractError(
            "final cache dimension order differs"
        )
    if (
        metadata.get("feature_names")
        != list(PHYSICAL_CONTEXT_FEATURE_NAMES)
        or metadata.get("feature_units")
        != dict(PHYSICAL_CONTEXT_FEATURE_UNITS)
        or metadata.get("feature_definitions")
        != dict(PHYSICAL_CONTEXT_FEATURE_DEFINITIONS)
        or metadata.get("transforms") != dict(PHYSICAL_CONTEXT_TRANSFORMS)
        or metadata.get("feature_contract_sha256")
        != FEATURE_CONTRACT_SHA256
        or manifest.get("feature_contract_sha256")
        != FEATURE_CONTRACT_SHA256
    ):
        raise PhysicalContextCacheContractError(
            "final cache physical-feature contract differs"
        )
    if (
        metadata.get("source_variables") != list(RAW_CHANNEL_NAMES)
        or metadata.get("temporal_aggregation") != TEMPORAL_AGGREGATION
        or metadata.get("ensemble_aggregation") != ENSEMBLE_AGGREGATION
        or metadata.get("flux_aggregation") != FLUX_AGGREGATION
        or metadata.get("normalization") != "none"
    ):
        raise PhysicalContextCacheContractError(
            "final cache reduction or normalization contract differs"
        )
    if metadata.get("lead_week_labels") != list(
        range(1, LEAD_WEEK_COUNT + 1)
    ):
        raise PhysicalContextCacheContractError(
            "final cache lead labels differ"
        )
    if (
        metadata.get("latitude") != EXPECTED_LATITUDE.tolist()
        or metadata.get("longitude") != EXPECTED_LONGITUDE.tolist()
    ):
        raise PhysicalContextCacheContractError("final cache grid differs")
    if (
        metadata.get("data_file") != output.name
        or metadata.get("manifest_file") != manifest_path.name
        or metadata.get("checksums_file") != checksums_path.name
    ):
        raise PhysicalContextCacheContractError(
            "final cache sidecar names differ"
        )

    try:
        data = np.load(output, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PhysicalContextCacheContractError(
            f"cannot memory-map final cache: {error}"
        ) from error
    count = int(metadata.get("initialization_count", -1))
    expected_shape = (count, *CONTEXT_FIELD_SHAPE)
    if (
        tuple(metadata.get("shape", ())) != expected_shape
        or data.shape != expected_shape
    ):
        raise PhysicalContextCacheContractError(
            f"final cache shape differs: metadata={metadata.get('shape')}, "
            f"data={data.shape}"
        )
    if (
        data.dtype != np.float32
        or metadata.get("dtype") != np.dtype(np.float32).str
    ):
        raise PhysicalContextCacheContractError(
            "final cache is not float32"
        )

    initializations = np.asarray(
        metadata.get("initializations", []), dtype="datetime64[D]"
    )
    source_indices = np.asarray(
        metadata.get("source_init_indices", []), dtype=np.int64
    )
    if initializations.shape != (count,) or source_indices.shape != (count,):
        raise PhysicalContextCacheContractError(
            "final cache initialization metadata differs"
        )
    if count and (
        np.isnat(initializations).any()
        or np.unique(initializations).size != count
        or np.any(initializations[1:] <= initializations[:-1])
    ):
        raise PhysicalContextCacheContractError(
            "final cache initializations are invalid"
        )
    if count and (
        np.unique(source_indices).size != count
        or np.any(source_indices[1:] <= source_indices[:-1])
        or source_indices[0] < 0
        or source_indices[-1] >= INITIALIZATION_COUNT
    ):
        raise PhysicalContextCacheContractError(
            "final cache source indices are invalid"
        )
    _validate_scope_metadata(metadata, initializations, source_indices)

    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != count
        or manifest.get("initialization_count") != count
    ):
        raise PhysicalContextCacheContractError(
            "final manifest record count differs"
        )
    for index, (record, initialization) in enumerate(
        zip(records, initializations, strict=True)
    ):
        if not isinstance(record, dict):
            raise PhysicalContextCacheContractError(
                "final manifest contains a non-object record"
            )
        if (
            record.get("output_index") != index
            or record.get("source_init_index") != int(source_indices[index])
            or record.get("initialization")
            != np.datetime_as_string(initialization, unit="D")
        ):
            raise PhysicalContextCacheContractError(
                f"final manifest alignment differs at output index {index}"
            )
        for hash_name in (
            "part_sha256",
            "physical_context_weekly_sha256",
        ):
            checksum = record.get(hash_name)
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in checksum
                )
            ):
                raise PhysicalContextCacheContractError(
                    f"final manifest {hash_name} differs at output index "
                    f"{index}"
                )

    for start in range(0, count, 8):
        block = np.asarray(data[start : start + 8])
        _validate_context_values(
            block,
            expected_shape=(len(block), *CONTEXT_FIELD_SHAPE),
        )
    return {
        "status": "verified",
        "data": str(output),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "source_fingerprint": source_fingerprint,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "data_sha256": actual_checksums[output.name],
        "full_archive": bool(metadata.get("full_archive")),
    }


def load_physical_context_cache(
    output: Path = DEFAULT_CACHE,
    *,
    verify: bool = True,
    expected_source_fingerprint: str | None = None,
) -> tuple[np.memmap, Mapping[str, Any]]:
    """Return the read-only physical-context memmap and alignment metadata."""

    output = Path(output).resolve()
    metadata_path, _, _ = _sidecar_paths(output)
    if verify:
        verify_cache(
            output,
            expected_source_fingerprint=expected_source_fingerprint,
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
            contract,
            max_initializations=args.max_inits,
            smoke=args.smoke,
        )
        print(
            json.dumps(
                {
                    "source_store": contract.source_store,
                    "source_fingerprint": contract.source_fingerprint,
                    "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                    "source_initialization_count": len(
                        contract.initializations
                    ),
                    "selected_initialization_count": len(scope),
                    "first_initialization": np.datetime_as_string(
                        scope[0][1], unit="D"
                    ),
                    "last_initialization": np.datetime_as_string(
                        scope[-1][1], unit="D"
                    ),
                    "lead_weeks": LEAD_WEEK_COUNT,
                    "feature_names": list(
                        PHYSICAL_CONTEXT_FEATURE_NAMES
                    ),
                    "grid_shape": list(GRID_SHAPE),
                    "expected_output_shape": [
                        len(scope),
                        *CONTEXT_FIELD_SHAPE,
                    ],
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
        for ordinal, (source_index, initialization) in enumerate(
            records, start=1
        ):
            path, created = build_part(
                group,
                contract,
                source_index,
                initialization,
                args.parts_dir,
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
