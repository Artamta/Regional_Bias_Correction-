#!/usr/bin/env python3
"""Independent alignment audit for the corrected FuXi training experiment.

The auditor does not import the training or cache-builder implementations.  It
reopens the frozen 2002--2021 FuXi member cache, native FuXi metadata, and only
the 2002--2022 IMD stores.  It independently reconstructs end-labelled
initialization+1..+42 weekly targets, split/embargo IDs, and scoring support.
One native initialization is also reduced from 42 daily lead fields to six
weekly fields and compared byte-for-byte with the frozen cache.

The original training run did not persist logical hashes of the weekly target
tensors it consumed.  This audit therefore proves the current source bytes,
code snapshot, contract reconstruction, and every comparison recoverable from
the frozen manifest/cache; it cannot retroactively prove equality to an
unrecorded historical target tensor.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_training_alignment_audit_v1"
PARENT_EXPERIMENT = "fuxi_allseason_ensemble_calibration_v2_aligned"
CONTRACT_REVISION = "imd_end_labelled_init_plus_1_through_42_v2"
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "a98bf491a41c7245ef2ee0904e4882eb0e3ca3081db2b0291b1f0fa1669fd5df"
)

TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
REUSED_DEVELOPMENT_YEARS = (2020, 2021)
OBSERVATION_YEARS = tuple(range(2002, 2023))
TARGET_DAY_OFFSETS = tuple(range(1, 43))
EXPECTED_SPLIT_COUNTS = {
    "train": 1652,
    "validation": 196,
    "test": 208,
    "embargo": 24,
}
EXPECTED_CACHE_SHAPE = (2080, 51, 6, 27, 27)
EXPECTED_SUPPORT_CELLS = 171
MAX_OPEN_OBSERVATION_YEAR = 2022

DEFAULT_PARENT_MANIFEST = (
    PROJECT_ROOT
    / "resultsv3/fuxi_allseason_ensemble_calibration/"
    "full_20260825T224711Z/manifest.json"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "cache/fuxi_tp_members_weekly_2002_2021_allseason_v1.npy"
)
DEFAULT_NATIVE_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
DEFAULT_IMD_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/observations/ground_truth_v1/daily/imd/tp/"
    "india_1p5_27x27_v1"
)
DEFAULT_SPATIAL_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/spatial/spatial_support.zarr"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv3/fuxi_allseason_training_alignment_audit"
)
DEFAULT_SPOT_INITIALIZATION = "2017-11-17"
SOURCE_PATH = Path(__file__).resolve()
TEST_SOURCE_PATH = (
    PROJECT_ROOT / "tests/test_audit_fuxi_allseason_training_alignment.py"
)
_RENAME_NOREPLACE = 1


class AlignmentAuditError(RuntimeError):
    """Raised when an independently reconstructed contract does not match."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignmentAuditError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of every byte in one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order logical bytes."""

    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    view = memoryview(array.view(np.uint8).reshape(-1))
    block_size = 8 * 1024 * 1024
    for start in range(0, view.nbytes, block_size):
        digest.update(view[start : start + block_size])
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignmentAuditError(f"cannot read JSON: {path}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def comparison(actual: Any, expected: Any) -> dict[str, Any]:
    return {"actual": actual, "expected": expected, "match": actual == expected}


def target_dates(initializations: Iterable[Any]) -> np.ndarray:
    """Build W1 +1..+7 through W6 +36..+42 without training code."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    require(starts.ndim == 1 and starts.size > 0, "initializations must be 1-D")
    require(not np.isnat(starts).any(), "initializations contain NaT")
    offsets = np.asarray(TARGET_DAY_OFFSETS, dtype="timedelta64[D]").reshape(
        1, 6, 7
    )
    return starts[:, None, None] + offsets


@dataclass(frozen=True)
class SplitReconstruction:
    indices: Mapping[str, np.ndarray]
    outcome_ends: np.ndarray
    candidate_roles: np.ndarray
    final_roles: np.ndarray


def reconstruct_splits(initializations: Iterable[Any]) -> SplitReconstruction:
    """Independently purge windows crossing the 2018 and 2020 boundaries."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    require(starts.ndim == 1 and starts.size > 0, "initializations must be 1-D")
    require(not np.isnat(starts).any(), "initializations contain NaT")
    require(np.unique(starts).size == starts.size, "initializations are duplicated")
    require(
        np.all(np.diff(starts) > np.timedelta64(0, "D")),
        "initializations are not strictly increasing",
    )
    years = starts.astype("datetime64[Y]").astype(np.int64) + 1970
    ends = starts + np.timedelta64(42, "D")
    train_candidate = np.isin(years, TRAIN_YEARS)
    validation_candidate = np.isin(years, VALIDATION_YEARS)
    test_candidate = np.isin(years, REUSED_DEVELOPMENT_YEARS)
    in_scope = train_candidate | validation_candidate | test_candidate
    require(np.all(in_scope), "cache includes an out-of-contract initialization year")

    train_mask = train_candidate & (ends < np.datetime64("2018-01-01", "D"))
    validation_mask = validation_candidate & (
        ends < np.datetime64("2020-01-01", "D")
    )
    test_mask = test_candidate
    retained = train_mask | validation_mask | test_mask
    embargo_mask = in_scope & ~retained

    candidate_roles = np.full(starts.shape, "", dtype="U18")
    candidate_roles[train_candidate] = "train"
    candidate_roles[validation_candidate] = "validation"
    candidate_roles[test_candidate] = "test"
    final_roles = candidate_roles.copy()
    final_roles[embargo_mask] = "embargo"
    indices = {
        "train": np.flatnonzero(train_mask).astype(np.int64),
        "validation": np.flatnonzero(validation_mask).astype(np.int64),
        "test": np.flatnonzero(test_mask).astype(np.int64),
        "embargo": np.flatnonzero(embargo_mask).astype(np.int64),
    }
    return SplitReconstruction(indices, ends, candidate_roles, final_roles)


def decode_cf_days(values: np.ndarray, attrs: Mapping[str, Any]) -> np.ndarray:
    """Decode the narrow integer ``days since YYYY-MM-DD`` IMD contract."""

    raw = np.asarray(values)
    require(np.issubdtype(raw.dtype, np.integer), "IMD time is not integer days")
    units = str(attrs.get("units", ""))
    match = re.fullmatch(
        r"days since (\d{4}-\d{2}-\d{2})(?: 00:00:00)?", units
    )
    require(match is not None, f"unsupported IMD time units: {units!r}")
    require(
        str(attrs.get("calendar")) == "proleptic_gregorian",
        "unexpected IMD calendar",
    )
    origin = np.datetime64(match.group(1), "D")
    return origin + raw.astype("timedelta64[D]")


def weekly_target_mean(daily: np.ndarray) -> np.ndarray:
    """Average an explicit [case, week, day, y, x] daily target tensor."""

    values = np.asarray(daily)
    require(values.ndim == 5 and values.shape[1:3] == (6, 7), "daily target shape differs")
    require(values.dtype == np.float32, "daily IMD target is not float32")
    return values.mean(axis=2, dtype=np.float64).astype(np.float32)


def weekly_native_tp(native_daily_rate: np.ndarray) -> np.ndarray:
    """Convert native mm h-1 [member,42,y,x] to weekly mm day-1."""

    values = np.asarray(native_daily_rate)
    require(
        values.ndim == 4 and values.shape[:2] == (51, 42),
        "native TP spot shape differs",
    )
    require(values.dtype == np.float32, "native TP spot is not float32")
    require(np.isfinite(values).all(), "native TP spot contains non-finite values")
    require(np.all(values >= 0.0), "native TP spot contains negative values")
    height, width = values.shape[-2:]
    return (
        (values * np.float32(24.0))
        .reshape(51, 6, 7, height, width)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )


def import_zarr() -> Any:
    try:
        import zarr
    except ImportError as error:
        raise AlignmentAuditError("zarr is required for the alignment audit") from error
    require(hasattr(zarr, "open_consolidated"), "zarr v2 compatibility is required")
    return zarr


def _zmetadata_sha256(store: Path) -> str:
    metadata = Path(store) / ".zmetadata"
    require(metadata.is_file(), f"consolidated metadata is absent: {store}")
    return sha256_file(metadata)


def _exact_array(first: np.ndarray, second: np.ndarray, label: str) -> None:
    require(first.shape == second.shape, f"{label} shape differs")
    require(first.dtype == second.dtype, f"{label} dtype differs")
    require(array_sha256(first) == array_sha256(second), f"{label} bytes differ")


def verify_declared_artifacts(
    root: Path, declared: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Verify every file hash declared by a frozen manifest mapping."""

    require(isinstance(declared, Mapping), f"{label} hash mapping is absent")
    mismatches: list[str] = []
    for relative, expected in sorted(declared.items()):
        require(valid_sha256(expected), f"invalid {label} hash for {relative}")
        child = (root / str(relative)).resolve()
        require(root.resolve() in child.parents, f"{label} path escapes run: {relative}")
        if not child.is_file() or sha256_file(child) != expected:
            mismatches.append(str(relative))
    require(not mismatches, f"{label} byte mismatches: {mismatches[:5]}")
    return {"declared_file_count": len(declared), "failure_count": 0, "passed": True}


@dataclass(frozen=True)
class CacheContract:
    values: np.ndarray
    metadata: Mapping[str, Any]
    cache_manifest: Mapping[str, Any]
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    source_indices: np.ndarray
    paths: Mapping[str, Path]
    byte_comparisons: Mapping[str, Any]


def load_and_verify_cache(
    cache_path: Path, parent: Mapping[str, Any]
) -> CacheContract:
    cache_path = Path(cache_path).resolve()
    metadata_path = cache_path.with_suffix(".metadata.json")
    cache_manifest_path = cache_path.with_suffix(".manifest.json")
    checksums_path = cache_path.with_suffix(".sha256")
    for path in (cache_path, metadata_path, cache_manifest_path, checksums_path):
        require(path.is_file(), f"cache artifact is absent: {path}")
    metadata = read_json(metadata_path)
    cache_manifest = read_json(cache_manifest_path)
    parent_cache = parent.get("cache", {})
    require(isinstance(parent_cache, Mapping), "parent cache contract is absent")
    require(
        Path(str(parent_cache.get("data_file", ""))).resolve() == cache_path,
        "CLI cache differs from frozen training cache",
    )

    actual_hashes = {
        "data_sha256": sha256_file(cache_path),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(cache_manifest_path),
    }
    byte_comparisons = {
        name: comparison(actual, parent_cache.get(name))
        for name, actual in actual_hashes.items()
    }
    require(
        all(item["match"] for item in byte_comparisons.values()),
        "cache bytes differ from frozen training manifest",
    )
    require(
        metadata.get("data_sha256") == actual_hashes["data_sha256"],
        "cache metadata data hash differs",
    )
    checksum_lines = {
        line.split()[1]: line.split()[0]
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected_checksum_lines = {
        cache_path.name: actual_hashes["data_sha256"],
        metadata_path.name: actual_hashes["metadata_sha256"],
        cache_manifest_path.name: actual_hashes["manifest_sha256"],
    }
    require(checksum_lines == expected_checksum_lines, "cache SHA-256 sidecar differs")

    values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    require(values.shape == EXPECTED_CACHE_SHAPE, "frozen cache shape differs")
    require(values.dtype == np.float32, "frozen cache dtype differs")
    require(metadata.get("shape") == list(EXPECTED_CACHE_SHAPE), "cache metadata shape differs")
    require(
        metadata.get("dims") == ["init", "member", "lead_week", "lat", "lon"],
        "cache dimension contract differs",
    )
    require(metadata.get("output_units") == "mm day-1", "cache units differ")
    initializations = np.asarray(metadata.get("initializations", []), dtype="datetime64[D]")
    latitude = np.asarray(metadata.get("latitude", []), dtype=np.float64)
    longitude = np.asarray(metadata.get("longitude", []), dtype=np.float64)
    source_indices = np.asarray(metadata.get("source_init_indices", []), dtype=np.int64)
    require(initializations.shape == (2080,), "cache initialization coordinate differs")
    require(latitude.shape == (27,) and longitude.shape == (27,), "cache grid differs")
    require(source_indices.shape == (2080,), "cache source indices differ")
    require(
        np.array_equal(source_indices, np.arange(2080, dtype=np.int64)),
        "cache source indices are not 0..2079",
    )
    require(
        cache_manifest.get("source_fingerprint") == metadata.get("source_fingerprint"),
        "cache manifest/metadata source fingerprints differ",
    )
    require(
        parent_cache.get("source_fingerprint") == metadata.get("source_fingerprint"),
        "training/cache source fingerprints differ",
    )
    return CacheContract(
        values=values,
        metadata=metadata,
        cache_manifest=cache_manifest,
        initializations=initializations,
        latitude=latitude,
        longitude=longitude,
        source_indices=source_indices,
        paths={
            "data": cache_path,
            "metadata": metadata_path,
            "manifest": cache_manifest_path,
            "checksums": checksums_path,
        },
        byte_comparisons=byte_comparisons,
    )


@dataclass(frozen=True)
class ImdReconstruction:
    weekly_truth: np.ndarray
    target_dates: np.ndarray
    observation_fraction: np.ndarray
    support_mask: np.ndarray
    source_rows: Sequence[Mapping[str, Any]]
    logical_hashes: Mapping[str, str]
    opened_stores: Sequence[str]


def load_imd_and_reconstruct_targets(
    imd_root: Path,
    initializations: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> ImdReconstruction:
    """Reopen fixed IMD years and reconstruct weekly targets independently."""

    require(max(OBSERVATION_YEARS) == MAX_OPEN_OBSERVATION_YEAR, "IMD year guard differs")
    zarr = import_zarr()
    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    opened_stores: list[str] = []
    reference_fraction: np.ndarray | None = None

    for year in OBSERVATION_YEARS:
        require(year <= MAX_OPEN_OBSERVATION_YEAR, "observation-year firewall failed")
        store = (Path(imd_root) / f"{year}.zarr").resolve()
        zmetadata_sha = _zmetadata_sha256(store)
        group = zarr.open_consolidated(str(store), mode="r")
        attrs = dict(group.attrs)
        require(attrs.get("source") == "imd", f"{year} source is not IMD")
        require(attrs.get("units") == "mm day-1", f"{year} IMD units differ")
        require(attrs.get("variable") == "tp", f"{year} IMD variable differs")
        require(int(attrs.get("schema_version")) == 1, f"{year} IMD schema differs")
        required = {"time", "latitude", "longitude", "observation", "observation_fraction"}
        require(required.issubset(set(group.array_keys())), f"{year} IMD arrays are incomplete")

        dates = decode_cf_days(group["time"][:], dict(group["time"].attrs))
        expected_dates = np.arange(
            np.datetime64(f"{year}-01-01", "D"),
            np.datetime64(f"{year + 1}-01-01", "D"),
            np.timedelta64(1, "D"),
        )
        require(np.array_equal(dates, expected_dates), f"{year} IMD calendar is incomplete")
        store_latitude = np.asarray(group["latitude"][:], dtype=np.float64)
        store_longitude = np.asarray(group["longitude"][:], dtype=np.float64)
        require(np.array_equal(store_latitude, latitude), f"{year} IMD latitude differs")
        require(np.array_equal(store_longitude, longitude), f"{year} IMD longitude differs")
        values = np.asarray(group["observation"][:])
        fraction = np.asarray(group["observation_fraction"][:])
        require(values.dtype == np.float32, f"{year} IMD observation dtype differs")
        require(fraction.dtype == np.float32, f"{year} IMD fraction dtype differs")
        require(values.shape == (dates.size, 27, 27), f"{year} IMD shape differs")
        require(fraction.shape == (27, 27), f"{year} IMD support shape differs")
        if reference_fraction is None:
            reference_fraction = fraction.copy()
        else:
            _exact_array(fraction, reference_fraction, f"{year} IMD support")
        support = fraction > 0.0
        require(
            np.isfinite(values[:, support]).all() and np.all(values[:, support] >= 0.0),
            f"{year} supported IMD values are invalid",
        )

        upstream_source = Path(str(attrs.get("source_path", ""))).resolve()
        declared_source_sha = attrs.get("source_sha256")
        require(upstream_source.is_file(), f"{year} upstream IMD source is absent")
        actual_source_sha = sha256_file(upstream_source)
        require(
            valid_sha256(declared_source_sha) and actual_source_sha == declared_source_sha,
            f"{year} upstream IMD source bytes differ",
        )
        source_rows.append(
            {
                "year": year,
                "store": str(store),
                "zmetadata_sha256": zmetadata_sha,
                "root_attrs_sha256": canonical_json_sha256(attrs),
                "declared_spatial_support_zmetadata_sha256": attrs.get(
                    "spatial_support_zmetadata_sha256"
                ),
                "upstream_source": str(upstream_source),
                "declared_upstream_source_sha256": declared_source_sha,
                "actual_upstream_source_sha256": actual_source_sha,
                "upstream_source_hash_match": True,
                "dates_sha256": array_sha256(dates),
                "observation_sha256": array_sha256(values),
                "observation_fraction_sha256": array_sha256(fraction),
                "latitude_sha256": array_sha256(store_latitude),
                "longitude_sha256": array_sha256(store_longitude),
            }
        )
        all_dates.append(dates)
        all_values.append(values)
        opened_stores.append(str(store))

    assert reference_fraction is not None
    daily_dates = np.concatenate(all_dates)
    daily_values = np.concatenate(all_values)
    require(np.unique(daily_dates).size == daily_dates.size, "IMD dates are duplicated")
    targets = target_dates(initializations)
    requested = targets.reshape(-1)
    positions = np.searchsorted(daily_dates, requested)
    require(np.all(positions < daily_dates.size), "target exceeds allowed IMD calendar")
    require(np.array_equal(daily_dates[positions], requested), "target date is absent from IMD")
    selected_daily = daily_values[positions].reshape(
        len(initializations), 6, 7, 27, 27
    )
    weekly = weekly_target_mean(selected_daily)
    support_mask = reference_fraction > 0.0
    require(np.count_nonzero(support_mask) == EXPECTED_SUPPORT_CELLS, "IMD support count differs")
    require(
        np.isfinite(weekly[..., support_mask]).all()
        and np.all(weekly[..., support_mask] >= 0.0),
        "weekly IMD target is invalid on support",
    )
    logical_hashes = {
        "all_daily_dates_2002_2022_sha256": array_sha256(daily_dates),
        "all_daily_observations_2002_2022_sha256": array_sha256(daily_values),
        "target_dates_plus_1_through_42_sha256": array_sha256(targets),
        "selected_daily_target_tensor_sha256": array_sha256(selected_daily),
        "weekly_target_tensor_sha256": array_sha256(weekly),
        "weekly_target_supported_values_sha256": array_sha256(weekly[..., support_mask]),
        "observation_fraction_sha256": array_sha256(reference_fraction),
        "support_mask_sha256": array_sha256(support_mask),
    }
    return ImdReconstruction(
        weekly_truth=weekly,
        target_dates=targets,
        observation_fraction=reference_fraction,
        support_mask=support_mask,
        source_rows=source_rows,
        logical_hashes=logical_hashes,
        opened_stores=opened_stores,
    )


def audit_scoring_support(
    spatial_store: Path,
    imd: ImdReconstruction,
    cache: CacheContract,
    parent_path: Path,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild support weights and compare the saved NPZ arrays."""

    zarr = import_zarr()
    spatial_store = Path(spatial_store).resolve()
    zmetadata_sha = _zmetadata_sha256(spatial_store)
    group = zarr.open_consolidated(str(spatial_store), mode="r")
    attrs = dict(group.attrs)
    latitude = np.asarray(group["latitude"][:], dtype=np.float64)
    longitude = np.asarray(group["longitude"][:], dtype=np.float64)
    area = np.asarray(group["india_area_weight_km2"][:], dtype=np.float64)
    require(np.array_equal(latitude, cache.latitude), "spatial latitude differs")
    require(np.array_equal(longitude, cache.longitude), "spatial longitude differs")
    require(area.shape == (27, 27), "spatial area shape differs")
    declared_source_sha = attrs.get("source_sha256")
    upstream_source = Path(str(attrs.get("source_path", ""))).resolve()
    require(upstream_source.is_file(), "upstream spatial source is absent")
    actual_source_sha = sha256_file(upstream_source)
    require(
        valid_sha256(declared_source_sha) and actual_source_sha == declared_source_sha,
        "upstream spatial source bytes differ",
    )
    imd_declared_spatial_hashes = {
        str(row.get("declared_spatial_support_zmetadata_sha256"))
        for row in imd.source_rows
    }
    require(
        imd_declared_spatial_hashes == {zmetadata_sha},
        "IMD stores do not bind the current spatial support metadata",
    )

    weights = area * imd.observation_fraction.astype(np.float64)
    weights[~np.isfinite(weights) | (weights <= 0.0)] = 0.0
    normalized = weights / weights.sum(dtype=np.float64)
    evaluation = parent.get("evaluation", {})
    relative = str(evaluation.get("scoring_support_artifact", ""))
    require(relative == "evaluation/scoring_support.npz", "parent support artifact differs")
    saved_path = (parent_path.parent / relative).resolve()
    require(saved_path.is_file(), "parent scoring support artifact is absent")
    actual_saved_sha = sha256_file(saved_path)
    require(
        actual_saved_sha == evaluation.get("scoring_support_sha256"),
        "parent scoring support artifact hash differs",
    )
    require(
        actual_saved_sha == parent.get("artifact_sha256", {}).get(relative),
        "parent artifact ledger support hash differs",
    )
    expected_arrays = {
        "latitude": cache.latitude,
        "longitude": cache.longitude,
        "observation_fraction": imd.observation_fraction,
        "support_mask": imd.support_mask,
        "scoring_weight_km2_fraction": weights,
        "normalized_scoring_weight": normalized,
    }
    comparisons: dict[str, Any] = {}
    with np.load(saved_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected_arrays), "saved support keys differ")
        for name, expected in expected_arrays.items():
            actual = np.asarray(archive[name])
            _exact_array(actual, expected, f"saved scoring support {name}")
            comparisons[name] = {
                "match": True,
                "saved_sha256": array_sha256(actual),
                "reconstructed_sha256": array_sha256(expected),
            }
    return {
        "spatial_store": str(spatial_store),
        "spatial_zmetadata_sha256": zmetadata_sha,
        "spatial_root_attrs_sha256": canonical_json_sha256(attrs),
        "upstream_source": str(upstream_source),
        "declared_upstream_source_sha256": declared_source_sha,
        "actual_upstream_source_sha256": actual_source_sha,
        "upstream_source_hash_match": True,
        "india_area_weight_km2_sha256": array_sha256(area),
        "reconstructed_scoring_weight_sha256": array_sha256(weights),
        "reconstructed_normalized_weight_sha256": array_sha256(normalized),
        "saved_support_artifact": str(saved_path),
        "saved_support_artifact_sha256": actual_saved_sha,
        "saved_array_comparisons": comparisons,
        "support_cells": int(np.count_nonzero(weights > 0.0)),
        "passed": True,
    }


def audit_native_spot(
    native_store: Path,
    spot_initialization: str,
    cache: CacheContract,
) -> dict[str, Any]:
    """Reduce one native +1..+42 forecast and compare it with the cache row."""

    zarr = import_zarr()
    native_store = Path(native_store).resolve()
    zmetadata_sha = _zmetadata_sha256(native_store)
    group = zarr.open_consolidated(str(native_store), mode="r")
    attrs = dict(group.attrs)
    require(attrs.get("status") == "complete", "native FuXi store is incomplete")
    require(str(attrs.get("schema_version")) == "1.0", "native FuXi schema differs")
    lead_days = np.asarray(group["lead_day"][:], dtype=np.int16)
    require(
        np.array_equal(lead_days, np.arange(1, 43, dtype=np.int16)),
        "native lead-day labels are not 1..42",
    )
    members = np.asarray(group["member"][:], dtype=np.int16)
    require(np.array_equal(members, np.arange(51, dtype=np.int16)), "native members differ")
    channels = tuple(np.asarray(group["channel"][:]).astype(str).tolist())
    require(channels.count("tp") == 1, "native TP channel is ambiguous")
    tp_index = channels.index("tp")
    native_initializations = np.asarray(group["init"][:]).astype("datetime64[ns]").astype("datetime64[D]")
    spot = np.datetime64(spot_initialization, "D")
    matches = np.flatnonzero(native_initializations == spot)
    require(matches.size == 1, "spot initialization is absent or duplicated in native FuXi")
    source_index = int(matches[0])
    cache_matches = np.flatnonzero(cache.initializations == spot)
    require(cache_matches.size == 1, "spot initialization is absent or duplicated in cache")
    cache_index = int(cache_matches[0])
    require(
        int(cache.source_indices[cache_index]) == source_index,
        "spot native/cache source indices differ",
    )

    native_latitude = np.asarray(group["lat"][:], dtype=np.float64)
    native_longitude = np.asarray(group["lon"][:], dtype=np.float64)
    latitude_indices = np.flatnonzero(np.isin(native_latitude, cache.latitude))
    longitude_indices = np.flatnonzero(np.isin(native_longitude, cache.longitude))
    require(latitude_indices.size == 27 and longitude_indices.size == 27, "native India grid is absent")
    require(
        np.array_equal(native_latitude[latitude_indices], cache.latitude),
        "native/cache latitude differs",
    )
    require(
        np.array_equal(native_longitude[longitude_indices], cache.longitude),
        "native/cache longitude differs",
    )
    require(np.all(np.diff(latitude_indices) == 1), "native latitude selection is not contiguous")
    require(np.all(np.diff(longitude_indices) == 1), "native longitude selection is not contiguous")
    latitude_slice = slice(int(latitude_indices[0]), int(latitude_indices[-1]) + 1)
    longitude_slice = slice(int(longitude_indices[0]), int(longitude_indices[-1]) + 1)

    period_starts = np.asarray(group["forecast_period_start"][source_index]).astype("datetime64[ns]").astype("datetime64[D]")
    period_ends = np.asarray(group["forecast_period_end"][source_index]).astype("datetime64[ns]").astype("datetime64[D]")
    expected_starts = spot + np.arange(0, 42).astype("timedelta64[D]")
    expected_ends = spot + np.arange(1, 43).astype("timedelta64[D]")
    require(np.array_equal(period_starts, expected_starts), "native period starts differ")
    require(np.array_equal(period_ends, expected_ends), "native period ends differ")

    raw = np.asarray(
        group["forecast"][
            source_index,
            slice(None),
            slice(None),
            tp_index,
            latitude_slice,
            longitude_slice,
        ]
    )
    reconstructed = weekly_native_tp(raw)
    cached = np.asarray(cache.values[cache_index]).copy()
    _exact_array(reconstructed, cached, "native/cache weekly spot")

    records = cache.cache_manifest.get("records", [])
    require(isinstance(records, list) and len(records) == 2080, "cache record ledger differs")
    record = records[cache_index]
    require(int(record.get("output_index")) == cache_index, "spot cache record index differs")
    require(int(record.get("source_init_index")) == source_index, "spot source record index differs")
    require(record.get("initialization") == spot_initialization, "spot cache record date differs")
    reconstructed_hash = array_sha256(reconstructed)
    cached_hash = array_sha256(cached)
    require(reconstructed_hash == cached_hash, "spot logical weekly hashes differ")
    require(
        cached_hash == record.get("tp_members_weekly_sha256"),
        "spot logical hash differs from cache manifest",
    )
    part_path = Path(str(record.get("part_path", ""))).resolve()
    require(part_path.is_file(), "spot cache part is absent")
    part_sha = sha256_file(part_path)
    require(part_sha == record.get("part_sha256"), "spot cache part bytes differ")

    archive_manifest = Path(str(attrs.get("archive_manifest", ""))).resolve()
    require(archive_manifest.is_file(), "native archive manifest is absent")
    archive_manifest_sha = sha256_file(archive_manifest)
    require(
        archive_manifest_sha == attrs.get("archive_manifest_sha256"),
        "native archive manifest bytes differ",
    )
    return {
        "native_store": str(native_store),
        "native_zmetadata_sha256": zmetadata_sha,
        "native_root_attrs_sha256": canonical_json_sha256(attrs),
        "native_archive_manifest": str(archive_manifest),
        "native_archive_manifest_sha256": archive_manifest_sha,
        "lead_day_labels": lead_days.tolist(),
        "lead_day_labels_sha256": array_sha256(lead_days),
        "lead_day_bounds": [int(lead_days.min()), int(lead_days.max())],
        "forecast_period_start_sha256": array_sha256(period_starts),
        "forecast_period_end_sha256": array_sha256(period_ends),
        "spot_initialization": spot_initialization,
        "spot_role": "last retained 2002-2017 training initialization",
        "source_init_index": source_index,
        "cache_index": cache_index,
        "first_period_start": np.datetime_as_string(period_starts[0], unit="D"),
        "first_period_end": np.datetime_as_string(period_ends[0], unit="D"),
        "last_period_start": np.datetime_as_string(period_starts[-1], unit="D"),
        "last_period_end": np.datetime_as_string(period_ends[-1], unit="D"),
        "native_daily_tp_sha256": array_sha256(raw),
        "reconstructed_weekly_tp_sha256": reconstructed_hash,
        "cached_weekly_tp_sha256": cached_hash,
        "cache_manifest_weekly_tp_sha256": record.get("tp_members_weekly_sha256"),
        "maximum_absolute_difference": float(
            np.max(np.abs(reconstructed.astype(np.float64) - cached.astype(np.float64)))
        ),
        "spot_part": str(part_path),
        "spot_part_sha256": part_sha,
        "byte_exact_match": True,
        "passed": True,
    }


def split_comparisons(
    splits: SplitReconstruction,
    initializations: np.ndarray,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    retained = parent.get("retained_initializations", {})
    require(isinstance(retained, Mapping), "parent retained-initialization ledger is absent")
    result: dict[str, Any] = {}
    for name, indices in splits.indices.items():
        actual_dates = initializations[indices]
        expected_dates = np.asarray(retained.get(name, []), dtype="datetime64[D]")
        actual_hash = array_sha256(actual_dates)
        expected_hash = array_sha256(expected_dates)
        item = {
            "count": int(indices.size),
            "indices_sha256": array_sha256(indices),
            "initializations_sha256": actual_hash,
            "frozen_initializations_sha256": expected_hash,
            "exact_match_to_frozen_manifest": bool(np.array_equal(actual_dates, expected_dates)),
            "first_initialization": np.datetime_as_string(actual_dates[0], unit="D"),
            "last_initialization": np.datetime_as_string(actual_dates[-1], unit="D"),
            "last_outcome_end": np.datetime_as_string(
                splits.outcome_ends[indices].max(), unit="D"
            ),
        }
        require(item["exact_match_to_frozen_manifest"], f"{name} IDs differ from parent")
        result[name] = item

    counts = {name: int(indices.size) for name, indices in splits.indices.items()}
    require(counts == EXPECTED_SPLIT_COUNTS, "independent split counts differ")
    require(counts == parent.get("split_counts_archive"), "parent split counts differ")
    fit_indices = np.asarray(parent.get("normalization", {}).get("fit_indices", []), dtype=np.int64)
    require(
        np.array_equal(fit_indices, splits.indices["train"]),
        "parent normalization fit indices differ from training split",
    )
    result["counts"] = counts
    result["normalization_fit_indices_sha256"] = array_sha256(fit_indices)
    result["normalization_fit_indices_exact_match"] = True
    result["purge_rule"] = {
        "outcome_end": "initialization + 42 days",
        "train_retained_if": "outcome_end < 2018-01-01",
        "validation_retained_if": "outcome_end < 2020-01-01",
        "reused_development_2020_2021": (
            "all retained; outcome labels may enter 2022, which is inside the fixed "
            "2002-2022 observation-source contract"
        ),
        "latest_reused_development_outcome_end": np.datetime_as_string(
            splits.outcome_ends[splits.indices["test"]].max(), unit="D"
        ),
    }
    result["passed"] = True
    return result


def write_split_table(
    path: Path, splits: SplitReconstruction, initializations: np.ndarray
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cache_index",
                "initialization",
                "outcome_end_plus_42",
                "candidate_role",
                "final_role",
                "retained",
                "boundary_crossing_embargo",
            ),
        )
        writer.writeheader()
        for index, start in enumerate(initializations):
            final_role = str(splits.final_roles[index])
            writer.writerow(
                {
                    "cache_index": index,
                    "initialization": np.datetime_as_string(start, unit="D"),
                    "outcome_end_plus_42": np.datetime_as_string(
                        splits.outcome_ends[index], unit="D"
                    ),
                    "candidate_role": str(splits.candidate_roles[index]),
                    "final_role": final_role,
                    "retained": str(final_role != "embargo").lower(),
                    "boundary_crossing_embargo": str(final_role == "embargo").lower(),
                }
            )


def write_source_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "source table is empty")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "audit_receipt.json" and ".tmp" not in path.name
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling output directory without replacement."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    require(source.parent == destination.parent, "publication directories are not siblings")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        require(stat.S_ISDIR(metadata.st_mode), "staging output is not a directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        require(renameat2 is not None, "Linux renameat2 is required")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            descriptor,
            os.fsencode(source.name),
            descriptor,
            os.fsencode(destination.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(descriptor)


def verify_parent_contract(parent_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_path = Path(parent_path).resolve()
    require(parent_path.is_file(), "frozen training manifest is absent")
    parent_sha = sha256_file(parent_path)
    require(
        parent_sha == EXPECTED_PARENT_MANIFEST_SHA256,
        "frozen training manifest bytes differ from the accepted run",
    )
    parent = read_json(parent_path)
    require(parent.get("experiment") == PARENT_EXPERIMENT, "parent experiment differs")
    require(parent.get("status") == "complete", "parent training is not complete")
    require(parent.get("mode") == "full", "parent training is not a full run")
    contract = parent.get("contract", {})
    require(contract.get("revision") == CONTRACT_REVISION, "parent alignment revision differs")
    require(
        contract.get("target_day_offsets") == list(TARGET_DAY_OFFSETS),
        "parent target offsets differ",
    )
    require(
        contract.get("initialization_day_included_in_target") is False,
        "parent target includes initialization day",
    )
    require(contract.get("sealed_2025_target_opened") is False, "parent opened sealed target")
    artifact_result = verify_declared_artifacts(
        parent_path.parent, parent.get("artifact_sha256", {}), "parent artifact"
    )
    snapshot_result = verify_declared_artifacts(
        parent_path.parent,
        parent.get("source_snapshot_sha256", {}),
        "parent source snapshot",
    )
    return parent, {
        "manifest": str(parent_path),
        "manifest_sha256": parent_sha,
        "artifact_verification": artifact_result,
        "source_snapshot_verification": snapshot_result,
    }


def run_alignment_audit(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output).resolve()
    output_root = DEFAULT_OUTPUT_ROOT.resolve()
    require(output_root in output.parents, "output must remain under the alignment-audit resultsv3 root")
    require(not output.exists(), f"audit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    require(not staging.exists(), "staging output already exists")
    staging.mkdir()
    try:
        parent_path = Path(args.parent_manifest).resolve()
        parent, parent_receipt = verify_parent_contract(parent_path)
        cache = load_and_verify_cache(args.cache, parent)
        splits = reconstruct_splits(cache.initializations)
        split_receipt = split_comparisons(splits, cache.initializations, parent)
        require(
            args.spot_initialization
            == np.datetime_as_string(
                cache.initializations[splits.indices["train"]][-1], unit="D"
            ),
            "spot must be the last retained training initialization",
        )

        imd = load_imd_and_reconstruct_targets(
            args.imd_root, cache.initializations, cache.latitude, cache.longitude
        )
        require(
            list(imd.opened_stores) == parent.get("observation_stores"),
            "independent IMD store list differs from frozen training manifest",
        )
        support_receipt = audit_scoring_support(
            args.spatial_store, imd, cache, parent_path, parent
        )
        native_receipt = audit_native_spot(
            args.native_store, args.spot_initialization, cache
        )

        source_copy = staging / "code/src" / SOURCE_PATH.name
        test_copy = staging / "code/tests" / TEST_SOURCE_PATH.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        test_copy.parent.mkdir(parents=True, exist_ok=True)
        require(TEST_SOURCE_PATH.is_file(), "focused audit test source is absent")
        shutil.copy2(SOURCE_PATH, source_copy)
        shutil.copy2(TEST_SOURCE_PATH, test_copy)
        require(
            sha256_file(source_copy) == sha256_file(SOURCE_PATH),
            "auditor source snapshot differs",
        )
        require(
            sha256_file(test_copy) == sha256_file(TEST_SOURCE_PATH),
            "auditor test snapshot differs",
        )

        tables = staging / "tables"
        receipts = staging / "receipts"
        tables.mkdir()
        receipts.mkdir()
        write_split_table(tables / "split_ids.csv", splits, cache.initializations)
        write_source_table(tables / "imd_source_hashes.csv", imd.source_rows)
        write_json(receipts / "logical_array_hashes.json", imd.logical_hashes)
        write_json(receipts / "split_comparison.json", split_receipt)
        write_json(receipts / "support_comparison.json", support_receipt)
        write_json(receipts / "native_spot_check.json", native_receipt)
        write_json(
            receipts / "cache_comparison.json",
            {
                "paths": {name: str(path) for name, path in cache.paths.items()},
                "byte_comparisons": cache.byte_comparisons,
                "logical_coordinate_hashes": {
                    "initializations_sha256": array_sha256(cache.initializations),
                    "source_init_indices_sha256": array_sha256(cache.source_indices),
                    "latitude_sha256": array_sha256(cache.latitude),
                    "longitude_sha256": array_sha256(cache.longitude),
                },
                "source_fingerprint": cache.metadata.get("source_fingerprint"),
                "passed": True,
            },
        )

        receipt: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "status": "passed",
            "created_utc": utc_now(),
            "output_path": str(output),
            "independence": {
                "imports_training_implementation": False,
                "imports_member_cache_builder": False,
                "reopened_daily_imd": True,
                "reimplemented_cf_time_decode": True,
                "reimplemented_plus_1_through_42_target_construction": True,
                "reimplemented_split_and_embargo_rules": True,
                "reimplemented_native_daily_to_weekly_conversion": True,
            },
            "safety": {
                "opened_observation_years": list(OBSERVATION_YEARS),
                "maximum_observation_year_opened": MAX_OPEN_OBSERVATION_YEAR,
                "maximum_target_label_opened": np.datetime_as_string(
                    imd.target_dates.max(), unit="D"
                ),
                "opened_2025_observation": False,
                "sealed_2025_target_opened": False,
                "used_glob_to_discover_observation_years": False,
            },
            "contract": {
                "revision": CONTRACT_REVISION,
                "reforecast_initialization_years": list(
                    range(2002, 2022)
                ),
                "observation_source_years": list(OBSERVATION_YEARS),
                "target_day_offsets": list(TARGET_DAY_OFFSETS),
                "initialization_day_included": False,
                "weekly_target_shape": list(imd.weekly_truth.shape),
                "support_cells": int(np.count_nonzero(imd.support_mask)),
                "split_counts": split_receipt["counts"],
                "minimum_target_label": np.datetime_as_string(
                    imd.target_dates.min(), unit="D"
                ),
                "maximum_target_label": np.datetime_as_string(
                    imd.target_dates.max(), unit="D"
                ),
            },
            "inputs": {
                "parent_training": parent_receipt,
                "cache": {name: str(path) for name, path in cache.paths.items()},
                "cache_byte_comparisons": cache.byte_comparisons,
                "native_store": str(Path(args.native_store).resolve()),
                "imd_root": str(Path(args.imd_root).resolve()),
                "spatial_store": str(Path(args.spatial_store).resolve()),
            },
            "results": {
                "weekly_target_reconstruction": "passed",
                "source_and_zmetadata_hashing": "passed",
                "split_and_embargo_comparison": "passed",
                "scoring_support_byte_comparison": "passed",
                "native_lead_1_through_42_spot_check": "passed",
                "cache_file_byte_comparison": "passed",
                "all_frozen_parent_artifact_hashes": "passed",
            },
            "logical_array_hashes": imd.logical_hashes,
            "split_audit": split_receipt,
            "support_audit": support_receipt,
            "native_spot_check": native_receipt,
            "historical_target_equality_limitation": {
                "original_training_run_persisted_logical_weekly_target_hashes": False,
                "retrospective_equality_to_consumed_target_tensors_provable": False,
                "statement": (
                    "The frozen training run did not persist logical hashes or copies of "
                    "its weekly IMD target tensors. This receipt independently reconstructs "
                    "the +1..+42 targets from currently hash-bound source bytes and verifies "
                    "the frozen source code, cache bytes, split IDs, support arrays, and a "
                    "native-to-cache spot row. Historical byte equality to the unrecorded "
                    "target tensors consumed during training cannot be proven beyond this "
                    "source+code+contract reconstruction."
                ),
            },
            "source_snapshot_sha256": {
                f"code/src/{SOURCE_PATH.name}": sha256_file(source_copy),
                f"code/tests/{TEST_SOURCE_PATH.name}": sha256_file(test_copy),
            },
            "artifact_sha256": artifact_hashes(staging),
        }
        write_json(staging / "audit_receipt.json", receipt)
        rename_noreplace(staging, output)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"audit_full_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--native-store", type=Path, default=DEFAULT_NATIVE_STORE)
    parser.add_argument("--imd-root", type=Path, default=DEFAULT_IMD_ROOT)
    parser.add_argument("--spatial-store", type=Path, default=DEFAULT_SPATIAL_STORE)
    parser.add_argument("--spot-initialization", default=DEFAULT_SPOT_INITIALIZATION)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is None:
        args.output = default_output()
    receipt = run_alignment_audit(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
