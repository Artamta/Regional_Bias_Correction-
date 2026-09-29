#!/usr/bin/env python3
"""Frozen 2024 adapter-to-station external-target robustness evaluation.

This evaluator is deliberately narrower than the historical PiggyCast station
analysis.  It never trains, selects, calibrates, or blends a forecast.  It
scores immutable gridded predictions on the exact 30 preregistered 2024 JJAS
initializations and six lead weeks against a separately cleaned rain-gauge
target.  Gauge values are median-combined inside fixed 1.5-degree cells, and
every forecast is scored on the same cells in each case.

The cleaned recent-station container also holds later rows.  To preserve the
2025 boundary, the loader first checks a row's date and station identifier and
converts rainfall only for the exact union of required 2024 verification days.
Those selected rows are written to a deterministic, hashed 2024-only snapshot
before any metric is computed.  No 2025 rainfall value enters an array, metric,
or output.

This is a frozen independent-observational-target sensitivity analysis, not an
untouched temporal test: 2024 IMD performance and the older station analysis
were already known.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = PROJECT_ROOT.parent
BASE_RESULT_ROOT = (
    CLEAN_ROOT
    / "studies"
    / "fuxi_imd_adapter_benchmark_v1"
    / "results"
    / "full_context_jjas_2022_2024_job91439"
)
BASE_PREDICTION_STORE = BASE_RESULT_ROOT / "predictions.zarr"
BASE_MANIFEST = BASE_RESULT_ROOT / "manifest.json"
STATION_ROOT = Path("/home/raj.ayush/saptarishi_stuff/station_architecture_comparison")
STATION_DATA = STATION_ROOT / "data"
COVERAGE_REFERENCE = STATION_ROOT / "metrics" / "station_case_coverage.csv"
SPATIAL_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/spatial/spatial_support.zarr"
)

BASE_METHODS = ("raw_fuxi", "log_bias", "selected_adapter")
EXTENDED_METHODS = ("raw_identity", "raw_identity_raw_mean_preserved")
EXTENDED_STORE_METHODS = (
    "raw_fuxi",
    "log_bias",
    "legacy_anchored_adapter",
    *EXTENDED_METHODS,
)
LEAD_WEEKS = (1, 2, 3, 4, 5, 6)
GRID_SHAPE = (27, 27)
SUPPORT_CELLS = 171
MAPPED_STATIONS = 380
MAPPED_CELLS = 100
ELIGIBLE_STATIONS = 382
MINIMUM_WEEK_DAYS = 6
MINIMUM_COMMON_CELLS = 20
MAXIMUM_WEEKLY_STATION_RAIN_MM_DAY = 150.0
MAXIMUM_GRID_DISTANCE_KM = 150.0
EXPECTED_MAXIMUM_GRID_DISTANCE_KM = 119.27221839722655
EXPECTED_CELL_COUNT_RANGE = (92, 99)
EXPECTED_STATION_COUNT_RANGE = (295, 339)
RECENT_CONTAINER_ROWS = 543_518
RECENT_CONTAINER_DATE_MIN = "2023-12-31"
RECENT_CONTAINER_DATE_MAX = "2025-02-10"
RECENT_CONTAINER_2025_PLUS_ROWS = 45_910
BOOTSTRAP_DRAWS = 2_000
PRIMARY_BLOCK_LENGTH = 13
SENSITIVITY_BLOCK_LENGTHS = (4, 8)
BOOTSTRAP_SEED = 20_260_822
PROJECTION_CLOSURE_TOLERANCE = 2.0e-6

EXACT_INITIALIZATION_STRINGS = (
    "2024-06-03",
    "2024-06-20",
    "2024-06-24",
    "2024-06-27",
    "2024-07-01",
    "2024-07-04",
    "2024-07-08",
    "2024-07-11",
    "2024-07-15",
    "2024-07-18",
    "2024-07-22",
    "2024-07-25",
    "2024-07-29",
    "2024-08-01",
    "2024-08-08",
    "2024-08-12",
    "2024-08-15",
    "2024-08-19",
    "2024-08-22",
    "2024-08-26",
    "2024-08-29",
    "2024-09-02",
    "2024-09-05",
    "2024-09-09",
    "2024-09-12",
    "2024-09-16",
    "2024-09-19",
    "2024-09-23",
    "2024-09-26",
    "2024-09-30",
)
EXACT_INITIALIZATIONS = np.asarray(EXACT_INITIALIZATION_STRINGS, dtype="datetime64[D]")

BASE_TREE_SHA256 = "109ffe000bb2a4e68c60ee9dc3b4dc50af9285ebc554f2b1dd23400b6186271a"
BASE_MANIFEST_SHA256 = (
    "260fd5e344dead5359c482e74479ccbe1e3775fc5f0d040a4bbffdd412ac83e9"
)
SPATIAL_TREE_SHA256 = "2ceb0bbbe0c032d5435ca218d9d962e0c3bee6e900c7562c006bbca2d2fe7a37"
FROZEN_FILE_SHA256 = {
    STATION_DATA
    / "recent_daily_rainfall_clean.csv.gz": "202240d369ab69956f2712baaa39bbc1c6487e5c06d64adfe34f1749bf00c070",
    STATION_DATA
    / "station_rainfall_climatology.csv.gz": "8038dbabd5ac77d71b1f433070a4469dcb02ee07af119dc19520035ac66f759c",
    STATION_DATA
    / "station_locations.csv": "c1de404b4f1c38cb994149da59f5b034b7e57164afc189cd8443e4af23573b42",
    STATION_DATA
    / "station_grid_collocation.csv": "7c14d735ba27d9e031ccaf3b65800c9113e9667df1a3a463239aaff192c28d09",
    STATION_DATA
    / "cleaning_manifest.json": "ba5b511cace7063d6ad6f748950f72179cfd73125f1e8e1487cd8cf8b5388412",
    COVERAGE_REFERENCE: "ab22e5b14c8e24bd7bbc22c011b0962bf9a0f01326c3a80086e981a859cf4ad6",
}

GRID_CELL_PATTERN = re.compile(r"^grid_(\d{2})_(\d{2})$")


class StationEvaluationContractError(RuntimeError):
    """Raised when frozen inputs or derived outputs violate the E3 contract."""


@dataclass(frozen=True)
class PredictionBundle:
    all_initializations: np.ndarray
    initializations: np.ndarray
    lead_weeks: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    adapter_support: np.ndarray
    full_legacy_predictions: Mapping[str, np.ndarray]
    predictions: Mapping[str, np.ndarray]
    extended_manifest: Mapping[str, Any] | None
    extended_manifest_sha256: str | None
    extended_tree_sha256: str | None


@dataclass(frozen=True)
class StationTruthSelection:
    rows: pd.DataFrame
    container_rows: int
    container_date_min: str
    container_date_max: str
    unselected_2025_plus_rows: int


@dataclass(frozen=True)
class StationArrays:
    location_ids: tuple[str, ...]
    cell_flat_indices: np.ndarray
    daily_dates: np.ndarray
    daily_rain: np.ndarray
    climatology: np.ndarray


@dataclass(frozen=True)
class CellTarget:
    truth: np.ndarray
    climatology: np.ndarray
    station_count_by_cell: np.ndarray
    station_location_count: int
    station_grid_cell_count: int


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    candidate: str
    reference: str
    primary: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise StationEvaluationContractError(f"empty artifact tree: {root}")
    for item in files:
        if item.is_symlink():
            raise StationEvaluationContractError(
                f"artifact tree contains symlink: {item}"
            )
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def float32_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values).astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StationEvaluationContractError(
            f"cannot read JSON object: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise StationEvaluationContractError(f"JSON root is not an object: {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_write_deterministic_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")
    os.replace(temporary, path)


@contextmanager
def fresh_atomic_output(output: Path) -> Iterable[Path]:
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"fresh output required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:10]}"
    )
    staging.mkdir(mode=0o750)
    try:
        yield staging
        if not (staging / "manifest.json").is_file():
            raise StationEvaluationContractError("staging output has no manifest")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_frozen_sources() -> dict[str, str]:
    """Hash every preregistered source before loading scientific arrays."""

    observed: dict[str, str] = {}
    file_contract = {BASE_MANIFEST: BASE_MANIFEST_SHA256, **FROZEN_FILE_SHA256}
    for path, expected in file_contract.items():
        if not path.is_file() or path.is_symlink():
            raise StationEvaluationContractError(
                f"missing/nonregular frozen file: {path}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise StationEvaluationContractError(
                f"frozen input hash mismatch for {path}: expected {expected}, found {actual}"
            )
        observed[str(path)] = actual
    for path, expected in (
        (BASE_PREDICTION_STORE, BASE_TREE_SHA256),
        (SPATIAL_STORE, SPATIAL_TREE_SHA256),
    ):
        if not path.is_dir() or path.is_symlink():
            raise StationEvaluationContractError(
                f"missing frozen artifact tree: {path}"
            )
        actual = sha256_tree(path)
        if actual != expected:
            raise StationEvaluationContractError(
                f"frozen tree hash mismatch for {path}: expected {expected}, found {actual}"
            )
        observed[str(path)] = actual
    return observed


def validate_exact_initializations(values: np.ndarray) -> np.ndarray:
    dates = np.asarray(values).astype("datetime64[D]")
    if dates.shape != (30,) or np.isnat(dates).any():
        raise StationEvaluationContractError("E3 requires exactly 30 valid dates")
    if not np.array_equal(dates, EXACT_INITIALIZATIONS):
        raise StationEvaluationContractError(
            "initialization dates/order changed from E3"
        )
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=int)
    if not np.all(years == 2024) or np.any(years >= 2025):
        raise StationEvaluationContractError("only 2024 initializations are admissible")
    return dates


def _validate_base_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "status": "complete",
        "support_cells": SUPPORT_CELLS,
        "audit_case_count": 100,
        "audit_initialization_years": [2022, 2023, 2024],
        "audit_initialization_months": [6, 7, 8, 9],
        "final_initialization_year_quarantined": 2025,
        "selected_model": "normal_climo_model",
        "selected_alpha": 1.0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise StationEvaluationContractError(
                f"base manifest {key!r} changed: {manifest.get(key)!r}"
            )
    counts = {
        str(key): int(value) for key, value in manifest.get("audit_counts", {}).items()
    }
    if counts != {"2022": 35, "2023": 35, "2024": 30}:
        raise StationEvaluationContractError(f"base audit counts changed: {counts}")
    outputs = manifest.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or outputs.get("predictions.zarr") != BASE_TREE_SHA256
    ):
        raise StationEvaluationContractError(
            "base manifest does not bind the frozen Zarr"
        )


def _dataset_coord(dataset: xr.Dataset, name: str) -> np.ndarray:
    if name not in dataset.coords:
        raise StationEvaluationContractError(
            f"prediction store lacks coordinate {name}"
        )
    return np.asarray(dataset.coords[name].values)


def load_base_predictions() -> PredictionBundle:
    manifest = _read_json(BASE_MANIFEST)
    _validate_base_manifest(manifest)
    with xr.open_zarr(BASE_PREDICTION_STORE, consolidated=True) as dataset:
        required_variables = {
            "prediction",
            "adapter_support",
            "truth_imd",
            "fixed_imd_climatology",
        }
        if set(dataset.data_vars) != required_variables:
            raise StationEvaluationContractError(
                f"base Zarr variables changed: {sorted(dataset.data_vars)}"
            )
        methods = tuple(str(value) for value in _dataset_coord(dataset, "method"))
        leads = _dataset_coord(dataset, "lead_week").astype(np.int16)
        all_dates = _dataset_coord(dataset, "init").astype("datetime64[D]")
        latitude = _dataset_coord(dataset, "latitude").astype(np.float64)
        longitude = _dataset_coord(dataset, "longitude").astype(np.float64)
        if methods != BASE_METHODS or tuple(leads.tolist()) != LEAD_WEEKS:
            raise StationEvaluationContractError(
                "base method or lead coordinate changed"
            )
        expected_latitude = 39.0 - 1.5 * np.arange(27)
        expected_longitude = 60.0 + 1.5 * np.arange(27)
        if not np.array_equal(latitude, expected_latitude) or not np.array_equal(
            longitude, expected_longitude
        ):
            raise StationEvaluationContractError("base 1.5-degree coordinates changed")
        years = pd.DatetimeIndex(all_dates).year.to_numpy(dtype=int)
        counts = {
            int(year): int(np.count_nonzero(years == year)) for year in np.unique(years)
        }
        if counts != {2022: 35, 2023: 35, 2024: 30} or np.any(years >= 2025):
            raise StationEvaluationContractError(f"base date tier changed: {counts}")
        selected_positions = np.flatnonzero(years == 2024)
        selected_dates = validate_exact_initializations(all_dates[selected_positions])
        support = np.asarray(dataset.adapter_support.load().values, dtype=bool)
        if support.shape != GRID_SHAPE or int(support.sum()) != SUPPORT_CELLS:
            raise StationEvaluationContractError("base adapter support changed")
        full_values = np.asarray(
            dataset.prediction.sel({"method": list(BASE_METHODS)}).load().values,
            dtype=np.float32,
        )
        if full_values.shape != (3, 100, 6, 27, 27):
            raise StationEvaluationContractError(
                f"base prediction shape changed: {full_values.shape}"
            )
        if (
            dataset.attrs.get("units") != "mm day-1"
            or dataset.attrs.get("weekly_alignment")
            != "W1 init+0..6 through W6 init+35..41"
        ):
            raise StationEvaluationContractError("base units/alignment changed")
    legacy = {name: full_values[index] for index, name in enumerate(BASE_METHODS)}
    for name, values in legacy.items():
        if not np.isfinite(values[..., support]).all() or np.any(
            values[..., support] < 0.0
        ):
            raise StationEvaluationContractError(f"invalid frozen values for {name}")
    predictions = {
        name: np.asarray(values[selected_positions], dtype=np.float32)
        for name, values in legacy.items()
    }
    return PredictionBundle(
        all_initializations=all_dates,
        initializations=selected_dates,
        lead_weeks=leads,
        latitude=latitude,
        longitude=longitude,
        adapter_support=support,
        full_legacy_predictions=legacy,
        predictions=predictions,
        extended_manifest=None,
        extended_manifest_sha256=None,
        extended_tree_sha256=None,
    )


def load_spatial_weights(bundle: PredictionBundle) -> np.ndarray:
    with xr.open_zarr(SPATIAL_STORE, consolidated=True) as dataset:
        if "india_area_weight_km2" not in dataset:
            raise StationEvaluationContractError(
                "spatial store lacks India area weights"
            )
        if not np.array_equal(
            _dataset_coord(dataset, "latitude"), bundle.latitude
        ) or not np.array_equal(_dataset_coord(dataset, "longitude"), bundle.longitude):
            raise StationEvaluationContractError(
                "spatial/prediction coordinates differ"
            )
        weights = np.asarray(
            dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
    if (
        weights.shape != GRID_SHAPE
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
    ):
        raise StationEvaluationContractError("India area weights are invalid")
    if int(np.count_nonzero(weights > 0.0)) != 174:
        raise StationEvaluationContractError("India area-weight support changed")
    if int(np.count_nonzero((weights > 0.0) & bundle.adapter_support)) != SUPPORT_CELLS:
        raise StationEvaluationContractError(
            "adapter support is not fully area weighted"
        )
    return weights


def _resolve_extended_store(value: Path) -> tuple[Path, Path]:
    path = Path(value).expanduser().resolve()
    if path.name == "predictions.zarr" and path.is_dir():
        store, root = path, path.parent
    else:
        root, store = path, path / "predictions.zarr"
    manifest = root / "manifest.json"
    if not root.is_dir() or not store.is_dir() or not manifest.is_file():
        raise StationEvaluationContractError(
            "extended predictions must be a completed root (or predictions.zarr) "
            "with sibling manifest.json"
        )
    if root.is_symlink() or store.is_symlink() or manifest.is_symlink():
        raise StationEvaluationContractError("extended artifact may not be a symlink")
    return store, manifest


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _flatten_strings(item)


def _validate_extended_manifest(
    manifest: Mapping[str, Any],
    tree_hash: str,
    *,
    require_canonical: bool = True,
) -> None:
    if require_canonical:
        canonical_expected = {
            "status": "complete",
            "canonical": True,
            "scientific_eligible": True,
            "smoke": False,
        }
        for key, expected in canonical_expected.items():
            if manifest.get(key) != expected:
                raise StationEvaluationContractError(
                    f"extended E2 production gate {key!r} changed: "
                    f"{manifest.get(key)!r}"
                )
        bootstrap = manifest.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise StationEvaluationContractError(
                "extended E2 manifest lacks bootstrap contract"
            )
        canonical_bootstrap = {
            "draws": 10_000,
            "block_length_initializations": 13,
            "seed": 20_260_822,
        }
        for key, expected in canonical_bootstrap.items():
            if bootstrap.get(key) != expected:
                raise StationEvaluationContractError(
                    f"extended E2 canonical bootstrap {key!r} changed"
                )
        if bootstrap.get("canonical_contract") != canonical_bootstrap:
            raise StationEvaluationContractError(
                "extended E2 canonical bootstrap contract changed"
            )
        diagnostics = bootstrap.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise StationEvaluationContractError(
                "extended E2 manifest lacks bootstrap diagnostics"
            )
        for key in (
            "year_stratified",
            "circular_within_year",
            "equal_marginal_inclusion_by_design",
            "no_year_crossing",
            "all_six_leads_retained_per_start",
        ):
            if diagnostics.get(key) is not True:
                raise StationEvaluationContractError(
                    f"extended E2 bootstrap diagnostic {key!r} is not true"
                )
        try:
            mean_multiplicity = float(
                diagnostics["mean_multiplicity_across_initializations"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StationEvaluationContractError(
                "extended E2 mean multiplicity diagnostic is invalid"
            ) from exc
        if not np.isfinite(mean_multiplicity) or abs(mean_multiplicity - 1.0) > 1.0e-12:
            raise StationEvaluationContractError(
                "extended E2 mean initialization multiplicity is not one"
            )
    elif manifest.get("status") not in {"complete", "complete_noncanonical_smoke"}:
        raise StationEvaluationContractError(
            "extended E2 schema-only artifact is not complete"
        )
    if manifest.get("final_2025_store_opened") is not False:
        raise StationEvaluationContractError("extended E2 artifact does not seal 2025")
    if manifest.get("audit_years") != [2022, 2023, 2024]:
        raise StationEvaluationContractError("extended E2 audit years changed")
    counts = {
        str(key): int(value) for key, value in manifest.get("audit_counts", {}).items()
    }
    if counts != {"2022": 35, "2023": 35, "2024": 30}:
        raise StationEvaluationContractError("extended E2 case counts changed")
    if manifest.get("methods") != list(EXTENDED_STORE_METHODS):
        raise StationEvaluationContractError("extended E2 method contract changed")
    projection = manifest.get("projection")
    projection_expected = {
        "weighting_contract": "fixed_forecast_time_india_area_x_frozen_adapter_support",
        "uses_weekly_imd_coverage": False,
        "uses_observed_rainfall_values": False,
        "support_cells": SUPPORT_CELLS,
        "post_hoc": True,
    }
    if not isinstance(projection, Mapping):
        raise StationEvaluationContractError(
            "extended manifest lacks projection contract"
        )
    for key, expected in projection_expected.items():
        if projection.get(key) != expected:
            raise StationEvaluationContractError(
                f"extended projection {key!r} changed: {projection.get(key)!r}"
            )
    try:
        recorded_closure = float(projection["maximum_float32_closure_mm_day"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StationEvaluationContractError(
            "invalid extended projection closure"
        ) from exc
    if (
        not np.isfinite(recorded_closure)
        or recorded_closure > PROJECTION_CLOSURE_TOLERANCE
    ):
        raise StationEvaluationContractError(
            "extended projection closure exceeds tolerance"
        )
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("predictions.zarr") != tree_hash
    ):
        raise StationEvaluationContractError("extended manifest does not bind its Zarr")
    if manifest.get("array_sha256_contract") != (
        "sha256 of contiguous C-order little-endian float32 raw bytes; no header"
    ):
        raise StationEvaluationContractError("extended array hash contract changed")
    array_hashes = manifest.get("array_sha256")
    if (
        not isinstance(array_hashes, Mapping)
        or set(array_hashes)
        != {
            "raw_identity",
            "raw_identity_raw_mean_preserved",
            "raw_identity_residual",
        }
        or not all(_is_sha256(value) for value in array_hashes.values())
    ):
        raise StationEvaluationContractError("extended array hash inventory changed")
    provenance = manifest.get("input_provenance")
    if BASE_TREE_SHA256 not in set(_flatten_strings(provenance)):
        raise StationEvaluationContractError(
            "extended input provenance does not name the immutable base prediction hash"
        )


def attach_extended_predictions(
    bundle: PredictionBundle,
    weights: np.ndarray,
    extended: Path,
    *,
    require_canonical: bool = True,
) -> PredictionBundle:
    store, manifest_path = _resolve_extended_store(extended)
    manifest = _read_json(manifest_path)
    tree_hash = sha256_tree(store)
    _validate_extended_manifest(
        manifest, tree_hash, require_canonical=require_canonical
    )
    with xr.open_zarr(store, consolidated=True) as dataset:
        expected_variables = {
            "prediction",
            "truth_imd",
            "fixed_imd_climatology",
            "weekly_imd_coverage",
            "india_area_weight_km2",
            "adapter_support",
        }
        if set(dataset.data_vars) != expected_variables:
            raise StationEvaluationContractError(
                f"extended Zarr variables changed: {sorted(dataset.data_vars)}"
            )
        methods = tuple(str(value) for value in _dataset_coord(dataset, "method"))
        if methods != EXTENDED_STORE_METHODS:
            raise StationEvaluationContractError("extended method order changed")
        if not np.array_equal(
            _dataset_coord(dataset, "init").astype("datetime64[D]"),
            bundle.all_initializations,
        ):
            raise StationEvaluationContractError(
                "extended initialization coordinate changed"
            )
        if not np.array_equal(_dataset_coord(dataset, "lead_week"), bundle.lead_weeks):
            raise StationEvaluationContractError("extended lead coordinate changed")
        if not np.array_equal(
            _dataset_coord(dataset, "latitude"), bundle.latitude
        ) or not np.array_equal(_dataset_coord(dataset, "longitude"), bundle.longitude):
            raise StationEvaluationContractError("extended grid coordinate changed")
        support = np.asarray(dataset.adapter_support.load().values, dtype=bool)
        extended_weights = np.asarray(
            dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
        expected_projection_weights = np.where(bundle.adapter_support, weights, 0.0)
        if not np.array_equal(support, bundle.adapter_support) or not np.array_equal(
            extended_weights, expected_projection_weights
        ):
            raise StationEvaluationContractError(
                "extended support/area weights changed"
            )
        values = np.asarray(
            dataset.prediction.sel({"method": list(EXTENDED_STORE_METHODS)})
            .load()
            .values,
            dtype=np.float32,
        )
    if values.shape != (5, 100, 6, 27, 27):
        raise StationEvaluationContractError(
            f"extended prediction shape changed: {values.shape}"
        )
    legacy_mapping = {
        "raw_fuxi": "raw_fuxi",
        "log_bias": "log_bias",
        "legacy_anchored_adapter": "selected_adapter",
    }
    for extended_name, base_name in legacy_mapping.items():
        left = values[EXTENDED_STORE_METHODS.index(extended_name)]
        right = bundle.full_legacy_predictions[base_name]
        if not np.array_equal(left, right, equal_nan=True):
            raise StationEvaluationContractError(
                f"extended {extended_name} is not the immutable base {base_name}"
            )
    array_hashes = manifest["array_sha256"]
    for method in EXTENDED_METHODS:
        current = values[EXTENDED_STORE_METHODS.index(method)]
        if float32_array_sha256(current) != array_hashes[method]:
            raise StationEvaluationContractError(
                f"extended array hash mismatch: {method}"
            )
        if not np.isfinite(current[..., support]).all() or np.any(
            current[..., support] < 0.0
        ):
            raise StationEvaluationContractError(f"invalid extended values: {method}")
    fixed_weights = weights[support]
    denominator = fixed_weights.sum(dtype=np.float64)
    raw = values[EXTENDED_STORE_METHODS.index("raw_fuxi")]
    projected = values[EXTENDED_STORE_METHODS.index("raw_identity_raw_mean_preserved")]
    raw_means = (
        np.sum(raw[..., support] * fixed_weights, axis=-1, dtype=np.float64)
        / denominator
    )
    projected_means = (
        np.sum(projected[..., support] * fixed_weights, axis=-1, dtype=np.float64)
        / denominator
    )
    closure = np.abs(projected_means - raw_means)
    if (
        not np.isfinite(closure).all()
        or float(closure.max()) > PROJECTION_CLOSURE_TOLERANCE
    ):
        raise StationEvaluationContractError(
            "extended amount-preserving field does not close to raw FuXi under "
            "fixed forecast-time weights"
        )
    recorded = float(manifest["projection"]["maximum_float32_closure_mm_day"])
    if not np.isclose(float(closure.max()), recorded, rtol=0.0, atol=1.0e-12):
        raise StationEvaluationContractError(
            "extended recorded/recomputed closure differs"
        )
    selected_positions = np.flatnonzero(
        pd.DatetimeIndex(bundle.all_initializations).year.to_numpy(dtype=int) == 2024
    )
    predictions = dict(bundle.predictions)
    for method in EXTENDED_METHODS:
        predictions[method] = np.asarray(
            values[EXTENDED_STORE_METHODS.index(method), selected_positions],
            dtype=np.float32,
        )
    return PredictionBundle(
        all_initializations=bundle.all_initializations,
        initializations=bundle.initializations,
        lead_weeks=bundle.lead_weeks,
        latitude=bundle.latitude,
        longitude=bundle.longitude,
        adapter_support=bundle.adapter_support,
        full_legacy_predictions=bundle.full_legacy_predictions,
        predictions=predictions,
        extended_manifest=manifest,
        extended_manifest_sha256=sha256_file(manifest_path),
        extended_tree_sha256=tree_hash,
    )


def verification_dates(initialization: np.datetime64, lead_week: int) -> np.ndarray:
    if int(lead_week) not in LEAD_WEEKS:
        raise ValueError(f"invalid lead week: {lead_week}")
    start = np.datetime64(initialization, "D") + np.timedelta64(
        (int(lead_week) - 1) * 7, "D"
    )
    dates = start + np.arange(7) * np.timedelta64(1, "D")
    if not np.all(pd.DatetimeIndex(dates).year.to_numpy(dtype=int) == 2024):
        raise StationEvaluationContractError("a verification window leaves 2024")
    return dates.astype("datetime64[D]")


def required_verification_dates(initializations: np.ndarray) -> np.ndarray:
    dates = np.unique(
        np.concatenate(
            [
                verification_dates(date, lead)
                for date in initializations
                for lead in LEAD_WEEKS
            ]
        )
    )
    if dates[0] != np.datetime64("2024-06-03", "D") or dates[-1] != np.datetime64(
        "2024-11-10", "D"
    ):
        raise StationEvaluationContractError("verification date boundary changed")
    if np.any(pd.DatetimeIndex(dates).year.to_numpy(dtype=int) >= 2025):
        raise StationEvaluationContractError("2025 entered the verification dates")
    return dates


def stream_exact_2024_station_truth(
    path: Path,
    location_ids: Iterable[str],
    required_dates: np.ndarray,
) -> StationTruthSelection:
    allowed_locations = frozenset(str(value) for value in location_ids)
    allowed_dates = frozenset(
        np.datetime_as_string(value, unit="D") for value in required_dates
    )
    if not allowed_dates or any(
        not value.startswith("2024-") for value in allowed_dates
    ):
        raise StationEvaluationContractError(
            "station truth filter is not exclusively 2024"
        )
    selected: list[tuple[str, str, float]] = []
    total_rows = 0
    later_rows = 0
    date_min: str | None = None
    date_max: str | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise StationEvaluationContractError(
                "station truth container is empty"
            ) from exc
        required_columns = {"location_id", "rain_day", "rain_mm"}
        if not required_columns.issubset(header) or len(header) != len(set(header)):
            raise StationEvaluationContractError("station truth columns changed")
        location_index = header.index("location_id")
        date_index = header.index("rain_day")
        rain_index = header.index("rain_mm")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise StationEvaluationContractError(
                    f"malformed station truth row {row_number}"
                )
            total_rows += 1
            date_text = row[date_index]
            date_min = date_text if date_min is None else min(date_min, date_text)
            date_max = date_text if date_max is None else max(date_max, date_text)
            if date_text[:4].isdigit() and int(date_text[:4]) >= 2025:
                later_rows += 1
            location_id = row[location_index]
            # The rainfall token is deliberately untouched for every non-E3 row.
            if date_text not in allowed_dates or location_id not in allowed_locations:
                continue
            try:
                rain = float(row[rain_index])
            except ValueError as exc:
                raise StationEvaluationContractError(
                    f"non-numeric selected rainfall at row {row_number}"
                ) from exc
            if not np.isfinite(rain) or rain < 0.0:
                raise StationEvaluationContractError(
                    f"invalid selected rainfall at row {row_number}"
                )
            selected.append((location_id, date_text, rain))
    if not selected or date_min is None or date_max is None:
        raise StationEvaluationContractError("station truth selection is empty")
    frame = pd.DataFrame(selected, columns=["location_id", "rain_day", "rain_mm"])
    if frame.duplicated(["location_id", "rain_day"]).any():
        raise StationEvaluationContractError(
            "selected station truth has duplicate days"
        )
    frame = frame.sort_values(["location_id", "rain_day"]).reset_index(drop=True)
    if not frame.rain_day.str.startswith("2024-").all():
        raise StationEvaluationContractError(
            "selected station snapshot exposes a later row"
        )
    return StationTruthSelection(
        rows=frame,
        container_rows=total_rows,
        container_date_min=date_min,
        container_date_max=date_max,
        unselected_2025_plus_rows=later_rows,
    )


def haversine_km(
    latitude_1: np.ndarray,
    longitude_1: np.ndarray,
    latitude_2: np.ndarray,
    longitude_2: np.ndarray,
) -> np.ndarray:
    radius_km = 6_371.0
    lat_1 = np.radians(np.asarray(latitude_1, dtype=float))
    lat_2 = np.radians(np.asarray(latitude_2, dtype=float))
    delta_latitude = lat_2 - lat_1
    delta_longitude = np.radians(longitude_2) - np.radians(longitude_1)
    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def parse_grid_cell_ids(
    grid_cell_ids: Sequence[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latitude_indices: list[int] = []
    longitude_indices: list[int] = []
    for identifier in grid_cell_ids:
        match = GRID_CELL_PATTERN.fullmatch(str(identifier))
        if match is None:
            raise StationEvaluationContractError(f"invalid grid_cell_id: {identifier}")
        latitude_index, longitude_index = map(int, match.groups())
        if latitude_index >= len(latitude) or longitude_index >= len(longitude):
            raise StationEvaluationContractError(
                f"out-of-range grid_cell_id: {identifier}"
            )
        latitude_indices.append(latitude_index)
        longitude_indices.append(longitude_index)
    return np.asarray(latitude_indices, dtype=int), np.asarray(
        longitude_indices, dtype=int
    )


def load_and_validate_mapping(
    bundle: PredictionBundle, weights: np.ndarray
) -> pd.DataFrame:
    location_columns = [
        "location_id",
        "history_location_id",
        "latitude",
        "longitude",
        "eligible_for_verification",
    ]
    locations = pd.read_csv(
        STATION_DATA / "station_locations.csv", usecols=location_columns
    )
    if len(locations) != 2_053 or locations.location_id.nunique() != 2_053:
        raise StationEvaluationContractError("station location inventory changed")
    eligible_mask = (
        locations.eligible_for_verification.astype(str).str.lower().eq("true")
    )
    if int(eligible_mask.sum()) != ELIGIBLE_STATIONS:
        raise StationEvaluationContractError("eligible station count changed")
    mapping_columns = [
        "location_id",
        "history_location_id",
        "latitude",
        "longitude",
        "eligible_for_verification",
        "grid_cell_id",
        "grid_cell_id_distance_km",
    ]
    # grid_position is intentionally not read: it belongs to an old compressed mask.
    mapping = pd.read_csv(
        STATION_DATA / "station_grid_collocation.csv", usecols=mapping_columns
    )
    if (
        len(mapping) != MAPPED_STATIONS
        or mapping.location_id.nunique() != MAPPED_STATIONS
    ):
        raise StationEvaluationContractError("fixed collocation station count changed")
    if mapping.grid_cell_id.nunique() != MAPPED_CELLS:
        raise StationEvaluationContractError("fixed collocation cell count changed")
    if not mapping.eligible_for_verification.astype(str).str.lower().eq("true").all():
        raise StationEvaluationContractError(
            "collocation contains an ineligible station"
        )
    eligible = locations.loc[eligible_mask, location_columns].copy()
    joined = mapping.merge(
        eligible,
        on="location_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_inventory"),
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise StationEvaluationContractError(
            "collocation is not a subset of eligible stations"
        )
    if joined.history_location_id.isna().any() or not joined.history_location_id.astype(
        str
    ).equals(joined.history_location_id_inventory.astype(str)):
        raise StationEvaluationContractError("collocation/history linkage changed")
    for coordinate in ("latitude", "longitude"):
        if not np.array_equal(
            joined[coordinate].to_numpy(dtype=float),
            joined[f"{coordinate}_inventory"].to_numpy(dtype=float),
        ):
            raise StationEvaluationContractError(
                f"collocation/{coordinate} inventory changed"
            )
    ii, jj = parse_grid_cell_ids(
        joined.grid_cell_id.astype(str).tolist(), bundle.latitude, bundle.longitude
    )
    stored_distance = joined.grid_cell_id_distance_km.to_numpy(dtype=float)
    computed_distance = haversine_km(
        joined.latitude.to_numpy(dtype=float),
        joined.longitude.to_numpy(dtype=float),
        bundle.latitude[ii],
        bundle.longitude[jj],
    )
    if not np.allclose(computed_distance, stored_distance, rtol=0.0, atol=1.0e-9):
        raise StationEvaluationContractError(
            "grid_II_JJ distance does not match mapping"
        )
    if not np.isclose(
        stored_distance.max(),
        EXPECTED_MAXIMUM_GRID_DISTANCE_KM,
        rtol=0.0,
        atol=1.0e-9,
    ) or np.any(stored_distance > MAXIMUM_GRID_DISTANCE_KM):
        raise StationEvaluationContractError(
            "fixed collocation distance contract changed"
        )
    if not bundle.adapter_support[ii, jj].all() or not np.all(weights[ii, jj] > 0.0):
        raise StationEvaluationContractError(
            "mapped cells leave fixed adapter/India support"
        )
    joined["latitude_index"] = ii
    joined["longitude_index"] = jj
    joined["cell_flat_index"] = ii * GRID_SHAPE[1] + jj
    return joined.sort_values("location_id").reset_index(drop=True)


def _month_day_positions(values: np.ndarray) -> np.ndarray:
    output = []
    for value in np.asarray(values, dtype="datetime64[D]"):
        timestamp = pd.Timestamp(value)
        output.append(pd.Timestamp(2000, timestamp.month, timestamp.day).dayofyear - 1)
    return np.asarray(output, dtype=int)


def prepare_station_arrays(
    mapping: pd.DataFrame,
    selection: StationTruthSelection,
    required_dates: np.ndarray,
) -> StationArrays:
    location_ids = tuple(mapping.location_id.astype(str))
    date_strings = [np.datetime_as_string(value, unit="D") for value in required_dates]
    daily = selection.rows.pivot(
        index="location_id", columns="rain_day", values="rain_mm"
    ).reindex(index=location_ids, columns=date_strings)
    daily_values = daily.to_numpy(dtype=float)
    if daily_values.shape != (MAPPED_STATIONS, len(required_dates)):
        raise StationEvaluationContractError(
            "selected station daily matrix changed shape"
        )
    climate = pd.read_csv(
        STATION_DATA / "station_rainfall_climatology.csv.gz",
        usecols=["location_id", "climatology_day", "climatology_rain_mm"],
    )
    history_ids = tuple(mapping.history_location_id.astype(str))
    climate = climate.loc[climate.location_id.astype(str).isin(history_ids)].copy()
    if climate.duplicated(["location_id", "climatology_day"]).any():
        raise StationEvaluationContractError("station climatology contains duplicates")
    climate_pivot = climate.pivot(
        index="location_id", columns="climatology_day", values="climatology_rain_mm"
    ).reindex(index=history_ids, columns=range(1, 367))
    climate_values = climate_pivot.to_numpy(dtype=float)
    if climate_values.shape != (MAPPED_STATIONS, 366):
        raise StationEvaluationContractError("station climatology matrix changed shape")
    if not np.isfinite(climate_values).all() or np.any(climate_values < 0.0):
        raise StationEvaluationContractError("mapped station climatology is incomplete")
    return StationArrays(
        location_ids=location_ids,
        cell_flat_indices=mapping.cell_flat_index.to_numpy(dtype=int),
        daily_dates=np.asarray(required_dates, dtype="datetime64[D]"),
        daily_rain=daily_values,
        climatology=climate_values,
    )


def aggregate_station_week_to_cells(
    daily_rain: np.ndarray,
    climatology: np.ndarray,
    daily_positions: np.ndarray,
    climatology_positions: np.ndarray,
    cell_flat_indices: np.ndarray,
    *,
    n_cells: int = 27 * 27,
    minimum_week_days: int = MINIMUM_WEEK_DAYS,
    maximum_weekly_rain: float = MAXIMUM_WEEKLY_STATION_RAIN_MM_DAY,
) -> CellTarget:
    values = np.asarray(daily_rain, dtype=float)[:, daily_positions]
    day_counts = np.isfinite(values).sum(axis=1)
    truth = np.divide(
        np.nansum(values, axis=1),
        day_counts,
        out=np.full(values.shape[0], np.nan),
        where=day_counts >= minimum_week_days,
    )
    climate_days = np.asarray(climatology, dtype=float)[:, climatology_positions]
    climate_counts = np.isfinite(climate_days).sum(axis=1)
    weekly_climate = np.divide(
        np.nansum(climate_days, axis=1),
        climate_counts,
        out=np.full(values.shape[0], np.nan),
        where=climate_counts > 0,
    )
    location_valid = (
        np.isfinite(truth)
        & np.isfinite(weekly_climate)
        & (truth >= 0.0)
        & (truth <= maximum_weekly_rain)
    )
    cell_truth = np.full(n_cells, np.nan, dtype=float)
    cell_climate = np.full(n_cells, np.nan, dtype=float)
    station_counts = np.zeros(n_cells, dtype=int)
    cells = np.asarray(cell_flat_indices, dtype=int)
    if (
        cells.shape != (values.shape[0],)
        or np.any(cells < 0)
        or np.any(cells >= n_cells)
    ):
        raise StationEvaluationContractError("station-to-cell index array is invalid")
    for cell in np.unique(cells):
        usable = np.flatnonzero((cells == cell) & location_valid)
        if usable.size:
            cell_truth[cell] = float(np.median(truth[usable]))
            cell_climate[cell] = float(np.median(weekly_climate[usable]))
            station_counts[cell] = int(usable.size)
    return CellTarget(
        truth=cell_truth,
        climatology=cell_climate,
        station_count_by_cell=station_counts,
        station_location_count=int(location_valid.sum()),
        station_grid_cell_count=int(np.isfinite(cell_truth).sum()),
    )


def weighted_station_metrics(
    truth_absolute: np.ndarray,
    prediction_absolute: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray,
    common_valid: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth_absolute, dtype=float)
    prediction = np.asarray(prediction_absolute, dtype=float)
    climate = np.asarray(climatology, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.asarray(common_valid, dtype=bool)
    if not (
        truth.shape == prediction.shape == climate.shape == weight.shape == valid.shape
    ):
        raise ValueError("station metric arrays must have identical shapes")
    exact_valid = (
        valid
        & np.isfinite(truth)
        & np.isfinite(prediction)
        & np.isfinite(climate)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    if not np.array_equal(exact_valid, valid):
        raise StationEvaluationContractError("method-specific validity entered scoring")
    if int(valid.sum()) < MINIMUM_COMMON_CELLS:
        raise StationEvaluationContractError("fewer than 20 common station cells")
    truth = truth[valid]
    prediction = prediction[valid]
    climate = climate[valid]
    weight = weight[valid]
    weight = weight / weight.sum(dtype=np.float64)
    error = prediction - truth
    truth_anomaly = truth - climate
    prediction_anomaly = prediction - climate
    truth_centered = truth_anomaly - np.sum(weight * truth_anomaly)
    prediction_centered = prediction_anomaly - np.sum(weight * prediction_anomaly)
    denominator = np.sqrt(
        np.sum(weight * truth_centered**2) * np.sum(weight * prediction_centered**2)
    )
    acc = (
        float(np.sum(weight * truth_centered * prediction_centered) / denominator)
        if denominator > 0.0
        else np.nan
    )
    bias = float(np.sum(weight * error))
    return {
        "acc": acc,
        "rmse": float(np.sqrt(np.sum(weight * error**2))),
        "mae": float(np.sum(weight * np.abs(error))),
        "bias": bias,
        "absolute_bias": abs(bias),
    }


def load_coverage_reference() -> Mapping[tuple[str, int], tuple[int, int]]:
    frame = pd.read_csv(
        COVERAGE_REFERENCE,
        usecols=[
            "initialization",
            "lead_week",
            "station_location_count",
            "station_grid_cell_count",
        ],
    )
    frame["initialization"] = frame.initialization.astype(str)
    selected = frame.loc[frame.initialization.isin(EXACT_INITIALIZATION_STRINGS)].copy()
    if (
        len(selected) != 180
        or selected.duplicated(["initialization", "lead_week"]).any()
    ):
        raise StationEvaluationContractError(
            "coverage reference lacks the exact 180 cases"
        )
    if set(selected.lead_week.astype(int)) != set(LEAD_WEEKS):
        raise StationEvaluationContractError("coverage reference lead set changed")
    if not selected.station_grid_cell_count.between(*EXPECTED_CELL_COUNT_RANGE).all():
        raise StationEvaluationContractError("coverage reference cell range changed")
    if not selected.station_location_count.between(*EXPECTED_STATION_COUNT_RANGE).all():
        raise StationEvaluationContractError("coverage reference station range changed")
    return {
        (row.initialization, int(row.lead_week)): (
            int(row.station_location_count),
            int(row.station_grid_cell_count),
        )
        for row in selected.itertuples(index=False)
    }


def score_station_cases(
    bundle: PredictionBundle,
    weights_2d: np.ndarray,
    station: StationArrays,
    coverage_reference: Mapping[tuple[str, int], tuple[int, int]],
) -> pd.DataFrame:
    methods = tuple(bundle.predictions)
    if methods[:3] != BASE_METHODS or any(
        method not in BASE_METHODS + EXTENDED_METHODS for method in methods
    ):
        raise StationEvaluationContractError("scoring method list changed")
    date_lookup = {date: index for index, date in enumerate(station.daily_dates)}
    weights = np.asarray(weights_2d, dtype=float).reshape(-1)
    support = bundle.adapter_support.reshape(-1)
    rows: list[dict[str, Any]] = []
    for init_index, initialization in enumerate(bundle.initializations):
        init_string = np.datetime_as_string(initialization, unit="D")
        for lead_index, lead_week in enumerate(bundle.lead_weeks):
            dates = verification_dates(initialization, int(lead_week))
            positions = np.asarray([date_lookup[date] for date in dates], dtype=int)
            target = aggregate_station_week_to_cells(
                station.daily_rain,
                station.climatology,
                positions,
                _month_day_positions(dates),
                station.cell_flat_indices,
            )
            expected = coverage_reference[(init_string, int(lead_week))]
            observed = (target.station_location_count, target.station_grid_cell_count)
            if observed != expected:
                raise StationEvaluationContractError(
                    f"reconstructed station coverage differs for {init_string} W{lead_week}: "
                    f"expected {expected}, found {observed}"
                )
            prediction_vectors = {
                method: np.asarray(values[init_index, lead_index], dtype=float).reshape(
                    -1
                )
                for method, values in bundle.predictions.items()
            }
            method_finite = np.logical_and.reduce(
                [np.isfinite(values) for values in prediction_vectors.values()]
            )
            common = (
                np.isfinite(target.truth)
                & np.isfinite(target.climatology)
                & np.isfinite(weights)
                & (weights > 0.0)
                & support
                & method_finite
            )
            if int(common.sum()) != target.station_grid_cell_count:
                raise StationEvaluationContractError(
                    f"shared method cells differ from station cells for {init_string} W{lead_week}"
                )
            if (
                not EXPECTED_CELL_COUNT_RANGE[0]
                <= int(common.sum())
                <= EXPECTED_CELL_COUNT_RANGE[1]
            ):
                raise StationEvaluationContractError(
                    "common cell coverage left frozen range"
                )
            base = {
                "initialization": init_string,
                "lead_week": int(lead_week),
                "verification_start": np.datetime_as_string(dates[0], unit="D"),
                "verification_end": np.datetime_as_string(dates[-1], unit="D"),
                "station_location_count": target.station_location_count,
                "common_grid_cell_count": int(common.sum()),
            }
            for method, prediction in prediction_vectors.items():
                rows.append(
                    {
                        **base,
                        "method": method,
                        **weighted_station_metrics(
                            target.truth,
                            prediction,
                            target.climatology,
                            weights,
                            common,
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    expected_rows = 180 * len(methods)
    if (
        len(frame) != expected_rows
        or frame.duplicated(["method", "initialization", "lead_week"]).any()
    ):
        raise StationEvaluationContractError(
            "case metric table is not complete and unique"
        )
    if frame[["rmse", "mae", "bias", "absolute_bias", "acc"]].isna().any().any():
        raise StationEvaluationContractError(
            "case metric table contains a nonfinite metric"
        )
    if frame.verification_end.str.startswith("2025-").any():
        raise StationEvaluationContractError("2025 entered the metric table")
    return frame


def summarize_methods(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in case_metrics.groupby("method", sort=False):
        scopes = [("pooled", "W1-W6", group)] + [
            ("lead", f"W{lead}", group.loc[group.lead_week.eq(lead)])
            for lead in LEAD_WEEKS
        ]
        for scope_type, scope, selected in scopes:
            rows.append(
                {
                    "method": method,
                    "scope_type": scope_type,
                    "scope": scope,
                    "initializations": int(selected.initialization.nunique()),
                    "case_leads": int(len(selected)),
                    "mean_common_grid_cells": float(
                        selected.common_grid_cell_count.mean()
                    ),
                    "mean_station_locations": float(
                        selected.station_location_count.mean()
                    ),
                    **{
                        f"{metric}_mean": float(selected[metric].mean())
                        for metric in ("rmse", "mae", "bias", "absolute_bias", "acc")
                    },
                }
            )
    return pd.DataFrame(rows)


def circular_moving_block_indices(
    n_initializations: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    block_length: int = PRIMARY_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if n_initializations <= 0 or draws <= 0 or block_length <= 0:
        raise ValueError("bootstrap sizes must be positive")
    if block_length > n_initializations:
        raise ValueError("block length exceeds initialization count")
    blocks_per_draw = math.ceil(n_initializations / block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_initializations, size=(draws, blocks_per_draw))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (
        (starts[:, :, None] + offsets[None, None, :]) % n_initializations
    ).reshape(draws, -1)[:, :n_initializations]
    if indices.shape != (draws, n_initializations):
        raise StationEvaluationContractError("circular bootstrap shape changed")
    for start in range(0, n_initializations, block_length):
        block = indices[:, start : min(start + block_length, n_initializations)]
        if block.shape[1] > 1 and not np.all(
            np.diff(block, axis=1) % n_initializations == 1
        ):
            raise StationEvaluationContractError(
                "bootstrap block is not circular-consecutive"
            )
    multiplicity = mean_date_multiplicity(indices, n_initializations)
    allowed_deviation = max(0.05, 6.0 / math.sqrt(draws))
    maximum_deviation = float(np.max(np.abs(multiplicity - 1.0)))
    if maximum_deviation > allowed_deviation:
        raise StationEvaluationContractError(
            "circular bootstrap date multiplicity is inconsistent with equal "
            f"marginal inclusion: maximum deviation={maximum_deviation:.6g}, "
            f"allowed={allowed_deviation:.6g}"
        )
    return indices


def mean_date_multiplicity(indices: np.ndarray, n_initializations: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 2 or values.shape[1] != n_initializations:
        raise ValueError("bootstrap index matrix has the wrong shape")
    if np.any(values < 0) or np.any(values >= n_initializations):
        raise ValueError("bootstrap index matrix contains an out-of-range date")
    multiplicity = np.bincount(values.reshape(-1), minlength=n_initializations).astype(
        np.float64
    ) / float(values.shape[0])
    if not np.isclose(multiplicity.mean(), 1.0, rtol=0.0, atol=1.0e-12):
        raise StationEvaluationContractError(
            "bootstrap mean per-date multiplicity is not one"
        )
    return multiplicity


def bootstrap_multiplicity_table(
    plans: Mapping[int, np.ndarray], initializations: np.ndarray
) -> pd.DataFrame:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    rows: list[dict[str, Any]] = []
    for block_length, indices in plans.items():
        multiplicity = mean_date_multiplicity(indices, len(dates))
        rows.extend(
            {
                "block_length_initializations": int(block_length),
                "initialization": np.datetime_as_string(date, unit="D"),
                "mean_draw_multiplicity": float(value),
                "deviation_from_one": float(value - 1.0),
            }
            for date, value in zip(dates, multiplicity)
        )
    return pd.DataFrame(rows)


def _metric_cube(
    case_metrics: pd.DataFrame,
    method: str,
    metric: str,
    initializations: np.ndarray,
) -> np.ndarray:
    selected = case_metrics.loc[case_metrics.method.eq(method)].copy()
    selected["initialization"] = pd.to_datetime(selected.initialization)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(initializations), LEAD_WEEKS],
        names=["initialization", "lead_week"],
    )
    series = selected.set_index(["initialization", "lead_week"])[metric].reindex(index)
    values = series.to_numpy(dtype=float).reshape(len(initializations), len(LEAD_WEEKS))
    if not np.isfinite(values).all():
        raise StationEvaluationContractError(
            f"incomplete {method}/{metric} metric cube"
        )
    return values


def comparison_specs(methods: Sequence[str]) -> tuple[ComparisonSpec, ...]:
    specs = [
        ComparisonSpec(
            "selected_adapter_vs_raw_fuxi",
            "selected_adapter",
            "raw_fuxi",
            primary=True,
        ),
        ComparisonSpec("selected_adapter_vs_log_bias", "selected_adapter", "log_bias"),
        ComparisonSpec("log_bias_vs_raw_fuxi", "log_bias", "raw_fuxi"),
    ]
    if "raw_identity" in methods:
        specs.append(
            ComparisonSpec("raw_identity_vs_raw_fuxi", "raw_identity", "raw_fuxi")
        )
    if "raw_identity_raw_mean_preserved" in methods:
        specs.extend(
            (
                ComparisonSpec(
                    "raw_identity_raw_mean_preserved_vs_raw_fuxi",
                    "raw_identity_raw_mean_preserved",
                    "raw_fuxi",
                ),
                ComparisonSpec(
                    "raw_identity_raw_mean_preserved_vs_raw_identity",
                    "raw_identity_raw_mean_preserved",
                    "raw_identity",
                ),
            )
        )
    return tuple(specs)


def paired_bootstrap_effects(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    plans: Mapping[int, np.ndarray],
    comparisons: Sequence[ComparisonSpec],
) -> pd.DataFrame:
    metric_contract = {
        "rmse": ("reference_minus_candidate", 1.0),
        "mae": ("reference_minus_candidate", 1.0),
        "acc": ("candidate_minus_reference", -1.0),
        "bias": ("candidate_minus_reference", -1.0),
        "absolute_bias": ("reference_minus_candidate", 1.0),
    }
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric, (definition, multiplier) in metric_contract.items():
            candidate = _metric_cube(
                case_metrics, comparison.candidate, metric, initializations
            )
            reference = _metric_cube(
                case_metrics, comparison.reference, metric, initializations
            )
            difference = multiplier * (reference - candidate)
            point = float(difference.mean())
            for block_length, indices in plans.items():
                draws = difference[indices, :].mean(axis=(1, 2))
                if not np.isfinite(draws).all():
                    raise StationEvaluationContractError(
                        "bootstrap produced nonfinite effects"
                    )
                lower, upper = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "comparison": comparison.name,
                        "candidate": comparison.candidate,
                        "reference": comparison.reference,
                        "metric": metric,
                        "effect_definition": definition,
                        "positive_favors_candidate": (
                            None if metric == "bias" else True
                        ),
                        "block_length_initializations": int(block_length),
                        "analysis_role": (
                            "primary_uncertainty"
                            if block_length == PRIMARY_BLOCK_LENGTH
                            else "predeclared_sensitivity"
                        ),
                        "bootstrap_draws": int(len(draws)),
                        "initializations": int(len(initializations)),
                        "case_leads": int(len(initializations) * len(LEAD_WEEKS)),
                        "point_effect": point,
                        "bootstrap_mean": float(draws.mean()),
                        "ci_lower_2p5": float(lower),
                        "ci_upper_97p5": float(upper),
                        "probability_effect_gt_zero": float(np.mean(draws > 0.0)),
                        "primary_estimand": bool(
                            comparison.primary
                            and metric == "rmse"
                            and block_length == PRIMARY_BLOCK_LENGTH
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    primary = output.loc[output.primary_estimand]
    if len(primary) != 1 or int(primary.iloc[0].case_leads) != 180:
        raise StationEvaluationContractError("primary 180-case estimand is not unique")
    return output


def _copy_source_snapshots(
    staging: Path,
    extended: Path | None,
) -> None:
    code = staging / "code"
    source = staging / "source_snapshot"
    code.mkdir()
    source.mkdir()
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    launcher = (
        PROJECT_ROOT / "slurm" / "evaluate_adapter_station_external_target.sbatch"
    )
    if not launcher.is_file():
        raise StationEvaluationContractError("E3 Slurm launcher is missing")
    shutil.copy2(launcher, code / launcher.name)
    shutil.copy2(BASE_MANIFEST, source / "base_prediction_manifest.json")
    shutil.copy2(
        BASE_PREDICTION_STORE / ".zmetadata",
        source / "base_predictions_zmetadata.json",
    )
    shutil.copy2(
        STATION_DATA / "cleaning_manifest.json",
        source / "station_cleaning_manifest.json",
    )
    shutil.copy2(SPATIAL_STORE / ".zmetadata", source / "spatial_zmetadata.json")
    if extended is not None:
        _, extended_manifest = _resolve_extended_store(extended)
        shutil.copy2(extended_manifest, source / "extended_prediction_manifest.json")


def _artifact_hashes(staging: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(
        candidate for candidate in staging.rglob("*") if candidate.is_file()
    ):
        if path.name == "manifest.json" or path.name.startswith("."):
            continue
        artifacts[path.relative_to(staging).as_posix()] = sha256_file(path)
    return artifacts


def run_evaluation(
    output: Path,
    *,
    extended_predictions: Path | None = None,
    require_extended: bool = False,
) -> Path:
    if require_extended and extended_predictions is None:
        raise StationEvaluationContractError(
            "--require-extended needs the completed E2 prediction artifact"
        )
    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh output required: {output}")
    forbidden = (
        BASE_RESULT_ROOT.resolve(),
        STATION_ROOT.resolve(),
        SPATIAL_STORE.resolve(),
    )
    if any(output == root or root in output.parents for root in forbidden):
        raise StationEvaluationContractError("output may not be inside a frozen source")
    launcher = (
        PROJECT_ROOT / "slurm" / "evaluate_adapter_station_external_target.sbatch"
    )
    execution_source_hashes = {
        Path(__file__).name: sha256_file(Path(__file__)),
        launcher.name: sha256_file(launcher),
    }
    input_hashes = verify_frozen_sources()
    bundle = load_base_predictions()
    weights = load_spatial_weights(bundle)
    if extended_predictions is not None:
        bundle = attach_extended_predictions(bundle, weights, extended_predictions)
        extended_store, extended_manifest = _resolve_extended_store(
            extended_predictions
        )
        assert bundle.extended_manifest_sha256 is not None
        assert bundle.extended_tree_sha256 is not None
        input_hashes[str(extended_manifest)] = bundle.extended_manifest_sha256
        input_hashes[str(extended_store)] = bundle.extended_tree_sha256
    mapping = load_and_validate_mapping(bundle, weights)
    required_dates = required_verification_dates(bundle.initializations)
    selection = stream_exact_2024_station_truth(
        STATION_DATA / "recent_daily_rainfall_clean.csv.gz",
        mapping.location_id.astype(str),
        required_dates,
    )
    observed_container_contract = (
        selection.container_rows,
        selection.container_date_min,
        selection.container_date_max,
        selection.unselected_2025_plus_rows,
    )
    expected_container_contract = (
        RECENT_CONTAINER_ROWS,
        RECENT_CONTAINER_DATE_MIN,
        RECENT_CONTAINER_DATE_MAX,
        RECENT_CONTAINER_2025_PLUS_ROWS,
    )
    if observed_container_contract != expected_container_contract:
        raise StationEvaluationContractError(
            "mixed station container date/count boundary changed"
        )
    station = prepare_station_arrays(mapping, selection, required_dates)
    coverage = load_coverage_reference()

    with fresh_atomic_output(output) as staging:
        _copy_source_snapshots(staging, extended_predictions)
        for name, expected_hash in execution_source_hashes.items():
            observed_hash = sha256_file(staging / "code" / name)
            if observed_hash != expected_hash:
                raise StationEvaluationContractError(
                    f"executed/source-snapshot hash mismatch: {name}"
                )
        (staging / "inputs").mkdir()
        atomic_write_deterministic_gzip_csv(
            staging / "inputs" / "station_truth_selected_2024.csv.gz",
            selection.rows,
        )
        protocol = {
            "schema_version": 1,
            "experiment": "E3_adapter_station_external_target",
            "scientific_status": (
                "frozen independent-observational-target sensitivity; "
                "not untouched temporal final"
            ),
            "methods": list(bundle.predictions),
            "initializations": list(EXACT_INITIALIZATION_STRINGS),
            "lead_weeks": list(LEAD_WEEKS),
            "primary_estimand": (
                "equal-case mean RMSE(raw_fuxi) - RMSE(selected_adapter) "
                "over 30 initializations x 6 leads"
            ),
            "station_aggregation": {
                "minimum_daily_values_per_week": MINIMUM_WEEK_DAYS,
                "within_cell_aggregation": "median",
                "acc_reference": "common cleaned 2018-2023 station climatology",
                "minimum_common_cells": MINIMUM_COMMON_CELLS,
                "weights": "India land-area weight",
                "identical_cells_for_every_method": True,
            },
            "uncertainty": {
                "design": "paired circular moving blocks of ordered initializations; all six leads attached",
                "draws": BOOTSTRAP_DRAWS,
                "primary_block_length": PRIMARY_BLOCK_LENGTH,
                "sensitivity_block_lengths": list(SENSITIVITY_BLOCK_LENGTHS),
                "seed": BOOTSTRAP_SEED,
                "equal_marginal_date_inclusion": True,
            },
            "boundaries": {
                "training_or_selection_on_stations": False,
                "2025_target_values_selected": False,
                "station_rain_day": "03:00-03:00 UTC",
                "forecast_day": "00:00-00:00 UTC",
                "three_hour_convention_mismatch_disclosed": True,
                "station_network_selection_uses_full_2024_QC": True,
                "point_gauge_vs_grid_cell_representativeness_error": True,
            },
        }
        atomic_write_json(staging / "protocol.json", protocol)

        # Scientific scoring starts only after all frozen hashes, dates, support,
        # mapping, and selected 2024 station rows have passed their contracts.
        case_metrics = score_station_cases(bundle, weights, station, coverage)
        summary = summarize_methods(case_metrics)
        block_lengths = (PRIMARY_BLOCK_LENGTH, *SENSITIVITY_BLOCK_LENGTHS)
        plans = {
            length: circular_moving_block_indices(
                len(bundle.initializations),
                draws=BOOTSTRAP_DRAWS,
                block_length=length,
                seed=BOOTSTRAP_SEED,
            )
            for length in block_lengths
        }
        multiplicity = bootstrap_multiplicity_table(plans, bundle.initializations)
        effects = paired_bootstrap_effects(
            case_metrics,
            bundle.initializations,
            plans,
            comparison_specs(tuple(bundle.predictions)),
        )
        atomic_write_csv(staging / "case_metrics.csv", case_metrics)
        atomic_write_csv(staging / "method_summary.csv", summary)
        atomic_write_csv(staging / "paired_bootstrap_effects.csv", effects)
        atomic_write_csv(staging / "bootstrap_date_multiplicity.csv", multiplicity)
        atomic_write_npz(
            staging / "bootstrap_indices.npz",
            **{
                f"circular_block_{length}": indices for length, indices in plans.items()
            },
        )
        artifacts = _artifact_hashes(staging)
        primary = effects.loc[effects.primary_estimand].iloc[0]
        canonical_artifact = bundle.extended_manifest is not None
        manifest = {
            "schema_version": 1,
            "status": (
                "complete_frozen_external_target_sensitivity"
                if canonical_artifact
                else "complete_frozen_external_target_sensitivity_base_only_noncanonical"
            ),
            "canonical_artifact": canonical_artifact,
            "execution_tier": (
                "canonical_five_method_e3"
                if canonical_artifact
                else "noncanonical_base_three_method_only"
            ),
            "created_utc": utc_now(),
            "experiment": "E3_adapter_station_external_target",
            "scientific_status": protocol["scientific_status"],
            "training_performed": False,
            "selection_calibration_or_blending_performed": False,
            "2025_prediction_opened": False,
            "2025_station_value_selected": False,
            "2025_metric_computed": False,
            "methods": list(bundle.predictions),
            "dates": {
                "initializations": list(EXACT_INITIALIZATION_STRINGS),
                "initialization_years": [2024],
                "verification_date_min": "2024-06-03",
                "verification_date_max": "2024-11-10",
                "initialization_count": 30,
                "lead_weeks": list(LEAD_WEEKS),
                "case_leads": 180,
            },
            "coverage": {
                "mapped_station_locations": MAPPED_STATIONS,
                "mapped_grid_cells": MAPPED_CELLS,
                "adapter_support_cells": SUPPORT_CELLS,
                "case_station_location_min": int(
                    case_metrics.station_location_count.min()
                ),
                "case_station_location_max": int(
                    case_metrics.station_location_count.max()
                ),
                "case_common_grid_cell_min": int(
                    case_metrics.common_grid_cell_count.min()
                ),
                "case_common_grid_cell_max": int(
                    case_metrics.common_grid_cell_count.max()
                ),
                "reference_counts_reconstructed_exactly": True,
                "grid_index_contract": "parsed grid_II_JJ; old grid_position never loaded",
            },
            "station_truth_boundary": {
                "mixed_clean_container": str(
                    STATION_DATA / "recent_daily_rainfall_clean.csv.gz"
                ),
                "container_rows_scanned": selection.container_rows,
                "container_date_min": selection.container_date_min,
                "container_date_max": selection.container_date_max,
                "unselected_2025_plus_rows": selection.unselected_2025_plus_rows,
                "rainfall_converted_only_after_exact_2024_date_and_station_filter": True,
                "selected_snapshot": "inputs/station_truth_selected_2024.csv.gz",
            },
            "primary_estimand": {
                "comparison": primary.comparison,
                "metric": primary.metric,
                "definition": primary.effect_definition,
                "case_weighting": "equal over 180 initialization-by-lead cases",
                "point_effect": float(primary.point_effect),
                "ci_lower_2p5": float(primary.ci_lower_2p5),
                "ci_upper_97p5": float(primary.ci_upper_97p5),
                "probability_effect_gt_zero": float(primary.probability_effect_gt_zero),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "circular_block_length_initializations": PRIMARY_BLOCK_LENGTH,
                "all_six_leads_attached": True,
            },
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "primary_block_length": PRIMARY_BLOCK_LENGTH,
                "sensitivity_block_lengths": list(SENSITIVITY_BLOCK_LENGTHS),
                "seed": BOOTSTRAP_SEED,
                "shared_within_each_block_length_across_all_methods_metrics": True,
                "equal_marginal_date_inclusion": True,
                "date_multiplicity": {
                    str(length): {
                        "mean": float(
                            multiplicity.loc[
                                multiplicity.block_length_initializations.eq(length),
                                "mean_draw_multiplicity",
                            ].mean()
                        ),
                        "minimum": float(
                            multiplicity.loc[
                                multiplicity.block_length_initializations.eq(length),
                                "mean_draw_multiplicity",
                            ].min()
                        ),
                        "maximum": float(
                            multiplicity.loc[
                                multiplicity.block_length_initializations.eq(length),
                                "mean_draw_multiplicity",
                            ].max()
                        ),
                        "maximum_absolute_deviation": float(
                            multiplicity.loc[
                                multiplicity.block_length_initializations.eq(length),
                                "deviation_from_one",
                            ]
                            .abs()
                            .max()
                        ),
                    }
                    for length in block_lengths
                },
            },
            "input_hashes": input_hashes,
            "base_prediction_manifest_sha256": BASE_MANIFEST_SHA256,
            "base_prediction_tree_sha256": BASE_TREE_SHA256,
            "extended_prediction_manifest_sha256": bundle.extended_manifest_sha256,
            "extended_prediction_tree_sha256": bundle.extended_tree_sha256,
            "extended_projection_verified_without_target_coverage": (
                bundle.extended_manifest is not None
            ),
            "extended_prediction_canonical_contract_verified": canonical_artifact,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xarray": xr.__version__,
            },
            "command": [str(value) for value in sys.argv],
            "executed_source_sha256": execution_source_hashes,
            "artifacts": artifacts,
        }
        atomic_write_json(staging / "manifest.json", manifest)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="fresh output directory"
    )
    parser.add_argument(
        "--extended-predictions",
        type=Path,
        help="optional completed E2 root (or predictions.zarr) with identity methods",
    )
    parser.add_argument(
        "--require-extended",
        action="store_true",
        help="fail unless the completed E2 identity-method artifact is supplied",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_evaluation(
        args.output,
        extended_predictions=args.extended_predictions,
        require_extended=args.require_extended,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
