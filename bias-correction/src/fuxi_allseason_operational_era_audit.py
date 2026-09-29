#!/usr/bin/env python3
"""Locked all-season 2022--2024 audit of the selected FuXi adapter.

This evaluator cannot train or select a model.  It restores only the
validation-selected ``base_42k`` checkpoints (seeds 42/43/44), applies them
unchanged to the individual 50-member operational-era FuXi ensemble, and
compares their scores with the same raw members.  Forecast and verification
access is an exact 2022/2023/2024 whitelist.  Late-2024 starts whose 42-day
verification would enter 2025 are rejected before any forecast values load.

The evidence label is deliberately conservative: post-hoc operational-era
retrospective audit, with no retraining or selection.  It is not an untouched
final test and it never opens a 2025 forecast or observation store.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

import fuxi_allseason_capacity_ablation as capacity
import fuxi_allseason_capacity_development_evaluation as capacity_gate
import fuxi_allseason_ensemble_calibration as base
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_operational_era_audit_v1"
AUDIT_CONTRACT_VERSION = "operational_era_audit_contract_v3"
POSTRUN_AUDIT_VERSION = "operational_era_postrun_v3"
EVIDENCE_LABEL = (
    "post-hoc 2022-2024 operational-era retrospective audit; "
    "no retraining or selection"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_operational_era_audit"
)
DEFAULT_CAPACITY_MANIFEST = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z/manifest.json"
)
PROTOCOL_PATH = PROJECT_ROOT / "plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md"
SLURM_PATH = PROJECT_ROOT / "slurm/evaluate_allseason_operational_era.sbatch"

# The parent model-run label includes ``2020_2025``.  That label is metadata,
# not a year-store wildcard.  Code below constructs only the three literal
# whitelisted child paths and never enumerates this directory.
OPERATIONAL_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/forecasts/fuxi_s2s/"
    "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/"
    "tp/common_1p5"
)
IMD_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/observations/ground_truth_v1/daily/imd/tp/"
    "india_1p5_27x27_v1"
)

AUDIT_YEARS = (2022, 2023, 2024)
TRAIN_YEARS = tuple(range(2002, 2018))
SEEDS = (42, 43, 44)
EXPECTED_STORED_COUNTS = {2022: 104, 2023: 104, 2024: 100}
EXPECTED_ELIGIBLE_COUNTS = {2022: 104, 2023: 104, 2024: 88}
EXPECTED_FULL_CASES = 296
FORECAST_MEMBER_COUNT = 50
HINDCAST_MEMBER_COUNT = 51
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_822
SMOKE_CASES_PER_YEAR = 4
DAILY_VALIDATION_CHUNK_CASES = 8
METHODS = ("raw_fuxi", capacity.BASE_CANDIDATE)
CORE_METRICS = ("crps", "rmse", "mae", "acc", "bias")

EXPECTED_NORMALIZATION_MEAN = np.asarray(
    [
        0.9928932189941406,
        0.9928548336029053,
        0.9927545189857483,
        0.992607057094574,
        0.9922419190406799,
        0.9919546246528625,
    ],
    dtype=np.float32,
)
EXPECTED_NORMALIZATION_STD = np.asarray(
    [
        0.8253104090690613,
        0.8253679871559143,
        0.8254558444023132,
        0.8255149126052856,
        0.8256960511207581,
        0.825825035572052,
    ],
    dtype=np.float32,
)
EXPECTED_NORMALIZATION_MEAN_SHA256 = (
    "0fdceee6daa9bf0469967f9ac3e7df5a9ffb0bedfc53930a4dcf8184b89fc911"
)
EXPECTED_NORMALIZATION_STD_SHA256 = (
    "051a34ffe3842bdd6089db1d270b4a8dc1d403afa7b43ea917e4a4c5f9a19926"
)
NORMALIZATION_REFERENCE = (
    "resultsv2/fuxi_allseason_ensemble_calibration/"
    "full_publication_20260822T115253Z/manifest.json:"
    "normalization.climatology_log1p_mean_by_lead/std_by_lead"
)

# This verification-data exception was frozen during the read-only prelaunch
# inspection, before forecast scores existed.  The loader derives the actual
# deviations from the arrays and compares the complete derived identity set
# with this contract; these constants are not merely copied into the manifest.
EXPECTED_COVERAGE_DEVIATION_DATE = np.datetime64("2023-10-12", "D")
EXPECTED_COVERAGE_DEVIATIONS = (
    # (latitude index, longitude index, latitude, longitude,
    #  frozen fraction, observed daily fraction)
    (10, 6, 24.0, 69.0, 1.0, 0.0),
    (10, 7, 24.0, 70.5, 1.0, 0.7498814463615417),
    (11, 6, 22.5, 69.0, 0.7812473773956299, 0.0),
    (11, 7, 22.5, 70.5, 1.0, 0.6565456390380859),
    (12, 7, 21.0, 70.5, 0.9373154640197754, 0.8279865980148315),
)
EXPECTED_ZERO_DAILY_COVERAGE_CELLS = 2
EXPECTED_AFFECTED_CASE_LEADS = (
    # (initialization, lead week, one-based day within the seven-day block)
    ("2023-09-04", 6, 4),
    ("2023-09-07", 6, 1),
    ("2023-09-11", 5, 4),
    ("2023-09-14", 5, 1),
    ("2023-09-18", 4, 4),
    ("2023-09-21", 4, 1),
    ("2023-09-25", 3, 4),
    ("2023-09-28", 3, 1),
    ("2023-10-02", 2, 4),
    ("2023-10-05", 2, 1),
    ("2023-10-09", 1, 4),
    ("2023-10-12", 1, 1),
)
EXPECTED_SMOKE_AFFECTED_CASE_LEADS = (("2023-10-12", 1, 1),)
EXPECTED_FULL_MINIMUM_WEIGHT_RATIO = 6.0 / 7.0

EXPECTED_CAPACITY_MANIFEST_SHA256 = (
    "2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c"
)
EXPECTED_SELECTION_SHA256 = (
    "69057167ac0a784ea5209375114d32077f38273547dfc8412e82f182532cb747"
)
EXPECTED_SCORING_SUPPORT_SHA256 = (
    "492d7c54167163f8a77cb7ea3c3f16a142be5427e88b477f367bc6278f2659c4"
)
EXPECTED_CHECKPOINT_SHA256 = {
    42: "5b44762ff3d0e2a93834dc2dfd514f8ba7a40d3ce0f4ed74e29c71b2c49e7dba",
    43: "76b4b3d88018d8c26bc82f84e1409770e3472e706e0736d079b46cbe15013e3a",
    44: "afbf758500a5c510c742fdf0d1a3d310255f2b435a777844dbf4c907059dc646",
}


class OperationalAuditError(RuntimeError):
    """Raised when a locked model, data, or evaluation invariant changes."""


@dataclass(frozen=True)
class OperationalMembers:
    members: np.ndarray
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    member_labels: np.ndarray
    member_available: np.ndarray
    store_paths: tuple[str, ...]
    content_inventory: Mapping[str, Any]
    stored_counts: Mapping[int, int]
    eligible_counts: Mapping[int, int]
    evaluated_counts: Mapping[int, int]
    excluded_2024_initializations: tuple[str, ...]
    lead_week_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class DailyObservations:
    training_dates: np.ndarray
    training_values: np.ndarray
    verification_dates: np.ndarray
    verification_values: np.ndarray
    verification_fraction: np.ndarray
    observation_fraction: np.ndarray
    training_stores: tuple[str, ...]
    verification_stores: tuple[str, ...]
    content_inventory: Mapping[str, Any]
    coverage_deviation_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class AuditTargets:
    truth: np.ndarray
    climatology: np.ndarray
    weights: np.ndarray
    frozen_spatial_weights: np.ndarray
    context: base.ContextBundle
    training_climatology_daily: np.ndarray
    normalization_inventory: Mapping[str, Any]
    content_inventory: Mapping[str, Any]
    coverage_contract: Mapping[str, Any]


@dataclass(frozen=True)
class TwoStageBootstrap:
    draws: tuple[np.ndarray, ...]
    sampled_years: np.ndarray
    source_years: tuple[int, ...]
    year_sizes: Mapping[int, int]
    block_length: int
    seed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Content hash that binds array dtype, shape, and C-order bytes."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_receipt(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "sha256": array_sha256(array),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _reject_forbidden_year_path(path: Path) -> None:
    """Reject an exact year-store component without substring false positives."""

    if any(part == "2025.zarr" for part in Path(path).parts):
        raise OperationalAuditError(f"forbidden 2025 year store: {path}")


def exact_year_store_paths(root: Path, years: Sequence[int]) -> tuple[Path, ...]:
    """Build direct paths only; callers must never replace this with discovery."""

    normalized = tuple(int(year) for year in years)
    allowed = set(TRAIN_YEARS) | set(AUDIT_YEARS)
    if len(set(normalized)) != len(normalized) or not set(normalized).issubset(allowed):
        raise OperationalAuditError(f"year-store whitelist violation: {normalized}")
    paths = tuple(Path(root) / f"{year}.zarr" for year in normalized)
    for year, path in zip(normalized, paths, strict=True):
        _reject_forbidden_year_path(path)
        if path.name != f"{year}.zarr" or not (path / ".zmetadata").is_file():
            raise FileNotFoundError(path)
    return paths


def operational_store_paths() -> tuple[Path, ...]:
    return exact_year_store_paths(OPERATIONAL_ROOT, AUDIT_YEARS)


def observation_store_paths(years: Sequence[int]) -> tuple[Path, ...]:
    return exact_year_store_paths(IMD_ROOT, years)


def validate_daily_coverage_deviations(
    dates: np.ndarray,
    daily_fraction: np.ndarray,
    frozen_fraction: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> dict[str, Any]:
    """Derive and exact-gate the frozen prelaunch IMD coverage exception."""

    days = np.asarray(dates, dtype="datetime64[D]")
    fractions = np.asarray(daily_fraction, dtype=np.float32)
    frozen = np.asarray(frozen_fraction, dtype=np.float32)
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    _require(
        days.ndim == 1
        and fractions.shape == (days.size, 27, 27)
        and frozen.shape == (27, 27)
        and lat.shape == (27,)
        and lon.shape == (27,),
        "coverage-deviation arrays are not on the exact daily 27x27 contract",
    )
    changed = ~np.isclose(
        fractions,
        frozen[None],
        rtol=0.0,
        atol=1.0e-7,
        equal_nan=False,
    )
    locations = np.argwhere(changed)
    actual_identity = tuple(
        (
            np.datetime_as_string(days[time_index], unit="D"),
            int(latitude_index),
            int(longitude_index),
        )
        for time_index, latitude_index, longitude_index in locations
    )
    expected_date = np.datetime_as_string(EXPECTED_COVERAGE_DEVIATION_DATE, unit="D")
    expected_identity = tuple(
        (expected_date, int(latitude_index), int(longitude_index))
        for latitude_index, longitude_index, *_ in EXPECTED_COVERAGE_DEVIATIONS
    )
    _require(
        actual_identity == expected_identity,
        "derived IMD coverage-deviation date/cell identities differ from the frozen amendment: "
        f"{actual_identity}",
    )

    records: list[dict[str, Any]] = []
    zero_count = 0
    for location, expected in zip(locations, EXPECTED_COVERAGE_DEVIATIONS, strict=True):
        time_index, latitude_index, longitude_index = (int(value) for value in location)
        (
            expected_latitude_index,
            expected_longitude_index,
            expected_latitude,
            expected_longitude,
            expected_frozen,
            expected_daily,
        ) = expected
        actual_frozen = float(frozen[latitude_index, longitude_index])
        actual_daily = float(fractions[time_index, latitude_index, longitude_index])
        _require(
            latitude_index == expected_latitude_index
            and longitude_index == expected_longitude_index
            and float(lat[latitude_index]) == expected_latitude
            and float(lon[longitude_index]) == expected_longitude
            and math.isclose(actual_frozen, expected_frozen, rel_tol=0.0, abs_tol=1.0e-7)
            and math.isclose(actual_daily, expected_daily, rel_tol=0.0, abs_tol=1.0e-7),
            "derived IMD coverage-deviation coordinate/fraction differs from the frozen amendment",
        )
        is_zero = actual_daily == 0.0
        zero_count += int(is_zero)
        records.append(
            {
                "date": expected_date,
                "latitude_index": latitude_index,
                "longitude_index": longitude_index,
                "latitude": float(lat[latitude_index]),
                "longitude": float(lon[longitude_index]),
                "frozen_observation_fraction": actual_frozen,
                "daily_observation_fraction": actual_daily,
                "zero_daily_coverage": is_zero,
            }
        )
    _require(
        zero_count == EXPECTED_ZERO_DAILY_COVERAGE_CELLS,
        f"derived zero-daily-coverage count {zero_count} differs from the frozen amendment",
    )
    return {
        "contract_version": AUDIT_CONTRACT_VERSION,
        "derivation": "all verification day/cell fractions compared with frozen fraction at atol=1e-7",
        "affected_dates": [expected_date],
        "daily_deviation_count": len(records),
        "reduced_coverage_cell_count": len(records),
        "zero_daily_coverage_cell_count": zero_count,
        "deviation_records": records,
        "derived_deviation_mask": array_receipt(changed.astype(np.uint8)),
        "exact_identity_gate_passed": True,
    }


def validate_evaluated_coverage_contract(
    initializations: np.ndarray,
    valid_dates: np.ndarray,
    daily_fraction: np.ndarray,
    frozen_fraction: np.ndarray,
    weight_ratio: np.ndarray,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Exact-gate which evaluated case/lead blocks contain the known exception."""

    inits = np.asarray(initializations, dtype="datetime64[D]")
    dates = np.asarray(valid_dates, dtype="datetime64[D]")
    fractions = np.asarray(daily_fraction, dtype=np.float32)
    frozen = np.asarray(frozen_fraction, dtype=np.float32)
    ratios = np.asarray(weight_ratio, dtype=np.float64)
    expected_cases = 3 * SMOKE_CASES_PER_YEAR if smoke else EXPECTED_FULL_CASES
    _require(len(inits) == expected_cases, f"coverage contract received {len(inits)} cases")
    _require(
        dates.shape == (len(inits), 6, 7)
        and fractions.shape == (len(inits), 6, 7, 27, 27),
        "evaluated daily coverage arrays are not aligned",
    )
    changed = ~np.isclose(
        fractions,
        frozen[None, None, None],
        rtol=0.0,
        atol=1.0e-7,
        equal_nan=False,
    )
    affected_days = np.argwhere(np.any(changed, axis=(-2, -1)))
    actual = tuple(
        (
            np.datetime_as_string(inits[case_index], unit="D"),
            int(lead_index) + 1,
            int(day_index) + 1,
        )
        for case_index, lead_index, day_index in affected_days
    )
    expected = (
        EXPECTED_SMOKE_AFFECTED_CASE_LEADS
        if smoke
        else EXPECTED_AFFECTED_CASE_LEADS
    )
    _require(
        actual == expected,
        "evaluated case/lead/day coverage identities differ from the frozen amendment: "
        f"{actual}",
    )
    if affected_days.size:
        affected_valid_dates = tuple(
            np.datetime_as_string(dates[tuple(location)], unit="D")
            for location in affected_days
        )
        _require(
            set(affected_valid_dates)
            == {np.datetime_as_string(EXPECTED_COVERAGE_DEVIATION_DATE, unit="D")},
            "evaluated coverage deviation appears on an unexpected verification date",
        )
    observed_minimum = float(np.min(ratios))
    expected_minimum = EXPECTED_FULL_MINIMUM_WEIGHT_RATIO
    _require(
        math.isclose(observed_minimum, expected_minimum, rel_tol=0.0, abs_tol=1.0e-12),
        f"minimum dynamic/frozen weight ratio {observed_minimum} != {expected_minimum}",
    )
    return {
        "mode_contract": (
            "fixed_12_case_smoke_with_real_2023_coverage_path"
            if smoke
            else "all_296_eligible_cases"
        ),
        "affected_case_lead_count": len(actual),
        "expected_affected_case_lead_count": len(expected),
        "affected_case_leads": [
            {
                "initialization": initialization,
                "lead_week": lead_week,
                "day_within_week": day_within_week,
                "affected_verification_date": np.datetime_as_string(
                    EXPECTED_COVERAGE_DEVIATION_DATE, unit="D"
                ),
            }
            for initialization, lead_week, day_within_week in actual
        ],
        "minimum_dynamic_to_frozen_weight_ratio": observed_minimum,
        "expected_minimum_dynamic_to_frozen_weight_ratio": expected_minimum,
        "exact_case_lead_gate_passed": True,
    }


def validate_locked_model_receipt(
    manifest_path: Path,
) -> capacity_gate.CapacityReceipt:
    """Pin the selected architecture/configuration and all three checkpoints."""

    manifest_path = Path(manifest_path).resolve()
    actual_manifest_hash = sha256_file(manifest_path)
    _require(
        actual_manifest_hash == EXPECTED_CAPACITY_MANIFEST_SHA256,
        "capacity manifest is not the frozen pre-audit receipt",
    )
    receipt = capacity_gate.validate_capacity_manifest(manifest_path)
    _require(
        receipt.selected_candidate == capacity.BASE_CANDIDATE,
        "validation selection no longer retains base_42k",
    )
    _require(receipt.selected_distinct_from_base is False, "selected/base identity changed")
    selection_path = receipt.root / "selection.json"
    _require(
        sha256_file(selection_path) == EXPECTED_SELECTION_SHA256,
        "capacity selection receipt changed",
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(
        selection.get("status") == "validation_selection_locked"
        and selection.get("scientific_selection") is True
        and selection.get("test_metrics_consulted") is False,
        "capacity selection is not validation-only and locked",
    )
    _require(
        int(selection.get("selected_parameter_count", -1)) == 42_434,
        "selected model parameter count changed",
    )
    scoring_support = receipt.root / "evaluation/scoring_support.npz"
    _require(
        sha256_file(scoring_support) == EXPECTED_SCORING_SUPPORT_SHA256,
        "capacity scoring-support artifact changed",
    )
    for seed in SEEDS:
        record = receipt.checkpoint_records[(capacity.BASE_CANDIDATE, seed)]
        relative = Path(str(record["checkpoint"]))
        checkpoint = receipt.root / relative
        expected = EXPECTED_CHECKPOINT_SHA256[seed]
        _require(record.get("checkpoint_sha256") == expected, f"seed {seed} record changed")
        _require(sha256_file(checkpoint) == expected, f"seed {seed} checkpoint changed")
        _require(int(record.get("parameter_count", -1)) == 42_434, "architecture changed")
        _require(int(record.get("member_hidden_channels", -1)) == 8, "member width changed")
        _require(int(record.get("backbone_channels", -1)) == 24, "backbone width changed")
    training = receipt.manifest.get("training", {})
    _require(training.get("objective") == "area-weighted empirical finite-ensemble CRPS", "loss changed")
    _require(training.get("member_subsample") == 16, "training member subsample changed")
    _require(training.get("full_members_for_validation") == 51, "hindcast validation member count changed")
    return receipt


def eligible_initialization_mask(
    initializations: np.ndarray, year: int
) -> np.ndarray:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise OperationalAuditError("initializations must be a non-empty 1-D array")
    if int(year) not in AUDIT_YEARS:
        raise OperationalAuditError(f"non-audit initialization year: {year}")
    observed_years = pd.DatetimeIndex(dates).year.to_numpy()
    _require(np.all(observed_years == year), f"{year} store contains another init year")
    if year < 2024:
        return np.ones(dates.size, dtype=bool)
    verification_end = dates + np.timedelta64(41, "D")
    return verification_end <= np.datetime64("2024-12-31", "D")


def _select_smoke_indices(
    initializations: np.ndarray,
    indices: np.ndarray,
    year: int,
    count: int,
) -> np.ndarray:
    """Fixed year-balanced smoke with the real 2023 coverage path exercised."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    values = np.asarray(indices, dtype=np.int64)
    if values.size <= count:
        return values
    selected = values[np.linspace(0, values.size - 1, count, dtype=np.int64)]
    if int(year) == 2023:
        required = np.flatnonzero(dates == EXPECTED_COVERAGE_DEVIATION_DATE)
        _require(
            required.size == 1 and int(required[0]) in set(values.tolist()),
            "fixed smoke coverage initialization is absent or ineligible",
        )
        required_index = int(required[0])
        if required_index not in set(selected.tolist()):
            replaceable = np.arange(1, len(selected) - 1, dtype=np.int64)
            replacement = int(
                replaceable[
                    np.argmin(np.abs(selected[replaceable] - required_index))
                ]
            )
            selected[replacement] = required_index
            selected = np.sort(selected)
        _require(
            np.count_nonzero(dates[selected] == EXPECTED_COVERAGE_DEVIATION_DATE) == 1,
            "fixed smoke does not exercise the 2023 coverage deviation exactly once",
        )
    _require(np.unique(selected).size == count, "smoke selection is not unique")
    return selected


def validate_all_selected_weekly_means(
    dataset: xr.Dataset,
    selected_indices: np.ndarray,
    stored_weekly: np.ndarray,
    *,
    chunk_cases: int = DAILY_VALIDATION_CHUNK_CASES,
) -> dict[str, Any]:
    """Stream every selected native-daily case and verify all weekly grid means."""

    selected = np.asarray(selected_indices, dtype=np.int64)
    weekly_values = np.asarray(stored_weekly, dtype=np.float32)
    if selected.ndim != 1 or selected.size == 0 or chunk_cases < 1:
        raise ValueError("selected indices and positive daily-validation chunk are required")
    if weekly_values.ndim != 5 or weekly_values.shape[0] != selected.size:
        raise ValueError("stored weekly fields must be [selected,member,week,y,x]")
    case_count, member_count, week_count, height, width = weekly_values.shape
    day_count = week_count * 7
    expected_daily_shape = (case_count, member_count, day_count, height, width)
    digest = hashlib.sha256()
    digest.update(np.dtype("<f4").str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(expected_daily_shape), separators=(",", ":")).encode()
    )
    digest.update(b"\0")
    maximum_difference = 0.0
    checked_cases = 0
    checked_chunks = 0
    for start in range(0, case_count, chunk_cases):
        stop = min(start + chunk_cases, case_count)
        source_indices = selected[start:stop]
        daily = np.asarray(
            dataset["forecast"]
            .isel(init=source_indices)
            .transpose("init", "member", "lead_day", "latitude", "longitude")
            .load()
            .values,
            dtype=np.float32,
        )
        expected_chunk_shape = (
            stop - start,
            member_count,
            day_count,
            height,
            width,
        )
        _require(
            daily.shape == expected_chunk_shape,
            f"native-daily chunk shape {daily.shape} != {expected_chunk_shape}",
        )
        _require(
            np.isfinite(daily).all() and np.all(daily >= 0.0),
            "native-daily forecast contains invalid precipitation",
        )
        little_endian = np.ascontiguousarray(daily.astype("<f4", copy=False))
        digest.update(little_endian.tobytes(order="C"))
        recomputed = daily.reshape(
            stop - start,
            member_count,
            week_count,
            7,
            height,
            width,
        ).mean(axis=3, dtype=np.float64).astype(np.float32)
        difference = float(
            np.max(
                np.abs(
                    recomputed.astype(np.float64)
                    - weekly_values[start:stop].astype(np.float64)
                )
            )
        )
        maximum_difference = max(maximum_difference, difference)
        _require(
            difference <= 2.0e-6,
            "stored weekly/native-daily grouping differs in selected chunk "
            f"{start}:{stop}",
        )
        checked_cases += stop - start
        checked_chunks += 1
    _require(checked_cases == case_count, "not every selected native-daily case was checked")
    return {
        "sha256": digest.hexdigest(),
        "dtype": np.dtype("<f4").str,
        "shape": list(expected_daily_shape),
        "validation": {
            "streamed_in_bounded_chunks": True,
            "chunk_case_limit": int(chunk_cases),
            "checked_chunks": checked_chunks,
            "checked_initialization_count": case_count,
            "checked_member_count_per_initialization": member_count,
            "checked_lead_day_count": day_count,
            "checked_lead_week_count": week_count,
            "checked_case_member_week_blocks": case_count
            * member_count
            * week_count,
            "checked_weekly_grid_values": case_count
            * member_count
            * week_count
            * height
            * width,
            "maximum_weekly_vs_daily_absolute_difference_mm_day": maximum_difference,
            "tolerance_mm_day": 2.0e-6,
        },
    }


def validate_stored_ensemble_mean_product(
    members: np.ndarray,
    stored_ensemble_mean: np.ndarray,
) -> dict[str, Any]:
    """Verify the archive's exact float64-reduction-to-float32 mean product.

    The benchmark writer uses ``np.nanmean(..., dtype=np.float64)`` over the
    member axis and then casts the result to float32 before storage.  Comparing
    that stored product with the *unrounded* float64 reduction under a fixed
    absolute tolerance is magnitude dependent and can reject a correct heavy
    rainfall field.  The scientific gate is therefore bitwise identity after
    reproducing the writer's canonical float32 cast.  The pre-round residual is
    retained only as a diagnostic, both in physical units and as a fraction of
    one float32 ULP.
    """

    raw_members = np.asarray(members)
    raw_stored = np.asarray(stored_ensemble_mean)
    if raw_members.dtype != np.dtype(np.float32):
        raise ValueError("ensemble members must retain archive float32 dtype")
    if raw_stored.dtype != np.dtype(np.float32):
        raise ValueError("stored ensemble mean must retain archive float32 dtype")
    if raw_members.ndim != 5 or raw_stored.shape != (
        raw_members.shape[0],
        *raw_members.shape[2:],
    ):
        raise ValueError(
            "members/stored mean must be [case,member,lead,y,x] and [case,lead,y,x]"
        )
    _require(
        np.isfinite(raw_members).all()
        and np.all(raw_members >= 0.0)
        and np.isfinite(raw_stored).all()
        and np.all(raw_stored >= 0.0),
        "member/stored ensemble-mean product contains invalid precipitation",
    )

    unrounded = np.nanmean(raw_members, axis=1, dtype=np.float64)
    canonical = np.ascontiguousarray(unrounded.astype(np.float32))
    stored = np.ascontiguousarray(raw_stored)
    mismatch = canonical.view(np.uint32) != stored.view(np.uint32)
    mismatch_count = int(np.count_nonzero(mismatch))

    pre_round_difference = np.abs(unrounded - stored.astype(np.float64))
    spacing = np.spacing(stored).astype(np.float64)
    fraction_of_ulp = np.divide(
        pre_round_difference,
        spacing,
        out=np.zeros_like(pre_round_difference),
        where=spacing > 0.0,
    )
    maximum_pre_round_difference = float(np.max(pre_round_difference))
    maximum_fraction_of_ulp = float(np.max(fraction_of_ulp))
    _require(
        mismatch_count == 0,
        "canonical float64-nanmean-to-float32 ensemble mean differs bitwise "
        f"from stored product at {mismatch_count} grid values",
    )
    _require(
        maximum_fraction_of_ulp <= 0.5 + np.finfo(np.float64).eps,
        "pre-round ensemble-mean residual exceeds half a stored float32 ULP",
    )
    return {
        "stored_ensemble_mean_reduction_contract": (
            "np.nanmean(member_axis,dtype=float64).astype(float32)"
        ),
        "stored_ensemble_mean_exact_post_float32_cast_identity": True,
        "stored_ensemble_mean_bitwise_mismatch_count": mismatch_count,
        "checked_member_mean_grid_values": int(stored.size),
        "maximum_pre_round_member_mean_absolute_difference_mm_day": (
            maximum_pre_round_difference
        ),
        "maximum_pre_round_member_mean_fraction_of_float32_ulp": (
            maximum_fraction_of_ulp
        ),
    }


def load_operational_members(
    canonical_cache: base.MemberCache,
    *,
    smoke: bool,
) -> OperationalMembers:
    """Load every eligible member from the exact three-store whitelist."""

    member_chunks: list[np.ndarray] = []
    init_chunks: list[np.ndarray] = []
    availability_chunks: list[np.ndarray] = []
    inventories: dict[str, Any] = {}
    paths = operational_store_paths()
    stored_counts: dict[int, int] = {}
    eligible_counts: dict[int, int] = {}
    evaluated_counts: dict[int, int] = {}
    excluded_2024: list[str] = []
    lead_diagnostics: dict[str, Any] = {}
    reference_members: np.ndarray | None = None
    for year, store in zip(AUDIT_YEARS, paths, strict=True):
        # ``chunks=None`` keeps these direct local reads independent of a Dask
        # scheduler and ensures the arrays hashed below are the arrays scored.
        with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
            _require(dataset.attrs.get("model") == "fuxi_s2s", f"model metadata changed: {store}")
            _require(dataset.attrs.get("distribution_representation") == "members", f"not a member store: {store}")
            _require(dataset.attrs.get("variable") == "tp", f"wrong variable: {store}")
            required_dims = {
                "init": EXPECTED_STORED_COUNTS[year],
                "member": FORECAST_MEMBER_COUNT,
                "lead_week": 6,
                "lead_day": 42,
                "latitude": 27,
                "longitude": 27,
            }
            _require(
                all(int(dataset.sizes.get(name, -1)) == size for name, size in required_dims.items()),
                f"unexpected operational dimensions in {store}: {dict(dataset.sizes)}",
            )
            weekly = dataset["forecast_weekly_mean"]
            _require(
                weekly.dims == ("init", "member", "lead_week", "latitude", "longitude"),
                f"weekly member dimension order changed: {store}",
            )
            _require(weekly.attrs.get("units") == "mm day-1", f"weekly units changed: {store}")
            _require(
                weekly.attrs.get("temporal_statistic") == "mean_of_complete_7_day_block",
                f"weekly temporal statistic changed: {store}",
            )
            _require(
                dataset["forecast"].attrs.get("units") == "mm day-1"
                and dataset["forecast"].attrs.get("temporal_statistic") == "daily_mean_rate",
                f"daily forecast convention changed: {store}",
            )
            latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
            longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
            member_labels = np.asarray(dataset.member.values, dtype=np.int64)
            lead_week = np.asarray(dataset.lead_week.values, dtype=np.int64)
            lead_day = np.asarray(dataset.lead_day.values, dtype=np.int64)
            _require(np.array_equal(latitude, canonical_cache.latitude), f"latitude/order differs: {store}")
            _require(np.array_equal(longitude, canonical_cache.longitude), f"longitude/order differs: {store}")
            _require(np.array_equal(lead_week, np.arange(1, 7)), f"lead weeks differ: {store}")
            _require(np.array_equal(lead_day, np.arange(1, 43)), f"lead days differ: {store}")
            _require(
                np.array_equal(member_labels, np.arange(FORECAST_MEMBER_COUNT)),
                f"member labels differ: {store}",
            )
            if reference_members is None:
                reference_members = member_labels.copy()
            else:
                _require(np.array_equal(member_labels, reference_members), "member labels vary by year")

            initializations = np.asarray(dataset.init.values, dtype="datetime64[D]")
            _require(
                np.unique(initializations).size == initializations.size
                and np.all(np.diff(initializations) > np.timedelta64(0, "D")),
                f"initializations are not unique/sorted: {store}",
            )
            retained_mask = eligible_initialization_mask(initializations, year)
            stored_counts[year] = int(initializations.size)
            full_eligible = np.flatnonzero(retained_mask)
            _require(
                full_eligible.size == EXPECTED_ELIGIBLE_COUNTS[year],
                f"{year} eligible count {full_eligible.size} changed",
            )
            eligible_counts[year] = int(full_eligible.size)
            if year == 2024:
                excluded_2024 = [
                    np.datetime_as_string(value, unit="D")
                    for value in initializations[~retained_mask]
                ]
                _require(len(excluded_2024) == 12, "late-2024 exclusion count changed")
                _require(
                    np.all(initializations[~retained_mask] + np.timedelta64(41, "D") >= np.datetime64("2025-01-01")),
                    "a 2024 exclusion does not cross into 2025",
                )
            selected = (
                _select_smoke_indices(
                    initializations,
                    full_eligible,
                    year,
                    SMOKE_CASES_PER_YEAR,
                )
                if smoke
                else full_eligible
            )
            evaluated_counts[year] = int(selected.size)
            selected_inits = initializations[selected]
            available = np.asarray(
                dataset["member_available"].isel(init=selected).load().values,
                dtype=bool,
            )
            _require(
                available.shape == (selected.size, FORECAST_MEMBER_COUNT) and available.all(),
                f"one or more operational members unavailable: {store}",
            )
            values = np.asarray(
                weekly.isel(init=selected)
                .transpose("init", "member", "lead_week", "latitude", "longitude")
                .load()
                .values,
                dtype=np.float32,
            )
            _require(
                values.shape == (selected.size, FORECAST_MEMBER_COUNT, 6, 27, 27),
                f"weekly member shape changed: {store}",
            )
            _require(np.isfinite(values).all() and np.all(values >= 0.0), f"invalid rainfall members: {store}")

            # Verify every selected member/case/week against all 42 native
            # daily leads.  The native daily product is streamed in bounded
            # case chunks and content-hashed without materializing it in full.
            daily_receipt = validate_all_selected_weekly_means(
                dataset,
                selected,
                values,
                chunk_cases=DAILY_VALIDATION_CHUNK_CASES,
            )
            ensemble_mean = np.asarray(
                dataset["ensemble_mean_weekly"].isel(init=selected).load().values,
                dtype=np.float32,
            )
            member_mean_validation = validate_stored_ensemble_mean_product(
                values, ensemble_mean
            )
            daily_receipt["validation"].update(member_mean_validation)

            prefix = f"operational_{year}"
            inventories[f"{prefix}.zmetadata"] = {
                "sha256": sha256_file(store / ".zmetadata"),
                "path": str(store / ".zmetadata"),
            }
            for name, array in (
                ("selected_initializations", selected_inits.astype("datetime64[D]").astype("<i8")),
                ("latitude", latitude.astype("<f8")),
                ("longitude", longitude.astype("<f8")),
                ("member_labels", member_labels.astype("<i8")),
                ("lead_week", lead_week.astype("<i8")),
                ("lead_day", lead_day.astype("<i8")),
                ("member_available", available.astype("|u1")),
                ("forecast_weekly_mean", values.astype("<f4", copy=False)),
                ("ensemble_mean_weekly", ensemble_mean.astype("<f4", copy=False)),
            ):
                inventories[f"{prefix}.{name}"] = array_receipt(array)
            inventories[f"{prefix}.forecast_daily_selected_stream"] = daily_receipt
            lead_diagnostics[str(year)] = {
                "lead_day_values": lead_day.tolist(),
                "lead_week_values": lead_week.tolist(),
                "weekly_definition": "lead days 1-42 reshaped into six successive 7-day blocks",
                "truth_date_offsets_days": list(range(42)),
                **dict(daily_receipt["validation"]),
            }
            member_chunks.append(values)
            init_chunks.append(selected_inits)
            availability_chunks.append(available)

    members = np.concatenate(member_chunks, axis=0)
    initializations = np.concatenate(init_chunks).astype("datetime64[D]")
    member_available = np.concatenate(availability_chunks, axis=0)
    _require(
        np.unique(initializations).size == initializations.size
        and np.all(np.diff(initializations) > np.timedelta64(0, "D")),
        "combined operational initializations are not unique/sorted",
    )
    expected = 3 * SMOKE_CASES_PER_YEAR if smoke else EXPECTED_FULL_CASES
    _require(len(initializations) == expected, f"combined case count {len(initializations)} != {expected}")
    verification_end = initializations + np.timedelta64(41, "D")
    _require(
        np.max(verification_end) <= np.datetime64("2024-12-31"),
        "retained verification enters 2025",
    )
    inventories["combined.initializations"] = array_receipt(
        initializations.astype("datetime64[D]").astype("<i8")
    )
    inventories["combined.forecast_weekly_mean"] = array_receipt(
        members.astype("<f4", copy=False)
    )
    inventories["combined.member_available"] = array_receipt(
        member_available.astype("|u1")
    )
    assert reference_members is not None
    return OperationalMembers(
        members=members,
        initializations=initializations,
        latitude=canonical_cache.latitude.astype(np.float64),
        longitude=canonical_cache.longitude.astype(np.float64),
        member_labels=reference_members,
        member_available=member_available,
        store_paths=tuple(str(path) for path in paths),
        content_inventory=inventories,
        stored_counts=stored_counts,
        eligible_counts=eligible_counts,
        evaluated_counts=evaluated_counts,
        excluded_2024_initializations=tuple(excluded_2024),
        lead_week_diagnostics=lead_diagnostics,
    )


def load_daily_observations(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> DailyObservations:
    """Load only training IMD (2002--2017) and audit IMD (2022--2024)."""

    training_paths = observation_store_paths(TRAIN_YEARS)
    verification_paths = observation_store_paths(AUDIT_YEARS)
    all_paths = (*training_paths, *verification_paths)
    inventories: dict[str, Any] = {}
    training_dates: list[np.ndarray] = []
    training_values: list[np.ndarray] = []
    verification_dates: list[np.ndarray] = []
    verification_values: list[np.ndarray] = []
    verification_fractions: list[np.ndarray] = []
    reference_fraction: np.ndarray | None = None
    for year, store in zip((*TRAIN_YEARS, *AUDIT_YEARS), all_paths, strict=True):
        _reject_forbidden_year_path(store)
        with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
            _require(
                dataset.attrs.get("source") == "imd"
                and dataset.attrs.get("variable") == "tp"
                and dataset.attrs.get("units") == "mm day-1",
                f"IMD metadata changed: {store}",
            )
            store_latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
            store_longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
            _require(np.array_equal(store_latitude, latitude), f"IMD latitude/order differs: {store}")
            _require(np.array_equal(store_longitude, longitude), f"IMD longitude/order differs: {store}")
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            fraction_values = np.asarray(
                dataset.observation_fraction.load().values, dtype=np.float32
            )
            if dataset.observation_fraction.dims == ("latitude", "longitude"):
                _require(fraction_values.shape == (27, 27), f"IMD fraction shape changed: {store}")
                daily_fraction = np.broadcast_to(
                    fraction_values[None], (dates.size, 27, 27)
                ).copy()
                static_fraction = fraction_values
            elif dataset.observation_fraction.dims == (
                "time",
                "latitude",
                "longitude",
            ):
                _require(
                    fraction_values.shape == (dates.size, 27, 27),
                    f"dynamic IMD fraction shape changed: {store}",
                )
                daily_fraction = fraction_values
                static_fraction = fraction_values[0]
            else:
                raise OperationalAuditError(
                    f"unexpected IMD observation_fraction dimensions: {store}"
                )
            _require(
                dates.size in (365, 366)
                and np.all(np.diff(dates) == np.timedelta64(1, "D"))
                and np.all(pd.DatetimeIndex(dates).year.to_numpy() == year),
                f"IMD calendar is incomplete: {store}",
            )
            _require(values.shape == (dates.size, 27, 27), f"IMD shape changed: {store}")
            if reference_fraction is None:
                reference_fraction = static_fraction.copy()
            else:
                _require(
                    np.allclose(
                        reference_fraction,
                        static_fraction,
                        rtol=0.0,
                        atol=1.0e-7,
                        equal_nan=True,
                    ),
                    f"baseline IMD support changed: {store}",
                )
            _require(
                np.isfinite(daily_fraction).all()
                and np.all(daily_fraction >= 0.0)
                and np.all(daily_fraction <= reference_fraction[None] + 1.0e-6),
                f"invalid/dilated daily IMD coverage: {store}",
            )
            support = daily_fraction > 0.0
            _require(
                np.isfinite(values[support]).all() and np.all(values[support] >= 0.0),
                f"invalid supported IMD values: {store}",
            )
            prefix = f"imd_{year}"
            inventories[f"{prefix}.zmetadata"] = {
                "sha256": sha256_file(store / ".zmetadata"),
                "path": str(store / ".zmetadata"),
            }
            inventories[f"{prefix}.dates"] = array_receipt(
                dates.astype("datetime64[D]").astype("<i8")
            )
            inventories[f"{prefix}.observation"] = array_receipt(
                values.astype("<f4", copy=False)
            )
            inventories[f"{prefix}.observation_fraction"] = array_receipt(
                fraction_values.astype("<f4", copy=False)
            )
            if year in TRAIN_YEARS:
                training_dates.append(dates)
                training_values.append(values)
            else:
                verification_dates.append(dates)
                verification_values.append(values)
                verification_fractions.append(daily_fraction)
    assert reference_fraction is not None
    train_dates = np.concatenate(training_dates).astype("datetime64[D]")
    train_values = np.concatenate(training_values).astype(np.float32)
    verify_dates = np.concatenate(verification_dates).astype("datetime64[D]")
    verify_values = np.concatenate(verification_values).astype(np.float32)
    verify_fraction = np.concatenate(verification_fractions).astype(np.float32)
    _require(np.unique(train_dates).size == train_dates.size, "duplicate training IMD dates")
    _require(np.unique(verify_dates).size == verify_dates.size, "duplicate verification IMD dates")
    coverage_deviation_receipt = validate_daily_coverage_deviations(
        verify_dates,
        verify_fraction,
        reference_fraction,
        latitude,
        longitude,
    )
    inventories["combined.training_dates"] = array_receipt(train_dates.astype("<i8"))
    inventories["combined.training_observation"] = array_receipt(train_values.astype("<f4", copy=False))
    inventories["combined.verification_dates"] = array_receipt(verify_dates.astype("<i8"))
    inventories["combined.verification_observation"] = array_receipt(verify_values.astype("<f4", copy=False))
    inventories["combined.verification_observation_fraction"] = array_receipt(
        verify_fraction.astype("<f4", copy=False)
    )
    inventories["combined.observation_fraction"] = array_receipt(reference_fraction.astype("<f4", copy=False))
    return DailyObservations(
        training_dates=train_dates,
        training_values=train_values,
        verification_dates=verify_dates,
        verification_values=verify_values,
        verification_fraction=verify_fraction,
        observation_fraction=reference_fraction,
        training_stores=tuple(str(path) for path in training_paths),
        verification_stores=tuple(str(path) for path in verification_paths),
        content_inventory=inventories,
        coverage_deviation_receipt=coverage_deviation_receipt,
    )


def weekly_climatology_for_initializations(
    daily_climatology: np.ndarray,
    initializations: np.ndarray,
) -> np.ndarray:
    valid_dates = base.derive_valid_dates(initializations)
    positions = base.calendar_positions(valid_dates)
    result = np.mean(
        np.asarray(daily_climatology, dtype=np.float32)[positions],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    if result.shape != (len(initializations), 6, 27, 27):
        raise OperationalAuditError(f"unexpected weekly climatology shape {result.shape}")
    return result


def _context_from_frozen_normalization(
    operational: OperationalMembers,
    weekly_climatology: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    support: np.ndarray,
) -> base.ContextBundle:
    log_climatology = np.log1p(weekly_climatology).astype(np.float32)
    normalized = (log_climatology - means[None, :, None, None]) / stds[
        None, :, None, None
    ]
    normalized = np.where(
        support[None, None] & np.isfinite(normalized), normalized, 0.0
    ).astype(np.float32)
    latitude = operational.latitude.astype(np.float32)
    longitude = operational.longitude.astype(np.float32)
    lat_scaled = (
        2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    )
    lon_scaled = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    midpoints = base.derive_valid_dates(operational.initializations)[:, :, 3]
    midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
    day_of_year = (midpoint_index.dayofyear.to_numpy() - 1).reshape(
        len(operational.initializations), 6
    )
    angle = 2.0 * np.pi * day_of_year / 365.2425
    context = base.ContextBundle(
        normalized_climatology=normalized,
        climatology_mean_by_lead=np.asarray(means, dtype=np.float32),
        climatology_std_by_lead=np.asarray(stds, dtype=np.float32),
        latitude_scaled=lat_scaled.astype(np.float32),
        longitude_scaled=lon_scaled.astype(np.float32),
        season_sin=np.sin(angle).astype(np.float32),
        season_cos=np.cos(angle).astype(np.float32),
        lead_scaled=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        support=np.asarray(support, dtype=bool),
    )
    # Exercise the exact seven-channel runtime contract at both ends.
    for index in (0, len(operational.initializations) - 1):
        _require(
            base.context_for_case(context, index).shape == (6, 7, 27, 27),
            "operational model context contract changed",
        )
    return context


def aggregate_weekly_observations(
    daily_truth: np.ndarray,
    daily_fraction: np.ndarray,
    cell_area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate seven days using actual space-time observation coverage."""

    values = np.asarray(daily_truth, dtype=np.float32)
    fractions = np.asarray(daily_fraction, dtype=np.float32)
    area = np.asarray(cell_area, dtype=np.float64)
    if values.shape != fractions.shape or values.ndim != 5 or values.shape[2] != 7:
        raise ValueError("daily truth/fraction must share [case,lead,7,y,x]")
    if area.shape != values.shape[-2:]:
        raise ValueError("cell area does not match daily observation grid")
    valid = (fractions > 0.0) & np.isfinite(values) & np.isfinite(fractions)
    weighted = np.where(valid, values, 0.0).astype(np.float64) * np.where(
        valid, fractions, 0.0
    ).astype(np.float64)
    fraction_sum = np.where(valid, fractions, 0.0).sum(axis=2, dtype=np.float64)
    truth = np.full(fraction_sum.shape, np.nan, dtype=np.float64)
    np.divide(
        weighted.sum(axis=2, dtype=np.float64),
        fraction_sum,
        out=truth,
        where=fraction_sum > 0.0,
    )
    weekly_fraction = fraction_sum / 7.0
    weights = area[None, None] * weekly_fraction
    weights[~np.isfinite(weights) | (weights <= 0.0)] = 0.0
    return truth.astype(np.float32), weekly_fraction, weights


def build_audit_targets(
    canonical_cache: base.MemberCache,
    operational: OperationalMembers,
    observations: DailyObservations,
    receipt: capacity_gate.CapacityReceipt,
    *,
    smoke: bool,
) -> AuditTargets:
    """Rebuild the frozen training context, then extend it without refitting."""

    support = observations.observation_fraction > 0.0
    daily_climatology = base.build_training_climatology(
        observations.training_dates,
        observations.training_values,
        support,
    )
    canonical_weekly_climatology = weekly_climatology_for_initializations(
        daily_climatology, canonical_cache.initializations
    )
    split = base.make_split_indices(canonical_cache.initializations)
    _require(len(split.train) == base.EXPECTED_COUNTS["train"], "training split changed")

    spatial_path = base.SPATIAL_STORE
    with xr.open_zarr(spatial_path, consolidated=True, chunks=None) as spatial:
        spatial_latitude = np.asarray(spatial.latitude.values, dtype=np.float64)
        spatial_longitude = np.asarray(spatial.longitude.values, dtype=np.float64)
        area = np.asarray(spatial.india_area_weight_km2.load().values, dtype=np.float64)
    _require(np.array_equal(spatial_latitude, operational.latitude), "spatial latitude differs")
    _require(np.array_equal(spatial_longitude, operational.longitude), "spatial longitude differs")
    frozen_weights = area * observations.observation_fraction.astype(np.float64)
    frozen_weights[~np.isfinite(frozen_weights) | (frozen_weights <= 0.0)] = 0.0
    _require(
        np.count_nonzero(frozen_weights) == base.SUPPORT_CELLS,
        "frozen scoring support changed",
    )

    scoring_support_path = receipt.root / "evaluation/scoring_support.npz"
    with np.load(scoring_support_path, allow_pickle=False) as frozen:
        _require(np.array_equal(frozen["latitude"], operational.latitude), "frozen scoring latitude differs")
        _require(np.array_equal(frozen["longitude"], operational.longitude), "frozen scoring longitude differs")
        _require(
            np.allclose(
                frozen["observation_fraction"],
                observations.observation_fraction,
                rtol=0.0,
                atol=1.0e-7,
                equal_nan=True,
            ),
            "frozen observation fraction differs",
        )
        _require(np.array_equal(frozen["support_mask"], support), "frozen support mask differs")
        _require(
            np.array_equal(frozen["scoring_weight_km2_fraction"], frozen_weights),
            "frozen scoring weights differ",
        )

    log_canonical = np.log1p(canonical_weekly_climatology).astype(np.float32)
    means, stds = base.weighted_lead_moments(
        log_canonical, split.train, frozen_weights
    )
    _require(means.shape == (6,) and stds.shape == (6,), "normalization shape changed")
    _require(np.isfinite(means).all() and np.all(stds > 0.0), "invalid frozen normalization")
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    actual_mean_sha256 = array_sha256(means.astype("<f4", copy=False))
    actual_std_sha256 = array_sha256(stds.astype("<f4", copy=False))
    _require(
        np.array_equal(means, EXPECTED_NORMALIZATION_MEAN)
        and actual_mean_sha256 == EXPECTED_NORMALIZATION_MEAN_SHA256,
        "reconstructed normalization mean differs from the frozen neural receipt",
    )
    _require(
        np.array_equal(stds, EXPECTED_NORMALIZATION_STD)
        and actual_std_sha256 == EXPECTED_NORMALIZATION_STD_SHA256,
        "reconstructed normalization std differs from the frozen neural receipt",
    )

    audit_climatology = weekly_climatology_for_initializations(
        daily_climatology, operational.initializations
    )
    valid_dates = base.derive_valid_dates(operational.initializations)
    requested = valid_dates.reshape(-1)
    _require(
        np.min(requested) >= np.datetime64("2022-01-01")
        and np.max(requested) <= np.datetime64("2024-12-31"),
        "verification date firewall failed",
    )
    positions = np.searchsorted(observations.verification_dates, requested)
    _require(
        np.all(positions < observations.verification_dates.size)
        and np.array_equal(observations.verification_dates[positions], requested),
        "one or more whitelisted verification days are absent",
    )
    daily_truth = observations.verification_values[positions].reshape(
        *valid_dates.shape, 27, 27
    )
    daily_fraction = observations.verification_fraction[positions].reshape(
        *valid_dates.shape, 27, 27
    )
    truth, weekly_fraction, weights = aggregate_weekly_observations(
        daily_truth, daily_fraction, area
    )
    _require(
        truth.shape == audit_climatology.shape
        == (len(operational.initializations), 6, 27, 27),
        "truth/climatology alignment changed",
    )
    _require(
        np.isfinite(truth[weights > 0.0]).all()
        and np.all(truth[weights > 0.0] >= 0.0),
        "invalid audit truth on dynamic scoring support",
    )
    _require(
        np.all(weights <= frozen_weights[None, None] + 1.0e-6),
        "dynamic verification weights exceed the frozen spatial support",
    )
    support_counts = np.count_nonzero(weights > 0.0, axis=(-2, -1))
    _require(
        int(support_counts.min()) == base.SUPPORT_CELLS
        and int(support_counts.max()) == base.SUPPORT_CELLS,
        f"unexpected dynamic support range {support_counts.min()}-{support_counts.max()}",
    )
    positive_frozen = frozen_weights > 0.0
    weight_ratio = weights[..., positive_frozen] / frozen_weights[positive_frozen]
    _require(
        np.all(weight_ratio > 0.0) and np.all(weight_ratio <= 1.0 + 1.0e-6),
        "dynamic verification-weight ratios are invalid",
    )
    coverage_contract = validate_evaluated_coverage_contract(
        operational.initializations,
        valid_dates,
        daily_fraction,
        observations.observation_fraction,
        weight_ratio,
        smoke=smoke,
    )
    context = _context_from_frozen_normalization(
        operational, audit_climatology, means, stds, support
    )

    normalization_inventory = {
        "fit_years": list(TRAIN_YEARS),
        "train_case_count": len(split.train),
        "formula": "log1p calendar climatology normalized by area-weighted lead moments from 2002-2017 train cases",
        "climatology_window_days": 31,
        "equal_year_weighting": True,
        "climatology_mean_by_lead": means.tolist(),
        "climatology_std_by_lead": stds.tolist(),
        "mean_sha256": actual_mean_sha256,
        "std_sha256": actual_std_sha256,
        "expected_climatology_mean_by_lead": EXPECTED_NORMALIZATION_MEAN.tolist(),
        "expected_climatology_std_by_lead": EXPECTED_NORMALIZATION_STD.tolist(),
        "expected_mean_sha256": EXPECTED_NORMALIZATION_MEAN_SHA256,
        "expected_std_sha256": EXPECTED_NORMALIZATION_STD_SHA256,
        "frozen_reference": NORMALIZATION_REFERENCE,
        "exact_frozen_value_and_hash_gate_passed": True,
        "normalization_refit_on_2022_2024": False,
        "evaluation_coverage_used_for_normalization": False,
    }
    content = {
        "spatial.zmetadata": {
            "path": str(spatial_path / ".zmetadata"),
            "sha256": sha256_file(spatial_path / ".zmetadata"),
        },
        "spatial.latitude": array_receipt(spatial_latitude.astype("<f8")),
        "spatial.longitude": array_receipt(spatial_longitude.astype("<f8")),
        "spatial.india_area_weight_km2": array_receipt(area.astype("<f8")),
        "derived.frozen_spatial_weights": array_receipt(
            frozen_weights.astype("<f8")
        ),
        "derived.dynamic_scoring_weights": array_receipt(weights.astype("<f8")),
        "derived.audit_weekly_observation_fraction": array_receipt(
            weekly_fraction.astype("<f8")
        ),
        "derived.dynamic_to_frozen_weight_ratio": array_receipt(
            weight_ratio.astype("<f8")
        ),
        "derived.training_climatology_daily": array_receipt(
            daily_climatology.astype("<f4", copy=False)
        ),
        "derived.canonical_weekly_climatology": array_receipt(
            canonical_weekly_climatology.astype("<f4", copy=False)
        ),
        "derived.audit_weekly_climatology": array_receipt(
            audit_climatology.astype("<f4", copy=False)
        ),
        "derived.audit_weekly_truth": array_receipt(truth.astype("<f4", copy=False)),
        "derived.audit_valid_dates": array_receipt(valid_dates.astype("<i8")),
        "derived.normalized_audit_climatology": array_receipt(
            context.normalized_climatology.astype("<f4", copy=False)
        ),
    }
    return AuditTargets(
        truth=truth,
        climatology=audit_climatology,
        weights=weights,
        frozen_spatial_weights=frozen_weights,
        context=context,
        training_climatology_daily=daily_climatology,
        normalization_inventory=normalization_inventory,
        content_inventory=content,
        coverage_contract=coverage_contract,
    )


def _dynamic_weighted_field_mean(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    fields = np.asarray(values, dtype=np.float64)
    dynamic = np.asarray(weights, dtype=np.float64)
    if fields.shape != dynamic.shape or fields.ndim != 4:
        raise ValueError(
            f"dynamic fields/weights must share [case,lead,y,x], got {fields.shape}/{dynamic.shape}"
        )
    valid = (dynamic > 0.0) & np.isfinite(dynamic)
    if not np.isfinite(fields[valid]).all():
        raise OperationalAuditError("metric field is invalid on dynamic support")
    numerator = np.sum(
        np.where(valid, fields * dynamic, 0.0), axis=(-2, -1), dtype=np.float64
    )
    denominator = np.sum(
        np.where(valid, dynamic, 0.0), axis=(-2, -1), dtype=np.float64
    )
    if np.any(denominator <= 0.0):
        raise OperationalAuditError("a case/lead has zero verification weight")
    return numerator / denominator


def _dynamic_centred_acc(
    truth_anomaly: np.ndarray,
    prediction_anomaly: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(truth_anomaly, dtype=np.float64)
    prediction = np.asarray(prediction_anomaly, dtype=np.float64)
    dynamic = np.asarray(weights, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape != dynamic.shape:
        raise ValueError("dynamic ACC arrays must share a shape")
    valid = (dynamic > 0.0) & np.isfinite(dynamic)
    if not np.isfinite(truth[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise OperationalAuditError("ACC anomaly is invalid on dynamic support")
    safe_weights = np.where(valid, dynamic, 0.0)
    denominator_weight = safe_weights.sum(axis=(-2, -1), dtype=np.float64)
    truth_mean = np.sum(
        np.where(valid, truth * dynamic, 0.0), axis=(-2, -1), dtype=np.float64
    ) / denominator_weight
    prediction_mean = np.sum(
        np.where(valid, prediction * dynamic, 0.0),
        axis=(-2, -1),
        dtype=np.float64,
    ) / denominator_weight
    truth_centred = np.where(valid, truth - truth_mean[..., None, None], 0.0)
    prediction_centred = np.where(
        valid, prediction - prediction_mean[..., None, None], 0.0
    )
    covariance = np.sum(
        truth_centred * prediction_centred * safe_weights,
        axis=(-2, -1),
        dtype=np.float64,
    )
    truth_variance = np.sum(
        truth_centred**2 * safe_weights, axis=(-2, -1), dtype=np.float64
    )
    prediction_variance = np.sum(
        prediction_centred**2 * safe_weights, axis=(-2, -1), dtype=np.float64
    )
    denominator = np.sqrt(truth_variance * prediction_variance)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    return np.clip(result, -1.0, 1.0)


def evaluate_ensemble_dynamic(
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
    """Canonical ensemble metrics with case/lead-specific observation weights."""

    n_cases, member_count, n_leads, height, width = members.shape
    expected = (n_cases, n_leads, height, width)
    if truth.shape != expected or climatology.shape != expected or weights.shape != expected:
        raise OperationalAuditError("dynamic evaluation arrays are not aligned")
    rng = np.random.default_rng(rank_seed)
    rank_counts = np.zeros((n_leads, member_count + 1), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    labels = base.season_labels(initializations)
    init_strings = [np.datetime_as_string(value, unit="D") for value in initializations]
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    for start in range(0, n_cases, chunk_size):
        stop = min(start + chunk_size, n_cases)
        ensemble = np.asarray(members[start:stop], dtype=np.float32)
        target = np.asarray(truth[start:stop], dtype=np.float32)
        climate = np.asarray(climatology[start:stop], dtype=np.float32)
        dynamic = np.asarray(weights[start:stop], dtype=np.float64)
        valid = dynamic > 0.0
        if not np.isfinite(ensemble).all() or np.any(ensemble < 0.0):
            raise OperationalAuditError(f"{method} ensemble contains invalid values")
        if not np.isfinite(target[valid]).all() or np.any(target[valid] < 0.0):
            raise OperationalAuditError("truth is invalid on dynamic support")
        if not np.isfinite(climate[valid]).all():
            raise OperationalAuditError("climatology is invalid on dynamic support")
        mean = ensemble.mean(axis=1, dtype=np.float64)
        error = mean - target
        crps = _dynamic_weighted_field_mean(
            base.numpy_ensemble_crps(ensemble, target), dynamic
        )
        mean_squared_error = _dynamic_weighted_field_mean(error**2, dynamic)
        rmse = np.sqrt(mean_squared_error)
        mae = _dynamic_weighted_field_mean(np.abs(error), dynamic)
        bias = _dynamic_weighted_field_mean(error, dynamic)
        acc = _dynamic_centred_acc(target - climate, mean - climate, dynamic)
        ensemble_variance = _dynamic_weighted_field_mean(
            ensemble.var(axis=1, ddof=0, dtype=np.float64), dynamic
        )
        spread = np.sqrt(ensemble_variance)
        extras: dict[str, np.ndarray] = {}
        for threshold in base.THRESHOLDS_MM_DAY:
            probability = np.mean(ensemble >= threshold, axis=1, dtype=np.float64)
            event = (target >= threshold).astype(np.float64)
            suffix = f"{threshold:g}"
            extras[f"brier_{suffix}"] = _dynamic_weighted_field_mean(
                (probability - event) ** 2, dynamic
            )
            extras[f"forecast_probability_{suffix}"] = _dynamic_weighted_field_mean(
                probability, dynamic
            )
            extras[f"observed_frequency_{suffix}"] = _dynamic_weighted_field_mean(
                event, dynamic
            )
        for coverage in base.COVERAGES:
            alpha = (1.0 - coverage) / 2.0
            lower, upper = np.quantile(
                ensemble, (alpha, 1.0 - alpha), axis=1, method="linear"
            )
            covered = ((target >= lower) & (target <= upper)).astype(np.float64)
            suffix = f"{int(round(100 * coverage))}"
            extras[f"coverage_{suffix}"] = _dynamic_weighted_field_mean(
                covered, dynamic
            )
            extras[f"width_{suffix}"] = _dynamic_weighted_field_mean(
                upper - lower, dynamic
            )
        for lead in range(n_leads):
            lead_valid = valid[:, lead]
            member_values = np.moveaxis(ensemble[:, :, lead], 1, -1)[lead_valid]
            target_values = target[:, lead][lead_valid]
            rank_weights = dynamic[:, lead][lead_valid]
            lower = np.sum(member_values < target_values[:, None], axis=1)
            ties = np.sum(member_values == target_values[:, None], axis=1)
            tie_offsets = np.asarray(
                [rng.integers(0, int(tie) + 1) for tie in ties], dtype=np.int64
            )
            ranks = lower + tie_offsets
            rank_counts[lead] += np.bincount(
                ranks, weights=rank_weights, minlength=member_count + 1
            )
        support_counts = np.count_nonzero(valid, axis=(-2, -1))
        weight_sums = dynamic.sum(axis=(-2, -1), dtype=np.float64)
        for local in range(stop - start):
            global_index = start + local
            for lead in range(n_leads):
                row: dict[str, Any] = {
                    "split": "operational_era_retrospective",
                    "method": method,
                    "method_label": base.METHOD_LABELS[method],
                    "seed": seed_label,
                    "init": init_strings[global_index],
                    "year": int(years[global_index]),
                    "season": str(labels[global_index]),
                    "lead_week": lead + 1,
                    "member_count": member_count,
                    "support_cells": int(support_counts[local, lead]),
                    "scoring_weight_sum_km2_fraction": float(weight_sums[local, lead]),
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
            "weighting": "dynamic india_area_weight_km2 x weekly observation_fraction",
        }
        for lead in range(n_leads)
        for rank in range(member_count + 1)
    ]
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def reliability_bins_dynamic(
    method: str,
    members: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    *,
    bin_count: int = 10,
    chunk_size: int = 8,
) -> pd.DataFrame:
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    n_cases, _, n_leads, _, _ = members.shape
    if truth.shape != weights.shape or truth.shape[0] != n_cases:
        raise ValueError("dynamic reliability arrays are not aligned")
    shape = (n_leads, len(base.THRESHOLDS_MM_DAY), bin_count)
    weight_sum = np.zeros(shape, dtype=np.float64)
    probability_sum = np.zeros(shape, dtype=np.float64)
    event_sum = np.zeros(shape, dtype=np.float64)
    sample_count = np.zeros(shape, dtype=np.int64)
    for start in range(0, n_cases, chunk_size):
        stop = min(start + chunk_size, n_cases)
        ensemble = np.asarray(members[start:stop], dtype=np.float32)
        target = np.asarray(truth[start:stop], dtype=np.float32)
        dynamic = np.asarray(weights[start:stop], dtype=np.float64)
        for threshold_index, threshold in enumerate(base.THRESHOLDS_MM_DAY):
            probability = np.mean(ensemble >= threshold, axis=1, dtype=np.float64)
            event = target >= threshold
            for lead in range(n_leads):
                valid = dynamic[:, lead] > 0.0
                lead_probability = probability[:, lead][valid]
                lead_event = event[:, lead][valid].astype(np.float64)
                lead_weights = dynamic[:, lead][valid]
                bins = np.minimum(
                    (lead_probability * bin_count).astype(np.int64), bin_count - 1
                )
                for bin_index in range(bin_count):
                    selected = bins == bin_index
                    if not np.any(selected):
                        continue
                    selected_weights = lead_weights[selected]
                    weight_sum[lead, threshold_index, bin_index] += selected_weights.sum()
                    probability_sum[lead, threshold_index, bin_index] += np.sum(
                        selected_weights * lead_probability[selected], dtype=np.float64
                    )
                    event_sum[lead, threshold_index, bin_index] += np.sum(
                        selected_weights * lead_event[selected], dtype=np.float64
                    )
                    sample_count[lead, threshold_index, bin_index] += int(
                        np.count_nonzero(selected)
                    )
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    rows: list[dict[str, Any]] = []
    for lead in range(n_leads):
        for threshold_index, threshold in enumerate(base.THRESHOLDS_MM_DAY):
            for bin_index in range(bin_count):
                denominator = weight_sum[lead, threshold_index, bin_index]
                rows.append(
                    {
                        "method": method,
                        "method_label": base.METHOD_LABELS[method],
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


def verification_seasons(initializations: np.ndarray) -> np.ndarray:
    midpoints = base.derive_valid_dates(initializations)[:, :, 3]
    months = pd.DatetimeIndex(midpoints.reshape(-1)).month.to_numpy().reshape(-1, 6)
    return np.asarray(
        [
            "DJF"
            if month in (12, 1, 2)
            else "MAM"
            if month in (3, 4, 5)
            else "JJA"
            if month in (6, 7, 8)
            else "SON"
            for month in months.reshape(-1)
        ],
        dtype=object,
    ).reshape(months.shape)


def relabel_case_metrics(
    frame: pd.DataFrame, initializations: np.ndarray
) -> pd.DataFrame:
    """Replace the inherited development/init-season labels with audit labels."""

    result = frame.copy()
    result["split"] = "operational_era_retrospective"
    init_strings = [np.datetime_as_string(value, unit="D") for value in initializations]
    initialization_labels = base.season_labels(initializations)
    verification_labels = verification_seasons(initializations)
    init_season = dict(zip(init_strings, initialization_labels, strict=True))
    verify_season = {
        (init_strings[case], lead + 1): verification_labels[case, lead]
        for case in range(len(init_strings))
        for lead in range(6)
    }
    result["initialization_season"] = result["init"].map(init_season)
    result["season"] = [
        verify_season[(str(init), int(lead))]
        for init, lead in zip(result["init"], result["lead_week"], strict=True)
    ]
    _require(not result["season"].isna().any(), "verification season mapping failed")
    return result


def two_stage_year_block_indices(
    initializations: np.ndarray,
    *,
    n_resamples: int,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> TwoStageBootstrap:
    """Resample years, then circular blocks within every sampled year."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0 or np.unique(dates).size != dates.size:
        raise ValueError("bootstrap initializations must be a non-empty unique 1-D array")
    if n_resamples < 1 or block_length < 1:
        raise ValueError("bootstrap draws and block length must be positive")
    years_array = pd.DatetimeIndex(dates).year.to_numpy()
    source_years = tuple(int(year) for year in np.sort(np.unique(years_array)))
    if source_years != AUDIT_YEARS:
        raise ValueError(f"bootstrap years {source_years} differ from {AUDIT_YEARS}")
    groups = {year: np.flatnonzero(years_array == year) for year in source_years}
    if any(len(group) == 0 for group in groups.values()):
        raise ValueError("bootstrap year group is empty")
    rng = np.random.default_rng(seed)
    sampled_years = rng.choice(
        np.asarray(source_years, dtype=np.int64),
        size=(n_resamples, len(source_years)),
        replace=True,
    )
    draws: list[np.ndarray] = []
    for sampled in sampled_years:
        segments: list[np.ndarray] = []
        for sampled_year in sampled:
            group = groups[int(sampled_year)]
            effective_block = min(block_length, len(group))
            block_count = int(math.ceil(len(group) / effective_block))
            starts = rng.integers(0, len(group), size=block_count)
            offsets = np.arange(effective_block, dtype=np.int64)
            local = ((starts[:, None] + offsets[None]) % len(group)).reshape(-1)
            segments.append(group[local[: len(group)]])
        draws.append(np.concatenate(segments).astype(np.int64, copy=False))
    return TwoStageBootstrap(
        draws=tuple(draws),
        sampled_years=sampled_years,
        source_years=source_years,
        year_sizes={year: int(len(group)) for year, group in groups.items()},
        block_length=int(block_length),
        seed=int(seed),
    )


def bootstrap_diagnostics(plan: TwoStageBootstrap) -> dict[str, Any]:
    draw_lengths = np.asarray([len(draw) for draw in plan.draws], dtype=np.int64)
    repeated_year_draws = np.asarray(
        [len(set(row.tolist())) < len(row) for row in plan.sampled_years], dtype=bool
    )
    return {
        "two_stage": True,
        "stage_one": "sample initialization years with replacement",
        "stage_two": "sample circular moving blocks within every sampled year",
        "source_years": list(plan.source_years),
        "year_sizes": dict(plan.year_sizes),
        "block_length_initializations": plan.block_length,
        "seed": plan.seed,
        "draws": len(plan.draws),
        "minimum_initializations_per_draw": int(draw_lengths.min()),
        "maximum_initializations_per_draw": int(draw_lengths.max()),
        "draws_with_repeated_year_cluster": int(repeated_year_draws.sum()),
        "all_six_leads_and_all_members_grouped_with_initialization": True,
        "small_number_of_year_clusters_limitation": len(plan.source_years),
    }


def paired_two_stage_bootstrap(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    plan: TwoStageBootstrap,
    *,
    method: str = capacity.BASE_CANDIDATE,
    baseline: str = "raw_fuxi",
    optimization_seed: str | int = "mean_of_seed_scores",
) -> pd.DataFrame:
    """Paired CIs with initialization/member/lead dependence preserved."""

    init_order = [np.datetime_as_string(value, unit="D") for value in initializations]
    rows: list[dict[str, Any]] = []
    lead_scopes = [("W1-W6", tuple(range(1, 7)))] + [
        (f"W{lead}", (lead,)) for lead in range(1, 7)
    ]
    for lead_scope, leads in lead_scopes:
        selected = case_metrics.loc[
            case_metrics.lead_week.isin(leads)
            & case_metrics.method.isin((baseline, method))
        ]
        for metric in CORE_METRICS:
            pivot = selected.pivot_table(
                index="init", columns="method", values=metric, aggfunc="mean"
            ).reindex(init_order)
            if set((baseline, method)) - set(pivot.columns) or pivot[[baseline, method]].isna().any().any():
                raise OperationalAuditError("bootstrap comparison is not exactly paired")
            reference = pivot[baseline].to_numpy(dtype=np.float64)
            calibrated = pivot[method].to_numpy(dtype=np.float64)
            if metric in {"crps", "rmse", "mae"}:
                effect_name = f"{metric}_skill_pct_vs_{baseline}"
                observed = 100.0 * (reference.mean() - calibrated.mean()) / reference.mean()
                effects = np.asarray(
                    [
                        100.0
                        * (reference[draw].mean() - calibrated[draw].mean())
                        / reference[draw].mean()
                        for draw in plan.draws
                    ],
                    dtype=np.float64,
                )
            else:
                effect_name = f"delta_{metric}_vs_{baseline}"
                observed = calibrated.mean() - reference.mean()
                effects = np.asarray(
                    [calibrated[draw].mean() - reference[draw].mean() for draw in plan.draws],
                    dtype=np.float64,
                )
            rows.append(
                {
                    "comparison_scope": "paired_seed_score_average_vs_raw"
                    if optimization_seed == "mean_of_seed_scores"
                    else "paired_individual_seed_vs_raw",
                    "optimization_seed": optimization_seed,
                    "method": method,
                    "baseline": baseline,
                    "lead_scope": lead_scope,
                    "metric": metric,
                    "effect_name": effect_name,
                    "effect": float(observed),
                    "ci_lower_2p5": float(np.percentile(effects, 2.5)),
                    "ci_upper_97p5": float(np.percentile(effects, 97.5)),
                    "bootstrap_probability_effect_positive": float(np.mean(effects > 0.0)),
                    "paired_initializations": len(initializations),
                    "source_year_clusters": len(plan.source_years),
                    "n_resamples": len(plan.draws),
                    "block_length_initializations": plan.block_length,
                    "bootstrap_seed": plan.seed,
                    "resampling_unit": "initialization; 50 members and all selected lead weeks remain grouped",
                    "bootstrap": "year resampling plus circular 13-initialization blocks within sampled year",
                }
            )
    return pd.DataFrame(rows)


def build_headline_case_metrics(
    raw_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Average scores across seeds, never parameters, fields, or predictions."""

    averaged = base.mean_seed_case_metrics(seed_metrics, SEEDS)
    result = pd.concat((raw_metrics, averaged), ignore_index=True)
    expected = raw_metrics["init"].nunique() * 6 * len(METHODS)
    _require(len(result) == expected, f"headline case rows {len(result)} != {expected}")
    keys = ["method", "init", "lead_week"]
    _require(not result.duplicated(keys).any(), "headline method/case/lead rows duplicate")
    return result


def summarize_years(
    headline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_frames: list[pd.DataFrame] = []
    week_frames: list[pd.DataFrame] = []
    for year in AUDIT_YEARS:
        selected = headline.loc[headline.year.eq(year)].drop(
            columns=["initialization_season"], errors="ignore"
        )
        _require(not selected.empty, f"year {year} is absent from metrics")
        weekwise, pooled, _, _ = base.summarize_metrics(selected)
        weekwise.insert(1, "year", year)
        pooled.insert(0, "year", year)
        weekwise["split"] = "operational_era_retrospective"
        pooled_frames.append(pooled)
        week_frames.append(weekwise)
    return (
        pd.concat(pooled_frames, ignore_index=True),
        pd.concat(week_frames, ignore_index=True),
    )


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_operational_era_audit.py": Path(__file__).resolve(),
        "src/fuxi_allseason_capacity_development_evaluation.py": Path(
            capacity_gate.__file__
        ).resolve(),
        "src/fuxi_allseason_capacity_ablation.py": Path(capacity.__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": Path(base.__file__).resolve(),
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md": PROTOCOL_PATH,
        "slurm/evaluate_allseason_operational_era.sbatch": SLURM_PATH,
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"source snapshot input is missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    excluded = {"manifest.json", "failure.json", "slurm_gate_receipt.json"}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative not in excluded:
            result[relative] = sha256_file(path)
    return result


def _read_contract_csv(path: Path) -> pd.DataFrame:
    _require(path.is_file(), f"semantic audit artifact is missing: {path}")
    frame = pd.read_csv(path)
    _require(not frame.empty, f"semantic audit artifact is empty: {path}")
    _require(
        not any(str(column).startswith("Unnamed:") for column in frame.columns),
        f"semantic audit artifact has an unnamed column: {path}",
    )
    return frame


def _semantic_sort(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    _require(set(keys).issubset(frame.columns), f"semantic comparison keys absent: {keys}")
    order = sorted(
        range(len(frame)),
        key=lambda index: tuple(str(frame.iloc[index][key]) for key in keys),
    )
    return frame.iloc[order].reset_index(drop=True)


def _require_frames_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
    atol: float = 2.0e-10,
) -> None:
    """Compare regenerated tables after CSV serialization without trusting row order."""

    _require(set(actual.columns) == set(expected.columns), f"{label} columns differ")
    _require(len(actual) == len(expected), f"{label} row count differs")
    _require(not actual.duplicated(list(keys)).any(), f"{label} has duplicate row identities")
    _require(not expected.duplicated(list(keys)).any(), f"regenerated {label} has duplicate identities")
    left = _semantic_sort(actual, keys)
    right = _semantic_sort(expected, keys)
    for column in sorted(left.columns):
        left_column = left[column]
        right_column = right[column]
        if pd.api.types.is_numeric_dtype(left_column) and pd.api.types.is_numeric_dtype(
            right_column
        ):
            _require(
                np.allclose(
                    left_column.to_numpy(dtype=np.float64),
                    right_column.to_numpy(dtype=np.float64),
                    rtol=2.0e-10,
                    atol=atol,
                    equal_nan=True,
                ),
                f"{label} numeric column differs: {column}",
            )
        else:
            left_values = left_column.fillna("<NA>").astype(str).to_numpy()
            right_values = right_column.fillna("<NA>").astype(str).to_numpy()
            _require(
                np.array_equal(left_values, right_values),
                f"{label} categorical column differs: {column}",
            )


def _require_finite_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    missing = set(columns) - set(frame.columns)
    _require(not missing, f"{label} required numeric columns absent: {sorted(missing)}")
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), f"{label} contains non-finite required scores")


def _validate_case_metric_semantics(
    headline: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    evaluated: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    smoke: bool,
) -> np.ndarray:
    expected_total = 3 * SMOKE_CASES_PER_YEAR if smoke else EXPECTED_FULL_CASES
    expected_year_counts = (
        {2022: 4, 2023: 4, 2024: 4}
        if smoke
        else EXPECTED_ELIGIBLE_COUNTS
    )
    required_identifiers = {
        "split",
        "method",
        "method_label",
        "seed",
        "init",
        "year",
        "season",
        "initialization_season",
        "lead_week",
        "member_count",
        "support_cells",
        "scoring_weight_sum_km2_fraction",
    }
    _require(required_identifiers.issubset(headline.columns), "headline identifiers absent")
    _require(required_identifiers.issubset(seed_metrics.columns), "seed identifiers absent")
    _require(
        len(headline) == expected_total * 6 * len(METHODS),
        "headline case/lead/method row count differs",
    )
    _require(
        len(seed_metrics) == expected_total * 6 * len(SEEDS),
        "per-seed case/lead row count differs",
    )
    _require(
        set(headline["split"].astype(str)) == {"operational_era_retrospective"}
        and set(seed_metrics["split"].astype(str)) == {"operational_era_retrospective"},
        "case metric evidence label differs",
    )
    _require(set(headline.method.astype(str)) == set(METHODS), "headline methods differ")
    _require(
        set(seed_metrics.method.astype(str)) == {capacity.BASE_CANDIDATE},
        "per-seed method differs",
    )
    seed_values = set(pd.to_numeric(seed_metrics.seed, errors="raise").astype(int))
    _require(seed_values == set(SEEDS), "per-seed identities differ")
    _require(
        set(headline.loc[headline.method.eq("raw_fuxi"), "seed"].astype(str))
        == {"not_applicable"}
        and set(
            headline.loc[headline.method.eq(capacity.BASE_CANDIDATE), "seed"].astype(str)
        )
        == {"mean_of_seed_metrics_42_43_44"},
        "headline seed-score labels differ",
    )
    for frame, label in ((headline, "headline"), (seed_metrics, "per-seed")):
        _require(
            set(pd.to_numeric(frame.lead_week, errors="raise").astype(int))
            == set(range(1, 7)),
            f"{label} lead identities differ",
        )
        _require(
            set(pd.to_numeric(frame.member_count, errors="raise").astype(int)) == {50},
            f"{label} member count differs",
        )
        _require(
            set(pd.to_numeric(frame.support_cells, errors="raise").astype(int)) == {171},
            f"{label} support count differs",
        )
        _require(
            set(frame.season.astype(str)).issubset({"DJF", "MAM", "JJA", "SON"})
            and set(frame.initialization_season.astype(str)).issubset(
                {"DJF", "MAM", "JJA", "SON"}
            ),
            f"{label} season labels differ",
        )

    initialization_strings = sorted(headline.init.astype(str).unique().tolist())
    _require(len(initialization_strings) == expected_total, "headline initialization count differs")
    initializations = np.asarray(initialization_strings, dtype="datetime64[D]")
    _require(
        np.unique(initializations).size == expected_total
        and np.all(np.diff(initializations) > np.timedelta64(0, "D")),
        "headline initialization dates are not unique/sorted",
    )
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    actual_year_counts = {
        year: int(np.count_nonzero(years == year)) for year in AUDIT_YEARS
    }
    _require(actual_year_counts == expected_year_counts, "headline year counts differ")
    _require(
        np.max(initializations + np.timedelta64(41, "D")) <= np.datetime64("2024-12-31"),
        "headline verification dates cross into 2025",
    )
    manifest_initializations = manifest.get("cases", {}).get(
        "evaluated_initializations", []
    )
    _require(
        manifest_initializations == initialization_strings,
        "manifest initialization identities differ from case metrics",
    )

    headline_keys = {
        (str(initialization), method, lead)
        for initialization in initialization_strings
        for method in METHODS
        for lead in range(1, 7)
    }
    actual_headline_keys = set(
        zip(
            headline.init.astype(str),
            headline.method.astype(str),
            pd.to_numeric(headline.lead_week).astype(int),
            strict=True,
        )
    )
    _require(actual_headline_keys == headline_keys, "headline row identities are incomplete")
    seed_keys = {
        (str(initialization), seed, lead)
        for initialization in initialization_strings
        for seed in SEEDS
        for lead in range(1, 7)
    }
    actual_seed_keys = set(
        zip(
            seed_metrics.init.astype(str),
            pd.to_numeric(seed_metrics.seed).astype(int),
            pd.to_numeric(seed_metrics.lead_week).astype(int),
            strict=True,
        )
    )
    _require(actual_seed_keys == seed_keys, "per-seed row identities are incomplete")

    expected_initialization_columns = {
        "initialization",
        "initialization_year",
        "verification_start",
        "verification_end",
        "retained",
    }
    _require(
        set(evaluated.columns) == expected_initialization_columns
        and len(evaluated) == expected_total,
        "evaluated-initialization receipt shape differs",
    )
    _require(
        evaluated.initialization.astype(str).tolist() == initialization_strings,
        "evaluated-initialization receipt identities differ",
    )
    evaluation_dates = pd.to_datetime(evaluated.initialization, format="%Y-%m-%d")
    expected_end = (evaluation_dates + pd.to_timedelta(41, unit="D")).dt.strftime(
        "%Y-%m-%d"
    )
    _require(
        np.array_equal(
            evaluated.verification_start.astype(str).to_numpy(),
            evaluated.initialization.astype(str).to_numpy(),
        )
        and np.array_equal(
            evaluated.verification_end.astype(str).to_numpy(), expected_end.to_numpy()
        )
        and evaluated.retained.astype(bool).all(),
        "evaluated-initialization verification windows differ",
    )

    required_scores = (
        *CORE_METRICS,
        "absolute_bias",
        "ensemble_spread",
        "ensemble_variance",
        "mean_squared_error",
        "spread_skill_ratio",
        "coverage_50",
        "coverage_80",
        "coverage_90",
    )
    _require_finite_columns(headline, required_scores, label="headline case metrics")
    _require_finite_columns(seed_metrics, required_scores, label="per-seed case metrics")
    for frame, label in ((headline, "headline"), (seed_metrics, "per-seed")):
        _require(
            np.all(frame[["crps", "rmse", "mae", "ensemble_variance", "mean_squared_error"]].to_numpy(dtype=float) >= 0.0)
            and np.all(np.abs(frame.acc.to_numpy(dtype=float)) <= 1.0 + 1.0e-12)
            and np.all(
                (frame[["coverage_50", "coverage_80", "coverage_90"]].to_numpy(dtype=float) >= 0.0)
                & (frame[["coverage_50", "coverage_80", "coverage_90"]].to_numpy(dtype=float) <= 1.0)
            ),
            f"{label} score domains differ",
        )

    pairing_columns = [
        "init",
        "lead_week",
        "year",
        "season",
        "initialization_season",
        "member_count",
        "support_cells",
        "scoring_weight_sum_km2_fraction",
    ]
    raw_identity = _semantic_sort(
        headline.loc[headline.method.eq("raw_fuxi"), pairing_columns],
        ("init", "lead_week"),
    )
    adapter_identity = _semantic_sort(
        headline.loc[headline.method.eq(capacity.BASE_CANDIDATE), pairing_columns],
        ("init", "lead_week"),
    )
    _require_frames_equivalent(
        adapter_identity,
        raw_identity,
        keys=("init", "lead_week"),
        label="raw/calibrated case-support pairing",
    )
    for seed in SEEDS:
        selected = seed_metrics.loc[pd.to_numeric(seed_metrics.seed).astype(int).eq(seed)]
        seed_identity = _semantic_sort(selected[pairing_columns], ("init", "lead_week"))
        _require_frames_equivalent(
            seed_identity,
            raw_identity,
            keys=("init", "lead_week"),
            label=f"raw/seed-{seed} case-support pairing",
        )

    adapter = headline.loc[headline.method.eq(capacity.BASE_CANDIDATE)].copy()
    numeric_columns = [
        column
        for column in seed_metrics.columns
        if column not in required_identifiers
        and pd.api.types.is_numeric_dtype(seed_metrics[column])
        and column in adapter.columns
    ]
    averaged = seed_metrics.groupby(["init", "lead_week"], as_index=False)[
        numeric_columns
    ].mean()
    averaged["ensemble_spread"] = np.sqrt(
        np.maximum(averaged.ensemble_variance.to_numpy(dtype=np.float64), 0.0)
    )
    averaged["spread_skill_ratio"] = np.sqrt(
        averaged.ensemble_variance.to_numpy(dtype=np.float64)
        / averaged.mean_squared_error.to_numpy(dtype=np.float64)
    )
    comparison = adapter[["init", "lead_week", *numeric_columns]].copy()
    _require_frames_equivalent(
        comparison,
        averaged,
        keys=("init", "lead_week"),
        label="headline arithmetic mean of seed scores",
    )
    return initializations


def _validate_rank_and_reliability_semantics(
    root: Path,
    headline: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> None:
    rank_histograms = _read_contract_csv(root / "metrics/rank_histograms.csv")
    seed_ranks = _read_contract_csv(root / "metrics/seed_rank_histograms.csv")
    expected_rank_keys = {
        (method, lead, rank)
        for method in METHODS
        for lead in range(1, 7)
        for rank in range(FORECAST_MEMBER_COUNT + 1)
    }
    actual_rank_keys = set(
        zip(
            rank_histograms.method.astype(str),
            pd.to_numeric(rank_histograms.lead_week).astype(int),
            pd.to_numeric(rank_histograms["rank"]).astype(int),
            strict=True,
        )
    )
    _require(actual_rank_keys == expected_rank_keys, "headline rank identities differ")
    expected_seed_rank_keys = {
        (seed, lead, rank)
        for seed in SEEDS
        for lead in range(1, 7)
        for rank in range(FORECAST_MEMBER_COUNT + 1)
    }
    actual_seed_rank_keys = set(
        zip(
            pd.to_numeric(seed_ranks.seed).astype(int),
            pd.to_numeric(seed_ranks.lead_week).astype(int),
            pd.to_numeric(seed_ranks["rank"]).astype(int),
            strict=True,
        )
    )
    _require(
        set(seed_ranks.method.astype(str)) == {capacity.BASE_CANDIDATE}
        and actual_seed_rank_keys == expected_seed_rank_keys,
        "per-seed rank identities differ",
    )
    _require_finite_columns(rank_histograms, ("count",), label="headline ranks")
    _require_finite_columns(seed_ranks, ("count",), label="per-seed ranks")
    _require(
        np.all(rank_histograms["count"].to_numpy(dtype=float) >= 0.0)
        and np.all(seed_ranks["count"].to_numpy(dtype=float) >= 0.0),
        "rank counts are negative",
    )
    expected_adapter_ranks = base.mean_seed_rank_histograms(
        seed_ranks.drop(columns=["seed"])
    )
    _require_frames_equivalent(
        rank_histograms.loc[rank_histograms.method.eq(capacity.BASE_CANDIDATE)],
        expected_adapter_ranks,
        keys=("method", "lead_week", "rank"),
        label="headline arithmetic mean of seed rank counts",
    )
    expected_weight = (
        headline.loc[headline.method.eq("raw_fuxi")]
        .groupby("lead_week")["scoring_weight_sum_km2_fraction"]
        .sum()
    )
    for frame, methods, label in (
        (rank_histograms, METHODS, "headline"),
        (seed_ranks, (capacity.BASE_CANDIDATE,), "per-seed"),
    ):
        grouping = ["method", "lead_week"] + (["seed"] if "seed" in frame.columns else [])
        totals = frame.groupby(grouping)["count"].sum().reset_index()
        for row in totals.itertuples(index=False):
            _require(
                math.isclose(
                    float(row.count),
                    float(expected_weight.loc[int(row.lead_week)]),
                    rel_tol=2.0e-10,
                    abs_tol=1.0e-5,
                ),
                f"{label} rank weights do not match case scoring weights",
            )

    reliability = _read_contract_csv(root / "metrics/reliability_bins.csv")
    seed_reliability = _read_contract_csv(root / "metrics/seed_reliability_bins.csv")
    thresholds = {float(value) for value in base.THRESHOLDS_MM_DAY}
    expected_reliability_keys = {
        (method, lead, threshold, bin_index)
        for method in METHODS
        for lead in range(1, 7)
        for threshold in thresholds
        for bin_index in range(10)
    }
    actual_reliability_keys = set(
        zip(
            reliability.method.astype(str),
            pd.to_numeric(reliability.lead_week).astype(int),
            pd.to_numeric(reliability.threshold_mm_day).astype(float),
            pd.to_numeric(reliability.probability_bin).astype(int),
            strict=True,
        )
    )
    _require(actual_reliability_keys == expected_reliability_keys, "reliability identities differ")
    expected_seed_reliability_keys = {
        (seed, lead, threshold, bin_index)
        for seed in SEEDS
        for lead in range(1, 7)
        for threshold in thresholds
        for bin_index in range(10)
    }
    actual_seed_reliability_keys = set(
        zip(
            pd.to_numeric(seed_reliability.seed).astype(int),
            pd.to_numeric(seed_reliability.lead_week).astype(int),
            pd.to_numeric(seed_reliability.threshold_mm_day).astype(float),
            pd.to_numeric(seed_reliability.probability_bin).astype(int),
            strict=True,
        )
    )
    _require(
        set(seed_reliability.method.astype(str)) == {capacity.BASE_CANDIDATE}
        and actual_seed_reliability_keys == expected_seed_reliability_keys,
        "per-seed reliability identities differ",
    )
    reliability_sums = (
        "cell_case_count",
        "area_weight_sum",
        "forecast_probability_weighted_sum",
        "observed_event_weighted_sum",
    )
    for frame, label in ((reliability, "headline"), (seed_reliability, "per-seed")):
        _require_finite_columns(frame, reliability_sums, label=f"{label} reliability")
        _require(
            np.all(frame[list(reliability_sums)].to_numpy(dtype=float) >= 0.0),
            f"{label} reliability contains negative sums",
        )
        positive = frame.area_weight_sum.to_numpy(dtype=float) > 0.0
        probability = frame.mean_forecast_probability.to_numpy(dtype=float)
        observed = frame.observed_frequency.to_numpy(dtype=float)
        _require(
            np.isfinite(probability[positive]).all()
            and np.isfinite(observed[positive]).all()
            and np.isnan(probability[~positive]).all()
            and np.isnan(observed[~positive]).all()
            and np.all((probability[positive] >= 0.0) & (probability[positive] <= 1.0))
            and np.all((observed[positive] >= 0.0) & (observed[positive] <= 1.0)),
            f"{label} reliability conditional finiteness differs",
        )
        _require(
            np.allclose(
                probability[positive],
                frame.forecast_probability_weighted_sum.to_numpy(dtype=float)[positive]
                / frame.area_weight_sum.to_numpy(dtype=float)[positive],
                rtol=2.0e-10,
                atol=2.0e-10,
            )
            and np.allclose(
                observed[positive],
                frame.observed_event_weighted_sum.to_numpy(dtype=float)[positive]
                / frame.area_weight_sum.to_numpy(dtype=float)[positive],
                rtol=2.0e-10,
                atol=2.0e-10,
            ),
            f"{label} reliability ratios differ from stored sums",
        )
    expected_adapter_reliability = base.mean_seed_reliability_bins(
        seed_reliability.drop(columns=["seed"])
    )
    _require_frames_equivalent(
        reliability.loc[reliability.method.eq(capacity.BASE_CANDIDATE)],
        expected_adapter_reliability,
        keys=("method", "lead_week", "threshold_mm_day", "probability_bin"),
        label="headline arithmetic mean of seed reliability bins",
    )


def validate_output_semantics(
    output: Path,
    mode: str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently regenerate table relationships before a Slurm receipt."""

    root = Path(output).resolve()
    _require(mode in {"smoke", "full"}, f"unknown semantic-audit mode: {mode}")
    if manifest is None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    smoke = mode == "smoke"
    _require(
        manifest.get("audit_contract_version") == AUDIT_CONTRACT_VERSION,
        "manifest audit-contract version differs",
    )
    headline = _read_contract_csv(root / "metrics/case_metrics.csv")
    seed_metrics = _read_contract_csv(root / "metrics/seed_case_metrics.csv")
    evaluated = _read_contract_csv(root / "evaluation/evaluated_initializations.csv")
    initializations = _validate_case_metric_semantics(
        headline, seed_metrics, evaluated, manifest, smoke=smoke
    )

    summary_input = headline.drop(columns=["initialization_season"])
    seed_summary_input = seed_metrics.drop(columns=["initialization_season"])
    expected_weekwise, expected_pooled, expected_seasonal, expected_reliability = (
        base.summarize_metrics(summary_input)
    )
    expected_weekwise["split"] = "operational_era_retrospective"
    expected_pooled.insert(0, "split", "operational_era_retrospective")
    expected_seasonal = base.add_seasonal_raw_comparisons(expected_seasonal)
    expected_seasonal.insert(0, "split", "operational_era_retrospective")
    expected_reliability["split"] = "operational_era_retrospective"
    actual_weekwise = _read_contract_csv(root / "metrics/weekwise_metrics.csv")
    actual_pooled = _read_contract_csv(root / "metrics/pooled_metrics.csv")
    actual_seasonal = _read_contract_csv(
        root / "metrics/verification_season_weekwise_metrics.csv"
    )
    actual_reliability = _read_contract_csv(
        root / "metrics/weekwise_reliability_summary.csv"
    )
    _require_frames_equivalent(
        actual_weekwise,
        expected_weekwise,
        keys=("method", "lead_week"),
        label="weekwise summary recomputation",
    )
    _require_frames_equivalent(
        actual_pooled,
        expected_pooled,
        keys=("method",),
        label="pooled summary recomputation",
    )
    _require_frames_equivalent(
        actual_seasonal,
        expected_seasonal,
        keys=("season", "method", "lead_week"),
        label="verification-season summary recomputation",
    )
    _require_frames_equivalent(
        actual_reliability,
        expected_reliability,
        keys=("method", "lead_week"),
        label="weekwise reliability summary recomputation",
    )

    expected_seed_weekwise = base.summarize_seed_metrics(
        seed_summary_input,
        expected_weekwise.loc[expected_weekwise.method.eq("raw_fuxi")],
    )
    expected_seed_weekwise["split"] = "operational_era_retrospective"
    actual_seed_weekwise = _read_contract_csv(
        root / "metrics/seed_weekwise_metrics.csv"
    )
    _require_frames_equivalent(
        actual_seed_weekwise,
        expected_seed_weekwise,
        keys=("seed", "lead_week"),
        label="per-seed weekwise summary recomputation",
    )
    expected_seed_variability = base.summarize_seed_variability(
        expected_seed_weekwise
    )
    _require_frames_equivalent(
        _read_contract_csv(root / "metrics/seed_variability_by_week.csv"),
        expected_seed_variability,
        keys=("method", "lead_week"),
        label="seed variability recomputation",
    )
    expected_yearwise, expected_year_weekwise = summarize_years(headline)
    _require_frames_equivalent(
        _read_contract_csv(root / "metrics/yearwise_pooled_metrics.csv"),
        expected_yearwise,
        keys=("year", "method"),
        label="yearwise pooled summary recomputation",
    )
    _require_frames_equivalent(
        _read_contract_csv(root / "metrics/year_weekwise_metrics.csv"),
        expected_year_weekwise,
        keys=("year", "method", "lead_week"),
        label="year-week summary recomputation",
    )

    draws = 200 if smoke else BOOTSTRAP_DRAWS
    bootstrap_plan = two_stage_year_block_indices(
        initializations,
        n_resamples=draws,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )
    expected_bootstrap_frames = [
        paired_two_stage_bootstrap(headline, initializations, bootstrap_plan)
    ]
    raw_rows = headline.loc[headline.method.eq("raw_fuxi")]
    numeric_seed = pd.to_numeric(seed_metrics.seed).astype(int)
    for seed in SEEDS:
        paired = pd.concat(
            (raw_rows, seed_metrics.loc[numeric_seed.eq(seed)]), ignore_index=True
        )
        expected_bootstrap_frames.append(
            paired_two_stage_bootstrap(
                paired,
                initializations,
                bootstrap_plan,
                optimization_seed=seed,
            )
        )
    expected_bootstrap = pd.concat(expected_bootstrap_frames, ignore_index=True)
    actual_bootstrap = _read_contract_csv(
        root / "metrics/paired_two_stage_bootstrap.csv"
    )
    _require(len(actual_bootstrap) == 4 * 7 * len(CORE_METRICS), "bootstrap row count differs")
    _require_finite_columns(
        actual_bootstrap,
        (
            "effect",
            "ci_lower_2p5",
            "ci_upper_97p5",
            "bootstrap_probability_effect_positive",
        ),
        label="paired bootstrap",
    )
    _require(
        np.all(
            actual_bootstrap.ci_lower_2p5.to_numpy(dtype=float)
            <= actual_bootstrap.ci_upper_97p5.to_numpy(dtype=float)
        )
        and np.all(
            (actual_bootstrap.bootstrap_probability_effect_positive.to_numpy(dtype=float) >= 0.0)
            & (actual_bootstrap.bootstrap_probability_effect_positive.to_numpy(dtype=float) <= 1.0)
        ),
        "bootstrap interval/probability domains differ",
    )
    _require_frames_equivalent(
        actual_bootstrap,
        expected_bootstrap,
        keys=("optimization_seed", "lead_scope", "metric"),
        label="paired two-stage bootstrap recomputation",
        atol=2.0e-9,
    )
    expected_design = _json_safe(bootstrap_diagnostics(bootstrap_plan))
    stored_design = json.loads(
        (root / "evaluation/bootstrap_design.json").read_text(encoding="utf-8")
    )
    _require(stored_design == expected_design, "bootstrap design receipt differs")
    _require(
        manifest.get("evaluation", {}).get("bootstrap") == expected_design,
        "manifest bootstrap design differs",
    )

    _validate_rank_and_reliability_semantics(root, headline, seed_metrics)
    return {
        "post_run_audit_version": POSTRUN_AUDIT_VERSION,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "mode": mode,
        "status": "passed",
        "evaluated_initializations": len(initializations),
        "headline_case_rows": len(headline),
        "per_seed_case_rows": len(seed_metrics),
        "bootstrap_rows_recomputed": len(actual_bootstrap),
        "summary_tables_recomputed": [
            "weekwise",
            "pooled",
            "verification_season_weekwise",
            "yearwise_pooled",
            "year_weekwise",
            "seed_weekwise",
            "seed_variability",
            "weekwise_reliability",
        ],
        "rank_and_reliability_identities_checked": True,
        "headline_seed_score_arithmetic_checked": True,
        "raw_calibrated_pairing_checked": True,
    }


def plot_weekwise_metrics(
    weekwise: pd.DataFrame,
    seed_weekwise: pd.DataFrame,
    output: Path,
    *,
    smoke: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    specs = (
        ("crps", "CRPS (mm day⁻¹)"),
        ("rmse", "RMSE (mm day⁻¹)"),
        ("acc", "ACC"),
        ("bias", "Signed bias (mm day⁻¹)"),
    )
    colors = {"raw_fuxi": "#4D4D4D", capacity.BASE_CANDIDATE: "#009E73"}
    for axis, (metric, ylabel) in zip(axes.ravel(), specs, strict=True):
        for method in METHODS:
            rows = weekwise.loc[weekwise.method.eq(method)].sort_values("lead_week")
            axis.plot(
                rows.lead_week,
                rows[metric],
                marker="o",
                linewidth=1.45,
                color=colors[method],
                label=str(rows.method_label.iloc[0]),
            )
            if method == capacity.BASE_CANDIDATE:
                seed_rows = seed_weekwise.loc[seed_weekwise.method.eq(method)]
                grouped = seed_rows.groupby("lead_week")[metric]
                lower = grouped.min().reindex(rows.lead_week).to_numpy()
                upper = grouped.max().reindex(rows.lead_week).to_numpy()
                axis.fill_between(
                    rows.lead_week,
                    lower,
                    upper,
                    color=colors[method],
                    alpha=0.14,
                    linewidth=0.0,
                )
        if metric == "bias":
            axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        ("SMOKE — " if smoke else "")
        + "2022–2024 operational-era audit · 50-member FuXi",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_pooled_bootstrap(bootstrap: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    selected = bootstrap.loc[
        bootstrap.lead_scope.eq("W1-W6")
        & bootstrap.optimization_seed.eq("mean_of_seed_scores")
    ].copy()
    labels = {
        "crps": "CRPS skill (%)",
        "rmse": "RMSE skill (%)",
        "mae": "MAE skill (%)",
        "acc": "ACC difference",
        "bias": "Bias difference",
    }
    figure, axes = plt.subplots(1, 5, figsize=(8.2, 2.2), constrained_layout=True)
    for axis, metric in zip(axes, CORE_METRICS, strict=True):
        row = selected.loc[selected.metric.eq(metric)].iloc[0]
        value = float(row.effect)
        lower = float(row.ci_lower_2p5)
        upper = float(row.ci_upper_97p5)
        axis.errorbar(
            [0],
            [value],
            yerr=[[value - lower], [upper - value]],
            fmt="o",
            color="#009E73",
            capsize=3,
        )
        axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        axis.set_xticks([])
        axis.set_ylabel(labels[metric], fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        ("SMOKE — " if smoke else "")
        + "Locked base_42k versus raw FuXi · paired two-stage 95% CIs",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_readme(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    operational: OperationalMembers,
    *,
    smoke: bool,
) -> str:
    status = "NON-SCIENTIFIC SMOKE" if smoke else EVIDENCE_LABEL.upper()
    lines = [
        "# FuXi 2022--2024 operational-era ensemble audit",
        "",
        f"Status: **{status}**",
        "",
        "This evaluation applies the already validation-selected 42,434-parameter "
        "adapter to the later all-season FuXi archive without retraining, fine-tuning, "
        "or model selection. The operational archive has 50 members; the 2002--2019 "
        "hindcast training/validation archive had 51.",
        "",
        f"The full design contains {sum(operational.eligible_counts.values())} eligible "
        "initializations (104 in 2022, 104 in 2023, 88 in 2024). Twelve late-2024 "
        "initializations are excluded because initialization + 41 days would enter "
        "2025. No 2025 forecast or observation store is opened.",
        "",
        "Headline adapter values are arithmetic means of scores from seeds 42/43/44. "
        "Parameters, adjustment fields, and predictions are not averaged.",
        "",
        "For every loaded case/lead/grid point, the archived ensemble mean is "
        "independently reproduced with the benchmark writer's float64 member reduction "
        "followed by float32 storage cast and must match bit for bit.",
        "",
        "## Pooled metrics",
        "",
        "| Method | CRPS | CRPSS vs raw | RMSE | MAE | Bias | ACC | Spread / pooled error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.crps:.4f} | {row.crpss_vs_raw:+.3f} | "
            f"{row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.4f} | {row.acc:.4f} | "
            f"{row.spread_skill_ratio:.3f} |"
        )
    if not bootstrap.empty:
        lines.extend(["", "## Paired uncertainty", ""])
        selected = bootstrap.loc[
            bootstrap.lead_scope.eq("W1-W6")
            & bootstrap.optimization_seed.eq("mean_of_seed_scores")
        ]
        for row in selected.itertuples(index=False):
            lines.append(
                f"- `{row.effect_name}`: {row.effect:+.3f} "
                f"(95% CI {row.ci_lower_2p5:+.3f} to {row.ci_upper_97p5:+.3f})."
            )
    lines.extend(
        [
            "",
        "The paired bootstrap first resamples the three years and then circular "
            "13-initialization blocks within every sampled year. Three year clusters "
            "provide limited information about interannual uncertainty, so these "
            "intervals should be interpreted cautiously.",
            "",
            "IMD 2023-10-12 has reduced coverage at five coastal cells (two missing). "
            "Evaluation therefore uses identical case/lead-specific area × weekly "
            "observation-coverage weights for raw and calibrated forecasts, while the "
            "model context remains on the frozen 171-cell training support.",
            "",
            "## Artifact map",
            "",
            "- `metrics/case_metrics.csv`: raw and mean-of-seed-score case/lead rows.",
            "- `metrics/seed_case_metrics.csv`: each locked checkpoint separately.",
            "- `metrics/weekwise_metrics.csv`, `pooled_metrics.csv`, and year/season tables.",
            "- `metrics/paired_two_stage_bootstrap.csv`: paired headline and per-seed CIs.",
            "- `metrics/rank_histograms.csv` and `reliability_bins.csv`: ensemble diagnostics.",
            "- `evaluation/input_content_receipts.json`: dtype/shape/content hashes for loaded arrays.",
            "- `evaluation/scoring_support.npz`: exact grid, support, and weights.",
            "- `manifest.json`: locked model receipt, data firewall, and artifact hashes.",
            "",
            "This is complementary to, not pooled with, the earlier 100-start "
            "deterministic/raw-identity audit. The 2025 final-control workflow remains sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    args: argparse.Namespace,
    output: Path,
    receipt: capacity_gate.CapacityReceipt | None = None,
) -> Mapping[str, Any]:
    started = time.monotonic()
    if receipt is None:
        receipt = validate_locked_model_receipt(Path(args.capacity_manifest))
    snapshot_hashes = source_snapshot(output)
    base.METHOD_LABELS[capacity.BASE_CANDIDATE] = "Locked base adapter (42,434 params)"
    base.PLOT_METHOD_LABELS[capacity.BASE_CANDIDATE] = "Locked base adapter"
    base.METHOD_COLORS[capacity.BASE_CANDIDATE] = "#009E73"
    base.METHOD_MARKERS[capacity.BASE_CANDIDATE] = "P"

    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise OperationalAuditError(f"canonical {EXPERIMENT} requires CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)

    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(receipt.manifest_path, inputs / "capacity_manifest.json")
    shutil.copy2(receipt.root / "selection.json", inputs / "capacity_selection.json")

    print("Verifying canonical hindcast cache and frozen normalization inputs...", flush=True)
    canonical_cache = base.load_member_cache(receipt.cache_path, allow_partial=False)
    _require(
        canonical_cache.members.shape[1] == HINDCAST_MEMBER_COUNT,
        "hindcast member-count contract changed",
    )
    _require(
        np.array_equal(canonical_cache.member_labels, np.arange(HINDCAST_MEMBER_COUNT)),
        "hindcast member labels changed",
    )
    print("Loading exact 2022/2023/2024 operational member stores...", flush=True)
    operational = load_operational_members(canonical_cache, smoke=args.smoke)
    print(
        f"Operational cases={len(operational.initializations)}; "
        f"members={operational.members.shape[1]}; max verification="
        f"{np.datetime_as_string(np.max(operational.initializations + np.timedelta64(41, 'D')), unit='D')}",
        flush=True,
    )
    print("Loading only 2002-2017 training IMD and 2022-2024 verification IMD...", flush=True)
    daily = load_daily_observations(operational.latitude, operational.longitude)
    targets = build_audit_targets(
        canonical_cache, operational, daily, receipt, smoke=args.smoke
    )

    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    weight_denominator = targets.weights.sum(
        axis=(-2, -1), keepdims=True, dtype=np.float64
    )
    normalized_weights = targets.weights / weight_denominator
    scoring_path = evaluation_dir / "scoring_support.npz"
    np.savez_compressed(
        scoring_path,
        latitude=operational.latitude.astype(np.float64),
        longitude=operational.longitude.astype(np.float64),
        frozen_observation_fraction=daily.observation_fraction.astype(np.float32),
        frozen_support_mask=(targets.frozen_spatial_weights > 0.0),
        frozen_scoring_weight_km2_fraction=targets.frozen_spatial_weights.astype(
            np.float64
        ),
        dynamic_support_mask=(targets.weights > 0.0),
        dynamic_scoring_weight_km2_fraction=targets.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights.astype(np.float64),
    )
    content_receipts = {
        "forecast_inputs": dict(operational.content_inventory),
        "observation_inputs": dict(daily.content_inventory),
        "derived_inputs": dict(targets.content_inventory),
    }
    write_json(evaluation_dir / "input_content_receipts.json", content_receipts)
    write_json(evaluation_dir / "normalization_receipt.json", targets.normalization_inventory)
    write_json(evaluation_dir / "lead_week_diagnostics.json", operational.lead_week_diagnostics)
    case_dates = pd.DataFrame(
        {
            "initialization": operational.initializations.astype("datetime64[D]").astype(str),
            "initialization_year": pd.DatetimeIndex(operational.initializations).year,
            "verification_start": operational.initializations.astype("datetime64[D]").astype(str),
            "verification_end": (
                operational.initializations + np.timedelta64(41, "D")
            ).astype("datetime64[D]").astype(str),
            "retained": True,
        }
    )
    case_dates.to_csv(evaluation_dir / "evaluated_initializations.csv", index=False)

    print("Evaluating raw 50-member FuXi on the exact audit cases...", flush=True)
    raw_metrics, raw_ranks = evaluate_ensemble_dynamic(
        "raw_fuxi",
        operational.members,
        targets.truth,
        targets.climatology,
        operational.initializations,
        targets.weights,
        chunk_size=args.evaluation_batch_size,
        rank_seed=BOOTSTRAP_SEED,
    )
    raw_reliability = reliability_bins_dynamic(
        "raw_fuxi",
        operational.members,
        targets.truth,
        targets.weights,
        chunk_size=args.evaluation_batch_size,
    )

    indices = np.arange(len(operational.initializations), dtype=np.int64)
    seed_metric_frames: list[pd.DataFrame] = []
    seed_rank_frames: list[pd.DataFrame] = []
    seed_reliability_frames: list[pd.DataFrame] = []
    checkpoint_receipts: list[dict[str, Any]] = []
    output_prediction_receipts: dict[str, Any] = {}
    for seed in SEEDS:
        record = receipt.checkpoint_records[(capacity.BASE_CANDIDATE, seed)]
        checkpoint = receipt.root / Path(str(record["checkpoint"]))
        checkpoint_hash = sha256_file(checkpoint)
        print(f"Applying locked base_42k seed {seed} to 50 members...", flush=True)
        model = capacity.load_checkpoint_model(
            checkpoint,
            capacity.CANDIDATE_BY_NAME[capacity.BASE_CANDIDATE],
            seed,
            device,
        )
        _require(
            sum(parameter.numel() for parameter in model.parameters()) == 42_434,
            "restored architecture parameter count changed",
        )
        delta, log_spread = base.predict_adjustments(
            model,
            operational.members,
            targets.truth,
            targets.context,
            indices,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=True,
        )
        _require(np.isfinite(delta).all() and np.isfinite(log_spread).all(), "non-finite model adjustment")
        spread = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
        corrected = base.apply_affine_log_calibration(
            operational.members, delta, spread
        )
        _require(corrected.shape == operational.members.shape, "calibrated member shape changed")
        adjustment_path = (
            output / "models" / capacity.BASE_CANDIDATE / f"seed_{seed}" / "audit_adjustments.npz"
        )
        adjustment_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            adjustment_path,
            initializations=operational.initializations,
            delta_log_location=delta,
            log_spread=log_spread,
            spread_factor=spread,
            source_checkpoint_sha256=checkpoint_hash,
            source_member_count=np.int64(FORECAST_MEMBER_COUNT),
            seed=np.int64(seed),
        )
        seed_metrics, seed_ranks = evaluate_ensemble_dynamic(
            capacity.BASE_CANDIDATE,
            corrected,
            targets.truth,
            targets.climatology,
            operational.initializations,
            targets.weights,
            chunk_size=args.evaluation_batch_size,
            rank_seed=seed,
            seed_label=seed,
        )
        seed_reliability = reliability_bins_dynamic(
            capacity.BASE_CANDIDATE,
            corrected,
            targets.truth,
            targets.weights,
            chunk_size=args.evaluation_batch_size,
        )
        seed_reliability.insert(2, "seed", seed)
        seed_metric_frames.append(seed_metrics)
        seed_rank_frames.append(seed_ranks.assign(seed=seed))
        seed_reliability_frames.append(seed_reliability)
        checkpoint_receipts.append(
            {
                "candidate": capacity.BASE_CANDIDATE,
                "seed": seed,
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "parameter_count": 42_434,
                "member_hidden_channels": 8,
                "backbone_channels": 24,
                "mode": "location_spread",
            }
        )
        output_prediction_receipts[f"seed_{seed}"] = {
            "delta_log_location": array_receipt(delta.astype("<f4", copy=False)),
            "log_spread": array_receipt(log_spread.astype("<f4", copy=False)),
            "corrected_members": array_receipt(corrected.astype("<f4", copy=False)),
        }
        del model, delta, log_spread, spread, corrected, seed_metrics, seed_ranks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_metrics_unlabelled = pd.concat(seed_metric_frames, ignore_index=True)
    headline_unlabelled = build_headline_case_metrics(
        raw_metrics, seed_metrics_unlabelled
    )
    headline = relabel_case_metrics(headline_unlabelled, operational.initializations)
    seed_case_metrics = relabel_case_metrics(
        seed_metrics_unlabelled, operational.initializations
    )
    summary_input = headline.drop(columns=["initialization_season"])
    seed_summary_input = seed_case_metrics.drop(columns=["initialization_season"])
    weekwise, pooled, seasonal, reliability_summary = base.summarize_metrics(
        summary_input
    )
    weekwise["split"] = "operational_era_retrospective"
    pooled.insert(0, "split", "operational_era_retrospective")
    seasonal = base.add_seasonal_raw_comparisons(seasonal)
    seasonal.insert(0, "split", "operational_era_retrospective")
    reliability_summary["split"] = "operational_era_retrospective"
    seed_weekwise = base.summarize_seed_metrics(
        seed_summary_input,
        weekwise.loc[weekwise.method.eq("raw_fuxi")],
    )
    seed_weekwise["split"] = "operational_era_retrospective"
    seed_variability = base.summarize_seed_variability(seed_weekwise)
    yearwise, year_weekwise = summarize_years(headline)

    seed_ranks = pd.concat(seed_rank_frames, ignore_index=True)
    mean_ranks = base.mean_seed_rank_histograms(seed_ranks.drop(columns=["seed"]))
    rank_histograms = pd.concat((raw_ranks, mean_ranks), ignore_index=True)
    seed_reliability = pd.concat(seed_reliability_frames, ignore_index=True)
    mean_reliability = base.mean_seed_reliability_bins(
        seed_reliability.drop(columns=["seed"])
    )
    reliability_bins = pd.concat((raw_reliability, mean_reliability), ignore_index=True)

    bootstrap_draws = args.bootstrap_draws
    plan = two_stage_year_block_indices(
        operational.initializations,
        n_resamples=bootstrap_draws,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )
    headline_bootstrap = paired_two_stage_bootstrap(
        headline,
        operational.initializations,
        plan,
    )
    per_seed_bootstrap = []
    for seed in SEEDS:
        paired = pd.concat(
            (
                raw_metrics,
                seed_metrics_unlabelled.loc[seed_metrics_unlabelled.seed.eq(seed)],
            ),
            ignore_index=True,
        )
        per_seed_bootstrap.append(
            paired_two_stage_bootstrap(
                paired,
                operational.initializations,
                plan,
                optimization_seed=seed,
            )
        )
    bootstrap = pd.concat((headline_bootstrap, *per_seed_bootstrap), ignore_index=True)
    diagnostics = bootstrap_diagnostics(plan)

    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(metrics_dir / "case_metrics.csv", index=False)
    seed_case_metrics.to_csv(metrics_dir / "seed_case_metrics.csv", index=False)
    weekwise.to_csv(metrics_dir / "weekwise_metrics.csv", index=False)
    pooled.to_csv(metrics_dir / "pooled_metrics.csv", index=False)
    seasonal.to_csv(metrics_dir / "verification_season_weekwise_metrics.csv", index=False)
    yearwise.to_csv(metrics_dir / "yearwise_pooled_metrics.csv", index=False)
    year_weekwise.to_csv(metrics_dir / "year_weekwise_metrics.csv", index=False)
    seed_weekwise.to_csv(metrics_dir / "seed_weekwise_metrics.csv", index=False)
    seed_variability.to_csv(metrics_dir / "seed_variability_by_week.csv", index=False)
    reliability_summary.to_csv(metrics_dir / "weekwise_reliability_summary.csv", index=False)
    rank_histograms.to_csv(metrics_dir / "rank_histograms.csv", index=False)
    seed_ranks.to_csv(metrics_dir / "seed_rank_histograms.csv", index=False)
    reliability_bins.to_csv(metrics_dir / "reliability_bins.csv", index=False)
    seed_reliability.to_csv(metrics_dir / "seed_reliability_bins.csv", index=False)
    bootstrap.to_csv(metrics_dir / "paired_two_stage_bootstrap.csv", index=False)
    base.write_metric_matrices(weekwise, metrics_dir / "matrices")
    write_json(evaluation_dir / "bootstrap_design.json", diagnostics)
    write_json(evaluation_dir / "output_prediction_receipts.json", output_prediction_receipts)

    figures = output / "figures"
    plot_weekwise_metrics(
        weekwise,
        seed_weekwise,
        figures / "weekwise_core_metrics",
        smoke=args.smoke,
    )
    plot_pooled_bootstrap(
        bootstrap,
        figures / "pooled_paired_effects",
        smoke=args.smoke,
    )
    (output / "README.md").write_text(
        build_readme(pooled, bootstrap, operational, smoke=args.smoke),
        encoding="utf-8",
    )

    opened_paths = [
        *operational.store_paths,
        *daily.training_stores,
        *daily.verification_stores,
        str(base.SPATIAL_STORE),
    ]
    _require(
        all(not any(part == "2025.zarr" for part in Path(path).parts) for path in opened_paths),
        "opened-store audit contains a forbidden 2025 year store",
    )
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "canonical": not args.smoke,
        "scientific_eligible": not args.smoke,
        "scientific_status": (
            "non-scientific plumbing smoke over fixed year-balanced subsets"
            if args.smoke
            else EVIDENCE_LABEL
        ),
        "evidence_label": EVIDENCE_LABEL,
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "workflow_can_train": False,
            "workflow_can_finetune": False,
            "workflow_can_select": False,
            "workflow_can_blend": False,
            "selection_data": "2018-2019 validation only, completed before this audit",
            "audit_metrics_used_for_selection": False,
            "audit_years": list(AUDIT_YEARS),
            "train_years": list(TRAIN_YEARS),
            "forecast_year_store_whitelist": list(AUDIT_YEARS),
            "verification_year_store_whitelist": list(AUDIT_YEARS),
            "year_store_discovery_or_globbing_used": False,
            "final_2025_store_opened": False,
            "sealed_2025_target_opened": False,
            "maximum_verification_date": np.datetime_as_string(
                np.max(operational.initializations + np.timedelta64(41, "D")),
                unit="D",
            ),
            "late_2024_rule": "retain only initialization + 41 days <= 2024-12-31",
            "all_eligible_initializations_retained_in_full": not args.smoke,
            "arbitrary_scientific_subsample": False,
            "region": "39N-0N, 60E-99E, 27x27 India box",
            "target": "IMD weekly mean precipitation, mm day-1",
            "lead_week_definition": "six successive complete 7-day blocks spanning initialization offsets 0-41",
            "stored_ensemble_mean_gate": (
                "np.nanmean(member_axis,dtype=float64).astype(float32) must match "
                "the stored float32 product bitwise"
            ),
        },
        "relationship_to_prior_evidence": {
            "existing_audit": "frozen 100-start 2022-2024 deterministic/raw-identity matched audit",
            "relationship": "complementary; metrics are not pooled",
            "differences": [
                "all eligible all-season initializations rather than a selected 100-start set",
                "individual 50-member distributions rather than ensemble summaries",
                "capacity-selected base_42k lineage rather than raw-identity deterministic lineage",
                "probabilistic CRPS/coverage/rank diagnostics in addition to deterministic metrics",
            ],
            "untouched_final_test_claim": False,
        },
        "capacity_receipt": {
            "path": str(receipt.manifest_path),
            "sha256": receipt.manifest_sha256,
            "expected_sha256": EXPECTED_CAPACITY_MANIFEST_SHA256,
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "artifact_inventory_verified": True,
            "selected_candidate": capacity.BASE_CANDIDATE,
            "selected_parameter_count": 42_434,
        },
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "seed_handling": {
            "parameters_averaged": False,
            "adjustment_fields_averaged": False,
            "predictions_averaged": False,
            "headline_aggregation": "arithmetic mean of per-seed scores on identical case/lead rows",
            "per_seed_scores_retained": True,
        },
        "member_count_shift": {
            "training_validation_hindcast_members": HINDCAST_MEMBER_COUNT,
            "operational_audit_members": FORECAST_MEMBER_COUNT,
            "adapter_accepts_variable_member_set_size": True,
            "member_count_domain_shift_explicit": True,
        },
        "input_checkpoints": checkpoint_receipts,
        "cases": {
            "stored_counts": dict(operational.stored_counts),
            "full_eligible_counts": dict(operational.eligible_counts),
            "evaluated_counts": dict(operational.evaluated_counts),
            "evaluated_total": len(operational.initializations),
            "excluded_late_2024_count": len(operational.excluded_2024_initializations),
            "excluded_late_2024_initializations": list(operational.excluded_2024_initializations),
            "evaluated_initializations": operational.initializations.astype(
                "datetime64[D]"
            ).astype(str).tolist(),
        },
        "evaluation": {
            "metrics": list(CORE_METRICS),
            "calibration_diagnostics": [
                "coverage_50",
                "coverage_80",
                "coverage_90",
                "spread_skill_ratio",
                "rank_histogram",
                "threshold_reliability",
            ],
            "raw_and_calibrated_case_member_support_identity": True,
            "frozen_model_context_support_cells": int(
                np.count_nonzero(targets.frozen_spatial_weights > 0.0)
            ),
            "dynamic_scoring_support_cells_minimum": int(
                np.count_nonzero(targets.weights > 0.0, axis=(-2, -1)).min()
            ),
            "dynamic_scoring_support_cells_maximum": int(
                np.count_nonzero(targets.weights > 0.0, axis=(-2, -1)).max()
            ),
            "dynamic_to_frozen_weight_ratio_minimum": float(
                np.min(
                    targets.weights[..., targets.frozen_spatial_weights > 0.0]
                    / targets.frozen_spatial_weights[
                        targets.frozen_spatial_weights > 0.0
                    ]
                )
            ),
            "dynamic_to_frozen_weight_ratio_maximum": float(
                np.max(
                    targets.weights[..., targets.frozen_spatial_weights > 0.0]
                    / targets.frozen_spatial_weights[
                        targets.frozen_spatial_weights > 0.0
                    ]
                )
            ),
            "dynamic_observation_coverage_amendment": {
                "frozen_before_forecast_scores": True,
                **dict(daily.coverage_deviation_receipt),
                **dict(targets.coverage_contract),
                "weekly_truth": "daily observation-fraction-weighted mean",
                "scoring_weight": "cell area x seven-day mean observation fraction",
                "raw_and_calibrated_weights_identical": True,
            },
            "scoring_support_sha256": sha256_file(scoring_path),
            "verification_season_definition": "season of initialization + lead midpoint offset (3,10,17,24,31,38 days)",
            "bootstrap": diagnostics,
            "bootstrap_full_draws": BOOTSTRAP_DRAWS,
        },
        "normalization": dict(targets.normalization_inventory),
        "opened_store_paths_exact": opened_paths,
        "forecast_store_paths": list(operational.store_paths),
        "training_imd_store_paths": list(daily.training_stores),
        "verification_imd_store_paths": list(daily.verification_stores),
        "content_receipt_artifact": "evaluation/input_content_receipts.json",
        "lead_week_diagnostics": dict(operational.lead_week_diagnostics),
        "cache": base.cache_provenance(canonical_cache),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "xarray": xr.__version__,
            "zarr": importlib.metadata.version("zarr"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot_hashes,
    }
    manifest["artifact_sha256"] = output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked base_42k adapter on whitelisted 2022-2024 operational members."
    )
    parser.add_argument(
        "--capacity-manifest", type=Path, default=DEFAULT_CAPACITY_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_draws is None:
        args.bootstrap_draws = 200 if args.smoke else BOOTSTRAP_DRAWS
    fixed = {
        "batch_size": (args.batch_size, 8),
        "evaluation_batch_size": (args.evaluation_batch_size, 8),
        "bootstrap_draws": (
            args.bootstrap_draws,
            200 if args.smoke else BOOTSTRAP_DRAWS,
        ),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical audit settings differ: {mismatch}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if not args.preflight_only and args.device != "cuda":
        raise ValueError("canonical operational-era evaluation requires --device cuda")


def preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    receipt = validate_locked_model_receipt(Path(args.capacity_manifest))
    forecast_paths = operational_store_paths()
    training_paths = observation_store_paths(TRAIN_YEARS)
    verification_paths = observation_store_paths(AUDIT_YEARS)
    exact_paths = (*forecast_paths, *training_paths, *verification_paths)
    _require(
        all(not any(part == "2025.zarr" for part in path.parts) for path in exact_paths),
        "preflight exact-path whitelist contains 2025",
    )
    return {
        "experiment": EXPERIMENT,
        "status": "preflight_passed",
        "capacity_manifest_sha256": receipt.manifest_sha256,
        "selected_candidate": receipt.selected_candidate,
        "seeds": list(SEEDS),
        "forecast_year_stores": [str(path) for path in forecast_paths],
        "training_imd_years": list(TRAIN_YEARS),
        "verification_imd_years": list(AUDIT_YEARS),
        "final_2025_store_opened": False,
        "workflow_can_train": False,
        "workflow_can_select": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.preflight_only:
        print(json.dumps(_json_safe(preflight(args)), indent=2, sort_keys=True))
        return 0
    receipt = validate_locked_model_receipt(Path(args.capacity_manifest))
    requested_output = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(f"refusing to overwrite output: {requested_output}")
    staging = requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging, receipt)
        os.replace(staging, requested_output)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "capacity_manifest": str(receipt.manifest_path),
                "capacity_manifest_sha256": receipt.manifest_sha256,
                "requested_output": str(requested_output),
                "final_2025_store_opened": False,
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} operational-era audit at {requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
