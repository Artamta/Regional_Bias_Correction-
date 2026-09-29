#!/usr/bin/env python3
"""Independent prediction-level audit of the probabilistic FuXi bridge.

This executable intentionally does not import either bridge scoring module.  It
reopens the frozen 2020--2024 FuXi and IMD stores, reconstructs the two
calibrated 50-member ensembles from compact inference shards, independently
forms +1..+42 weekly targets and climatologies, and reproduces every saved
case metric.  The 2025 observation is never opened.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import india_s2s_probabilistic_bridge as bridge
from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_probabilistic_bridge_prediction_audit_v1"
INFERENCE_EXPERIMENT = "india_s2s_probabilistic_bridge_inference_v1"
SCORING_EXPERIMENT = "india_s2s_probabilistic_bridge_scoring_v1"
PREFLIGHT_EXPERIMENT = "india_s2s_probabilistic_bridge_v1"
PARENT_EXPERIMENT = "fuxi_allseason_ensemble_calibration_v2_aligned"
CONTRACT_REVISION = "imd_end_labelled_init_plus_1_through_42_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_probabilistic_bridge"
SOURCE_PATH = Path(__file__).resolve()

YEARS = (2020, 2021, 2022, 2023, 2024)
FORECAST_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 100}
SCORING_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 88}
METHODS = ("raw_fuxi", "moment_calibration", "location_spread")
REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
LEADS = (1, 2, 3, 4, 5, 6)
TARGET_OFFSETS = tuple(range(1, 43))
LAST_ALLOWED_TARGET = np.datetime64("2024-12-31", "D")
FORECAST_DATES_SHA256 = (
    "e95a7f88fdc0ca13b775e1c8708b510310e4a647ff2040e099f65db71baeef56"
)
SCORING_DATES_SHA256 = (
    "41b67539c1d9196dfe5b598e0613a80a794dfea9da6ecbaae40cc102ebd66062"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a9a71465e2399773d5968bc94e4a5d535826b9ec5d1c4d3ec555fd8449586fee"
)
EXPECTED_MEMBER_COUNT = 50
EXPECTED_CASE_ROWS = 45_450
FLOAT_METRICS = (
    "crps",
    "acc",
    "rmse",
    "mae",
    "bias",
    "effective_area_km2",
    "coverage90",
    "ensemble_variance",
    "mean_squared_error",
    "ensemble_spread",
    "spread_skill_ratio",
)
KEY_COLUMNS = ("method", "init", "region", "lead_week")
_RENAME_NOREPLACE = 1


class PredictionAuditError(RuntimeError):
    """Raised when prediction-level reproduction violates a frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    """Hash logical arrays including dtype and shape, independent of containers."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"|")
        # ``memoryview.cast`` rejects NumPy datetime dtypes even though their
        # contiguous bytes are well-defined.  A uint8 view handles numeric and
        # datetime arrays identically without changing the logical hash.
        digest.update(memoryview(array.view(np.uint8)))
    return digest.hexdigest()


def _raw_content_update(digest: Any, values: np.ndarray) -> None:
    digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionAuditError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(str(relative))
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run: {relative}")
    return resolved


@dataclass
class HashLedger:
    """Files that must retain identical bytes for the entire audit."""

    expected: dict[Path, str] = field(default_factory=dict)
    labels: dict[Path, str] = field(default_factory=dict)

    def verify(self, path: Path, expected: Any, label: str) -> None:
        resolved = Path(path).resolve()
        _require(_valid_sha256(expected), f"{label} has an invalid SHA-256")
        _require(resolved.is_file(), f"{label} is missing: {resolved}")
        expected_text = str(expected)
        previous = self.expected.get(resolved)
        _require(previous in (None, expected_text), f"conflicting hash for {resolved}")
        observed = sha256_file(resolved)
        _require(observed == expected_text, f"{label} SHA-256 differs")
        self.expected[resolved] = expected_text
        self.labels[resolved] = label

    def reverify(self) -> None:
        for path, expected in self.expected.items():
            _require(path.is_file(), f"audited input disappeared: {path}")
            _require(
                sha256_file(path) == expected,
                f"audited input changed during audit: {self.labels[path]}",
            )


@dataclass(frozen=True)
class Contracts:
    inference_path: Path
    scoring_path: Path
    parent_path: Path
    preflight_path: Path
    inference: Mapping[str, Any]
    scoring: Mapping[str, Any]
    parent: Mapping[str, Any]
    preflight: Mapping[str, Any]
    shard_arrays: Mapping[int, Mapping[str, np.ndarray]]
    starts: np.ndarray
    score_starts: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    moment_delta: np.ndarray
    moment_spread: np.ndarray


@dataclass(frozen=True)
class WeeklyTruth:
    values: np.ndarray
    normal: np.ndarray
    support: np.ndarray
    target_dates: np.ndarray
    region_weights: Mapping[str, np.ndarray]
    source_receipt: Mapping[str, Any]


def initialization_dates_sha256(values: Iterable[Any]) -> str:
    dates = np.asarray(values, dtype="datetime64[D]")
    _require(dates.ndim == 1 and dates.size > 0, "dates must be non-empty and 1-D")
    _require(not np.isnat(dates).any(), "dates contain NaT")
    _require(np.unique(dates).size == dates.size, "dates are duplicated")
    _require(np.all(np.diff(dates) > np.timedelta64(0, "D")), "dates are unsorted")
    text = "".join(f"{np.datetime_as_string(date, unit='D')}\n" for date in dates)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def target_dates(initializations: Iterable[Any]) -> np.ndarray:
    starts = np.asarray(initializations, dtype="datetime64[D]")
    initialization_dates_sha256(starts)
    offsets = np.asarray(TARGET_OFFSETS, dtype="timedelta64[D]").reshape(1, 6, 7)
    return starts[:, None, None] + offsets


def _fixed_day(values: Iterable[Any]) -> np.ndarray:
    dates = pd.DatetimeIndex(values)
    _require(not dates.hasnans, "climatology labels contain NaT")
    return np.asarray(
        [pd.Timestamp(2000, value.month, value.day).dayofyear for value in dates],
        dtype=np.int16,
    )


def _artifact_path(manifest_path: Path, manifest: Mapping[str, Any], relative: str) -> Path:
    hashes = manifest.get("artifact_sha256", {})
    _require(relative in hashes, f"manifest does not bind {relative}")
    return _resolve_child(manifest_path.parent, relative)


def _verify_selected_parent(
    parent_path: Path, parent: Mapping[str, Any], ledger: HashLedger
) -> tuple[np.ndarray, np.ndarray]:
    _require(parent.get("experiment") == PARENT_EXPERIMENT, "parent experiment differs")
    _require(parent.get("status") == "complete", "parent is not complete")
    contract = parent.get("contract", {})
    _require(contract.get("revision") == CONTRACT_REVISION, "parent alignment differs")
    _require(contract.get("target_day_offsets") == list(TARGET_OFFSETS), "parent offsets differ")
    _require(contract.get("initialization_day_included_in_target") is False, "parent uses init day")
    _require(contract.get("sealed_2025_target_opened") is False, "parent opened 2025")
    selection = parent.get("deployment_selection", {})
    _require(selection.get("selected_seed") == 43, "parent did not select seed 43")
    _require(selection.get("parameter_averaging") is False, "parent averaged parameters")
    _require(selection.get("prediction_averaging") is False, "parent averaged predictions")
    _require(
        selection.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "parent checkpoint differs",
    )
    checkpoint_relative = str(selection.get("checkpoint", ""))
    checkpoint = _artifact_path(parent_path, parent, checkpoint_relative)
    ledger.verify(
        checkpoint,
        parent["artifact_sha256"][checkpoint_relative],
        "selected seed-43 checkpoint",
    )
    moment_relative = "models/moment_calibration_fit.npz"
    moment_path = _artifact_path(parent_path, parent, moment_relative)
    ledger.verify(moment_path, parent["artifact_sha256"][moment_relative], "moment fit")
    with np.load(moment_path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == {"delta_log_location", "spread_factor", "shrinkage"},
            "moment fit keys differ",
        )
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        shrinkage = float(archive["shrinkage"])
    _require(delta.shape == (6, 12, 27, 27), "moment delta shape differs")
    _require(spread.shape == (6, 12), "moment spread shape differs")
    _require(np.isfinite(delta).all() and np.isfinite(spread).all(), "moment fit is non-finite")
    _require(np.all((spread >= 0.25) & (spread <= 4.0)), "moment spread bounds differ")
    _require(shrinkage == 10.0, "moment shrinkage differs")
    _require(
        parent.get("split_counts_selected", {}).get("train") == 1652,
        "moment fit is not bound to 1,652 training starts",
    )
    retained = parent.get("retained_initializations", {})
    train = np.asarray(retained.get("train", []), dtype="datetime64[D]")
    _require(train.shape == (1652,), "parent training cohort differs")
    _require(train.max() + np.timedelta64(42, "D") == np.datetime64("2017-12-29"), "training purge differs")
    return delta, spread


def _validate_shard(
    year: int,
    inference_path: Path,
    inference: Mapping[str, Any],
    receipt: Mapping[str, Any],
    latitude: np.ndarray,
    longitude: np.ndarray,
    ledger: HashLedger,
) -> Mapping[str, np.ndarray]:
    _require(int(receipt.get("year", -1)) == year, f"{year} shard year differs")
    _require(receipt.get("selected_seed") == 43, f"{year} shard seed differs")
    _require(receipt.get("member_count") == 50, f"{year} shard member count differs")
    _require(receipt.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, f"{year} checkpoint differs")
    _require(receipt.get("verification_truth_opened") is False, f"{year} inference opened truth")
    _require(receipt.get("sealed_2025_target_opened") is False, f"{year} inference opened 2025")
    relative = str(receipt.get("artifact", ""))
    path = _artifact_path(inference_path, inference, relative)
    _require(inference["artifact_sha256"][relative] == receipt.get("artifact_sha256"), f"{year} shard hash bindings differ")
    ledger.verify(path, receipt.get("artifact_sha256"), f"{year} compact prediction shard")
    receipt_relative = f"inference/adjustments_{year}.receipt.json"
    receipt_path = _artifact_path(inference_path, inference, receipt_relative)
    ledger.verify(receipt_path, inference["artifact_sha256"][receipt_relative], f"{year} shard receipt")
    _require(_read_json(receipt_path) == receipt, f"{year} embedded/on-disk receipts differ")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            "initializations",
            "scoreable_for_truth",
            "delta_log_location",
            "log_spread",
            "raw_ensemble_mean",
            "corrected_ensemble_mean",
            "latitude",
            "longitude",
            "member_count",
            "selected_seed",
        }
        _require(set(archive.files) == expected_keys, f"{year} shard keys differ")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    starts = np.asarray(arrays["initializations"], dtype="datetime64[D]")
    count = FORECAST_COUNTS[year]
    _require(starts.shape == (count,), f"{year} forecast count differs")
    _require(set(pd.DatetimeIndex(starts).year) == {year}, f"{year} shard crosses years")
    _require(np.array_equal(arrays["latitude"], latitude), f"{year} latitude differs")
    _require(np.array_equal(arrays["longitude"], longitude), f"{year} longitude differs")
    _require(int(arrays["member_count"]) == 50, f"{year} stored members differ")
    _require(int(arrays["selected_seed"]) == 43, f"{year} stored seed differs")
    expected_scoreable = starts + np.timedelta64(42, "D") <= LAST_ALLOWED_TARGET
    _require(np.array_equal(arrays["scoreable_for_truth"], expected_scoreable), f"{year} scoreable mask differs")
    _require(int(expected_scoreable.sum()) == SCORING_COUNTS[year], f"{year} scoring count differs")
    expected_shape = (count, 6, 27, 27)
    for name in ("delta_log_location", "log_spread", "raw_ensemble_mean", "corrected_ensemble_mean"):
        value = arrays[name]
        _require(value.shape == expected_shape and value.dtype == np.float32, f"{year} {name} shape/dtype differs")
        _require(np.isfinite(value).all(), f"{year} {name} is non-finite")
    _require(np.max(np.abs(arrays["log_spread"])) <= 2.00001, f"{year} log spread differs")
    raw_source = receipt.get("raw_source", {})
    _require(raw_source.get("year") == year, f"{year} raw source year differs")
    _require(raw_source.get("selection_indices") == list(range(count)), f"{year} selection is not complete")
    _require(raw_source.get("raw_member_shape") == [count, 50, 6, 27, 27], f"{year} raw shape receipt differs")
    _require(
        raw_source.get("selected_initialization_dates_sha256")
        == initialization_dates_sha256(starts),
        f"{year} initialization receipt differs",
    )
    for key, label in (("manifest", "forecast manifest"), ("store", "forecast store")):
        value = Path(str(raw_source.get(key, ""))).resolve()
        _require("2025" not in value.name, f"{year} {label} names 2025")
    forecast_manifest = Path(str(raw_source["manifest"]))
    ledger.verify(forecast_manifest, raw_source.get("manifest_sha256"), f"{year} forecast source manifest")
    store = Path(str(raw_source["store"]))
    _require(store.name == f"{year}.zarr", f"{year} forecast store differs")
    ledger.verify(store / ".zmetadata", raw_source.get("zmetadata_sha256"), f"{year} forecast metadata")
    return arrays


def load_contracts(
    *,
    inference_manifest: Path,
    scoring_manifest: Path,
    parent_manifest: Path,
    preflight_manifest: Path,
    ledger: HashLedger,
) -> Contracts:
    """Validate all four frozen manifests and every directly consumed artifact."""

    inference_path = Path(inference_manifest).resolve()
    scoring_path = Path(scoring_manifest).resolve()
    parent_path = Path(parent_manifest).resolve()
    preflight_path = Path(preflight_manifest).resolve()
    inference = _read_json(inference_path)
    scoring = _read_json(scoring_path)
    parent = _read_json(parent_path)
    preflight = _read_json(preflight_path)

    _require(inference.get("experiment") == INFERENCE_EXPERIMENT, "inference experiment differs")
    _require(inference.get("status") == "complete" and inference.get("mode") == "full", "inference is not full/complete")
    _require(inference.get("verification_truth_opened") is False, "inference opened verification truth")
    _require(inference.get("sealed_2025_target_opened") is False, "inference opened 2025")
    _require(scoring.get("experiment") == SCORING_EXPERIMENT and scoring.get("status") == "complete", "scoring is not complete")
    _require(preflight.get("experiment") == PREFLIGHT_EXPERIMENT and preflight.get("status") == "preflight_complete", "preflight differs")
    _require(scoring.get("opened_observation_years") == list(YEARS), "scoring observation years differ")
    _require(scoring.get("opened_2025_observation") is False, "scoring opened 2025")
    _require(scoring.get("sealed_2025_target_opened") is False, "scoring opened sealed truth")
    _require(scoring.get("member_count_each_method") == 50, "scoring members differ")
    _require(scoring.get("methods") == list(METHODS), "scoring methods differ")

    inference_hash = sha256_file(inference_path)
    parent_hash = sha256_file(parent_path)
    preflight_hash = sha256_file(preflight_path)
    _require(Path(str(scoring.get("input_inference_manifest"))).resolve() == inference_path, "scoring points to another inference")
    _require(scoring.get("input_inference_manifest_sha256") == inference_hash, "scoring inference hash differs")
    _require(Path(str(scoring.get("parent_manifest"))).resolve() == parent_path, "scoring points to another parent")
    _require(scoring.get("parent_manifest_sha256") == parent_hash, "scoring parent hash differs")
    _require(Path(str(scoring.get("input_preflight_manifest"))).resolve() == preflight_path, "scoring points to another preflight")
    _require(scoring.get("input_preflight_manifest_sha256") == preflight_hash, "scoring preflight hash differs")
    parent_receipt = inference.get("parent_receipt", {})
    _require(Path(str(parent_receipt.get("parent_manifest"))).resolve() == parent_path, "inference parent path differs")
    _require(parent_receipt.get("parent_manifest_sha256") == parent_hash, "inference parent hash differs")
    _require(parent_receipt.get("selected_seed") == 43, "inference receipt seed differs")
    _require(parent_receipt.get("selected_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "inference receipt checkpoint differs")
    _require(parent_receipt.get("target_day_offsets") == list(TARGET_OFFSETS), "inference receipt offsets differ")
    _require(parent_receipt.get("sealed_2025_target_opened") is False, "inference receipt opened 2025")
    _require(preflight.get("parent_receipt") == parent_receipt, "preflight/inference parent receipts differ")
    _require(scoring.get("preflight_parent_receipt") == parent_receipt, "scoring parent receipt differs")
    _require(scoring.get("selected_seed") == 43, "scoring seed differs")
    _require(scoring.get("selected_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "scoring checkpoint differs")

    moment_delta, moment_spread = _verify_selected_parent(parent_path, parent, ledger)
    moment_path = _resolve_child(parent_path.parent, "models/moment_calibration_fit.npz")
    _require(Path(str(scoring.get("moment_fit"))).resolve() == moment_path, "scoring moment path differs")
    _require(scoring.get("moment_fit_sha256") == sha256_file(moment_path), "scoring moment hash differs")

    for manifest_path, manifest, label in (
        (inference_path, inference, "inference"),
        (scoring_path, scoring, "scoring"),
        (preflight_path, preflight, "preflight"),
    ):
        hashes = manifest.get("artifact_sha256", {})
        _require(isinstance(hashes, dict) and hashes, f"{label} artifact hashes absent")
        for relative, expected in hashes.items():
            ledger.verify(_resolve_child(manifest_path.parent, relative), expected, f"{label} artifact {relative}")
        for relative, expected in manifest.get("source_snapshot_sha256", {}).items():
            _require(hashes.get(relative) == expected, f"{label} source is not artifact-bound")

    context_relative = "context/training_context.npz"
    context_path = _artifact_path(inference_path, inference, context_relative)
    with np.load(context_path, allow_pickle=False) as archive:
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
    _require(latitude.shape == (27,) and longitude.shape == (27,), "context grid differs")

    receipts = inference.get("inference_shards", [])
    _require(isinstance(receipts, list) and len(receipts) == 5, "inference shards differ")
    by_year = {int(item.get("year", -1)): item for item in receipts}
    _require(tuple(sorted(by_year)) == YEARS, "inference years differ")
    shards = {
        year: _validate_shard(
            year, inference_path, inference, by_year[year], latitude, longitude, ledger
        )
        for year in YEARS
    }
    starts = np.concatenate([shards[year]["initializations"] for year in YEARS]).astype("datetime64[D]")
    _require(starts.size == 517, "forecast cohort does not contain 517 starts")
    _require(initialization_dates_sha256(starts) == FORECAST_DATES_SHA256, "forecast cohort hash differs")
    mask = starts + np.timedelta64(42, "D") <= LAST_ALLOWED_TARGET
    score_starts = starts[mask]
    _require(score_starts.size == 505, "scoring cohort does not contain 505 starts")
    _require(initialization_dates_sha256(score_starts) == SCORING_DATES_SHA256, "scoring cohort hash differs")
    _require(target_dates(score_starts).max() == np.datetime64("2024-12-30"), "maximum target differs")

    table_relative = "tables/case_metrics.csv"
    _artifact_path(scoring_path, scoring, table_relative)
    _require(scoring.get("table_rows", {}).get("case_metrics") == EXPECTED_CASE_ROWS, "saved case-row count differs")
    truth_access = scoring.get("truth_access", {})
    _require(truth_access.get("opened_years") == list(YEARS), "truth receipt years differ")
    _require(truth_access.get("opened_2025") is False, "truth receipt opened 2025")
    _require(truth_access.get("maximum_target_label") == "2024-12-30", "truth maximum label differs")
    _require(truth_access.get("target_day_offsets") == list(TARGET_OFFSETS), "truth offsets differ")

    return Contracts(
        inference_path=inference_path,
        scoring_path=scoring_path,
        parent_path=parent_path,
        preflight_path=preflight_path,
        inference=inference,
        scoring=scoring,
        parent=parent,
        preflight=preflight,
        shard_arrays=shards,
        starts=starts,
        score_starts=score_starts,
        latitude=latitude,
        longitude=longitude,
        moment_delta=moment_delta,
        moment_spread=moment_spread,
    )


def reconstruct_affine_members(
    raw_members: np.ndarray,
    delta_log_location: np.ndarray,
    spread_factor: np.ndarray,
) -> np.ndarray:
    """Apply the independently implemented rank-preserving log-space affine map."""

    raw = np.asarray(raw_members, dtype=np.float32)
    delta = np.asarray(delta_log_location, dtype=np.float32)
    spread = np.asarray(spread_factor, dtype=np.float32)
    _require(raw.ndim == 5 and raw.shape[1:] == (50, 6, 27, 27), "raw member shape differs")
    _require(delta.shape == (raw.shape[0], 6, 27, 27), "affine delta shape differs")
    _require(spread.shape in {(raw.shape[0], 6), delta.shape}, "affine spread shape differs")
    _require(np.isfinite(raw).all() and np.all(raw >= 0.0), "raw members are invalid")
    transformed = np.log1p(raw).astype(np.float32)
    centre = transformed.mean(axis=1, dtype=np.float64).astype(np.float32)
    multiplier = spread[:, None]
    if spread.ndim == 2:
        multiplier = multiplier[..., None, None]
    corrected_log = centre[:, None] + delta[:, None] + multiplier * (
        transformed - centre[:, None]
    )
    corrected = np.expm1(np.clip(corrected_log, 0.0, 20.0)).astype(np.float32)
    _require(np.isfinite(corrected).all() and np.all(corrected >= 0.0), "reconstruction is invalid")
    return corrected


def reconstruct_neural_members(
    raw_members: np.ndarray, delta: np.ndarray, log_spread: np.ndarray
) -> np.ndarray:
    spread = np.exp(np.clip(np.asarray(log_spread, dtype=np.float32), -2.0, 2.0)).astype(np.float32)
    return reconstruct_affine_members(raw_members, delta, spread)


def reconstruct_moment_members(
    raw_members: np.ndarray,
    initializations: np.ndarray,
    delta_fit: np.ndarray,
    spread_fit: np.ndarray,
) -> np.ndarray:
    starts = np.asarray(initializations, dtype="datetime64[D]")
    midpoint_offsets = np.asarray((4, 11, 18, 25, 32, 39), dtype="timedelta64[D]")
    midpoint_dates = starts[:, None] + midpoint_offsets[None]
    months = pd.DatetimeIndex(midpoint_dates.reshape(-1)).month.to_numpy().reshape(len(starts), 6)
    delta = np.empty((len(starts), 6, 27, 27), dtype=np.float32)
    spread = np.empty((len(starts), 6), dtype=np.float32)
    for lead_index in range(6):
        delta[:, lead_index] = delta_fit[lead_index, months[:, lead_index] - 1]
        spread[:, lead_index] = spread_fit[lead_index, months[:, lead_index] - 1]
    return reconstruct_affine_members(raw_members, delta, spread)


def _open_forecast(loader: Any, year: int) -> Any:
    _require(year in YEARS, "forecast request escapes 2020--2024")
    return loader.open_forecast_dataset(
        model="fuxi_s2s",
        variable="tp",
        year=year,
        grid="common_1p5",
        experiment_id=bridge.FUXI_EXPERIMENT_ID,
    )


def _raw_receipt_by_year(contracts: Contracts) -> dict[int, Mapping[str, Any]]:
    return {
        int(item["year"]): item["raw_source"]
        for item in contracts.inference["inference_shards"]
    }


def build_forecast_means(
    contracts: Contracts, loader: Any, *, batch_size: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Reopen all 517 forecasts, validate compact predictions, and retain means."""

    _require(batch_size > 0, "batch size must be positive")
    by_method = {
        method: np.empty((517, 6, 27, 27), dtype=np.float32) for method in METHODS
    }
    receipts = _raw_receipt_by_year(contracts)
    source_rows: list[dict[str, Any]] = []
    offset = 0
    for year in YEARS:
        shard = contracts.shard_arrays[year]
        starts = np.asarray(shard["initializations"], dtype="datetime64[D]")
        count = len(starts)
        digest = hashlib.sha256()
        dataset = _open_forecast(loader, year)
        try:
            _require(np.array_equal(np.asarray(dataset.init.values, dtype="datetime64[D]"), starts), f"{year} live initializations differ")
            _require(np.array_equal(np.asarray(dataset.latitude.values), contracts.latitude), f"{year} live latitude differs")
            _require(np.array_equal(np.asarray(dataset.longitude.values), contracts.longitude), f"{year} live longitude differs")
            variable = dataset["forecast_weekly_mean"]
            _require(variable.dims == ("init", "member", "lead_week", "latitude", "longitude"), f"{year} raw dimensions differ")
            _require(variable.shape == (count, 50, 6, 27, 27), f"{year} live member shape differs")
            for begin in range(0, count, batch_size):
                end = min(begin + batch_size, count)
                raw = np.asarray(variable.isel(init=slice(begin, end)).load().values, dtype=np.float32)
                _raw_content_update(digest, raw)
                neural = reconstruct_neural_members(
                    raw,
                    shard["delta_log_location"][begin:end],
                    shard["log_spread"][begin:end],
                )
                moment = reconstruct_moment_members(
                    raw,
                    starts[begin:end],
                    contracts.moment_delta,
                    contracts.moment_spread,
                )
                raw_mean = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
                neural_mean = neural.mean(axis=1, dtype=np.float64).astype(np.float32)
                moment_mean = moment.mean(axis=1, dtype=np.float64).astype(np.float32)
                _require(np.array_equal(raw_mean, shard["raw_ensemble_mean"][begin:end]), f"{year} compact raw means differ")
                _require(np.array_equal(neural_mean, shard["corrected_ensemble_mean"][begin:end]), f"{year} compact neural means differ")
                slc = slice(offset + begin, offset + end)
                by_method["raw_fuxi"][slc] = raw_mean
                by_method["location_spread"][slc] = neural_mean
                by_method["moment_calibration"][slc] = moment_mean
        finally:
            dataset.close()
        observed = digest.hexdigest()
        expected = str(receipts[year]["raw_weekly_members_sha256"])
        _require(observed == expected, f"{year} raw member content hash differs")
        source_rows.append(
            {
                "year": year,
                "forecast_count": count,
                "member_count": 50,
                "raw_weekly_members_expected_sha256": expected,
                "raw_weekly_members_climatology_pass_sha256": observed,
                "raw_weekly_members_scoring_pass_sha256": None,
            }
        )
        offset += count
    _require(offset == 517, "forecast mean pass did not process 517 starts")
    return by_method, source_rows


def _circular_year_mean(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    days = _fixed_day(starts).astype(np.int64)
    _require(np.unique(days).size == len(days), "forecast year duplicates a calendar day")
    daily = np.zeros((366,) + values.shape[1:], dtype=np.float64)
    counts = np.zeros(366, dtype=np.int16)
    daily[days - 1] = np.asarray(values, dtype=np.float64)
    counts[days - 1] = 1
    extended = np.concatenate((daily[-15:], daily, daily[:15]), axis=0)
    extended_counts = np.concatenate((counts[-15:], counts, counts[:15]))
    cumulative = np.concatenate(
        (np.zeros((1,) + daily.shape[1:], dtype=np.float64), np.cumsum(extended, axis=0)),
        axis=0,
    )
    count_cumulative = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(extended_counts, dtype=np.int32)))
    total = cumulative[31:] - cumulative[:-31]
    count = count_cumulative[31:] - count_cumulative[:-31]
    _require(total.shape[0] == 366 and np.all(count > 0), "forecast climatology window differs")
    return (total / count.reshape((366,) + (1,) * (values.ndim - 1))).astype(np.float32)


def build_loyo_climatologies(
    starts: np.ndarray, means: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    years = pd.DatetimeIndex(starts).year.to_numpy()
    result: dict[str, np.ndarray] = {}
    for method in METHODS:
        yearly = np.stack(
            [_circular_year_mean(means[method][years == year], starts[years == year]) for year in YEARS]
        ).astype(np.float32)
        total = yearly.sum(axis=0, dtype=np.float64)
        loyo = np.empty_like(yearly)
        for index in range(5):
            loyo[index] = ((total - yearly[index].astype(np.float64)) / 4.0).astype(np.float32)
        result[method] = loyo
    return result


def _source_metadata_path(store: Path) -> Path:
    return Path(store) / ".zmetadata"


def load_weekly_truth(
    contracts: Contracts, loader: Any, ledger: HashLedger
) -> WeeklyTruth:
    """Open only IMD 2020--2024 and independently construct exact weekly truth."""

    expected_receipt = contracts.scoring["truth_access"]
    expected_daily = {int(item["year"]): item for item in expected_receipt["daily_sources"]}
    _require(tuple(sorted(expected_daily)) == YEARS, "IMD receipt years differ")
    daily_dates: list[np.ndarray] = []
    daily_values: list[np.ndarray] = []
    daily_fraction: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    for year in YEARS:
        _require(year != 2025, "2025 truth access is forbidden")
        records = loader.list_observations(source="imd", variable="tp", year=year)
        _require(len(records) == 1, f"IMD {year} catalog record differs")
        record = records.iloc[0].to_dict()
        store = Path(str(record["store"])).resolve()
        expected = expected_daily[year]
        _require(store == Path(str(expected["store"])).resolve(), f"IMD {year} store differs")
        _require(store.name == f"{year}.zarr" and year in YEARS, f"IMD {year} escapes firewall")
        ledger.verify(_source_metadata_path(store), expected["zmetadata_sha256"], f"IMD {year} metadata")
        dataset = loader.open_observation_dataset(source="imd", variable="tp", year=year)
        try:
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            fraction = np.asarray(dataset.observation_fraction.load().values, dtype=np.float32)
            _require(np.array_equal(np.asarray(dataset.latitude.values), contracts.latitude), f"IMD {year} latitude differs")
            _require(np.array_equal(np.asarray(dataset.longitude.values), contracts.longitude), f"IMD {year} longitude differs")
        finally:
            dataset.close()
        _require(dates.min() == np.datetime64(f"{year}-01-01") and dates.max() == np.datetime64(f"{year}-12-31"), f"IMD {year} calendar differs")
        if fraction.ndim == 2:
            fraction = np.broadcast_to(fraction, values.shape).copy()
        _require(fraction.shape == values.shape, f"IMD {year} fractions differ")
        _require(np.isfinite(fraction).all() and np.all((fraction >= 0.0) & (fraction <= 1.0)), f"IMD {year} support invalid")
        daily_dates.append(dates)
        daily_values.append(values)
        daily_fraction.append(fraction)
        source_rows.append(
            {
                "year": year,
                "store": str(store),
                "zmetadata_sha256": expected["zmetadata_sha256"],
                "logical_dates_observation_fraction_sha256": _array_sha256(dates.astype("datetime64[D]"), values, fraction),
            }
        )

    climate_expected = expected_receipt["climatology_source"]
    climate_store = Path(str(climate_expected["store"])).resolve()
    _require("2025" not in climate_store.name, "climatology path names 2025")
    ledger.verify(_source_metadata_path(climate_store), climate_expected["zmetadata_sha256"], "IMD 1991--2019 climatology metadata")
    climate = loader.open_observation_climatology_dataset(source="imd", baseline=(1991, 2019))
    try:
        daily_normal = np.asarray(climate.climatology_mean.load().values, dtype=np.float32)
        _require(np.array_equal(np.asarray(climate.latitude.values), contracts.latitude), "IMD normal latitude differs")
        _require(np.array_equal(np.asarray(climate.longitude.values), contracts.longitude), "IMD normal longitude differs")
    finally:
        climate.close()
    _require(daily_normal.shape == (366, 27, 27), "IMD normal shape differs")

    support_expected = Path(str(expected_receipt["spatial_support_store"])).resolve()
    ledger.verify(
        _source_metadata_path(support_expected),
        expected_receipt["spatial_support_zmetadata_sha256"],
        "spatial support metadata",
    )
    support_dataset = loader.open_spatial_support()
    try:
        india = np.asarray(support_dataset.india_area_weight_km2.load().values, dtype=np.float64)
        cell = np.asarray(support_dataset.cell_area_km2.load().values, dtype=np.float64)
        fractions = {
            region: np.asarray(support_dataset[f"{region}_fraction"].load().values, dtype=np.float64)
            for region in REGIONS[1:]
        }
        _require(np.array_equal(np.asarray(support_dataset.latitude.values), contracts.latitude), "support latitude differs")
        _require(np.array_equal(np.asarray(support_dataset.longitude.values), contracts.longitude), "support longitude differs")
    finally:
        support_dataset.close()
    region_weights = {"all_india": india}
    region_weights.update({region: cell * fractions[region] for region in REGIONS[1:]})
    # The geometric India mask has 174 positive cells.  IMD availability is a
    # separate dynamic factor and reduces scored support to 171 (occasionally
    # 169) cells; conflating those masks would incorrectly alter the weights.
    _require(np.count_nonzero(india > 0.0) == 174, "All-India geometry differs")

    dates = np.concatenate(daily_dates)
    values = np.concatenate(daily_values)
    fraction = np.concatenate(daily_fraction)
    targets = target_dates(contracts.score_starts)
    _require(targets.min() == contracts.score_starts.min() + np.timedelta64(1, "D"), "W1 includes init day")
    _require(targets.max() == np.datetime64("2024-12-30"), "truth request crosses firewall")
    positions = pd.Index(dates).get_indexer(targets.reshape(-1))
    _require(np.all(positions >= 0), "a weekly IMD target date is missing")
    shape = (505, 6, 7, 27, 27)
    selected_values = values[positions].reshape(shape).astype(np.float64, copy=False)
    selected_fraction = fraction[positions].reshape(shape).astype(np.float64, copy=False)
    denominator = selected_fraction.sum(axis=2, dtype=np.float64)
    numerator = (np.nan_to_num(selected_values, nan=0.0) * selected_fraction).sum(axis=2, dtype=np.float64)
    truth = np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0.0)
    weekly_support = selected_fraction.min(axis=2)
    supported_counts = np.sum(
        weekly_support * india[None, None] > 0.0, axis=(-2, -1), dtype=np.int64
    )
    _require(
        set(np.unique(supported_counts)).issubset({169, 171}),
        "dynamic All-India observation support differs",
    )
    normal_days = _fixed_day(targets.reshape(-1)).astype(np.int64) - 1
    selected_normal = daily_normal[normal_days].reshape(shape).astype(np.float64, copy=False)
    finite_count = np.isfinite(selected_normal).sum(axis=2)
    weekly_normal = np.divide(
        np.nansum(selected_normal, axis=2, dtype=np.float64),
        finite_count,
        out=np.full(finite_count.shape, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    truth = truth.astype(np.float32)
    weekly_support = weekly_support.astype(np.float32)
    weekly_normal = weekly_normal.astype(np.float32)
    receipt = {
        "opened_years": list(YEARS),
        "opened_2025": False,
        "sealed_2025_target_opened": False,
        "target_day_offsets": list(TARGET_OFFSETS),
        "minimum_target_label": np.datetime_as_string(targets.min(), unit="D"),
        "maximum_target_label": np.datetime_as_string(targets.max(), unit="D"),
        "scoreable_case_count": 505,
        "daily_sources": source_rows,
        "climatology_source": {
            "store": str(climate_store),
            "zmetadata_sha256": climate_expected["zmetadata_sha256"],
            "logical_climatology_sha256": _array_sha256(daily_normal),
        },
        "spatial_support_source": {
            "store": str(support_expected),
            "zmetadata_sha256": expected_receipt["spatial_support_zmetadata_sha256"],
            "logical_support_sha256": _array_sha256(india, cell, *[fractions[name] for name in REGIONS[1:]]),
        },
        "weekly_truth_sha256": _array_sha256(truth),
        "weekly_normal_sha256": _array_sha256(weekly_normal),
        "weekly_observation_support_sha256": _array_sha256(weekly_support),
        "target_dates_sha256": _array_sha256(targets),
    }
    return WeeklyTruth(truth, weekly_normal, weekly_support, targets, region_weights, receipt)


def empirical_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    ensemble = np.moveaxis(np.asarray(members, dtype=np.float64), 1, 0)
    target = np.asarray(truth, dtype=np.float64)
    _require(ensemble.shape[0] == 50 and ensemble.shape[1:] == target.shape, "CRPS dimensions differ")
    first = np.mean(np.abs(ensemble - target[None]), axis=0, dtype=np.float64)
    ordered = np.sort(ensemble, axis=0)
    coefficients = (2.0 * np.arange(50) - 49.0).reshape((50,) + (1,) * target.ndim)
    dispersion = np.sum(ordered * coefficients, axis=0, dtype=np.float64) / 2500.0
    return first - dispersion


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    spatial = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    represented = spatial > 0.0
    _require(np.all(np.sum(spatial, axis=(-2, -1)) > 0.0), "metric has no spatial support")
    _require(np.isfinite(values[represented]).all(), "metric is non-finite on support")
    safe = np.where(represented, values, 0.0)
    return np.sum(safe * spatial, axis=(-2, -1), dtype=np.float64) / np.sum(spatial, axis=(-2, -1), dtype=np.float64)


def weighted_acc(forecast: np.ndarray, observation: np.ndarray, weights: np.ndarray) -> np.ndarray:
    first = np.asarray(forecast, dtype=np.float64)
    second = np.asarray(observation, dtype=np.float64)
    spatial = np.broadcast_to(np.asarray(weights, dtype=np.float64), first.shape)
    represented = spatial > 0.0
    total = np.sum(spatial, axis=(-2, -1), dtype=np.float64)
    first_mean = np.sum(np.where(represented, first, 0.0) * spatial, axis=(-2, -1)) / total
    second_mean = np.sum(np.where(represented, second, 0.0) * spatial, axis=(-2, -1)) / total
    first_centered = np.where(represented, first - first_mean[..., None, None], 0.0)
    second_centered = np.where(represented, second - second_mean[..., None, None], 0.0)
    covariance = np.sum(spatial * first_centered * second_centered, axis=(-2, -1), dtype=np.float64)
    first_variance = np.sum(spatial * first_centered**2, axis=(-2, -1), dtype=np.float64)
    second_variance = np.sum(spatial * second_centered**2, axis=(-2, -1), dtype=np.float64)
    denominator = np.sqrt(first_variance * second_variance)
    return np.divide(covariance, denominator, out=np.full(covariance.shape, np.nan), where=denominator > 0.0)


def _select_loyo(climatology: np.ndarray, initializations: np.ndarray) -> np.ndarray:
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    year_index = np.asarray([YEARS.index(int(year)) for year in years], dtype=np.int64)
    day_index = _fixed_day(initializations).astype(np.int64) - 1
    return climatology[year_index, day_index]


def score_member_batch(
    *,
    method: str,
    starts: np.ndarray,
    members: np.ndarray,
    truth: np.ndarray,
    observation_normal: np.ndarray,
    observation_support: np.ndarray,
    forecast_normal: np.ndarray,
    region_weights: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Independently calculate all deterministic and probabilistic case metrics."""

    ensemble = np.asarray(members, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    support = np.asarray(observation_support, dtype=np.float64)
    _require(ensemble.shape == (len(starts), 50, 6, 27, 27), "scoring members differ")
    point_crps = empirical_crps(ensemble, target)
    ensemble_mean = ensemble.mean(axis=1, dtype=np.float64)
    forecast_anomaly = ensemble_mean - np.asarray(forecast_normal)
    observation_anomaly = target - np.asarray(observation_normal)
    lower, upper = np.quantile(ensemble, (0.05, 0.95), axis=1, method="linear")
    covered = ((target >= lower) & (target <= upper)).astype(np.float64)
    variance_field = ensemble.var(axis=1, ddof=0, dtype=np.float64)
    squared_error = (ensemble_mean - target) ** 2
    rows: list[dict[str, Any]] = []
    for region in REGIONS:
        dynamic = support * np.asarray(region_weights[region], dtype=np.float64)
        error = np.where(dynamic > 0.0, ensemble_mean - target, 0.0)
        rmse = np.sqrt(weighted_mean(error**2, dynamic))
        mae = weighted_mean(np.abs(error), dynamic)
        bias = weighted_mean(error, dynamic)
        crps = weighted_mean(point_crps, dynamic)
        acc = weighted_acc(forecast_anomaly, observation_anomaly, dynamic)
        coverage = weighted_mean(covered, dynamic)
        variance = weighted_mean(variance_field, dynamic)
        mse = weighted_mean(squared_error, dynamic)
        spread = np.sqrt(np.maximum(variance, 0.0))
        ratio = np.divide(spread, np.sqrt(mse), out=np.full(spread.shape, np.nan), where=mse > 0.0)
        valid_count = np.sum(dynamic > 0.0, axis=(-2, -1), dtype=np.int64)
        effective_area = np.sum(dynamic, axis=(-2, -1), dtype=np.float64)
        for case_index, start in enumerate(starts):
            for lead_index, lead in enumerate(LEADS):
                rows.append(
                    {
                        "method": method,
                        "init": np.datetime_as_string(start, unit="D"),
                        "region": region,
                        "lead_week": lead,
                        "crps": float(crps[case_index, lead_index]),
                        "acc": float(acc[case_index, lead_index]),
                        "rmse": float(rmse[case_index, lead_index]),
                        "mae": float(mae[case_index, lead_index]),
                        "bias": float(bias[case_index, lead_index]),
                        "valid_cell_count": int(valid_count[case_index, lead_index]),
                        "effective_area_km2": float(effective_area[case_index, lead_index]),
                        "coverage90": float(coverage[case_index, lead_index]),
                        "ensemble_variance": float(variance[case_index, lead_index]),
                        "mean_squared_error": float(mse[case_index, lead_index]),
                        "ensemble_spread": float(spread[case_index, lead_index]),
                        "spread_skill_ratio": float(ratio[case_index, lead_index]),
                    }
                )
    return rows


def reproduce_case_metrics(
    contracts: Contracts,
    loader: Any,
    truth: WeeklyTruth,
    climatologies: Mapping[str, np.ndarray],
    forecast_sources: list[dict[str, Any]],
    *,
    batch_size: int,
) -> pd.DataFrame:
    truth_lookup = {date: index for index, date in enumerate(contracts.score_starts)}
    source_lookup = {int(item["year"]): item for item in forecast_sources}
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        shard = contracts.shard_arrays[year]
        starts = np.asarray(shard["initializations"], dtype="datetime64[D]")
        scoreable = np.asarray(shard["scoreable_for_truth"], dtype=bool)
        digest = hashlib.sha256()
        dataset = _open_forecast(loader, year)
        try:
            variable = dataset["forecast_weekly_mean"]
            for begin in range(0, len(starts), batch_size):
                end = min(begin + batch_size, len(starts))
                raw_all = np.asarray(variable.isel(init=slice(begin, end)).load().values, dtype=np.float32)
                _raw_content_update(digest, raw_all)
                local = np.flatnonzero(scoreable[begin:end])
                if not local.size:
                    continue
                selected = begin + local
                batch_starts = starts[selected]
                truth_indices = np.asarray([truth_lookup[value] for value in batch_starts], dtype=np.int64)
                raw = raw_all[local]
                members = {
                    "raw_fuxi": raw,
                    "moment_calibration": reconstruct_moment_members(
                        raw, batch_starts, contracts.moment_delta, contracts.moment_spread
                    ),
                    "location_spread": reconstruct_neural_members(
                        raw,
                        shard["delta_log_location"][selected],
                        shard["log_spread"][selected],
                    ),
                }
                for method in METHODS:
                    rows.extend(
                        score_member_batch(
                            method=method,
                            starts=batch_starts,
                            members=members[method],
                            truth=truth.values[truth_indices],
                            observation_normal=truth.normal[truth_indices],
                            observation_support=truth.support[truth_indices],
                            forecast_normal=_select_loyo(climatologies[method], batch_starts),
                            region_weights=truth.region_weights,
                        )
                    )
        finally:
            dataset.close()
        observed = digest.hexdigest()
        expected = str(_raw_receipt_by_year(contracts)[year]["raw_weekly_members_sha256"])
        _require(observed == expected, f"{year} raw content changed between passes")
        source_lookup[year]["raw_weekly_members_scoring_pass_sha256"] = observed
    frame = pd.DataFrame(rows)
    _require(len(frame) == EXPECTED_CASE_ROWS, f"reproduced {len(frame)} rows, expected {EXPECTED_CASE_ROWS}")
    _require(not frame.duplicated(list(KEY_COLUMNS)).any(), "reproduced case keys are duplicated")
    _require(frame.groupby("method").size().to_dict() == {method: 15_150 for method in METHODS}, "reproduced method counts differ")
    return frame


def compare_case_metrics(
    reproduced: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require(len(saved) == EXPECTED_CASE_ROWS, "saved case table row count differs")
    required = {*KEY_COLUMNS, *FLOAT_METRICS, "valid_cell_count"}
    _require(required.issubset(saved.columns), "saved case table lacks required metrics")
    _require(not saved.duplicated(list(KEY_COLUMNS)).any(), "saved case keys are duplicated")
    left = reproduced.set_index(list(KEY_COLUMNS)).sort_index()
    right = saved.set_index(list(KEY_COLUMNS)).sort_index()
    _require(left.index.equals(right.index), "reproduced and saved case IDs differ")
    summaries: list[dict[str, Any]] = []
    overall_failures = 0
    for metric in (*FLOAT_METRICS, "valid_cell_count"):
        first = left[metric].to_numpy(dtype=np.float64)
        second = right[metric].to_numpy(dtype=np.float64)
        both_nan = np.isnan(first) & np.isnan(second)
        comparable = np.isfinite(first) & np.isfinite(second)
        invalid = ~(both_nan | comparable)
        if metric == "valid_cell_count":
            close = first == second
            metric_atol, metric_rtol = 0.0, 0.0
        elif metric == "effective_area_km2":
            metric_atol, metric_rtol = 1.0e-6, 1.0e-12
            close = np.isclose(first, second, atol=metric_atol, rtol=metric_rtol, equal_nan=True)
        else:
            metric_atol, metric_rtol = atol, rtol
            close = np.isclose(first, second, atol=metric_atol, rtol=metric_rtol, equal_nan=True)
        failures = int(np.count_nonzero(invalid | ~close))
        overall_failures += failures
        maximum = float(np.max(np.abs(first[comparable] - second[comparable]))) if np.any(comparable) else 0.0
        summaries.append(
            {
                "metric": metric,
                "row_count": len(first),
                "both_nan_count": int(np.count_nonzero(both_nan)),
                "failure_count": failures,
                "max_absolute_difference": maximum,
                "atol": metric_atol,
                "rtol": metric_rtol,
            }
        )
    comparison = pd.DataFrame(summaries)
    return comparison, {
        "passed": overall_failures == 0,
        "matched_case_rows": len(left),
        "failure_count": overall_failures,
        "max_absolute_difference_by_metric": {
            row["metric"]: row["max_absolute_difference"] for row in summaries
        },
        "failure_count_by_metric": {
            row["metric"]: row["failure_count"] for row in summaries
        },
        "atol": atol,
        "rtol": rtol,
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "audit_receipt.json" and ".tmp" not in path.name
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing a race winner."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "publication directories are not siblings")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "staging output is not a directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        _require(renameat2 is not None, "Linux renameat2 is required")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
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


def run_prediction_audit(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output).resolve()
    root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(root in output.parents, "audit output must remain under the resultsv3 bridge root")
    _require(not output.exists(), f"audit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    _require(not staging.exists(), "staging directory already exists")
    staging.mkdir()
    try:
        ledger = HashLedger()
        source_hash = sha256_file(SOURCE_PATH)
        source_copy = staging / "code/src" / SOURCE_PATH.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_PATH, source_copy)
        _require(sha256_file(source_copy) == source_hash, "auditor source snapshot differs")

        contracts = load_contracts(
            inference_manifest=args.inference_manifest,
            scoring_manifest=args.scoring_manifest,
            parent_manifest=args.parent_manifest,
            preflight_manifest=args.preflight_manifest,
            ledger=ledger,
        )
        loader = bridge.build_benchmark_loader()
        means, forecast_sources = build_forecast_means(
            contracts, loader, batch_size=args.batch_size
        )
        climatologies = build_loyo_climatologies(contracts.starts, means)
        truth = load_weekly_truth(contracts, loader, ledger)
        reproduced = reproduce_case_metrics(
            contracts,
            loader,
            truth,
            climatologies,
            forecast_sources,
            batch_size=args.batch_size,
        )
        saved_path = _artifact_path(
            contracts.scoring_path, contracts.scoring, "tables/case_metrics.csv"
        )
        saved = pd.read_csv(saved_path)
        comparison, result = compare_case_metrics(
            reproduced, saved, atol=args.atol, rtol=args.rtol
        )
        _require(result["passed"], f"prediction reproduction differs: {result['failure_count_by_metric']}")
        ledger.reverify()
        _require(sha256_file(SOURCE_PATH) == source_hash, "auditor source changed during run")

        tables = staging / "tables"
        receipts = staging / "receipts"
        tables.mkdir()
        receipts.mkdir()
        normalized = reproduced.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        normalized.to_csv(tables / "recomputed_case_metrics.csv", index=False)
        comparison.to_csv(tables / "metric_comparison.csv", index=False)
        _write_json(receipts / "truth_sources.json", truth.source_receipt)
        _write_json(
            receipts / "forecast_sources.json",
            {
                "forecast_only_case_count": 517,
                "scoreable_case_count": 505,
                "member_count": 50,
                "selected_seed": 43,
                "parameter_averaging": False,
                "prediction_averaging": False,
                "years": forecast_sources,
                "sealed_2025_target_opened": False,
            },
        )
        receipt = {
            "experiment": EXPERIMENT,
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output),
            "independence": {
                "imports_bridge_score_module": False,
                "imports_benchmark_scoring_module": False,
                "reopened_raw_members": True,
                "reopened_imd_daily_truth": True,
                "reimplemented_member_reconstruction": True,
                "reimplemented_target_alignment": True,
                "reimplemented_metric_equations": True,
            },
            "inputs": {
                "inference_manifest": str(contracts.inference_path),
                "inference_manifest_sha256": sha256_file(contracts.inference_path),
                "scoring_manifest": str(contracts.scoring_path),
                "scoring_manifest_sha256": sha256_file(contracts.scoring_path),
                "parent_manifest": str(contracts.parent_path),
                "parent_manifest_sha256": sha256_file(contracts.parent_path),
                "preflight_manifest": str(contracts.preflight_path),
                "preflight_manifest_sha256": sha256_file(contracts.preflight_path),
                "saved_case_metrics": str(saved_path),
                "saved_case_metrics_sha256": sha256_file(saved_path),
            },
            "contract": {
                "forecast_years": list(YEARS),
                "opened_observation_years": list(YEARS),
                "opened_2025_observation": False,
                "sealed_2025_target_opened": False,
                "target_day_offsets": list(TARGET_OFFSETS),
                "maximum_target_label_opened": "2024-12-30",
                "forecast_only_case_count": 517,
                "scoreable_case_count": 505,
                "method_count": 3,
                "region_count": 5,
                "lead_count": 6,
                "member_count_each_method": 50,
                "selected_seed": 43,
                "selected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "seed_or_prediction_averaging": False,
            },
            "reproduction": result,
            "source_snapshot_sha256": {
                f"code/src/{SOURCE_PATH.name}": source_hash
            },
            "artifact_sha256": _artifact_hashes(staging),
        }
        _write_json(staging / "audit_receipt.json", receipt)
        rename_noreplace(staging, output)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_prediction_audit(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
