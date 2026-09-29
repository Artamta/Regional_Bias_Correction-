#!/usr/bin/env python3
"""Score the aligned FuXi bridge under the frozen India S2S contract.

This is the only truth-bearing stage of the probabilistic bridge.  It accepts
only a complete, hash-valid 517-case inference run, opens IMD for 2020--2024
only, and scores the 505 starts whose ``init+42`` label remains in 2024.
Forecast climatologies are method-specific, use all 517 forecast-only starts,
and follow the benchmark's fixed-366-day equal-year four-year LOYO estimator.
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
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import fuxi_allseason_ensemble_calibration as calibration
import india_s2s_benchmark_scoring as scoring
import india_s2s_probabilistic_bridge as bridge
import india_s2s_probabilistic_bridge_run as inference
from india_s2s_block_bootstrap import circular_year_stratified_index_matrix


EXPERIMENT = "india_s2s_probabilistic_bridge_scoring_v1"
METHOD_ORDER = ("raw_fuxi", "moment_calibration", "location_spread")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "moment_calibration": "Train-only moment calibration",
    "location_spread": "Selected neural location-spread adapter",
}
PRIMARY_METHOD = "location_spread"
CASE_METRICS = (
    "crps",
    "acc",
    "rmse",
    "mae",
    "bias",
    "coverage90",
    "spread_skill_ratio",
)
LOSS_METRICS = ("crps", "rmse", "mae")
BLOCK_LENGTHS = (16, 13)
COMPARISON_PAIRS = (
    ("moment_calibration", "raw_fuxi"),
    ("location_spread", "raw_fuxi"),
    ("location_spread", "moment_calibration"),
)
DEFAULT_BENCHMARK_CASES = (
    bridge.BENCHMARK_CODE_ROOT
    / "results"
    / "imd_tp_sensitivity"
    / "tables"
    / "case_metrics.csv"
)
EXPECTED_BENCHMARK_CASES_SHA256 = (
    "000544615d004669b41dd87fc647bd979cd92c3638e7deee5ea77992046e1a14"
)
MOMENT_FIT_RELATIVE = "models/moment_calibration_fit.npz"
MOMENT_TRAIN_INITIALIZATIONS_SHA256 = (
    "fd763b61bb75509c81a5526d7dde93241e10484f8d89f7c9107c0d17ca52c235"
)
_RENAME_NOREPLACE = 1


class BridgeScoringError(RuntimeError):
    """Raised when scoring would violate a provenance or science contract."""


@dataclass(frozen=True)
class ValidatedInference:
    """Complete hash-gated forecast-only input to the scoring stage."""

    manifest_path: Path
    root: Path
    manifest: Mapping[str, Any]
    parent: inference.FrozenParent
    context: inference.ContextInputs
    canonical_initializations: np.ndarray
    scoring_initializations: np.ndarray
    cohort: Mapping[str, Any]


@dataclass(frozen=True)
class TruthBundle:
    """The only verification-truth object constructed by this module."""

    weekly: scoring.WeeklyIMDTargets
    region_weights: Mapping[str, np.ndarray]
    latitude: np.ndarray
    longitude: np.ndarray
    receipt: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeScoringError(f"cannot read JSON contract: {path}") from error
    if not isinstance(payload, dict):
        raise BridgeScoringError(f"JSON contract is not an object: {path}")
    return payload


def _json_normalized(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not Path(path).is_file():
        raise BridgeScoringError(f"{label} is missing: {path}")
    observed = bridge.sha256_file(Path(path))
    if observed != expected:
        raise BridgeScoringError(f"{label} SHA-256 differs: {observed} != {expected}")


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    if child.is_absolute():
        raise BridgeScoringError(f"artifact path must be relative: {relative}")
    resolved = (Path(root) / child).resolve()
    if Path(root).resolve() not in resolved.parents:
        raise BridgeScoringError(f"artifact path escapes its run: {relative}")
    return resolved


def _assert_bridge_output(path: Path) -> None:
    resolved = Path(path).resolve()
    root = bridge.DEFAULT_OUTPUT_ROOT.resolve()
    if root not in resolved.parents:
        raise BridgeScoringError("scoring output must be under the resultsv3 bridge root")


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a raced destination."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source.parent != destination.parent:
        raise BridgeScoringError("no-clobber publication requires one parent directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise BridgeScoringError("publication source is not a real directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("Linux renameat2 is required for no-clobber publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_descriptor,
            os.fsencode(source.name),
            parent_descriptor,
            os.fsencode(destination.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(parent_descriptor)


def _validate_inference_header(manifest: Mapping[str, Any]) -> None:
    expected = {
        "experiment": inference.EXPERIMENT,
        "status": "complete",
        "mode": "full",
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BridgeScoringError(
                f"inference manifest {key} differs: {manifest.get(key)!r} != {value!r}"
            )
    selection = manifest.get("inference_selection", {})
    required_selection = {
        "archive_count": 517,
        "selected_count": 517,
        "scoreable_count": 505,
        "forecast_only_count": 12,
        "selected_dates_sha256": scoring.FORECAST_INITIALIZATION_SHA256,
        "scoreable_dates_sha256": scoring.SCORING_INITIALIZATION_SHA256,
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    for key, value in required_selection.items():
        if selection.get(key) != value:
            raise BridgeScoringError(f"inference selection {key} differs")
    storage = manifest.get("storage_contract", {})
    if storage.get("operational_member_count") != 50:
        raise BridgeScoringError("inference does not represent exactly 50 members")
    if storage.get("corrected_members_exactly_reconstructible") is not True:
        raise BridgeScoringError("inference corrected members are not reconstructible")
    if storage.get("corrected_members_stored") is not False:
        raise BridgeScoringError("unexpected materialized corrected-member artifact")


def validate_inference_manifest(path: Path) -> ValidatedInference:
    """Re-run the complete inference validator and bind its exact artifacts."""

    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    _assert_bridge_output(root)
    manifest = _read_json(manifest_path)
    _validate_inference_header(manifest)
    if manifest.get("output_path") != str(root):
        raise BridgeScoringError("inference output path differs from its manifest location")

    parent_receipt = manifest.get("parent_receipt", {})
    parent_path = Path(str(parent_receipt.get("parent_manifest", "")))
    parent = inference.load_frozen_parent(parent_path)
    if _json_normalized(parent.receipt) != _json_normalized(parent_receipt):
        raise BridgeScoringError("inference parent receipt differs from a fresh validation")

    canonical = bridge.load_canonical_initializations()
    cohort = bridge.build_cohort_receipt(canonical, strict=True)
    score_inits = canonical[bridge.scoring_mask(canonical)]
    scoring.validate_bridge_cohorts(canonical, score_inits, strict_counts=True)
    selection = manifest["inference_selection"]
    execution = manifest.get("execution", {})
    execution_contract = {
        key: execution.get(key)
        for key in (
            "requested_device",
            "resolved_device",
            "automatic_mixed_precision",
            "batch_size",
        )
    }
    complete = inference._validate_complete_run(
        root,
        parent=parent,
        full_cohort=cohort,
        selected=canonical,
        selection=selection,
        execution_contract=execution_contract,
        source_hashes=manifest.get("source_snapshot_sha256", {}),
    )
    if complete is None:
        raise BridgeScoringError("inference manifest disappeared during validation")
    if _json_normalized(complete) != _json_normalized(manifest):
        raise BridgeScoringError("inference manifest changed during validation")
    context = inference._load_context_artifact(root, parent)
    return ValidatedInference(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        parent=parent,
        context=context,
        canonical_initializations=canonical,
        scoring_initializations=score_inits,
        cohort=cohort,
    )


def validate_preflight_manifest(
    path: Path,
    *,
    validated: ValidatedInference,
) -> Mapping[str, Any]:
    """Validate the immutable preflight against the same parent and cohort."""

    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    _assert_bridge_output(root)
    manifest = _read_json(manifest_path)
    expected = {
        "experiment": bridge.EXPERIMENT,
        "status": "preflight_complete",
        "output_path": str(root),
        "sealed_2025_target_opened": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BridgeScoringError(f"preflight {key} differs")
    if manifest.get("parent_receipt") != validated.parent.receipt:
        raise BridgeScoringError("preflight and inference parent receipts differ")
    expected_hashes = bridge._artifact_hashes(root)
    if manifest.get("artifact_sha256") != expected_hashes:
        raise BridgeScoringError("preflight artifact hashes differ")
    parent_receipt = _read_json(root / "parent_receipt.json")
    if parent_receipt != validated.parent.receipt:
        raise BridgeScoringError("preflight parent receipt artifact differs")
    cohort_receipt = _read_json(root / "cohort_receipt.json")
    expected_cohort = _json_normalized(validated.cohort)
    if cohort_receipt != expected_cohort:
        raise BridgeScoringError("preflight cohort receipt differs")
    summary = manifest.get("cohort_contract", {})
    expected_summary = {
        "forecast_only_count": 517,
        "scoring_count": 505,
        "jjas_initialization_count": 170,
        "jjas_valid_midpoint_count_by_lead": {
            str(lead): 169 for lead in scoring.LEAD_WEEKS
        },
    }
    if _json_normalized(summary) != expected_summary:
        raise BridgeScoringError("preflight cohort summary differs")
    return manifest


def load_moment_fit(validated: ValidatedInference) -> calibration.MomentFit:
    """Load only the corrected parent's hash-bound train-only moment fit."""

    parent_manifest = validated.parent.manifest
    expected_contract = {
        "equation": "u'_m = u_bar + delta_lead_month_grid + s_lead_month*(u_m-u_bar)",
        "fit_scope": "effective training split only",
        "location_shrinkage": 10.0,
        "spread_clip": [0.25, 4.0],
    }
    if parent_manifest.get("moment_calibration") != expected_contract:
        raise BridgeScoringError("parent moment calibration is not the train-only contract")
    if parent_manifest.get("split_counts_selected", {}).get("train") != 1652:
        raise BridgeScoringError("parent moment fit does not bind 1,652 training starts")
    retained = parent_manifest.get("retained_initializations", {})
    train = np.asarray(retained.get("train", []), dtype="datetime64[D]")
    validation = np.asarray(retained.get("validation", []), dtype="datetime64[D]")
    test = np.asarray(retained.get("test", []), dtype="datetime64[D]")
    if (
        train.shape != (1652,)
        or validation.shape != (196,)
        or test.shape != (208,)
        or bridge.initialization_dates_sha256(train)
        != MOMENT_TRAIN_INITIALIZATIONS_SHA256
        or train[0] != np.datetime64("2002-01-03")
        or train[-1] != np.datetime64("2017-11-17")
        or bridge.target_valid_dates(train)[:, -1, -1].max()
        != np.datetime64("2017-12-29")
        or validation.min() < np.datetime64("2018-01-01")
        or test.min() < np.datetime64("2020-01-01")
    ):
        raise BridgeScoringError("parent moment fit training-cohort binding differs")
    expected_hash = parent_manifest.get("artifact_sha256", {}).get(MOMENT_FIT_RELATIVE)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise BridgeScoringError("parent does not bind the moment calibration fit")
    path = _resolve_child(validated.parent.run_root, MOMENT_FIT_RELATIVE)
    _require_hash(path, expected_hash, "moment calibration fit")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"delta_log_location", "spread_factor", "shrinkage"}:
            raise BridgeScoringError("moment calibration fit keys differ")
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        shrinkage = float(archive["shrinkage"])
    if delta.shape != (6, 12, 27, 27) or spread.shape != (6, 12):
        raise BridgeScoringError("moment calibration fit shapes differ")
    if not np.isfinite(delta).all() or not np.isfinite(spread).all():
        raise BridgeScoringError("moment calibration fit is non-finite")
    if np.any((spread < 0.25) | (spread > 4.0)) or shrinkage != 10.0:
        raise BridgeScoringError("moment calibration fit bounds differ")
    return calibration.MomentFit(delta, spread, shrinkage)


def independent_affine_log_calibration(
    raw_members: np.ndarray,
    delta_log_location: np.ndarray,
    spread_factor: np.ndarray,
) -> np.ndarray:
    """Independently reproduce the frozen rank-preserving log-space transform."""

    raw = np.asarray(raw_members, dtype=np.float32)
    delta = np.asarray(delta_log_location, dtype=np.float32)
    spread = np.asarray(spread_factor, dtype=np.float32)
    if raw.ndim != 5 or raw.shape[1:] != (50, 6, 27, 27):
        raise BridgeScoringError("independent reconstruction requires 50 FuXi members")
    if delta.shape != (raw.shape[0], 6, 27, 27):
        raise BridgeScoringError("independent reconstruction delta shape differs")
    if spread.shape not in {
        (raw.shape[0], 6),
        (raw.shape[0], 6, 27, 27),
    }:
        raise BridgeScoringError("independent reconstruction spread shape differs")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise BridgeScoringError("independent reconstruction raw members are invalid")
    raw_log = np.log1p(raw).astype(np.float32)
    centre = raw_log.mean(axis=1, dtype=np.float64).astype(np.float32)
    expanded_spread = spread[:, None]
    if spread.ndim == 2:
        expanded_spread = expanded_spread[..., None, None]
    corrected_log = centre[:, None] + delta[:, None] + expanded_spread * (
        raw_log - centre[:, None]
    )
    corrected = np.expm1(np.clip(corrected_log, 0.0, 20.0)).astype(np.float32)
    if not np.isfinite(corrected).all() or np.any(corrected < 0.0):
        raise BridgeScoringError("independent reconstruction produced invalid members")
    return corrected


def independent_neural_members(
    raw_members: np.ndarray,
    delta_log_location: np.ndarray,
    log_spread: np.ndarray,
) -> np.ndarray:
    """Reconstruct compact neural shards without calling the inference helper."""

    spread = np.exp(
        np.clip(np.asarray(log_spread, dtype=np.float32), -2.0, 2.0)
    ).astype(np.float32)
    return independent_affine_log_calibration(raw_members, delta_log_location, spread)


def independent_moment_members(
    raw_members: np.ndarray,
    initializations: np.ndarray,
    fit: calibration.MomentFit,
) -> np.ndarray:
    """Apply the train-only lead/month fit using an independent month lookup."""

    inits = np.asarray(initializations, dtype="datetime64[D]")
    midpoint_labels = bridge.target_valid_dates(inits)[:, :, 3]
    months = pd.DatetimeIndex(midpoint_labels.reshape(-1)).month.to_numpy().reshape(
        len(inits), 6
    )
    delta = np.empty((len(inits), 6, 27, 27), dtype=np.float32)
    spread = np.empty((len(inits), 6), dtype=np.float32)
    for lead in range(6):
        delta[:, lead] = fit.delta_log_location[lead, months[:, lead] - 1]
        spread[:, lead] = fit.spread_factor[lead, months[:, lead] - 1]
    return independent_affine_log_calibration(raw_members, delta, spread)


def _shard_receipts(validated: ValidatedInference) -> dict[int, Mapping[str, Any]]:
    receipts = validated.manifest.get("inference_shards", [])
    by_year = {int(item.get("year", -1)): item for item in receipts}
    if tuple(sorted(by_year)) != tuple(scoring.FORECAST_YEARS) or len(receipts) != 5:
        raise BridgeScoringError("inference does not contain exactly five yearly shards")
    return by_year


def _load_shard(validated: ValidatedInference, year: int) -> dict[str, np.ndarray]:
    receipt = _shard_receipts(validated)[year]
    artifact = _resolve_child(validated.root, str(receipt.get("artifact", "")))
    _require_hash(artifact, str(receipt.get("artifact_sha256", "")), f"{year} shard")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    year_mask = (
        pd.DatetimeIndex(validated.canonical_initializations).year.to_numpy() == year
    )
    expected_inits = validated.canonical_initializations[year_mask]
    inference.validate_adjustment_shard(
        artifact,
        receipt,
        expected_initializations=expected_inits,
        expected_latitude=validated.context.latitude,
        expected_longitude=validated.context.longitude,
        checkpoint_sha256=str(validated.parent.receipt["selected_checkpoint_sha256"]),
    )
    return arrays


def _validate_live_forecast_source(
    loader: Any,
    year: int,
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    current = inference._forecast_record(loader, year)
    frozen = receipt.get("raw_source", {})
    for key in ("year", "store", "manifest", "manifest_sha256", "zmetadata_sha256"):
        if frozen.get(key) != current.get(key):
            raise BridgeScoringError(f"FuXi {year} raw-source {key} differs")
    store = Path(str(current["store"]))
    if store.name != f"{year}.zarr" or year not in scoring.FORECAST_YEARS:
        raise BridgeScoringError(f"FuXi source is not the requested 2020--2024 year: {store}")
    selection_indices = frozen.get("selection_indices", [])
    expected_count = scoring.FORECAST_COUNTS[year]
    if selection_indices != list(range(expected_count)):
        raise BridgeScoringError(f"FuXi {year} full-archive selection indices differ")
    return current


def build_forecast_climatologies(
    validated: ValidatedInference,
    moment_fit: calibration.MomentFit,
    *,
    batch_size: int,
) -> tuple[dict[str, scoring.ForecastClimatology], dict[int, np.ndarray], list[dict[str, Any]]]:
    """Build all three method climatologies from all 517 forecast-only starts."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loader = inference._forecast_loader()
    accumulators = {
        method: scoring.EqualYearForecastClimatologyAccumulator(
            expected_counts=scoring.FORECAST_COUNTS
        )
        for method in METHOD_ORDER
    }
    moment_means: dict[int, np.ndarray] = {}
    sources: list[dict[str, Any]] = []
    receipts = _shard_receipts(validated)
    for year in scoring.FORECAST_YEARS:
        shard = _load_shard(validated, year)
        inits = np.asarray(shard["initializations"], dtype="datetime64[D]")
        raw_stored = np.asarray(shard["raw_ensemble_mean"], dtype=np.float32)
        neural_stored = np.asarray(shard["corrected_ensemble_mean"], dtype=np.float32)
        accumulators["raw_fuxi"].update(inits, raw_stored)
        accumulators["location_spread"].update(inits, neural_stored)
        source = dict(_validate_live_forecast_source(loader, year, receipts[year]))
        dataset = loader.open_forecast_dataset(
            model="fuxi_s2s",
            variable="tp",
            year=year,
            grid="common_1p5",
            experiment_id=bridge.FUXI_EXPERIMENT_ID,
        )
        try:
            inference.validate_forecast_dataset(
                dataset,
                year=year,
                canonical_initializations=inits,
                context=validated.context,
            )
            variable = dataset["forecast_weekly_mean"]
            moment_mean = np.empty_like(raw_stored)
            digest = hashlib.sha256()
            for begin in range(0, len(inits), batch_size):
                end = min(begin + batch_size, len(inits))
                raw = np.asarray(variable.isel(init=slice(begin, end)).load().values, dtype=np.float32)
                inference._content_digest_update(digest, raw)
                observed_raw_mean = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
                if not np.array_equal(observed_raw_mean, raw_stored[begin:end]):
                    raise BridgeScoringError(f"FuXi {year} stored raw means differ")
                corrected = independent_neural_members(
                    raw,
                    shard["delta_log_location"][begin:end],
                    shard["log_spread"][begin:end],
                )
                observed_neural_mean = corrected.mean(axis=1, dtype=np.float64).astype(np.float32)
                if not np.array_equal(observed_neural_mean, neural_stored[begin:end]):
                    raise BridgeScoringError(f"FuXi {year} stored neural means differ")
                moment = independent_moment_members(
                    raw, inits[begin:end], moment_fit
                )
                moment_mean[begin:end] = moment.mean(axis=1, dtype=np.float64).astype(np.float32)
            expected_digest = receipts[year]["raw_source"]["raw_weekly_members_sha256"]
            if digest.hexdigest() != expected_digest:
                raise BridgeScoringError(f"FuXi {year} raw member content hash differs")
        finally:
            dataset.close()
        moment_means[year] = moment_mean
        accumulators["moment_calibration"].update(inits, moment_mean)
        source["raw_weekly_members_sha256"] = digest.hexdigest()
        sources.append(source)
    climates = {method: accumulator.finalize() for method, accumulator in accumulators.items()}
    for method, climate in climates.items():
        if climate.initialization_counts_by_year != scoring.FORECAST_COUNTS:
            raise BridgeScoringError(f"{method} climatology did not use all 517 starts")
    return climates, moment_means, sources


def _catalog_record_hash(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    store = Path(str(record["store"]))
    zmetadata = store / ".zmetadata"
    expected = str(record["zmetadata_sha256"])
    _require_hash(zmetadata, expected, f"{label} .zmetadata")
    return {
        "store": str(store),
        "zmetadata_sha256": expected,
    }


def load_imd_truth(
    validated: ValidatedInference,
    loader: Any,
) -> TruthBundle:
    """Open IMD 2020--2024 only and build the exact 505 +1..+42 targets."""

    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for year in scoring.FORECAST_YEARS:
        records = loader.list_observations(source="imd", variable="tp", year=year)
        if len(records) != 1:
            raise BridgeScoringError(f"expected one IMD observation record for {year}")
        record = records.iloc[0].to_dict()
        store = Path(str(record["store"]))
        if store.name != f"{year}.zarr" or year == 2025:
            raise BridgeScoringError(f"IMD store escapes the 2020--2024 firewall: {store}")
        source = {"year": year, **_catalog_record_hash(record, label=f"IMD {year}")}
        dataset = loader.open_observation_dataset(source="imd", variable="tp", year=year)
        try:
            current_dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            current_values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            current_fraction = np.asarray(
                dataset.observation_fraction.load().values, dtype=np.float32
            )
            if not np.array_equal(dataset.latitude.values, validated.context.latitude):
                raise BridgeScoringError(f"IMD {year} latitude differs")
            if not np.array_equal(dataset.longitude.values, validated.context.longitude):
                raise BridgeScoringError(f"IMD {year} longitude differs")
        finally:
            dataset.close()
        if current_dates.min() != np.datetime64(f"{year}-01-01") or current_dates.max() != np.datetime64(
            f"{year}-12-31"
        ):
            raise BridgeScoringError(f"IMD {year} daily calendar is incomplete")
        if current_fraction.ndim == 2:
            current_fraction = np.broadcast_to(current_fraction, current_values.shape).copy()
        if current_fraction.shape != current_values.shape:
            raise BridgeScoringError(f"IMD {year} observation fraction shape differs")
        dates.append(current_dates)
        values.append(current_values)
        fractions.append(current_fraction)
        sources.append(source)

    climatology_records = [
        item
        for item in loader.catalog.observation_climatologies
        if item.get("source") == "imd" and item.get("baseline") == [1991, 2019]
    ]
    if len(climatology_records) != 1:
        raise BridgeScoringError("IMD 1991--2019 normal is ambiguous")
    climatology_source = _catalog_record_hash(
        climatology_records[0], label="IMD 1991--2019 climatology"
    )
    climate_dataset = loader.open_observation_climatology_dataset(
        source="imd", baseline=(1991, 2019)
    )
    try:
        daily_climatology = np.asarray(
            climate_dataset.climatology_mean.load().values, dtype=np.float32
        )
        if not np.array_equal(climate_dataset.latitude.values, validated.context.latitude):
            raise BridgeScoringError("IMD climatology latitude differs")
        if not np.array_equal(climate_dataset.longitude.values, validated.context.longitude):
            raise BridgeScoringError("IMD climatology longitude differs")
    finally:
        climate_dataset.close()

    combined_dates = np.concatenate(dates)
    combined_values = np.concatenate(values)
    combined_fractions = np.concatenate(fractions)
    weekly = scoring.weekly_imd_targets(
        validated.scoring_initializations,
        combined_dates,
        combined_values,
        combined_fractions,
        daily_climatology,
        reject_daily_dates_after_cutoff=True,
    )
    maximum_target = weekly.target_dates[:, -1, -1].max()
    if maximum_target != np.datetime64("2024-12-30"):
        raise BridgeScoringError(f"maximum opened target is not 2024-12-30: {maximum_target}")

    support_dataset = loader.open_spatial_support()
    try:
        latitude = np.asarray(support_dataset.latitude.values, dtype=np.float64)
        longitude = np.asarray(support_dataset.longitude.values, dtype=np.float64)
        india = np.asarray(support_dataset.india_area_weight_km2.load().values, dtype=np.float64)
        cell = np.asarray(support_dataset.cell_area_km2.load().values, dtype=np.float64)
        region_fractions = {
            name: np.asarray(
                support_dataset[f"{name}_fraction"].load().values, dtype=np.float64
            )
            for name in scoring.REGION_ORDER[1:]
        }
    finally:
        support_dataset.close()
    if not np.array_equal(latitude, validated.context.latitude) or not np.array_equal(
        longitude, validated.context.longitude
    ):
        raise BridgeScoringError("spatial-support grid differs from inference")
    region_weights = scoring.build_region_base_weights(india, cell, region_fractions)
    support_store = (
        loader.catalog.archive_root / loader.catalog.config["spatial_support"]
    ).resolve()
    support_hash = bridge.sha256_file(support_store / ".zmetadata")
    receipt = {
        "source": "imd",
        "variable": "tp",
        "opened_years": list(scoring.FORECAST_YEARS),
        "opened_2025": False,
        "daily_sources": sources,
        "climatology_baseline": [1991, 2019],
        "climatology_source": climatology_source,
        "spatial_support_store": str(support_store),
        "spatial_support_zmetadata_sha256": support_hash,
        "target_day_offsets": list(scoring.TARGET_DAY_OFFSETS),
        "scoreable_case_count": 505,
        "maximum_target_label": "2024-12-30",
        "sealed_2025_target_opened": False,
    }
    return TruthBundle(weekly, region_weights, latitude, longitude, receipt)


def probabilistic_case_rows(
    *,
    method: str,
    initializations: Iterable[Any],
    members: np.ndarray,
    truth: np.ndarray,
    weekly_observation_support: np.ndarray,
    region_base_weights: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Return case-wise 90% coverage and spread/error diagnostics."""

    inits = np.asarray(initializations, dtype="datetime64[D]")
    ensemble = np.asarray(members, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    support = np.asarray(weekly_observation_support, dtype=np.float64)
    if ensemble.shape != (len(inits), 50, 6, *target.shape[-2:]):
        raise BridgeScoringError("probabilistic metrics require [case,50,6,y,x]")
    if target.shape != (len(inits), 6, *ensemble.shape[-2:]) or support.shape != target.shape:
        raise BridgeScoringError("probabilistic truth/support shapes differ")
    lower, upper = np.quantile(ensemble, (0.05, 0.95), axis=1, method="linear")
    covered = ((target >= lower) & (target <= upper)).astype(np.float64)
    ensemble_mean = ensemble.mean(axis=1, dtype=np.float64)
    ensemble_variance_field = ensemble.var(axis=1, ddof=0, dtype=np.float64)
    mse_field = (ensemble_mean - target) ** 2
    rows: list[dict[str, Any]] = []
    for region in scoring.REGION_ORDER:
        dynamic = support * np.asarray(region_base_weights[region], dtype=np.float64)
        coverage = scoring.weighted_field_mean(covered, dynamic)
        variance = scoring.weighted_field_mean(ensemble_variance_field, dynamic)
        mse = scoring.weighted_field_mean(mse_field, dynamic)
        spread = np.sqrt(np.maximum(variance, 0.0))
        ratio = np.divide(
            spread,
            np.sqrt(mse),
            out=np.full(spread.shape, np.nan, dtype=np.float64),
            where=mse > 0.0,
        )
        for case_index, init in enumerate(inits):
            for lead_index, lead in enumerate(scoring.LEAD_WEEKS):
                rows.append(
                    {
                        "method": method,
                        "init": np.datetime_as_string(init, unit="D"),
                        "region": region,
                        "lead_week": lead,
                        "coverage90": float(coverage[case_index, lead_index]),
                        "ensemble_variance": float(variance[case_index, lead_index]),
                        "mean_squared_error": float(mse[case_index, lead_index]),
                        "ensemble_spread": float(spread[case_index, lead_index]),
                        "spread_skill_ratio": float(ratio[case_index, lead_index]),
                    }
                )
    return pd.DataFrame(rows)


def _score_members(
    *,
    method: str,
    initializations: np.ndarray,
    members: np.ndarray,
    truth: scoring.WeeklyIMDTargets,
    truth_indices: np.ndarray,
    climatology: scoring.ForecastClimatology,
    region_weights: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    target = truth.truth[truth_indices]
    observation_climate = truth.climatology[truth_indices]
    observation_support = truth.observation_support[truth_indices]
    cases = scoring.score_case_chunk(
        method=method,
        initializations=initializations,
        members=members,
        truth=target,
        forecast_climatology=climatology,
        observation_climatology=observation_climate,
        weekly_observation_support=observation_support,
        region_base_weights=region_weights,
        expected_member_count=50,
    )
    probability = probabilistic_case_rows(
        method=method,
        initializations=initializations,
        members=members,
        truth=target,
        weekly_observation_support=observation_support,
        region_base_weights=region_weights,
    )
    cases = cases.merge(
        probability,
        on=["method", "init", "region", "lead_week"],
        validate="one_to_one",
    )
    cases.insert(0, "track", "tp_imd")
    cases["model"] = method
    cases["model_label"] = METHOD_LABELS[method]
    cases["member_count"] = 50
    cases["selected_seed"] = 43 if method == PRIMARY_METHOD else "not_applicable"
    return cases


def score_all_methods(
    validated: ValidatedInference,
    climates: Mapping[str, scoring.ForecastClimatology],
    moment_fit: calibration.MomentFit,
    truth: TruthBundle,
    *,
    batch_size: int,
) -> pd.DataFrame:
    """Reopen raw members and score all methods on identical 50-member cases."""

    loader = inference._forecast_loader()
    truth_lookup = {
        value: index for index, value in enumerate(validated.scoring_initializations)
    }
    frames: list[pd.DataFrame] = []
    for year in scoring.FORECAST_YEARS:
        shard = _load_shard(validated, year)
        inits = np.asarray(shard["initializations"], dtype="datetime64[D]")
        scoreable = np.asarray(shard["scoreable_for_truth"], dtype=bool)
        expected_digest = _shard_receipts(validated)[year]["raw_source"][
            "raw_weekly_members_sha256"
        ]
        digest = hashlib.sha256()
        dataset = loader.open_forecast_dataset(
            model="fuxi_s2s",
            variable="tp",
            year=year,
            grid="common_1p5",
            experiment_id=bridge.FUXI_EXPERIMENT_ID,
        )
        try:
            variable = dataset["forecast_weekly_mean"]
            # Re-hash the complete forecast-only year during the truth-bearing
            # pass.  This closes the race between climatology construction and
            # scoring while still withholding truth for the final 12 starts.
            for begin in range(0, len(inits), batch_size):
                end = min(begin + batch_size, len(inits))
                raw_all = np.asarray(
                    variable.isel(init=slice(begin, end)).load().values,
                    dtype=np.float32,
                )
                inference._content_digest_update(digest, raw_all)
                local = np.flatnonzero(scoreable[begin:end])
                if local.size == 0:
                    continue
                selected = begin + local
                batch_inits = inits[selected]
                if any(value not in truth_lookup for value in batch_inits):
                    raise BridgeScoringError(f"FuXi {year} scoreable start lacks IMD truth")
                truth_indices = np.asarray(
                    [truth_lookup[value] for value in batch_inits], dtype=np.int64
                )
                raw = raw_all[local]
                neural = independent_neural_members(
                    raw,
                    shard["delta_log_location"][selected],
                    shard["log_spread"][selected],
                )
                moment = independent_moment_members(raw, batch_inits, moment_fit)
                members_by_method = {
                    "raw_fuxi": raw,
                    "moment_calibration": moment,
                    "location_spread": neural,
                }
                for method in METHOD_ORDER:
                    frames.append(
                        _score_members(
                            method=method,
                            initializations=batch_inits,
                            members=members_by_method[method],
                            truth=truth.weekly,
                            truth_indices=truth_indices,
                            climatology=climates[method],
                            region_weights=truth.region_weights,
                        )
                    )
            if digest.hexdigest() != expected_digest:
                raise BridgeScoringError(
                    f"FuXi {year} raw content changed before scoring"
                )
        finally:
            dataset.close()
    cases = pd.concat(frames, ignore_index=True)
    expected_rows = 3 * scoring.RAW_IDENTITY_CASE_COUNT
    if len(cases) != expected_rows:
        raise BridgeScoringError(f"case table has {len(cases)} rows, expected {expected_rows}")
    expected_per_method = {method: scoring.RAW_IDENTITY_CASE_COUNT for method in METHOD_ORDER}
    if cases.groupby("method").size().to_dict() != expected_per_method:
        raise BridgeScoringError("method case counts differ")
    return cases


def aggregate_case_metrics(
    cases: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("method", "region", "lead_week"),
) -> pd.DataFrame:
    """Arithmetic means of case scores, plus a separately pooled spread ratio."""

    required = {*group_by, "init", *CASE_METRICS, "ensemble_variance", "mean_squared_error"}
    missing = sorted(required.difference(cases.columns))
    if missing:
        raise BridgeScoringError(f"case table lacks aggregate columns: {missing}")
    if cases.duplicated([*group_by, "init"]).any():
        raise BridgeScoringError("case table has duplicate initialization rows per group")
    rows: list[dict[str, Any]] = []
    for key, group in cases.groupby(list(group_by), sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_by, values, strict=True))
        row["case_count"] = int(len(group))
        for metric in CASE_METRICS:
            data = group[metric].to_numpy(dtype=np.float64)
            finite = data[np.isfinite(data)]
            row[f"{metric}_valid_case_count"] = int(len(finite))
            if metric == "spread_skill_ratio":
                row["mean_case_spread_skill_ratio"] = (
                    float(finite.mean()) if finite.size else np.nan
                )
            else:
                row[metric] = float(finite.mean()) if finite.size else np.nan
        variance = float(group["ensemble_variance"].mean())
        mse = float(group["mean_squared_error"].mean())
        row["ensemble_variance"] = variance
        row["mean_squared_error"] = mse
        row["ensemble_spread"] = float(np.sqrt(max(variance, 0.0)))
        row["spread_skill_ratio"] = (
            float(np.sqrt(max(variance, 0.0) / mse)) if mse > 0.0 else np.nan
        )
        row["pooled_spread_skill_ratio"] = row["spread_skill_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def add_raw_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Add within-region/lead effects relative to the identical raw forecast."""

    keys = [column for column in ("region", "lead_week") if column in summary.columns]
    raw = summary[summary.method.eq("raw_fuxi")].set_index(keys)
    result = summary.copy()
    result["crps_skill_pct_vs_raw"] = np.nan
    result["rmse_skill_pct_vs_raw"] = np.nan
    result["mae_skill_pct_vs_raw"] = np.nan
    for index, row in result.iterrows():
        key = tuple(row[column] for column in keys)
        if len(key) == 1:
            key = key[0]
        reference = raw.loc[key]
        for metric in LOSS_METRICS:
            baseline = float(reference[metric])
            result.loc[index, f"{metric}_skill_pct_vs_raw"] = (
                100.0 * (1.0 - float(row[metric]) / baseline)
                if baseline > 0.0
                else np.nan
            )
        for metric in ("acc", "bias", "coverage90", "spread_skill_ratio"):
            result.loc[index, f"delta_{metric}_vs_raw"] = float(row[metric]) - float(
                reference[metric]
            )
    return result


def validate_raw_identity(
    cases: pd.DataFrame,
    benchmark_path: Path,
    scoring_initializations: np.ndarray,
) -> Mapping[str, Any]:
    """Require exact 15,150-row raw deterministic identity with the benchmark."""

    source = Path(benchmark_path).resolve()
    _require_hash(source, EXPECTED_BENCHMARK_CASES_SHA256, "frozen benchmark case table")
    benchmark = pd.read_csv(source)
    allowed = {
        np.datetime_as_string(value, unit="D") for value in scoring_initializations
    }
    benchmark_raw = benchmark[
        benchmark.model.eq("fuxi_s2s")
        & benchmark.reference.eq("imd")
        & benchmark.score_status.eq("available")
        & benchmark.init.isin(allowed)
    ].copy()
    independent_raw = cases[cases.method.eq("raw_fuxi")].copy()
    receipt = scoring.assert_raw_fuxi_identity(independent_raw, benchmark_raw)
    return {
        **receipt,
        "benchmark_case_metrics": str(source),
        "benchmark_case_metrics_sha256": EXPECTED_BENCHMARK_CASES_SHA256,
    }


def _paired_method_arrays(
    selected: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = selected[selected.method.isin((candidate, baseline))]
    if subset.duplicated(["method", "init"]).any():
        raise BridgeScoringError("paired interval input has duplicate method/init rows")
    wide = subset.pivot(index="init", columns="method", values=metric).sort_index()
    if set(wide.columns) != {candidate, baseline} or wide.isna().any().any():
        raise BridgeScoringError(
            f"incomplete {candidate}/{baseline} pairing for {metric}"
        )
    dates = wide.index.to_numpy(dtype="datetime64[D]")
    return (
        dates,
        wide[candidate].to_numpy(dtype=np.float64),
        wide[baseline].to_numpy(dtype=np.float64),
    )


def compute_paired_intervals(
    jjas_cases: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 42,
    block_lengths: Sequence[int] = BLOCK_LENGTHS,
    regions: Sequence[str] = scoring.REGION_ORDER,
    lead_weeks: Sequence[int] = scoring.LEAD_WEEKS,
    require_169: bool = True,
    analysis_cohort: str = "2020_2024_valid_midpoint_jjas",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute paired block intervals for both calibrated forecasts vs raw."""

    if replicates < 1 or any(int(value) < 1 for value in block_lengths):
        raise ValueError("bootstrap replicate and block counts must be positive")
    required = {
        "method",
        "region",
        "lead_week",
        "init",
        "season",
        "ensemble_variance",
        "mean_squared_error",
        *CASE_METRICS,
    }
    missing = sorted(required.difference(jjas_cases.columns))
    if missing:
        raise BridgeScoringError(f"paired interval table lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    sampling_receipts: dict[str, Any] = {}
    for lead in lead_weeks:
        lead_cases = jjas_cases[jjas_cases.lead_week.eq(lead)]
        reference_dates: np.ndarray | None = None
        index_by_block: dict[int, np.ndarray] = {}
        for region in regions:
            selected = lead_cases[lead_cases.region.eq(region)]
            raw_dates = np.asarray(
                sorted(selected.loc[selected.method.eq("raw_fuxi"), "init"].unique()),
                dtype="datetime64[D]",
            )
            if require_169 and len(raw_dates) != 169:
                raise BridgeScoringError(
                    f"valid-midpoint JJAS requires 169 cases for {region}, W{lead}"
                )
            if reference_dates is None:
                reference_dates = raw_dates
                for block in block_lengths:
                    draw_seed = int(seed + 1000 * int(lead) + int(block))
                    indices = circular_year_stratified_index_matrix(
                        raw_dates,
                        block_length=int(block),
                        replicates=replicates,
                        seed=draw_seed,
                    )
                    index_by_block[int(block)] = indices
                    key = f"w{lead}__block{block}"
                    sampling_receipts[key] = {
                        "dates_sha256": scoring.initialization_dates_sha256(raw_dates),
                        "index_matrix_sha256": hashlib.sha256(
                            np.ascontiguousarray(indices).tobytes()
                        ).hexdigest(),
                        "shape": list(indices.shape),
                        "seed": draw_seed,
                    }
            elif not np.array_equal(raw_dates, reference_dates):
                raise BridgeScoringError(f"regional JJAS date cohorts differ at W{lead}")

            for candidate, baseline in COMPARISON_PAIRS:
                for metric in CASE_METRICS:
                    if metric == "spread_skill_ratio":
                        dates, candidate_variance, baseline_variance = (
                            _paired_method_arrays(
                                selected, candidate, baseline, "ensemble_variance"
                            )
                        )
                        mse_dates, candidate_mse, baseline_mse = _paired_method_arrays(
                            selected, candidate, baseline, "mean_squared_error"
                        )
                        if not np.array_equal(dates, mse_dates):
                            raise BridgeScoringError(
                                f"spread/error dates differ for {candidate}/{baseline}"
                            )
                        candidate_estimate = float(
                            np.sqrt(candidate_variance.mean() / candidate_mse.mean())
                        )
                        baseline_estimate = float(
                            np.sqrt(baseline_variance.mean() / baseline_mse.mean())
                        )
                        candidate_values = baseline_values = None
                    else:
                        dates, candidate_values, baseline_values = (
                            _paired_method_arrays(
                                selected, candidate, baseline, metric
                            )
                        )
                        candidate_estimate = float(candidate_values.mean())
                        baseline_estimate = float(baseline_values.mean())
                    if not np.array_equal(dates, reference_dates):
                        raise BridgeScoringError(
                            "paired dates differ for "
                            f"{candidate}/{baseline}/{region}/W{lead}/{metric}"
                        )
                    for block in block_lengths:
                        indices = index_by_block[int(block)]
                        if metric == "spread_skill_ratio":
                            candidate_samples = np.sqrt(
                                candidate_variance[indices].mean(axis=1)
                                / candidate_mse[indices].mean(axis=1)
                            )
                            baseline_samples = np.sqrt(
                                baseline_variance[indices].mean(axis=1)
                                / baseline_mse[indices].mean(axis=1)
                            )
                            samples = candidate_samples - baseline_samples
                            estimate = candidate_estimate - baseline_estimate
                        else:
                            assert candidate_values is not None
                            assert baseline_values is not None
                            difference = candidate_values - baseline_values
                            samples = difference[indices].mean(
                                axis=1, dtype=np.float64
                            )
                            estimate = float(difference.mean())
                        lower, upper = np.quantile(samples, (0.025, 0.975))
                        common = {
                            "reference": "imd",
                            "season": "JJAS_valid_midpoint",
                            "analysis_cohort": analysis_cohort,
                            "region": region,
                            "lead_week": int(lead),
                            "metric": metric,
                            "candidate": candidate,
                            "baseline": baseline,
                            "candidate_mean": candidate_estimate,
                            "baseline_mean": baseline_estimate,
                            "n_cases": int(len(dates)),
                            "confidence": 0.95,
                            "block_length_starts": int(block),
                            "replicates": int(replicates),
                            "seed": int(seed + 1000 * int(lead) + int(block)),
                        }
                        rows.append(
                            {
                                **common,
                                "effect": "candidate_minus_baseline",
                                "estimate": estimate,
                                "ci_lower": float(lower),
                                "ci_upper": float(upper),
                            }
                        )
                        if metric in LOSS_METRICS:
                            assert candidate_values is not None
                            assert baseline_values is not None
                            candidate_sample_mean = candidate_values[indices].mean(axis=1)
                            baseline_sample_mean = baseline_values[indices].mean(axis=1)
                            if np.any(baseline_sample_mean <= 0.0):
                                raise BridgeScoringError(
                                    f"non-positive baseline {metric} in bootstrap"
                                )
                            skill_samples = 100.0 * (
                                1.0 - candidate_sample_mean / baseline_sample_mean
                            )
                            skill_lower, skill_upper = np.quantile(
                                skill_samples, (0.025, 0.975)
                            )
                            rows.append(
                                {
                                    **common,
                                    "effect": "skill_pct_vs_baseline",
                                    "estimate": float(
                                        100.0
                                        * (
                                            1.0
                                            - candidate_values.mean()
                                            / baseline_values.mean()
                                        )
                                    ),
                                    "ci_lower": float(skill_lower),
                                    "ci_upper": float(skill_upper),
                                }
                            )
    return pd.DataFrame(rows), sampling_receipts


def build_story_intervals(
    jjas_cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    regions: Sequence[str] = scoring.REGION_ORDER,
    lead_weeks: Sequence[int] = scoring.LEAD_WEEKS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Publish both the five-year primary and 2022--2024 story-gate cohorts."""

    if "verification_year" not in jjas_cases:
        raise BridgeScoringError("story intervals require verification_year")
    primary, primary_receipt = compute_paired_intervals(
        jjas_cases,
        replicates=replicates,
        seed=seed,
        regions=regions,
        lead_weeks=lead_weeks,
    )
    development = jjas_cases[jjas_cases.verification_year.between(2022, 2024)].copy()
    development_counts = development.groupby(
        ["method", "region", "lead_week"]
    ).size()
    expected_groups = len(METHOD_ORDER) * len(regions) * len(lead_weeks)
    if len(development_counts) != expected_groups or set(development_counts) != {100}:
        raise BridgeScoringError(
            "2022--2024 valid-midpoint JJAS must contain 100 cases per lead"
        )
    development_intervals, development_receipt = compute_paired_intervals(
        development,
        replicates=replicates,
        seed=seed,
        regions=regions,
        lead_weeks=lead_weeks,
        require_169=False,
        analysis_cohort="2022_2024_valid_midpoint_jjas",
    )
    return (
        pd.concat([primary, development_intervals], ignore_index=True),
        {
            "2020_2024_valid_midpoint_jjas": primary_receipt,
            "2022_2024_valid_midpoint_jjas": development_receipt,
        },
    )


def compute_pooled_w1_w6_story_intervals(
    jjas_cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_lengths: Sequence[int] = BLOCK_LENGTHS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bootstrap the synchronized 2022--2024 valid-midpoint W1--W6 gate.

    Each lead has a distinct 100-case valid-midpoint cohort.  One block draw is
    made from W1's within-year ordinal layout and the same ordinal indices are
    applied to every lead before averaging all 600 case-lead values.  This
    preserves the main season contract without pretending the leads share 100
    initialization dates.
    """

    required = {
        "method",
        "region",
        "lead_week",
        "init",
        "verification_year",
        "season",
        "crps",
        "acc",
        "bias",
    }
    missing = sorted(required.difference(jjas_cases.columns))
    if missing:
        raise BridgeScoringError(f"pooled story table lacks columns: {missing}")
    selected = jjas_cases[
        jjas_cases.region.eq("all_india")
        & jjas_cases.verification_year.between(2022, 2024)
        & jjas_cases.season.eq("JJAS")
        & jjas_cases.method.isin(METHOD_ORDER)
        & jjas_cases.lead_week.isin(scoring.LEAD_WEEKS)
    ].copy()
    selected["init"] = pd.to_datetime(selected["init"]).dt.strftime("%Y-%m-%d")
    if selected.duplicated(["method", "lead_week", "init"]).any():
        raise BridgeScoringError("pooled story table has duplicate method/init/lead rows")

    group_counts = selected.groupby(["method", "lead_week"]).size()
    expected_groups = len(METHOD_ORDER) * len(scoring.LEAD_WEEKS)
    if len(group_counts) != expected_groups or set(group_counts) != {100}:
        raise BridgeScoringError(
            "pooled story gate requires 100 valid-midpoint cases per method and lead"
        )

    expected_year_counts = {2022: 35, 2023: 35, 2024: 30}
    lead_dates: dict[int, np.ndarray] = {}
    lead_year_layout: np.ndarray | None = None
    for lead in scoring.LEAD_WEEKS:
        lead_selected = selected[selected.lead_week.eq(lead)]
        raw = lead_selected[lead_selected.method.eq("raw_fuxi")].sort_values(
            ["verification_year", "init"]
        )
        year_counts = {
            int(year): int(count)
            for year, count in raw.groupby("verification_year").size().items()
        }
        if year_counts != expected_year_counts:
            raise BridgeScoringError(
                f"pooled story W{lead} year counts differ: {year_counts}"
            )
        dates = raw["init"].to_numpy(dtype="datetime64[D]")
        date_years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
        verification_years = raw["verification_year"].to_numpy(dtype=np.int64)
        if not np.array_equal(date_years, verification_years):
            raise BridgeScoringError(
                f"pooled story W{lead} initialization/verification years differ"
            )
        if lead_year_layout is None:
            lead_year_layout = verification_years
        elif not np.array_equal(verification_years, lead_year_layout):
            raise BridgeScoringError(
                "pooled story lead cohorts lack identical year/ordinal layout"
            )
        for method in METHOD_ORDER:
            method_rows = lead_selected[lead_selected.method.eq(method)].sort_values(
                ["verification_year", "init"]
            )
            method_dates = method_rows["init"].to_numpy(dtype="datetime64[D]")
            if not np.array_equal(method_dates, dates):
                raise BridgeScoringError(
                    f"pooled story method dates differ for {method}/W{lead}"
                )
        lead_dates[int(lead)] = dates

    date_sets = [set(values.tolist()) for values in lead_dates.values()]
    intersection_count = len(set.intersection(*date_sets))
    union_count = len(set.union(*date_sets))
    if intersection_count != 70 or union_count != 130:
        raise BridgeScoringError(
            "pooled valid-midpoint lead-date geometry differs: "
            f"intersection={intersection_count}, union={union_count}"
        )

    w1_dates = lead_dates[1]
    rows: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {}
    for block in block_lengths:
        draw_seed = int(seed + 9000 + int(block))
        indices = circular_year_stratified_index_matrix(
            w1_dates,
            block_length=int(block),
            replicates=replicates,
            seed=draw_seed,
        )
        receipt[f"pooled_w1_w6__block{block}"] = {
            "ordinal_reference_lead": 1,
            "lead_dates_sha256": {
                f"w{lead}": scoring.initialization_dates_sha256(dates)
                for lead, dates in lead_dates.items()
            },
            "index_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(indices).tobytes()
            ).hexdigest(),
            "shape": list(indices.shape),
            "seed": draw_seed,
            "year_counts_per_lead": expected_year_counts,
            "n_cases_per_lead": 100,
            "n_case_lead_rows": 600,
            "lead_date_intersection_count": intersection_count,
            "lead_date_union_count": union_count,
            "sampling_unit": (
                "within-verification-year ordinal synchronized across "
                "six lead-specific valid-midpoint cohorts"
            ),
        }
        for candidate, baseline in (
            ("location_spread", "raw_fuxi"),
            ("location_spread", "moment_calibration"),
        ):
            for metric in ("crps", "acc", "bias"):
                candidate_by_lead: list[np.ndarray] = []
                baseline_by_lead: list[np.ndarray] = []
                for lead in scoring.LEAD_WEEKS:
                    paired_dates, candidate_values, baseline_values = (
                        _paired_method_arrays(
                            selected[selected.lead_week.eq(lead)],
                            candidate,
                            baseline,
                            metric,
                        )
                    )
                    if not np.array_equal(paired_dates, lead_dates[int(lead)]):
                        raise BridgeScoringError(
                            "pooled story dates differ for "
                            f"{candidate}/{baseline}/W{lead}/{metric}"
                        )
                    candidate_by_lead.append(candidate_values)
                    baseline_by_lead.append(baseline_values)
                candidate_matrix = np.stack(candidate_by_lead, axis=0)
                baseline_matrix = np.stack(baseline_by_lead, axis=0)
                candidate_samples = np.take(candidate_matrix, indices, axis=1).mean(
                    axis=(0, 2), dtype=np.float64
                )
                baseline_samples = np.take(baseline_matrix, indices, axis=1).mean(
                    axis=(0, 2), dtype=np.float64
                )
                samples = candidate_samples - baseline_samples
                lower, upper = np.quantile(samples, (0.025, 0.975))
                common = {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": (
                        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
                    ),
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "lead_aggregation": (
                        "arithmetic mean over six lead-specific 100-case cohorts; "
                        "bootstrap draws synchronized by within-year ordinal"
                    ),
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_mean": float(candidate_matrix.mean()),
                    "baseline_mean": float(baseline_matrix.mean()),
                    "n_cases_per_lead": 100,
                    "n_case_lead_rows": 600,
                    "lead_date_intersection_count": intersection_count,
                    "lead_date_union_count": union_count,
                    "confidence": 0.95,
                    "block_length_starts": int(block),
                    "replicates": int(replicates),
                    "seed": draw_seed,
                }
                rows.append(
                    {
                        **common,
                        "effect": "candidate_minus_baseline",
                        "estimate": float(
                            candidate_matrix.mean() - baseline_matrix.mean()
                        ),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
                if metric == "crps":
                    if np.any(baseline_samples <= 0.0):
                        raise BridgeScoringError("pooled raw/moment CRPS is non-positive")
                    skill = 100.0 * (1.0 - candidate_samples / baseline_samples)
                    skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                    rows.append(
                        {
                            **common,
                            "effect": "skill_pct_vs_baseline",
                            "estimate": float(
                                100.0
                                * (
                                    1.0
                                    - candidate_matrix.mean()
                                    / baseline_matrix.mean()
                                )
                            ),
                            "ci_lower": float(skill_lower),
                            "ci_upper": float(skill_upper),
                        }
                    )
    return pd.DataFrame(rows), receipt


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): bridge.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "failure.json"}
        and ".tmp" not in path.name
    }


def _scoring_sources() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(scoring.__file__).resolve(),
        Path(inference.__file__).resolve(),
        Path(bridge.__file__).resolve(),
        Path(calibration.__file__).resolve(),
        Path(circular_year_stratified_index_matrix.__code__.co_filename).resolve(),
    )


def _snapshot_sources(staging: Path) -> dict[str, str]:
    sources = _scoring_sources()
    for source in sources:
        destination = staging / "code" / "src" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        str(path.relative_to(staging)): bridge.sha256_file(path)
        for path in sorted((staging / "code" / "src").glob("*.py"))
    }


def _verify_source_snapshot(staging: Path, hashes: Mapping[str, str]) -> None:
    """Fail if any executed source changed after the startup snapshot."""

    by_name = {path.name: path for path in _scoring_sources()}
    if len(by_name) != len(_scoring_sources()):
        raise BridgeScoringError("scoring source snapshot names are ambiguous")
    for relative, expected in hashes.items():
        snapshot = staging / relative
        _require_hash(snapshot, expected, "scoring source snapshot")
        source = by_name.get(snapshot.name)
        if source is None or bridge.sha256_file(source) != expected:
            raise BridgeScoringError(f"scoring source changed during run: {snapshot.name}")


def _verify_unchanged_inputs(inputs: Mapping[Path, str]) -> None:
    for path, expected in inputs.items():
        _require_hash(path, expected, "scoring input changed during run")


def publish_scoring(
    *,
    inference_manifest: Path,
    preflight_manifest: Path,
    benchmark_case_metrics: Path,
    output: Path,
    batch_size: int = 8,
    replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> Path:
    """Run the complete scoring workflow and atomically publish its manifest."""

    destination = Path(output).resolve()
    _assert_bridge_output(destination)
    if destination.exists():
        raise FileExistsError(f"fresh output path required: {destination}")
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    truth_opened = False
    try:
        source_hashes = _snapshot_sources(staging)
        frozen_inputs = {
            Path(inference_manifest).resolve(): bridge.sha256_file(
                Path(inference_manifest)
            ),
            Path(preflight_manifest).resolve(): bridge.sha256_file(
                Path(preflight_manifest)
            ),
            Path(benchmark_case_metrics).resolve(): bridge.sha256_file(
                Path(benchmark_case_metrics)
            ),
        }
        validated = validate_inference_manifest(inference_manifest)
        preflight = validate_preflight_manifest(preflight_manifest, validated=validated)
        _require_hash(
            Path(benchmark_case_metrics),
            EXPECTED_BENCHMARK_CASES_SHA256,
            "frozen benchmark case table",
        )
        moment_fit = load_moment_fit(validated)
        moment_path = _resolve_child(validated.parent.run_root, MOMENT_FIT_RELATIVE)
        frozen_inputs[validated.parent.manifest_path] = bridge.sha256_file(
            validated.parent.manifest_path
        )
        frozen_inputs[moment_path] = bridge.sha256_file(moment_path)
        climates, _, forecast_sources = build_forecast_climatologies(
            validated, moment_fit, batch_size=batch_size
        )
        loader = inference._forecast_loader()
        truth = load_imd_truth(validated, loader)
        truth_opened = True
        cases = score_all_methods(
            validated,
            climates,
            moment_fit,
            truth,
            batch_size=batch_size,
        )
        raw_identity = validate_raw_identity(
            cases, benchmark_case_metrics, validated.scoring_initializations
        )
        allseason = add_raw_comparisons(aggregate_case_metrics(cases))
        jjas = cases[cases.season.eq("JJAS")].copy()
        expected_jjas_rows = 3 * 5 * 6 * 169
        if len(jjas) != expected_jjas_rows:
            raise BridgeScoringError(
                f"valid-midpoint JJAS has {len(jjas)} rows, expected {expected_jjas_rows}"
            )
        jjas_summary = add_raw_comparisons(aggregate_case_metrics(jjas))
        if set(jjas_summary.case_count) != {169}:
            raise BridgeScoringError("valid-midpoint JJAS summary is not 169 cases per lead")
        intervals, resampling = build_story_intervals(
            jjas,
            replicates=replicates,
            seed=bootstrap_seed,
        )
        pooled_intervals, pooled_resampling = (
            compute_pooled_w1_w6_story_intervals(
                jjas,
                replicates=replicates,
                seed=bootstrap_seed,
            )
        )
        intervals = pd.concat([intervals, pooled_intervals], ignore_index=True)
        resampling[
            "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
        ] = pooled_resampling

        tables = staging / "tables"
        receipts = staging / "receipts"
        tables.mkdir(parents=True)
        receipts.mkdir(parents=True)
        cases.to_csv(tables / "case_metrics.csv", index=False)
        allseason.to_csv(tables / "allseason_summary.csv", index=False)
        jjas_summary.to_csv(tables / "jjas_valid_midpoint_summary.csv", index=False)
        intervals.to_csv(tables / "paired_block_intervals.csv", index=False)
        bridge.write_json(receipts / "raw_fuxi_identity.json", raw_identity)
        bridge.write_json(receipts / "truth_access.json", truth.receipt)
        bridge.write_json(receipts / "cohort.json", validated.cohort)
        bridge.write_json(receipts / "bootstrap_sampling.json", resampling)
        _verify_source_snapshot(staging, source_hashes)
        _verify_unchanged_inputs(frozen_inputs)
        manifest: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "scientific_status": "retrospective 2020-2024 benchmark evidence",
            "created_utc": bridge.utc_now(),
            "output_path": str(destination),
            "input_inference_manifest": str(validated.manifest_path),
            "input_inference_manifest_sha256": frozen_inputs[
                validated.manifest_path
            ],
            "input_preflight_manifest": str(Path(preflight_manifest).resolve()),
            "input_preflight_manifest_sha256": frozen_inputs[
                Path(preflight_manifest).resolve()
            ],
            "parent_manifest": validated.parent.receipt["parent_manifest"],
            "parent_manifest_sha256": validated.parent.receipt[
                "parent_manifest_sha256"
            ],
            "selected_seed": 43,
            "selected_checkpoint_sha256": validated.parent.receipt[
                "selected_checkpoint_sha256"
            ],
            "moment_fit": str(moment_path),
            "moment_fit_sha256": frozen_inputs[moment_path],
            "benchmark_case_metrics": str(Path(benchmark_case_metrics).resolve()),
            "benchmark_case_metrics_sha256": EXPECTED_BENCHMARK_CASES_SHA256,
            "raw_fuxi_identity": raw_identity,
            "cohort": {
                "forecast_only_climatology_count": 517,
                "scoring_count": 505,
                "valid_midpoint_jjas_count_per_lead": 169,
                "target_day_offsets": list(scoring.TARGET_DAY_OFFSETS),
                "maximum_target_label_opened": "2024-12-30",
            },
            "methods": list(METHOD_ORDER),
            "member_count_each_method": 50,
            "metrics": list(CASE_METRICS),
            "aggregation": "arithmetic mean of per-initialization spatial scores",
            "forecast_climatology": (
                "method-specific all-517 fixed-366-day centered-31-day "
                "equal-year four-year LOYO"
            ),
            "forecast_sources": forecast_sources,
            "truth_access": truth.receipt,
            "uncertainty": {
                "cohorts": {
                    "2020_2024_valid_midpoint_jjas": 169,
                    "2022_2024_valid_midpoint_jjas": 100,
                    "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6": {
                        "n_cases_per_lead": 100,
                        "n_case_lead_rows": 600,
                        "lead_date_intersection_count": 70,
                        "lead_date_union_count": 130,
                    },
                },
                "resampling": (
                    "paired year-stratified circular initialization blocks; pooled "
                    "W1-W6 uses one W1 within-year ordinal draw synchronized across "
                    "six lead-specific valid-midpoint cohorts"
                ),
                "block_lengths_starts": list(BLOCK_LENGTHS),
                "replicates": int(replicates),
                "seed": int(bootstrap_seed),
                "comparisons": [list(values) for values in COMPARISON_PAIRS],
            },
            "table_rows": {
                "case_metrics": int(len(cases)),
                "allseason_summary": int(len(allseason)),
                "jjas_valid_midpoint_summary": int(len(jjas_summary)),
                "paired_block_intervals": int(len(intervals)),
            },
            "preflight_parent_receipt": preflight["parent_receipt"],
            "source_snapshot_sha256": source_hashes,
            "opened_observation_years": list(scoring.FORECAST_YEARS),
            "opened_2025_observation": False,
            "sealed_2025_target_opened": False,
        }
        manifest["artifact_sha256"] = _artifact_hashes(staging)
        bridge.write_json(staging / "manifest.json", manifest)
        _verify_source_snapshot(staging, source_hashes)
        _verify_unchanged_inputs(frozen_inputs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staging, destination)
    except BaseException as error:
        bridge.write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "failed_utc": bridge.utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "verification_truth_opened_before_failure": truth_opened,
                "opened_2025_observation": False,
                "sealed_2025_target_opened": False,
            },
        )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument(
        "--benchmark-case-metrics", type=Path, default=DEFAULT_BENCHMARK_CASES
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.replicates < 1:
        raise ValueError("batch size and bootstrap replicates must be positive")
    manifest = publish_scoring(
        inference_manifest=args.inference_manifest,
        preflight_manifest=args.preflight_manifest,
        benchmark_case_metrics=args.benchmark_case_metrics,
        output=args.output,
        batch_size=args.batch_size,
        replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"PASS: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
