#!/usr/bin/env python3
"""Score the frozen FuXi bridge against unfitted 2020--2024 IMERG.

The calibrated forecasts, forecast climatologies, and 505-case cohort are
identical to the IMD bridge score.  Only the observation reference changes:
IMERG is period-start labelled, so W1 uses ``init..init+6`` and W6 uses
``init+35..init+41``.  The command never opens a 2025 observation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import shutil
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
import india_s2s_probabilistic_bridge_score as imd_score


EXPERIMENT = "india_s2s_probabilistic_bridge_imerg_sensitivity_v1"
REFERENCE = "imerg"
REFERENCE_LABEL = "IMERG"
TRACK = "tp_imerg"
TARGET_DAY_OFFSETS = tuple(range(42))
LAST_ALLOWED_SOURCE_DATE = np.datetime64("2024-12-31", "D")
EXPECTED_MAXIMUM_TARGET_DATE = np.datetime64("2024-12-29", "D")
EXPECTED_BENCHMARK_CASES_SHA256 = (
    "6f3c6433de533a319f38523a88c9f19f6b376a03f1b3bde67ec36a59fe8dd26a"
)
EXPECTED_DATA_CONFIG_SHA256 = (
    "1d38fdd3c3e651db458d33f75ccb92e218ab61f3365a957a43faf5f923202825"
)
EXPECTED_EVALUATION_CONFIG_SHA256 = (
    "002d943ce673e2c620a89d81cca18eaf1204ae8afafbf51d0e8acbc8b9770721"
)
EXPECTED_INFERENCE_MANIFEST_SHA256 = (
    "c3d4fb825e8549a00ee393e3b522d8b8e9bb289338bbea383ea693bceb797eca"
)
EXPECTED_PREFLIGHT_MANIFEST_SHA256 = (
    "8c8df561ff347d17a8e865e455678c0d8e6d9b6b62e134a267f0d1180ba43e6a"
)
DEFAULT_INFERENCE_MANIFEST = (
    bridge.DEFAULT_OUTPUT_ROOT
    / "inference_full_20260825T233121Z"
    / "manifest.json"
)
DEFAULT_PREFLIGHT_MANIFEST = (
    bridge.DEFAULT_OUTPUT_ROOT
    / "preflight_20260825T225552Z"
    / "manifest.json"
)
DEFAULT_BENCHMARK_CASES = (
    bridge.BENCHMARK_CODE_ROOT
    / "results"
    / "imerg_era5_primary"
    / "tables"
    / "case_metrics.csv"
)


class IMERGSensitivityError(imd_score.BridgeScoringError):
    """Raised when the independent-reference contract would be violated."""


@dataclass(frozen=True)
class WeeklyIMERGTargets:
    """Period-start weekly IMERG truth, normal, support, and target labels."""

    truth: np.ndarray
    climatology: np.ndarray
    observation_support: np.ndarray
    target_dates: np.ndarray


@dataclass(frozen=True)
class IMERGTruthBundle:
    """The only truth-bearing object constructed by this evaluator."""

    weekly: WeeklyIMERGTargets
    region_weights: Mapping[str, np.ndarray]
    latitude: np.ndarray
    longitude: np.ndarray
    receipt: Mapping[str, Any]


def imerg_target_dates(initializations: Iterable[Any]) -> np.ndarray:
    """Return period-start labels shaped ``[case, 6, 7]`` at offsets 0..41."""

    inits = np.asarray(initializations, dtype="datetime64[D]")
    if inits.ndim != 1 or inits.size == 0:
        raise IMERGSensitivityError("initializations must be a non-empty 1-D array")
    if np.unique(inits).size != inits.size or np.any(
        np.diff(inits) <= np.timedelta64(0, "D")
    ):
        raise IMERGSensitivityError(
            "initializations must be unique and strictly increasing"
        )
    offsets = np.asarray(TARGET_DAY_OFFSETS, dtype="timedelta64[D]").reshape(
        1, 6, 7
    )
    return inits[:, None, None] + offsets


def weekly_imerg_targets(
    initializations: Iterable[Any],
    daily_dates: Iterable[Any],
    daily_values: np.ndarray,
    daily_observation_fraction: np.ndarray,
    imerg_daily_climatology_2001_2019: np.ndarray,
    *,
    last_allowed_source_date: np.datetime64 = LAST_ALLOWED_SOURCE_DATE,
) -> WeeklyIMERGTargets:
    """Build coverage-weighted IMERG weeks and minimum seven-day support."""

    targets = imerg_target_dates(initializations)
    dates = np.asarray(daily_dates, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise IMERGSensitivityError("IMERG daily dates must be non-empty and 1-D")
    if np.unique(dates).size != dates.size or np.any(
        np.diff(dates) <= np.timedelta64(0, "D")
    ):
        raise IMERGSensitivityError("IMERG daily dates must be unique and sorted")
    cutoff = np.datetime64(last_allowed_source_date, "D")
    if dates.max() > cutoff:
        raise IMERGSensitivityError("IMERG daily input crosses the 2024 firewall")
    if targets[:, -1, -1].max() > cutoff:
        raise IMERGSensitivityError("requested IMERG target crosses the 2024 firewall")

    values = np.asarray(daily_values)
    fraction = np.asarray(daily_observation_fraction)
    if values.ndim < 2 or values.shape[0] != len(dates):
        raise IMERGSensitivityError("daily values must be [date, ...field]")
    field_shape = values.shape[1:]
    if fraction.shape == field_shape:
        fraction = np.broadcast_to(fraction, values.shape)
    if fraction.shape != values.shape:
        raise IMERGSensitivityError("daily IMERG fractions do not match values")
    if not np.isfinite(fraction).all() or np.any(
        (fraction < 0.0) | (fraction > 1.0)
    ):
        raise IMERGSensitivityError("daily IMERG fractions must be finite in [0,1]")
    represented = fraction > 0.0
    if not np.isfinite(values[represented]).all():
        raise IMERGSensitivityError("represented IMERG values are non-finite")

    normal = np.asarray(imerg_daily_climatology_2001_2019)
    if normal.shape != (366,) + field_shape:
        raise IMERGSensitivityError(
            "IMERG 2001--2019 climatology must be [366, ...field]"
        )
    positions = pd.Index(dates).get_indexer(targets.reshape(-1))
    if np.any(positions < 0):
        missing = targets.reshape(-1)[positions < 0][0]
        raise IMERGSensitivityError(
            "IMERG target date is unavailable: "
            + np.datetime_as_string(missing, unit="D")
        )
    shape = (len(targets), 6, 7) + field_shape
    selected_values = values[positions].reshape(shape).astype(np.float64, copy=False)
    selected_fraction = (
        fraction[positions].reshape(shape).astype(np.float64, copy=False)
    )
    denominator = selected_fraction.sum(axis=2, dtype=np.float64)
    numerator = (
        np.nan_to_num(selected_values, nan=0.0) * selected_fraction
    ).sum(axis=2, dtype=np.float64)
    truth = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    support = selected_fraction.min(axis=2)

    normal_days = scoring.fixed_climatology_day(targets.reshape(-1)) - 1
    selected_normal = normal[normal_days].reshape(shape).astype(np.float64, copy=False)
    finite_count = np.isfinite(selected_normal).sum(axis=2)
    weekly_normal = np.divide(
        np.nansum(selected_normal, axis=2, dtype=np.float64),
        finite_count,
        out=np.full(finite_count.shape, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    return WeeklyIMERGTargets(
        truth=truth.astype(np.float32),
        climatology=weekly_normal.astype(np.float32),
        observation_support=support.astype(np.float32),
        target_dates=targets,
    )


def _catalog_input_paths(loader: Any) -> tuple[Path, ...]:
    """Return the catalog/config files whose content selects every data store."""

    config = bridge.BENCHMARK_DATA_CONFIG.resolve()
    archive = Path(loader.catalog.archive_root).resolve()
    keys = ("forecast_catalog", "forecast_climatology_catalog", "observation_catalog")
    return (config, bridge.BENCHMARK_EVALUATION_CONFIG.resolve()) + tuple(
        (archive / str(loader.catalog.config[key])).resolve() for key in keys
    )


def load_imerg_truth(
    validated: imd_score.ValidatedInference,
    loader: Any,
) -> IMERGTruthBundle:
    """Open only IMERG 2020--2024 and construct exact period-start targets."""

    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for year in scoring.FORECAST_YEARS:
        records = loader.list_observations(source=REFERENCE, variable="tp", year=year)
        if len(records) != 1:
            raise IMERGSensitivityError(f"expected one IMERG record for {year}")
        record = records.iloc[0].to_dict()
        store = Path(str(record["store"]))
        if store.name != f"{year}.zarr" or year == 2025:
            raise IMERGSensitivityError(
                f"IMERG store escapes the 2020--2024 firewall: {store}"
            )
        source = {
            "year": year,
            **imd_score._catalog_record_hash(record, label=f"IMERG {year}"),
        }
        dataset = loader.open_observation_dataset(
            source=REFERENCE, variable="tp", year=year
        )
        try:
            current_dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            current_values = np.asarray(
                dataset.observation.load().values, dtype=np.float32
            )
            current_fraction = np.asarray(
                dataset.observation_fraction.load().values, dtype=np.float32
            )
            if not np.array_equal(
                dataset.latitude.values, validated.context.latitude
            ):
                raise IMERGSensitivityError(f"IMERG {year} latitude differs")
            if not np.array_equal(
                dataset.longitude.values, validated.context.longitude
            ):
                raise IMERGSensitivityError(f"IMERG {year} longitude differs")
        finally:
            dataset.close()
        if (
            current_dates.min() != np.datetime64(f"{year}-01-01")
            or current_dates.max() != np.datetime64(f"{year}-12-31")
        ):
            raise IMERGSensitivityError(f"IMERG {year} daily calendar is incomplete")
        if current_fraction.ndim == 2:
            current_fraction = np.broadcast_to(
                current_fraction, current_values.shape
            ).copy()
        if current_fraction.shape != current_values.shape:
            raise IMERGSensitivityError(
                f"IMERG {year} observation fraction shape differs"
            )
        dates.append(current_dates)
        values.append(current_values)
        fractions.append(current_fraction)
        sources.append(source)

    climate_records = [
        item
        for item in loader.catalog.observation_climatologies
        if item.get("source") == REFERENCE and item.get("baseline") == [2001, 2019]
    ]
    if len(climate_records) != 1:
        raise IMERGSensitivityError("IMERG 2001--2019 normal is ambiguous")
    climate_source = imd_score._catalog_record_hash(
        climate_records[0], label="IMERG 2001--2019 climatology"
    )
    climate_dataset = loader.open_observation_climatology_dataset(
        source=REFERENCE, baseline=(2001, 2019)
    )
    try:
        daily_climatology = np.asarray(
            climate_dataset.climatology_mean.load().values, dtype=np.float32
        )
        if not np.array_equal(
            climate_dataset.latitude.values, validated.context.latitude
        ):
            raise IMERGSensitivityError("IMERG climatology latitude differs")
        if not np.array_equal(
            climate_dataset.longitude.values, validated.context.longitude
        ):
            raise IMERGSensitivityError("IMERG climatology longitude differs")
    finally:
        climate_dataset.close()

    weekly = weekly_imerg_targets(
        validated.scoring_initializations,
        np.concatenate(dates),
        np.concatenate(values),
        np.concatenate(fractions),
        daily_climatology,
    )
    maximum_target = weekly.target_dates[:, -1, -1].max()
    if maximum_target != EXPECTED_MAXIMUM_TARGET_DATE:
        raise IMERGSensitivityError(
            f"maximum IMERG target differs: {maximum_target}"
        )

    support_dataset = loader.open_spatial_support()
    try:
        latitude = np.asarray(support_dataset.latitude.values, dtype=np.float64)
        longitude = np.asarray(support_dataset.longitude.values, dtype=np.float64)
        india = np.asarray(
            support_dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
        cell = np.asarray(
            support_dataset.cell_area_km2.load().values, dtype=np.float64
        )
        region_fractions = {
            name: np.asarray(
                support_dataset[f"{name}_fraction"].load().values,
                dtype=np.float64,
            )
            for name in scoring.REGION_ORDER[1:]
        }
    finally:
        support_dataset.close()
    if not np.array_equal(latitude, validated.context.latitude) or not np.array_equal(
        longitude, validated.context.longitude
    ):
        raise IMERGSensitivityError("spatial-support grid differs from inference")
    region_weights = scoring.build_region_base_weights(
        india, cell, region_fractions
    )
    support_store = (
        loader.catalog.archive_root / loader.catalog.config["spatial_support"]
    ).resolve()
    support_hash = bridge.sha256_file(support_store / ".zmetadata")
    receipt = {
        "source": REFERENCE,
        "variable": "tp",
        "label_convention": "UTC period-start",
        "opened_years": list(scoring.FORECAST_YEARS),
        "opened_2025": False,
        "daily_sources": sources,
        "climatology_baseline": [2001, 2019],
        "climatology_source": climate_source,
        "spatial_support_store": str(support_store),
        "spatial_support_zmetadata_sha256": support_hash,
        "target_day_offsets": list(TARGET_DAY_OFFSETS),
        "weekly_observation_support": "minimum of seven daily fractions",
        "scoreable_case_count": 505,
        "maximum_target_label": "2024-12-29",
        "maximum_source_label_opened": "2024-12-31",
        "sealed_2025_target_opened": False,
    }
    return IMERGTruthBundle(
        weekly=weekly,
        region_weights=region_weights,
        latitude=latitude,
        longitude=longitude,
        receipt=receipt,
    )


def _score_members(
    *,
    method: str,
    initializations: np.ndarray,
    members: np.ndarray,
    truth: WeeklyIMERGTargets,
    truth_indices: np.ndarray,
    climatology: scoring.ForecastClimatology,
    region_weights: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Score one method, replacing the generic scorer's IMD metadata."""

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
    probability = imd_score.probabilistic_case_rows(
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
    cases.insert(0, "track", TRACK)
    cases["reference"] = REFERENCE
    cases["reference_label"] = REFERENCE_LABEL
    cases["model"] = method
    cases["model_label"] = imd_score.METHOD_LABELS[method]
    cases["member_count"] = 50
    cases["selected_seed"] = (
        43 if method == imd_score.PRIMARY_METHOD else "not_applicable"
    )
    return cases


def score_all_methods(
    validated: imd_score.ValidatedInference,
    climates: Mapping[str, scoring.ForecastClimatology],
    moment_fit: calibration.MomentFit,
    truth: IMERGTruthBundle,
    *,
    batch_size: int,
) -> pd.DataFrame:
    """Reopen and re-hash raw members, then score three identical ensembles."""

    loader = inference._forecast_loader()
    truth_lookup = {
        value: index for index, value in enumerate(validated.scoring_initializations)
    }
    frames: list[pd.DataFrame] = []
    for year in scoring.FORECAST_YEARS:
        shard = imd_score._load_shard(validated, year)
        inits = np.asarray(shard["initializations"], dtype="datetime64[D]")
        scoreable = np.asarray(shard["scoreable_for_truth"], dtype=bool)
        expected_digest = imd_score._shard_receipts(validated)[year]["raw_source"][
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
                    raise IMERGSensitivityError(
                        f"FuXi {year} scoreable start lacks IMERG truth"
                    )
                truth_indices = np.asarray(
                    [truth_lookup[value] for value in batch_inits], dtype=np.int64
                )
                raw = raw_all[local]
                neural = imd_score.independent_neural_members(
                    raw,
                    shard["delta_log_location"][selected],
                    shard["log_spread"][selected],
                )
                moment = imd_score.independent_moment_members(
                    raw, batch_inits, moment_fit
                )
                members_by_method = {
                    "raw_fuxi": raw,
                    "moment_calibration": moment,
                    "location_spread": neural,
                }
                for method in imd_score.METHOD_ORDER:
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
                raise IMERGSensitivityError(
                    f"FuXi {year} raw content changed before IMERG scoring"
                )
        finally:
            dataset.close()
    cases = pd.concat(frames, ignore_index=True)
    expected_rows = len(imd_score.METHOD_ORDER) * scoring.RAW_IDENTITY_CASE_COUNT
    if len(cases) != expected_rows:
        raise IMERGSensitivityError(
            f"case table has {len(cases)} rows, expected {expected_rows}"
        )
    expected_by_method = {
        method: scoring.RAW_IDENTITY_CASE_COUNT for method in imd_score.METHOD_ORDER
    }
    if cases.groupby("method").size().to_dict() != expected_by_method:
        raise IMERGSensitivityError("method case counts differ")
    if set(cases.reference) != {REFERENCE} or set(cases.track) != {TRACK}:
        raise IMERGSensitivityError("case table observation-reference labels differ")
    return cases


def select_common_benchmark_raw(
    benchmark: pd.DataFrame,
    scoring_initializations: np.ndarray,
) -> pd.DataFrame:
    """Select the exact 505-start FuXi/IMERG subset without late-2024 cases."""

    allowed = {
        np.datetime_as_string(value, unit="D") for value in scoring_initializations
    }
    selected = benchmark[
        benchmark.model.eq("fuxi_s2s")
        & benchmark.track.eq(TRACK)
        & benchmark.reference.eq(REFERENCE)
        & benchmark.score_status.eq("available")
        & benchmark.init.isin(allowed)
    ].copy()
    if len(selected) != scoring.RAW_IDENTITY_CASE_COUNT:
        raise IMERGSensitivityError(
            f"IMERG benchmark common subset has {len(selected)} rows"
        )
    if selected.init.nunique() != 505:
        raise IMERGSensitivityError("IMERG benchmark common subset is not 505 starts")
    return selected


def validate_raw_identity(
    cases: pd.DataFrame,
    benchmark_path: Path,
    scoring_initializations: np.ndarray,
) -> Mapping[str, Any]:
    """Require ACC/RMSE/MAE/bias identity on all 15,150 common raw rows."""

    source = Path(benchmark_path).resolve()
    imd_score._require_hash(
        source, EXPECTED_BENCHMARK_CASES_SHA256, "frozen IMERG benchmark table"
    )
    benchmark = pd.read_csv(source)
    benchmark_raw = select_common_benchmark_raw(
        benchmark, scoring_initializations
    )
    benchmark_all_raw = benchmark[
        benchmark.model.eq("fuxi_s2s")
        & benchmark.track.eq(TRACK)
        & benchmark.reference.eq(REFERENCE)
        & benchmark.score_status.eq("available")
    ]
    if len(benchmark_all_raw) != 15_510 or benchmark_all_raw.init.nunique() != 517:
        raise IMERGSensitivityError("full frozen IMERG benchmark geometry differs")
    independent_raw = cases[cases.method.eq("raw_fuxi")].copy()
    receipt = scoring.assert_raw_fuxi_identity(independent_raw, benchmark_raw)
    return {
        **receipt,
        "reference": REFERENCE,
        "benchmark_total_fuxi_rows": int(len(benchmark_all_raw)),
        "excluded_late_2024_benchmark_rows": int(
            len(benchmark_all_raw) - len(benchmark_raw)
        ),
        "benchmark_case_metrics": str(source),
        "benchmark_case_metrics_sha256": EXPECTED_BENCHMARK_CASES_SHA256,
    }


def build_imerg_intervals(
    jjas_cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse paired draws, then bind every interval to IMERG explicitly."""

    intervals, receipts = imd_score.build_story_intervals(
        jjas_cases, replicates=replicates, seed=seed
    )
    pooled, pooled_receipts = imd_score.compute_pooled_w1_w6_story_intervals(
        jjas_cases, replicates=replicates, seed=seed
    )
    intervals = pd.concat([intervals, pooled], ignore_index=True)
    intervals["reference"] = REFERENCE
    receipts[
        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
    ] = pooled_receipts
    return intervals, receipts


def _scoring_sources() -> tuple[Path, ...]:
    sources = (
        Path(__file__).resolve(),
        Path(imd_score.__file__).resolve(),
        Path(scoring.__file__).resolve(),
        Path(inference.__file__).resolve(),
        Path(bridge.__file__).resolve(),
        Path(calibration.__file__).resolve(),
        Path(
            imd_score.circular_year_stratified_index_matrix.__code__.co_filename
        ).resolve(),
    )
    if len({path.name for path in sources}) != len(sources):
        raise IMERGSensitivityError("source snapshot basenames are ambiguous")
    return sources


def _snapshot_sources(staging: Path) -> dict[str, str]:
    for source in _scoring_sources():
        destination = staging / "code" / "src" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        str(path.relative_to(staging)): bridge.sha256_file(path)
        for path in sorted((staging / "code" / "src").glob("*.py"))
    }


def _verify_source_snapshot(staging: Path, hashes: Mapping[str, str]) -> None:
    live = {path.name: path for path in _scoring_sources()}
    for relative, expected in hashes.items():
        snapshot = staging / relative
        imd_score._require_hash(snapshot, expected, "IMERG source snapshot")
        source = live.get(snapshot.name)
        if source is None or bridge.sha256_file(source) != expected:
            raise IMERGSensitivityError(
                f"IMERG scoring source changed during run: {snapshot.name}"
            )


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package in ("xarray", "zarr", "numcodecs", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _truth_input_paths(receipt: Mapping[str, Any]) -> tuple[Path, ...]:
    paths = [
        Path(str(item["store"])) / ".zmetadata"
        for item in receipt["daily_sources"]
    ]
    paths.append(Path(str(receipt["climatology_source"]["store"])) / ".zmetadata")
    paths.append(Path(str(receipt["spatial_support_store"])) / ".zmetadata")
    return tuple(path.resolve() for path in paths)


def publish_sensitivity(
    *,
    inference_manifest: Path,
    preflight_manifest: Path,
    benchmark_case_metrics: Path,
    output: Path,
    batch_size: int = 8,
    replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> Path:
    """Run the complete IMERG sensitivity and publish atomically."""

    destination = Path(output).resolve()
    imd_score._assert_bridge_output(destination)
    if destination.exists():
        raise FileExistsError(f"fresh output path required: {destination}")
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    truth_opened = False
    try:
        source_hashes = _snapshot_sources(staging)
        exact_inputs = {
            Path(inference_manifest).resolve(): EXPECTED_INFERENCE_MANIFEST_SHA256,
            Path(preflight_manifest).resolve(): EXPECTED_PREFLIGHT_MANIFEST_SHA256,
            Path(benchmark_case_metrics).resolve(): EXPECTED_BENCHMARK_CASES_SHA256,
            bridge.BENCHMARK_DATA_CONFIG.resolve(): EXPECTED_DATA_CONFIG_SHA256,
            bridge.BENCHMARK_EVALUATION_CONFIG.resolve(): (
                EXPECTED_EVALUATION_CONFIG_SHA256
            ),
        }
        for path, expected in exact_inputs.items():
            imd_score._require_hash(path, expected, "frozen IMERG sensitivity input")
        frozen_inputs = dict(exact_inputs)

        validated = imd_score.validate_inference_manifest(inference_manifest)
        preflight = imd_score.validate_preflight_manifest(
            preflight_manifest, validated=validated
        )
        if validated.parent.receipt.get("selected_seed") != 43:
            raise IMERGSensitivityError("inference parent is not selected seed 43")
        moment_fit = imd_score.load_moment_fit(validated)
        moment_path = imd_score._resolve_child(
            validated.parent.run_root, imd_score.MOMENT_FIT_RELATIVE
        )
        frozen_inputs[validated.parent.manifest_path] = bridge.sha256_file(
            validated.parent.manifest_path
        )
        frozen_inputs[moment_path] = bridge.sha256_file(moment_path)

        climates, _, forecast_sources = imd_score.build_forecast_climatologies(
            validated, moment_fit, batch_size=batch_size
        )
        loader = inference._forecast_loader()
        for path in _catalog_input_paths(loader):
            frozen_inputs.setdefault(path, bridge.sha256_file(path))
        truth = load_imerg_truth(validated, loader)
        truth_opened = True
        for path in _truth_input_paths(truth.receipt):
            frozen_inputs.setdefault(path, bridge.sha256_file(path))

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
        allseason = imd_score.add_raw_comparisons(
            imd_score.aggregate_case_metrics(cases)
        )
        jjas = cases[cases.season.eq("JJAS")].copy()
        expected_jjas_rows = len(imd_score.METHOD_ORDER) * 5 * 6 * 169
        if len(jjas) != expected_jjas_rows:
            raise IMERGSensitivityError(
                f"valid-midpoint JJAS has {len(jjas)} rows, expected {expected_jjas_rows}"
            )
        jjas_summary = imd_score.add_raw_comparisons(
            imd_score.aggregate_case_metrics(jjas)
        )
        if set(jjas_summary.case_count) != {169}:
            raise IMERGSensitivityError(
                "valid-midpoint JJAS is not exactly 169 cases per lead"
            )
        intervals, resampling = build_imerg_intervals(
            jjas, replicates=replicates, seed=bootstrap_seed
        )
        if set(intervals.reference) != {REFERENCE}:
            raise IMERGSensitivityError("paired intervals are not bound to IMERG")

        tables = staging / "tables"
        receipts = staging / "receipts"
        tables.mkdir(parents=True)
        receipts.mkdir(parents=True)
        cases.to_csv(tables / "case_metrics.csv", index=False)
        allseason.to_csv(tables / "allseason_summary.csv", index=False)
        jjas_summary.to_csv(
            tables / "jjas_valid_midpoint_summary.csv", index=False
        )
        intervals.to_csv(tables / "paired_block_intervals.csv", index=False)
        bridge.write_json(receipts / "raw_fuxi_benchmark_identity.json", raw_identity)
        bridge.write_json(receipts / "truth_access.json", truth.receipt)
        bridge.write_json(receipts / "cohort.json", validated.cohort)
        bridge.write_json(receipts / "bootstrap_sampling.json", resampling)

        _verify_source_snapshot(staging, source_hashes)
        imd_score._verify_unchanged_inputs(frozen_inputs)
        manifest: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "scientific_status": "unfitted observation-reference sensitivity",
            "created_utc": bridge.utc_now(),
            "output_path": str(destination),
            "input_inference_manifest": str(validated.manifest_path),
            "input_inference_manifest_sha256": EXPECTED_INFERENCE_MANIFEST_SHA256,
            "input_preflight_manifest": str(Path(preflight_manifest).resolve()),
            "input_preflight_manifest_sha256": EXPECTED_PREFLIGHT_MANIFEST_SHA256,
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
            "reference": {
                "source": REFERENCE,
                "role": "unfitted independent observation sensitivity",
                "label_convention": "UTC period-start",
                "climatology_baseline": [2001, 2019],
            },
            "cohort": {
                "forecast_only_climatology_count": 517,
                "scoring_count": 505,
                "valid_midpoint_jjas_count_per_lead": 169,
                "target_day_offsets": list(TARGET_DAY_OFFSETS),
                "maximum_target_label_opened": "2024-12-29",
                "benchmark_late_2024_cases_requiring_2025_truth_excluded": 12,
            },
            "methods": list(imd_score.METHOD_ORDER),
            "member_count_each_method": 50,
            "metrics": list(imd_score.CASE_METRICS),
            "aggregation": (
                "arithmetic mean of per-initialization spatial scores; "
                "spread-skill pooled from mean variance and mean squared error"
            ),
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
                    "W1-W6 synchronizes within-year ordinal draws across leads"
                ),
                "block_lengths_starts": list(imd_score.BLOCK_LENGTHS),
                "replicates": int(replicates),
                "seed": int(bootstrap_seed),
                "comparisons": [list(values) for values in imd_score.COMPARISON_PAIRS],
            },
            "table_rows": {
                "case_metrics": int(len(cases)),
                "allseason_summary": int(len(allseason)),
                "jjas_valid_midpoint_summary": int(len(jjas_summary)),
                "paired_block_intervals": int(len(intervals)),
            },
            "preflight_parent_receipt": preflight["parent_receipt"],
            "configuration_sha256": {
                str(bridge.BENCHMARK_DATA_CONFIG.resolve()): (
                    EXPECTED_DATA_CONFIG_SHA256
                ),
                str(bridge.BENCHMARK_EVALUATION_CONFIG.resolve()): (
                    EXPECTED_EVALUATION_CONFIG_SHA256
                ),
            },
            "software_versions": _software_versions(),
            "source_snapshot_sha256": source_hashes,
            "opened_observation_years": list(scoring.FORECAST_YEARS),
            "opened_2025_observation": False,
            "sealed_2025_target_opened": False,
        }
        manifest["artifact_sha256"] = imd_score._artifact_hashes(staging)
        bridge.write_json(staging / "manifest.json", manifest)
        _verify_source_snapshot(staging, source_hashes)
        imd_score._verify_unchanged_inputs(frozen_inputs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        imd_score.rename_noreplace(staging, destination)
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
    parser.add_argument(
        "--inference-manifest", type=Path, default=DEFAULT_INFERENCE_MANIFEST
    )
    parser.add_argument(
        "--preflight-manifest", type=Path, default=DEFAULT_PREFLIGHT_MANIFEST
    )
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
    manifest = publish_sensitivity(
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
