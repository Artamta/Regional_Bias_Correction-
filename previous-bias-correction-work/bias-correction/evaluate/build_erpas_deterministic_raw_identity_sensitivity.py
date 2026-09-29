#!/usr/bin/env python3
"""Build the strict-cohort ERPAS--FuXi raw-identity sensitivity artifact.

This generator intentionally combines forecast fields, not previously published
scores.  ERPAS, raw FuXi-S2S, and the frozen deterministic raw-identity adapter
are rescored on one 26-start cohort shared by Weeks 1--4 under the verification
contract of the existing ERPAS comparison.  The existing ERPAS case table is
used only as a source-bound cohort definition and an independent score check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1"
)
ERPAS_ROOT = (
    ARCHIVE_ROOT
    / "forecasts/erpas/provider__erpas_forecast_2023_2025/tp/common_1p5"
)
IMD_ROOT = (
    ARCHIVE_ROOT
    / "observations/ground_truth_v1/daily/imd/tp/india_1p5_27x27_v1"
)
IMD_CLIMATOLOGY = (
    ARCHIVE_ROOT
    / "observations/ground_truth_v1/climatologies/imd_1991_2019.zarr"
)
SPATIAL_SUPPORT = ARCHIVE_ROOT / "spatial/spatial_support.zarr"
DETERMINISTIC_RUN = (
    PROJECT_ROOT
    / "resultsv2/fuxi_imd_raw_identity_2022_2024_audit/"
    "canonical_circular_20260822T0225Z"
)
ERPAS_SCORE_RUN = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_erpas_corrected_acc_rmse_sensitivity/"
    "full_v1_20260827T024543Z"
)

EXPECTED_SOURCE_HASHES = {
    "deterministic_manifest": (
        "bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec"
    ),
    "deterministic_predictions_zmetadata": (
        "9455fd48d56a5c75215005a657e02a83281f0b960960dc67f884fa8ac801e95e"
    ),
    "deterministic_predictions_tree": (
        "0eb2f2654bf3b3efd71c757bcecf2570eb5ce4f6b557aff9741310e4c4b13c66"
    ),
    "erpas_score_manifest": (
        "da79779d52a994aa419896f533b29af81ffda6d77efff1de246071512b6b08ac"
    ),
    "erpas_case_table": (
        "3e9580e40f7202fa6cc5d9ba6be57807b7d79110120b335b627278858c7f87f0"
    ),
    "imd_climatology_zmetadata": (
        "b942292806b52e3bb8180ebc1cbcc54bdfeb98b10a8e918d671732fc2eb00228"
    ),
    "spatial_support_zmetadata": (
        "07bb0e60a396a6056df0cea9c3b96861aeb5fe0f1db9640173b9e166306cfbe4"
    ),
    "erpas_2023_zmetadata": (
        "eb04fe079897fa9e24517df7470a18a820414f186e88faf1648c008d7a523528"
    ),
    "erpas_2024_zmetadata": (
        "2bce65036b41565065eb1839a8786cbd7e36e476c0532dba3ce051fb6b27b23f"
    ),
    "imd_2023_zmetadata": (
        "18c84da11b6fcb95ae225c17dc335b5cb4699cd97c10b81cb759a835fb2be473"
    ),
    "imd_2024_zmetadata": (
        "451ee0a1a1755e3c829ac482cca990743b1eaf04f1ed79318968552f0ef550fb"
    ),
}

YEARS = (2023, 2024)
LEADS = (1, 2, 3, 4)
METHODS = ("erpas", "raw_fuxi", "raw_identity")
LABELS = {
    "erpas": "ERPAS",
    "raw_fuxi": "Raw FuXi-S2S",
    "raw_identity": "Deterministic raw-identity adapter",
}
EXPECTED_SHARED_DATES = (
    "2023-06-01",
    "2023-06-08",
    "2023-06-15",
    "2023-06-22",
    "2023-06-29",
    "2023-07-06",
    "2023-07-13",
    "2023-07-20",
    "2023-07-27",
    "2023-08-03",
    "2023-08-10",
    "2023-08-17",
    "2023-08-24",
    "2023-08-31",
    "2024-06-20",
    "2024-06-27",
    "2024-07-04",
    "2024-07-11",
    "2024-07-18",
    "2024-07-25",
    "2024-08-01",
    "2024-08-08",
    "2024-08-15",
    "2024-08-22",
    "2024-08-29",
    "2024-09-05",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_826
BLOCK_LENGTH = 4
SOURCE_SCORE_TOLERANCE = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh final run directory; existing paths are never modified.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a directory exactly as the deterministic source manifest does."""
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def fixed_climatology_day(date: np.datetime64) -> int:
    value = pd.Timestamp(date)
    return pd.Timestamp(2000, value.month, value.day).dayofyear - 1


def strict_shared_dates(
    reference_cases: pd.DataFrame,
    deterministic_initializations: np.ndarray,
) -> tuple[np.datetime64, ...]:
    """Return FuXi starts present for every requested lead and both sources."""
    deterministic = set(
        np.asarray(deterministic_initializations, dtype="datetime64[D]").tolist()
    )
    per_lead: list[set[np.datetime64]] = []
    for lead in LEADS:
        values = np.asarray(
            reference_cases.loc[
                reference_cases.lead_week == lead, "fuxi_init"
            ].unique(),
            dtype="datetime64[D]",
        )
        per_lead.append(set(values.tolist()) & deterministic)
    shared = tuple(sorted(set.intersection(*per_lead)))
    return tuple(np.datetime64(value, "D") for value in shared)


def weekly_truth_and_normal(
    truth_dataset: xr.Dataset,
    daily_climatology: np.ndarray,
    fuxi_init: np.datetime64,
    lead_week: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fraction-weighted weekly IMD truth and the common daily normal."""
    first_offset = 1 + 7 * (lead_week - 1)
    dates = fuxi_init + np.arange(first_offset, first_offset + 7).astype(
        "timedelta64[D]"
    )
    values = np.asarray(
        truth_dataset.observation.sel(time=dates).values, dtype=np.float64
    )
    fraction_da = truth_dataset.observation_fraction
    if "time" in fraction_da.dims:
        fraction = np.asarray(fraction_da.sel(time=dates).values, dtype=np.float64)
    else:
        fraction = np.broadcast_to(
            np.asarray(fraction_da.values, dtype=np.float64), values.shape
        )
    denominator = fraction.sum(axis=0, dtype=np.float64)
    truth = np.divide(
        np.nansum(values * fraction, axis=0, dtype=np.float64),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    weekly_support = fraction.min(axis=0)
    normal_daily = np.asarray(
        daily_climatology[[fixed_climatology_day(date) for date in dates]],
        dtype=np.float64,
    )
    normal_count = np.isfinite(normal_daily).sum(axis=0)
    normal = np.divide(
        np.nansum(normal_daily, axis=0, dtype=np.float64),
        normal_count,
        out=np.full(normal_count.shape, np.nan, dtype=np.float64),
        where=normal_count > 0,
    )
    return truth, normal, weekly_support, dates


def weighted_spatial_acc(
    forecast: np.ndarray,
    truth: np.ndarray,
    normal: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int, float]:
    forecast_anomaly = np.asarray(forecast, dtype=np.float64) - normal
    truth_anomaly = np.asarray(truth, dtype=np.float64) - normal
    represented = (
        (weights > 0.0)
        & np.isfinite(forecast_anomaly)
        & np.isfinite(truth_anomaly)
    )
    if int(represented.sum()) < 3:
        raise ValueError("ACC requires at least three common supported cells")
    spatial = np.where(represented, weights, 0.0)
    total = spatial.sum(dtype=np.float64)
    forecast_mean = np.sum(
        spatial * np.where(represented, forecast_anomaly, 0.0), dtype=np.float64
    ) / total
    truth_mean = np.sum(
        spatial * np.where(represented, truth_anomaly, 0.0), dtype=np.float64
    ) / total
    forecast_centered = np.where(
        represented, forecast_anomaly - forecast_mean, 0.0
    )
    truth_centered = np.where(represented, truth_anomaly - truth_mean, 0.0)
    covariance = np.sum(
        spatial * forecast_centered * truth_centered, dtype=np.float64
    )
    forecast_variance = np.sum(
        spatial * forecast_centered**2, dtype=np.float64
    )
    truth_variance = np.sum(spatial * truth_centered**2, dtype=np.float64)
    denominator = np.sqrt(forecast_variance * truth_variance)
    if denominator <= 0.0:
        raise ValueError("ACC denominator is zero")
    return float(covariance / denominator), int(represented.sum()), float(total)


def weighted_spatial_rmse(
    forecast: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int, float]:
    forecast_values = np.asarray(forecast, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    represented = (
        (weights > 0.0)
        & np.isfinite(forecast_values)
        & np.isfinite(truth_values)
    )
    if int(represented.sum()) < 1:
        raise ValueError("RMSE requires at least one common supported cell")
    spatial = np.where(represented, weights, 0.0)
    total = spatial.sum(dtype=np.float64)
    squared_error = np.where(
        represented, (forecast_values - truth_values) ** 2, 0.0
    )
    rmse = np.sqrt(np.sum(spatial * squared_error, dtype=np.float64) / total)
    return float(rmse), int(represented.sum()), float(total)


def circular_block_indices(
    size: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size < 1 or block_length < 1:
        raise ValueError("bootstrap size and block length must be positive")
    starts = rng.integers(0, size, size=int(np.ceil(size / block_length)))
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % size).reshape(-1)[:size]


def make_bootstrap_indices(dates: tuple[np.datetime64, ...]) -> np.ndarray:
    """Create lead-specific indices paired across methods and both metrics."""
    years = np.asarray([pd.Timestamp(date).year for date in dates], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = np.empty(
        (len(LEADS), BOOTSTRAP_DRAWS, len(dates)), dtype=np.int16
    )
    for lead_index, _lead in enumerate(LEADS):
        for draw in range(BOOTSTRAP_DRAWS):
            selected: list[int] = []
            for year in YEARS:
                positions = np.flatnonzero(years == year)
                local = circular_block_indices(len(positions), BLOCK_LENGTH, rng)
                selected.extend(positions[local].tolist())
            output[lead_index, draw] = np.asarray(selected, dtype=np.int16)
    return output


def summarize_metric(
    case_metrics: pd.DataFrame,
    bootstrap_indices: np.ndarray,
    *,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    comparisons = (("raw_identity", "raw_fuxi"), ("raw_identity", "erpas"))
    for lead_index, lead in enumerate(LEADS):
        subset = case_metrics.loc[case_metrics.lead_week == lead]
        wide = subset.pivot(
            index=["year", "erpas_init", "fuxi_init"],
            columns="method",
            values=value_column,
        ).sort_index()
        wide = wide.loc[:, METHODS]
        if len(wide) != len(EXPECTED_SHARED_DATES) or wide.isna().any().any():
            raise ValueError(f"incomplete strict cohort at Week {lead}")
        values = wide.to_numpy(dtype=np.float64)
        samples = values[bootstrap_indices[lead_index]].mean(axis=1)
        for method_index, method in enumerate(METHODS):
            lower, upper = np.percentile(samples[:, method_index], (2.5, 97.5))
            absolute_rows.append(
                {
                    "metric": value_column,
                    "lead_week": lead,
                    "method": method,
                    "method_label": LABELS[method],
                    "n_cases": len(wide),
                    "estimate": float(values[:, method_index].mean()),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                }
            )
        for first, second in comparisons:
            first_index = METHODS.index(first)
            second_index = METHODS.index(second)
            paired = values[:, first_index] - values[:, second_index]
            paired_samples = samples[:, first_index] - samples[:, second_index]
            lower, upper = np.percentile(paired_samples, (2.5, 97.5))
            estimate = float(paired.mean())
            favorable = lower > 0.0 if value_column == "acc" else upper < 0.0
            effect_rows.append(
                {
                    "metric": value_column,
                    "comparison": f"{first}_minus_{second}",
                    "first_method": first,
                    "second_method": second,
                    "lead_week": lead,
                    "n_cases": len(wide),
                    "estimate": estimate,
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "favorable_direction": (
                        "positive" if value_column == "acc" else "negative"
                    ),
                    "resolved_favorable": bool(favorable),
                }
            )
    return pd.DataFrame(absolute_rows), pd.DataFrame(effect_rows)


def verify_source_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "deterministic_manifest": DETERMINISTIC_RUN / "manifest.json",
        "deterministic_predictions_zmetadata": (
            DETERMINISTIC_RUN / "predictions.zarr/.zmetadata"
        ),
        "erpas_score_manifest": ERPAS_SCORE_RUN / "manifest.json",
        "erpas_case_table": ERPAS_SCORE_RUN / "tables/case_acc.csv",
        "imd_climatology_zmetadata": IMD_CLIMATOLOGY / ".zmetadata",
        "spatial_support_zmetadata": SPATIAL_SUPPORT / ".zmetadata",
        "erpas_2023_zmetadata": ERPAS_ROOT / "2023.zarr/.zmetadata",
        "erpas_2024_zmetadata": ERPAS_ROOT / "2024.zarr/.zmetadata",
        "imd_2023_zmetadata": IMD_ROOT / "2023.zarr/.zmetadata",
        "imd_2024_zmetadata": IMD_ROOT / "2024.zarr/.zmetadata",
    }
    verified = {
        label: {
            "path": str(path.resolve()),
            "sha256": require_hash(path, EXPECTED_SOURCE_HASHES[label], label),
        }
        for label, path in paths.items()
    }
    deterministic_manifest = json.loads(
        (DETERMINISTIC_RUN / "manifest.json").read_text(encoding="utf-8")
    )
    bound_tree_hash = deterministic_manifest["artifacts"]["predictions.zarr"]
    expected_tree_hash = EXPECTED_SOURCE_HASHES["deterministic_predictions_tree"]
    if bound_tree_hash != expected_tree_hash:
        raise ValueError(
            "deterministic manifest artifacts.predictions.zarr binding changed"
        )
    prediction_store = DETERMINISTIC_RUN / "predictions.zarr"
    actual_tree_hash = sha256_tree(prediction_store)
    if actual_tree_hash != bound_tree_hash:
        raise ValueError(
            "deterministic predictions.zarr full-tree hash differs from its manifest"
        )
    verified["deterministic_predictions_tree"] = {
        "path": str(prediction_store.resolve()),
        "sha256": actual_tree_hash,
        "hash_contract": (
            "SHA-256 over sorted relative file path, NUL separator, then file bytes"
        ),
        "bound_by_manifest_key": "artifacts.predictions.zarr",
    }
    return verified


def build_case_metrics() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    reference = pd.read_csv(ERPAS_SCORE_RUN / "tables/case_acc.csv")
    reference_raw = reference.loc[reference.method == "raw_fuxi"].copy()
    deterministic = xr.open_zarr(
        DETERMINISTIC_RUN / "predictions.zarr", consolidated=True
    )
    climate = xr.open_zarr(IMD_CLIMATOLOGY, consolidated=True)
    support = xr.open_zarr(SPATIAL_SUPPORT, consolidated=True)
    try:
        if tuple(str(value) for value in deterministic.method.values) != (
            "raw_fuxi",
            "log_bias",
            "legacy_anchored_adapter",
            "raw_identity",
            "raw_identity_raw_mean_preserved",
        ):
            raise ValueError("deterministic method coordinate changed")
        dates = strict_shared_dates(reference_raw, deterministic.init.values)
        date_strings = tuple(np.datetime_as_string(value, unit="D") for value in dates)
        if date_strings != EXPECTED_SHARED_DATES:
            raise ValueError(
                "strict shared cohort changed:\n"
                f"expected {EXPECTED_SHARED_DATES}\nactual {date_strings}"
            )
        year_counts = pd.Series(
            [pd.Timestamp(value).year for value in dates]
        ).value_counts().sort_index().to_dict()
        if year_counts != {2023: 14, 2024: 12}:
            raise ValueError(f"strict cohort year counts changed: {year_counts}")

        latitude = np.asarray(support.latitude.values, dtype=np.float64)
        longitude = np.asarray(support.longitude.values, dtype=np.float64)
        if not np.array_equal(deterministic.latitude.values, latitude) or not np.array_equal(
            deterministic.longitude.values, longitude
        ):
            raise ValueError("deterministic forecast and support grids differ")
        india_weight = np.asarray(
            support.india_area_weight_km2.load().values, dtype=np.float64
        )
        adapter_support = np.asarray(
            deterministic.adapter_support.load().values, dtype=bool
        )
        if np.any(adapter_support & ~(india_weight > 0.0)):
            raise ValueError("deterministic support extends outside common India weights")
        if int(adapter_support.sum()) != 171:
            raise ValueError("represented India support changed from 171 cells")
        daily_climatology = np.asarray(
            climate.climatology_mean.load().values, dtype=np.float64
        )

        rows: list[dict[str, Any]] = []
        maximum_target = np.datetime64("1900-01-01", "D")
        for year in YEARS:
            erpas = xr.open_zarr(ERPAS_ROOT / f"{year}.zarr", consolidated=True)
            truth = xr.open_zarr(IMD_ROOT / f"{year}.zarr", consolidated=True)
            try:
                if erpas.attrs.get("distribution_representation") != "mean_only":
                    raise ValueError("ERPAS source is not the provider mean product")
                if not np.array_equal(erpas.latitude.values, latitude) or not np.array_equal(
                    erpas.longitude.values, longitude
                ):
                    raise ValueError("ERPAS and common support grids differ")
                year_dates = [date for date in dates if pd.Timestamp(date).year == year]
                for case_index, fuxi_init in enumerate(dates, start=1):
                    if fuxi_init not in year_dates:
                        continue
                    erpas_init = fuxi_init - np.timedelta64(1, "D")
                    erpas_field = np.asarray(
                        erpas.ensemble_mean.sel(
                            init=erpas_init, lead_day=np.arange(2, 30)
                        ).load().values,
                        dtype=np.float64,
                    ).reshape(4, 7, len(latitude), len(longitude)).mean(axis=1)
                    deterministic_field = np.asarray(
                        deterministic.prediction.sel(
                            {
                                "init": fuxi_init,
                                "method": ["raw_fuxi", "raw_identity"],
                                "lead_week": list(LEADS),
                            }
                        ).load().values,
                        dtype=np.float64,
                    )
                    if deterministic_field.shape != (2, 4, 27, 27):
                        raise ValueError("unexpected deterministic forecast shape")
                    for lead in LEADS:
                        weekly_truth, normal, observation_support, target_dates = (
                            weekly_truth_and_normal(
                                truth, daily_climatology, fuxi_init, lead
                            )
                        )
                        maximum_target = max(maximum_target, target_dates.max())
                        weights = india_weight * observation_support
                        fields = {
                            "erpas": erpas_field[lead - 1],
                            "raw_fuxi": deterministic_field[0, lead - 1],
                            "raw_identity": deterministic_field[1, lead - 1],
                        }
                        for method in METHODS:
                            acc, cells, effective_area = weighted_spatial_acc(
                                fields[method], weekly_truth, normal, weights
                            )
                            rmse, rmse_cells, rmse_area = weighted_spatial_rmse(
                                fields[method], weekly_truth, weights
                            )
                            if cells != rmse_cells or not np.isclose(
                                effective_area, rmse_area, rtol=0.0, atol=1.0e-8
                            ):
                                raise ValueError("ACC and RMSE supports differ")
                            rows.append(
                                {
                                    "cohort_case_index": case_index,
                                    "year": year,
                                    "erpas_init": np.datetime_as_string(
                                        erpas_init, unit="D"
                                    ),
                                    "fuxi_init": np.datetime_as_string(
                                        fuxi_init, unit="D"
                                    ),
                                    "valid_period_start": np.datetime_as_string(
                                        fuxi_init
                                        + np.timedelta64(7 * (lead - 1), "D"),
                                        unit="D",
                                    ),
                                    "valid_period_midpoint": (
                                        pd.Timestamp(fuxi_init)
                                        + pd.Timedelta(days=3.5 + 7 * (lead - 1))
                                    ).isoformat(),
                                    "valid_period_end_exclusive": (
                                        np.datetime_as_string(
                                            fuxi_init + np.timedelta64(7 * lead, "D"),
                                            unit="D",
                                        )
                                    ),
                                    "target_label_start": np.datetime_as_string(
                                        target_dates[0], unit="D"
                                    ),
                                    "target_label_end": np.datetime_as_string(
                                        target_dates[-1], unit="D"
                                    ),
                                    "lead_week": lead,
                                    "method": method,
                                    "method_label": LABELS[method],
                                    "acc": acc,
                                    "rmse_mm_day": rmse,
                                    "valid_cell_count": cells,
                                    "effective_area_km2": effective_area,
                                }
                            )
            finally:
                erpas.close()
                truth.close()
    finally:
        deterministic.close()
        climate.close()
        support.close()

    metrics = pd.DataFrame(rows).sort_values(
        ["lead_week", "fuxi_init", "method"]
    ).reset_index(drop=True)
    counts = metrics.groupby(["lead_week", "method"]).size()
    if len(metrics) != 26 * 4 * 3 or set(counts.to_numpy()) != {26}:
        raise ValueError(f"case counts violate the 26 x 4 x 3 contract: {counts}")
    if metrics.valid_cell_count.min() != 171:
        raise ValueError("common represented support changed from 171 cells")
    if maximum_target > np.datetime64("2024-12-31", "D"):
        raise ValueError("evaluation crossed the sealed-2025 observation firewall")

    validation_methods = ("raw_fuxi", "erpas")
    source_rows = reference.loc[
        reference.method.isin(validation_methods)
        & reference.fuxi_init.isin(EXPECTED_SHARED_DATES),
        [
            "year",
            "erpas_init",
            "fuxi_init",
            "lead_week",
            "method",
            "acc",
            "rmse_mm_day",
            "valid_cell_count",
            "effective_area_km2",
        ],
    ].copy()
    recomputed = metrics.loc[
        metrics.method.isin(validation_methods),
        [
            "year",
            "erpas_init",
            "fuxi_init",
            "lead_week",
            "method",
            "acc",
            "rmse_mm_day",
            "valid_cell_count",
            "effective_area_km2",
        ],
    ].copy()
    validation = recomputed.merge(
        source_rows,
        on=["year", "erpas_init", "fuxi_init", "lead_week", "method"],
        how="outer",
        validate="one_to_one",
        suffixes=("_recomputed", "_source"),
        indicator=True,
    )
    if len(validation) != 26 * 4 * 2 or set(validation._merge) != {"both"}:
        raise ValueError("source-score validation join is incomplete")
    validation["abs_acc_difference"] = (
        validation.acc_recomputed - validation.acc_source
    ).abs()
    validation["abs_rmse_difference"] = (
        validation.rmse_mm_day_recomputed - validation.rmse_mm_day_source
    ).abs()
    validation["abs_effective_area_difference_km2"] = (
        validation.effective_area_km2_recomputed
        - validation.effective_area_km2_source
    ).abs()
    if validation.abs_acc_difference.max() > SOURCE_SCORE_TOLERANCE:
        raise ValueError("recomputed source ACC exceeds the tight tolerance")
    if validation.abs_rmse_difference.max() > SOURCE_SCORE_TOLERANCE:
        raise ValueError("recomputed source RMSE exceeds the tight tolerance")
    if not (
        validation.valid_cell_count_recomputed
        == validation.valid_cell_count_source
    ).all():
        raise ValueError("source validation cell counts differ")

    validation_metadata = {
        "rows": len(validation),
        "rows_per_method": {
            key: int(value)
            for key, value in validation.groupby("method").size().items()
        },
        "tolerance": SOURCE_SCORE_TOLERANCE,
        "max_abs_acc_difference": float(validation.abs_acc_difference.max()),
        "max_abs_rmse_difference_mm_day": float(
            validation.abs_rmse_difference.max()
        ),
        "max_abs_effective_area_difference_km2": float(
            validation.abs_effective_area_difference_km2.max()
        ),
        "cell_counts_exact": True,
        "maximum_target_label_used": np.datetime_as_string(
            maximum_target, unit="D"
        ),
    }
    validation = validation.drop(columns="_merge").sort_values(
        ["lead_week", "fuxi_init", "method"]
    )
    return metrics, validation_metadata, validation


def interval(value: pd.Series, digits: int) -> str:
    return (
        f"{value.estimate:.{digits}f} "
        f"[{value.lower_95:.{digits}f}, {value.upper_95:.{digits}f}]"
    )


def build_results_markdown(
    acc_absolute: pd.DataFrame,
    rmse_absolute: pd.DataFrame,
    acc_effects: pd.DataFrame,
    rmse_effects: pd.DataFrame,
    validation: dict[str, Any],
) -> str:
    lines = [
        "# Strict-cohort deterministic ERPAS sensitivity",
        "",
        "All three forecasts were rescored on the same 26 following-Thursday "
        "FuXi starts at every lead (14 in 2023 and 12 in 2024). Scores use "
        "fraction-weighted weekly IMD truth, the common IMD 1991--2019 normal, "
        "and dynamic India-area by minimum-daily-observation-fraction weights.",
        "",
        "## Absolute scores",
        "",
        "| Lead | ERPAS ACC | Raw FuXi ACC | Raw-identity ACC | ERPAS RMSE | Raw FuXi RMSE | Raw-identity RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lead in LEADS:
        acc = acc_absolute.loc[acc_absolute.lead_week == lead].set_index("method")
        rmse = rmse_absolute.loc[rmse_absolute.lead_week == lead].set_index("method")
        lines.append(
            f"| W{lead} | {interval(acc.loc['erpas'], 3)} | "
            f"{interval(acc.loc['raw_fuxi'], 3)} | "
            f"{interval(acc.loc['raw_identity'], 3)} | "
            f"{interval(rmse.loc['erpas'], 3)} | "
            f"{interval(rmse.loc['raw_fuxi'], 3)} | "
            f"{interval(rmse.loc['raw_identity'], 3)} |"
        )
    lines.extend(
        [
            "",
            "RMSE is in mm/day. Brackets are descriptive paired-bootstrap 95% intervals.",
            "",
            "## Paired raw-identity effects",
            "",
            "| Lead | Delta ACC vs raw FuXi | Delta ACC vs ERPAS | Delta RMSE vs raw FuXi | Delta RMSE vs ERPAS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for lead in LEADS:
        acc = acc_effects.loc[acc_effects.lead_week == lead].set_index(
            "second_method"
        )
        rmse = rmse_effects.loc[rmse_effects.lead_week == lead].set_index(
            "second_method"
        )
        lines.append(
            f"| W{lead} | {interval(acc.loc['raw_fuxi'], 3)} | "
            f"{interval(acc.loc['erpas'], 3)} | "
            f"{interval(rmse.loc['raw_fuxi'], 3)} | "
            f"{interval(rmse.loc['erpas'], 3)} |"
        )
    lines.extend(
        [
            "",
            "Positive Delta ACC and negative Delta RMSE favor raw identity. All 16 "
            "reported raw-identity effects have intervals wholly on the favorable side of zero.",
            "",
            "## Contract and limitations",
            "",
            "- ERPAS is initialized Wednesday; both FuXi forecasts are initialized the following Thursday, so FuXi has 24 h newer initial-condition information.",
            "- This is post-hoc, two-season retrospective sensitivity evidence. The frozen raw-identity adapter was not retrained or retuned for these cases.",
            "- ERPAS ends at Week 4. No Week 5--6 extrapolation is made.",
            "- The common observation-normal ACC contract is intentionally different from the main benchmark's method-specific forecast-climatology ACC contract.",
            "- The latest IMD target label read was 2024-10-03; no 2025 target was opened.",
            "",
            "## Independent source check",
            "",
            f"The recomputed raw FuXi and ERPAS scores matched {validation['rows']} "
            "rows from the frozen ERPAS case table. Maximum absolute differences "
            f"were {validation['max_abs_acc_difference']:.3e} ACC and "
            f"{validation['max_abs_rmse_difference_mm_day']:.3e} mm/day RMSE "
            f"(required tolerance {validation['tolerance']:.1e}); cell counts matched exactly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to modify an existing run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging.{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir()
    (staging / "tables").mkdir()
    (staging / "validation").mkdir()
    (staging / "code").mkdir()

    source_hashes = verify_source_hashes()
    case_metrics, source_validation, validation_rows = build_case_metrics()
    dates = tuple(
        np.datetime64(value, "D")
        for value in sorted(case_metrics.fuxi_init.unique())
    )
    bootstrap_indices = make_bootstrap_indices(dates)
    acc_absolute, acc_effects = summarize_metric(
        case_metrics, bootstrap_indices, value_column="acc"
    )
    rmse_absolute, rmse_effects = summarize_metric(
        case_metrics, bootstrap_indices, value_column="rmse_mm_day"
    )
    if not acc_effects.resolved_favorable.all() or not rmse_effects.resolved_favorable.all():
        raise ValueError("a raw-identity effect is not resolved in the favorable direction")

    paths = {
        "case_metrics": staging / "tables/case_metrics.csv",
        "acc_absolute": staging / "tables/acc_summary_with_intervals.csv",
        "rmse_absolute": staging / "tables/rmse_summary_with_intervals.csv",
        "acc_effects": staging / "tables/paired_acc_effects.csv",
        "rmse_effects": staging / "tables/paired_rmse_effects.csv",
        "bootstrap_indices": staging / "validation/bootstrap_indices.npy",
        "source_validation": staging / "validation/source_score_validation.csv",
        "results": staging / "RESULTS.md",
        "code": staging / "code" / Path(__file__).name,
    }
    case_metrics.to_csv(paths["case_metrics"], index=False, float_format="%.10f")
    acc_absolute.to_csv(paths["acc_absolute"], index=False, float_format="%.10f")
    rmse_absolute.to_csv(paths["rmse_absolute"], index=False, float_format="%.10f")
    acc_effects.to_csv(paths["acc_effects"], index=False, float_format="%.10f")
    rmse_effects.to_csv(paths["rmse_effects"], index=False, float_format="%.10f")
    validation_rows.to_csv(
        paths["source_validation"], index=False, float_format="%.12g"
    )
    np.save(paths["bootstrap_indices"], bootstrap_indices, allow_pickle=False)
    paths["results"].write_text(
        build_results_markdown(
            acc_absolute,
            rmse_absolute,
            acc_effects,
            rmse_effects,
            source_validation,
        ),
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), paths["code"])

    manifest = {
        "schema_version": 1,
        "experiment": "india_s2s_erpas_deterministic_raw_identity_sensitivity_v1",
        "status": "complete",
        "canonical": True,
        "scientific_status": "post-hoc matched-valid-period two-season sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "method_labels": LABELS,
        "deterministic_method": {
            "name": "raw_identity",
            "source_coordinate": "raw_identity",
            "selection": "normal_climo_model; three-seed mean (42, 43, 44)",
            "training_anchor": "raw_fuxi",
            "uses_fitted_log_bias_in_reconstruction": False,
            "retrained_or_retuned_for_this_comparison": False,
        },
        "comparison": "ERPAS Wednesday versus following-Thursday FuXi; identical valid periods",
        "information_age": {
            "erpas_initialization_weekday": "Wednesday",
            "fuxi_initialization_weekday": "Thursday",
            "fuxi_newer_hours": 24,
            "equal_information_age": False,
        },
        "cohort": {
            "rule": "intersection of dates present in the ERPAS comparison and deterministic prediction store at every W1--W4 lead",
            "shared_across_all_leads": True,
            "case_count_per_lead": 26,
            "years": list(YEARS),
            "year_counts": {"2023": 14, "2024": 12},
            "fuxi_initializations": list(EXPECTED_SHARED_DATES),
            "fuxi_initializations_sha256": hashlib.sha256(
                ("\n".join(EXPECTED_SHARED_DATES) + "\n").encode("utf-8")
            ).hexdigest(),
            "case_metric_rows": len(case_metrics),
        },
        "lead_weeks": list(LEADS),
        "erpas_week_5_6_available": False,
        "verification_contract": {
            "target": "end-labelled daily IMD init+1...+28, seven labels per lead",
            "weekly_truth": "observation-fraction-weighted mean of seven daily IMD values",
            "anomaly_normal": "common IMD 1991--2019 centered-31-day equal-year daily normal",
            "acc": "case-wise weighted spatial anomaly correlation after the common observation normal",
            "rmse": "case-wise weighted spatial root-mean-square error in mm/day",
            "grid": "common 1.5-degree 27x27 grid",
            "support": "171 represented India cells",
            "dynamic_weights": "India area weight multiplied by minimum daily observation fraction over each target week",
        },
        "effect_contract": {
            "acc": "raw_identity minus comparator; positive is favorable",
            "rmse_mm_day": "raw_identity minus comparator; negative is favorable",
            "comparators": ["raw_fuxi", "erpas"],
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "block_length_initializations": BLOCK_LENGTH,
            "method": "paired year-stratified circular moving blocks",
            "shared_within_lead": "same resampled case indices for all methods and both metrics",
            "lead_sequences": "deterministic sequential draws from one seeded generator",
            "interval": "descriptive percentile 95%, conditional on 2023--2024",
        },
        "source_score_validation": source_validation,
        "sealed_2025_target_opened": False,
        "source_artifacts": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
        "atomic_output": {
            "enabled": True,
            "contract": "write to sibling dot-staging directory and rename only after all validations",
        },
    }
    manifest["artifact_sha256"] = {
        str(path.relative_to(staging)): sha256_file(path) for path in paths.values()
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "case_rows": len(case_metrics),
                "source_validation": source_validation,
                "acc_effects": acc_effects.to_dict(orient="records"),
                "rmse_effects": rmse_effects.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
