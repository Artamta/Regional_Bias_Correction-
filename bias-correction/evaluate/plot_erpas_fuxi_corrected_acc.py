#!/usr/bin/env python3
"""Build the aligned ERPAS--FuXi--INDRA ACC and RMSE sensitivity.

ERPAS is initialized on Wednesday.  Lead days 2--29 are averaged into four
Thursday--Wednesday windows and paired with the following-Thursday FuXi start.
FuXi therefore has 24 hours newer initial-condition information.  All three
forecasts are scored against the same end-labelled IMD targets (+1...+28),
1991--2019 IMD anomaly normal, dynamic observation support, and area weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ARCHIVE_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1"
)
FUXI_ROOT = (
    ARCHIVE_ROOT
    / "forecasts/fuxi_s2s/"
    "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/"
    "tp/common_1p5"
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
INFERENCE_ROOT = Path(
    "resultsv3/india_s2s_probabilistic_bridge/"
    "inference_full_20260825T233121Z"
)
DEFAULT_OUTPUT = Path(
    "resultsv3/india_s2s_erpas_corrected_acc_sensitivity/"
    "full_v2_20260826T232800Z"
)

YEARS = (2023, 2024)
LEADS = (1, 2, 3, 4)
METHODS = ("erpas", "raw_fuxi", "location_spread")
LABELS = {
    "erpas": "ERPAS",
    "raw_fuxi": "Raw FuXi-S2S",
    "location_spread": "INDRA-S2S v1 (calibrated mean)",
}
COLORS = {
    "erpas": "#D55E00",
    "raw_fuxi": "#666666",
    "location_spread": "#0072B2",
}
MARKERS = {"erpas": "s", "raw_fuxi": "o", "location_spread": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_826)
    parser.add_argument("--block-length", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_climatology_day(date: np.datetime64) -> int:
    value = pd.Timestamp(date)
    return pd.Timestamp(2000, value.month, value.day).dayofyear - 1


def valid_midpoint_is_jjas(fuxi_init: np.datetime64, lead_week: int) -> bool:
    midpoint = pd.Timestamp(fuxi_init) + pd.Timedelta(
        days=3.5 + 7 * (lead_week - 1)
    )
    return midpoint.month in (6, 7, 8, 9)


def collect_pairs(
    erpas_initializations: np.ndarray, fuxi_initializations: np.ndarray
) -> list[tuple[np.datetime64, np.datetime64]]:
    erpas = np.asarray(erpas_initializations, dtype="datetime64[D]")
    fuxi = set(np.asarray(fuxi_initializations, dtype="datetime64[D]"))
    pairs = [
        (init, init + np.timedelta64(1, "D"))
        for init in np.sort(erpas)
        if init + np.timedelta64(1, "D") in fuxi
    ]
    if not pairs:
        raise ValueError("no Wednesday ERPAS/following-Thursday FuXi pairs")
    if any(pd.Timestamp(left).weekday() != 2 for left, _ in pairs):
        raise ValueError("ERPAS cohort contains a non-Wednesday initialization")
    if any(right - left != np.timedelta64(1, "D") for left, right in pairs):
        raise ValueError("FuXi must initialize exactly 24 hours after ERPAS")
    return pairs


def weekly_truth_and_normal(
    truth_dataset: xr.Dataset,
    daily_climatology: np.ndarray,
    fuxi_init: np.datetime64,
    lead_week: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    support = fraction.min(axis=0)
    days = [fixed_climatology_day(date) for date in dates]
    normal_daily = np.asarray(daily_climatology[days], dtype=np.float64)
    count = np.isfinite(normal_daily).sum(axis=0)
    normal = np.divide(
        np.nansum(normal_daily, axis=0, dtype=np.float64),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return truth, normal, support, dates


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
    forecast_mean = (
        spatial * np.where(represented, forecast_anomaly, 0.0)
    ).sum(dtype=np.float64) / total
    truth_mean = (
        spatial * np.where(represented, truth_anomaly, 0.0)
    ).sum(dtype=np.float64) / total
    forecast_centered = np.where(
        represented, forecast_anomaly - forecast_mean, 0.0
    )
    truth_centered = np.where(represented, truth_anomaly - truth_mean, 0.0)
    covariance = (spatial * forecast_centered * truth_centered).sum(dtype=np.float64)
    forecast_variance = (spatial * forecast_centered**2).sum(dtype=np.float64)
    truth_variance = (spatial * truth_centered**2).sum(dtype=np.float64)
    denominator = np.sqrt(forecast_variance * truth_variance)
    if denominator <= 0.0:
        raise ValueError("ACC denominator is zero")
    return float(covariance / denominator), int(represented.sum()), float(total)


def weighted_spatial_rmse(
    forecast: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, int, float]:
    """Return area-and-observation-weighted RMSE on common finite support."""
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
    rmse = np.sqrt((spatial * squared_error).sum(dtype=np.float64) / total)
    return float(rmse), int(represented.sum()), float(total)


def circular_block_indices(
    size: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if size < 1 or block_length < 1:
        raise ValueError("bootstrap size and block length must be positive")
    starts = rng.integers(0, size, size=int(np.ceil(size / block_length)))
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % size).reshape(-1)[:size]


def paired_bootstrap(
    case_metrics: pd.DataFrame,
    *,
    draws: int,
    block_length: int,
    seed: int,
    value_column: str = "acc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    absolute_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    comparisons = (
        ("location_spread", "raw_fuxi"),
        ("location_spread", "erpas"),
        ("raw_fuxi", "erpas"),
    )
    for lead in LEADS:
        subset = case_metrics[case_metrics.lead_week == lead]
        wide = subset.pivot(
            index=["year", "erpas_init", "fuxi_init"],
            columns="method",
            values=value_column,
        ).sort_index()
        if list(wide.columns) != list(METHODS):
            wide = wide.loc[:, METHODS]
        values = wide.to_numpy(dtype=np.float64)
        years = wide.index.get_level_values("year").to_numpy(dtype=np.int64)
        samples = np.empty((draws, len(METHODS)), dtype=np.float64)
        for draw in range(draws):
            selected: list[int] = []
            for year in YEARS:
                positions = np.flatnonzero(years == year)
                local = circular_block_indices(len(positions), block_length, rng)
                selected.extend(positions[local].tolist())
            samples[draw] = values[selected].mean(axis=0)
        for method_index, method in enumerate(METHODS):
            estimate = float(values[:, method_index].mean())
            lower, upper = np.percentile(samples[:, method_index], (2.5, 97.5))
            absolute_rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "lead_week": lead,
                    "n_cases": len(wide),
                    "estimate": estimate,
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "metric": value_column,
                }
            )
        for first, second in comparisons:
            first_index = METHODS.index(first)
            second_index = METHODS.index(second)
            paired_values = values[:, first_index] - values[:, second_index]
            paired_samples = samples[:, first_index] - samples[:, second_index]
            lower, upper = np.percentile(paired_samples, (2.5, 97.5))
            effect_rows.append(
                {
                    "comparison": f"{first}_minus_{second}",
                    "first_method": first,
                    "second_method": second,
                    "lead_week": lead,
                    "n_cases": len(wide),
                    "estimate": float(paired_values.mean()),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "metric": value_column,
                }
            )
    return pd.DataFrame(absolute_rows), pd.DataFrame(effect_rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.titleweight": "semibold",
            "legend.fontsize": 8.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
        }
    )


def make_figure(
    absolute: pd.DataFrame, effects: pd.DataFrame, output: Path
) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.45),
        gridspec_kw={"width_ratios": (1.08, 0.92)},
    )
    weeks = np.asarray(LEADS)
    for method in METHODS:
        rows = absolute[absolute.method == method].sort_values("lead_week")
        values = rows.estimate.to_numpy()
        lower = rows.lower_95.to_numpy()
        upper = rows.upper_95.to_numpy()
        axes[0].plot(
            weeks,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.25 if method == "location_spread" else 1.65,
            markersize=5.2,
            label=LABELS[method],
            zorder=4 if method == "location_spread" else 3,
        )
        axes[0].fill_between(
            weeks, lower, upper, color=COLORS[method], alpha=0.10, linewidth=0
        )
    axes[0].axhline(0.0, color="#222222", linewidth=0.7, alpha=0.5)
    axes[0].set_title("(a) Mean spatial ACC")
    axes[0].set_ylabel("Anomaly correlation coefficient")
    axes[0].set_xlabel("Lead week")
    axes[0].set_xticks(weeks, [f"W{week}" for week in weeks])
    axes[0].set_ylim(-0.10, 0.82)
    axes[0].legend(loc="upper right", frameon=False)

    comparisons = (
        ("location_spread_minus_raw_fuxi", "INDRA − raw FuXi", "#0072B2", "o"),
        ("location_spread_minus_erpas", "INDRA − ERPAS", "#009E73", "D"),
    )
    for comparison, label, color, marker in comparisons:
        rows = effects[effects.comparison == comparison].sort_values("lead_week")
        values = rows.estimate.to_numpy()
        errors = np.vstack(
            (values - rows.lower_95.to_numpy(), rows.upper_95.to_numpy() - values)
        )
        axes[1].errorbar(
            weeks,
            values,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=5.0,
            linewidth=1.8,
            capsize=3.0,
            label=label,
            zorder=3,
        )
    axes[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_title("(b) Paired ACC improvement")
    axes[1].set_ylabel("ΔACC (positive favours INDRA)")
    axes[1].set_xlabel("Lead week")
    axes[1].set_xticks(weeks, [f"W{week}" for week in weeks])
    axes[1].set_ylim(-0.04, 0.32)
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Matched-valid-time rainfall skill over India · 2023–2024 JJAS",
        fontsize=12.4,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "33 periods per lead; ERPAS Wed, FuXi Thu (+24 h newer); common IMD 1991–2019 normal. "
        "Shading/bars: paired year-stratified block-bootstrap 95% intervals.",
        ha="center",
        va="bottom",
        fontsize=7.35,
        color="#333333",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.22, wspace=0.31)
    base = output / "Figure_S_erpas_fuxi_indra_acc"
    files: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        files.append(path)
    plt.close(fig)
    return files


def build_case_metrics() -> tuple[pd.DataFrame, dict[str, Any]]:
    climate_dataset = xr.open_zarr(IMD_CLIMATOLOGY, consolidated=True)
    support_dataset = xr.open_zarr(SPATIAL_SUPPORT, consolidated=True)
    try:
        daily_climatology = np.asarray(
            climate_dataset.climatology_mean.load().values, dtype=np.float64
        )
        india_weight = np.asarray(
            support_dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
        latitude = np.asarray(support_dataset.latitude.values, dtype=np.float64)
        longitude = np.asarray(support_dataset.longitude.values, dtype=np.float64)
    finally:
        climate_dataset.close()
        support_dataset.close()

    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    maximum_target = np.datetime64("1900-01-01", "D")
    total_pairs = 0
    for year in YEARS:
        adjustment_path = INFERENCE_ROOT / f"inference/adjustments_{year}.npz"
        adjustment = np.load(adjustment_path)
        fuxi_initializations = np.asarray(
            adjustment["initializations"], dtype="datetime64[D]"
        )
        if int(adjustment["selected_seed"]) != 43:
            raise ValueError("frozen selected seed changed")
        if int(adjustment["member_count"]) != 50:
            raise ValueError("frozen member count changed")
        if not np.array_equal(adjustment["latitude"], latitude) or not np.array_equal(
            adjustment["longitude"], longitude
        ):
            raise ValueError("inference and spatial-support grids differ")

        erpas_path = ERPAS_ROOT / f"{year}.zarr"
        truth_path = IMD_ROOT / f"{year}.zarr"
        erpas_dataset = xr.open_zarr(erpas_path, consolidated=True)
        truth_dataset = xr.open_zarr(truth_path, consolidated=True)
        try:
            if erpas_dataset.attrs.get("distribution_representation") != "mean_only":
                raise ValueError("ERPAS is not the provider mean-only product")
            if not np.array_equal(erpas_dataset.latitude.values, latitude) or not np.array_equal(
                erpas_dataset.longitude.values, longitude
            ):
                raise ValueError("ERPAS and benchmark grids differ")
            pairs = collect_pairs(erpas_dataset.init.values, fuxi_initializations)
            total_pairs += len(pairs)
            lookup = {value: index for index, value in enumerate(fuxi_initializations)}
            for erpas_init, fuxi_init in pairs:
                fuxi_index = lookup[fuxi_init]
                raw = np.asarray(
                    adjustment["raw_ensemble_mean"][fuxi_index], dtype=np.float64
                )
                corrected = np.asarray(
                    adjustment["corrected_ensemble_mean"][fuxi_index],
                    dtype=np.float64,
                )
                erpas = np.asarray(
                    erpas_dataset.ensemble_mean.sel(
                        init=erpas_init, lead_day=np.arange(2, 30)
                    ).load().values,
                    dtype=np.float64,
                ).reshape(4, 7, len(latitude), len(longitude)).mean(axis=1)
                for lead in LEADS:
                    if not valid_midpoint_is_jjas(fuxi_init, lead):
                        continue
                    truth, normal, observation_support, dates = weekly_truth_and_normal(
                        truth_dataset,
                        daily_climatology,
                        fuxi_init,
                        lead,
                    )
                    maximum_target = max(maximum_target, dates.max())
                    weights = india_weight * observation_support
                    fields = {
                        "erpas": erpas[lead - 1],
                        "raw_fuxi": raw[lead - 1],
                        "location_spread": corrected[lead - 1],
                    }
                    for method in METHODS:
                        acc, cells, effective_area = weighted_spatial_acc(
                            fields[method], truth, normal, weights
                        )
                        rmse, rmse_cells, rmse_effective_area = weighted_spatial_rmse(
                            fields[method], truth, weights
                        )
                        if cells != rmse_cells or not np.isclose(
                            effective_area, rmse_effective_area
                        ):
                            raise ValueError("ACC and RMSE supports differ")
                        rows.append(
                            {
                                "year": year,
                                "erpas_init": np.datetime_as_string(
                                    erpas_init, unit="D"
                                ),
                                "fuxi_init": np.datetime_as_string(fuxi_init, unit="D"),
                                "valid_period_start": np.datetime_as_string(
                                    fuxi_init + np.timedelta64(7 * (lead - 1), "D"),
                                    unit="D",
                                ),
                                "valid_period_midpoint": (
                                    pd.Timestamp(fuxi_init)
                                    + pd.Timedelta(days=3.5 + 7 * (lead - 1))
                                ).isoformat(),
                                "valid_period_end_exclusive": np.datetime_as_string(
                                    fuxi_init + np.timedelta64(7 * lead, "D"), unit="D"
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
            erpas_dataset.close()
            truth_dataset.close()
        sources[str(year)] = {
            "adjustments": str(adjustment_path.resolve()),
            "adjustments_sha256": sha256_file(adjustment_path),
            "erpas_store": str(erpas_path),
            "erpas_zmetadata_sha256": sha256_file(erpas_path / ".zmetadata"),
            "imd_store": str(truth_path),
            "imd_zmetadata_sha256": sha256_file(truth_path / ".zmetadata"),
            "matched_allseason_cycles": len(pairs),
        }
    metrics = pd.DataFrame(rows).sort_values(
        ["lead_week", "fuxi_init", "method"]
    ).reset_index(drop=True)
    counts = metrics.groupby(["lead_week", "method"]).size()
    if set(counts.to_numpy()) != {33}:
        raise ValueError(f"expected 33 valid-midpoint JJAS cases per row: {counts}")
    if metrics.valid_cell_count.min() != 171:
        raise ValueError("common represented support changed from 171 cells")
    if maximum_target > np.datetime64("2024-12-31", "D"):
        raise ValueError("evaluation crossed the sealed 2025 observation firewall")
    metadata = {
        "sources": sources,
        "total_matched_allseason_cycles": total_pairs,
        "maximum_target_label_used": np.datetime_as_string(maximum_target, unit="D"),
        "imd_climatology": str(IMD_CLIMATOLOGY),
        "imd_climatology_zmetadata_sha256": sha256_file(
            IMD_CLIMATOLOGY / ".zmetadata"
        ),
        "spatial_support": str(SPATIAL_SUPPORT),
        "spatial_support_zmetadata_sha256": sha256_file(
            SPATIAL_SUPPORT / ".zmetadata"
        ),
    }
    return metrics, metadata


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to modify an existing run: {output}")
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "code").mkdir()

    case_metrics, source_metadata = build_case_metrics()
    absolute, effects = paired_bootstrap(
        case_metrics,
        draws=args.bootstrap_draws,
        block_length=args.block_length,
        seed=args.bootstrap_seed,
    )
    rmse_absolute, rmse_effects = paired_bootstrap(
        case_metrics,
        draws=args.bootstrap_draws,
        block_length=args.block_length,
        seed=args.bootstrap_seed,
        value_column="rmse_mm_day",
    )
    case_path = output / "tables/case_acc.csv"
    summary_path = output / "tables/acc_summary_with_intervals.csv"
    effect_path = output / "tables/paired_acc_effects.csv"
    rmse_summary_path = output / "tables/rmse_summary_with_intervals.csv"
    rmse_effect_path = output / "tables/paired_rmse_effects.csv"
    case_metrics.to_csv(case_path, index=False, float_format="%.10f")
    absolute.to_csv(summary_path, index=False, float_format="%.10f")
    effects.to_csv(effect_path, index=False, float_format="%.10f")
    rmse_absolute.to_csv(rmse_summary_path, index=False, float_format="%.10f")
    rmse_effects.to_csv(rmse_effect_path, index=False, float_format="%.10f")
    figure_paths = make_figure(absolute, effects, output)

    caption = (
        "Limited-period matched-valid-time comparison of All-India weekly rainfall "
        "spatial anomaly correlation (ACC) and RMSE for ERPAS, raw FuXi-S2S, and the "
        "INDRA-S2S v1 calibrated FuXi ensemble mean. ERPAS Wednesday lead days "
        "2--29 and the following-Thursday FuXi Weeks 1--4 verify over identical "
        "seven-day periods; FuXi therefore uses initial conditions 24 h newer than "
        "ERPAS. Seasons are assigned by valid-period midpoint, giving 33 paired JJAS "
        "periods per lead across 2023--2024. All methods use the same end-labelled "
        "IMD targets (+1...+28), 1991--2019 IMD anomaly normal, 171-cell support, and "
        "area-by-observation-coverage weighting. The ACC and RMSE tables report "
        "absolute scores and paired differences; "
        "positive values favour INDRA. Shading and error bars are descriptive 95% "
        "percentile intervals from 10,000 paired, year-stratified circular moving-block "
        "resamples of four weekly cycles. ERPAS has no complete Week 5--6 fields, and "
        "the intervals are conditional on these two retrospective years."
    )
    (output / "CAPTION.md").write_text(caption + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), output / "code" / Path(__file__).name)

    files = [
        case_path,
        summary_path,
        effect_path,
        rmse_summary_path,
        rmse_effect_path,
        *figure_paths,
        output / "CAPTION.md",
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "india_s2s_erpas_corrected_acc_rmse_sensitivity_v1",
        "status": "complete",
        "scientific_status": "retrospective limited-period appendix sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "ERPAS Wednesday versus following-Thursday FuXi; identical valid times",
        "initialization_age_difference_hours": 24,
        "newer_system": "Raw FuXi-S2S and INDRA-S2S v1",
        "not_equal_information_age": True,
        "years": list(YEARS),
        "lead_weeks": list(LEADS),
        "erpas_week_5_6_available": False,
        "season_assignment": "exact seven-day valid-period midpoint in June--September",
        "case_count_per_lead": 33,
        "target_contract": "end-labelled IMD init+1...+28; seven days per lead",
        "anomaly_contract": "common IMD 1991--2019 centered-31-day equal-year daily normal",
        "spatial_contract": "common 27x27 grid; 171 represented India cells; India area x weekly minimum observation coverage",
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "block_length_weekly_cycles": args.block_length,
            "method": "paired year-stratified circular moving blocks",
            "interval": "descriptive percentile 95%; conditional on 2023--2024",
        },
        "selected_method": {
            "name": "location_spread",
            "paper_label": LABELS["location_spread"],
            "seed": 43,
            "member_count": 50,
            "evidence_role": "ensemble mean for deterministic ACC and RMSE",
        },
        "sealed_2025_target_opened": False,
        "source_metadata": source_metadata,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    manifest_path = output / "manifest.json"
    manifest["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path) for path in files
    }
    manifest["code_sha256"] = sha256_file(output / "code" / Path(__file__).name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "figure": str(figure_paths[0]),
        "summary": absolute.to_dict(orient="records"),
        "effects": effects.to_dict(orient="records"),
        "rmse_summary": rmse_absolute.to_dict(orient="records"),
        "rmse_effects": rmse_effects.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
