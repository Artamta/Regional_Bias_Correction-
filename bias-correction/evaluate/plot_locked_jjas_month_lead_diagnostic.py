#!/usr/bin/env python3
"""Derive an initialization-month × lead diagnostic from locked case metrics.

Only two files are read from a completed locked evaluation: ``manifest.json``
and ``metrics/test_case_metrics.csv``.  Forecast, prediction, observation,
target, checkpoint, and source-data arrays are never opened.  The 2020--2021
period is exploratory/reused and is not independent confirmation.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


SCHEMA_NAME = "fuxi_imd_locked_jjas_month_lead_diagnostic"
SCHEMA_VERSION = 1
EVALUATION_ROLE = "exploratory_reused_hindcast_evaluation"
EVALUATION_SCOPE = (
    "2020-2021 exploratory/reused hindcast diagnostic; not independent confirmation"
)
TEST_YEARS = (2020, 2021)
INITIALIZATION_MONTHS = (6, 7, 8, 9)
LEAD_WEEKS = (1, 2, 3, 4, 5, 6)
EXPECTED_MONTH_COUNTS_PER_YEAR = {6: 9, 7: 8, 8: 9, 9: 9}
RAW_METHOD = "raw_fuxi"
CORRECTED_METHOD = "corrected"
DEFAULT_REPLICATES = 5000
DEFAULT_BLOCK_LENGTH = 3
DEFAULT_SEED = 20260812
REQUIRED_COLUMNS = {
    "method",
    "case_id",
    "lead",
    "region",
    "season",
    "valid_cells",
    "weight_sum",
    "acc",
    "rmse",
    "bias",
}
MONTH_COLORS = {
    6: "#007C91",
    7: "#E08A1E",
    8: "#5B5F97",
    9: "#B94C55",
}
IMPROVEMENT_CMAP = LinearSegmentedColormap.from_list(
    "terracotta_to_teal",
    ("#B44738", "#F5F2EA", "#007C83"),
)


class DiagnosticContractError(ValueError):
    """Raised when locked artifact or table structure differs from contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticContractError(f"expected a JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(figure: plt.Figure, output_stem: Path, dpi: int) -> None:
    for suffix, options in (
        (".png", {"dpi": dpi, "facecolor": figure.get_facecolor()}),
        (".pdf", {"facecolor": figure.get_facecolor()}),
    ):
        target = output_stem.with_suffix(suffix).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=suffix, dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(
                temporary,
                format=suffix.lstrip("."),
                bbox_inches="tight",
                **options,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_locked_manifest(
    locked_directory: Path,
) -> tuple[Mapping[str, Any], Path]:
    manifest_path = locked_directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise DiagnosticContractError("locked evaluation is not complete")
    if manifest.get("evaluation_role") != EVALUATION_ROLE:
        raise DiagnosticContractError("locked evaluation role differs")
    scope = str(manifest.get("evaluation_scope", "")).lower()
    for required in ("exploratory", "reused", "not independent"):
        if required not in scope:
            raise DiagnosticContractError(
                f"locked evaluation scope lacks {required!r}"
            )
    if tuple(manifest.get("test_years", ())) != TEST_YEARS:
        raise DiagnosticContractError("test years must be exactly 2020--2021")
    if manifest.get("selection_locked_before_target_access") is not True:
        raise DiagnosticContractError("selection was not locked before target access")
    if not (
        manifest.get("selection_locked_before_test") is True
        or manifest.get("model_selection_locked") is True
    ):
        raise DiagnosticContractError("model selection is not declared locked")
    if manifest.get("test_used_for_selection") is not False:
        raise DiagnosticContractError("test_used_for_selection must be false")
    if manifest.get("parameter_updates") != 0:
        raise DiagnosticContractError("parameter_updates must equal zero")
    if manifest.get("reused_test_period") is not True:
        raise DiagnosticContractError("reused_test_period must be true")
    if manifest.get("genuine_independent_test") is not False:
        raise DiagnosticContractError("genuine_independent_test must be false")
    if not str(manifest.get("selected_configuration", "")):
        raise DiagnosticContractError("selected configuration is missing")

    case_metrics_path = locked_directory / "metrics" / "test_case_metrics.csv"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DiagnosticContractError("locked manifest lacks artifact checksums")
    expected = artifacts.get("metrics/test_case_metrics.csv")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DiagnosticContractError("case-metrics checksum is not declared")
    if not case_metrics_path.is_file() or sha256_file(case_metrics_path) != expected:
        raise DiagnosticContractError("locked case-metrics checksum differs")
    return manifest, case_metrics_path


def load_locked_case_metrics(
    locked_evaluation_directory: Path,
) -> tuple[Mapping[str, Any], pd.DataFrame, Path]:
    """Read only the locked manifest and case-level metric CSV."""

    locked_directory = Path(locked_evaluation_directory).expanduser().resolve()
    manifest, case_metrics_path = _validate_locked_manifest(locked_directory)
    frame = pd.read_csv(case_metrics_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DiagnosticContractError(f"case metrics lack columns: {missing}")
    frame = frame.loc[frame.method.isin((RAW_METHOD, CORRECTED_METHOD))].copy()
    frame["case_id"] = pd.to_datetime(frame.case_id, errors="coerce").dt.normalize()
    if frame.case_id.isna().any():
        raise DiagnosticContractError("case metrics contain invalid dates")
    if frame.duplicated(["method", "case_id", "lead"]).any():
        raise DiagnosticContractError("case metrics contain duplicate method/date/lead")
    if set(frame.method) != {RAW_METHOD, CORRECTED_METHOD}:
        raise DiagnosticContractError("raw and corrected case metrics are both required")
    if set(frame.lead.astype(int)) != set(LEAD_WEEKS):
        raise DiagnosticContractError("case-metric leads are not exactly weeks 1--6")
    if set(frame.region.astype(str).str.lower()) != {"india"}:
        raise DiagnosticContractError("case metrics are not India-only")
    if set(frame.season.astype(str)) != {"ALL"}:
        raise DiagnosticContractError("case metrics do not use the expected ALL season")
    for metric in ("acc", "rmse", "bias", "valid_cells", "weight_sum"):
        values = frame[metric].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DiagnosticContractError(f"case metrics contain non-finite {metric}")
    if np.any(frame.rmse.to_numpy(dtype=np.float64) < 0.0):
        raise DiagnosticContractError("RMSE cannot be negative")
    if np.any(np.abs(frame.acc.to_numpy(dtype=np.float64)) > 1.0 + 1.0e-8):
        raise DiagnosticContractError("ACC lies outside [-1, 1]")
    if np.any(frame.valid_cells.to_numpy(dtype=np.float64) <= 0.0) or np.any(
        frame.weight_sum.to_numpy(dtype=np.float64) <= 0.0
    ):
        raise DiagnosticContractError("spatial support is empty")

    dates = pd.DatetimeIndex(sorted(frame.case_id.unique()))
    if len(dates) != 70 or not dates.is_monotonic_increasing:
        raise DiagnosticContractError("expected exactly 70 chronological dates")
    if tuple(sorted(dates.year.unique())) != TEST_YEARS:
        raise DiagnosticContractError("case dates are not exactly 2020--2021")
    if tuple(sorted(dates.month.unique())) != INITIALIZATION_MONTHS:
        raise DiagnosticContractError("case dates are not exactly JJAS")
    for year in TEST_YEARS:
        year_dates = dates[dates.year == year]
        if len(year_dates) != 35:
            raise DiagnosticContractError(f"{year} does not contain 35 starts")
        counts = pd.Series(year_dates.month).value_counts().sort_index().to_dict()
        if counts != EXPECTED_MONTH_COUNTS_PER_YEAR:
            raise DiagnosticContractError(
                f"{year} initialization-month counts differ: {counts}"
            )
    expected_rows = 2 * len(dates) * len(LEAD_WEEKS)
    if len(frame) != expected_rows:
        raise DiagnosticContractError(
            f"expected {expected_rows} paired rows; found {len(frame)}"
        )
    pair_counts = frame.groupby(["case_id", "lead"]).method.nunique()
    if len(pair_counts) != len(dates) * len(LEAD_WEEKS) or not pair_counts.eq(2).all():
        raise DiagnosticContractError("raw/corrected metric pairs are incomplete")
    return manifest, frame.sort_values(["case_id", "lead", "method"]), case_metrics_path


def _metric_matrix(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    method: str,
    metric: str,
) -> np.ndarray:
    selected = frame.loc[frame.method.eq(method), ["case_id", "lead", metric]]
    matrix = selected.pivot(index="case_id", columns="lead", values=metric)
    matrix = matrix.reindex(index=dates, columns=LEAD_WEEKS)
    values = matrix.to_numpy(dtype=np.float64)
    if values.shape != (70, 6) or not np.isfinite(values).all():
        raise DiagnosticContractError(f"incomplete {method} {metric} matrix")
    return values


def _circular_block_sample(
    indices: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_size = len(indices)
    block_count = int(math.ceil(sample_size / block_length))
    starts = rng.integers(0, sample_size, size=block_count)
    offsets = np.arange(block_length, dtype=np.int64)
    positions = np.concatenate(
        [((int(start) + offsets) % sample_size) for start in starts]
    )[:sample_size]
    return indices[positions]


def month_stratified_paired_draws(
    dates: pd.DatetimeIndex,
    month: int,
    *,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample years, then paired initialization blocks within one month."""

    if month not in INITIALIZATION_MONTHS:
        raise ValueError(f"month must be one of {INITIALIZATION_MONTHS}")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    groups = {
        year: np.flatnonzero((dates.year == year) & (dates.month == month)).astype(
            np.int64
        )
        for year in TEST_YEARS
    }
    sizes = {year: len(indices) for year, indices in groups.items()}
    if len(set(sizes.values())) != 1 or not all(sizes.values()):
        raise DiagnosticContractError(
            f"month {month} does not have balanced nonempty year groups: {sizes}"
        )
    if not 1 <= block_length <= min(sizes.values()):
        raise ValueError("block_length exceeds a month/year stratum")
    output = np.empty((replicates, sum(sizes.values())), dtype=np.int16)
    years = np.asarray(TEST_YEARS, dtype=np.int16)
    for replicate in range(replicates):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        output[replicate] = np.concatenate(
            [
                _circular_block_sample(groups[int(year)], block_length, rng)
                for year in sampled_years
            ]
        )
    return output


def summarize_month_lead(
    frame: pd.DataFrame,
    *,
    replicates: int = DEFAULT_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compute paired raw/corrected points and approximate percentile intervals."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must lie in 0..2^63-1")
    dates = pd.DatetimeIndex(sorted(frame.case_id.unique()))
    raw_acc = _metric_matrix(frame, dates, RAW_METHOD, "acc")
    corrected_acc = _metric_matrix(frame, dates, CORRECTED_METHOD, "acc")
    raw_rmse = _metric_matrix(frame, dates, RAW_METHOD, "rmse")
    corrected_rmse = _metric_matrix(frame, dates, CORRECTED_METHOD, "rmse")
    raw_bias = _metric_matrix(frame, dates, RAW_METHOD, "bias")
    corrected_bias = _metric_matrix(frame, dates, CORRECTED_METHOD, "bias")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for month in INITIALIZATION_MONTHS:
        month_indices = np.flatnonzero(dates.month == month)
        draws = month_stratified_paired_draws(
            dates,
            month,
            replicates=replicates,
            block_length=block_length,
            rng=rng,
        )
        raw_acc_point = np.mean(raw_acc[month_indices], axis=0)
        corrected_acc_point = np.mean(corrected_acc[month_indices], axis=0)
        acc_effect = corrected_acc_point - raw_acc_point
        raw_rmse_point = np.mean(raw_rmse[month_indices], axis=0)
        corrected_rmse_point = np.mean(corrected_rmse[month_indices], axis=0)
        if np.any(raw_rmse_point <= 0.0):
            raise DiagnosticContractError("raw mean RMSE must be positive")
        rmse_effect = 100.0 * (
            raw_rmse_point - corrected_rmse_point
        ) / raw_rmse_point
        raw_bias_point = np.mean(raw_bias[month_indices], axis=0)
        corrected_bias_point = np.mean(corrected_bias[month_indices], axis=0)
        absolute_bias_effect = np.abs(raw_bias_point) - np.abs(corrected_bias_point)

        acc_samples = np.mean(corrected_acc[draws] - raw_acc[draws], axis=1)
        sampled_raw_rmse = np.mean(raw_rmse[draws], axis=1)
        sampled_corrected_rmse = np.mean(corrected_rmse[draws], axis=1)
        if np.any(sampled_raw_rmse <= 0.0):
            raise DiagnosticContractError("bootstrap raw mean RMSE must be positive")
        rmse_samples = 100.0 * (
            sampled_raw_rmse - sampled_corrected_rmse
        ) / sampled_raw_rmse
        sampled_raw_bias = np.mean(raw_bias[draws], axis=1)
        sampled_corrected_bias = np.mean(corrected_bias[draws], axis=1)
        absolute_bias_samples = np.abs(sampled_raw_bias) - np.abs(
            sampled_corrected_bias
        )
        acc_lower, acc_upper = np.quantile(acc_samples, (0.025, 0.975), axis=0)
        rmse_lower, rmse_upper = np.quantile(
            rmse_samples, (0.025, 0.975), axis=0
        )
        bias_lower, bias_upper = np.quantile(
            absolute_bias_samples, (0.025, 0.975), axis=0
        )
        counts_by_year = {
            year: int(np.count_nonzero((dates.year == year) & (dates.month == month)))
            for year in TEST_YEARS
        }
        for lead_index, lead in enumerate(LEAD_WEEKS):
            rows.append(
                {
                    "evaluation_role": EVALUATION_ROLE,
                    "evaluation_scope": EVALUATION_SCOPE,
                    "test_years": "2020,2021",
                    "initialization_month": month,
                    "initialization_month_name": calendar.month_name[month],
                    "lead_week": lead,
                    "n_years": len(TEST_YEARS),
                    "n_initializations": len(month_indices),
                    "n_initializations_2020": counts_by_year[2020],
                    "n_initializations_2021": counts_by_year[2021],
                    "raw_acc_mean": float(raw_acc_point[lead_index]),
                    "corrected_acc_mean": float(corrected_acc_point[lead_index]),
                    "acc_difference": float(acc_effect[lead_index]),
                    "acc_difference_ci95_lower": float(acc_lower[lead_index]),
                    "acc_difference_ci95_upper": float(acc_upper[lead_index]),
                    "acc_bootstrap_fraction_positive": float(
                        np.mean(acc_samples[:, lead_index] > 0.0)
                    ),
                    "acc_approximate_interval_above_zero": bool(
                        acc_lower[lead_index] > 0.0
                    ),
                    "raw_rmse_mean_mm_day": float(raw_rmse_point[lead_index]),
                    "corrected_rmse_mean_mm_day": float(
                        corrected_rmse_point[lead_index]
                    ),
                    "rmse_reduction_pct": float(rmse_effect[lead_index]),
                    "rmse_reduction_ci95_lower_pct": float(rmse_lower[lead_index]),
                    "rmse_reduction_ci95_upper_pct": float(rmse_upper[lead_index]),
                    "rmse_bootstrap_fraction_positive": float(
                        np.mean(rmse_samples[:, lead_index] > 0.0)
                    ),
                    "rmse_approximate_interval_above_zero": bool(
                        rmse_lower[lead_index] > 0.0
                    ),
                    "raw_bias_mean_mm_day": float(raw_bias_point[lead_index]),
                    "corrected_bias_mean_mm_day": float(
                        corrected_bias_point[lead_index]
                    ),
                    "absolute_bias_improvement_mm_day": float(
                        absolute_bias_effect[lead_index]
                    ),
                    "absolute_bias_improvement_ci95_lower_mm_day": float(
                        bias_lower[lead_index]
                    ),
                    "absolute_bias_improvement_ci95_upper_mm_day": float(
                        bias_upper[lead_index]
                    ),
                    "absolute_bias_bootstrap_fraction_positive": float(
                        np.mean(absolute_bias_samples[:, lead_index] > 0.0)
                    ),
                    "absolute_bias_approximate_interval_above_zero": bool(
                        bias_lower[lead_index] > 0.0
                    ),
                    "bootstrap_replicates": replicates,
                    "bootstrap_block_length_initializations": block_length,
                    "bootstrap_seed": seed,
                    "uncertainty_interpretation": (
                        "approximate paired percentile interval; two years only; "
                        "not a significance test"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _matrix(summary: pd.DataFrame, column: str) -> np.ndarray:
    values = summary.pivot(
        index="initialization_month", columns="lead_week", values=column
    ).reindex(index=INITIALIZATION_MONTHS, columns=LEAD_WEEKS)
    array = values.to_numpy(dtype=np.float64)
    if array.shape != (4, 6) or not np.isfinite(array).all():
        raise DiagnosticContractError(f"cannot form month/lead matrix for {column}")
    return array


def _month_tick_labels(summary: pd.DataFrame) -> list[str]:
    counts = summary.groupby("initialization_month").n_initializations.first()
    return [
        f"{calendar.month_name[month]}  (n={int(counts.loc[month])})"
        for month in INITIALIZATION_MONTHS
    ]


def _draw_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    value_format: str,
    cmap: Any,
    norm: Normalize,
    month_labels: Sequence[str] | None,
    bold_mask: np.ndarray | None = None,
) -> Any:
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    axis.set_title(title, fontsize=12.2, fontweight="semibold", pad=9)
    axis.set_xticks(range(6), [f"W{lead}" for lead in LEAD_WEEKS])
    axis.set_yticks(
        range(4), month_labels if month_labels is not None else [""] * 4
    )
    axis.tick_params(length=0, labelsize=9.5)
    axis.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.6)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(4):
        for column in range(6):
            rgba = cmap(norm(float(values[row, column])))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column,
                row,
                value_format.format(values[row, column]),
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight=(
                    "bold"
                    if bold_mask is not None and bool(bold_mask[row, column])
                    else "normal"
                ),
                color="white" if luminance < 0.48 else "#152632",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def plot_improvement_heatmaps(
    summary: pd.DataFrame,
    output_stem: Path,
    *,
    selected_configuration: str,
    dpi: int,
) -> None:
    acc_difference = _matrix(summary, "acc_difference")
    rmse_reduction = _matrix(summary, "rmse_reduction_pct")
    absolute_bias_improvement = _matrix(
        summary, "absolute_bias_improvement_mm_day"
    )
    acc_supported = _matrix(
        summary, "acc_approximate_interval_above_zero"
    ).astype(bool)
    rmse_supported = _matrix(
        summary, "rmse_approximate_interval_above_zero"
    ).astype(bool)
    bias_supported = _matrix(
        summary, "absolute_bias_approximate_interval_above_zero"
    ).astype(bool)
    month_labels = _month_tick_labels(summary)
    acc_limit = max(0.02, float(np.max(np.abs(acc_difference))))
    rmse_limit = max(2.0, float(np.max(np.abs(rmse_reduction))))
    bias_limit = max(0.1, float(np.max(np.abs(absolute_bias_improvement))))
    acc_effect_norm = TwoSlopeNorm(vmin=-acc_limit, vcenter=0.0, vmax=acc_limit)
    rmse_effect_norm = TwoSlopeNorm(vmin=-rmse_limit, vcenter=0.0, vmax=rmse_limit)
    bias_effect_norm = TwoSlopeNorm(vmin=-bias_limit, vcenter=0.0, vmax=bias_limit)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#152632",
            "text.color": "#152632",
            "xtick.color": "#314A59",
            "ytick.color": "#314A59",
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(16.7, 6.0),
            facecolor="#F4F7F8",
            gridspec_kw={"wspace": 0.17},
        )
        panels = (
            (
                axes[0],
                acc_difference,
                "Spatial pattern skill · ΔACC",
                "{:+.3f}",
                IMPROVEMENT_CMAP,
                acc_effect_norm,
                acc_supported,
            ),
            (
                axes[1],
                rmse_reduction,
                "Magnitude error · RMSE reduction",
                "{:+.1f}%",
                IMPROVEMENT_CMAP,
                rmse_effect_norm,
                rmse_supported,
            ),
            (
                axes[2],
                absolute_bias_improvement,
                "National mean · absolute-bias improvement",
                "{:+.2f}",
                IMPROVEMENT_CMAP,
                bias_effect_norm,
                bias_supported,
            ),
        )
        images = []
        for index, (axis, values, title, fmt, cmap, norm, bold) in enumerate(panels):
            images.append(
                _draw_heatmap(
                    axis,
                    values,
                    title=title,
                    value_format=fmt,
                    cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
                    norm=norm,
                    month_labels=month_labels if index == 0 else None,
                    bold_mask=bold,
                )
            )
            axis.set_xlabel("Lead week", fontsize=9.5)
        colorbar_labels = (
            "Corrected Forecast − Raw FuXi-S2S ACC",
            "Reduction (%) · positive is better",
            r"$|\overline{bias}_{Raw\ FuXi}|-|\overline{bias}_{Corrected\ Forecast}|$ (mm day$^{-1}$)",
        )
        for axis, image, label in zip(axes, images, colorbar_labels):
            colorbar = figure.colorbar(
                image,
                ax=axis,
                orientation="horizontal",
                fraction=0.055,
                pad=0.10,
                aspect=28,
            )
            colorbar.set_label(label, fontsize=8.2)
            colorbar.ax.tick_params(labelsize=7.8, length=2)
            colorbar.outline.set_visible(False)
        figure.suptitle(
            "When does Corrected Forecast improve skill?  Month × lead week",
            x=0.055,
            y=0.975,
            ha="left",
            fontsize=19,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.905,
            "India-area-weighted rainfall · JJAS 2020–2021 exploratory/reused hindcasts · not independent confirmation",
            ha="left",
            fontsize=11,
            color="#526A78",
        )
        figure.text(
            0.985,
            0.905,
            f"Corrected Forecast configuration: {selected_configuration}",
            ha="right",
            fontsize=9.3,
            color="#526A78",
        )
        figure.text(
            0.055,
            0.018,
            "Positive = Corrected Forecast improves on Raw FuXi-S2S. Bias compares the absolute value of each month/lead national mean bias, not mean case-wise |bias|. Bold cells: approximate paired 95% percentile interval above zero; descriptive only, no significance claim. No refitting or target/source reopening.",
            ha="left",
            fontsize=8.3,
            color="#526A78",
        )
        figure.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.18)
        _save_figure(figure, output_stem, dpi)
        plt.close(figure)


def plot_paired_uncertainty(
    summary: pd.DataFrame,
    output_stem: Path,
    *,
    replicates: int,
    block_length: int,
    dpi: int,
) -> None:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#AEBCC5",
            "axes.labelcolor": "#203643",
            "xtick.color": "#3D5360",
            "ytick.color": "#3D5360",
            "text.color": "#152632",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(18.2, 6.45), facecolor="white")
        offsets = {6: -0.18, 7: -0.06, 8: 0.06, 9: 0.18}
        specifications = (
            (
                axes[0],
                "acc_difference",
                "acc_difference_ci95_lower",
                "acc_difference_ci95_upper",
                "acc_approximate_interval_above_zero",
                "ACC improvement",
                "Corrected Forecast − Raw FuXi-S2S ACC",
            ),
            (
                axes[1],
                "rmse_reduction_pct",
                "rmse_reduction_ci95_lower_pct",
                "rmse_reduction_ci95_upper_pct",
                "rmse_approximate_interval_above_zero",
                "RMSE improvement",
                "Mean RMSE reduction (%)",
            ),
            (
                axes[2],
                "absolute_bias_improvement_mm_day",
                "absolute_bias_improvement_ci95_lower_mm_day",
                "absolute_bias_improvement_ci95_upper_mm_day",
                "absolute_bias_approximate_interval_above_zero",
                "Absolute-bias tradeoff",
                r"$|\overline{bias}_{Raw\ FuXi}|-|\overline{bias}_{Corrected\ Forecast}|$ (mm day$^{-1}$)",
            ),
        )
        for axis, point_name, lower_name, upper_name, support_name, title, ylabel in specifications:
            for month in INITIALIZATION_MONTHS:
                group = summary.loc[
                    summary.initialization_month.eq(month)
                ].sort_values("lead_week")
                point = group[point_name].to_numpy(dtype=np.float64)
                lower = group[lower_name].to_numpy(dtype=np.float64)
                upper = group[upper_name].to_numpy(dtype=np.float64)
                supported = group[support_name].to_numpy(dtype=bool)
                x = np.asarray(LEAD_WEEKS, dtype=np.float64) + offsets[month]
                color = MONTH_COLORS[month]
                count = int(group.n_initializations.iloc[0])
                axis.plot(
                    x,
                    point,
                    color=color,
                    linewidth=1.45,
                    alpha=0.80,
                    zorder=2,
                )
                axis.errorbar(
                    x,
                    point,
                    yerr=np.maximum(
                        0.0, np.vstack((point - lower, upper - point))
                    ),
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.35,
                    capsize=2.8,
                    capthick=1.2,
                    alpha=0.88,
                    zorder=2,
                )
                axis.scatter(
                    x,
                    point,
                    s=48,
                    marker="o",
                    facecolors=np.where(supported, color, "white"),
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                    label=f"{calendar.month_name[month]}  n={count}",
                )
            axis.axhline(0.0, color="#263B47", linewidth=1.0, linestyle="--")
            axis.set_title(title, fontsize=14, fontweight="semibold", pad=10)
            axis.set_xlabel("Lead week")
            axis.set_ylabel(ylabel)
            axis.set_xticks(LEAD_WEEKS, [f"W{lead}" for lead in LEAD_WEEKS])
            axis.grid(axis="y", color="#D6E0E5", linewidth=0.8, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(
                0.01,
                0.98,
                "positive = Corrected Forecast improves",
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                color="#607480",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.89),
            ncol=4,
            frameon=False,
            fontsize=9.5,
        )
        figure.suptitle(
            "Corrected Forecast: paired month × lead uncertainty",
            x=0.055,
            y=0.975,
            ha="left",
            fontsize=19,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.915,
            "JJAS 2020–2021 exploratory/reused hindcasts · not independent confirmation",
            ha="left",
            fontsize=11,
            color="#526A78",
        )
        figure.text(
            0.5,
            0.018,
            f"Whiskers: 95% percentile intervals from {replicates:,} paired month-stratified two-stage circular moving-block resamples (2 years, then {block_length}-initialization blocks; all six leads retained). Filled markers: interval above zero. Bias uses |month/lead mean bias|, not mean |case bias|. Approximate with only two years; no p-values or significance claim.",
            ha="center",
            fontsize=8.3,
            color="#526A78",
        )
        figure.subplots_adjust(left=0.065, right=0.985, top=0.75, bottom=0.15, wspace=0.28)
        _save_figure(figure, output_stem, dpi)
        plt.close(figure)


def build_diagnostic(
    locked_evaluation_directory: Path,
    output_directory: Path,
    *,
    replicates: int = DEFAULT_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_SEED,
    dpi: int = 220,
) -> Path:
    """Validate locked artifacts and write a separate derived diagnostic."""

    locked_directory = Path(locked_evaluation_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh output is required: {output}")
    if output == locked_directory or locked_directory in output.parents:
        raise ValueError("diagnostic output must not be inside locked evaluation")
    if dpi < 100:
        raise ValueError("dpi must be at least 100")
    output.mkdir(parents=True, exist_ok=False)
    manifest, frame, case_metrics_path = load_locked_case_metrics(locked_directory)
    summary = summarize_month_lead(
        frame,
        replicates=replicates,
        block_length=block_length,
        seed=seed,
    )
    summary.insert(3, "selected_configuration", manifest["selected_configuration"])
    csv_path = output / "jjas_initialization_month_by_lead_summary.csv"
    _atomic_to_csv(summary, csv_path)
    plot_improvement_heatmaps(
        summary,
        output / "jjas_month_lead_improvement_tradeoffs",
        selected_configuration=str(manifest["selected_configuration"]),
        dpi=dpi,
    )
    plot_paired_uncertainty(
        summary,
        output / "jjas_month_lead_paired_uncertainty",
        replicates=replicates,
        block_length=block_length,
        dpi=dpi,
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "diagnostic_manifest.json"
    }
    sample_counts = {
        calendar.month_name[month]: int(
            summary.loc[
                summary.initialization_month.eq(month), "n_initializations"
            ].iloc[0]
        )
        for month in INITIALIZATION_MONTHS
    }
    diagnostic_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        "evaluation_role": EVALUATION_ROLE,
        "evaluation_scope": EVALUATION_SCOPE,
        "test_years": list(TEST_YEARS),
        "initialization_months": list(INITIALIZATION_MONTHS),
        "lead_weeks": list(LEAD_WEEKS),
        "selected_configuration": manifest["selected_configuration"],
        "method_labels": {
            RAW_METHOD: "Raw FuXi-S2S",
            CORRECTED_METHOD: "Corrected Forecast",
        },
        "selection_locked_before_target_access": True,
        "test_used_for_selection": False,
        "parameter_updates": 0,
        "reused_test_period": True,
        "genuine_independent_test": False,
        "source_locked_evaluation_directory": str(locked_directory),
        "source_manifest": str(locked_directory / "manifest.json"),
        "source_manifest_sha256": sha256_file(locked_directory / "manifest.json"),
        "source_case_metrics": str(case_metrics_path),
        "source_case_metrics_sha256": sha256_file(case_metrics_path),
        "source_artifacts_read": ["manifest.json", "metrics/test_case_metrics.csv"],
        "forecast_prediction_observation_or_target_arrays_opened": False,
        "metrics_computed_from_locked_case_metrics_only": True,
        "sample_counts_by_initialization_month": sample_counts,
        "uncertainty": {
            "design": (
                "paired month-stratified two-stage circular moving-block bootstrap: "
                "resample the two years, then initialization blocks within each "
                "month; retain all six leads together"
            ),
            "replicates": replicates,
            "block_length_initializations": block_length,
            "seed": seed,
            "interval": "two-sided 95% percentile interval",
            "p_values_computed": False,
            "multiple_testing_claim": False,
            "significance_claimed": False,
            "interpretation": (
                "approximate paired uncertainty only; the reused diagnostic has "
                "two year clusters and is not independent confirmation"
            ),
        },
        "artifacts": artifacts,
    }
    _atomic_write_text(
        output / "diagnostic_manifest.json",
        json.dumps(diagnostic_manifest, indent=2) + "\n",
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locked_evaluation_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_diagnostic(
        args.locked_evaluation_directory,
        args.output_directory,
        replicates=int(args.replicates),
        block_length=int(args.block_length),
        seed=int(args.seed),
        dpi=int(args.dpi),
    )
    print(f"PASS: locked JJAS month/lead diagnostic: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
