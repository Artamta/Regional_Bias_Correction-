#!/usr/bin/env python3
"""Create validation-only publication figures for a physical FuXi--IMD sweep.

The input is a *completed* ``fuxi_imd_compact_validation_sweep.py`` output
directory.  Only the blocked 2018--2019 validation residuals are reconstructed;
the quarantined 2020+ cases are never selected, scored, or written.

The figures deliberately remain descriptive.  In particular, the spatial maps
do not imply pixel-wise statistical significance.  A paired, year-stratified
moving-block bootstrap is supplied separately for India-area-weighted aggregate
scores so overlapping twice-weekly initializations are not treated as IID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


HERE = Path(__file__).resolve().parents[1]
DEFAULT_INDIA_BOUNDARY = (
    HERE.parent / "fuxi-dashboard" / "public" / "data" / "india-admin.json"
)
CONTROL_CONFIGURATION = "physical_control"
VALIDATION_YEARS = (2018, 2019)
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#0072B2",
    CONTROL_CONFIGURATION: "#E69F00",
    "best_physical": "#009E73",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    CONTROL_CONFIGURATION: "^",
    "best_physical": "D",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_india_boundary(
    path: Path = DEFAULT_INDIA_BOUNDARY,
) -> tuple[tuple[np.ndarray, ...], Mapping[str, Any]]:
    """Load the checked Survey of India ABDB display derivative.

    The dashboard asset is a WGS84 GeoJSON derivative of the locally archived
    Survey of India state/UT shapefile.  Reading the compact derivative keeps
    this plotting path independent of GDAL while retaining the checked source
    checksum and complete supplied depiction.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"India boundary is required for spatial figures: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("India boundary is not a GeoJSON FeatureCollection")
    source = payload.get("source")
    if not isinstance(source, Mapping) or "Survey of India" not in str(
        source.get("name", "")
    ):
        raise ValueError("India boundary lacks Survey of India provenance")
    features = payload.get("features")
    if not isinstance(features, list) or len(features) < 30:
        raise ValueError("India boundary has incomplete state/UT coverage")

    segments: list[np.ndarray] = []
    for feature in features:
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            raise ValueError(f"unsupported India geometry type: {geometry_type!r}")
        for polygon in polygons:
            for ring in polygon:
                values = np.asarray(ring, dtype=np.float64)
                if (
                    values.ndim != 2
                    or values.shape[0] < 4
                    or values.shape[1] != 2
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("India boundary contains an invalid ring")
                if values[:, 0].min() < 60.0 or values[:, 0].max() > 100.0:
                    raise ValueError("India boundary longitude is outside the study grid")
                if values[:, 1].min() < 0.0 or values[:, 1].max() > 40.0:
                    raise ValueError("India boundary latitude is outside the study grid")
                segments.append(values)
    if len(segments) < len(features):
        raise ValueError("India boundary contains too few polygon rings")
    provenance = {
        "boundary_path": str(path),
        "boundary_sha256": sha256_file(path),
        "feature_count": len(features),
        "ring_count": len(segments),
        "source": dict(source),
    }
    return tuple(segments), provenance


def select_best_physical_candidate(
    ranking: pd.DataFrame,
    available_configurations: Sequence[str],
    *,
    control_configuration: str = CONTROL_CONFIGURATION,
    explicit_candidate: str | None = None,
) -> str:
    """Select the best ranked non-control physical candidate with residuals."""

    if "configuration" not in ranking:
        raise ValueError("ranking lacks a configuration column")
    available = set(str(value) for value in available_configurations)
    physical = ranking.loc[
        ranking.configuration.astype(str).str.startswith("physical_")
        & ranking.configuration.ne(control_configuration)
        & ranking.configuration.isin(available)
    ].copy()
    if explicit_candidate is not None:
        if explicit_candidate == control_configuration:
            raise ValueError("the best physical candidate cannot be the control")
        if explicit_candidate not in available:
            raise ValueError(
                f"requested candidate {explicit_candidate!r} has no ensemble residual"
            )
        if explicit_candidate not in set(physical.configuration):
            raise ValueError(
                f"requested candidate {explicit_candidate!r} is not a physical candidate"
            )
        return explicit_candidate
    if physical.empty:
        raise ValueError("no completed non-control physical candidate was found")
    if "rank" in physical and np.isfinite(
        pd.to_numeric(physical["rank"], errors="coerce")
    ).any():
        physical["_selection"] = pd.to_numeric(physical["rank"], errors="coerce")
    elif "mean_best_validation_loss" in physical:
        physical["_selection"] = pd.to_numeric(
            physical["mean_best_validation_loss"], errors="coerce"
        )
    elif "ensemble_validation_loss" in physical:
        physical["_selection"] = pd.to_numeric(
            physical["ensemble_validation_loss"], errors="coerce"
        )
    else:
        raise ValueError("ranking has no usable rank or validation-loss column")
    if not np.isfinite(physical._selection).any():
        raise ValueError("physical candidate ranking values are all non-finite")
    return str(
        physical.sort_values(["_selection", "configuration"], kind="stable")
        .iloc[0]
        .configuration
    )


def _validate_fields(
    prediction: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and truth must share [case, lead, latitude, longitude] shape"
        )
    if weights.shape != prediction.shape[-2:]:
        raise ValueError("weights do not match the spatial grid")
    support = np.isfinite(weights) & (weights > 0.0)
    if not np.any(support):
        raise ValueError("positive verification support is empty")
    if not np.isfinite(prediction[..., support]).all():
        raise ValueError("prediction is non-finite on verification support")
    if not np.isfinite(truth[..., support]).all():
        raise ValueError("truth is non-finite on verification support")
    return prediction, truth, weights, support


def spatial_mean_bias(
    prediction: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return validation-case mean error at every lead and supported cell."""

    prediction, truth, _, support = _validate_fields(prediction, truth, weights)
    result = np.mean(prediction - truth, axis=0, dtype=np.float64)
    result[:, ~support] = np.nan
    return result


def spatial_rmse_skill_vs_control(
    candidate: np.ndarray,
    control: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    *,
    minimum_control_rmse: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local RMSE skill percentage, candidate RMSE, and control RMSE.

    Positive skill means the candidate has lower validation RMSE.  RMSE is
    reduced over initialization cases independently at each lead and grid cell.
    """

    candidate, truth, _, support = _validate_fields(candidate, truth, weights)
    control, _, _, _ = _validate_fields(control, truth, weights)
    candidate_rmse = np.sqrt(
        np.mean((candidate - truth) ** 2, axis=0, dtype=np.float64)
    )
    control_rmse = np.sqrt(np.mean((control - truth) ** 2, axis=0, dtype=np.float64))
    skill = np.full(candidate_rmse.shape, np.nan, dtype=np.float64)
    valid = support[None] & (control_rmse > minimum_control_rmse)
    skill[valid] = (
        100.0
        * (control_rmse[valid] - candidate_rmse[valid])
        / control_rmse[valid]
    )
    candidate_rmse[:, ~support] = np.nan
    control_rmse[:, ~support] = np.nan
    return skill, candidate_rmse, control_rmse


def area_weighted_case_rmse(
    prediction: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return one India-area-weighted RMSE per case and lead."""

    prediction, truth, weights, support = _validate_fields(
        prediction, truth, weights
    )
    normalized_weights = weights[support] / weights[support].sum()
    squared_error = (prediction[..., support] - truth[..., support]) ** 2
    return np.sqrt(
        np.sum(squared_error * normalized_weights[None, None], axis=-1)
    )


def _metric_matrices(
    case_metrics: pd.DataFrame, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"method", "case_id", "lead", "rmse", "acc", "bias"}
    missing = required.difference(case_metrics.columns)
    if missing:
        raise ValueError(f"case metrics lack columns: {sorted(missing)}")
    selected = case_metrics.loc[case_metrics.method.eq(method)].copy()
    selected["case_id"] = pd.to_datetime(selected.case_id).dt.normalize()
    if selected.duplicated(["case_id", "lead"]).any():
        raise ValueError(f"duplicate case/lead metrics for {method}")
    dates = np.sort(selected.case_id.unique())
    leads = np.arange(1, 7)
    matrices = []
    for metric in ("rmse", "acc", "bias"):
        matrix = (
            selected.pivot(index="case_id", columns="lead", values=metric)
            .reindex(index=dates, columns=leads)
            .to_numpy(dtype=np.float64)
        )
        if matrix.shape != (len(dates), 6) or not np.isfinite(matrix).all():
            raise ValueError(f"incomplete {metric} matrix for {method}")
        matrices.append(matrix)
    return np.asarray(dates), matrices[0], matrices[1], matrices[2]


def build_all_lead_guard_table(
    case_metrics: pd.DataFrame,
    candidate: str,
    control: str = CONTROL_CONFIGURATION,
) -> tuple[pd.DataFrame, Mapping[str, bool]]:
    """Build the six-lead candidate-minus-control performance guard table."""

    dates, control_rmse, control_acc, control_bias = _metric_matrices(
        case_metrics, control
    )
    candidate_dates, candidate_rmse, candidate_acc, candidate_bias = (
        _metric_matrices(case_metrics, candidate)
    )
    if not np.array_equal(dates, candidate_dates):
        raise ValueError("candidate and control case dates differ")
    rows: list[dict[str, Any]] = []
    for lead_index in range(6):
        control_rmse_mean = float(control_rmse[:, lead_index].mean())
        candidate_rmse_mean = float(candidate_rmse[:, lead_index].mean())
        control_acc_mean = float(control_acc[:, lead_index].mean())
        candidate_acc_mean = float(candidate_acc[:, lead_index].mean())
        control_abs_bias = abs(float(control_bias[:, lead_index].mean()))
        candidate_abs_bias = abs(float(candidate_bias[:, lead_index].mean()))
        rmse_delta = candidate_rmse_mean - control_rmse_mean
        acc_delta = candidate_acc_mean - control_acc_mean
        abs_bias_delta = candidate_abs_bias - control_abs_bias
        rows.append(
            {
                "lead": f"W{lead_index + 1}",
                "control_rmse": control_rmse_mean,
                "candidate_rmse": candidate_rmse_mean,
                "rmse_delta_candidate_minus_control": rmse_delta,
                "rmse_improved": bool(rmse_delta < 0.0),
                "control_acc": control_acc_mean,
                "candidate_acc": candidate_acc_mean,
                "acc_delta_candidate_minus_control": acc_delta,
                "acc_improved": bool(acc_delta > 0.0),
                "control_abs_bias": control_abs_bias,
                "candidate_abs_bias": candidate_abs_bias,
                "abs_bias_delta_candidate_minus_control": abs_bias_delta,
                "abs_bias_improved": bool(abs_bias_delta < 0.0),
                "all_three_improved": bool(
                    rmse_delta < 0.0 and acc_delta > 0.0 and abs_bias_delta < 0.0
                ),
            }
        )
    table = pd.DataFrame(rows)
    guards = {
        "rmse_improved_every_lead": bool(table.rmse_improved.all()),
        "acc_improved_every_lead": bool(table.acc_improved.all()),
        "absolute_bias_improved_every_lead": bool(table.abs_bias_improved.all()),
        "all_three_metrics_improved_every_lead": bool(
            table.all_three_improved.all()
        ),
    }
    for name, value in guards.items():
        table[name] = value
    return table, guards


def stratified_paired_bootstrap(
    case_metrics: pd.DataFrame,
    candidate: str,
    control: str = CONTROL_CONFIGURATION,
    *,
    replicates: int = 2000,
    seed: int = 20260812,
    block_length: int = 13,
) -> pd.DataFrame:
    """Moving-block bootstrap paired metric deltas within validation years.

    Circular blocks of consecutive initialization cases are sampled separately
    within 2018 and 2019.  All six lead scores from an initialization move
    together.  This preserves method pairing, year balance, serial dependence,
    and within-case lead dependence.
    """

    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    if block_length < 1:
        raise ValueError("bootstrap block length must be positive")
    dates, control_rmse, control_acc, control_bias = _metric_matrices(
        case_metrics, control
    )
    candidate_dates, candidate_rmse, candidate_acc, candidate_bias = (
        _metric_matrices(case_metrics, candidate)
    )
    if not np.array_equal(dates, candidate_dates):
        raise ValueError("candidate and control case dates differ")
    years = pd.DatetimeIndex(dates).year.to_numpy()
    if tuple(sorted(np.unique(years))) != VALIDATION_YEARS:
        raise ValueError("bootstrap inputs must contain only 2018 and 2019")
    rng = np.random.default_rng(seed)
    draws = []
    for year in VALIDATION_YEARS:
        positions = np.flatnonzero(years == year)
        if positions.size == 0:
            raise ValueError(f"validation year {year} has no cases")
        effective_block = min(int(block_length), int(positions.size))
        block_count = int(np.ceil(positions.size / effective_block))
        starts = rng.integers(0, positions.size, size=(replicates, block_count))
        offsets = np.arange(effective_block, dtype=np.int64)
        within_year = (starts[..., None] + offsets) % positions.size
        within_year = within_year.reshape(replicates, -1)[:, : positions.size]
        draws.append(positions[within_year])
    sample = np.concatenate(draws, axis=1)
    candidate_rmse_draw = candidate_rmse[sample]
    control_rmse_draw = control_rmse[sample]
    candidate_acc_draw = candidate_acc[sample]
    control_acc_draw = control_acc[sample]
    candidate_bias_draw = candidate_bias[sample]
    control_bias_draw = control_bias[sample]

    rmse_delta = (candidate_rmse_draw - control_rmse_draw).mean(axis=1)
    acc_delta = (candidate_acc_draw - control_acc_draw).mean(axis=1)
    absolute_bias_delta = np.abs(candidate_bias_draw.mean(axis=1)) - np.abs(
        control_bias_draw.mean(axis=1)
    )
    point_rmse = (candidate_rmse - control_rmse).mean(axis=0)
    point_acc = (candidate_acc - control_acc).mean(axis=0)
    point_absolute_bias = np.abs(candidate_bias.mean(axis=0)) - np.abs(
        control_bias.mean(axis=0)
    )

    rows: list[dict[str, Any]] = []
    metric_values = (
        ("rmse_delta_candidate_minus_control", rmse_delta, point_rmse, "lower"),
        ("acc_delta_candidate_minus_control", acc_delta, point_acc, "higher"),
        (
            "abs_bias_delta_candidate_minus_control",
            absolute_bias_delta,
            point_absolute_bias,
            "lower",
        ),
    )
    for metric, distribution, point, direction in metric_values:
        for lead_index in range(6):
            values = distribution[:, lead_index]
            rows.append(
                {
                    "scope": f"W{lead_index + 1}",
                    "metric": metric,
                    "point_delta": float(point[lead_index]),
                    "ci_lower_2p5": float(np.quantile(values, 0.025)),
                    "ci_upper_97p5": float(np.quantile(values, 0.975)),
                    "bootstrap_probability_improved": float(
                        np.mean(values < 0.0)
                        if direction == "lower"
                        else np.mean(values > 0.0)
                    ),
                    "replicates": int(replicates),
                    "seed": int(seed),
                    "block_length_initializations": int(block_length),
                }
            )
        if metric.startswith("rmse"):
            pooled_distribution = distribution.mean(axis=1)
            pooled_point = float(point.mean())
        elif metric.startswith("acc"):
            pooled_distribution = distribution.mean(axis=1)
            pooled_point = float(point.mean())
        else:
            pooled_distribution = np.abs(candidate_bias_draw.mean(axis=(1, 2))) - np.abs(
                control_bias_draw.mean(axis=(1, 2))
            )
            pooled_point = float(
                abs(candidate_bias.mean()) - abs(control_bias.mean())
            )
        rows.append(
            {
                "scope": "ALL_WEEKS",
                "metric": metric,
                "point_delta": pooled_point,
                "ci_lower_2p5": float(np.quantile(pooled_distribution, 0.025)),
                "ci_upper_97p5": float(np.quantile(pooled_distribution, 0.975)),
                "bootstrap_probability_improved": float(
                    np.mean(pooled_distribution < 0.0)
                    if direction == "lower"
                    else np.mean(pooled_distribution > 0.0)
                ),
                "replicates": int(replicates),
                "seed": int(seed),
                "block_length_initializations": int(block_length),
            }
        )
    return pd.DataFrame(rows)


def _robust_symmetric_limit(values: np.ndarray, minimum: float) -> float:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return minimum
    raw = max(minimum, float(np.quantile(finite, 0.98)))
    magnitude = 10.0 ** np.floor(np.log10(raw))
    return float(np.ceil(raw / magnitude * 2.0) / 2.0 * magnitude)


def _map_backend() -> tuple[Any | None, Any | None]:
    try:
        from cartopy import crs as ccrs
        from cartopy import feature as cfeature

        return ccrs, cfeature
    except ImportError:
        return None, None


def _decorate_map(
    axis: Any,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ccrs: Any | None,
    cfeature: Any | None,
    boundary_segments: Sequence[np.ndarray],
    *,
    left_labels: bool,
    bottom_labels: bool,
) -> None:
    if ccrs is None:
        axis.set_xlim(float(longitude.min()), float(longitude.max()))
        axis.set_ylim(float(latitude.min()), float(latitude.max()))
        axis.set_aspect("equal", adjustable="box")
        if left_labels:
            axis.set_ylabel("Latitude (°N)")
        if bottom_labels:
            axis.set_xlabel("Longitude (°E)")
        axis.grid(linewidth=0.35, alpha=0.25)
    else:
        projection = ccrs.PlateCarree()
        axis.coastlines(resolution="50m", linewidth=0.4, color="0.45")
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="0.55"
        )
        axis.set_extent(
            [
                float(longitude.min()) - 0.75,
                float(longitude.max()) + 0.75,
                float(latitude.min()) - 0.75,
                float(latitude.max()) + 0.75,
            ],
            crs=projection,
        )
        grid = axis.gridlines(
            draw_labels=True,
            linewidth=0.25,
            color="0.55",
            alpha=0.45,
            x_inline=False,
            y_inline=False,
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.left_labels = left_labels
        grid.bottom_labels = bottom_labels
    if boundary_segments:
        transform = (
            axis.transData
            if ccrs is None
            else ccrs.PlateCarree()._as_mpl_transform(axis)
        )
        boundaries = LineCollection(
            boundary_segments,
            colors="#161616",
            linewidths=0.28,
            alpha=0.90,
            zorder=6,
            transform=transform,
        )
        axis.add_collection(boundaries, autolim=False)


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_spatial_bias_maps(
    bias_fields: Mapping[str, np.ndarray],
    labels: Mapping[str, str],
    latitude: np.ndarray,
    longitude: np.ndarray,
    output: Path,
    boundary_segments: Sequence[np.ndarray] = (),
) -> None:
    methods = tuple(bias_fields)
    if len(methods) != 4:
        raise ValueError("spatial bias figure requires exactly four methods")
    fields = np.stack([bias_fields[method] for method in methods])
    limit = _robust_symmetric_limit(fields, 0.5)
    ccrs, cfeature = _map_backend()
    subplot_kw = {} if ccrs is None else {"projection": ccrs.PlateCarree()}
    figure, axes = plt.subplots(
        4, 6, figsize=(18.0, 11.0), subplot_kw=subplot_kw, squeeze=False
    )
    image = None
    for row, method in enumerate(methods):
        for lead in range(6):
            axis = axes[row, lead]
            kwargs = {}
            if ccrs is not None:
                kwargs["transform"] = ccrs.PlateCarree()
            image = axis.pcolormesh(
                longitude,
                latitude,
                np.ma.masked_invalid(fields[row, lead]),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                **kwargs,
            )
            _decorate_map(
                axis,
                latitude,
                longitude,
                ccrs,
                cfeature,
                boundary_segments,
                left_labels=lead == 0,
                bottom_labels=row == len(methods) - 1,
            )
            if row == 0:
                axis.set_title(f"Week {lead + 1}", fontsize=10.4, fontweight="semibold")
            if lead == 0:
                axis.text(
                    -0.24,
                    0.5,
                    labels[method],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10.0,
                    fontweight="semibold",
                )
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.045,
        aspect=55,
        extend="both",
    )
    colorbar.set_label("Mean bias (prediction − IMD; mm day$^{-1}$)")
    figure.suptitle(
        "Spatial mean bias across blocked validation cases\n"
        "JJAS 2018–2019 · IMD-supported India cells",
        fontsize=15,
        fontweight="semibold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.10, wspace=0.04, hspace=0.12)
    _save_figure(figure, output)


def plot_spatial_rmse_skill(
    skill: np.ndarray,
    summary: pd.DataFrame,
    candidate_label: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    output: Path,
    boundary_segments: Sequence[np.ndarray] = (),
) -> None:
    limit = _robust_symmetric_limit(skill, 2.0)
    ccrs, cfeature = _map_backend()
    subplot_kw = {} if ccrs is None else {"projection": ccrs.PlateCarree()}
    figure, axes = plt.subplots(
        2, 3, figsize=(12.8, 7.8), subplot_kw=subplot_kw, squeeze=False
    )
    image = None
    for lead, axis in enumerate(axes.ravel()):
        kwargs = {}
        if ccrs is not None:
            kwargs["transform"] = ccrs.PlateCarree()
        image = axis.pcolormesh(
            longitude,
            latitude,
            np.ma.masked_invalid(skill[lead]),
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
            **kwargs,
        )
        _decorate_map(
            axis,
            latitude,
            longitude,
            ccrs,
            cfeature,
            boundary_segments,
            left_labels=lead % 3 == 0,
            bottom_labels=lead >= 3,
        )
        row = summary.loc[summary.lead.eq(f"W{lead + 1}")].iloc[0]
        axis.set_title(f"Week {lead + 1}", fontsize=11, fontweight="semibold")
        axis.text(
            0.98,
            0.03,
            f"{row.area_fraction_improved_pct:.1f}% area improved\n"
            f"mean skill {row.area_weighted_mean_skill_pct:+.2f}%",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
        )
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.08,
        aspect=45,
        extend="both",
    )
    colorbar.set_label(
        "Local RMSE skill vs compact control (%) · positive is better"
    )
    figure.suptitle(
        f"{candidate_label} vs compact control\n"
        "Spatial RMSE skill · JJAS 2018–2019 blocked validation",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Descriptive point estimates over 70 initialization cases; no cell-wise significance claim.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.16, wspace=0.07, hspace=0.14)
    _save_figure(figure, output)


def plot_lead_metrics(
    summary: pd.DataFrame,
    methods: Sequence[str],
    labels: Mapping[str, str],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    specifications = (
        ("rmse", "RMSE (mm day$^{-1}$)", "Lower is better"),
        ("acc", "ACC", "Higher is better"),
        ("bias", "Bias (mm day$^{-1}$)", "Closer to zero is better"),
    )
    for method in methods:
        selected = summary.loc[summary.method.eq(method)].sort_values("lead")
        style_key = "best_physical" if method not in METHOD_COLORS else method
        for axis, (metric, ylabel, title) in zip(axes, specifications):
            axis.plot(
                selected.lead,
                selected[metric],
                color=METHOD_COLORS[style_key],
                marker=METHOD_MARKERS[style_key],
                linewidth=1.8,
                markersize=5,
                label=labels[method],
            )
            axis.set_xlabel("Lead week")
            axis.set_ylabel(ylabel)
            axis.set_title(title, fontsize=10.5, fontweight="semibold")
            axis.set_xticks(np.arange(1, 7))
            axis.grid(alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
    axes[2].axhline(0.0, color="0.25", linestyle="--", linewidth=0.9)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        "India-area-weighted skill by lead · blocked validation 2018–2019",
        fontsize=13.5,
        fontweight="semibold",
        y=1.02,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    _save_figure(figure, output)


def plot_time_lead_comparison(
    case_rmse: Mapping[str, np.ndarray],
    dates: np.ndarray,
    methods: Sequence[str],
    labels: Mapping[str, str],
    candidate: str,
    output: Path,
) -> None:
    dates = np.asarray(dates, dtype="datetime64[D]")
    years = pd.DatetimeIndex(dates).year.to_numpy()
    if tuple(sorted(np.unique(years))) != VALIDATION_YEARS:
        raise ValueError("time/lead plot must contain 2018 and 2019 only")
    x = np.arange(len(dates))
    control = case_rmse[CONTROL_CONFIGURATION]
    candidate_values = case_rmse[candidate]
    skill = 100.0 * (control - candidate_values) / np.maximum(control, 1.0e-8)
    limit = _robust_symmetric_limit(skill, 2.0)

    figure = plt.figure(figsize=(14.0, 7.5))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.05, 1.0), hspace=0.22)
    time_axis = figure.add_subplot(grid[0])
    heat_axis = figure.add_subplot(grid[1])
    for method in methods:
        style_key = "best_physical" if method not in METHOD_COLORS else method
        values = case_rmse[method].mean(axis=1)
        time_axis.plot(x, values, color=METHOD_COLORS[style_key], alpha=0.18, linewidth=0.7)
        smoothed = np.full(values.shape, np.nan, dtype=np.float64)
        for year in VALIDATION_YEARS:
            positions = np.flatnonzero(years == year)
            smoothed[positions] = (
                pd.Series(values[positions]).rolling(5, center=True, min_periods=2).mean()
            )
        time_axis.plot(
            x,
            smoothed,
            color=METHOD_COLORS[style_key],
            linewidth=2.0,
            label=labels[method],
        )
    boundary = np.flatnonzero(years == 2018).size - 0.5
    time_axis.axvline(boundary, color="0.4", linestyle="--", linewidth=0.9)
    time_axis.set_ylabel("Mean W1–W6 RMSE\n(mm day$^{-1}$)")
    time_axis.set_xlim(-0.5, len(dates) - 0.5)
    time_axis.grid(axis="y", alpha=0.22)
    time_axis.spines[["top", "right"]].set_visible(False)
    time_axis.legend(frameon=False, ncol=4, loc="upper center")
    time_axis.set_title(
        "Five-initialization running mean (faint lines show individual cases)",
        fontsize=10.3,
        fontweight="semibold",
    )

    image = heat_axis.imshow(
        skill.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        extent=(-0.5, len(dates) - 0.5, 6.5, 0.5),
    )
    heat_axis.axvline(boundary, color="0.15", linestyle="--", linewidth=0.9)
    heat_axis.set_yticks(np.arange(1, 7), [f"W{lead}" for lead in range(1, 7)])
    heat_axis.set_ylabel("Lead week")
    tick_positions = np.unique(
        np.r_[np.arange(0, len(dates), 7), len(dates) - 1]
    )
    tick_labels = [pd.Timestamp(dates[index]).strftime("%Y-%m-%d") for index in tick_positions]
    heat_axis.set_xticks(tick_positions, tick_labels, rotation=38, ha="right")
    heat_axis.set_xlabel("FuXi initialization date")
    colorbar = figure.colorbar(image, ax=heat_axis, pad=0.015, aspect=28)
    colorbar.set_label("Case RMSE skill vs compact control (%)")
    figure.suptitle(
        f"Temporal and lead-wise validation behavior · {labels[candidate]}\n"
        "India-area-weighted scores · JJAS 2018–2019",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    figure.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.12)
    _save_figure(figure, output)


def _validate_completed_sweep(
    sweep_directory: Path, *, allow_smoke: bool
) -> Mapping[str, Any]:
    manifest_path = sweep_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("sweep manifest is not complete")
    if manifest.get("test_predictions_created") is not False:
        raise ValueError("sweep does not declare its test period untouched")
    if tuple(manifest.get("validation_years", ())) != VALIDATION_YEARS:
        raise ValueError("expected the blocked 2018--2019 validation split")
    if manifest.get("smoke") and not allow_smoke:
        raise ValueError("refusing publication plots from a smoke run; use --allow-smoke")
    if int(manifest.get("split_counts", {}).get("validation", -1)) != 70:
        raise ValueError("expected exactly 70 validation initialization cases")
    return manifest


def _available_ensemble_residuals(sweep_directory: Path) -> tuple[str, ...]:
    models = sweep_directory / "models"
    if not models.is_dir():
        raise FileNotFoundError(models)
    return tuple(
        sorted(
            path.parent.name
            for path in models.glob("*/validation_residual_ensemble.npy")
            if path.is_file()
        )
    )


def _canonical_grid_from_features(
    features: np.ndarray, validation_index: int, normalization: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(normalization.get("input_channels", ()))
    if len(names) < 5 or names[3:5] != ("latitude", "longitude"):
        raise ValueError("feature coordinate-channel contract differs")
    lat_scaled = np.asarray(features[validation_index, 0, 3, :, 0], dtype=np.float64)
    lon_scaled = np.asarray(features[validation_index, 0, 4, 0, :], dtype=np.float64)
    latitude = 19.5 * (lat_scaled + 1.0)
    longitude = 60.0 + 19.5 * (lon_scaled + 1.0)
    if (
        latitude.shape != (27,)
        or longitude.shape != (27,)
        or not np.allclose(np.sort(latitude), np.linspace(0.0, 39.0, 27), atol=2e-5)
        or not np.allclose(longitude, np.linspace(60.0, 99.0, 27), atol=2e-5)
    ):
        raise ValueError("could not recover the frozen 0--39N, 60--99E grid")
    return latitude, longitude


def _build_spatial_summary(
    skill: np.ndarray,
    candidate_rmse: np.ndarray,
    control_rmse: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    support = weights > 0.0
    rows = []
    denominator = float(weights[support].sum())
    for lead in range(6):
        field = skill[lead]
        valid = support & np.isfinite(field)
        rows.append(
            {
                "lead": f"W{lead + 1}",
                "area_fraction_improved_pct": float(
                    100.0 * weights[valid & (field > 0.0)].sum() / denominator
                ),
                "area_weighted_mean_skill_pct": float(
                    np.sum(weights[valid] * field[valid]) / weights[valid].sum()
                ),
                "area_weighted_mean_control_local_rmse": float(
                    np.sum(weights[support] * control_rmse[lead, support]) / denominator
                ),
                "area_weighted_mean_candidate_local_rmse": float(
                    np.sum(weights[support] * candidate_rmse[lead, support]) / denominator
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    sweep_directory = Path(args.sweep_directory).expanduser().resolve()
    manifest = _validate_completed_sweep(
        sweep_directory, allow_smoke=bool(args.allow_smoke)
    )
    ranking_path = sweep_directory / "metrics" / "ranked_configurations.csv"
    ranking = pd.read_csv(ranking_path)
    available = _available_ensemble_residuals(sweep_directory)
    if CONTROL_CONFIGURATION not in available:
        raise ValueError(
            f"completed sweep lacks {CONTROL_CONFIGURATION!r} ensemble residual"
        )
    candidate = select_best_physical_candidate(
        ranking,
        available,
        explicit_candidate=args.candidate,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else sweep_directory / "publication_validation"
    )
    if output.exists():
        raise FileExistsError(
            f"output already exists (existing files are preserved): {output}"
        )
    if output == sweep_directory or sweep_directory in output.parents and output.name in {
        "models",
        "metrics",
        "figures",
        "code",
    }:
        raise ValueError("output must not replace a sweep artifact directory")
    (output / "figures").mkdir(parents=True, exist_ok=False)
    (output / "metrics").mkdir(parents=True, exist_ok=False)

    source_root = HERE / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import fuxi_imd_compact_validation_sweep as sweep
    from fuxi_adapter.anchored import reconstruct_anchored_precipitation

    selected_candidates = (
        sweep.CANDIDATE_BY_NAME[CONTROL_CONFIGURATION],
        sweep.CANDIDATE_BY_NAME[candidate],
    )
    prepared, normalization, _ = sweep.prepare_data(selected_candidates)
    indices = np.asarray(prepared.validation_indices, dtype=np.int64)
    dates = np.asarray(prepared.initializations[indices], dtype="datetime64[D]")
    years = pd.DatetimeIndex(dates).year.to_numpy()
    if indices.shape != (70,) or tuple(sorted(np.unique(years))) != VALIDATION_YEARS:
        raise ValueError("prepared validation selection is not exactly 2018--2019")
    if np.any(years >= 2020):
        raise ValueError("quarantined cases entered validation postprocessing")
    latitude, longitude = _canonical_grid_from_features(
        prepared.features, int(indices[0]), normalization
    )
    boundary_segments, boundary_provenance = load_india_boundary(
        args.india_boundary
    )
    anchor_path = sweep_directory / "models" / "log_bias_anchor.npz"
    with np.load(anchor_path, allow_pickle=False) as anchor:
        stored_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
    if not np.array_equal(stored_scale, prepared.target_scale):
        raise ValueError("saved anchor target scale differs from current preparation")

    truth = np.asarray(prepared.truth[indices], dtype=np.float32)
    climatology = np.asarray(prepared.climatology[indices], dtype=np.float32)
    valid = np.asarray(prepared.valid_mask[indices], dtype=bool)
    predictions: dict[str, np.ndarray] = {
        "raw_fuxi": np.asarray(prepared.raw_fuxi[indices], dtype=np.float32),
        "log_bias": np.asarray(prepared.bias_baseline[indices], dtype=np.float32),
    }
    for configuration in (CONTROL_CONFIGURATION, candidate):
        residual_path = (
            sweep_directory
            / "models"
            / configuration
            / "validation_residual_ensemble.npy"
        )
        residual = np.load(residual_path, allow_pickle=False)
        if residual.shape != truth.shape or not np.isfinite(residual).all():
            raise ValueError(f"invalid validation residual: {residual_path}")
        predictions[configuration] = reconstruct_anchored_precipitation(
            prepared.bias_baseline[indices],
            residual,
            prepared.target_scale,
            valid_mask=valid,
        )
    methods = ("raw_fuxi", "log_bias", CONTROL_CONFIGURATION, candidate)
    candidate_row = ranking.loc[ranking.configuration.eq(candidate)].iloc[0]
    candidate_label = str(candidate_row.get("label", candidate))
    labels = {
        "raw_fuxi": "Raw FuXi",
        "log_bias": "Log-bias",
        CONTROL_CONFIGURATION: "Compact control",
        candidate: candidate_label,
    }

    case_frames = []
    for method in methods:
        frame = sweep.compute_case_metrics(
            truth,
            predictions[method],
            truth - climatology,
            predictions[method] - climatology,
            prepared.weights,
            predictor=method,
            case_ids=dates,
            leads=np.arange(1, 7),
            valid_mask=valid,
        )
        frame.insert(0, "method", method)
        case_frames.append(frame)
    case_metrics = pd.concat(case_frames, ignore_index=True)
    case_metrics.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    lead_summary = (
        case_metrics.groupby(["method", "lead"], as_index=False)[
            ["rmse", "mae", "bias", "acc"]
        ]
        .mean()
        .sort_values(["method", "lead"])
    )
    lead_summary.to_csv(output / "metrics" / "metrics_by_lead.csv", index=False)

    guard_table, guards = build_all_lead_guard_table(
        case_metrics, candidate, CONTROL_CONFIGURATION
    )
    guard_table.insert(0, "candidate", candidate)
    guard_table.insert(1, "control", CONTROL_CONFIGURATION)
    guard_table.to_csv(
        output / "metrics" / "all_six_lead_guard_table.csv", index=False
    )
    (output / "metrics" / "all_six_lead_guard_summary.json").write_text(
        json.dumps({"candidate": candidate, "control": CONTROL_CONFIGURATION, **guards}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    bootstrap = stratified_paired_bootstrap(
        case_metrics,
        candidate,
        CONTROL_CONFIGURATION,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        block_length=int(args.bootstrap_block_length),
    )
    bootstrap.insert(0, "candidate", candidate)
    bootstrap.insert(1, "control", CONTROL_CONFIGURATION)
    bootstrap.to_csv(
        output / "metrics" / "year_stratified_paired_moving_block_bootstrap.csv",
        index=False,
    )

    bias_fields = {
        method: spatial_mean_bias(predictions[method], truth, prepared.weights)
        for method in methods
    }
    skill, candidate_rmse, control_rmse = spatial_rmse_skill_vs_control(
        predictions[candidate],
        predictions[CONTROL_CONFIGURATION],
        truth,
        prepared.weights,
    )
    spatial_summary = _build_spatial_summary(
        skill, candidate_rmse, control_rmse, prepared.weights
    )
    spatial_summary.insert(0, "candidate", candidate)
    spatial_summary.to_csv(
        output / "metrics" / "spatial_rmse_skill_summary.csv", index=False
    )
    np.savez_compressed(
        output / "metrics" / "spatial_validation_fields.npz",
        methods=np.asarray(methods, dtype="U64"),
        latitude=latitude.astype(np.float64),
        longitude=longitude.astype(np.float64),
        spatial_mean_bias=np.stack([bias_fields[method] for method in methods]).astype(
            np.float32
        ),
        candidate_rmse_skill_vs_control_pct=skill.astype(np.float32),
        candidate_local_rmse=candidate_rmse.astype(np.float32),
        control_local_rmse=control_rmse.astype(np.float32),
    )

    plot_spatial_bias_maps(
        bias_fields,
        labels,
        latitude,
        longitude,
        output / "figures" / "01_spatial_mean_bias_by_lead",
        boundary_segments,
    )
    plot_spatial_rmse_skill(
        skill,
        spatial_summary,
        candidate_label,
        latitude,
        longitude,
        output / "figures" / "02_spatial_rmse_skill_vs_control",
        boundary_segments,
    )
    plot_lead_metrics(
        lead_summary,
        methods,
        labels,
        output / "figures" / "03_india_weighted_metrics_by_lead",
    )
    case_rmse = {
        method: area_weighted_case_rmse(
            predictions[method], truth, prepared.weights
        )
        for method in methods
    }
    plot_time_lead_comparison(
        case_rmse,
        dates,
        methods,
        labels,
        candidate,
        output / "figures" / "04_validation_time_lead_comparison",
    )

    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    postprocess_manifest = {
        "status": "complete",
        "created_utc": utc_now(),
        "source_sweep": str(sweep_directory),
        "source_manifest_sha256": sha256_file(sweep_directory / "manifest.json"),
        "source_training_mode": manifest.get("training_mode"),
        "validation_years": list(VALIDATION_YEARS),
        "validation_initialization_count": int(len(dates)),
        "evaluation_scope": "blocked validation only; no 2020+ prediction or metric",
        "control_configuration": CONTROL_CONFIGURATION,
        "selected_best_physical_candidate": candidate,
        "selection_source": "metrics/ranked_configurations.csv",
        "spatial_inference_status": "descriptive; no pixel-wise significance inference",
        "india_boundary": boundary_provenance,
        "bootstrap": {
            "method": "circular moving-block bootstrap",
            "unit": "paired initialization case with all leads retained",
            "strata": list(VALIDATION_YEARS),
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "block_length_initializations": int(args.bootstrap_block_length),
        },
        "all_six_lead_guards": guards,
        "artifacts": artifacts,
    }
    (output / "postprocessing_manifest.json").write_text(
        json.dumps(postprocess_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: physical validation publication plots: {output}", flush=True)
    print(f"Best physical candidate: {candidate}", flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--candidate",
        default=None,
        help="Explicit completed physical candidate; default is best ranked.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument(
        "--bootstrap-block-length",
        type=int,
        default=13,
        help="Circular moving-block length in consecutive initialization cases.",
    )
    parser.add_argument(
        "--india-boundary",
        type=Path,
        default=DEFAULT_INDIA_BOUNDARY,
        help=(
            "WGS84 Survey of India ABDB-derived state/UT GeoJSON used for "
            "all spatial panels."
        ),
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow diagnostic plots from a smoke run (not publication evidence).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
