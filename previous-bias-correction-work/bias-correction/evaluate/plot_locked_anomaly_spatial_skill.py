#!/usr/bin/env python3
"""Plot leakage-safe IMD-referenced anomaly composites over India.

This presentation diagnostic reads only the immutable prediction bundle from
the locked 2020--2021 exploratory evaluation.  Every anomaly uses the same
case- and lead-matched, training-only IMD calendar climatology stored by that
evaluator.  The maps remain on the native 1.5-degree evaluation grid.

Two complementary figures are written:

* a six-row atlas with lead weeks down and IMD/raw/corrected columns across;
* a compact all-case/all-lead JJAS summary for a presentation slide.

The small ``r_map`` labels are correlations of the displayed composite maps.
They are descriptive and are not substitutes for mean per-case ACC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
SOURCE_ROOT = WORK_ROOT / "src"
for source in (HERE, SOURCE_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


EXPECTED_ROLE = "exploratory_reused_hindcast_evaluation"
EXPECTED_CONFIGURATION = "physical_full_compact"
EXPECTED_TEST_YEARS = (2020, 2021)
EXPECTED_TRAIN_YEARS = tuple(range(2002, 2018))
PREDICTION_RELATIVE_PATH = Path("predictions/exploratory_test_predictions.npz")
FIELD_KEYS = ("truth_imd", "raw_fuxi", "corrected")
COLUMN_LABELS = (
    "IMD\nanomaly",
    "Raw FuXi-S2S\nanomaly",
    "Corrected Forecast\nanomaly",
)
ANOMALY_CMAP = "RdBu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_manifest(directory: Path) -> Mapping[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requirements = {
        "status": "complete",
        "evaluation_role": EXPECTED_ROLE,
        "selected_configuration": EXPECTED_CONFIGURATION,
        "reused_test_period": True,
        "genuine_independent_test": False,
        "selection_locked_before_target_access": True,
        "test_used_for_selection": False,
        "normalization_fit_on_test": False,
    }
    for key, expected in requirements.items():
        if manifest.get(key) != expected:
            raise ValueError(f"locked evaluation has unsafe {key!r}")
    if tuple(manifest.get("test_years", ())) != EXPECTED_TEST_YEARS:
        raise ValueError("anomaly maps require exactly reused 2020--2021 hindcasts")
    if int(manifest.get("test_initialization_count", -1)) != 70:
        raise ValueError("locked evaluation must contain exactly 70 starts")
    return manifest


def load_locked_predictions(directory: Path) -> dict[str, np.ndarray]:
    """Load and contract-check the immutable evaluation prediction bundle."""

    directory = Path(directory).expanduser().resolve()
    manifest = _validated_manifest(directory)
    path = directory / PREDICTION_RELATIVE_PATH
    expected_hash = manifest.get("artifacts", {}).get(str(PREDICTION_RELATIVE_PATH))
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError("prediction bundle checksum differs from locked manifest")
    required = {
        "initializations",
        "lead_week",
        "latitude",
        "longitude",
        "weights",
        "training_climatology",
        *FIELD_KEYS,
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"locked prediction bundle lacks {sorted(missing)}")
        values = {name: np.asarray(source[name]).copy() for name in required}

    initializations = np.asarray(values["initializations"], dtype="datetime64[D]")
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    if initializations.shape != (70,) or np.any(initializations[1:] <= initializations[:-1]):
        raise ValueError("initialization contract differs")
    counts = pd.Series(years).value_counts().sort_index().to_dict()
    if counts != {2020: 35, 2021: 35}:
        raise ValueError("expected 35 starts in each reused evaluation year")
    months = pd.DatetimeIndex(initializations).month.to_numpy()
    if not np.isin(months, (6, 7, 8, 9)).all():
        raise ValueError("initializations are not confined to JJAS")
    if not np.array_equal(values["lead_week"], np.arange(1, 7)):
        raise ValueError("lead weeks differ from 1--6")

    latitude = np.asarray(values["latitude"], dtype=np.float64)
    longitude = np.asarray(values["longitude"], dtype=np.float64)
    weights = np.asarray(values["weights"], dtype=np.float64)
    if latitude.shape != (27,) or longitude.shape != (27,):
        raise ValueError("native FuXi grid must be 27x27")
    if not np.all(np.diff(latitude) < 0.0) or not np.all(np.diff(longitude) > 0.0):
        raise ValueError("coordinate ordering differs from the evaluated grid")
    if weights.shape != (27, 27) or np.count_nonzero(weights > 0.0) != 171:
        raise ValueError("IMD-supported native grid must contain 171 cells")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("area/support weights are invalid")
    support = weights > 0.0
    for key in ("training_climatology", *FIELD_KEYS):
        array = np.asarray(values[key])
        if array.shape != (70, 6, 27, 27):
            raise ValueError(f"{key} has an unexpected shape")
        if not np.isfinite(array[:, :, support]).all():
            raise ValueError(f"{key} contains non-finite supported values")
    if np.any(values["training_climatology"][:, :, support] < 0.0):
        raise ValueError("training climatology contains negative rainfall")
    return values


def anomaly_composites(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average case anomalies after subtracting case-matched climatology."""

    climatology = np.asarray(values["training_climatology"], dtype=np.float64)
    support = np.asarray(values["weights"], dtype=np.float64) > 0.0
    composites: dict[str, np.ndarray] = {}
    for key in FIELD_KEYS:
        field = np.asarray(values[key], dtype=np.float64)
        if field.shape != climatology.shape:
            raise ValueError(f"{key} and climatology shapes differ")
        composite = np.mean(field - climatology, axis=0, dtype=np.float64)
        composite[:, ~support] = np.nan
        composites[key] = composite
    return composites


def area_weighted_pattern_correlation(
    reference: np.ndarray,
    forecast: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted, spatially centered correlation of two displayed maps."""

    reference = np.asarray(reference, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if reference.shape != forecast.shape or reference.shape != weights.shape:
        raise ValueError("pattern-correlation fields and weights must align")
    valid = (
        np.isfinite(reference)
        & np.isfinite(forecast)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        return float("nan")
    weight = weights[valid]
    weight = weight / np.sum(weight)
    x = reference[valid]
    y = forecast[valid]
    x = x - np.sum(weight * x)
    y = y - np.sum(weight * y)
    denominator = np.sqrt(np.sum(weight * x * x) * np.sum(weight * y * y))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.sum(weight * x * y) / denominator)


def displayed_skill(
    composites: Mapping[str, np.ndarray], weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Return per-lead r_map values for raw and corrected composites."""

    reference = np.asarray(composites["truth_imd"], dtype=np.float64)
    if reference.ndim != 3:
        raise ValueError("displayed composites must have lead, latitude, longitude axes")
    result: dict[str, np.ndarray] = {}
    for key in ("raw_fuxi", "corrected"):
        forecast = np.asarray(composites[key], dtype=np.float64)
        if forecast.shape != reference.shape:
            raise ValueError(f"{key} displayed composite shape differs from IMD")
        result[key] = np.asarray(
            [
                area_weighted_pattern_correlation(
                    reference[lead], forecast[lead], weights
                )
                for lead in range(reference.shape[0])
            ],
            dtype=np.float64,
        )
    return result


def _color_limit(composites: Mapping[str, np.ndarray], support: np.ndarray) -> float:
    absolute = np.concatenate(
        [np.abs(np.asarray(composites[key])[:, support]).ravel() for key in FIELD_KEYS]
    )
    return float(max(1.0, np.ceil(np.quantile(absolute, 0.99))))


def _decorate_map(axis: plt.Axes, boundary: Sequence[np.ndarray]) -> None:
    axis.set_xlim(67.0, 98.7)
    axis.set_ylim(6.0, 38.7)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#F4F7F9")
    axis.add_collection(
        LineCollection(
            boundary,
            colors="#142936",
            linewidths=0.30,
            alpha=0.95,
            zorder=5,
        ),
        autolim=False,
    )
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_color("#B7C4CC")
        spine.set_linewidth(0.52)


def _plot_native(
    axis: plt.Axes,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    boundary: Sequence[np.ndarray],
    norm: TwoSlopeNorm,
) -> Any:
    shown = np.asarray(field, dtype=np.float64).copy()
    shown[~support] = np.nan
    image = axis.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(shown),
        cmap=ANOMALY_CMAP,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )
    _decorate_map(axis, boundary)
    return image


def _skill_badge(axis: plt.Axes, value: float) -> None:
    axis.text(
        0.965,
        0.965,
        f"$r_{{map}}$ = {value:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        fontweight="semibold",
        color="#142936",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "#CBD5DB",
            "linewidth": 0.5,
            "alpha": 0.92,
        },
        zorder=8,
    )


def _atomic_save_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=path.suffix[1:], **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_png_pdf(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    png = Path(stem).with_suffix(".png")
    pdf = Path(stem).with_suffix(".pdf")
    _atomic_save_figure(figure, png, dpi=300, bbox_inches="tight", facecolor="white")
    _atomic_save_figure(figure, pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def _base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def render_weekwise_atlas(
    values: Mapping[str, np.ndarray],
    composites: Mapping[str, np.ndarray],
    boundary: Sequence[np.ndarray],
    output_stem: Path,
) -> tuple[Path, Path, dict[str, np.ndarray], float]:
    """Render weeks as rows and IMD/raw/corrected anomalies as columns."""

    _base_style()
    latitude = np.asarray(values["latitude"], dtype=np.float64)
    longitude = np.asarray(values["longitude"], dtype=np.float64)
    weights = np.asarray(values["weights"], dtype=np.float64)
    support = weights > 0.0
    skill = displayed_skill(composites, weights)
    limit = _color_limit(composites, support)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(6, 3, figsize=(10.8, 18.2), facecolor="white")
    image = None
    for lead in range(6):
        for column, key in enumerate(FIELD_KEYS):
            axis = axes[lead, column]
            image = _plot_native(
                axis,
                composites[key][lead],
                latitude,
                longitude,
                support,
                boundary,
                norm,
            )
            if lead == 0:
                axis.set_title(COLUMN_LABELS[column], fontsize=13.0, pad=8)
            if column == 0:
                axis.text(
                    -0.26,
                    0.50,
                    f"Week {lead + 1}",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11.0,
                    fontweight="bold",
                    color="#173447",
                )
            if key in skill:
                _skill_badge(axis, float(skill[key][lead]))
    if image is None:
        raise RuntimeError("no anomaly map was rendered")
    figure.subplots_adjust(
        left=0.105,
        right=0.982,
        top=0.918,
        bottom=0.087,
        wspace=0.035,
        hspace=0.055,
    )
    colorbar = figure.colorbar(
        image,
        cax=figure.add_axes((0.24, 0.051, 0.54, 0.014)),
        orientation="horizontal",
        extend="both",
    )
    colorbar.set_label(
        "Weekly rainfall anomaly from the 2002–2017 IMD climatology (mm day$^{-1}$)",
        fontsize=9.4,
    )
    colorbar.ax.tick_params(labelsize=8.2, length=2.5)
    figure.suptitle(
        "Spatial anomaly patterns across six lead weeks\n"
        "70 JJAS starts · 2020–2021 exploratory/reused hindcasts · native 1.5° grid",
        fontsize=16.5,
        fontweight="bold",
        color="#142936",
        y=0.972,
    )
    figure.text(
        0.5,
        0.017,
        "Composite = mean of case-wise [weekly field − matched training-only IMD climatology].  "
        "$r_{map}$ compares the displayed forecast and IMD composite maps; verification ACC instead "
        "averages 70 case-wise spatial correlations at each lead.\n"
        "Survey of India ABDB-derived boundary.  Reused evaluation; not independent confirmation.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#526777",
    )
    png, pdf = _save_png_pdf(figure, output_stem)
    return png, pdf, skill, limit


def render_jjas_summary(
    values: Mapping[str, np.ndarray],
    composites: Mapping[str, np.ndarray],
    boundary: Sequence[np.ndarray],
    output_stem: Path,
    *,
    color_limit: float,
) -> tuple[Path, Path, dict[str, float]]:
    """Render a compact mean over all 70 starts and all six lead weeks."""

    _base_style()
    latitude = np.asarray(values["latitude"], dtype=np.float64)
    longitude = np.asarray(values["longitude"], dtype=np.float64)
    weights = np.asarray(values["weights"], dtype=np.float64)
    support = weights > 0.0
    summary = {key: np.mean(composites[key], axis=0) for key in FIELD_KEYS}
    skill = {
        key: area_weighted_pattern_correlation(
            summary["truth_imd"], summary[key], weights
        )
        for key in ("raw_fuxi", "corrected")
    }
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.9), facecolor="white")
    image = None
    for column, key in enumerate(FIELD_KEYS):
        axis = axes[column]
        image = _plot_native(
            axis,
            summary[key],
            latitude,
            longitude,
            support,
            boundary,
            norm,
        )
        axis.set_title(COLUMN_LABELS[column], fontsize=12.2, pad=7)
        if key in skill:
            _skill_badge(axis, skill[key])
    if image is None:
        raise RuntimeError("no JJAS anomaly summary was rendered")
    figure.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.73,
        bottom=0.24,
        wspace=0.04,
    )
    colorbar = figure.colorbar(
        image,
        cax=figure.add_axes((0.29, 0.145, 0.42, 0.025)),
        orientation="horizontal",
        extend="both",
    )
    colorbar.set_label(
        "Rainfall anomaly (mm day$^{-1}$)", fontsize=9.2
    )
    colorbar.ax.tick_params(labelsize=8.0, length=2.5)
    figure.suptitle(
        "JJAS-initialized spatial anomaly composite against IMD\n"
        "Mean over 70 starts × 6 lead weeks · common 2002–2017 IMD climatology",
        fontsize=15.4,
        fontweight="bold",
        color="#142936",
        y=0.975,
    )
    figure.text(
        0.5,
        0.020,
        "$r_{map}$ describes these displayed composites; verification ACC instead averages the "
        "70 case-wise spatial correlations at each lead.  "
        "Native 1.5° grid · 2020–2021 reused exploratory hindcasts; not independent confirmation.",
        ha="center",
        fontsize=7.8,
        color="#526777",
    )
    png, pdf = _save_png_pdf(figure, output_stem)
    return png, pdf, skill


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--india-boundary", type=Path, default=DEFAULT_INDIA_BOUNDARY
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluation = Path(args.evaluation_directory).expanduser().resolve()
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = load_locked_predictions(evaluation)
    composites = anomaly_composites(values)
    boundary, boundary_provenance = load_india_boundary(args.india_boundary)
    if int(boundary_provenance.get("feature_count", 0)) < 30:
        raise ValueError("official India boundary coverage is incomplete")

    week_png, week_pdf, week_skill, color_limit = render_weekwise_atlas(
        values,
        composites,
        boundary,
        output / "07_weekwise_imd_referenced_anomaly_spatial_skill",
    )
    summary_png, summary_pdf, summary_skill = render_jjas_summary(
        values,
        composites,
        boundary,
        output / "08_jjas_mean_imd_referenced_anomaly_spatial_skill",
        color_limit=color_limit,
    )
    outputs = (week_png, week_pdf, summary_png, summary_pdf)
    prediction_path = evaluation / PREDICTION_RELATIVE_PATH
    boundary_path = Path(args.india_boundary).expanduser().resolve()
    _atomic_write_json(
        output / "anomaly_spatial_skill_manifest.json",
        {
            "schema_name": "fuxi_imd_locked_anomaly_spatial_skill",
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_role": EXPECTED_ROLE,
            "evaluation_scope": (
                "2020-2021 exploratory/reused locked hindcasts; "
                "not independent confirmation"
            ),
            "selected_configuration": EXPECTED_CONFIGURATION,
            "display_name": "Corrected Forecast",
            "train_years_for_climatology": list(EXPECTED_TRAIN_YEARS),
            "climatology": (
                "common fixed 2002-2017 training-only, equal-year centered "
                "31-day IMD calendar climatology; matched to every valid day "
                "and averaged within lead week"
            ),
            "anomaly_definition": "weekly field minus case- and lead-matched training climatology",
            "weekwise_composite": "mean anomaly across the same 70 JJAS starts",
            "jjas_summary_composite": "mean anomaly across 70 starts and six lead weeks",
            "cases": 70,
            "case_counts_by_year": {"2020": 35, "2021": 35},
            "lead_weeks": list(range(1, 7)),
            "native_grid_degrees": 1.5,
            "support_cells": 171,
            "spatial_interpolation": False,
            "display_color_limit_mm_day": color_limit,
            "displayed_map_correlation": {
                "definition": (
                    "area-weighted spatially centered correlation between "
                    "displayed forecast and IMD composite anomalies"
                ),
                "warning": "descriptive r_map; not mean per-case ACC",
                "weekwise": {
                    key: [float(value) for value in week_skill[key]]
                    for key in ("raw_fuxi", "corrected")
                },
                "all_case_all_lead_summary": {
                    key: float(summary_skill[key])
                    for key in ("raw_fuxi", "corrected")
                },
            },
            "inference": "none; no p-values, multiplicity tests, or significance claims",
            "source_evaluation_manifest": {
                "path": str(evaluation / "manifest.json"),
                "sha256": sha256_file(evaluation / "manifest.json"),
            },
            "source_prediction_bundle": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "india_boundary": {
                "path": str(boundary_path),
                "sha256": sha256_file(boundary_path),
                "provenance": boundary_provenance,
            },
            "artifacts": {path.name: sha256_file(path) for path in outputs},
        },
    )
    for path in outputs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
