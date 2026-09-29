#!/usr/bin/env python3
"""Render the six-week IMD/FuXi/corrected spatial story in two styles.

The source is the immutable ``spatial_test_fields.npz`` produced by the
locked 2020--2021 exploratory evaluator.  No forecast or observation store is
opened and no metric is recomputed.  The native figure shows the actual 1.5°
cells.  Its smooth companion applies interpolation for display only and says
so prominently on the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata


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
EXPECTED_YEARS = (2020, 2021)
EXPECTED_CONFIGURATION = "physical_full_compact"
RAIN_COLOR = "YlGnBu"
DIFFERENCE_COLOR = "RdBu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(directory: Path) -> Mapping[str, Any]:
    path = directory / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("locked evaluation manifest must be a JSON object")
    required = {
        "status": "complete",
        "evaluation_role": EXPECTED_ROLE,
        "selected_configuration": EXPECTED_CONFIGURATION,
        "reused_test_period": True,
        "genuine_independent_test": False,
        "selection_locked_before_target_access": True,
        "test_used_for_selection": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"locked evaluation manifest has unsafe {key!r}")
    if tuple(payload.get("test_years", ())) != EXPECTED_YEARS:
        raise ValueError("spatial atlas requires exactly the 2020--2021 reused set")
    return payload


def load_spatial_fields(directory: Path) -> dict[str, np.ndarray]:
    """Load and validate only the evaluator's already-derived spatial fields."""

    directory = Path(directory).expanduser().resolve()
    manifest = _read_manifest(directory)
    relative = "metrics/spatial_test_fields.npz"
    path = directory / relative
    expected = manifest.get("artifacts", {}).get(relative)
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError("spatial field artifact checksum differs from the manifest")
    required = (
        "latitude",
        "longitude",
        "weights",
        "observed_mean",
        "raw_mean",
        "corrected_mean",
    )
    with np.load(path, allow_pickle=False) as source:
        missing = set(required).difference(source.files)
        if missing:
            raise ValueError(f"spatial fields lack {sorted(missing)}")
        values = {name: np.asarray(source[name]).copy() for name in required}
    latitude = values["latitude"]
    longitude = values["longitude"]
    weights = values["weights"]
    if latitude.shape != (27,) or longitude.shape != (27,):
        raise ValueError("spatial atlas requires the native 27x27 FuXi grid")
    if not np.all(np.diff(latitude) < 0.0) or not np.all(np.diff(longitude) > 0.0):
        raise ValueError("latitude/longitude ordering differs from the trained grid")
    if weights.shape != (27, 27) or np.count_nonzero(weights > 0.0) != 171:
        raise ValueError("IMD support must contain exactly 171 native cells")
    for name in ("observed_mean", "raw_mean", "corrected_mean"):
        if values[name].shape != (6, 27, 27):
            raise ValueError(f"{name} must have six 27x27 weekly maps")
        if not np.isfinite(values[name][:, weights > 0.0]).all():
            raise ValueError(f"{name} contains non-finite supported values")
    return values


def story_fields(values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return the five requested rows with the requested IMD-minus sign."""

    observed = np.asarray(values["observed_mean"], dtype=np.float64)
    raw = np.asarray(values["raw_mean"], dtype=np.float64)
    corrected = np.asarray(values["corrected_mean"], dtype=np.float64)
    return observed, raw, corrected, observed - raw, observed - corrected


def smooth_display_field(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    *,
    points_per_axis: int = 280,
    nonnegative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic display interpolation, masked to nearest native support cells."""

    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if field.shape != support.shape or field.shape != (len(latitude), len(longitude)):
        raise ValueError("field/support coordinates are inconsistent")
    if points_per_axis < 50:
        raise ValueError("display interpolation grid is too coarse")
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    points = np.column_stack((lon2d[support], lat2d[support]))
    samples = field[support]
    dense_longitude = np.linspace(longitude.min(), longitude.max(), points_per_axis)
    dense_latitude = np.linspace(latitude.min(), latitude.max(), points_per_axis)
    dense_lon2d, dense_lat2d = np.meshgrid(dense_longitude, dense_latitude)
    dense = griddata(
        points,
        samples,
        (dense_lon2d, dense_lat2d),
        method="cubic",
    )
    linear = griddata(
        points,
        samples,
        (dense_lon2d, dense_lat2d),
        method="linear",
    )
    dense = np.where(np.isfinite(dense), dense, linear)
    nearest = griddata(
        points,
        samples,
        (dense_lon2d, dense_lat2d),
        method="nearest",
    )
    dense = np.where(np.isfinite(dense), dense, nearest)

    # The interpolation is visual only.  Retain the native IMD support
    # footprint through nearest-cell lookup, rather than inventing offshore
    # or unsupported values from a polygon fill.
    latitude_ascending = latitude[::-1]
    support_ascending = support[::-1].astype(np.float64)
    support_lookup = RegularGridInterpolator(
        (latitude_ascending, longitude),
        support_ascending,
        method="nearest",
        bounds_error=False,
        fill_value=0.0,
    )
    dense_points = np.column_stack((dense_lat2d.ravel(), dense_lon2d.ravel()))
    dense_support = support_lookup(dense_points).reshape(dense.shape) > 0.5
    if nonnegative:
        dense = np.maximum(dense, 0.0)
    dense[~dense_support] = np.nan
    return dense_latitude, dense_longitude, dense


def _nice_upper(value: float, step: float = 2.0) -> float:
    return float(np.ceil(max(value, step) / step) * step)


def _plot_map(
    axis: plt.Axes,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    smooth: bool,
    support: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    norm: Any = None,
) -> Any:
    if smooth:
        dense_lat, dense_lon, shown = smooth_display_field(
            field,
            latitude,
            longitude,
            support,
            nonnegative=vmin == 0.0,
        )
        return axis.pcolormesh(
            dense_lon,
            dense_lat,
            np.ma.masked_invalid(shown),
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            shading="auto",
            rasterized=True,
        )
    shown = np.asarray(field, dtype=np.float64).copy()
    shown[~support] = np.nan
    return axis.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(shown),
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        shading="nearest",
        rasterized=True,
    )


def _decorate(
    axis: plt.Axes,
    boundary_segments: Sequence[np.ndarray],
    *,
    left: bool,
    bottom: bool,
) -> None:
    axis.set_xlim(67.0, 98.7)
    axis.set_ylim(6.0, 38.7)
    axis.set_aspect("equal", adjustable="box")
    axis.add_collection(
        LineCollection(
            boundary_segments,
            colors="#172B3A",
            linewidths=0.28,
            alpha=0.92,
            zorder=6,
        ),
        autolim=False,
    )
    axis.set_xticks((70, 80, 90))
    axis.set_yticks((10, 20, 30))
    axis.tick_params(
        labelsize=7,
        length=2.5,
        labelleft=left,
        labelbottom=bottom,
        colors="#4B6070",
    )
    if left:
        axis.set_ylabel("Latitude (°N)", fontsize=8)
    if bottom:
        axis.set_xlabel("Longitude (°E)", fontsize=8)
    axis.grid(color="#8596A3", alpha=0.14, linewidth=0.35)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#728491")


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small provenance manifest without exposing a partial file."""

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


def render_atlas(
    values: Mapping[str, np.ndarray],
    boundary_segments: Sequence[np.ndarray],
    output_stem: Path,
    *,
    smooth: bool,
) -> tuple[Path, Path]:
    """Render one 5-row × 6-week atlas."""

    latitude = np.asarray(values["latitude"], dtype=np.float64)
    longitude = np.asarray(values["longitude"], dtype=np.float64)
    support = np.asarray(values["weights"]) > 0.0
    fields = story_fields(values)
    rain_values = np.concatenate([field[:, support].ravel() for field in fields[:3]])
    difference_values = np.concatenate(
        [np.abs(field[:, support]).ravel() for field in fields[3:]]
    )
    rain_max = _nice_upper(float(np.quantile(rain_values, 0.99)), step=2.0)
    difference_max = _nice_upper(
        float(np.quantile(difference_values, 0.99)), step=1.0
    )
    difference_norm = TwoSlopeNorm(
        vmin=-difference_max,
        vcenter=0.0,
        vmax=difference_max,
    )
    row_labels = (
        "IMD observed",
        "Raw FuXi-S2S",
        "Frozen corrected",
        "IMD − raw FuXi",
        "IMD − corrected",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # The requested "6x3" story is six lead-week columns over three primary
    # rainfall rows, followed by the two diagnostic difference rows.
    figure, axes = plt.subplots(5, 6, figsize=(18.4, 15.1), facecolor="white")
    rain_image = None
    difference_image = None
    for row, (row_name, weekly) in enumerate(zip(row_labels, fields)):
        for lead in range(6):
            axis = axes[row, lead]
            is_difference = row >= 3
            image = _plot_map(
                axis,
                weekly[lead],
                latitude,
                longitude,
                smooth=smooth,
                support=support,
                cmap=DIFFERENCE_COLOR if is_difference else RAIN_COLOR,
                vmin=-difference_max if is_difference else 0.0,
                vmax=difference_max if is_difference else rain_max,
                norm=difference_norm if is_difference else None,
            )
            if is_difference:
                difference_image = image
            else:
                rain_image = image
            _decorate(
                axis,
                boundary_segments,
                left=lead == 0,
                bottom=row == 4,
            )
            if row == 0:
                axis.set_title(f"Lead week {lead + 1}", fontsize=10.5, pad=5)
            if lead == 0:
                axis.text(
                    -0.34,
                    0.50,
                    row_name,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.2,
                    fontweight="semibold",
                    color="#172B3A",
                )
    if rain_image is None or difference_image is None:
        raise RuntimeError("atlas did not create its shared color scales")
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.895,
        bottom=0.115,
        wspace=0.055,
        hspace=0.10,
    )
    rain_bar = figure.colorbar(
        rain_image,
        cax=figure.add_axes((0.14, 0.064, 0.32, 0.014)),
        orientation="horizontal",
    )
    rain_bar.set_label("Weekly-mean rainfall (mm day$^{-1}$)", fontsize=9)
    difference_bar = figure.colorbar(
        difference_image,
        cax=figure.add_axes((0.58, 0.064, 0.32, 0.014)),
        orientation="horizontal",
        extend="both",
    )
    difference_bar.set_label(
        "IMD − forecast (mm day$^{-1}$) · positive = forecast too dry",
        fontsize=9,
    )
    mode = (
        "Display-only smooth interpolation"
        if smooth
        else "Native 1.5° grid · no interpolation"
    )
    figure.suptitle(
        "Six-week rainfall composites: IMD, raw FuXi-S2S, and frozen correction\n"
        f"Mean across 70 JJAS starts in 2020–2021 · {mode} · exploratory/reused",
        fontsize=17,
        fontweight="bold",
        color="#172B3A",
        y=0.972,
    )
    footer = (
        "Visual interpolation only; values between native cell centres are not new observations or forecasts. "
        "All evaluation metrics use the native 1.5° grid."
        if smooth
        else "Every colored tile is one native 1.5° evaluation cell; no spatial interpolation or smoothing."
    )
    figure.text(
        0.5,
        0.018,
        footer
        + "  Survey of India ABDB-derived state/UT boundary. "
        "2020–2021 is exploratory/reused, not independent confirmation.",
        ha="center",
        fontsize=8.2,
        color="#526777",
    )
    output_stem = Path(output_stem).expanduser().resolve()
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    _atomic_save_figure(
        figure,
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    _atomic_save_figure(
        figure,
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return png, pdf


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--india-boundary",
        type=Path,
        default=DEFAULT_INDIA_BOUNDARY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    values = load_spatial_fields(args.evaluation_directory)
    segments, provenance = load_india_boundary(args.india_boundary)
    if provenance.get("feature_count", 0) < 30:
        raise ValueError("official India boundary coverage is incomplete")
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(
        render_atlas(
            values,
            segments,
            output / "01_six_week_spatial_atlas_native_grid",
            smooth=False,
        )
    )
    outputs.extend(
        render_atlas(
            values,
            segments,
            output / "02_six_week_spatial_atlas_visual_interpolation",
            smooth=True,
        )
    )
    evaluation = Path(args.evaluation_directory).expanduser().resolve()
    boundary = Path(args.india_boundary).expanduser().resolve()
    relative_spatial = Path("metrics/spatial_test_fields.npz")
    _atomic_write_json(
        output / "spatial_atlas_manifest.json",
        {
            "schema_name": "fuxi_imd_clear_story_spatial_atlas",
            "schema_version": 1,
            "status": "complete",
            "evaluation_scope": (
                "2020-2021 exploratory/reused locked hindcasts; "
                "not independent confirmation"
            ),
            "selected_configuration": EXPECTED_CONFIGURATION,
            "cases": 70,
            "lead_weeks": list(range(1, 7)),
            "layout": {
                "columns": "lead weeks 1-6",
                "primary_rows": [
                    "IMD observed",
                    "Raw FuXi-S2S",
                    "Frozen corrected",
                ],
                "diagnostic_rows": [
                    "IMD minus raw FuXi-S2S",
                    "IMD minus frozen corrected",
                ],
            },
            "aggregation": "mean across the same 70 JJAS initializations",
            "native_grid_degrees": 1.5,
            "metrics_recomputed": False,
            "visual_interpolation_used_for_metrics": False,
            "source_evaluation_manifest": {
                "path": str(evaluation / "manifest.json"),
                "sha256": _sha256(evaluation / "manifest.json"),
            },
            "source_spatial_fields": {
                "path": str(evaluation / relative_spatial),
                "sha256": _sha256(evaluation / relative_spatial),
            },
            "india_boundary": {
                "path": str(boundary),
                "sha256": _sha256(boundary),
                "provenance": provenance,
            },
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(outputs, key=lambda item: item.name)
            },
        },
    )
    for path in outputs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
