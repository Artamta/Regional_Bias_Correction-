#!/usr/bin/env python3
"""Render the locked JJAS spatial composites with lead weeks as rows.

This is a presentation-only re-layout of the checksummed spatial fields from
the locked 2020--2021 exploratory evaluator.  It never opens the forecast or
IMD source stores and never recomputes a metric.  The second output uses
interpolation only to make the same native fields easier to view; the support
mask and all underlying evaluation remain on the native 1.5-degree grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
SOURCE_ROOT = WORK_ROOT / "src"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from plot_clear_story_spatial_atlas import (  # noqa: E402
    DIFFERENCE_COLOR,
    EXPECTED_CONFIGURATION,
    RAIN_COLOR,
    _atomic_save_figure,
    _atomic_write_json,
    _decorate,
    _nice_upper,
    _plot_map,
    _sha256,
    load_spatial_fields,
    story_fields,
)
from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


COLUMN_TITLES = (
    "IMD Observation",
    "Raw FuXi-S2S",
    "Corrected Forecast",
    "IMD − Raw FuXi",
    "IMD − Corrected Forecast",
)
LEAD_LABELS = tuple(f"WEEK {lead}" for lead in range(1, 7))
CASE_COUNT = 70


def layout_contract() -> dict[str, Any]:
    """Return the explicit semantic layout recorded in the sidecar manifest."""

    return {
        "rows": list(LEAD_LABELS),
        "columns": list(COLUMN_TITLES),
        "difference_sign": "IMD minus forecast; positive means forecast too dry",
    }


def render_transposed_atlas(
    values: Mapping[str, np.ndarray],
    boundary_segments: Sequence[np.ndarray],
    output_stem: Path,
    *,
    smooth: bool,
) -> tuple[Path, Path]:
    """Render six lead-week rows by five field columns."""

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

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(6, 5, figsize=(16.2, 18.7), facecolor="white")
    rain_image = None
    difference_image = None
    for lead in range(6):
        for column, weekly in enumerate(fields):
            axis = axes[lead, column]
            is_difference = column >= 3
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
                left=column == 0,
                bottom=lead == 5,
            )
            if lead == 0:
                axis.set_title(COLUMN_TITLES[column], fontsize=11.1, pad=7)

    if rain_image is None or difference_image is None:
        raise RuntimeError("transposed atlas did not create its shared color scales")

    figure.subplots_adjust(
        left=0.088,
        right=0.988,
        top=0.887,
        bottom=0.105,
        wspace=0.060,
        hspace=0.085,
    )
    for lead, label in enumerate(LEAD_LABELS):
        box = axes[lead, 0].get_position()
        figure.text(
            0.027,
            box.y0 + box.height / 2.0,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.3,
            fontweight="bold",
            color="#172B3A",
        )

    rain_bar = figure.colorbar(
        rain_image,
        cax=figure.add_axes((0.145, 0.055, 0.31, 0.013)),
        orientation="horizontal",
    )
    rain_bar.set_label("JJAS weekly-mean rainfall (mm day$^{-1}$)", fontsize=9)
    difference_bar = figure.colorbar(
        difference_image,
        cax=figure.add_axes((0.565, 0.055, 0.31, 0.013)),
        orientation="horizontal",
        extend="both",
    )
    difference_bar.set_label(
        "IMD − forecast (mm day$^{-1}$) · positive = forecast too dry",
        fontsize=9,
    )

    mode = (
        "Display-only visual interpolation"
        if smooth
        else "Native 1.5° grid · no interpolation"
    )
    figure.suptitle(
        "India JJAS rainfall composites across six forecast weeks\n"
        f"Mean across {CASE_COUNT} starts in 2020–2021 · {mode}",
        fontsize=17,
        fontweight="bold",
        color="#172B3A",
        y=0.966,
    )
    display_note = (
        "Visual interpolation is display-only; it creates no new observations or forecasts. "
        "All evaluation uses the native 1.5° grid."
        if smooth
        else "Each colored tile is a native 1.5° evaluation cell; no spatial smoothing is applied."
    )
    figure.text(
        0.5,
        0.014,
        display_note
        + "  Survey of India ABDB-derived state/UT boundary. "
        "Exploratory/reused 2020–2021 hindcasts; not independent confirmation.",
        ha="center",
        fontsize=8.25,
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
    outputs: list[Path] = []
    outputs.extend(
        render_transposed_atlas(
            values,
            segments,
            output / "01_week_rows_spatial_composites_native_grid",
            smooth=False,
        )
    )
    outputs.extend(
        render_transposed_atlas(
            values,
            segments,
            output / "02_week_rows_spatial_composites_interpolated_display_only",
            smooth=True,
        )
    )

    evaluation = Path(args.evaluation_directory).expanduser().resolve()
    boundary = Path(args.india_boundary).expanduser().resolve()
    spatial = evaluation / "metrics" / "spatial_test_fields.npz"
    _atomic_write_json(
        output / "transposed_spatial_composites_manifest.json",
        {
            "schema_name": "fuxi_imd_transposed_spatial_composites",
            "schema_version": 1,
            "status": "complete",
            "selected_configuration": EXPECTED_CONFIGURATION,
            "evaluation_scope": (
                "2020-2021 exploratory/reused locked hindcasts; "
                "not independent confirmation"
            ),
            "aggregation": f"mean across the same {CASE_COUNT} JJAS starts",
            "test_initialization_count": CASE_COUNT,
            "lead_weeks": 6,
            "native_grid_degrees": 1.5,
            "layout": layout_contract(),
            "corrected_product_display_name": "Corrected Forecast",
            "metrics_recomputed": False,
            "visual_interpolation_used_for_metrics": False,
            "source_evaluation_manifest": {
                "path": str(evaluation / "manifest.json"),
                "sha256": _sha256(evaluation / "manifest.json"),
            },
            "source_spatial_fields": {
                "path": str(spatial),
                "sha256": _sha256(spatial),
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
