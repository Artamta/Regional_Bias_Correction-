#!/usr/bin/env python3
"""Render a display-only smooth preview of aligned INDRA spatial fields.

Interpolation changes only the appearance of the two descriptive maps.  The
input fields, support, and every reported metric remain on the native 1.5°
grid.  This script intentionally produces a preview outside the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
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
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


DEFAULT_RUN = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_spatial_appendix/full_20260827T140000Z"
)
EXPECTED_FIELDS_SHA256 = (
    "d98cd3b10fc9d26dd3724bd456d2c89fbfc628bf7e3fbd8e2e5ab35c6a4be383"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def interpolate_for_display(
    latitude: np.ndarray,
    longitude: np.ndarray,
    field: np.ndarray,
    support: np.ndarray,
    *,
    step: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Linearly interpolate values and retain a nearest-cell native mask."""

    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    valid = np.asarray(support, dtype=bool) & np.isfinite(values)
    if lat.ndim != 1 or lon.ndim != 1 or values.shape != (lat.size, lon.size):
        raise ValueError("field does not match the latitude/longitude grid")
    if np.count_nonzero(valid) < 3 or step <= 0.0:
        raise ValueError("interpolation requires supported values and positive step")

    fine_lat = np.arange(float(lat.min()), float(lat.max()) + step / 2.0, step)
    fine_lon = np.arange(float(lon.min()), float(lon.max()) + step / 2.0, step)
    fine_x, fine_y = np.meshgrid(fine_lon, fine_lat)
    native_x, native_y = np.meshgrid(lon, lat)
    points = np.column_stack((native_x[valid], native_y[valid]))
    smooth = griddata(
        points,
        values[valid],
        (fine_x, fine_y),
        method="linear",
        fill_value=np.nan,
    )

    # RegularGridInterpolator requires increasing coordinates.  Its nearest
    # lookup retains the exact native support decision for every display pixel
    # and prevents smooth values from appearing over unsupported cells.
    order_lat = np.argsort(lat)
    order_lon = np.argsort(lon)
    native_mask = np.asarray(support, dtype=np.float64)[np.ix_(order_lat, order_lon)]
    mask_lookup = RegularGridInterpolator(
        (lat[order_lat], lon[order_lon]),
        native_mask,
        method="nearest",
        bounds_error=False,
        fill_value=0.0,
    )
    fine_points = np.column_stack((fine_y.ravel(), fine_x.ravel()))
    fine_support = mask_lookup(fine_points).reshape(fine_y.shape) > 0.5
    return fine_lat, fine_lon, np.ma.masked_where(~fine_support | ~np.isfinite(smooth), smooth)


def render(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    crps: np.ndarray,
    mae: np.ndarray,
    boundary: tuple[np.ndarray, ...],
    output: Path,
) -> tuple[Path, Path]:
    fine_lat, fine_lon, smooth_crps = interpolate_for_display(
        latitude, longitude, crps, support
    )
    _, _, smooth_mae = interpolate_for_display(
        latitude, longitude, mae, support
    )
    fine_x, fine_y = np.meshgrid(fine_lon, fine_lat)
    native_x, native_y = np.meshgrid(longitude, latitude)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), facecolor="#F8FAFC")
    figure.subplots_adjust(left=0.055, right=0.965, top=0.80, bottom=0.16, wspace=0.16)
    panels = (
        (
            axes[0], smooth_crps, crps, "a  Local CRPS reduction", "probabilistic score",
            "CRPS(raw FuXi) − CRPS(INDRA-S2S v1)",
        ),
        (
            axes[1], smooth_mae, mae, "b  Ensemble-mean absolute-error reduction", "point forecast",
            "|raw FuXi mean − IMD| − |INDRA-S2S v1 mean − IMD|",
        ),
    )
    for axis, smooth, native, title, subtitle, scale_label in panels:
        displayed = np.asarray(smooth.compressed(), dtype=np.float64)
        limit = max(float(np.quantile(np.abs(displayed), 0.975)), 0.05)
        levels = np.linspace(-limit, limit, 25)
        image = axis.contourf(
            fine_x,
            fine_y,
            smooth,
            levels=levels,
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            extend="both",
            antialiased=True,
        )
        axis.add_collection(
            LineCollection(boundary, colors="#172B3A", linewidths=0.43, alpha=0.88)
        )
        native_valid = support & np.isfinite(native)
        axis.scatter(
            native_x[native_valid],
            native_y[native_valid],
            s=2.0,
            facecolors="white",
            edgecolors="none",
            alpha=0.24,
            zorder=3,
        )
        axis.set_xlim(66.4, 98.6)
        axis.set_ylim(6.0, 38.7)
        axis.set_aspect("equal")
        axis.set_xticks((70, 80, 90))
        axis.set_yticks((10, 20, 30))
        axis.set_xticklabels(("70°E", "80°E", "90°E"), fontsize=8)
        axis.set_yticklabels(("10°N", "20°N", "30°N"), fontsize=8)
        axis.tick_params(length=0, colors="#526777")
        axis.grid(color="#C9D6DF", linewidth=0.45, alpha=0.6)
        axis.set_facecolor("#EDF2F6")
        axis.set_title(title, loc="left", fontsize=12.0, color="#172B3A", pad=15)
        axis.text(
            0.0, 1.018, subtitle, transform=axis.transAxes, ha="left", va="bottom",
            fontsize=8.4, color="#637887"
        )
        colorbar = figure.colorbar(
            image, ax=axis, orientation="horizontal", pad=0.055, fraction=0.05,
            aspect=34, ticks=np.linspace(-limit, limit, 5), format="%.1f",
        )
        colorbar.ax.tick_params(labelsize=7.6, length=2, colors="#526777")
        colorbar.outline.set_linewidth(0.5)
        colorbar.set_label(
            scale_label + "  (mm day$^{-1}$)  ·  positive is better",
            fontsize=8.2,
            color="#344B5C",
        )

    figure.suptitle(
        "Where INDRA-S2S v1 changes subseasonal monsoon rainfall errors",
        fontsize=17.0,
        fontweight="bold",
        color="#122738",
        y=0.955,
    )
    figure.text(
        0.5,
        0.872,
        "2020–2024 valid-midpoint JJAS · 169 cases per lead · pooled across Weeks 1–6",
        ha="center",
        fontsize=10.0,
        color="#526777",
    )
    figure.text(
        0.5,
        0.047,
        "DISPLAY-ONLY INTERPOLATION  •  All values and metrics originate on the native 1.5° grid  •  "
        "No pixelwise or state-level significance claim",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#9A3412",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF7ED", "edgecolor": "#FDBA74", "linewidth": 0.8},
    )
    png = output / "indra_spatial_interpolated_display_preview.png"
    pdf = output / "indra_spatial_interpolated_display_preview.pdf"
    figure.savefig(png, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def run(args: argparse.Namespace) -> Path:
    run_root = Path(args.run).resolve()
    manifest_path = run_root / "manifest.json"
    fields_path = run_root / "tables/spatial_fields.npz"
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifact_sha256", {}).get("tables/spatial_fields.npz")
    if (
        manifest.get("status") != "complete_descriptive_appendix_evidence"
        or expected != EXPECTED_FIELDS_SHA256
        or sha256_file(fields_path) != EXPECTED_FIELDS_SHA256
        or manifest.get("contract", {}).get("sealed_2025_target_opened") is not False
        or manifest.get("interpretation", {}).get("pixelwise_inference") is not False
    ):
        raise ValueError("aligned spatial source contract differs")
    with np.load(fields_path, allow_pickle=False) as archive:
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
        crps = np.asarray(archive["pooled_crps_reduction"], dtype=np.float64)
        mae = np.asarray(archive["pooled_mae_reduction"], dtype=np.float64)
        weights = np.asarray(archive["all_india_area_weight_km2"], dtype=np.float64)
        temporal = np.asarray(archive["temporal_support"], dtype=np.float64)
    support = (weights > 0.0) & (np.sum(temporal, axis=0) > 0.0)
    boundary, boundary_receipt = load_india_boundary(args.india_boundary)
    output.mkdir(parents=True)
    png, pdf = render(
        latitude=latitude,
        longitude=longitude,
        support=support,
        crps=crps,
        mae=mae,
        boundary=boundary,
        output=output,
    )
    receipt = {
        "schema_name": "indra_spatial_interpolated_display_preview",
        "schema_version": 1,
        "status": "display_only_preview_not_paper_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "native_fields": str(fields_path),
            "native_fields_sha256": sha256_file(fields_path),
        },
        "contract": {
            "native_grid_degrees": 1.5,
            "display_interpolation_degrees": 0.12,
            "interpolation_used_for_metrics": False,
            "native_support_retained_by_nearest_cell": True,
            "pixelwise_inference": False,
            "state_level_inference": False,
            "opened_2025_observation": False,
            "sealed_2025_target_opened": False,
        },
        "india_boundary": boundary_receipt,
        "artifacts": {
            png.name: sha256_file(png),
            pdf.name: sha256_file(pdf),
        },
    }
    atomic_json(output / "manifest.json", receipt)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--india-boundary", type=Path, default=DEFAULT_INDIA_BOUNDARY)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = run(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
