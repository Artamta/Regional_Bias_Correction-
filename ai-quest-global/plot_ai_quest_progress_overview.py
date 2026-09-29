#!/usr/bin/env python3
"""Create a one-page overview of the live AI Quest precipitation forecast."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import xarray as xr


INK = "#17212b"
MUTED = "#607080"
TEAL = "#2a9d8f"
BLUE = "#277da1"
PURPLE = "#7b6fd0"
ORANGE = "#e76f51"
PALE = "#f3f7f8"
CATEGORY_COLORS = ("#9a6324", "#e9c46a", "#f4f1de", "#72b7b2", "#176d8c")
CATEGORY_LABELS = ("Q1 driest", "Q2", "Q3 near-normal", "Q4", "Q5 wettest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card(axis, title: str, body: str, color: str) -> None:
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            transform=axis.transAxes,
            facecolor=color,
            edgecolor=color,
            alpha=0.11,
            clip_on=False,
        )
    )
    axis.text(0.04, 0.78, title, color=color, fontsize=11, weight="bold", va="top")
    axis.text(0.04, 0.56, body, color=INK, fontsize=10, va="top", linespacing=1.35)


def _load_bundle(bundle: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = bundle / "leader_handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = []
    latitude = longitude = None
    for record in sorted(manifest["files"], key=lambda item: item["period"]):
        path = bundle / record["filename"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"forecast checksum mismatch: {path}")
        with xr.open_dataarray(path) as data:
            values = np.asarray(data.values, dtype=np.float32)
            if values.shape != (5, 121, 240):
                raise ValueError(f"unexpected forecast shape {values.shape} in {path}")
            fields.append(values)
            current_lat = np.asarray(data.latitude.values, dtype=np.float32)
            current_lon = np.asarray(data.longitude.values, dtype=np.float32)
            if latitude is None:
                latitude, longitude = current_lat, current_lon
            else:
                np.testing.assert_array_equal(current_lat, latitude)
                np.testing.assert_array_equal(current_lon, longitude)
    probability = np.stack(fields)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("forecast contains invalid probabilities")
    if float(np.max(np.abs(probability.sum(axis=1, dtype=np.float64) - 1.0))) > 1e-6:
        raise ValueError("forecast probabilities do not sum to one")
    assert latitude is not None and longitude is not None
    return manifest, probability, latitude, longitude


def _map_axis(figure, rect):
    projection = ccrs.Robinson(central_longitude=180)
    axis = figure.add_axes(rect, projection=projection)
    axis.set_global()
    axis.coastlines(resolution="110m", linewidth=0.55, color="#465560")
    axis.gridlines(linewidth=0.35, color="#87939b", alpha=0.35, linestyle=":")
    return axis


def run(args: argparse.Namespace) -> None:
    manifest, probability, latitude, longitude = _load_bundle(args.bundle)
    performance = manifest["performance"]
    dominant = np.argmax(probability, axis=1) + 1
    confidence = np.max(probability, axis=1)
    weights = np.cos(np.deg2rad(latitude))[:, None]
    confidence_mean = [
        float(np.sum(confidence[lead] * weights) / (weights.sum() * len(longitude)))
        for lead in range(2)
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(16, 10))
    figure.text(
        0.045,
        0.965,
        "AI Quest precipitation adapter · what we have now",
        fontsize=22,
        weight="bold",
        va="top",
    )
    figure.text(
        0.045,
        0.930,
        "Issue 13 August 2026 · calibrated FuXi-S2S probabilities for five rainfall quintiles",
        fontsize=11,
        color=MUTED,
        va="top",
    )

    card_width = 0.285
    cards = [
        figure.add_axes([0.045, 0.810, card_width, 0.095]),
        figure.add_axes([0.3575, 0.810, card_width, 0.095]),
        figure.add_axes([0.670, 0.810, card_width, 0.095]),
    ]
    _card(
        cards[0],
        "DATA & MODEL",
        "Train 2002–18 · validation 2019–20\n2021 test sealed · 51 FuXi members\n151,573 parameters · frozen seed 42",
        PURPLE,
    )
    _card(
        cards[1],
        "VALIDATION RESULT",
        f"RPS {performance['validation_rps_pooled']:.4f} (lower is better)\n"
        f"RPSS {100 * performance['validation_rpss_pooled']:+.2f}% vs uniform\n"
        f"D19–25 {100 * performance['validation_rpss_d19_25']:+.2f}% · "
        f"D26–32 {100 * performance['validation_rpss_d26_32']:+.2f}%",
        TEAL,
    )
    _card(
        cards[2],
        "LIVE FORECAST STATUS",
        "51/51 unique members · source QC passed\n2 weekly fields × 5 categories · sums valid\nLive competition score pending observations",
        BLUE,
    )

    warning = figure.add_axes([0.045, 0.755, 0.910, 0.038])
    warning.set_axis_off()
    warning.add_patch(
        FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.04",
            transform=warning.transAxes, facecolor=ORANGE, edgecolor="none", alpha=0.13
        )
    )
    warning.text(
        0.5,
        0.5,
        "PROXY PREVIEW — uses public ERA5 2003–2022 climatology; replace with official AI-WQ boundaries before packaging/upload",
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=10.2,
        weight="bold",
    )

    lead_titles = (
        "D19–25 · valid 31 Aug–6 Sep 2026",
        "D26–32 · valid 7–13 Sep 2026",
    )
    dominant_axes = (
        _map_axis(figure, [0.045, 0.440, 0.435, 0.275]),
        _map_axis(figure, [0.520, 0.440, 0.435, 0.275]),
    )
    category_cmap = ListedColormap(CATEGORY_COLORS)
    category_norm = BoundaryNorm(np.arange(0.5, 6.5, 1), category_cmap.N)
    dominant_mesh = None
    for lead, axis in enumerate(dominant_axes):
        dominant_mesh = axis.pcolormesh(
            longitude,
            latitude,
            dominant[lead],
            cmap=category_cmap,
            norm=category_norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        axis.set_title(f"Most likely rainfall category · {lead_titles[lead]}", pad=8, weight="bold")
    category_bar = figure.add_axes([0.180, 0.423, 0.640, 0.015])
    colorbar = figure.colorbar(dominant_mesh, cax=category_bar, orientation="horizontal", ticks=range(1, 6))
    colorbar.ax.set_xticklabels(CATEGORY_LABELS, fontsize=9)
    colorbar.outline.set_visible(False)

    confidence_axes = (
        _map_axis(figure, [0.045, 0.105, 0.435, 0.265]),
        _map_axis(figure, [0.520, 0.105, 0.435, 0.265]),
    )
    confidence_mesh = None
    for lead, axis in enumerate(confidence_axes):
        confidence_mesh = axis.pcolormesh(
            longitude,
            latitude,
            confidence[lead],
            cmap="YlGnBu",
            vmin=0.20,
            vmax=0.72,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        axis.set_title(
            f"Confidence: probability of winning category · area-weighted mean {confidence_mean[lead]:.1%}",
            pad=8,
            weight="bold",
        )
    confidence_bar = figure.add_axes([0.180, 0.086, 0.640, 0.015])
    confidence_colorbar = figure.colorbar(confidence_mesh, cax=confidence_bar, orientation="horizontal")
    confidence_colorbar.set_label("Maximum category probability (20% = uniform baseline)", fontsize=9)
    confidence_colorbar.outline.set_visible(False)

    figure.text(
        0.5,
        0.026,
        "Interpretation: upper maps show the locally most probable rainfall quintile; lower maps show confidence, not expected rainfall amount. "
        "Validation improvement is evidence from 2019–20 only, not the score of this live issue.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )
    figure.text(
        0.955,
        0.012,
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ha="right",
        fontsize=7.8,
        color=MUTED,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = args.output.with_suffix(suffix)
        figure.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        outputs.append({"path": str(path), "sha256": sha256_file(path)})
    plt.close(figure)
    plot_manifest = {
        "status": "complete",
        "source_bundle_manifest": str(args.bundle / "leader_handoff_manifest.json"),
        "source_bundle_manifest_sha256": sha256_file(args.bundle / "leader_handoff_manifest.json"),
        "outputs": outputs,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(plot_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plot_manifest, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="Output stem without extension")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
