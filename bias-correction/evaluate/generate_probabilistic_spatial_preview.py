#!/usr/bin/env python3
"""Generate an honest spatial preview of the FuXi probabilistic calibrator.

This is a diagnostic renderer, not a paper-result generator.  It reconstructs
one fixed-seed corrected ensemble from the archived test adjustment fields and
verifies it against IMD using the corrected end-date-labelled convention:
W1 = init+1..init+7, ..., W6 = init+36..init+42.

The representative JJAS initialization is selected mechanically: its pooled
six-week CRPSS is closest to the median over all JJAS 2020--2021 starts.  This
avoids choosing a visually impressive case by hand.
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
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


DEFAULT_RUN = (
    WORK_ROOT
    / "resultsv2/fuxi_allseason_ensemble_calibration/"
    "full_publication_20260822T115253Z"
)
DEFAULT_CACHE = WORK_ROOT / "cache/fuxi_tp_members_weekly_2002_2021_allseason_v1.npy"
DEFAULT_IMD = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/observations/ground_truth_v1/daily/imd/tp/"
    "india_1p5_27x27_v1"
)
DEFAULT_OUTPUT = WORK_ROOT / "diagnostics/probabilistic_spatial_preview_20260824"
THRESHOLD_MM_DAY = 10.0
EXPECTED_MEMBERS = 51
EXPECTED_LEADS = 6
EXPECTED_GRID = (27, 27)
EXPECTED_SUPPORT_CELLS = 171


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_figure(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def apply_affine_log_calibration(
    members: np.ndarray,
    delta_log_location: np.ndarray,
    spread_factor: np.ndarray,
) -> np.ndarray:
    """Apply the archived rank-preserving location/spread transformation."""

    raw = np.asarray(members, dtype=np.float32)
    delta = np.asarray(delta_log_location, dtype=np.float32)
    spread = np.asarray(spread_factor, dtype=np.float32)
    if raw.ndim != 4 or raw.shape[1:] != (EXPECTED_LEADS, *EXPECTED_GRID):
        raise ValueError(f"members must be [member,6,27,27], found {raw.shape}")
    if delta.shape != raw.shape[1:] or spread.shape != raw.shape[1:]:
        raise ValueError("delta/spread fields are incompatible with members")
    if np.any(raw < 0.0) or not np.isfinite(raw).all():
        raise ValueError("raw members contain invalid rainfall")
    if np.any(spread <= 0.0) or not np.isfinite(spread).all():
        raise ValueError("spread factors must be finite and positive")
    log_members = np.log1p(raw).astype(np.float32)
    centre = log_members.mean(axis=0, dtype=np.float64).astype(np.float32)
    corrected_log = centre[None] + delta[None] + spread[None] * (
        log_members - centre[None]
    )
    corrected = np.expm1(np.clip(corrected_log, 0.0, 20.0)).astype(np.float32)
    if np.any(corrected < 0.0) or not np.isfinite(corrected).all():
        raise ValueError("calibration produced invalid rainfall")
    return corrected


def ensemble_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Finite-ensemble CRPS at each lead/grid cell without an M x M array."""

    values = np.asarray(members, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim < 2 or target.shape != values.shape[1:]:
        raise ValueError("members/truth shapes are incompatible")
    count = values.shape[0]
    first = np.mean(np.abs(values - target[None]), axis=0, dtype=np.float64)
    ordered = np.sort(values, axis=0)
    coefficients = (2.0 * np.arange(count) - count + 1.0).reshape(
        (count,) + (1,) * (values.ndim - 1)
    )
    dispersion = np.sum(ordered * coefficients, axis=0, dtype=np.float64) / count**2
    return first - dispersion


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    spatial_weights = np.asarray(weights, dtype=np.float64)
    support = spatial_weights > 0.0
    if values.shape[-2:] != spatial_weights.shape:
        raise ValueError("field and spatial weights are incompatible")
    expanded = np.broadcast_to(spatial_weights, values.shape)
    valid = np.broadcast_to(support, values.shape)
    if not np.isfinite(values[valid]).all():
        raise ValueError("field is non-finite on supported cells")
    return float(np.sum(values[valid] * expanded[valid]) / np.sum(expanded[valid]))


def select_median_skill_case(skills: np.ndarray, candidate_indices: np.ndarray) -> int:
    values = np.asarray(skills, dtype=np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 1 or candidates.ndim != 1 or values.size != candidates.size:
        raise ValueError("skills and candidate indices must be paired 1-D arrays")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("candidate skills must be non-empty and finite")
    median = float(np.median(values))
    position = int(np.argmin(np.abs(values - median)))
    return int(candidates[position])


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(cache_path).resolve()
    metadata_path = path.with_suffix("").with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    members = np.load(path, mmap_mode="r", allow_pickle=False)
    initializations = np.asarray(metadata["initializations"], dtype="datetime64[D]")
    latitude = np.asarray(metadata["latitude"], dtype=np.float64)
    longitude = np.asarray(metadata["longitude"], dtype=np.float64)
    if members.shape[1:] != (EXPECTED_MEMBERS, EXPECTED_LEADS, *EXPECTED_GRID):
        raise ValueError(f"unexpected member cache shape {members.shape}")
    if initializations.shape != (members.shape[0],):
        raise ValueError("cache dates do not match member cases")
    if latitude.shape != (27,) or longitude.shape != (27,):
        raise ValueError("preview requires the native 27x27 India grid")
    return members, initializations, latitude, longitude


def load_aligned_imd_truth(
    initializations: np.ndarray,
    imd_root: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load weekly IMD means with W1=init+1..+7 and W6=init+36..+42."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    offsets = np.arange(1, 43, dtype="timedelta64[D]").reshape(1, 6, 7)
    verification_dates = starts[:, None, None] + offsets
    first_year = int(str(verification_dates.min())[:4])
    final_year = int(str(verification_dates.max())[:4])
    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    stores: list[str] = []
    for year in range(first_year, final_year + 1):
        store = Path(imd_root) / f"{year}.zarr"
        with xr.open_zarr(store, consolidated=True) as dataset:
            if not np.array_equal(dataset.latitude.values, latitude):
                raise ValueError(f"IMD latitude differs in {store}")
            if not np.array_equal(dataset.longitude.values, longitude):
                raise ValueError(f"IMD longitude differs in {store}")
            all_dates.append(np.asarray(dataset.time.values, dtype="datetime64[D]"))
            all_values.append(np.asarray(dataset.observation.load().values, dtype=np.float32))
        stores.append(str(store))
    dates = np.concatenate(all_dates)
    daily = np.concatenate(all_values)
    order = np.argsort(dates)
    dates = dates[order]
    daily = daily[order]
    requested = verification_dates.reshape(-1)
    positions = np.searchsorted(dates, requested)
    if np.any(positions >= dates.size) or not np.array_equal(dates[positions], requested):
        raise ValueError("aligned IMD verification dates are incomplete")
    selected = daily[positions].reshape(*verification_dates.shape, *EXPECTED_GRID)
    truth = np.mean(selected, axis=2, dtype=np.float64).astype(np.float32)
    return truth, tuple(stores)


def decorate_map(
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


def map_field(
    axis: plt.Axes,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    *,
    cmap: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any = None,
) -> Any:
    shown = np.asarray(field, dtype=np.float64).copy()
    shown[~support] = np.nan
    return axis.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(shown),
        shading="nearest",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        rasterized=True,
    )


def render_probability_atlas(
    raw: np.ndarray,
    corrected: np.ndarray,
    truth: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    boundary_segments: Sequence[np.ndarray],
    initialization: np.datetime64,
    output: Path,
) -> tuple[Path, Path]:
    raw_probability = np.mean(raw > THRESHOLD_MM_DAY, axis=0)
    corrected_probability = np.mean(corrected > THRESHOLD_MM_DAY, axis=0)
    probability_change = corrected_probability - raw_probability
    observed_event = (truth > THRESHOLD_MM_DAY).astype(np.float64)
    change_limit = max(0.1, float(np.ceil(np.nanmax(np.abs(probability_change)) * 10) / 10))
    change_norm = TwoSlopeNorm(vmin=-change_limit, vcenter=0.0, vmax=change_limit)
    event_cmap = ListedColormap(("#F4F1E8", "#08519C"))
    event_norm = BoundaryNorm((-0.5, 0.5, 1.5), event_cmap.N)

    figure, axes = plt.subplots(6, 4, figsize=(12.8, 17.8), facecolor="white")
    probability_image = change_image = event_image = None
    for lead in range(6):
        fields = (
            raw_probability[lead],
            corrected_probability[lead],
            probability_change[lead],
            observed_event[lead],
        )
        for column, field in enumerate(fields):
            if column < 2:
                image = map_field(
                    axes[lead, column], field, latitude, longitude, support,
                    cmap="YlGnBu", vmin=0.0, vmax=1.0,
                )
                probability_image = image
            elif column == 2:
                image = map_field(
                    axes[lead, column], field, latitude, longitude, support,
                    cmap="RdBu", norm=change_norm,
                )
                change_image = image
            else:
                image = map_field(
                    axes[lead, column], field, latitude, longitude, support,
                    cmap=event_cmap, norm=event_norm,
                )
                event_image = image
            decorate_map(
                axes[lead, column], boundary_segments,
                left=column == 0, bottom=lead == 5,
            )
            if lead == 0:
                axes[lead, column].set_title(
                    (
                        "Raw P(rain > 10)",
                        "Corrected P(rain > 10)",
                        "Corrected − raw probability",
                        "IMD event (> 10)",
                    )[column],
                    fontsize=10.5,
                    pad=7,
                )
        axes[lead, 0].text(
            -0.30, 0.5, f"W{lead + 1}", transform=axes[lead, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
        )
    if probability_image is None or change_image is None or event_image is None:
        raise RuntimeError("probability atlas did not render all scales")
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.09, wspace=0.08, hspace=0.10)
    probability_bar = figure.colorbar(
        probability_image,
        cax=figure.add_axes((0.12, 0.046, 0.30, 0.012)),
        orientation="horizontal",
    )
    probability_bar.set_label("Ensemble exceedance probability", fontsize=9)
    change_bar = figure.colorbar(
        change_image,
        cax=figure.add_axes((0.49, 0.046, 0.24, 0.012)),
        orientation="horizontal",
        extend="both",
    )
    change_bar.set_label("Probability change", fontsize=9)
    event_bar = figure.colorbar(
        event_image,
        cax=figure.add_axes((0.79, 0.046, 0.13, 0.012)),
        orientation="horizontal",
        ticks=(0, 1),
    )
    event_bar.ax.set_xticklabels(("No", "Yes"))
    event_bar.set_label("Observed event", fontsize=9)
    figure.suptitle(
        "Probabilistic rainfall calibration over India\n"
        f"Initialization {str(initialization)} · 51 members · native 1.5° grid",
        fontsize=16,
        fontweight="bold",
        color="#172B3A",
        y=0.965,
    )
    figure.text(
        0.5,
        0.012,
        "DIAGNOSTIC ONLY: archived pre-fix checkpoint; IMD verification corrected to init+1…init+42. "
        "Not a final paper result.",
        ha="center",
        fontsize=8.3,
        color="#9C2F2F",
    )
    png = output / "01_probability_atlas_all_six_weeks.png"
    pdf = output / "01_probability_atlas_all_six_weeks.pdf"
    save_figure(figure, png, dpi=220, bbox_inches="tight", facecolor="white")
    save_figure(figure, pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def render_distribution_detail(
    raw: np.ndarray,
    corrected: np.ndarray,
    truth: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
    boundary_segments: Sequence[np.ndarray],
    initialization: np.datetime64,
    output: Path,
    *,
    lead: int = 2,
) -> tuple[Path, Path, dict[str, float]]:
    raw_mean = raw[:, lead].mean(axis=0)
    corrected_mean = corrected[:, lead].mean(axis=0)
    observed = truth[lead]
    error_improvement = np.abs(raw_mean - observed) - np.abs(corrected_mean - observed)
    raw_probability = np.mean(raw[:, lead] > THRESHOLD_MM_DAY, axis=0)
    corrected_probability = np.mean(corrected[:, lead] > THRESHOLD_MM_DAY, axis=0)
    probability_change = corrected_probability - raw_probability
    observed_event = (observed > THRESHOLD_MM_DAY).astype(np.float64)
    raw_width = np.quantile(raw[:, lead], 0.9, axis=0) - np.quantile(raw[:, lead], 0.1, axis=0)
    corrected_width = np.quantile(corrected[:, lead], 0.9, axis=0) - np.quantile(corrected[:, lead], 0.1, axis=0)
    width_change = corrected_width - raw_width
    raw_crps = ensemble_crps(raw[:, lead], observed)
    corrected_crps = ensemble_crps(corrected[:, lead], observed)
    crps_improvement = raw_crps - corrected_crps

    rain_limit = float(np.ceil(np.quantile(np.concatenate((observed[support], raw_mean[support], corrected_mean[support])), 0.99) / 2) * 2)
    width_limit = float(np.ceil(np.quantile(np.concatenate((raw_width[support], corrected_width[support])), 0.99) / 2) * 2)
    crps_limit = float(np.ceil(np.quantile(np.concatenate((raw_crps[support], corrected_crps[support])), 0.99) / 2) * 2)
    signed_limits = {
        "error": max(1.0, float(np.ceil(np.quantile(np.abs(error_improvement[support]), 0.98)))),
        "probability": max(0.1, float(np.ceil(np.quantile(np.abs(probability_change[support]), 0.98) * 10) / 10)),
        "width": max(1.0, float(np.ceil(np.quantile(np.abs(width_change[support]), 0.98)))),
        "crps": max(1.0, float(np.ceil(np.quantile(np.abs(crps_improvement[support]), 0.98)))),
    }
    event_cmap = ListedColormap(("#F4F1E8", "#08519C"))
    event_norm = BoundaryNorm((-0.5, 0.5, 1.5), event_cmap.N)

    rows = (
        (observed, raw_mean, corrected_mean, error_improvement),
        (observed_event, raw_probability, corrected_probability, probability_change),
        (None, raw_width, corrected_width, width_change),
        (None, raw_crps, corrected_crps, crps_improvement),
    )
    titles = (
        (
            f"IMD observation [0–{rain_limit:g}]",
            f"Raw ensemble mean [0–{rain_limit:g}]",
            f"Corrected ensemble mean [0–{rain_limit:g}]",
            f"Absolute-error reduction [±{signed_limits['error']:g}]",
        ),
        (
            "IMD event (>10)",
            "Raw P(>10) [0–1]",
            "Corrected P(>10) [0–1]",
            f"Probability change [±{signed_limits['probability']:g}]",
        ),
        (
            "Observation has no interval",
            f"Raw 80% interval width [0–{width_limit:g}]",
            f"Corrected 80% interval width [0–{width_limit:g}]",
            f"Width change [±{signed_limits['width']:g}]",
        ),
        (
            "CRPS uses IMD as truth",
            f"Raw pointwise CRPS [0–{crps_limit:g}]",
            f"Corrected pointwise CRPS [0–{crps_limit:g}]",
            f"CRPS improvement [±{signed_limits['crps']:g}]",
        ),
    )
    figure, axes = plt.subplots(4, 4, figsize=(14.6, 13.2), facecolor="white")
    images: list[Any] = []
    for row, fields in enumerate(rows):
        for column, field in enumerate(fields):
            axis = axes[row, column]
            axis.set_title(titles[row][column], fontsize=9.5, pad=6)
            if field is None:
                axis.text(
                    0.5, 0.52,
                    "The observation is one\nrealized rainfall field.",
                    transform=axis.transAxes, ha="center", va="center",
                    fontsize=10, color="#526777",
                )
                axis.set_xlim(67.0, 98.7)
                axis.set_ylim(6.0, 38.7)
                axis.set_xticks(())
                axis.set_yticks(())
                continue
            if row == 0 and column < 3:
                image = map_field(axis, field, latitude, longitude, support, cmap="YlGnBu", vmin=0.0, vmax=rain_limit)
            elif row == 1 and column == 0:
                image = map_field(axis, field, latitude, longitude, support, cmap=event_cmap, norm=event_norm)
            elif row == 1 and column in (1, 2):
                image = map_field(axis, field, latitude, longitude, support, cmap="YlGnBu", vmin=0.0, vmax=1.0)
            elif row == 2 and column in (1, 2):
                image = map_field(axis, field, latitude, longitude, support, cmap="magma", vmin=0.0, vmax=width_limit)
            elif row == 3 and column in (1, 2):
                image = map_field(axis, field, latitude, longitude, support, cmap="viridis", vmin=0.0, vmax=crps_limit)
            else:
                key = ("error", "probability", "width", "crps")[row]
                limit = signed_limits[key]
                image = map_field(axis, field, latitude, longitude, support, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
            images.append(image)
            decorate_map(axis, boundary_segments, left=column == 0, bottom=row == 3)
    figure.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.07, wspace=0.13, hspace=0.18)
    figure.suptitle(
        "What the probabilistic correction changes\n"
        f"Initialization {str(initialization)} · Week {lead + 1} · positive improvement maps mean corrected is better",
        fontsize=15.5,
        fontweight="bold",
        color="#172B3A",
        y=0.97,
    )
    figure.text(
        0.5,
        0.018,
        "DIAGNOSTIC ONLY: seed 43 archived pre-fix checkpoint; correctly shifted IMD verification. "
        "All fields shown on the native grid.",
        ha="center",
        fontsize=8.4,
        color="#9C2F2F",
    )
    png = output / "02_distribution_detail_week3.png"
    pdf = output / "02_distribution_detail_week3.pdf"
    save_figure(figure, png, dpi=220, bbox_inches="tight", facecolor="white")
    save_figure(figure, pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metrics = {
        "raw_crps": weighted_mean(raw_crps, weights),
        "corrected_crps": weighted_mean(corrected_crps, weights),
        "crpss_vs_raw": 1.0 - weighted_mean(corrected_crps, weights) / weighted_mean(raw_crps, weights),
        "raw_mean_absolute_error": weighted_mean(np.abs(raw_mean - observed), weights),
        "corrected_mean_absolute_error": weighted_mean(np.abs(corrected_mean - observed), weights),
        "raw_mean_80pct_width": weighted_mean(raw_width, weights),
        "corrected_mean_80pct_width": weighted_mean(corrected_width, weights),
    }
    return png, pdf, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--imd-root", type=Path, default=DEFAULT_IMD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--india-boundary", type=Path, default=DEFAULT_INDIA_BOUNDARY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adjustment_path = run / f"models/location_spread/seed_{args.seed}/test_adjustments.npz"
    support_path = run / "evaluation/scoring_support.npz"
    checkpoint_path = run / f"models/location_spread/seed_{args.seed}/best.pt"
    members, cache_initializations, latitude, longitude = load_cache(args.cache)
    with np.load(adjustment_path, allow_pickle=False) as source:
        test_initializations = np.asarray(source["initializations"], dtype="datetime64[D]")
        deltas = np.asarray(source["delta_log_location"], dtype=np.float32)
        spreads = np.asarray(source["spread_factor"], dtype=np.float32)
    with np.load(support_path, allow_pickle=False) as source:
        support = np.asarray(source["support_mask"], dtype=bool)
        weights = np.asarray(source["scoring_weight_km2_fraction"], dtype=np.float64)
    if np.count_nonzero(support) != EXPECTED_SUPPORT_CELLS:
        raise ValueError("scoring support does not contain 171 cells")
    positions = np.searchsorted(cache_initializations, test_initializations)
    if not np.array_equal(cache_initializations[positions], test_initializations):
        raise ValueError("adjustment initializations are absent from member cache")
    truth, imd_stores = load_aligned_imd_truth(
        test_initializations,
        args.imd_root,
        latitude,
        longitude,
    )
    months = np.asarray([int(str(value)[5:7]) for value in test_initializations])
    candidate_positions = np.flatnonzero(np.isin(months, (6, 7, 8, 9)))
    skills = np.empty(candidate_positions.size, dtype=np.float64)
    for item, test_position in enumerate(candidate_positions):
        raw_case = np.asarray(members[positions[test_position]], dtype=np.float32)
        corrected_case = apply_affine_log_calibration(
            raw_case,
            deltas[test_position],
            spreads[test_position],
        )
        raw_score = weighted_mean(ensemble_crps(raw_case, truth[test_position]), weights)
        corrected_score = weighted_mean(
            ensemble_crps(corrected_case, truth[test_position]), weights
        )
        skills[item] = 1.0 - corrected_score / raw_score
    selected = select_median_skill_case(skills, candidate_positions)
    raw = np.asarray(members[positions[selected]], dtype=np.float32)
    corrected = apply_affine_log_calibration(raw, deltas[selected], spreads[selected])
    selected_truth = truth[selected]
    boundary_segments, boundary_provenance = load_india_boundary(args.india_boundary)
    outputs: list[Path] = []
    outputs.extend(
        render_probability_atlas(
            raw,
            corrected,
            selected_truth,
            latitude,
            longitude,
            support,
            boundary_segments,
            test_initializations[selected],
            output,
        )
    )
    detail_png, detail_pdf, detail_metrics = render_distribution_detail(
        raw,
        corrected,
        selected_truth,
        latitude,
        longitude,
        support,
        weights,
        boundary_segments,
        test_initializations[selected],
        output,
    )
    outputs.extend((detail_png, detail_pdf))
    fields_path = output / "diagnostic_fields.npz"
    np.savez_compressed(
        fields_path,
        initialization=test_initializations[selected],
        latitude=latitude,
        longitude=longitude,
        support_mask=support,
        raw_members=raw,
        corrected_members=corrected,
        aligned_imd_truth=selected_truth,
        threshold_mm_day=np.float64(THRESHOLD_MM_DAY),
        seed=np.int64(args.seed),
    )
    outputs.append(fields_path)
    raw_all = weighted_mean(ensemble_crps(raw, selected_truth), weights)
    corrected_all = weighted_mean(ensemble_crps(corrected, selected_truth), weights)
    manifest = {
        "schema_name": "fuxi_probabilistic_spatial_preview",
        "schema_version": 1,
        "status": "complete_diagnostic_not_paper_evidence",
        "created_utc": utc_now(),
        "scientific_warning": (
            "The checkpoint was trained before the IMD end-date alignment defect was fixed. "
            "This preview uses correctly aligned IMD only for visualization and diagnostic scoring."
        ),
        "verification_windows": "W1=init+1..+7; W2=+8..+14; ...; W6=+36..+42",
        "model": "location_spread",
        "seed": args.seed,
        "threshold_mm_day": THRESHOLD_MM_DAY,
        "selection": {
            "candidate_scope": "all June-September 2020-2021 development starts",
            "rule": "pooled six-week CRPSS closest to the candidate median",
            "candidate_count": int(candidate_positions.size),
            "median_candidate_crpss": float(np.median(skills)),
            "selected_initialization": str(test_initializations[selected]),
            "selected_crpss": float(1.0 - corrected_all / raw_all),
        },
        "selected_case_pooled_scores": {
            "raw_crps": raw_all,
            "corrected_crps": corrected_all,
            "crpss_vs_raw": 1.0 - corrected_all / raw_all,
        },
        "week3_detail_scores": detail_metrics,
        "sources": {
            "cache": {"path": str(args.cache.resolve()), "sha256": sha256_file(args.cache)},
            "adjustments": {"path": str(adjustment_path), "sha256": sha256_file(adjustment_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "scoring_support": {"path": str(support_path), "sha256": sha256_file(support_path)},
            "imd_stores": list(imd_stores),
            "india_boundary": boundary_provenance,
        },
        "artifacts": {path.name: sha256_file(path) for path in outputs},
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    for path in (*outputs, manifest_path):
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
