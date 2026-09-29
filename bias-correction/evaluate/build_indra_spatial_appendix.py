#!/usr/bin/env python3
"""Build aligned, descriptive spatial evidence for the INDRA-S2S appendix.

The renderer reopens the frozen 2020--2024 FuXi members, reconstructs the
selected seed-43 INDRA ensemble, and independently rebuilds IMD targets using
the end-labelled +1...+42-day contract.  Spatial panels are descriptive: the
paper ACC is a per-case spatial correlation and is deliberately not rendered
as a cellwise "ACC map".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "src"
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_india_s2s_probabilistic_bridge_predictions as audit  # noqa: E402
import india_s2s_probabilistic_bridge as bridge  # noqa: E402
from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


DEFAULT_INFERENCE = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge/"
    "inference_full_20260825T233121Z/manifest.json"
)
DEFAULT_SCORING = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge/"
    "scoring_full_20260825T234808Z/manifest.json"
)
DEFAULT_PARENT = (
    PROJECT_ROOT
    / "resultsv3/fuxi_allseason_ensemble_calibration/"
    "full_20260825T224711Z/manifest.json"
)
DEFAULT_PREFLIGHT = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge/"
    "preflight_20260825T225552Z/manifest.json"
)
EXPECTED_SCORE_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 88}
CASE_QUANTILES = (0.50, 0.10)


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


def empirical_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return finite-ensemble CRPS for [case,member,lead,y,x] arrays."""

    ensemble = np.asarray(members, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if ensemble.ndim != 5 or target.shape != (
        ensemble.shape[0],
        ensemble.shape[2],
        ensemble.shape[3],
        ensemble.shape[4],
    ):
        raise ValueError("members and truth have incompatible dimensions")
    count = ensemble.shape[1]
    first = np.mean(np.abs(ensemble - target[:, None]), axis=1, dtype=np.float64)
    ordered = np.sort(ensemble, axis=1)
    coefficients = (2.0 * np.arange(count) - count + 1.0).reshape(
        (1, count, 1, 1, 1)
    )
    dispersion = np.sum(ordered * coefficients, axis=1, dtype=np.float64) / count**2
    return first - dispersion


def select_nearest_quantile(values: np.ndarray, quantile: float) -> int:
    """Select the first finite value nearest a fixed quantile."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("selection scores must be a finite non-empty vector")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    target = float(np.quantile(scores, quantile))
    return int(np.argmin(np.abs(scores - target)))


def weighted_spatial_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    spatial = np.asarray(weights, dtype=np.float64)
    represented = spatial > 0.0
    safe = np.where(represented, field, 0.0)
    numerator = np.sum(safe * spatial, axis=(-2, -1), dtype=np.float64)
    denominator = np.sum(spatial, axis=(-2, -1), dtype=np.float64)
    if np.any(denominator <= 0.0):
        raise ValueError("spatial mean has no represented cells")
    return numerator / denominator


def jjas_midpoint_mask(target_dates: np.ndarray) -> np.ndarray:
    # Forecast windows use [init, init+7), while end-labelled IMD verification
    # uses init+1...+7.  The physical midpoint is therefore init+3.5 days; its
    # calendar date is the third IMD target label, not the fourth.
    midpoint = np.asarray(target_dates, dtype="datetime64[D]")[:, :, 2]
    months = pd.DatetimeIndex(midpoint.reshape(-1)).month.to_numpy().reshape(
        midpoint.shape
    )
    return np.isin(months, (6, 7, 8, 9))


def _map_field(
    axis: plt.Axes,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    boundary: tuple[np.ndarray, ...],
    *,
    limit: float,
    title: str,
) -> Any:
    masked = np.ma.masked_where(~support, field)
    image = axis.pcolormesh(
        longitude,
        latitude,
        masked,
        shading="nearest",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True,
    )
    axis.add_collection(
        LineCollection(boundary, colors="#263A49", linewidths=0.30, alpha=0.78)
    )
    axis.set_xlim(66.5, 98.5)
    axis.set_ylim(6.0, 38.5)
    axis.set_aspect("equal")
    axis.set_xticks((70, 80, 90))
    axis.set_yticks((10, 20, 30))
    axis.tick_params(labelsize=7, length=2)
    axis.grid(color="#D9E2E8", linewidth=0.35, alpha=0.6)
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold")
    return image


def _rain_field(
    axis: plt.Axes,
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    support: np.ndarray,
    boundary: tuple[np.ndarray, ...],
    *,
    limit: float,
    title: str,
) -> Any:
    image = axis.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_where(~support, field),
        shading="nearest",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=limit,
        rasterized=True,
    )
    axis.add_collection(
        LineCollection(boundary, colors="#263A49", linewidths=0.25, alpha=0.72)
    )
    axis.set_xlim(66.5, 98.5)
    axis.set_ylim(6.0, 38.5)
    axis.set_aspect("equal")
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_title(title, fontsize=7.6)
    return image


def render_figure(
    *,
    output: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    geometric_support: np.ndarray,
    boundary: tuple[np.ndarray, ...],
    pooled_crps_delta: np.ndarray,
    pooled_mae_delta: np.ndarray,
    case_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    figure = plt.figure(figsize=(7.05, 7.4), facecolor="white")
    grid = figure.add_gridspec(
        3, 2, height_ratios=(1.0, 0.83, 0.83), hspace=0.30, wspace=0.18
    )
    aggregate_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    crps_limit = max(float(np.nanpercentile(np.abs(pooled_crps_delta[geometric_support]), 97)), 0.05)
    mae_limit = max(float(np.nanpercentile(np.abs(pooled_mae_delta[geometric_support]), 97)), 0.05)
    crps_image = _map_field(
        aggregate_axes[0], pooled_crps_delta, latitude, longitude, geometric_support,
        boundary, limit=crps_limit, title="a  Continuous CRPS reduction",
    )
    mae_image = _map_field(
        aggregate_axes[1], pooled_mae_delta, latitude, longitude, geometric_support,
        boundary, limit=mae_limit, title="b  Deterministic MAE reduction",
    )
    aggregate_axes[1].annotate(
        "Uttarakhand\n($>+3.5$ mm/d)",
        xy=(79.5, 31.0),
        xytext=(83.5, 34.5),
        arrowprops=dict(arrowstyle="->", color="#8B0000", lw=0.8),
        fontsize=6.2,
        fontweight="bold",
        color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.90, edgecolor="#8B0000", lw=0.4),
    )
    aggregate_axes[1].annotate(
        "Western Ghats\n($>+2.5$ mm/d)",
        xy=(74.0, 16.5),
        xytext=(66.5, 12.5),
        arrowprops=dict(arrowstyle="->", color="#003366", lw=0.8),
        fontsize=6.2,
        fontweight="bold",
        color="#003366",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.90, edgecolor="#003366", lw=0.4),
    )
    for axis, image in zip(aggregate_axes, (crps_image, mae_image)):
        colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.06, fraction=0.05)
        colorbar.ax.tick_params(labelsize=6.5, length=2)
        colorbar.set_label("mm day$^{-1}$", fontsize=7)

    for row_index, record in enumerate(case_rows, start=1):
        subgrid = grid[row_index, :].subgridspec(1, 4, wspace=0.08)
        axes = [figure.add_subplot(subgrid[0, column]) for column in range(4)]
        rain_limit = max(
            float(np.nanpercentile(record["truth"][record["support"]], 98)),
            float(np.nanpercentile(record["raw_mean"][record["support"]], 98)),
            float(np.nanpercentile(record["corrected_mean"][record["support"]], 98)),
            1.0,
        )
        rain_images = []
        for axis, field, title in zip(
            axes[:3],
            (record["truth"], record["raw_mean"], record["corrected_mean"]),
            ("IMD Observed", "Raw FuXi", "Calibrated Adapter"),
        ):
            rain_images.append(
                _rain_field(
                    axis, field, latitude, longitude, record["support"], boundary,
                    limit=rain_limit, title=f"{title} [0–{rain_limit:.0f}]",
                )
            )
        local_limit = max(
            float(np.nanpercentile(np.abs(record["crps_delta"])[record["support"]], 97)),
            0.05,
        )
        delta_image = _map_field(
            axes[3], record["crps_delta"], latitude, longitude, record["support"],
            boundary, limit=local_limit, title="Local CRPS reduction",
        )
        axes[3].set_xticks(())
        axes[3].set_yticks(())
        label = "Representative (median)" if record["quantile"] == 0.5 else "Lower-tail (10th percentile)"
        figure.text(
            0.012,
            0.475 if row_index == 1 else 0.195,
            f"{chr(98 + row_index)}  {label}\n{record['initialization']} · W{record['lead']}\n"
            f"paired $\\Delta$CRPS={record['score']:+.3f}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="#243746",
        )
        cbar_delta = figure.colorbar(
            delta_image, ax=axes[3], orientation="horizontal", pad=0.025, fraction=0.045
        )
        cbar_delta.ax.tick_params(labelsize=6, length=2)

    figure.suptitle(
        "Where Neural Adapters Improve JJAS Rainfall Forecasts over India",
        y=0.986,
        fontsize=13.2,
        fontweight="bold",
        color="#172B3A",
    )
    figure.text(
        0.5,
        0.006,
        "2020–2024 valid-midpoint JJAS; exact +1…+42-day IMD alignment. "
        "Rainfall panel brackets and all reduction scales use mm day$^{-1}$. Maps are descriptive "
        "(no pixelwise significance claim); state borders are for orientation.",
        ha="center",
        fontsize=6.8,
        color="#50616E",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.915, bottom=0.035)
    png = output / "fig03_spatial_improvement_and_cases.png"
    pdf = output / "fig03_spatial_improvement_and_cases.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    root = (PROJECT_ROOT / "resultsv3/india_s2s_spatial_appendix").resolve()
    if root not in output.parents:
        raise ValueError("output must be a fresh child of resultsv3/india_s2s_spatial_appendix")
    if output.exists():
        raise FileExistsError(output)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        ledger = audit.HashLedger()
        contracts = audit.load_contracts(
            inference_manifest=args.inference_manifest,
            scoring_manifest=args.scoring_manifest,
            parent_manifest=args.parent_manifest,
            preflight_manifest=args.preflight_manifest,
            ledger=ledger,
        )
        loader = bridge.build_benchmark_loader()
        truth = audit.load_weekly_truth(contracts, loader, ledger)
        midpoint_mask = jjas_midpoint_mask(truth.target_dates)
        if not np.array_equal(midpoint_mask.sum(axis=0), np.full(6, 169)):
            raise ValueError("valid-midpoint JJAS count is not 169 per lead")

        all_india = np.asarray(truth.region_weights["all_india"], dtype=np.float64)
        temporal_weight = np.zeros((6, 27, 27), dtype=np.float64)
        crps_delta_sum = np.zeros_like(temporal_weight)
        mae_delta_sum = np.zeros_like(temporal_weight)
        candidates: list[dict[str, Any]] = []
        score_offset = 0
        raw_receipts = audit._raw_receipt_by_year(contracts)
        for year in audit.YEARS:
            shard = contracts.shard_arrays[year]
            score_count = EXPECTED_SCORE_COUNTS[year]
            starts = np.asarray(shard["initializations"][:score_count], dtype="datetime64[D]")
            if not np.array_equal(starts, contracts.score_starts[score_offset : score_offset + score_count]):
                raise ValueError(f"{year} scoreable cohort is not the expected prefix")
            dataset = audit._open_forecast(loader, year)
            digest = hashlib.sha256()
            try:
                variable = dataset["forecast_weekly_mean"]
                for begin in range(0, len(shard["initializations"]), args.batch_size):
                    end = min(begin + args.batch_size, len(shard["initializations"]))
                    raw_all = np.asarray(variable.isel(init=slice(begin, end)).load().values, dtype=np.float32)
                    audit._raw_content_update(digest, raw_all)
                    if begin >= score_count:
                        continue
                    usable = min(end, score_count) - begin
                    raw = raw_all[:usable]
                    corrected = audit.reconstruct_neural_members(
                        raw,
                        shard["delta_log_location"][begin : begin + usable],
                        shard["log_spread"][begin : begin + usable],
                    )
                    global_slice = slice(score_offset + begin, score_offset + begin + usable)
                    observed = np.asarray(truth.values[global_slice], dtype=np.float32)
                    obs_support = np.asarray(truth.support[global_slice], dtype=np.float64)
                    raw_crps = empirical_crps(raw, observed)
                    corrected_crps = empirical_crps(corrected, observed)
                    raw_mean = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
                    corrected_mean = corrected.mean(axis=1, dtype=np.float64).astype(np.float32)
                    crps_delta = raw_crps - corrected_crps
                    mae_delta = np.abs(raw_mean - observed) - np.abs(corrected_mean - observed)
                    selected = midpoint_mask[global_slice]
                    for lead in range(6):
                        choose = selected[:, lead]
                        if not np.any(choose):
                            continue
                        weights = obs_support[choose, lead]
                        temporal_weight[lead] += np.sum(weights, axis=0, dtype=np.float64)
                        crps_delta_sum[lead] += np.sum(
                            crps_delta[choose, lead] * weights, axis=0, dtype=np.float64
                        )
                        mae_delta_sum[lead] += np.sum(
                            mae_delta[choose, lead] * weights, axis=0, dtype=np.float64
                        )
                    local_starts = np.asarray(shard["initializations"][begin : begin + usable], dtype="datetime64[D]")
                    for case_index, lead_index in np.argwhere(selected):
                        weights = all_india * obs_support[case_index, lead_index]
                        score = float(weighted_spatial_mean(crps_delta[case_index, lead_index], weights))
                        mae_score = float(weighted_spatial_mean(mae_delta[case_index, lead_index], weights))
                        candidates.append(
                            {
                                "initialization": np.datetime_as_string(local_starts[case_index], unit="D"),
                                "lead": int(lead_index + 1),
                                "score": score,
                                "mae_score": mae_score,
                                "truth": observed[case_index, lead_index].copy(),
                                "raw_mean": raw_mean[case_index, lead_index].copy(),
                                "corrected_mean": corrected_mean[case_index, lead_index].copy(),
                                "crps_delta": crps_delta[case_index, lead_index].copy(),
                                "support": weights > 0.0,
                            }
                        )
            finally:
                dataset.close()
            if digest.hexdigest() != str(raw_receipts[year]["raw_weekly_members_sha256"]):
                raise ValueError(f"{year} raw-member content digest differs")
            score_offset += score_count
        if score_offset != 505 or len(candidates) != 1014:
            raise ValueError("spatial pass did not process the frozen JJAS cohort")

        local_crps = np.divide(
            crps_delta_sum,
            temporal_weight,
            out=np.full_like(crps_delta_sum, np.nan),
            where=temporal_weight > 0.0,
        )
        local_mae = np.divide(
            mae_delta_sum,
            temporal_weight,
            out=np.full_like(mae_delta_sum, np.nan),
            where=temporal_weight > 0.0,
        )
        pooled_weight = temporal_weight.sum(axis=0)
        pooled_crps = np.divide(
            crps_delta_sum.sum(axis=0), pooled_weight,
            out=np.full((27, 27), np.nan), where=pooled_weight > 0.0,
        )
        pooled_mae = np.divide(
            mae_delta_sum.sum(axis=0), pooled_weight,
            out=np.full((27, 27), np.nan), where=pooled_weight > 0.0,
        )
        geometric_support = (all_india > 0.0) & (pooled_weight > 0.0)
        case_scores = np.asarray([row["score"] for row in candidates], dtype=np.float64)
        selected_rows: list[dict[str, Any]] = []
        for quantile in CASE_QUANTILES:
            chosen = dict(candidates[select_nearest_quantile(case_scores, quantile)])
            chosen["quantile"] = quantile
            selected_rows.append(chosen)

        boundary, boundary_receipt = load_india_boundary(args.india_boundary)
        figure_dir = staging / "figures"
        table_dir = staging / "tables"
        figure_dir.mkdir()
        table_dir.mkdir()
        png, pdf = render_figure(
            output=figure_dir,
            latitude=contracts.latitude,
            longitude=contracts.longitude,
            geometric_support=geometric_support,
            boundary=boundary,
            pooled_crps_delta=pooled_crps,
            pooled_mae_delta=pooled_mae,
            case_rows=selected_rows,
        )

        rows = []
        for lead in range(6):
            represented = temporal_weight[lead] > 0.0
            area = np.where(represented, all_india, 0.0)
            lead_candidates = [row for row in candidates if row["lead"] == lead + 1]
            if len(lead_candidates) != 169:
                raise ValueError(f"Week {lead + 1} does not contain 169 case metrics")
            rows.append(
                {
                    "lead_week": lead + 1,
                    "case_count": int(midpoint_mask[:, lead].sum()),
                    "area_fraction_crps_improved": float(np.sum(area[local_crps[lead] > 0.0]) / np.sum(area)),
                    "area_fraction_mae_improved": float(np.sum(area[local_mae[lead] > 0.0]) / np.sum(area)),
                    "case_mean_area_weighted_crps_reduction": float(
                        np.mean([row["score"] for row in lead_candidates])
                    ),
                    "case_mean_area_weighted_mae_reduction": float(
                        np.mean([row["mae_score"] for row in lead_candidates])
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(table_dir / "leadwise_spatial_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "selection_quantile": row["quantile"],
                    "initialization": row["initialization"],
                    "lead_week": row["lead"],
                    "paired_crps_reduction_mm_day": row["score"],
                    "candidate_count": len(candidates),
                }
                for row in selected_rows
            ]
        ).to_csv(table_dir / "selected_case_diagnostics.csv", index=False)
        np.savez_compressed(
            table_dir / "spatial_fields.npz",
            latitude=contracts.latitude,
            longitude=contracts.longitude,
            pooled_crps_reduction=pooled_crps,
            pooled_mae_reduction=pooled_mae,
            leadwise_crps_reduction=local_crps,
            leadwise_mae_reduction=local_mae,
            temporal_support=temporal_weight,
            all_india_area_weight_km2=all_india,
        )
        source_copy = staging / "code/evaluate" / Path(__file__).name
        source_copy.parent.mkdir(parents=True)
        shutil.copy2(Path(__file__), source_copy)
        ledger.reverify()
        artifacts = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "experiment": "india_s2s_indra_spatial_appendix_v1",
            "status": "complete_descriptive_appendix_evidence",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract": {
                "forecast_years": list(audit.YEARS),
                "valid_midpoint_season": "JJAS",
                "case_count_per_lead": 169,
                "case_lead_count": len(candidates),
                "target_day_offsets": list(audit.TARGET_OFFSETS),
                "member_count": 50,
                "selected_seed": 43,
                "opened_2025_observation": False,
                "sealed_2025_target_opened": False,
            },
            "interpretation": {
                "positive_map_values": "lower INDRA error or CRPS than raw FuXi",
                "acc_map_omitted": (
                    "Paper ACC is a per-case spatial correlation and has no cellwise map; "
                    "rendering a grid-cell ACC improvement would change the metric definition."
                ),
                "pixelwise_inference": False,
                "state_boundaries": "display orientation only; no state-level inference",
                "case_panels": (
                    "Outcome-ranked descriptive diagnostics selected mechanically at the median "
                    "and 10th percentile of paired all-India CRPS reduction; not independent case studies."
                ),
            },
            "selection": [
                {
                    "quantile": row["quantile"],
                    "initialization": row["initialization"],
                    "lead_week": row["lead"],
                    "paired_crps_reduction_mm_day": row["score"],
                }
                for row in selected_rows
            ],
            "inputs": {
                "inference_manifest": str(Path(args.inference_manifest).resolve()),
                "inference_manifest_sha256": sha256_file(args.inference_manifest),
                "scoring_manifest": str(Path(args.scoring_manifest).resolve()),
                "scoring_manifest_sha256": sha256_file(args.scoring_manifest),
                "parent_manifest": str(Path(args.parent_manifest).resolve()),
                "parent_manifest_sha256": sha256_file(args.parent_manifest),
                "preflight_manifest": str(Path(args.preflight_manifest).resolve()),
                "preflight_manifest_sha256": sha256_file(args.preflight_manifest),
                "india_boundary": boundary_receipt,
            },
            "artifact_sha256": artifacts,
        }
        atomic_json(staging / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--scoring-manifest", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--preflight-manifest", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--india-boundary", type=Path, default=DEFAULT_INDIA_BOUNDARY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = run(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
