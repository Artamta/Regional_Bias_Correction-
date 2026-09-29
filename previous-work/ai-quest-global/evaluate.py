#!/usr/bin/env python3
"""Offline RPS/RPSS comparison for uniform, FuXi-p0, and model forecasts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .predict import (
        FUXI_PERMISSION_WARNING,
        LATITUDE,
        LONGITUDE,
        N_LATITUDE,
        N_LEADS,
        N_LONGITUDE,
        N_QUINTILES,
        NoMatchingPreparedCases,
        is_zarr_store,
        load_checkpoint_model,
        load_prepared,
        run_model,
        validate_probability_cube,
    )
except (ImportError, ValueError):
    from predict import (  # type: ignore
        FUXI_PERMISSION_WARNING,
        LATITUDE,
        LONGITUDE,
        N_LATITUDE,
        N_LEADS,
        N_LONGITUDE,
        N_QUINTILES,
        NoMatchingPreparedCases,
        is_zarr_store,
        load_checkpoint_model,
        load_prepared,
        run_model,
        validate_probability_cube,
    )


SYSTEMS = ("uniform", "p0", "model")
RELIABILITY_SYSTEMS = ("p0", "model")
RELIABILITY_BIN_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _rps_field(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the four-threshold RPS at every grid cell."""

    probs = validate_probability_cube(probabilities)
    truth = np.asarray(target)
    expected_grid = (N_LATITUDE, N_LONGITUDE)
    if truth.shape != expected_grid:
        raise ValueError(f"target must have shape {expected_grid}; got {truth.shape}")
    forecast_cdf = np.cumsum(np.asarray(probs, dtype=np.float64), axis=0)[:-1]
    thresholds = np.arange(N_QUINTILES - 1, dtype=np.int16)[:, None, None]
    observed_cdf = truth[None, :, :] <= thresholds
    return np.sum((forecast_cdf - observed_cdf) ** 2, axis=0)


def _weighted_rps_components(
    rps_field: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> tuple[float, float, int]:
    """Aggregate an already-computed RPS field over valid weighted cells."""

    truth = np.asarray(target)
    weights = np.asarray(spatial_weight, dtype=np.float64)
    expected_grid = (N_LATITUDE, N_LONGITUDE)
    if np.asarray(rps_field).shape != expected_grid:
        raise ValueError(f"rps_field must have shape {expected_grid}")
    if truth.shape != expected_grid:
        raise ValueError(f"target must have shape {expected_grid}; got {truth.shape}")
    if weights.shape != expected_grid:
        raise ValueError(f"spatial weights must have shape {expected_grid}; got {weights.shape}")

    valid = (
        (truth >= 0)
        & (truth < N_QUINTILES)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    count = int(valid.sum())
    if count == 0:
        return 0.0, 0.0, 0
    denominator = float(weights[valid].sum())
    numerator = float(np.sum(np.asarray(rps_field)[valid] * weights[valid]))
    return numerator, denominator, count


def _rps_components(
    probabilities: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> tuple[float, float, int]:
    """Return weighted RPS numerator, denominator, and valid-cell count.

    RPS is the sum over the four non-trivial cumulative thresholds.  This is
    the same convention used by ``train.rps_loss``; the fifth cumulative term
    is identically zero and is therefore omitted here.
    """

    return _weighted_rps_components(
        _rps_field(probabilities, target), target, spatial_weight
    )


def _accumulate_reliability(
    probability_sum: np.ndarray,
    observation_sum: np.ndarray,
    weight_sum: np.ndarray,
    *,
    lead_index: int,
    system_index: int,
    probabilities: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> None:
    """Accumulate area-weighted multicategory reliability-bin statistics."""

    truth = np.asarray(target)
    weights = np.asarray(spatial_weight, dtype=np.float64)
    valid = (
        (truth >= 0)
        & (truth < N_QUINTILES)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return
    valid_truth = truth[valid]
    valid_weight = weights[valid]
    n_bins = len(RELIABILITY_BIN_EDGES) - 1
    for category in range(N_QUINTILES):
        forecast = np.asarray(probabilities[category], dtype=np.float64)[valid]
        bins = np.searchsorted(RELIABILITY_BIN_EDGES, forecast, side="right") - 1
        bins = np.clip(bins, 0, n_bins - 1)
        observed = valid_truth == category
        index = (lead_index, system_index)
        probability_sum[index] += np.bincount(
            bins, weights=valid_weight * forecast, minlength=n_bins
        )
        observation_sum[index] += np.bincount(
            bins, weights=valid_weight * observed, minlength=n_bins
        )
        weight_sum[index] += np.bincount(
            bins, weights=valid_weight, minlength=n_bins
        )


def _base_spatial_weight(
    *,
    weighting: str,
    land_fraction: np.ndarray | None,
) -> np.ndarray:
    area = np.clip(np.cos(np.deg2rad(LATITUDE.astype(np.float64))), 0.0, None)[:, None]
    weights = np.broadcast_to(area, (N_LATITUDE, N_LONGITUDE)).copy()
    if weighting == "area-land":
        if land_fraction is None:
            raise ValueError("--weighting area-land requires land_fraction in every NPZ")
        # Match the training and official precipitation scoring mask rather
        # than treating coastal land fraction as a fractional weight.
        weights *= np.asarray(land_fraction, dtype=np.float64) >= 0.5
    elif weighting != "area":
        raise ValueError(f"Unknown weighting mode: {weighting}")
    return weights


def _resolve_case_files(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if is_zarr_store(path):
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.glob("*.npz")))
            paths.extend(sorted(candidate for candidate in path.glob("*.zarr") if is_zarr_store(candidate)))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Evaluation input does not exist: {path}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("No prepared NPZ files or Zarr stores were found")
    return unique


def _cache_contract_fingerprint(path: Path) -> str | None:
    """Return a prepared artifact's scientific-contract fingerprint, if present."""

    if path.suffix.lower() == ".npz" and path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if "cache_contract_sha256" not in archive:
                return None
            return str(np.asarray(archive["cache_contract_sha256"]).item())
    if is_zarr_store(path):
        try:
            import zarr
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("zarr is required to verify cache provenance") from error
        group = zarr.open_group(str(path), mode="r")
        value = group.attrs.get("cache_contract_sha256")
        return None if value in (None, "") else str(value)
    return None


def _checkpoint_contract_fingerprint(payload: Mapping[str, Any]) -> str | None:
    normalization = payload.get("normalization", {})
    if not isinstance(normalization, Mapping):
        return None
    value = normalization.get("cache_contract_sha256")
    return None if value in (None, "") else str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    note: str | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("matplotlib is required to write the evaluation PNG") from exc

    lookup = {(int(row["lead"]), str(row["system"])): row for row in rows}
    colors = {"uniform": "#777777", "p0": "#cc79a7", "model": "#009e73"}
    labels = {"uniform": "Uniform", "p0": "FuXi p0", "model": "Model"}
    x = np.arange(N_LEADS, dtype=np.float64)
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for system_index, system in enumerate(SYSTEMS):
        offset = (system_index - 1) * width
        rps = [float(lookup[(lead, system)]["rps"]) for lead in range(1, N_LEADS + 1)]
        rpss = [float(lookup[(lead, system)]["rpss"]) for lead in range(1, N_LEADS + 1)]
        axes[0].bar(x + offset, rps, width, label=labels[system], color=colors[system])
        axes[1].bar(x + offset, rpss, width, label=labels[system], color=colors[system])

    for axis in axes:
        axis.set_xticks(x, [f"Lead {lead}" for lead in range(1, N_LEADS + 1)])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Area-weighted RPS (lower is better)")
    axes[0].set_title("Ranked Probability Score")
    axes[0].set_ylim(bottom=0.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("RPSS vs uniform (higher is better)")
    axes[1].set_title("Ranked Probability Skill Score")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22))
    title = "Global precipitation quintile forecast evaluation"
    if note:
        title = f"{title}\n{note}"
    fig.suptitle(title, y=1.06 if note else 1.02)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(
    probability_sum: np.ndarray,
    observation_sum: np.ndarray,
    weight_sum: np.ndarray,
    destination: Path,
    *,
    note: str | None = None,
) -> None:
    """Plot category-pooled forecast probability against observed frequency."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("matplotlib is required to write reliability.png") from exc

    colors = {"p0": "#cc79a7", "model": "#009e73"}
    labels = {"p0": "FuXi p0", "model": "Model"}
    fig, axes = plt.subplots(
        1, N_LEADS, figsize=(4.5 * N_LEADS, 4.1), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for lead_index, axis in enumerate(axes):
        axis.plot([0.0, 1.0], [0.0, 1.0], color="0.45", linestyle="--", linewidth=1.0)
        for system_index, system in enumerate(RELIABILITY_SYSTEMS):
            denominator = weight_sum[lead_index, system_index]
            valid = denominator > 0.0
            mean_probability = np.divide(
                probability_sum[lead_index, system_index],
                denominator,
                out=np.full_like(denominator, np.nan),
                where=valid,
            )
            observed_frequency = np.divide(
                observation_sum[lead_index, system_index],
                denominator,
                out=np.full_like(denominator, np.nan),
                where=valid,
            )
            axis.plot(
                mean_probability[valid],
                observed_frequency[valid],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=colors[system],
                label=labels[system],
            )
        axis.set(
            title=f"Lead {lead_index + 1}",
            xlabel="Mean forecast probability",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Observed category frequency")
    axes[-1].legend(frameon=False, loc="lower right")
    title = "Global precipitation reliability (five categories pooled)"
    if note:
        title = f"{title}\n{note}"
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_spatial_rps_improvement(
    improvement_sum: np.ndarray,
    improvement_count: np.ndarray,
    destination: Path,
    *,
    note: str | None = None,
) -> None:
    """Plot case-mean FuXi-p0 RPS minus model RPS for each lead."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib is required to write spatial_rps_improvement.png"
        ) from exc

    mean_improvement = np.divide(
        improvement_sum,
        improvement_count,
        out=np.full_like(improvement_sum, np.nan, dtype=np.float64),
        where=improvement_count > 0,
    )
    finite = np.isfinite(mean_improvement)
    color_limit = (
        float(np.nanpercentile(np.abs(mean_improvement[finite]), 98.0))
        if np.any(finite)
        else 0.0
    )
    color_limit = max(color_limit, 1.0e-6)

    fig, axes = plt.subplots(
        N_LEADS, 1, figsize=(10.2, 3.2 * N_LEADS), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for lead_index, axis in enumerate(axes):
        field = np.ma.masked_where(
            improvement_count[lead_index] == 0,
            mean_improvement[lead_index],
        )
        mesh = axis.pcolormesh(
            LONGITUDE,
            LATITUDE,
            field,
            shading="auto",
            cmap="PiYG",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set(
            title=f"Lead {lead_index + 1}",
            xlabel="Longitude (degrees east)",
            ylabel="Latitude",
            xlim=(0.0, 358.5),
            ylim=(-90.0, 90.0),
        )
        axis.grid(alpha=0.15)
    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.88, pad=0.02)
    colorbar.set_label("FuXi p0 RPS - model RPS (positive = model better)")
    title = "Case-mean spatial RPS improvement"
    if note:
        title = f"{title}\n{note}"
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_cases(
    case_files: Iterable[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1,
    weighting: str = "area-land",
    years: Sequence[int] | None = None,
    thursday_only: bool = False,
    plot_note: str | None = None,
) -> tuple[Path, Path, Path]:
    """Evaluate all cases and write per-case CSV, summary CSV, and summary PNG."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    paths = _resolve_case_files(case_files)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    model: Any | None = None
    model_channels: int | None = None
    checkpoint_fingerprint: str | None = None
    case_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[int, str], list[float]] = {
        (lead, system): [0.0, 0.0, 0.0]
        for lead in range(1, N_LEADS + 1)
        for system in SYSTEMS
    }
    uniform = np.full(
        (N_QUINTILES, N_LATITUDE, N_LONGITUDE),
        1.0 / N_QUINTILES,
        dtype=np.float32,
    )
    n_reliability_bins = len(RELIABILITY_BIN_EDGES) - 1
    reliability_probability_sum = np.zeros(
        (N_LEADS, len(RELIABILITY_SYSTEMS), n_reliability_bins), dtype=np.float64
    )
    reliability_observation_sum = np.zeros_like(reliability_probability_sum)
    reliability_weight_sum = np.zeros_like(reliability_probability_sum)
    spatial_improvement_sum = np.zeros(
        (N_LEADS, N_LATITUDE, N_LONGITUDE), dtype=np.float64
    )
    spatial_improvement_count = np.zeros(
        (N_LEADS, N_LATITUDE, N_LONGITUDE), dtype=np.int32
    )
    india_mask = (
        (LATITUDE[:, None] >= 5.0)
        & (LATITUDE[:, None] <= 40.0)
        & (LONGITUDE[None, :] >= 65.0)
        & (LONGITUDE[None, :] <= 100.0)
    )
    india_aggregate: dict[tuple[int, str], list[float]] = {
        (lead, system): [0.0, 0.0, 0.0]
        for lead in range(1, N_LEADS + 1)
        for system in RELIABILITY_SYSTEMS
    }

    for path in paths:
        source_fingerprint = _cache_contract_fingerprint(path)
        try:
            prepared = load_prepared(
                path,
                require_target=True,
                years=years,
                thursday_only=thursday_only,
            )
        except NoMatchingPreparedCases:
            continue
        assert prepared.target is not None
        in_channels = int(prepared.features.shape[2])
        if model is None:
            model, checkpoint_payload = load_checkpoint_model(
                checkpoint_path,
                device=device,
                in_channels=in_channels,
            )
            model_channels = in_channels
            checkpoint_fingerprint = _checkpoint_contract_fingerprint(
                checkpoint_payload
            )
        elif in_channels != model_channels:
            raise ValueError(
                f"All files must use {model_channels} feature channels; {path} uses {in_channels}"
            )
        if source_fingerprint is not None or checkpoint_fingerprint is not None:
            if source_fingerprint is None or checkpoint_fingerprint is None:
                raise ValueError(
                    "Checkpoint/cache provenance mismatch: both artifacts must "
                    "declare cache_contract_sha256"
                )
            if source_fingerprint != checkpoint_fingerprint:
                raise ValueError(
                    "Checkpoint/cache provenance mismatch: cache_contract_sha256 differs"
                )

        model_probabilities = np.empty_like(prepared.p0)
        for start in range(0, prepared.n_cases, batch_size):
            stop = min(start + batch_size, prepared.n_cases)
            model_probabilities[start:stop] = run_model(
                model,
                prepared.features[start:stop],
                prepared.p0[start:stop],
                device=device,
            )
        weights = _base_spatial_weight(
            weighting=weighting,
            land_fraction=prepared.land_fraction,
        )

        for case_index in range(prepared.n_cases):
            init_date = prepared.init_dates[case_index]
            for lead_index in range(N_LEADS):
                lead = lead_index + 1
                target = prepared.target[case_index, lead_index]
                forecasts = {
                    "uniform": uniform,
                    "p0": prepared.p0[case_index, lead_index],
                    "model": model_probabilities[case_index, lead_index],
                }
                rps_fields = {
                    system: _rps_field(probabilities, target)
                    for system, probabilities in forecasts.items()
                }
                components = {
                    system: _weighted_rps_components(field, target, weights)
                    for system, field in rps_fields.items()
                }
                reference_num, reference_den, _ = components["uniform"]
                if reference_den <= 0.0:
                    raise ValueError(
                        f"No valid evaluation cells for {path}, case {case_index}, lead {lead}"
                    )
                reference_rps = reference_num / reference_den

                for system in SYSTEMS:
                    numerator, denominator, valid_cells = components[system]
                    rps = numerator / denominator
                    rpss = 1.0 - (rps / reference_rps) if reference_rps > 0.0 else float("nan")
                    case_rows.append(
                        {
                            "case_file": str(path),
                            "case_index": case_index,
                            "init_date": init_date,
                            "lead": lead,
                            "system": system,
                            "rps": rps,
                            "rpss": rpss,
                            "valid_cells": valid_cells,
                            "valid_weight": denominator,
                            "weighting": weighting,
                        }
                    )
                    totals = aggregate[(lead, system)]
                    totals[0] += numerator
                    totals[1] += denominator
                    totals[2] += 1.0

                valid = (
                    (target >= 0)
                    & (target < N_QUINTILES)
                    & np.isfinite(weights)
                    & (weights > 0.0)
                )
                improvement = rps_fields["p0"] - rps_fields["model"]
                lead_improvement_sum = spatial_improvement_sum[lead_index]
                lead_improvement_count = spatial_improvement_count[lead_index]
                lead_improvement_sum[valid] += improvement[valid]
                lead_improvement_count[valid] += 1

                for system_index, system in enumerate(RELIABILITY_SYSTEMS):
                    _accumulate_reliability(
                        reliability_probability_sum,
                        reliability_observation_sum,
                        reliability_weight_sum,
                        lead_index=lead_index,
                        system_index=system_index,
                        probabilities=forecasts[system],
                        target=target,
                        spatial_weight=weights,
                    )

                india_valid = valid & india_mask
                if np.any(india_valid):
                    india_denominator = float(weights[india_valid].sum())
                    for system in RELIABILITY_SYSTEMS:
                        india_numerator = float(
                            np.sum(rps_fields[system][india_valid] * weights[india_valid])
                        )
                        totals = india_aggregate[(lead, system)]
                        totals[0] += india_numerator
                        totals[1] += india_denominator
                        totals[2] += 1.0

    if not case_rows:
        raise NoMatchingPreparedCases(
            f"No evaluation cases matched years={years}, thursday_only={thursday_only}"
        )

    summary_rows: list[dict[str, Any]] = []
    aggregate_rps = {
        key: values[0] / values[1] if values[1] > 0.0 else float("nan")
        for key, values in aggregate.items()
    }
    for lead in range(1, N_LEADS + 1):
        reference_rps = aggregate_rps[(lead, "uniform")]
        for system in SYSTEMS:
            numerator, denominator, n_cases = aggregate[(lead, system)]
            rps = aggregate_rps[(lead, system)]
            rpss = 1.0 - (rps / reference_rps) if reference_rps > 0.0 else float("nan")
            summary_rows.append(
                {
                    "lead": lead,
                    "system": system,
                    "rps": rps,
                    "rpss": rpss,
                    "valid_weight": denominator,
                    "n_cases": int(n_cases),
                    "weighting": weighting,
                }
            )

    india_rows: list[dict[str, Any]] = []
    for lead in range(1, N_LEADS + 1):
        raw_numerator, raw_denominator, n_cases = india_aggregate[(lead, "p0")]
        model_numerator, model_denominator, _ = india_aggregate[(lead, "model")]
        raw_rps = (
            raw_numerator / raw_denominator if raw_denominator > 0.0 else float("nan")
        )
        model_rps = (
            model_numerator / model_denominator
            if model_denominator > 0.0
            else float("nan")
        )
        model_vs_raw_skill = (
            1.0 - model_rps / raw_rps if np.isfinite(raw_rps) and raw_rps > 0.0 else float("nan")
        )
        india_rows.append(
            {
                "lead": lead,
                "raw_p0_rps": raw_rps,
                "model_rps": model_rps,
                "model_vs_raw_skill": model_vs_raw_skill,
                "valid_weight": raw_denominator,
                "n_cases": int(n_cases),
                "latitude_bounds": "5N-40N",
                "longitude_bounds": "65E-100E",
                "weighting": weighting,
            }
        )

    case_csv = destination / "evaluation_by_case.csv"
    summary_csv = destination / "evaluation_summary.csv"
    plot_png = destination / "evaluation_by_lead.png"
    reliability_png = destination / "reliability.png"
    spatial_improvement_png = destination / "spatial_rps_improvement.png"
    india_csv = destination / "india_summary.csv"
    _write_csv(
        case_csv,
        case_rows,
        (
            "case_file",
            "case_index",
            "init_date",
            "lead",
            "system",
            "rps",
            "rpss",
            "valid_cells",
            "valid_weight",
            "weighting",
        ),
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ("lead", "system", "rps", "rpss", "valid_weight", "n_cases", "weighting"),
    )
    _write_csv(
        india_csv,
        india_rows,
        (
            "lead",
            "raw_p0_rps",
            "model_rps",
            "model_vs_raw_skill",
            "valid_weight",
            "n_cases",
            "latitude_bounds",
            "longitude_bounds",
            "weighting",
        ),
    )
    _plot_summary(summary_rows, plot_png, note=plot_note)
    _plot_reliability(
        reliability_probability_sum,
        reliability_observation_sum,
        reliability_weight_sum,
        reliability_png,
        note=plot_note,
    )
    _plot_spatial_rps_improvement(
        spatial_improvement_sum,
        spatial_improvement_count,
        spatial_improvement_png,
        note=plot_note,
    )
    return case_csv, summary_csv, plot_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare uniform, FuXi p0, and model RPS/RPSS on prepared NPZ/Zarr cases.",
        epilog=FUXI_PERMISSION_WARNING,
    )
    parser.add_argument(
        "--cases",
        required=True,
        nargs="+",
        type=Path,
        help="Prepared NPZ files, project Zarr cache, or directories containing them",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local model checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path, help="CSV/PNG output directory")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--batch-size", default=1, type=int, help="Inference batch size")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021],
        help="Initialization years to evaluate (default: 2020 2021)",
    )
    parser.add_argument(
        "--thursday-only",
        action="store_true",
        help="Diagnostic only: retain literal calendar Thursdays (default: all dates)",
    )
    parser.add_argument(
        "--weighting",
        choices=("area", "area-land"),
        default="area-land",
        help="Cos(latitude) over land_fraction>=0.5 (official) or the full grid",
    )
    parser.add_argument(
        "--plot-note",
        help="Optional subtitle written onto every evaluation figure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(FUXI_PERMISSION_WARNING, file=sys.stderr)
    outputs = evaluate_cases(
        args.cases,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        weighting=args.weighting,
        years=args.years,
        thursday_only=args.thursday_only,
        plot_note=args.plot_note,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
