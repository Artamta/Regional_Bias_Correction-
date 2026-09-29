#!/usr/bin/env python3
"""Fit and evaluate a validation-only uniform-blend probability calibration.

The fitted forecast is

    P_cal = (1 - alpha) * Uniform(5) + alpha * P

where one scalar ``alpha`` is fitted independently for FuXi ``p0`` and the
model forecast.  The fit minimizes the pooled area-weighted RPS numerator over
the requested fit years.  Because the scoring denominator is independent of
``alpha``, this is also the exact minimizer of the aggregate RPS ratio.

This is an offline exploratory utility.  It does not submit forecasts and its
figure is deliberately labelled as non-official validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .evaluate import (
        _base_spatial_weight,
        _cache_contract_fingerprint,
        _checkpoint_contract_fingerprint,
        _resolve_case_files,
        _rps_components,
    )
    from .predict import (
        FUXI_PERMISSION_WARNING,
        N_LEADS,
        N_QUINTILES,
        NoMatchingPreparedCases,
        load_checkpoint_model,
        load_prepared,
        normalize_probabilities,
        run_model,
    )
except (ImportError, ValueError):
    from evaluate import (  # type: ignore
        _base_spatial_weight,
        _cache_contract_fingerprint,
        _checkpoint_contract_fingerprint,
        _resolve_case_files,
        _rps_components,
    )
    from predict import (  # type: ignore
        FUXI_PERMISSION_WARNING,
        N_LEADS,
        N_QUINTILES,
        NoMatchingPreparedCases,
        load_checkpoint_model,
        load_prepared,
        normalize_probabilities,
        run_model,
    )


SYSTEMS = ("uniform", "p0", "p0_calibrated", "model", "model_calibrated")
CALIBRATED_SYSTEMS = ("p0", "model")
DEFAULT_PLOT_NOTE = "Exploratory; not official AI Weather Quest validation"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RPSAggregate:
    """Pooled RPS totals using the repository's ratio-of-weighted-sums rule."""

    numerator: float
    denominator: float
    valid_cells: int

    @property
    def rps(self) -> float:
        if self.denominator <= 0.0:
            raise ValueError("RPS aggregate contains no valid weighted cells")
        return self.numerator / self.denominator


@dataclass(frozen=True)
class ForecastCase:
    """One prepared case with raw and model probability forecasts."""

    source: Path
    case_index: int
    init_date: str
    p0: np.ndarray
    model: np.ndarray
    target: np.ndarray
    spatial_weight: np.ndarray


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Paths written by :func:`run_calibration`."""

    calibration_json: Path
    summary_csv: Path
    per_case_csv: Path
    plot_png: Path

    @property
    def paths(self) -> tuple[Path, ...]:
        return (
            self.calibration_json,
            self.summary_csv,
            self.per_case_csv,
            self.plot_png,
        )


def _probability_axis(values: np.ndarray, category_axis: int) -> int:
    axis = category_axis if category_axis >= 0 else values.ndim + category_axis
    if axis < 0 or axis >= values.ndim:
        raise ValueError(f"invalid category axis {category_axis} for {values.shape}")
    if values.shape[axis] != N_QUINTILES:
        raise ValueError(
            f"probabilities need {N_QUINTILES} categories on axis {category_axis}"
        )
    return axis


def blend_probabilities(
    probabilities: np.ndarray,
    alpha: float,
    *,
    category_axis: int = -3,
) -> np.ndarray:
    """Return a finite non-negative normalized uniform blend.

    The input must already be a probability forecast (within the float16
    tolerance used by prepared caches).  For ``alpha < 1`` every returned
    category is strictly positive, including when the input contains zeros.
    """

    coefficient = float(alpha)
    if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    values = np.asarray(probabilities, dtype=np.float64)
    axis = _probability_axis(values, category_axis)
    if not np.isfinite(values).all() or float(values.min()) < -1.0e-7:
        raise ValueError("probabilities must be finite and non-negative")
    totals = values.sum(axis=axis)
    if float(np.max(np.abs(totals - 1.0))) > 2.0e-3:
        raise ValueError("input probabilities must sum to one over categories")
    normalized = normalize_probabilities(values, category_axis=axis)
    blended = (1.0 - coefficient) / N_QUINTILES + coefficient * normalized
    return normalize_probabilities(blended, category_axis=axis)


def _validated_scoring_arrays(
    probabilities: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and broadcast generic ``[case, lead, category, y, x]`` arrays."""

    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(target)
    if probs.ndim != 5 or probs.shape[2] != N_QUINTILES:
        raise ValueError("probabilities must be [case, lead, 5, latitude, longitude]")
    expected_target = probs.shape[:2] + probs.shape[-2:]
    if truth.shape != expected_target:
        raise ValueError(f"target must have shape {expected_target}; got {truth.shape}")
    if not np.isfinite(probs).all() or float(probs.min()) < -1.0e-7:
        raise ValueError("probabilities must be finite and non-negative")
    if float(np.max(np.abs(probs.sum(axis=2) - 1.0))) > 2.0e-3:
        raise ValueError("probabilities must sum to one over categories")
    if not np.isfinite(truth).all() or not np.equal(truth, np.round(truth)).all():
        raise ValueError("target must contain integer categories -1 or 0..4")

    raw_weight = np.asarray(spatial_weight, dtype=np.float64)
    if raw_weight.ndim == 2:
        raw_weight = raw_weight[None, None, :, :]
    elif raw_weight.ndim == 3:
        raw_weight = raw_weight[:, None, :, :]
    elif raw_weight.ndim != 4:
        raise ValueError(
            "spatial_weight must be [lat,lon], [case,lat,lon], or "
            "[case,lead,lat,lon]"
        )
    try:
        weights = np.broadcast_to(raw_weight, truth.shape)
    except ValueError as error:
        raise ValueError(
            f"spatial_weight shape {np.asarray(spatial_weight).shape} cannot broadcast "
            f"to target shape {truth.shape}"
        ) from error
    valid = (
        (truth >= 0)
        & (truth < N_QUINTILES)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        raise ValueError("data contain no valid weighted scoring cells")
    return probs, truth.astype(np.int8, copy=False), weights, valid


def aggregate_rps(
    probabilities: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> RPSAggregate:
    """Compute pooled RPS as one ratio of weighted sums, never a case mean."""

    probs, truth, weights, valid = _validated_scoring_arrays(
        probabilities, target, spatial_weight
    )
    forecast_cdf = np.cumsum(probs, axis=2)[:, :, :-1]
    thresholds = np.arange(N_QUINTILES - 1, dtype=np.int8)[None, None, :, None, None]
    observed_cdf = truth[:, :, None, :, :] <= thresholds
    cell_rps = np.sum((forecast_cdf - observed_cdf) ** 2, axis=2)
    combined_weight = np.where(valid, weights, 0.0)
    return RPSAggregate(
        numerator=float(np.sum(cell_rps * combined_weight)),
        denominator=float(np.sum(combined_weight)),
        valid_cells=int(valid.sum()),
    )


def fit_uniform_blend_alpha(
    probabilities: np.ndarray,
    target: np.ndarray,
    spatial_weight: np.ndarray,
) -> float:
    """Return the analytic pooled-RPS optimum, clamped to ``[0, 1]``.

    Let ``e`` be the uniform-CDF error and ``d`` the forecast CDF minus the
    uniform CDF.  The weighted RPS numerator is ``A*alpha**2 + 2*B*alpha + C``
    with ``A=sum(w*d**2)`` and ``B=sum(w*e*d)``.  Its unconstrained optimum is
    therefore ``-B/A``.  When the forecast equals uniform on every fitted cell,
    the objective is flat and the conservative representative ``alpha=0`` is
    returned.
    """

    probs, truth, weights, valid = _validated_scoring_arrays(
        probabilities, target, spatial_weight
    )
    uniform_cdf = (
        np.arange(1, N_QUINTILES, dtype=np.float64)[None, None, :, None, None]
        / N_QUINTILES
    )
    forecast_cdf = np.cumsum(probs, axis=2)[:, :, :-1]
    thresholds = np.arange(N_QUINTILES - 1, dtype=np.int8)[None, None, :, None, None]
    observed_cdf = truth[:, :, None, :, :] <= thresholds
    delta = forecast_cdf - uniform_cdf
    uniform_error = uniform_cdf - observed_cdf
    combined_weight = np.where(valid, weights, 0.0)[:, :, None, :, :]
    # Float32 encodes 0.2 inexactly.  Treat that representation of a uniform
    # forecast as the analytically flat objective instead of fitting noise.
    active_delta = np.abs(delta) * (combined_weight > 0.0)
    if float(active_delta.max()) <= 1.0e-7:
        return 0.0
    quadratic = float(np.sum(combined_weight * delta * delta))
    if quadratic <= np.finfo(np.float64).eps:
        return 0.0
    linear_half = float(np.sum(combined_weight * uniform_error * delta))
    return float(np.clip(-linear_half / quadratic, 0.0, 1.0))


def _canonical_years(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    years = tuple(sorted({int(value) for value in values}))
    if not years or any(year < 1 or year > 9999 for year in years):
        raise ValueError(f"{name} must contain valid calendar years")
    return years


def _check_artifact_contract(
    source_fingerprint: str | None,
    checkpoint_fingerprint: str | None,
) -> None:
    if source_fingerprint is None and checkpoint_fingerprint is None:
        return
    if source_fingerprint is None or checkpoint_fingerprint is None:
        raise ValueError(
            "Checkpoint/cache provenance mismatch: both artifacts must declare "
            "cache_contract_sha256"
        )
    if source_fingerprint != checkpoint_fingerprint:
        raise ValueError(
            "Checkpoint/cache provenance mismatch: cache_contract_sha256 differs"
        )


def _load_forecast_cases(
    paths: Sequence[Path],
    checkpoint_path: str | Path,
    years: Sequence[int],
    *,
    device: str,
    batch_size: int,
    weighting: str,
    thursday_only: bool,
    model: Any | None = None,
    model_channels: int | None = None,
    checkpoint_fingerprint: str | None = None,
) -> tuple[list[ForecastCase], Any, int, str | None, Mapping[str, Any]]:
    records: list[ForecastCase] = []
    checkpoint_payload: Mapping[str, Any] = {}
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
        channels = int(prepared.features.shape[2])
        if model is None:
            model, checkpoint_payload = load_checkpoint_model(
                checkpoint_path,
                device=device,
                in_channels=channels,
            )
            model_channels = channels
            checkpoint_fingerprint = _checkpoint_contract_fingerprint(
                checkpoint_payload
            )
        elif channels != model_channels:
            raise ValueError(
                f"All files must use {model_channels} feature channels; "
                f"{path} uses {channels}"
            )
        _check_artifact_contract(source_fingerprint, checkpoint_fingerprint)

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
            records.append(
                ForecastCase(
                    source=path,
                    case_index=case_index,
                    init_date=prepared.init_dates[case_index],
                    p0=prepared.p0[case_index],
                    model=model_probabilities[case_index],
                    target=prepared.target[case_index],
                    spatial_weight=weights,
                )
            )
    if not records:
        raise NoMatchingPreparedCases(
            f"No prepared cases matched years={tuple(years)}, "
            f"thursday_only={thursday_only}"
        )
    assert model is not None and model_channels is not None
    return records, model, model_channels, checkpoint_fingerprint, checkpoint_payload


def _fit_alphas(cases: Sequence[ForecastCase]) -> dict[str, float]:
    target = np.stack([case.target for case in cases], axis=0)
    weights = np.stack([case.spatial_weight for case in cases], axis=0)
    return {
        system: fit_uniform_blend_alpha(
            np.stack([getattr(case, system) for case in cases], axis=0),
            target,
            weights,
        )
        for system in CALIBRATED_SYSTEMS
    }


def _case_forecasts(
    case: ForecastCase,
    alphas: Mapping[str, float],
) -> dict[str, np.ndarray]:
    uniform = np.full_like(case.p0, 1.0 / N_QUINTILES, dtype=np.float32)
    return {
        "uniform": uniform,
        "p0": case.p0,
        "p0_calibrated": blend_probabilities(case.p0, alphas["p0"]),
        "model": case.model,
        "model_calibrated": blend_probabilities(case.model, alphas["model"]),
    }


def _relative_skill(score: float, raw_score: float) -> float | None:
    """Return skill versus raw, including the equal-perfect edge case."""

    if raw_score > 0.0:
        return 1.0 - score / raw_score
    if abs(score - raw_score) <= 1.0e-15:
        return 0.0
    return None


def _score_cases(
    cases: Sequence[ForecastCase],
    alphas: Mapping[str, float],
    *,
    split: str,
    weighting: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: dict[tuple[str, str], list[float]] = {
        (lead, system): [0.0, 0.0, 0.0]
        for lead in ("1", "2", "all")
        for system in SYSTEMS
    }
    case_rows: list[dict[str, Any]] = []
    alpha_by_system: Mapping[str, float | None] = {
        "uniform": 0.0,
        "p0": 1.0,
        "p0_calibrated": alphas["p0"],
        "model": 1.0,
        "model_calibrated": alphas["model"],
    }
    for case in cases:
        forecasts = _case_forecasts(case, alphas)
        for lead_index in range(N_LEADS):
            lead = str(lead_index + 1)
            target = case.target[lead_index]
            components = {
                system: _rps_components(
                    probabilities[lead_index], target, case.spatial_weight
                )
                for system, probabilities in forecasts.items()
            }
            uniform_num, uniform_den, _ = components["uniform"]
            uniform_rps = uniform_num / uniform_den
            raw_rps = {
                "p0": components["p0"][0] / components["p0"][1],
                "model": components["model"][0] / components["model"][1],
            }
            for system in SYSTEMS:
                numerator, denominator, valid_cells = components[system]
                rps = numerator / denominator
                if system == "p0_calibrated":
                    skill_vs_raw: float | None = _relative_skill(rps, raw_rps["p0"])
                elif system == "model_calibrated":
                    skill_vs_raw = _relative_skill(rps, raw_rps["model"])
                elif system in CALIBRATED_SYSTEMS:
                    skill_vs_raw = 0.0
                else:
                    skill_vs_raw = None
                case_rows.append(
                    {
                        "split": split,
                        "case_file": str(case.source),
                        "case_index": case.case_index,
                        "init_date": case.init_date,
                        "lead": lead,
                        "system": system,
                        "alpha": alpha_by_system[system],
                        "rps": rps,
                        "rpss_vs_uniform": 1.0 - rps / uniform_rps,
                        "skill_vs_raw": skill_vs_raw,
                        "valid_cells": valid_cells,
                        "valid_weight": denominator,
                        "weighting": weighting,
                    }
                )
                for aggregate_lead in (lead, "all"):
                    totals = aggregates[(aggregate_lead, system)]
                    totals[0] += numerator
                    totals[1] += denominator
                    totals[2] += 1.0

    summary_rows: list[dict[str, Any]] = []
    for lead in ("1", "2", "all"):
        uniform_values = aggregates[(lead, "uniform")]
        uniform_rps = uniform_values[0] / uniform_values[1]
        raw_scores = {
            system: aggregates[(lead, system)][0] / aggregates[(lead, system)][1]
            for system in CALIBRATED_SYSTEMS
        }
        for system in SYSTEMS:
            numerator, denominator, count = aggregates[(lead, system)]
            rps = numerator / denominator
            if system == "p0_calibrated":
                skill_vs_raw = _relative_skill(rps, raw_scores["p0"])
            elif system == "model_calibrated":
                skill_vs_raw = _relative_skill(rps, raw_scores["model"])
            elif system in CALIBRATED_SYSTEMS:
                skill_vs_raw = 0.0
            else:
                skill_vs_raw = None
            summary_rows.append(
                {
                    "split": split,
                    "lead": lead,
                    "system": system,
                    "alpha": alpha_by_system[system],
                    "rps": rps,
                    "rpss_vs_uniform": 1.0 - rps / uniform_rps,
                    "skill_vs_raw": skill_vs_raw,
                    "rps_numerator": numerator,
                    "valid_weight": denominator,
                    "n_case_leads": int(count),
                    "weighting": weighting,
                }
            )
    return case_rows, summary_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    fit_years: Sequence[int],
    evaluation_years: Sequence[int],
    weighting: str,
    alphas: Mapping[str, float],
    note: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("matplotlib is required to write the calibration PNG") from error

    lookup = {
        (str(row["split"]), str(row["system"])): row
        for row in rows
        if str(row["lead"]) == "all"
    }
    labels = {
        "uniform": "Uniform",
        "p0": "FuXi p0",
        "p0_calibrated": "p0 + uniform",
        "model": "Model",
        "model_calibrated": "Model + uniform",
    }
    colors = {
        "uniform": "#777777",
        "p0": "#cc79a7",
        "p0_calibrated": "#e6a7cf",
        "model": "#009e73",
        "model_calibrated": "#75c9ad",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True, constrained_layout=True)
    for axis, split, years in zip(
        axes,
        ("fit", "evaluation"),
        (fit_years, evaluation_years),
    ):
        x = np.arange(len(SYSTEMS))
        values = [float(lookup[(split, system)]["rps"]) for system in SYSTEMS]
        bars = axis.bar(x, values, color=[colors[system] for system in SYSTEMS])
        for bar, system, value in zip(bars, SYSTEMS, values):
            annotation = f"{value:.3f}"
            if system.endswith("_calibrated"):
                rpss = float(lookup[(split, system)]["rpss_vs_uniform"])
                annotation = f"{annotation}\nRPSS {rpss:+.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                annotation,
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        axis.set_xticks(x, [labels[system] for system in SYSTEMS], rotation=24, ha="right")
        axis.set_title(f"{split.title()} years: {', '.join(map(str, years))}")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0.0)
        axis.margins(y=0.18)
    axes[0].set_ylabel("Pooled area-weighted RPS (lower is better)")
    title = (
        "Exploratory uniform-blend calibration — not official Quest validation\n"
        f"alpha(p0)={alphas['p0']:.3f}, alpha(model)={alphas['model']:.3f}; "
        f"weighting={weighting}"
    )
    if note and note != DEFAULT_PLOT_NOTE:
        title = f"{title}\n{note}"
    fig.suptitle(title)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _all_lead_results(
    rows: Sequence[Mapping[str, Any]], split: str
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if row["split"] != split or row["lead"] != "all":
            continue
        result[str(row["system"])] = {
            "alpha": None if row["alpha"] is None else float(row["alpha"]),
            "rps": float(row["rps"]),
            "rpss_vs_uniform": float(row["rpss_vs_uniform"]),
            "skill_vs_raw": (
                None if row["skill_vs_raw"] is None else float(row["skill_vs_raw"])
            ),
            "valid_weight": float(row["valid_weight"]),
        }
    return result


def run_calibration(
    case_files: Iterable[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    fit_years: Sequence[int],
    evaluation_years: Sequence[int],
    device: str = "cpu",
    batch_size: int = 1,
    weighting: str = "area-land",
    thursday_only: bool = False,
    plot_note: str = DEFAULT_PLOT_NOTE,
) -> CalibrationArtifacts:
    """Fit on explicit years, freeze alphas, and score explicit evaluation years."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    fit = _canonical_years(fit_years, name="fit_years")
    evaluation = _canonical_years(evaluation_years, name="evaluation_years")
    overlap = sorted(set(fit).intersection(evaluation))
    if overlap:
        raise ValueError(f"fit and evaluation years must be disjoint; overlap={overlap}")
    paths = _resolve_case_files(case_files)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    (
        fit_cases,
        model,
        model_channels,
        checkpoint_fingerprint,
        checkpoint_payload,
    ) = _load_forecast_cases(
        paths,
        checkpoint_path,
        fit,
        device=device,
        batch_size=batch_size,
        weighting=weighting,
        thursday_only=thursday_only,
    )
    evaluation_cases, _, _, _, _ = _load_forecast_cases(
        paths,
        checkpoint_path,
        evaluation,
        device=device,
        batch_size=batch_size,
        weighting=weighting,
        thursday_only=thursday_only,
        model=model,
        model_channels=model_channels,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    alphas = _fit_alphas(fit_cases)
    fit_case_rows, fit_summary_rows = _score_cases(
        fit_cases, alphas, split="fit", weighting=weighting
    )
    evaluation_case_rows, evaluation_summary_rows = _score_cases(
        evaluation_cases, alphas, split="evaluation", weighting=weighting
    )
    case_rows = fit_case_rows + evaluation_case_rows
    summary_rows = fit_summary_rows + evaluation_summary_rows

    json_path = destination / "uniform_blend_calibration.json"
    summary_path = destination / "uniform_blend_summary.csv"
    per_case_path = destination / "uniform_blend_by_case.csv"
    plot_path = destination / "uniform_blend_exploratory.png"
    _write_csv(
        summary_path,
        summary_rows,
        (
            "split",
            "lead",
            "system",
            "alpha",
            "rps",
            "rpss_vs_uniform",
            "skill_vs_raw",
            "rps_numerator",
            "valid_weight",
            "n_case_leads",
            "weighting",
        ),
    )
    _write_csv(
        per_case_path,
        case_rows,
        (
            "split",
            "case_file",
            "case_index",
            "init_date",
            "lead",
            "system",
            "alpha",
            "rps",
            "rpss_vs_uniform",
            "skill_vs_raw",
            "valid_cells",
            "valid_weight",
            "weighting",
        ),
    )
    checkpoint_model_name = checkpoint_payload.get("model_name")
    checkpoint_metadata = checkpoint_payload.get("metadata", {})
    if checkpoint_model_name is None and isinstance(checkpoint_metadata, Mapping):
        checkpoint_model_name = checkpoint_metadata.get("model_name")
    if checkpoint_model_name is not None:
        checkpoint_model_name = str(checkpoint_model_name)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = {
        "schema_version": 2,
        "method": "P_cal=(1-alpha)*uniform+alpha*P",
        "selection_metric": "pooled area-weighted four-threshold RPS",
        "alpha_scope": "one scalar per forecast source across both lead periods",
        "fit_years": list(fit),
        "evaluation_years": list(evaluation),
        "disjoint_years": True,
        "weighting": weighting,
        "thursday_only": bool(thursday_only),
        "alphas": {name: float(value) for name, value in alphas.items()},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_cache_contract_sha256": checkpoint_fingerprint,
        "checkpoint_model_name": checkpoint_model_name,
        "case_files": [str(path) for path in paths],
        "n_fit_cases": len(fit_cases),
        "n_evaluation_cases": len(evaluation_cases),
        "results": {
            "fit": _all_lead_results(summary_rows, "fit"),
            "evaluation": _all_lead_results(summary_rows, "evaluation"),
        },
        "plot_note": plot_note,
        "status": "exploratory; not official AI Weather Quest validation",
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _plot_summary(
        summary_rows,
        plot_path,
        fit_years=fit,
        evaluation_years=evaluation,
        weighting=weighting,
        alphas=alphas,
        note=plot_note,
    )
    return CalibrationArtifacts(
        calibration_json=json_path,
        summary_csv=summary_path,
        per_case_csv=per_case_path,
        plot_png=plot_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit scalar uniform blends on explicit years and report frozen "
            "calibration results on disjoint evaluation years."
        ),
        epilog=FUXI_PERMISSION_WARNING,
    )
    parser.add_argument(
        "--cases",
        required=True,
        nargs="+",
        type=Path,
        help="Prepared NPZ files, project Zarr cache, or directories containing them",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-years", required=True, nargs="+", type=int)
    parser.add_argument("--evaluation-years", required=True, nargs="+", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument(
        "--weighting",
        choices=("area", "area-land"),
        default="area-land",
        help="Cos(latitude) globally or with the official land_fraction>=0.5 mask",
    )
    parser.add_argument(
        "--thursday-only",
        action="store_true",
        help="Retain literal calendar Thursdays in both splits",
    )
    parser.add_argument("--plot-note", default=DEFAULT_PLOT_NOTE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(FUXI_PERMISSION_WARNING, file=sys.stderr)
    artifacts = run_calibration(
        args.cases,
        args.checkpoint,
        args.output_dir,
        fit_years=args.fit_years,
        evaluation_years=args.evaluation_years,
        device=args.device,
        batch_size=args.batch_size,
        weighting=args.weighting,
        thursday_only=args.thursday_only,
        plot_note=args.plot_note,
    )
    for path in artifacts.paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
