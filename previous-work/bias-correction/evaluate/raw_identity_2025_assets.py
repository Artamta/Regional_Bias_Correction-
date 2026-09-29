#!/usr/bin/env python3
"""Primary scores and frozen secondary intensity diagnostics for 2025."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import raw_identity_2025_contract as contract


METRICS = ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")


def _weighted_scalar(
    forecast: np.ndarray, observation: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    valid = (
        np.isfinite(forecast)
        & np.isfinite(observation)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    if int(valid.sum()) < 3:
        raise ValueError("fewer than three weighted cells")
    predicted = np.asarray(forecast[valid], dtype=np.float64)
    observed = np.asarray(observation[valid], dtype=np.float64)
    weights = np.asarray(weight[valid], dtype=np.float64)
    total = float(weights.sum(dtype=np.float64))
    error = predicted - observed
    predicted_centered = predicted - float(np.sum(weights * predicted) / total)
    observed_centered = observed - float(np.sum(weights * observed) / total)
    acc_denominator = float(
        np.sqrt(
            np.sum(weights * predicted_centered**2)
            * np.sum(weights * observed_centered**2)
        )
    )
    values = {
        "acc": (
            float(
                np.sum(weights * predicted_centered * observed_centered)
                / acc_denominator
            )
            if acc_denominator > 0.0
            else float("nan")
        ),
        "rmse_mm_day": float(np.sqrt(np.sum(weights * error**2) / total)),
        "mae_mm_day": float(np.sum(weights * np.abs(error)) / total),
        "bias_mm_day": float(np.sum(weights * error) / total),
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("case/lead metric is non-finite")
    return values


def weighted_case_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    coverage: np.ndarray,
    area_weight_km2: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Match E2 case/lead scores, including float32 anomaly formation."""

    prediction32 = np.asarray(prediction, dtype=np.float32)
    truth32 = np.asarray(truth, dtype=np.float32)
    climatology32 = np.asarray(climatology, dtype=np.float32)
    coverage32 = np.asarray(coverage, dtype=np.float32)
    area = np.asarray(area_weight_km2, dtype=np.float64)
    if prediction32.shape != truth32.shape or truth32.shape != climatology32.shape:
        raise ValueError("prediction, truth, and climatology shapes differ")
    if coverage32.shape != truth32.shape or area.shape != truth32.shape[-2:]:
        raise ValueError("verification weights do not match the evaluated fields")
    if not np.isfinite(coverage32).all() or np.any(
        (coverage32 < 0.0) | (coverage32 > 1.0)
    ):
        raise ValueError("coverage must be finite and in [0,1]")
    if not np.isfinite(area).all() or np.any(area < 0.0):
        raise ValueError("area weights must be finite and nonnegative")
    result = {
        metric: np.empty(prediction32.shape[:2], dtype=np.float64) for metric in METRICS
    }
    forecast_anomaly = prediction32 - climatology32
    truth_anomaly = truth32 - climatology32
    for case in range(prediction32.shape[0]):
        for lead in range(prediction32.shape[1]):
            weight = area * coverage32[case, lead]
            absolute = _weighted_scalar(
                prediction32[case, lead], truth32[case, lead], weight
            )
            anomaly = _weighted_scalar(
                forecast_anomaly[case, lead], truth_anomaly[case, lead], weight
            )
            for metric in ("rmse_mm_day", "mae_mm_day", "bias_mm_day"):
                result[metric][case, lead] = absolute[metric]
            result["acc"][case, lead] = anomaly["acc"]
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("primary case/lead metric cube is non-finite")
    return result


def circular_moving_block_indices(
    case_count: int,
    *,
    draws: int = contract.BOOTSTRAP_DRAWS,
    block_length: int = contract.BOOTSTRAP_BLOCK_LENGTH,
    seed: int = contract.BOOTSTRAP_SEED,
) -> np.ndarray:
    if case_count < 2 or draws < 1:
        raise ValueError("case_count and draws must be positive")
    if not 1 <= block_length <= case_count:
        raise ValueError("block_length must be between one and case_count")
    generator = np.random.default_rng(seed)
    block_count = math.ceil(case_count / block_length)
    starts = generator.integers(0, case_count, size=(draws, block_count))
    offsets = np.arange(block_length, dtype=np.int64)
    values = (starts[:, :, None] + offsets[None, None, :]) % case_count
    return values.reshape(draws, -1)[:, :case_count].astype(np.int16)


def _effect_draws(
    metric: str, raw: np.ndarray, candidate: np.ndarray, indices: np.ndarray
) -> tuple[float, np.ndarray, float | None, np.ndarray | None]:
    if not np.isfinite(raw).all() or not np.isfinite(candidate).all():
        raise ValueError(f"non-finite {metric} entered primary bootstrap")
    raw_point = float(np.mean(raw))
    candidate_point = float(np.mean(candidate))
    axes = tuple(range(1, raw[indices].ndim))
    sampled_raw = np.mean(raw[indices], axis=axes)
    sampled_candidate = np.mean(candidate[indices], axis=axes)
    if metric in ("rmse_mm_day", "mae_mm_day"):
        if raw_point <= 0.0 or np.any(sampled_raw <= 0.0):
            raise ValueError(f"nonpositive raw {metric} prevents relative effects")
        effect = raw_point - candidate_point
        effects = sampled_raw - sampled_candidate
        percent = 100.0 * effect / raw_point
        percents = 100.0 * effects / sampled_raw
    elif metric == "acc":
        effect = candidate_point - raw_point
        effects = sampled_candidate - sampled_raw
        percent = None
        percents = None
    elif metric == "bias_mm_day":
        effect = abs(raw_point) - abs(candidate_point)
        effects = np.abs(sampled_raw) - np.abs(sampled_candidate)
        percent = None
        percents = None
    else:  # pragma: no cover
        raise ValueError(metric)
    if not np.isfinite(effect) or not np.isfinite(effects).all():
        raise ValueError(f"non-finite {metric} bootstrap effect")
    return effect, effects, percent, percents


def paired_effects_table(
    raw: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        raw_values = np.asarray(raw[metric], dtype=np.float64)
        candidate_values = np.asarray(candidate[metric], dtype=np.float64)
        if raw_values.shape != candidate_values.shape or raw_values.ndim != 2:
            raise ValueError(f"{metric} must share [case, lead] arrays")
        scopes = [
            (f"W{lead + 1}", raw_values[:, lead], candidate_values[:, lead])
            for lead in range(raw_values.shape[1])
        ]
        scopes.append(("W1-W6", raw_values, candidate_values))
        for scope, current_raw, current_candidate in scopes:
            effect, effects, percent, percents = _effect_draws(
                metric, current_raw, current_candidate, indices
            )
            low = float(np.quantile(effects, 0.025))
            high = float(np.quantile(effects, 0.975))
            rows.append(
                {
                    "candidate": "raw_identity",
                    "baseline": "raw_fuxi",
                    "lead_scope": scope,
                    "metric": metric,
                    "raw_fuxi": float(np.mean(current_raw)),
                    "raw_identity": float(np.mean(current_candidate)),
                    "improvement": effect,
                    "improvement_ci_low": low,
                    "improvement_ci_high": high,
                    "improvement_pct": percent,
                    "improvement_pct_ci_low": (
                        float(np.quantile(percents, 0.025))
                        if percents is not None
                        else None
                    ),
                    "improvement_pct_ci_high": (
                        float(np.quantile(percents, 0.975))
                        if percents is not None
                        else None
                    ),
                    "descriptive_interval_wholly_above_zero": bool(
                        effect > 0.0 and low > 0.0
                    ),
                    "effect_direction": (
                        "baseline minus candidate (positive is better)"
                        if metric in ("rmse_mm_day", "mae_mm_day")
                        else (
                            "candidate minus baseline (positive is better)"
                            if metric == "acc"
                            else "absolute baseline bias minus absolute candidate bias"
                        )
                    ),
                }
            )
    result = pd.DataFrame(rows)
    numeric = [
        "raw_fuxi",
        "raw_identity",
        "improvement",
        "improvement_ci_low",
        "improvement_ci_high",
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("primary bootstrap table is non-finite")
    return result


def case_metrics_table(
    initializations: np.ndarray,
    metrics: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    if tuple(metrics) != contract.METHOD_HIERARCHY:
        raise ValueError("result assets permit only raw_fuxi and raw_identity")
    rows: list[dict[str, Any]] = []
    for method in contract.METHOD_HIERARCHY:
        for case, initialization in enumerate(initializations):
            for lead in range(contract.EXPECTED_LEADS):
                rows.append(
                    {
                        "method": method,
                        "initialization": np.datetime_as_string(
                            initialization, unit="D"
                        ),
                        "lead_week": lead + 1,
                        **{
                            metric: float(metrics[method][metric][case, lead])
                            for metric in METRICS
                        },
                    }
                )
    result = pd.DataFrame(rows)
    if result.duplicated(["method", "initialization", "lead_week"]).any():
        raise ValueError("case metric table contains duplicates")
    if not np.isfinite(result[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("case metric table contains non-finite values")
    return result


def _bootstrap_counts(indices: np.ndarray, case_count: int) -> np.ndarray:
    counts = np.zeros((indices.shape[0], case_count), dtype=np.int16)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    np.add.at(counts, (rows, indices.reshape(-1)), 1)
    return counts


def intensity_diagnostics(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    dynamic_weights: np.ndarray,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen secondary E2-style pooled intensity diagnostics."""

    forecast = np.stack(
        [
            np.asarray(predictions[method], dtype=np.float64)
            for method in contract.METHOD_HIERARCHY
        ]
    )
    truth64 = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(dynamic_weights, dtype=np.float64)
    if weights.shape != truth64.shape or forecast.shape[1:] != truth64.shape:
        raise ValueError("intensity fields and dynamic weights differ")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("intensity dynamic weights must be finite and nonnegative")
    common_valid = (
        (weights > 0.0)
        & np.isfinite(weights)
        & np.isfinite(truth64)
        & np.all(np.isfinite(forecast), axis=0)
    )
    counts = _bootstrap_counts(indices, truth64.shape[0])
    metric_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for key, label, lower, upper in contract.INTENSITY_STRATA:
        upper_mask = (
            np.ones(truth64.shape, dtype=bool) if upper is None else truth64 < upper
        )
        selected = common_valid & (truth64 >= lower) & upper_mask
        denominator_by_case = np.sum(
            np.where(selected, weights, 0.0), axis=(1, 2, 3), dtype=np.float64
        )
        denominator = float(denominator_by_case.sum(dtype=np.float64))
        empty = not np.isfinite(denominator) or denominator <= 0.0
        numerators: dict[tuple[str, str], np.ndarray] = {}
        for method_index, method in enumerate(contract.METHOD_HIERARCHY):
            error = forecast[method_index] - truth64
            for metric, values in (
                ("rmse_mm_day", error**2),
                ("mae_mm_day", np.abs(error)),
                ("bias_mm_day", error),
            ):
                numerators[(method, metric)] = np.sum(
                    np.where(selected, weights * values, 0.0),
                    axis=(1, 2, 3),
                    dtype=np.float64,
                )
            if empty:
                rmse = mae = bias = None
                status = "insufficient_weight/no_estimate"
            else:
                rmse = float(
                    np.sqrt(numerators[(method, "rmse_mm_day")].sum() / denominator)
                )
                mae = float(numerators[(method, "mae_mm_day")].sum() / denominator)
                bias = float(numerators[(method, "bias_mm_day")].sum() / denominator)
                status = "estimated"
            metric_rows.append(
                {
                    "analysis_role": "secondary_exploratory_within_final",
                    "method": method,
                    "stratum": key,
                    "stratum_label": label,
                    "lower_mm_day_inclusive": lower,
                    "upper_mm_day_exclusive": upper,
                    "estimate_status": status,
                    "cell_case_lead_count": int(np.count_nonzero(selected)),
                    "dynamic_area_weight_sum_km2_case_lead": denominator,
                    "rmse_mm_day": rmse,
                    "mae_mm_day": mae,
                    "bias_mm_day": bias,
                    "definition": (
                        "pooled cell-lead error stratified by verifying weekly-mean IMD; "
                        "India cell area x exact weekly IMD coverage weighted"
                    ),
                }
            )
        draw_denominator = counts @ denominator_by_case
        for metric in ("rmse_mm_day", "mae_mm_day", "bias_mm_day"):
            raw_num = numerators[("raw_fuxi", metric)]
            candidate_num = numerators[("raw_identity", metric)]
            interval_available = (not empty) and np.all(draw_denominator > 0.0)
            if empty:
                raw_point = candidate_point = effect = None
                status = "insufficient_weight/no_estimate"
                distribution = None
            else:
                if metric == "rmse_mm_day":
                    raw_point = float(np.sqrt(raw_num.sum() / denominator))
                    candidate_point = float(np.sqrt(candidate_num.sum() / denominator))
                else:
                    raw_point = float(raw_num.sum() / denominator)
                    candidate_point = float(candidate_num.sum() / denominator)
                effect = (
                    abs(raw_point) - abs(candidate_point)
                    if metric == "bias_mm_day"
                    else raw_point - candidate_point
                )
                if interval_available:
                    if metric == "rmse_mm_day":
                        raw_draw = np.sqrt((counts @ raw_num) / draw_denominator)
                        candidate_draw = np.sqrt(
                            (counts @ candidate_num) / draw_denominator
                        )
                    else:
                        raw_draw = (counts @ raw_num) / draw_denominator
                        candidate_draw = (counts @ candidate_num) / draw_denominator
                    distribution = (
                        np.abs(raw_draw) - np.abs(candidate_draw)
                        if metric == "bias_mm_day"
                        else raw_draw - candidate_draw
                    )
                    status = "estimated"
                else:
                    distribution = None
                    status = "point_estimate_only/insufficient_resampled_weight"
            effect_rows.append(
                {
                    "analysis_role": "secondary_exploratory_within_final",
                    "selection_or_multiplicity_claim": False,
                    "stratum": key,
                    "candidate": "raw_identity",
                    "baseline": "raw_fuxi",
                    "metric": metric,
                    "estimate_status": status,
                    "raw_fuxi": raw_point,
                    "raw_identity": candidate_point,
                    "improvement": effect,
                    "improvement_ci_low": (
                        float(np.quantile(distribution, 0.025))
                        if distribution is not None
                        else None
                    ),
                    "improvement_ci_high": (
                        float(np.quantile(distribution, 0.975))
                        if distribution is not None
                        else None
                    ),
                    "bootstrap_draws": int(indices.shape[0]),
                    "block_length_initializations": contract.BOOTSTRAP_BLOCK_LENGTH,
                    "definition": (
                        "baseline minus candidate for RMSE/MAE; absolute baseline "
                        "bias minus absolute candidate bias; descriptive only"
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(effect_rows)


def _write_results_markdown(path: Path, effects: pd.DataFrame) -> None:
    pooled = effects.loc[effects.lead_scope.eq("W1-W6")].set_index("metric")
    lines = [
        "# Sealed independent 2025 raw-identity evaluation",
        "",
        "> Frozen primary: the three-seed raw-identity residual-mean ensemble "
        "versus raw FuXi.",
        "",
        "## Pooled W1-W6",
        "",
        "| Metric | Raw FuXi | Raw identity | Improvement | 95% paired block interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in (
        ("rmse_mm_day", "RMSE (mm/day)"),
        ("mae_mm_day", "MAE (mm/day)"),
        ("bias_mm_day", "Signed bias (mm/day)"),
        ("acc", "ACC"),
    ):
        row = pooled.loc[metric]
        if metric in ("rmse_mm_day", "mae_mm_day"):
            effect = f"{row.improvement_pct:+.2f}%"
            interval = (
                f"[{row.improvement_pct_ci_low:+.2f}%, "
                f"{row.improvement_pct_ci_high:+.2f}%]"
            )
        else:
            effect = f"{row.improvement:+.4f}"
            interval = (
                f"[{row.improvement_ci_low:+.4f}, {row.improvement_ci_high:+.4f}]"
            )
        lines.append(
            f"| {label} | {row.raw_fuxi:.4f} | {row.raw_identity:.4f} | "
            f"{effect} | {interval} |"
        )
    lines.extend(
        [
            "",
            "Truth is the ordinary arithmetic mean of seven daily IMD values. "
            "Scores use India area times the minimum daily coverage in each week, "
            "then equally average case/lead spatial scores.",
            "",
            "Bias improvement is abs(raw pooled signed bias) minus abs(candidate "
            "pooled signed bias). Intervals use 10,000 paired circular moving-block "
            "draws of 13 starts (seed 20260822). They are descriptive and conditional "
            "on one 2025 JJAS season, not interannual uncertainty or hypothesis tests.",
            "",
            "Intensity-stratum results are frozen secondary exploratory diagnostics; "
            "they were not used for selection and carry no multiplicity claim.",
        ]
    )
    contract.write_bytes_exclusive(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _write_npy(path: Path, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    contract.write_bytes_exclusive(path, stream.getvalue())


def _write_npz(path: Path, **values: np.ndarray) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    contract.write_bytes_exclusive(path, stream.getvalue())


def write_assets(
    output: Path,
    *,
    initializations: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weekly_coverage: np.ndarray,
    area_weight_km2: np.ndarray,
    support: np.ndarray,
    raw_fuxi: np.ndarray,
    fuxi_spread: np.ndarray,
    fuxi_t2m_kelvin: np.ndarray,
    fuxi_member_count: np.ndarray,
    fuxi_t2m_member_count: np.ndarray,
    valid_dates: np.ndarray,
    selected_daily_imd: np.ndarray,
    selected_daily_coverage: np.ndarray,
    raw_identity: np.ndarray,
    ensemble_standardized_residual: np.ndarray,
) -> Mapping[str, Any]:
    expected_shape = (
        contract.EXPECTED_CASES,
        contract.EXPECTED_LEADS,
        *contract.EXPECTED_GRID,
    )
    if not np.array_equal(
        np.asarray(initializations, dtype="datetime64[D]"),
        contract.expected_initialization_dates(),
    ):
        raise ValueError("asset initializations differ from the frozen schedule")
    expected_valid_dates = contract.expected_initialization_dates()[
        :, None, None
    ] + np.arange(42, dtype="timedelta64[D]").reshape(1, 6, 7)
    if (
        np.asarray(raw_fuxi).shape != expected_shape
        or np.asarray(fuxi_spread).shape != expected_shape
        or np.asarray(fuxi_t2m_kelvin).shape != expected_shape
        or np.asarray(fuxi_member_count).shape != expected_shape
        or np.asarray(fuxi_t2m_member_count).shape != expected_shape
        or np.asarray(raw_identity).shape != expected_shape
        or np.asarray(truth).shape != expected_shape
        or np.asarray(climatology).shape != expected_shape
        or np.asarray(weekly_coverage).shape != expected_shape
        or np.asarray(ensemble_standardized_residual).shape != expected_shape
        or np.asarray(valid_dates).shape
        != (contract.EXPECTED_CASES, contract.EXPECTED_LEADS, 7)
        or np.asarray(selected_daily_imd).shape
        != (contract.EXPECTED_CASES, contract.EXPECTED_LEADS, 7, 27, 27)
        or np.asarray(selected_daily_coverage).shape
        != (contract.EXPECTED_CASES, contract.EXPECTED_LEADS, 7, 27, 27)
        or np.asarray(support).shape != contract.EXPECTED_GRID
        or int(np.asarray(support, dtype=bool).sum()) != contract.EXPECTED_SUPPORT_CELLS
        or np.asarray(area_weight_km2).shape != contract.EXPECTED_GRID
        or not np.array_equal(
            np.asarray(latitude, dtype=np.float64), np.linspace(39.0, 0.0, 27)
        )
        or not np.array_equal(
            np.asarray(longitude, dtype=np.float64), np.linspace(60.0, 99.0, 27)
        )
    ):
        raise ValueError("final asset shape/grid/support contract changed")
    support_array = np.asarray(support, dtype=bool)
    daily = np.asarray(selected_daily_imd, dtype=np.float32)
    daily_coverage = np.asarray(selected_daily_coverage, dtype=np.float32)
    if not np.array_equal(
        np.asarray(valid_dates, dtype="datetime64[D]"), expected_valid_dates
    ):
        raise ValueError("saved daily verification dates changed")
    if (
        not np.isfinite(np.asarray(raw_fuxi, dtype=np.float32)).all()
        or not np.isfinite(np.asarray(fuxi_spread, dtype=np.float32)).all()
        or not np.isfinite(np.asarray(fuxi_t2m_kelvin, dtype=np.float32)).all()
        or np.any(np.asarray(raw_fuxi, dtype=np.float32) < 0.0)
        or np.any(np.asarray(fuxi_spread, dtype=np.float32) < 0.0)
        or not np.all(np.asarray(fuxi_member_count) == 50)
        or not np.all(np.asarray(fuxi_t2m_member_count) == 50)
        or not np.isfinite(daily_coverage).all()
        or np.any((daily_coverage < 0.0) | (daily_coverage > 1.0))
        or not np.isfinite(np.asarray(area_weight_km2, dtype=np.float64)).all()
        or np.any(np.asarray(area_weight_km2, dtype=np.float64) < 0.0)
        or not np.array_equal(
            np.asarray(area_weight_km2, dtype=np.float64) > 0.0,
            support_array,
        )
    ):
        raise ValueError("saved predictor/weight arrays are invalid")
    represented_daily = daily_coverage > 0.0
    if (
        not np.isfinite(daily[represented_daily]).all()
        or np.any(daily[represented_daily] < 0.0)
        or not np.array_equal(
            np.mean(daily, axis=2, dtype=np.float64).astype(np.float32),
            np.asarray(truth, dtype=np.float32),
            equal_nan=True,
        )
        or not np.array_equal(
            np.min(daily_coverage, axis=2).astype(np.float32),
            np.asarray(weekly_coverage, dtype=np.float32),
        )
    ):
        raise ValueError("saved daily IMD arrays do not reconstruct scored fields")
    predictions = {
        "raw_fuxi": np.asarray(raw_fuxi, dtype=np.float32),
        "raw_identity": np.asarray(raw_identity, dtype=np.float32),
    }
    metrics = {
        method: weighted_case_metrics(
            prediction,
            truth,
            climatology,
            weekly_coverage,
            area_weight_km2,
        )
        for method, prediction in predictions.items()
    }
    indices = circular_moving_block_indices(contract.EXPECTED_CASES)
    effects = paired_effects_table(
        metrics["raw_fuxi"], metrics["raw_identity"], indices
    )
    cases = case_metrics_table(initializations, metrics)
    dynamic_weight = np.asarray(area_weight_km2, dtype=np.float64)[
        None, None
    ] * np.asarray(weekly_coverage, dtype=np.float32)
    intensity_metrics, intensity_effects = intensity_diagnostics(
        predictions, truth, dynamic_weight, indices
    )
    contract.write_bytes_exclusive(
        output / "case_metrics.csv", cases.to_csv(index=False).encode("utf-8")
    )
    contract.write_bytes_exclusive(
        output / "paired_block_bootstrap_effects.csv",
        effects.to_csv(index=False).encode("utf-8"),
    )
    contract.write_bytes_exclusive(
        output / "secondary_intensity_metrics.csv",
        intensity_metrics.to_csv(index=False).encode("utf-8"),
    )
    contract.write_bytes_exclusive(
        output / "secondary_intensity_block_effects.csv",
        intensity_effects.to_csv(index=False).encode("utf-8"),
    )
    _write_npy(output / "bootstrap_initialization_indices.npy", indices)
    _write_npz(
        output / "independent_2025_fields.npz",
        initializations=np.asarray(initializations, dtype="datetime64[D]"),
        latitude=np.asarray(latitude, dtype=np.float64),
        longitude=np.asarray(longitude, dtype=np.float64),
        lead_week=np.arange(1, 7, dtype=np.int16),
        valid_dates=np.asarray(valid_dates, dtype="datetime64[D]"),
        truth_imd=np.asarray(truth, dtype=np.float32),
        fixed_training_climatology=np.asarray(climatology, dtype=np.float32),
        weekly_coverage=np.asarray(weekly_coverage, dtype=np.float32),
        india_area_weight_km2=np.asarray(area_weight_km2, dtype=np.float64),
        support=np.asarray(support, dtype=bool),
        raw_fuxi=predictions["raw_fuxi"],
        fuxi_spread=np.asarray(fuxi_spread, dtype=np.float32),
        fuxi_t2m_kelvin=np.asarray(fuxi_t2m_kelvin, dtype=np.float32),
        fuxi_member_count=np.asarray(fuxi_member_count, dtype=np.int16),
        fuxi_t2m_member_count=np.asarray(fuxi_t2m_member_count, dtype=np.int16),
        selected_daily_imd=np.asarray(selected_daily_imd, dtype=np.float32),
        selected_daily_coverage=np.asarray(selected_daily_coverage, dtype=np.float32),
        raw_identity=predictions["raw_identity"],
        ensemble_standardized_residual=np.asarray(
            ensemble_standardized_residual, dtype=np.float32
        ),
    )
    _write_results_markdown(output / "RESULTS.md", effects)
    pooled = effects.loc[effects.lead_scope.eq("W1-W6")]
    summary = {
        str(row.metric): {
            "raw_fuxi": float(row.raw_fuxi),
            "raw_identity": float(row.raw_identity),
            "improvement": float(row.improvement),
            "improvement_ci_low": float(row.improvement_ci_low),
            "improvement_ci_high": float(row.improvement_ci_high),
            "improvement_pct": (
                None if pd.isna(row.improvement_pct) else float(row.improvement_pct)
            ),
        }
        for row in pooled.itertuples(index=False)
    }
    contract.write_json_exclusive(output / "summary.json", summary)
    contract.fsync_directory(output)
    return {
        "method_hierarchy": list(contract.METHOD_HIERARCHY),
        "bootstrap_indices_sha256": contract.array_sha256(indices, "<i2"),
        "pooled_summary": summary,
        "secondary_intensity": {
            "role": "secondary_exploratory_within_final",
            "selection_or_multiplicity_claim": False,
            "strata": [key for key, _, _, _ in contract.INTENSITY_STRATA],
        },
    }


def artifact_hashes(output: Path) -> Mapping[str, str]:
    root = contract.assert_secure_directory(output, "staged artifact directory")
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise contract.SealContractError(
                        f"symlink in staged artifacts: {entry.path}"
                    )
                item = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(item)
                elif (
                    entry.is_file(follow_symlinks=False)
                    and entry.name != "manifest.json"
                ):
                    files.append(item)
                elif not entry.is_file(follow_symlinks=False):
                    raise contract.SealContractError(
                        f"non-regular staged artifact: {entry.path}"
                    )
    return {
        path.relative_to(root).as_posix(): contract.sha256_file(path)
        for path in sorted(files)
    }
