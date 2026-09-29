#!/usr/bin/env python
"""Verify a completed FuXi–IMERG experiment from its saved Zarr and CSVs.

Run with the same environment as the experiment::

    /home/raj.ayush/.conda/envs/fuxi/bin/python evaluate/verify_fuxi_imerg_results.py

Pass a result directory explicitly, or add ``--check-sources`` to re-hash the
2.2 GB source collection as well as the delivered artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
RESULTS_ROOT = HERE / "results" / "fuxi_imerg_jjas_5yr"
METHODS = ("raw_fuxi", "log_bias", "quantile_mapping", "residual_unet")
METRICS = ("acc", "rmse", "mae", "bias")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def weighted_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & np.isfinite(climatology)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    truth = truth[valid].astype(np.float64)
    prediction = prediction[valid].astype(np.float64)
    climatology = climatology[valid].astype(np.float64)
    weight = weights[valid].astype(np.float64)
    weight_sum = weight.sum()

    error = prediction - truth
    rmse = np.sqrt(np.sum(weight * error**2) / weight_sum)
    mae = np.sum(weight * np.abs(error)) / weight_sum
    bias = np.sum(weight * error) / weight_sum

    truth_anomaly = truth - climatology
    prediction_anomaly = prediction - climatology
    truth_anomaly -= np.sum(weight * truth_anomaly) / weight_sum
    prediction_anomaly -= np.sum(weight * prediction_anomaly) / weight_sum
    denominator = np.sqrt(
        np.sum(weight * truth_anomaly**2)
        * np.sum(weight * prediction_anomaly**2)
    )
    acc = np.sum(weight * truth_anomaly * prediction_anomaly) / denominator
    return float(acc), float(rmse), float(mae), float(bias)


def recompute_case_metrics(dataset: xr.Dataset) -> pd.DataFrame:
    truth = dataset.truth_imerg.load().values
    climatology = dataset.imerg_climatology.load().values
    predictions = dataset.prediction.load().values
    weights = dataset.area_weight_km2.load().values
    dates = dataset.init.values.astype("datetime64[D]")

    rows = []
    for method_index, method in enumerate(METHODS):
        for case_index, date in enumerate(dates):
            init = np.datetime_as_string(date, unit="D")
            for lead_index in range(6):
                acc, rmse, mae, bias = weighted_scores(
                    truth[case_index, lead_index],
                    predictions[method_index, case_index, lead_index],
                    climatology[case_index, lead_index],
                    weights,
                )
                rows.append(
                    {
                        "method": method,
                        "init": init,
                        "lead_week": lead_index + 1,
                        "acc": acc,
                        "rmse": rmse,
                        "mae": mae,
                        "bias": bias,
                    }
                )
    return pd.DataFrame(rows)


def block_indices(
    initializations: np.ndarray,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(years == year) for year in sorted(np.unique(years))]
    case_counts = {len(group) for group in groups}
    if len(case_counts) != 1:
        raise AssertionError("unequal yearly test counts")
    n_cases = case_counts.pop()
    blocks = int(np.ceil(n_cases / block_length))
    offsets = np.arange(block_length)
    sampled_years = rng.integers(0, len(groups), size=(samples, len(groups)))
    sampled = np.empty((samples, len(groups) * n_cases), dtype=np.int64)
    for draw in range(samples):
        for slot, year_index in enumerate(sampled_years[draw]):
            starts = rng.integers(0, n_cases - block_length + 1, size=blocks)
            local = (starts[:, None] + offsets).reshape(-1)[:n_cases]
            sampled[draw, slot * n_cases : (slot + 1) * n_cases] = groups[
                year_index
            ][local]
    return sampled


def effect(metric: str, model: np.ndarray, baseline: np.ndarray) -> float:
    model_mean = float(np.mean(model))
    baseline_mean = float(np.mean(baseline))
    if metric == "acc":
        return model_mean - baseline_mean
    return 100.0 * (baseline_mean - model_mean) / baseline_mean


def verify_bootstrap(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    saved: pd.DataFrame,
) -> None:
    first = saved.iloc[0]
    sampled = block_indices(
        initializations,
        int(first.n_resamples),
        int(first.block_length),
        int(first.seed),
    )
    case_order = [
        np.datetime_as_string(value, unit="D") for value in initializations
    ]
    for row in saved.itertuples(index=False):
        leads = range(1, 7) if row.lead_scope == "W1-W6" else [int(row.lead_scope[1:])]
        selected = case_metrics.loc[
            case_metrics.lead_week.isin(leads)
            & case_metrics.method.isin([row.method, row.baseline])
        ]
        pivot = selected.pivot_table(
            index="init", columns="method", values=row.metric, aggfunc="mean"
        ).reindex(case_order)
        model = pivot[row.method].to_numpy(dtype=np.float64)
        baseline = pivot[row.baseline].to_numpy(dtype=np.float64)
        draws = np.asarray(
            [effect(row.metric, model[index], baseline[index]) for index in sampled]
        )
        actual = np.asarray(
            [
                model.mean(),
                baseline.mean(),
                effect(row.metric, model, baseline),
                np.percentile(draws, 2.5),
                np.percentile(draws, 97.5),
            ]
        )
        expected = np.asarray(
            [row.model_mean, row.baseline_mean, row.effect, row.ci_lower, row.ci_upper]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)


def verify_result(run: Path, check_sources: bool) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["smoke"]:
        raise AssertionError("expected a complete non-smoke result")

    for relative, expected in manifest["artifacts"].items():
        actual = sha256_file(run / relative)
        if actual != expected:
            raise AssertionError(f"artifact hash differs: {relative}")

    if check_sources:
        inventory = pd.read_csv(run / "source_inventory.csv")
        for row in inventory.itertuples(index=False):
            path = Path(row.path)
            actual = sha256_file(path) if path.is_file() else sha256_tree(path)
            if actual != row.sha256:
                raise AssertionError(f"source hash differs: {path}")

    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as dataset:
        if dict(dataset.sizes) != {
            "method": 4,
            "init": 70,
            "lead_week": 6,
            "latitude": 27,
            "longitude": 27,
        }:
            raise AssertionError(f"unexpected Zarr sizes: {dataset.sizes}")
        if dataset.method.values.tolist() != list(METHODS):
            raise AssertionError("method ordering differs")
        if np.count_nonzero(dataset.area_weight_km2.values > 0.0) != 174:
            raise AssertionError("IMERG support is not 174 cells")
        years, counts = np.unique(
            pd.DatetimeIndex(dataset.init.values).year, return_counts=True
        )
        if dict(zip(years, counts)) != {2020: 35, 2021: 35}:
            raise AssertionError("test years/counts differ")
        expected_start = (
            dataset.init.values[:, None].astype("datetime64[D]")
            + 7 * np.arange(6)[None, :].astype("timedelta64[D]")
        )
        if not np.array_equal(
            dataset.valid_start.values.astype("datetime64[D]"), expected_start
        ):
            raise AssertionError("weekly valid starts differ")
        if not np.array_equal(
            dataset.valid_end_exclusive.values.astype("datetime64[D]"),
            expected_start + np.timedelta64(7, "D"),
        ):
            raise AssertionError("weekly valid ends differ")
        recomputed = recompute_case_metrics(dataset)
        initializations = dataset.init.values.astype("datetime64[D]")

    saved_cases = pd.read_csv(run / "metrics" / "case_metrics.csv")
    keys = ["method", "init", "lead_week"]
    recomputed = recomputed.sort_values(keys).reset_index(drop=True)
    saved_cases = saved_cases.sort_values(keys).reset_index(drop=True)
    if not recomputed[keys].equals(saved_cases[keys]):
        raise AssertionError("case-metric row ordering differs")
    np.testing.assert_allclose(
        recomputed[list(METRICS)].to_numpy(),
        saved_cases[list(METRICS)].to_numpy(),
        rtol=0.0,
        atol=2.0e-8,
    )

    saved_headline = pd.read_csv(run / "metrics" / "headline_metrics.csv").set_index(
        "method"
    )
    means = recomputed.groupby("method")[list(METRICS)].mean()
    np.testing.assert_allclose(
        means.loc[list(METHODS)].to_numpy(),
        saved_headline.loc[
            list(METHODS), ["acc", "rmse_mm_day", "mae_mm_day", "bias_mm_day"]
        ].to_numpy(),
        rtol=0.0,
        atol=2.0e-8,
    )
    verify_bootstrap(
        recomputed,
        initializations,
        pd.read_csv(run / "metrics" / "paired_skill.csv"),
    )

    print("PASS: artifacts, Zarr schema, alignment, metrics, and bootstrap intervals")
    print(saved_headline[["acc", "rmse_mm_day", "mae_mm_day", "bias_mm_day"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    run = args.run
    if run is None:
        candidates = sorted(RESULTS_ROOT.glob("full_*"))
        if not candidates:
            raise FileNotFoundError("no full result directory found")
        run = candidates[-1]
    verify_result(run.resolve(), args.check_sources)


if __name__ == "__main__":
    main()
