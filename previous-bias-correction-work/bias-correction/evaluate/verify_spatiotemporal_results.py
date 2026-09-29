#!/usr/bin/env python
"""Independently rebuild and verify a FuXi–IMERG spatiotemporal result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

import fuxi_imerg_experiment as base
import fuxi_imerg_spatiotemporal as experiment
from fuxi_adapter.models import TemporalAttentionUNet


HERE = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_result() -> Path:
    candidates = sorted(experiment.RESULTS_ROOT.glob("full_scratch_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError("no completed full scratch result found")
    return candidates[-1].parent


def assert_frame_equal(
    actual: pd.DataFrame,
    expected_path: Path,
    *,
    rtol: float = 2.0e-11,
    atol: float = 2.0e-11,
) -> None:
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=rtol,
        atol=atol,
        check_dtype=False,
    )


def verify(result: Path) -> None:
    result = Path(result).resolve()
    manifest = json.loads((result / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["smoke"]:
        raise AssertionError("result is not a complete full run")
    if manifest["methods"] != list(experiment.METHOD_ORDER):
        raise AssertionError("method contract differs")

    for relative, expected_hash in manifest["artifacts"].items():
        path = result / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise AssertionError(f"artifact hash differs: {relative}")

    with xr.open_zarr(result / "predictions.zarr", consolidated=True) as dataset:
        dataset = dataset.load()
    expected_sizes = {
        "method": 5,
        "init": 70,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }
    if dict(dataset.sizes) != expected_sizes:
        raise AssertionError(f"prediction sizes differ: {dataset.sizes}")
    if dataset.method.values.tolist() != list(experiment.METHOD_ORDER):
        raise AssertionError("prediction method ordering differs")
    weights = np.asarray(dataset.area_weight_km2, dtype=np.float64)
    support = weights > 0.0
    if support.sum() != 174:
        raise AssertionError("support is not 174 cells")

    predictions = {
        method: np.asarray(
            dataset.prediction.sel({"method": method}), dtype=np.float32
        )
        for method in experiment.METHOD_ORDER
    }
    truth = np.asarray(dataset.truth_imerg, dtype=np.float32)
    climatology = np.asarray(dataset.imerg_climatology, dtype=np.float32)
    initializations = dataset.init.values.astype("datetime64[D]")
    hybrid = predictions["lead_adaptive_hybrid"]
    spatial = predictions["spatial_unet"]
    temporal = predictions["spatiotemporal_unet"]
    if not np.array_equal(hybrid[:, :4], spatial[:, :4], equal_nan=True):
        raise AssertionError("hybrid Weeks 1-4 are not exact spatial predictions")
    if not np.array_equal(hybrid[:, 4:], temporal[:, 4:], equal_nan=True):
        raise AssertionError("hybrid Weeks 5-6 are not exact temporal predictions")

    case_metrics = experiment.evaluate_predictions(
        truth, climatology, predictions, initializations, weights
    )
    assert_frame_equal(case_metrics, result / "metrics/case_metrics.csv")
    summary = experiment.summarize_by_lead(case_metrics)
    assert_frame_equal(summary, result / "metrics/summary_by_lead.csv")
    intervals = experiment.paired_intervals(
        case_metrics, initializations, smoke=False
    )
    assert_frame_equal(intervals, result / "metrics/paired_skill.csv")
    headline = experiment.headline_table(case_metrics, intervals)
    assert_frame_equal(headline, result / "metrics/headline_metrics.csv")
    yearly = experiment.yearly_table(case_metrics)
    assert_frame_equal(yearly, result / "metrics/yearly_skill.csv")

    spatial_data, spatial_summary, spatial_audit = experiment.build_spatial_metrics(
        predictions,
        truth,
        dataset.latitude.values,
        dataset.longitude.values,
        weights,
    )
    with xr.open_dataset(result / "metrics/spatial_rmse.nc") as stored_spatial:
        xr.testing.assert_allclose(spatial_data, stored_spatial.load())
    assert_frame_equal(spatial_summary, result / "metrics/spatial_summary.csv")
    stored_audit = json.loads((result / "metrics/spatial_audit.json").read_text())
    if json.dumps(spatial_audit, sort_keys=True) != json.dumps(stored_audit, sort_keys=True):
        raise AssertionError("spatial audit differs")

    print("Rebuilding source alignment and 2019 lead selection...", flush=True)
    forecast = base.load_fuxi()
    observations = base.load_imerg(forecast)
    rebuilt_weights = base.load_area_weights(
        forecast, observations.observation_fraction
    )
    splits = base.split_indices(forecast.initializations)
    arrays = base.make_neural_arrays(
        forecast, observations, rebuilt_weights, splits["train"]
    )
    if arrays.feature_stats != json.loads((result / "normalization.json").read_text()):
        raise AssertionError("train-only normalization does not rebuild exactly")
    if not np.array_equal(
        forecast.initializations[splits["test"]], initializations
    ):
        raise AssertionError("rebuilt test initialization order differs")
    if not np.array_equal(
        observations.weekly_truth[splits["test"]], truth, equal_nan=True
    ):
        raise AssertionError("rebuilt IMERG truth differs")

    spatial_validation = experiment.load_checkpoint_ensemble(
        experiment.PARENT_RESULT,
        "spatial",
        arrays,
        forecast,
        rebuilt_weights,
        splits["validation"],
    )
    temporal_validation = experiment.load_checkpoint_ensemble(
        result,
        "scratch_temporal",
        arrays,
        forecast,
        rebuilt_weights,
        splits["validation"],
    )
    rebuilt_hybrid, selection = experiment.select_lead_adaptive_hybrid(
        spatial_validation,
        temporal_validation,
        observations.weekly_truth[splits["validation"]],
        observations.weekly_climatology[splits["validation"]],
        forecast.initializations[splits["validation"]],
        rebuilt_weights,
        spatial,
        temporal,
    )
    if not np.array_equal(rebuilt_hybrid, hybrid, equal_nan=True):
        raise AssertionError("validation-selected hybrid does not rebuild exactly")
    assert_frame_equal(
        selection,
        result / "metrics/validation_lead_selection.csv",
        rtol=2.0e-6,
        atol=2.0e-7,
    )

    training = manifest["training"]
    if training["parameter_count"] != 144401:
        raise AssertionError("unexpected temporal parameter count")
    for record in training["runs"]:
        checkpoint = torch.load(
            result / record["checkpoint"], map_location="cpu", weights_only=False
        )
        model = TemporalAttentionUNet(9, base_channels=16, dropout=0.1, max_leads=6)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if checkpoint["best_epoch"] != record["best_epoch"]:
            raise AssertionError("checkpoint epoch differs from manifest")

    print("PASS: hashes, splits, checkpoints, predictions, metrics, bootstrap, and maps")
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    verify(args.result or default_result())


if __name__ == "__main__":
    main()
