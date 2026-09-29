#!/usr/bin/env python
"""Verify a completed A100 large-temporal experiment from saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402


METHODS = (
    "raw_fuxi",
    "log_bias",
    "small_temporal",
    "big_spatial",
    "big_temporal",
    "selected_model",
)
CANDIDATES = ("small_temporal", "big_spatial", "big_temporal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def latest_full() -> Path:
    runs = sorted(
        (HERE / "results" / "fuxi_imerg_a100_big_temporal").glob("full_*")
    )
    if not runs:
        raise FileNotFoundError("no full A100 large-temporal result found")
    return runs[-1]


def parse_utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise AssertionError(f"timestamp is not timezone-aware: {value}")
    return timestamp


def rebuild_case_metrics(dataset: xr.Dataset) -> pd.DataFrame:
    truth = np.asarray(dataset.truth_imerg.values, dtype=np.float32)
    climatology = np.asarray(dataset.imerg_climatology.values, dtype=np.float32)
    weights = np.asarray(dataset.area_weight_km2.values, dtype=np.float64)
    initializations = dataset.init.values.astype("datetime64[D]")
    case_ids = [np.datetime_as_string(value, unit="D") for value in initializations]
    frames = []
    for method in METHODS:
        prediction = np.asarray(
            dataset.prediction.sel({"method": method}).values, dtype=np.float32
        )
        frame = compute_case_metrics(
            truth,
            prediction,
            truth - climatology,
            prediction - climatology,
            weights,
            predictor=method,
            case_ids=case_ids,
            leads=np.arange(1, 7),
            valid_mask=weights > 0.0,
        ).rename(
            columns={"predictor": "method", "case_id": "init", "lead": "lead_week"}
        )
        frame.insert(0, "split", "test")
        frame.insert(2, "year", pd.DatetimeIndex(frame.init).year)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def verify_selection(run: Path, manifest: dict) -> dict:
    selection_path = run / manifest.get("selection_file", "selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen" or selection.get("smoke"):
        raise AssertionError("selection is not a frozen full-run decision")
    if selection.get("selection_scope") != "validation_only":
        raise AssertionError("model selection was not scoped to validation only")
    if selection.get("test_predictions_created") is not False:
        raise AssertionError("selection record was not frozen before test prediction")
    if selection.get("train_years") != list(range(2002, 2018)):
        raise AssertionError("selection train years differ")
    if selection.get("validation_years") != [2018, 2019]:
        raise AssertionError("selection validation years differ")
    if selection.get("test_years_quarantined_during_selection") != [2020, 2021]:
        raise AssertionError("test years were not declared quarantined")

    selected = selection.get("selected_model")
    if selected not in ("log_bias", *CANDIDATES):
        raise AssertionError(f"unknown selected model: {selected}")
    if not 0.0 <= float(selection.get("selected_alpha", -1.0)) <= 1.0:
        raise AssertionError("selected alpha is outside [0, 1]")
    if manifest.get("selected_model") != selected or not np.isclose(
        float(manifest.get("selected_alpha")),
        float(selection["selected_alpha"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("manifest and frozen selection differ")
    if manifest.get("selection_frozen_utc") != selection.get("frozen_utc"):
        raise AssertionError("manifest selection timestamp differs")
    if sha256_file(selection_path) != manifest.get("selection_sha256"):
        raise AssertionError("frozen selection hash differs")
    checkpoint_hashes = {
        candidate: [
            record["checkpoint_sha256"]
            for record in manifest["training"][candidate]["runs"]
        ]
        for candidate in CANDIDATES
    }
    if selection.get("checkpoint_sha256") != checkpoint_hashes:
        raise AssertionError("frozen selection references different checkpoints")

    frozen = parse_utc(selection["frozen_utc"])
    test_started = parse_utc(manifest["test_evaluation_started_utc"])
    completed = parse_utc(manifest["created_utc"])
    if not frozen < test_started <= completed:
        raise AssertionError("selection was not frozen before test evaluation")

    table = pd.read_csv(run / "metrics" / "validation_selection.csv")
    if set(table.candidate) != set(CANDIDATES) or len(table) != len(CANDIDATES):
        raise AssertionError("validation candidate table differs")
    robust = table.loc[table.improves_every_validation_year.astype(bool)]
    if robust.empty:
        expected_model, expected_alpha = "log_bias", 0.0
    else:
        row = robust.loc[robust.w5_w6_case_mean_rmse.idxmin()]
        expected_model, expected_alpha = str(row.candidate), float(row.alpha)
    if selected != expected_model or not np.isclose(
        float(selection["selected_alpha"]), expected_alpha, rtol=0.0, atol=1.0e-12
    ):
        raise AssertionError("frozen decision cannot be rebuilt from validation scores")
    return selection


def verify_training(run: Path, manifest: dict) -> None:
    training = manifest.get("training", {})
    if set(training) != set(CANDIDATES):
        raise AssertionError("training candidates differ")
    parameter_counts = {}
    required_history = {
        "epoch",
        "train_loss",
        "validation_loss",
        "train_smooth_l1",
        "validation_smooth_l1",
        "train_mean_spatial_acc",
        "validation_mean_spatial_acc",
        "train_mean_bias_squared",
        "validation_mean_bias_squared",
    }
    for candidate in CANDIDATES:
        metadata = training[candidate]
        if metadata.get("train_case_count") != 560:
            raise AssertionError(f"wrong training count for {candidate}")
        if metadata.get("validation_case_count") != 70:
            raise AssertionError(f"wrong validation count for {candidate}")
        if metadata.get("seeds") != [42, 43, 44] or len(metadata.get("runs", [])) != 3:
            raise AssertionError(f"fixed-seed ensemble differs for {candidate}")
        parameter_counts[candidate] = int(metadata.get("parameter_count", 0))
        for record in metadata["runs"]:
            checkpoint = run / record["checkpoint"]
            history_path = run / record["history"]
            if not checkpoint.is_file() or not history_path.is_file():
                raise AssertionError(f"missing model artifact for {candidate}")
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise AssertionError(f"checkpoint hash differs: {checkpoint}")
            history = pd.read_csv(history_path)
            if not required_history.issubset(history.columns):
                raise AssertionError(f"training history columns differ: {history_path}")
            if not np.isfinite(history[list(required_history)].to_numpy(dtype=np.float64)).all():
                raise AssertionError(f"non-finite training history: {history_path}")
            selected = history.loc[
                history.epoch.astype(int).eq(int(record["best_epoch"]))
            ]
            if len(selected) != 1 or not np.isclose(
                float(selected.validation_loss.iloc[0]),
                float(record["best_validation_loss"]),
                rtol=0.0,
                atol=2.0e-12,
            ):
                raise AssertionError(f"best history row differs: {history_path}")
            if not np.isclose(
                float(history.validation_loss.min()),
                float(record["best_validation_loss"]),
                rtol=0.0,
                atol=2.0e-12,
            ):
                raise AssertionError(f"checkpoint is not validation-selected: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu")
            if (
                int(payload["seed"]) != int(record["seed"])
                or int(payload["best_epoch"]) != int(record["best_epoch"])
                or not np.isclose(
                    float(payload["best_validation_loss"]),
                    float(record["best_validation_loss"]),
                    rtol=0.0,
                    atol=2.0e-12,
                )
                or not payload.get("model_state_dict")
            ):
                raise AssertionError(f"checkpoint metadata differs: {checkpoint}")
    if not (
        parameter_counts["big_spatial"] > parameter_counts["small_temporal"]
        and parameter_counts["big_temporal"] > parameter_counts["small_temporal"]
    ):
        raise AssertionError("large candidates are not larger than the control")


def verify_predictions(run: Path, selection: dict) -> xr.Dataset:
    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as source:
        dataset = source.load()
    if tuple(dataset.method.values.tolist()) != METHODS:
        raise AssertionError("prediction method ordering differs")
    expected_sizes = {
        "method": 6,
        "init": 70,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }
    if dict(dataset.sizes) != expected_sizes:
        raise AssertionError(f"prediction dimensions differ: {dict(dataset.sizes)}")
    years, counts = np.unique(pd.DatetimeIndex(dataset.init.values).year, return_counts=True)
    if not np.array_equal(years, [2020, 2021]) or not np.array_equal(counts, [35, 35]):
        raise AssertionError("test initialization years differ")
    if (
        dataset.attrs.get("selection_scope") != "validation_only"
        or dataset.attrs.get("selected_model") != selection["selected_model"]
        or not np.isclose(
            float(dataset.attrs.get("selected_alpha")),
            float(selection["selected_alpha"]),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise AssertionError("prediction-store selection metadata differs")

    weights = np.asarray(dataset.area_weight_km2.values, dtype=np.float64)
    support = weights > 0.0
    if int(support.sum()) != 174:
        raise AssertionError("supported-cell count differs")
    values = np.asarray(dataset.prediction.values, dtype=np.float32)
    log_bias = np.asarray(dataset.prediction.sel({"method": "log_bias"}).values)
    for method in METHODS[2:]:
        prediction = np.asarray(dataset.prediction.sel({"method": method}).values)
        if not np.array_equal(prediction[:, :4], log_bias[:, :4], equal_nan=True):
            raise AssertionError(f"{method} violates W1-W4 identity")
    if not np.isfinite(values[:, :, 4:, support]).all():
        raise AssertionError("W5-W6 predictions are not finite on support")
    if np.any(values[:, :, 4:, support] < 0.0):
        raise AssertionError("W5-W6 predictions are negative on support")
    if not np.isnan(values[..., ~support]).all():
        raise AssertionError("predictions outside support are not NaN")
    selected_source = selection["selected_model"]
    selected_prediction = np.asarray(
        dataset.prediction.sel({"method": "selected_model"}).values
    )
    expected_prediction = np.asarray(
        dataset.prediction.sel({"method": selected_source}).values
    )
    if not np.array_equal(selected_prediction, expected_prediction, equal_nan=True):
        raise AssertionError("selected-model alias differs from its frozen source")
    return dataset


def verify_metrics(run: Path, dataset: xr.Dataset) -> None:
    rebuilt = rebuild_case_metrics(dataset)
    stored = pd.read_csv(run / "metrics" / "case_metrics.csv")
    pd.testing.assert_frame_equal(
        stored,
        rebuilt,
        check_dtype=False,
        check_exact=False,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def verify_hashes(run: Path, manifest: dict) -> None:
    required = {
        "selection.json",
        "metrics/case_metrics.csv",
        "metrics/validation_selection.csv",
        "predictions.zarr",
    }
    artifacts = manifest.get("artifacts", {})
    if not required.issubset(artifacts):
        raise AssertionError("required artifact hashes are missing")
    for relative, expected in artifacts.items():
        path = run / relative
        if not path.exists():
            raise AssertionError(f"missing hashed artifact: {relative}")
        actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
        if actual != expected:
            raise AssertionError(f"artifact hash differs: {relative}")


def verify(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise AssertionError("run is not a completed full experiment")
    if manifest.get("split_counts") != {"train": 560, "validation": 70, "test": 70}:
        raise AssertionError("split counts differ")
    if manifest.get("split_years") != {
        "train": list(range(2002, 2018)),
        "validation": [2018, 2019],
        "test": [2020, 2021],
    }:
        raise AssertionError("split years differ")
    if (
        manifest.get("test_count_used") != 70
        or manifest.get("active_leads") != [5, 6]
        or manifest.get("support_cells") != 174
    ):
        raise AssertionError("test, lead, or support contract differs")

    selection = verify_selection(run, manifest)
    verify_training(run, manifest)
    dataset = verify_predictions(run, selection)
    verify_metrics(run, dataset)
    verify_hashes(run, manifest)
    print(
        "PASS: manifest, 560/70/70 splits, validation-only selection timing, "
        "histories, checkpoints, W1-W4 identity, W5-W6 values, recomputed metrics, "
        "and artifact hashes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    verify((args.run or latest_full()).resolve())


if __name__ == "__main__":
    main()
