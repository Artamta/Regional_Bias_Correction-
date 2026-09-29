#!/usr/bin/env python
"""Verify a completed full-archive late-lead experiment from saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    "spatial_unet",
    "spatiotemporal_unet",
    "lead_adaptive_hybrid",
)


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
        (HERE / "results" / "fuxi_imerg_full_archive_latelead").glob("full_*")
    )
    if not runs:
        raise FileNotFoundError("no full late-lead result found")
    return runs[-1]


def rebuild_case_metrics(dataset: xr.Dataset) -> pd.DataFrame:
    truth = np.asarray(dataset.truth_imerg.load(), dtype=np.float32)
    climatology = np.asarray(dataset.imerg_climatology.load(), dtype=np.float32)
    weights = np.asarray(dataset.area_weight_km2.load(), dtype=np.float64)
    initializations = dataset.init.values.astype("datetime64[D]")
    case_ids = [np.datetime_as_string(value, unit="D") for value in initializations]
    frames = []
    for method in METHODS:
        prediction = np.asarray(
            dataset.prediction.sel({"method": method}).load(), dtype=np.float32
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


def verify(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise AssertionError("run is not a completed full experiment")
    if manifest["split_counts"] != {"train": 595, "validation": 35, "test": 70}:
        raise AssertionError("split counts differ")
    if manifest["active_leads"] != [5, 6] or manifest["support_cells"] != 174:
        raise AssertionError("lead or spatial contract differs")

    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as source:
        dataset = source.load()
    if tuple(dataset.method.values.tolist()) != METHODS:
        raise AssertionError("method ordering differs")
    if dataset.sizes != {
        "method": 5,
        "init": 70,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }:
        raise AssertionError(f"prediction dimensions differ: {dataset.sizes}")
    years, counts = np.unique(
        pd.DatetimeIndex(dataset.init.values).year, return_counts=True
    )
    if not np.array_equal(years, [2020, 2021]) or not np.array_equal(counts, [35, 35]):
        raise AssertionError("test initialization years differ")
    weights = np.asarray(dataset.area_weight_km2.values, dtype=np.float64)
    support = weights > 0.0
    if int(support.sum()) != 174:
        raise AssertionError("supported-cell count differs")
    log_bias = np.asarray(
        dataset.prediction.sel({"method": "log_bias"}).values, dtype=np.float32
    )
    for method in ("spatial_unet", "spatiotemporal_unet", "lead_adaptive_hybrid"):
        prediction = np.asarray(
            dataset.prediction.sel({"method": method}).values, dtype=np.float32
        )
        if not np.array_equal(prediction[:, :4], log_bias[:, :4], equal_nan=True):
            raise AssertionError(f"{method} violates W1-W4 identity")
    values = np.asarray(dataset.prediction.values, dtype=np.float32)
    if not np.isfinite(values[..., support]).all() or np.any(values[..., support] < 0.0):
        raise AssertionError("prediction values are invalid on support")
    if not np.isnan(values[..., ~support]).all():
        raise AssertionError("prediction values outside support are not NaN")

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
    headline = pd.read_csv(run / "metrics" / "headline_metrics.csv").set_index("method")
    for method in METHODS:
        selected = rebuilt.loc[rebuilt.method.eq(method)]
        checks = {
            "acc": selected.acc.mean(),
            "rmse_mm_day": selected.rmse.mean(),
            "mae_mm_day": selected.mae.mean(),
            "bias_mm_day": selected.bias.mean(),
        }
        for column, expected in checks.items():
            if not np.isclose(headline.loc[method, column], expected, rtol=0.0, atol=2.0e-13):
                raise AssertionError(f"headline mismatch: {method}, {column}")

    intervals = pd.read_csv(run / "metrics" / "paired_skill.csv")
    case_order = [np.datetime_as_string(value, unit="D") for value in dataset.init.values]
    scopes = {
        "W1-W6": (1, 2, 3, 4, 5, 6),
        "W1-W2": (1, 2),
        "W3-W4": (3, 4),
        "W5-W6": (5, 6),
        **{f"W{lead}": (lead,) for lead in range(1, 7)},
    }
    for row in intervals.itertuples(index=False):
        leads = scopes[row.lead_scope]
        selected = rebuilt.loc[
            rebuilt.lead_week.isin(leads)
            & rebuilt.method.isin((row.method, row.baseline))
        ]
        pivot = selected.pivot_table(
            index="init", columns="method", values=row.metric, aggfunc="mean"
        ).reindex(case_order)
        model = pivot[row.method].to_numpy(dtype=np.float64)
        baseline = pivot[row.baseline].to_numpy(dtype=np.float64)
        if row.metric == "acc":
            effect = model.mean() - baseline.mean()
        else:
            effect = 100.0 * (baseline.mean() - model.mean()) / baseline.mean()
        if not np.isclose(effect, row.effect, rtol=0.0, atol=2.0e-13):
            raise AssertionError(
                f"paired point effect mismatch: {row.method}, {row.baseline}, "
                f"{row.lead_scope}, {row.metric}"
            )

    for name in ("spatial", "temporal"):
        for record in manifest["training"][name]["runs"]:
            checkpoint = run / record["checkpoint"]
            history = pd.read_csv(
                str(checkpoint).replace("checkpoints/best.pt", "logs/training_history.csv")
            )
            selected = history.loc[history.epoch.eq(record["best_epoch"])]
            if len(selected) != 1 or not np.isclose(
                selected.validation_loss.iloc[0],
                record["best_validation_loss"],
                rtol=0.0,
                atol=2.0e-13,
            ):
                raise AssertionError(f"training selection differs: {name}, {record['seed']}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload["best_epoch"]) != int(record["best_epoch"]):
                raise AssertionError("checkpoint epoch differs")

    for stem in (
        "00_training_curves",
        "01_skill_by_lead",
        "02_direct_added_value",
        "03_late_spatial_added_value",
    ):
        for suffix in (".png", ".pdf"):
            path = run / "figures" / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size < 10_000:
                raise AssertionError(f"missing or empty figure: {path}")

    for relative, expected in manifest["artifacts"].items():
        path = run / relative
        actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
        if actual != expected:
            raise AssertionError(f"artifact hash differs: {relative}")
    print(
        "PASS: manifest, dimensions, splits, W1-W4 identity, values, metrics, "
        "paired effects, checkpoints, figures, and artifact hashes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    verify((args.run or latest_full()).resolve())


if __name__ == "__main__":
    main()
