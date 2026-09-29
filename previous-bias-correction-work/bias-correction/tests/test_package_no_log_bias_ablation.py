"""Focused tests for the no-log-bias anchor-ablation packager."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


EVALUATE = Path(__file__).resolve().parents[1] / "evaluate"
if str(EVALUATE) not in sys.path:
    sys.path.insert(0, str(EVALUATE))

import package_no_log_bias_ablation as package_ablation  # noqa: E402


def _dates(years: tuple[int, int]) -> list[str]:
    return [
        f"{years[0]}-06-02",
        f"{years[0]}-06-09",
        f"{years[1]}-06-02",
        f"{years[1]}-06-09",
    ]


def _case_metrics(
    *,
    split: str,
    dates: list[str],
    raw_identity: bool,
) -> pd.DataFrame:
    rows = []
    for method in (
        package_ablation.RAW,
        package_ablation.LOG_BIAS,
        package_ablation.MATCHED_MODEL,
    ):
        for case_index, initialization in enumerate(dates):
            for lead in package_ablation.EXPECTED_LEADS:
                raw_rmse = 4.0 + 0.08 * case_index + 0.12 * lead
                raw_mae = 2.5 + 0.04 * case_index + 0.06 * lead
                if method == package_ablation.RAW:
                    rmse, mae, acc, pcc, bias = (
                        raw_rmse,
                        raw_mae,
                        0.18 + 0.02 * lead,
                        0.42 + 0.01 * lead,
                        -0.10,
                    )
                elif method == package_ablation.LOG_BIAS:
                    rmse, mae, acc, pcc, bias = (
                        0.92 * raw_rmse,
                        0.91 * raw_mae,
                        0.25 + 0.02 * lead,
                        0.48 + 0.01 * lead,
                        -0.25,
                    )
                else:
                    multiplier = 0.865 if raw_identity else 0.87
                    rmse, mae, acc, pcc, bias = (
                        multiplier * raw_rmse,
                        multiplier * raw_mae,
                        (0.34 if raw_identity else 0.33) + 0.02 * lead,
                        (0.55 if raw_identity else 0.54) + 0.01 * lead,
                        -0.30,
                    )
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "year": int(initialization[:4]),
                        "init": initialization,
                        "lead_week": lead,
                        "region": "india",
                        "season": "ALL",
                        "valid_cells": 11,
                        "weight_sum": 11.0,
                        "acc": acc,
                        "rmse": rmse,
                        "mae": mae,
                        "bias": bias,
                        "negative_fraction": 0.0,
                        "spatial_acc_common_imd": acc,
                        "mse": rmse**2,
                        "climatology_mse": 25.0,
                        "pcc": pcc,
                    }
                )
    return pd.DataFrame(rows)


def _prediction_store(
    path: Path,
    *,
    dates: list[str],
    raw_identity: bool,
) -> None:
    latitude = np.array([3.0, 1.5, 0.0], dtype=np.float64)
    longitude = np.array([60.0, 61.5, 63.0, 64.5], dtype=np.float64)
    shape = (len(dates), len(package_ablation.EXPECTED_LEADS), 3, 4)
    truth = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 25.0 + 3.0
    climatology = np.full(shape, 3.0, dtype=np.float32)
    pattern = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 1, 3, 4)
    raw = np.maximum(0.0, truth + 1.4 * pattern)
    log_bias = np.maximum(0.0, truth + 1.0 * pattern)
    neural_scale = 0.72 if raw_identity else 0.78
    normal = np.maximum(0.0, truth + neural_scale * pattern)
    prediction = np.stack((raw, log_bias, normal)).astype(np.float32)
    weights = np.ones((3, 4), dtype=np.float64)
    weights[0, 0] = 0.0
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                prediction,
            ),
            "truth_imd": (
                ("init", "lead_week", "latitude", "longitude"),
                truth,
            ),
            "fixed_imd_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                climatology,
            ),
            "area_weight_km2": (("latitude", "longitude"), weights),
        },
        coords={
            "method": [
                package_ablation.RAW,
                package_ablation.LOG_BIAS,
                package_ablation.MATCHED_MODEL,
            ],
            "init": np.asarray(dates, dtype="datetime64[ns]"),
            "lead_week": np.asarray(package_ablation.EXPECTED_LEADS, dtype=np.int16),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    dataset.to_zarr(path, mode="w", consolidated=True)


def _source_run(root: Path, *, role: str) -> Path:
    run = root / role
    (run / "metrics").mkdir(parents=True)
    (run / "models").mkdir()
    validation_dates = _dates(package_ablation.EXPECTED_VALIDATION_YEARS)
    test_dates = _dates(package_ablation.EXPECTED_TEST_YEARS)
    is_raw_identity = role == "raw_identity"
    _case_metrics(
        split="validation",
        dates=validation_dates,
        raw_identity=is_raw_identity,
    ).to_csv(run / "metrics" / "validation_case_metrics.csv", index=False)
    _case_metrics(split="test", dates=test_dates, raw_identity=is_raw_identity).to_csv(
        run / "metrics" / "case_metrics.csv", index=False
    )
    (run / "normalization.json").write_text(
        json.dumps({"same_training_only_normalization": True}) + "\n",
        encoding="utf-8",
    )
    selection = {
        "status": "frozen",
        "smoke": False,
        "selected_model": package_ablation.MATCHED_MODEL,
        "selected_alpha": 1.0,
    }
    (run / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    if is_raw_identity:
        np.savez_compressed(
            run / "models" / "training_anchor_contract.npz",
            anchor_kind=np.asarray(package_ablation.RAW),
            target_scale=np.ones(6, dtype=np.float32),
            fitted_target_years=np.asarray(
                package_ablation.EXPECTED_TRAIN_YEARS, dtype=np.int16
            ),
        )
    else:
        np.savez_compressed(
            run / "models" / "log_bias_anchor.npz",
            target_scale=np.ones(6, dtype=np.float32),
        )
    _prediction_store(
        run / "predictions.zarr",
        dates=test_dates,
        raw_identity=is_raw_identity,
    )

    training = {
        "parameter_count": 144_689,
        "seeds": [42, 43, 44],
        "batch_size": 32,
        "learning_rate": 2.0e-4,
        "weight_decay": 2.0e-3,
        "dropout": 0.30,
        "max_epochs": 100,
        "patience": 15,
    }
    manifest = {
        "status": "complete",
        "smoke": False,
        "test_count_used": 4,
        "selected_model": package_ablation.MATCHED_MODEL,
        "selected_alpha": 1.0,
        "split_years": {
            "train": list(package_ablation.EXPECTED_TRAIN_YEARS),
            "validation": list(package_ablation.EXPECTED_VALIDATION_YEARS),
            "test": list(package_ablation.EXPECTED_TEST_YEARS),
        },
        "active_leads": list(package_ablation.EXPECTED_LEADS),
        "quarantined_final_initialization_years": [2025],
        "training": {package_ablation.MATCHED_MODEL: training},
        "artifacts": {},
    }
    if is_raw_identity:
        manifest.update(
            {
                "training_anchor": package_ablation.RAW,
                "uses_fitted_log_bias_in_neural_training": False,
                "log_bias_role": "reporting_only",
            }
        )
    for relative in (
        "selection.json",
        "normalization.json",
        "metrics/case_metrics.csv",
        "metrics/validation_case_metrics.csv",
        "predictions.zarr",
    ):
        path = run / relative
        manifest["artifacts"][relative] = (
            package_ablation.sha256_tree(path)
            if path.is_dir()
            else package_ablation.sha256_file(path)
        )
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return run


def test_circular_bootstrap_is_deterministic_and_year_stratified() -> None:
    dates = [
        *(f"2020-06-{day:02d}" for day in range(1, 7)),
        *(f"2021-06-{day:02d}" for day in range(1, 7)),
    ]
    first = package_ablation.circular_stratified_indices(
        dates, n_resamples=40, block_length=4, seed=17
    )
    second = package_ablation.circular_stratified_indices(
        dates, n_resamples=40, block_length=4, seed=17
    )
    assert np.array_equal(first, second)
    assert first.shape == (40, 12)
    assert np.all(first[:, :6] < 6)
    assert np.all(first[:, 6:] >= 6)
    for row in first:
        assert np.all(row[1:4] == (row[:3] + 1) % 6)
        local = row[6:10] - 6
        assert np.all(local[1:] == (local[:3] + 1) % 6)


def test_paired_candidate_skill_uses_positive_raw_identity_convention() -> None:
    def frame(years: tuple[int, int], split: str) -> pd.DataFrame:
        rows = []
        for initialization in _dates(years):
            for lead in (1, 2):
                for method, rmse, mae, acc, pcc in (
                    (package_ablation.ANCHORED, 10.0, 5.0, 0.20, 0.40),
                    (package_ablation.RAW_IDENTITY, 9.0, 4.5, 0.25, 0.42),
                ):
                    rows.append(
                        {
                            "split": split,
                            "method": method,
                            "init": initialization,
                            "lead_week": lead,
                            "rmse": rmse,
                            "mae": mae,
                            "acc": acc,
                            "pcc": pcc,
                        }
                    )
        return pd.DataFrame(rows)

    effects = package_ablation.build_paired_effects(
        frame(package_ablation.EXPECTED_VALIDATION_YEARS, "validation"),
        frame(package_ablation.EXPECTED_TEST_YEARS, "test"),
        n_resamples=50,
        block_length=2,
        seed=package_ablation.BOOTSTRAP_SEED,
        leads=(1, 2),
    )
    rmse = effects.loc[
        effects.split.eq("test") & effects.scope.eq("W1-W6") & effects.metric.eq("rmse")
    ].iloc[0]
    acc = effects.loc[
        effects.split.eq("test") & effects.scope.eq("W1-W6") & effects.metric.eq("acc")
    ].iloc[0]
    assert rmse.raw_identity_candidate_skill == pytest.approx(10.0)
    assert rmse.interval_lower == pytest.approx(10.0)
    assert rmse.interval_upper == pytest.approx(10.0)
    assert acc.raw_identity_candidate_skill == pytest.approx(0.05)
    assert (effects.bootstrap_seed == package_ablation.BOOTSTRAP_SEED).all()


def test_baseline_identity_rejects_metric_drift() -> None:
    dates = _dates(package_ablation.EXPECTED_TEST_YEARS)
    anchored = _case_metrics(split="test", dates=dates, raw_identity=False)
    raw_identity = _case_metrics(split="test", dates=dates, raw_identity=True)
    target = raw_identity.index[raw_identity.method.eq(package_ablation.RAW)][0]
    raw_identity.loc[target, "rmse"] += 0.01
    with pytest.raises(ValueError, match="baseline identity"):
        package_ablation.validate_baseline_identity(
            anchored, raw_identity, split="test"
        )


def test_package_writes_atomic_honest_comparison_with_ten_figures(
    tmp_path: Path,
) -> None:
    anchored = _source_run(tmp_path / "sources", role="anchored")
    raw_identity = _source_run(tmp_path / "sources", role="raw_identity")
    output = package_ablation.package(
        anchored,
        raw_identity,
        tmp_path / "comparison",
        expected_cases=4,
        n_resamples=40,
        block_length=2,
        seed=package_ablation.BOOTSTRAP_SEED,
    )
    assert len(list((output / "figures").glob("*.png"))) == 5
    assert len(list((output / "figures").glob("*.pdf"))) == 5
    assert all(path.stat().st_size > 1_000 for path in (output / "figures").glob("*"))

    absolute = pd.read_csv(output / "metrics" / "absolute_metrics_long.csv")
    paired = pd.read_csv(output / "metrics" / "paired_anchor_effects.csv")
    assert set(absolute.method) == set(package_ablation.METHOD_ORDER)
    assert len(paired) == 2 * 7 * 4
    assert (paired.bootstrap_seed == package_ablation.BOOTSTRAP_SEED).all()
    assert (paired.p_value_computed == False).all()  # noqa: E712

    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "not independent confirmation" in report
    assert (
        "Positive candidate skill always means the raw-identity adapter is better"
        in report
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert (
        manifest["identity_checks"]["test_raw_and_log_bias_prediction_grids_exact"]
        is True
    )
    assert manifest["evidence_scope"]["final_2025_accessed"] is False
    assert manifest["uncertainty"]["seed"] == package_ablation.BOOTSTRAP_SEED
    assert len(manifest["artifacts"]) == 14

    with pytest.raises(FileExistsError):
        package_ablation.package(
            anchored,
            raw_identity,
            output,
            expected_cases=4,
            n_resamples=10,
            block_length=2,
        )
