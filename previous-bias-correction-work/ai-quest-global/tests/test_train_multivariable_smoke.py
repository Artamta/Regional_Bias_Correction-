"""Regression tests for the multi-variable synthetic optimization smoke."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_multivariable_smoke import (
    build_parser,
    make_synthetic_split,
    run_smoke,
    synthetic_spatial_weights,
)


def test_synthetic_split_has_complete_independent_contract() -> None:
    first = make_synthetic_split(3, 11, height=8, width=12)
    repeated = make_synthetic_split(3, 11, height=8, width=12)
    independent = make_synthetic_split(3, 12, height=8, width=12)

    assert first.features.shape == (3, 2, 38, 8, 12)
    assert first.anchor.shape == (3, 3, 2, 5, 8, 12)
    assert first.target.shape == (3, 3, 2, 8, 12)
    torch.testing.assert_close(first.anchor.sum(dim=3), torch.ones(3, 3, 2, 8, 12))
    torch.testing.assert_close(first.features, repeated.features)
    assert not torch.equal(first.features, independent.features)
    assert int(first.target.min()) >= 0 and int(first.target.max()) <= 4

    weights = synthetic_spatial_weights(8, 12)
    assert weights.shape == (3, 2, 8, 12)
    assert torch.all(weights >= 0.0)
    assert torch.all(weights.reshape(3, -1).sum(dim=1) > 0.0)


def test_training_smoke_writes_curves_checkpoint_and_numeric_evidence(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "multivariable_smoke"
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_directory),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--train-cases",
            "6",
            "--validation-cases",
            "3",
            "--height",
            "8",
            "--width",
            "8",
            "--batch-size",
            "3",
            "--base-channels",
            "1",
            "--attention-heads",
            "1",
            "--minimum-improvement",
            "0",
            "--seed",
            "9",
            "--skip-native-grid-check",
        ]
    )

    result = run_smoke(args)

    assert result == run_directory.resolve()
    assert (result / "best.pt").is_file()
    assert (result / "history.csv").is_file()
    expected_figures = {
        "training_history.png",
        "modal_quintile_accuracy.png",
        "validation_rps_by_variable.png",
        "validation_reliability.png",
    }
    assert {path.name for path in (result / "figures").glob("*.png")} == expected_figures
    assert all((result / "figures" / name).stat().st_size > 10_000 for name in expected_figures)

    with (result / "history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert float(rows[0]["validation_skill_vs_p0"]) == 0.0
    assert np.isfinite([float(value) for value in rows[-1].values()]).all()

    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "verified_synthetic_optimization_smoke"
    assert summary["submission_ready"] is False
    assert summary["initial_anchor_identity_error"] <= 1.0e-6
    assert summary["checkpoint_roundtrip_max_error"] <= 1.0e-7
    assert summary["validation_rps_improvement"] >= 0.0
    assert summary["native_grid_backward_check"] is None
    assert "not meteorological ACC" in " ".join(summary["limitations"])


def test_failed_improvement_retains_explicit_epoch_zero_fallback(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "fallback_smoke"
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_directory),
            "--epochs",
            "1",
            "--train-cases",
            "2",
            "--validation-cases",
            "2",
            "--height",
            "8",
            "--width",
            "8",
            "--batch-size",
            "2",
            "--base-channels",
            "1",
            "--attention-heads",
            "1",
            "--learning-rate",
            "0",
            "--minimum-improvement",
            "0.01",
            "--skip-native-grid-check",
        ]
    )

    result = run_smoke(args)

    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "synthetic_optimization_below_required_improvement"
    assert summary["minimum_improvement_passed"] is False
    assert summary["selected_system"] == "p0_anchor"
    assert summary["best_epoch"] == 0
    checkpoint = torch.load(result / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["metadata"]["trained"] is False
    assert checkpoint["metadata"]["optimization_attempted"] is True
    assert checkpoint["metadata"]["selected_system"] == "p0_anchor"
    assert checkpoint["metadata"]["minimum_improvement_passed"] is False
