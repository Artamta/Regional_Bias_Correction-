"""Tests for the artifact-only locked JJAS month/lead diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import plot_locked_jjas_month_lead_diagnostic as diagnostic  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dates() -> pd.DatetimeIndex:
    values: list[pd.Timestamp] = []
    for year in diagnostic.TEST_YEARS:
        for month, count in diagnostic.EXPECTED_MONTH_COUNTS_PER_YEAR.items():
            values.extend(pd.date_range(f"{year}-{month:02d}-02", periods=count, freq="3D"))
    return pd.DatetimeIndex(values)


def _locked_fixture(root: Path) -> Path:
    metrics = root / "metrics"
    metrics.mkdir(parents=True)
    rows = []
    for method in ("raw_fuxi", "log_bias", "corrected"):
        for date in _dates():
            for lead in diagnostic.LEAD_WEEKS:
                if method == "raw_fuxi":
                    acc, rmse, bias = 0.20 + 0.01 * lead, 4.0, 1.0
                elif method == "corrected":
                    acc, rmse, bias = 0.30 + 0.01 * lead, 3.0, 0.25
                else:
                    acc, rmse, bias = 0.22 + 0.01 * lead, 3.8, 0.5
                rows.append(
                    {
                        "method": method,
                        "predictor": method,
                        "case_id": date.strftime("%Y-%m-%d"),
                        "lead": lead,
                        "region": "india",
                        "season": "ALL",
                        "valid_cells": 171,
                        "weight_sum": 3_207_490.0,
                        "acc": acc,
                        "rmse": rmse,
                        "mae": rmse * 0.7,
                        "bias": bias,
                        "negative_fraction": 0.0,
                    }
                )
    case_path = metrics / "test_case_metrics.csv"
    pd.DataFrame(rows).to_csv(case_path, index=False)
    manifest = {
        "status": "complete",
        "evaluation_role": diagnostic.EVALUATION_ROLE,
        "evaluation_scope": (
            "2020-2021 exploratory/reused hindcast test; not independent confirmation"
        ),
        "test_years": [2020, 2021],
        "selected_configuration": "physical_full_compact",
        "selection_locked_before_target_access": True,
        "selection_locked_before_test": True,
        "test_used_for_selection": False,
        "parameter_updates": 0,
        "reused_test_period": True,
        "genuine_independent_test": False,
        "artifacts": {"metrics/test_case_metrics.csv": _sha256(case_path)},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root


def test_summary_has_exact_month_lead_counts_and_paired_effects(tmp_path: Path) -> None:
    root = _locked_fixture(tmp_path / "locked")
    _, frame, _ = diagnostic.load_locked_case_metrics(root)
    first = diagnostic.summarize_month_lead(
        frame, replicates=80, block_length=3, seed=9
    )
    second = diagnostic.summarize_month_lead(
        frame, replicates=80, block_length=3, seed=9
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.shape[0] == 24
    counts = first.groupby("initialization_month").n_initializations.first().to_dict()
    assert counts == {6: 18, 7: 16, 8: 18, 9: 18}
    assert np.allclose(first.acc_difference, 0.10)
    assert np.allclose(first.acc_difference_ci95_lower, 0.10)
    assert np.allclose(first.acc_difference_ci95_upper, 0.10)
    assert np.allclose(first.rmse_reduction_pct, 25.0)
    assert np.allclose(first.rmse_reduction_ci95_lower_pct, 25.0)
    assert np.allclose(first.rmse_reduction_ci95_upper_pct, 25.0)
    assert first.acc_approximate_interval_above_zero.all()
    assert first.rmse_approximate_interval_above_zero.all()
    assert np.allclose(first.raw_bias_mean_mm_day, 1.0)
    assert np.allclose(first.corrected_bias_mean_mm_day, 0.25)
    assert np.allclose(first.absolute_bias_improvement_mm_day, 0.75)
    assert np.allclose(first.absolute_bias_improvement_ci95_lower_mm_day, 0.75)
    assert np.allclose(first.absolute_bias_improvement_ci95_upper_mm_day, 0.75)
    assert first.absolute_bias_approximate_interval_above_zero.all()


def test_loader_rejects_tampered_case_metrics(tmp_path: Path) -> None:
    root = _locked_fixture(tmp_path / "locked")
    case_path = root / "metrics" / "test_case_metrics.csv"
    case_path.write_text(
        case_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(diagnostic.DiagnosticContractError, match="checksum"):
        diagnostic.load_locked_case_metrics(root)


def test_build_writes_clean_artifact_only_package(tmp_path: Path) -> None:
    root = _locked_fixture(tmp_path / "locked")
    output = tmp_path / "diagnostic"
    result = diagnostic.build_diagnostic(
        root,
        output,
        replicates=80,
        block_length=3,
        seed=11,
        dpi=110,
    )

    assert result == output.resolve()
    expected = {
        "diagnostic_manifest.json",
        "jjas_initialization_month_by_lead_summary.csv",
        "jjas_month_lead_improvement_tradeoffs.png",
        "jjas_month_lead_improvement_tradeoffs.pdf",
        "jjas_month_lead_paired_uncertainty.png",
        "jjas_month_lead_paired_uncertainty.pdf",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads(
        (output / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["evaluation_role"] == diagnostic.EVALUATION_ROLE
    assert manifest["genuine_independent_test"] is False
    assert manifest["forecast_prediction_observation_or_target_arrays_opened"] is False
    assert manifest["metrics_computed_from_locked_case_metrics_only"] is True
    assert manifest["source_artifacts_read"] == [
        "manifest.json",
        "metrics/test_case_metrics.csv",
    ]
    assert manifest["uncertainty"]["p_values_computed"] is False
    assert manifest["uncertainty"]["significance_claimed"] is False
    assert len(manifest["artifacts"]) == 5
    summary = pd.read_csv(output / "jjas_initialization_month_by_lead_summary.csv")
    assert not any("p_value" in name.lower() for name in summary.columns)
    with pytest.raises(FileExistsError, match="fresh output"):
        diagnostic.build_diagnostic(root, output, replicates=10)


def test_output_cannot_modify_locked_evaluation(tmp_path: Path) -> None:
    root = _locked_fixture(tmp_path / "locked")
    with pytest.raises(ValueError, match="must not be inside"):
        diagnostic.build_diagnostic(root, root / "derived", replicates=10)
