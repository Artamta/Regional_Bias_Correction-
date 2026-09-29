"""Focused tests for the independent prediction-level bridge auditor."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import audit_india_s2s_probabilistic_bridge_predictions as audit


def _members(value: float = 1.0) -> np.ndarray:
    return np.full((1, 50, 6, 27, 27), value, dtype=np.float32)


def test_target_dates_are_exactly_init_plus_1_through_42() -> None:
    starts = np.asarray(("2024-11-18",), dtype="datetime64[D]")
    dates = audit.target_dates(starts)
    assert dates.shape == (1, 6, 7)
    assert dates[0, 0, 0] == np.datetime64("2024-11-19")
    assert dates[0, 0, -1] == np.datetime64("2024-11-25")
    assert dates[0, -1, 0] == np.datetime64("2024-12-24")
    assert dates[0, -1, -1] == np.datetime64("2024-12-30")
    assert not np.any(dates == starts[0])
    assert len(audit._array_sha256(dates)) == 64


def test_affine_identity_and_neural_log_spread_are_independent() -> None:
    raw = np.linspace(0.0, 20.0, num=50 * 6 * 27 * 27, dtype=np.float32).reshape(
        1, 50, 6, 27, 27
    )
    delta = np.zeros((1, 6, 27, 27), dtype=np.float32)
    unit = np.ones_like(delta)
    identity = audit.reconstruct_affine_members(raw, delta, unit)
    neural = audit.reconstruct_neural_members(raw, delta, np.zeros_like(delta))
    np.testing.assert_allclose(identity, raw, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(neural, identity)


def test_moment_reconstruction_uses_valid_midpoint_month() -> None:
    raw = _members(0.0)
    delta = np.zeros((6, 12, 27, 27), dtype=np.float32)
    spread = np.ones((6, 12), dtype=np.float32)
    for month in range(12):
        delta[:, month] = np.float32((month + 1) / 100.0)
    starts = np.asarray(("2020-01-20",), dtype="datetime64[D]")
    corrected = audit.reconstruct_moment_members(raw, starts, delta, spread)
    midpoint_months = pd.DatetimeIndex(
        (
            starts[:, None]
            + np.asarray((4, 11, 18, 25, 32, 39), dtype="timedelta64[D]")[None]
        ).reshape(-1)
    ).month
    expected = np.expm1(np.asarray(midpoint_months, dtype=np.float32) / 100.0)
    for lead_index in range(6):
        np.testing.assert_allclose(
            corrected[0, :, lead_index], expected[lead_index], rtol=1.0e-6
        )


def test_case_metrics_match_hand_calculation() -> None:
    y, x = np.indices((27, 27))
    field = (5.0 + 0.1 * y + 0.03 * x).astype(np.float32)
    truth = np.broadcast_to(field, (1, 6, 27, 27)).copy()
    offsets = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    members = truth[:, None] + np.float32(0.5) + offsets[None, :, None, None, None]
    weights = {name: np.ones((27, 27), dtype=np.float64) for name in audit.REGIONS}
    rows = audit.score_member_batch(
        method="raw_fuxi",
        starts=np.asarray(("2020-01-02",), dtype="datetime64[D]"),
        members=members,
        truth=truth,
        observation_normal=np.zeros_like(truth),
        observation_support=np.ones_like(truth),
        forecast_normal=np.zeros_like(truth),
        region_weights=weights,
    )
    assert len(rows) == 30
    row = rows[0]
    member_error = 0.5 + offsets.astype(np.float64)
    pairwise = np.abs(member_error[:, None] - member_error[None, :])
    expected_crps = np.mean(np.abs(member_error)) - 0.5 * np.mean(pairwise)
    expected_variance = np.var(member_error, ddof=0)
    assert row["rmse"] == pytest.approx(0.5, abs=2.0e-7)
    assert row["mae"] == pytest.approx(0.5, abs=2.0e-7)
    assert row["bias"] == pytest.approx(0.5, abs=2.0e-7)
    assert row["acc"] == pytest.approx(1.0, abs=1.0e-12)
    assert row["crps"] == pytest.approx(expected_crps, abs=5.0e-8)
    assert row["coverage90"] == pytest.approx(1.0)
    assert row["ensemble_variance"] == pytest.approx(expected_variance, abs=5.0e-8)
    assert row["mean_squared_error"] == pytest.approx(0.25, abs=2.0e-7)
    assert row["spread_skill_ratio"] == pytest.approx(
        np.sqrt(expected_variance) / 0.5, abs=2.0e-7
    )


def test_comparison_accepts_matching_nan_and_rejects_changed_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_CASE_ROWS", 1)
    row = {
        "method": "raw_fuxi",
        "init": "2020-01-02",
        "region": "all_india",
        "lead_week": 1,
        "valid_cell_count": 171,
        **{metric: 1.0 for metric in audit.FLOAT_METRICS},
    }
    row["spread_skill_ratio"] = np.nan
    first = pd.DataFrame([row])
    comparison, receipt = audit.compare_case_metrics(first, first.copy(), atol=1e-6, rtol=1e-7)
    assert receipt["passed"]
    assert comparison.failure_count.sum() == 0
    changed = first.copy()
    changed.loc[0, "crps"] = 2.0
    _, failed = audit.compare_case_metrics(first, changed, atol=1e-6, rtol=1e-7)
    assert not failed["passed"]
    assert failed["failure_count_by_metric"]["crps"] == 1


def test_auditor_does_not_import_either_scoring_implementation() -> None:
    source = inspect.getsource(audit)
    assert "import india_s2s_probabilistic_bridge_score" not in source
    assert "import india_s2s_benchmark_scoring" not in source
