"""Focused leakage, recentering, and physical-ranking tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_bias_aware_validation_sweep as sweep  # noqa: E402
from fuxi_adapter.baselines import apply_log_bias_correction  # noqa: E402


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    support = weights > 0.0
    selected = values[..., support].astype(np.float64)
    spatial = weights[support].astype(np.float64)
    return float(
        np.sum(selected * spatial, dtype=np.float64)
        / (selected.shape[0] * spatial.sum(dtype=np.float64))
    )


def _recenter_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dates = np.asarray(
        ["2002-06-01", "2003-06-01", "2018-06-01", "2019-06-01"],
        dtype="datetime64[D]",
    )
    raw = np.asarray(
        [
            [[[1.0, 3.0], [999.0, 2.0]]],
            [[[2.0, 4.0], [999.0, 1.0]]],
            [[[5.0, 5.0], [999.0, 5.0]]],
            [[[7.0, 7.0], [999.0, 7.0]]],
        ],
        dtype=np.float32,
    )
    truth = np.asarray(
        [
            [[[2.0, 5.0], [777.0, 3.0]]],
            [[[4.0, 7.0], [777.0, 2.0]]],
            [[[50.0, 50.0], [777.0, 50.0]]],
            [[[60.0, 60.0], [777.0, 60.0]]],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    delta = np.zeros((1, 12, 2, 2), dtype=np.float32)
    # Explicitly mirror the real contract: correction is NaN off IMD support.
    delta[:, :, 1, 0] = np.nan
    train = np.asarray([0, 1], dtype=np.int64)
    validation = np.asarray([2, 3], dtype=np.int64)
    return raw, truth, dates, weights, delta, train, validation


def test_recenter_closes_training_mean_and_ignores_nan_outside_support() -> None:
    raw, truth, dates, weights, delta, train, _ = _recenter_fixture()

    fitted = sweep.fit_physical_recentered_log_correction(
        raw,
        truth,
        dates,
        train,
        weights,
        delta,
        split_name="train",
    )
    prediction = apply_log_bias_correction(raw[train], dates[train], fitted.correction)

    assert np.isnan(fitted.correction.lead_month_residual[:, :, 1, 0]).all()
    assert np.isfinite(fitted.scalar_by_lead_month).all()
    assert _weighted_mean(prediction, weights) == pytest.approx(
        _weighted_mean(truth[train], weights), rel=2.0e-6, abs=2.0e-6
    )


def test_recenter_fit_is_train_only_when_validation_truth_changes() -> None:
    raw, truth, dates, weights, delta, train, validation = _recenter_fixture()
    first = sweep.fit_physical_recentered_log_correction(
        raw, truth, dates, train, weights, delta, split_name="train"
    )
    changed_truth = truth.copy()
    changed_truth[validation] = 1_000_000.0
    second = sweep.fit_physical_recentered_log_correction(
        raw, changed_truth, dates, train, weights, delta, split_name="train"
    )

    np.testing.assert_array_equal(
        first.scalar_by_lead_month, second.scalar_by_lead_month
    )
    np.testing.assert_array_equal(
        first.correction.lead_month_residual,
        second.correction.lead_month_residual,
    )


def test_recenter_solver_handles_zero_and_expands_upper_endpoint() -> None:
    base_log = np.asarray([[0.2, 0.4], [0.3, 0.5]], dtype=np.float64)
    weights = np.asarray([1.0, 3.0], dtype=np.float64)

    zero_scalar, zero_mean = sweep.solve_physical_mean_recenter_scalar(
        base_log, weights, 0.0
    )
    high_scalar, high_mean = sweep.solve_physical_mean_recenter_scalar(
        base_log, weights, 10_000.0
    )

    assert zero_scalar < -0.5
    assert zero_mean == 0.0
    assert high_scalar > 0.0
    assert high_mean == pytest.approx(10_000.0, rel=2.0e-9)


def test_split_contract_rejects_quarantined_validation_year() -> None:
    dates = np.asarray(
        ["2002-06-01", "2017-06-01", "2018-06-01", "2019-06-01", "2020-06-01"],
        dtype="datetime64[D]",
    )
    quarantined = sweep.validate_quarantined_splits(
        dates,
        np.asarray([0, 1]),
        np.asarray([2, 3]),
    )

    np.testing.assert_array_equal(quarantined, np.asarray([4]))
    with pytest.raises(ValueError, match="2018--2019"):
        sweep.validate_quarantined_splits(
            dates,
            np.asarray([0, 1]),
            np.asarray([2, 4]),
        )


def _ranking_case_metrics() -> pd.DataFrame:
    dates = (pd.Timestamp("2018-06-01"), pd.Timestamp("2019-06-01"))
    rows = []
    values = {
        sweep.REFERENCE_CONFIGURATION: {
            "rmse": 5.00,
            "mae": 3.00,
            "bias": -1.00,
            "acc": 0.350,
        },
        "log_anchor_bias_aware_loss": {
            "rmse": 5.02,
            "mae": 3.01,
            "bias": -0.70,
            "acc": 0.348,
        },
        "recentered_anchor_current_loss": {
            "rmse": 5.20,
            "mae": 3.20,
            "bias": -0.60,
            "acc": 0.330,
        },
        # This is numerically lower but has a larger absolute negative bias;
        # it must fail the bias reduction guard.
        "recentered_anchor_bias_aware_loss": {
            "rmse": 4.95,
            "mae": 2.95,
            "bias": -1.30,
            "acc": 0.355,
        },
    }
    for configuration, metrics in values.items():
        for member in ("ensemble", "seed_42", "seed_43", "seed_44"):
            member_metrics = dict(metrics)
            if configuration == "log_anchor_bias_aware_loss" and member == "seed_44":
                member_metrics["bias"] = -1.20
            for date in dates:
                for lead in range(1, 7):
                    rows.append(
                        {
                            "configuration": configuration,
                            "member": member,
                            "case_id": date,
                            "lead": lead,
                            **member_metrics,
                        }
                    )
    for date in dates:
        for lead in range(1, 7):
            rows.append(
                {
                    "configuration": "raw_fuxi",
                    "member": "deterministic",
                    "case_id": date,
                    "lead": lead,
                    "rmse": 5.50,
                    "mae": 3.50,
                    "bias": -2.00,
                    "acc": 0.300,
                }
            )
    anchor_values = {
        "log_anchor": {"rmse": 5.30, "mae": 3.30, "bias": -1.20, "acc": 0.320},
        "physical_recentered_anchor": {
            "rmse": 5.25,
            "mae": 3.25,
            "bias": -0.80,
            "acc": 0.325,
        },
    }
    for configuration, metrics in anchor_values.items():
        for date in dates:
            for lead in range(1, 7):
                rows.append(
                    {
                        "configuration": configuration,
                        "member": "deterministic",
                        "case_id": date,
                        "lead": lead,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def test_physical_ranking_uses_absolute_bias_and_predeclared_guards() -> None:
    records = pd.DataFrame(
        [
            {
                "configuration": candidate.name,
                "parameter_count": 123,
                "best_validation_objective": 0.1 + 0.01 * index,
                "elapsed_seconds": 10.0,
            }
            for index, candidate in enumerate(sweep.CANDIDATES)
        ]
    )

    ranking = sweep.build_physical_ranking(
        records, _ranking_case_metrics(), sweep.CANDIDATES
    ).set_index("configuration")

    good = ranking.loc["log_anchor_bias_aware_loss"]
    sign_trap = ranking.loc["recentered_anchor_bias_aware_loss"]
    assert good.pooled_abs_bias_reduction_pct == pytest.approx(30.0)
    assert bool(good.qualifies)
    assert good.seed_guard_passes == 2
    assert good.seed_guard_required == 2
    assert bool(good.bias_beats_raw_guard)
    assert good.raw_pooled_bias == pytest.approx(-2.0)
    assert good.mean_abs_lead_bias_reduction_pct == pytest.approx(30.0)
    assert sign_trap.pooled_abs_bias_reduction_pct == pytest.approx(-30.0)
    assert not bool(sign_trap.qualifies)
    assert not bool(ranking.loc["recentered_anchor_current_loss"].pooled_rmse_guard)


def test_matching_lead_bias_guard_catches_cross_lead_max_loophole() -> None:
    frame = _ranking_case_metrics()
    reference = frame.configuration.eq(sweep.REFERENCE_CONFIGURATION)
    good = frame.configuration.eq("log_anchor_bias_aware_loss")
    frame.loc[reference & frame.lead.eq(1), "bias"] = -2.0
    frame.loc[good & frame.lead.eq(1), "bias"] = -1.0
    frame.loc[good & frame.lead.eq(2), "bias"] = -1.1
    records = pd.DataFrame(
        [
            {
                "configuration": candidate.name,
                "parameter_count": 123,
                "best_validation_objective": 0.1,
                "elapsed_seconds": 10.0,
            }
            for candidate in sweep.CANDIDATES
        ]
    )

    ranking = sweep.build_physical_ranking(
        records, frame, sweep.CANDIDATES
    ).set_index("configuration")
    row = ranking.loc["log_anchor_bias_aware_loss"]

    assert max(abs(row[f"W{lead}_bias"]) for lead in range(1, 7)) <= 2.0
    assert row.mean_abs_lead_bias_reduction_pct >= 20.0
    assert not bool(row.all_lead_abs_bias_not_worse)
    assert not bool(row.qualifies)


def test_selection_retains_reference_when_no_candidate_qualifies() -> None:
    records = pd.DataFrame(
        [
            {
                "configuration": candidate.name,
                "parameter_count": 123,
                "best_validation_objective": 0.1,
                "elapsed_seconds": 10.0,
            }
            for candidate in sweep.CANDIDATES
        ]
    )
    ranking = sweep.build_physical_ranking(
        records, _ranking_case_metrics(), sweep.CANDIDATES
    )
    ranking.loc[:, "qualifies"] = False
    ranking = ranking.sort_values("pooled_rmse").reset_index(drop=True)

    status, selected = sweep.select_configuration(ranking)

    assert status == "no_candidate_qualified_reference_retained"
    assert selected.configuration == sweep.REFERENCE_CONFIGURATION


def test_seed_guard_allows_one_of_one_for_smoke() -> None:
    frame = _ranking_case_metrics()
    frame = frame.loc[~frame.member.isin(("seed_43", "seed_44"))]

    guards = sweep.build_seed_physical_guards(frame, sweep.CANDIDATES)
    good = guards.loc[guards.configuration.eq("log_anchor_bias_aware_loss")]

    assert len(good) == 1
    assert bool(good.seed_passes_all_guards.iloc[0])


def test_candidate_comparison_uses_its_own_deterministic_anchor() -> None:
    comparison = sweep.candidate_vs_own_anchor(
        _ranking_case_metrics(), sweep.CANDIDATES
    )

    log_rows = comparison.loc[
        comparison.configuration.eq("log_anchor_bias_aware_loss")
    ]
    recentered_rows = comparison.loc[
        comparison.configuration.eq("recentered_anchor_current_loss")
    ]
    assert set(log_rows.anchor_configuration) == {"log_anchor"}
    assert set(recentered_rows.anchor_configuration) == {
        "physical_recentered_anchor"
    }
    pooled_bias = log_rows.loc[
        log_rows.scope.eq("ALL_WEEKS") & log_rows.metric.eq("bias")
    ].iloc[0]
    assert pooled_bias.absolute_bias_delta_candidate_minus_anchor == pytest.approx(
        -0.50
    )


def test_candidate_grid_is_exact_two_by_two() -> None:
    assert len(sweep.CANDIDATES) == 4
    assert {
        (candidate.anchor_kind, candidate.loss_kind)
        for candidate in sweep.CANDIDATES
    } == {
        ("log_anchor", "current"),
        ("log_anchor", "bias_aware"),
        ("physical_recentered", "current"),
        ("physical_recentered", "bias_aware"),
    }
