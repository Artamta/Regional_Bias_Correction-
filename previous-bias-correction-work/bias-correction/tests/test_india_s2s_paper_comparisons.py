from __future__ import annotations

import numpy as np
import pandas as pd

import india_s2s_paper_comparisons as comparisons


def _sensitivity_cases() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "raw_fuxi": (2.0, 0.2, 0.4),
        "moment_calibration": (1.8, 0.25, 0.2),
        "location_spread": (1.6, 0.3, 0.1),
    }
    common = pd.date_range("2022-06-01", periods=4, freq="2D")
    for lead in comparisons.LEADS:
        dates = list(common) + [pd.Timestamp("2022-07-01") + pd.Timedelta(days=lead)]
        for method, (crps, acc, bias) in values.items():
            for midpoint in dates:
                rows.append(
                    {
                        "method": method,
                        "region": "all_india",
                        "lead_week": lead,
                        "verification_year": 2022,
                        "valid_period_midpoint": midpoint.isoformat(),
                        "season": "JJAS",
                        "crps": crps,
                        "acc": acc,
                        "bias": bias,
                    }
                )
    return pd.DataFrame(rows)


def test_exact_common_midpoint_sensitivity_uses_only_shared_dates() -> None:
    intervals, receipt = comparisons.exact_common_midpoint_sensitivity(
        _sensitivity_cases(),
        replicates=8,
        seed=3,
        block_lengths=(2,),
        strict=False,
    )

    assert len(intervals) == 8
    assert receipt["common_midpoints_per_lead"] == 4
    assert receipt["case_lead_rows"] == 24
    assert receipt["year_counts"] == {2022: 4}
    assert set(intervals.n_common_midpoints_per_lead) == {4}
    assert set(intervals.n_case_lead_rows) == {24}
    raw = intervals[intervals.baseline.eq("raw_fuxi")]
    skill = raw[
        raw.metric.eq("crps") & raw.effect.eq("skill_pct_vs_baseline")
    ].iloc[0]
    assert np.isclose(skill.estimate, 20.0)
    acc = raw[
        raw.metric.eq("acc") & raw.effect.eq("candidate_minus_baseline")
    ].iloc[0]
    assert np.isclose(acc.estimate, 0.1)


def _mme_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    neural_rows: list[dict[str, object]] = []
    mme_rows: list[dict[str, object]] = []
    dates = pd.date_range("2022-06-01", periods=4, freq="7D")
    for lead in comparisons.LEADS:
        for init in dates:
            midpoint = init + pd.Timedelta(days=7 * (lead - 1) + 3, hours=12)
            neural_rows.append(
                {
                    "method": "location_spread",
                    "region": "all_india",
                    "season": "JJAS",
                    "lead_week": lead,
                    "init": init.strftime("%Y-%m-%d"),
                    "valid_period_midpoint": midpoint.isoformat(),
                    "acc": 0.4,
                    "rmse": 4.0,
                    "mae": 3.0,
                    "bias": 0.1,
                }
            )
            mme_rows.append(
                {
                    "model": "mme",
                    "model_label": "MME (no ECMWF)",
                    "region": "all_india",
                    "season": "JJAS",
                    "lead_week": lead,
                    "score_status": "available",
                    "init": init.strftime("%Y-%m-%d"),
                    "valid_period_midpoint": midpoint.isoformat(),
                    "acc": 0.3,
                    "rmse": 5.0,
                    "mae": 4.0,
                    "bias": 0.2,
                }
            )
    return pd.DataFrame(neural_rows), pd.DataFrame(mme_rows)


def test_calibrated_fuxi_vs_mme_is_paired_by_initialization() -> None:
    neural, mme = _mme_cases()
    intervals, receipt = comparisons.calibrated_fuxi_vs_mme(
        neural,
        mme,
        replicates=8,
        seed=5,
        block_lengths=(2,),
        strict=False,
    )

    assert len(intervals) == 36
    assert receipt["ecmwf_in_mme"] is False
    assert receipt["mme_members"] == [
        "cma",
        "dlesym_v0",
        "fuxi_s2s",
        "ncep",
        "neuralgcm",
        "ukmo",
    ]
    differences = intervals[intervals.effect.eq("candidate_minus_baseline")]
    np.testing.assert_allclose(
        differences[differences.metric.eq("acc")].estimate, 0.1
    )
    np.testing.assert_allclose(
        differences[differences.metric.eq("rmse")].estimate, -1.0
    )
    skills = intervals[
        intervals.metric.eq("rmse")
        & intervals.effect.eq("skill_pct_vs_baseline")
    ]
    np.testing.assert_allclose(skills.estimate, 20.0)
