from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import india_s2s_regional_seasonal_comparison as comparison


def _cases() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    season_month = {"JF": 1, "MAM": 3, "JJAS": 6, "OND": 10}
    for season, month in season_month.items():
        for year in (2020, 2021):
            for case in range(4):
                init = pd.Timestamp(year, month, 1) + pd.Timedelta(days=7 * case)
                for region_index, region in enumerate(comparison.REGIONS):
                    for lead in comparison.LEADS:
                        base = 2.0 + 0.1 * lead + 0.05 * region_index + 0.01 * case
                        for method in comparison.METHODS:
                            corrected = method == comparison.CANDIDATE
                            rows.append(
                                {
                                    "method": method,
                                    "reference": "imd",
                                    "score_status": "available",
                                    "verification_year": year,
                                    "init": init.strftime("%Y-%m-%d"),
                                    "valid_period_midpoint": (
                                        init + pd.Timedelta(days=3.5 + 7 * (lead - 1))
                                    ).isoformat(),
                                    "season": season,
                                    "region": region,
                                    "region_label": comparison.REGION_LABELS[region],
                                    "lead_week": lead,
                                    "crps": base - (0.2 if corrected else 0.0),
                                    "acc": 0.3 + 0.01 * lead + (0.05 if corrected else 0.0),
                                    "rmse": base + 3.0 - (0.4 if corrected else 0.0),
                                    "mae": base + 1.0 - (0.3 if corrected else 0.0),
                                    "bias": -0.2 - (0.1 if corrected else 0.0),
                                }
                            )
    return pd.DataFrame(rows)


def test_tables_cover_all_regions_seasons_leads_and_show_improvement() -> None:
    cases = comparison._validated_cases(_cases(), frozen=False)
    descriptive = comparison.descriptive_metrics(cases)
    intervals, receipts = comparison.paired_intervals(
        cases, replicates=50, block_length=2, seed=7
    )

    assert len(descriptive) == 4 * 4 * 6 * 2
    assert len(intervals) == 4 * 4 * 6 * 8
    assert len(receipts) == 4 * 6
    assert set(descriptive.method_label) == {
        "Model v1",
        "Probabilistic Correction v1",
    }
    for metric in comparison.LOSS_METRICS:
        selected = intervals[
            intervals.metric.eq(metric)
            & intervals.effect.eq("skill_pct_vs_baseline")
        ]
        assert selected.resolved_improvement.all()
        assert (selected.ci_lower > 0.0).all()
    acc = intervals[
        intervals.metric.eq("acc")
        & intervals.effect.eq("candidate_minus_baseline")
    ]
    assert acc.resolved_improvement.all()
    assert (acc.ci_lower > 0.0).all()


def test_shared_regional_date_contract_rejects_missing_case() -> None:
    cases = comparison._validated_cases(_cases(), frozen=False)
    index = cases.index[
        cases.region.eq("central_india")
        & cases.season.eq("MAM")
        & cases.lead_week.eq(2)
        & cases.method.eq(comparison.CANDIDATE)
    ][0]
    malformed = cases.drop(index=index)

    with pytest.raises(comparison.RegionalSeasonalError, match="pairing|cohorts"):
        comparison.paired_intervals(
            malformed, replicates=5, block_length=2, seed=3
        )


def test_figure_and_results_are_written(tmp_path: Path) -> None:
    cases = comparison._validated_cases(_cases(), frozen=False)
    descriptive = comparison.descriptive_metrics(cases)
    intervals, _ = comparison.paired_intervals(
        cases, replicates=10, block_length=2, seed=9
    )
    stem = tmp_path / "comparison"
    comparison.comparison_figure(intervals, stem)
    comparison._write_results(tmp_path / "RESULTS.md", descriptive, intervals)

    assert stem.with_suffix(".png").stat().st_size > 10_000
    assert stem.with_suffix(".pdf").stat().st_size > 1_000
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "IMD homogeneous-region" in text
    assert "No 2025 observation was opened" in text
