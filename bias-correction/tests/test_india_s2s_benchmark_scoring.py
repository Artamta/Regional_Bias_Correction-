"""Synthetic contract tests for independent India S2S benchmark scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import india_s2s_benchmark_scoring as scoring


def _canonical_schedule() -> np.ndarray:
    dates: list[pd.Timestamp] = []
    for year in scoring.FORECAST_YEARS:
        mondays = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")
        thursdays = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-THU")
        dates.extend(mondays.union(thursdays).sort_values())
    missing_2024 = pd.to_datetime(
        ["2024-06-06", "2024-06-10", "2024-06-13", "2024-06-17", "2024-08-05"]
    )
    return np.asarray(pd.DatetimeIndex(dates).difference(missing_2024), dtype="datetime64[D]")


def _region_weights(shape: tuple[int, int] = (1, 3)) -> dict[str, np.ndarray]:
    india = np.ones(shape, dtype=np.float64)
    cell = np.ones(shape, dtype=np.float64)
    fractions = {
        name: np.ones(shape, dtype=np.float64) for name in scoring.REGION_ORDER[1:]
    }
    return scoring.build_region_base_weights(india, cell, fractions)


def test_imd_targets_are_exact_plus_1_through_plus_42_with_minimum_support() -> None:
    init = np.asarray(["2024-01-01"], dtype="datetime64[D]")
    dates = pd.date_range("2024-01-02", periods=42, freq="D")
    values = np.arange(1, 43, dtype=np.float32)[:, None, None]
    values = np.broadcast_to(values, (42, 1, 3)).copy()
    fractions = np.ones_like(values)
    fractions[0, 0, 0] = 0.5
    normal = np.full((366, 1, 3), 9.0, dtype=np.float32)

    weekly = scoring.weekly_imd_targets(init, dates, values, fractions, normal)

    assert weekly.target_dates.shape == (1, 6, 7)
    assert weekly.target_dates[0, 0, 0] == np.datetime64("2024-01-02")
    assert weekly.target_dates[0, -1, -1] == np.datetime64("2024-02-12")
    np.testing.assert_allclose(weekly.truth[0, :, 0, 1], [4, 11, 18, 25, 32, 39])
    weighted_w1 = (0.5 + np.arange(2, 8).sum()) / 6.5
    assert weekly.truth[0, 0, 0, 0] == pytest.approx(weighted_w1)
    assert weekly.observation_support[0, 0, 0, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(weekly.climatology, 9.0)


def test_imd_target_builder_rejects_any_2025_truth_boundary_crossing() -> None:
    init = np.asarray(["2024-11-20"], dtype="datetime64[D]")
    dates = pd.date_range("2024-11-21", periods=42, freq="D")
    values = np.ones((42, 1, 3), dtype=np.float32)
    with pytest.raises(scoring.ScoringContractError, match="sealed 2025"):
        scoring.weekly_imd_targets(
            init,
            dates,
            values,
            np.ones_like(values),
            np.ones((366, 1, 3), dtype=np.float32),
            reject_daily_dates_after_cutoff=False,
        )


def test_canonical_cohorts_keep_517_climatology_and_505_scoring_cases() -> None:
    forecast = _canonical_schedule()
    score = forecast[scoring.scoreable_initialization_mask(forecast)]

    receipt = scoring.validate_bridge_cohorts(forecast, score, strict_counts=True)

    assert receipt["forecast_only_count"] == 517
    assert receipt["scoring_count"] == 505
    assert receipt["forecast_only_year_counts"] == scoring.FORECAST_COUNTS
    assert receipt["scoring_year_counts"] == scoring.SCORING_COUNTS
    assert (
        receipt["forecast_only_dates_sha256"]
        == scoring.FORECAST_INITIALIZATION_SHA256
    )
    assert receipt["scoring_dates_sha256"] == scoring.SCORING_INITIALIZATION_SHA256
    assert set(receipt["valid_midpoint_jjas_count_by_lead"].values()) == {169}
    assert receipt["last_scoring_target_date"] == "2024-12-30"
    np.testing.assert_array_equal(forecast[receipt["score_indices"]], score)


def test_canonical_cohort_rejects_a_same_count_date_mutation() -> None:
    forecast = _canonical_schedule()
    forecast[0] = np.datetime64("2020-01-01")
    score = forecast[scoring.scoreable_initialization_mask(forecast)]

    with pytest.raises(scoring.ScoringContractError, match="frozen hash"):
        scoring.validate_bridge_cohorts(forecast, score, strict_counts=True)


def test_fixed_calendar_and_circular_window_match_frozen_invariants() -> None:
    np.testing.assert_array_equal(
        scoring.fixed_climatology_day(["2020-03-01", "2021-03-01", "2024-02-29"]),
        [61, 61, 60],
    )
    values = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)
    mean, count = scoring.circular_window_mean(
        values,
        np.asarray([5, 1, 2]),
        half_width_days=1,
        calendar_days=5,
    )
    assert count.tolist() == [3, 2, 1, 1, 2]
    assert mean[0, 0] == pytest.approx(20.0)
    assert mean[4, 0] == pytest.approx(15.0)


def test_streaming_climatology_is_equal_year_loyo_and_matches_batch() -> None:
    dates: list[np.datetime64] = []
    fields: list[np.ndarray] = []
    for value, year in enumerate(scoring.FORECAST_YEARS):
        yearly = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="14D")
        dates.extend(np.asarray(yearly, dtype="datetime64[D]"))
        fields.extend(
            [np.full((6, 1, 3), value, dtype=np.float32) for _ in range(len(yearly))]
        )
    dates_array = np.asarray(dates, dtype="datetime64[D]")
    field_array = np.stack(fields)
    order = np.argsort(dates_array)
    dates_array = dates_array[order]
    field_array = field_array[order]

    batch = scoring.build_equal_year_forecast_climatology(
        dates_array, field_array, strict_bridge_counts=False
    )
    stream = scoring.EqualYearForecastClimatologyAccumulator()
    for indices in np.array_split(np.arange(len(dates_array)), 7):
        stream.update(dates_array[indices], field_array[indices])
    streamed = stream.finalize()

    np.testing.assert_array_equal(streamed.initialization_count, batch.initialization_count)
    np.testing.assert_allclose(streamed.mean_by_year, batch.mean_by_year)
    np.testing.assert_allclose(streamed.mean_loyo, batch.mean_loyo)
    selected = batch.select_loyo(["2020-06-01", "2024-06-03"])
    np.testing.assert_allclose(selected[0], 2.5)
    np.testing.assert_allclose(selected[1], 1.5)


def test_climatology_uses_unscored_forecast_only_starts() -> None:
    forecast = _canonical_schedule()
    score_mask = scoring.scoreable_initialization_mask(forecast)
    fields = np.zeros((len(forecast), 6, 1, 3), dtype=np.float32)
    fields[~score_mask] = 10.0

    climate = scoring.build_equal_year_forecast_climatology(
        forecast, fields, strict_bridge_counts=True
    )
    # A 2023 score excludes 2023 but includes the 2024 yearly component.  Its
    # centered window therefore sees late-2024 forecast-only starts for which
    # truth is sealed and absent from the 505-case scoring cohort.
    selected = climate.select_loyo(["2023-11-20"])

    assert climate.initialization_counts_by_year == scoring.FORECAST_COUNTS
    assert np.all(selected > 0.0)


def test_empirical_50_member_crps_matches_quadratic_definition() -> None:
    members = np.concatenate((np.zeros(25), np.full(25, 2.0))).reshape(1, 50, 1, 1, 1)
    truth = np.ones((1, 1, 1, 1), dtype=np.float64)

    actual = scoring.empirical_crps(members, truth)
    values = members[0, :, 0, 0, 0]
    brute = np.mean(np.abs(values - 1.0)) - 0.5 * np.mean(
        np.abs(values[:, None] - values[None, :])
    )

    assert actual.item() == pytest.approx(brute)
    assert actual.item() == pytest.approx(0.5)
    with pytest.raises(scoring.ScoringContractError, match="expected 50"):
        scoring.empirical_crps(members[:, :-1], truth)


def test_chunk_scoring_uses_separate_climatologies_and_all_five_regions() -> None:
    init = np.asarray(["2024-06-03"], dtype="datetime64[D]")
    signal = np.asarray([1.0, 3.0, 2.0], dtype=np.float32).reshape(1, 1, 1, 3)
    forecast_normal = np.asarray([100.0, 0.0, 50.0], dtype=np.float32).reshape(
        1, 1, 1, 3
    )
    observation_normal = np.asarray([0.0, 50.0, 100.0], dtype=np.float32).reshape(
        1, 1, 1, 3
    )
    forecast = np.broadcast_to(forecast_normal + signal, (1, 6, 1, 3)).copy()
    truth = np.broadcast_to(observation_normal + signal, (1, 6, 1, 3)).copy()
    members = np.repeat(forecast[:, None], 50, axis=1)
    forecast_climo = np.broadcast_to(forecast_normal, forecast.shape).copy()
    observation_climo = np.broadcast_to(observation_normal, truth.shape).copy()
    support = np.ones_like(truth)
    support[..., 0] = 0.5

    cases = scoring.score_case_chunk(
        method="raw_fuxi",
        initializations=init,
        members=members,
        truth=truth,
        forecast_climatology=forecast_climo,
        observation_climatology=observation_climo,
        weekly_observation_support=support,
        region_base_weights=_region_weights(),
    )

    assert len(cases) == 5 * 6
    assert tuple(cases.region.drop_duplicates()) == scoring.REGION_ORDER
    np.testing.assert_allclose(cases.acc, 1.0)
    np.testing.assert_allclose(cases.crps, cases.mae)
    assert set(cases.season) == {"JJAS"}
    assert cases.valid_cell_count.eq(3).all()
    assert cases.effective_area_km2.eq(2.5).all()


def test_aggregation_is_mean_case_rmse_not_pooled_rmse() -> None:
    cases = pd.DataFrame(
        {
            "method": ["raw_fuxi", "raw_fuxi"],
            "region": ["all_india", "all_india"],
            "lead_week": [1, 1],
            "init": ["2020-01-02", "2020-01-06"],
            "crps": [0.0, 10.0],
            "acc": [0.0, 1.0],
            "rmse": [0.0, 10.0],
            "mae": [0.0, 10.0],
            "bias": [0.0, 10.0],
        }
    )

    summary = scoring.aggregate_case_scores(cases).iloc[0]

    assert summary.rmse == pytest.approx(5.0)
    assert summary.rmse != pytest.approx(np.sqrt(50.0))
    assert summary.case_count == 2


def test_raw_fuxi_identity_receipt_requires_metrics_and_exact_case_ids() -> None:
    benchmark = pd.DataFrame(
        {
            "init": ["2020-01-02", "2020-01-02"],
            "lead_week": [1, 2],
            "region": ["all_india", "all_india"],
            "acc": [0.3, np.nan],
            "rmse": [2.0, 3.0],
            "mae": [1.0, 2.0],
            "bias": [-0.1, 0.2],
            "valid_cell_count": [171, 170],
            "effective_area_km2": [3_200_000.0, 3_100_000.0],
        }
    )
    independent = benchmark.sample(frac=1.0, random_state=7).reset_index(drop=True)

    receipt = scoring.assert_raw_fuxi_identity(
        independent, benchmark, require_complete_bridge=False
    )

    assert receipt["identity"] is True
    assert receipt["matched_case_rows"] == 2
    changed = independent.copy()
    changed.loc[changed.lead_week.eq(1), "rmse"] += 0.1
    with pytest.raises(scoring.ScoringContractError, match="metrics differ"):
        scoring.assert_raw_fuxi_identity(
            changed, benchmark, require_complete_bridge=False
        )
    changed_support = independent.copy()
    changed_support.loc[changed_support.lead_week.eq(1), "effective_area_km2"] += 1.0
    with pytest.raises(scoring.ScoringContractError, match="metrics differ"):
        scoring.assert_raw_fuxi_identity(
            changed_support, benchmark, require_complete_bridge=False
        )
    missing = independent.iloc[:1]
    with pytest.raises(scoring.ScoringContractError, match="case IDs differ"):
        scoring.compare_raw_fuxi_identity(
            missing, benchmark, require_complete_bridge=False
        )
    with pytest.raises(scoring.ScoringContractError, match="exactly 15150"):
        scoring.compare_raw_fuxi_identity(independent, benchmark)
    with pytest.raises(scoring.ScoringContractError, match="empty"):
        scoring.compare_raw_fuxi_identity(
            independent.iloc[:0], benchmark.iloc[:0], require_complete_bridge=False
        )


def test_complete_raw_identity_is_bound_to_frozen_case_grid_hash() -> None:
    forecast = _canonical_schedule()
    score = forecast[scoring.scoreable_initialization_mask(forecast)]
    rows = [
        {
            "init": np.datetime_as_string(init, unit="D"),
            "lead_week": lead,
            "region": region,
            "acc": 0.1,
            "rmse": 1.0,
            "mae": 0.8,
            "bias": 0.0,
            "valid_cell_count": 171,
            "effective_area_km2": 3_200_000.0,
        }
        for init in score
        for lead in scoring.LEAD_WEEKS
        for region in scoring.REGION_ORDER
    ]
    benchmark = pd.DataFrame(rows)
    independent = benchmark.sample(frac=1.0, random_state=9).reset_index(drop=True)

    receipt = scoring.assert_raw_fuxi_identity(independent, benchmark)

    assert receipt["identity"] is True
    assert receipt["matched_case_rows"] == scoring.RAW_IDENTITY_CASE_COUNT
    assert receipt["case_ids_sha256"] == scoring.RAW_IDENTITY_CASE_IDS_SHA256
    assert receipt["complete_bridge_contract"] is True
