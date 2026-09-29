"""Focused mathematical and leakage tests for categorical FuXi PBC."""

from __future__ import annotations

import numpy as np

import fuxi_pbc_core as pbc


def _weekly_dates(count: int, start: str = "2001-12-20") -> np.ndarray:
    return np.datetime64(start, "D") + np.arange(count) * np.timedelta64(7, "D")


def _rainfall(count: int, *, leads: int = 2) -> np.ndarray:
    rng = np.random.default_rng(42)
    unique_windows = rng.gamma(1.3, 2.0, size=(count + leads - 1, 2, 3)).astype(
        np.float32
    )
    unique_windows[rng.random(unique_windows.shape) < 0.35] = 0.0
    return np.stack(
        [unique_windows[case : case + leads] for case in range(count)], axis=0
    )


def test_training_quantiles_ignore_every_heldout_value_and_retain_ties() -> None:
    dates = _weekly_dates(32)
    truth = _rainfall(32)
    train = np.arange(20, dtype=np.int64)
    support = np.ones((2, 3), dtype=bool)
    first = pbc.fit_calendar_quantiles(
        truth,
        dates,
        train,
        (0.05, 0.2, 0.5, 0.95),
        support,
        window_radius_days=30,
        minimum_samples=6,
    )
    mutated = truth.copy()
    mutated[20:] = 1.0e6
    second = pbc.fit_calendar_quantiles(
        mutated,
        dates,
        train,
        (0.05, 0.2, 0.5, 0.95),
        support,
        window_radius_days=30,
        minimum_samples=6,
    )
    np.testing.assert_array_equal(first.thresholds, second.thresholds)
    np.testing.assert_array_equal(first.empirical_cdf, second.empirical_cdf)
    assert np.any(first.thresholds[:, 0, support] == 0.0)
    zero = first.thresholds == 0.0
    # Strict P(Y < 0) is zero even though the nominal cut is five percent.
    np.testing.assert_array_equal(first.empirical_cdf[zero], 0.0)


def test_raw_cdf_uses_strict_less_than_and_preserves_nan_support() -> None:
    members = np.asarray([0.0, 0.0, 1.0, 3.0], dtype=np.float32).reshape(1, 4, 1, 1, 1)
    thresholds = np.asarray([0.0, 1.0, 2.0], dtype=np.float32).reshape(1, 1, 3, 1, 1)
    actual = pbc.ensemble_cdf(members, thresholds, chunk_size=1)
    np.testing.assert_allclose(actual.reshape(-1), [0.0, 0.5, 0.75])
    observed = pbc.observation_cdf(
        np.asarray([1.0], dtype=np.float32).reshape(1, 1, 1, 1), thresholds
    )
    np.testing.assert_array_equal(observed.reshape(-1), [0.0, 0.0, 1.0])
    thresholds[..., 0, 0] = np.nan
    assert np.isnan(pbc.ensemble_cdf(members, thresholds)).all()


def test_isotonic_projection_is_exact_valid_and_never_hurts_any_rps_outcome() -> None:
    raw = np.asarray(
        [
            [0.8, 0.2, 0.4, 1.2],
            [-0.2, 0.6, 0.5, 0.9],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    projected = pbc.project_cdf(raw)
    np.testing.assert_allclose(projected[0], [0.46666667, 0.46666667, 0.46666667, 1.0])
    np.testing.assert_allclose(projected[1], [0.0, 0.55, 0.55, 0.9])
    np.testing.assert_allclose(projected[2], raw[2])
    assert pbc.is_valid_cdf(projected)
    clipped = np.clip(raw, 0.0, 1.0)
    # Every possible categorical observation has cumulative indicators
    # [0,...,0,1,...,1].  Projection onto the CDF cone cannot increase SSE.
    for category in range(5):
        outcome = (np.arange(4) >= category).astype(np.float64)
        before = np.sum((clipped - outcome) ** 2, axis=1)
        after = np.sum((projected - outcome) ** 2, axis=1)
        assert np.all(after <= before + 1.0e-10)


def test_threshold_projection_collapses_ties_anchors_zero_and_is_order_robust() -> None:
    thresholds = np.asarray(
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(3, 1, 4, 1, 1)
    raw = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.8],
            [0.9, 0.8, 0.2, 0.1],
            [0.9, 0.4, 0.2, 0.7],
        ],
        dtype=np.float32,
    ).reshape(thresholds.shape)
    projected = pbc.project_cdf_for_thresholds(raw, thresholds)
    np.testing.assert_allclose(projected[0].reshape(-1), [0.0, 0.0, 0.1, 0.8])
    np.testing.assert_allclose(projected[1].reshape(-1), [0.0, 0.3, 0.3, 0.3])
    np.testing.assert_array_equal(projected[2], 0.0)
    assert pbc.is_valid_cdf_for_thresholds(projected, thresholds)

    # The deterministic q=0 channel is excluded before PAVA and cannot pull a
    # positive-cut prediction upward.
    anchored = pbc.project_cdf_for_thresholds(
        np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 1, 2, 1, 1),
        np.asarray([0.0, 1.0], dtype=np.float32).reshape(1, 1, 2, 1, 1),
    )
    np.testing.assert_array_equal(anchored.reshape(-1), [0.0, 0.0])

    # Averaging an exact-threshold group is symmetric in its nominal slots.
    swapped = raw.copy()
    swapped[1, 0, 1], swapped[1, 0, 2] = (
        raw[1, 0, 2].copy(),
        raw[1, 0, 1].copy(),
    )
    np.testing.assert_array_equal(
        pbc.project_cdf_for_thresholds(swapped, thresholds)[1], projected[1]
    )
    permutation = np.asarray([2, 0, 1])
    permuted = pbc.project_cdf_for_thresholds(raw[permutation], thresholds[permutation])
    np.testing.assert_array_equal(permuted, projected[permutation])

    unsorted = thresholds.copy()
    unsorted[0, 0, :, 0, 0] = [0.0, 2.0, 1.0, 1.0]
    with np.testing.assert_raises_regex(pbc.PBCContractError, "nondecreasing"):
        pbc.project_cdf_for_thresholds(raw, unsorted)


def test_effective_rps_counts_unique_positive_cuts_once_and_excludes_all_zero() -> None:
    thresholds = np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float32).reshape(
        1, 1, 4, 1, 1
    )
    forecast = np.asarray([0.0, 0.5, 0.5, 0.9], dtype=np.float32).reshape(
        thresholds.shape
    )
    observed = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(
        thresholds.shape
    )
    score = pbc.ranked_probability_score(forecast, observed, thresholds)
    # q=0 is deterministic and excluded; duplicate q=1 is counted once.
    np.testing.assert_allclose(score, (0.5**2 + 0.1**2) / 2.0)
    assert 0.0 <= float(score.item()) <= 1.0

    all_zero_thresholds = np.zeros((1, 1, 4, 1, 1), dtype=np.float32)
    all_zero_cdf = np.zeros_like(all_zero_thresholds)
    assert np.isnan(
        pbc.ranked_probability_score(all_zero_cdf, all_zero_cdf, all_zero_thresholds)
    ).all()
    bad = forecast.copy()
    bad[:, :, 2] += 0.1
    with np.testing.assert_raises_regex(
        pbc.PBCContractError, "physical-threshold contract"
    ):
        pbc.ranked_probability_score(bad, observed, thresholds)


def test_effective_rps_is_invariant_to_replication_of_an_identical_cut() -> None:
    compact_thresholds = np.asarray([1.0, 3.0], dtype=np.float32).reshape(1, 1, 2, 1, 1)
    compact_forecast = np.asarray([0.5, 0.9], dtype=np.float32).reshape(
        compact_thresholds.shape
    )
    compact_observed = np.asarray([0.0, 1.0], dtype=np.float32).reshape(
        compact_thresholds.shape
    )
    repeated_thresholds = np.asarray([1.0, 1.0, 1.0, 3.0], dtype=np.float32).reshape(
        1, 1, 4, 1, 1
    )
    repeated_forecast = np.asarray([0.5, 0.5, 0.5, 0.9], dtype=np.float32).reshape(
        repeated_thresholds.shape
    )
    repeated_observed = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(
        repeated_thresholds.shape
    )
    np.testing.assert_array_equal(
        pbc.ranked_probability_score(
            compact_forecast, compact_observed, compact_thresholds
        ),
        pbc.ranked_probability_score(
            repeated_forecast, repeated_observed, repeated_thresholds
        ),
    )


def test_paper_nominal_cut_sensitivity_uses_sum_q80_mask_and_literal_reference() -> (
    None
):
    thresholds = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(1, 1, 4, 1, 2)
    forecast = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [0.9, 0.0],
        ],
        dtype=np.float32,
    ).reshape(thresholds.shape)
    observed = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(thresholds.shape)
    score = pbc.paper_nominal_cut_rps(forecast, observed, thresholds)
    np.testing.assert_allclose(score[..., 0], 0.5**2 + 0.1**2)
    assert np.isnan(score[..., 1]).all()
    reference = pbc.paper_fixed_nominal_reference_rps(observed, thresholds)
    np.testing.assert_allclose(
        reference[..., 0], 0.2**2 + 0.4**2 + 0.6**2 + (0.8 - 1.0) ** 2
    )
    assert np.isnan(reference[..., 1]).all()


def test_training_empirical_cdf_is_recomputed_at_persisted_float32_threshold() -> None:
    dates = _weekly_dates(6)
    next_one = np.nextafter(np.float32(1.0), np.float32(np.inf))
    values = np.asarray([0.0, 0.2, 0.7, 1.0, next_one, 3.0], dtype=np.float32)
    truth = values.reshape(6, 1, 1, 1)
    model = pbc.fit_calendar_quantiles(
        truth,
        dates,
        np.arange(6),
        (0.60, 0.65),
        np.ones((1, 1), dtype=bool),
        window_radius_days=183,
        minimum_samples=6,
    )
    assert np.any(model.thresholds[:, 0] == model.thresholds[:, 1])
    for day in range(366):
        for cut in range(2):
            persisted = model.thresholds[day, cut, 0, 0]
            expected = np.mean(values < persisted, dtype=np.float64)
            assert model.empirical_cdf[day, cut, 0, 0] == np.float32(expected)
        if model.thresholds[day, 0, 0, 0] == model.thresholds[day, 1, 0, 0]:
            assert (
                model.empirical_cdf[day, 0, 0, 0] == model.empirical_cdf[day, 1, 0, 0]
            )


def test_issue_time_lags_are_exact_complete_and_future_mutation_safe() -> None:
    dates = _weekly_dates(8)
    truth = np.arange(8 * 2 * 2 * 3, dtype=np.float32).reshape(8, 2, 2, 3)
    lags = pbc.build_issue_time_lags(dates, truth)
    assert lags.source_indices[0].tolist() == [-1, -1]
    assert lags.source_indices[1].tolist() == [0, -1]
    assert lags.source_indices[2].tolist() == [1, 0]
    np.testing.assert_array_equal(lags.values[5, 0], truth[4, 0])
    np.testing.assert_array_equal(lags.values[5, 1], truth[3, 0])
    issue = np.broadcast_to(dates[:, None], lags.window_end.shape)
    available = lags.source_indices >= 0
    assert np.all(lags.window_end[available] < issue[available])

    mutated = truth.copy()
    mutated[5:] = -9999.0
    changed = pbc.build_issue_time_lags(dates, mutated)
    # The forecast issued for case five cannot see case five or later truth.
    np.testing.assert_array_equal(changed.values[5], lags.values[5])


def test_daily_issue_time_lags_bridge_forecast_cadence_and_year_boundaries() -> None:
    daily_dates = np.arange(
        np.datetime64("2019-12-15"),
        np.datetime64("2020-03-20"),
        np.timedelta64(1, "D"),
    )
    daily_values = np.arange(daily_dates.size, dtype=np.float32).reshape(-1, 1, 1)
    # These issues deliberately do not have init-7/init-14 entries in the
    # forecast inventory. The complete observation calendar still defines both
    # operational lag windows exactly, including across New Year and leap day.
    issues = np.asarray(
        ["2020-01-03", "2020-01-06", "2020-02-29", "2020-03-03"],
        dtype="datetime64[D]",
    )
    lags = pbc.build_daily_issue_time_lags(issues, daily_dates, daily_values)
    assert lags.available.all()
    for case, issue in enumerate(issues):
        for column, lag in enumerate((1, 2)):
            start = issue - np.timedelta64(7 * lag, "D")
            stop = start + np.timedelta64(7, "D")
            chosen = (daily_dates >= start) & (daily_dates < stop)
            np.testing.assert_allclose(
                lags.values[case, column], daily_values[chosen].mean(axis=0)
            )
            assert lags.window_start[case, column] == start
            assert lags.window_end[case, column] == issue - np.timedelta64(
                7 * (lag - 1) + 1, "D"
            )
            assert lags.window_end[case, column] < issue

    mutated = daily_values.copy()
    mutated[daily_dates >= issues[1]] = -9999.0
    changed = pbc.build_daily_issue_time_lags(issues, daily_dates, mutated)
    # Nothing at or after issuance can alter an issue's lag features.
    np.testing.assert_array_equal(changed.values[1], lags.values[1])


def test_debias_and_persistence_use_only_fit_indices_and_return_valid_cdfs() -> None:
    count = 28
    dates = _weekly_dates(count)
    support = np.ones((2, 3), dtype=bool)
    levels = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    shape = (count, 2, 4, 2, 3)
    raw = np.broadcast_to(levels[None, None, :, None, None] + 0.12, shape).copy()
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)
    observed = np.broadcast_to(levels[None, None, :, None, None], shape).copy()
    observed = (observed > 0.45).astype(np.float32)
    # Alternate labels over time so the regression is nonsingular enough for
    # a deterministic unit test.
    observed[::2, :, :2] = 0.0
    train = np.arange(20, dtype=np.int64)
    debias = pbc.fit_debias(
        raw, observed, dates, train, support, half_window_days=35, minimum_samples=6
    )
    changed_observed = observed.copy()
    changed_observed[20:] = 1.0 - changed_observed[20:]
    debias_again = pbc.fit_debias(
        raw,
        changed_observed,
        dates,
        train,
        support,
        half_window_days=35,
        minimum_samples=6,
    )
    np.testing.assert_array_equal(debias.correction, debias_again.correction)
    truth = _rainfall(count)
    quantiles = pbc.fit_calendar_quantiles(
        truth, dates, train, levels, support, window_radius_days=60, minimum_samples=6
    )
    threshold = pbc.calendar_fields(quantiles, dates, 2)
    obs = pbc.observation_cdf(truth, threshold)
    climate = pbc.calendar_fields(quantiles, dates, 2, empirical=True)
    lags = pbc.build_issue_time_lags(dates, truth)
    lag_cdf = pbc.lag_observation_cdf(lags, quantiles)
    debias_prediction = pbc.apply_debias(
        raw[20:],
        dates[20:],
        debias,
        threshold[20:],
        project=True,
    )
    assert pbc.is_valid_cdf_for_thresholds(debias_prediction, threshold[20:], axis=2)
    persistence = pbc.fit_persistence(
        raw, obs, climate, lag_cdf, lags.available, train, support, ridge=1.0e-2
    )
    persistence_again = pbc.fit_persistence(
        raw,
        np.concatenate((obs[:20], 1.0 - obs[20:])),
        climate,
        lag_cdf,
        lags.available,
        train,
        support,
        ridge=1.0e-2,
    )
    np.testing.assert_array_equal(
        persistence.coefficients, persistence_again.coefficients
    )
    persistence_prediction = pbc.apply_persistence(
        raw[20:],
        climate[20:],
        lag_cdf[20:],
        lags.available[20:],
        persistence,
        threshold[20:],
        project=True,
    )
    assert pbc.is_valid_cdf_for_thresholds(
        persistence_prediction, threshold[20:], axis=2
    )
    combined = pbc.combine_projected_components(
        debias_prediction, persistence_prediction, threshold[20:]
    )
    assert pbc.is_valid_cdf_for_thresholds(combined, threshold[20:], axis=2)


def test_debias_calendar_accumulator_matches_direct_circular_windows() -> None:
    rng = np.random.default_rng(3)
    count = 40
    dates = _weekly_dates(count, start="2001-12-01")
    raw = rng.random((count, 2, 4, 2, 2), dtype=np.float32)
    observed = (rng.random(raw.shape) > 0.5).astype(np.float32)
    support = np.ones((2, 2), dtype=bool)
    fit = pbc.fit_debias(
        raw,
        observed,
        dates,
        np.arange(count),
        support,
        half_window_days=14,
        minimum_samples=1,
    )
    positions = pbc._calendar_positions(pbc.verification_midpoints(dates, 2))
    for day in range(366):
        for lead in range(2):
            distance = np.minimum(
                (positions[:, lead] - day) % 366,
                (day - positions[:, lead]) % 366,
            )
            chosen = np.flatnonzero(distance <= 14)
            if chosen.size == 0:
                chosen = np.argsort(distance, kind="stable")[:1]
            expected = np.mean(observed[chosen, lead] - raw[chosen, lead], axis=0)
            np.testing.assert_allclose(
                fit.correction[day, lead], expected, rtol=1.0e-6, atol=1.0e-7
            )


def test_rps_probability_bias_and_upper_tail_brier_are_area_weighted() -> None:
    observed = np.asarray([0.0, 1.0], dtype=np.float32).reshape(1, 1, 2, 1, 1)
    perfect = observed.copy()
    climate = np.asarray([0.25, 0.75], dtype=np.float32).reshape(1, 1, 2, 1, 1)
    thresholds = np.asarray([1.0, 2.0], dtype=np.float32).reshape(1, 1, 2, 1, 1)
    weights = np.ones((1, 1), dtype=np.float64)
    np.testing.assert_allclose(
        pbc.mean_rps_by_lead(perfect, observed, thresholds, weights), 0.0
    )
    np.testing.assert_allclose(
        pbc.probability_bias(climate, observed, weights).reshape(-1), [0.25, -0.25]
    )
    # Final CDF is one, so the upper-tail event and prediction are both zero.
    np.testing.assert_allclose(
        pbc.upper_tail_brier_score(perfect, observed, thresholds, weights), 0.0
    )
