"""Focused contracts for the validation-only persistence neural ablation."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import fuxi_allseason_ensemble_calibration as frozen
import fuxi_allseason_persistence_augmented as driver
import fuxi_persistence_context as persistence
from fuxi_pbc_core import (
    IssueTimeLags,
    build_daily_issue_time_lags,
    observation_cdf,
)


CANONICAL_SEEDS = (42, 43, 44)
VALIDATION_YEARS = (2018, 2019)


@dataclass(frozen=True)
class SyntheticPersistenceInputs:
    initializations: np.ndarray
    daily_dates: np.ndarray
    daily_values: np.ndarray
    lags: IssueTimeLags
    base_context: frozen.ContextBundle
    train_indices: np.ndarray
    weights: np.ndarray


def _synthetic_inputs() -> SyntheticPersistenceInputs:
    initializations = np.datetime64("2002-01-08") + np.arange(6).astype(
        "timedelta64[W]"
    )
    initializations = initializations.astype("datetime64[D]")
    daily_dates = np.arange(
        initializations[0] - np.timedelta64(7, "D"),
        initializations[-1],
        dtype="datetime64[D]",
    )
    day = np.arange(daily_dates.size, dtype=np.float32)[:, None, None]
    latitude = np.arange(27, dtype=np.float32)[None, :, None]
    longitude = np.arange(27, dtype=np.float32)[None, None, :]
    daily_values = (0.2 + 0.11 * day + 0.01 * latitude + 0.001 * longitude).astype(
        np.float32
    )
    lags = build_daily_issue_time_lags(
        initializations, daily_dates, daily_values, lag_weeks=(1, 2)
    )

    case = np.arange(initializations.size, dtype=np.float32)[:, None, None, None]
    lead = np.arange(6, dtype=np.float32)[None, :, None, None]
    row = np.arange(27, dtype=np.float32)[None, None, :, None]
    column = np.arange(27, dtype=np.float32)[None, None, None, :]
    normalized_climatology = case + 0.1 * lead + 0.01 * row + 0.001 * column
    phase = (
        np.arange(initializations.size, dtype=np.float32)[:, None]
        + np.arange(6, dtype=np.float32)[None]
    ) / 9.0
    support = np.ones((27, 27), dtype=bool)
    support[0, 0] = False
    support[5, 7] = False
    weights = np.broadcast_to(
        np.linspace(1.0, 1.5, 27, dtype=np.float64)[:, None], (27, 27)
    ).copy()
    weights[~support] = 0.0
    base_context = frozen.ContextBundle(
        normalized_climatology=normalized_climatology.astype(np.float32),
        climatology_mean_by_lead=np.zeros(6, dtype=np.float32),
        climatology_std_by_lead=np.ones(6, dtype=np.float32),
        latitude_scaled=np.linspace(-1.0, 1.0, 27, dtype=np.float32),
        longitude_scaled=np.linspace(-1.0, 1.0, 27, dtype=np.float32),
        season_sin=np.sin(phase).astype(np.float32),
        season_cos=np.cos(phase).astype(np.float32),
        lead_scaled=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        support=support,
    )
    return SyntheticPersistenceInputs(
        initializations=initializations,
        daily_dates=daily_dates,
        daily_values=daily_values,
        lags=lags,
        base_context=base_context,
        train_indices=np.asarray([1, 2, 3], dtype=np.int64),
        weights=weights,
    )


def _copy_lags(
    lags: IssueTimeLags,
    *,
    values: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    window_start: np.ndarray | None = None,
    window_end: np.ndarray | None = None,
) -> IssueTimeLags:
    return IssueTimeLags(
        values=lags.values.copy() if values is None else values,
        source_indices=(
            lags.source_indices.copy() if source_indices is None else source_indices
        ),
        window_start=lags.window_start.copy() if window_start is None else window_start,
        window_end=lags.window_end.copy() if window_end is None else window_end,
    )


def _context_bundle(
    inputs: SyntheticPersistenceInputs,
    lags: IssueTimeLags | None = None,
) -> persistence.PersistenceContextBundle:
    return persistence.build_persistence_context_bundle(
        inputs.base_context,
        inputs.initializations,
        inputs.lags if lags is None else lags,
        inputs.train_indices,
        inputs.weights,
    )


def _validation_metrics() -> pd.DataFrame:
    scores = {
        persistence.BASE_ARM: (1.0, 1.0),
        persistence.ZERO_LAG_ARM: (1.0, 1.0),
        persistence.PERSISTENCE_LAG_ARM: (0.99, 0.98),
    }
    rows: list[dict[str, object]] = []
    for arm in persistence.ARMS:
        crps, rps = scores[arm]
        for seed in CANONICAL_SEEDS:
            for year in VALIDATION_YEARS:
                for lead in range(1, 7):
                    rows.append(
                        {
                            "split": "validation",
                            "arm": arm,
                            "seed": seed,
                            "init": f"{year}-01-04",
                            "year": year,
                            "lead_week": lead,
                            "crps": crps,
                            "quintile_rps": rps,
                            "score_contract": driver.SCORING_CONTRACT_VERSION,
                        }
                    )
    return pd.DataFrame(rows)


def test_daily_lags_are_exact_preissuance_windows_and_ignore_future_mutations() -> None:
    inputs = _synthetic_inputs()
    available = inputs.lags.source_indices >= 0
    expected_start = inputs.initializations[:, None] - np.asarray(
        [7, 14], dtype="timedelta64[D]"
    )[None]
    expected_end = expected_start + np.timedelta64(6, "D")

    np.testing.assert_array_equal(
        inputs.lags.window_start[available], expected_start[available]
    )
    np.testing.assert_array_equal(
        inputs.lags.window_end[available], expected_end[available]
    )
    issue_grid = np.broadcast_to(inputs.initializations[:, None], available.shape)
    assert np.all(inputs.lags.window_end[available] < issue_grid[available])
    np.testing.assert_array_equal(
        persistence.validate_issue_time_lags(
            inputs.initializations, inputs.lags, inputs.base_context.support
        ),
        available,
    )

    issue_index = 2
    changed_daily = inputs.daily_values.copy()
    changed_daily[inputs.daily_dates >= inputs.initializations[issue_index]] += 10_000.0
    changed_lags = build_daily_issue_time_lags(
        inputs.initializations,
        inputs.daily_dates,
        changed_daily,
        lag_weeks=(1, 2),
    )
    np.testing.assert_array_equal(
        changed_lags.values[issue_index], inputs.lags.values[issue_index]
    )
    np.testing.assert_array_equal(
        changed_lags.source_indices[issue_index],
        inputs.lags.source_indices[issue_index],
    )
    assert not np.array_equal(
        changed_lags.values[issue_index + 1, 0],
        inputs.lags.values[issue_index + 1, 0],
    )


def test_context_rejects_a_lag_window_that_reaches_issuance() -> None:
    inputs = _synthetic_inputs()
    bad_end = inputs.lags.window_end.copy()
    bad_end[2, 0] = inputs.initializations[2]
    malicious = _copy_lags(inputs.lags, window_end=bad_end)

    with pytest.raises(
        persistence.PersistenceContextError, match="incorrect end date|issuance"
    ):
        _context_bundle(inputs, malicious)


def test_lag_normalization_is_log1p_separate_by_lag_and_train_only() -> None:
    inputs = _synthetic_inputs()
    original = _context_bundle(inputs)
    changed_values = inputs.lags.values.copy()
    for case in (4, 5):
        for column in range(2):
            if inputs.lags.source_indices[case, column] >= 0:
                changed_values[case, column] += 50_000.0
    changed = _context_bundle(inputs, _copy_lags(inputs.lags, values=changed_values))

    np.testing.assert_array_equal(
        original.normalization_fit_indices, inputs.train_indices
    )
    np.testing.assert_array_equal(
        original.lag_log1p_mean_by_lag, changed.lag_log1p_mean_by_lag
    )
    np.testing.assert_array_equal(
        original.lag_log1p_std_by_lag, changed.lag_log1p_std_by_lag
    )
    np.testing.assert_array_equal(
        original.normalized_lag_log1p[inputs.train_indices],
        changed.normalized_lag_log1p[inputs.train_indices],
    )
    assert not np.array_equal(
        original.normalized_lag_log1p[4:, :, inputs.base_context.support],
        changed.normalized_lag_log1p[4:, :, inputs.base_context.support],
    )

    usable = (inputs.lags.source_indices >= 0)[..., None, None] & (
        inputs.base_context.support[None, None]
    )
    log_lags = np.full(inputs.lags.values.shape, np.nan, dtype=np.float32)
    np.log1p(inputs.lags.values, out=log_lags, where=usable)
    expected_mean, expected_std = frozen.weighted_lead_moments(
        log_lags, inputs.train_indices, inputs.weights
    )
    np.testing.assert_array_equal(original.lag_log1p_mean_by_lag, expected_mean)
    np.testing.assert_array_equal(original.lag_log1p_std_by_lag, expected_std)
    assert not original.normalized_lag_log1p.flags.writeable
    assert not original.lag_available.flags.writeable
    assert not original.normalization_fit_indices.flags.writeable


def test_partial_lags_and_all_three_context_schemas_are_exact() -> None:
    inputs = _synthetic_inputs()
    bundle = _context_bundle(inputs)

    # The daily archive begins seven days before the first issuance: lag 1 is
    # usable, while lag 2 is independently absent. Never collapse this to
    # IssueTimeLags.available, which would discard the valid first lag.
    np.testing.assert_array_equal(bundle.lag_available[0], [True, False])
    assert not inputs.lags.available[0]
    base_context = persistence.context_for_case(bundle, 0, persistence.BASE_ARM)
    frozen_context = frozen.context_for_case(inputs.base_context, 0)
    zero_context = persistence.context_for_case(bundle, 0, persistence.ZERO_LAG_ARM)
    lag_context = persistence.context_for_case(
        bundle, 0, persistence.PERSISTENCE_LAG_ARM
    )

    np.testing.assert_array_equal(base_context, frozen_context)
    assert base_context.shape == (6, 7, 27, 27)
    assert zero_context.shape == lag_context.shape == (6, 11, 27, 27)
    assert zero_context.dtype == lag_context.dtype == np.float32
    assert zero_context.flags.c_contiguous and lag_context.flags.c_contiguous
    assert np.isfinite(zero_context).all() and np.isfinite(lag_context).all()
    np.testing.assert_array_equal(zero_context[:, :7], frozen_context)
    np.testing.assert_array_equal(lag_context[:, :7], frozen_context)
    assert np.count_nonzero(zero_context[:, 7:]) == 0

    # Added order is lag1-z, lag2-z, lag1-mask, lag2-mask. All four fields are
    # issue-time fields and therefore broadcast unchanged over forecast lead.
    assert persistence.context_channel_names_for_arm(
        persistence.PERSISTENCE_LAG_ARM
    ) == (
        *persistence.BASE_CONTEXT_CHANNEL_NAMES,
        *persistence.LAG_CONTEXT_CHANNEL_NAMES,
    )
    assert persistence.LAG_CONTEXT_CHANNEL_NAMES == (
        "lag_1week_log1p_z",
        "lag_2week_log1p_z",
        "lag_1week_available",
        "lag_2week_available",
    )
    for lead in range(1, 6):
        np.testing.assert_array_equal(lag_context[lead, 7:], lag_context[0, 7:])
    assert np.count_nonzero(lag_context[:, 7]) > 0
    assert np.count_nonzero(lag_context[:, 8]) == 0
    assert np.all(lag_context[:, 9, inputs.base_context.support] == 1.0)
    assert np.count_nonzero(lag_context[:, 10]) == 0
    assert np.count_nonzero(lag_context[:, 7:, ~inputs.base_context.support]) == 0

    members = np.zeros((6, 3, 6, 27, 27), dtype=np.float32)
    truth = np.ones((6, 6, 27, 27), dtype=np.float32)
    dataset = persistence.PersistenceCaseDataset(
        members,
        truth,
        bundle,
        np.asarray([0, 2], dtype=np.int64),
        persistence.PERSISTENCE_LAG_ARM,
    )
    member_tensor, context_tensor, truth_tensor = dataset[0]
    assert member_tensor.shape == (3, 6, 27, 27)
    assert context_tensor.shape == (6, 11, 27, 27)
    assert truth_tensor.shape == (6, 27, 27)
    assert member_tensor.dtype == context_tensor.dtype == truth_tensor.dtype == torch.float32


def test_seeded_models_have_exact_counts_transplant_and_matched_45k_state() -> None:
    base_model = driver.build_seeded_model(persistence.BASE_ARM, 42)
    zero_model = driver.build_seeded_model(persistence.ZERO_LAG_ARM, 42)
    lag_model = driver.build_seeded_model(persistence.PERSISTENCE_LAG_ARM, 42)

    models = {
        persistence.BASE_ARM: base_model,
        persistence.ZERO_LAG_ARM: zero_model,
        persistence.PERSISTENCE_LAG_ARM: lag_model,
    }
    for arm, model in models.items():
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            driver.EXPECTED_PARAMETER_COUNTS[arm]
        )
        assert model.context_channels == persistence.context_channels_for_arm(arm)
    assert driver.EXPECTED_PARAMETER_COUNTS == {
        persistence.BASE_ARM: 42_434,
        persistence.ZERO_LAG_ARM: 45_026,
        persistence.PERSISTENCE_LAG_ARM: 45_026,
    }
    assert driver.model_state_sha256(zero_model) == driver.model_state_sha256(lag_model)

    base_state = base_model.state_dict()
    zero_state = zero_model.state_dict()
    changed_shape = "backbone.input.weight"
    original_width = base_state[changed_shape].shape[1]
    for name, tensor in base_state.items():
        if name == changed_shape:
            np.testing.assert_array_equal(
                zero_state[name][:, :original_width].numpy(), tensor.numpy()
            )
            assert torch.count_nonzero(zero_state[name][:, original_width:]) == 0
        else:
            assert torch.equal(zero_state[name], tensor), name

    # Construction resets the seed, so unrelated RNG consumption cannot alter a
    # same-seed control receipt.
    _ = torch.rand(29)
    rebuilt = driver.build_seeded_model(persistence.BASE_ARM, 42)
    assert driver.model_state_sha256(rebuilt) == driver.model_state_sha256(base_model)
    assert driver.model_state_sha256(
        driver.build_seeded_model(persistence.BASE_ARM, 43)
    ) != driver.model_state_sha256(base_model)


def test_training_rng_receipt_is_resettable_without_advancing_the_stream() -> None:
    device = torch.device("cpu")
    frozen.set_deterministic_seed(42)
    first = driver.training_rng_state_sha256(device)
    assert driver.training_rng_state_sha256(device) == first
    _ = torch.rand(1)
    assert driver.training_rng_state_sha256(device) != first
    frozen.set_deterministic_seed(42)
    assert driver.training_rng_state_sha256(device) == first

    receipts = {}
    for arm in persistence.ARMS:
        # Model construction consumes a different number of draws for 7 versus
        # 11 input channels. The explicit pre-training reset must erase that
        # difference for every arm, not merely for the two widened controls.
        driver.build_seeded_model(arm, 42)
        frozen.set_deterministic_seed(42)
        receipts[arm] = driver.training_rng_state_sha256(device)
    assert set(receipts) == set(persistence.ARMS)
    assert len(set(receipts.values())) == 1
    assert next(iter(receipts.values())) == first

    source = textwrap.dedent(inspect.getsource(driver.train_one_arm))
    assert source.index("base.set_deterministic_seed(int(seed))") > source.index(
        "initial_validation"
    )
    assert source.index("training_rng_state_sha256(device)") < source.index(
        "for epoch in range"
    )


@pytest.mark.parametrize("arm", persistence.ARMS)
def test_every_arm_starts_as_an_exact_physical_noop(arm: str) -> None:
    generator = torch.Generator().manual_seed(811)
    members = 10.0 * torch.rand(1, 4, 6, 3, 2, generator=generator)
    context = torch.randn(
        1,
        6,
        persistence.context_channels_for_arm(arm),
        3,
        2,
        generator=generator,
    )
    output = driver.build_seeded_model(arm, 42).eval()(members, context)

    assert torch.equal(output.corrected_members, members)
    assert torch.count_nonzero(output.delta_log_location) == 0
    assert torch.count_nonzero(output.log_spread) == 0
    assert torch.equal(output.spread_factor, torch.ones_like(output.spread_factor))


def test_selector_promotes_only_persistence_after_every_frozen_guard() -> None:
    selection = driver.select_persistence_arm(
        _validation_metrics(),
        expected_initializations=np.asarray(
            ["2018-01-04", "2019-01-04"], dtype="datetime64[D]"
        ),
    )

    assert selection["status"] == "validation_selection_locked"
    assert selection["selected_arm"] == persistence.PERSISTENCE_LAG_ARM
    assert selection["candidate_promoted"] is True
    assert selection["test_metrics_consulted"] is False
    assert selection["zero_lag_selectable"] is False
    assert selection["zero_lag_control_reproduction"]["passes"] is True
    for reference in (persistence.ZERO_LAG_ARM, persistence.BASE_ARM):
        comparison = selection["candidate_comparisons"][reference]
        assert comparison["pooled_rps_minimum_improvement_guard"] is True
        assert comparison["w2_w6_rps_minimum_improvement_guard"] is True
        assert comparison["pooled_crps_noninferiority_guard"] is True
        assert comparison["all_years_rps_noninferiority_guard"] is True
        assert comparison["matched_seed_rps_guard"] is True
        assert comparison["matched_seed_rps_improvement_passes"] == 3
        assert comparison["passes"] is True


@pytest.mark.parametrize(
    ("scenario", "failed_guard"),
    (
        ("pooled_rps", "pooled_rps_minimum_improvement_guard"),
        ("w2_w6_rps", "w2_w6_rps_minimum_improvement_guard"),
        ("pooled_crps", "pooled_crps_noninferiority_guard"),
        ("year_rps", "all_years_rps_noninferiority_guard"),
        ("matched_seed_rps", "matched_seed_rps_guard"),
    ),
)
def test_each_candidate_promotion_gate_fails_closed(
    scenario: str, failed_guard: str
) -> None:
    metrics = _validation_metrics()
    candidate = metrics.arm.eq(persistence.PERSISTENCE_LAG_ARM)
    if scenario == "pooled_rps":
        metrics.loc[candidate & metrics.lead_week.eq(1), "quintile_rps"] = 1.047
        metrics.loc[candidate & metrics.lead_week.ge(2), "quintile_rps"] = 0.99
    elif scenario == "w2_w6_rps":
        metrics.loc[candidate & metrics.lead_week.eq(1), "quintile_rps"] = 0.90
        metrics.loc[candidate & metrics.lead_week.ge(2), "quintile_rps"] = 0.997
    elif scenario == "pooled_crps":
        metrics.loc[candidate, "crps"] = 1.001
    elif scenario == "year_rps":
        metrics.loc[candidate & metrics.year.eq(2018), "quintile_rps"] = 0.95
        metrics.loc[candidate & metrics.year.eq(2019), "quintile_rps"] = 1.001
    elif scenario == "matched_seed_rps":
        metrics.loc[candidate & metrics.seed.eq(42), "quintile_rps"] = 0.94
        metrics.loc[candidate & metrics.seed.ne(42), "quintile_rps"] = 1.0
    else:  # pragma: no cover - parameterization is frozen above
        raise AssertionError(scenario)

    selection = driver.select_persistence_arm(metrics)

    assert selection["selected_arm"] == persistence.BASE_ARM
    assert selection["candidate_promoted"] is False
    guard_names = {
        "pooled_rps_minimum_improvement_guard",
        "w2_w6_rps_minimum_improvement_guard",
        "pooled_crps_noninferiority_guard",
        "all_years_rps_noninferiority_guard",
        "matched_seed_rps_guard",
    }
    for reference in (persistence.ZERO_LAG_ARM, persistence.BASE_ARM):
        comparison = selection["candidate_comparisons"][reference]
        assert comparison[failed_guard] is False
        for other in guard_names - {failed_guard}:
            assert comparison[other] is True
        assert comparison["passes"] is False


@pytest.mark.parametrize("metric", ("crps", "quintile_rps"))
def test_zero_control_reproduction_gate_fails_closed(metric: str) -> None:
    metrics = _validation_metrics()
    metrics.loc[metrics.arm.eq(persistence.ZERO_LAG_ARM), metric] = 1.003

    selection = driver.select_persistence_arm(metrics)

    reproduction = selection["zero_lag_control_reproduction"]
    assert reproduction["passes"] is False
    assert reproduction[f"{metric}_within_tolerance"] is False
    assert selection["candidate_promoted"] is False
    assert selection["selected_arm"] == persistence.BASE_ARM
    assert all(
        comparison["passes"]
        for comparison in selection["candidate_comparisons"].values()
    )


@pytest.mark.parametrize(
    "scenario",
    (
        "missing_column",
        "development_row",
        "missing_arm",
        "wrong_score_contract",
        "nonfinite_score",
        "fractional_lead",
        "missing_global_lead",
        "incomplete_case_leads",
        "missing_seed",
        "sealed_year",
        "duplicate_key",
        "misaligned_arm_case",
        "misaligned_seed_inventory",
    ),
)
def test_selector_rejects_incomplete_or_contaminated_inputs(scenario: str) -> None:
    metrics = _validation_metrics()
    if scenario == "missing_column":
        metrics = metrics.drop(columns="quintile_rps")
    elif scenario == "development_row":
        metrics.loc[0, "split"] = "test_development"
    elif scenario == "missing_arm":
        metrics = metrics.loc[metrics.arm.ne(persistence.ZERO_LAG_ARM)]
    elif scenario == "wrong_score_contract":
        metrics.loc[:, "score_contract"] = "legacy_rps"
    elif scenario == "nonfinite_score":
        metrics.loc[0, "crps"] = np.nan
    elif scenario == "fractional_lead":
        metrics["lead_week"] = metrics.lead_week.astype(np.float64)
        metrics.loc[metrics.lead_week.eq(1), "lead_week"] = 1.5
    elif scenario == "missing_global_lead":
        metrics = metrics.loc[metrics.lead_week.ne(6)]
    elif scenario == "incomplete_case_leads":
        drop = (
            metrics.seed.eq(42)
            & metrics.year.eq(2018)
            & metrics.lead_week.eq(6)
        )
        metrics = metrics.loc[~drop]
    elif scenario == "missing_seed":
        metrics = metrics.loc[metrics.seed.ne(44)]
    elif scenario == "sealed_year":
        selected = metrics.year.eq(2019)
        metrics.loc[selected, "year"] = 2025
        metrics.loc[selected, "init"] = "2025-01-04"
    elif scenario == "duplicate_key":
        metrics = pd.concat((metrics, metrics.iloc[[0]]), ignore_index=True)
    elif scenario == "misaligned_arm_case":
        selected = metrics.arm.eq(persistence.PERSISTENCE_LAG_ARM) & metrics.seed.eq(42)
        metrics.loc[selected & metrics.year.eq(2018), "init"] = "2018-02-01"
    elif scenario == "misaligned_seed_inventory":
        selected = metrics.seed.eq(42) & metrics.year.eq(2018)
        metrics.loc[selected, "init"] = "2018-02-01"
    else:  # pragma: no cover - parameterization is frozen above
        raise AssertionError(scenario)

    with pytest.raises(ValueError):
        driver.select_persistence_arm(metrics)


def test_selector_requires_the_exact_expected_initialization_inventory() -> None:
    metrics = _validation_metrics()
    expected = np.asarray(
        ["2018-01-04", "2019-01-04"], dtype="datetime64[D]"
    )
    assert driver.select_persistence_arm(
        metrics, expected_initializations=expected
    )["candidate_promoted"]

    with pytest.raises(ValueError, match="expected validation initialization inventory"):
        driver.select_persistence_arm(
            metrics,
            expected_initializations=np.asarray(
                ["2018-01-04", "2019-01-04", "2019-02-01"],
                dtype="datetime64[D]",
            ),
        )
    with pytest.raises(ValueError, match="inventory is not unique"):
        driver.select_persistence_arm(
            metrics,
            expected_initializations=np.asarray(
                ["2018-01-04", "2019-01-04", "2019-01-04"],
                dtype="datetime64[D]",
            ),
        )


def test_validation_scorer_is_validation_only_and_emits_complete_leads() -> None:
    cases = 4
    members = np.empty((cases, 3, 6, 27, 27), dtype=np.float32)
    members[:, 0] = 0.5
    members[:, 1] = 1.5
    members[:, 2] = 2.5
    truth = np.full((cases, 6, 27, 27), 2.25, dtype=np.float32)
    initializations = np.asarray(
        ["2017-01-05", "2018-01-04", "2019-01-03", "2020-01-02"],
        dtype="datetime64[D]",
    )
    validation_indices = np.asarray([1, 2], dtype=np.int64)
    adjustment = np.zeros((2, 6, 27, 27), dtype=np.float32)
    threshold_values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    thresholds = np.broadcast_to(
        threshold_values[None, None, :, None, None], (2, 6, 4, 27, 27)
    ).copy()
    observed = observation_cdf(truth[validation_indices], thresholds)
    weights = np.ones((27, 27), dtype=np.float64)

    metrics = driver.validation_case_metrics(
        persistence.BASE_ARM,
        42,
        members,
        truth,
        initializations,
        validation_indices,
        adjustment,
        adjustment,
        weights,
        thresholds,
        observed,
        chunk_size=1,
    )

    assert len(metrics) == 12
    assert set(metrics.split) == {"validation"}
    assert set(metrics.year) == {2018, 2019}
    assert set(metrics.lead_week) == set(range(1, 7))
    assert set(metrics.score_contract) == {driver.SCORING_CONTRACT_VERSION}
    assert np.isfinite(metrics[["crps", "quintile_rps"]]).all().all()
    assert (metrics[["crps", "quintile_rps"]] >= 0.0).all().all()
    evidence_columns = (
        "cdf_shape_matches_thresholds",
        "forecast_cdf_finite_where_threshold_defined",
        "observed_cdf_finite_where_threshold_defined",
        "forecast_cdf_valid_for_thresholds",
        "observed_cdf_valid_for_thresholds",
    )
    assert all(column in metrics for column in evidence_columns)
    assert metrics[list(evidence_columns)].to_numpy(dtype=bool).all()

    malformed_observed = observed.copy()
    malformed_observed[:, :, 1] = 1.0
    malformed_observed[:, :, 2] = 0.0
    with pytest.raises(
        driver.PersistenceExperimentError, match="categorical CDF"
    ):
        driver.validation_case_metrics(
            persistence.BASE_ARM,
            42,
            members,
            truth,
            initializations,
            validation_indices,
            adjustment,
            adjustment,
            weights,
            thresholds,
            malformed_observed,
            chunk_size=1,
        )

    with pytest.raises(driver.PersistenceExperimentError, match="validation years"):
        driver.validation_case_metrics(
            persistence.BASE_ARM,
            42,
            members,
            truth,
            initializations,
            np.asarray([2, 3], dtype=np.int64),
            adjustment,
            adjustment,
            weights,
            thresholds,
            observed,
            chunk_size=1,
        )


def test_driver_source_has_no_development_or_sealed_scoring_path() -> None:
    source = textwrap.dedent(inspect.getsource(driver.run_experiment))
    tree = ast.parse(source)
    split_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "splits"
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "test" not in split_attributes
    assert "test_indices" not in referenced_names
    assert "development_indices" not in referenced_names

    guarded_calls: dict[str, list[ast.Call]] = {
        name: []
        for name in (
            "fit_calendar_quantiles",
            "predict_adjustments",
            "validation_case_metrics",
            "select_persistence_arm",
        )
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in guarded_calls:
            guarded_calls[name].append(node)
    assert len(guarded_calls["fit_calendar_quantiles"]) == 1
    assert ast.unparse(guarded_calls["fit_calendar_quantiles"][0].args[2]) == (
        "train_indices"
    )
    assert len(guarded_calls["predict_adjustments"]) == 1
    assert ast.unparse(guarded_calls["predict_adjustments"][0].args[5]) == (
        "validation_indices"
    )
    assert len(guarded_calls["validation_case_metrics"]) == 1
    assert ast.unparse(guarded_calls["validation_case_metrics"][0].args[5]) == (
        "validation_indices"
    )
    assert len(guarded_calls["select_persistence_arm"]) == 1
    assert ast.unparse(guarded_calls["select_persistence_arm"][0].args[0]) == (
        "validation_frame"
    )
    assert frozen.SEALED_YEARS == (2025,)
    options = {
        option
        for action in driver.build_parser()._actions
        for option in action.option_strings
    }
    assert not any("development" in option or "test-year" in option for option in options)


def test_validate_args_freezes_smoke_and_hash_bound_full_runs() -> None:
    parser = driver.build_parser()
    smoke = parser.parse_args(["--smoke"])
    driver.validate_args(smoke)
    assert smoke.arms == ",".join(persistence.ARMS)
    assert smoke.seeds == "42"
    assert smoke.max_epochs == 2
    assert smoke.patience == 1
    assert smoke.expected_persistence_lag_sha256 is None
    assert smoke.expected_observation_bundle_sha256 is None

    digest_a = "a" * 64
    digest_b = "b" * 64
    full = parser.parse_args(
        [
            "--expected-persistence-lag-sha256",
            digest_a,
            "--expected-observation-bundle-sha256",
            digest_b,
        ]
    )
    driver.validate_args(full)
    assert full.seeds == "42,43,44"
    assert full.max_epochs == 100
    assert full.patience == 15

    invalid_argv = (
        (),
        ("--smoke", "--expected-persistence-lag-sha256", digest_a),
        ("--expected-persistence-lag-sha256", digest_a),
        (
            "--expected-persistence-lag-sha256",
            "A" * 64,
            "--expected-observation-bundle-sha256",
            digest_b,
        ),
        (
            "--arms",
            persistence.BASE_ARM,
            "--expected-persistence-lag-sha256",
            digest_a,
            "--expected-observation-bundle-sha256",
            digest_b,
        ),
        (
            "--device",
            "cpu",
            "--expected-persistence-lag-sha256",
            digest_a,
            "--expected-observation-bundle-sha256",
            digest_b,
        ),
        (
            "--no-amp",
            "--expected-persistence-lag-sha256",
            digest_a,
            "--expected-observation-bundle-sha256",
            digest_b,
        ),
    )
    for argv in invalid_argv:
        with pytest.raises(ValueError):
            driver.validate_args(parser.parse_args(argv))


def test_main_is_atomic_and_retains_failure_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "persistence-smoke"

    def fake_run(args, staging: Path):
        assert args.output == published.resolve()
        assert staging.name.startswith(".persistence-smoke.incomplete-")
        driver.write_json(
            staging / "manifest.json",
            {"experiment": driver.EXPERIMENT, "status": "complete", "mode": "smoke"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(driver, "run_experiment", fake_run)
    assert driver.main(["--smoke", "--output", str(published)]) == 0
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "experiment": driver.EXPERIMENT,
        "mode": "smoke",
        "status": "complete",
    }
    assert not list(tmp_path.glob(".persistence-smoke.incomplete-*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        driver.main(["--smoke", "--output", str(published)])

    failed = tmp_path / "failed"

    def fail_run(args, staging: Path):
        raise RuntimeError("synthetic persistence failure")

    monkeypatch.setattr(driver, "run_experiment", fail_run)
    with pytest.raises(RuntimeError, match="synthetic persistence failure"):
        driver.main(["--smoke", "--output", str(failed)])
    retained = list(tmp_path.glob(".failed.incomplete-*"))
    assert len(retained) == 1
    failure = json.loads((retained[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["experiment"] == driver.EXPERIMENT
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "synthetic persistence failure"
    assert failure["requested_output"] == str(failed.resolve())
