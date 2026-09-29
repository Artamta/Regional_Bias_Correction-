"""Focused contracts for the frozen active-mask persistence addendum."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_persistence_mask_control as driver
import fuxi_allseason_persistence_augmented as v1
import fuxi_persistence_context as persistence
import fuxi_persistence_mask_control as mask_control


SEEDS = (42, 43, 44)
VALIDATION_DATES = np.asarray(
    ["2018-01-04", "2019-01-03"], dtype="datetime64[D]"
)
CDF_EVIDENCE = (
    "cdf_shape_matches_thresholds",
    "forecast_cdf_finite_where_threshold_defined",
    "observed_cdf_finite_where_threshold_defined",
    "forecast_cdf_valid_for_thresholds",
    "observed_cdf_valid_for_thresholds",
)


def _context_bundle() -> persistence.PersistenceContextBundle:
    cases = 4
    support = np.ones((27, 27), dtype=bool)
    support[0, 0] = False
    case = np.arange(cases, dtype=np.float32)[:, None, None, None]
    lead = np.arange(6, dtype=np.float32)[None, :, None, None]
    normalized_climatology = np.broadcast_to(
        case + 0.1 * lead, (cases, 6, 27, 27)
    ).copy()
    base_context = base.ContextBundle(
        normalized_climatology=normalized_climatology,
        climatology_mean_by_lead=np.zeros(6, dtype=np.float32),
        climatology_std_by_lead=np.ones(6, dtype=np.float32),
        latitude_scaled=np.linspace(-1.0, 1.0, 27, dtype=np.float32),
        longitude_scaled=np.linspace(-1.0, 1.0, 27, dtype=np.float32),
        season_sin=np.zeros((cases, 6), dtype=np.float32),
        season_cos=np.ones((cases, 6), dtype=np.float32),
        lead_scaled=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        support=support,
    )
    availability = np.asarray(
        [[False, False], [True, False], [True, True], [False, True]],
        dtype=np.bool_,
    )
    normalized_lags = np.zeros((cases, 2, 27, 27), dtype=np.float32)
    for case_index in range(cases):
        for lag_index in range(2):
            if availability[case_index, lag_index]:
                normalized_lags[case_index, lag_index, support] = (
                    10.0 * case_index + lag_index + 1.0
                )
    return persistence.PersistenceContextBundle(
        base_context=base_context,
        normalized_lag_log1p=normalized_lags,
        lag_available=availability,
        lag_log1p_mean_by_lag=np.asarray([0.5, 0.7], dtype=np.float32),
        lag_log1p_std_by_lag=np.asarray([0.2, 0.3], dtype=np.float32),
        normalization_fit_indices=np.asarray([0, 1], dtype=np.int64),
    )


def _metric_frames(
    *, seeds: tuple[int, ...] = SEEDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_scores = {
        persistence.BASE_ARM: (1.0, 1.0),
        persistence.ZERO_LAG_ARM: (1.0, 1.0),
        persistence.PERSISTENCE_LAG_ARM: (0.99, 0.98),
    }
    parent_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    for seed in seeds:
        for value in VALIDATION_DATES:
            date = np.datetime_as_string(value, unit="D")
            year = int(date[:4])
            for lead_week in range(1, 7):
                common: dict[str, object] = {
                    "split": "validation",
                    "seed": seed,
                    "init": date,
                    "year": year,
                    "lead_week": lead_week,
                    "score_contract": v1.SCORING_CONTRACT_VERSION,
                    **{name: True for name in CDF_EVIDENCE},
                }
                for arm in persistence.ARMS:
                    crps, rps = parent_scores[arm]
                    parent_rows.append(
                        {**common, "arm": arm, "crps": crps, "quintile_rps": rps}
                    )
                mask_rows.append(
                    {
                        **common,
                        "arm": mask_control.MASK_ONLY_ARM,
                        "crps": 1.0,
                        "quintile_rps": 1.0,
                    }
                )
    return pd.DataFrame(parent_rows), pd.DataFrame(mask_rows)


def _select(
    parent: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    return driver.select_joint_arm(
        parent,
        mask,
        expected_seeds=seeds,
        expected_initializations=VALIDATION_DATES,
    )


def test_mask_context_has_exact_zero_rain_and_true_candidate_masks() -> None:
    bundle = _context_bundle()
    for case_index in range(bundle.case_count):
        mask_fields = mask_control.mask_only_context_for_case(bundle, case_index)
        candidate = persistence.context_for_case(
            bundle, case_index, persistence.PERSISTENCE_LAG_ARM
        )
        frozen_base = base.context_for_case(bundle.base_context, case_index)
        assert mask_fields.shape == (6, 11, 27, 27)
        assert mask_fields.dtype == np.float32 and mask_fields.flags.c_contiguous
        np.testing.assert_array_equal(mask_fields[:, :7], frozen_base)
        assert np.count_nonzero(mask_fields[:, 7:9]) == 0
        np.testing.assert_array_equal(mask_fields[:, 9:11], candidate[:, 9:11])
        assert np.count_nonzero(mask_fields[:, 7:, ~bundle.base_context.support]) == 0
        for lead in range(1, 6):
            np.testing.assert_array_equal(mask_fields[lead, 7:], mask_fields[0, 7:])


def test_mask_dataset_is_lazy_and_retains_only_explicit_indices() -> None:
    bundle = _context_bundle()
    members = np.zeros((4, 3, 6, 27, 27), dtype=np.float32)
    truth = np.ones((4, 6, 27, 27), dtype=np.float32)
    dataset = mask_control.MaskOnlyCaseDataset(
        members, truth, bundle, np.asarray([0, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(dataset.indices, [0, 2])
    member_tensor, context_tensor, truth_tensor = dataset[1]
    assert member_tensor.shape == (3, 6, 27, 27)
    assert context_tensor.shape == (6, 11, 27, 27)
    assert truth_tensor.shape == (6, 27, 27)
    assert member_tensor.dtype == context_tensor.dtype == truth_tensor.dtype == torch.float32
    with pytest.raises(persistence.PersistenceContextError, match="strictly increasing"):
        mask_control.MaskOnlyCaseDataset(
            members, truth, bundle, np.asarray([2, 0], dtype=np.int64)
        )


def test_mask_model_matches_the_original_widened_initial_state_and_noop() -> None:
    mask_model = driver.new_mask_model(42)
    original = v1.build_seeded_model(persistence.PERSISTENCE_LAG_ARM, 42)
    assert sum(parameter.numel() for parameter in mask_model.parameters()) == 45_026
    assert v1.model_state_sha256(mask_model) == v1.model_state_sha256(original)
    widened = mask_model.state_dict()["backbone.input.weight"]
    original_width = v1.build_seeded_model(
        persistence.BASE_ARM, 42
    ).state_dict()["backbone.input.weight"].shape[1]
    assert torch.count_nonzero(widened[:, original_width:]) == 0

    generator = torch.Generator().manual_seed(19)
    members = torch.rand(1, 4, 6, 3, 2, generator=generator)
    context = torch.rand(1, 6, 11, 3, 2, generator=generator)
    result = mask_model.eval()(members, context)
    assert torch.equal(result.corrected_members, members)
    assert torch.count_nonzero(result.delta_log_location) == 0
    assert torch.count_nonzero(result.log_spread) == 0


def test_mask_training_resets_the_same_rng_stream_after_initial_validation() -> None:
    source = textwrap.dedent(inspect.getsource(driver.train_mask_only))
    assert source.index("base.set_deterministic_seed(int(seed))") > source.index(
        "initial_validation"
    )
    assert source.index("training_rng_state_sha256(device)") < source.index(
        "for epoch in range"
    )
    device = torch.device("cpu")
    receipts = []
    for seed in SEEDS:
        driver.new_mask_model(seed)
        base.set_deterministic_seed(seed)
        mask_receipt = v1.training_rng_state_sha256(device)
        base.set_deterministic_seed(seed)
        parent_receipt = v1.training_rng_state_sha256(device)
        assert mask_receipt == parent_receipt
        receipts.append(mask_receipt)
    assert len(set(receipts)) == len(SEEDS)


def test_joint_selector_promotes_only_after_original_and_mask_gates() -> None:
    parent, mask = _metric_frames()
    selection = _select(parent, mask)
    assert selection["status"] == driver.JOINT_SELECTION_STATUS
    assert selection["selected_arm"] == persistence.PERSISTENCE_LAG_ARM
    assert selection["candidate_promoted"] is True
    assert selection["mask_only_selectable"] is False
    assert selection["test_metrics_consulted"] is False
    assert selection["development_indices_accessed"] is False
    assert selection["original_three_arm_gates"]["passes"] is True
    comparison = selection["mask_only_comparison"]
    for gate in (
        "pooled_rps_minimum_improvement_guard",
        "w2_w6_rps_minimum_improvement_guard",
        "pooled_crps_noninferiority_guard",
        "all_years_rps_noninferiority_guard",
        "matched_seed_rps_guard",
        "passes",
    ):
        assert comparison[gate] is True


@pytest.mark.parametrize(
    "scenario",
    ("pooled_rps", "late_rps", "crps", "year", "matched_seed"),
)
def test_every_mask_attribution_gate_fails_closed(scenario: str) -> None:
    parent, mask = _metric_frames()
    candidate = parent.arm.eq(persistence.PERSISTENCE_LAG_ARM)
    if scenario == "pooled_rps":
        mask.loc[mask.lead_week.eq(1), "quintile_rps"] = 0.89
    elif scenario == "late_rps":
        mask.loc[mask.lead_week.eq(1), "quintile_rps"] = 1.5
        mask.loc[mask.lead_week.ge(2), "quintile_rps"] = 0.982
    elif scenario == "crps":
        mask.loc[:, "crps"] = 0.98
    elif scenario == "year":
        mask.loc[mask.year.eq(2018), "quintile_rps"] = 0.97
        mask.loc[mask.year.eq(2019), "quintile_rps"] = 1.1
    elif scenario == "matched_seed":
        mask.loc[mask.seed.eq(42), "quintile_rps"] = 1.0
        mask.loc[mask.seed.ne(42), "quintile_rps"] = 0.98
    else:  # pragma: no cover
        raise AssertionError(scenario)
    selection = _select(parent, mask)
    assert selection["selected_arm"] == persistence.BASE_ARM
    assert selection["candidate_promoted"] is False
    assert selection["mask_only_selectable"] is False
    assert selection["mask_only_comparison"]["passes"] is False
    assert parent.loc[candidate, "quintile_rps"].mean() == pytest.approx(0.98)


def test_joint_selector_requires_complete_true_cdf_evidence_and_exact_inventory() -> None:
    parent, mask = _metric_frames()
    mask.loc[0, "forecast_cdf_valid_for_thresholds"] = False
    with pytest.raises(driver.MaskControlError, match="CDF evidence failed"):
        _select(parent, mask)

    parent, mask = _metric_frames()
    mask = mask.drop(index=0)
    with pytest.raises(driver.MaskControlError, match="validation keys"):
        _select(parent, mask)

    parent, mask = _metric_frames()
    parent.loc[parent.arm.eq(persistence.PERSISTENCE_LAG_ARM), "split"] = "development"
    with pytest.raises(driver.MaskControlError, match="validation-only"):
        _select(parent, mask)


def test_smoke_selector_requires_one_of_one_matched_seed_but_never_selects_mask() -> None:
    parent, mask = _metric_frames(seeds=(42,))
    selection = _select(parent, mask, seeds=(42,))
    assert selection["candidate_promoted"] is True
    assert selection["selected_arm"] == persistence.PERSISTENCE_LAG_ARM
    assert selection["rules"][
        "minimum_matched_seed_rps_improvements_vs_mask_only"
    ] == 1
    assert selection["mask_only_selectable"] is False


def test_reproduction_precedes_training_and_no_development_split_is_indexed() -> None:
    source = textwrap.dedent(inspect.getsource(driver.run_experiment))
    assert source.index("reproduce_parent_validation(") < source.index(
        "train_mask_only("
    )
    tree = ast.parse(source)
    split_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "splits"
    }
    assert split_attributes == {"as_dict", "train", "validation"}
    assert '"development_or_later": []' in source


def test_source_and_artifact_contract_includes_both_plans_launcher_and_project_paths() -> None:
    assert set(driver.SOURCE_PATHS) == {
        "src/fuxi_allseason_persistence_mask_control.py",
        "src/fuxi_persistence_mask_control.py",
        "tests/test_allseason_persistence_mask_control.py",
        "src/fuxi_allseason_persistence_augmented.py",
        "src/fuxi_persistence_context.py",
        "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_pbc_baseline.py",
        "src/fuxi_pbc_core.py",
        "src/fuxi_allseason_member_cache.py",
        "src/project_paths.py",
        "slurm/run_allseason_persistence_mask_control.sbatch",
        "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md",
        "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
    }
    assert driver.ADDENDUM_PLAN_SHA256 == (
        "c429ee96bfd9b9fc1ddc8fa88da653eeb4923ea037d516d8d6a42071dd3c3ba0"
    )
    source = inspect.getsource(driver.run_experiment)
    assert '"mask_smoke_parent": mask_smoke_binding' in source
    assert '"base_context_and_weekly_climatology_provenance"' in source


def test_array_equality_is_byte_schema_strict_for_dates_strings_and_nan() -> None:
    dates = np.asarray(["2018-01-04", "NaT"], dtype="datetime64[D]")
    assert driver.array_equal(dates, dates.copy())
    assert not driver.array_equal(dates, dates.astype("datetime64[ns]"))
    strings = np.asarray(["mask", "rain"])
    assert driver.array_equal(strings, strings.copy())
    first = np.asarray([1.0, np.nan], dtype=np.float32)
    assert driver.array_equal(first, first.copy())
    assert not driver.array_equal(first, first.astype(np.float64))
