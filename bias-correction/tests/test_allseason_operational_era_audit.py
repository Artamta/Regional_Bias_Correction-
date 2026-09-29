"""Focused scientific-contract tests for the locked operational-era audit."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import fuxi_allseason_operational_era_audit as audit


def _synthetic_initializations() -> np.ndarray:
    return np.concatenate(
        (
            np.arange("2022-01-03", "2022-01-08", dtype="datetime64[D]"),
            np.arange("2023-01-02", "2023-01-08", dtype="datetime64[D]"),
            np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
        )
    )


def _synthetic_case_metrics(initializations: np.ndarray) -> pd.DataFrame:
    rows = []
    for date in initializations:
        init = np.datetime_as_string(date, unit="D")
        year = int(pd.Timestamp(date).year)
        for lead in range(1, 7):
            for method in audit.METHODS:
                calibrated = method == audit.capacity.BASE_CANDIDATE
                rows.append(
                    {
                        "split": "operational_era_retrospective",
                        "method": method,
                        "method_label": method,
                        "seed": "mean_of_seed_scores" if calibrated else "not_applicable",
                        "init": init,
                        "year": year,
                        "season": "DJF",
                        "lead_week": lead,
                        "member_count": 50,
                        "support_cells": 171,
                        "crps": 8.0 if calibrated else 10.0,
                        "rmse": 4.0 if calibrated else 5.0,
                        "mae": 2.0 if calibrated else 2.5,
                        "acc": 0.3 if calibrated else 0.2,
                        "bias": -0.2 if calibrated else -0.1,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_semantic_contract_frames() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]
]:
    initialization_strings = [
        "2022-01-03",
        "2022-05-02",
        "2022-08-29",
        "2022-12-29",
        "2023-01-02",
        "2023-05-01",
        "2023-10-12",
        "2023-12-28",
        "2024-01-01",
        "2024-04-11",
        "2024-08-08",
        "2024-11-18",
    ]
    scores = {
        "crps": 1.0,
        "rmse": 2.0,
        "mae": 1.0,
        "acc": 0.25,
        "bias": -0.1,
        "absolute_bias": 0.1,
        "ensemble_spread": 2.0,
        "ensemble_variance": 4.0,
        "mean_squared_error": 4.0,
        "spread_skill_ratio": 1.0,
        "coverage_50": 0.5,
        "coverage_80": 0.8,
        "coverage_90": 0.9,
    }
    headline_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for initialization in initialization_strings:
        year = int(initialization[:4])
        for lead_week in range(1, 7):
            common = {
                "split": "operational_era_retrospective",
                "init": initialization,
                "year": year,
                "season": "DJF",
                "initialization_season": "DJF",
                "lead_week": lead_week,
                "member_count": 50,
                "support_cells": 171,
                "scoring_weight_sum_km2_fraction": 100.0,
                **scores,
            }
            headline_rows.extend(
                (
                    {
                        **common,
                        "method": "raw_fuxi",
                        "method_label": "Raw FuXi",
                        "seed": "not_applicable",
                    },
                    {
                        **common,
                        "method": audit.capacity.BASE_CANDIDATE,
                        "method_label": "Locked base adapter",
                        "seed": "mean_of_seed_metrics_42_43_44",
                    },
                )
            )
            for seed in audit.SEEDS:
                seed_rows.append(
                    {
                        **common,
                        "method": audit.capacity.BASE_CANDIDATE,
                        "method_label": "Locked base adapter",
                        "seed": seed,
                    }
                )
    evaluated = pd.DataFrame(
        {
            "initialization": initialization_strings,
            "initialization_year": [int(value[:4]) for value in initialization_strings],
            "verification_start": initialization_strings,
            "verification_end": [
                (pd.Timestamp(value) + pd.Timedelta(days=41)).strftime("%Y-%m-%d")
                for value in initialization_strings
            ],
            "retained": True,
        }
    )
    manifest: dict[str, object] = {
        "cases": {"evaluated_initializations": initialization_strings}
    }
    return pd.DataFrame(headline_rows), pd.DataFrame(seed_rows), evaluated, manifest


def test_frozen_protocol_exists_and_forbids_audit_selection() -> None:
    text = audit.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "frozen before launch, 2026-08-22" in text
    assert "may not train, fine-tune, select, blend, or promote" in text
    assert "post-hoc 2022--2024 operational-era" in text
    assert "100-start" in text
    assert "2025" in text
    assert "Prelaunch observation-coverage amendment" in text
    assert "2023-10-12" in text
    assert "Pre-score stored ensemble-mean amendment" in text
    assert "1,294,704" in text
    assert "one-ULP corruption" in text


def test_exact_store_whitelist_has_no_2025_year_component() -> None:
    paths = audit.operational_store_paths()
    assert tuple(path.name for path in paths) == (
        "2022.zarr",
        "2023.zarr",
        "2024.zarr",
    )
    assert all("2025.zarr" not in path.parts for path in paths)
    assert "2020_2025" in str(audit.OPERATIONAL_ROOT)
    with pytest.raises(audit.OperationalAuditError, match="whitelist"):
        audit.exact_year_store_paths(audit.IMD_ROOT, (2025,))


def test_late_2024_guard_is_based_on_full_42_day_window() -> None:
    dates = np.asarray(
        ["2024-11-18", "2024-11-20", "2024-11-21", "2024-12-30"],
        dtype="datetime64[D]",
    )
    mask = audit.eligible_initialization_mask(dates, 2024)
    assert mask.tolist() == [True, True, False, False]
    assert dates[mask][0] + np.timedelta64(41, "D") == np.datetime64("2024-12-29")
    for year in (2022, 2023):
        year_dates = np.asarray([f"{year}-12-28"], dtype="datetime64[D]")
        assert audit.eligible_initialization_mask(year_dates, year).tolist() == [True]


def test_full_case_contract_uses_every_eligible_allseason_start() -> None:
    assert audit.EXPECTED_STORED_COUNTS == {2022: 104, 2023: 104, 2024: 100}
    assert audit.EXPECTED_ELIGIBLE_COUNTS == {2022: 104, 2023: 104, 2024: 88}
    assert sum(audit.EXPECTED_ELIGIBLE_COUNTS.values()) == audit.EXPECTED_FULL_CASES == 296
    assert audit.SMOKE_CASES_PER_YEAR == 4
    assert audit.DAILY_VALIDATION_CHUNK_CASES == 8


def test_all_selected_weekly_means_are_streamed_and_middle_case_is_checked() -> None:
    daily = np.arange(3 * 2 * 14 * 2 * 3, dtype=np.float32).reshape(3, 2, 14, 2, 3)
    weekly = daily.reshape(3, 2, 2, 7, 2, 3).mean(
        axis=3, dtype=np.float64
    ).astype(np.float32)
    dataset = xr.Dataset(
        {
            "forecast": (
                ("init", "member", "lead_day", "latitude", "longitude"),
                daily,
            )
        }
    )
    receipt = audit.validate_all_selected_weekly_means(
        dataset,
        np.arange(3, dtype=np.int64),
        weekly,
        chunk_cases=1,
    )
    assert receipt["sha256"] == audit.array_sha256(daily.astype("<f4"))
    assert receipt["shape"] == [3, 2, 14, 2, 3]
    assert receipt["validation"]["checked_chunks"] == 3
    assert receipt["validation"]["checked_initialization_count"] == 3
    assert receipt["validation"]["checked_case_member_week_blocks"] == 12
    assert receipt["validation"]["checked_weekly_grid_values"] == 72
    assert receipt["validation"]["maximum_weekly_vs_daily_absolute_difference_mm_day"] == 0.0

    corrupted = weekly.copy()
    corrupted[1, 0, 0, 0, 0] += 1.0e-3
    with pytest.raises(audit.OperationalAuditError, match="chunk 1:2"):
        audit.validate_all_selected_weekly_means(
            dataset,
            np.arange(3, dtype=np.int64),
            corrupted,
            chunk_cases=1,
        )


def _heavy_member_mean_fixture() -> tuple[np.ndarray, np.ndarray, float]:
    members = np.full((1, 50, 1, 1, 1), 70.0, dtype=np.float32)
    members[0, -1, 0, 0, 0] = np.float32(71.0)
    unrounded = np.nanmean(members, axis=1, dtype=np.float64)
    stored = unrounded.astype(np.float32)
    old_absolute_difference = float(
        np.max(np.abs(unrounded - stored.astype(np.float64)))
    )
    return members, stored, old_absolute_difference


def test_stored_ensemble_mean_uses_exact_canonical_post_cast_identity() -> None:
    members, stored, old_absolute_difference = _heavy_member_mean_fixture()
    assert old_absolute_difference > 2.0e-6

    receipt = audit.validate_stored_ensemble_mean_product(members, stored)
    assert receipt["stored_ensemble_mean_reduction_contract"] == (
        "np.nanmean(member_axis,dtype=float64).astype(float32)"
    )
    assert receipt["stored_ensemble_mean_exact_post_float32_cast_identity"] is True
    assert receipt["stored_ensemble_mean_bitwise_mismatch_count"] == 0
    assert receipt["checked_member_mean_grid_values"] == 1
    assert receipt["maximum_pre_round_member_mean_absolute_difference_mm_day"] == (
        pytest.approx(old_absolute_difference)
    )
    assert 0.0 < receipt[
        "maximum_pre_round_member_mean_fraction_of_float32_ulp"
    ] <= 0.5


def test_stored_ensemble_mean_rejects_one_float32_ulp_corruption() -> None:
    members, stored, _ = _heavy_member_mean_fixture()
    corrupted = stored.copy()
    corrupted.flat[0] = np.nextafter(
        corrupted.flat[0], np.float32(np.inf)
    )
    with pytest.raises(
        audit.OperationalAuditError,
        match="differs bitwise from stored product at 1 grid values",
    ):
        audit.validate_stored_ensemble_mean_product(members, corrupted)


def test_frozen_normalization_values_and_full_hashes_are_exact() -> None:
    assert audit.EXPECTED_NORMALIZATION_MEAN.tolist() == [
        0.9928932189941406,
        0.9928548336029053,
        0.9927545189857483,
        0.992607057094574,
        0.9922419190406799,
        0.9919546246528625,
    ]
    assert audit.EXPECTED_NORMALIZATION_STD.tolist() == [
        0.8253104090690613,
        0.8253679871559143,
        0.8254558444023132,
        0.8255149126052856,
        0.8256960511207581,
        0.825825035572052,
    ]
    assert (
        audit.array_sha256(audit.EXPECTED_NORMALIZATION_MEAN)
        == audit.EXPECTED_NORMALIZATION_MEAN_SHA256
        == "0fdceee6daa9bf0469967f9ac3e7df5a9ffb0bedfc53930a4dcf8184b89fc911"
    )
    assert (
        audit.array_sha256(audit.EXPECTED_NORMALIZATION_STD)
        == audit.EXPECTED_NORMALIZATION_STD_SHA256
        == "051a34ffe3842bdd6089db1d270b4a8dc1d403afa7b43ea917e4a4c5f9a19926"
    )


def test_fixed_smoke_selection_exercises_real_coverage_date() -> None:
    dates = np.arange("2023-01-01", "2024-01-01", dtype="datetime64[D]")
    selected = audit._select_smoke_indices(
        dates, np.arange(len(dates), dtype=np.int64), 2023, 4
    )
    assert len(selected) == 4
    assert np.unique(selected).size == 4
    assert np.datetime64("2023-10-12") in dates[selected]


def test_daily_coverage_deviation_is_derived_and_exact_gated() -> None:
    dates = np.asarray(["2023-10-12"], dtype="datetime64[D]")
    latitude = np.linspace(39.0, 0.0, 27)
    longitude = np.linspace(60.0, 99.0, 27)
    frozen = np.ones((27, 27), dtype=np.float32)
    daily = frozen[None].copy()
    for y, x, _, _, frozen_value, daily_value in audit.EXPECTED_COVERAGE_DEVIATIONS:
        frozen[y, x] = frozen_value
        daily[0, y, x] = daily_value
    receipt = audit.validate_daily_coverage_deviations(
        dates, daily, frozen, latitude, longitude
    )
    assert receipt["affected_dates"] == ["2023-10-12"]
    assert receipt["daily_deviation_count"] == 5
    assert receipt["zero_daily_coverage_cell_count"] == 2
    assert receipt["exact_identity_gate_passed"] is True

    changed = daily.copy()
    changed[0, 0, 0] = 0.5
    with pytest.raises(audit.OperationalAuditError, match="date/cell identities"):
        audit.validate_daily_coverage_deviations(
            dates, changed, frozen, latitude, longitude
        )


def test_withdrawn_audit_rejects_corrected_one_based_verification_dates() -> None:
    initializations = np.asarray(
        [
            "2022-01-03",
            "2022-05-02",
            "2022-09-01",
            "2022-12-01",
            "2023-01-02",
            "2023-05-01",
            "2023-10-12",
            "2023-12-01",
            "2024-01-01",
            "2024-05-02",
            "2024-09-02",
            "2024-11-18",
        ],
        dtype="datetime64[D]",
    )
    valid_dates = audit.base.derive_valid_dates(initializations)
    frozen = np.ones((27, 27), dtype=np.float32)
    fractions = np.broadcast_to(
        frozen[None, None, None], (12, 6, 7, 27, 27)
    ).copy()
    case = int(np.flatnonzero(initializations == np.datetime64("2023-10-12"))[0])
    for y, x, _, _, frozen_value, daily_value in audit.EXPECTED_COVERAGE_DEVIATIONS:
        frozen[y, x] = frozen_value
        fractions[:, :, :, y, x] = frozen_value
        fractions[case, 0, 0, y, x] = daily_value
    with pytest.raises(
        audit.OperationalAuditError,
        match="unexpected verification date",
    ):
        audit.validate_evaluated_coverage_contract(
            initializations,
            valid_dates,
            fractions,
            frozen,
            np.asarray([1.0, audit.EXPECTED_FULL_MINIMUM_WEIGHT_RATIO]),
            smoke=True,
        )


def test_verification_season_is_lead_specific_not_initialization_only() -> None:
    initializations = np.asarray(["2024-02-27", "2024-08-25"], dtype="datetime64[D]")
    seasons = audit.verification_seasons(initializations)
    assert seasons.shape == (2, 6)
    assert seasons[0, 0] == "MAM"
    assert seasons[1, 0] == "JJA"
    assert seasons[1, 1] == "SON"


def test_array_hash_binds_dtype_shape_and_content() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert audit.array_sha256(values) == audit.array_sha256(values.copy())
    assert audit.array_sha256(values) != audit.array_sha256(values.reshape(2, 6))
    assert audit.array_sha256(values) != audit.array_sha256(values.astype(np.float64))
    changed = values.copy()
    changed[0, 0] = 99.0
    assert audit.array_sha256(values) != audit.array_sha256(changed)


def test_dynamic_weights_ignore_unobserved_nan_and_retain_case_lead_pairing() -> None:
    fields = np.asarray(
        [[[[1.0, 3.0], [np.nan, 9.0]]]], dtype=np.float64
    )
    weights = np.asarray(
        [[[[1.0, 3.0], [0.0, 0.0]]]], dtype=np.float64
    )
    result = audit._dynamic_weighted_field_mean(fields, weights)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(2.5)
    with pytest.raises(audit.OperationalAuditError, match="zero verification weight"):
        audit._dynamic_weighted_field_mean(fields, np.zeros_like(weights))


def test_weekly_observation_aggregation_downweights_a_missing_day() -> None:
    values = np.arange(1.0, 8.0, dtype=np.float32).reshape(1, 1, 7, 1, 1)
    fractions = np.ones_like(values)
    values[0, 0, 3, 0, 0] = np.nan
    fractions[0, 0, 3, 0, 0] = 0.0
    truth, weekly_fraction, weights = audit.aggregate_weekly_observations(
        values, fractions, np.asarray([[10.0]])
    )
    assert truth[0, 0, 0, 0] == pytest.approx((1 + 2 + 3 + 5 + 6 + 7) / 6)
    assert weekly_fraction[0, 0, 0, 0] == pytest.approx(6 / 7)
    assert weights[0, 0, 0, 0] == pytest.approx(60 / 7)


def test_dynamic_ensemble_metrics_use_identical_case_lead_support() -> None:
    members = np.asarray(
        [[[[[1.0, 2.0], [3.0, 4.0]]] * 6, [[[2.0, 3.0], [4.0, 5.0]]] * 6]],
        dtype=np.float32,
    )
    truth = np.full((1, 6, 2, 2), 2.5, dtype=np.float32)
    climatology = np.full_like(truth, 2.0)
    weights = np.ones_like(truth, dtype=np.float64)
    truth[:, :, 1, 1] = np.nan
    weights[:, :, 1, 1] = 0.0
    frame, ranks = audit.evaluate_ensemble_dynamic(
        "raw_fuxi",
        members,
        truth,
        climatology,
        np.asarray(["2022-01-03"], dtype="datetime64[D]"),
        weights,
        chunk_size=1,
    )
    assert len(frame) == 6
    assert set(frame.support_cells) == {3}
    assert set(frame.member_count) == {2}
    assert np.isfinite(frame[["crps", "rmse", "mae", "bias"]].to_numpy()).all()
    assert len(ranks) == 6 * 3


def test_two_stage_bootstrap_is_deterministic_and_uses_circular_year_blocks() -> None:
    initializations = _synthetic_initializations()
    first = audit.two_stage_year_block_indices(
        initializations, n_resamples=200, block_length=3, seed=17
    )
    second = audit.two_stage_year_block_indices(
        initializations, n_resamples=200, block_length=3, seed=17
    )
    assert np.array_equal(first.sampled_years, second.sampled_years)
    assert all(np.array_equal(a, b) for a, b in zip(first.draws, second.draws, strict=True))
    assert first.year_sizes == {2022: 5, 2023: 6, 2024: 4}
    assert any(len(set(row.tolist())) < 3 for row in first.sampled_years)

    years = pd.DatetimeIndex(initializations).year.to_numpy()
    for draw, sampled_years in zip(first.draws[:25], first.sampled_years[:25], strict=True):
        cursor = 0
        for sampled_year in sampled_years:
            source = np.flatnonzero(years == sampled_year)
            segment = draw[cursor : cursor + len(source)]
            cursor += len(source)
            assert np.all(years[segment] == sampled_year)
            lookup = {global_index: local for local, global_index in enumerate(source)}
            local = np.asarray([lookup[int(index)] for index in segment])
            for start in range(0, len(local), 3):
                block = local[start : start + 3]
                if len(block) > 1:
                    assert np.all(np.mod(np.diff(block), len(source)) == 1)
        assert cursor == len(draw)


def test_paired_bootstrap_recovers_constant_effect_with_unequal_year_sizes() -> None:
    initializations = _synthetic_initializations()
    metrics = _synthetic_case_metrics(initializations)
    plan = audit.two_stage_year_block_indices(
        initializations, n_resamples=200, block_length=3, seed=7
    )
    result = audit.paired_two_stage_bootstrap(metrics, initializations, plan)
    pooled = result.loc[result.lead_scope.eq("W1-W6")].set_index("metric")
    assert pooled.loc["crps", "effect"] == pytest.approx(20.0)
    assert pooled.loc["crps", "ci_lower_2p5"] == pytest.approx(20.0)
    assert pooled.loc["rmse", "effect"] == pytest.approx(20.0)
    assert pooled.loc["mae", "effect"] == pytest.approx(20.0)
    assert pooled.loc["acc", "effect"] == pytest.approx(0.1)
    assert pooled.loc["bias", "effect"] == pytest.approx(-0.1)
    diagnostics = audit.bootstrap_diagnostics(plan)
    assert diagnostics["two_stage"] is True
    assert diagnostics["block_length_initializations"] == 3
    assert diagnostics["minimum_initializations_per_draw"] < diagnostics["maximum_initializations_per_draw"]


def test_headline_averages_scores_not_predictions() -> None:
    keys = {
        "split": "test_development",
        "method": audit.capacity.BASE_CANDIDATE,
        "method_label": "base",
        "init": "2022-01-03",
        "year": 2022,
        "season": "DJF",
        "lead_week": 1,
        "member_count": 50,
        "support_cells": 171,
    }
    seed_rows = []
    raw_rows = []
    for lead in range(1, 7):
        for seed, value in zip(audit.SEEDS, (1.0, 2.0, 3.0), strict=True):
            seed_rows.append(
                {
                    **keys,
                    "lead_week": lead,
                    "seed": seed,
                    "crps": value,
                    "rmse": value,
                    "ensemble_variance": value**2,
                    "mean_squared_error": value**2,
                }
            )
        raw_rows.append(
            {
                **keys,
                "lead_week": lead,
                "method": "raw_fuxi",
                "method_label": "raw",
                "seed": "not_applicable",
                "crps": 4.0,
                "rmse": 4.0,
                "ensemble_variance": 16.0,
                "mean_squared_error": 16.0,
            }
        )
    raw = pd.DataFrame(raw_rows)
    result = audit.build_headline_case_metrics(raw, pd.DataFrame(seed_rows))
    adapter = result.loc[result.method.eq(audit.capacity.BASE_CANDIDATE)].iloc[0]
    assert adapter.crps == pytest.approx(2.0)
    assert adapter.seed == "mean_of_seed_metrics_42_43_44"


def test_withdrawn_locked_receipt_rejects_corrected_live_source() -> None:
    with pytest.raises(
        audit.capacity_gate.DevelopmentEvaluationError,
        match="current source differs from frozen capacity run",
    ):
        audit.validate_locked_model_receipt(audit.DEFAULT_CAPACITY_MANIFEST)


def test_driver_has_no_training_or_selection_entrypoint() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(name.startswith("train") for name in function_names)
    assert "select_capacity" not in function_names
    parser = audit.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--learning-rate" not in option_strings
    assert "--max-epochs" not in option_strings
    assert "--candidate" not in option_strings
    assert "truth" not in inspect.signature(audit._context_from_frozen_normalization).parameters


def test_slurm_launcher_is_blacklisted_smoke_gated_and_atomic() -> None:
    text = audit.SLURM_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17" in text
    assert "slurm_gate_receipt.json" in text
    assert "os.replace(temporary, destination)" in text
    assert "full mode requires an absolute successful-smoke manifest" in text
    assert "--preflight-only" in text
    assert "operational_era_postrun_v3" in text
    assert "validate_output_semantics" in text
    assert "complete smoke source snapshot" in text
    assert "code/plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md" in text
    assert "EXPECTED_SMOKE_AFFECTED_CASE_LEADS" in text
    assert "EXPECTED_NORMALIZATION_MEAN_SHA256" in text
    assert "EXPECTED_NORMALIZATION_STD_SHA256" in text
    assert "checked_case_member_week_blocks" in text
    assert "stored_ensemble_mean_bitwise_mismatch_count" in text
    assert "maximum_pre_round_member_mean_fraction_of_float32_ulp" in text
    assert (
        "maximum_member_mean_vs_stored_ensemble_mean_absolute_difference_mm_day"
        not in text
    )
    assert "sbatch " not in text.replace("usage: sbatch", "")


def test_semantic_frame_comparison_rejects_score_tampering() -> None:
    expected = pd.DataFrame(
        {"method": ["raw_fuxi", "base_42k"], "lead_week": [1, 1], "crps": [2.0, 1.0]}
    )
    actual = expected.copy()
    audit._require_frames_equivalent(
        actual,
        expected,
        keys=("method", "lead_week"),
        label="test table",
    )
    actual.loc[1, "crps"] = 1.25
    with pytest.raises(audit.OperationalAuditError, match="numeric column differs"):
        audit._require_frames_equivalent(
            actual,
            expected,
            keys=("method", "lead_week"),
            label="test table",
        )


def test_case_semantic_verifier_formats_series_verification_end_dates() -> None:
    headline, seed_metrics, evaluated, manifest = (
        _synthetic_semantic_contract_frames()
    )
    result = audit._validate_case_metric_semantics(
        headline,
        seed_metrics,
        evaluated,
        manifest,
        smoke=True,
    )
    assert result.dtype == np.dtype("datetime64[D]")
    assert result.shape == (12,)
    assert result[-1] == np.datetime64("2024-11-18")
