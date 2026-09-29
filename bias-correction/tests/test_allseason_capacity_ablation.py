"""Focused contracts for the validation-only adapter capacity ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_capacity_ablation as capacity


CANONICAL_SEEDS = (42, 43, 44)


def _case_metrics(
    values: dict[str, dict[int, float]],
    *,
    seeds: tuple[int, ...] = CANONICAL_SEEDS,
) -> pd.DataFrame:
    rows = []
    for arm in capacity.EXPERIMENT_ARMS:
        for seed in seeds:
            for year in (2018, 2019):
                for lead in (1, 2):
                    rows.append(
                        {
                            "split": "validation",
                            "candidate": arm.name,
                            "seed": seed,
                            "init": f"{year}-01-0{lead}",
                            "year": year,
                            "lead_week": lead,
                            "crps": values[arm.name][year],
                        }
                    )
    return pd.DataFrame(rows)


def _default_values() -> dict[str, dict[int, float]]:
    return {
        "small_20k": {2018: 0.997, 2019: 0.997},
        "base_42k": {2018: 1.000, 2019: 1.000},
        "medium_158k": {2018: 0.990, 2019: 0.990},
        "large_294k": {2018: 0.988, 2019: 0.988},
        # This diagnostic may be excellent but is never selection-eligible.
        "summary_matched_43k": {2018: 0.970, 2019: 0.970},
    }


def test_frozen_width_grid_preserves_base_and_expected_parameter_counts() -> None:
    assert capacity.CANDIDATE_NAMES == (
        "small_20k",
        "base_42k",
        "medium_158k",
        "large_294k",
    )
    assert [
        (arm.member_hidden_channels, arm.backbone_channels)
        for arm in capacity.CANDIDATES
    ] == [(4, 16), (8, 24), (16, 48), (32, 64)]
    assert [arm.expected_parameter_count for arm in capacity.CANDIDATES] == [
        19_618,
        42_434,
        157_570,
        293_762,
    ]
    for arm in capacity.EXPERIMENT_ARMS:
        model = capacity.build_model(arm)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            arm.expected_parameter_count
        )
    assert (
        capacity.CANDIDATE_BY_NAME[capacity.BASE_CANDIDATE].expected_parameter_count
        == 42_434
    )


def test_parameter_matched_summary_control_is_separate_and_nonselectable() -> None:
    control = capacity.SUMMARY_CONTROL
    assert control.name not in capacity.CANDIDATE_NAMES
    assert control.mode == "summary_only"
    assert control.role == "parameter_matched_summary_control"
    assert control.backbone_channels == 26
    assert control.expected_parameter_count == 43_058
    assert abs(control.expected_parameter_count / 42_434 - 1.0) < 0.015
    assert capacity.build_model(control).member_encoder is None


def test_selection_uses_year_guards_and_parsimony_without_selecting_control() -> None:
    result = capacity.select_capacity(_case_metrics(_default_values()))

    # Large is numerically best, but medium is within the frozen 0.25% tie and
    # therefore wins on parsimony. The even better summary control cannot win.
    assert result["selected_candidate"] == "medium_158k"
    assert result["selected_parameter_count"] == 157_570
    assert result["test_metrics_consulted"] is False
    assert result["parameter_matched_summary_control"]["eligible_promotion"] is False
    assert result["parameter_matched_summary_control"]["member_encoder_used"] is False
    medium = next(
        row
        for row in result["validation_candidates"]
        if row["candidate"] == "medium_158k"
    )
    assert medium["pooled_minimum_improvement_guard"] is True
    assert medium["all_years_noninferior_guard"] is True
    assert medium["eligible_promotion"] is True


def test_yearwise_regression_retains_base_despite_pooled_improvement() -> None:
    values = _default_values()
    values["small_20k"] = {2018: 1.002, 2019: 1.002}
    values["medium_158k"] = {2018: 0.980, 2019: 1.005}
    values["large_294k"] = {2018: 1.010, 2019: 1.010}

    result = capacity.select_capacity(_case_metrics(values))

    assert result["selected_candidate"] == capacity.BASE_CANDIDATE
    medium = next(
        row
        for row in result["validation_candidates"]
        if row["candidate"] == "medium_158k"
    )
    assert medium["crps_skill_pct_vs_base"] > 0.5
    assert medium["pooled_minimum_improvement_guard"] is True
    assert medium["all_years_noninferior_guard"] is False
    assert medium["eligible_promotion"] is False


def test_matched_seed_guard_requires_two_of_three_base_improvements() -> None:
    values = _default_values()
    values["small_20k"] = {2018: 1.01, 2019: 1.01}
    values["medium_158k"] = {2018: 1.00, 2019: 1.00}
    values["large_294k"] = {2018: 1.01, 2019: 1.01}
    metrics = _case_metrics(values)
    # One very good seed creates >0.5% pooled skill and both-year skill, but the
    # other two merely tie the base. It must not be a paper promotion.
    metrics.loc[metrics.candidate.eq("medium_158k") & metrics.seed.eq(42), "crps"] = (
        0.95
    )

    result = capacity.select_capacity(metrics)

    assert result["selected_candidate"] == capacity.BASE_CANDIDATE
    medium = next(
        row
        for row in result["validation_candidates"]
        if row["candidate"] == "medium_158k"
    )
    assert medium["crps_skill_pct_vs_base"] > 0.5
    assert medium["all_years_noninferior_guard"] is True
    assert medium["matched_seed_improvement_passes"] == 1
    assert medium["matched_seed_improvement_required"] == 2
    assert medium["matched_seed_guard"] is False
    assert medium["eligible_promotion"] is False


def test_selection_rejects_test_rows_missing_seeds_and_misaligned_cases() -> None:
    complete = _case_metrics(_default_values())
    contaminated = complete.copy()
    contaminated.loc[0, "split"] = "test_development"
    with pytest.raises(ValueError, match="validation rows only"):
        capacity.select_capacity(contaminated)

    missing_seed = complete.loc[complete.seed != 44]
    with pytest.raises(ValueError, match="seeds"):
        capacity.select_capacity(missing_seed)

    misaligned = complete.copy()
    selected = misaligned.candidate.eq("small_20k") & (misaligned.index == 0)
    misaligned.loc[selected, "init"] = "2018-12-31"
    with pytest.raises(ValueError, match="identical validation cases"):
        capacity.select_capacity(misaligned)


def test_validation_case_scoring_is_chunked_and_validation_only() -> None:
    cases = 4
    members = np.zeros((cases, 3, 6, 27, 27), dtype=np.float32)
    members[:, 0] = 1.0
    members[:, 1] = 2.0
    members[:, 2] = 4.0
    truth = np.full((cases, 6, 27, 27), 2.0, dtype=np.float32)
    initializations = np.asarray(
        ["2017-01-01", "2018-01-03", "2019-01-03", "2020-01-03"],
        dtype="datetime64[D]",
    )
    selected = np.asarray([1, 2], dtype=np.int64)
    adjustment = np.zeros((2, 6, 27, 27), dtype=np.float32)
    weights = np.ones((27, 27), dtype=np.float64)

    metrics = capacity.validation_case_crps(
        capacity.CANDIDATE_BY_NAME["small_20k"],
        42,
        members,
        truth,
        initializations,
        selected,
        adjustment,
        adjustment,
        weights,
        chunk_size=1,
    )

    assert len(metrics) == 2 * 6
    assert set(metrics.split) == {"validation"}
    assert set(metrics.year) == {2018, 2019}
    assert set(metrics.seed) == {42}
    assert np.isfinite(metrics.crps).all()
    assert (metrics.crps >= 0.0).all()

    with pytest.raises(capacity.CapacityAblationError, match="non-validation year"):
        capacity.validation_case_crps(
            capacity.CANDIDATE_BY_NAME["small_20k"],
            42,
            members,
            truth,
            initializations,
            np.asarray([2, 3]),
            adjustment,
            adjustment,
            weights,
            chunk_size=2,
        )


def test_summary_frames_keep_control_distinct() -> None:
    metrics = _case_metrics(_default_values())
    selection = capacity.select_capacity(metrics)
    summary, by_year, by_seed = capacity.validation_summary_frames(metrics, selection)

    assert set(summary.candidate) == set(capacity.ARM_NAMES)
    control = summary.loc[summary.candidate == capacity.SUMMARY_CONTROL.name].iloc[0]
    assert control.role == "parameter_matched_summary_control"
    assert not bool(control.eligible_promotion)
    assert len(by_year) == len(capacity.EXPERIMENT_ARMS) * 2
    assert len(by_seed) == len(capacity.EXPERIMENT_ARMS) * len(CANONICAL_SEEDS)


def test_validate_args_freezes_capacity_only_factor_and_member_subsample() -> None:
    parser = capacity.build_parser()
    full = parser.parse_args([])
    capacity.validate_args(full)
    assert full.candidates == ",".join(capacity.CANDIDATE_NAMES)
    assert full.seeds == "42,43,44"
    assert full.member_subsample == 16

    smoke = parser.parse_args(["--smoke"])
    capacity.validate_args(smoke)
    assert smoke.seeds == "42"
    assert smoke.max_epochs == 2
    assert smoke.patience == 1

    subset = parser.parse_args(["--candidates", "base_42k"])
    with pytest.raises(ValueError, match="capacity candidates"):
        capacity.validate_args(subset)
    member_variant = parser.parse_args(["--member-subsample", "32"])
    with pytest.raises(ValueError, match="canonical run settings differ"):
        capacity.validate_args(member_variant)
    with pytest.raises(ValueError, match="seeds"):
        capacity.validate_args(parser.parse_args(["--seeds", "42"]))


def test_main_publishes_fresh_directory_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capacity-smoke"

    def fake_run(args, staging: Path):
        assert args.output == output.resolve()
        capacity.write_json(
            staging / "manifest.json",
            {
                "experiment": capacity.EXPERIMENT,
                "status": "complete",
                "mode": "smoke",
            },
        )
        return {"status": "complete"}

    monkeypatch.setattr(capacity, "run_experiment", fake_run)
    assert capacity.main(["--smoke", "--output", str(output)]) == 0
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".*.incomplete-*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        capacity.main(["--smoke", "--output", str(output)])
