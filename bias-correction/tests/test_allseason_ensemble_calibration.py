"""Focused contracts for the all-season ensemble-calibration driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_ensemble_calibration as experiment


def test_valid_dates_follow_one_based_fuxi_lead_days() -> None:
    initializations = np.asarray(["2020-06-01", "2020-12-29"], dtype="datetime64[D]")
    valid = experiment.derive_valid_dates(initializations)
    assert valid.shape == (2, 6, 7)
    np.testing.assert_array_equal(
        valid[0, 0],
        np.arange(
            np.datetime64("2020-06-02"),
            np.datetime64("2020-06-09"),
            dtype="datetime64[D]",
        ),
    )
    np.testing.assert_array_equal(
        valid[0, 5],
        np.arange(
            np.datetime64("2020-07-07"),
            np.datetime64("2020-07-14"),
            dtype="datetime64[D]",
        ),
    )
    assert valid[1, 0, 0] == np.datetime64("2020-12-30")
    assert valid[1, 0, -1] == np.datetime64("2021-01-05")
    assert valid[1, -1, -1] == np.datetime64("2021-02-09")


def test_corrected_alignment_is_separately_versioned_and_isolated() -> None:
    assert experiment.EXPERIMENT == "fuxi_allseason_ensemble_calibration_v2_aligned"
    assert experiment.CONTRACT_REVISION == "imd_end_labelled_init_plus_1_through_42_v2"
    assert experiment.TARGET_DAY_OFFSETS == tuple(range(1, 43))
    assert experiment.CONFIGURATIONS == ("location_spread",)
    assert experiment.DEFAULT_OUTPUT_ROOT == (
        experiment.HERE / "resultsv3/fuxi_allseason_ensemble_calibration"
    )


def test_deployable_seed_uses_validation_crps_and_lowest_seed_tiebreak() -> None:
    def candidate(seed: int, validation_crps: float) -> experiment.TrainingRun:
        return experiment.TrainingRun(
            configuration="location_spread",
            seed=seed,
            best_epoch=3,
            stopped_epoch=4,
            stopping_reason="early_stopping_patience",
            best_validation_crps=validation_crps,
            elapsed_seconds=1.0,
            parameter_count=42_434,
            checkpoint=f"seed_{seed}/best.pt",
        )

    selected = experiment.select_deployable_run(
        [candidate(44, 0.9), candidate(43, 0.8), candidate(42, 0.8)]
    )

    assert selected.seed == 42
    assert selected.best_validation_crps == pytest.approx(0.8)


def test_corrected_slurm_gate_rejects_v1_and_resultsv2() -> None:
    launcher = (
        experiment.HERE / "slurm/run_fuxi_allseason_ensemble_calibration.sbatch"
    ).read_text(encoding="utf-8")

    assert "fuxi_allseason_ensemble_calibration_v2_aligned" in launcher
    assert "--configs location_spread" in launcher
    assert "target_day_offsets" in launcher
    assert "resultsv3/fuxi_allseason_ensemble_calibration" in launcher
    assert "fuxi_allseason_ensemble_calibration_v1" not in launcher
    assert '"summary_only",' not in launcher
    assert '"location_only",' not in launcher


def test_outcome_window_purge_keeps_right_split_and_purges_crossers() -> None:
    initializations = np.asarray(
        [
            "2017-11-17",  # outcome ends before the validation boundary
            "2017-12-01",  # outcome crosses into 2018: purge
            "2018-01-03",
            "2019-11-17",  # outcome ends before the test boundary
            "2019-12-01",  # outcome crosses into 2020: purge
            "2020-01-03",
            "2021-12-29",  # retained; truth is allowed to extend into 2022
        ],
        dtype="datetime64[D]",
    )
    splits = experiment.make_split_indices(initializations)
    assert splits.train.tolist() == [0]
    assert splits.validation.tolist() == [2, 3]
    assert splits.test.tolist() == [5, 6]
    assert splits.embargo.tolist() == [1, 4]


def test_canonical_json_cache_sidecars_use_strict_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cache.npy"
    shape = (2, 51, 6, 27, 27)
    np.save(path, np.zeros(shape, dtype=np.float32), allow_pickle=False)
    metadata = {
        "initializations": ["2002-01-03", "2020-01-03"],
        "source_init_indices": [0, 1872],
        "latitude": np.arange(39.0, -0.01, -1.5).tolist(),
        "longitude": np.arange(60.0, 99.01, 1.5).tolist(),
        "member_labels": list(range(51)),
        "shape": list(shape),
        "dims": ["init", "member", "lead_week", "lat", "lon"],
        "output_units": "mm day-1",
        "scope": "unit_test",
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    path.with_suffix(".manifest.json").write_text(json.dumps({"records": []}))
    calls: list[tuple[Path, bool]] = []

    def strict_loader(output: Path, *, verify: bool):
        calls.append((Path(output), verify))
        return np.load(output, mmap_mode="r", allow_pickle=False), metadata

    monkeypatch.setattr(
        experiment.canonical_member_cache, "load_member_cache", strict_loader
    )
    loaded = experiment.load_member_cache(path, allow_partial=True)
    assert calls == [(path, True)]
    assert loaded.members.shape == shape
    assert isinstance(loaded.members, np.memmap)
    assert loaded.members.flags.writeable is False
    assert loaded.initializations.tolist() == [
        np.datetime64("2002-01-03"),
        np.datetime64("2020-01-03"),
    ]


def test_legacy_cache_metadata_cannot_enter_canonical_experiment(tmp_path: Path) -> None:
    path = tmp_path / "cache.npy"
    np.save(path, np.zeros((2, 51, 6, 27, 27), dtype=np.float32), allow_pickle=False)
    np.savez_compressed(
        tmp_path / "metadata.npz",
        initializations=np.asarray(["2002-01-03", "2020-01-03"], dtype="datetime64[D]"),
        latitude=np.arange(39.0, -0.01, -1.5),
        longitude=np.arange(60.0, 99.01, 1.5),
        member=np.arange(51),
    )
    with pytest.raises(experiment.DataContractError, match="legacy metadata are rejected"):
        experiment.load_member_cache(path, allow_partial=True)


def test_affine_log_calibration_identity_and_rank_preservation() -> None:
    rng = np.random.default_rng(12)
    members = rng.gamma(1.5, 2.0, size=(3, 9, 6, 27, 27)).astype(np.float32)
    delta = np.zeros((3, 6, 27, 27), dtype=np.float32)
    scale = np.ones((3, 6, 27, 27), dtype=np.float32)
    identity = experiment.apply_affine_log_calibration(members, delta, scale)
    np.testing.assert_allclose(identity, members, rtol=1.0e-6, atol=1.0e-6)
    corrected = experiment.apply_affine_log_calibration(
        members,
        np.full_like(delta, 0.2),
        np.full_like(scale, 1.4),
    )
    raw_order = np.argsort(members, axis=1)
    corrected_in_raw_order = np.take_along_axis(corrected, raw_order, axis=1)
    # Zero censoring can create ties, but a positive spread cannot invert members.
    assert np.all(np.diff(corrected_in_raw_order, axis=1) >= -1.0e-6)
    assert np.isfinite(corrected).all()
    assert np.all(corrected >= 0.0)


def test_numpy_crps_matches_bruteforce_definition() -> None:
    members = np.asarray([0.0, 2.0, 5.0], dtype=np.float32).reshape(1, 3, 1, 1, 1)
    truth = np.asarray([1.0], dtype=np.float32).reshape(1, 1, 1, 1)
    actual = experiment.numpy_ensemble_crps(members, truth).item()
    values = members.reshape(-1).astype(np.float64)
    expected = np.mean(np.abs(values - 1.0)) - 0.5 * np.mean(
        np.abs(values[:, None] - values[None, :])
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_evaluation_groups_members_and_reports_every_metric() -> None:
    rng = np.random.default_rng(5)
    cases = 3
    full_truth = rng.gamma(1.5, 2.0, size=(cases, 6, 27, 27)).astype(np.float32)
    members = np.repeat(full_truth[:, None], 7, axis=1)
    truth = full_truth.copy()
    climatology = np.full_like(full_truth, 1.0)
    weights = np.zeros((27, 27), dtype=np.float64)
    weights[5:8, 7:10] = 1.0
    truth[..., weights == 0.0] = np.nan
    climatology[..., weights == 0.0] = np.nan
    initializations = np.asarray(
        ["2020-01-03", "2020-04-02", "2020-07-02"], dtype="datetime64[D]"
    )
    metrics, ranks = experiment.evaluate_ensemble(
        "raw_fuxi",
        members,
        truth,
        climatology,
        initializations,
        weights,
        chunk_size=2,
    )
    assert len(metrics) == cases * 6
    assert len(ranks) == 6 * 8
    np.testing.assert_allclose(
        ranks.groupby("lead_week")["count"].sum().to_numpy(),
        cases * weights.sum(),
    )
    for column in (
        "crps",
        "rmse",
        "mae",
        "bias",
        "acc",
        "ensemble_spread",
        "brier_1",
        "brier_5",
        "brier_10",
        "brier_20",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "width_50",
        "width_80",
        "width_90",
    ):
        assert column in metrics
    np.testing.assert_allclose(metrics.crps, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(metrics.rmse, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(metrics.coverage_90, 1.0, atol=1.0e-6)
    reliability = experiment.reliability_bins(
        "raw_fuxi", members, truth, weights, bin_count=5, chunk_size=2
    )
    assert len(reliability) == 6 * 4 * 5
    populated = reliability.loc[reliability.area_weight_sum > 0.0]
    np.testing.assert_allclose(
        populated.mean_forecast_probability,
        populated.observed_frequency,
        atol=1.0e-12,
    )
    weekwise, _, _, _ = experiment.summarize_metrics(metrics)
    for column in (
        "split",
        "seed",
        "week",
        "n_initializations",
        "n_valid_cells",
    ):
        assert column in weekwise
    assert set(weekwise.n_initializations) == {cases}
    assert set(weekwise.n_valid_cells) == {int(np.count_nonzero(weights))}
    neural_seed = metrics.copy()
    neural_seed["method"] = "location_spread"
    neural_seed["method_label"] = experiment.METHOD_LABELS["location_spread"]
    neural_seed["seed"] = 42
    seed_weekwise = experiment.summarize_seed_metrics(neural_seed, weekwise)
    assert "crpss_vs_raw" in seed_weekwise
    np.testing.assert_allclose(seed_weekwise.crpss_vs_raw, 0.0)


def test_moment_fit_uses_train_only_and_returns_finite_rank_preserving_fields() -> None:
    rng = np.random.default_rng(9)
    cases = 48
    initializations = np.arange(
        np.datetime64("2002-01-03"),
        np.datetime64("2002-01-03") + cases * np.timedelta64(7, "D"),
        np.timedelta64(7, "D"),
    )
    members = rng.gamma(1.8, 1.5, size=(cases, 5, 6, 27, 27)).astype(np.float32)
    truth = (members.mean(axis=1) * 1.25).astype(np.float32)
    weights = np.zeros((27, 27), dtype=np.float64)
    weights[4:20, 6:19] = 1.0
    truth[..., weights == 0.0] = np.nan
    train = np.arange(36, dtype=np.int64)
    original_tail = truth[36:].copy()
    fit = experiment.fit_moment_calibration(
        members, truth, initializations, train, weights
    )
    np.testing.assert_array_equal(truth[36:], original_tail)
    assert fit.delta_log_location.shape == (6, 12, 27, 27)
    assert fit.spread_factor.shape == (6, 12)
    assert np.isfinite(fit.delta_log_location).all()
    assert np.all((fit.spread_factor >= 0.25) & (fit.spread_factor <= 4.0))
    corrected = experiment.apply_moment_fit(
        members[36:], initializations[36:], fit
    )
    raw_order = np.argsort(members[36:], axis=1)
    corrected_in_raw_order = np.take_along_axis(corrected, raw_order, axis=1)
    assert np.all(np.diff(corrected_in_raw_order, axis=1) >= -1.0e-6)


def test_block_bootstrap_keeps_complete_initialization_vectors() -> None:
    initializations = np.concatenate(
        (
            np.arange(
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-03") + 20 * np.timedelta64(7, "D"),
                np.timedelta64(7, "D"),
            ),
            np.arange(
                np.datetime64("2021-01-03"),
                np.datetime64("2021-01-03") + 20 * np.timedelta64(7, "D"),
                np.timedelta64(7, "D"),
            ),
        )
    )
    draws = experiment._block_bootstrap_indices(
        initializations, n_resamples=25, block_length=7, seed=42
    )
    assert draws.shape == (25, 40)
    assert np.all((draws >= 0) & (draws < 40))


def test_paired_bootstrap_accepts_a_nonraw_ablation_baseline() -> None:
    initializations = np.asarray(
        ["2020-01-03", "2020-01-10", "2021-01-03", "2021-01-10"],
        dtype="datetime64[D]",
    )
    rows = []
    for initialization in initializations:
        init_string = np.datetime_as_string(initialization, unit="D")
        for lead_week in range(1, 7):
            for method, factor, acc, bias in (
                ("summary_only", 1.0, 0.2, -0.1),
                ("location_spread", 0.9, 0.3, -0.3),
            ):
                rows.append(
                    {
                        "init": init_string,
                        "lead_week": lead_week,
                        "method": method,
                        "crps": factor * lead_week,
                        "rmse": factor * (lead_week + 1.0),
                        "mae": factor * (lead_week + 2.0),
                        "acc": acc,
                        "bias": bias,
                    }
                )
    result = experiment.paired_block_bootstrap(
        pd.DataFrame(rows),
        initializations,
        ("location_spread",),
        n_resamples=50,
        block_length=2,
        seed=7,
        baseline="summary_only",
    )
    pooled = result.loc[result.lead_scope == "W1-W6"].set_index("metric")
    assert set(result.baseline) == {"summary_only"}
    assert pooled.loc["crps", "effect_name"] == "crps_skill_pct_vs_summary_only"
    np.testing.assert_allclose(
        pooled.loc[["crps", "rmse", "mae"], "effect"],
        10.0,
    )
    np.testing.assert_allclose(pooled.loc["acc", "effect"], 0.1)
    np.testing.assert_allclose(pooled.loc["bias", "effect"], -0.2)


def test_headline_neural_scores_are_means_of_seed_scores() -> None:
    common = {
        "split": "test_development",
        "method": "location_spread",
        "method_label": "Neural location + spread",
        "init": "2020-01-03",
        "year": 2020,
        "season": "DJF",
        "lead_week": 1,
        "member_count": 51,
        "support_cells": 10,
    }
    seed_metrics = pd.DataFrame(
        [
            {
                **common,
                "seed": 42,
                "crps": 1.0,
                "ensemble_spread": 2.0,
                "ensemble_variance": 4.0,
                "mean_squared_error": 4.0,
                "spread_skill_ratio": 1.0,
            },
            {
                **common,
                "seed": 43,
                "crps": 3.0,
                "ensemble_spread": 4.0,
                "ensemble_variance": 16.0,
                "mean_squared_error": 25.0,
                "spread_skill_ratio": 0.8,
            },
        ]
    )
    headline = experiment.mean_seed_case_metrics(seed_metrics, (42, 43))
    assert len(headline) == 1
    assert headline.iloc[0].crps == pytest.approx(2.0)
    assert headline.iloc[0].seed == "mean_of_seed_metrics_42_43"
    assert headline.iloc[0].ensemble_spread == pytest.approx(np.sqrt(10.0))
    assert headline.iloc[0].spread_skill_ratio == pytest.approx(np.sqrt(10.0 / 14.5))

    with pytest.raises(experiment.DataContractError, match="missing or duplicated"):
        experiment.mean_seed_case_metrics(
            pd.concat((seed_metrics, seed_metrics.iloc[[0]]), ignore_index=True), (42, 43)
        )


def test_seed_variability_uses_population_standard_deviation() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test_development", "test_development"],
            "method": ["location_spread", "location_spread"],
            "method_label": ["Neural location + spread", "Neural location + spread"],
            "seed": [42, 43],
            "lead_week": [1, 1],
            "week": [1, 1],
            "n_initializations": [208, 208],
            "n_valid_cells": [100, 100],
            "crps": [1.0, 3.0],
        }
    )
    summary = experiment.summarize_seed_variability(frame)
    assert summary.iloc[0].seed_count == 2
    assert summary.iloc[0].crps_mean == pytest.approx(2.0)
    assert summary.iloc[0].crps_std == pytest.approx(1.0)
    assert summary.iloc[0].crps_min == pytest.approx(1.0)
    assert summary.iloc[0].crps_max == pytest.approx(3.0)


def test_only_frozen_settings_can_claim_the_canonical_experiment() -> None:
    parser = experiment.build_parser()
    full = parser.parse_args([])
    experiment.validate_args(full)
    smoke = parser.parse_args(["--smoke"])
    experiment.validate_args(smoke)
    assert smoke.seeds == "42"
    assert smoke.max_epochs == 2
    assert smoke.patience == 1
    variant = parser.parse_args(["--configs", "summary_only"])
    with pytest.raises(ValueError, match="invalid configurations"):
        experiment.validate_args(variant)
    wrong_smoke = parser.parse_args(["--smoke", "--seeds", "42,43,44"])
    with pytest.raises(ValueError, match="requires seeds"):
        experiment.validate_args(wrong_smoke)
