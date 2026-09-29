"""Focused safety and numerical tests for the truth-bearing bridge scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import india_s2s_benchmark_scoring as benchmark
import india_s2s_probabilistic_bridge as bridge
import india_s2s_probabilistic_bridge_run as inference
import india_s2s_probabilistic_bridge_score as score


def _full_inference_header() -> dict[str, object]:
    return {
        "experiment": inference.EXPERIMENT,
        "status": "complete",
        "mode": "full",
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
        "inference_selection": {
            "archive_count": 517,
            "selected_count": 517,
            "scoreable_count": 505,
            "forecast_only_count": 12,
            "selected_dates_sha256": benchmark.FORECAST_INITIALIZATION_SHA256,
            "scoreable_dates_sha256": benchmark.SCORING_INITIALIZATION_SHA256,
            "verification_truth_opened": False,
            "sealed_2025_target_opened": False,
        },
        "storage_contract": {
            "operational_member_count": 50,
            "corrected_members_exactly_reconstructible": True,
            "corrected_members_stored": False,
        },
    }


def _unit_region_weights(shape: tuple[int, int] = (1, 3)) -> dict[str, np.ndarray]:
    return {name: np.ones(shape, dtype=np.float64) for name in benchmark.REGION_ORDER}


def test_inference_header_accepts_only_complete_517_case_run() -> None:
    manifest = _full_inference_header()
    score._validate_inference_header(manifest)

    smoke = json.loads(json.dumps(manifest))
    smoke["status"] = "smoke_complete"
    smoke["mode"] = "bounded_smoke"
    smoke["inference_selection"]["selected_count"] = 5
    with pytest.raises(score.BridgeScoringError, match="status differs"):
        score._validate_inference_header(smoke)

    wrong_members = json.loads(json.dumps(manifest))
    wrong_members["storage_contract"]["operational_member_count"] = 51
    with pytest.raises(score.BridgeScoringError, match="exactly 50"):
        score._validate_inference_header(wrong_members)


def test_scoring_output_guard_requires_resultsv3_bridge_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "resultsv3" / "india_s2s_probabilistic_bridge"
    monkeypatch.setattr(score.bridge, "DEFAULT_OUTPUT_ROOT", root)

    score._assert_bridge_output(root / "scoring_fresh")
    with pytest.raises(score.BridgeScoringError, match="resultsv3"):
        score._assert_bridge_output(tmp_path / "outside")


def test_no_clobber_publication_rejects_raced_destination(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "marker").write_text("first", encoding="utf-8")
    destination = tmp_path / "published"

    score.rename_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "marker").read_text(encoding="utf-8") == "first"
    second = tmp_path / "second"
    second.mkdir()
    (second / "marker").write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError):
        score.rename_noreplace(second, destination)
    assert (destination / "marker").read_text(encoding="utf-8") == "first"
    assert (second / "marker").read_text(encoding="utf-8") == "second"


def test_probabilistic_case_rows_use_50_members_dynamic_regions_and_linear_coverage() -> None:
    inits = np.asarray(["2024-06-03"], dtype="datetime64[D]")
    member_values = np.concatenate((np.zeros(25), np.full(25, 2.0))).astype(np.float32)
    members = np.broadcast_to(
        member_values.reshape(1, 50, 1, 1, 1), (1, 50, 6, 1, 3)
    ).copy()
    truth = np.zeros((1, 6, 1, 3), dtype=np.float32)
    support = np.ones_like(truth)
    support[..., 0] = 0.5

    result = score.probabilistic_case_rows(
        method="raw_fuxi",
        initializations=inits,
        members=members,
        truth=truth,
        weekly_observation_support=support,
        region_base_weights=_unit_region_weights(),
    )

    assert len(result) == 5 * 6
    assert tuple(result.region.drop_duplicates()) == benchmark.REGION_ORDER
    np.testing.assert_allclose(result.coverage90, 1.0)
    np.testing.assert_allclose(result.ensemble_variance, 1.0)
    np.testing.assert_allclose(result.mean_squared_error, 1.0)
    np.testing.assert_allclose(result.ensemble_spread, 1.0)
    np.testing.assert_allclose(result.spread_skill_ratio, 1.0)
    assert result.lead_week.tolist()[:6] == [1, 2, 3, 4, 5, 6]


def test_probabilistic_case_rows_rejects_49_members() -> None:
    inits = np.asarray(["2024-06-03"], dtype="datetime64[D]")
    members = np.zeros((1, 49, 6, 1, 3), dtype=np.float32)
    truth = np.zeros((1, 6, 1, 3), dtype=np.float32)
    with pytest.raises(score.BridgeScoringError, match="case,50"):
        score.probabilistic_case_rows(
            method="raw_fuxi",
            initializations=inits,
            members=members,
            truth=truth,
            weekly_observation_support=np.ones_like(truth),
            region_base_weights=_unit_region_weights(),
        )


def test_independent_member_reconstruction_matches_frozen_formula() -> None:
    random = np.random.default_rng(17)
    raw = random.uniform(0.0, 12.0, size=(1, 50, 6, 27, 27)).astype(np.float32)
    delta = random.normal(0.0, 0.05, size=(1, 6, 27, 27)).astype(np.float32)
    log_spread = random.uniform(-0.2, 0.2, size=(1, 6, 27, 27)).astype(np.float32)

    independent = score.independent_neural_members(raw, delta, log_spread)
    production = inference.reconstruct_corrected_members(raw, delta, log_spread)

    np.testing.assert_array_equal(independent, production)
    fit = score.calibration.MomentFit(
        delta_log_location=np.broadcast_to(delta[0, :, None], (6, 12, 27, 27)).copy(),
        spread_factor=np.ones((6, 12), dtype=np.float32),
        shrinkage=10.0,
    )
    inits = np.asarray(["2024-05-28"], dtype="datetime64[D]")
    np.testing.assert_array_equal(
        score.independent_moment_members(raw, inits, fit),
        score.calibration.apply_moment_fit(raw, inits, fit),
    )


def test_aggregate_is_arithmetic_for_case_scores_and_labels_pooled_ratio() -> None:
    cases = pd.DataFrame(
        {
            "method": ["raw_fuxi", "raw_fuxi"],
            "region": ["all_india", "all_india"],
            "lead_week": [1, 1],
            "init": ["2020-06-01", "2020-06-04"],
            "crps": [1.0, 3.0],
            "acc": [0.1, 0.3],
            "rmse": [2.0, 6.0],
            "mae": [1.5, 4.5],
            "bias": [-1.0, 1.0],
            "coverage90": [0.5, 1.0],
            "spread_skill_ratio": [0.5, 1.5],
            "ensemble_variance": [1.0, 9.0],
            "mean_squared_error": [4.0, 4.0],
        }
    )

    result = score.aggregate_case_metrics(cases).iloc[0]

    assert result.crps == pytest.approx(2.0)
    assert result.rmse == pytest.approx(4.0)
    assert result.coverage90 == pytest.approx(0.75)
    assert result.mean_case_spread_skill_ratio == pytest.approx(1.0)
    assert result.ensemble_variance == pytest.approx(5.0)
    assert result.mean_squared_error == pytest.approx(4.0)
    assert result.spread_skill_ratio == pytest.approx(np.sqrt(5.0 / 4.0))
    assert result.pooled_spread_skill_ratio == pytest.approx(np.sqrt(5.0 / 4.0))
    assert result.case_count == 2


def _interval_cases() -> pd.DataFrame:
    dates = np.asarray(
        [
            "2020-06-01",
            "2020-06-04",
            "2021-06-03",
            "2021-06-07",
            "2022-06-02",
            "2022-06-06",
        ],
        dtype="datetime64[D]",
    )
    rows: list[dict[str, object]] = []
    for init in dates:
        for method, offset, variance in (
            ("raw_fuxi", 0.0, 0.64),
            ("moment_calibration", -0.1, 0.81),
            ("location_spread", -0.2, 1.0),
        ):
            rows.append(
                {
                    "method": method,
                    "region": "all_india",
                    "lead_week": 1,
                    "init": np.datetime_as_string(init, unit="D"),
                    "season": "JJAS",
                    "crps": 2.0 + offset,
                    "acc": 0.2 - offset,
                    "rmse": 3.0 + offset,
                    "mae": 2.5 + offset,
                    "bias": -0.3 + offset,
                    "coverage90": 0.7 - offset,
                    "spread_skill_ratio": float(np.sqrt(variance)),
                    "ensemble_variance": variance,
                    "mean_squared_error": 1.0,
                    "verification_year": int(str(init)[:4]),
                }
            )
    return pd.DataFrame(rows)


def test_paired_intervals_use_both_blocks_and_preserve_constant_effects() -> None:
    intervals, receipt = score.compute_paired_intervals(
        _interval_cases(),
        replicates=40,
        seed=7,
        block_lengths=(2, 3),
        regions=("all_india",),
        lead_weeks=(1,),
        require_169=False,
    )

    assert set(intervals.block_length_starts) == {2, 3}
    assert set(intervals.candidate) == {"moment_calibration", "location_spread"}
    assert set(intervals.effect) == {
        "candidate_minus_baseline",
        "skill_pct_vs_baseline",
    }
    assert len(intervals) == 3 * 2 * (
        len(score.CASE_METRICS) + len(score.LOSS_METRICS)
    )
    selected = intervals[
        intervals.candidate.eq("location_spread")
        & intervals.baseline.eq("raw_fuxi")
        & intervals.metric.eq("crps")
        & intervals.effect.eq("candidate_minus_baseline")
    ]
    np.testing.assert_allclose(selected.estimate, -0.2)
    np.testing.assert_allclose(selected.ci_lower, -0.2)
    np.testing.assert_allclose(selected.ci_upper, -0.2)
    assert set(receipt) == {"w1__block2", "w1__block3"}
    assert receipt["w1__block2"]["shape"] == [40, 6]
    assert len(receipt["w1__block2"]["index_matrix_sha256"]) == 64
    spread = intervals[
        intervals.candidate.eq("location_spread")
        & intervals.baseline.eq("raw_fuxi")
        & intervals.metric.eq("spread_skill_ratio")
    ]
    np.testing.assert_allclose(spread.estimate, 0.2)


def test_paired_intervals_rejects_primary_cohort_shorter_than_169() -> None:
    with pytest.raises(score.BridgeScoringError, match="169 cases"):
        score.compute_paired_intervals(
            _interval_cases(),
            replicates=2,
            regions=("all_india",),
            lead_weeks=(1,),
        )


def test_story_intervals_include_2022_2024_and_neural_vs_moment() -> None:
    dates: list[np.datetime64] = []
    for year, count in ((2020, 35), (2021, 34), (2022, 35), (2023, 35), (2024, 30)):
        dates.extend(
            np.datetime64(f"{year}-06-01")
            + np.arange(count).astype("timedelta64[D]")
        )
    rows: list[dict[str, object]] = []
    for init in dates:
        for method, value, variance in (
            ("raw_fuxi", 2.0, 0.64),
            ("moment_calibration", 1.9, 0.81),
            ("location_spread", 1.8, 1.0),
        ):
            rows.append(
                {
                    "method": method,
                    "region": "all_india",
                    "lead_week": 1,
                    "init": np.datetime_as_string(init, unit="D"),
                    "verification_year": int(str(init)[:4]),
                    "season": "JJAS",
                    "crps": value,
                    "acc": 0.2 + (2.0 - value),
                    "rmse": value + 1.0,
                    "mae": value + 0.5,
                    "bias": value - 2.0,
                    "coverage90": 0.7 + (2.0 - value),
                    "spread_skill_ratio": float(np.sqrt(variance)),
                    "ensemble_variance": variance,
                    "mean_squared_error": 1.0,
                }
            )
    intervals, receipts = score.build_story_intervals(
        pd.DataFrame(rows),
        replicates=4,
        seed=3,
        regions=("all_india",),
        lead_weeks=(1,),
    )

    assert set(intervals.analysis_cohort) == {
        "2020_2024_valid_midpoint_jjas",
        "2022_2024_valid_midpoint_jjas",
    }
    assert set(intervals.n_cases) == {100, 169}
    assert (
        intervals.candidate.eq("location_spread")
        & intervals.baseline.eq("moment_calibration")
    ).any()
    assert set(receipts) == {
        "2020_2024_valid_midpoint_jjas",
        "2022_2024_valid_midpoint_jjas",
    }


def _pooled_valid_midpoint_cases() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_values = {
        "raw_fuxi": {"crps": 2.0, "acc": 0.2, "bias": 0.4},
        "moment_calibration": {"crps": 1.8, "acc": 0.25, "bias": 0.2},
        "location_spread": {"crps": 1.6, "acc": 0.3, "bias": 0.1},
    }
    for lead in benchmark.LEAD_WEEKS:
        offset = 2 * (int(lead) - 1)
        for year, count in ((2022, 35), (2023, 35), (2024, 30)):
            master = np.datetime64(f"{year}-06-01") + np.arange(
                count + 10
            ).astype("timedelta64[D]")
            for init in master[offset : offset + count]:
                for method, values in method_values.items():
                    rows.append(
                        {
                            "method": method,
                            "region": "all_india",
                            "lead_week": int(lead),
                            "init": np.datetime_as_string(init, unit="D"),
                            "verification_year": year,
                            "season": "JJAS",
                            **values,
                        }
                    )
    return pd.DataFrame(rows)


def test_pooled_story_gate_synchronizes_distinct_valid_midpoint_leads() -> None:
    intervals, receipts = score.compute_pooled_w1_w6_story_intervals(
        _pooled_valid_midpoint_cases(),
        replicates=8,
        seed=7,
        block_lengths=(2, 3),
    )

    assert len(intervals) == 16
    assert set(intervals.analysis_cohort) == {
        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
    }
    assert set(intervals.season) == {"JJAS_valid_midpoint"}
    assert set(intervals.lead_week) == {"pooled_w1_w6"}
    assert set(intervals.n_cases_per_lead) == {100}
    assert set(intervals.n_case_lead_rows) == {600}
    assert set(intervals.lead_date_intersection_count) == {70}
    assert set(intervals.lead_date_union_count) == {130}
    assert set(intervals.metric) == {"crps", "acc", "bias"}
    assert set(intervals.baseline) == {"raw_fuxi", "moment_calibration"}
    assert set(intervals.candidate) == {"location_spread"}
    assert set(intervals.block_length_starts) == {2, 3}
    assert set(intervals.loc[intervals.metric.eq("crps"), "effect"]) == {
        "candidate_minus_baseline",
        "skill_pct_vs_baseline",
    }
    assert set(intervals.loc[intervals.metric.eq("acc"), "effect"]) == {
        "candidate_minus_baseline"
    }
    assert set(intervals.loc[intervals.metric.eq("bias"), "effect"]) == {
        "candidate_minus_baseline"
    }
    neural_raw = intervals[
        intervals.baseline.eq("raw_fuxi")
        & intervals.effect.eq("candidate_minus_baseline")
    ]
    np.testing.assert_allclose(
        neural_raw.loc[neural_raw.metric.eq("crps"), "estimate"], -0.4
    )
    np.testing.assert_allclose(
        neural_raw.loc[neural_raw.metric.eq("acc"), "estimate"], 0.1
    )
    np.testing.assert_allclose(
        neural_raw.loc[neural_raw.metric.eq("bias"), "estimate"], -0.3
    )
    skill = intervals[
        intervals.baseline.eq("raw_fuxi")
        & intervals.metric.eq("crps")
        & intervals.effect.eq("skill_pct_vs_baseline")
    ]
    np.testing.assert_allclose(skill.estimate, 20.0)
    assert set(receipts) == {"pooled_w1_w6__block2", "pooled_w1_w6__block3"}
    for receipt in receipts.values():
        assert receipt["shape"] == [8, 100]
        assert receipt["year_counts_per_lead"] == {2022: 35, 2023: 35, 2024: 30}
        assert receipt["n_cases_per_lead"] == 100
        assert receipt["n_case_lead_rows"] == 600
        assert receipt["lead_date_intersection_count"] == 70
        assert receipt["lead_date_union_count"] == 130
        assert len(set(receipt["lead_dates_sha256"].values())) == 6


def test_pooled_story_gate_rejects_incomplete_lead_cohort() -> None:
    cases = _pooled_valid_midpoint_cases()
    cases = cases.drop(
        cases[
            cases.method.eq("location_spread") & cases.lead_week.eq(6)
        ].index[0]
    )
    with pytest.raises(score.BridgeScoringError, match="100 valid-midpoint cases"):
        score.compute_pooled_w1_w6_story_intervals(
            cases,
            replicates=2,
            seed=1,
            block_lengths=(2,),
        )


def test_moment_fit_loader_requires_parent_bound_hash_and_exact_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "parent"
    fit_path = run / score.MOMENT_FIT_RELATIVE
    fit_path.parent.mkdir(parents=True)
    with fit_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            delta_log_location=np.zeros((6, 12, 27, 27), dtype=np.float32),
            spread_factor=np.ones((6, 12), dtype=np.float32),
            shrinkage=np.float32(10.0),
        )
    digest = bridge.sha256_file(fit_path)
    train_values: list[np.datetime64] = []
    for year in range(2002, 2018):
        date = np.datetime64(f"{year}-01-03")
        count = 92 if year == 2017 else 104
        for index in range(count):
            train_values.append(date)
            date += np.timedelta64(3 if index % 2 == 0 else 4, "D")
    train = np.asarray(train_values, dtype="datetime64[D]")
    monkeypatch.setattr(
        score,
        "MOMENT_TRAIN_INITIALIZATIONS_SHA256",
        bridge.initialization_dates_sha256(train),
    )
    parent_manifest = {
        "artifact_sha256": {score.MOMENT_FIT_RELATIVE: digest},
        "moment_calibration": {
            "equation": "u'_m = u_bar + delta_lead_month_grid + s_lead_month*(u_m-u_bar)",
            "fit_scope": "effective training split only",
            "location_shrinkage": 10.0,
            "spread_clip": [0.25, 4.0],
        },
        "split_counts_selected": {"train": 1652},
        "retained_initializations": {
            "train": [np.datetime_as_string(value, unit="D") for value in train],
            "validation": [
                np.datetime_as_string(value, unit="D")
                for value in np.datetime64("2018-01-01")
                + np.arange(196).astype("timedelta64[D]")
            ],
            "test": [
                np.datetime_as_string(value, unit="D")
                for value in np.datetime64("2020-01-01")
                + np.arange(208).astype("timedelta64[D]")
            ],
        },
    }
    parent = inference.FrozenParent(
        manifest_path=run / "manifest.json",
        run_root=run,
        manifest=parent_manifest,
        receipt={},
        checkpoint=run / "best.pt",
        support_artifact=run / "support.npz",
        normalization_mean=np.zeros(6, dtype=np.float32),
        normalization_std=np.ones(6, dtype=np.float32),
    )
    validated = score.ValidatedInference(
        manifest_path=tmp_path / "inference.json",
        root=tmp_path,
        manifest={},
        parent=parent,
        context=None,  # type: ignore[arg-type]
        canonical_initializations=np.asarray(["2020-01-02"], dtype="datetime64[D]"),
        scoring_initializations=np.asarray(["2020-01-02"], dtype="datetime64[D]"),
        cohort={},
    )

    fit = score.load_moment_fit(validated)

    assert fit.delta_log_location.shape == (6, 12, 27, 27)
    assert fit.spread_factor.shape == (6, 12)
    assert fit.shrinkage == 10.0
    bad_manifest = json.loads(json.dumps(parent_manifest))
    bad_manifest["artifact_sha256"][score.MOMENT_FIT_RELATIVE] = "0" * 64
    bad_parent = inference.FrozenParent(**{**parent.__dict__, "manifest": bad_manifest})
    bad = score.ValidatedInference(**{**validated.__dict__, "parent": bad_parent})
    with pytest.raises(score.BridgeScoringError, match="SHA-256 differs"):
        score.load_moment_fit(bad)
    leaked_manifest = json.loads(json.dumps(parent_manifest))
    leaked_manifest["moment_calibration"]["fit_scope"] = "all years"
    leaked_parent = inference.FrozenParent(
        **{**parent.__dict__, "manifest": leaked_manifest}
    )
    leaked = score.ValidatedInference(**{**validated.__dict__, "parent": leaked_parent})
    with pytest.raises(score.BridgeScoringError, match="train-only"):
        score.load_moment_fit(leaked)


def test_catalog_record_hash_binds_zmetadata(tmp_path: Path) -> None:
    store = tmp_path / "2024.zarr"
    store.mkdir()
    metadata = store / ".zmetadata"
    metadata.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()

    receipt = score._catalog_record_hash(
        {"store": str(store), "zmetadata_sha256": digest}, label="synthetic"
    )

    assert receipt == {"store": str(store), "zmetadata_sha256": digest}
    with pytest.raises(score.BridgeScoringError, match="SHA-256 differs"):
        score._catalog_record_hash(
            {"store": str(store), "zmetadata_sha256": "0" * 64}, label="synthetic"
        )


def test_unchanged_input_gate_detects_midrun_mutation(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"status":"complete"}\n', encoding="utf-8")
    frozen = {source.resolve(): bridge.sha256_file(source)}

    score._verify_unchanged_inputs(frozen)
    source.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(score.BridgeScoringError, match="changed during run"):
        score._verify_unchanged_inputs(frozen)


def test_cli_requires_both_hash_gated_inputs_and_defaults_to_10000() -> None:
    parser = score.build_parser()
    args = parser.parse_args(
        [
            "--inference-manifest",
            "inference/manifest.json",
            "--preflight-manifest",
            "preflight/manifest.json",
            "--output",
            "resultsv3/india_s2s_probabilistic_bridge/scoring_test",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.replicates == 10_000
    assert args.bootstrap_seed == 42
    assert args.batch_size == 8
    assert args.benchmark_case_metrics == score.DEFAULT_BENCHMARK_CASES
    assert 2025 not in benchmark.FORECAST_YEARS
    assert score.BLOCK_LENGTHS == (16, 13)
