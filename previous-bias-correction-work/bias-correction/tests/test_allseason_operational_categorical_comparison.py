"""Contract tests for the frozen operational-era categorical comparison."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_operational_categorical_comparison as comparison


def _dates_by_year() -> np.ndarray:
    chunks = []
    for year, count in comparison.EXPECTED_YEAR_COUNTS.items():
        chunks.append(
            np.datetime64(f"{year}-01-01", "D")
            + (3 * np.arange(count)).astype("timedelta64[D]")
        )
    return np.concatenate(chunks).astype("datetime64[D]")


def _bootstrap_case_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    dates = _dates_by_year()
    quintile_rows = []
    semidecile_rows = []
    upper_rows = []
    scores = {
        "raw_fuxi_categorical": 2.0,
        "base_42k": 1.0,
        "debias_plus_plus": 1.4,
        "persistence_plus_plus": 1.2,
        "pbc_combined": 1.1,
    }
    for date in dates:
        label = np.datetime_as_string(date, unit="D")
        for lead in range(1, 7):
            for method, score in scores.items():
                common = {
                    "method": method,
                    "method_label": comparison.METHOD_LABELS[method],
                    "initialization": label,
                    "lead_week": lead,
                }
                quintile_rows.append(
                    {
                        **common,
                        "rps": score,
                        "paper_nominal_cut_q80_positive_rps": 2.0 * score,
                        "paper_q80_retained_scoring_weight": float(100 + lead),
                    }
                )
                semidecile_rows.append({**common, "rps": 0.5 * score})
                upper_rows.append({**common, "brier_score": 0.25 * score})
    return (
        pd.DataFrame(quintile_rows),
        pd.DataFrame(semidecile_rows),
        pd.DataFrame(upper_rows),
        dates,
    )


def test_immutable_parent_hashes_and_probability_levels_are_exact() -> None:
    assert comparison.PBC_MANIFEST_SHA256 == (
        "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6"
    )
    assert comparison.PBC_RECEIPT_SHA256 == (
        "1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2"
    )
    assert comparison.PBC_FIT_SHA256 == (
        "0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36"
    )
    assert comparison.OPERATIONAL_MANIFEST_SHA256 == (
        "7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad"
    )
    assert comparison.OPERATIONAL_RECEIPT_SHA256 == (
        "65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd"
    )
    assert comparison.MEMBER_CACHE_DATA_SHA256 == (
        "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
    )
    assert comparison.MEMBER_CACHE_CHECKSUMS_SHA256 == (
        "c4bb28ea67c650910280a9ec6ed340de01e5d54b82848cd4c592e277a6732603"
    )
    assert comparison.DEFAULT_OPERATIONAL_MANIFEST == (
        comparison.PROJECT_ROOT
        / "resultsv2/fuxi_allseason_operational_era_audit/"
        "full_20260822T190829Z/manifest.json"
    )
    assert comparison.DEFAULT_OPERATIONAL_RECEIPT == (
        comparison.PROJECT_ROOT
        / "resultsv2/fuxi_allseason_operational_era_audit/"
        "full_20260822T190829Z/slurm_gate_receipt.json"
    )
    np.testing.assert_array_equal(
        comparison.LEVELS["quintile"],
        np.asarray(np.arange(0.2, 1.0, 0.2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        comparison.LEVELS["semidecile"],
        np.asarray(np.arange(0.05, 1.0, 0.05), dtype=np.float32),
    )


def test_dynamic_weighted_spatial_mean_uses_case_lead_specific_denominators() -> None:
    values = np.zeros((2, 1, 2, 2), dtype=np.float64)
    values[0, 0] = [[1.0, 10.0], [100.0, np.nan]]
    values[1, 0] = [[4.0, 8.0], [12.0, 16.0]]
    weights = np.zeros_like(values)
    weights[0, 0] = [[1.0, 0.0], [3.0, 2.0]]
    weights[1, 0] = [[0.0, 1.0], [0.0, 1.0]]
    actual = comparison.dynamic_weighted_spatial_mean(values, weights)
    np.testing.assert_allclose(actual[:, 0], [(1.0 + 300.0) / 4.0, 12.0])


def test_fraction_weighted_lags_require_all_days_and_correct_issue_windows() -> None:
    initialization = np.asarray(["2022-01-15"], dtype="datetime64[D]")
    dates = np.arange(
        np.datetime64("2022-01-01"),
        np.datetime64("2022-01-15"),
        dtype="datetime64[D]",
    )
    values = np.broadcast_to(
        np.arange(1, 15, dtype=np.float32)[:, None, None],
        (14, 27, 27),
    ).copy()
    fractions = np.ones_like(values)
    support = np.ones((27, 27), dtype=bool)
    lags, receipt = comparison.build_fraction_weighted_issue_lags(
        initialization,
        dates,
        values,
        fractions,
        np.ones((27, 27), dtype=np.float32),
        support,
        smoke=True,
    )
    assert lags.available.all()
    np.testing.assert_allclose(lags.values[0, 0], 11.0)
    np.testing.assert_allclose(lags.values[0, 1], 4.0)
    assert lags.window_start[0, 0] == np.datetime64("2022-01-08")
    assert lags.window_end[0, 0] == np.datetime64("2022-01-14")
    assert lags.window_start[0, 1] == np.datetime64("2022-01-01")
    assert lags.window_end[0, 1] == np.datetime64("2022-01-07")
    assert receipt["affected_coverage_block_count"] == 0

    with pytest.raises(comparison.OperationalCategoricalError, match="continuous"):
        comparison.build_fraction_weighted_issue_lags(
            initialization,
            np.delete(dates, 3),
            np.delete(values, 3, axis=0),
            np.delete(fractions, 3, axis=0),
            np.ones((27, 27), dtype=np.float32),
            support,
            smoke=True,
        )


def test_lag_weighting_receipts_the_exact_2023_coverage_anomaly_blocks() -> None:
    initializations = np.asarray(
        ["2023-10-16", "2023-10-19", "2023-10-23", "2023-10-26"],
        dtype="datetime64[D]",
    )
    dates = np.arange(
        np.datetime64("2023-10-02"),
        np.datetime64("2023-10-26"),
        dtype="datetime64[D]",
    )
    values = np.ones((dates.size, 27, 27), dtype=np.float32)
    anomaly = int(np.flatnonzero(dates == np.datetime64("2023-10-12"))[0])
    values[anomaly, 0, 0] = 100.0
    fractions = np.ones_like(values)
    fractions[anomaly, 0, 0] = 0.0
    support = np.ones((27, 27), dtype=bool)
    lags, receipt = comparison.build_fraction_weighted_issue_lags(
        initializations,
        dates,
        values,
        fractions,
        np.ones((27, 27), dtype=np.float32),
        support,
        smoke=False,
    )
    assert receipt["affected_coverage_block_count"] == 4
    actual = tuple(
        (
            row["initialization"],
            row["lag_week"],
            row["day_within_lag"],
            row["date"],
        )
        for row in receipt["affected_coverage_blocks"]
    )
    assert actual == comparison.EXPECTED_FULL_LAG_COVERAGE_BLOCKS
    # The 100 mm/day value has zero observation fraction and cannot contaminate
    # either lag block containing 2023-10-12.
    np.testing.assert_allclose(lags.values[:, :, 0, 0], 1.0)


def test_equality_aware_dynamic_scoring_covers_k4_and_q80() -> None:
    thresholds = np.full((2, 1, 4, 27, 27), np.nan, dtype=np.float32)
    support = np.zeros((27, 27), dtype=bool)
    support[0, 0] = True
    support[0, 1] = True
    thresholds[..., support] = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)[
        None, None, :, None
    ]
    truth = np.zeros((2, 1, 27, 27), dtype=np.float32)
    truth[:, :, 0, 0] = 0.5
    truth[:, :, 0, 1] = 3.0
    observed = comparison.pbc.observation_cdf(truth, thresholds)
    forecast = comparison.pbc.project_cdf_for_thresholds(
        np.where(
            np.isfinite(thresholds),
            np.asarray([0.0, 0.25, 0.75, 0.9], dtype=np.float32)[
                None, None, :, None, None
            ],
            np.nan,
        ),
        thresholds,
        axis=2,
    )
    weights = np.zeros((2, 1, 27, 27), dtype=np.float64)
    weights[0, 0, 0, :2] = [1.0, 3.0]
    weights[1, 0, 0, :2] = [3.0, 1.0]
    score = comparison.score_cdf(
        "quintile", forecast, observed, thresholds, weights
    )
    assert score.rps.shape == (2, 1)
    assert score.paper_q80_rps is not None
    assert score.upper_brier is None
    assert np.all((score.rps >= 0.0) & (score.rps <= 1.0))
    assert comparison.pbc.is_valid_cdf_for_thresholds(forecast, thresholds, axis=2)
    np.testing.assert_array_equal(forecast[:, :, 1], forecast[:, :, 2])
    np.testing.assert_array_equal(forecast[:, :, 0][..., support], 0.0)


def test_paired_bootstrap_preserves_constant_effect_with_unequal_year_sizes() -> None:
    quintile, semidecile, upper, dates = _bootstrap_case_frames()
    plan = comparison.operational.two_stage_year_block_indices(
        dates,
        n_resamples=40,
        block_length=comparison.BOOTSTRAP_BLOCK_LENGTH,
        seed=comparison.BOOTSTRAP_SEED,
    )
    result = comparison.paired_two_stage_bootstrap(
        quintile, semidecile, upper, dates, plan
    )
    assert len(result) == 4 * 7 * len(comparison.COMPARISONS)
    selected = result.loc[
        result.metric.eq("quintile_effective_positive_cut_rps")
        & result.lead_scope.eq("W1-W6")
        & result.method.eq("base_42k")
        & result.baseline.eq("raw_fuxi_categorical")
    ].iloc[0]
    assert selected.score_reduction_fraction == pytest.approx(0.5)
    assert selected.ci_lower_95 == pytest.approx(0.5)
    assert selected.ci_upper_95 == pytest.approx(0.5)
    assert selected.bootstrap_probability_improvement == 1.0
    assert set(len(draw) for draw in plan.draws).issubset({264, 280, 296, 312})


def test_neural_headline_averages_only_per_seed_scores() -> None:
    rows = []
    for seed, score in zip(comparison.SEEDS, (1.0, 2.0, 3.0), strict=True):
        rows.append(
            {
                "family": "quintile",
                "method": "base_42k",
                "method_label": comparison.METHOD_LABELS["base_42k"],
                "seed": seed,
                "score_contract": comparison.SCORING_CONTRACT_VERSION,
                "nominal_reference_contract": comparison.NOMINAL_REFERENCE_CONTRACT,
                "initialization": "2022-01-03",
                "year": 2022,
                "lead_week": 1,
                "verification_start": "2022-01-03",
                "verification_midpoint": "2022-01-06",
                "verification_end": "2022-01-09",
                "season": "DJF",
                "dynamic_scoring_weight": 100.0,
                "rps": score,
                "nominal_climatology_rps": 4.0,
                "training_empirical_climatology_rps": 5.0,
                "rpss_vs_nominal_climatology_case": 1.0 - score / 4.0,
                "rpss_vs_training_empirical_climatology_case": 1.0 - score / 5.0,
                "paper_nominal_cut_q80_positive_rps": 2.0 * score,
                "paper_fixed_nominal_climatology_rps": 10.0,
                "paper_q80_retained_scoring_weight": 80.0,
                "paper_q80_total_supported_scoring_weight": 100.0,
                "paper_q80_retained_scoring_weight_fraction": 0.8,
                "paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case": (
                    1.0 - 2.0 * score / 10.0
                ),
            }
        )
    result = comparison.average_neural_rps_scores(
        pd.DataFrame(rows), quintile=True
    )
    assert len(result) == 1
    assert result.columns.is_unique
    assert result.iloc[0].seed == "mean_of_seed_scores_42_43_44"
    assert result.iloc[0].rps == pytest.approx(2.0)
    assert result.iloc[0].paper_nominal_cut_q80_positive_rps == pytest.approx(4.0)
    assert result.iloc[0].rpss_vs_training_empirical_climatology_case == pytest.approx(0.6)


def test_all_neural_diagnostics_require_exact_seed_identities() -> None:
    upper_rows = []
    bias_rows = []
    for seed in comparison.SEEDS:
        upper_rows.append(
            {
                "event": "upper_5pct_informative_positive_q95",
                "method": "base_42k",
                "method_label": comparison.METHOD_LABELS["base_42k"],
                "seed": seed,
                "initialization": "2022-01-03",
                "year": 2022,
                "lead_week": 1,
                "verification_midpoint": "2022-01-06",
                "season": "DJF",
                "brier_score": 0.1,
                "nominal_climatology_brier_score": 0.2,
                "training_empirical_climatology_brier_score": 0.3,
                "q95_retained_scoring_weight": 80.0,
                "q95_total_supported_scoring_weight": 100.0,
                "q95_retained_scoring_weight_fraction": 0.8,
            }
        )
        bias_rows.append(
            {
                "family": "semidecile",
                "method": "base_42k",
                "method_label": comparison.METHOD_LABELS["base_42k"],
                "seed": seed,
                "lead_week": 1,
                "season": "ALL",
                "nominal_cumulative_probability": 0.95,
                "probability_bias": 0.01,
                "mean_absolute_case_probability_bias": 0.02,
            }
        )
    assert len(comparison.average_neural_upper_scores(pd.DataFrame(upper_rows))) == 1
    assert len(comparison.average_neural_probability_bias(pd.DataFrame(bias_rows))) == 1
    for rows, function in (
        (upper_rows, comparison.average_neural_upper_scores),
        (bias_rows, comparison.average_neural_probability_bias),
    ):
        wrong = pd.DataFrame(rows).assign(seed=[1, 2, 3])
        with pytest.raises(
            comparison.OperationalCategoricalError, match="seed identities"
        ):
            function(wrong)


def test_operational_adjustment_loader_is_hash_seed_member_and_checkpoint_bound(
    tmp_path: Path,
) -> None:
    dates = np.datetime64("2022-01-01", "D") + np.arange(
        comparison.EXPECTED_CASES
    ).astype("timedelta64[D]")
    shape = (comparison.EXPECTED_CASES, 6, 27, 27)
    relative = "models/base_42k/seed_42/audit_adjustments.npz"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    checkpoint = "a" * 64
    np.savez_compressed(
        path,
        initializations=dates,
        delta_log_location=np.zeros(shape, dtype=np.float32),
        log_spread=np.zeros(shape, dtype=np.float32),
        spread_factor=np.ones(shape, dtype=np.float32),
        source_checkpoint_sha256=np.asarray(checkpoint),
        source_member_count=np.int64(50),
        seed=np.int64(42),
    )
    manifest = {
        "artifact_sha256": {relative: comparison.sha256_file(path)},
        "input_checkpoints": [{"seed": 42, "sha256": checkpoint}],
    }
    delta, spread, receipt = comparison.load_operational_adjustment(
        tmp_path, manifest, 42, dates
    )
    assert delta.shape == spread.shape == shape
    assert receipt["source_member_count"] == 50
    assert receipt["source_checkpoint_sha256"] == checkpoint

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["source_member_count"] = np.int64(51)
    np.savez_compressed(path, **payload)
    manifest["artifact_sha256"][relative] = comparison.sha256_file(path)
    with pytest.raises(
        comparison.OperationalCategoricalError, match="identity/shape"
    ):
        comparison.load_operational_adjustment(tmp_path, manifest, 42, dates)


def test_neural_member_cache_access_is_direct_parent_bound_and_year_receipted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "members_2002_2021.npy"
    metadata = tmp_path / "members_2002_2021.metadata.json"
    cache_manifest = tmp_path / "members_2002_2021.manifest.json"
    capacity_manifest = tmp_path / "capacity_manifest.json"
    checksums = tmp_path / "members_2002_2021.sha256"
    data.write_bytes(b"synthetic-parent-gated-cache")
    metadata.write_text("{}\n", encoding="utf-8")
    cache_manifest.write_text("{}\n", encoding="utf-8")
    capacity_manifest.write_text("{}\n", encoding="utf-8")
    checksums.write_text("synthetic checksums\n", encoding="utf-8")
    monkeypatch.setattr(
        comparison,
        "MEMBER_CACHE_CHECKSUMS_SHA256",
        comparison.sha256_file(checksums),
    )
    dates = np.concatenate(
        [
            np.datetime64(f"{year}-01-01", "D")
            + (3 * np.arange(104)).astype("timedelta64[D]")
            for year in comparison.MEMBER_CACHE_YEARS
        ]
    )
    cache = argparse.Namespace(
        members_path=data,
        metadata_path=metadata,
        manifest_path=cache_manifest,
        initializations=dates,
    )
    capacity = argparse.Namespace(
        manifest={
            "cache": {
                "data_file": str(data),
                "data_sha256": comparison.MEMBER_CACHE_DATA_SHA256,
                "metadata_file": str(metadata),
                "metadata_sha256": comparison.sha256_file(metadata),
                "manifest_file": str(cache_manifest),
                "manifest_sha256": comparison.sha256_file(cache_manifest),
            }
        },
        manifest_path=capacity_manifest,
        manifest_sha256="e" * 64,
    )
    receipt = comparison.neural_member_cache_access_receipt(cache, capacity)
    assert receipt["years"] == list(range(2002, 2022))
    assert receipt["initialization_count"] == 2080
    assert receipt["whole_npy_integrity_scanned"] is True
    assert receipt["checksums_file"] == str(checksums)
    assert receipt["hindcast_member_values_scored_in_this_comparison"] is False
    assert receipt["parent_bound"] is True and receipt["read_only"] is True


def test_driver_has_no_fit_training_selection_or_prediction_entrypoint() -> None:
    source = Path(comparison.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "fit_calendar_quantiles",
        "fit_debias",
        "fit_persistence",
        "select_debias_spans",
        "predict_adjustments",
        "load_checkpoint_model",
    }
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert calls.isdisjoint(forbidden_calls)
    options = {
        option
        for action in comparison.build_parser()._actions
        for option in action.option_strings
    }
    assert "--learning-rate" not in options
    assert "--epochs" not in options
    assert "--candidate" not in options
    assert "--blend-weight" not in options


def test_protocol_and_launcher_freeze_smoke_full_receipts_and_node_blacklist() -> None:
    protocol = comparison.PROTOCOL_PATH.read_text(encoding="utf-8")
    launcher = comparison.SLURM_PATH.read_text(encoding="utf-8")
    for digest in (
        comparison.PBC_MANIFEST_SHA256,
        comparison.PBC_RECEIPT_SHA256,
        comparison.PBC_FIT_SHA256,
        comparison.OPERATIONAL_MANIFEST_SHA256,
        comparison.OPERATIONAL_RECEIPT_SHA256,
    ):
        assert digest in protocol
        assert digest in launcher
    assert "issue-14" in protocol and "2023-10-12" in protocol
    assert "10,000" in protocol
    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17" in launcher
    assert "full mode requires an absolute successful-smoke manifest" in launcher
    assert "full_20260822T190829Z" in protocol
    assert "full_20260822T190829Z" in launcher
    assert "sbatch $0 smoke OUTPUT" in launcher
    assert "sbatch $0 full  OUTPUT SUCCESSFUL_SMOKE_MANIFEST" in launcher
    assert "OPERATIONAL_MANIFEST_RAW" not in launcher
    assert "--operational-manifest" not in launcher
    assert "did not exist when" not in protocol
    assert "did not exist when" not in launcher
    assert "independent semantic reconstruction" not in protocol
    assert "score-table semantic reconstruction" in protocol
    assert "whole 1.86 GB NPY" in protocol
    assert "hindcast member" in protocol
    assert 'require(set(snapshot) == set(live)' in launcher
    assert 'inventory.get(relative) == expected' in launcher
    assert 'smoke_manifest.get("source_snapshot_sha256") == source_hashes' in launcher
    assert '--output "${PUBLISH_STAGING}"' in launcher
    assert '--publication-output "${OUTPUT}"' in launcher
    assert 'mv -T -- "${PUBLISH_STAGING}" "${OUTPUT}"' in launcher
    assert launcher.rindex('mv -T -- "${PUBLISH_STAGING}" "${OUTPUT}"') > launcher.rindex(
        'destination = root / "slurm_gate_receipt.json"'
    )
    assert "path.name not in" not in launcher
    assert 'get("receipt_sha256") == expected_pbc_receipt' in launcher
    assert 'get("fit_sha256") == expected_pbc_fit' in launcher
    assert 'require(actual_files == set(inventory), "complete smoke artifact inventory")' in launcher
    assert "validate_output_semantics" in launcher
    assert "operational_categorical_postrun_v1" in launcher
    assert "slurm_gate_receipt.json" in launcher
    assert "os.replace(temporary, destination)" in launcher
    assert "srun" in launcher
    assert launcher.count("sbatch ") == 2  # the two usage examples only


def test_source_snapshot_binds_all_new_and_reused_runtime_files(tmp_path: Path) -> None:
    snapshot = comparison.source_snapshot(tmp_path)
    expected = {
        "code/src/fuxi_allseason_operational_categorical_comparison.py",
        "code/src/fuxi_allseason_operational_era_audit.py",
        "code/src/fuxi_allseason_ensemble_calibration.py",
        "code/src/fuxi_pbc_core.py",
        "code/src/fuxi_allseason_capacity_ablation.py",
        "code/src/fuxi_allseason_capacity_development_evaluation.py",
        "code/src/fuxi_allseason_member_cache.py",
        "code/src/fuxi_ensemble_calibration_core.py",
        "code/plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md",
        "code/slurm/evaluate_allseason_operational_categorical_comparison.sbatch",
        "code/tests/test_allseason_operational_categorical_comparison.py",
    }
    assert set(snapshot) == expected
    assert snapshot == comparison.output_checksums(tmp_path)
    for relative, digest in snapshot.items():
        assert digest == comparison.sha256_file(tmp_path / relative)


def test_cli_requires_exact_mode_specific_bootstrap_draws() -> None:
    options = {
        option
        for action in comparison.build_parser()._actions
        for option in action.option_strings
    }
    assert "--operational-manifest" not in options
    assert "--operational-manifest-sha256" not in options
    assert "--operational-receipt" not in options
    assert "--operational-receipt-sha256" not in options
    assert "--publication-output" in options
    full = comparison.build_parser().parse_args([])
    comparison.validate_args(full)
    assert full.bootstrap_draws == 10_000
    smoke = comparison.build_parser().parse_args(["--smoke"])
    comparison.validate_args(smoke)
    assert smoke.bootstrap_draws == 200
    invalid = comparison.build_parser().parse_args(
        ["--bootstrap-draws", "9999"]
    )
    with pytest.raises(
        comparison.OperationalCategoricalError, match="bootstrap-draws"
    ):
        comparison.validate_args(invalid)


def test_atomic_main_publishes_fresh_output_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / ".published.gate-incomplete-test"
    publication = tmp_path / "published"
    with pytest.raises(
        comparison.OperationalCategoricalError, match="publication-output"
    ):
        comparison.main(["--output", str(output)])
    observed_publication_paths: list[Path] = []

    def successful(args: argparse.Namespace, staging: Path):
        observed_publication_paths.append(args.publication_output)
        comparison.write_json(staging / "manifest.json", {"status": "complete"})
        return {"status": "complete"}

    monkeypatch.setattr(comparison, "run_experiment", successful)
    args = [
        "--output",
        str(output),
        "--publication-output",
        str(publication),
    ]
    assert comparison.main(args) == 0
    assert observed_publication_paths == [publication.resolve()]
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert not publication.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.main(args)

    failed = tmp_path / ".failed.gate-incomplete-test"
    failed_publication = tmp_path / "failed"

    def broken(args: argparse.Namespace, staging: Path):
        raise RuntimeError("synthetic operational categorical failure")

    monkeypatch.setattr(comparison, "run_experiment", broken)
    with pytest.raises(RuntimeError, match="synthetic operational categorical failure"):
        comparison.main(
            [
                "--output",
                str(failed),
                "--publication-output",
                str(failed_publication),
            ]
        )
    retained = list(tmp_path.glob("..failed.gate-incomplete-test.incomplete-*"))
    assert len(retained) == 1
    assert json.loads((retained[0] / "failure.json").read_text())["status"] == "failed"
