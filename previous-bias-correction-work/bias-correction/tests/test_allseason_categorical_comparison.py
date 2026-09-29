"""Focused tests for the receipt-gated PBC V2 categorical comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_categorical_comparison as comparison
import fuxi_allseason_ensemble_calibration as frozen
from fuxi_pbc_core import observation_cdf


def _sha(path: Path) -> str:
    return frozen.sha256_file(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def test_probability_levels_match_frozen_pbc_bit_for_bit() -> None:
    canonical_quintiles = np.asarray(
        np.arange(0.2, 1.0, 0.2, dtype=np.float64), dtype=np.float32
    )
    canonical_semideciles = np.asarray(
        np.arange(0.05, 1.0, 0.05, dtype=np.float64), dtype=np.float32
    )
    np.testing.assert_array_equal(comparison.QUINTILE_LEVELS, canonical_quintiles)
    np.testing.assert_array_equal(
        comparison.SEMIDECILE_LEVELS, canonical_semideciles
    )
    np.testing.assert_array_equal(
        comparison.SEMIDECILE_LEVELS.view(np.uint32),
        np.asarray(
            [
                1028443341,
                1036831949,
                1041865114,
                1045220557,
                1048576000,
                1050253722,
                1051931443,
                1053609165,
                1055286886,
                1056964608,
                1057803469,
                1058642330,
                1059481190,
                1060320051,
                1061158912,
                1061997773,
                1062836634,
                1063675494,
                1064514355,
            ],
            dtype=np.uint32,
        ),
    )
    assert not np.array_equal(
        comparison.SEMIDECILE_LEVELS,
        np.arange(0.05, 1.0, 0.05, dtype=np.float32),
    )


def test_adjustment_loader_requires_hash_dates_seed_and_log_spread_identity(
    tmp_path: Path,
) -> None:
    dates = np.asarray(["2020-01-02", "2020-01-09"], dtype="datetime64[D]")
    shape = (2, 6, 27, 27)
    relative = "models/summary_only/seed_42/test_adjustments.npz"
    path = tmp_path / relative
    log_spread = np.full(shape, np.float32(0.2))
    _write_npz(
        path,
        initializations=dates,
        delta_log_location=np.zeros(shape, dtype=np.float32),
        log_spread=log_spread,
        spread_factor=np.exp(log_spread).astype(np.float32),
        seed=np.int64(42),
    )
    manifest = {"artifact_sha256": {relative: _sha(path)}}
    delta, spread, receipt = comparison.load_adjustment(
        tmp_path, manifest, "summary_only", 42, dates
    )
    assert delta.shape == shape
    np.testing.assert_allclose(spread, np.exp(np.float32(0.2)))
    assert receipt["sha256"] == _sha(path)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["spread_factor"] = np.ones(shape, dtype=np.float32)
    _write_npz(path, **payload)
    manifest["artifact_sha256"][relative] = _sha(path)
    with pytest.raises(comparison.ComparisonContractError, match="spread/log-spread"):
        comparison.load_adjustment(tmp_path, manifest, "summary_only", 42, dates)


def test_moment_reconstruction_uses_stored_fit_without_training(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    members = rng.gamma(1.2, 2.0, size=(2, 5, 6, 27, 27)).astype(np.float32)
    dates = np.asarray(["2020-01-02", "2020-07-02"], dtype="datetime64[D]")
    relative = "models/moment_calibration_fit.npz"
    path = tmp_path / relative
    _write_npz(
        path,
        delta_log_location=np.zeros((6, 12, 27, 27), dtype=np.float32),
        spread_factor=np.ones((6, 12), dtype=np.float32),
        shrinkage=np.float32(10.0),
    )
    manifest = {
        "artifact_sha256": {relative: _sha(path)},
        "moment_calibration": {"location_shrinkage": 10.0},
    }
    corrected, receipt = comparison.reconstruct_moment(
        tmp_path, manifest, members, dates
    )
    np.testing.assert_allclose(corrected, members, rtol=1.0e-6, atol=1.0e-6)
    assert receipt["sha256"] == _sha(path)


def _thresholds(level_count: int, positive: np.ndarray) -> np.ndarray:
    result = np.full((2, 1, level_count, 1, 2), np.nan, dtype=np.float32)
    result[..., 0] = np.broadcast_to(
        positive.reshape(1, 1, level_count, 1), (2, 1, level_count, 1)
    )
    return result


def test_categorical_scoring_uses_v2_geometry_and_unsupported_mask() -> None:
    truth = np.asarray([[[[1.0, 9.0]]], [[[3.0, 9.0]]]], dtype=np.float32)
    members = np.repeat(truth[:, None], 5, axis=1)
    quintile_thresholds = _thresholds(
        4, np.asarray([0.0, 1.5, 1.5, 3.5], dtype=np.float32)
    )
    semidecile_thresholds = _thresholds(
        19,
        np.concatenate(
            (np.zeros(3, dtype=np.float32), np.linspace(0.2, 4.0, 16, dtype=np.float32))
        ),
    )
    quintile_observed = observation_cdf(truth, quintile_thresholds)
    semidecile_observed = observation_cdf(truth, semidecile_thresholds)
    scored = comparison.score_ensemble(
        members,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        chunk_size=1,
    )
    np.testing.assert_allclose(scored.quintile_rps, 0.0)
    np.testing.assert_allclose(scored.paper_nominal_cut_rps, 0.0)
    np.testing.assert_allclose(scored.upper_brier, 0.0)
    np.testing.assert_allclose(scored.quintile_probability_bias, 0.0)
    np.testing.assert_allclose(scored.semidecile_probability_bias, 0.0)


def test_reference_scores_project_nominal_and_keep_literal_paper_denominator() -> None:
    truth = np.asarray([[[[1.2]]]], dtype=np.float32)
    quintile_thresholds = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32).reshape(
        1, 1, 4, 1, 1
    )
    semidecile_thresholds = np.linspace(0.0, 2.0, 19, dtype=np.float32).reshape(
        1, 1, 19, 1, 1
    )
    quintile_observed = observation_cdf(truth, quintile_thresholds)
    semidecile_observed = observation_cdf(truth, semidecile_thresholds)
    quintile_climatology = np.asarray([0.0, 0.3, 0.3, 0.8], dtype=np.float32).reshape(
        1, 1, 4, 1, 1
    )
    semidecile_climatology = np.linspace(0.0, 0.9, 19, dtype=np.float32).reshape(
        1, 1, 19, 1, 1
    )
    references = comparison.reference_scores(
        SimpleNamespace(levels=comparison.QUINTILE_LEVELS),
        SimpleNamespace(levels=comparison.SEMIDECILE_LEVELS),
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        quintile_climatology,
        semidecile_climatology,
        np.ones((1, 1), dtype=np.float64),
    )
    assert references.projected_nominal_rps.shape == (1, 1)
    assert references.training_empirical_rps.shape == (1, 1)
    # Literal paper reference counts all four nominal slots, including tied slots.
    manual = np.sum(
        (comparison.QUINTILE_LEVELS.astype(np.float64) - quintile_observed.reshape(4))
        ** 2
    )
    assert references.paper_fixed_nominal_rps.item() == pytest.approx(manual)


def _seed_case_frame() -> pd.DataFrame:
    records = []
    dates = ("2020-01-02", "2020-01-09")
    for method_index, method in enumerate(comparison.NEURAL_CONFIGURATIONS):
        for seed_index, seed in enumerate(comparison.SEEDS):
            for date in dates:
                for lead in (1, 2):
                    score = float(method_index + seed_index + lead)
                    paper_score = score + 0.5
                    records.append(
                        {
                            "method": method,
                            "method_label": comparison.METHOD_LABELS[method],
                            "seed": seed,
                            "score_contract": comparison.SCORING_CONTRACT_VERSION,
                            "nominal_reference_contract": comparison.NOMINAL_REFERENCE_CONTRACT,
                            "initialization": date,
                            "lead_week": lead,
                            "verification_start": date,
                            "verification_midpoint": date,
                            "verification_end": date,
                            "season": "DJF",
                            "rps": score,
                            "paper_nominal_cut_q80_positive_rps": paper_score,
                            "paper_fixed_nominal_climatology_rps": 10.0,
                            "paper_q80_retained_scoring_weight": 2.0,
                            "paper_q80_total_supported_scoring_weight": 4.0,
                            "paper_q80_retained_scoring_weight_fraction": 0.5,
                            "nominal_climatology_rps": 10.0,
                            "training_empirical_climatology_rps": 8.0,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_seed_aggregation_averages_scores_only_and_recomputes_both_skills() -> None:
    averaged = comparison.average_seed_case_scores(_seed_case_frame())
    assert len(averaged) == 2 * 2 * 2
    row = averaged.loc[
        (averaged.method == "summary_only") & (averaged.lead_week == 1)
    ].iloc[0]
    assert row.rps == pytest.approx(2.0)
    assert row.paper_nominal_cut_q80_positive_rps == pytest.approx(2.5)
    assert row.rpss_vs_training_empirical_climatology_case == pytest.approx(0.75)
    assert (
        row.paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case
        == pytest.approx(0.75)
    )
    assert set(averaged.seed) == {"mean_of_seed_scores_42_43_44"}


def test_aggregate_uses_retained_q80_weight_for_paper_formula() -> None:
    frame = pd.DataFrame(
        {
            "method": ["summary_only", "summary_only"],
            "method_label": [comparison.METHOD_LABELS["summary_only"]] * 2,
            "score_contract": [comparison.SCORING_CONTRACT_VERSION] * 2,
            "nominal_reference_contract": [comparison.NOMINAL_REFERENCE_CONTRACT] * 2,
            "initialization": ["2020-01-02", "2020-01-09"],
            "rps": [1.0, 3.0],
            "paper_nominal_cut_q80_positive_rps": [1.0, 3.0],
            "paper_fixed_nominal_climatology_rps": [2.0, 4.0],
            "paper_q80_retained_scoring_weight": [1.0, 3.0],
            "nominal_climatology_rps": [2.0, 4.0],
            "training_empirical_climatology_rps": [2.0, 4.0],
        }
    )
    row = comparison.aggregate_quintile_scores(frame, ()).iloc[0]
    assert row.rps == pytest.approx(2.0)
    assert row.paper_nominal_cut_q80_positive_rps == pytest.approx(2.5)
    assert row.paper_fixed_nominal_climatology_rps == pytest.approx(3.5)


def _bootstrap_frame() -> pd.DataFrame:
    records = []
    dates = [
        *pd.date_range("2020-01-02", periods=8, freq="7D").strftime("%Y-%m-%d"),
        *pd.date_range("2021-01-07", periods=8, freq="7D").strftime("%Y-%m-%d"),
    ]
    for method_index, method in enumerate(comparison.COMPARISON_METHODS):
        for case, date in enumerate(dates):
            for lead in range(1, 7):
                records.append(
                    {
                        "method": method,
                        "initialization": date,
                        "lead_week": lead,
                        "rps": 0.1 + 0.01 * method_index + 0.001 * lead,
                        "paper_nominal_cut_q80_positive_rps": (
                            0.4 + 0.02 * method_index + 0.002 * lead
                        ),
                        "paper_q80_retained_scoring_weight": 1.0 + case % 3,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_paired_bootstrap_has_two_stage_pooled_and_both_metrics() -> None:
    result = comparison.paired_block_bootstrap(
        _bootstrap_frame(),
        samples=20,
        block_length=5,
        seed=comparison.BOOTSTRAP_SEED,
    )
    assert len(result) == 15 * 7 * 2
    assert set(result.lead_scope) == {
        "W1-W6",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
    }
    assert set(result.metric) == {
        "primary_normalized_informative_positive_cut_rps",
        "secondary_guan_nominal_cut_q80_positive_rps",
    }
    assert set(result.bootstrap_scheme) == {comparison.BOOTSTRAP_SCHEME}
    assert set(result.bootstrap_seed) == {comparison.BOOTSTRAP_SEED}
    assert set(result.test_year_count) == {2}
    pooled = result.loc[result.lead_scope == "W1-W6"]
    assert set(zip(pooled.method, pooled.baseline)) >= {
        ("summary_only", "pbc_combined"),
        ("location_spread", "raw_fuxi_categorical"),
    }
    np.testing.assert_allclose(
        result.rps_reduction_fraction,
        1.0 - result.method_score / result.baseline_score,
        rtol=0.0,
        atol=1.0e-14,
    )


def _case_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    for lead in range(1, 7):
        raw_rows.append(
            {
                "method": "raw_fuxi_categorical",
                "method_label": comparison.METHOD_LABELS["raw_fuxi_categorical"],
                "score_contract": comparison.SCORING_CONTRACT_VERSION,
                "nominal_reference_contract": comparison.NOMINAL_REFERENCE_CONTRACT,
                "initialization": "2020-01-02",
                "lead_week": lead,
                "rps": 0.5,
                "paper_nominal_cut_q80_positive_rps": 1.0,
                "paper_fixed_nominal_climatology_rps": 2.0,
                "paper_q80_retained_scoring_weight": 3.0,
                "paper_q80_total_supported_scoring_weight": 4.0,
                "paper_q80_retained_scoring_weight_fraction": 0.75,
                "nominal_climatology_rps": 0.8,
                "training_empirical_climatology_rps": 0.7,
            }
        )
    raw = pd.DataFrame(raw_rows)
    pbc = pd.concat(
        [
            raw.assign(method=method, method_label=comparison.METHOD_LABELS[method])
            for method in comparison.PBC_METHODS
        ],
        ignore_index=True,
    )
    return raw, pbc


def test_pbc_raw_identity_covers_primary_paper_references_and_support() -> None:
    raw, pbc = _case_rows()
    receipts = comparison.validate_pbc_case_scores(
        pbc,
        np.asarray(["2020-01-02"], dtype="datetime64[D]"),
        raw,
    )
    assert receipts["maximum_absolute_rps_difference"] == 0.0
    assert (
        receipts["maximum_absolute_paper_q80_retained_scoring_weight_difference"] == 0.0
    )
    pbc.loc[
        pbc.method == "raw_fuxi_categorical",
        "paper_nominal_cut_q80_positive_rps",
    ] += 0.01
    with pytest.raises(comparison.ComparisonContractError, match="raw identity"):
        comparison.validate_pbc_case_scores(
            pbc,
            np.asarray(["2020-01-02"], dtype="datetime64[D]"),
            raw,
        )


def _pbc_receipt_fixture(tmp_path: Path) -> tuple[dict[str, object], str]:
    provenance = {
        "schema_version": "named_c_order_array_bundle_sha256_v1",
        "bundle_name": "synthetic",
        "sha256": "a" * 64,
        "arrays": {"x": {"dtype": "<f4", "shape": [1]}},
    }
    artifacts = {}
    for name in ("persistence_lag", "observation_bundle"):
        path = tmp_path / f"evaluation/{name}_provenance.json"
        comparison._atomic_json(path, provenance)
        artifacts[f"evaluation/{name}_provenance.json"] = _sha(path)
    validity = {
        family: {
            method: {
                key: True
                for key in (
                    "valid_probability_cdf",
                    "valid_effective_threshold_cdf",
                    "all_supported_values_finite",
                    "all_supported_values_bounded_0_1",
                    "all_supported_rows_nondecreasing",
                    "all_identical_threshold_cdfs_exactly_equal",
                    "all_zero_threshold_cdfs_exactly_zero",
                    "all_unsupported_values_nan",
                )
            }
            for method in comparison.PBC_METHODS
        }
        for family in ("quintile", "semidecile")
    }
    manifest: dict[str, object] = {
        "scoring_contract_version": comparison.SCORING_CONTRACT_VERSION,
        "split_counts_archive": {
            "train": 1652,
            "validation": 196,
            "test": 208,
            "embargo": 24,
        },
        "split_counts_selected": {
            "train": 1652,
            "validation": 196,
            "test": 208,
        },
        "temporal_evidence": {
            "persistence_lag_source": "complete daily IMD calendar; exact seven-day means",
            "persistence_usable_training_cases": 1648,
            "persistence_usable_validation_cases": 196,
            "persistence_usable_development_cases": 208,
        },
        "evaluation": {
            "primary_metric": "normalized informative-positive-cut quintile RPSS synthetic",
            "effective_cut_support": {
                "quintile": {"all_case_leads_have_positive_scoring_weight": True}
            },
            "paper_nominal_cut_support": {
                "all_case_leads_have_positive_q80_scoring_weight": True
            },
            "informative_upper_tail_support": {
                "all_case_leads_have_positive_q95_scoring_weight": True
            },
        },
        "fitting": {"cdf_validity": validity},
        "persistence_lag_provenance": provenance,
        "observation_bundle_provenance": provenance,
        "artifact_sha256": artifacts,
    }
    manifest_sha = "b" * 64
    comparison._atomic_json(
        tmp_path / "slurm_gate_receipt.json",
        {
            "experiment": comparison.PBC_EXPERIMENT,
            "gate_status": "passed",
            "mode": "full",
            "post_run_audit_version": comparison.PBC_POSTRUN_AUDIT_VERSION,
            "scoring_contract_version": comparison.SCORING_CONTRACT_VERSION,
            "manifest_sha256": manifest_sha,
            "persistence_lag_sha256": provenance["sha256"],
            "observation_bundle_sha256": provenance["sha256"],
        },
    )
    return manifest, manifest_sha


def test_pbc_v2_receipt_gate_rejects_stale_contract(tmp_path: Path) -> None:
    manifest, manifest_sha = _pbc_receipt_fixture(tmp_path)
    accepted = comparison.validate_pbc_full_receipt(tmp_path, manifest, manifest_sha)
    assert accepted["post_run_audit_version"] == "pbc_postrun_v2"
    manifest["scoring_contract_version"] = "stale_v1"
    with pytest.raises(comparison.ComparisonContractError, match="scoring contract"):
        comparison.validate_pbc_full_receipt(tmp_path, manifest, manifest_sha)


def test_array_bundle_provenance_binds_dtype_shape_name_and_bytes() -> None:
    first = comparison.array_bundle_provenance(
        {"x": np.asarray([1.0, 2.0], dtype=np.float32)}, bundle_name="test"
    )
    second = comparison.array_bundle_provenance(
        {"x": np.asarray([1.0, 3.0], dtype=np.float32)}, bundle_name="test"
    )
    assert first["schema_version"] == "named_c_order_array_bundle_sha256_v1"
    assert first["sha256"] != second["sha256"]


def test_source_snapshot_binds_the_frozen_protocol(tmp_path: Path) -> None:
    snapshot = comparison._source_snapshot(tmp_path)
    relative = "code/plan/CATEGORICAL_COMPARISON_20260822.md"
    copied = tmp_path / relative
    assert copied.is_file()
    assert snapshot[relative] == _sha(copied)


def test_slurm_launcher_is_receipt_gated_to_canonical_inputs() -> None:
    launcher = (
        Path(comparison.PROJECT_ROOT)
        / "slurm/evaluate_allseason_categorical_comparison.sbatch"
    )
    text = launcher.read_text(encoding="utf-8")
    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17" in text
    assert "#SBATCH --gres" not in text
    assert "full_20260822T173656Z/manifest.json" in text
    assert "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6" in text
    assert comparison.FROZEN_NEURAL_MANIFEST_SHA256 in text
    assert "pbc_postrun_v2" in text
    assert "categorical_comparison_postrun_v2" in text
    assert comparison.BOOTSTRAP_SCHEME in text
    assert "code/plan/CATEGORICAL_COMPARISON_20260822.md" in text
    assert "mid-run live source drift" in text
    assert "source_snapshot_bundle_sha256" in text
    assert "def within_two_float64_ulps" in text


def test_postflight_csv_reaggregation_tolerance_is_strictly_two_float64_ulps() -> None:
    stored = np.float64(3_802_304_942.6868734)
    two_steps = np.nextafter(np.nextafter(stored, np.inf), np.inf)
    three_steps = np.nextafter(two_steps, np.inf)
    tolerance = 2.0 * max(abs(np.spacing(stored)), abs(np.spacing(two_steps)))
    assert abs(two_steps - stored) > 1.0e-8
    assert abs(two_steps - stored) <= tolerance
    assert abs(three_steps - stored) > tolerance


def test_cli_keeps_predeclared_thirteen_initialization_blocks() -> None:
    args = comparison.build_parser().parse_args(
        ["--pbc-manifest", "/tmp/manifest.json", "--block-length", "12"]
    )
    with pytest.raises(ValueError, match="predeclared 13"):
        comparison.validate_args(args)


def test_atomic_main_publishes_fresh_output_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pbc_manifest = tmp_path / "pbc" / "manifest.json"
    pbc_manifest.parent.mkdir()
    pbc_manifest.write_text("{}")
    output = tmp_path / "published"

    def fake_run(args: argparse.Namespace, staging: Path):
        comparison._atomic_json(
            staging / "manifest.json",
            {"experiment": comparison.EXPERIMENT, "status": "complete"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(comparison, "run_comparison", fake_run)
    assert (
        comparison.main(["--pbc-manifest", str(pbc_manifest), "--output", str(output)])
        == 0
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.main(["--pbc-manifest", str(pbc_manifest), "--output", str(output)])

    failed = tmp_path / "failed"

    def fail(args: argparse.Namespace, staging: Path):
        raise RuntimeError("synthetic comparison failure")

    monkeypatch.setattr(comparison, "run_comparison", fail)
    with pytest.raises(RuntimeError, match="synthetic comparison failure"):
        comparison.main(["--pbc-manifest", str(pbc_manifest), "--output", str(failed)])
    retained = list(tmp_path.glob(".failed.incomplete-*"))
    assert len(retained) == 1
    assert json.loads((retained[0] / "failure.json").read_text())["status"] == "failed"
