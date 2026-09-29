"""Contracts for the frozen raw-identity 2022--2024 matched audit."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate_raw_identity_2022_2024_audit as audit


def _frozen_initializations() -> np.ndarray:
    cases = pd.read_csv(audit.DEFAULT_AUDIT_RUN / "case_metrics.csv")
    return np.sort(cases.init.drop_duplicates().to_numpy(dtype="datetime64[D]"))


def test_method_contract_excludes_unpromoted_global_candidate() -> None:
    assert audit.METHODS == (
        "raw_fuxi",
        "log_bias",
        "legacy_anchored_adapter",
        "raw_identity",
        "raw_identity_raw_mean_preserved",
    )
    assert all("global" not in method for method in audit.METHODS)
    assert audit.AUDIT_YEARS == (2022, 2023, 2024)
    assert 2025 not in audit.AUDIT_YEARS


def test_projection_uses_no_truth_and_preserves_dynamic_weighted_mean() -> None:
    assert (
        "truth"
        not in inspect.signature(audit.project_log_offset_to_reference_mean).parameters
    )
    support = np.ones((2, 3), dtype=bool)
    candidate = np.asarray(
        [
            [
                [[0.0, 1.0, 4.0], [2.0, 3.0, 8.0]],
                [[8.0, 2.0, 0.0], [1.0, 3.0, 5.0]],
            ]
        ],
        dtype=np.float32,
    )
    reference = np.asarray(
        [
            [
                [[2.0, 3.0, 6.0], [4.0, 5.0, 9.0]],
                [[3.0, 1.0, 0.0], [0.5, 2.0, 4.0]],
            ]
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            [
                [[1.0, 1.0, 25.0], [1.0, 1.0, 1.0]],
                [[3.0, 1.0, 2.0], [7.0, 1.0, 4.0]],
            ]
        ],
        dtype=np.float64,
    )
    projected, diagnostics = audit.project_log_offset_to_reference_mean(
        candidate, reference, weights, support
    )
    assert np.all(projected[..., support] >= 0.0)
    assert diagnostics.float32_absolute_mean_closure_mm_day.max() < 2.0e-6
    for lead in range(2):
        target = np.average(reference[0, lead], weights=weights[0, lead])
        actual = np.average(projected[0, lead], weights=weights[0, lead])
        assert actual == pytest.approx(target, abs=2.0e-6)


def test_projection_is_deterministic_and_dynamic_weights_change_the_answer() -> None:
    support = np.ones((2, 2), dtype=bool)
    candidate = np.asarray([[[[0.0, 0.0], [10.0, 10.0]]]], dtype=np.float32)
    reference = np.asarray([[[[10.0, 10.0], [0.0, 0.0]]]], dtype=np.float32)
    dynamic = np.asarray([[[[20.0, 20.0], [1.0, 1.0]]]], dtype=np.float64)
    fixed = np.ones_like(dynamic)
    first, _ = audit.project_log_offset_to_reference_mean(
        candidate, reference, dynamic, support
    )
    second, _ = audit.project_log_offset_to_reference_mean(
        candidate, reference, dynamic, support
    )
    fixed_result, _ = audit.project_log_offset_to_reference_mean(
        candidate, reference, fixed, support
    )
    assert np.array_equal(first, second, equal_nan=True)
    assert not np.allclose(first, fixed_result, equal_nan=True)
    assert np.average(first[0, 0], weights=dynamic[0, 0]) == pytest.approx(
        np.average(reference[0, 0], weights=dynamic[0, 0]), abs=2.0e-6
    )


def test_projection_handles_zero_candidate_and_positive_raw_mean() -> None:
    support = np.ones((2, 2), dtype=bool)
    candidate = np.zeros((1, 1, 2, 2), dtype=np.float32)
    reference = np.full_like(candidate, 4.0)
    weights = np.ones_like(candidate, dtype=np.float64)
    projected, _ = audit.project_log_offset_to_reference_mean(
        candidate, reference, weights, support
    )
    assert np.allclose(projected, 4.0, rtol=0.0, atol=2.0e-6)


def test_fixed_projection_is_invariant_to_truth_and_weekly_coverage() -> None:
    assert "truth" not in inspect.signature(audit.fixed_projection_weights).parameters
    assert (
        "coverage" not in inspect.signature(audit.fixed_projection_weights).parameters
    )
    support = np.ones((9, 19), dtype=bool)
    candidate = np.linspace(0.0, 12.0, 171, dtype=np.float32).reshape(1, 1, 9, 19)
    raw = np.flip(candidate, axis=-1).copy()
    area = np.linspace(1.0, 2.0, 171, dtype=np.float64).reshape(9, 19)
    fixed = audit.fixed_projection_weights(area, support, candidate.shape)

    truth_a = np.zeros_like(candidate)
    coverage_a = np.ones_like(candidate)
    first, _ = audit.project_log_offset_to_reference_mean(
        candidate, raw, fixed, support
    )
    # These verification-only fields change completely, but are intentionally
    # absent from both the fixed-weight builder and projection call.
    truth_b = truth_a + 999.0
    coverage_b = np.zeros_like(coverage_a)
    second, _ = audit.project_log_offset_to_reference_mean(
        candidate, raw, fixed, support
    )
    assert not np.array_equal(truth_a, truth_b)
    assert not np.array_equal(coverage_a, coverage_b)
    assert np.array_equal(first, second, equal_nan=True)


def test_bootstrap_is_deterministic_year_stratified_circular_and_balanced() -> None:
    initializations = _frozen_initializations()
    first, slices = audit.year_stratified_circular_block_indices(
        initializations, draws=1_000, block_length=13, seed=17
    )
    second, second_slices = audit.year_stratified_circular_block_indices(
        initializations, draws=1_000, block_length=13, seed=17
    )
    assert np.array_equal(first, second)
    assert slices == second_slices
    diagnostics = audit.bootstrap_index_diagnostics(initializations, first, slices, 13)
    assert diagnostics["year_stratified"] is True
    assert diagnostics["circular_within_year"] is True
    assert diagnostics["equal_marginal_inclusion_by_design"] is True
    assert diagnostics["no_year_crossing"] is True
    assert diagnostics["all_six_leads_retained_per_start"] is True
    assert diagnostics["mean_multiplicity_across_initializations"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert diagnostics["maximum_absolute_mean_multiplicity_deviation_from_one"] < 0.1
    assert len(diagnostics["mean_multiplicity_per_initialization"]) == 100
    years = initializations.astype("datetime64[Y]").astype(int) + 1970
    saw_wrap = False
    for year, segment_slice in slices.items():
        segment = first[:, segment_slice]
        assert np.all(years[segment] == year)
        positions = np.flatnonzero(years == year)
        lookup = np.full(100, -1, dtype=np.int16)
        lookup[positions] = np.arange(len(positions), dtype=np.int16)
        local = lookup[segment]
        for start in range(0, segment.shape[1], 13):
            block = local[:, start : min(start + 13, segment.shape[1])]
            if block.shape[1] > 1:
                differences = np.diff(block, axis=1)
                assert np.all(np.mod(differences, len(positions)) == 1)
                saw_wrap = saw_wrap or bool(np.any(differences < 0))
    assert saw_wrap is True


def test_bootstrap_rejects_too_few_draws() -> None:
    with pytest.raises(ValueError, match="at least 1,000"):
        audit.year_stratified_circular_block_indices(
            _frozen_initializations(), draws=999
        )


def test_paired_circular_bootstrap_centers_a_constant_synthetic_effect() -> None:
    initializations = _frozen_initializations()
    indices, slices = audit.year_stratified_circular_block_indices(
        initializations, draws=1_000, block_length=13, seed=17
    )
    gains = {
        "raw_fuxi": 0.0,
        "log_bias": 1.0,
        "legacy_anchored_adapter": 1.5,
        "raw_identity": 2.0,
        "raw_identity_raw_mean_preserved": 2.1,
    }
    frame = pd.MultiIndex.from_product(
        [audit.EXPECTED_REGIONS, audit.METHODS, range(100), range(1, 7)],
        names=("region", "method", "case_index", "lead_week"),
    ).to_frame(index=False)
    base = 10.0 + frame.case_index.to_numpy() / 100.0 + frame.lead_week / 10.0
    gain = frame.method.map(gains).to_numpy(dtype=np.float64)
    frame["rmse_mm_day"] = base - gain
    frame["mae_mm_day"] = base / 2.0 - gain / 4.0
    frame["bias_mm_day"] = 1.0 + base / 100.0 - gain / 100.0
    frame["acc"] = 0.1 + gain / 100.0

    effects = audit.paired_block_effects(frame, indices, slices, block_length=13)
    selected = effects.loc[
        effects.scope_type.eq("pooled")
        & effects.candidate.eq("raw_identity")
        & effects.baseline.eq("raw_fuxi")
        & effects.source_metric.eq("rmse_mm_day")
    ].iloc[0]
    assert selected.effect == pytest.approx(2.0, abs=1.0e-12)
    assert selected.ci_lower_2p5 == pytest.approx(2.0, abs=1.0e-12)
    assert selected.ci_upper_97p5 == pytest.approx(2.0, abs=1.0e-12)
    assert selected.bootstrap_probability_improved == 1.0
    assert selected.n_starts == 100
    assert selected.n_leads_per_start == 6
    assert "actual block length 13 initializations" in selected.definition


def test_intensity_metrics_use_dynamic_weights() -> None:
    truth = np.asarray([[[[0.5, 2.0, 7.0, 15.0, 25.0]]]], dtype=np.float32)
    weights = np.asarray([[[[20.0, 1.0, 1.0, 1.0, 2.0]]]], dtype=np.float64)
    predictions = {
        method: truth + np.float32(index) for index, method in enumerate(audit.METHODS)
    }
    table, thresholds, aggregates = audit.intensity_metrics(predictions, truth, weights)
    assert set(table.stratum) == {item[0] for item in audit.INTENSITY_STRATA}
    raw_dry = table.loc[table.method.eq("raw_fuxi") & table.stratum.eq("dry_lt1")].iloc[
        0
    ]
    assert raw_dry.dynamic_area_weight_sum_km2_case_lead == 20.0
    assert thresholds.empty is False
    assert "dry_lt1" in aggregates


def test_full_intensity_strata_require_nonempty_verification_bins() -> None:
    truth = np.asarray([[[[0.5, 2.0, 7.0, 15.0, 25.0]]]], dtype=np.float32)
    weights = np.ones_like(truth, dtype=np.float64)
    predictions = {method: truth.copy() for method in audit.METHODS}
    table, thresholds, aggregates = audit.intensity_metrics(predictions, truth, weights)
    assert len(table) == len(audit.METHODS) * len(audit.INTENSITY_STRATA)
    assert len(thresholds) == len(audit.METHODS) * len(audit.THRESHOLDS_MM_DAY)
    assert set(aggregates) == {item[0] for item in audit.INTENSITY_STRATA}
    assert np.allclose(table.rmse_mm_day, 0.0)


def test_artifact_preflight_accepts_only_the_pinned_runs() -> None:
    provenance = audit.validate_input_artifacts(
        audit.DEFAULT_AUDIT_RUN, audit.DEFAULT_RAW_IDENTITY_RUN
    )
    assert provenance["audit_manifest_sha256"] == audit.EXPECTED_AUDIT_MANIFEST_SHA256
    assert (
        provenance["raw_identity_manifest_sha256"] == audit.EXPECTED_RAW_MANIFEST_SHA256
    )
    assert [item["sha256"] for item in provenance["raw_identity_checkpoints"]] == [
        "05739bb81a26694ccf5946daee9e4d0fc2bcacbdbfc976e5fc5c4c437f19cdd0",
        "90914d67cd807d118b57f295917a66662a59fa94157eb471fd18b988b629e193",
        "c7c346a12dd781e7b043c84887dc4a947f95f888e410a272d334e6f517b37f52",
    ]


def test_hash_guard_rejects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "changed.txt"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(audit.AuditContractError, match="hash changed"):
        audit._require_hash(path, "0" * 64, "fixture")


def test_cli_requires_output_except_for_preflight() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])
    args = audit.parse_args(["--preflight-only"])
    assert args.output is None
    assert args.preflight_only is True
    assert args.noncanonical_smoke is False
    assert args.bootstrap_draws == 10_000
    assert args.bootstrap_block_length == 13
    assert args.bootstrap_seed == 20_260_822


def test_cli_locks_canonical_bootstrap_and_labels_smoke_overrides() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args(["--preflight-only", "--bootstrap-draws", "1000"])
    with pytest.raises(SystemExit):
        audit.parse_args(["--preflight-only", "--bootstrap-block-length", "12"])
    with pytest.raises(SystemExit):
        audit.parse_args(["--preflight-only", "--bootstrap-seed", "7"])

    args = audit.parse_args(
        [
            "--preflight-only",
            "--bootstrap-draws",
            "1000",
            "--noncanonical-smoke",
        ]
    )
    assert args.noncanonical_smoke is True
    assert args.bootstrap_draws == 1_000
