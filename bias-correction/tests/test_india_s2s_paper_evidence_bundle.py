"""Contract tests for the immutable India S2S paper-table compiler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import india_s2s_paper_evidence_bundle as bundle


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, receipt_name: str) -> dict[str, str]:
    return {
        str(path.relative_to(root)): bundle.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != receipt_name
    }


def _benchmark_cases() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    initializations = pd.date_range("2020-01-01", periods=bundle.CASE_COUNT_PER_LEAD)
    for model_index, model in enumerate(bundle.MODELS):
        for lead in bundle.LEADS:
            available = (model, lead) != ("ecmwf", 3)
            for case_index, initialization in enumerate(initializations):
                perturbation = 0.0001 * case_index
                rows.append(
                    {
                        "track": "tp_imd",
                        "variable": "tp",
                        "reference": "imd",
                        "model": model,
                        "init": initialization.strftime("%Y-%m-%d"),
                        "valid_period_midpoint": "2020-07-15T12:00:00",
                        "season": "JJAS",
                        "region": "all_india",
                        "lead_week": lead,
                        "score_status": (
                            "available"
                            if available
                            else "unavailable_documented_archive_artifact"
                        ),
                        "acc": 0.65 - 0.05 * lead - 0.01 * model_index + perturbation
                        if available
                        else np.nan,
                        "rmse": 3.5 + 0.25 * lead + 0.05 * model_index + perturbation
                        if available
                        else np.nan,
                        "mae": 2.2 + 0.15 * lead + 0.03 * model_index + perturbation
                        if available
                        else np.nan,
                        "bias": -0.2 + 0.02 * lead + 0.01 * model_index + perturbation
                        if available
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _benchmark_intervals(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lead in bundle.LEADS:
        for metric in ("acc", "rmse", "mae", "bias"):
            means = cases[
                cases["lead_week"].eq(lead)
                & cases["model"].isin(("mme", "fuxi_s2s"))
            ].groupby("model")[metric].mean()
            for block in (16, 13):
                rows.append(
                    {
                        "region": "all_india",
                        "lead_week": lead,
                        "metric": metric,
                        "candidate_mean": means["mme"],
                        "baseline_mean": means["fuxi_s2s"],
                        "n_cases": bundle.CASE_COUNT_PER_LEAD,
                        "block_length_starts": block,
                    }
                )
    return pd.DataFrame(rows)


def _score_summary(cases: pd.DataFrame) -> pd.DataFrame:
    raw_means = (
        cases[cases["model"].eq("fuxi_s2s")]
        .groupby("lead_week")[["acc", "rmse", "mae", "bias"]]
        .mean()
    )
    rows: list[dict[str, object]] = []
    effects = {
        "raw_fuxi": (1.0, 0.0),
        "moment_calibration": (0.90, 0.02),
        "location_spread": (0.85, 0.04),
    }
    for method in bundle.METHODS:
        scale, acc_gain = effects[method]
        for region in bundle.REGIONS:
            for lead in bundle.LEADS:
                raw = raw_means.loc[lead]
                rows.append(
                    {
                        "method": method,
                        "region": region,
                        "lead_week": lead,
                        "case_count": bundle.CASE_COUNT_PER_LEAD,
                        "crps_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "crps": (2.0 + 0.2 * lead) * scale,
                        "acc_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "acc": raw["acc"] + acc_gain,
                        "rmse_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "rmse": raw["rmse"] * scale,
                        "mae_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "mae": raw["mae"] * scale,
                        "bias_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "bias": raw["bias"],
                        "coverage90_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "coverage90": 0.7 + 0.1 * (1.0 - scale),
                        "spread_skill_ratio_valid_case_count": bundle.CASE_COUNT_PER_LEAD,
                        "pooled_spread_skill_ratio": 0.7 + 0.2 * (1.0 - scale),
                        "crps_skill_pct_vs_raw": 100.0 * (1.0 - scale),
                        "rmse_skill_pct_vs_raw": 100.0 * (1.0 - scale),
                        "delta_acc_vs_raw": acc_gain,
                    }
                )
    return pd.DataFrame(rows)


def _story_intervals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("raw_fuxi", "crps", "candidate_minus_baseline", -0.51, -0.57, -0.45),
        ("raw_fuxi", "crps", "skill_pct_vs_baseline", 16.2, 14.8, 17.7),
        ("raw_fuxi", "acc", "candidate_minus_baseline", 0.026, 0.009, 0.044),
        ("raw_fuxi", "bias", "candidate_minus_baseline", 0.020, -0.21, 0.25),
        ("moment_calibration", "crps", "candidate_minus_baseline", -0.11, -0.15, -0.07),
        ("moment_calibration", "crps", "skill_pct_vs_baseline", 4.26, 2.99, 5.70),
        ("moment_calibration", "acc", "candidate_minus_baseline", 0.01, -0.01, 0.03),
        ("moment_calibration", "bias", "candidate_minus_baseline", 0.01, -0.10, 0.12),
    )
    for block in (16, 13):
        for baseline, metric, effect, estimate, lower, upper in specifications:
            rows.append(
                {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": bundle.STORY_COHORT,
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "lead_aggregation": "synchronized lead cohorts",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline,
                    "candidate_mean": 1.0,
                    "baseline_mean": 1.1,
                    "n_cases_per_lead": 100,
                    "n_case_lead_rows": 600,
                    "lead_date_intersection_count": 70,
                    "lead_date_union_count": 130,
                    "confidence": 0.95,
                    "block_length_starts": block,
                    "replicates": 10_000,
                    "seed": 42,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _story_gate() -> dict[str, object]:
    return {
        "analysis_cohort": bundle.STORY_COHORT,
        "cases_per_lead": 100,
        "case_lead_rows": 600,
        "primary_block_length_starts": 16,
        "retrospective_2022_2024_component_passes": True,
        "final_headline_gate": "pending_sealed_2025_point-estimate_direction_check",
        "sealed_2025_target_opened": False,
        "crps_skill_pct_vs_raw": {"estimate": 16.2, "ci_lower": 14.8, "ci_upper": 17.7},
        "acc_delta_vs_raw": {"estimate": 0.026, "ci_lower": 0.009, "ci_upper": 0.044},
        "bias_delta_vs_raw": {"estimate": 0.020, "ci_lower": -0.21, "ci_upper": 0.25},
        "crps_skill_pct_vs_moment": {"estimate": 4.26, "ci_lower": 2.99, "ci_upper": 5.70},
    }


def _build_inputs(root: Path, *, opened_2025: bool = False) -> dict[str, object]:
    cases = _benchmark_cases()
    case_path = root / "benchmark_case_metrics.csv"
    cases.to_csv(case_path, index=False)
    case_digest = bundle.sha256_file(case_path)

    benchmark_root = root / "benchmark"
    benchmark_root.mkdir()
    _benchmark_intervals(cases).to_csv(
        benchmark_root / "mme_vs_fuxi_paired_intervals.csv", index=False
    )
    benchmark_manifest: dict[str, object] = {
        "experiment": bundle.BENCHMARK_EXPERIMENT,
        "status": "complete",
        "cohort": "IMD valid-midpoint JJAS",
        "case_count_per_lead_region": bundle.CASE_COUNT_PER_LEAD,
        "block_lengths_starts": [16, 13],
        "replicates": 10_000,
        "source_case_metrics": str(case_path.resolve()),
        "source_case_metrics_sha256": case_digest,
        "source_snapshot_sha256": {},
    }
    benchmark_manifest["artifact_sha256"] = _artifact_hashes(benchmark_root, "manifest.json")
    benchmark_path = benchmark_root / "manifest.json"
    _write_json(benchmark_path, benchmark_manifest)
    benchmark_digest = bundle.sha256_file(benchmark_path)

    scoring_root = root / "scoring"
    (scoring_root / "tables").mkdir(parents=True)
    summary = _score_summary(cases)
    summary.to_csv(scoring_root / "tables/jjas_valid_midpoint_summary.csv", index=False)
    story_intervals = _story_intervals()
    scoring_intervals = story_intervals.copy()
    scoring_intervals.insert(10, "n_cases", np.nan)
    scoring_intervals.to_csv(scoring_root / "tables/paired_block_intervals.csv", index=False)
    scoring_manifest: dict[str, object] = {
        "experiment": bundle.SCORING_EXPERIMENT,
        "status": "complete",
        "benchmark_case_metrics": str(case_path.resolve()),
        "benchmark_case_metrics_sha256": case_digest,
        "cohort": {
            "valid_midpoint_jjas_count_per_lead": bundle.CASE_COUNT_PER_LEAD,
            "maximum_target_label_opened": "2025-01-01" if opened_2025 else "2024-12-30",
        },
        "opened_2025_observation": opened_2025,
        "sealed_2025_target_opened": opened_2025,
        "opened_observation_years": list(range(2020, 2026 if opened_2025 else 2025)),
        "truth_access": {
            "opened_2025": opened_2025,
            "sealed_2025_target_opened": opened_2025,
            "maximum_target_label": "2025-01-01" if opened_2025 else "2024-12-30",
        },
        "raw_fuxi_identity": {
            "identity": True,
            "complete_bridge_contract": True,
            "matched_case_rows": bundle.RAW_IDENTITY_ROWS,
            "benchmark_case_metrics": str(case_path.resolve()),
            "benchmark_case_metrics_sha256": case_digest,
        },
        "table_rows": {"case_metrics": bundle.CASE_ROWS},
        "uncertainty": {
            "block_lengths_starts": [16, 13],
            "replicates": 10_000,
            "cohorts": {"2020_2024_valid_midpoint_jjas": bundle.CASE_COUNT_PER_LEAD},
        },
        "source_snapshot_sha256": {},
    }
    scoring_manifest["artifact_sha256"] = _artifact_hashes(scoring_root, "manifest.json")
    scoring_path = scoring_root / "manifest.json"
    _write_json(scoring_path, scoring_manifest)
    scoring_digest = bundle.sha256_file(scoring_path)

    audit_root = root / "audit"
    (audit_root / "reconstructed").mkdir(parents=True)
    summary.to_csv(audit_root / "reconstructed/jjas_valid_midpoint_summary.csv", index=False)
    story_intervals.to_csv(audit_root / "reconstructed/story_gate_intervals.csv", index=False)
    audit_receipt: dict[str, object] = {
        "experiment": bundle.AUDIT_EXPERIMENT,
        "status": "passed",
        "scoring_manifest": str(scoring_path.resolve()),
        "scoring_manifest_sha256": scoring_digest,
        "input_hashes_reverified_after_semantic_audit": True,
        "semantic_checks": {
            "full_case_cartesian_rows": bundle.CASE_ROWS,
            "raw_identity_rows": bundle.RAW_IDENTITY_ROWS,
            "sealed_2025_target_opened": False,
            "forecast_or_truth_arrays_opened_by_auditor": False,
            "valid_midpoint_jjas_summary_reconstructed": True,
            "paired_intervals_reconstructed_for_both_cohorts": True,
            "synchronized_pooled_story_gate_reconstructed": True,
        },
        "story_gate": _story_gate(),
        "source_snapshot_sha256": {},
    }
    audit_receipt["artifact_sha256"] = _artifact_hashes(audit_root, "audit_receipt.json")
    audit_path = audit_root / "audit_receipt.json"
    _write_json(audit_path, audit_receipt)
    audit_digest = bundle.sha256_file(audit_path)
    return {
        "benchmark": benchmark_path,
        "benchmark_sha256": benchmark_digest,
        "scoring": scoring_path,
        "scoring_sha256": scoring_digest,
        "audit": audit_path,
        "audit_sha256": audit_digest,
    }


def _compile(inputs: dict[str, object], output: Path, root: Path) -> Path:
    return bundle.compile_bundle(
        inputs["benchmark"],
        inputs["benchmark_sha256"],
        inputs["scoring"],
        inputs["scoring_sha256"],
        inputs["audit"],
        inputs["audit_sha256"],
        output,
        output_root=root,
    )


def test_compiles_machine_and_publication_tables_from_hash_pinned_inputs(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    output_root = tmp_path / "bundles"
    manifest_path = _compile(inputs, output_root / "run", output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["table_rows"] == {
        "main_deterministic_benchmark": 48,
        "calibrated_fuxi_by_lead": 18,
        "story_gate": 4,
    }
    assert manifest["hard_gates"]["sealed_2025_target_opened"] is False
    assert manifest["hard_gates"]["forecast_or_truth_arrays_opened_by_compiler"] is False
    assert len(manifest["artifact_sha256"]) == 10
    for relative, digest in manifest["artifact_sha256"].items():
        assert bundle.sha256_file(manifest_path.parent / relative) == digest
    story = pd.read_csv(manifest_path.parent / "tables/story_gate.csv")
    assert story["decision"].tolist() == ["pass", "pass", "no_significant_change", "pass"]


def test_rejects_an_artifact_changed_after_scoring_manifest(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    summary = Path(inputs["scoring"]).parent / "tables/jjas_valid_midpoint_summary.csv"
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(bundle.PaperBundleError, match="SHA-256 differs"):
        _compile(inputs, tmp_path / "bundles/run", tmp_path / "bundles")


def test_rejects_any_receipt_that_reports_opened_2025_truth(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path, opened_2025=True)

    with pytest.raises(bundle.PaperBundleError, match="scoring opened 2025"):
        _compile(inputs, tmp_path / "bundles/run", tmp_path / "bundles")


def test_completed_bundle_is_no_clobber(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    output_root = tmp_path / "bundles"
    output = output_root / "run"
    _compile(inputs, output, output_root)

    with pytest.raises(FileExistsError, match="fresh paper-bundle output"):
        _compile(inputs, output, output_root)


def test_atomic_rename_cannot_replace_a_raced_destination(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    (source / "marker").write_text("new", encoding="utf-8")
    (destination / "marker").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        bundle.rename_noreplace(source, destination)

    assert (source / "marker").read_text(encoding="utf-8") == "new"
    assert (destination / "marker").read_text(encoding="utf-8") == "existing"
