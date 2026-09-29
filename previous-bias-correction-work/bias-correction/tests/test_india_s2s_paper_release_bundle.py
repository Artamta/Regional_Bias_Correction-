"""Synthetic contract tests for the comprehensive paper release compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import india_s2s_paper_release_bundle as release


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, receipt_name: str) -> dict[str, str]:
    return {
        str(path.relative_to(root)): release.core.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != receipt_name
    }


def _finish_package(root: Path, name: str, payload: dict[str, object]) -> tuple[Path, str]:
    payload["artifact_sha256"] = _artifact_hashes(root, name)
    path = root / name
    _write_json(path, payload)
    return path, release.core.sha256_file(path)


def _simple_manifest(path: Path, **values: object) -> tuple[Path, str]:
    _write_json(path, dict(values))
    return path, release.core.sha256_file(path)


def _calibrated_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scales = {"raw_fuxi": 1.0, "moment_calibration": 0.9, "location_spread": 0.8}
    labels = {
        "raw_fuxi": "Raw FuXi-S2S",
        "moment_calibration": "Train-only moment calibration",
        "location_spread": "Neural location-spread adapter",
    }
    for method, scale in scales.items():
        for lead in range(1, 7):
            rows.append(
                {
                    "method": method,
                    "method_label": labels[method],
                    "region": "all_india",
                    "lead_week": lead,
                    "case_count": 169,
                    "crps": (3.0 + 0.1 * lead) * scale,
                    "acc": 0.5 - 0.05 * lead + (1.0 - scale) * 0.1,
                    "rmse": (5.0 + 0.1 * lead) * scale,
                    "mae": (3.0 + 0.1 * lead) * scale,
                    "bias": -0.2 + (1.0 - scale) * 0.1,
                    "coverage90": 0.4 + (1.0 - scale),
                    "pooled_spread_skill_ratio": 0.5 + (1.0 - scale),
                    "crps_skill_pct_vs_raw": 100.0 * (1.0 - scale),
                    "rmse_skill_pct_vs_raw": 100.0 * (1.0 - scale),
                    "delta_acc_vs_raw": (1.0 - scale) * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _deterministic_benchmark() -> pd.DataFrame:
    labels = {
        "cma": "CMA",
        "dlesym_v0": "DLESyM-v0",
        "ecmwf": "ECMWF",
        "fuxi_s2s": "FuXi-S2S",
        "ncep": "NCEP",
        "neuralgcm": "NeuralGCM",
        "ukmo": "UKMO",
        "mme": "Equal-system MME",
    }
    model_offsets = {
        "cma": 0.00,
        "dlesym_v0": 0.01,
        "ecmwf": 0.02,
        "fuxi_s2s": 0.03,
        "ncep": 0.04,
        "neuralgcm": 0.05,
        "ukmo": 0.06,
        "mme": 0.07,
    }
    rows: list[dict[str, object]] = []
    for model, label in labels.items():
        for lead in range(1, 7):
            unavailable = model == "ecmwf" and lead == 3
            offset = model_offsets[model]
            rows.append(
                {
                    "model": model,
                    "model_label": label,
                    "lead_week": lead,
                    "cohort_case_count": 169,
                    "valid_case_count": 0 if unavailable else 169,
                    "availability_note": "unavailable" if unavailable else "available",
                    "acc": float("nan") if unavailable else 0.60 - 0.05 * lead + offset,
                    "rmse": float("nan") if unavailable else 4.0 + 0.2 * lead - offset,
                    "mae": float("nan") if unavailable else 3.0 + 0.1 * lead - offset,
                    "bias": float("nan") if unavailable else -0.3 + 0.02 * lead + offset,
                }
            )
    return pd.DataFrame(rows)


def _benchmark_interval_rows() -> pd.DataFrame:
    deterministic = _deterministic_benchmark().set_index(["model", "lead_week"])
    rows: list[dict[str, object]] = []
    for block in (16, 13):
        for lead in range(1, 7):
            for metric in ("acc", "rmse", "mae", "bias"):
                candidate = float(deterministic.loc[("mme", lead), metric])
                baseline = float(deterministic.loc[("fuxi_s2s", lead), metric])
                estimate = candidate - baseline
                rows.append(
                    {
                        "reference": "imd",
                        "season": "JJAS_valid_midpoint",
                        "region": "all_india",
                        "lead_week": lead,
                        "metric": metric,
                        "candidate": "mme",
                        "baseline": "fuxi_s2s",
                        "effect": "mme_minus_fuxi",
                        "candidate_mean": candidate,
                        "baseline_mean": baseline,
                        "estimate": estimate,
                        "ci_lower": estimate - 0.01,
                        "ci_upper": estimate + 0.01,
                        "confidence": 0.95,
                        "n_cases": 169,
                        "block_length_starts": block,
                        "replicates": 10_000,
                        "seed": 42,
                    }
                )
    return pd.DataFrame(rows)


def _core_tables(root: Path) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True)
    frames = {
        "main_deterministic_benchmark": _deterministic_benchmark(),
        "calibrated_fuxi_by_lead": _calibrated_scores(),
        "story_gate": pd.DataFrame(
            {"gate": [f"gate-{index}" for index in range(4)], "decision": ["pass"] * 4}
        ),
    }
    for name, frame in frames.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
        (tables / f"{name}.tex").write_text(f"% {name}\n", encoding="utf-8")


def _exact_common_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("raw_fuxi", "crps", "candidate_minus_baseline", -0.5, -0.6, -0.4),
        ("raw_fuxi", "crps", "skill_pct_vs_baseline", 15.8, 14.0, 17.0),
        ("raw_fuxi", "acc", "candidate_minus_baseline", 0.03, 0.01, 0.05),
        ("raw_fuxi", "bias", "candidate_minus_baseline", 0.04, -0.2, 0.3),
        ("moment_calibration", "crps", "candidate_minus_baseline", -0.1, -0.15, -0.05),
        ("moment_calibration", "crps", "skill_pct_vs_baseline", 4.0, 2.5, 5.5),
        ("moment_calibration", "acc", "candidate_minus_baseline", 0.03, 0.01, 0.05),
        ("moment_calibration", "bias", "candidate_minus_baseline", -0.4, -0.5, -0.3),
    )
    for block in (16, 13):
        for baseline, metric, effect, estimate, lower, upper in specifications:
            rows.append(
                {
                    "reference": "imd",
                    "analysis_cohort": "exact-common",
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline,
                    "candidate_mean": 1.0,
                    "baseline_mean": 1.1,
                    "n_common_midpoints_per_lead": 85,
                    "n_case_lead_rows": 510,
                    "confidence": 0.95,
                    "block_length_valid_midpoints": block,
                    "replicates": 10_000,
                    "seed": 42,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _mme_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block in (16, 13):
        for lead in range(1, 7):
            acc_lower = -0.01 if lead == 2 else 0.01
            specifications = (
                ("acc", "candidate_minus_baseline", 0.03, acc_lower, 0.05),
                ("rmse", "candidate_minus_baseline", -0.3, -0.5, -0.1),
                ("rmse", "skill_pct_vs_baseline", 5.0, 2.0, 8.0),
                ("mae", "candidate_minus_baseline", -0.2, -0.3, -0.1),
                ("mae", "skill_pct_vs_baseline", 4.0, 1.0, 7.0),
                ("bias", "candidate_minus_baseline", 0.1, -0.1, 0.3),
            )
            for metric, effect, estimate, lower, upper in specifications:
                rows.append(
                    {
                        "reference": "imd",
                        "analysis_cohort": "2020-2024",
                        "region": "all_india",
                        "lead_week": lead,
                        "metric": metric,
                        "candidate": "location_spread",
                        "baseline": "equal_system_mme_no_ecmwf",
                        "candidate_mean": 1.0,
                        "baseline_mean": 1.1,
                        "n_cases": 169,
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


def _imerg_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("raw_fuxi", "crps", "candidate_minus_baseline", -0.38, -0.43, -0.33),
        ("raw_fuxi", "crps", "skill_pct_vs_baseline", 12.2, 10.5, 14.0),
        ("raw_fuxi", "acc", "candidate_minus_baseline", 0.014, -0.006, 0.034),
        ("raw_fuxi", "bias", "candidate_minus_baseline", 0.018, -0.21, 0.25),
        ("moment_calibration", "crps", "candidate_minus_baseline", -0.06, -0.10, -0.02),
        ("moment_calibration", "crps", "skill_pct_vs_baseline", 2.08, 0.67, 3.66),
        ("moment_calibration", "acc", "candidate_minus_baseline", 0.01, -0.01, 0.03),
        ("moment_calibration", "bias", "candidate_minus_baseline", -0.38, -0.46, -0.30),
    )
    for block in (16, 13):
        for baseline, metric, effect, estimate, lower, upper in specifications:
            rows.append(
                {
                    "reference": "imerg",
                    "analysis_cohort": release.STORY_COHORT,
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline,
                    "n_cases_per_lead": 100,
                    "n_case_lead_rows": 600,
                    "confidence": 0.95,
                    "block_length_starts": block,
                    "replicates": 10_000,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _score_interval_rows() -> pd.DataFrame:
    scores = _calibrated_scores().set_index(["method", "lead_week"])
    rows: list[dict[str, object]] = []
    metric_columns = {
        "acc": "acc",
        "rmse": "rmse",
        "bias": "bias",
        "coverage90": "coverage90",
        "spread_skill_ratio": "pooled_spread_skill_ratio",
    }
    for lead in range(1, 7):
        raw = scores.loc[("raw_fuxi", lead)]
        moment = scores.loc[("moment_calibration", lead)]
        for method in ("moment_calibration", "location_spread"):
            candidate = scores.loc[(method, lead)]
            crps_skill = 100.0 * (1.0 - float(candidate["crps"]) / float(raw["crps"]))
            specifications: list[tuple[str, str, float, float, float, float, float]] = [
                (
                    "crps",
                    "skill_pct_vs_baseline",
                    crps_skill,
                    crps_skill - 1.0,
                    crps_skill + 1.0,
                    float(candidate["crps"]),
                    float(raw["crps"]),
                )
            ]
            for metric, column in metric_columns.items():
                estimate = float(candidate[column]) - float(raw[column])
                specifications.append(
                    (
                        metric,
                        "candidate_minus_baseline",
                        estimate,
                        estimate - 0.01,
                        estimate + 0.01,
                        float(candidate[column]),
                        float(raw[column]),
                    )
                )
            for metric, effect, estimate, lower, upper, candidate_mean, baseline_mean in specifications:
                rows.append(
                    {
                        "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                        "region": "all_india",
                        "lead_week": lead,
                        "candidate": method,
                        "baseline": "raw_fuxi",
                        "metric": metric,
                        "effect": effect,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "candidate_mean": candidate_mean,
                        "baseline_mean": baseline_mean,
                        "n_cases": 169,
                        "confidence": 0.95,
                        "block_length_starts": 16,
                        "replicates": 10_000,
                    }
                )
        neural = scores.loc[("location_spread", lead)]
        moment_skill = 100.0 * (1.0 - float(neural["crps"]) / float(moment["crps"]))
        rows.append(
            {
                "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                "region": "all_india",
                "lead_week": lead,
                "candidate": "location_spread",
                "baseline": "moment_calibration",
                "metric": "crps",
                "effect": "skill_pct_vs_baseline",
                "estimate": moment_skill,
                "ci_lower": moment_skill - 1.0,
                "ci_upper": moment_skill + 1.0,
                "candidate_mean": float(neural["crps"]),
                "baseline_mean": float(moment["crps"]),
                "n_cases": 169,
                "confidence": 0.95,
                "block_length_starts": 16,
                "replicates": 10_000,
            }
        )
    return pd.DataFrame(rows)


def _build_release_inputs(root: Path) -> dict[str, object]:
    upstream = root / "upstream"
    benchmark_root = upstream / "benchmark"
    benchmark_root.mkdir(parents=True)
    _benchmark_interval_rows().to_csv(
        benchmark_root / "mme_vs_fuxi_paired_intervals.csv", index=False
    )
    benchmark, benchmark_sha = _finish_package(
        benchmark_root,
        "manifest.json",
        {"software_versions": {"python": "test"}},
    )

    data_config, data_config_sha = _simple_manifest(
        upstream / "data_sources.json",
        schema_version=1,
        verification_years=[2020, 2021, 2022, 2023, 2024],
        known_data_rules=[
            {
                "model": "ecmwf",
                "variable": "tp",
                "lead_week": 3,
                "status": "exclude_from_legacy_verification",
                "reason": "Documented synthetic resolution-transition artifact.",
            }
        ],
    )
    methods_manifest, methods_manifest_sha = _simple_manifest(
        upstream / "methods_manifest.json",
        schema_version=1,
        data_config=str(data_config.resolve()),
        data_config_sha256=data_config_sha,
        methods={
            "forecast_field": "ensemble_mean_weekly",
            "mme": "equal-weight mean of component-system ensemble means and their climatologies",
            "system_comparison_caveat": (
                "Models have unequal ensemble sizes; scores compare archived ensemble-mean "
                "systems, not equal-size intrinsic model skill."
            ),
        },
        tracks={
            "tp_imd": {
                "variable": "tp",
                "models": [
                    "cma",
                    "dlesym_v0",
                    "ecmwf",
                    "fuxi_s2s",
                    "ncep",
                    "neuralgcm",
                    "ukmo",
                ],
                "mme_components": [
                    "cma",
                    "dlesym_v0",
                    "fuxi_s2s",
                    "ncep",
                    "neuralgcm",
                    "ukmo",
                ],
                "unavailable_lead_weeks": {"ecmwf": [3]},
                "experiments": {
                    "cma": "physics/cma_operational_2020_2025",
                    "dlesym_v0": "model-run/dlesym/test_ens1",
                    "ecmwf": "physics/ecmwf_operational_2020_2025",
                    "fuxi_s2s": "model-run/fuxi/test_ens50",
                    "ncep": "physics/ncep_operational_2020_2025",
                    "neuralgcm": "model-run/neural-gcm/test_ens10",
                    "ukmo": "physics/ukmo_operational_2020_2025",
                },
            }
        },
    )
    scoring_root = upstream / "scoring"
    (scoring_root / "tables").mkdir(parents=True)
    _score_interval_rows().to_csv(
        scoring_root / "tables/paired_block_intervals.csv", index=False
    )
    scoring, scoring_sha = _finish_package(
        scoring_root,
        "manifest.json",
        {
            "cohort": {
                "maximum_target_label_opened": "2024-12-30",
                "valid_midpoint_jjas_count_per_lead": 169,
            },
            "sealed_2025_target_opened": False,
            "member_count_each_method": 50,
            "selected_checkpoint_sha256": "a" * 64,
            "uncertainty": {
                "replicates": 10_000,
                "block_lengths_starts": [16, 13],
            },
            "source_snapshot_sha256": {},
        },
    )
    training, training_sha = _simple_manifest(
        upstream / "training.json", software={"python": "test", "torch": "test"}
    )
    inference, inference_sha = _simple_manifest(upstream / "inference.json")
    preflight, preflight_sha = _simple_manifest(upstream / "preflight.json")

    table_root = root / "table_audit"
    table_root.mkdir()
    (table_root / "detail.txt").write_text("passed\n", encoding="utf-8")
    table_audit, table_sha = _finish_package(
        table_root,
        "audit_receipt.json",
        {
            "experiment": release.TABLE_AUDIT_EXPERIMENT,
            "status": "passed",
            "scoring_manifest": str(scoring.resolve()),
            "scoring_manifest_sha256": scoring_sha,
            "semantic_checks": {
                "full_case_cartesian_rows": 45_450,
                "sealed_2025_target_opened": False,
            },
            "source_snapshot_sha256": {},
        },
    )

    prediction_root = root / "prediction_audit"
    prediction_root.mkdir()
    (prediction_root / "metric_comparison.csv").write_text("metric,delta\ncrps,0\n", encoding="utf-8")
    prediction_audit, prediction_sha = _finish_package(
        prediction_root,
        "audit_receipt.json",
        {
            "experiment": release.PREDICTION_AUDIT_EXPERIMENT,
            "status": "passed",
            "inputs": {
                "scoring_manifest": str(scoring.resolve()),
                "scoring_manifest_sha256": scoring_sha,
                "parent_manifest": str(training.resolve()),
                "parent_manifest_sha256": training_sha,
                "inference_manifest": str(inference.resolve()),
                "inference_manifest_sha256": inference_sha,
                "preflight_manifest": str(preflight.resolve()),
                "preflight_manifest_sha256": preflight_sha,
            },
            "contract": {
                "sealed_2025_target_opened": False,
                "maximum_target_label_opened": "2024-12-30",
            },
            "reproduction": {
                "passed": True,
                "failure_count": 0,
                "matched_case_rows": 45_450,
                "max_absolute_difference_by_metric": {"crps": 1e-15, "variance": 2e-13},
            },
            "source_snapshot_sha256": {},
        },
    )

    alignment_root = root / "alignment_audit"
    alignment_root.mkdir()
    (alignment_root / "detail.txt").write_text("passed\n", encoding="utf-8")
    alignment_results = {
        "all_frozen_parent_artifact_hashes": "passed",
        "cache_file_byte_comparison": "passed",
        "native_lead_1_through_42_spot_check": "passed",
        "scoring_support_byte_comparison": "passed",
        "source_and_zmetadata_hashing": "passed",
        "split_and_embargo_comparison": "passed",
        "weekly_target_reconstruction": "passed",
    }
    alignment_audit, alignment_sha = _finish_package(
        alignment_root,
        "audit_receipt.json",
        {
            "experiment": release.ALIGNMENT_AUDIT_EXPERIMENT,
            "status": "passed",
            "inputs": {
                "parent_training": {
                    "manifest": str(training.resolve()),
                    "manifest_sha256": training_sha,
                }
            },
            "contract": {"weekly_target_shape": [2080, 6, 27, 27]},
            "results": alignment_results,
            "safety": {
                "sealed_2025_target_opened": False,
                "maximum_target_label_opened": "2022-02-09",
            },
            "historical_target_equality_limitation": {
                "retrospective_equality_to_consumed_target_tensors_provable": False,
                "statement": "Historical tensor equality is not provable because logical hashes were not persisted.",
            },
            "source_snapshot_sha256": {},
        },
    )

    core_root = root / "core"
    core_root.mkdir()
    _core_tables(core_root)
    core_manifest, core_sha = _finish_package(
        core_root,
        "manifest.json",
        {
            "experiment": release.CORE_EXPERIMENT,
            "status": "complete",
            "inputs": {
                "benchmark_manifest": str(benchmark.resolve()),
                "benchmark_manifest_sha256": benchmark_sha,
                "scoring_manifest": str(scoring.resolve()),
                "scoring_manifest_sha256": scoring_sha,
                "audit_receipt": str(table_audit.resolve()),
                "audit_receipt_sha256": table_sha,
            },
            "hard_gates": {
                "sealed_2025_target_opened": False,
                "final_headline_gate": release.FINAL_GATE,
            },
            "table_rows": {
                "main_deterministic_benchmark": 48,
                "calibrated_fuxi_by_lead": 18,
                "story_gate": 4,
            },
            "source_snapshot_sha256": {},
        },
    )

    comparison_root = root / "comparison"
    (comparison_root / "tables").mkdir(parents=True)
    _exact_common_rows().to_csv(
        comparison_root / "tables/exact_common_midpoint_sensitivity.csv", index=False
    )
    _mme_rows().to_csv(comparison_root / "tables/calibrated_fuxi_vs_mme.csv", index=False)
    comparison_manifest, comparison_sha = _finish_package(
        comparison_root,
        "manifest.json",
        {
            "experiment": release.COMPARISON_EXPERIMENT,
            "status": "complete",
            "inputs": {
                "scoring_manifest": str(scoring.resolve()),
                "scoring_manifest_sha256": scoring_sha,
                "audit_receipt": str(table_audit.resolve()),
                "audit_receipt_sha256": table_sha,
            },
            "exact_common_midpoint": {
                "common_midpoints_per_lead": 85,
                "case_lead_rows": 510,
                "year_counts": {"2022": 35, "2023": 35, "2024": 15},
            },
            "calibrated_fuxi_vs_mme": {
                "cases_per_lead": 169,
                "ecmwf_in_mme": False,
                "mme_members": ["cma", "dlesym_v0", "fuxi_s2s", "ncep", "neuralgcm", "ukmo"],
            },
            "sealed_2025_target_opened": False,
            "scoring_safety_contract": {"maximum_target_label_opened": "2024-12-30"},
            "software_versions": {"python": "test"},
            "source_snapshot_sha256": {},
        },
    )

    imerg_root = root / "imerg"
    (imerg_root / "tables").mkdir(parents=True)
    _imerg_rows().to_csv(imerg_root / "tables/paired_block_intervals.csv", index=False)
    imerg_manifest, imerg_sha = _finish_package(
        imerg_root,
        "manifest.json",
        {
            "experiment": release.IMERG_EXPERIMENT,
            "status": "complete",
            "parent_manifest": str(training.resolve()),
            "parent_manifest_sha256": training_sha,
            "input_inference_manifest": str(inference.resolve()),
            "input_inference_manifest_sha256": inference_sha,
            "input_preflight_manifest": str(preflight.resolve()),
            "input_preflight_manifest_sha256": preflight_sha,
            "cohort": {"maximum_target_label_opened": "2024-12-29"},
            "sealed_2025_target_opened": False,
            "software_versions": {"python": "test"},
            "source_snapshot_sha256": {},
        },
    )
    return {
        "core": core_manifest,
        "core_sha": core_sha,
        "comparison": comparison_manifest,
        "comparison_sha": comparison_sha,
        "imerg": imerg_manifest,
        "imerg_sha": imerg_sha,
        "table_audit": table_audit,
        "table_sha": table_sha,
        "prediction_audit": prediction_audit,
        "prediction_sha": prediction_sha,
        "alignment_audit": alignment_audit,
        "alignment_sha": alignment_sha,
        "methods_manifest": methods_manifest,
        "methods_manifest_sha": methods_manifest_sha,
        "data_config": data_config,
        "data_config_sha": data_config_sha,
    }


def _compile(inputs: dict[str, object], output: Path, output_root: Path) -> Path:
    return release.compile_release_bundle(
        inputs["core"],
        inputs["core_sha"],
        inputs["comparison"],
        inputs["comparison_sha"],
        inputs["imerg"],
        inputs["imerg_sha"],
        inputs["table_audit"],
        inputs["table_sha"],
        inputs["prediction_audit"],
        inputs["prediction_sha"],
        inputs["alignment_audit"],
        inputs["alignment_sha"],
        inputs["methods_manifest"],
        inputs["methods_manifest_sha"],
        inputs["data_config"],
        inputs["data_config_sha"],
        output,
        output_root=output_root,
    )


def test_compiles_complete_release_with_receipts_and_safety_gate(tmp_path: Path) -> None:
    inputs = _build_release_inputs(tmp_path)
    output_root = tmp_path / "release"
    manifest_path = _compile(inputs, output_root / "full", output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["hard_gates"]["sealed_2025_target_opened"] is False
    assert manifest["hard_gates"]["prediction_level_audit"] == "passed"
    assert manifest["hard_gates"]["comparison_findings"]["acc_unresolved_leads"] == [2]
    assert manifest["hard_gates"]["unfitted_imerg_gate"] == {
        "crps_skill_pct_vs_raw": "positive",
        "crps_skill_pct_vs_moment": "positive",
        "acc_delta_vs_raw": "unresolved",
        "bias_delta_vs_raw": "unresolved",
    }
    assert manifest["table_rows"] == {
        "main_deterministic_benchmark": 48,
        "calibrated_fuxi_by_lead": 18,
        "story_gate": 4,
        "table1_system_inventory": 8,
        "table2a_deterministic_benchmark": 48,
        "table2b_retrospective_calibrated_fuxi": 18,
        "exact_common_sensitivity": 16,
        "calibrated_fuxi_vs_fixed_mme": 72,
        "unfitted_imerg_gate": 4,
        "audit_summary": 3,
        "training_alignment_audit": 7,
        "safety_2025_seal": 8,
    }
    assert len(manifest["bound_upstream_files"]) >= 20
    table2b = pd.read_csv(
        manifest_path.parent / "tables/table2b_retrospective_calibrated_fuxi.csv"
    )
    assert len(table2b) == 18
    assert table2b["n_cases"].eq(169).all()
    assert table2b["member_count"].eq(50).all()
    assert table2b["pooled_gate_not_joined"].all()
    assert set(table2b["neural_checkpoint_sha256"]) == {"a" * 64}
    assert manifest["hard_gates"]["table2b_contract"]["pooled_gate_is_separate"] is True
    assert manifest["hard_gates"]["table1_contract"]["row_count"] == 8
    assert manifest["hard_gates"]["table2a_contract"]["row_count"] == 48
    table1 = pd.read_csv(manifest_path.parent / "tables/table1_system_inventory.csv")
    assert dict(zip(table1["system"], table1["native_ensemble_size"].astype(str))) == {
        "cma": "varies/not recoverable from bound manifest",
        "dlesym_v0": "1",
        "ecmwf": "varies/not recoverable from bound manifest",
        "fuxi_s2s": "50",
        "ncep": "varies/not recoverable from bound manifest",
        "neuralgcm": "10",
        "ukmo": "varies/not recoverable from bound manifest",
        "mme": "not applicable (derived)",
    }
    ecmwf = table1.loc[table1["system"].eq("ecmwf")].iloc[0]
    assert ecmwf["unavailable_leads"] == "W3"
    assert "resolution-transition" in ecmwf["exclusions_or_notes"]
    mme_inventory = table1.loc[table1["system"].eq("mme")].iloc[0]
    assert mme_inventory["family"] == "derived"
    assert not bool(mme_inventory["mme_member"])
    assert table1["comparison_caveat"].str.contains("unequal ensemble sizes").all()

    table2a = pd.read_csv(
        manifest_path.parent / "tables/table2a_deterministic_benchmark.csv"
    )
    assert len(table2a) == 48
    assert table2a.loc[table2a["model"].eq("mme"), "paired_effect_fields_populated"].all()
    assert table2a.loc[
        ~table2a["model"].eq("mme"),
        [
            "mme_minus_fuxi_acc_effect",
            "mme_minus_fuxi_rmse_effect",
            "mme_minus_fuxi_mae_effect",
            "mme_minus_fuxi_bias_effect",
        ],
    ].isna().all().all()
    assert table2a.loc[
        table2a["model"].eq("ecmwf") & table2a["lead_week"].eq(3),
        "valid_case_count",
    ].item() == 0
    assert table2a["paired_interval_scope"].str.contains("no multiplicity adjustment").all()
    assert set(
        table2a.loc[
            ~table2a["model"].isin(["mme", "fuxi_s2s"]), "paired_contrast_status"
        ]
    ) == {"pending_not_available"}
    assert "training_alignment_audit" in manifest["direct_inputs"]
    assert "benchmark_methods_manifest" in manifest["direct_inputs"]
    assert "benchmark_data_configuration" in manifest["direct_inputs"]
    assert "evidence_provenance" in manifest
    assert (
        manifest["evidence_provenance"]["byte_exact_imports"]
        ["receipts/training_alignment_audit_receipt.json"]["sha256"]
        == inputs["alignment_sha"]
    )
    for relative, digest in manifest["artifact_sha256"].items():
        assert release.core.sha256_file(manifest_path.parent / relative) == digest
    copied_receipt = manifest_path.parent / "receipts/prediction_audit_receipt.json"
    assert release.core.sha256_file(copied_receipt) == inputs["prediction_sha"]


def test_rejects_tampered_comparison_artifact(tmp_path: Path) -> None:
    inputs = _build_release_inputs(tmp_path)
    table = Path(inputs["comparison"]).parent / "tables/calibrated_fuxi_vs_mme.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(release.core.PaperBundleError, match="SHA-256 differs"):
        _compile(inputs, tmp_path / "release/full", tmp_path / "release")


def test_rejects_any_component_that_reports_opened_2025(tmp_path: Path) -> None:
    inputs = _build_release_inputs(tmp_path)
    receipt_path = Path(inputs["prediction_audit"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"]["sealed_2025_target_opened"] = True
    _write_json(receipt_path, receipt)
    inputs["prediction_sha"] = release.core.sha256_file(receipt_path)

    with pytest.raises(release.core.PaperBundleError, match="opened sealed 2025"):
        _compile(inputs, tmp_path / "release/full", tmp_path / "release")


def test_completed_release_is_no_clobber(tmp_path: Path) -> None:
    inputs = _build_release_inputs(tmp_path)
    output_root = tmp_path / "release"
    output = output_root / "full"
    _compile(inputs, output, output_root)

    with pytest.raises(FileExistsError, match="fresh paper-release output"):
        _compile(inputs, output, output_root)
