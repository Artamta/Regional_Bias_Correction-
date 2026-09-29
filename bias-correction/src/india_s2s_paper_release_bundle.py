#!/usr/bin/env python3
"""Publish the complete, receipt-bound India S2S paper evidence release.

This is a tabular release compiler.  It opens only immutable JSON, CSV,
Markdown, LaTeX, and manifest files; it never opens forecast, checkpoint, or
observation arrays.  Every direct input and every file declared by an input
artifact inventory is hash-verified before and after compilation.  Publication
uses an atomic no-replace rename so an existing release can never be clobbered.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import india_s2s_paper_evidence_bundle as core
from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_paper_release_bundle_v1"
CORE_EXPERIMENT = "india_s2s_paper_evidence_bundle_v1"
COMPARISON_EXPERIMENT = "india_s2s_paper_comparisons_v1"
IMERG_EXPERIMENT = "india_s2s_probabilistic_bridge_imerg_sensitivity_v1"
TABLE_AUDIT_EXPERIMENT = "india_s2s_probabilistic_bridge_score_audit_v1"
PREDICTION_AUDIT_EXPERIMENT = "india_s2s_probabilistic_bridge_prediction_audit_v1"
ALIGNMENT_AUDIT_EXPERIMENT = "fuxi_allseason_training_alignment_audit_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_paper_release_bundle"
PRIMARY_BLOCK_LENGTH = 16
FINAL_GATE = "pending_sealed_2025_point-estimate_direction_check"
STORY_COHORT = "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"


def _load_package(
    path: Path,
    expected_sha256: str,
    *,
    experiment: str,
    status: str,
    label: str,
    ledger: core.HashLedger,
) -> tuple[dict[str, Any], int]:
    resolved = Path(path).resolve()
    ledger.verify(resolved, expected_sha256, label)
    payload = core._read_json(resolved)
    core._require(payload.get("experiment") == experiment, f"{label} experiment differs")
    core._require(payload.get("status") == status, f"{label} status is not {status}")
    count = core._verify_artifacts(resolved, payload, ledger, label=label)
    return payload, count


def _same_binding(
    payload: Mapping[str, Any],
    path_key: str,
    sha_key: str,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    core._require(
        Path(str(payload.get(path_key, ""))).resolve() == Path(expected_path).resolve(),
        f"{label} path differs",
    )
    core._require(payload.get(sha_key) == expected_sha256, f"{label} SHA-256 differs")


def _bind_declared_file(
    payload: Mapping[str, Any],
    path_key: str,
    sha_key: str,
    label: str,
    ledger: core.HashLedger,
) -> Path:
    path = Path(str(payload.get(path_key, ""))).resolve()
    ledger.verify(path, payload.get(sha_key), label)
    return path


def _assert_release_output(destination: Path, output_root: Path) -> None:
    resolved = Path(destination).resolve()
    root = Path(output_root).resolve()
    core._require(root in resolved.parents, "release output must be a fresh child of its output root")
    core._require(resolved != root, "release output cannot equal its output root")


def _copy_bound_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination)


def _write_table_formats(
    staging: Path,
    name: str,
    frame: pd.DataFrame,
    publication_columns: Sequence[str],
) -> None:
    core._require(
        set(publication_columns).issubset(frame.columns),
        f"{name} lacks publication columns",
    )
    tables = staging / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv", index=False, lineterminator="\n")
    (tables / f"{name}.md").write_text(
        core._markdown_table(frame, publication_columns) + "\n", encoding="utf-8"
    )
    (tables / f"{name}.tex").write_text(
        core._latex_table(frame, publication_columns) + "\n", encoding="utf-8"
    )


def _copy_core_tables(
    core_manifest_path: Path,
    manifest: Mapping[str, Any],
    staging: Path,
) -> dict[str, int]:
    expected_rows = {
        "main_deterministic_benchmark": 48,
        "calibrated_fuxi_by_lead": 18,
        "story_gate": 4,
    }
    core._require(manifest.get("table_rows") == expected_rows, "core table inventory differs")
    root = Path(core_manifest_path).resolve().parent
    for name, expected_count in expected_rows.items():
        csv_source = core._resolve_child(root, f"tables/{name}.csv")
        frame = core._read_csv(csv_source, f"core {name}")
        core._require(len(frame) == expected_count, f"core {name} row count differs")
        for extension in ("csv", "md", "tex"):
            relative = f"tables/{name}.{extension}"
            source = core._resolve_child(root, relative)
            core._require(
                manifest.get("artifact_sha256", {}).get(relative) == core.sha256_file(source),
                f"core {relative} is not artifact-bound",
            )
            _copy_bound_file(source, staging / relative)
    return expected_rows


def _table1_system_inventory(
    methods_manifest_path: Path,
    methods_manifest: Mapping[str, Any],
    data_config_path: Path,
    data_config: Mapping[str, Any],
    staging: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build system metadata without inferring unavailable ensemble sizes."""

    core._require(
        Path(str(methods_manifest.get("data_config", ""))).resolve()
        == Path(data_config_path).resolve(),
        "methods manifest points to a different data configuration",
    )
    track = methods_manifest.get("tracks", {}).get("tp_imd", {})
    experiments = track.get("experiments", {})
    models = track.get("models", [])
    mme_components = track.get("mme_components", [])
    core._require(
        models == ["cma", "dlesym_v0", "ecmwf", "fuxi_s2s", "ncep", "neuralgcm", "ukmo"]
        and mme_components == ["cma", "dlesym_v0", "fuxi_s2s", "ncep", "neuralgcm", "ukmo"],
        "Table 1 benchmark system inventory differs",
    )
    years = data_config.get("verification_years")
    core._require(years == [2020, 2021, 2022, 2023, 2024], "Table 1 verification years differ")
    known_rules = data_config.get("known_data_rules", [])
    ecmwf_rule = [
        rule
        for rule in known_rules
        if rule.get("model") == "ecmwf"
        and rule.get("variable") == "tp"
        and rule.get("lead_week") == 3
    ]
    core._require(len(ecmwf_rule) == 1, "Table 1 lacks the bound ECMWF-W3 exclusion rule")
    methods = methods_manifest.get("methods", {})
    archived_treatment = methods.get("forecast_field")
    mme_treatment = methods.get("mme")
    comparison_caveat = methods.get("system_comparison_caveat")
    core._require(
        archived_treatment == "ensemble_mean_weekly"
        and isinstance(mme_treatment, str)
        and isinstance(comparison_caveat, str)
        and bool(comparison_caveat.strip()),
        "Table 1 archived ensemble treatment differs",
    )

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

    def family(experiment: str) -> tuple[str, str]:
        if experiment.startswith("physics/"):
            return "dynamical", "explicit physics/ archive namespace"
        if experiment.startswith(("model-run/dlesym/", "model-run/fuxi/", "model-run/neural-gcm/")):
            return "AI", "explicit model-run archive namespace for named AI system"
        return "not recoverable from bound manifest", "archive namespace is insufficient"

    def ensemble_size(experiment: str) -> str:
        match = re.search(r"(?:^|_)ens([0-9]+)(?:$|_)", experiment)
        return match.group(1) if match else "varies/not recoverable from bound manifest"

    unavailable_by_model = track.get("unavailable_lead_weeks", {})
    rows: list[dict[str, Any]] = []
    for model in (*models, "mme"):
        if model == "mme":
            experiment = "not applicable (derived equal-system MME)"
            model_family = "derived"
            family_basis = "methods manifest defines an equal-weight MME"
            native_size = "not applicable (derived)"
            treatment = str(mme_treatment)
            unavailable: list[int] = []
            notes = "Derived from CMA, DLESyM-v0, FuXi-S2S, NCEP, NeuralGCM, and UKMO; ECMWF excluded at every lead."
        else:
            experiment = str(experiments.get(model, ""))
            core._require(bool(experiment), f"Table 1 lacks experiment identifier for {model}")
            model_family, family_basis = family(experiment)
            native_size = ensemble_size(experiment)
            treatment = str(archived_treatment)
            unavailable = [int(value) for value in unavailable_by_model.get(model, [])]
            if model == "ecmwf":
                notes = (
                    "W3 excluded: "
                    + str(ecmwf_rule[0].get("reason"))
                    + " Excluded from the fixed MME at every lead."
                )
            elif model in mme_components:
                notes = "Included in the fixed six-system equal-system MME."
            else:
                notes = "Not included in the fixed MME."
        available = [lead for lead in range(1, 7) if lead not in unavailable]
        rows.append(
            {
                "system": model,
                "system_label": labels[model],
                "family": model_family,
                "family_basis": family_basis,
                "experiment_or_archive_identifier": experiment,
                "archived_ensemble_treatment": treatment,
                "native_ensemble_size": native_size,
                "mme_member": model in mme_components,
                "variable": str(track.get("variable")),
                "verification_years": f"{years[0]}-{years[-1]}",
                "available_leads": ",".join(f"W{lead}" for lead in available),
                "unavailable_leads": (
                    ",".join(f"W{lead}" for lead in unavailable) if unavailable else "none"
                ),
                "exclusions_or_notes": notes,
                "comparison_caveat": comparison_caveat,
            }
        )
    output = pd.DataFrame(rows)
    core._require(len(output) == 8, "Table 1 row count differs")
    _write_table_formats(
        staging,
        "table1_system_inventory",
        output,
        (
            "system_label",
            "family",
            "experiment_or_archive_identifier",
            "archived_ensemble_treatment",
            "native_ensemble_size",
            "mme_member",
            "variable",
            "verification_years",
            "available_leads",
            "unavailable_leads",
            "exclusions_or_notes",
            "comparison_caveat",
        ),
    )
    contract = {
        "status": "complete",
        "table": "tables/table1_system_inventory.csv",
        "row_count": 8,
        "methods_manifest": str(Path(methods_manifest_path).resolve()),
        "data_config": str(Path(data_config_path).resolve()),
        "native_ensemble_size_policy": (
            "Parse only explicit _ensN tokens in bound experiment identifiers; otherwise "
            "report varies/not recoverable from bound manifest."
        ),
        "family_policy": (
            "Use explicit physics/ namespaces for dynamical systems, explicit named model-run "
            "namespaces for AI systems, and derived for the MME; otherwise report unrecoverable."
        ),
        "comparison_caveat": comparison_caveat,
    }
    core._write_json(staging / "receipts/table1_contract.json", contract)
    return output, contract


def _table2a_deterministic_benchmark(
    benchmark_manifest_path: Path,
    benchmark: Mapping[str, Any],
    staging: Path,
    ledger: core.HashLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach only the supported pointwise MME-minus-FuXi intervals."""

    deterministic = core._read_csv(
        staging / "tables/main_deterministic_benchmark.csv",
        "core deterministic benchmark table",
    )
    core._require(
        len(deterministic) == 48
        and set(deterministic["model"]) == {
            "cma",
            "dlesym_v0",
            "ecmwf",
            "fuxi_s2s",
            "ncep",
            "neuralgcm",
            "ukmo",
            "mme",
        }
        and set(deterministic["lead_week"].astype(int)) == set(range(1, 7)),
        "Table 2A deterministic inventory differs",
    )
    benchmark_root = Path(benchmark_manifest_path).resolve().parent
    interval_relative = "mme_vs_fuxi_paired_intervals.csv"
    interval_path = core._resolve_child(benchmark_root, interval_relative)
    interval_sha = benchmark.get("artifact_sha256", {}).get(interval_relative)
    ledger.verify(interval_path, interval_sha, "Table 2A pointwise interval source")
    intervals = core._read_csv(interval_path, "Table 2A MME-minus-FuXi intervals")
    primary = intervals[
        intervals["reference"].eq("imd")
        & intervals["season"].eq("JJAS_valid_midpoint")
        & intervals["region"].eq("all_india")
        & intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ]
    core._require(len(primary) == 24, "Table 2A focal interval inventory differs")
    output = deterministic.copy()
    output.insert(0, "reference", "imd")
    output.insert(1, "season", "JJAS_valid_midpoint")
    output.insert(2, "analysis_cohort", "2020_2024_valid_midpoint_jjas")
    output["paired_contrast"] = output["model"].map(
        lambda model: "mme_minus_fuxi_s2s" if model in {"mme", "fuxi_s2s"} else "not_applicable"
    )
    output["paired_contrast_role"] = output["model"].map(
        lambda model: "candidate_effect_reported_here"
        if model == "mme"
        else "baseline_reference_effect_reported_on_mme_row"
        if model == "fuxi_s2s"
        else "unsupported_contrast"
    )
    output["paired_contrast_status"] = output["model"].map(
        lambda model: "available_on_this_candidate_row"
        if model == "mme"
        else "baseline_reference_only"
        if model == "fuxi_s2s"
        else "pending_not_available"
    )
    output["paired_effect_fields_populated"] = output["model"].eq("mme")
    output["paired_interval_scope"] = (
        "pointwise paired 95% interval; 16-start blocks; no multiplicity adjustment; "
        "only MME-minus-FuXi is supported"
    )
    output["paired_interval_confidence"] = 0.95
    output["paired_interval_block_length_starts"] = PRIMARY_BLOCK_LENGTH
    output["paired_interval_replicates"] = 10_000
    output["unsupported_contrast_note"] = output["model"].map(
        lambda model: "not applicable"
        if model == "mme"
        else "effect is reported once on the MME candidate row"
        if model == "fuxi_s2s"
        else "pending; no bound paired interval for this system contrast"
    )
    for metric in ("acc", "rmse", "mae", "bias"):
        for suffix in ("effect", "ci_lower", "ci_upper"):
            output[f"mme_minus_fuxi_{metric}_{suffix}"] = np.nan

    for lead in range(1, 7):
        mme_index = output.index[
            output["model"].eq("mme") & output["lead_week"].eq(lead)
        ]
        fuxi = output[output["model"].eq("fuxi_s2s") & output["lead_week"].eq(lead)]
        core._require(len(mme_index) == 1 and len(fuxi) == 1, f"Table 2A lacks focal W{lead} rows")
        for metric in ("acc", "rmse", "mae", "bias"):
            match = primary[
                primary["lead_week"].eq(lead)
                & primary["metric"].eq(metric)
                & primary["candidate"].eq("mme")
                & primary["baseline"].eq("fuxi_s2s")
                & primary["effect"].eq("mme_minus_fuxi")
            ]
            core._require(
                len(match) == 1
                and int(match.iloc[0]["n_cases"]) == 169
                and int(match.iloc[0]["replicates"]) == 10_000,
                f"Table 2A focal interval differs: W{lead} {metric}",
            )
            interval = match.iloc[0]
            mme_value = float(output.loc[mme_index[0], metric])
            fuxi_value = float(fuxi.iloc[0][metric])
            core._require(
                np.isclose(float(interval["candidate_mean"]), mme_value, atol=1e-12)
                and np.isclose(float(interval["baseline_mean"]), fuxi_value, atol=1e-12),
                f"Table 2A point estimate differs from interval source: W{lead} {metric}",
            )
            output.loc[mme_index, f"mme_minus_fuxi_{metric}_effect"] = float(
                interval["estimate"]
            )
            output.loc[mme_index, f"mme_minus_fuxi_{metric}_ci_lower"] = float(
                interval["ci_lower"]
            )
            output.loc[mme_index, f"mme_minus_fuxi_{metric}_ci_upper"] = float(
                interval["ci_upper"]
            )

    unsupported = ~output["model"].eq("mme")
    contrast_columns = [
        column
        for column in output.columns
        if column.startswith("mme_minus_fuxi_")
        and column.endswith(("_effect", "_ci_lower", "_ci_upper"))
    ]
    core._require(
        output.loc[unsupported, contrast_columns].isna().all().all(),
        "Table 2A populated an unsupported contrast",
    )
    _write_table_formats(
        staging,
        "table2a_deterministic_benchmark",
        output,
        (
            "model_label",
            "lead_week",
            "valid_case_count",
            "acc",
            "rmse",
            "mae",
            "bias",
            "paired_contrast_role",
            "paired_contrast_status",
            "mme_minus_fuxi_acc_effect",
            "mme_minus_fuxi_acc_ci_lower",
            "mme_minus_fuxi_acc_ci_upper",
            "mme_minus_fuxi_rmse_effect",
            "mme_minus_fuxi_rmse_ci_lower",
            "mme_minus_fuxi_rmse_ci_upper",
            "paired_interval_scope",
        ),
    )
    contract = {
        "status": "complete",
        "table": "tables/table2a_deterministic_benchmark.csv",
        "row_count": 48,
        "all_system_metrics": ["acc", "rmse", "mae", "bias"],
        "all_system_case_count_field": "valid_case_count",
        "supported_paired_contrast": "mme_minus_fuxi_s2s",
        "effect_fields_populated_only_on": "mme candidate rows",
        "unsupported_contrast_fields": "null with explicit role/note",
        "other_system_contrast_status": "pending_not_available",
        "interval_scope": (
            "pointwise paired 95% intervals from 10,000 year-stratified circular "
            "block resamples with 16-start blocks; no multiplicity adjustment"
        ),
    }
    core._write_json(staging / "receipts/table2a_contract.json", contract)
    return output, contract


def _table2b_retrospective_calibrated_fuxi(
    scoring_manifest_path: Path,
    scoring: Mapping[str, Any],
    staging: Path,
    ledger: core.HashLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join lead scores to leadwise intervals without mixing in the pooled gate."""

    scores = core._read_csv(
        staging / "tables/calibrated_fuxi_by_lead.csv",
        "core calibrated-FuXi lead table",
    )
    required_scores = {
        "method",
        "method_label",
        "lead_week",
        "case_count",
        "crps",
        "acc",
        "rmse",
        "bias",
        "coverage90",
        "pooled_spread_skill_ratio",
    }
    core._require(required_scores.issubset(scores.columns), "core calibrated table schema differs")
    core._require(
        len(scores) == 18
        and set(scores["method"]) == {"raw_fuxi", "moment_calibration", "location_spread"}
        and set(scores["lead_week"].astype(int)) == set(range(1, 7))
        and (scores["case_count"] == 169).all(),
        "core calibrated table cohort differs",
    )
    scoring_root = Path(scoring_manifest_path).resolve().parent
    interval_relative = "tables/paired_block_intervals.csv"
    interval_path = core._resolve_child(scoring_root, interval_relative)
    interval_sha = scoring.get("artifact_sha256", {}).get(interval_relative)
    ledger.verify(interval_path, interval_sha, "Table 2B leadwise interval source")
    intervals = core._read_csv(interval_path, "Table 2B leadwise intervals")
    primary = intervals[
        intervals["analysis_cohort"].eq("2020_2024_valid_midpoint_jjas")
        & intervals["region"].eq("all_india")
        & intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ]
    member_count = scoring.get("member_count_each_method")
    checkpoint_sha256 = scoring.get("selected_checkpoint_sha256")
    core._require(member_count == 50, "Table 2B member count differs")
    core._require(core._valid_sha256(checkpoint_sha256), "Table 2B checkpoint hash is invalid")
    core._require(
        scoring.get("cohort", {}).get("valid_midpoint_jjas_count_per_lead") == 169,
        "Table 2B scoring cohort differs",
    )
    uncertainty = scoring.get("uncertainty", {})
    core._require(
        uncertainty.get("replicates") == 10_000
        and uncertainty.get("block_lengths_starts") == [16, 13],
        "Table 2B uncertainty contract differs",
    )

    score_to_interval_metric = {
        "acc": "acc",
        "rmse": "rmse",
        "bias": "bias",
        "coverage90": "coverage90",
        "pooled_spread_skill_ratio": "spread_skill_ratio",
    }

    def interval_row(
        lead: int,
        candidate: str,
        baseline: str,
        metric: str,
        effect: str,
    ) -> pd.Series:
        match = primary[
            primary["lead_week"].astype(str).eq(str(lead))
            & primary["candidate"].eq(candidate)
            & primary["baseline"].eq(baseline)
            & primary["metric"].eq(metric)
            & primary["effect"].eq(effect)
        ]
        core._require(
            len(match) == 1
            and int(match.iloc[0]["n_cases"]) == 169
            and int(match.iloc[0]["replicates"]) == 10_000
            and np.isclose(float(match.iloc[0]["confidence"]), 0.95),
            f"Table 2B interval row differs: W{lead} {candidate}/{baseline} {metric} {effect}",
        )
        return match.iloc[0]

    rows: list[dict[str, Any]] = []
    for score in scores.itertuples(index=False):
        lead = int(score.lead_week)
        method = str(score.method)
        row: dict[str, Any] = {
            "reference": "imd",
            "season": "JJAS_valid_midpoint",
            "analysis_cohort": "2020_2024_valid_midpoint_jjas",
            "method": method,
            "method_label": str(score.method_label),
            "lead_week": lead,
            "n_cases": int(score.case_count),
            "member_count": int(member_count),
            "neural_checkpoint_sha256": str(checkpoint_sha256),
            "method_uses_selected_checkpoint": method == "location_spread",
            "crps": float(score.crps),
            "acc": float(score.acc),
            "rmse": float(score.rmse),
            "bias": float(score.bias),
            "coverage90": float(score.coverage90),
            "pooled_spread_skill_ratio": float(score.pooled_spread_skill_ratio),
            "paired_effect_baseline": "raw_fuxi" if method != "raw_fuxi" else "not_applicable",
            "primary_interval_block_length_starts": PRIMARY_BLOCK_LENGTH,
            "confidence": 0.95,
            "replicates": 10_000,
            "leadwise_interval_scope": "2020-2024; 169 cases at this lead",
            "pooled_gate_not_joined": True,
            "pooled_gate_location": "tables/story_gate.csv",
            "pooled_gate_scope": "2022-2024 synchronized; 100 cases/lead; 600 case-lead rows",
        }
        for prefix in (
            "crps_skill_pct_vs_raw",
            "acc_delta_vs_raw",
            "rmse_delta_vs_raw",
            "bias_delta_vs_raw",
            "coverage90_delta_vs_raw",
            "spread_skill_ratio_delta_vs_raw",
            "crps_skill_pct_vs_moment",
        ):
            row[f"{prefix}_estimate"] = np.nan
            row[f"{prefix}_ci_lower"] = np.nan
            row[f"{prefix}_ci_upper"] = np.nan

        if method != "raw_fuxi":
            crps_effect = interval_row(
                lead, method, "raw_fuxi", "crps", "skill_pct_vs_baseline"
            )
            for column in ("estimate", "ci_lower", "ci_upper"):
                row[f"crps_skill_pct_vs_raw_{column}"] = float(crps_effect[column])
            for score_column, interval_metric in score_to_interval_metric.items():
                effect = interval_row(
                    lead, method, "raw_fuxi", interval_metric, "candidate_minus_baseline"
                )
                prefix = (
                    "spread_skill_ratio_delta_vs_raw"
                    if score_column == "pooled_spread_skill_ratio"
                    else f"{score_column}_delta_vs_raw"
                )
                for column in ("estimate", "ci_lower", "ci_upper"):
                    row[f"{prefix}_{column}"] = float(effect[column])
                core._require(
                    np.isclose(float(effect["candidate_mean"]), float(getattr(score, score_column)), atol=1e-12),
                    f"Table 2B score/interval candidate mean differs: W{lead} {method} {score_column}",
                )
        if method == "location_spread":
            moment_effect = interval_row(
                lead,
                "location_spread",
                "moment_calibration",
                "crps",
                "skill_pct_vs_baseline",
            )
            for column in ("estimate", "ci_lower", "ci_upper"):
                row[f"crps_skill_pct_vs_moment_{column}"] = float(moment_effect[column])
        rows.append(row)

    output = pd.DataFrame(rows)
    output["_method_order"] = output["method"].map(
        {"raw_fuxi": 0, "moment_calibration": 1, "location_spread": 2}
    )
    output = (
        output.sort_values(["_method_order", "lead_week"], kind="stable")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )
    core._require(len(output) == 18, "Table 2B row count differs")
    _write_table_formats(
        staging,
        "table2b_retrospective_calibrated_fuxi",
        output,
        (
            "method_label",
            "lead_week",
            "n_cases",
            "member_count",
            "crps",
            "acc",
            "rmse",
            "bias",
            "coverage90",
            "pooled_spread_skill_ratio",
            "crps_skill_pct_vs_raw_estimate",
            "crps_skill_pct_vs_raw_ci_lower",
            "crps_skill_pct_vs_raw_ci_upper",
            "acc_delta_vs_raw_estimate",
            "acc_delta_vs_raw_ci_lower",
            "acc_delta_vs_raw_ci_upper",
            "neural_checkpoint_sha256",
        ),
    )
    contract = {
        "status": "complete",
        "leadwise_table": "tables/table2b_retrospective_calibrated_fuxi.csv",
        "leadwise_cohort": "2020_2024_valid_midpoint_jjas",
        "leadwise_cases_per_method_and_lead": 169,
        "member_count_each_method": 50,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "leadwise_primary_interval": {
            "block_length_starts": 16,
            "confidence": 0.95,
            "replicates": 10_000,
        },
        "pooled_gate_is_separate": True,
        "pooled_gate_table": "tables/story_gate.csv",
        "pooled_gate_cohort": STORY_COHORT,
        "pooled_gate_cases_per_lead": 100,
        "pooled_gate_case_lead_rows": 600,
        "warning": "Never attach pooled story-gate intervals to individual leadwise rows.",
    }
    core._write_json(staging / "receipts/table2b_contract.json", contract)
    return output, contract


def _comparison_tables(
    comparison_manifest_path: Path,
    manifest: Mapping[str, Any],
    staging: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    root = Path(comparison_manifest_path).resolve().parent
    exact_path = core._resolve_child(root, "tables/exact_common_midpoint_sensitivity.csv")
    mme_path = core._resolve_child(root, "tables/calibrated_fuxi_vs_mme.csv")
    exact = core._read_csv(exact_path, "exact-common sensitivity")
    mme = core._read_csv(mme_path, "calibrated FuXi versus MME")
    core._require(len(exact) == 16, "exact-common sensitivity must contain 16 rows")
    core._require(len(mme) == 72, "calibrated-FuXi versus MME must contain 72 rows")
    exact_contract = manifest.get("exact_common_midpoint", {})
    core._require(
        exact_contract.get("common_midpoints_per_lead") == 85
        and exact_contract.get("case_lead_rows") == 510
        and exact_contract.get("year_counts") == {"2022": 35, "2023": 35, "2024": 15},
        "exact-common cohort contract differs",
    )
    core._require(
        set(exact["block_length_valid_midpoints"].astype(int)) == {13, 16}
        and (exact["n_common_midpoints_per_lead"] == 85).all()
        and (exact["n_case_lead_rows"] == 510).all(),
        "exact-common table sampling contract differs",
    )
    mme_contract = manifest.get("calibrated_fuxi_vs_mme", {})
    core._require(
        mme_contract.get("cases_per_lead") == 169
        and mme_contract.get("ecmwf_in_mme") is False
        and mme_contract.get("mme_members")
        == ["cma", "dlesym_v0", "fuxi_s2s", "ncep", "neuralgcm", "ukmo"],
        "fixed-MME contract differs",
    )
    core._require(
        set(mme["block_length_starts"].astype(int)) == {13, 16}
        and (mme["n_cases"] == 169).all()
        and set(mme["lead_week"].astype(int)) == set(range(1, 7)),
        "calibrated-FuXi versus MME sampling contract differs",
    )
    _write_table_formats(
        staging,
        "exact_common_sensitivity",
        exact,
        (
            "baseline",
            "metric",
            "effect",
            "estimate",
            "ci_lower",
            "ci_upper",
            "block_length_valid_midpoints",
            "n_common_midpoints_per_lead",
            "n_case_lead_rows",
        ),
    )
    _write_table_formats(
        staging,
        "calibrated_fuxi_vs_fixed_mme",
        mme,
        (
            "lead_week",
            "metric",
            "effect",
            "candidate_mean",
            "baseline_mean",
            "estimate",
            "ci_lower",
            "ci_upper",
            "block_length_starts",
            "n_cases",
        ),
    )

    primary = mme[mme["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)]
    rmse_skill = primary[
        primary["metric"].eq("rmse") & primary["effect"].eq("skill_pct_vs_baseline")
    ]
    acc_delta = primary[
        primary["metric"].eq("acc") & primary["effect"].eq("candidate_minus_baseline")
    ]
    rmse_better = sorted(rmse_skill.loc[rmse_skill["ci_lower"] > 0.0, "lead_week"].astype(int))
    acc_better = sorted(acc_delta.loc[acc_delta["ci_lower"] > 0.0, "lead_week"].astype(int))
    acc_unresolved = sorted(
        acc_delta.loc[
            (acc_delta["ci_lower"] <= 0.0) & (acc_delta["ci_upper"] >= 0.0), "lead_week"
        ].astype(int)
    )
    core._require(rmse_better == [1, 2, 3, 4, 5, 6], "fixed-MME RMSE finding differs")
    core._require(acc_better == [1, 3, 4, 5, 6] and acc_unresolved == [2], "fixed-MME ACC finding differs")
    findings = {
        "rmse_lower_primary_interval_leads": rmse_better,
        "acc_higher_primary_interval_leads": acc_better,
        "acc_unresolved_leads": acc_unresolved,
        "fixed_mme_excludes_ecmwf": True,
        "exact_common_midpoints_per_lead": 85,
        "exact_common_year_counts": exact_contract["year_counts"],
    }
    return {
        "exact_common_sensitivity": len(exact),
        "calibrated_fuxi_vs_fixed_mme": len(mme),
    }, findings


def _imerg_gate_table(
    imerg_manifest_path: Path,
    manifest: Mapping[str, Any],
    staging: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    root = Path(imerg_manifest_path).resolve().parent
    intervals = core._read_csv(
        core._resolve_child(root, "tables/paired_block_intervals.csv"),
        "IMERG paired intervals",
    )
    primary = intervals[
        intervals["analysis_cohort"].eq(STORY_COHORT)
        & intervals["region"].eq("all_india")
        & intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ]
    specifications = (
        ("crps_skill_pct_vs_raw", "crps", "raw_fuxi", "skill_pct_vs_baseline", "positive"),
        (
            "crps_skill_pct_vs_moment",
            "crps",
            "moment_calibration",
            "skill_pct_vs_baseline",
            "positive",
        ),
        ("acc_delta_vs_raw", "acc", "raw_fuxi", "candidate_minus_baseline", "unresolved"),
        ("bias_delta_vs_raw", "bias", "raw_fuxi", "candidate_minus_baseline", "unresolved"),
    )
    rows: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    for gate, metric, baseline, effect, expected_decision in specifications:
        match = primary[
            primary["metric"].eq(metric)
            & primary["candidate"].eq("location_spread")
            & primary["baseline"].eq(baseline)
            & primary["effect"].eq(effect)
        ]
        core._require(len(match) == 1, f"missing IMERG gate row: {gate}")
        row = match.iloc[0]
        lower, upper = float(row["ci_lower"]), float(row["ci_upper"])
        decision = "positive" if lower > 0.0 else "unresolved" if lower <= 0.0 <= upper else "negative"
        core._require(decision == expected_decision, f"IMERG gate decision differs: {gate}")
        core._require(
            int(row["n_cases_per_lead"]) == 100
            and int(row["n_case_lead_rows"]) == 600
            and int(row["replicates"]) == 10_000,
            f"IMERG sampling contract differs: {gate}",
        )
        decisions[gate] = decision
        rows.append(
            {
                "gate": gate,
                "reference": "imerg",
                "baseline": baseline,
                "metric": metric,
                "estimate": float(row["estimate"]),
                "ci_lower": lower,
                "ci_upper": upper,
                "decision": decision,
                "primary_block_length_starts": PRIMARY_BLOCK_LENGTH,
                "cases_per_lead": int(row["n_cases_per_lead"]),
                "case_lead_rows": int(row["n_case_lead_rows"]),
                "interpretation": "unfitted observation-reference sensitivity",
            }
        )
    frame = pd.DataFrame(rows)
    _write_table_formats(
        staging,
        "unfitted_imerg_gate",
        frame,
        (
            "gate",
            "estimate",
            "ci_lower",
            "ci_upper",
            "decision",
            "cases_per_lead",
            "case_lead_rows",
        ),
    )
    core._require(
        manifest.get("cohort", {}).get("maximum_target_label_opened") == "2024-12-29",
        "IMERG maximum target label differs",
    )
    return frame, decisions


def _audit_tables(
    table_audit: Mapping[str, Any],
    prediction_audit: Mapping[str, Any],
    alignment_audit: Mapping[str, Any],
    staging: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_semantic = table_audit.get("semantic_checks", {})
    prediction_reproduction = prediction_audit.get("reproduction", {})
    alignment_results = alignment_audit.get("results", {})
    core._require(
        table_semantic.get("full_case_cartesian_rows") == 45_450
        and table_semantic.get("sealed_2025_target_opened") is False,
        "table audit contract differs",
    )
    core._require(
        prediction_reproduction.get("passed") is True
        and prediction_reproduction.get("failure_count") == 0
        and prediction_reproduction.get("matched_case_rows") == 45_450,
        "prediction audit did not reproduce all rows",
    )
    core._require(
        alignment_results
        and all(value == "passed" for value in alignment_results.values()),
        "training-alignment audit has a failed result",
    )
    maximum_prediction_delta = max(
        float(value)
        for value in prediction_reproduction.get("max_absolute_difference_by_metric", {}).values()
    )
    audit_summary = pd.DataFrame(
        (
            {
                "audit": "table_level",
                "status": "passed",
                "matched_or_checked_rows": int(table_semantic["full_case_cartesian_rows"]),
                "failure_count": 0,
                "maximum_absolute_metric_difference": np.nan,
                "independence": "reconstructed summaries, intervals, and story gate",
                "limitation": "does not reopen forecast or truth arrays",
            },
            {
                "audit": "prediction_level",
                "status": "passed",
                "matched_or_checked_rows": int(prediction_reproduction["matched_case_rows"]),
                "failure_count": int(prediction_reproduction["failure_count"]),
                "maximum_absolute_metric_difference": maximum_prediction_delta,
                "independence": "reopened raw members and IMD; reimplemented alignment, members, and metrics",
                "limitation": "retrospective 2020-2024 only",
            },
            {
                "audit": "training_alignment",
                "status": "passed_with_recorded_limitation",
                "matched_or_checked_rows": int(
                    alignment_audit.get("contract", {}).get("weekly_target_shape", [0])[0]
                ),
                "failure_count": 0,
                "maximum_absolute_metric_difference": 0.0,
                "independence": "reimplemented target construction, native conversion, splits, and support",
                "limitation": alignment_audit.get("historical_target_equality_limitation", {}).get(
                    "statement", ""
                ),
            },
        )
    )
    _write_table_formats(
        staging,
        "audit_summary",
        audit_summary,
        (
            "audit",
            "status",
            "matched_or_checked_rows",
            "failure_count",
            "maximum_absolute_metric_difference",
            "limitation",
        ),
    )

    alignment_rows = pd.DataFrame(
        [
            {"check": check, "status": status, "target_offsets": "+1...+42", "sealed_2025": True}
            for check, status in sorted(alignment_results.items())
        ]
    )
    _write_table_formats(
        staging,
        "training_alignment_audit",
        alignment_rows,
        ("check", "status", "target_offsets", "sealed_2025"),
    )
    limitation = alignment_audit.get("historical_target_equality_limitation", {})
    core._write_json(staging / "receipts/training_alignment_limitation.json", limitation)
    (staging / "receipts/training_alignment_limitation.md").write_text(
        "# Training-alignment limitation\n\n"
        + str(limitation.get("statement", ""))
        + "\n",
        encoding="utf-8",
    )
    return audit_summary, alignment_rows


def _safety_receipt(
    core_manifest: Mapping[str, Any],
    scoring: Mapping[str, Any],
    table_audit: Mapping[str, Any],
    prediction_audit: Mapping[str, Any],
    comparison: Mapping[str, Any],
    imerg: Mapping[str, Any],
    alignment: Mapping[str, Any],
    staging: Path,
) -> pd.DataFrame:
    components = (
        ("core_bundle", core_manifest.get("hard_gates", {}).get("sealed_2025_target_opened"), "2024-12-30"),
        ("imd_scoring", scoring.get("sealed_2025_target_opened"), scoring.get("cohort", {}).get("maximum_target_label_opened")),
        ("table_audit", table_audit.get("semantic_checks", {}).get("sealed_2025_target_opened"), "2024-12-30"),
        ("prediction_audit", prediction_audit.get("contract", {}).get("sealed_2025_target_opened"), prediction_audit.get("contract", {}).get("maximum_target_label_opened")),
        ("paper_comparisons", comparison.get("sealed_2025_target_opened"), comparison.get("scoring_safety_contract", {}).get("maximum_target_label_opened")),
        ("imerg_sensitivity", imerg.get("sealed_2025_target_opened"), imerg.get("cohort", {}).get("maximum_target_label_opened")),
        ("training_alignment", alignment.get("safety", {}).get("sealed_2025_target_opened"), alignment.get("safety", {}).get("maximum_target_label_opened")),
        ("release_compiler", False, "not_opened"),
    )
    rows = [
        {
            "component": name,
            "sealed_2025_target_opened": bool(opened),
            "maximum_target_label_opened": maximum,
            "status": "sealed" if opened is False else "violation",
        }
        for name, opened, maximum in components
    ]
    core._require(all(row["status"] == "sealed" for row in rows), "a component opened sealed 2025")
    frame = pd.DataFrame(rows)
    _write_table_formats(
        staging,
        "safety_2025_seal",
        frame,
        (
            "component",
            "sealed_2025_target_opened",
            "maximum_target_label_opened",
            "status",
        ),
    )
    core._write_json(
        staging / "receipts/safety_2025_sealed.json",
        {
            "status": "passed",
            "sealed_2025_target_opened": False,
            "final_headline_gate": FINAL_GATE,
            "compiler_opened_forecast_or_truth_arrays": False,
            "components": rows,
        },
    )
    return frame


def _reported_software(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = payload.get("software_versions", payload.get("software"))
    return value if isinstance(value, Mapping) else None


def _project_relative(path: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return None


def _source_record(path: Path, sha256: str, *, role: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "role": role,
        "path": str(resolved),
        "project_relative_path": _project_relative(resolved),
        "sha256": sha256,
    }


def _bound_upstream_files(ledger: core.HashLedger) -> list[dict[str, Any]]:
    return [
        {
            "label": ledger.labels[path],
            "path": str(path),
            "project_relative_path": _project_relative(path),
            "sha256": digest,
        }
        for path, digest in sorted(ledger.expected.items(), key=lambda item: str(item[0]))
    ]


def compile_release_bundle(
    core_manifest_path: Path,
    core_manifest_sha256: str,
    comparison_manifest_path: Path,
    comparison_manifest_sha256: str,
    imerg_manifest_path: Path,
    imerg_manifest_sha256: str,
    table_audit_receipt_path: Path,
    table_audit_receipt_sha256: str,
    prediction_audit_receipt_path: Path,
    prediction_audit_receipt_sha256: str,
    alignment_audit_receipt_path: Path,
    alignment_audit_receipt_sha256: str,
    methods_manifest_path: Path,
    methods_manifest_sha256: str,
    data_config_path: Path,
    data_config_sha256: str,
    output: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Verify all paper evidence and atomically publish one complete release."""

    destination = Path(output).resolve()
    _assert_release_output(destination, output_root)
    if destination.exists():
        raise FileExistsError(f"fresh paper-release output required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    ledger = core.HashLedger()
    try:
        core_manifest, core_count = _load_package(
            core_manifest_path,
            core_manifest_sha256,
            experiment=CORE_EXPERIMENT,
            status="complete",
            label="core bundle",
            ledger=ledger,
        )
        comparison, comparison_count = _load_package(
            comparison_manifest_path,
            comparison_manifest_sha256,
            experiment=COMPARISON_EXPERIMENT,
            status="complete",
            label="paper comparisons",
            ledger=ledger,
        )
        imerg, imerg_count = _load_package(
            imerg_manifest_path,
            imerg_manifest_sha256,
            experiment=IMERG_EXPERIMENT,
            status="complete",
            label="IMERG sensitivity",
            ledger=ledger,
        )
        table_audit, table_count = _load_package(
            table_audit_receipt_path,
            table_audit_receipt_sha256,
            experiment=TABLE_AUDIT_EXPERIMENT,
            status="passed",
            label="table audit",
            ledger=ledger,
        )
        prediction_audit, prediction_count = _load_package(
            prediction_audit_receipt_path,
            prediction_audit_receipt_sha256,
            experiment=PREDICTION_AUDIT_EXPERIMENT,
            status="passed",
            label="prediction audit",
            ledger=ledger,
        )
        alignment_audit, alignment_count = _load_package(
            alignment_audit_receipt_path,
            alignment_audit_receipt_sha256,
            experiment=ALIGNMENT_AUDIT_EXPERIMENT,
            status="passed",
            label="training-alignment audit",
            ledger=ledger,
        )
        resolved_methods_manifest = Path(methods_manifest_path).resolve()
        resolved_data_config = Path(data_config_path).resolve()
        ledger.verify(
            resolved_methods_manifest,
            methods_manifest_sha256,
            "benchmark methods manifest",
        )
        ledger.verify(
            resolved_data_config,
            data_config_sha256,
            "benchmark data configuration",
        )
        methods_manifest = core._read_json(resolved_methods_manifest)
        data_config = core._read_json(resolved_data_config)
        core._require(
            methods_manifest.get("data_config_sha256") == data_config_sha256,
            "methods-manifest/data-configuration SHA-256 differs",
        )

        core_gates = core_manifest.get("hard_gates", {})
        core._require(
            core_gates.get("sealed_2025_target_opened") is False
            and core_gates.get("final_headline_gate") == FINAL_GATE,
            "core bundle safety contract differs",
        )
        core_inputs = core_manifest.get("inputs", {})
        benchmark_path = _bind_declared_file(
            core_inputs,
            "benchmark_manifest",
            "benchmark_manifest_sha256",
            "transitive benchmark manifest",
            ledger,
        )
        scoring_path = _bind_declared_file(
            core_inputs,
            "scoring_manifest",
            "scoring_manifest_sha256",
            "transitive scoring manifest",
            ledger,
        )
        scoring = core._read_json(scoring_path)
        benchmark = core._read_json(benchmark_path)
        _same_binding(
            core_inputs,
            "audit_receipt",
            "audit_receipt_sha256",
            table_audit_receipt_path,
            table_audit_receipt_sha256,
            "core/table-audit binding",
        )
        _same_binding(
            table_audit,
            "scoring_manifest",
            "scoring_manifest_sha256",
            scoring_path,
            str(core_inputs["scoring_manifest_sha256"]),
            "table-audit/scoring binding",
        )
        comparison_inputs = comparison.get("inputs", {})
        _same_binding(
            comparison_inputs,
            "scoring_manifest",
            "scoring_manifest_sha256",
            scoring_path,
            str(core_inputs["scoring_manifest_sha256"]),
            "comparison/scoring binding",
        )
        _same_binding(
            comparison_inputs,
            "audit_receipt",
            "audit_receipt_sha256",
            table_audit_receipt_path,
            table_audit_receipt_sha256,
            "comparison/table-audit binding",
        )
        prediction_inputs = prediction_audit.get("inputs", {})
        _same_binding(
            prediction_inputs,
            "scoring_manifest",
            "scoring_manifest_sha256",
            scoring_path,
            str(core_inputs["scoring_manifest_sha256"]),
            "prediction-audit/scoring binding",
        )
        training_path = _bind_declared_file(
            prediction_inputs,
            "parent_manifest",
            "parent_manifest_sha256",
            "transitive training manifest",
            ledger,
        )
        inference_path = _bind_declared_file(
            prediction_inputs,
            "inference_manifest",
            "inference_manifest_sha256",
            "transitive inference manifest",
            ledger,
        )
        preflight_path = _bind_declared_file(
            prediction_inputs,
            "preflight_manifest",
            "preflight_manifest_sha256",
            "transitive preflight manifest",
            ledger,
        )
        alignment_parent = alignment_audit.get("inputs", {}).get("parent_training", {})
        _same_binding(
            alignment_parent,
            "manifest",
            "manifest_sha256",
            training_path,
            str(prediction_inputs["parent_manifest_sha256"]),
            "alignment-audit/training binding",
        )
        _same_binding(
            imerg,
            "parent_manifest",
            "parent_manifest_sha256",
            training_path,
            str(prediction_inputs["parent_manifest_sha256"]),
            "IMERG/training binding",
        )
        _same_binding(
            imerg,
            "input_inference_manifest",
            "input_inference_manifest_sha256",
            inference_path,
            str(prediction_inputs["inference_manifest_sha256"]),
            "IMERG/inference binding",
        )
        _same_binding(
            imerg,
            "input_preflight_manifest",
            "input_preflight_manifest_sha256",
            preflight_path,
            str(prediction_inputs["preflight_manifest_sha256"]),
            "IMERG/preflight binding",
        )
        training = core._read_json(training_path)

        row_counts = _copy_core_tables(core_manifest_path, core_manifest, staging)
        table1, table1_contract = _table1_system_inventory(
            resolved_methods_manifest,
            methods_manifest,
            resolved_data_config,
            data_config,
            staging,
        )
        row_counts["table1_system_inventory"] = len(table1)
        table2a, table2a_contract = _table2a_deterministic_benchmark(
            benchmark_path,
            benchmark,
            staging,
            ledger,
        )
        row_counts["table2a_deterministic_benchmark"] = len(table2a)
        table2b, table2b_contract = _table2b_retrospective_calibrated_fuxi(
            scoring_path, scoring, staging, ledger
        )
        row_counts["table2b_retrospective_calibrated_fuxi"] = len(table2b)
        comparison_counts, comparison_findings = _comparison_tables(
            comparison_manifest_path, comparison, staging
        )
        row_counts.update(comparison_counts)
        imerg_gate, imerg_decisions = _imerg_gate_table(imerg_manifest_path, imerg, staging)
        row_counts["unfitted_imerg_gate"] = len(imerg_gate)
        audit_summary, alignment_summary = _audit_tables(
            table_audit, prediction_audit, alignment_audit, staging
        )
        row_counts["audit_summary"] = len(audit_summary)
        row_counts["training_alignment_audit"] = len(alignment_summary)
        safety = _safety_receipt(
            core_manifest,
            scoring,
            table_audit,
            prediction_audit,
            comparison,
            imerg,
            alignment_audit,
            staging,
        )
        row_counts["safety_2025_seal"] = len(safety)

        receipt_sources = {
            "table_audit_receipt.json": Path(table_audit_receipt_path),
            "prediction_audit_receipt.json": Path(prediction_audit_receipt_path),
            "training_alignment_audit_receipt.json": Path(alignment_audit_receipt_path),
        }
        for name, source in receipt_sources.items():
            _copy_bound_file(source, staging / "receipts" / name)

        source_paths = (Path(__file__).resolve(), Path(core.__file__).resolve())
        for source in source_paths:
            _copy_bound_file(source, staging / "code/src" / source.name)

        ledger.reverify()
        artifacts = {
            str(path.relative_to(staging)): core.sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "failure.json"}
        }
        reported_software = {
            "benchmark": _reported_software(benchmark),
            "training": _reported_software(training),
            "comparison": _reported_software(comparison),
            "imerg_sensitivity": _reported_software(imerg),
        }
        missing_software = sorted(
            label
            for label, payload in {
                "core_bundle": core_manifest,
                "scoring": scoring,
                "table_audit": table_audit,
                "prediction_audit": prediction_audit,
                "training_alignment_audit": alignment_audit,
            }.items()
            if _reported_software(payload) is None
        )
        core_root = Path(core_manifest_path).resolve().parent
        comparison_root = Path(comparison_manifest_path).resolve().parent
        imerg_root = Path(imerg_manifest_path).resolve().parent
        imported_files: dict[str, dict[str, Any]] = {}
        for table_name in (
            "main_deterministic_benchmark",
            "calibrated_fuxi_by_lead",
            "story_gate",
        ):
            for extension in ("csv", "md", "tex"):
                relative = f"tables/{table_name}.{extension}"
                imported_files[relative] = _source_record(
                    core_root / relative,
                    str(core_manifest["artifact_sha256"][relative]),
                    role="byte-exact copy from immutable core bundle",
                )
        for receipt_name, source in receipt_sources.items():
            imported_files[f"receipts/{receipt_name}"] = _source_record(
                source,
                ledger.expected[Path(source).resolve()],
                role="byte-exact copy of independent audit receipt",
            )
        exact_source = comparison_root / "tables/exact_common_midpoint_sensitivity.csv"
        mme_source = comparison_root / "tables/calibrated_fuxi_vs_mme.csv"
        imerg_interval_source = imerg_root / "tables/paired_block_intervals.csv"
        scoring_interval_source = scoring_path.parent / "tables/paired_block_intervals.csv"
        benchmark_interval_source = benchmark_path.parent / "mme_vs_fuxi_paired_intervals.csv"
        derived_evidence = {
            "tables/table1_system_inventory.{csv,md,tex}": [
                _source_record(
                    resolved_methods_manifest,
                    methods_manifest_sha256,
                    role="system/archive/method inventory source",
                ),
                _source_record(
                    resolved_data_config,
                    data_config_sha256,
                    role="verification-year and known-exclusion source",
                ),
            ],
            "receipts/table1_contract.json": [
                _source_record(
                    resolved_methods_manifest,
                    methods_manifest_sha256,
                    role="system/archive/method inventory source",
                ),
                _source_record(
                    resolved_data_config,
                    data_config_sha256,
                    role="verification-year and known-exclusion source",
                ),
            ],
            "tables/table2a_deterministic_benchmark.{csv,md,tex}": [
                _source_record(
                    core_root / "tables/main_deterministic_benchmark.csv",
                    str(
                        core_manifest["artifact_sha256"][
                            "tables/main_deterministic_benchmark.csv"
                        ]
                    ),
                    role="all-system point estimates and case-count source",
                ),
                _source_record(
                    benchmark_interval_source,
                    str(benchmark["artifact_sha256"]["mme_vs_fuxi_paired_intervals.csv"]),
                    role="pointwise MME-minus-FuXi paired-interval source",
                ),
                _source_record(
                    benchmark_path,
                    str(core_inputs["benchmark_manifest_sha256"]),
                    role="benchmark interval artifact manifest",
                ),
            ],
            "receipts/table2a_contract.json": [
                _source_record(
                    core_root / "tables/main_deterministic_benchmark.csv",
                    str(
                        core_manifest["artifact_sha256"][
                            "tables/main_deterministic_benchmark.csv"
                        ]
                    ),
                    role="all-system point estimates and case-count source",
                ),
                _source_record(
                    benchmark_interval_source,
                    str(benchmark["artifact_sha256"]["mme_vs_fuxi_paired_intervals.csv"]),
                    role="pointwise MME-minus-FuXi paired-interval source",
                ),
            ],
            "tables/table2b_retrospective_calibrated_fuxi.{csv,md,tex}": [
                _source_record(
                    core_root / "tables/calibrated_fuxi_by_lead.csv",
                    str(
                        core_manifest["artifact_sha256"][
                            "tables/calibrated_fuxi_by_lead.csv"
                        ]
                    ),
                    role="leadwise score source",
                ),
                _source_record(
                    scoring_interval_source,
                    str(scoring["artifact_sha256"]["tables/paired_block_intervals.csv"]),
                    role="leadwise paired-interval source",
                ),
                _source_record(
                    scoring_path,
                    str(core_inputs["scoring_manifest_sha256"]),
                    role="member-count and checkpoint provenance",
                ),
            ],
            "receipts/table2b_contract.json": [
                _source_record(
                    core_root / "tables/calibrated_fuxi_by_lead.csv",
                    str(
                        core_manifest["artifact_sha256"][
                            "tables/calibrated_fuxi_by_lead.csv"
                        ]
                    ),
                    role="leadwise score source",
                ),
                _source_record(
                    scoring_interval_source,
                    str(scoring["artifact_sha256"]["tables/paired_block_intervals.csv"]),
                    role="leadwise paired-interval source",
                ),
                _source_record(
                    core_root / "tables/story_gate.csv",
                    str(core_manifest["artifact_sha256"]["tables/story_gate.csv"]),
                    role="separate pooled-gate source",
                ),
            ],
            "tables/exact_common_sensitivity.{csv,md,tex}": [
                _source_record(
                    exact_source,
                    str(
                        comparison["artifact_sha256"][
                            "tables/exact_common_midpoint_sensitivity.csv"
                        ]
                    ),
                    role="source table",
                )
            ],
            "tables/calibrated_fuxi_vs_fixed_mme.{csv,md,tex}": [
                _source_record(
                    mme_source,
                    str(comparison["artifact_sha256"]["tables/calibrated_fuxi_vs_mme.csv"]),
                    role="source table",
                )
            ],
            "tables/unfitted_imerg_gate.{csv,md,tex}": [
                _source_record(
                    imerg_interval_source,
                    str(imerg["artifact_sha256"]["tables/paired_block_intervals.csv"]),
                    role="source interval table",
                )
            ],
            "tables/audit_summary.{csv,md,tex}": [
                _source_record(
                    Path(table_audit_receipt_path),
                    table_audit_receipt_sha256,
                    role="table-level audit",
                ),
                _source_record(
                    Path(prediction_audit_receipt_path),
                    prediction_audit_receipt_sha256,
                    role="prediction-level audit",
                ),
                _source_record(
                    Path(alignment_audit_receipt_path),
                    alignment_audit_receipt_sha256,
                    role="training-alignment audit",
                ),
            ],
            "tables/training_alignment_audit.{csv,md,tex}": [
                _source_record(
                    Path(alignment_audit_receipt_path),
                    alignment_audit_receipt_sha256,
                    role="training-alignment results and limitation",
                )
            ],
            "receipts/training_alignment_limitation.{json,md}": [
                _source_record(
                    Path(alignment_audit_receipt_path),
                    alignment_audit_receipt_sha256,
                    role="historical target-tensor equality limitation",
                )
            ],
            "tables/safety_2025_seal.{csv,md,tex}": [
                _source_record(Path(core_manifest_path), core_manifest_sha256, role="core seal"),
                _source_record(scoring_path, str(core_inputs["scoring_manifest_sha256"]), role="IMD scoring seal"),
                _source_record(Path(table_audit_receipt_path), table_audit_receipt_sha256, role="table-audit seal"),
                _source_record(Path(prediction_audit_receipt_path), prediction_audit_receipt_sha256, role="prediction-audit seal"),
                _source_record(Path(comparison_manifest_path), comparison_manifest_sha256, role="comparison seal"),
                _source_record(Path(imerg_manifest_path), imerg_manifest_sha256, role="IMERG seal"),
                _source_record(Path(alignment_audit_receipt_path), alignment_audit_receipt_sha256, role="training-alignment seal"),
            ],
            "receipts/safety_2025_sealed.json": [
                _source_record(Path(core_manifest_path), core_manifest_sha256, role="core seal"),
                _source_record(scoring_path, str(core_inputs["scoring_manifest_sha256"]), role="IMD scoring seal"),
                _source_record(Path(table_audit_receipt_path), table_audit_receipt_sha256, role="table-audit seal"),
                _source_record(Path(prediction_audit_receipt_path), prediction_audit_receipt_sha256, role="prediction-audit seal"),
                _source_record(Path(comparison_manifest_path), comparison_manifest_sha256, role="comparison seal"),
                _source_record(Path(imerg_manifest_path), imerg_manifest_sha256, role="IMERG seal"),
                _source_record(Path(alignment_audit_receipt_path), alignment_audit_receipt_sha256, role="training-alignment seal"),
            ],
        }
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(destination),
            "scientific_status": (
                "complete retrospective 2020-2024 paper evidence release; "
                "final headline gate pending sealed 2025 check"
            ),
            "supersedes": {
                "core_bundle": str(Path(core_manifest_path).resolve()),
                "core_bundle_sha256": core_manifest_sha256,
                "scope": "paper release packaging; immutable upstream core remains valid",
            },
            "direct_inputs": {
                "core_bundle": _source_record(Path(core_manifest_path), core_manifest_sha256, role="direct input"),
                "paper_comparisons": _source_record(Path(comparison_manifest_path), comparison_manifest_sha256, role="direct input"),
                "imerg_sensitivity": _source_record(Path(imerg_manifest_path), imerg_manifest_sha256, role="direct input"),
                "table_audit": _source_record(Path(table_audit_receipt_path), table_audit_receipt_sha256, role="direct input"),
                "prediction_audit": _source_record(Path(prediction_audit_receipt_path), prediction_audit_receipt_sha256, role="direct input"),
                "training_alignment_audit": _source_record(Path(alignment_audit_receipt_path), alignment_audit_receipt_sha256, role="direct input"),
                "benchmark_methods_manifest": _source_record(resolved_methods_manifest, methods_manifest_sha256, role="direct input"),
                "benchmark_data_configuration": _source_record(resolved_data_config, data_config_sha256, role="direct input"),
            },
            "transitive_manifests": {
                "benchmark": _source_record(benchmark_path, str(core_inputs["benchmark_manifest_sha256"]), role="transitive manifest"),
                "scoring": _source_record(scoring_path, str(core_inputs["scoring_manifest_sha256"]), role="transitive manifest"),
                "training": _source_record(training_path, str(prediction_inputs["parent_manifest_sha256"]), role="transitive manifest"),
                "inference": _source_record(inference_path, str(prediction_inputs["inference_manifest_sha256"]), role="transitive manifest"),
                "preflight": _source_record(preflight_path, str(prediction_inputs["preflight_manifest_sha256"]), role="transitive manifest"),
            },
            "evidence_provenance": {
                "byte_exact_imports": imported_files,
                "derived_outputs": derived_evidence,
            },
            "bound_upstream_files": _bound_upstream_files(ledger),
            "software": {
                "compiler": {
                    "python": sys.version,
                    "python_implementation": platform.python_implementation(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "upstream_reported": reported_software,
                "upstream_receipts_without_software_fields": missing_software,
                "limitation": (
                    "Missing entries mean the immutable upstream receipt did not persist "
                    "software versions; this compiler does not infer historical environments."
                ),
            },
            "hard_gates": {
                "all_direct_input_sha256_verified": True,
                "all_declared_artifact_inventories_verified": True,
                "all_input_hashes_reverified_before_publication": True,
                "validated_direct_artifact_counts": {
                    "core_bundle": core_count,
                    "paper_comparisons": comparison_count,
                    "imerg_sensitivity": imerg_count,
                    "table_audit": table_count,
                    "prediction_audit": prediction_count,
                    "training_alignment_audit": alignment_count,
                },
                "validated_configuration_inputs": {
                    "benchmark_methods_manifest_sha256": methods_manifest_sha256,
                    "benchmark_data_configuration_sha256": data_config_sha256,
                    "methods_manifest_binds_data_configuration": True,
                },
                "table_level_audit": "passed",
                "prediction_level_audit": "passed",
                "training_alignment_audit": "passed_with_recorded_historical_tensor_limitation",
                "historical_training_target_tensor_equality_provable": False,
                "retrospective_imd_story_gate": "passed",
                "unfitted_imerg_gate": imerg_decisions,
                "comparison_findings": comparison_findings,
                "table1_contract": table1_contract,
                "table2a_contract": table2a_contract,
                "table2b_contract": table2b_contract,
                "final_headline_gate": FINAL_GATE,
                "sealed_2025_target_opened": False,
                "compiler_opened_forecast_checkpoint_or_truth_arrays": False,
            },
            "table_rows": row_counts,
            "formats": ["csv", "markdown", "latex", "json"],
            "source_snapshot_sha256": {
                f"code/src/{source.name}": artifacts[f"code/src/{source.name}"]
                for source in source_paths
            },
            "artifact_sha256": artifacts,
        }
        core._write_json(staging / "manifest.json", manifest)
        ledger.reverify()
        core.rename_noreplace(staging, destination)
    except BaseException as error:
        if staging.exists():
            core._write_json(
                staging / "failure.json",
                {
                    "experiment": EXPERIMENT,
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "compiler_opened_forecast_checkpoint_or_truth_arrays": False,
                    "sealed_2025_target_opened": False,
                },
            )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--core-manifest-sha256", required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--comparison-manifest-sha256", required=True)
    parser.add_argument("--imerg-manifest", type=Path, required=True)
    parser.add_argument("--imerg-manifest-sha256", required=True)
    parser.add_argument("--table-audit-receipt", type=Path, required=True)
    parser.add_argument("--table-audit-receipt-sha256", required=True)
    parser.add_argument("--prediction-audit-receipt", type=Path, required=True)
    parser.add_argument("--prediction-audit-receipt-sha256", required=True)
    parser.add_argument("--alignment-audit-receipt", type=Path, required=True)
    parser.add_argument("--alignment-audit-receipt-sha256", required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--methods-manifest-sha256", required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--data-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    manifest = compile_release_bundle(
        arguments.core_manifest,
        arguments.core_manifest_sha256,
        arguments.comparison_manifest,
        arguments.comparison_manifest_sha256,
        arguments.imerg_manifest,
        arguments.imerg_manifest_sha256,
        arguments.table_audit_receipt,
        arguments.table_audit_receipt_sha256,
        arguments.prediction_audit_receipt,
        arguments.prediction_audit_receipt_sha256,
        arguments.alignment_audit_receipt,
        arguments.alignment_audit_receipt_sha256,
        arguments.methods_manifest,
        arguments.methods_manifest_sha256,
        arguments.data_config,
        arguments.data_config_sha256,
        arguments.output,
    )
    print(f"PASS: comprehensive paper release published at {manifest}")


if __name__ == "__main__":
    main()
