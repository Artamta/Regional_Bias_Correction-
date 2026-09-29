#!/usr/bin/env python3
"""Build a provenance-bound paper supplement for the operational categorical audit.

This is a derived reporting step.  It reads only receipted CSV/JSON artifacts
from the one accepted full run, never opens forecast or target stores, and
atomically installs a fresh bundle under ``presentation/deliverables``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


BUNDLE_EXPERIMENT = "fuxi_allseason_operational_categorical_paper_bundle_v1"
BUNDLE_CONTRACT = "operational_categorical_paper_bundle_contract_v1"
SOURCE_EXPERIMENT = "fuxi_allseason_operational_categorical_comparison_v1"
SOURCE_CONTRACT = "operational_categorical_contract_v1"
SCORING_CONTRACT = "normalized_informative_positive_cut_dynamic_v1"
CANONICAL_SOURCE_DIR = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_operational_categorical_comparison"
    / "full_20260822T203000Z"
)
CANONICAL_MANIFEST_SHA256 = (
    "cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d"
)
CANONICAL_SEMANTIC_SHA256 = (
    "00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c"
)
CANONICAL_GATE_SHA256 = (
    "cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c"
)
EXPECTED_CASES = 296
EXPECTED_BOOTSTRAP_ROWS = 252
EXPECTED_BOOTSTRAP_DRAWS = 10_000
EXPECTED_YEARS = (2022, 2023, 2024)
EXPECTED_YEAR_COUNTS = {"2022": 104, "2023": 104, "2024": 88}
EXPECTED_METHODS = (
    "raw_fuxi_categorical",
    "base_42k",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
)
PAPER_METHODS = (
    "raw_fuxi_categorical",
    "base_42k",
    "persistence_plus_plus",
    "pbc_combined",
)
METHOD_LABELS = {
    "raw_fuxi_categorical": "Raw FuXi",
    "base_42k": "Neural adapter",
    "debias_plus_plus": "Debias++",
    "persistence_plus_plus": "Persistence++",
    "pbc_combined": "Combined PBC",
}
FAMILY_METRICS = {
    "quintile": "quintile_effective_positive_cut_rps",
    "semidecile": "semidecile_effective_positive_cut_rps",
}
EXPECTED_DECLARED_ARTIFACTS = frozenset(
    {
        "README.md",
        "code/plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md",
        "code/slurm/evaluate_allseason_operational_categorical_comparison.sbatch",
        "code/src/fuxi_allseason_capacity_ablation.py",
        "code/src/fuxi_allseason_capacity_development_evaluation.py",
        "code/src/fuxi_allseason_ensemble_calibration.py",
        "code/src/fuxi_allseason_member_cache.py",
        "code/src/fuxi_allseason_operational_categorical_comparison.py",
        "code/src/fuxi_allseason_operational_era_audit.py",
        "code/src/fuxi_ensemble_calibration_core.py",
        "code/src/fuxi_pbc_core.py",
        "code/tests/test_allseason_operational_categorical_comparison.py",
        "evaluation/bootstrap_design.json",
        "evaluation/late_2021_lag_only_receipt.json",
        "evaluation/persistence_lag_provenance.json",
        "evaluation/persistence_lags.npz",
        "evaluation/scoring_support.npz",
        "metrics/neural_seed_probability_bias.csv",
        "metrics/neural_seed_quintile_case_scores.csv",
        "metrics/neural_seed_semidecile_case_scores.csv",
        "metrics/neural_seed_upper_case_scores.csv",
        "metrics/paired_two_stage_bootstrap.csv",
        "metrics/pooled_rps.csv",
        "metrics/probability_bias.csv",
        "metrics/quintile_case_scores.csv",
        "metrics/seasonal_weekwise_rps.csv",
        "metrics/semidecile_case_scores.csv",
        "metrics/upper_q95_case_scores.csv",
        "metrics/upper_q95_summary.csv",
        "metrics/weekwise_rps.csv",
        "receipts/input_receipts.json",
    }
)
EXPECTED_PHYSICAL_EXTRAS = frozenset(
    {"manifest.json", "evaluation/postflight_semantic_audit.json", "slurm_gate_receipt.json"}
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
YEAR_TOKEN = re.compile(r"(?:^|[^0-9])2025(?:[^0-9]|$)")


class PaperBundleError(RuntimeError):
    """Raised when source evidence or derived output violates the contract."""


@dataclass(frozen=True)
class VerifiedSource:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    semantic: Mapping[str, Any]
    semantic_sha256: str
    gate: Mapping[str, Any]
    gate_sha256: str
    artifact_sha256: Mapping[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperBundleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperBundleError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.temporary")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe_source_artifact(root: Path, relative: str) -> Path:
    item = Path(relative)
    _require(
        bool(relative) and not item.is_absolute() and ".." not in item.parts,
        f"unsafe source artifact path: {relative!r}",
    )
    _require(
        YEAR_TOKEN.search(relative) is None,
        f"sealed-year token in source artifact path: {relative}",
    )
    path = root / item
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise PaperBundleError(f"source artifact escapes run: {relative}") from error
    _require(not path.is_symlink(), f"source artifact is a symlink: {relative}")
    return path


def _physical_inventory(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }


def _opened_store_targets_2025(value: str) -> bool:
    name = Path(value).name.lower()
    return name == "2025" or name.startswith("2025-") or (
        Path(name).suffix in {".zarr", ".nc", ".nc4", ".grib", ".grib2"}
        and YEAR_TOKEN.search(name) is not None
    )


def _verify_no_2025(manifest: Mapping[str, Any], semantic: Mapping[str, Any]) -> None:
    contract = manifest.get("contract")
    _require(isinstance(contract, Mapping), "source contract is absent")
    for key in ("sealed_2025_target_opened", "final_2025_store_opened"):
        _require(contract.get(key) is False, f"source does not prove {key}=false")
    for key in ("forecast_year_store_whitelist", "verification_year_store_whitelist"):
        _require(
            tuple(contract.get(key, ())) == EXPECTED_YEARS,
            f"unexpected {key}: {contract.get(key)!r}",
        )
    _require(
        str(contract.get("maximum_verification_date", "")).startswith("2024-"),
        "maximum verification date is not in 2024",
    )
    _require(semantic.get("no_2025_year_store_component") is True, "semantic no-2025 gate failed")
    cases = manifest.get("cases", {})
    _require(cases.get("evaluated_counts") == EXPECTED_YEAR_COUNTS, "year counts differ")
    initializations = cases.get("evaluated_initializations", [])
    _require(len(initializations) == EXPECTED_CASES, "initialization count differs")
    years = {int(str(value)[:4]) for value in initializations}
    _require(years == set(EXPECTED_YEARS), f"unexpected initialization years: {years}")
    for key in ("opened_input_paths_exact", "opened_store_paths_exact"):
        paths = manifest.get(key)
        _require(isinstance(paths, list) and paths, f"{key} is absent")
        _require(
            not any(_opened_store_targets_2025(str(path)) for path in paths),
            f"{key} contains an opened 2025 store",
        )


def verify_source(source: Path, *, expected_manifest_sha256: str) -> VerifiedSource:
    """Verify the exact accepted full source without opening scientific arrays."""

    _require(
        HEX64.fullmatch(expected_manifest_sha256) is not None,
        "expected manifest SHA-256 must be 64 lowercase hex characters",
    )
    _require(
        expected_manifest_sha256 == CANONICAL_MANIFEST_SHA256,
        "expected manifest hash is not the frozen accepted hash",
    )
    root = Path(source).resolve(strict=True)
    canonical = Path(CANONICAL_SOURCE_DIR).resolve(strict=True)
    _require(root == canonical, f"source is not the frozen accepted full run: {root}")
    _require(root.is_dir(), f"source is not a directory: {root}")
    _require(
        "smoke" not in root.name.lower() and "incomplete" not in root.name.lower(),
        "smoke/incomplete source is forbidden",
    )
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest is absent or linked")
    manifest_sha = sha256_file(manifest_path)
    _require(manifest_sha == expected_manifest_sha256, f"manifest hash differs: {manifest_sha}")
    manifest = _read_json(manifest_path)
    for key, expected in {
        "experiment": SOURCE_EXPERIMENT,
        "audit_contract_version": SOURCE_CONTRACT,
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "canonical": True,
        "scientific_eligible": True,
        "scoring_contract_version": SCORING_CONTRACT,
    }.items():
        _require(manifest.get(key) == expected, f"source field {key!r} differs")
    _require(
        manifest.get("methods") == list(EXPECTED_METHODS),
        "source method inventory differs",
    )
    _require(
        manifest.get("seeds") == [42, 43, 44],
        "source seed inventory differs",
    )
    seed_handling = manifest.get("seed_handling", {})
    _require(
        seed_handling.get("headline_aggregation")
        == "arithmetic mean of per-seed scores on identical case/lead rows"
        and seed_handling.get("probabilities_averaged_across_seeds") is False,
        "source seed-score aggregation contract differs",
    )

    artifacts = manifest.get("artifact_sha256")
    _require(isinstance(artifacts, Mapping), "source artifact inventory is absent")
    _require(set(artifacts) == set(EXPECTED_DECLARED_ARTIFACTS), "declared source inventory differs")
    checked: dict[str, str] = {}
    for relative, expected in sorted(artifacts.items()):
        _require(
            isinstance(relative, str)
            and isinstance(expected, str)
            and HEX64.fullmatch(expected) is not None,
            f"malformed artifact receipt: {relative!r}",
        )
        path = _safe_source_artifact(root, relative)
        _require(path.is_file(), f"missing source artifact: {relative}")
        actual = sha256_file(path)
        _require(actual == expected, f"source artifact hash differs: {relative}")
        checked[relative] = actual
    _require(
        _physical_inventory(root) == set(EXPECTED_DECLARED_ARTIFACTS) | set(EXPECTED_PHYSICAL_EXTRAS),
        "physical source inventory differs from declared artifacts plus gates",
    )

    semantic_path = root / "evaluation/postflight_semantic_audit.json"
    semantic_sha = sha256_file(semantic_path)
    _require(semantic_sha == CANONICAL_SEMANTIC_SHA256, "semantic audit hash differs")
    semantic = _read_json(semantic_path)
    for key, expected in {
        "experiment": SOURCE_EXPERIMENT,
        "post_run_audit_version": "operational_categorical_postrun_v1",
        "status": "passed",
        "mode": "full",
        "manifest_sha256": manifest_sha,
        "case_count": EXPECTED_CASES,
        "bootstrap_rows_recomputed": EXPECTED_BOOTSTRAP_ROWS,
        "bootstrap_draws_recomputed": EXPECTED_BOOTSTRAP_DRAWS,
        "aggregate_tables_recomputed": True,
        "headline_neural_score_average_recomputed": True,
        "smoke_full_exact_source_snapshot_identity": True,
        "lag_receipts_rechecked": True,
    }.items():
        _require(semantic.get(key) == expected, f"semantic field {key!r} differs")

    gate_path = root / "slurm_gate_receipt.json"
    gate_sha = sha256_file(gate_path)
    _require(gate_sha == CANONICAL_GATE_SHA256, "Slurm gate receipt hash differs")
    gate = _read_json(gate_path)
    for key, expected in {
        "experiment": SOURCE_EXPERIMENT,
        "audit_contract_version": SOURCE_CONTRACT,
        "post_run_audit_version": "operational_categorical_postrun_v1",
        "gate_status": "passed",
        "mode": "full",
        "manifest_sha256": manifest_sha,
        "postflight_semantic_audit_sha256": semantic_sha,
    }.items():
        _require(gate.get(key) == expected, f"Slurm gate field {key!r} differs")
    for key in ("job_id", "node", "partition", "completed_utc"):
        _require(isinstance(gate.get(key), str) and gate[key].strip(), f"Slurm gate lacks {key}")
    _require(
        gate.get("source_snapshot_sha256") == manifest.get("source_snapshot_sha256")
        == semantic.get("source_snapshot_sha256"),
        "manifest/semantic/gate source snapshots differ",
    )
    _verify_no_2025(manifest, semantic)
    return VerifiedSource(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        semantic=semantic,
        semantic_sha256=semantic_sha,
        gate=gate,
        gate_sha256=gate_sha,
        artifact_sha256=checked,
    )


def _verified_csv(source: VerifiedSource, relative: str, required: Sequence[str]) -> pd.DataFrame:
    _require(relative in source.artifact_sha256, f"unreceipted CSV requested: {relative}")
    frame = pd.read_csv(_safe_source_artifact(source.root, relative))
    _require(not frame.empty, f"empty CSV: {relative}")
    missing = set(required) - set(frame.columns)
    _require(not missing, f"{relative} lacks columns {sorted(missing)}")
    _require(not any(str(column).startswith("Unnamed:") for column in frame.columns), f"unnamed column in {relative}")
    for column in ("year", "initialization", "verification_midpoint"):
        if column in frame:
            values = frame[column].dropna().astype(str)
            _require(not values.str.startswith("2025").any(), f"sealed-year row in {relative}:{column}")
    return frame


def _finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy()
    _require(np.isfinite(values).all(), f"non-finite values in {label}")


def build_tables(source: VerifiedSource) -> dict[str, pd.DataFrame]:
    pooled = _verified_csv(
        source,
        "metrics/pooled_rps.csv",
        ("family", "method", "method_label", "score_contract", "rps", "rpss_vs_training_empirical_climatology", "n_initializations"),
    )
    weekwise = _verified_csv(
        source,
        "metrics/weekwise_rps.csv",
        ("family", "lead_week", "method", "method_label", "score_contract", "rps", "rpss_vs_training_empirical_climatology", "n_initializations"),
    )
    bootstrap = _verified_csv(
        source,
        "metrics/paired_two_stage_bootstrap.csv",
        ("score_contract", "metric", "lead_scope", "lead_week", "method", "baseline", "method_score", "baseline_score", "score_reduction_fraction", "ci_lower_95", "ci_upper_95", "bootstrap_probability_improvement", "bootstrap_samples", "block_length_initializations", "bootstrap_seed", "bootstrap_scheme", "source_year_clusters", "small_year_cluster_limitation"),
    )
    _require(len(bootstrap) == EXPECTED_BOOTSTRAP_ROWS, "bootstrap row count differs from semantic receipt")
    _require(set(pooled.family) == set(FAMILY_METRICS), "pooled families differ")
    _require(set(weekwise.family) == set(FAMILY_METRICS), "weekwise families differ")
    for frame, label, expected_rows in ((pooled, "pooled", 10), (weekwise, "weekwise", 60)):
        _require(len(frame) == expected_rows, f"{label} row count differs")
        _require(set(frame.method) == set(EXPECTED_METHODS), f"{label} methods differ")
        _require(set(frame.score_contract) == {SCORING_CONTRACT}, f"{label} scoring contract differs")
        _require(set(frame.n_initializations.astype(int)) == {EXPECTED_CASES}, f"{label} case count differs")
    _require(set(weekwise.lead_week.astype(int)) == set(range(1, 7)), "weekwise leads differ")
    _require(not pooled.duplicated(["family", "method"]).any(), "duplicate pooled rows")
    _require(not weekwise.duplicated(["family", "lead_week", "method"]).any(), "duplicate weekwise rows")
    _finite(pooled, ("rps", "rpss_vs_training_empirical_climatology"), "pooled RPS")
    _finite(weekwise, ("rps", "rpss_vs_training_empirical_climatology"), "weekwise RPS")
    _finite(bootstrap, ("method_score", "baseline_score", "score_reduction_fraction", "ci_lower_95", "ci_upper_95"), "bootstrap")

    output_pooled = pooled.loc[pooled.method.isin(PAPER_METHODS)].copy()
    output_pooled["method_short_label"] = output_pooled.method.map(METHOD_LABELS)
    output_pooled["evidence_period"] = "2022-2024_retrospective"
    output_pooled["source_year_clusters"] = 3
    output_pooled = output_pooled[
        ["evidence_period", "family", "method", "method_short_label", "rps", "rpss_vs_training_empirical_climatology", "n_initializations", "source_year_clusters", "score_contract"]
    ].sort_values(["family", "rps"])

    by_lead: dict[str, pd.DataFrame] = {}
    contrasts: dict[str, pd.DataFrame] = {}
    score_lookup = {
        (str(row.family), int(row.lead_week), str(row.method)): float(row.rps)
        for row in weekwise.itertuples(index=False)
    }
    score_lookup.update(
        {
            (str(row.family), 0, str(row.method)): float(row.rps)
            for row in pooled.itertuples(index=False)
        }
    )
    for family, metric in FAMILY_METRICS.items():
        lead = weekwise.loc[weekwise.family.eq(family) & weekwise.method.isin(PAPER_METHODS)].copy()
        _require(len(lead) == 24, f"{family} paper lead table is incomplete")
        lead["method_short_label"] = lead.method.map(METHOD_LABELS)
        lead["rank_within_lead"] = lead.groupby("lead_week").rps.rank(method="min").astype(int)
        by_lead[family] = lead[
            ["family", "lead_week", "method", "method_short_label", "rps", "rpss_vs_training_empirical_climatology", "rank_within_lead", "n_initializations", "score_contract"]
        ].sort_values(["lead_week", "rank_within_lead", "method"])

        selected = bootstrap.loc[
            bootstrap.metric.eq(metric)
            & bootstrap.method.eq("base_42k")
            & bootstrap.baseline.isin(("raw_fuxi_categorical", "persistence_plus_plus", "pbc_combined"))
        ].copy()
        _require(len(selected) == 21, f"{family} neural contrast rows are incomplete")
        _require(set(selected.lead_week.astype(int)) == set(range(0, 7)), f"{family} contrast leads differ")
        _require(not selected.duplicated(["lead_week", "baseline"]).any(), f"duplicate {family} contrast")
        for row in selected.itertuples(index=False):
            lead_week = int(row.lead_week)
            _require(
                math.isclose(float(row.method_score), score_lookup[(family, lead_week, "base_42k")], rel_tol=1e-10, abs_tol=1e-10),
                f"{family} neural bootstrap score differs from aggregate",
            )
            _require(
                math.isclose(float(row.baseline_score), score_lookup[(family, lead_week, str(row.baseline))], rel_tol=1e-10, abs_tol=1e-10),
                f"{family} baseline bootstrap score differs from aggregate",
            )
            expected_reduction = 1.0 - float(row.method_score) / float(row.baseline_score)
            _require(math.isclose(float(row.score_reduction_fraction), expected_reduction, rel_tol=1e-10, abs_tol=1e-10), f"{family} score-reduction identity differs")
        _require((selected.ci_lower_95 <= selected.ci_upper_95).all(), f"{family} intervals are unordered")
        _require(set(selected.bootstrap_samples.astype(int)) == {EXPECTED_BOOTSTRAP_DRAWS}, f"{family} bootstrap draws differ")
        _require(set(selected.block_length_initializations.astype(int)) == {13}, f"{family} block length differs")
        _require(set(selected.bootstrap_seed.astype(int)) == {20260823}, f"{family} bootstrap seed differs")
        _require(set(selected.source_year_clusters.astype(int)) == {3}, f"{family} year clusters differ")
        _require(selected.small_year_cluster_limitation.astype(bool).all(), f"{family} cluster limitation missing")
        selected["baseline_short_label"] = selected.baseline.map(METHOD_LABELS)
        selected["score_reduction_pct"] = 100.0 * selected.score_reduction_fraction
        selected["ci_lower_95_pct"] = 100.0 * selected.ci_lower_95
        selected["ci_upper_95_pct"] = 100.0 * selected.ci_upper_95
        selected["inference"] = np.where(
            selected.ci_lower_95.gt(0),
            "neural_better_95pct_interval",
            np.where(selected.ci_upper_95.lt(0), "baseline_better_95pct_interval", "unresolved_95pct_interval"),
        )
        contrasts[family] = selected[
            ["metric", "lead_scope", "lead_week", "method", "baseline", "baseline_short_label", "method_score", "baseline_score", "score_reduction_pct", "ci_lower_95_pct", "ci_upper_95_pct", "bootstrap_probability_improvement", "inference", "bootstrap_samples", "block_length_initializations", "bootstrap_seed", "bootstrap_scheme", "source_year_clusters", "small_year_cluster_limitation"]
        ].sort_values(["lead_week", "baseline"])

    return {
        "primary_pooled_rps": output_pooled.reset_index(drop=True),
        "quintile_rps_by_lead": by_lead["quintile"].reset_index(drop=True),
        "semidecile_rps_by_lead": by_lead["semidecile"].reset_index(drop=True),
        "quintile_neural_contrasts": contrasts["quintile"].reset_index(drop=True),
        "semidecile_neural_contrasts": contrasts["semidecile"].reset_index(drop=True),
    }


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def plot_transfer_story(tables: Mapping[str, pd.DataFrame], stem: Path) -> None:
    """Render the quintile lead curves and paired neural contrasts."""

    _paper_style()
    lead = tables["quintile_rps_by_lead"]
    contrast = tables["quintile_neural_contrasts"]
    styles = {
        "raw_fuxi_categorical": ("#666666", "--", "o", 1.5),
        "base_42k": ("#0072B2", "-", "o", 2.3),
        "persistence_plus_plus": ("#D55E00", "-", "s", 1.9),
        "pbc_combined": ("#009E73", "-", "D", 1.9),
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.55), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.0]})
    for method in PAPER_METHODS:
        selected = lead.loc[lead.method.eq(method)].sort_values("lead_week")
        _require(len(selected) == 6, f"figure lacks six leads for {method}")
        color, linestyle, marker, width = styles[method]
        axes[0].plot(selected.lead_week, selected.rps, color=color, linestyle=linestyle, marker=marker, linewidth=width, markersize=4.6, label=METHOD_LABELS[method])
    axes[0].set_ylabel("Quintile RPS\n(lower is better)")
    axes[0].set_title("a  Same-case categorical scores")
    axes[0].grid(axis="y", color="0.91", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=4, loc="upper center", columnspacing=1.0, handlelength=2.2)

    comparison_styles = {
        "raw_fuxi_categorical": ("#0072B2", "o", -0.06),
        "persistence_plus_plus": ("#D55E00", "s", 0.0),
        "pbc_combined": ("#009E73", "D", 0.06),
    }
    for baseline, (color, marker, offset) in comparison_styles.items():
        selected = contrast.loc[contrast.baseline.eq(baseline) & contrast.lead_week.between(1, 6)].sort_values("lead_week")
        _require(len(selected) == 6, f"figure lacks six contrasts against {baseline}")
        x = selected.lead_week.to_numpy(dtype=float) + offset
        y = selected.score_reduction_pct.to_numpy(dtype=float)
        lower = y - selected.ci_lower_95_pct.to_numpy(dtype=float)
        upper = selected.ci_upper_95_pct.to_numpy(dtype=float) - y
        axes[1].plot(x, y, color=color, linewidth=1.25, alpha=0.85)
        for index, row in enumerate(selected.itertuples(index=False)):
            resolved = row.inference == "neural_better_95pct_interval"
            axes[1].errorbar(
                x[index], y[index], yerr=np.array([[lower[index]], [upper[index]]]),
                color=color, marker=marker, markersize=4.8, markerfacecolor=color if resolved else "white",
                markeredgecolor=color, linewidth=1.0, capsize=2.2,
                label=f"vs {METHOD_LABELS[baseline]}" if index == 0 else None,
            )
    axes[1].axhline(0.0, color="0.35", linewidth=0.85, linestyle="--")
    axes[1].set_xticks(range(1, 7), [f"W{week}" for week in range(1, 7)])
    axes[1].set_xlabel("Lead week")
    axes[1].set_ylabel("Neural RPS reduction (%)\n(positive favors neural)")
    axes[1].set_title("b  Paired two-stage bootstrap intervals")
    axes[1].grid(axis="y", color="0.91", linewidth=0.6)
    axes[1].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].text(0.01, 0.03, "Filled: 95% interval above zero; open: unresolved", transform=axes[1].transAxes, fontsize=7.5, color="0.3")
    fig.suptitle("2022–2024 retrospective transfer: neural leads at W1; PBC catches up later", y=0.995, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _find_contrast(table: pd.DataFrame, lead_week: int, baseline: str) -> Mapping[str, Any]:
    selected = table.loc[table.lead_week.eq(lead_week) & table.baseline.eq(baseline)]
    _require(len(selected) == 1, f"missing contrast W{lead_week} against {baseline}")
    return selected.iloc[0].to_dict()


def _claim_boundaries(source: VerifiedSource, tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    quintile = tables["quintile_neural_contrasts"]
    pooled_raw = _find_contrast(quintile, 0, "raw_fuxi_categorical")
    week1_raw = _find_contrast(quintile, 1, "raw_fuxi_categorical")
    week1_persistence = _find_contrast(quintile, 1, "persistence_plus_plus")
    week1_combined = _find_contrast(quintile, 1, "pbc_combined")
    later = quintile.loc[
        quintile.lead_week.between(2, 6)
        & quintile.baseline.isin(("persistence_plus_plus", "pbc_combined"))
    ]
    _require(set(later.inference) == {"unresolved_95pct_interval"}, "later PBC contrasts are not all unresolved")
    return {
        "evidence_label": source.manifest["evidence_label"],
        "allowed_primary_claims": {
            "pooled_neural_vs_raw_quintile_rps_reduction_pct": {
                "estimate": pooled_raw["score_reduction_pct"],
                "ci_lower_95": pooled_raw["ci_lower_95_pct"],
                "ci_upper_95": pooled_raw["ci_upper_95_pct"],
            },
            "week1_neural_vs_raw_quintile_rps_reduction_pct": {
                "estimate": week1_raw["score_reduction_pct"],
                "ci_lower_95": week1_raw["ci_lower_95_pct"],
                "ci_upper_95": week1_raw["ci_upper_95_pct"],
            },
            "week1_neural_vs_persistence_quintile_rps_reduction_pct": {
                "estimate": week1_persistence["score_reduction_pct"],
                "ci_lower_95": week1_persistence["ci_lower_95_pct"],
                "ci_upper_95": week1_persistence["ci_upper_95_pct"],
            },
            "week1_neural_vs_combined_quintile_rps_reduction_pct": {
                "estimate": week1_combined["score_reduction_pct"],
                "ci_lower_95": week1_combined["ci_lower_95_pct"],
                "ci_upper_95": week1_combined["ci_upper_95_pct"],
            },
            "lead_dependence": "W2 neural-vs-PBC intervals are unresolved; W3-W6 point estimates favor PBC, but every neural-vs-Persistence++ and neural-vs-combined interval crosses zero.",
        },
        "mandatory_limitations": [
            "Post-hoc 2022-2024 retrospective audit; not a prospective operational trial.",
            "No retraining, model selection, PBC refit, or fitted blend was performed on 2022-2024.",
            "Only three source-year clusters are available; paired bootstrap intervals are exploratory.",
            "Neural headline values average per-seed scores, not forecasts, probabilities, parameters, or members.",
            "The India-domain evidence does not establish whole-world performance.",
            "The sealed 2025 forecast and target were not opened.",
        ],
        "forbidden_claims": [
            "prospective or live operational validation",
            "untouched final-test evidence",
            "universal neural dominance over Persistence++ or combined PBC",
            "extreme-rain, flood, drought, or event skill",
            "whole-world training or evaluation",
            "2025 evaluation",
        ],
    }


def _source_receipt(source: VerifiedSource) -> dict[str, Any]:
    return {
        "source_path": str(source.root),
        "source_experiment": SOURCE_EXPERIMENT,
        "source_scientific_status": source.manifest["scientific_status"],
        "manifest_sha256": source.manifest_sha256,
        "postflight_semantic_audit_sha256": source.semantic_sha256,
        "slurm_gate_receipt_sha256": source.gate_sha256,
        "declared_artifacts_verified": len(source.artifact_sha256),
        "physical_inventory_exact": True,
        "case_count": EXPECTED_CASES,
        "year_counts": EXPECTED_YEAR_COUNTS,
        "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
        "source_year_clusters": 3,
        "sealed_2025_target_opened": False,
        "forecast_or_target_arrays_opened_by_builder": False,
        "source_run_modified": False,
        "job_id": source.gate["job_id"],
        "node": source.gate["node"],
        "partition": source.gate["partition"],
    }


def _readme(source: VerifiedSource, claims: Mapping[str, Any]) -> str:
    values = claims["allowed_primary_claims"]
    pooled = values["pooled_neural_vs_raw_quintile_rps_reduction_pct"]
    w1_raw = values["week1_neural_vs_raw_quintile_rps_reduction_pct"]
    w1_p = values["week1_neural_vs_persistence_quintile_rps_reduction_pct"]
    w1_c = values["week1_neural_vs_combined_quintile_rps_reduction_pct"]
    fmt = lambda value: f"{float(value):.2f}"
    return "\n".join(
        [
            "# Operational-era categorical paper supplement",
            "",
            source.manifest["evidence_label"],
            "",
            "This derived bundle is locked to one accepted full source manifest. The builder re-hashed its exact declared artifact inventory, semantic audit, and Slurm gate before reading receipted aggregate CSVs. It did not open forecast or target arrays and did not modify the source run.",
            "",
            "## Claim-safe result",
            "",
            f"For the primary quintile RPS, the neural adapter improves on raw FuXi by {fmt(pooled['estimate'])}% pooled (95% interval {fmt(pooled['ci_lower_95'])}% to {fmt(pooled['ci_upper_95'])}%). At W1 the reduction is {fmt(w1_raw['estimate'])}% ({fmt(w1_raw['ci_lower_95'])}% to {fmt(w1_raw['ci_upper_95'])}%).",
            "",
            f"At W1 the neural adapter also improves on Persistence++ by {fmt(w1_p['estimate'])}% ({fmt(w1_p['ci_lower_95'])}% to {fmt(w1_p['ci_upper_95'])}%) and combined PBC by {fmt(w1_c['estimate'])}% ({fmt(w1_c['ci_lower_95'])}% to {fmt(w1_c['ci_upper_95'])}%). W2 is unresolved. At W3-W6 the point estimates favor PBC, but every neural-vs-Persistence++ and neural-vs-combined interval crosses zero.",
            "",
            "## Contents",
            "",
            "- `tables/primary_pooled_rps.csv`: pooled quintile and semidecile RPS for the four paper-facing methods.",
            "- `tables/*_rps_by_lead.csv`: same-case weekwise RPS and ranks.",
            "- `tables/*_neural_contrasts.csv`: paired neural comparisons, 95% intervals, and resolved/unresolved labels.",
            "- `figures/operational_categorical_transfer.{pdf,png}`: vector and 300-dpi views of the quintile transfer story.",
            "- `source_receipt.json`, `claim_boundaries.json`, and `manifest.json`: provenance, interpretation limits, and exact output hashes.",
            "",
            "## Scope",
            "",
            "This is a post-hoc 2022-2024 retrospective with only three year clusters. It is not a prospective operational test, untouched final test, event/extremes study, global evaluation, or 2025 result.",
            "",
        ]
    )


def _artifact_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def build_bundle(
    source_path: Path,
    expected_manifest_sha256: str,
    output: Path,
    *,
    allowed_output_root: Path,
) -> Path:
    """Verify the source and atomically install a fresh derived bundle."""

    output = Path(output).resolve(strict=False)
    allowed = Path(allowed_output_root).resolve(strict=False)
    try:
        output.relative_to(allowed)
    except ValueError as error:
        raise PaperBundleError(f"output must be under {allowed}") from error
    _require(output != allowed, "output must be a child of the deliverables root")
    _require(output.parent == allowed, "output must be a direct deliverables child")
    _require(output.name.startswith("fuxi_allseason_operational_categorical_paper_"), "output name differs from bundle contract")
    _require(YEAR_TOKEN.search(str(output)) is None, "sealed-year token in output path")
    _require(not output.exists(), f"refusing to overwrite output: {output}")

    source = verify_source(source_path, expected_manifest_sha256=expected_manifest_sha256)
    tables = build_tables(source)
    claims = _claim_boundaries(source, tables)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    _require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir()
    try:
        (staging / "tables").mkdir()
        for name, frame in tables.items():
            frame.to_csv(staging / "tables" / f"{name}.csv", index=False, lineterminator="\n", float_format="%.12g")
        plot_transfer_story(tables, staging / "figures/operational_categorical_transfer")
        _write_json(staging / "source_receipt.json", _source_receipt(source))
        _write_json(staging / "claim_boundaries.json", _json_safe(claims))
        (staging / "README.md").write_text(_readme(source, claims), encoding="utf-8")
        expected_outputs = {
            "README.md",
            "source_receipt.json",
            "claim_boundaries.json",
            "tables/primary_pooled_rps.csv",
            "tables/quintile_rps_by_lead.csv",
            "tables/semidecile_rps_by_lead.csv",
            "tables/quintile_neural_contrasts.csv",
            "tables/semidecile_neural_contrasts.csv",
            "figures/operational_categorical_transfer.pdf",
            "figures/operational_categorical_transfer.png",
        }
        artifacts = _artifact_inventory(staging)
        _require(set(artifacts) == expected_outputs, "derived output inventory differs")
        manifest = {
            "experiment": BUNDLE_EXPERIMENT,
            "contract_version": BUNDLE_CONTRACT,
            "status": "complete",
            "mode": "derived_full",
            "smoke": False,
            "scientific_status": "paper supplement derived from the accepted post-hoc 2022-2024 operational-era categorical retrospective",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output),
            "source": {
                "path": str(source.root),
                "manifest_sha256": source.manifest_sha256,
                "postflight_semantic_audit_sha256": source.semantic_sha256,
                "slurm_gate_receipt_sha256": source.gate_sha256,
            },
            "contract": {
                "source_run_modified": False,
                "forecast_or_target_arrays_opened": False,
                "sealed_2025_target_opened": False,
                "sealed_unopened_years": [2025],
                "tables_derived_only_from_receipted_aggregate_csvs": True,
                "figures_derived_only_from_bundle_tables": True,
                "claim_boundaries_machine_readable": True,
            },
            "tables": sorted(f"tables/{name}.csv" for name in tables),
            "figures": ["figures/operational_categorical_transfer"],
            "artifact_sha256": artifacts,
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        _require(_artifact_inventory(staging) == artifacts, "derived artifact changed before installation")
        os.replace(staging, output)
    except Exception:
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(f".{output.name}.incomplete-"):
            shutil.rmtree(staging)
        raise
    return output


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "presentation/deliverables" / f"fuxi_allseason_operational_categorical_paper_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen operational categorical paper supplement.")
    parser.add_argument("--source-full", type=Path, default=CANONICAL_SOURCE_DIR)
    parser.add_argument("--expected-manifest-sha256", required=True, help="Exact accepted full-run manifest SHA-256.")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = default_output() if args.output is None else args.output
    completed = build_bundle(
        args.source_full,
        args.expected_manifest_sha256,
        output,
        allowed_output_root=PROJECT_ROOT / "presentation/deliverables",
    )
    print(f"PASS: {completed}")
    print(f"manifest_sha256={sha256_file(completed / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
