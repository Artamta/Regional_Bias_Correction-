#!/usr/bin/env python3
"""Compile immutable India S2S paper tables from audited tabular evidence only.

The compiler never opens a forecast, checkpoint, or observation array.  It
accepts an explicitly hash-pinned benchmark receipt, score receipt, and
independent audit receipt; validates their complete artifact inventories and
scientific contracts; and atomically publishes CSV, Markdown, and LaTeX tables.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_paper_evidence_bundle_v1"
BENCHMARK_EXPERIMENT = "india_s2s_benchmark_uncertainty_v2"
SCORING_EXPERIMENT = "india_s2s_probabilistic_bridge_scoring_v1"
AUDIT_EXPERIMENT = "india_s2s_probabilistic_bridge_score_audit_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_paper_bundle"

MODELS = (
    "cma",
    "dlesym_v0",
    "ecmwf",
    "fuxi_s2s",
    "ncep",
    "neuralgcm",
    "ukmo",
    "mme",
)
MODEL_LABELS = {
    "cma": "CMA",
    "dlesym_v0": "DLESyM-v0",
    "ecmwf": "ECMWF",
    "fuxi_s2s": "FuXi-S2S",
    "ncep": "NCEP",
    "neuralgcm": "NeuralGCM",
    "ukmo": "UKMO",
    "mme": "Equal-system MME",
}
METHODS = ("raw_fuxi", "moment_calibration", "location_spread")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "moment_calibration": "Train-only moment calibration",
    "location_spread": "Neural location-spread adapter",
}
REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
LEADS = tuple(range(1, 7))
CASE_COUNT_PER_LEAD = 169
RAW_IDENTITY_ROWS = 15_150
CASE_ROWS = 45_450
STORY_COHORT = "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
PRIMARY_BLOCK_LENGTH = 16
_RENAME_NOREPLACE = 1


class PaperBundleError(RuntimeError):
    """Raised when an input or publication violates the frozen paper contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperBundleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperBundleError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(str(relative))
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run: {relative}")
    return resolved


@dataclass
class HashLedger:
    """Hash-pinned files that must remain unchanged through publication."""

    expected: dict[Path, str] = field(default_factory=dict)
    labels: dict[Path, str] = field(default_factory=dict)

    def verify(self, path: Path, expected: Any, label: str) -> None:
        resolved = Path(path).resolve()
        _require(_valid_sha256(expected), f"{label} has an invalid SHA-256")
        _require(resolved.is_file(), f"{label} is missing: {resolved}")
        digest = str(expected)
        previous = self.expected.get(resolved)
        _require(
            previous in (None, digest),
            f"{label} conflicts with another hash binding for {resolved}",
        )
        observed = sha256_file(resolved)
        _require(observed == digest, f"{label} SHA-256 differs: {observed} != {digest}")
        self.expected[resolved] = digest
        self.labels[resolved] = label

    def reverify(self) -> None:
        for path, digest in self.expected.items():
            _require(path.is_file(), f"pinned input disappeared: {path}")
            _require(
                sha256_file(path) == digest,
                f"pinned input changed during compilation: {self.labels[path]}",
            )


def _verify_artifacts(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    ledger: HashLedger,
    *,
    label: str,
) -> int:
    root = Path(receipt_path).resolve().parent
    hashes = receipt.get("artifact_sha256")
    _require(isinstance(hashes, dict) and hashes, f"{label} has no artifact hashes")
    expected = {str(relative) for relative in hashes}
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {receipt_path.name, "failure.json"}
    }
    _require(
        actual == expected,
        f"{label} artifact inventory differs: "
        f"extra={sorted(actual - expected)}, missing={sorted(expected - actual)}",
    )
    for relative, digest in hashes.items():
        ledger.verify(_resolve_child(root, str(relative)), digest, f"{label} {relative}")
    source_hashes = receipt.get("source_snapshot_sha256", {})
    _require(isinstance(source_hashes, dict), f"{label} source hashes are invalid")
    for relative, digest in source_hashes.items():
        _require(
            hashes.get(relative) == digest,
            f"{label} source snapshot is not artifact-bound: {relative}",
        )
    return len(hashes)


def _assert_output_path(destination: Path, output_root: Path) -> None:
    resolved = Path(destination).resolve()
    root = Path(output_root).resolve()
    _require(root in resolved.parents, "paper bundle output must be under resultsv3/india_s2s_paper_bundle")
    _require(resolved != root, "paper bundle output must be a fresh child directory")


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without replacing a raced destination."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "publication requires one parent directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "publication source is not a real directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("Linux renameat2 is required for no-clobber publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            descriptor,
            os.fsencode(source.name),
            descriptor,
            os.fsencode(destination.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(descriptor)


def _load_inputs(
    benchmark_manifest_path: Path,
    benchmark_manifest_sha256: str,
    scoring_manifest_path: Path,
    scoring_manifest_sha256: str,
    audit_receipt_path: Path,
    audit_receipt_sha256: str,
    ledger: HashLedger,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    specs = (
        (benchmark_manifest_path, benchmark_manifest_sha256, "benchmark manifest"),
        (scoring_manifest_path, scoring_manifest_sha256, "scoring manifest"),
        (audit_receipt_path, audit_receipt_sha256, "audit receipt"),
    )
    payloads: list[dict[str, Any]] = []
    for path, digest, label in specs:
        ledger.verify(Path(path), digest, label)
        payloads.append(_read_json(Path(path)))
    benchmark, scoring, audit = payloads
    _require(
        benchmark.get("experiment") == BENCHMARK_EXPERIMENT
        and benchmark.get("status") == "complete",
        "benchmark receipt is not the completed v2 uncertainty run",
    )
    _require(
        scoring.get("experiment") == SCORING_EXPERIMENT
        and scoring.get("status") == "complete",
        "scoring receipt is not complete",
    )
    _require(
        audit.get("experiment") == AUDIT_EXPERIMENT and audit.get("status") == "passed",
        "independent score audit did not pass",
    )
    artifact_counts = {
        "benchmark": _verify_artifacts(
            Path(benchmark_manifest_path), benchmark, ledger, label="benchmark"
        ),
        "scoring": _verify_artifacts(
            Path(scoring_manifest_path), scoring, ledger, label="scoring"
        ),
        "audit": _verify_artifacts(Path(audit_receipt_path), audit, ledger, label="audit"),
    }
    _require(
        Path(str(audit.get("scoring_manifest", ""))).resolve()
        == Path(scoring_manifest_path).resolve(),
        "audit points to a different scoring manifest",
    )
    _require(
        audit.get("scoring_manifest_sha256") == scoring_manifest_sha256,
        "audit does not bind the requested scoring manifest digest",
    )
    _require(
        audit.get("input_hashes_reverified_after_semantic_audit") is True,
        "audit did not reverify its inputs after semantic checks",
    )
    return benchmark, scoring, audit, artifact_counts


def _validate_seal_and_cohort(
    benchmark: Mapping[str, Any],
    scoring: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    _require(
        benchmark.get("cohort") == "IMD valid-midpoint JJAS"
        and benchmark.get("case_count_per_lead_region") == CASE_COUNT_PER_LEAD,
        "benchmark is not the frozen 169-case valid-midpoint JJAS cohort",
    )
    _require(
        benchmark.get("block_lengths_starts") == [16, 13]
        and benchmark.get("replicates") == 10_000,
        "benchmark uncertainty is not the frozen 10,000-replicate block-16/13 run",
    )
    cohort = scoring.get("cohort", {})
    _require(
        cohort.get("valid_midpoint_jjas_count_per_lead") == CASE_COUNT_PER_LEAD,
        "scoring valid-midpoint JJAS count is not 169 per lead",
    )
    _require(scoring.get("opened_2025_observation") is False, "scoring opened 2025")
    _require(scoring.get("sealed_2025_target_opened") is False, "scoring opened sealed 2025")
    _require(
        cohort.get("maximum_target_label_opened") == "2024-12-30",
        "scoring maximum target label differs",
    )
    _require(scoring.get("opened_observation_years") == list(range(2020, 2025)), "opened years differ")
    truth = scoring.get("truth_access", {})
    _require(
        truth.get("opened_2025") is False
        and truth.get("sealed_2025_target_opened") is False
        and truth.get("maximum_target_label") == "2024-12-30",
        "truth-access receipt violates the 2025 seal",
    )
    raw = scoring.get("raw_fuxi_identity", {})
    _require(
        raw.get("identity") is True
        and raw.get("complete_bridge_contract") is True
        and raw.get("matched_case_rows") == RAW_IDENTITY_ROWS,
        "raw FuXi identity gate failed",
    )
    _require(
        scoring.get("table_rows", {}).get("case_metrics") == CASE_ROWS,
        "scoring case-table row count differs",
    )
    uncertainty = scoring.get("uncertainty", {})
    _require(
        uncertainty.get("block_lengths_starts") == [16, 13]
        and uncertainty.get("replicates") == 10_000
        and uncertainty.get("cohorts", {}).get("2020_2024_valid_midpoint_jjas")
        == CASE_COUNT_PER_LEAD,
        "scoring uncertainty contract differs",
    )
    semantic = audit.get("semantic_checks", {})
    _require(
        semantic.get("full_case_cartesian_rows") == CASE_ROWS
        and semantic.get("raw_identity_rows") == RAW_IDENTITY_ROWS
        and semantic.get("sealed_2025_target_opened") is False
        and semantic.get("forecast_or_truth_arrays_opened_by_auditor") is False
        and semantic.get("valid_midpoint_jjas_summary_reconstructed") is True
        and semantic.get("paired_intervals_reconstructed_for_both_cohorts") is True
        and semantic.get("synchronized_pooled_story_gate_reconstructed") is True,
        "independent semantic-audit gates failed",
    )
    story = audit.get("story_gate", {})
    _require(
        story.get("analysis_cohort") == STORY_COHORT
        and story.get("cases_per_lead") == 100
        and story.get("case_lead_rows") == 600
        and story.get("primary_block_length_starts") == PRIMARY_BLOCK_LENGTH
        and story.get("sealed_2025_target_opened") is False,
        "story-gate cohort or seal differs",
    )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise PaperBundleError(f"cannot read {label}: {path}") from error


def _benchmark_table(
    benchmark_manifest_path: Path,
    benchmark: Mapping[str, Any],
    scoring: Mapping[str, Any],
    ledger: HashLedger,
) -> pd.DataFrame:
    source = Path(str(benchmark.get("source_case_metrics", ""))).resolve()
    source_digest = benchmark.get("source_case_metrics_sha256")
    ledger.verify(source, source_digest, "benchmark case metrics")
    _require(
        scoring.get("benchmark_case_metrics_sha256") == source_digest,
        "benchmark and scoring receipts bind different case-table bytes",
    )
    _require(
        Path(str(scoring.get("benchmark_case_metrics", ""))).resolve() == source,
        "benchmark and scoring receipts point to different case tables",
    )
    raw = scoring.get("raw_fuxi_identity", {})
    _require(
        raw.get("benchmark_case_metrics_sha256") == source_digest
        and Path(str(raw.get("benchmark_case_metrics", ""))).resolve() == source,
        "raw-identity receipt is not bound to the benchmark case table",
    )

    frame = _read_csv(source, "benchmark case metrics")
    required = {
        "track",
        "variable",
        "reference",
        "model",
        "init",
        "valid_period_midpoint",
        "season",
        "region",
        "lead_week",
        "score_status",
        "acc",
        "rmse",
        "mae",
        "bias",
    }
    _require(required.issubset(frame.columns), "benchmark case table lacks required columns")
    midpoint = pd.to_datetime(frame["valid_period_midpoint"], errors="raise")
    selected = frame[
        frame["track"].eq("tp_imd")
        & frame["variable"].eq("tp")
        & frame["reference"].eq("imd")
        & frame["region"].eq("all_india")
        & frame["season"].eq("JJAS")
        & midpoint.dt.month.isin((6, 7, 8, 9))
    ].copy()
    _require(set(selected["model"]) == set(MODELS), "benchmark model inventory differs")
    _require(set(selected["lead_week"].astype(int)) == set(LEADS), "benchmark lead inventory differs")
    _require(
        not selected.duplicated(["model", "lead_week", "init"]).any(),
        "benchmark contains duplicate model/lead/case rows",
    )
    counts = selected.groupby(["model", "lead_week"], observed=True).size()
    _require(
        len(counts) == len(MODELS) * len(LEADS)
        and (counts == CASE_COUNT_PER_LEAD).all(),
        "benchmark does not contain exactly 169 cohort rows per model and lead",
    )

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for lead in LEADS:
            group = selected[(selected["model"] == model) & (selected["lead_week"] == lead)]
            available = group[group["score_status"] == "available"]
            valid_count = len(available)
            expected_valid = 0 if (model, lead) == ("ecmwf", 3) else CASE_COUNT_PER_LEAD
            _require(valid_count == expected_valid, f"unexpected availability for {model} W{lead}")
            if valid_count == 0:
                _require(
                    group[["acc", "rmse", "mae", "bias"]].isna().all().all(),
                    f"unavailable {model} W{lead} contains metric values",
                )
            metrics: dict[str, float] = {}
            for metric in ("acc", "rmse", "mae", "bias"):
                values = available[metric].to_numpy(dtype=np.float64)
                _require(
                    (valid_count == 0 and values.size == 0)
                    or (values.size == CASE_COUNT_PER_LEAD and np.isfinite(values).all()),
                    f"invalid {metric} values for {model} W{lead}",
                )
                metrics[metric] = float(values.mean()) if values.size else np.nan
            if (model, lead) == ("ecmwf", 3):
                note = "unavailable: documented archive artifact"
            elif (model, lead) == ("mme", 3):
                note = "equal-system MME; ECMWF excluded"
            else:
                note = "available"
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "lead_week": lead,
                    "cohort_case_count": CASE_COUNT_PER_LEAD,
                    "valid_case_count": valid_count,
                    "availability_note": note,
                    **metrics,
                }
            )
    output = pd.DataFrame(rows)

    interval_relative = "mme_vs_fuxi_paired_intervals.csv"
    interval_path = _resolve_child(
        Path(benchmark_manifest_path).resolve().parent, interval_relative
    )
    _require(
        benchmark.get("artifact_sha256", {}).get(interval_relative)
        == sha256_file(interval_path),
        "benchmark interval table is not artifact-bound",
    )
    intervals = _read_csv(interval_path, "benchmark intervals")
    primary = intervals[
        intervals["region"].eq("all_india")
        & intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ]
    for lead in LEADS:
        for metric in ("acc", "rmse", "mae", "bias"):
            row = primary[(primary["lead_week"] == lead) & (primary["metric"] == metric)]
            _require(len(row) == 1 and int(row.iloc[0]["n_cases"]) == CASE_COUNT_PER_LEAD, "benchmark interval contract differs")
            mme = output[(output["model"] == "mme") & (output["lead_week"] == lead)].iloc[0]
            fuxi = output[(output["model"] == "fuxi_s2s") & (output["lead_week"] == lead)].iloc[0]
            _require(
                np.isclose(float(row.iloc[0]["candidate_mean"]), float(mme[metric]), atol=1e-12)
                and np.isclose(float(row.iloc[0]["baseline_mean"]), float(fuxi[metric]), atol=1e-12),
                f"benchmark interval means disagree for {metric} W{lead}",
            )
    return output


def _assert_frame_values_equal(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    _require(set(left.columns) == set(right.columns), f"{label} columns differ")
    columns = sorted(left.columns)
    left_canonical = left[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    right_canonical = right[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left_canonical,
            right_canonical,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise PaperBundleError(f"{label} values differ") from error


def _calibrated_table(
    scoring_manifest_path: Path,
    scoring: Mapping[str, Any],
    audit_receipt_path: Path,
    audit: Mapping[str, Any],
    deterministic: pd.DataFrame,
) -> pd.DataFrame:
    scoring_path = _resolve_child(
        Path(scoring_manifest_path).resolve().parent,
        "tables/jjas_valid_midpoint_summary.csv",
    )
    audit_path = _resolve_child(
        Path(audit_receipt_path).resolve().parent,
        "reconstructed/jjas_valid_midpoint_summary.csv",
    )
    score_summary = _read_csv(scoring_path, "scoring JJAS summary")
    audit_summary = _read_csv(audit_path, "audited JJAS summary")
    _assert_frame_values_equal(score_summary, audit_summary, "scoring/audit JJAS summary")
    _require(len(score_summary) == len(METHODS) * len(REGIONS) * len(LEADS), "JJAS summary row count differs")
    _require(
        set(score_summary["method"]) == set(METHODS)
        and set(score_summary["region"]) == set(REGIONS)
        and set(score_summary["lead_week"].astype(int)) == set(LEADS),
        "JJAS summary Cartesian inventory differs",
    )
    count_columns = [column for column in score_summary if column.endswith("_valid_case_count")]
    _require(
        (score_summary["case_count"] == CASE_COUNT_PER_LEAD).all()
        and all((score_summary[column] == CASE_COUNT_PER_LEAD).all() for column in count_columns),
        "calibrated-FuXi summary is not 169 cases per method/region/lead",
    )
    selected_columns = [
        "method",
        "region",
        "lead_week",
        "case_count",
        "crps",
        "acc",
        "rmse",
        "mae",
        "bias",
        "coverage90",
        "pooled_spread_skill_ratio",
        "crps_skill_pct_vs_raw",
        "rmse_skill_pct_vs_raw",
        "delta_acc_vs_raw",
    ]
    _require(set(selected_columns).issubset(score_summary.columns), "JJAS summary lacks publication columns")
    output = score_summary[score_summary["region"].eq("all_india")][selected_columns].copy()
    output.insert(1, "method_label", output["method"].map(METHOD_LABELS))
    output = output.sort_values(
        ["method", "lead_week"],
        key=lambda values: values.map({name: index for index, name in enumerate(METHODS)})
        if values.name == "method"
        else values,
        kind="stable",
    ).reset_index(drop=True)
    _require(len(output) == len(METHODS) * len(LEADS), "all-India calibrated table row count differs")

    raw = output[output["method"].eq("raw_fuxi")].set_index("lead_week")
    frozen = deterministic[deterministic["model"].eq("fuxi_s2s")].set_index("lead_week")
    for metric in ("acc", "rmse", "mae", "bias"):
        _require(
            np.allclose(raw.loc[list(LEADS), metric], frozen.loc[list(LEADS), metric], rtol=1e-7, atol=1e-6),
            f"raw FuXi {metric} summary differs from the frozen benchmark",
        )
    return output


def _story_gate_table(
    scoring_manifest_path: Path,
    audit_receipt_path: Path,
    audit: Mapping[str, Any],
) -> pd.DataFrame:
    scoring_intervals = _read_csv(
        _resolve_child(
            Path(scoring_manifest_path).resolve().parent,
            "tables/paired_block_intervals.csv",
        ),
        "scoring paired intervals",
    )
    audit_intervals = _read_csv(
        _resolve_child(
            Path(audit_receipt_path).resolve().parent,
            "reconstructed/story_gate_intervals.csv",
        ),
        "audited story-gate intervals",
    )
    scoring_story = scoring_intervals[
        scoring_intervals["analysis_cohort"].eq(STORY_COHORT)
    ].copy()
    _require(
        "n_cases" in scoring_story.columns and scoring_story["n_cases"].isna().all(),
        "pooled story rows unexpectedly use a scalar case count",
    )
    scoring_story = scoring_story.drop(columns="n_cases")
    _assert_frame_values_equal(scoring_story, audit_intervals, "scoring/audit story intervals")
    primary = audit_intervals[
        audit_intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ]
    specifications = (
        (
            "crps_skill_pct_vs_raw",
            "crps",
            "raw_fuxi",
            "skill_pct_vs_baseline",
            "retrospective CRPS gate",
        ),
        (
            "acc_delta_vs_raw",
            "acc",
            "raw_fuxi",
            "candidate_minus_baseline",
            "retrospective ACC gate",
        ),
        (
            "bias_delta_vs_raw",
            "bias",
            "raw_fuxi",
            "candidate_minus_baseline",
            "bias guardrail",
        ),
        (
            "crps_skill_pct_vs_moment",
            "crps",
            "moment_calibration",
            "skill_pct_vs_baseline",
            "moment-baseline comparison",
        ),
    )
    story = audit.get("story_gate", {})
    rows: list[dict[str, Any]] = []
    for key, metric, baseline, effect, role in specifications:
        match = primary[
            primary["metric"].eq(metric)
            & primary["candidate"].eq("location_spread")
            & primary["baseline"].eq(baseline)
            & primary["effect"].eq(effect)
        ]
        _require(len(match) == 1, f"missing primary story-gate interval: {key}")
        interval = match.iloc[0]
        _require(
            int(interval["n_cases_per_lead"]) == 100
            and int(interval["n_case_lead_rows"]) == 600
            and int(interval["lead_date_intersection_count"]) == 70
            and int(interval["lead_date_union_count"]) == 130
            and int(interval["replicates"]) == 10_000
            and np.isclose(float(interval["confidence"]), 0.95),
            f"story-gate sampling contract differs for {key}",
        )
        values = story.get(key, {})
        for column in ("estimate", "ci_lower", "ci_upper"):
            _require(
                np.isclose(float(interval[column]), float(values.get(column, np.nan)), atol=1e-12),
                f"story-gate JSON disagrees for {key}.{column}",
            )
        lower = float(interval["ci_lower"])
        upper = float(interval["ci_upper"])
        if role == "bias guardrail":
            decision = "no_significant_change" if lower <= 0.0 <= upper else "review_required"
        else:
            decision = "pass" if lower > 0.0 else "fail"
        rows.append(
            {
                "gate": key,
                "decision_role": role,
                "estimate": float(interval["estimate"]),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence": float(interval["confidence"]),
                "primary_block_length_starts": PRIMARY_BLOCK_LENGTH,
                "cases_per_lead": int(interval["n_cases_per_lead"]),
                "case_lead_rows": int(interval["n_case_lead_rows"]),
                "decision": decision,
                "retrospective_component_passes": bool(
                    story.get("retrospective_2022_2024_component_passes")
                ),
                "final_headline_gate": str(story.get("final_headline_gate")),
                "sealed_2025_target_opened": False,
            }
        )
    _require(
        story.get("retrospective_2022_2024_component_passes") is True
        and all(row["decision"] == "pass" for row in rows if row["decision_role"] != "bias guardrail")
        and rows[2]["decision"] == "no_significant_change",
        "published retrospective story-gate decision is not supported",
    )
    _require(
        story.get("final_headline_gate")
        == "pending_sealed_2025_point-estimate_direction_check",
        "final headline gate is not explicitly pending the sealed 2025 check",
    )
    return pd.DataFrame(rows)


def _display_value(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    labels = [column.replace("_", " ").title() for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_display_value(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    alignment = "l" + "r" * (len(columns) - 1)
    header = " & ".join(_latex_escape(column.replace("_", " ").title()) for column in columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule", header + r" \\", "\\midrule"]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(_display_value(value)) for value in values) + r" \\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines)


def _write_publication_tables(
    staging: Path,
    deterministic: pd.DataFrame,
    calibrated: pd.DataFrame,
    story: pd.DataFrame,
) -> dict[str, int]:
    tables = staging / "tables"
    tables.mkdir(parents=True)
    frames = {
        "main_deterministic_benchmark": deterministic,
        "calibrated_fuxi_by_lead": calibrated,
        "story_gate": story,
    }
    columns = {
        "main_deterministic_benchmark": (
            "model_label",
            "lead_week",
            "cohort_case_count",
            "valid_case_count",
            "acc",
            "rmse",
            "mae",
            "bias",
            "availability_note",
        ),
        "calibrated_fuxi_by_lead": (
            "method_label",
            "lead_week",
            "case_count",
            "crps",
            "acc",
            "rmse",
            "bias",
            "coverage90",
            "pooled_spread_skill_ratio",
            "crps_skill_pct_vs_raw",
            "delta_acc_vs_raw",
        ),
        "story_gate": (
            "gate",
            "estimate",
            "ci_lower",
            "ci_upper",
            "decision",
            "final_headline_gate",
        ),
    }
    for name, frame in frames.items():
        frame.to_csv(tables / f"{name}.csv", index=False, lineterminator="\n")
        (tables / f"{name}.md").write_text(
            _markdown_table(frame, columns[name]) + "\n", encoding="utf-8"
        )
        (tables / f"{name}.tex").write_text(
            _latex_table(frame, columns[name]) + "\n", encoding="utf-8"
        )
    return {name: len(frame) for name, frame in frames.items()}


def compile_bundle(
    benchmark_manifest: Path,
    benchmark_manifest_sha256: str,
    scoring_manifest: Path,
    scoring_manifest_sha256: str,
    audit_receipt: Path,
    audit_receipt_sha256: str,
    output: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Validate frozen receipts and atomically publish paper-facing tables."""

    destination = Path(output).resolve()
    _assert_output_path(destination, output_root)
    if destination.exists():
        raise FileExistsError(f"fresh paper-bundle output required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    ledger = HashLedger()
    try:
        benchmark, scoring, audit, artifact_counts = _load_inputs(
            Path(benchmark_manifest),
            benchmark_manifest_sha256,
            Path(scoring_manifest),
            scoring_manifest_sha256,
            Path(audit_receipt),
            audit_receipt_sha256,
            ledger,
        )
        _validate_seal_and_cohort(benchmark, scoring, audit)
        deterministic = _benchmark_table(
            Path(benchmark_manifest), benchmark, scoring, ledger
        )
        calibrated = _calibrated_table(
            Path(scoring_manifest), scoring, Path(audit_receipt), audit, deterministic
        )
        story = _story_gate_table(Path(scoring_manifest), Path(audit_receipt), audit)
        row_counts = _write_publication_tables(staging, deterministic, calibrated, story)

        source = Path(__file__).resolve()
        snapshot = staging / "code/src" / source.name
        snapshot.parent.mkdir(parents=True)
        shutil.copy2(source, snapshot)

        ledger.reverify()
        artifacts = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "failure.json"}
        }
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(destination),
            "scientific_status": "retrospective 2020-2024 evidence; final headline gate pending sealed 2025 check",
            "inputs": {
                "benchmark_manifest": str(Path(benchmark_manifest).resolve()),
                "benchmark_manifest_sha256": benchmark_manifest_sha256,
                "scoring_manifest": str(Path(scoring_manifest).resolve()),
                "scoring_manifest_sha256": scoring_manifest_sha256,
                "audit_receipt": str(Path(audit_receipt).resolve()),
                "audit_receipt_sha256": audit_receipt_sha256,
                "benchmark_case_metrics": str(Path(str(benchmark["source_case_metrics"])).resolve()),
                "benchmark_case_metrics_sha256": benchmark["source_case_metrics_sha256"],
            },
            "hard_gates": {
                "all_input_sha256_verified": True,
                "all_input_hashes_reverified_before_publication": True,
                "validated_input_artifact_counts": artifact_counts,
                "benchmark_valid_midpoint_jjas_cases_per_lead": CASE_COUNT_PER_LEAD,
                "calibrated_valid_midpoint_jjas_cases_per_method_region_lead": CASE_COUNT_PER_LEAD,
                "raw_fuxi_identity": True,
                "raw_fuxi_identity_rows": RAW_IDENTITY_ROWS,
                "independent_scoring_audit": "passed",
                "retrospective_story_gate": "passed",
                "final_headline_gate": "pending_sealed_2025_point-estimate_direction_check",
                "sealed_2025_target_opened": False,
                "forecast_or_truth_arrays_opened_by_compiler": False,
            },
            "table_rows": row_counts,
            "formats": ["csv", "markdown", "latex"],
            "source_snapshot_sha256": {str(snapshot.relative_to(staging)): artifacts[str(snapshot.relative_to(staging))]},
            "artifact_sha256": artifacts,
        }
        _write_json(staging / "manifest.json", manifest)
        ledger.reverify()
        rename_noreplace(staging, destination)
    except BaseException as error:
        if staging.exists():
            _write_json(
                staging / "failure.json",
                {
                    "experiment": EXPERIMENT,
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "forecast_or_truth_arrays_opened_by_compiler": False,
                    "sealed_2025_target_opened": False,
                },
            )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-manifest-sha256", required=True)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--scoring-manifest-sha256", required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    manifest = compile_bundle(
        arguments.benchmark_manifest,
        arguments.benchmark_manifest_sha256,
        arguments.scoring_manifest,
        arguments.scoring_manifest_sha256,
        arguments.audit_receipt,
        arguments.audit_receipt_sha256,
        arguments.output,
    )
    print(f"PASS: paper evidence bundle published at {manifest}")


if __name__ == "__main__":
    main()
