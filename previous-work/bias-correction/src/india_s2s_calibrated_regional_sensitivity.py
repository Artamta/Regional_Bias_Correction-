#!/usr/bin/env python3
"""Publish calibrated-FuXi regional sensitivity from frozen scoring tables.

This compiler reads only the immutable IMD bridge manifest, CSV tables, and
bootstrap receipt.  It never opens forecast, checkpoint, or observation
arrays and never performs a new resample.  The published intervals are an
exact filtered view of the already-bound pointwise bridge intervals.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_calibrated_regional_sensitivity_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_calibrated_regional_sensitivity"
DEFAULT_SCORING_MANIFEST = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge"
    / "scoring_full_20260825T234808Z/manifest.json"
)
SCORING_MANIFEST_SHA256 = (
    "4ab303a3fa0025bfb2b8654bf34770091f117df18f236c0c724d8e1078450943"
)
SOURCE_ARTIFACT_SHA256 = {
    "tables/case_metrics.csv": (
        "36e2e1ccadf43e29144c4267e290ca001fc3d1f41658b630f665fd158b1d4f8c"
    ),
    "tables/jjas_valid_midpoint_summary.csv": (
        "46afe63646495ee51c64fa8566deae632eaf1e218c083dba57d434cc00cef383"
    ),
    "tables/paired_block_intervals.csv": (
        "a29e6a6c694c74938ea9d5e342dd0bfd359588bf9e9533a09d2de6215a49283e"
    ),
    "receipts/bootstrap_sampling.json": (
        "b8a21b2d06829e120f9a13661f3f7d4916ecab646fb834174e308ea3679f4f94"
    ),
}
ANALYSIS_COHORT = "2020_2024_valid_midpoint_jjas"
REGION_LABELS = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}
REGIONS = tuple(REGION_LABELS)
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "moment_calibration": "Train-only moment calibration",
    "location_spread": "Neural location-spread adapter",
}
METHODS = tuple(METHOD_LABELS)
LEADS = (1, 2, 3, 4, 5, 6)
METRICS = (
    "crps",
    "acc",
    "rmse",
    "mae",
    "bias",
    "coverage90",
    "spread_skill_ratio",
)
LOSS_METRICS = ("crps", "rmse", "mae")
BLOCK_LENGTHS = (16, 13)
PRIMARY_BLOCK_LENGTH = 16
EXPECTED_CASE_ROWS = 45_450
EXPECTED_REGIONAL_JJAS_CASE_ROWS = 12_168
EXPECTED_DESCRIPTIVE_ROWS = 72
EXPECTED_INTERVAL_ROWS = 480
EXPECTED_PRIMARY_INTERVAL_ROWS = 240
_RENAME_NOREPLACE = 1

METRIC_SEMANTICS: Mapping[str, Mapping[str, Any]] = {
    "crps": {
        "target": "minimum",
        "candidate_minus_baseline_favorable_sign": "negative",
        "skill_pct_favorable_sign": "positive",
    },
    "acc": {
        "target": "maximum",
        "candidate_minus_baseline_favorable_sign": "positive",
        "skill_pct_favorable_sign": "not_published",
    },
    "rmse": {
        "target": "minimum",
        "candidate_minus_baseline_favorable_sign": "negative",
        "skill_pct_favorable_sign": "positive",
    },
    "mae": {
        "target": "minimum",
        "candidate_minus_baseline_favorable_sign": "negative",
        "skill_pct_favorable_sign": "positive",
    },
    "bias": {
        "target": "zero",
        "candidate_minus_baseline_favorable_sign": "not_monotone_signed_change",
        "skill_pct_favorable_sign": "not_published",
    },
    "coverage90": {
        "target": 0.9,
        "candidate_minus_baseline_favorable_sign": "not_monotone_target_0.9",
        "skill_pct_favorable_sign": "not_published",
    },
    "spread_skill_ratio": {
        "target": 1.0,
        "candidate_minus_baseline_favorable_sign": "not_monotone_target_1.0",
        "skill_pct_favorable_sign": "not_published",
    },
}


class RegionalSensitivityError(RuntimeError):
    """Raised when the table-only regional contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegionalSensitivityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    _require(Path(path).is_file(), f"{label} is missing: {path}")
    observed = sha256_file(Path(path))
    _require(observed == expected, f"{label} SHA-256 differs: {observed} != {expected}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegionalSensitivityError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run: {relative}")
    return resolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dates_sha256(values: Iterable[Any]) -> str:
    dates = np.asarray(list(values), dtype="datetime64[D]")
    _require(dates.ndim == 1 and dates.size > 0, "date receipt is empty")
    _require(np.unique(dates).size == dates.size, "date receipt has duplicates")
    payload = "".join(
        f"{np.datetime_as_string(value, unit='D')}\n" for value in np.sort(dates)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "publication parents differ")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "publication source is not a directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        _require(renameat2 is not None, "Linux renameat2 is required")
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


def validate_scoring_inputs(
    scoring_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Hash-gate the frozen scoring tables and their bootstrap lineage."""

    manifest_path = Path(scoring_manifest_path).resolve()
    _require_hash(manifest_path, SCORING_MANIFEST_SHA256, "scoring manifest")
    manifest = _read_json(manifest_path)
    required_header = {
        "experiment": "india_s2s_probabilistic_bridge_scoring_v1",
        "status": "complete",
        "scientific_status": "retrospective 2020-2024 benchmark evidence",
        "selected_seed": 43,
        "member_count_each_method": 50,
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
    }
    for key, expected in required_header.items():
        _require(manifest.get(key) == expected, f"scoring manifest {key} differs")
    _require(manifest.get("opened_observation_years") == [2020, 2021, 2022, 2023, 2024],
             "scoring observation years differ")
    _require(manifest.get("methods") == list(METHODS), "scoring methods differ")
    _require(manifest.get("metrics") == list(METRICS), "scoring metrics differ")
    cohort = manifest.get("cohort", {})
    _require(cohort.get("valid_midpoint_jjas_count_per_lead") == 169,
             "scoring JJAS count differs")
    _require(cohort.get("maximum_target_label_opened") == "2024-12-30",
             "scoring maximum target label differs")
    uncertainty = manifest.get("uncertainty", {})
    _require(uncertainty.get("block_lengths_starts") == [16, 13],
             "scoring block lengths differ")
    _require(uncertainty.get("replicates") == 10_000,
             "scoring replicate count differs")
    _require(uncertainty.get("cohorts", {}).get(ANALYSIS_COHORT) == 169,
             "scoring primary interval cohort differs")

    paths: dict[str, Path] = {}
    inventory = manifest.get("artifact_sha256", {})
    for relative, expected in SOURCE_ARTIFACT_SHA256.items():
        _require(inventory.get(relative) == expected,
                 f"scoring manifest does not bind {relative}")
        path = _resolve_child(manifest_path.parent, relative)
        _require_hash(path, expected, f"scoring artifact {relative}")
        paths[relative] = path
    bootstrap = _read_json(paths["receipts/bootstrap_sampling.json"])
    primary_sampling = bootstrap.get(ANALYSIS_COHORT, {})
    for lead in LEADS:
        for block in BLOCK_LENGTHS:
            key = f"w{lead}__block{block}"
            receipt = primary_sampling.get(key, {})
            _require(receipt.get("shape") == [10_000, 169],
                     f"bootstrap shape differs for {key}")
            _require(receipt.get("seed") == 42 + 1000 * lead + block,
                     f"bootstrap seed differs for {key}")
            _require(
                isinstance(receipt.get("index_matrix_sha256"), str)
                and len(receipt["index_matrix_sha256"]) == 64,
                f"bootstrap index hash is invalid for {key}",
            )
    return manifest, paths, bootstrap


def _validate_regional_cases(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "method",
        "region",
        "region_label",
        "lead_week",
        "verification_year",
        "init",
        "valid_period_midpoint",
        "season",
        "crps",
        "acc",
        "rmse",
        "mae",
        "bias",
        "coverage90",
        "ensemble_variance",
        "mean_squared_error",
        "spread_skill_ratio",
    }
    _require(required.issubset(cases.columns), "case table lacks regional columns")
    _require(len(cases) == EXPECTED_CASE_ROWS, "source case-row count differs")
    frame = cases[
        cases["method"].isin(METHODS)
        & cases["region"].isin(REGIONS)
        & cases["lead_week"].isin(LEADS)
        & cases["season"].eq("JJAS")
    ].copy()
    _require(len(frame) == EXPECTED_REGIONAL_JJAS_CASE_ROWS,
             "regional JJAS case-row count differs")
    _require(
        not frame.duplicated(["method", "region", "lead_week", "init"]).any(),
        "regional JJAS cases are duplicated",
    )
    counts = frame.groupby(["method", "region", "lead_week"]).size()
    _require(len(counts) == EXPECTED_DESCRIPTIVE_ROWS and set(counts) == {169},
             "regional JJAS group counts differ")
    _require(set(frame["verification_year"].astype(int)) == {2020, 2021, 2022, 2023, 2024},
             "regional verification years differ")
    midpoints = pd.to_datetime(frame["valid_period_midpoint"], errors="raise")
    _require(set(midpoints.dt.month).issubset({6, 7, 8, 9}),
             "JJAS labels are inconsistent with valid midpoints")
    _require(midpoints.max().year <= 2024, "regional table crosses the 2025 firewall")
    labels = frame[["region", "region_label"]].drop_duplicates()
    _require(labels.set_index("region")["region_label"].to_dict() == REGION_LABELS,
             "regional labels differ")
    for metric in METRICS:
        _require(np.isfinite(frame[metric].to_numpy(dtype=np.float64)).all(),
                 f"regional case metric is non-finite: {metric}")
    receipt = {
        "analysis_cohort": ANALYSIS_COHORT,
        "reference": "imd",
        "season_assignment": "valid-period midpoint in June-September",
        "verification_years": [2020, 2021, 2022, 2023, 2024],
        "regions": REGION_LABELS,
        "methods": METHOD_LABELS,
        "leads": list(LEADS),
        "n_cases_per_method_region_lead": 169,
        "source_case_rows": int(len(cases)),
        "selected_regional_case_rows": int(len(frame)),
        "initialization_dates_sha256_by_lead": {},
        "latest_valid_midpoint": midpoints.max().isoformat(),
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
    }
    raw = frame[frame["method"].eq("raw_fuxi")]
    for lead in LEADS:
        dates = raw[raw["lead_week"].eq(lead)]["init"].drop_duplicates()
        _require(len(dates) == 169, f"W{lead} initialization count differs")
        receipt["initialization_dates_sha256_by_lead"][f"w{lead}"] = _dates_sha256(dates)
    return frame, receipt


def _regional_descriptive_table(
    cases: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    required = {
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
        "ensemble_variance",
        "mean_squared_error",
        "ensemble_spread",
        "pooled_spread_skill_ratio",
        "crps_skill_pct_vs_raw",
        "rmse_skill_pct_vs_raw",
        "mae_skill_pct_vs_raw",
        "delta_acc_vs_raw",
        "delta_bias_vs_raw",
        "delta_coverage90_vs_raw",
        "delta_spread_skill_ratio_vs_raw",
    }
    _require(required.issubset(summary.columns), "JJAS summary lacks regional columns")
    selected = summary[
        summary["method"].isin(METHODS)
        & summary["region"].isin(REGIONS)
        & summary["lead_week"].isin(LEADS)
    ].copy()
    _require(len(selected) == EXPECTED_DESCRIPTIVE_ROWS,
             "regional descriptive row count differs")
    _require(
        not selected.duplicated(["method", "region", "lead_week"]).any(),
        "regional descriptive rows are duplicated",
    )
    _require(set(selected["case_count"].astype(int)) == {169},
             "regional descriptive case count differs")

    grouped = cases.groupby(["method", "region", "lead_week"], sort=False)
    means = grouped[["crps", "acc", "rmse", "mae", "bias", "coverage90",
                     "ensemble_variance", "mean_squared_error"]].mean()
    for row in selected.itertuples(index=False):
        key = (str(row.method), str(row.region), int(row.lead_week))
        expected = means.loc[key]
        for metric in ("crps", "acc", "rmse", "mae", "bias", "coverage90",
                       "ensemble_variance", "mean_squared_error"):
            _require(
                np.isclose(float(getattr(row, metric)), float(expected[metric]),
                           rtol=1e-12, atol=1e-12),
                f"regional summary does not reproduce case mean: {key} {metric}",
            )
        pooled = float(np.sqrt(expected["ensemble_variance"] / expected["mean_squared_error"]))
        _require(np.isclose(float(row.pooled_spread_skill_ratio), pooled,
                            rtol=1e-12, atol=1e-12),
                 f"regional pooled spread-skill ratio differs: {key}")

    selected.insert(0, "analysis_cohort", ANALYSIS_COHORT)
    selected.insert(0, "season", "JJAS_valid_midpoint")
    selected.insert(0, "reference", "imd")
    selected["method_label"] = selected["method"].map(METHOD_LABELS)
    selected["region_label"] = selected["region"].map(REGION_LABELS)
    selected = selected.rename(columns={"case_count": "n_cases"})
    columns = [
        "reference",
        "season",
        "analysis_cohort",
        "region",
        "region_label",
        "method",
        "method_label",
        "lead_week",
        "n_cases",
        "crps",
        "acc",
        "rmse",
        "mae",
        "bias",
        "coverage90",
        "ensemble_spread",
        "pooled_spread_skill_ratio",
        "crps_skill_pct_vs_raw",
        "rmse_skill_pct_vs_raw",
        "mae_skill_pct_vs_raw",
        "delta_acc_vs_raw",
        "delta_bias_vs_raw",
        "delta_coverage90_vs_raw",
        "delta_spread_skill_ratio_vs_raw",
    ]
    return selected[columns].sort_values(
        ["region", "method", "lead_week"], kind="stable"
    ).reset_index(drop=True)


def _direction_label(metric: str, effect: str) -> str:
    if effect == "skill_pct_vs_baseline":
        return "positive_favors_neural"
    favorable = METRIC_SEMANTICS[metric]["candidate_minus_baseline_favorable_sign"]
    if favorable == "negative":
        return "negative_favors_neural"
    if favorable == "positive":
        return "positive_favors_neural"
    return str(favorable)


def _regional_interval_table(
    intervals: pd.DataFrame,
    descriptive: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "reference",
        "season",
        "analysis_cohort",
        "region",
        "lead_week",
        "metric",
        "candidate",
        "baseline",
        "candidate_mean",
        "baseline_mean",
        "n_cases",
        "confidence",
        "block_length_starts",
        "replicates",
        "seed",
        "effect",
        "estimate",
        "ci_lower",
        "ci_upper",
    }
    _require(required.issubset(intervals.columns), "interval table lacks regional columns")
    lead_numeric = pd.to_numeric(intervals["lead_week"], errors="coerce")
    selected = intervals[
        intervals["analysis_cohort"].eq(ANALYSIS_COHORT)
        & intervals["region"].isin(REGIONS)
        & intervals["candidate"].eq("location_spread")
        & intervals["baseline"].eq("raw_fuxi")
        & lead_numeric.isin(LEADS)
    ].copy()
    selected["lead_week"] = pd.to_numeric(selected["lead_week"], errors="raise").astype(int)
    _require(len(selected) == EXPECTED_INTERVAL_ROWS,
             "regional neural-vs-raw interval row count differs")
    _require(set(selected["metric"]) == set(METRICS), "regional interval metrics differ")
    _require(set(selected["block_length_starts"].astype(int)) == set(BLOCK_LENGTHS),
             "regional interval block lengths differ")
    _require(set(selected["confidence"].astype(float)) == {0.95},
             "regional interval confidence differs")
    _require(set(selected["replicates"].astype(int)) == {10_000},
             "regional interval replicate count differs")
    _require(set(selected["n_cases"].astype(int)) == {169},
             "regional interval case count differs")
    key_columns = [
        "region",
        "lead_week",
        "metric",
        "effect",
        "block_length_starts",
    ]
    _require(not selected.duplicated(key_columns).any(),
             "regional interval rows are duplicated")
    expected_effects = {
        metric: ({"candidate_minus_baseline", "skill_pct_vs_baseline"}
                 if metric in LOSS_METRICS else {"candidate_minus_baseline"})
        for metric in METRICS
    }
    grouped_effects = selected.groupby("metric")["effect"].agg(set).to_dict()
    _require(grouped_effects == expected_effects, "regional interval effects differ")
    _require(np.isfinite(selected[["estimate", "ci_lower", "ci_upper"]].to_numpy(
        dtype=np.float64)).all(), "regional intervals contain non-finite values")

    scores = descriptive.set_index(["region", "method", "lead_week"])
    metric_columns = {**{metric: metric for metric in METRICS},
                      "spread_skill_ratio": "pooled_spread_skill_ratio"}
    for row in selected.itertuples(index=False):
        lead = int(row.lead_week)
        metric = str(row.metric)
        column = metric_columns[metric]
        candidate = float(scores.loc[(row.region, "location_spread", lead), column])
        baseline = float(scores.loc[(row.region, "raw_fuxi", lead), column])
        _require(np.isclose(float(row.candidate_mean), candidate, rtol=1e-12, atol=1e-12),
                 f"interval candidate mean differs: {row.region} W{lead} {metric}")
        _require(np.isclose(float(row.baseline_mean), baseline, rtol=1e-12, atol=1e-12),
                 f"interval baseline mean differs: {row.region} W{lead} {metric}")
        expected = (100.0 * (1.0 - candidate / baseline)
                    if row.effect == "skill_pct_vs_baseline" else candidate - baseline)
        _require(np.isclose(float(row.estimate), expected, rtol=1e-11, atol=1e-11),
                 f"interval estimate differs: {row.region} W{lead} {metric} {row.effect}")
        expected_seed = 42 + 1000 * lead + int(row.block_length_starts)
        _require(int(row.seed) == expected_seed,
                 f"interval seed differs: {row.region} W{lead} block {row.block_length_starts}")

    selected["region_label"] = selected["region"].map(REGION_LABELS)
    selected["candidate_label"] = METHOD_LABELS["location_spread"]
    selected["baseline_label"] = METHOD_LABELS["raw_fuxi"]
    selected["interval_scope"] = "pointwise_95pct"
    selected["multiplicity_adjustment"] = "none"
    selected["effect_direction_semantics"] = [
        _direction_label(metric, effect)
        for metric, effect in zip(selected["metric"], selected["effect"], strict=True)
    ]
    selected["ci_excludes_zero"] = (
        (selected["ci_lower"] > 0.0) | (selected["ci_upper"] < 0.0)
    )
    columns = [
        "reference",
        "season",
        "analysis_cohort",
        "region",
        "region_label",
        "lead_week",
        "metric",
        "candidate",
        "candidate_label",
        "baseline",
        "baseline_label",
        "candidate_mean",
        "baseline_mean",
        "n_cases",
        "confidence",
        "block_length_starts",
        "replicates",
        "seed",
        "effect",
        "effect_direction_semantics",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_excludes_zero",
        "interval_scope",
        "multiplicity_adjustment",
    ]
    return selected[columns].sort_values(
        ["region", "lead_week", "metric", "effect", "block_length_starts"],
        kind="stable",
    ).reset_index(drop=True)


def build_regional_tables(
    cases: pd.DataFrame,
    summary: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and filter all four regional table products without resampling."""

    regional_cases, cohort = _validate_regional_cases(cases)
    descriptive = _regional_descriptive_table(regional_cases, summary)
    all_intervals = _regional_interval_table(intervals, descriptive)
    primary = all_intervals[
        all_intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
    ].reset_index(drop=True)
    _require(len(primary) == EXPECTED_PRIMARY_INTERVAL_ROWS,
             "primary regional interval row count differs")
    return descriptive, all_intervals, primary, cohort


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "failure.json"}
    }


def publish(*, scoring_manifest: Path, output: Path) -> Path:
    """Publish a fresh, atomic, manifest-bound regional sensitivity artifact."""

    destination = Path(output).resolve()
    output_root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(output_root in destination.parents,
             "output must be a fresh child of the regional resultsv3 root")
    _require(destination != output_root, "output cannot equal the regional output root")
    _require(not destination.exists(), f"fresh output path required: {destination}")
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    manifest_path = Path(scoring_manifest).resolve()
    frozen_inputs: dict[Path, str] = {manifest_path: SCORING_MANIFEST_SHA256}
    try:
        scoring, source_paths, bootstrap = validate_scoring_inputs(manifest_path)
        frozen_inputs.update(
            {source_paths[relative]: expected
             for relative, expected in SOURCE_ARTIFACT_SHA256.items()}
        )
        cases = pd.read_csv(
            source_paths["tables/case_metrics.csv"], float_precision="round_trip"
        )
        summary = pd.read_csv(
            source_paths["tables/jjas_valid_midpoint_summary.csv"],
            float_precision="round_trip",
        )
        intervals = pd.read_csv(
            source_paths["tables/paired_block_intervals.csv"],
            float_precision="round_trip",
        )
        descriptive, all_intervals, primary, cohort = build_regional_tables(
            cases, summary, intervals
        )

        tables_dir = staging / "tables"
        receipts_dir = staging / "receipts"
        source_dir = staging / "code/src"
        tables_dir.mkdir(parents=True)
        receipts_dir.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        descriptive.to_csv(
            tables_dir / "regional_jjas_descriptive_metrics.csv",
            index=False,
            lineterminator="\n",
        )
        all_intervals.to_csv(
            tables_dir / "regional_neural_vs_raw_intervals_all_blocks.csv",
            index=False,
            lineterminator="\n",
        )
        primary.to_csv(
            tables_dir / "regional_neural_vs_raw_primary_block16.csv",
            index=False,
            lineterminator="\n",
        )
        _write_json(receipts_dir / "cohort.json", cohort)
        bootstrap_receipts = {
            f"w{lead}__block{block}": bootstrap[ANALYSIS_COHORT][f"w{lead}__block{block}"]
            for lead in LEADS
            for block in BLOCK_LENGTHS
        }
        interval_lineage = {
            "source_interval_table": str(
                source_paths["tables/paired_block_intervals.csv"]
            ),
            "source_interval_table_sha256": SOURCE_ARTIFACT_SHA256[
                "tables/paired_block_intervals.csv"
            ],
            "analysis_cohort": ANALYSIS_COHORT,
            "candidate": "location_spread",
            "baseline": "raw_fuxi",
            "regions": list(REGIONS),
            "lead_weeks": list(LEADS),
            "metrics": list(METRICS),
            "confidence": 0.95,
            "pointwise_intervals": True,
            "multiplicity_adjustment": "none",
            "primary_block_length_starts": PRIMARY_BLOCK_LENGTH,
            "sensitivity_block_length_starts": 13,
            "replicates": 10_000,
            "resampling_performed_by_this_artifact": False,
            "intervals_are_filtered_from_frozen_source": True,
            "bootstrap_sampling_receipts": bootstrap_receipts,
        }
        _write_json(receipts_dir / "interval_lineage.json", interval_lineage)
        source = Path(__file__).resolve()
        shutil.copy2(source, source_dir / source.name)
        source_hash = sha256_file(source)

        for path, expected in frozen_inputs.items():
            _require_hash(path, expected, "regional sensitivity input changed during run")
        _require_hash(source_dir / source.name, source_hash, "source snapshot")

        manifest: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "scientific_status": "retrospective regional sensitivity",
            "created_utc": _utc_now(),
            "output_path": str(destination),
            "inputs": {
                "scoring_manifest": str(manifest_path),
                "scoring_manifest_sha256": SCORING_MANIFEST_SHA256,
                **{
                    relative.replace("/", "_").replace(".", "_"): {
                        "path": str(source_paths[relative]),
                        "sha256": expected,
                    }
                    for relative, expected in SOURCE_ARTIFACT_SHA256.items()
                },
            },
            "table_only_compilation": True,
            "forecast_arrays_opened": False,
            "observation_arrays_opened": False,
            "checkpoint_arrays_opened": False,
            "new_resampling_performed": False,
            "retrospective_only": True,
            "opened_observation_years": [2020, 2021, 2022, 2023, 2024],
            "latest_target_label_in_source": "2024-12-30",
            "opened_2025_observation": False,
            "sealed_2025_target_opened": False,
            "reference": "imd",
            "season_assignment": "valid-period midpoint in June-September",
            "analysis_cohort": ANALYSIS_COHORT,
            "regions": REGION_LABELS,
            "methods": METHOD_LABELS,
            "method_member_count": 50,
            "selected_neural_seed": 43,
            "lead_weeks": list(LEADS),
            "metrics": list(METRICS),
            "metric_semantics": METRIC_SEMANTICS,
            "cohort": cohort,
            "uncertainty": {
                "source": "frozen scoring paired_block_intervals.csv",
                "resampling_performed_by_this_artifact": False,
                "confidence": 0.95,
                "scope": "pointwise by region, lead, metric, effect, and block length",
                "multiplicity_adjustment": "none",
                "primary_block_length_starts": PRIMARY_BLOCK_LENGTH,
                "sensitivity_block_length_starts": 13,
                "replicates": 10_000,
                "n_cases_per_interval": 169,
            },
            "table_rows": {
                "regional_jjas_descriptive_metrics": int(len(descriptive)),
                "regional_neural_vs_raw_intervals_all_blocks": int(len(all_intervals)),
                "regional_neural_vs_raw_primary_block16": int(len(primary)),
            },
            "source_snapshot_sha256": {f"code/src/{source.name}": source_hash},
            "software_versions": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "upstream_safety_contract": {
                "opened_2025_observation": scoring["opened_2025_observation"],
                "sealed_2025_target_opened": scoring["sealed_2025_target_opened"],
                "maximum_target_label_opened": scoring["cohort"][
                    "maximum_target_label_opened"
                ],
            },
        }
        manifest["artifact_sha256"] = _artifact_hashes(staging)
        _write_json(staging / "manifest.json", manifest)

        for path, expected in frozen_inputs.items():
            _require_hash(path, expected, "regional sensitivity input changed before publication")
        _require_hash(source, source_hash, "regional sensitivity source changed during run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _rename_noreplace(staging, destination)
    except BaseException as error:
        _write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "table_only_compilation": True,
                "forecast_arrays_opened": False,
                "observation_arrays_opened": False,
                "checkpoint_arrays_opened": False,
                "opened_2025_observation": False,
                "sealed_2025_target_opened": False,
            },
        )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring-manifest", type=Path, default=DEFAULT_SCORING_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = publish(scoring_manifest=args.scoring_manifest, output=args.output)
    print(f"PASS: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
