#!/usr/bin/env python3
"""Build venue-neutral figures and tables from sealed E2/E3 metric artifacts.

This program is deliberately a *metric-artifact-only* reporting layer.  It
verifies caller-pinned source manifests and the checksums declared for five
small CSV files before loading any table.  It never opens prediction stores,
forecast arrays, observation arrays, station source files, or 2025 data.

The output is written atomically to a fresh directory and includes a manifest
whose checksums can be verified with :func:`verify_package`.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

E2_REQUIRED_ARTIFACTS = (
    "metrics/summary_pooled.csv",
    "metrics/summary_by_lead.csv",
    "metrics/paired_block_bootstrap_effects.csv",
)
E3_REQUIRED_ARTIFACTS = (
    "method_summary.csv",
    "paired_bootstrap_effects.csv",
)

E2_METHODS = (
    "raw_fuxi",
    "log_bias",
    "legacy_anchored_adapter",
    "raw_identity",
    "raw_identity_raw_mean_preserved",
)
E3_METHODS = (
    "raw_fuxi",
    "log_bias",
    "selected_adapter",
    "raw_identity",
    "raw_identity_raw_mean_preserved",
)

METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Training-only log-bias",
    "legacy_anchored_adapter": "Anchored adapter",
    "selected_adapter": "Anchored adapter",
    "raw_identity": "Raw-identity adapter",
    "raw_identity_raw_mean_preserved": "Raw-mean projection",
}
SHORT_LABELS = {
    "raw_fuxi": "Raw",
    "log_bias": "Log-bias",
    "legacy_anchored_adapter": "Anchored",
    "selected_adapter": "Anchored",
    "raw_identity": "Raw identity",
    "raw_identity_raw_mean_preserved": "Raw-mean proj.",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#56B4E9",
    "legacy_anchored_adapter": "#E69F00",
    "selected_adapter": "#E69F00",
    "raw_identity": "#0072B2",
    "raw_identity_raw_mean_preserved": "#009E73",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    "legacy_anchored_adapter": "D",
    "selected_adapter": "D",
    "raw_identity": "^",
    "raw_identity_raw_mean_preserved": "P",
}

E2_FOREST_COMPARISONS = (
    ("log_bias", "raw_fuxi"),
    ("legacy_anchored_adapter", "raw_fuxi"),
    ("raw_identity", "raw_fuxi"),
    ("raw_identity_raw_mean_preserved", "raw_fuxi"),
    ("raw_identity", "legacy_anchored_adapter"),
    ("raw_identity_raw_mean_preserved", "raw_identity"),
)
E3_COMPARISONS = (
    ("selected_adapter", "raw_fuxi"),
    ("selected_adapter", "log_bias"),
    ("log_bias", "raw_fuxi"),
    ("raw_identity", "raw_fuxi"),
    ("raw_identity_raw_mean_preserved", "raw_fuxi"),
    ("raw_identity_raw_mean_preserved", "raw_identity"),
)

PLOT_METRICS = (
    ("rmse", "RMSE improvement", "mm/day"),
    ("mae", "MAE improvement", "mm/day"),
    ("absolute_bias", "Absolute-bias improvement", "mm/day"),
    ("acc", "ACC improvement", "ACC"),
)

PLOT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "figure.titlesize": 11.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class EvidenceInputs:
    """Verified manifests and the five loaded metric tables."""

    e2_directory: Path
    e3_directory: Path
    e2_manifest: Mapping[str, Any]
    e3_manifest: Mapping[str, Any]
    e2_manifest_sha256: str
    e3_manifest_sha256: str
    verified_source_artifacts: Mapping[str, Mapping[str, str]]
    e2_pooled: pd.DataFrame
    e2_by_lead: pd.DataFrame
    e2_effects: pd.DataFrame
    e3_summary: pd.DataFrame
    e3_effects: pd.DataFrame


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_sha256(value: str, label: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return result


def _read_hash_pinned_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    """Hash manifest bytes before parsing them as JSON."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing regular manifest file: {path}")
    expected = _normalise_sha256(expected_sha256, str(path))
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"manifest SHA-256 differs for {path}: expected {expected}, got {actual}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain one JSON object: {path}")
    return value, actual


def _contained_regular_file(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"missing regular source artifact: {candidate}")
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise ValueError(f"source artifact escapes its canonical directory: {relative}")
    return resolved


def _read_verified_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    required: Sequence[str],
) -> tuple[dict[str, str], dict[str, bytes]]:
    """Read every CSV once, binding its checksum to the bytes later parsed."""

    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError(f"source manifest lacks an artifacts mapping: {root}")
    verified: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for relative in required:
        expected_value = declared.get(relative)
        if not isinstance(expected_value, str):
            raise ValueError(
                f"source manifest does not declare required artifact {relative}"
            )
        expected = _normalise_sha256(expected_value, relative)
        path = _contained_regular_file(root, relative)
        payload = path.read_bytes()
        actual = _sha256_bytes(payload)
        if actual != expected:
            raise ValueError(
                f"artifact SHA-256 differs for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
        payloads[relative] = payload
    return verified, payloads


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} must be {expected!r}, got {value!r}")


def _validate_e2_manifest(manifest: Mapping[str, Any]) -> None:
    _require_exact(manifest.get("status"), "complete", "E2 status")
    _require_exact(manifest.get("canonical"), True, "E2 canonical")
    _require_exact(manifest.get("scientific_eligible"), True, "E2 scientific_eligible")
    _require_exact(manifest.get("smoke"), False, "E2 smoke")
    _require_exact(
        tuple(manifest.get("audit_years", ())), (2022, 2023, 2024), "E2 audit years"
    )
    _require_exact(
        manifest.get("final_initialization_year_quarantined"),
        2025,
        "E2 quarantined year",
    )
    _require_exact(manifest.get("final_2025_store_opened"), False, "E2 2025-store flag")
    _require_exact(tuple(manifest.get("methods", ())), E2_METHODS, "E2 methods")
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("E2 bootstrap contract is missing")
    _require_exact(bootstrap.get("draws"), 10_000, "E2 bootstrap draws")
    _require_exact(bootstrap.get("block_length_initializations"), 13, "E2 block length")
    _require_exact(bootstrap.get("seed"), 20260822, "E2 bootstrap seed")
    _require_exact(bootstrap.get("all_six_leads_retained"), True, "E2 lead retention")
    status = str(manifest.get("scientific_status", "")).lower()
    if "development audit" not in status or "untouched-final-test claim" not in status:
        raise ValueError(
            "E2 scientific_status lacks the retrospective development-audit boundary"
        )


def _validate_e3_manifest(manifest: Mapping[str, Any], e2_manifest_sha256: str) -> None:
    _require_exact(
        manifest.get("status"),
        "complete_frozen_external_target_sensitivity",
        "E3 status",
    )
    _require_exact(manifest.get("canonical_artifact"), True, "E3 canonical_artifact")
    _require_exact(manifest.get("training_performed"), False, "E3 training flag")
    _require_exact(
        manifest.get("selection_calibration_or_blending_performed"),
        False,
        "E3 selection/calibration flag",
    )
    for key in (
        "2025_metric_computed",
        "2025_prediction_opened",
        "2025_station_value_selected",
    ):
        _require_exact(manifest.get(key), False, f"E3 {key}")
    _require_exact(tuple(manifest.get("methods", ())), E3_METHODS, "E3 methods")
    dates = manifest.get("dates")
    if not isinstance(dates, dict):
        raise ValueError("E3 dates contract is missing")
    _require_exact(
        tuple(dates.get("initialization_years", ())), (2024,), "E3 initialization years"
    )
    _require_exact(tuple(dates.get("lead_weeks", ())), (1, 2, 3, 4, 5, 6), "E3 leads")
    _require_exact(dates.get("initialization_count"), 30, "E3 initialization count")
    _require_exact(dates.get("case_leads"), 180, "E3 case-lead count")
    for date_key in ("verification_date_min", "verification_date_max"):
        if not str(dates.get(date_key, "")).startswith("2024-"):
            raise ValueError(f"E3 {date_key} must remain in 2024")
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("E3 bootstrap contract is missing")
    _require_exact(bootstrap.get("draws"), 2_000, "E3 bootstrap draws")
    _require_exact(bootstrap.get("primary_block_length"), 13, "E3 primary block length")
    _require_exact(
        tuple(bootstrap.get("sensitivity_block_lengths", ())),
        (4, 8),
        "E3 sensitivity blocks",
    )
    _require_exact(bootstrap.get("seed"), 20260822, "E3 bootstrap seed")
    _require_exact(
        manifest.get("extended_prediction_manifest_sha256"),
        e2_manifest_sha256,
        "E3 binding to E2 manifest",
    )
    status = str(manifest.get("scientific_status", "")).lower()
    if (
        "target" not in status
        or "sensitivity" not in status
        or "not untouched" not in status
    ):
        raise ValueError(
            "E3 scientific_status lacks the external-target sensitivity boundary"
        )
    station_boundary = manifest.get("station_truth_boundary")
    if not isinstance(station_boundary, dict):
        raise ValueError("E3 station_truth_boundary disclosure is missing")
    _require_exact(
        station_boundary.get("unselected_2025_plus_rows"),
        45_910,
        "E3 unselected 2025+ station rows",
    )
    _require_exact(
        station_boundary.get("container_rows_scanned"),
        543_518,
        "E3 mixed station-container rows scanned",
    )
    _require_exact(
        station_boundary.get("container_date_max"),
        "2025-02-10",
        "E3 mixed station-container maximum date",
    )
    _require_exact(
        station_boundary.get(
            "rainfall_converted_only_after_exact_2024_date_and_station_filter"
        ),
        True,
        "E3 station filtering before rainfall conversion",
    )
    _require_exact(
        station_boundary.get("selected_snapshot"),
        "inputs/station_truth_selected_2024.csv.gz",
        "E3 selected station snapshot",
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} must be finite numeric data")


def _validate_e2_tables(
    pooled: pd.DataFrame,
    by_lead: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    metric_columns = ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
    _require_columns(
        pooled, ("method", "case_lead_count", *metric_columns), "E2 pooled"
    )
    _require_exact(
        tuple(sorted(pooled["method"])), tuple(sorted(E2_METHODS)), "E2 pooled methods"
    )
    if (
        pooled["method"].duplicated().any()
        or not pooled["case_lead_count"].eq(600).all()
    ):
        raise ValueError("E2 pooled table must have one 600-case-lead row per method")
    _require_finite(pooled, metric_columns, "E2 pooled")

    _require_columns(
        by_lead,
        ("lead_week", "method", "case_lead_count", *metric_columns),
        "E2 by-lead",
    )
    expected_pairs = {(method, lead) for method in E2_METHODS for lead in range(1, 7)}
    actual_pairs = set(zip(by_lead["method"], by_lead["lead_week"].astype(int)))
    _require_exact(actual_pairs, expected_pairs, "E2 method/lead coverage")
    if (
        by_lead.duplicated(["method", "lead_week"]).any()
        or not by_lead["case_lead_count"].eq(100).all()
    ):
        raise ValueError(
            "E2 by-lead table must have one 100-case row per method and lead"
        )
    _require_finite(by_lead, metric_columns, "E2 by-lead")

    effect_columns = (
        "scope_type",
        "candidate",
        "baseline",
        "source_metric",
        "effect",
        "ci_lower_2p5",
        "ci_upper_97p5",
        "bootstrap_probability_improved",
        "n_starts",
        "n_leads_per_start",
        "definition",
    )
    _require_columns(effects, effect_columns, "E2 effects")
    selected = effects.loc[
        effects["scope_type"].eq("pooled")
        & effects[["candidate", "baseline"]]
        .apply(tuple, axis=1)
        .isin(E2_FOREST_COMPARISONS)
        & effects["source_metric"].isin(
            ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
        )
    ].copy()
    expected = {
        (candidate, baseline, metric)
        for candidate, baseline in E2_FOREST_COMPARISONS
        for metric in ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
    }
    actual = set(
        zip(selected["candidate"], selected["baseline"], selected["source_metric"])
    )
    _require_exact(actual, expected, "E2 pooled forest effects")
    if selected.duplicated(["candidate", "baseline", "source_metric"]).any():
        raise ValueError("E2 pooled forest effects contain duplicate rows")
    _require_finite(
        selected,
        ("effect", "ci_lower_2p5", "ci_upper_97p5", "bootstrap_probability_improved"),
        "E2 effects",
    )
    if (
        not selected["n_starts"].eq(100).all()
        or not selected["n_leads_per_start"].eq(6).all()
    ):
        raise ValueError(
            "E2 effects must retain 100 starts with all six leads attached"
        )
    if (
        not selected["definition"]
        .str.contains("actual block length 13", regex=False)
        .all()
    ):
        raise ValueError(
            "E2 effects do not declare the canonical circular block-13 contract"
        )
    if (selected["ci_lower_2p5"] > selected["effect"]).any() or (
        selected["effect"] > selected["ci_upper_97p5"]
    ).any():
        raise ValueError(
            "E2 point effects must lie inside their reported percentile intervals"
        )
    _crosscheck_e2_effects(pooled, selected)


def _crosscheck_e2_effects(pooled: pd.DataFrame, effects: pd.DataFrame) -> None:
    rows = pooled.set_index("method")
    for row in effects.itertuples(index=False):
        candidate = rows.loc[row.candidate]
        baseline = rows.loc[row.baseline]
        if row.source_metric == "acc":
            expected = candidate["acc"] - baseline["acc"]
        elif row.source_metric == "bias_mm_day":
            expected = abs(baseline["bias_mm_day"]) - abs(candidate["bias_mm_day"])
        else:
            expected = baseline[row.source_metric] - candidate[row.source_metric]
        if not math.isclose(
            float(row.effect), float(expected), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"E2 effect does not reproduce pooled metrics: {row.candidate} vs "
                f"{row.baseline}, {row.source_metric}"
            )


def _validate_e3_tables(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    metric_columns = (
        "rmse_mean",
        "mae_mean",
        "bias_mean",
        "absolute_bias_mean",
        "acc_mean",
    )
    _require_columns(
        summary,
        (
            "method",
            "scope_type",
            "scope",
            "initializations",
            "case_leads",
            *metric_columns,
        ),
        "E3 summary",
    )
    pooled = summary.loc[summary["scope_type"].eq("pooled")]
    _require_exact(
        tuple(sorted(pooled["method"])), tuple(sorted(E3_METHODS)), "E3 pooled methods"
    )
    if pooled["method"].duplicated().any():
        raise ValueError("E3 pooled table has duplicate methods")
    if (
        not pooled["initializations"].eq(30).all()
        or not pooled["case_leads"].eq(180).all()
    ):
        raise ValueError(
            "E3 pooled table must use 30 initializations and 180 case-leads"
        )
    _require_finite(pooled, metric_columns, "E3 pooled")
    lead = summary.loc[summary["scope_type"].eq("lead")]
    expected_leads = {
        (method, f"W{week}") for method in E3_METHODS for week in range(1, 7)
    }
    _require_exact(
        set(zip(lead["method"], lead["scope"])), expected_leads, "E3 lead coverage"
    )

    effect_columns = (
        "comparison",
        "candidate",
        "reference",
        "metric",
        "effect_definition",
        "block_length_initializations",
        "analysis_role",
        "bootstrap_draws",
        "initializations",
        "case_leads",
        "point_effect",
        "ci_lower_2p5",
        "ci_upper_97p5",
        "probability_effect_gt_zero",
        "primary_estimand",
    )
    _require_columns(effects, effect_columns, "E3 effects")
    expected_all = {
        (candidate, reference, metric, block)
        for candidate, reference in E3_COMPARISONS
        for metric in ("rmse", "mae", "acc", "bias", "absolute_bias")
        for block in (4, 8, 13)
    }
    actual_all = set(
        zip(
            effects["candidate"],
            effects["reference"],
            effects["metric"],
            effects["block_length_initializations"].astype(int),
        )
    )
    _require_exact(actual_all, expected_all, "E3 comparison/metric/block coverage")
    if effects.duplicated(
        ["candidate", "reference", "metric", "block_length_initializations"]
    ).any():
        raise ValueError("E3 effects contain duplicate comparison/metric/block rows")
    _require_finite(
        effects,
        ("point_effect", "ci_lower_2p5", "ci_upper_97p5", "probability_effect_gt_zero"),
        "E3 effects",
    )
    if not effects["bootstrap_draws"].eq(2000).all():
        raise ValueError("E3 effects must use 2,000 bootstrap draws")
    if (
        not effects["initializations"].eq(30).all()
        or not effects["case_leads"].eq(180).all()
    ):
        raise ValueError("E3 effects must retain 30 starts with all six leads attached")
    if not pd.api.types.is_bool_dtype(effects["primary_estimand"].dtype):
        raise ValueError("E3 primary_estimand values must be strict booleans")
    if any(
        type(value) not in (bool, np.bool_)
        for value in effects["primary_estimand"].array
    ):
        raise ValueError("E3 primary_estimand values must be strict booleans")
    primary = effects.loc[effects["primary_estimand"]]
    if len(primary) != 1:
        raise ValueError("E3 effects must contain exactly one primary_estimand=True")
    primary_row = primary.iloc[0]
    expected_primary_key = (
        "selected_adapter_vs_raw_fuxi",
        "selected_adapter",
        "raw_fuxi",
        "rmse",
        13,
        "primary_uncertainty",
    )
    actual_primary_key = (
        primary_row["comparison"],
        primary_row["candidate"],
        primary_row["reference"],
        primary_row["metric"],
        int(primary_row["block_length_initializations"]),
        primary_row["analysis_role"],
    )
    _require_exact(
        actual_primary_key,
        expected_primary_key,
        "E3 canonical primary-estimand row",
    )
    _crosscheck_e3_primary_manifest(primary_row, manifest)
    block13 = effects.loc[effects["block_length_initializations"].eq(13)]
    if not block13["analysis_role"].eq("primary_uncertainty").all():
        raise ValueError("E3 block-13 rows must be primary_uncertainty")
    if (block13["ci_lower_2p5"] > block13["point_effect"]).any() or (
        block13["point_effect"] > block13["ci_upper_97p5"]
    ).any():
        raise ValueError("E3 point effects must lie inside their block-13 intervals")
    _crosscheck_e3_effects(pooled, block13)


def _crosscheck_e3_primary_manifest(
    primary_row: pd.Series,
    manifest: Mapping[str, Any],
) -> None:
    structured = manifest.get("primary_estimand")
    if not isinstance(structured, dict):
        raise ValueError("E3 manifest lacks its structured primary_estimand")
    exact_fields = {
        "comparison": "selected_adapter_vs_raw_fuxi",
        "metric": "rmse",
        "definition": "reference_minus_candidate",
        "circular_block_length_initializations": 13,
        "bootstrap_draws": 2000,
        "all_six_leads_attached": True,
    }
    for key, expected in exact_fields.items():
        _require_exact(structured.get(key), expected, f"E3 manifest primary {key}")
    row_fields = {
        "point_effect": "point_effect",
        "ci_lower_2p5": "ci_lower_2p5",
        "ci_upper_97p5": "ci_upper_97p5",
        "probability_effect_gt_zero": "probability_effect_gt_zero",
    }
    for manifest_key, row_key in row_fields.items():
        manifest_value = structured.get(manifest_key)
        if not isinstance(manifest_value, (int, float)) or not math.isclose(
            float(manifest_value),
            float(primary_row[row_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"E3 manifest primary {manifest_key} differs from the effect row"
            )


def _crosscheck_e3_effects(pooled: pd.DataFrame, effects: pd.DataFrame) -> None:
    rows = pooled.set_index("method")
    metric_column = {
        "rmse": "rmse_mean",
        "mae": "mae_mean",
        "bias": "bias_mean",
        "absolute_bias": "absolute_bias_mean",
        "acc": "acc_mean",
    }
    for row in effects.itertuples(index=False):
        candidate = rows.loc[row.candidate]
        reference = rows.loc[row.reference]
        column = metric_column[row.metric]
        if row.metric in ("rmse", "mae", "absolute_bias"):
            expected = reference[column] - candidate[column]
            required_definition = "reference_minus_candidate"
        else:
            expected = candidate[column] - reference[column]
            required_definition = "candidate_minus_reference"
        if row.effect_definition != required_definition:
            raise ValueError(f"E3 {row.metric} effect direction is inconsistent")
        if not math.isclose(
            float(row.point_effect), float(expected), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"E3 effect does not reproduce pooled metrics: {row.candidate} vs "
                f"{row.reference}, {row.metric}"
            )


def load_verified_inputs(
    e2_directory: Path,
    e3_directory: Path,
    *,
    e2_manifest_sha256: str,
    e3_manifest_sha256: str,
) -> EvidenceInputs:
    """Verify both sources completely before loading any metric CSV."""

    e2_directory = Path(e2_directory).resolve()
    e3_directory = Path(e3_directory).resolve()
    e2_manifest, e2_hash = _read_hash_pinned_manifest(
        e2_directory / "manifest.json", e2_manifest_sha256
    )
    e3_manifest, e3_hash = _read_hash_pinned_manifest(
        e3_directory / "manifest.json", e3_manifest_sha256
    )
    _validate_e2_manifest(e2_manifest)
    _validate_e3_manifest(e3_manifest, e2_hash)

    # Every CSV is captured once. The exact checksum-verified bytes, rather than
    # a reopened path, are passed to pandas below.
    e2_verified, e2_payloads = _read_verified_artifacts(
        e2_directory, e2_manifest, E2_REQUIRED_ARTIFACTS
    )
    e3_verified, e3_payloads = _read_verified_artifacts(
        e3_directory, e3_manifest, E3_REQUIRED_ARTIFACTS
    )

    e2_pooled = pd.read_csv(io.BytesIO(e2_payloads[E2_REQUIRED_ARTIFACTS[0]]))
    e2_by_lead = pd.read_csv(io.BytesIO(e2_payloads[E2_REQUIRED_ARTIFACTS[1]]))
    e2_effects = pd.read_csv(io.BytesIO(e2_payloads[E2_REQUIRED_ARTIFACTS[2]]))
    e3_summary = pd.read_csv(io.BytesIO(e3_payloads[E3_REQUIRED_ARTIFACTS[0]]))
    e3_effects = pd.read_csv(io.BytesIO(e3_payloads[E3_REQUIRED_ARTIFACTS[1]]))
    _validate_e2_tables(e2_pooled, e2_by_lead, e2_effects)
    _validate_e3_tables(e3_summary, e3_effects, e3_manifest)
    return EvidenceInputs(
        e2_directory=e2_directory,
        e3_directory=e3_directory,
        e2_manifest=e2_manifest,
        e3_manifest=e3_manifest,
        e2_manifest_sha256=e2_hash,
        e3_manifest_sha256=e3_hash,
        verified_source_artifacts={"e2": e2_verified, "e3": e3_verified},
        e2_pooled=e2_pooled,
        e2_by_lead=e2_by_lead,
        e2_effects=e2_effects,
        e3_summary=e3_summary,
        e3_effects=e3_effects,
    )


def _method_order(frame: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = pd.Categorical(
        result["method"], categories=methods, ordered=True
    )
    return result.sort_values("method").reset_index(drop=True)


def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(
        directory / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        directory / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": stem.replace("_", " ").title(),
            "Creator": "build_paper_evidence_package.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="both", color="#D9D9D9", linewidth=0.55, alpha=0.65)
    axis.set_axisbelow(True)


def _plot_pooled_tradeoff(
    frame: pd.DataFrame,
    methods: Sequence[str],
    *,
    title: str,
    subtitle: str,
    columns: Mapping[str, str],
    output: Path,
    stem: str,
) -> None:
    ordered = _method_order(frame, methods)
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.75))
        figure.subplots_adjust(
            left=0.16, right=0.985, bottom=0.17, top=0.74, wspace=0.29
        )
        y = np.arange(len(ordered))
        axis = axes[0]
        for index, row in ordered.iterrows():
            method = str(row["method"])
            axis.plot(
                [row[columns["mae"]], row[columns["rmse"]]],
                [index, index],
                color=METHOD_COLORS[method],
                linewidth=1.5,
                alpha=0.75,
            )
            axis.scatter(
                row[columns["rmse"]],
                index,
                color=METHOD_COLORS[method],
                marker="o",
                s=35,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            axis.scatter(
                row[columns["mae"]],
                index,
                color=METHOD_COLORS[method],
                marker="s",
                s=31,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        axis.set_yticks(y, [METHOD_LABELS[str(value)] for value in ordered["method"]])
        axis.invert_yaxis()
        axis.set_xlabel("Error (mm/day; lower is better)\nCircle = RMSE · square = MAE")
        axis.set_title("a  Pooled error")
        _clean_axis(axis)

        axis = axes[1]
        for _, row in ordered.iterrows():
            method = str(row["method"])
            x_value = (
                abs(float(row[columns["bias"]]))
                if columns["bias"] == "bias_mm_day"
                else float(row[columns["bias"]])
            )
            y_value = float(row[columns["acc"]])
            axis.scatter(
                x_value,
                y_value,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=52,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            if method == "raw_identity":
                annotation = {"xytext": (-4, 4), "ha": "right"}
            elif method == "log_bias":
                annotation = {"xytext": (4, -10), "ha": "left"}
            else:
                annotation = {"xytext": (4, 4), "ha": "left"}
            axis.annotate(
                SHORT_LABELS[method],
                (x_value, y_value),
                xytext=annotation["xytext"],
                textcoords="offset points",
                fontsize=7.2,
                color=METHOD_COLORS[method],
                ha=annotation["ha"],
            )
        axis.set_xlabel("Absolute bias (mm/day; lower is better)")
        axis.set_ylabel("ACC (higher is better)")
        axis.set_title("b  Calibration–pattern trade-off")
        _clean_axis(axis)

        figure.suptitle(title, y=0.97, fontweight="semibold")
        figure.text(
            0.5,
            0.895,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#555555",
        )
        _save_figure(figure, output, stem)


def _plot_imd_by_lead(frame: pd.DataFrame, output: Path) -> None:
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.65))
        figure.subplots_adjust(
            left=0.09, right=0.985, bottom=0.15, top=0.68, wspace=0.20
        )
        for method in E2_METHODS:
            subset = frame.loc[frame["method"].eq(method)].sort_values("lead_week")
            for axis, column in zip(axes, ("rmse_mm_day", "acc")):
                axis.plot(
                    subset["lead_week"],
                    subset[column],
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.55,
                    markersize=4.3,
                )
        axes[0].set_title("a  RMSE by lead")
        axes[0].set_ylabel("RMSE (mm/day; lower is better)")
        axes[1].set_title("b  Spatial anomaly correlation by lead")
        axes[1].set_ylabel("ACC (higher is better)")
        for axis in axes:
            axis.set_xlabel("Lead week")
            axis.set_xticks(range(1, 7))
            _clean_axis(axis)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.855),
        )
        figure.suptitle(
            "Lead-wise IMD-grid performance",
            y=0.985,
            fontweight="semibold",
        )
        figure.text(
            0.5,
            0.91,
            "2022–2024 retrospective development audit · 100 starts per lead",
            ha="center",
            fontsize=8.0,
            color="#555555",
        )
        _save_figure(figure, output, "figure_02_imd_by_lead")


def _comparison_label(candidate: str, reference: str) -> str:
    return f"{SHORT_LABELS[candidate]} vs {SHORT_LABELS[reference]}"


def _forest_plot(
    effects: pd.DataFrame,
    *,
    metric_column: str,
    candidate_column: str,
    reference_column: str,
    point_column: str,
    title: str,
    subtitle: str,
    output: Path,
    stem: str,
) -> None:
    metric_key = {
        "rmse_mm_day": "rmse",
        "mae_mm_day": "mae",
        "bias_mm_day": "absolute_bias",
        "acc": "acc",
    }
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(2, 2, figsize=(7.35, 5.55), constrained_layout=True)
        for panel, (axis, (metric, metric_title, unit)) in enumerate(
            zip(axes.flat, PLOT_METRICS)
        ):
            if metric_column == "source_metric":
                source_metric = next(
                    source
                    for source, normalized in metric_key.items()
                    if normalized == metric
                )
            else:
                source_metric = metric
            subset = effects.loc[effects[metric_column].eq(source_metric)].copy()
            y = np.arange(len(subset))
            for index, row in enumerate(subset.itertuples(index=False)):
                candidate = str(getattr(row, candidate_column))
                point = float(getattr(row, point_column))
                lower = float(row.ci_lower_2p5)
                upper = float(row.ci_upper_97p5)
                axis.errorbar(
                    point,
                    index,
                    xerr=np.array([[point - lower], [upper - point]]),
                    fmt=METHOD_MARKERS[candidate],
                    color=METHOD_COLORS[candidate],
                    markersize=5.0,
                    elinewidth=1.25,
                    capsize=2.2,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                )
            labels = [
                _comparison_label(
                    str(row[candidate_column]), str(row[reference_column])
                )
                for _, row in subset.iterrows()
            ]
            axis.set_yticks(y, labels if panel % 2 == 0 else [])
            axis.invert_yaxis()
            axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle="--")
            axis.set_xlabel(f"Effect ({unit}); positive favors candidate")
            axis.set_title(f"{chr(97 + panel)}  {metric_title}")
            _clean_axis(axis)
        figure.suptitle(title, y=1.045, fontweight="semibold")
        figure.text(0.5, 1.005, subtitle, ha="center", fontsize=8.0, color="#555555")
        _save_figure(figure, output, stem)


def _select_e2_forest(effects: pd.DataFrame) -> pd.DataFrame:
    ranking = {
        comparison: index for index, comparison in enumerate(E2_FOREST_COMPARISONS)
    }
    selected = effects.loc[
        effects["scope_type"].eq("pooled")
        & effects[["candidate", "baseline"]]
        .apply(tuple, axis=1)
        .isin(E2_FOREST_COMPARISONS)
        & effects["source_metric"].isin(
            ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
        )
    ].copy()
    selected["_comparison_order"] = [
        ranking[(a, b)] for a, b in zip(selected["candidate"], selected["baseline"])
    ]
    metric_order = {
        value: index
        for index, value in enumerate(
            ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
        )
    }
    selected["_metric_order"] = selected["source_metric"].map(metric_order)
    return selected.sort_values(["_metric_order", "_comparison_order"]).reset_index(
        drop=True
    )


def _select_e3_block13(effects: pd.DataFrame) -> pd.DataFrame:
    ranking = {comparison: index for index, comparison in enumerate(E3_COMPARISONS)}
    selected = effects.loc[effects["block_length_initializations"].eq(13)].copy()
    selected["_comparison_order"] = [
        ranking[(a, b)] for a, b in zip(selected["candidate"], selected["reference"])
    ]
    metric_order = {
        value: index
        for index, value in enumerate(("rmse", "mae", "absolute_bias", "acc", "bias"))
    }
    selected["_metric_order"] = selected["metric"].map(metric_order)
    return selected.sort_values(["_metric_order", "_comparison_order"]).reset_index(
        drop=True
    )


def _markdown_value(value: Any, precision: int = 4) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value).replace("|", "\\|")


def _write_markdown_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    note: str,
    precision: Mapping[str, int] | None = None,
) -> None:
    precision = precision or {}
    columns = list(frame.columns)
    lines = [f"# {title}", "", note, "", "| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in frame.itertuples(index=False, name=None):
        cells = [
            _markdown_value(value, precision.get(column, 4))
            for column, value in zip(columns, row)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_pair(
    frame: pd.DataFrame,
    output: Path,
    stem: str,
    *,
    title: str,
    note: str,
    precision: Mapping[str, int] | None = None,
) -> None:
    frame.to_csv(output / f"{stem}.csv", index=False, float_format="%.12g")
    _write_markdown_table(
        frame,
        output / f"{stem}.md",
        title=title,
        note=note,
        precision=precision,
    )


def _publication_tables(
    inputs: EvidenceInputs, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    e2_pooled = _method_order(inputs.e2_pooled, E2_METHODS)
    e2_pooled_table = pd.DataFrame(
        {
            "method": [METHOD_LABELS[str(value)] for value in e2_pooled["method"]],
            "rmse_mm_day": e2_pooled["rmse_mm_day"].astype(float),
            "mae_mm_day": e2_pooled["mae_mm_day"].astype(float),
            "bias_mm_day": e2_pooled["bias_mm_day"].astype(float),
            "absolute_bias_mm_day": e2_pooled["bias_mm_day"].abs().astype(float),
            "acc": e2_pooled["acc"].astype(float),
            "case_leads": e2_pooled["case_lead_count"].astype(int),
        }
    )
    _write_table_pair(
        e2_pooled_table,
        output,
        "table_01_imd_pooled",
        title="IMD-grid pooled performance",
        note=(
            "2022–2024 retrospective development audit. Metrics are pooled over "
            "100 initializations × six leads; this is not an untouched final test."
        ),
    )

    e2_lead = inputs.e2_by_lead.copy()
    e2_lead["method"] = e2_lead["method"].map(METHOD_LABELS)
    e2_lead_table = e2_lead[
        [
            "lead_week",
            "method",
            "rmse_mm_day",
            "mae_mm_day",
            "bias_mm_day",
            "acc",
            "case_lead_count",
        ]
    ].rename(columns={"case_lead_count": "cases"})
    _write_table_pair(
        e2_lead_table,
        output,
        "table_02_imd_by_lead",
        title="IMD-grid performance by lead week",
        note="Retrospective 2022–2024 development-audit metrics; 100 matched starts at every lead.",
    )

    e2_forest = _select_e2_forest(inputs.e2_effects)
    e2_effect_table = pd.DataFrame(
        {
            "candidate": e2_forest["candidate"].map(METHOD_LABELS),
            "reference": e2_forest["baseline"].map(METHOD_LABELS),
            "metric": e2_forest["source_metric"].replace(
                {
                    "rmse_mm_day": "RMSE",
                    "mae_mm_day": "MAE",
                    "bias_mm_day": "absolute bias",
                    "acc": "ACC",
                }
            ),
            "effect": e2_forest["effect"].astype(float),
            "ci_lower_2p5": e2_forest["ci_lower_2p5"].astype(float),
            "ci_upper_97p5": e2_forest["ci_upper_97p5"].astype(float),
            "probability_improved": e2_forest["bootstrap_probability_improved"].astype(
                float
            ),
            "starts": e2_forest["n_starts"].astype(int),
            "leads_per_start": e2_forest["n_leads_per_start"].astype(int),
        }
    )
    _write_table_pair(
        e2_effect_table,
        output,
        "table_03_imd_paired_effects",
        title="IMD-grid paired effects",
        note=(
            "Effects are baseline minus candidate for RMSE, MAE, and absolute bias, "
            "and candidate minus baseline for ACC; positive therefore favors the "
            "candidate. Descriptive 95% percentile intervals use 10,000 shared, "
            "year-stratified circular moving-block resamples of 13 initializations "
            "with all six leads attached."
        ),
    )

    e3_pooled = _method_order(
        inputs.e3_summary.loc[inputs.e3_summary["scope_type"].eq("pooled")],
        E3_METHODS,
    )
    e3_pooled_table = pd.DataFrame(
        {
            "method": [METHOD_LABELS[str(value)] for value in e3_pooled["method"]],
            "rmse_mm_day": e3_pooled["rmse_mean"].astype(float),
            "mae_mm_day": e3_pooled["mae_mean"].astype(float),
            "bias_mm_day": e3_pooled["bias_mean"].astype(float),
            "absolute_bias_mm_day": e3_pooled["absolute_bias_mean"].astype(float),
            "acc": e3_pooled["acc_mean"].astype(float),
            "initializations": e3_pooled["initializations"].astype(int),
            "case_leads": e3_pooled["case_leads"].astype(int),
        }
    )
    _write_table_pair(
        e3_pooled_table,
        output,
        "table_04_station_pooled",
        title="Rain-gauge external-target pooled performance",
        note=(
            "Frozen 2024 external-observational-target sensitivity: 30 starts × six leads. "
            "This is not an untouched temporal final test."
        ),
    )

    e3_block13 = _select_e3_block13(inputs.e3_effects)
    e3_effect_table = pd.DataFrame(
        {
            "candidate": e3_block13["candidate"].map(METHOD_LABELS),
            "reference": e3_block13["reference"].map(METHOD_LABELS),
            "metric": e3_block13["metric"].replace(
                {
                    "rmse": "RMSE",
                    "mae": "MAE",
                    "absolute_bias": "absolute bias",
                    "acc": "ACC",
                    "bias": "signed bias",
                }
            ),
            "effect_definition": e3_block13["effect_definition"],
            "effect": e3_block13["point_effect"].astype(float),
            "ci_lower_2p5": e3_block13["ci_lower_2p5"].astype(float),
            "ci_upper_97p5": e3_block13["ci_upper_97p5"].astype(float),
            "probability_effect_gt_zero": e3_block13[
                "probability_effect_gt_zero"
            ].astype(float),
            "primary_estimand": e3_block13["primary_estimand"].astype(bool),
            "block_length_initializations": e3_block13[
                "block_length_initializations"
            ].astype(int),
            "bootstrap_draws": e3_block13["bootstrap_draws"].astype(int),
        }
    )
    _write_table_pair(
        e3_effect_table,
        output,
        "table_05_station_block13_effects",
        title="Rain-gauge external-target paired effects",
        note=(
            "Primary block-length-13 uncertainty rows only. Positive favors the candidate "
            "for RMSE, MAE, absolute bias, and ACC; signed-bias effects retain their "
            "candidate-minus-reference definition."
        ),
    )
    return e2_forest, e3_block13


def _write_readme(inputs: EvidenceInputs, output: Path) -> None:
    e2 = inputs.e2_pooled.set_index("method")
    e3 = inputs.e3_summary.loc[inputs.e3_summary["scope_type"].eq("pooled")].set_index(
        "method"
    )
    e2_rmse = e2.loc["raw_fuxi", "rmse_mm_day"] - e2.loc["raw_identity", "rmse_mm_day"]
    e2_mae = e2.loc["raw_fuxi", "mae_mm_day"] - e2.loc["raw_identity", "mae_mm_day"]
    e2_acc = e2.loc["raw_identity", "acc"] - e2.loc["raw_fuxi", "acc"]
    e3_rmse = e3.loc["raw_fuxi", "rmse_mean"] - e3.loc["raw_identity", "rmse_mean"]
    e3_mae = e3.loc["raw_fuxi", "mae_mean"] - e3.loc["raw_identity", "mae_mean"]
    e3_acc = e3.loc["raw_identity", "acc_mean"] - e3.loc["raw_fuxi", "acc_mean"]
    text = f"""# Sealed E2/E3 paper evidence package

This is a venue-neutral reporting package built only from checksum-verified,
already-computed metric tables. This reporting builder did **not** open
prediction stores, forecast arrays, target arrays, raw station files, or any
2025 data. Upstream, E3 scanned a mixed station container containing 45,910
unselected 2025+ rows; it selected, materialized, and scored no 2025 station
value because rainfall conversion followed the exact 2024 date/station filter.

## Evidence boundary

- **E2 — retrospective development audit (2022–2024):** 100 matched
  initializations × six leads. It is strong held-period development evidence,
  not an untouched final test.
- **E3 — external-target sensitivity (2024 rain gauges):** 30 matched
  initializations × six leads on a frozen observational target. It tests target
  robustness, not temporal independence.
- The E2 intervals are descriptive year-stratified circular block-13 intervals.
  The E3 figure and main table use the preregistered circular block length 13.

## Compact readout

On the IMD-grid audit, the raw-identity adapter improves raw FuXi RMSE by
{e2_rmse:.3f} mm/day, MAE by {e2_mae:.3f} mm/day, and ACC by {e2_acc:.4f};
its negative mean bias is the main calibration weakness. The raw-mean projection
repairs IMD mean calibration but sacrifices MAE and some ACC relative to raw
identity, so it is a Pareto diagnostic rather than a universal replacement.

Against the external station target, raw identity improves raw FuXi RMSE by
{e3_rmse:.3f} mm/day, MAE by {e3_mae:.3f} mm/day, and ACC by {e3_acc:.4f}.
The raw-mean projection loses most of those gains, demonstrating that its
calibration benefit is target-dependent.

## Files

- `figure_01_imd_pooled_tradeoff.*`: pooled IMD error and calibration–pattern trade-off.
- `figure_02_imd_by_lead.*`: IMD RMSE and ACC by lead week.
- `figure_03_imd_paired_effects.*`: E2 paired block-13 effect forest.
- `figure_04_station_pooled.*`: pooled external-station comparison.
- `figure_05_station_block13_effects.*`: E3 paired block-13 effect forest.
- `table_01` through `table_05`: matching machine-readable CSV and manuscript-friendly Markdown tables.
- `FIGURE_CAPTIONS.md`: evidence-bounded figure captions.
- `PACKAGE_MANIFEST.json`: source pins, access boundary, and output checksums.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _write_captions(output: Path) -> None:
    text = """# Figure captions

## Figure 1 — IMD-grid pooled performance and trade-off

Pooled deterministic rainfall performance over the 2022–2024 retrospective
development audit (100 matched initializations and six lead weeks). The left
panel reports RMSE and MAE; the right panel places spatial anomaly correlation
against absolute mean bias. Raw identity reduces error and increases ACC but
has a dry-bias trade-off, while the post-hoc raw-mean projection restores mean
calibration at the cost of MAE and some ACC. This is not an untouched final test.

## Figure 2 — IMD-grid performance by lead

RMSE and spatial anomaly correlation for lead weeks 1–6 in the retrospective
2022–2024 development audit. Every point summarizes the same 100 matched
initializations at that lead.

## Figure 3 — IMD-grid paired effects

Effects are oriented as baseline minus candidate for RMSE, MAE, and absolute
bias, and as candidate minus baseline for ACC; positive values therefore favor
the named candidate in every panel. Points are observed paired effects; bars
are descriptive 95% percentile intervals from 10,000 shared, year-stratified
circular moving-block resamples of 13 initialization dates, with all six leads
retained per sampled start.

## Figure 4 — External rain-gauge target sensitivity

Pooled scores on the frozen 2024 station target (30 matched initializations and
six lead weeks). The comparison is an external-observational-target sensitivity,
not an untouched temporal final test. Raw identity retains error gains, whereas
the IMD raw-mean projection does not transfer as a general calibration remedy.

## Figure 5 — External-target paired effects

Paired effects on the 2024 station target. Points are observed effects and bars
are 95% percentile intervals from 2,000 shared circular moving-block resamples
using the preregistered primary block length of 13 initialization dates, with all
six leads retained. Positive values favor the named candidate.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(text, encoding="utf-8")


def _all_nonmanifest_files(directory: Path) -> list[Path]:
    root_manifest = directory / "PACKAGE_MANIFEST.json"
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != root_manifest
    ]


def verify_package(directory: Path) -> Mapping[str, Any]:
    """Re-hash every declared output and reject missing or extra files."""

    directory = Path(directory).resolve()
    manifest_path = directory / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("paper-evidence package manifest is not complete")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("paper-evidence package lacks an artifact checksum mapping")
    actual_paths = {
        str(path.relative_to(directory)): path
        for path in _all_nonmanifest_files(directory)
    }
    if set(actual_paths) != set(declared):
        missing = sorted(set(declared) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(declared))
        raise ValueError(
            f"package file inventory differs; missing={missing}, extra={extra}"
        )
    for relative, path in actual_paths.items():
        expected = _normalise_sha256(str(declared[relative]), relative)
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"package artifact SHA-256 differs: {relative}")
    return manifest


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish ``staging`` while refusing a raced destination.

    Linux ``renameat2(RENAME_NOREPLACE)`` closes the gap between an existence
    check and publication. The builder fails closed if that primitive is not
    available rather than weakening its no-overwrite guarantee.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-clobber publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "refusing to replace a destination created during publication",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def build_package(
    e2_directory: Path,
    e3_directory: Path,
    output_directory: Path,
    *,
    e2_manifest_sha256: str,
    e3_manifest_sha256: str,
) -> Path:
    """Build and verify a fresh, atomic paper-evidence directory."""

    e2_directory = Path(e2_directory).resolve()
    e3_directory = Path(e3_directory).resolve()
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_directory}"
        )
    for source in (e2_directory, e3_directory):
        if output_directory == source or source in output_directory.parents:
            raise ValueError(
                "output directory may not be inside a canonical source directory"
            )

    inputs = load_verified_inputs(
        e2_directory,
        e3_directory,
        e2_manifest_sha256=e2_manifest_sha256,
        e3_manifest_sha256=e3_manifest_sha256,
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        e2_forest, e3_block13 = _publication_tables(inputs, staging)
        _plot_pooled_tradeoff(
            inputs.e2_pooled,
            E2_METHODS,
            title="IMD-grid pooled performance and trade-off",
            subtitle="2022–2024 retrospective development audit · 100 starts × 6 leads",
            columns={
                "rmse": "rmse_mm_day",
                "mae": "mae_mm_day",
                "bias": "bias_mm_day",
                "acc": "acc",
            },
            output=staging,
            stem="figure_01_imd_pooled_tradeoff",
        )
        _plot_imd_by_lead(inputs.e2_by_lead, staging)
        _forest_plot(
            e2_forest,
            metric_column="source_metric",
            candidate_column="candidate",
            reference_column="baseline",
            point_column="effect",
            title="Paired effects on the IMD-grid target",
            subtitle="Retrospective development audit · 10,000 year-stratified circular block-13 resamples",
            output=staging,
            stem="figure_03_imd_paired_effects",
        )
        e3_pooled = inputs.e3_summary.loc[inputs.e3_summary["scope_type"].eq("pooled")]
        _plot_pooled_tradeoff(
            e3_pooled,
            E3_METHODS,
            title="Performance on an external rain-gauge target",
            subtitle="Frozen 2024 external-target sensitivity · 30 starts × 6 leads",
            columns={
                "rmse": "rmse_mean",
                "mae": "mae_mean",
                "bias": "absolute_bias_mean",
                "acc": "acc_mean",
            },
            output=staging,
            stem="figure_04_station_pooled",
        )
        e3_plot = e3_block13.loc[
            e3_block13["metric"].isin(("rmse", "mae", "absolute_bias", "acc"))
        ]
        _forest_plot(
            e3_plot,
            metric_column="metric",
            candidate_column="candidate",
            reference_column="reference",
            point_column="point_effect",
            title="Paired effects on the external rain-gauge target",
            subtitle="External-target sensitivity · 2,000 circular block-13 resamples",
            output=staging,
            stem="figure_05_station_block13_effects",
        )
        _write_readme(inputs, staging)
        _write_captions(staging)
        code_directory = staging / "code"
        code_directory.mkdir()
        builder_snapshot = code_directory / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), builder_snapshot)

        artifacts = {
            str(path.relative_to(staging)): _sha256_file(path)
            for path in _all_nonmanifest_files(staging)
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": _utc_now(),
            "package_role": "venue-neutral sealed E2/E3 paper evidence",
            "publication": {
                "atomic": True,
                "no_clobber": True,
                "primitive": "Linux renameat2(RENAME_NOREPLACE)",
            },
            "builder": {
                "source": str(builder_snapshot.relative_to(staging)),
                "sha256": _sha256_file(builder_snapshot),
            },
            "evidence_labels": {
                "e2": "2022-2024 retrospective development audit; not an untouched final test",
                "e3": "2024 external-observational-target sensitivity; not an untouched temporal final test",
            },
            "access_boundary": {
                "scope": "this reporting-builder invocation only",
                "metric_artifact_only": True,
                "builder_opened": {
                    "raw_data": False,
                    "prediction_store": False,
                    "forecast_array": False,
                    "target_array": False,
                    "raw_station_file": False,
                    "2025_data": False,
                },
                "upstream_e3_disclosure": {
                    "mixed_station_container_scanned": True,
                    "container_rows_scanned": 543_518,
                    "container_max_date": "2025-02-10",
                    "unselected_2025_plus_rows": 45_910,
                    "2025_station_values_selected": 0,
                    "2025_station_values_materialized": 0,
                    "2025_station_values_scored": 0,
                    "filtering_contract": (
                        "rainfall conversion occurred only after exact 2024 "
                        "date and station filtering"
                    ),
                },
                "loaded_source_files": [
                    *[f"E2/{value}" for value in E2_REQUIRED_ARTIFACTS],
                    *[f"E3/{value}" for value in E3_REQUIRED_ARTIFACTS],
                ],
            },
            "sources": {
                "e2": {
                    "directory": str(inputs.e2_directory),
                    "manifest_sha256": inputs.e2_manifest_sha256,
                    "verified_artifacts": dict(inputs.verified_source_artifacts["e2"]),
                },
                "e3": {
                    "directory": str(inputs.e3_directory),
                    "manifest_sha256": inputs.e3_manifest_sha256,
                    "verified_artifacts": dict(inputs.verified_source_artifacts["e3"]),
                    "e2_manifest_binding_verified": True,
                },
            },
            "uncertainty": {
                "e2": "10,000 shared year-stratified circular moving-block resamples; block length 13; six leads attached",
                "e3": "2,000 shared circular moving-block resamples; primary block length 13; six leads attached",
            },
            "artifacts": artifacts,
        }
        (staging / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_package(staging)
        _publish_directory_noreplace(staging, output_directory)
        verify_package(output_directory)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2", type=Path, help="Canonical completed E2 directory")
    parser.add_argument("--e3", type=Path, help="Canonical completed E3 directory")
    parser.add_argument(
        "--e2-manifest-sha256",
        help="Externally pinned SHA-256 of E2/manifest.json",
    )
    parser.add_argument(
        "--e3-manifest-sha256",
        help="Externally pinned SHA-256 of E3/manifest.json",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh output directory"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing --output package instead of building it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        manifest = verify_package(args.output)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "output": str(args.output.resolve()),
                    "artifacts": len(manifest["artifacts"]),
                    "package_manifest_sha256": _sha256_file(
                        args.output.resolve() / "PACKAGE_MANIFEST.json"
                    ),
                },
                indent=2,
            )
        )
        return 0
    required = {
        "--e2": args.e2,
        "--e3": args.e3,
        "--e2-manifest-sha256": args.e2_manifest_sha256,
        "--e3-manifest-sha256": args.e3_manifest_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        _parser().error(f"building requires: {', '.join(missing)}")
    result = build_package(
        args.e2,
        args.e3,
        args.output,
        e2_manifest_sha256=args.e2_manifest_sha256,
        e3_manifest_sha256=args.e3_manifest_sha256,
    )
    manifest = verify_package(result)
    print(
        json.dumps(
            {
                "status": "complete_and_verified",
                "output": str(result),
                "artifacts": len(manifest["artifacts"]),
                "package_manifest_sha256": _sha256_file(
                    result / "PACKAGE_MANIFEST.json"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
