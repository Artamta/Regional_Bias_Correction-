#!/usr/bin/env python3
"""Build a compact three-figure paper package from frozen metric evidence.

This is a standalone, reporting-only program.  It accepts externally pinned
manifests for the raw-identity training run, the E2 IMD-grid audit, and the E3
gauge-derived-cell target sensitivity.  It verifies the exact bytes of every loaded CSV
against the corresponding source manifest before parsing those same in-memory
bytes.  It never opens checkpoints, prediction stores, arrays, raw observations,
raw station files, or 2025 data.

The destination is created atomically with Linux ``renameat2`` and
``RENAME_NOREPLACE``.  An existing or concurrently created destination is never
overwritten.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


TRAINING_REQUIRED_ARTIFACTS: tuple[str, ...] = ()
E2_REQUIRED_ARTIFACTS = (
    "metrics/summary_pooled.csv",
    "metrics/summary_by_lead.csv",
    "metrics/paired_block_bootstrap_effects.csv",
    "metrics/intensity_strata_metrics.csv",
    "metrics/intensity_paired_block_bootstrap_effects.csv",
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
E3_PRIMARY_TUPLE = (
    "selected_adapter_vs_raw_fuxi",
    "selected_adapter",
    "raw_fuxi",
    "rmse_mm_day",
)
SELECTED_MODEL = "normal_climo_model"
SELECTED_ARCHITECTURE = "fixed_climatology_allweeks"
FROZEN_MODEL_CODE_SHA256 = (
    "35a70b0e05043841c7e5b62793da05819f4f2b7e5e3f7a8e375f3bd76941f569"
)
PARAMETER_COUNT = 144_689
SEEDS = (42, 43, 44)
CONSUMED_CHANNELS = (
    "log_fuxi_mean",
    "log_fuxi_spread",
    "log_imd_calendar_climatology",
    "latitude",
    "longitude",
    "season_sin",
    "season_cos",
    "lead_week",
    "support",
    "explicit_log_fuxi_minus_imd_climatology",
    "fuxi_t2m_weekly",
)
INTENSITY_ORDER = (
    "dry_lt1",
    "light_1_5",
    "moderate_5_10",
    "heavy_10_20",
    "extreme_ge20",
)
INTENSITY_LABELS = {
    "dry_lt1": "<1",
    "light_1_5": "1–5",
    "moderate_5_10": "5–10",
    "heavy_10_20": "10–20",
    "extreme_ge20": "≥20",
}

RAW_COLOR = "#4D4D4D"
IDENTITY_COLOR = "#0072B2"
PROJECTION_COLOR = "#009E73"
IMD_COLOR = "#0072B2"
STATION_COLOR = "#D55E00"
LIMIT_COLOR = "#F3C6C1"
PLOT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 12.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class CoreInputs:
    """All source manifests and tables after byte-level verification."""

    training_directory: Path
    e2_directory: Path
    e3_directory: Path
    training_manifest: Mapping[str, Any]
    e2_manifest: Mapping[str, Any]
    e3_manifest: Mapping[str, Any]
    training_manifest_sha256: str
    e2_manifest_sha256: str
    e3_manifest_sha256: str
    verified_source_artifacts: Mapping[str, Mapping[str, str]]
    e2_pooled: pd.DataFrame
    e2_by_lead: pd.DataFrame
    e2_effects: pd.DataFrame
    e2_intensity: pd.DataFrame
    e2_intensity_effects: pd.DataFrame
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
    """Hash the exact manifest bytes before parsing them."""

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
    """Read each required CSV once and bind parsing to those verified bytes."""

    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError(f"source manifest lacks an artifacts mapping: {root}")
    verified: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for relative in required:
        expected_value = declared.get(relative)
        if not isinstance(expected_value, str):
            raise ValueError(f"source manifest does not declare {relative}")
        expected = _normalise_sha256(expected_value, relative)
        payload = _contained_regular_file(root, relative).read_bytes()
        actual = _sha256_bytes(payload)
        if actual != expected:
            raise ValueError(
                f"artifact SHA-256 differs for {relative}: "
                f"expected {expected}, got {actual}"
            )
        verified[relative] = actual
        payloads[relative] = payload
    return verified, payloads


def _parse_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} must be {expected!r}, got {value!r}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} must be finite numeric data")


def _validate_interval_rows(frame: pd.DataFrame, label: str) -> None:
    _require_finite(
        frame,
        ("effect", "ci_lower_2p5", "ci_upper_97p5"),
        label,
    )
    if (frame["ci_lower_2p5"] > frame["effect"]).any() or (
        frame["effect"] > frame["ci_upper_97p5"]
    ).any():
        raise ValueError(f"{label} has a point estimate outside its interval")


def _validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    _require_exact(manifest.get("status"), "complete", "training status")
    _require_exact(manifest.get("smoke"), False, "training smoke")
    _require_exact(manifest.get("observation_source"), "IMD", "training target")
    _require_exact(manifest.get("selected_model"), SELECTED_MODEL, "selected model")
    if not math.isclose(float(manifest.get("selected_alpha", math.nan)), 1.0):
        raise ValueError("selected alpha must be exactly 1.0")
    _require_exact(manifest.get("training_anchor"), "raw_fuxi", "training anchor")
    _require_exact(
        manifest.get("uses_fitted_log_bias_in_neural_training"),
        False,
        "fitted-log-bias training flag",
    )
    _require_exact(manifest.get("log_bias_role"), "reporting_only", "log-bias role")
    _require_exact(
        tuple(manifest.get("active_leads", ())),
        (1, 2, 3, 4, 5, 6),
        "active leads",
    )
    _require_exact(
        tuple(manifest.get("quarantined_final_initialization_years", ())),
        (2025,),
        "quarantined final year",
    )
    split_years = manifest.get("split_years")
    if not isinstance(split_years, dict):
        raise ValueError("training split_years is missing")
    _require_exact(
        tuple(split_years.get("train", ())), tuple(range(2002, 2018)), "train years"
    )
    _require_exact(
        tuple(split_years.get("validation", ())), (2018, 2019), "validation years"
    )
    _require_exact(
        tuple(split_years.get("test", ())), (2020, 2021), "exploratory test years"
    )
    _require_exact(
        manifest.get("split_counts"),
        {"train": 560, "validation": 70, "test": 70},
        "training split counts",
    )
    features = tuple(manifest.get("features", ()))
    _require_exact(features[: len(CONSUMED_CHANNELS)], CONSUMED_CHANNELS, "channels")
    if len(features) < len(CONSUMED_CHANNELS):
        raise ValueError("training manifest does not declare all consumed channels")
    code_hashes = manifest.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("training code hashes are missing")
    _require_exact(
        code_hashes.get("models.py"), FROZEN_MODEL_CODE_SHA256, "frozen model code"
    )
    training = manifest.get("training")
    if not isinstance(training, dict) or not isinstance(
        training.get(SELECTED_MODEL), dict
    ):
        raise ValueError("selected training metadata is missing")
    model = training[SELECTED_MODEL]
    _require_exact(model.get("architecture"), SELECTED_ARCHITECTURE, "architecture")
    _require_exact(model.get("parameter_count"), PARAMETER_COUNT, "parameter count")
    _require_exact(tuple(model.get("seeds", ())), SEEDS, "ensemble seeds")
    _require_exact(model.get("train_case_count"), 560, "model train cases")
    _require_exact(model.get("validation_case_count"), 70, "model validation cases")
    _require_exact(model.get("inactive_lead_count"), 0, "inactive lead count")
    _require_exact(model.get("dropout"), 0.3, "dropout")
    _require_exact(
        model.get("loss_coefficients"),
        {"smooth_l1": 0.75, "acc": 0.2, "bias": 0.05},
        "loss coefficients",
    )
    runs = model.get("runs")
    if not isinstance(runs, list) or tuple(run.get("seed") for run in runs) != SEEDS:
        raise ValueError("selected model must have exactly the three frozen seed runs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("training artifact map is missing")
    for run in runs:
        if int(run.get("best_epoch", -1)) < 0 or not np.isfinite(
            float(run.get("best_validation_loss", math.nan))
        ):
            raise ValueError("seed run has invalid validation selection metadata")
        checkpoint = str(run.get("checkpoint", ""))
        checkpoint_hash = _normalise_sha256(
            str(run.get("checkpoint_sha256", "")), checkpoint
        )
        _require_exact(
            artifacts.get(checkpoint), checkpoint_hash, "checkpoint hash declaration"
        )


def _validate_e2_manifest(
    manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    training_manifest_sha256: str,
) -> None:
    _require_exact(manifest.get("status"), "complete", "E2 status")
    _require_exact(manifest.get("canonical"), True, "E2 canonical")
    _require_exact(manifest.get("scientific_eligible"), True, "E2 eligibility")
    _require_exact(manifest.get("smoke"), False, "E2 smoke")
    _require_exact(
        tuple(manifest.get("audit_years", ())), (2022, 2023, 2024), "E2 years"
    )
    _require_exact(
        manifest.get("final_initialization_year_quarantined"), 2025, "E2 sealed year"
    )
    _require_exact(manifest.get("final_2025_store_opened"), False, "E2 2025 flag")
    _require_exact(tuple(manifest.get("methods", ())), E2_METHODS, "E2 methods")
    _require_exact(
        manifest.get("audit_counts"), {"2022": 35, "2023": 35, "2024": 30}, "E2 counts"
    )
    selection = manifest.get("raw_identity_selection")
    if not isinstance(selection, dict):
        raise ValueError("E2 raw-identity selection contract is missing")
    _require_exact(selection.get("model"), SELECTED_MODEL, "E2 selected model")
    _require_exact(selection.get("alpha"), 1.0, "E2 selected alpha")
    _require_exact(selection.get("retrained_for_audit"), False, "E2 retraining")
    _require_exact(selection.get("retuned_on_audit"), False, "E2 retuning")
    _require_exact(selection.get("training_anchor"), "raw_fuxi", "E2 anchor")
    _require_exact(
        selection.get("uses_fitted_log_bias_in_neural_training"),
        False,
        "E2 fitted-log-bias flag",
    )
    provenance = manifest.get("input_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("E2 input provenance is missing")
    _require_exact(
        provenance.get("raw_identity_manifest_sha256"),
        training_manifest_sha256,
        "E2 binding to raw-identity training manifest",
    )
    checkpoints = provenance.get("raw_identity_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError("E2 must bind exactly three raw-identity checkpoints")
    training_runs = training_manifest["training"][SELECTED_MODEL]["runs"]
    expected_by_seed: dict[int, str] = {}
    checkpoint_by_seed: dict[int, str] = {}
    for run in training_runs:
        seed = int(run["seed"])
        expected_by_seed[seed] = _normalise_sha256(
            str(run["checkpoint_sha256"]), f"training checkpoint seed {seed}"
        )
        checkpoint_by_seed[seed] = str(run["checkpoint"])
    observed_by_seed: dict[int, str] = {}
    for record in checkpoints:
        if not isinstance(record, dict):
            raise ValueError("E2 checkpoint provenance record must be a mapping")
        record_path = str(record.get("path", ""))
        matched_seeds = [
            seed
            for seed, checkpoint in checkpoint_by_seed.items()
            if record_path == checkpoint or record_path.endswith(f"/{checkpoint}")
        ]
        if len(matched_seeds) != 1:
            raise ValueError(
                "E2 checkpoint path does not identify exactly one validated "
                "training checkpoint"
            )
        seed = matched_seeds[0]
        if seed in observed_by_seed:
            raise ValueError("E2 checkpoint provenance repeats a training seed")
        observed_by_seed[seed] = _normalise_sha256(
            str(record.get("sha256", "")), f"E2 checkpoint seed {seed}"
        )
    _require_exact(
        observed_by_seed,
        expected_by_seed,
        "E2 checkpoint seed/SHA-256 mapping",
    )
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("E2 bootstrap contract is missing")
    _require_exact(bootstrap.get("draws"), 10_000, "E2 bootstrap draws")
    _require_exact(bootstrap.get("block_length_initializations"), 13, "E2 block length")
    _require_exact(bootstrap.get("seed"), 20260822, "E2 bootstrap seed")
    _require_exact(bootstrap.get("all_six_leads_retained"), True, "E2 lead retention")
    projection = manifest.get("projection")
    if not isinstance(projection, dict):
        raise ValueError("E2 projection contract is missing")
    _require_exact(projection.get("post_hoc"), True, "E2 projection role")
    _require_exact(projection.get("operational_claim"), False, "E2 projection claim")


def _validate_e3_manifest(manifest: Mapping[str, Any], e2_manifest_sha256: str) -> None:
    _require_exact(
        manifest.get("status"),
        "complete_frozen_external_target_sensitivity",
        "E3 status",
    )
    _require_exact(manifest.get("canonical_artifact"), True, "E3 canonical")
    _require_exact(manifest.get("training_performed"), False, "E3 training")
    _require_exact(
        manifest.get("selection_calibration_or_blending_performed"),
        False,
        "E3 selection/calibration",
    )
    for key in (
        "2025_metric_computed",
        "2025_prediction_opened",
        "2025_station_value_selected",
    ):
        _require_exact(manifest.get(key), False, f"E3 {key}")
    _require_exact(tuple(manifest.get("methods", ())), E3_METHODS, "E3 methods")
    _require_exact(
        manifest.get("extended_prediction_manifest_sha256"),
        e2_manifest_sha256,
        "E3 binding to E2 manifest",
    )
    dates = manifest.get("dates")
    if not isinstance(dates, dict):
        raise ValueError("E3 date contract is missing")
    _require_exact(tuple(dates.get("initialization_years", ())), (2024,), "E3 years")
    _require_exact(tuple(dates.get("lead_weeks", ())), (1, 2, 3, 4, 5, 6), "E3 leads")
    _require_exact(dates.get("initialization_count"), 30, "E3 starts")
    _require_exact(dates.get("case_leads"), 180, "E3 case leads")
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("E3 bootstrap contract is missing")
    _require_exact(bootstrap.get("draws"), 2_000, "E3 bootstrap draws")
    _require_exact(bootstrap.get("primary_block_length"), 13, "E3 block length")
    _require_exact(bootstrap.get("seed"), 20260822, "E3 bootstrap seed")
    primary = manifest.get("primary_estimand")
    if not isinstance(primary, dict):
        raise ValueError("E3 primary estimand is missing")
    _require_exact(
        primary.get("comparison"),
        "selected_adapter_vs_raw_fuxi",
        "E3 primary comparison",
    )
    _require_exact(primary.get("metric"), "rmse", "E3 primary metric")
    boundary = manifest.get("station_truth_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("E3 station boundary is missing")
    _require_exact(boundary.get("unselected_2025_plus_rows"), 45_910, "E3 2025+ rows")
    _require_exact(
        boundary.get(
            "rainfall_converted_only_after_exact_2024_date_and_station_filter"
        ),
        True,
        "E3 filtering boundary",
    )


def _validate_e2_tables(
    pooled: pd.DataFrame,
    by_lead: pd.DataFrame,
    effects: pd.DataFrame,
    intensity: pd.DataFrame,
    intensity_effects: pd.DataFrame,
) -> None:
    score_columns = ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
    _require_columns(pooled, ("method", "case_lead_count", *score_columns), "E2 pooled")
    _require_exact(
        tuple(sorted(pooled["method"])), tuple(sorted(E2_METHODS)), "E2 pooled methods"
    )
    if (
        pooled["method"].duplicated().any()
        or not pooled["case_lead_count"].eq(600).all()
    ):
        raise ValueError(
            "E2 pooled table must contain one 600-case-lead row per method"
        )
    _require_finite(pooled, score_columns, "E2 pooled")

    _require_columns(
        by_lead,
        ("lead_week", "method", "case_lead_count", *score_columns),
        "E2 by lead",
    )
    expected_pairs = {(method, lead) for method in E2_METHODS for lead in range(1, 7)}
    actual_pairs = set(zip(by_lead["method"], by_lead["lead_week"].astype(int)))
    _require_exact(actual_pairs, expected_pairs, "E2 method/lead coverage")
    if (
        by_lead.duplicated(["method", "lead_week"]).any()
        or not by_lead["case_lead_count"].eq(100).all()
    ):
        raise ValueError("E2 by-lead table must retain 100 matched starts per lead")
    _require_finite(by_lead, score_columns, "E2 by lead")

    effect_columns = (
        "scope_type",
        "scope",
        "region",
        "candidate",
        "baseline",
        "source_metric",
        "effect",
        "ci_lower_2p5",
        "ci_upper_97p5",
        "n_starts",
        "n_leads_per_start",
        "definition",
    )
    _require_columns(effects, effect_columns, "E2 effects")
    lead_effects = effects.loc[
        effects["scope_type"].eq("lead")
        & effects["region"].eq("all_india")
        & effects["candidate"].eq("raw_identity")
        & effects["baseline"].eq("raw_fuxi")
        & effects["source_metric"].isin(("rmse_mm_day", "acc"))
    ].copy()
    expected = {
        (f"W{lead}", metric)
        for lead in range(1, 7)
        for metric in ("rmse_mm_day", "acc")
    }
    actual = set(zip(lead_effects["scope"], lead_effects["source_metric"]))
    _require_exact(actual, expected, "E2 lead effects")
    if lead_effects.duplicated(["scope", "source_metric"]).any():
        raise ValueError("E2 lead effects contain duplicates")
    _validate_interval_rows(lead_effects, "E2 lead effects")
    if (
        not lead_effects["n_starts"].eq(100).all()
        or not lead_effects["n_leads_per_start"].eq(1).all()
    ):
        raise ValueError(
            "E2 lead-specific effects must contain 100 starts and one lead per row"
        )
    if (
        not lead_effects["definition"]
        .str.contains("actual block length 13", regex=False)
        .all()
    ):
        raise ValueError("E2 lead effects do not use circular block length 13")
    scores = by_lead.set_index(["method", "lead_week"])
    for row in lead_effects.itertuples(index=False):
        lead = int(str(row.scope)[1:])
        if row.source_metric == "rmse_mm_day":
            expected_effect = (
                scores.loc[("raw_fuxi", lead), "rmse_mm_day"]
                - scores.loc[("raw_identity", lead), "rmse_mm_day"]
            )
        else:
            expected_effect = (
                scores.loc[("raw_identity", lead), "acc"]
                - scores.loc[("raw_fuxi", lead), "acc"]
            )
        if not math.isclose(float(row.effect), float(expected_effect), abs_tol=1e-9):
            raise ValueError("E2 lead effect does not reproduce the by-lead scores")

    intensity_columns = (
        "method",
        "stratum",
        "stratum_label",
        "cell_case_lead_count",
        "rmse_mm_day",
        "mae_mm_day",
        "bias_mm_day",
        "truth_mean_mm_day",
        "prediction_mean_mm_day",
    )
    _require_columns(intensity, intensity_columns, "E2 intensity")
    expected_intensity = {
        (method, stratum) for method in E2_METHODS for stratum in INTENSITY_ORDER
    }
    _require_exact(
        set(zip(intensity["method"], intensity["stratum"])),
        expected_intensity,
        "E2 intensity coverage",
    )
    if intensity.duplicated(["method", "stratum"]).any():
        raise ValueError("E2 intensity table contains duplicates")
    _require_finite(
        intensity,
        (
            "cell_case_lead_count",
            "rmse_mm_day",
            "mae_mm_day",
            "bias_mm_day",
            "truth_mean_mm_day",
            "prediction_mean_mm_day",
        ),
        "E2 intensity",
    )
    _require_columns(
        intensity_effects,
        (
            "stratum",
            "candidate",
            "baseline",
            "source_metric",
            "effect",
            "ci_lower_2p5",
            "ci_upper_97p5",
            "n_starts",
            "n_leads_per_start",
            "definition",
        ),
        "E2 intensity effects",
    )
    selected = intensity_effects.loc[
        intensity_effects["candidate"].eq("raw_identity")
        & intensity_effects["baseline"].eq("raw_fuxi")
        & intensity_effects["source_metric"].isin(("rmse_mm_day", "mae_mm_day"))
    ].copy()
    expected_selected = {
        (stratum, metric)
        for stratum in INTENSITY_ORDER
        for metric in ("rmse_mm_day", "mae_mm_day")
    }
    _require_exact(
        set(zip(selected["stratum"], selected["source_metric"])),
        expected_selected,
        "E2 intensity effects",
    )
    if selected.duplicated(["stratum", "source_metric"]).any():
        raise ValueError("E2 intensity effects contain duplicates")
    _validate_interval_rows(selected, "E2 intensity effects")
    if (
        not selected["n_starts"].eq(100).all()
        or not selected["n_leads_per_start"].eq(6).all()
    ):
        raise ValueError("E2 intensity effects violate the resampling contract")
    if (
        not selected["definition"]
        .str.contains("actual block length 13", regex=False)
        .all()
    ):
        raise ValueError("E2 intensity effects do not use circular block length 13")
    intensity_scores = intensity.set_index(["method", "stratum"])
    for row in selected.itertuples(index=False):
        expected_effect = (
            intensity_scores.loc[("raw_fuxi", row.stratum), row.source_metric]
            - intensity_scores.loc[("raw_identity", row.stratum), row.source_metric]
        )
        if not math.isclose(float(row.effect), float(expected_effect), abs_tol=1e-9):
            raise ValueError("E2 intensity effect does not reproduce stratum metrics")

    heavy_mae = selected.loc[
        selected["stratum"].eq("heavy_10_20")
        & selected["source_metric"].eq("mae_mm_day")
    ].iloc[0]
    extreme = selected.loc[selected["stratum"].eq("extreme_ge20")]
    if not (float(heavy_mae.ci_lower_2p5) <= 0.0 <= float(heavy_mae.ci_upper_97p5)):
        raise ValueError(
            "the declared heavy-rain MAE limitation is no longer supported"
        )
    if not ((extreme["ci_lower_2p5"] <= 0.0) & (extreme["ci_upper_97p5"] >= 0.0)).all():
        raise ValueError("the declared extreme-rain uncertainty is no longer supported")

    projection = effects.loc[
        effects["scope_type"].eq("pooled")
        & effects["scope"].eq("W1-W6")
        & effects["region"].eq("all_india")
        & effects["candidate"].eq("raw_identity_raw_mean_preserved")
        & effects["baseline"].eq("raw_identity")
        & effects["source_metric"].isin(
            ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
        )
    ].copy()
    _require_exact(
        set(projection["source_metric"]),
        {"rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"},
        "E2 projection effects",
    )
    if len(projection) != 4:
        raise ValueError("E2 projection comparison must have four unique effects")
    _validate_interval_rows(projection, "E2 projection effects")
    pooled_scores = pooled.set_index("method")
    for row in projection.itertuples(index=False):
        candidate = pooled_scores.loc["raw_identity_raw_mean_preserved"]
        baseline = pooled_scores.loc["raw_identity"]
        if row.source_metric == "acc":
            expected_effect = candidate.acc - baseline.acc
        elif row.source_metric == "bias_mm_day":
            expected_effect = abs(baseline.bias_mm_day) - abs(candidate.bias_mm_day)
            if "absolute pooled bias" not in str(row.definition):
                raise ValueError(
                    "E2 bias effect is not labelled as absolute pooled bias"
                )
        else:
            expected_effect = baseline[row.source_metric] - candidate[row.source_metric]
        if not math.isclose(float(row.effect), float(expected_effect), abs_tol=1e-9):
            raise ValueError("E2 projection effect does not reproduce pooled metrics")


def _strict_bool_series(series: pd.Series, label: str) -> pd.Series:
    if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{label} must contain strict Boolean values")
    return series.astype(bool)


def _validate_e3_tables(
    summary: pd.DataFrame, effects: pd.DataFrame, manifest: Mapping[str, Any]
) -> None:
    metrics = (
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
            *metrics,
        ),
        "E3 summary",
    )
    pooled = summary.loc[summary["scope_type"].eq("pooled")].copy()
    _require_exact(
        tuple(sorted(pooled["method"])), tuple(sorted(E3_METHODS)), "E3 pooled methods"
    )
    if (
        pooled["method"].duplicated().any()
        or not pooled["initializations"].eq(30).all()
        or not pooled["case_leads"].eq(180).all()
    ):
        raise ValueError("E3 pooled table violates the 30-start/six-lead contract")
    _require_finite(pooled, metrics, "E3 pooled")

    _require_columns(
        effects,
        (
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
            "primary_estimand",
        ),
        "E3 effects",
    )
    primary_flags = _strict_bool_series(
        effects["primary_estimand"], "E3 primary_estimand"
    )
    primary_rows = effects.loc[primary_flags]
    if len(primary_rows) != 1:
        raise ValueError("E3 must contain exactly one primary-estimand row")
    primary = primary_rows.iloc[0]
    normalized_primary_tuple = (
        str(primary["comparison"]),
        str(primary["candidate"]),
        str(primary["reference"]),
        {"rmse": "rmse_mm_day"}.get(str(primary["metric"]), str(primary["metric"])),
    )
    _require_exact(
        normalized_primary_tuple,
        E3_PRIMARY_TUPLE,
        "E3 normalized primary estimand tuple",
    )
    declared = manifest["primary_estimand"]
    checks = {
        "comparison": declared["comparison"],
        "metric": declared["metric"],
        "point_effect": declared["point_effect"],
        "ci_lower_2p5": declared["ci_lower_2p5"],
        "ci_upper_97p5": declared["ci_upper_97p5"],
        "block_length_initializations": declared[
            "circular_block_length_initializations"
        ],
        "bootstrap_draws": declared["bootstrap_draws"],
    }
    for column, expected in checks.items():
        actual = primary[column]
        if isinstance(expected, float):
            if not math.isclose(float(actual), float(expected), abs_tol=1e-12):
                raise ValueError(f"E3 primary row differs from manifest: {column}")
        else:
            _require_exact(actual, expected, f"E3 primary {column}")

    block13 = effects.loc[effects["block_length_initializations"].eq(13)].copy()
    projection = block13.loc[
        block13["candidate"].eq("raw_identity_raw_mean_preserved")
        & block13["reference"].eq("raw_identity")
        & block13["metric"].isin(("rmse", "mae", "acc", "absolute_bias"))
    ].copy()
    _require_exact(
        set(projection["metric"]),
        {"rmse", "mae", "acc", "absolute_bias"},
        "E3 projection effects",
    )
    if (
        len(projection) != 4
        or _strict_bool_series(
            projection["primary_estimand"], "E3 projection primary flags"
        ).any()
    ):
        raise ValueError("E3 projection effects must be four non-primary rows")
    secondary = block13.loc[
        block13["candidate"].eq("raw_identity")
        & block13["reference"].eq("raw_fuxi")
        & block13["metric"].isin(("rmse", "mae", "acc", "absolute_bias"))
    ].copy()
    _require_exact(
        set(secondary["metric"]),
        {"rmse", "mae", "acc", "absolute_bias"},
        "E3 raw-identity secondary effects",
    )
    if (
        len(secondary) != 4
        or _strict_bool_series(
            secondary["primary_estimand"], "E3 secondary primary flags"
        ).any()
    ):
        raise ValueError(
            "E3 raw-identity comparison must remain explicitly non-primary"
        )
    for selected, label in (
        (projection, "E3 projection"),
        (secondary, "E3 secondary"),
        (primary_rows, "E3 primary"),
    ):
        renamed = selected.rename(columns={"point_effect": "effect"})
        _validate_interval_rows(renamed, label)
        if (
            not selected["bootstrap_draws"].eq(2_000).all()
            or not selected["initializations"].eq(30).all()
            or not selected["case_leads"].eq(180).all()
        ):
            raise ValueError(f"{label} violates the E3 bootstrap contract")
    pooled_scores = pooled.set_index("method")
    mapping = {
        "rmse": "rmse_mean",
        "mae": "mae_mean",
        "acc": "acc_mean",
        "absolute_bias": "absolute_bias_mean",
    }
    for selected, label in (
        (projection, "E3 projection"),
        (secondary, "E3 secondary"),
        (primary_rows, "E3 primary"),
    ):
        for row in selected.itertuples(index=False):
            candidate = pooled_scores.loc[row.candidate]
            reference = pooled_scores.loc[row.reference]
            expected_definition = (
                "candidate_minus_reference"
                if row.metric == "acc"
                else "reference_minus_candidate"
            )
            _require_exact(
                row.effect_definition,
                expected_definition,
                f"{label} effect definition",
            )
            column = mapping[row.metric]
            expected_effect = (
                candidate[column] - reference[column]
                if row.metric == "acc"
                else reference[column] - candidate[column]
            )
            if not math.isclose(
                float(row.point_effect), float(expected_effect), abs_tol=1e-9
            ):
                raise ValueError(
                    f"{label} effect does not reproduce pooled method-summary scores"
                )


def load_verified_inputs(
    training_directory: Path,
    e2_directory: Path,
    e3_directory: Path,
    *,
    training_manifest_sha256: str,
    e2_manifest_sha256: str,
    e3_manifest_sha256: str,
) -> CoreInputs:
    """Load only hash-bound manifest and CSV bytes."""

    training_directory = Path(training_directory).resolve()
    e2_directory = Path(e2_directory).resolve()
    e3_directory = Path(e3_directory).resolve()
    training_manifest, training_hash = _read_hash_pinned_manifest(
        training_directory / "manifest.json", training_manifest_sha256
    )
    e2_manifest, e2_hash = _read_hash_pinned_manifest(
        e2_directory / "manifest.json", e2_manifest_sha256
    )
    e3_manifest, e3_hash = _read_hash_pinned_manifest(
        e3_directory / "manifest.json", e3_manifest_sha256
    )
    _validate_training_manifest(training_manifest)
    _validate_e2_manifest(e2_manifest, training_manifest, training_hash)
    _validate_e3_manifest(e3_manifest, e2_hash)

    training_verified, _ = _read_verified_artifacts(
        training_directory, training_manifest, TRAINING_REQUIRED_ARTIFACTS
    )
    e2_verified, e2_payloads = _read_verified_artifacts(
        e2_directory, e2_manifest, E2_REQUIRED_ARTIFACTS
    )
    e3_verified, e3_payloads = _read_verified_artifacts(
        e3_directory, e3_manifest, E3_REQUIRED_ARTIFACTS
    )
    e2_pooled = _parse_csv(e2_payloads[E2_REQUIRED_ARTIFACTS[0]])
    e2_by_lead = _parse_csv(e2_payloads[E2_REQUIRED_ARTIFACTS[1]])
    e2_effects = _parse_csv(e2_payloads[E2_REQUIRED_ARTIFACTS[2]])
    e2_intensity = _parse_csv(e2_payloads[E2_REQUIRED_ARTIFACTS[3]])
    e2_intensity_effects = _parse_csv(e2_payloads[E2_REQUIRED_ARTIFACTS[4]])
    e3_summary = _parse_csv(e3_payloads[E3_REQUIRED_ARTIFACTS[0]])
    e3_effects = _parse_csv(e3_payloads[E3_REQUIRED_ARTIFACTS[1]])
    _validate_e2_tables(
        e2_pooled,
        e2_by_lead,
        e2_effects,
        e2_intensity,
        e2_intensity_effects,
    )
    _validate_e3_tables(e3_summary, e3_effects, e3_manifest)
    return CoreInputs(
        training_directory=training_directory,
        e2_directory=e2_directory,
        e3_directory=e3_directory,
        training_manifest=training_manifest,
        e2_manifest=e2_manifest,
        e3_manifest=e3_manifest,
        training_manifest_sha256=training_hash,
        e2_manifest_sha256=e2_hash,
        e3_manifest_sha256=e3_hash,
        verified_source_artifacts={
            "training": training_verified,
            "e2": e2_verified,
            "e3": e3_verified,
        },
        e2_pooled=e2_pooled,
        e2_by_lead=e2_by_lead,
        e2_effects=e2_effects,
        e2_intensity=e2_intensity,
        e2_intensity_effects=e2_intensity_effects,
        e3_summary=e3_summary,
        e3_effects=e3_effects,
    )


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "build_core3_paper_package.py"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "build_core3_paper_package.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.0,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.25,
    )
    return patch


def _arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color="#555555",
            transform=axis.transAxes,
        )
    )


def _figure1_data(inputs: CoreInputs) -> pd.DataFrame:
    model = inputs.training_manifest["training"][SELECTED_MODEL]
    rows: list[dict[str, Any]] = []
    for run in model["runs"]:
        rows.append(
            {
                "row_type": "seed_member",
                "stage": "raw_identity_member",
                "seed": int(run["seed"]),
                "parameter_count": PARAMETER_COUNT,
                "architecture": SELECTED_ARCHITECTURE,
                "consumed_channels": len(CONSUMED_CHANNELS),
                "unet_widths": "16/32/64",
                "temporal_attention_heads": 4,
                "residual_head_initialization": "zero",
                "best_epoch": int(run["best_epoch"]),
                "best_validation_loss": float(run["best_validation_loss"]),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "training_anchor": "raw_fuxi",
                "selected_alpha": 1.0,
                "ensemble_operation": "mean of three seed residuals before reconstruction",
            }
        )
    timeline = (
        ("train", 2002, 2017, 560, "fit three members", "IMD grid"),
        ("select", 2018, 2019, 70, "validation-only selection", "IMD grid"),
        (
            "reused_exploratory",
            2020,
            2021,
            70,
            "reused exploratory evaluation",
            "IMD grid",
        ),
        (
            "frozen_audit",
            2022,
            2024,
            100,
            "retrospective development audit",
            "IMD grid",
        ),
        ("sealed_final", 2025, 2025, np.nan, "sealed; not opened or scored", "none"),
        (
            "station_sensitivity",
            2024,
            2024,
            30,
            "external-target sensitivity; no retraining",
            "gauge-derived 1.5° cells",
        ),
    )
    for stage, start, end, count, role, target in timeline:
        rows.append(
            {
                "row_type": "evidence_stage",
                "stage": stage,
                "start_year": start,
                "end_year": end,
                "initialization_count": count,
                "evidence_role": role,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def _plot_figure1(inputs: CoreInputs, output: Path) -> pd.DataFrame:
    data = _figure1_data(inputs)
    model = inputs.training_manifest["training"][SELECTED_MODEL]
    with plt.rc_context(PLOT_RC):
        figure = plt.figure(figsize=(11.4, 6.5), facecolor="white")
        grid = figure.add_gridspec(2, 1, height_ratios=(1.35, 1.0), hspace=0.25)
        architecture = figure.add_subplot(grid[0])
        timeline = figure.add_subplot(grid[1])
        for axis in (architecture, timeline):
            axis.set_axis_off()

        architecture.text(
            0.0,
            1.04,
            "A  Three-member raw-identity residual architecture",
            transform=architecture.transAxes,
            weight="bold",
            fontsize=10.0,
            va="bottom",
        )
        _box(
            architecture,
            (0.01, 0.31),
            0.14,
            0.38,
            "6 lead weeks\n29 prepared features\n11 channels consumed",
            facecolor="#F4F4F4",
            edgecolor="#777777",
            weight="bold",
        )
        _arrow(architecture, (0.155, 0.50), (0.205, 0.50))
        details = (
            "Each member\n"
            "compact U-Net 16 → 32 → 64\n"
            "4-head temporal transformer at bottleneck\n"
            "decoder 32 → 16\n"
            "zero-initialized 1×1 residual head\n"
            f"{PARAMETER_COUNT:,} trainable parameters"
        )
        _box(
            architecture,
            (0.205, 0.20),
            0.34,
            0.60,
            details,
            facecolor="#EAF2F8",
            edgecolor=IDENTITY_COLOR,
            fontsize=7.8,
            weight="bold",
        )
        seed_x = 0.565
        seed_width = 0.135
        seed_height = 0.16
        seed_y = (0.69, 0.42, 0.15)
        for y, run in zip(seed_y, model["runs"]):
            _arrow(architecture, (0.545, 0.50), (seed_x, y + seed_height / 2))
            _box(
                architecture,
                (seed_x, y),
                seed_width,
                seed_height,
                (
                    f"Seed {run['seed']} residual\n"
                    f"best epoch {run['best_epoch']} · val {run['best_validation_loss']:.4f}"
                ),
                facecolor="#FFFFFF",
                edgecolor=IDENTITY_COLOR,
                fontsize=7.2,
            )
        _arrow(architecture, (0.705, 0.50), (0.755, 0.50))
        _box(
            architecture,
            (0.755, 0.38),
            0.09,
            0.24,
            "Average\n3 residuals",
            facecolor="#FFF4D6",
            edgecolor="#E69F00",
            weight="bold",
        )
        _arrow(architecture, (0.847, 0.50), (0.875, 0.50))
        _box(
            architecture,
            (0.875, 0.31),
            0.115,
            0.38,
            "Reconstruct around\nraw FuXi\nα = 1\n\nDeterministic\nweekly rainfall",
            facecolor="#E8F5E9",
            edgecolor=PROJECTION_COLOR,
            fontsize=7.6,
            weight="bold",
        )
        architecture.text(
            0.205,
            0.08,
            "All three members share the same architecture and parameter count; they are independently trained seeds, not one model.",
            transform=architecture.transAxes,
            fontsize=7.4,
            color="#555555",
        )

        timeline.text(
            0.0,
            1.04,
            "B  Training, selection, and evidence boundary",
            transform=timeline.transAxes,
            weight="bold",
            fontsize=10.0,
            va="bottom",
        )
        segments = (
            (0.02, 0.29, "2002–17\nTRAIN\n560 starts", "#DCEAF7", IDENTITY_COLOR),
            (0.31, 0.15, "2018–19\nSELECT\n70 starts", "#FFF1CC", "#E69F00"),
            (0.46, 0.15, "2020–21\nREUSED\nexploratory · 70", "#EEEEEE", "#777777"),
            (
                0.61,
                0.23,
                "2022–24\nFROZEN AUDIT\n100 starts × 6 leads",
                "#DDF1EC",
                PROJECTION_COLOR,
            ),
            (0.84, 0.14, "2025\nSEALED\nnot opened/scored", "#F8D7DA", "#B22222"),
        )
        for x, width, text_value, face, edge in segments:
            _box(
                timeline,
                (x, 0.49),
                width - 0.008,
                0.34,
                text_value,
                facecolor=face,
                edgecolor=edge,
                fontsize=7.7,
                weight="bold",
            )
        timeline.add_patch(
            FancyArrowPatch(
                (0.72, 0.49),
                (0.72, 0.30),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.15,
                color=STATION_COLOR,
                transform=timeline.transAxes,
            )
        )
        _box(
            timeline,
            (0.61, 0.05),
            0.23,
            0.24,
            "2024 GAUGE-DERIVED CELL BRANCH\n1.5° external-target sensitivity\n30 starts × 6 leads · no retraining/selection",
            facecolor="#FCE8DE",
            edgecolor=STATION_COLOR,
            fontsize=7.4,
            weight="bold",
        )
        timeline.text(
            0.02,
            0.36,
            "fit",
            transform=timeline.transAxes,
            color=IDENTITY_COLOR,
            weight="bold",
        )
        timeline.text(
            0.32,
            0.36,
            "validation-only choice",
            transform=timeline.transAxes,
            color="#9A6700",
            weight="bold",
        )
        timeline.text(
            0.62,
            0.36,
            "retrospective development evidence",
            transform=timeline.transAxes,
            color="#006B54",
            weight="bold",
        )
        figure.suptitle(
            "Figure 1 | Raw-identity architecture and separated evidence timeline",
            x=0.01,
            ha="left",
            weight="bold",
        )
        _save_figure(figure, output, "figure_01_architecture_evidence_timeline")
    return data


def _select_e2_lead_effects(inputs: CoreInputs) -> pd.DataFrame:
    return inputs.e2_effects.loc[
        inputs.e2_effects["scope_type"].eq("lead")
        & inputs.e2_effects["region"].eq("all_india")
        & inputs.e2_effects["candidate"].eq("raw_identity")
        & inputs.e2_effects["baseline"].eq("raw_fuxi")
        & inputs.e2_effects["source_metric"].isin(("rmse_mm_day", "acc"))
    ].copy()


def _select_e2_intensity_effects(inputs: CoreInputs) -> pd.DataFrame:
    return inputs.e2_intensity_effects.loc[
        inputs.e2_intensity_effects["candidate"].eq("raw_identity")
        & inputs.e2_intensity_effects["baseline"].eq("raw_fuxi")
        & inputs.e2_intensity_effects["source_metric"].isin(
            ("rmse_mm_day", "mae_mm_day")
        )
    ].copy()


def _figure2_data(inputs: CoreInputs) -> pd.DataFrame:
    lead = _select_e2_lead_effects(inputs)
    lead_scores = inputs.e2_by_lead.set_index(["method", "lead_week"])
    lead_rows = []
    for row in lead.itertuples(index=False):
        lead_week = int(str(row.scope)[1:])
        column = "rmse_mm_day" if row.source_metric == "rmse_mm_day" else "acc"
        lead_rows.append(
            {
                "panel": "lead_effect",
                "scope": row.scope,
                "lead_week": lead_week,
                "stratum": "",
                "stratum_label": "",
                "metric": "rmse" if row.source_metric == "rmse_mm_day" else "acc",
                "effect": row.effect,
                "ci_lower_2p5": row.ci_lower_2p5,
                "ci_upper_97p5": row.ci_upper_97p5,
                "candidate_score": lead_scores.loc[("raw_identity", lead_week), column],
                "baseline_score": lead_scores.loc[("raw_fuxi", lead_week), column],
                "cell_case_lead_count": np.nan,
                "n_starts": row.n_starts,
                "n_leads_per_start": row.n_leads_per_start,
                "effect_orientation": (
                    "raw minus raw-identity"
                    if row.source_metric == "rmse_mm_day"
                    else "raw-identity minus raw"
                ),
                "bootstrap_contract": "10,000 year-stratified circular block-13 resamples; all six leads attached",
            }
        )
    intensity = _select_e2_intensity_effects(inputs)
    intensity_scores = inputs.e2_intensity.set_index(["method", "stratum"])
    intensity_rows = []
    for row in intensity.itertuples(index=False):
        metric = "rmse" if row.source_metric == "rmse_mm_day" else "mae"
        candidate = intensity_scores.loc[("raw_identity", row.stratum)]
        baseline = intensity_scores.loc[("raw_fuxi", row.stratum)]
        intensity_rows.append(
            {
                "panel": "intensity_effect",
                "scope": "W1-W6",
                "lead_week": np.nan,
                "stratum": row.stratum,
                "stratum_label": INTENSITY_LABELS[row.stratum],
                "metric": metric,
                "effect": row.effect,
                "ci_lower_2p5": row.ci_lower_2p5,
                "ci_upper_97p5": row.ci_upper_97p5,
                "candidate_score": candidate[row.source_metric],
                "baseline_score": baseline[row.source_metric],
                "cell_case_lead_count": candidate.cell_case_lead_count,
                "n_starts": row.n_starts,
                "n_leads_per_start": row.n_leads_per_start,
                "effect_orientation": "raw minus raw-identity",
                "bootstrap_contract": "10,000 year-stratified circular block-13 resamples; all six leads attached",
            }
        )
    return pd.DataFrame([*lead_rows, *intensity_rows])


def _errorbar(axis: plt.Axes, frame: pd.DataFrame, *, color: str, marker: str) -> None:
    x = frame["x"].to_numpy(dtype=float)
    point = frame["effect"].to_numpy(dtype=float)
    lower = frame["ci_lower_2p5"].to_numpy(dtype=float)
    upper = frame["ci_upper_97p5"].to_numpy(dtype=float)
    axis.errorbar(
        x,
        point,
        yerr=np.vstack((point - lower, upper - point)),
        fmt=marker,
        color=color,
        ecolor=color,
        elinewidth=1.3,
        capsize=2.4,
        markersize=5.0,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )


def _clean_effect_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#555555", linewidth=0.8, zorder=1)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_figure2(inputs: CoreInputs, output: Path) -> pd.DataFrame:
    data = _figure2_data(inputs)
    lead = data.loc[data["panel"].eq("lead_effect")].copy()
    intensity = data.loc[data["panel"].eq("intensity_effect")].copy()
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.75), facecolor="white")
        for axis in axes:
            _clean_effect_axis(axis)

        rmse = lead.loc[lead["metric"].eq("rmse")].sort_values("lead_week")
        rmse["x"] = rmse["lead_week"]
        _errorbar(axes[0], rmse, color=IDENTITY_COLOR, marker="o")
        axes[0].set_title("A  RMSE gain by lead", loc="left", weight="bold")
        axes[0].set_xlabel("Lead week")
        axes[0].set_ylabel("RMSE improvement (mm day⁻¹)")
        axes[0].set_xticks(range(1, 7))
        axes[0].set_xlim(0.6, 6.4)
        axes[0].text(
            0.04,
            0.95,
            "Gain persists W1–W6\nbut narrows with lead",
            transform=axes[0].transAxes,
            va="top",
            color=IDENTITY_COLOR,
            weight="bold",
            fontsize=7.7,
        )

        acc = lead.loc[lead["metric"].eq("acc")].sort_values("lead_week")
        acc["x"] = acc["lead_week"]
        _errorbar(axes[1], acc, color=IDENTITY_COLOR, marker="o")
        axes[1].set_title("B  Pattern-skill gain by lead", loc="left", weight="bold")
        axes[1].set_xlabel("Lead week")
        axes[1].set_ylabel("ACC improvement")
        axes[1].set_xticks(range(1, 7))
        axes[1].set_xlim(0.6, 6.4)
        axes[1].text(
            0.04,
            0.95,
            "Positive at every lead",
            transform=axes[1].transAxes,
            va="top",
            color=IDENTITY_COLOR,
            weight="bold",
            fontsize=7.7,
        )

        x = np.arange(len(INTENSITY_ORDER), dtype=float)
        axes[2].axvspan(2.5, 4.5, color=LIMIT_COLOR, alpha=0.35, zorder=0)
        for metric, offset, color, marker, label in (
            ("rmse", -0.09, IDENTITY_COLOR, "o", "RMSE"),
            ("mae", 0.09, "#E69F00", "s", "MAE"),
        ):
            selected = intensity.loc[intensity["metric"].eq(metric)].copy()
            selected["order"] = selected["stratum"].map(
                {value: index for index, value in enumerate(INTENSITY_ORDER)}
            )
            selected = selected.sort_values("order")
            selected["x"] = x + offset
            _errorbar(axes[2], selected, color=color, marker=marker)
            axes[2].plot(
                [], [], marker=marker, color=color, linestyle="none", label=label
            )
        axes[2].set_title("C  Intensity limitation", loc="left", weight="bold")
        axes[2].set_xlabel("Verifying weekly IMD (mm day⁻¹)")
        axes[2].set_ylabel("Error improvement (mm day⁻¹)")
        axes[2].set_xticks(x, [INTENSITY_LABELS[value] for value in INTENSITY_ORDER])
        axes[2].legend(frameon=False, ncol=2, loc="upper right")
        axes[2].text(
            0.61,
            0.08,
            "Heavy: MAE CI crosses 0\nExtreme: both CIs cross 0;\nMAE point estimate worsens",
            transform=axes[2].transAxes,
            fontsize=7.1,
            color="#8C2D23",
            weight="bold",
            va="bottom",
        )
        figure.suptitle(
            "Figure 2 | Raw identity improves average skill; heavy-rain evidence is weaker",
            x=0.01,
            y=1.03,
            ha="left",
            weight="bold",
        )
        figure.text(
            0.01,
            -0.035,
            "Positive favors raw identity. Points are paired effects; bars are descriptive 95% intervals from 10,000 shared, year-stratified circular block-13 resamples (100 starts; all six leads attached).",
            fontsize=7.2,
            color="#444444",
        )
        figure.tight_layout(w_pad=2.0)
        _save_figure(figure, output, "figure_02_lead_intensity_effects")
    return data


def _select_e2_projection(inputs: CoreInputs) -> pd.DataFrame:
    return inputs.e2_effects.loc[
        inputs.e2_effects["scope_type"].eq("pooled")
        & inputs.e2_effects["scope"].eq("W1-W6")
        & inputs.e2_effects["region"].eq("all_india")
        & inputs.e2_effects["candidate"].eq("raw_identity_raw_mean_preserved")
        & inputs.e2_effects["baseline"].eq("raw_identity")
        & inputs.e2_effects["source_metric"].isin(
            ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
        )
    ].copy()


def _select_e3_block13(
    inputs: CoreInputs, candidate: str, reference: str
) -> pd.DataFrame:
    return inputs.e3_effects.loc[
        inputs.e3_effects["block_length_initializations"].eq(13)
        & inputs.e3_effects["candidate"].eq(candidate)
        & inputs.e3_effects["reference"].eq(reference)
        & inputs.e3_effects["metric"].isin(("rmse", "mae", "acc", "absolute_bias"))
    ].copy()


def _figure3_data(inputs: CoreInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    e2 = inputs.e2_pooled.set_index("method")
    e3 = inputs.e3_summary.loc[inputs.e3_summary["scope_type"].eq("pooled")].set_index(
        "method"
    )
    for target, table, bias_column, bias_estimator in (
        ("E2 IMD grid", e2, "bias_mm_day", "pooled signed bias"),
        ("E3 gauge-derived cells", e3, "bias_mean", "mean case signed bias"),
    ):
        for method in (
            "raw_fuxi",
            "raw_identity",
            "raw_identity_raw_mean_preserved",
        ):
            rows.append(
                {
                    "row_type": "signed_bias_summary",
                    "target": target,
                    "method": method,
                    "candidate": "",
                    "reference": "",
                    "metric": "signed_bias",
                    "bias_estimand": bias_estimator,
                    "point_effect_or_score": table.loc[method, bias_column],
                    "ci_lower_2p5": np.nan,
                    "ci_upper_97p5": np.nan,
                    "effect_orientation": "score",
                    "bootstrap_draws": np.nan,
                    "block_length_initializations": np.nan,
                    "bootstrap_contract": "not applicable to displayed pooled score",
                    "comparison_role": "descriptive target-specific score",
                    "primary_estimand": False,
                }
            )
    e2_projection = _select_e2_projection(inputs)
    for row in e2_projection.itertuples(index=False):
        metric = {
            "rmse_mm_day": "rmse",
            "mae_mm_day": "mae",
            "bias_mm_day": "absolute_bias",
            "acc": "acc",
        }[row.source_metric]
        rows.append(
            {
                "row_type": "projection_effect",
                "target": "E2 IMD grid",
                "method": "",
                "candidate": "raw_identity_raw_mean_preserved",
                "reference": "raw_identity",
                "metric": metric,
                "bias_estimand": (
                    "|pooled signed bias|" if metric == "absolute_bias" else ""
                ),
                "point_effect_or_score": row.effect,
                "ci_lower_2p5": row.ci_lower_2p5,
                "ci_upper_97p5": row.ci_upper_97p5,
                "effect_orientation": (
                    "candidate minus reference"
                    if metric == "acc"
                    else "reference minus candidate"
                ),
                "bootstrap_draws": 10_000,
                "block_length_initializations": 13,
                "bootstrap_contract": "year-stratified circular blocks; 100 starts; six leads attached",
                "comparison_role": "post-hoc projection diagnostic",
                "primary_estimand": False,
            }
        )
    e3_projection = _select_e3_block13(
        inputs, "raw_identity_raw_mean_preserved", "raw_identity"
    )
    for row in e3_projection.itertuples(index=False):
        rows.append(
            {
                "row_type": "projection_effect",
                "target": "E3 gauge-derived cells",
                "method": "",
                "candidate": row.candidate,
                "reference": row.reference,
                "metric": row.metric,
                "bias_estimand": (
                    "mean |case bias|" if row.metric == "absolute_bias" else ""
                ),
                "point_effect_or_score": row.point_effect,
                "ci_lower_2p5": row.ci_lower_2p5,
                "ci_upper_97p5": row.ci_upper_97p5,
                "effect_orientation": row.effect_definition.replace("_", " "),
                "bootstrap_draws": row.bootstrap_draws,
                "block_length_initializations": row.block_length_initializations,
                "bootstrap_contract": "circular blocks; 30 starts; six leads attached",
                "comparison_role": "secondary external-target projection diagnostic",
                "primary_estimand": bool(row.primary_estimand),
            }
        )
    e3_secondary = _select_e3_block13(inputs, "raw_identity", "raw_fuxi")
    for row in e3_secondary.itertuples(index=False):
        rows.append(
            {
                "row_type": "secondary_context_effect",
                "target": "E3 gauge-derived cells",
                "method": "",
                "candidate": row.candidate,
                "reference": row.reference,
                "metric": row.metric,
                "bias_estimand": (
                    "mean |case bias|" if row.metric == "absolute_bias" else ""
                ),
                "point_effect_or_score": row.point_effect,
                "ci_lower_2p5": row.ci_lower_2p5,
                "ci_upper_97p5": row.ci_upper_97p5,
                "effect_orientation": row.effect_definition.replace("_", " "),
                "bootstrap_draws": row.bootstrap_draws,
                "block_length_initializations": row.block_length_initializations,
                "bootstrap_contract": "circular blocks; 30 starts; six leads attached",
                "comparison_role": "secondary external-target sensitivity; not E3 primary estimand",
                "primary_estimand": bool(row.primary_estimand),
            }
        )
    primary = inputs.e3_effects.loc[
        _strict_bool_series(inputs.e3_effects["primary_estimand"], "E3 primary flags")
    ].iloc[0]
    rows.append(
        {
            "row_type": "e3_primary_contract",
            "target": "E3 gauge-derived cells",
            "method": "",
            "candidate": primary.candidate,
            "reference": primary.reference,
            "metric": primary.metric,
            "bias_estimand": "",
            "point_effect_or_score": primary.point_effect,
            "ci_lower_2p5": primary.ci_lower_2p5,
            "ci_upper_97p5": primary.ci_upper_97p5,
            "effect_orientation": primary.effect_definition.replace("_", " "),
            "bootstrap_draws": primary.bootstrap_draws,
            "block_length_initializations": primary.block_length_initializations,
            "bootstrap_contract": "circular blocks; 30 starts; six leads attached",
            "comparison_role": "E3 primary estimand: selected anchored adapter vs raw FuXi RMSE",
            "primary_estimand": True,
        }
    )
    return pd.DataFrame(rows)


def _plot_figure3(inputs: CoreInputs, output: Path) -> pd.DataFrame:
    data = _figure3_data(inputs)
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            1,
            4,
            figsize=(11.7, 4.05),
            gridspec_kw={"width_ratios": (1.25, 1.05, 0.80, 1.05)},
            facecolor="white",
        )
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)

        summaries = data.loc[data["row_type"].eq("signed_bias_summary")]
        targets = ("E2 IMD grid", "E3 gauge-derived cells")
        methods = (
            "raw_fuxi",
            "raw_identity",
            "raw_identity_raw_mean_preserved",
        )
        method_labels = ("Raw", "Raw identity", "Projection")
        colors = (RAW_COLOR, IDENTITY_COLOR, PROJECTION_COLOR)
        x = np.arange(2, dtype=float)
        width = 0.22
        for offset, method, label, color in zip(
            (-width, 0.0, width), methods, method_labels, colors
        ):
            values = [
                summaries.loc[
                    summaries["target"].eq(target) & summaries["method"].eq(method),
                    "point_effect_or_score",
                ].iloc[0]
                for target in targets
            ]
            axes[0].bar(
                x + offset, values, width=width, color=color, label=label, zorder=2
            )
        axes[0].axhline(0.0, color="#333333", linewidth=0.9)
        axes[0].set_xticks(x, ["IMD grid\n2022–24", "Gauge-derived cells\n2024"])
        axes[0].set_ylabel("Signed bias (mm day⁻¹)")
        axes[0].set_title("A  Raw-mean target flips", loc="left", weight="bold")
        axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")
        axes[0].text(
            0.02,
            0.02,
            "Raw is slightly dry vs IMD\nbut wet vs gauge-derived cells",
            transform=axes[0].transAxes,
            fontsize=7.1,
            color="#7A3326",
            weight="bold",
        )

        projection = data.loc[data["row_type"].eq("projection_effect")]
        for axis_index, metrics, title, ylabel in (
            (1, ("rmse", "mae"), "B  Error effect", "Improvement (mm day⁻¹)"),
            (2, ("acc",), "C  Pattern effect", "ACC improvement"),
        ):
            subset = projection.loc[projection["metric"].isin(metrics)].copy()
            positions = []
            labels = []
            for target_index, target in enumerate(targets):
                for metric_index, metric in enumerate(metrics):
                    row = subset.loc[
                        subset["target"].eq(target) & subset["metric"].eq(metric)
                    ].iloc[0]
                    position = target_index * (len(metrics) + 0.7) + metric_index
                    positions.append(position)
                    labels.append(f"{target.split()[0]}\n{metric.upper()}")
                    point = float(row.point_effect_or_score)
                    axes[axis_index].errorbar(
                        position,
                        point,
                        yerr=np.array(
                            [
                                [point - float(row.ci_lower_2p5)],
                                [float(row.ci_upper_97p5) - point],
                            ]
                        ),
                        fmt="o",
                        color=IMD_COLOR if target.startswith("E2") else STATION_COLOR,
                        ecolor=IMD_COLOR if target.startswith("E2") else STATION_COLOR,
                        capsize=2.5,
                        markersize=5,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        zorder=3,
                    )
            axes[axis_index].axhline(0.0, color="#444444", linewidth=0.8)
            axes[axis_index].set_xticks(positions, labels)
            axes[axis_index].set_ylabel(ylabel)
            axes[axis_index].set_title(title, loc="left", weight="bold")
        axes[1].text(
            0.04,
            0.04,
            "Positive favors projection",
            transform=axes[1].transAxes,
            fontsize=7.0,
            color="#555555",
        )

        bias = projection.loc[projection["metric"].eq("absolute_bias")].copy()
        bias_positions = np.arange(2, dtype=float)
        bias_points = (
            bias.set_index("target")
            .loc[list(targets), "point_effect_or_score"]
            .to_numpy(dtype=float)
        )
        bias_lower = (
            bias.set_index("target")
            .loc[list(targets), "ci_lower_2p5"]
            .to_numpy(dtype=float)
        )
        bias_upper = (
            bias.set_index("target")
            .loc[list(targets), "ci_upper_97p5"]
            .to_numpy(dtype=float)
        )
        for index, target in enumerate(targets):
            color = IMD_COLOR if target.startswith("E2") else STATION_COLOR
            axes[3].errorbar(
                bias_positions[index],
                bias_points[index],
                yerr=np.array(
                    [
                        [bias_points[index] - bias_lower[index]],
                        [bias_upper[index] - bias_points[index]],
                    ]
                ),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
        axes[3].axhline(0.0, color="#444444", linewidth=0.8)
        axes[3].set_xticks(
            bias_positions,
            ["E2\n|pooled signed bias|", "E3\nmean |case bias|"],
        )
        axes[3].set_ylabel("Bias-estimand improvement (mm day⁻¹)")
        axes[3].set_title("D  Distinct bias estimands", loc="left", weight="bold")
        axes[3].text(
            0.04,
            0.04,
            "Do not treat these as\nthe same estimand",
            transform=axes[3].transAxes,
            fontsize=7.0,
            color="#7A3326",
            weight="bold",
        )

        figure.suptitle(
            "Figure 3 | Raw-mean projection is target-dependent, not a transferable correction",
            x=0.01,
            y=1.04,
            ha="left",
            weight="bold",
        )
        figure.text(
            0.01,
            -0.025,
            "E2: 10,000 year-stratified circular block-13 resamples, 100 starts.  E3: 2,000 circular block-13 resamples, 30 starts.  All six leads remain attached.  E3 raw identity vs raw FuXi is secondary; the E3 primary estimand is selected anchored adapter vs raw FuXi RMSE.",
            fontsize=6.9,
            color="#444444",
        )
        figure.tight_layout(w_pad=1.45)
        _save_figure(figure, output, "figure_03_cross_target_projection_failure")
    return data


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_captions(output: Path) -> None:
    text = """# Core three-figure captions

## Figure 1 — Raw-identity architecture and evidence timeline

The raw-identity adapter is an ensemble of exactly three independently trained
members (seeds 42, 43, and 44), each with 144,689 trainable parameters. Each
member consumes the first 11 prepared channels, applies a compact 16/32/64
U-Net with four-head temporal attention across the six lead weeks at the
bottleneck, and ends in a zero-initialized residual head. The three predicted
residuals are averaged and reconstructed around raw FuXi with the
validation-selected alpha of one. Training used 2002–2017; selection used only
2018–2019. The 2020–2021 result is reused exploratory evidence. E2 is a frozen
2022–2024 retrospective development audit, not an untouched final test. E3
branches from the frozen predictions to a 2024 gauge-derived 1.5°-cell external-target
sensitivity without retraining or selection. The 2025 final year remains sealed
and was not opened or scored by this reporting builder.

## Figure 2 — Lead-wise gains and intensity limitation

Paired raw-identity-minus-raw evidence on the E2 IMD-grid retrospective
development audit. Error effects are raw FuXi minus raw identity; ACC effects
are raw identity minus raw FuXi, so positive values favor raw identity in all
panels. Points are observed effects and bars are descriptive 95% percentile
intervals from 10,000 shared, year-stratified circular moving-block resamples
of 13 initialization dates (100 starts, with all six leads retained per sampled
start). RMSE and ACC gains persist across all six leads. The intensity-stratified
panel exposes the limitation: heavy-rain (10–20 mm/day) MAE has an interval that
crosses zero; for extreme rain (at least 20 mm/day), both RMSE and MAE intervals
cross zero and the MAE point estimate is adverse. Intensity strata are defined
by verifying weekly-mean IMD rainfall and use dynamic area-by-coverage weights.

## Figure 3 — Cross-target failure of raw-mean projection

The post-hoc projection forces the raw-identity spatial mean toward the raw FuXi
mean, so its behavior depends on the observational target: raw FuXi is slightly
dry against the E2 IMD grid but wet against the E3 gauge-derived cells. Positive effects
favor projection (reference minus candidate for RMSE, MAE, and the displayed
bias estimands; candidate minus reference for ACC). E2 intervals use 10,000
shared, year-stratified circular block-13 resamples over 100 starts; E3 uses
2,000 shared circular block-13 resamples over 30 starts. All six leads remain
attached in both contracts. Crucially, the bias panels use different estimands:
E2 is improvement in **|pooled signed bias|**, whereas E3 is improvement in
**mean |case bias|**. They must not be conflated. The E3 raw-identity-versus-raw
comparison is secondary; E3's primary estimand is selected anchored adapter
versus raw FuXi RMSE. E3 is an external-target sensitivity, not an untouched
temporal final test.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(text, encoding="utf-8")


def _write_readme(inputs: CoreInputs, output: Path) -> None:
    e2_intensity = _select_e2_intensity_effects(inputs)
    extreme_mae = e2_intensity.loc[
        e2_intensity["stratum"].eq("extreme_ge20")
        & e2_intensity["source_metric"].eq("mae_mm_day")
    ].iloc[0]
    text = f"""# Core three-figure paper package

This package reduces the paper story to three evidence-bounded figures:

1. the exact three-member raw-identity architecture and evidence timeline;
2. its lead-wise E2 gains alongside the heavy/extreme-rain limitation; and
3. the cross-target failure of the post-hoc raw-mean projection.

The extreme-rain MAE effect is {extreme_mae.effect:.3f} mm/day with a 95%
interval [{extreme_mae.ci_lower_2p5:.3f}, {extreme_mae.ci_upper_97p5:.3f}];
that limitation is shown explicitly rather than treated as a positive result.

## Evidence boundary

- Training/selection manifest: `{inputs.training_manifest_sha256}`.
- E2 manifest: `{inputs.e2_manifest_sha256}` (2022–2024 retrospective
  development audit; not an untouched final test).
- E3 manifest: `{inputs.e3_manifest_sha256}` (2024 external-observational-target
  sensitivity; not an untouched temporal final test).
- E3 raw identity versus raw FuXi is a secondary comparison. The E3 primary
  estimand remains selected anchored adapter versus raw FuXi RMSE.
- E2 and E3 use different bias estimands in Figure 3: `|pooled signed bias|`
  and `mean |case bias|`, respectively.

## Reporting access boundary

This builder loaded three JSON manifests and seven already-computed metric CSVs.
It did not open checkpoints, prediction stores, forecast/target arrays, raw IMD
data, raw station data, or 2025 data. Upstream E3 scanned a mixed station
container with 45,910 unselected 2025+ rows, but filtered exact 2024 dates and
stations before rainfall conversion; it selected and scored no 2025 station
value.

Each figure has a 300-dpi PNG, vector PDF, and matching source-data CSV.
`FIGURE_CAPTIONS.md` contains manuscript-ready evidence-bounded captions;
`PACKAGE_MANIFEST.json` pins every input and output byte; `code/` contains the
exact standalone builder snapshot.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _all_nonmanifest_files(directory: Path) -> list[Path]:
    """Inventory regular files without following any symlink in the tree."""

    root_manifest = directory / "PACKAGE_MANIFEST.json"
    files: list[Path] = []

    def walk(current: Path) -> None:
        with os.scandir(current) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(directory)
            if entry.is_symlink():
                raise ValueError(f"package tree contains symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                walk(path)
            elif entry.is_file(follow_symlinks=False):
                if path != root_manifest:
                    files.append(path)
            else:
                raise ValueError(f"package tree contains non-regular entry: {relative}")

    walk(directory)
    return files


def verify_package(directory: Path) -> Mapping[str, Any]:
    """Re-hash the complete package inventory and reject extras."""

    requested_directory = Path(directory)
    if requested_directory.is_symlink():
        raise ValueError("core3 package directory must not be a symlink")
    directory = requested_directory.resolve()
    manifest_path = directory / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("core3 package manifest is not complete")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("core3 package lacks an artifact checksum mapping")
    actual_paths = {
        str(path.relative_to(directory)): path
        for path in _all_nonmanifest_files(directory)
    }
    if any(Path(relative).name == "PACKAGE_MANIFEST.json" for relative in actual_paths):
        raise ValueError("nested PACKAGE_MANIFEST.json is forbidden")
    if set(actual_paths) != set(declared):
        missing = sorted(set(declared) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(declared))
        raise ValueError(f"package inventory differs; missing={missing}, extra={extra}")
    for relative, path in actual_paths.items():
        expected = _normalise_sha256(str(declared[relative]), relative)
        if _sha256_file(path) != expected:
            raise ValueError(f"package artifact SHA-256 differs: {relative}")
    builder = manifest.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("core3 package builder binding is missing")
    builder_source = str(builder.get("source", ""))
    _require_exact(
        declared.get(builder_source), builder.get("sha256"), "builder snapshot binding"
    )
    return manifest


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish staging while refusing a raced destination."""

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
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        1,
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
    training_directory: Path,
    e2_directory: Path,
    e3_directory: Path,
    output_directory: Path,
    *,
    training_manifest_sha256: str,
    e2_manifest_sha256: str,
    e3_manifest_sha256: str,
) -> Path:
    """Build, seal, atomically publish, and re-verify a fresh core3 package."""

    sources = tuple(
        Path(value).resolve()
        for value in (training_directory, e2_directory, e3_directory)
    )
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_directory}"
        )
    for source in sources:
        if output_directory == source or source in output_directory.parents:
            raise ValueError("output directory may not be inside a canonical source")
    inputs = load_verified_inputs(
        *sources,
        training_manifest_sha256=training_manifest_sha256,
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
        figure1 = _plot_figure1(inputs, staging)
        figure2 = _plot_figure2(inputs, staging)
        figure3 = _plot_figure3(inputs, staging)
        _write_csv(figure1, staging / "figure_01_architecture_evidence_timeline.csv")
        _write_csv(figure2, staging / "figure_02_lead_intensity_effects.csv")
        _write_csv(figure3, staging / "figure_03_cross_target_projection_failure.csv")
        _write_captions(staging)
        _write_readme(inputs, staging)
        code_directory = staging / "code"
        code_directory.mkdir()
        builder_payload = Path(__file__).resolve().read_bytes()
        builder_snapshot = code_directory / Path(__file__).name
        builder_snapshot.write_bytes(builder_payload)

        artifacts = {
            str(path.relative_to(staging)): _sha256_file(path)
            for path in _all_nonmanifest_files(staging)
        }
        station_boundary = inputs.e3_manifest["station_truth_boundary"]
        model = inputs.training_manifest["training"][SELECTED_MODEL]
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": _utc_now(),
            "package_role": "publication-focused core three-figure evidence",
            "publication": {
                "atomic": True,
                "no_clobber": True,
                "primitive": "Linux renameat2(RENAME_NOREPLACE)",
            },
            "builder": {
                "source": str(builder_snapshot.relative_to(staging)),
                "sha256": _sha256_bytes(builder_payload),
            },
            "architecture_contract": {
                "selected_model": SELECTED_MODEL,
                "architecture": SELECTED_ARCHITECTURE,
                "ensemble_members": 3,
                "seeds": list(SEEDS),
                "parameter_count_per_member": PARAMETER_COUNT,
                "prepared_channels": len(inputs.training_manifest["features"]),
                "consumed_channels": len(CONSUMED_CHANNELS),
                "consumed_channel_names": list(CONSUMED_CHANNELS),
                "unet_widths": [16, 32, 64],
                "temporal_transformer_heads": 4,
                "residual_head_initialization": "zero",
                "seed_aggregation": "arithmetic mean of residuals",
                "reconstruction_anchor": "raw_fuxi",
                "selected_alpha": 1.0,
                "declared_model_code_sha256": FROZEN_MODEL_CODE_SHA256,
                "seed_runs": [
                    {
                        "seed": run["seed"],
                        "best_epoch": run["best_epoch"],
                        "best_validation_loss": run["best_validation_loss"],
                        "checkpoint_sha256": run["checkpoint_sha256"],
                    }
                    for run in model["runs"]
                ],
            },
            "evidence_labels": {
                "training": "2002-2017 training; 2018-2019 validation-only selection",
                "exploratory": "2020-2021 reused exploratory evaluation",
                "e2": "2022-2024 retrospective development audit; not an untouched final test",
                "e3": "2024 external-observational-target sensitivity; not an untouched temporal final test",
                "final": "2025 sealed; not opened or scored",
            },
            "access_boundary": {
                "scope": "this core3 reporting-builder invocation only",
                "manifest_and_metric_csv_only": True,
                "loaded_source_files": [
                    "TRAINING/manifest.json",
                    "E2/manifest.json",
                    *[f"E2/{relative}" for relative in E2_REQUIRED_ARTIFACTS],
                    "E3/manifest.json",
                    *[f"E3/{relative}" for relative in E3_REQUIRED_ARTIFACTS],
                ],
                "builder_opened": {
                    "checkpoint": False,
                    "prediction_store": False,
                    "forecast_array": False,
                    "target_array": False,
                    "raw_imd_data": False,
                    "raw_station_file": False,
                    "2025_data": False,
                },
                "upstream_e3_disclosure": {
                    "mixed_station_container_scanned": True,
                    "container_rows_scanned": station_boundary[
                        "container_rows_scanned"
                    ],
                    "container_max_date": station_boundary["container_date_max"],
                    "unselected_2025_plus_rows": station_boundary[
                        "unselected_2025_plus_rows"
                    ],
                    "2025_station_values_selected": 0,
                    "2025_station_values_scored": 0,
                    "filtering_contract": "exact 2024 date/station filtering before rainfall conversion",
                },
            },
            "sources": {
                "training": {
                    "directory": str(inputs.training_directory),
                    "manifest_sha256": inputs.training_manifest_sha256,
                    "verified_artifacts": dict(
                        inputs.verified_source_artifacts["training"]
                    ),
                },
                "e2": {
                    "directory": str(inputs.e2_directory),
                    "manifest_sha256": inputs.e2_manifest_sha256,
                    "verified_artifacts": dict(inputs.verified_source_artifacts["e2"]),
                    "training_manifest_binding_verified": True,
                },
                "e3": {
                    "directory": str(inputs.e3_directory),
                    "manifest_sha256": inputs.e3_manifest_sha256,
                    "verified_artifacts": dict(inputs.verified_source_artifacts["e3"]),
                    "e2_manifest_binding_verified": True,
                },
            },
            "uncertainty": {
                "e2": "10,000 shared year-stratified circular moving-block resamples; block length 13; 100 starts; six leads attached",
                "e3": "2,000 shared circular moving-block resamples; primary block length 13; 30 starts; six leads attached",
            },
            "bias_estimands": {
                "e2": "absolute value of pooled signed bias",
                "e3": "mean absolute case bias",
                "distinct_and_not_interchangeable": True,
            },
            "e3_comparison_roles": {
                "primary": "selected_adapter_vs_raw_fuxi RMSE",
                "raw_identity_vs_raw_fuxi": "secondary external-target sensitivity",
                "projection_vs_raw_identity": "secondary diagnostic",
            },
            "artifacts": artifacts,
        }
        (staging / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    parser.add_argument(
        "--training", type=Path, help="Frozen raw-identity training run"
    )
    parser.add_argument("--e2", type=Path, help="Canonical completed E2 directory")
    parser.add_argument("--e3", type=Path, help="Canonical completed E3 directory")
    parser.add_argument("--training-manifest-sha256")
    parser.add_argument("--e2-manifest-sha256")
    parser.add_argument("--e3-manifest-sha256")
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
    parser = _parser()
    args = parser.parse_args(argv)
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
        "--training": args.training,
        "--e2": args.e2,
        "--e3": args.e3,
        "--training-manifest-sha256": args.training_manifest_sha256,
        "--e2-manifest-sha256": args.e2_manifest_sha256,
        "--e3-manifest-sha256": args.e3_manifest_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"building requires: {', '.join(missing)}")
    result = build_package(
        args.training,
        args.e2,
        args.e3,
        args.output,
        training_manifest_sha256=args.training_manifest_sha256,
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
