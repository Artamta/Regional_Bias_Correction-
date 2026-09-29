#!/usr/bin/env python
"""Verify the full FuXi-to-IMD all-week attention-climatology experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

from fuxi_adapter.metrics import (  # noqa: E402
    compute_case_metrics,
    weighted_spatial_acc,
)


TRAIN_YEARS = list(range(2002, 2018))
VALIDATION_YEARS = [2018, 2019]
TEST_YEARS = [2020, 2021]
OFFSETS_DAYS = [-28, -21, -14, -7, 0, 7, 14, 21, 28]
ACTIVE_WEEKS = [1, 2, 3, 4, 5, 6]
INACTIVE_WEEKS: list[int] = []
LEAD_WEIGHTS = [1.0 / 6.0] * 6
VALIDATION_SCORE_COLUMN = "w1_w6_case_mean_rmse"
METHODS = (
    "raw_fuxi",
    "log_bias",
    "normal_climo_model",
    "attention_climo_model",
    "selected_model",
)
CANDIDATES = ("normal_climo_model", "attention_climo_model")
PARAMETER_COUNTS = {
    "normal_climo_model": 144_689,
    "attention_climo_model": 145_115,
}
SEEDS = [42, 43, 44]
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
FULL_FUXI_CONTEXT = False
TRAINING_ANCHOR = "log_bias"
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    "normal_climo_model": "Normal-climatology temporal (W1–W6)",
    "attention_climo_model": "Attention-climatology temporal (W1–W6)",
    "selected_model": "Validation-selected",
}
PAIRED_COMPARISONS = (
    ("log_bias", "raw_fuxi"),
    ("normal_climo_model", "raw_fuxi"),
    ("attention_climo_model", "raw_fuxi"),
    ("selected_model", "raw_fuxi"),
    ("normal_climo_model", "log_bias"),
    ("attention_climo_model", "log_bias"),
    ("attention_climo_model", "normal_climo_model"),
    ("selected_model", "log_bias"),
    ("selected_model", "normal_climo_model"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def configure_variant(
    *,
    large_model: bool,
    regularized_large: bool = False,
    full_fuxi_context: bool = False,
    training_anchor: str = "log_bias",
) -> None:
    """Set the expected capacity and labels for the requested W1-W6 run."""
    global PARAMETER_COUNTS, METHOD_LABELS, FULL_FUXI_CONTEXT, TRAINING_ANCHOR
    if training_anchor not in {"log_bias", "raw_fuxi"}:
        raise ValueError(f"unsupported training anchor: {training_anchor}")
    TRAINING_ANCHOR = training_anchor
    if full_fuxi_context and large_model:
        raise ValueError("full FuXi context verification expects the compact model")
    FULL_FUXI_CONTEXT = full_fuxi_context
    if regularized_large:
        large_model = True
    if not large_model:
        if full_fuxi_context:
            METHOD_LABELS = {
                "raw_fuxi": "Raw FuXi",
                "log_bias": "Log-bias",
                "normal_climo_model": (
                    "Full-context normal-climatology temporal (W1–W6)"
                ),
                "attention_climo_model": (
                    "Full-context attention-climatology temporal (W1–W6)"
                ),
                "selected_model": "Validation-selected",
            }
        return
    PARAMETER_COUNTS = {
        "normal_climo_model": 2_544_049,
        "attention_climo_model": 2_544_475,
    }
    METHOD_LABELS = {
        "raw_fuxi": "Raw FuXi",
        "log_bias": "Log-bias",
        "normal_climo_model": "Large normal-climatology temporal (W1–W6)",
        "attention_climo_model": "Large attention-climatology temporal (W1–W6)",
        "selected_model": "Validation-selected",
    }
    if regularized_large:
        METHOD_LABELS = {
            "raw_fuxi": "Raw FuXi",
            "log_bias": "Log-bias",
            "normal_climo_model": (
                "Regularized large normal-climatology temporal (W1–W6)"
            ),
            "attention_climo_model": (
                "Regularized large attention-climatology temporal (W1–W6)"
            ),
            "selected_model": "Validation-selected",
        }


def latest_full(
    *,
    large_model: bool,
    regularized_large: bool = False,
    full_fuxi_context: bool = False,
) -> Path:
    result_name = (
        "fuxi_imd_full_context_compact_allweeks"
        if full_fuxi_context
        else (
            "fuxi_imd_attention_climatology_big_allweeks_regularized"
            if regularized_large
            else (
                "fuxi_imd_attention_climatology_big_allweeks"
                if large_model
                else "fuxi_imd_attention_climatology_allweeks"
            )
        )
    )
    root = HERE / "results" / result_name
    runs = sorted(
        candidate
        for candidate in root.glob("full_*")
        if (candidate / "manifest.json").is_file()
    )
    if not runs:
        raise FileNotFoundError(
            "no completed full W1-W6 IMD attention-climatology run found"
        )
    return runs[-1]


def parse_utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise AssertionError(f"timestamp is not timezone-aware: {value}")
    return timestamp


def require_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise AssertionError(f"{name} is not boolean: {value!r}")


def torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise AssertionError(f"checkpoint payload is not a mapping: {path}")
    return payload


def verify_manifest_contract(run: Path, manifest: Mapping[str, Any]) -> None:
    if not run.name.startswith("full_"):
        raise AssertionError(f"run directory is not a full run: {run.name}")
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise AssertionError("manifest is not a completed full experiment")
    if manifest.get("observation_source") != "IMD":
        raise AssertionError("manifest observation source is not IMD")
    if TRAINING_ANCHOR == "raw_fuxi":
        expected_anchor_metadata = {
            "training_anchor": "raw_fuxi",
            "uses_fitted_log_bias_in_neural_training": False,
            "log_bias_role": "reporting_only",
            "experiment_role": "matched no-fitted-log-bias neural ablation",
        }
        for key, expected in expected_anchor_metadata.items():
            if manifest.get(key) != expected:
                raise AssertionError(f"raw-identity manifest field differs: {key}")
    if manifest.get("split_counts") != {"train": 560, "validation": 70, "test": 70}:
        raise AssertionError("split counts differ from 560/70/70")
    if manifest.get("split_years") != {
        "train": TRAIN_YEARS,
        "validation": VALIDATION_YEARS,
        "test": TEST_YEARS,
    }:
        raise AssertionError("split years differ")
    if (
        manifest.get("test_count_used") != 70
        or manifest.get("active_leads") != ACTIVE_WEEKS
        or manifest.get("support_cells") != 171
    ):
        raise AssertionError("test count, active leads, or IMD support differs")
    if (
        manifest.get("inactive_lead_identity_verified") is not True
        or manifest.get("lead_weights") != LEAD_WEIGHTS
        or manifest.get("primary_validation_metric")
        != "equal-case W1-W6 area-weighted RMSE"
    ):
        raise AssertionError("W1-W6 training and verification scope differs")
    definitions = manifest.get("metric_definitions", {})
    expected_definitions = {
        "acc": "area-weighted spatial correlation after subtracting the common fixed 2002-2017 IMD climatology from forecast and observation",
        "pcc": "area-weighted spatial correlation of absolute weekly forecast and observation; no climatology",
        "mse_skill_vs_imd_climatology": "1 - mean forecast MSE / mean fixed-IMD-climatology MSE",
    }
    if definitions != expected_definitions:
        raise AssertionError("ACC, PCC, or MSESS manifest definition differs")
    if manifest.get("prediction_store_roundtrip_verified") is not True:
        raise AssertionError("prediction-store round trip was not recorded as verified")
    if manifest.get("quarantined_final_initialization_years") != [2025]:
        raise AssertionError(
            "2025 is not recorded as a quarantined final initialization year"
        )

    sources = manifest.get("source_stores")
    expected_names = [f"{year}.zarr" for year in range(2002, 2022)]
    if (
        not isinstance(sources, list)
        or [Path(value).name for value in sources] != expected_names
    ):
        raise AssertionError("IMD source-store years differ")
    if any("/daily/imd/tp/" not in str(value).replace("\\", "/") for value in sources):
        raise AssertionError(
            "one or more source stores are not IMD precipitation stores"
        )
    if any(Path(value).name == "2025.zarr" for value in sources):
        raise AssertionError("the quarantined 2025 IMD store was accessed")


def verify_climatology_contract(run: Path, manifest: Mapping[str, Any]) -> None:
    climatology = manifest.get("climatology", {})
    if climatology.get("attention_offsets_days") != OFFSETS_DAYS:
        raise AssertionError("manifest attention-climatology offsets differ")
    if climatology.get("verification_reference_fixed") is not True:
        raise AssertionError("verification climatology is not recorded as fixed")
    normal = str(climatology.get("normal", ""))
    if "2002-2017" not in normal or "31-day IMD" not in normal:
        raise AssertionError("normal IMD climatology provenance differs")
    difference = float(climatology.get("zero_offset_max_abs_difference", np.nan))
    if not np.isfinite(difference) or not 0.0 <= difference <= 2.0e-6:
        raise AssertionError(
            f"zero-offset climatology difference exceeds 2e-6: {difference}"
        )

    normalization = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    context = normalization.get("spatial_context", {})
    if bool(context.get("enabled", False)) is not FULL_FUXI_CONTEXT:
        raise AssertionError("full-domain FuXi context mode differs")
    if manifest.get("spatial_context") != context:
        raise AssertionError("manifest and normalization context contracts differ")
    if FULL_FUXI_CONTEXT:
        if context.get("full_domain_channels") != [
            "log_fuxi_mean",
            "log_fuxi_spread",
            "fuxi_t2m_weekly",
        ]:
            raise AssertionError("full-domain FuXi channel contract differs")
        if context.get("target_support_cells") != 171:
            raise AssertionError("full-context target support is not 171 cells")
        if context.get("normalization_fit") != (
            "training cases and positive target weights only"
        ):
            raise AssertionError("full-context normalization was not train-only")
    attention = normalization.get("climatology_attention", {})
    if attention.get("offsets_days") != OFFSETS_DAYS:
        raise AssertionError("normalization attention offsets differ")
    if attention.get("source") != "IMD training years 2002-2017 only":
        raise AssertionError("attention-bank source is not training-only IMD")
    if (
        attention.get("verification_reference")
        != "fixed zero-offset 31-day climatology"
    ):
        raise AssertionError("normalization verification reference differs")
    channels = normalization.get("input_channels")
    appended = [
        *[f"imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
        *[f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
    ]
    if (
        not isinstance(channels, list)
        or len(channels) != 29
        or channels[-18:] != appended
    ):
        raise AssertionError("29-channel climatology-bank feature contract differs")
    if manifest.get("features") != channels:
        raise AssertionError("manifest and normalization feature lists differ")


def verify_training_anchor_contract(run: Path) -> None:
    """Verify the explicit train-only residual-reference artifact."""

    if TRAINING_ANCHOR != "raw_fuxi":
        return
    path = run / "models" / "training_anchor_contract.npz"
    if not path.is_file():
        raise AssertionError("raw-identity training-anchor contract is missing")
    with np.load(path, allow_pickle=False) as contract:
        if set(contract.files) != {
            "anchor_kind",
            "target_scale",
            "fitted_target_years",
        }:
            raise AssertionError("training-anchor contract fields differ")
        if str(contract["anchor_kind"].item()) != "raw_fuxi":
            raise AssertionError("training-anchor contract is not raw FuXi")
        target_scale = np.asarray(contract["target_scale"], dtype=np.float64)
        fitted_years = np.asarray(contract["fitted_target_years"], dtype=np.int64)
    if target_scale.shape != (6,) or not np.isfinite(target_scale).all():
        raise AssertionError("raw-residual target scale is invalid")
    if np.any(target_scale <= 0.0):
        raise AssertionError("raw-residual target scale is not positive")
    if not np.array_equal(fitted_years, np.asarray(TRAIN_YEARS, dtype=np.int64)):
        raise AssertionError("raw-residual target scale was not fit on 2002-2017")


def verify_selection(run: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    selection_path = run / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen" or selection.get("smoke") is not False:
        raise AssertionError("selection is not a frozen full-run decision")
    if selection.get("selection_scope") != "validation_only":
        raise AssertionError("selection was not scoped to validation only")
    if selection.get("observation_source") != "IMD":
        raise AssertionError("selection observation source is not IMD")
    if TRAINING_ANCHOR == "raw_fuxi":
        expected_anchor_metadata = {
            "training_anchor": "raw_fuxi",
            "uses_fitted_log_bias_in_neural_training": False,
            "log_bias_role": "reporting_only",
            "selection_policy": (
                "matched_normal_architecture_primary; attention_secondary_only"
            ),
        }
        for key, expected in expected_anchor_metadata.items():
            if selection.get(key) != expected:
                raise AssertionError(f"raw-identity selection field differs: {key}")
    if selection.get("test_predictions_created") is not False:
        raise AssertionError("selection was not frozen before test prediction creation")
    if selection.get("train_years") != TRAIN_YEARS:
        raise AssertionError("selection train years differ")
    if selection.get("validation_years") != VALIDATION_YEARS:
        raise AssertionError("selection validation years differ")
    if selection.get("test_years_quarantined_during_selection") != TEST_YEARS:
        raise AssertionError("test years were not quarantined during selection")
    if selection.get("candidate_set") != list(CANDIDATES):
        raise AssertionError("frozen candidate set differs")
    if selection.get("attention_bank_offsets_days") != OFFSETS_DAYS:
        raise AssertionError("frozen attention offsets differ")
    inactive_key = (
        "inactive_weeks_exact_training_anchor"
        if TRAINING_ANCHOR == "raw_fuxi"
        else "inactive_weeks_exact_log_bias"
    )
    if (
        selection.get("active_weeks") != ACTIVE_WEEKS
        or selection.get(inactive_key) != INACTIVE_WEEKS
        or selection.get("lead_weights") != LEAD_WEIGHTS
        or selection.get("primary_validation_metric")
        != "equal-case W1-W6 area-weighted RMSE"
    ):
        raise AssertionError("frozen W1-W6 selection contract differs")

    selected_model = selection.get("selected_model")
    if selected_model not in (TRAINING_ANCHOR, *CANDIDATES):
        raise AssertionError(f"unknown selected model: {selected_model}")
    selected_alpha = float(selection.get("selected_alpha", np.nan))
    if not np.isfinite(selected_alpha) or not 0.0 <= selected_alpha <= 1.0:
        raise AssertionError("selected alpha is outside [0, 1]")
    if manifest.get("selected_model") != selected_model or not np.isclose(
        float(manifest.get("selected_alpha")), selected_alpha, rtol=0.0, atol=0.0
    ):
        raise AssertionError("manifest and frozen selection differ")
    if manifest.get("selection_frozen_utc") != selection.get("frozen_utc"):
        raise AssertionError("manifest selection timestamp differs")
    if sha256_file(selection_path) != manifest.get("selection_sha256"):
        raise AssertionError("frozen selection hash differs")

    training = manifest.get("training", {})
    checkpoint_hashes = {
        candidate: [
            record["checkpoint_sha256"] for record in training[candidate]["runs"]
        ]
        for candidate in CANDIDATES
    }
    if selection.get("checkpoint_sha256") != checkpoint_hashes:
        raise AssertionError("selection references different checkpoint hashes")

    frozen = parse_utc(str(selection["frozen_utc"]))
    test_started = parse_utc(str(manifest["test_evaluation_started_utc"]))
    completed = parse_utc(str(manifest["created_utc"]))
    if not frozen < test_started <= completed:
        raise AssertionError("selection was not frozen before test evaluation")

    table = pd.read_csv(run / "metrics" / "validation_selection.csv")
    if len(table) != 2 or tuple(table.candidate) != CANDIDATES:
        raise AssertionError("validation selection candidates differ")
    scan = pd.read_csv(run / "metrics" / "validation_alpha_scan.csv")
    if set(scan.candidate) != set(CANDIDATES):
        raise AssertionError("validation alpha-scan candidates differ")
    for candidate in CANDIDATES:
        subset = scan.loc[scan.candidate.eq(candidate)]
        if subset.empty or not subset.alpha.between(0.0, 1.0).all():
            raise AssertionError(f"invalid alpha scan for {candidate}")
        if VALIDATION_SCORE_COLUMN not in subset:
            raise AssertionError("W1-W6 validation score column is missing")
        best = subset.loc[subset[VALIDATION_SCORE_COLUMN].idxmin()]
        chosen = table.loc[table.candidate.eq(candidate)].iloc[0]
        if not np.isclose(
            float(chosen.alpha), float(best.alpha), rtol=0.0, atol=1.0e-12
        ):
            raise AssertionError(f"selected alpha is not scan-optimal for {candidate}")
        if not np.isclose(
            float(chosen[VALIDATION_SCORE_COLUMN]),
            float(best[VALIDATION_SCORE_COLUMN]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError(f"selected validation score differs for {candidate}")
        skills = [
            float(chosen[f"rmse_skill_vs_{TRAINING_ANCHOR}_{year}_pct"])
            for year in VALIDATION_YEARS
        ]
        recorded_guard = require_bool(
            chosen.improves_every_validation_year,
            name=f"{candidate}.improves_every_validation_year",
        )
        if recorded_guard != all(value > 0.0 for value in skills):
            raise AssertionError(
                f"{TRAINING_ANCHOR} validation guard differs for {candidate}"
            )

    normal = table.loc[table.candidate.eq(CANDIDATES[0])].iloc[0]
    attention = table.loc[table.candidate.eq(CANDIDATES[1])].iloc[0]
    normal_guard = require_bool(
        normal.improves_every_validation_year,
        name="normal.improves_every_validation_year",
    )
    attention_guard = require_bool(
        attention.improves_every_validation_year,
        name="attention.improves_every_validation_year",
    )
    attention_skills = [
        float(attention[f"attention_rmse_skill_vs_normal_{year}_pct"])
        for year in VALIDATION_YEARS
    ]
    attention_vs_normal_guard = require_bool(
        attention.attention_beats_normal_every_year,
        name="attention.attention_beats_normal_every_year",
    )
    if attention_vs_normal_guard != all(value > 0.0 for value in attention_skills):
        raise AssertionError("attention-vs-normal validation guard differs")
    if TRAINING_ANCHOR == "raw_fuxi":
        expected_model, expected_alpha = (
            (CANDIDATES[0], float(normal.alpha)) if normal_guard else ("raw_fuxi", 0.0)
        )
    elif attention_guard and attention_vs_normal_guard:
        expected_model, expected_alpha = CANDIDATES[1], float(attention.alpha)
    elif normal_guard:
        expected_model, expected_alpha = CANDIDATES[0], float(normal.alpha)
    elif attention_guard:
        expected_model, expected_alpha = CANDIDATES[1], float(attention.alpha)
    else:
        expected_model, expected_alpha = "log_bias", 0.0
    if selected_model != expected_model or not np.isclose(
        selected_alpha, expected_alpha, rtol=0.0, atol=1.0e-12
    ):
        raise AssertionError("frozen decision cannot be rebuilt from validation scores")
    return selection


def verify_training(run: Path, manifest: Mapping[str, Any]) -> None:
    training = manifest.get("training", {})
    if set(training) != set(CANDIDATES):
        raise AssertionError("training candidate set differs")
    required_history = {
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "train_smooth_l1",
        "validation_smooth_l1",
        "train_acc_loss",
        "validation_acc_loss",
        "train_mean_spatial_acc",
        "validation_mean_spatial_acc",
        "train_mean_bias_squared",
        "validation_mean_bias_squared",
    }
    for candidate in CANDIDATES:
        metadata = training[candidate]
        if int(metadata.get("parameter_count", -1)) != PARAMETER_COUNTS[candidate]:
            raise AssertionError(
                f"{candidate} parameter count differs from {PARAMETER_COUNTS[candidate]:,}"
            )
        if metadata.get("train_case_count") != 560:
            raise AssertionError(f"wrong training count for {candidate}")
        if metadata.get("validation_case_count") != 70:
            raise AssertionError(f"wrong validation count for {candidate}")
        if (
            metadata.get("lead_weights") != LEAD_WEIGHTS
            or metadata.get("inactive_lead_count") != 0
            or metadata.get("loss_coefficients") != LOSS_COEFFICIENTS
        ):
            raise AssertionError(f"W1-W6 loss contract differs for {candidate}")
        records = metadata.get("runs", [])
        if metadata.get("seeds") != SEEDS or len(records) != 3:
            raise AssertionError(f"three-seed ensemble differs for {candidate}")
        if [int(record.get("seed", -1)) for record in records] != SEEDS:
            raise AssertionError(f"checkpoint seed ordering differs for {candidate}")

        for record in records:
            checkpoint = run / str(record["checkpoint"])
            history_path = run / str(record["history"])
            if not checkpoint.is_file() or not history_path.is_file():
                raise AssertionError(f"missing model artifact for {candidate}")
            if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
                raise AssertionError(f"checkpoint hash differs: {checkpoint}")

            history = pd.read_csv(history_path)
            if not required_history.issubset(history.columns) or history.empty:
                raise AssertionError(f"training history columns differ: {history_path}")
            numeric = history[list(required_history)].to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise AssertionError(f"non-finite training history: {history_path}")
            epochs = history.epoch.to_numpy(dtype=np.float64)
            if not np.array_equal(epochs, np.arange(len(history), dtype=np.float64)):
                raise AssertionError(
                    f"training epochs are not contiguous: {history_path}"
                )
            best_epoch = int(record["best_epoch"])
            best_loss = float(record["best_validation_loss"])
            selected = history.loc[history.epoch.astype(int).eq(best_epoch)]
            if len(selected) != 1 or not np.isclose(
                float(selected.validation_loss.iloc[0]),
                best_loss,
                rtol=0.0,
                atol=2.0e-12,
            ):
                raise AssertionError(f"best history row differs: {history_path}")
            if not np.isclose(
                float(history.validation_loss.min()), best_loss, rtol=0.0, atol=2.0e-12
            ):
                raise AssertionError(
                    f"checkpoint is not validation-selected: {checkpoint}"
                )

            payload = torch_load(checkpoint)
            state = payload.get("model_state_dict")
            checkpoint_lead_weights = np.asarray(
                payload.get("lead_weights", []), dtype=np.float64
            )
            if (
                int(payload.get("seed", -1)) != int(record["seed"])
                or int(payload.get("best_epoch", -1)) != best_epoch
                or not np.isclose(
                    float(payload.get("best_validation_loss", np.nan)),
                    best_loss,
                    rtol=0.0,
                    atol=2.0e-12,
                )
                or not isinstance(state, Mapping)
                or not state
                or not np.allclose(
                    checkpoint_lead_weights,
                    np.asarray(LEAD_WEIGHTS, dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-8,
                )
                or payload.get("loss_coefficients") != LOSS_COEFFICIENTS
            ):
                raise AssertionError(f"checkpoint metadata differs: {checkpoint}")
            has_conditioner = any(str(key).startswith("conditioner.") for key in state)
            if has_conditioner != (candidate == "attention_climo_model"):
                raise AssertionError(f"checkpoint architecture differs: {checkpoint}")


def verify_predictions(run: Path, selection: Mapping[str, Any]) -> xr.Dataset:
    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as source:
        dataset = source.load()
    expected_sizes = {
        "method": 5,
        "init": 70,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }
    if dict(dataset.sizes) != expected_sizes:
        raise AssertionError(f"prediction dimensions differ: {dict(dataset.sizes)}")
    if tuple(dataset.method.values.tolist()) != METHODS:
        raise AssertionError("prediction method ordering differs")
    if not np.array_equal(dataset.lead_week.values, np.arange(1, 7)):
        raise AssertionError("prediction lead coordinates differ")
    expected_variables = {
        "prediction",
        "truth_imd",
        "fixed_imd_climatology",
        "area_weight_km2",
    }
    if set(dataset.data_vars) != expected_variables:
        raise AssertionError("prediction-store variables differ")
    expected_dimensions = {
        "prediction": ("method", "init", "lead_week", "latitude", "longitude"),
        "truth_imd": ("init", "lead_week", "latitude", "longitude"),
        "fixed_imd_climatology": ("init", "lead_week", "latitude", "longitude"),
        "area_weight_km2": ("latitude", "longitude"),
    }
    for variable, dimensions in expected_dimensions.items():
        if dataset[variable].dims != dimensions:
            raise AssertionError(f"{variable} dimensions differ")
    years, counts = np.unique(
        pd.DatetimeIndex(dataset.init.values).year, return_counts=True
    )
    if not np.array_equal(years, TEST_YEARS) or not np.array_equal(counts, [35, 35]):
        raise AssertionError("test initialization years differ")
    inactive_key = (
        "inactive_weeks_exact_training_anchor"
        if TRAINING_ANCHOR == "raw_fuxi"
        else "inactive_weeks_exact_log_bias"
    )
    if (
        dataset.attrs.get("selection_scope") != "validation_only"
        or dataset.attrs.get("selected_model") != selection["selected_model"]
        or not np.isclose(
            float(dataset.attrs.get("selected_alpha")),
            float(selection["selected_alpha"]),
            rtol=0.0,
            atol=0.0,
        )
        or list(dataset.attrs.get("attention_bank_offsets_days", [])) != OFFSETS_DAYS
        or list(dataset.attrs.get("active_weeks", [])) != ACTIVE_WEEKS
        or list(dataset.attrs.get(inactive_key, [])) != INACTIVE_WEEKS
        or list(dataset.attrs.get("lead_weights", [])) != LEAD_WEIGHTS
        or bool(dataset.attrs.get("full_fuxi_context", False)) is not FULL_FUXI_CONTEXT
        or dataset.attrs.get("verification_climatology")
        != "fixed training-only 31-day IMD climatology"
        or dataset.attrs.get("acc_definition")
        != "common-reference spatial ACC using fixed 2002-2017 IMD climatology"
        or dataset.attrs.get("pcc_definition")
        != "spatial PCC of absolute weekly rainfall; no climatology"
        or dataset.attrs.get("smoke") is not False
    ):
        raise AssertionError("prediction-store experiment metadata differs")
    if (
        TRAINING_ANCHOR == "raw_fuxi"
        and dataset.attrs.get("training_anchor") != "raw_fuxi"
    ):
        raise AssertionError("prediction store does not declare the raw FuXi anchor")

    weights = np.asarray(dataset.area_weight_km2.values, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise AssertionError("IMD area weights are invalid")
    support = weights > 0.0
    if int(support.sum()) != 171:
        raise AssertionError("IMD supported-cell count differs")
    truth = np.asarray(dataset.truth_imd.values, dtype=np.float32)
    climatology = np.asarray(dataset.fixed_imd_climatology.values, dtype=np.float32)
    if not np.isfinite(truth[..., support]).all():
        raise AssertionError("IMD truth is not finite on support")
    if not np.isfinite(climatology[..., support]).all() or np.any(
        climatology[..., support] < 0.0
    ):
        raise AssertionError("fixed IMD climatology is invalid on support")

    values = np.asarray(dataset.prediction.values, dtype=np.float32)
    if not np.isfinite(values[..., support]).all():
        raise AssertionError("W1-W6 predictions are not finite on IMD support")
    if np.any(values[..., support] < 0.0):
        raise AssertionError("W1-W6 predictions are negative on IMD support")
    if not np.isnan(values[..., ~support]).all():
        raise AssertionError("predictions outside IMD support are not NaN")
    selected = np.asarray(dataset.prediction.sel({"method": "selected_model"}).values)
    source = np.asarray(
        dataset.prediction.sel({"method": selection["selected_model"]}).values
    )
    if not np.array_equal(selected, source, equal_nan=True):
        raise AssertionError("selected-model alias differs from its frozen source")
    return dataset


def case_rmse(
    prediction: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Reproduce the experiment's case-wise area-weighted RMSE."""

    support = weights > 0.0
    error = prediction[..., support].astype(np.float64) - truth[..., support]
    normalized = weights[support] / weights[support].sum()
    return np.sqrt(np.sum(error**2 * normalized[None, None], axis=-1))


def rebuild_case_metrics(dataset: xr.Dataset) -> pd.DataFrame:
    truth = np.asarray(dataset.truth_imd.values, dtype=np.float32)
    climatology = np.asarray(dataset.fixed_imd_climatology.values, dtype=np.float32)
    weights = np.asarray(dataset.area_weight_km2.values, dtype=np.float64)
    initializations = dataset.init.values.astype("datetime64[D]")
    case_ids = [np.datetime_as_string(value, unit="D") for value in initializations]
    support = weights > 0.0
    climatology_mse = case_rmse(climatology, truth, weights).reshape(-1) ** 2
    frames = []
    for method in METHODS:
        prediction = np.asarray(
            dataset.prediction.sel({"method": method}).values, dtype=np.float32
        )
        frame = compute_case_metrics(
            truth,
            prediction,
            truth - climatology,
            prediction - climatology,
            weights,
            predictor=method,
            case_ids=case_ids,
            leads=np.arange(1, 7),
            valid_mask=support,
        ).rename(
            columns={"predictor": "method", "case_id": "init", "lead": "lead_week"}
        )
        frame.insert(0, "split", "test")
        frame.insert(2, "year", pd.DatetimeIndex(frame.init).year)
        frame["spatial_acc_common_imd"] = frame["acc"]
        frame["mse"] = frame["rmse"] ** 2
        frame["climatology_mse"] = climatology_mse
        frame["pcc"] = [
            weighted_spatial_acc(
                truth[case_index, lead_index],
                prediction[case_index, lead_index],
                weights,
                support,
            )
            for case_index in range(len(initializations))
            for lead_index in range(6)
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def verify_summary_metrics(run: Path, case_metrics: pd.DataFrame) -> None:
    grouped = case_metrics.groupby(["method", "lead_week"], as_index=False).agg(
        case_count=("init", "size"),
        pcc=("pcc", "mean"),
        mean_mse=("mse", "mean"),
        climatology_mean_mse=("climatology_mse", "mean"),
    )
    grouped["mse_skill_vs_imd_climatology"] = 1.0 - (
        grouped.mean_mse / grouped.climatology_mean_mse
    )
    raw_pcc = grouped.loc[grouped.method.eq("raw_fuxi")].set_index("lead_week").pcc
    grouped["delta_pcc_vs_raw"] = [
        float(row.pcc - raw_pcc.loc[row.lead_week])
        for row in grouped.itertuples(index=False)
    ]
    columns = [
        "method",
        "lead_week",
        "case_count",
        "pcc",
        "mean_mse",
        "climatology_mean_mse",
        "mse_skill_vs_imd_climatology",
        "delta_pcc_vs_raw",
    ]
    stored = pd.read_csv(run / "metrics" / "summary_by_lead.csv")
    if len(stored) != len(METHODS) * 6 or not set(columns).issubset(stored):
        raise AssertionError("lead-summary PCC or MSESS columns differ")
    expected = (
        grouped[columns].sort_values(["method", "lead_week"]).reset_index(drop=True)
    )
    actual = stored[columns].sort_values(["method", "lead_week"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    active = case_metrics.loc[case_metrics.lead_week.isin(ACTIVE_WEEKS)]
    headline = active.groupby("method", as_index=False).agg(
        cases=("init", "nunique"),
        acc=("acc", "mean"),
        pcc=("pcc", "mean"),
        rmse_mm_day=("rmse", "mean"),
        mae_mm_day=("mae", "mean"),
        bias_mm_day=("bias", "mean"),
        mean_mse=("mse", "mean"),
        climatology_mean_mse=("climatology_mse", "mean"),
    )
    headline["mse_skill_vs_imd_climatology"] = 1.0 - (
        headline.mean_mse / headline.climatology_mean_mse
    )
    headline["method_label"] = headline.method.map(METHOD_LABELS)
    headline_columns = [
        "method",
        "method_label",
        "cases",
        "acc",
        "pcc",
        "rmse_mm_day",
        "mae_mm_day",
        "bias_mm_day",
        "mean_mse",
        "climatology_mean_mse",
        "mse_skill_vs_imd_climatology",
    ]
    active_path = run / "metrics" / "active_headline_metrics.csv"
    spatial_path = run / "metrics" / "active_spatial_summary.csv"
    if not active_path.is_file() or not spatial_path.is_file():
        raise AssertionError("active_* W1-W6 metric files are missing")
    if (run / "metrics" / "late_headline_metrics.csv").exists() or (
        run / "metrics" / "late_spatial_summary.csv"
    ).exists():
        raise AssertionError("legacy late_* metric files are present in a W1-W6 run")
    stored_headline = pd.read_csv(active_path)
    if len(stored_headline) != len(METHODS) or not set(headline_columns).issubset(
        stored_headline
    ):
        raise AssertionError("active headline schema differs")
    expected_headline = (
        headline[headline_columns].sort_values("method").reset_index(drop=True)
    )
    actual_headline = (
        stored_headline[headline_columns].sort_values("method").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        actual_headline,
        expected_headline,
        check_dtype=False,
        check_exact=False,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def verify_paired_w1_w6(run: Path, case_metrics: pd.DataFrame) -> None:
    intervals = pd.read_csv(run / "metrics" / "paired_skill.csv")
    active = intervals.loc[intervals.lead_scope.eq("W1-W6")].copy()
    expected_keys = {
        (method, baseline, metric)
        for method, baseline in PAIRED_COMPARISONS
        for metric in ("acc", "rmse", "mae", "pcc")
    }
    actual_keys = set(zip(active.method, active.baseline, active.metric))
    if len(active) != len(expected_keys) or actual_keys != expected_keys:
        raise AssertionError("W1-W6 paired comparison rows differ")
    if (
        not np.all(active.paired_case_count == 70)
        or not np.all(active.block_length == 13)
        or not np.all(active.n_resamples == 2000)
        or not np.all(active.seed == 42)
    ):
        raise AssertionError("W1-W6 paired bootstrap contract differs")
    if not np.isfinite(
        active[
            ["model_mean", "baseline_mean", "effect", "ci_lower", "ci_upper"]
        ].to_numpy(dtype=np.float64)
    ).all() or np.any(active.ci_lower > active.ci_upper):
        raise AssertionError("W1-W6 paired results contain invalid values")

    selected = case_metrics.loc[case_metrics.lead_week.isin(ACTIVE_WEEKS)]
    case_order = sorted(selected.init.unique())
    for row in active.itertuples(index=False):
        pivot = (
            selected.loc[selected.method.isin((row.method, row.baseline))]
            .pivot_table(
                index="init", columns="method", values=row.metric, aggfunc="mean"
            )
            .reindex(case_order)
        )
        model = pivot[row.method].to_numpy(dtype=np.float64)
        baseline = pivot[row.baseline].to_numpy(dtype=np.float64)
        if not np.isfinite(model).all() or not np.isfinite(baseline).all():
            raise AssertionError("W1-W6 paired metric reconstruction is incomplete")
        model_mean = float(model.mean())
        baseline_mean = float(baseline.mean())
        effect = (
            model_mean - baseline_mean
            if row.metric in ("acc", "pcc")
            else 100.0 * (baseline_mean - model_mean) / baseline_mean
        )
        if not (
            np.isclose(float(row.model_mean), model_mean, rtol=0.0, atol=2.0e-13)
            and np.isclose(
                float(row.baseline_mean), baseline_mean, rtol=0.0, atol=2.0e-13
            )
            and np.isclose(float(row.effect), effect, rtol=0.0, atol=2.0e-13)
        ):
            raise AssertionError(
                f"W1-W6 paired value differs: {row.method} vs {row.baseline}, {row.metric}"
            )


def verify_metrics(run: Path, dataset: xr.Dataset) -> None:
    rebuilt = rebuild_case_metrics(dataset)
    stored = pd.read_csv(run / "metrics" / "case_metrics.csv")
    if len(stored) != 70 * 6 * len(METHODS):
        raise AssertionError("stored case-metric row count differs")
    if not np.all(stored.valid_cells.to_numpy() == 171):
        raise AssertionError("stored case metrics do not use all 171 IMD cells")
    if not np.array_equal(
        stored.spatial_acc_common_imd.to_numpy(), stored.acc.to_numpy(), equal_nan=True
    ):
        raise AssertionError("spatial_acc_common_imd is not the common-reference ACC")
    pd.testing.assert_frame_equal(
        stored,
        rebuilt,
        check_dtype=False,
        check_exact=False,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    verify_summary_metrics(run, rebuilt)
    verify_paired_w1_w6(run, rebuilt)


def verify_hashes(run: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts", {})
    required = {
        "normalization.json",
        "selection.json",
        "metrics/case_metrics.csv",
        "metrics/summary_by_lead.csv",
        "metrics/paired_skill.csv",
        "metrics/active_headline_metrics.csv",
        "metrics/active_spatial_summary.csv",
        "metrics/validation_alpha_scan.csv",
        "metrics/validation_selection.csv",
        "predictions.zarr",
    }
    if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
        raise AssertionError("required artifact hashes are missing")

    prediction_store = run / "predictions.zarr"
    actual_artifacts = {
        str(path.relative_to(run))
        for path in run.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and prediction_store not in path.parents
    }
    actual_artifacts.add("predictions.zarr")
    if set(artifacts) != actual_artifacts:
        missing = sorted(actual_artifacts - set(artifacts))
        stale = sorted(set(artifacts) - actual_artifacts)
        raise AssertionError(
            f"artifact hash inventory differs; unlisted={missing}, missing_files={stale}"
        )
    for relative, expected in artifacts.items():
        path = run / relative
        actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
        if actual != expected:
            raise AssertionError(f"artifact hash differs: {relative}")

    code_hashes = manifest.get("code_sha256", {})
    if not isinstance(code_hashes, Mapping) or not code_hashes:
        raise AssertionError("code hash inventory is missing")
    for filename, expected in code_hashes.items():
        path = run / "code" / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(f"archived code hash differs: {filename}")
        if artifacts.get(f"code/{filename}") != expected:
            raise AssertionError(f"code and artifact hashes disagree: {filename}")


def verify(run: Path) -> None:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_contract(run, manifest)
    verify_climatology_contract(run, manifest)
    verify_training_anchor_contract(run)
    selection = verify_selection(run, manifest)
    verify_training(run, manifest)
    dataset = verify_predictions(run, selection)
    verify_metrics(run, dataset)
    verify_hashes(run, manifest)
    print(
        "PASS: completed full manifest, 560/70/70 IMD split, 171-cell support, "
        "fixed climatology and offsets, frozen W1-W6 validation selection, exact model "
        "sizes, three-seed histories/checkpoints, all weeks active, recomputed common-IMD "
        "ACC/PCC/MSESS and paired W1-W6 metrics, and complete artifact hashes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument(
        "--large-model",
        action="store_true",
        help="verify the 2.54M-parameter W1-W6 model variant",
    )
    parser.add_argument(
        "--regularized-large",
        action="store_true",
        help="verify the strongly regularized 2.54M-parameter W1-W6 variant",
    )
    parser.add_argument(
        "--full-fuxi-context",
        action="store_true",
        help="verify the compact all-week full-domain FuXi-context variant",
    )
    parser.add_argument(
        "--training-anchor",
        choices=("log_bias", "raw_fuxi"),
        default="log_bias",
        help="expected neural reconstruction anchor",
    )
    args = parser.parse_args()
    configure_variant(
        large_model=args.large_model or args.regularized_large,
        regularized_large=args.regularized_large,
        full_fuxi_context=args.full_fuxi_context,
        training_anchor=args.training_anchor,
    )
    verify(
        (
            args.run
            or latest_full(
                large_model=args.large_model or args.regularized_large,
                regularized_large=args.regularized_large,
                full_fuxi_context=args.full_fuxi_context,
            )
        ).resolve()
    )


if __name__ == "__main__":
    main()
