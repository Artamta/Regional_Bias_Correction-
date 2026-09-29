#!/usr/bin/env python3
"""Package the matched FuXi--IMD log-bias-anchor ablation.

The two input runs are immutable.  This postprocessor validates that they use
the same cases, observations, deterministic baselines, and normalization,
then compares the matched ``normal_climo_model`` trained around either the
fitted log-bias forecast or raw FuXi.  It writes a fresh, atomic comparison
directory with tidy metrics, descriptive paired intervals, five figures, a
short report, and complete provenance hashes.

Validation (2018--2019) was used repeatedly for model selection.  Test years
2020--2021 are also reused exploratory hindcasts.  Neither split is an
independent confirmation set, and the percentile intervals produced here are
descriptive rather than significance tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORED_RUN = (
    ROOT
    / "results"
    / "fuxi_imd_full_context_compact_allweeks"
    / "full_20260811T152024Z"
)
DEFAULT_RAW_IDENTITY_RUN = (
    ROOT / "resultsv2" / "fuxi_imd_no_log_bias_ablation" / "full_20260822T010749Z"
)

EXPECTED_TRAIN_YEARS = tuple(range(2002, 2018))
EXPECTED_VALIDATION_YEARS = (2018, 2019)
EXPECTED_TEST_YEARS = (2020, 2021)
EXPECTED_LEADS = tuple(range(1, 7))
EXPECTED_CASES = 70
MATCHED_MODEL = "normal_climo_model"

BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_818

RAW = "raw_fuxi"
LOG_BIAS = "log_bias"
ANCHORED = "log_bias_anchored_adapter"
RAW_IDENTITY = "raw_identity_adapter"
METHOD_ORDER = (RAW, LOG_BIAS, ANCHORED, RAW_IDENTITY)
METHOD_LABELS = {
    RAW: "Raw FuXi",
    LOG_BIAS: "Training-only log-bias",
    ANCHORED: "Neural + log-bias anchor",
    RAW_IDENTITY: "Raw-identity neural (no log-bias anchor)",
}
METHOD_COLORS = {
    RAW: "#4D4D4D",
    LOG_BIAS: "#CC79A7",
    ANCHORED: "#0072B2",
    RAW_IDENTITY: "#D55E00",
}
METHOD_MARKERS = {RAW: "o", LOG_BIAS: "s", ANCHORED: "^", RAW_IDENTITY: "P"}
TRAINING_REFERENCES = {
    RAW: "none; uncorrected FuXi",
    LOG_BIAS: "deterministic training-only log-bias correction",
    ANCHORED: "fitted training-only log-bias reconstruction reference",
    RAW_IDENTITY: "raw FuXi reconstruction reference",
}

ABSOLUTE_METRICS = ("rmse", "mae", "bias", "acc", "pcc")
PAIRED_METRICS = ("rmse", "mae", "acc", "pcc")
METRIC_UNITS = {
    "rmse": "mm day-1",
    "mae": "mm day-1",
    "bias": "mm day-1",
    "acc": "correlation",
    "pcc": "correlation",
}
EVIDENCE_STATUS = {
    "validation": "reused 2018-2019 model-selection validation; not independent",
    "test": "reused exploratory 2020-2021 hindcasts; not independent confirmation",
}


@dataclass(frozen=True)
class SpatialComparison:
    latitude: np.ndarray
    longitude: np.ndarray
    area_weight_km2: np.ndarray
    truth: np.ndarray
    anchored_prediction: np.ndarray
    raw_identity_prediction: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in Path(path).rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _hash_source_artifact(
    run: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> str:
    path = run / relative
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or relative not in artifacts:
        raise ValueError(f"source manifest does not hash {relative}: {run}")
    if actual != artifacts[relative]:
        raise ValueError(f"source artifact checksum mismatch: {path}")
    return actual


def validate_run(
    run: Path,
    *,
    role: str,
    expected_cases: int = EXPECTED_CASES,
    expected_leads: Sequence[int] = EXPECTED_LEADS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one frozen source archive and return manifest provenance."""
    run = Path(run).resolve()
    manifest_path = run / "manifest.json"
    selection_path = run / "selection.json"
    manifest = read_json(manifest_path)
    selection = read_json(selection_path)

    if role not in {"anchored", "raw_identity"}:
        raise ValueError(f"unknown run role: {role}")
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError(f"source must be a completed non-smoke run: {run}")
    if manifest.get("test_count_used") != expected_cases:
        raise ValueError(f"unexpected test case count in {run}")
    if manifest.get("selected_model") != MATCHED_MODEL:
        raise ValueError(f"matched normal model was not selected in {run}")
    if selection.get("status") != "frozen" or selection.get("smoke") is not False:
        raise ValueError(f"selection is not frozen in {run}")
    if selection.get("selected_model") != MATCHED_MODEL:
        raise ValueError(f"selection does not retain the matched normal model: {run}")

    split_years = manifest.get("split_years", {})
    expected_splits = {
        "train": list(EXPECTED_TRAIN_YEARS),
        "validation": list(EXPECTED_VALIDATION_YEARS),
        "test": list(EXPECTED_TEST_YEARS),
    }
    if split_years != expected_splits:
        raise ValueError(f"unexpected split years in {run}: {split_years}")
    if tuple(manifest.get("active_leads", ())) != tuple(expected_leads):
        raise ValueError(f"source does not cover the expected lead weeks: {run}")
    if 2025 not in manifest.get("quarantined_final_initialization_years", []):
        raise ValueError(f"source does not record the 2025 quarantine: {run}")

    training = manifest.get("training", {}).get(MATCHED_MODEL, {})
    expected_training = {
        "parameter_count": 144_689,
        "seeds": [42, 43, 44],
        "batch_size": 32,
        "learning_rate": 2.0e-4,
        "weight_decay": 2.0e-3,
        "dropout": 0.30,
        "max_epochs": 100,
        "patience": 15,
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise ValueError(
                f"matched training contract differs for {field}: "
                f"{training.get(field)!r} != {expected!r}"
            )

    if role == "raw_identity":
        if manifest.get("training_anchor") != RAW:
            raise ValueError("raw-identity source did not train around raw FuXi")
        if manifest.get("uses_fitted_log_bias_in_neural_training") is not False:
            raise ValueError(
                "raw-identity source used fitted log bias in neural training"
            )
        if manifest.get("log_bias_role") != "reporting_only":
            raise ValueError("log-bias comparator is not reporting-only")
        contract_path = run / "models" / "training_anchor_contract.npz"
        with np.load(contract_path) as contract:
            if str(contract["anchor_kind"].item()) != RAW:
                raise ValueError("raw-identity target contract has the wrong anchor")
            if contract["target_scale"].shape != (len(expected_leads),):
                raise ValueError("raw-identity target scale has the wrong shape")
            if contract["fitted_target_years"].tolist() != list(EXPECTED_TRAIN_YEARS):
                raise ValueError("raw-identity target scale was not fit on 2002-2017")
    else:
        anchor_path = run / "models" / "log_bias_anchor.npz"
        with np.load(anchor_path) as anchor:
            if anchor["target_scale"].shape != (len(expected_leads),):
                raise ValueError("canonical log-bias target scale has the wrong shape")

    hashed = {}
    for relative in (
        "selection.json",
        "normalization.json",
        "metrics/case_metrics.csv",
        "metrics/validation_case_metrics.csv",
        "predictions.zarr",
    ):
        hashed[relative] = _hash_source_artifact(run, manifest, relative)
    provenance = {
        "path": str(run),
        "role": role,
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_model": manifest["selected_model"],
        "selected_alpha": float(manifest["selected_alpha"]),
        "artifacts": hashed,
    }
    return manifest, selection, provenance


def read_case_metrics(
    run: Path,
    *,
    split: str,
    expected_cases: int = EXPECTED_CASES,
    expected_leads: Sequence[int] = EXPECTED_LEADS,
) -> pd.DataFrame:
    filename = (
        "validation_case_metrics.csv" if split == "validation" else "case_metrics.csv"
    )
    frame = pd.read_csv(Path(run) / "metrics" / filename)
    required = {
        "split",
        "method",
        "year",
        "init",
        "lead_week",
        "region",
        "rmse",
        "mae",
        "bias",
        "acc",
        "spatial_acc_common_imd",
        "pcc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} lacks columns: {sorted(missing)}")
    if set(frame["split"]) != {split}:
        raise ValueError(f"unexpected split labels in {filename}")
    if set(frame["region"]) != {"india"}:
        raise ValueError(f"unexpected verification regions in {filename}")
    if not np.array_equal(
        frame["acc"].to_numpy(dtype=np.float64),
        frame["spatial_acc_common_imd"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise ValueError(f"ACC is not the explicit common-IMD score in {filename}")
    frame = frame.copy()
    frame["acc"] = frame["spatial_acc_common_imd"]

    expected_years = (
        EXPECTED_VALIDATION_YEARS if split == "validation" else EXPECTED_TEST_YEARS
    )
    expected_lead_set = set(int(value) for value in expected_leads)
    for method in (RAW, LOG_BIAS, MATCHED_MODEL):
        selected = frame.loc[frame.method.eq(method)]
        if selected["init"].nunique() != expected_cases:
            raise ValueError(f"{split} {method} has the wrong case count")
        if set(selected["lead_week"].astype(int)) != expected_lead_set:
            raise ValueError(f"{split} {method} has the wrong lead coverage")
        if set(selected["year"].astype(int)) != set(expected_years):
            raise ValueError(f"{split} {method} has the wrong years")
        if selected.duplicated(["init", "lead_week"]).any():
            raise ValueError(f"duplicate {split} case/lead rows for {method}")
        if len(selected) != expected_cases * len(expected_leads):
            raise ValueError(f"incomplete {split} metric table for {method}")
        if not np.isfinite(
            selected[list(ABSOLUTE_METRICS)].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(f"non-finite {split} metrics for {method}")
    return frame


def validate_baseline_identity(
    anchored: pd.DataFrame,
    raw_identity: pd.DataFrame,
    *,
    split: str,
) -> None:
    """Require exact raw-FuXi and log-bias rows across both archives."""
    keys = ["method", "init", "lead_week"]
    left = (
        anchored.loc[anchored.method.isin((RAW, LOG_BIAS))]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    right = (
        raw_identity.loc[raw_identity.method.isin((RAW, LOG_BIAS))]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise ValueError(f"{split} baseline schemas differ")
    for column in left.columns:
        lhs = left[column].to_numpy()
        rhs = right[column].to_numpy()
        if pd.api.types.is_numeric_dtype(left[column]):
            equal = np.array_equal(
                lhs.astype(np.float64), rhs.astype(np.float64), equal_nan=True
            )
        else:
            equal = np.array_equal(lhs.astype(str), rhs.astype(str))
        if not equal:
            raise ValueError(f"{split} raw/log baseline identity failed: {column}")


def combine_case_metrics(
    anchored: pd.DataFrame,
    raw_identity: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    validate_baseline_identity(anchored, raw_identity, split=split)
    pieces = []
    for source, source_method, target_method in (
        (anchored, RAW, RAW),
        (anchored, LOG_BIAS, LOG_BIAS),
        (anchored, MATCHED_MODEL, ANCHORED),
        (raw_identity, MATCHED_MODEL, RAW_IDENTITY),
    ):
        selected = source.loc[source.method.eq(source_method)].copy()
        selected["method"] = target_method
        pieces.append(selected)
    combined = pd.concat(pieces, ignore_index=True)
    combined["method_label"] = combined.method.map(METHOD_LABELS)
    combined["training_reference"] = combined.method.map(TRAINING_REFERENCES)
    return combined


def _array_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    return np.array_equal(
        np.asarray(left.load()), np.asarray(right.load()), equal_nan=True
    )


def validate_prediction_identity(
    anchored_run: Path,
    raw_identity_run: Path,
    *,
    expected_cases: int = EXPECTED_CASES,
    expected_leads: Sequence[int] = EXPECTED_LEADS,
) -> SpatialComparison:
    """Validate common grids and return only arrays needed for the spatial effect."""
    anchored_store = Path(anchored_run) / "predictions.zarr"
    raw_store = Path(raw_identity_run) / "predictions.zarr"
    with xr.open_zarr(anchored_store, consolidated=True) as left, xr.open_zarr(
        raw_store, consolidated=True
    ) as right:
        if (
            left.sizes.get("init") != expected_cases
            or right.sizes.get("init") != expected_cases
        ):
            raise ValueError("prediction stores have the wrong test case count")
        if tuple(left.lead_week.values.tolist()) != tuple(expected_leads):
            raise ValueError("canonical prediction store has wrong lead coverage")
        if tuple(right.lead_week.values.tolist()) != tuple(expected_leads):
            raise ValueError("raw-identity prediction store has wrong lead coverage")
        for coordinate in ("init", "lead_week", "latitude", "longitude"):
            if not np.array_equal(left[coordinate].values, right[coordinate].values):
                raise ValueError(f"prediction coordinate identity failed: {coordinate}")
        for variable in ("truth_imd", "fixed_imd_climatology", "area_weight_km2"):
            if not _array_equal(left[variable], right[variable]):
                raise ValueError(
                    f"shared prediction variable identity failed: {variable}"
                )
        for method in (RAW, LOG_BIAS):
            lhs = left.prediction.sel({"method": method})
            rhs = right.prediction.sel({"method": method})
            if not _array_equal(lhs, rhs):
                raise ValueError(f"test prediction baseline identity failed: {method}")

        return SpatialComparison(
            latitude=np.asarray(left.latitude.load(), dtype=np.float64),
            longitude=np.asarray(left.longitude.load(), dtype=np.float64),
            area_weight_km2=np.asarray(left.area_weight_km2.load(), dtype=np.float64),
            truth=np.asarray(left.truth_imd.load(), dtype=np.float32),
            anchored_prediction=np.asarray(
                left.prediction.sel({"method": MATCHED_MODEL}).load(), dtype=np.float32
            ),
            raw_identity_prediction=np.asarray(
                right.prediction.sel({"method": MATCHED_MODEL}).load(), dtype=np.float32
            ),
        )


def build_absolute_metrics(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Return one tidy table for all plotted absolute metrics."""
    rows: list[dict[str, Any]] = []

    def append_summary(
        frame: pd.DataFrame,
        *,
        split: str,
        aggregation: str,
        scope: str,
        year: int | None = None,
        lead_week: int | None = None,
    ) -> None:
        selected = frame
        if year is not None:
            selected = selected.loc[selected.year.eq(year)]
        if lead_week is not None:
            selected = selected.loc[selected.lead_week.eq(lead_week)]
        for method in METHOD_ORDER:
            values = selected.loc[selected.method.eq(method)]
            if values.empty:
                raise ValueError(f"no rows for {split} {scope} {method}")
            for metric in ABSOLUTE_METRICS:
                rows.append(
                    {
                        "split": split,
                        "evidence_status": EVIDENCE_STATUS[split],
                        "aggregation": aggregation,
                        "scope": scope,
                        "year": year,
                        "lead_week": lead_week,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "training_reference": TRAINING_REFERENCES[method],
                        "metric": metric,
                        "value": float(values[metric].mean()),
                        "units": METRIC_UNITS[metric],
                        "case_count": int(values["init"].nunique()),
                        "case_lead_rows": int(len(values)),
                    }
                )

    append_summary(validation, split="validation", aggregation="pooled", scope="W1-W6")
    for year in EXPECTED_VALIDATION_YEARS:
        append_summary(
            validation,
            split="validation",
            aggregation="by_year",
            scope="W1-W6",
            year=year,
        )
    append_summary(test, split="test", aggregation="pooled", scope="W1-W6")
    for lead in EXPECTED_LEADS:
        append_summary(
            test,
            split="test",
            aggregation="by_lead",
            scope=f"W{lead}",
            lead_week=lead,
        )
    result = pd.DataFrame(rows)
    result["year"] = result["year"].astype("Int64")
    result["lead_week"] = result["lead_week"].astype("Int64")
    return result


def circular_stratified_indices(
    initializations: Sequence[object],
    *,
    n_resamples: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Sample circular chronological blocks independently inside each year."""
    dates = pd.DatetimeIndex(initializations)
    if dates.has_duplicates:
        raise ValueError("bootstrap initializations must be unique")
    if n_resamples < 1 or block_length < 1:
        raise ValueError("bootstrap draws and block length must be positive")
    years = dates.year.to_numpy()
    groups = [np.flatnonzero(years == year) for year in sorted(np.unique(years))]
    if len(groups) < 2:
        raise ValueError("year-stratified bootstrap requires at least two years")
    if any(len(group) < block_length for group in groups):
        raise ValueError("block length exceeds a year stratum")

    rng = np.random.default_rng(seed)
    sampled = np.empty((n_resamples, len(dates)), dtype=np.int64)
    column = 0
    offsets = np.arange(block_length, dtype=np.int64)
    for group in groups:
        blocks = int(np.ceil(len(group) / block_length))
        starts = rng.integers(0, len(group), size=(n_resamples, blocks))
        local = (starts[..., None] + offsets) % len(group)
        local = local.reshape(n_resamples, -1)[:, : len(group)]
        sampled[:, column : column + len(group)] = group[local]
        column += len(group)
    return sampled


def build_paired_effects(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
    leads: Sequence[int] = EXPECTED_LEADS,
) -> pd.DataFrame:
    """Direct raw-identity versus anchored effects with descriptive intervals."""
    rows: list[dict[str, Any]] = []
    scopes = {"W1-W6": tuple(leads), **{f"W{lead}": (lead,) for lead in leads}}
    for split, frame in (("validation", validation), ("test", test)):
        case_order = sorted(frame["init"].unique())
        sampled = circular_stratified_indices(
            case_order,
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
        )
        for scope, selected_leads in scopes.items():
            subset = frame.loc[
                frame.lead_week.isin(selected_leads)
                & frame.method.isin((ANCHORED, RAW_IDENTITY))
            ]
            for metric in PAIRED_METRICS:
                pivot = subset.pivot_table(
                    index="init", columns="method", values=metric, aggfunc="mean"
                ).reindex(case_order)
                anchor = pivot[ANCHORED].to_numpy(dtype=np.float64)
                raw_identity = pivot[RAW_IDENTITY].to_numpy(dtype=np.float64)
                if not np.isfinite(anchor).all() or not np.isfinite(raw_identity).all():
                    raise ValueError(
                        f"incomplete paired values for {split} {scope} {metric}"
                    )
                anchor_draw = anchor[sampled].mean(axis=1)
                raw_draw = raw_identity[sampled].mean(axis=1)
                if metric in ("rmse", "mae"):
                    effect = (
                        100.0 * (anchor.mean() - raw_identity.mean()) / anchor.mean()
                    )
                    draws = 100.0 * (anchor_draw - raw_draw) / anchor_draw
                    effect_units = "percent reduction relative to anchored model"
                else:
                    effect = raw_identity.mean() - anchor.mean()
                    draws = raw_draw - anchor_draw
                    effect_units = "raw-identity minus anchored correlation"
                lower, upper = np.percentile(draws, (2.5, 97.5))
                rows.append(
                    {
                        "split": split,
                        "evidence_status": EVIDENCE_STATUS[split],
                        "comparison": "raw_identity_adapter_vs_log_bias_anchored_adapter",
                        "scope": scope,
                        "metric": metric,
                        "anchored_mean": float(anchor.mean()),
                        "raw_identity_mean": float(raw_identity.mean()),
                        "raw_identity_candidate_skill": float(effect),
                        "effect_units": effect_units,
                        "positive_direction": "positive means raw-identity adapter is better",
                        "interval_lower": float(lower),
                        "interval_upper": float(upper),
                        "interval_excludes_zero_descriptively": bool(
                            lower > 0.0 or upper < 0.0
                        ),
                        "paired_cases": len(case_order),
                        "bootstrap_draws": n_resamples,
                        "block_length": block_length,
                        "circular_blocks": True,
                        "stratified_by_year": True,
                        "bootstrap_seed": seed,
                        "p_value_computed": False,
                        "significance_claimed": False,
                    }
                )
    return pd.DataFrame(rows)


def build_spatial_effect(
    spatial: SpatialComparison,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute native-grid RMSE(anchor) - RMSE(raw identity)."""
    support = spatial.area_weight_km2 > 0.0
    anchor_rmse = np.full(support.shape, np.nan, dtype=np.float64)
    raw_rmse = np.full(support.shape, np.nan, dtype=np.float64)
    anchor_error = spatial.anchored_prediction[:, :, support].astype(
        np.float64
    ) - spatial.truth[:, :, support].astype(np.float64)
    raw_error = spatial.raw_identity_prediction[:, :, support].astype(
        np.float64
    ) - spatial.truth[:, :, support].astype(np.float64)
    if not np.isfinite(anchor_error).all() or not np.isfinite(raw_error).all():
        raise ValueError(
            "spatial comparison contains missing values inside IMD support"
        )
    anchor_rmse[support] = np.sqrt(np.mean(anchor_error**2, axis=(0, 1)))
    raw_rmse[support] = np.sqrt(np.mean(raw_error**2, axis=(0, 1)))
    effect = anchor_rmse - raw_rmse

    latitude, longitude = np.meshgrid(
        spatial.latitude, spatial.longitude, indexing="ij"
    )
    table = pd.DataFrame(
        {
            "latitude": latitude.ravel(),
            "longitude": longitude.ravel(),
            "imd_support": support.ravel(),
            "area_weight_km2": spatial.area_weight_km2.ravel(),
            "anchored_rmse_mm_day": anchor_rmse.ravel(),
            "raw_identity_rmse_mm_day": raw_rmse.ravel(),
            "raw_identity_rmse_reduction_mm_day": effect.ravel(),
        }
    )
    weights = spatial.area_weight_km2[support]
    summary = {
        "support_cells": int(support.sum()),
        "area_fraction_raw_identity_better_pct": float(
            100.0 * weights[effect[support] > 0.0].sum() / weights.sum()
        ),
        "area_weighted_mean_local_rmse_reduction_mm_day": float(
            np.sum(weights * effect[support]) / weights.sum()
        ),
    }
    return table, summary


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _metric_rows(
    absolute: pd.DataFrame,
    *,
    split: str,
    aggregation: str,
    metric: str,
) -> pd.DataFrame:
    return absolute.loc[
        absolute.split.eq(split)
        & absolute.aggregation.eq(aggregation)
        & absolute.metric.eq(metric)
    ].copy()


def plot_validation_rmse(absolute: pd.DataFrame, output: Path) -> None:
    values = _metric_rows(
        absolute, split="validation", aggregation="by_year", metric="rmse"
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for method in METHOD_ORDER:
        selected = values.loc[values.method.eq(method)].sort_values("year")
        axis.plot(
            selected.year.astype(int),
            selected.value,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(EXPECTED_VALIDATION_YEARS)
    axis.set_xlabel("Validation year")
    axis.set_ylabel("Mean case/lead RMSE (mm day$^{-1}$)")
    axis.set_title("Validation RMSE by year", fontweight="semibold")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    figure.suptitle(
        "Matched FuXi–IMD training-anchor ablation",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "2018–2019 were used repeatedly for model selection; descriptive, not independent evidence.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(bottom=0.16, top=0.88)
    _save_figure(figure, output)


def _plot_test_by_lead(
    absolute: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    values = _metric_rows(absolute, split="test", aggregation="by_lead", metric=metric)
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for method in METHOD_ORDER:
        selected = values.loc[values.method.eq(method)].sort_values("lead_week")
        axis.plot(
            selected.lead_week.astype(int),
            selected.value,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(EXPECTED_LEADS, [f"W{lead}" for lead in EXPECTED_LEADS])
    axis.set_xlabel("Lead week")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="semibold")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    figure.suptitle(
        "Matched FuXi–IMD training-anchor ablation",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "70 reused exploratory 2020–2021 hindcasts; not independent confirmation.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(bottom=0.16, top=0.88)
    _save_figure(figure, output)


def plot_paired_effects(paired: pd.DataFrame, output: Path) -> None:
    overall = paired.loc[paired.scope.eq("W1-W6")]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    split_styles = {
        "validation": ("2018–2019 validation", "#666666", -0.12),
        "test": ("2020–2021 exploratory test", "#D55E00", 0.12),
    }
    for axis, metrics, xlabel in (
        (axes[0], ("rmse", "mae"), "Raw-identity reduction vs anchored (%)"),
        (axes[1], ("acc", "pcc"), "Raw-identity minus anchored correlation"),
    ):
        y = np.arange(len(metrics), dtype=np.float64)
        axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        for split, (label, color, offset) in split_styles.items():
            selected = (
                overall.loc[overall.split.eq(split) & overall.metric.isin(metrics)]
                .set_index("metric")
                .reindex(metrics)
            )
            point = selected.raw_identity_candidate_skill.to_numpy(dtype=float)
            lower = selected.interval_lower.to_numpy(dtype=float)
            upper = selected.interval_upper.to_numpy(dtype=float)
            axis.errorbar(
                point,
                y + offset,
                xerr=np.vstack((point - lower, upper - point)),
                fmt="o",
                color=color,
                capsize=4,
                linewidth=1.8,
                markersize=6,
                label=label,
            )
        axis.set_yticks(y, [metric.upper() for metric in metrics])
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Error reduction", fontweight="semibold")
    axes[1].set_title("Pattern correlation gain", fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.88),
    )
    figure.suptitle(
        "Direct raw-identity candidate skill versus log-bias-anchored neural adapter",
        y=0.99,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Positive favors raw identity. Whiskers: descriptive 95% circular block-13, year-stratified percentile intervals; no p-values.",
        ha="center",
        fontsize=8.3,
        color="0.35",
    )
    figure.subplots_adjust(bottom=0.17, top=0.76, wspace=0.34)
    _save_figure(figure, output)


def plot_spatial_effect(
    table: pd.DataFrame,
    summary: Mapping[str, float],
    output: Path,
) -> None:
    shape = (table.latitude.nunique(), table.longitude.nunique())
    latitude = table.latitude.to_numpy().reshape(shape)[:, 0]
    longitude = table.longitude.to_numpy().reshape(shape)[0]
    values = table.raw_identity_rmse_reduction_mm_day.to_numpy().reshape(shape)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("spatial RMSE effect has no supported cells")
    limit = max(0.05, float(np.percentile(finite, 98.0)))

    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    image = axis.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="nearest",
    )
    colorbar = figure.colorbar(image, ax=axis, extend="both", shrink=0.86)
    colorbar.set_label("RMSE(anchor) − RMSE(raw identity) (mm day$^{-1}$)")
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("Native-grid W1–W6 spatial RMSE effect", fontweight="semibold")
    axis.grid(alpha=0.15)
    axis.text(
        0.02,
        0.03,
        f"Raw identity better over {summary['area_fraction_raw_identity_better_pct']:.1f}% of IMD-supported area",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.suptitle(
        "Raw-identity versus log-bias-anchored neural adapter",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Positive (red) favors raw identity; native 1.5° grid; reused exploratory 2020–2021 hindcasts.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(bottom=0.14, top=0.88)
    _save_figure(figure, output)


def _absolute_value(
    table: pd.DataFrame,
    *,
    split: str,
    aggregation: str,
    method: str,
    metric: str,
    year: int | None = None,
) -> float:
    selected = table.loc[
        table.split.eq(split)
        & table.aggregation.eq(aggregation)
        & table.method.eq(method)
        & table.metric.eq(metric)
    ]
    if year is not None:
        selected = selected.loc[selected.year.eq(year)]
    if len(selected) != 1:
        raise ValueError(f"ambiguous absolute metric lookup: {split} {method} {metric}")
    return float(selected.value.iloc[0])


def _paired_value(table: pd.DataFrame, *, split: str, metric: str) -> pd.Series:
    selected = table.loc[
        table.split.eq(split) & table.scope.eq("W1-W6") & table.metric.eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"ambiguous paired metric lookup: {split} {metric}")
    return selected.iloc[0]


def build_report(
    absolute: pd.DataFrame,
    paired: pd.DataFrame,
    spatial_summary: Mapping[str, float],
    *,
    anchored_run: Path,
    raw_identity_run: Path,
) -> str:
    validation_rows = []
    for method in METHOD_ORDER:
        validation_rows.append(
            "| {label} | {y18:.3f} | {y19:.3f} |".format(
                label=METHOD_LABELS[method],
                y18=_absolute_value(
                    absolute,
                    split="validation",
                    aggregation="by_year",
                    method=method,
                    metric="rmse",
                    year=2018,
                ),
                y19=_absolute_value(
                    absolute,
                    split="validation",
                    aggregation="by_year",
                    method=method,
                    metric="rmse",
                    year=2019,
                ),
            )
        )
    test_rows = []
    for method in METHOD_ORDER:
        values = {
            metric: _absolute_value(
                absolute,
                split="test",
                aggregation="pooled",
                method=method,
                metric=metric,
            )
            for metric in ABSOLUTE_METRICS
        }
        test_rows.append(
            "| {label} | {acc:.3f} | {pcc:.3f} | {rmse:.3f} | {mae:.3f} | {bias:+.3f} |".format(
                label=METHOD_LABELS[method], **values
            )
        )
    validation_rmse = _paired_value(paired, split="validation", metric="rmse")
    test_rmse = _paired_value(paired, split="test", metric="rmse")
    test_acc = _paired_value(paired, split="test", metric="acc")
    test_mae = _paired_value(paired, split="test", metric="mae")

    return f"""# FuXi–IMD no-log-bias anchor ablation

## Result in one sentence

With architecture, features, objective, seeds, splits, and validation gate held fixed, the raw-identity neural adapter is slightly better than the log-bias-anchored neural adapter on reused 2020–2021 hindcasts: W1–W6 RMSE candidate skill is **{test_rmse.raw_identity_candidate_skill:+.3f}%** with a descriptive interval **[{test_rmse.interval_lower:+.3f}%, {test_rmse.interval_upper:+.3f}%]**.

## Scientific scope

- Train and target-derived preprocessing: 2002–2017 only.
- Validation: 2018–2019, repeatedly reused for selection; not independent evidence.
- Evaluation: 70 reused exploratory 2020–2021 hindcasts; not independent confirmation.
- 2025 remains quarantined and unopened by this postprocessor.
- The one changed factor is the neural reconstruction reference: fitted log-bias versus raw FuXi.
- The raw-identity model still uses training-only IMD climatology features and a raw-FuXi residual skip. It is a **no-fitted-log-bias adapter**, not a target-free or forecast-only AI model.

The raw FuXi and deterministic log-bias rows were required to match exactly across both source archives in validation metrics, test metrics, and stored test grids before this package was written.

## Reused validation RMSE by year

| Method | 2018 RMSE | 2019 RMSE |
|---|---:|---:|
{os.linesep.join(validation_rows)}

The pooled raw-identity-versus-anchored validation RMSE candidate skill is **{validation_rmse.raw_identity_candidate_skill:+.3f}%** with descriptive interval **[{validation_rmse.interval_lower:+.3f}%, {validation_rmse.interval_upper:+.3f}%]**. This split chose the models and gate, so the interval is diagnostic only.

## Reused exploratory 2020–2021 test

| Method | Common-IMD ACC | PCC | RMSE | MAE | Signed bias |
|---|---:|---:|---:|---:|---:|
{os.linesep.join(test_rows)}

Direct raw-identity candidate skill versus the anchored neural adapter:

- RMSE: **{test_rmse.raw_identity_candidate_skill:+.3f}%** [{test_rmse.interval_lower:+.3f}%, {test_rmse.interval_upper:+.3f}%].
- MAE: **{test_mae.raw_identity_candidate_skill:+.3f}%** [{test_mae.interval_lower:+.3f}%, {test_mae.interval_upper:+.3f}%].
- Common-IMD ACC difference: **{test_acc.raw_identity_candidate_skill:+.4f}** [{test_acc.interval_lower:+.4f}, {test_acc.interval_upper:+.4f}].
- Raw identity has lower local RMSE over **{spatial_summary['area_fraction_raw_identity_better_pct']:.1f}%** of IMD-supported area; the area-weighted mean local reduction is **{spatial_summary['area_weighted_mean_local_rmse_reduction_mm_day']:+.3f} mm day⁻¹**.

Intervals use 2,000 paired, year-stratified circular moving-block resamples with block length 13 and RNG seed {BOOTSTRAP_SEED}. They describe sensitivity to the limited reused samples; no p-values or independent-significance claims are made. Positive candidate skill always means the raw-identity adapter is better.

## Figure guide

1. `01_validation_rmse_by_year`: reused selection-period RMSE by year.
2. `02_test_rmse_by_lead`: exploratory test RMSE across W1–W6.
3. `03_test_common_imd_acc_by_lead`: exploratory common-training-climatology ACC.
4. `04_paired_anchor_effects`: direct pooled candidate skill and descriptive intervals.
5. `05_spatial_rmse_effect`: native-grid RMSE(anchor) − RMSE(raw identity).

## Provenance

- Canonical log-bias-anchored run: `{Path(anchored_run).resolve()}`
- Raw-identity run: `{Path(raw_identity_run).resolve()}`

Exact source and output hashes are recorded in `manifest.json`.
"""


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package(
    anchored_run: Path,
    raw_identity_run: Path,
    output: Path,
    *,
    expected_cases: int = EXPECTED_CASES,
    expected_leads: Sequence[int] = EXPECTED_LEADS,
    n_resamples: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> Path:
    anchored_run = Path(anchored_run).expanduser().resolve()
    raw_identity_run = Path(raw_identity_run).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    if output == anchored_run or anchored_run in output.parents:
        raise ValueError("comparison output may not be inside the anchored source run")
    if output == raw_identity_run or raw_identity_run in output.parents:
        raise ValueError(
            "comparison output may not be inside the raw-identity source run"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    anchored_manifest, _, anchored_source = validate_run(
        anchored_run,
        role="anchored",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    raw_manifest, _, raw_source = validate_run(
        raw_identity_run,
        role="raw_identity",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    if (
        anchored_source["artifacts"]["normalization.json"]
        != raw_source["artifacts"]["normalization.json"]
    ):
        raise ValueError("feature normalization differs between source runs")

    anchored_validation = read_case_metrics(
        anchored_run,
        split="validation",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    raw_validation = read_case_metrics(
        raw_identity_run,
        split="validation",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    anchored_test = read_case_metrics(
        anchored_run,
        split="test",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    raw_test = read_case_metrics(
        raw_identity_run,
        split="test",
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )
    validation = combine_case_metrics(
        anchored_validation, raw_validation, split="validation"
    )
    test = combine_case_metrics(anchored_test, raw_test, split="test")
    spatial = validate_prediction_identity(
        anchored_run,
        raw_identity_run,
        expected_cases=expected_cases,
        expected_leads=expected_leads,
    )

    absolute = build_absolute_metrics(validation, test)
    paired = build_paired_effects(
        validation,
        test,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        leads=expected_leads,
    )
    spatial_table, spatial_summary = build_spatial_effect(spatial)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        metrics = temporary / "metrics"
        figures = temporary / "figures"
        metrics.mkdir()
        figures.mkdir()
        absolute.to_csv(metrics / "absolute_metrics_long.csv", index=False)
        paired.to_csv(metrics / "paired_anchor_effects.csv", index=False)
        spatial_table.to_csv(metrics / "spatial_rmse_effect.csv", index=False)

        plot_validation_rmse(absolute, figures / "01_validation_rmse_by_year")
        _plot_test_by_lead(
            absolute,
            metric="rmse",
            ylabel="Mean case RMSE (mm day$^{-1}$)",
            title="Exploratory test RMSE by lead",
            output=figures / "02_test_rmse_by_lead",
        )
        _plot_test_by_lead(
            absolute,
            metric="acc",
            ylabel="Common-IMD spatial ACC",
            title="Exploratory test common-IMD ACC by lead",
            output=figures / "03_test_common_imd_acc_by_lead",
        )
        plot_paired_effects(paired, figures / "04_paired_anchor_effects")
        plot_spatial_effect(
            spatial_table, spatial_summary, figures / "05_spatial_rmse_effect"
        )

        report = build_report(
            absolute,
            paired,
            spatial_summary,
            anchored_run=anchored_run,
            raw_identity_run=raw_identity_run,
        )
        _atomic_write_text(temporary / "REPORT.md", report)

        artifacts = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_name": "fuxi_imd_no_log_bias_anchor_ablation_comparison",
            "schema_version": 1,
            "status": "complete",
            "created_utc": utc_now(),
            "experiment_role": "matched one-factor neural reconstruction-reference ablation",
            "methods": {
                method: {
                    "label": METHOD_LABELS[method],
                    "training_reference": TRAINING_REFERENCES[method],
                }
                for method in METHOD_ORDER
            },
            "primary_comparison": {
                "candidate": RAW_IDENTITY,
                "reference": ANCHORED,
                "positive_direction": "positive means raw-identity adapter is better",
                "matched_source_method": MATCHED_MODEL,
            },
            "split_years": anchored_manifest["split_years"],
            "cases": expected_cases,
            "lead_weeks": list(expected_leads),
            "evidence_scope": {
                "validation": EVIDENCE_STATUS["validation"],
                "test": EVIDENCE_STATUS["test"],
                "independent_confirmation_claimed": False,
                "statistical_significance_claimed": False,
                "p_values_computed": False,
                "final_2025_accessed": False,
            },
            "uncertainty": {
                "name": "paired year-stratified circular moving-block descriptive percentile interval",
                "percentiles": [2.5, 97.5],
                "bootstrap_draws": n_resamples,
                "block_length": block_length,
                "seed": seed,
                "year_strata": {
                    "validation": list(EXPECTED_VALIDATION_YEARS),
                    "test": list(EXPECTED_TEST_YEARS),
                },
                "all_selected_leads_kept_within_each_case_value": True,
                "p_values_computed": False,
                "significance_claimed": False,
            },
            "identity_checks": {
                "normalization_sha256_equal": True,
                "validation_raw_and_log_bias_case_metrics_exact": True,
                "test_raw_and_log_bias_case_metrics_exact": True,
                "test_coordinates_truth_climatology_weights_exact": True,
                "test_raw_and_log_bias_prediction_grids_exact": True,
            },
            "spatial_summary": spatial_summary,
            "sources": {
                "log_bias_anchored": anchored_source,
                "raw_identity": raw_source,
                "postprocessor": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "source_selected_alpha": {
                "log_bias_anchored": float(anchored_manifest["selected_alpha"]),
                "raw_identity": float(raw_manifest["selected_alpha"]),
            },
            "artifacts": dict(sorted(artifacts.items())),
        }
        _atomic_write_text(
            temporary / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchored-run",
        type=Path,
        default=DEFAULT_ANCHORED_RUN,
        help="completed canonical log-bias-anchored run",
    )
    parser.add_argument(
        "--raw-identity-run",
        type=Path,
        default=DEFAULT_RAW_IDENTITY_RUN,
        help="completed no-fitted-log-bias run",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = package(args.anchored_run, args.raw_identity_run, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
