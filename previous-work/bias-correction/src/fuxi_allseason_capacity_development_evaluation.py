#!/usr/bin/env python3
"""Post-selection 2020--2021 evaluation for the capacity ablation.

This program cannot train or select a model.  It accepts only a completed,
hash-consistent *full* capacity-ablation manifest, restores the already locked
``base_42k`` and selected three-seed checkpoints, and evaluates the reused
2020--2021 development split.  If validation retained ``base_42k``, the base
is evaluated once and the absence of a distinct selected-model comparison is
reported explicitly.  The sealed 2025 target is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import fuxi_allseason_capacity_ablation as capacity
import fuxi_allseason_ensemble_calibration as base
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_capacity_development_evaluation_v1"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_capacity_development_evaluation"
)
SLURM_PATH = PROJECT_ROOT / "slurm/evaluate_allseason_capacity_development.sbatch"
SEEDS = (42, 43, 44)
CORE_METRICS = ("crps", "rmse", "acc", "bias")
BOOTSTRAP_COLUMNS = (
    "comparison_scope",
    "optimization_seed",
    "method",
    "baseline",
    "lead_scope",
    "metric",
    "effect_name",
    "effect",
    "ci_lower",
    "ci_upper",
    "paired_initializations",
    "n_resamples",
    "block_length_initializations",
    "seed",
    "resampling_unit",
    "bootstrap",
)


class DevelopmentEvaluationError(RuntimeError):
    """Raised when a frozen receipt or evaluation contract is violated."""


@dataclass(frozen=True)
class CapacityReceipt:
    manifest_path: Path
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    selected_candidate: str
    selected_distinct_from_base: bool
    checkpoint_records: Mapping[tuple[str, int], Mapping[str, Any]]
    cache_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_receipt_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DevelopmentEvaluationError(f"unsafe {label} path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise DevelopmentEvaluationError(f"{label} escapes receipt root: {relative!r}")
    return resolved


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentEvaluationError(message)


def validate_capacity_manifest(path: Path) -> CapacityReceipt:
    """Validate the complete capacity receipt before any development access."""

    manifest_path = Path(path).resolve()
    _require(
        manifest_path.name == "manifest.json", "capacity receipt must be manifest.json"
    )
    _require(manifest_path.is_file(), f"capacity manifest is missing: {manifest_path}")
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentEvaluationError(
            f"invalid capacity manifest: {error}"
        ) from error
    _require(
        manifest.get("experiment") == capacity.EXPERIMENT,
        "capacity experiment identity mismatch",
    )
    _require(manifest.get("status") == "complete", "capacity run is not complete")
    _require(
        manifest.get("mode") == "full" and manifest.get("smoke") is False,
        "development evaluation requires a completed full capacity run",
    )
    _require(
        manifest.get("candidates") == [asdict(item) for item in capacity.CANDIDATES],
        "capacity candidate contract mismatch",
    )
    _require(
        manifest.get("controls") == [asdict(capacity.SUMMARY_CONTROL)],
        "capacity summary-control contract mismatch",
    )
    _require(manifest.get("seeds") == list(SEEDS), "capacity seed contract mismatch")
    _require(
        manifest.get("split_counts_selected")
        == {
            "train": base.EXPECTED_COUNTS["train"],
            "validation": base.EXPECTED_COUNTS["validation"],
        },
        "capacity train/validation counts mismatch",
    )
    contract = manifest.get("contract", {})
    _require(
        contract.get("test_metrics_consulted") is False,
        "capacity selection used test metrics",
    )
    _require(
        contract.get("sealed_2025_target_opened") is False,
        "capacity receipt opened 2025",
    )
    _require(
        contract.get("train_years") == list(base.TRAIN_YEARS)
        and contract.get("validation_years") == list(base.VALIDATION_YEARS),
        "capacity temporal contract mismatch",
    )
    training = manifest.get("training", {})
    _require(
        training.get("member_subsample") == 16, "capacity member subsample changed"
    )
    _require(
        training.get("full_members_for_validation") == 51,
        "capacity validation did not use 51 members",
    )
    _require(
        training.get("objective") == "area-weighted empirical finite-ensemble CRPS",
        "capacity objective mismatch",
    )

    artifact_hashes = manifest.get("artifact_sha256")
    _require(
        isinstance(artifact_hashes, dict) and bool(artifact_hashes),
        "capacity artifact hashes are missing",
    )
    for relative, expected_hash in sorted(artifact_hashes.items()):
        _require(
            isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"invalid artifact hash for {relative}",
        )
        artifact = _safe_receipt_path(root, str(relative), label="artifact")
        _require(artifact.is_file(), f"capacity artifact is missing: {relative}")
        _require(
            sha256_file(artifact) == expected_hash,
            f"capacity artifact hash mismatch: {relative}",
        )

    selection_path = _safe_receipt_path(root, "selection.json", label="selection")
    _require("selection.json" in artifact_hashes, "selection.json is not hash-gated")
    try:
        selection_file = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentEvaluationError(
            f"invalid locked selection: {error}"
        ) from error
    selection = manifest.get("selection")
    _require(selection_file == selection, "manifest and selection.json disagree")
    _require(
        selection.get("status") == "validation_selection_locked",
        "capacity selection is not locked",
    )
    _require(
        selection.get("scientific_selection") is True,
        "capacity full selection is not marked scientific",
    )
    _require(
        selection.get("test_metrics_consulted") is False,
        "locked selection consulted test metrics",
    )
    selected = str(selection.get("selected_candidate", ""))
    _require(
        selected in capacity.CANDIDATE_NAMES, "locked selected candidate is invalid"
    )
    validation_metrics_relative = "metrics/validation_case_metrics.csv"
    _require(
        validation_metrics_relative in artifact_hashes,
        "capacity validation metrics are not hash-gated",
    )
    validation_metrics_path = _safe_receipt_path(
        root, validation_metrics_relative, label="validation metrics"
    )
    try:
        validation_metrics = pd.read_csv(validation_metrics_path)
        recomputed_selection = capacity.select_capacity(
            validation_metrics, expected_seeds=SEEDS
        )
    except (OSError, ValueError, KeyError, base.DataContractError) as error:
        raise DevelopmentEvaluationError(
            f"cannot reproduce capacity selection: {error}"
        ) from error
    _require(
        recomputed_selection["selected_candidate"] == selected,
        "locked capacity winner is not reproducible from validation metrics",
    )
    _require(
        recomputed_selection["selected_parameter_count"]
        == selection.get("selected_parameter_count"),
        "recomputed selected parameter count disagrees",
    )
    _require(
        recomputed_selection["rules"] == selection.get("rules"),
        "capacity selection rules changed",
    )
    selected_record = next(
        (
            row
            for row in selection.get("validation_candidates", [])
            if row.get("candidate") == selected
        ),
        None,
    )
    _require(selected_record is not None, "selected capacity record is missing")
    _require(
        int(selection.get("selected_parameter_count", -1))
        == capacity.CANDIDATE_BY_NAME[selected].expected_parameter_count,
        "selected parameter count mismatch",
    )
    if selected != capacity.BASE_CANDIDATE:
        _require(
            selected_record.get("eligible_promotion") is True,
            "selected challenger is not promotion-eligible",
        )
        _require(
            selected_record.get("all_years_noninferior_guard") is True,
            "selected challenger failed year guard",
        )
        _require(
            selected_record.get("matched_seed_guard") is True,
            "selected challenger failed seed guard",
        )

    expected_arms = (*capacity.CANDIDATE_NAMES, capacity.SUMMARY_CONTROL.name)
    expected_parameters = {
        arm.name: arm.expected_parameter_count for arm in capacity.EXPERIMENT_ARMS
    }
    runs = training.get("runs", [])
    _require(
        len(runs) == len(expected_arms) * len(SEEDS), "capacity run grid is incomplete"
    )
    checkpoint_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in runs:
        candidate_name = str(record.get("candidate", ""))
        seed = int(record.get("seed", -1))
        key = (candidate_name, seed)
        _require(
            candidate_name in expected_arms and seed in SEEDS,
            f"invalid capacity run identity: {key}",
        )
        _require(key not in checkpoint_records, f"duplicate capacity run: {key}")
        _require(
            int(record.get("parameter_count", -1))
            == expected_parameters[candidate_name],
            f"capacity parameter count mismatch: {key}",
        )
        relative = str(record.get("checkpoint", ""))
        checkpoint = _safe_receipt_path(root, relative, label="checkpoint")
        _require(
            relative in artifact_hashes, f"checkpoint is not artifact-hashed: {key}"
        )
        _require(checkpoint.is_file(), f"checkpoint is missing: {key}")
        checkpoint_hash = sha256_file(checkpoint)
        _require(
            checkpoint_hash == artifact_hashes[relative],
            f"checkpoint artifact hash mismatch: {key}",
        )
        _require(
            checkpoint_hash == record.get("checkpoint_sha256"),
            f"checkpoint run hash mismatch: {key}",
        )
        checkpoint_records[key] = record
    _require(
        set(checkpoint_records)
        == {(arm, seed) for arm in expected_arms for seed in SEEDS},
        "capacity checkpoint grid is incomplete",
    )

    source_hashes = manifest.get("source_snapshot_sha256", {})
    relevant_sources = {
        "code/src/fuxi_allseason_capacity_ablation.py": Path(
            capacity.__file__
        ).resolve(),
        "code/src/fuxi_allseason_ensemble_calibration.py": Path(
            base.__file__
        ).resolve(),
        "code/src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "code/src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
    }
    for relative, current_source in relevant_sources.items():
        expected_hash = source_hashes.get(relative)
        _require(
            isinstance(expected_hash, str),
            f"capacity source snapshot is missing: {relative}",
        )
        _require(
            artifact_hashes.get(relative) == expected_hash,
            f"capacity source hash is not artifact-gated: {relative}",
        )
        frozen = _safe_receipt_path(root, relative, label="source snapshot")
        _require(
            sha256_file(frozen) == expected_hash,
            f"frozen capacity source mismatch: {relative}",
        )
        _require(
            current_source.is_file(), f"current source is missing: {current_source}"
        )
        _require(
            sha256_file(current_source) == expected_hash,
            f"current source differs from frozen capacity run: {current_source}",
        )

    cache_info = manifest.get("cache", {})
    cache_path = Path(str(cache_info.get("data_file", ""))).resolve()
    _require(cache_path.is_file(), f"capacity cache is unavailable: {cache_path}")
    _require(
        cache_info.get("data_sha256") == capacity.EXPECTED_CACHE_SHA256,
        "capacity cache identity mismatch",
    )
    _require(
        sha256_file(cache_path) == capacity.EXPECTED_CACHE_SHA256,
        "capacity cache bytes changed",
    )
    return CapacityReceipt(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        selected_candidate=selected,
        selected_distinct_from_base=selected != capacity.BASE_CANDIDATE,
        checkpoint_records=checkpoint_records,
        cache_path=cache_path,
    )


def evaluation_candidate_names(selected_candidate: str) -> tuple[str, ...]:
    """Return distinct trained methods, never a fake alias of the base."""

    if selected_candidate not in capacity.CANDIDATE_NAMES:
        raise ValueError(f"unknown selected candidate: {selected_candidate!r}")
    if selected_candidate == capacity.BASE_CANDIDATE:
        return (capacity.BASE_CANDIDATE,)
    return (capacity.BASE_CANDIDATE, selected_candidate)


def _register_method_labels(methods: Sequence[str]) -> None:
    colors = {
        capacity.BASE_CANDIDATE: "#009E73",
        "small_20k": "#56B4E9",
        "medium_158k": "#E69F00",
        "large_294k": "#D55E00",
    }
    for name in methods:
        candidate = capacity.CANDIDATE_BY_NAME[name]
        label = (
            f"Base adapter ({candidate.expected_parameter_count:,} params)"
            if name == capacity.BASE_CANDIDATE
            else f"Selected adapter: {name} ({candidate.expected_parameter_count:,} params)"
        )
        base.METHOD_LABELS[name] = label
        base.PLOT_METHOD_LABELS[name] = label
        base.METHOD_COLORS[name] = colors[name]
        base.METHOD_MARKERS[name] = "P" if name == capacity.BASE_CANDIDATE else "D"


def checkpoint_path(receipt: CapacityReceipt, candidate: str, seed: int) -> Path:
    record = receipt.checkpoint_records[(candidate, seed)]
    return _safe_receipt_path(
        receipt.root, str(record["checkpoint"]), label="checkpoint"
    )


def build_headline_case_metrics(
    raw_metrics: pd.DataFrame,
    seed_case_metrics: pd.DataFrame,
    methods: Sequence[str],
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    """Average scores across seeds without averaging parameters or predictions."""

    frames = [raw_metrics]
    for method in methods:
        selected = seed_case_metrics.loc[seed_case_metrics.method == method]
        frames.append(base.mean_seed_case_metrics(selected, seeds))
    result = pd.concat(frames, ignore_index=True)
    expected = raw_metrics["init"].nunique() * 6 * (1 + len(methods))
    if len(result) != expected:
        raise DevelopmentEvaluationError(
            f"headline metrics have {len(result)} rows; expected {expected}"
        )
    return result


def selected_vs_base_weekwise(
    weekwise: pd.DataFrame,
    selected_candidate: str,
) -> pd.DataFrame:
    """Build an honest comparison table, including the selected-is-base case."""

    base_rows = weekwise.loc[weekwise.method == capacity.BASE_CANDIDATE]
    if len(base_rows) != 6:
        raise DevelopmentEvaluationError("base weekwise table is incomplete")
    if selected_candidate == capacity.BASE_CANDIDATE:
        return pd.DataFrame(
            {
                "lead_week": base_rows.lead_week.to_numpy(dtype=np.int64),
                "base_candidate": capacity.BASE_CANDIDATE,
                "selected_candidate": selected_candidate,
                "distinct_comparison": False,
                "comparison_status": "selected_is_base_no_distinct_improvement",
                "crps_skill_pct_selected_vs_base": np.nan,
                "rmse_skill_pct_selected_vs_base": np.nan,
                "acc_delta_selected_vs_base": np.nan,
                "signed_bias_delta_selected_vs_base": np.nan,
            }
        )
    selected_rows = weekwise.loc[weekwise.method == selected_candidate]
    if len(selected_rows) != 6:
        raise DevelopmentEvaluationError("selected weekwise table is incomplete")
    merged = base_rows[["lead_week", "crps", "rmse", "acc", "bias"]].merge(
        selected_rows[["lead_week", "crps", "rmse", "acc", "bias"]],
        on="lead_week",
        suffixes=("_base", "_selected"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "lead_week": merged.lead_week,
            "base_candidate": capacity.BASE_CANDIDATE,
            "selected_candidate": selected_candidate,
            "distinct_comparison": True,
            "comparison_status": "distinct_locked_models_compared",
            "crps_skill_pct_selected_vs_base": 100.0
            * (merged.crps_base - merged.crps_selected)
            / merged.crps_base,
            "rmse_skill_pct_selected_vs_base": 100.0
            * (merged.rmse_base - merged.rmse_selected)
            / merged.rmse_base,
            "acc_delta_selected_vs_base": merged.acc_selected - merged.acc_base,
            "signed_bias_delta_selected_vs_base": merged.bias_selected
            - merged.bias_base,
        }
    )


def _empty_bootstrap() -> pd.DataFrame:
    return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)


def _annotate_bootstrap(
    frame: pd.DataFrame,
    *,
    comparison_scope: str,
    optimization_seed: str | int,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_bootstrap()
    selected = frame.loc[frame.metric.isin(CORE_METRICS)].copy()
    selected.insert(0, "optimization_seed", optimization_seed)
    selected.insert(0, "comparison_scope", comparison_scope)
    return selected.loc[:, BOOTSTRAP_COLUMNS]


def build_bootstrap_tables(
    headline_case_metrics: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    seed_case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    methods: Sequence[str],
    selected_candidate: str,
    *,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return headline, per-seed/raw, and matched-seed selected/base CIs."""

    raw_comparisons = _annotate_bootstrap(
        base.paired_block_bootstrap(
            headline_case_metrics,
            initializations,
            methods,
            n_resamples=n_resamples,
        ),
        comparison_scope="mean_seed_scores_vs_raw",
        optimization_seed="mean_of_seed_scores",
    )
    if selected_candidate == capacity.BASE_CANDIDATE:
        selected_vs_base = _empty_bootstrap()
    else:
        selected_vs_base = _annotate_bootstrap(
            base.paired_block_bootstrap(
                headline_case_metrics,
                initializations,
                (selected_candidate,),
                n_resamples=n_resamples,
                baseline=capacity.BASE_CANDIDATE,
            ),
            comparison_scope="mean_seed_scores_selected_vs_base",
            optimization_seed="mean_of_seed_scores",
        )

    seed_raw_frames: list[pd.DataFrame] = []
    matched_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for method in methods:
            method_seed = seed_case_metrics.loc[
                seed_case_metrics.method.eq(method) & seed_case_metrics.seed.eq(seed)
            ]
            paired = pd.concat((raw_metrics, method_seed), ignore_index=True)
            seed_raw_frames.append(
                _annotate_bootstrap(
                    base.paired_block_bootstrap(
                        paired,
                        initializations,
                        (method,),
                        n_resamples=n_resamples,
                    ),
                    comparison_scope="individual_seed_vs_raw",
                    optimization_seed=seed,
                )
            )
        if selected_candidate != capacity.BASE_CANDIDATE:
            paired_models = seed_case_metrics.loc[
                seed_case_metrics.seed.eq(seed)
                & seed_case_metrics.method.isin(
                    (capacity.BASE_CANDIDATE, selected_candidate)
                )
            ]
            matched_frames.append(
                _annotate_bootstrap(
                    base.paired_block_bootstrap(
                        paired_models,
                        initializations,
                        (selected_candidate,),
                        n_resamples=n_resamples,
                        baseline=capacity.BASE_CANDIDATE,
                    ),
                    comparison_scope="matched_seed_selected_vs_base",
                    optimization_seed=seed,
                )
            )
    seed_raw = (
        pd.concat(seed_raw_frames, ignore_index=True)
        if seed_raw_frames
        else _empty_bootstrap()
    )
    matched = (
        pd.concat(matched_frames, ignore_index=True)
        if matched_frames
        else _empty_bootstrap()
    )
    headline = (
        raw_comparisons.reset_index(drop=True)
        if selected_vs_base.empty
        else pd.concat((raw_comparisons, selected_vs_base), ignore_index=True)
    )
    return headline, seed_raw, matched


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_capacity_development_evaluation.py": Path(
            __file__
        ).resolve(),
        "src/fuxi_allseason_capacity_ablation.py": Path(capacity.__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": Path(base.__file__).resolve(),
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "slurm/evaluate_allseason_capacity_development.sbatch": SLURM_PATH,
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"frozen evaluation source is missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative not in {"manifest.json", "failure.json"}:
            checksums[relative] = sha256_file(path)
    return checksums


def plot_weekwise_core_metrics(
    weekwise: pd.DataFrame,
    seed_weekwise: pd.DataFrame,
    methods: Sequence[str],
    selected_candidate: str,
    output: Path,
    *,
    smoke: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    specifications = (
        ("crps", "CRPS (mm day⁻¹)", False),
        ("rmse", "RMSE (mm day⁻¹)", False),
        ("acc", "ACC", False),
        ("bias", "Signed bias (mm day⁻¹)", True),
    )
    colors = {"raw_fuxi": "#4D4D4D", **capacity.CANDIDATE_COLORS}
    order = ("raw_fuxi", *methods)
    for axis, (metric, ylabel, zero_line) in zip(
        axes.ravel(), specifications, strict=True
    ):
        for method in order:
            rows = weekwise.loc[weekwise.method == method].sort_values("lead_week")
            axis.plot(
                rows.lead_week,
                rows[metric],
                marker="o",
                linewidth=1.35,
                color=colors[method],
                label=str(rows.method_label.iloc[0]),
            )
            if method != "raw_fuxi":
                variability = seed_weekwise.loc[
                    seed_weekwise.method == method
                ].sort_values("lead_week")
                if not variability.empty:
                    grouped = variability.groupby("lead_week")[metric]
                    lower = grouped.min().reindex(rows.lead_week).to_numpy()
                    upper = grouped.max().reindex(rows.lead_week).to_numpy()
                    axis.fill_between(
                        rows.lead_week,
                        lower,
                        upper,
                        color=colors[method],
                        alpha=0.12,
                        linewidth=0,
                    )
        if zero_line:
            axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    status = (
        "selected=base; no duplicate model comparison"
        if selected_candidate == capacity.BASE_CANDIDATE
        else f"locked selected={selected_candidate}"
    )
    figure.suptitle(
        "2020–2021 post-selection development evaluation · "
        + status
        + (" · smoke" if smoke else ""),
        fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_bootstrap_effects(
    bootstrap: pd.DataFrame,
    selected_candidate: str,
    output: Path,
    *,
    smoke: bool,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    pooled = bootstrap.loc[bootstrap.lead_scope.eq("W1-W6")]
    labels = {
        "crps": "CRPS skill (%)",
        "rmse": "RMSE skill (%)",
        "acc": "ACC difference",
        "bias": "Signed-bias difference",
    }
    for axis, metric in zip(axes.ravel(), CORE_METRICS, strict=True):
        rows = pooled.loc[pooled.metric.eq(metric)].reset_index(drop=True)
        if rows.empty:
            axis.text(0.5, 0.5, "No distinct comparison", ha="center", va="center")
            axis.set_axis_off()
            continue
        names = [f"{row.method}\nvs {row.baseline}" for row in rows.itertuples()]
        positions = np.arange(len(rows))
        values = rows.effect.to_numpy(dtype=np.float64)
        errors = np.vstack(
            (
                values - rows.ci_lower.to_numpy(dtype=np.float64),
                rows.ci_upper.to_numpy(dtype=np.float64) - values,
            )
        )
        axis.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="o",
            color="#0072B2",
            capsize=3,
            linewidth=1.0,
        )
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        axis.set_xticks(positions, names, fontsize=6)
        axis.set_ylabel(labels[metric])
        axis.grid(axis="y", alpha=0.2)
    status = (
        "selected=base; selected-vs-base omitted"
        if selected_candidate == capacity.BASE_CANDIDATE
        else f"selected={selected_candidate}"
    )
    figure.suptitle(
        "Paired initialization-block bootstrap · "
        + status
        + (" · smoke" if smoke else ""),
        fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def build_readme(
    selected_candidate: str,
    methods: Sequence[str],
    *,
    smoke: bool,
    case_count: int,
) -> str:
    selected_status = (
        "Validation retained `base_42k`; it is evaluated once and no selected-vs-base "
        "improvement is fabricated."
        if selected_candidate == capacity.BASE_CANDIDATE
        else f"The locked validation winner is `{selected_candidate}` and is compared with `base_42k`."
    )
    return "\n".join(
        [
            "# Capacity post-selection development evaluation",
            "",
            f"**Status:** {'NON-SCIENTIFIC PLUMBING SMOKE' if smoke else 'REUSED 2020–2021 DEVELOPMENT EVALUATION'}",
            "",
            selected_status,
            "",
            f"This run scores {case_count} development initializations using methods: "
            + ", ".join(("raw_fuxi", *methods))
            + ".",
            "",
            "The capacity choice was already locked from 2018–2019. This program cannot "
            "train or reselect it, and it never accesses the sealed 2025 target.",
            "",
            "Each neural seed is restored and scored separately. Headline curves average "
            "per-seed scores for the same cases; parameters, correction fields, and forecast "
            "members are never averaged across seeds.",
            "",
            "## Main artifacts",
            "",
            "- `metrics/weekwise_core_metrics.csv`: CRPS, RMSE, ACC, and signed bias by week.",
            "- `metrics/seed_weekwise_metrics.csv`: every optimization seed separately.",
            "- `metrics/paired_block_bootstrap.csv`: paired initialization-block intervals.",
            "- `metrics/matched_seed_selected_vs_base_bootstrap.csv`: matched-seed intervals when models are distinct.",
            "- `metrics/selected_vs_base_weekwise.csv`: explicit distinct-or-same-model status.",
            "- `models/*/seed_*/development_adjustments.npz`: per-seed correction fields.",
            "- `manifest.json`: receipt, checkpoints, source snapshots, and artifact hashes.",
            "",
        ]
    )


def run_experiment(
    args: argparse.Namespace,
    output: Path,
    receipt: CapacityReceipt | None = None,
) -> Mapping[str, Any]:
    started_at = time.monotonic()
    if receipt is None:
        receipt = validate_capacity_manifest(Path(args.capacity_manifest))
    snapshot_checksums = source_snapshot(output)
    _register_method_labels(evaluation_candidate_names(receipt.selected_candidate))
    methods = evaluation_candidate_names(receipt.selected_candidate)
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)

    inputs_directory = output / "inputs"
    inputs_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(receipt.manifest_path, inputs_directory / "capacity_manifest.json")
    shutil.copy2(
        receipt.root / "selection.json", inputs_directory / "capacity_selection.json"
    )

    cache = base.load_member_cache(receipt.cache_path, allow_partial=False)
    provenance = base.cache_provenance(cache)
    _require(
        provenance.get("data_sha256") == receipt.manifest["cache"]["data_sha256"],
        "loaded cache differs from capacity receipt",
    )
    _require(
        provenance.get("source_fingerprint")
        == receipt.manifest["cache"].get("source_fingerprint"),
        "loaded cache source fingerprint differs from capacity receipt",
    )
    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(indices) for name, indices in splits.as_dict().items()}
    _require(split_counts == base.EXPECTED_COUNTS, "archive split counts changed")
    train_indices = splits.train
    test_indices = splits.test
    if args.smoke:
        test_indices = base.select_evenly(test_indices, 16)
    _require(len(test_indices) > 0, "development split is empty")
    test_initializations = cache.initializations[test_indices]
    test_years = set(pd.DatetimeIndex(test_initializations).year.to_numpy())
    _require(
        test_years == set(base.TEST_YEARS), "evaluation indices are not 2020–2021 only"
    )

    print("Loading IMD and rebuilding the training-only model context...", flush=True)
    observations = base.load_imd_observations(cache)
    _require(
        all("2025" not in source for source in observations.source_stores),
        "observation loader exposed a 2025 source",
    )
    context = base.build_context_bundle(cache, observations, train_indices)
    test_truth = observations.weekly_truth[test_indices]
    test_climatology = observations.weekly_climatology[test_indices]
    test_members = base.materialize_cases(cache.members, test_indices)

    print("Evaluating raw FuXi once...", flush=True)
    raw_metrics, _ = base.evaluate_ensemble(
        "raw_fuxi",
        test_members,
        test_truth,
        test_climatology,
        test_initializations,
        observations.weights,
        chunk_size=args.evaluation_batch_size,
    )
    seed_frames: list[pd.DataFrame] = []
    input_checkpoints: list[dict[str, Any]] = []
    for method in methods:
        candidate = capacity.CANDIDATE_BY_NAME[method]
        for seed in SEEDS:
            source_checkpoint = checkpoint_path(receipt, method, seed)
            source_record = receipt.checkpoint_records[(method, seed)]
            source_hash = sha256_file(source_checkpoint)
            print(f"Evaluating {method}, seed {seed}...", flush=True)
            model = capacity.load_checkpoint_model(
                source_checkpoint, candidate, seed, device
            )
            delta, log_spread = base.predict_adjustments(
                model,
                cache.members,
                observations.weekly_truth,
                context,
                test_indices,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            if not np.isfinite(delta).all() or not np.isfinite(log_spread).all():
                raise DevelopmentEvaluationError(
                    f"non-finite adjustment field for {method}, seed {seed}"
                )
            spread = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
            adjustment_path = (
                output
                / "models"
                / method
                / f"seed_{seed}"
                / "development_adjustments.npz"
            )
            adjustment_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                adjustment_path,
                initializations=test_initializations,
                delta_log_location=delta,
                log_spread=log_spread,
                spread_factor=spread,
                candidate=method,
                seed=np.int64(seed),
                source_checkpoint_sha256=source_hash,
            )
            corrected = base.apply_affine_log_calibration(test_members, delta, spread)
            seed_metrics, _ = base.evaluate_ensemble(
                method,
                corrected,
                test_truth,
                test_climatology,
                test_initializations,
                observations.weights,
                chunk_size=args.evaluation_batch_size,
                seed_label=seed,
            )
            seed_frames.append(seed_metrics)
            input_checkpoints.append(
                {
                    "candidate": method,
                    "seed": seed,
                    "path": str(source_checkpoint),
                    "sha256": source_hash,
                    "capacity_record_sha256": source_record["checkpoint_sha256"],
                    "parameter_count": source_record["parameter_count"],
                }
            )
            del model, delta, log_spread, spread, corrected, seed_metrics
            if device.type == "cuda":
                torch.cuda.empty_cache()

    seed_case_metrics = pd.concat(seed_frames, ignore_index=True)
    headline_case_metrics = build_headline_case_metrics(
        raw_metrics, seed_case_metrics, methods
    )
    weekwise, pooled, seasonal, _ = base.summarize_metrics(headline_case_metrics)
    seed_weekwise = base.summarize_seed_metrics(
        seed_case_metrics,
        weekwise.loc[weekwise.method == "raw_fuxi"],
    )
    seed_variability = base.summarize_seed_variability(seed_weekwise)
    selected_comparison = selected_vs_base_weekwise(
        weekwise, receipt.selected_candidate
    )
    bootstrap_samples = 100 if args.smoke else args.bootstrap_samples
    headline_bootstrap, seed_bootstrap, matched_bootstrap = build_bootstrap_tables(
        headline_case_metrics,
        raw_metrics,
        seed_case_metrics,
        test_initializations,
        methods,
        receipt.selected_candidate,
        n_resamples=bootstrap_samples,
    )

    core_columns = [
        "split",
        "method",
        "method_label",
        "seed",
        "lead_week",
        "n_initializations",
        "n_valid_cells",
        "crps",
        "crps_skill_pct_vs_raw",
        "rmse",
        "rmse_skill_pct_vs_raw",
        "acc",
        "delta_acc_vs_raw",
        "bias",
        "delta_bias_vs_raw",
    ]
    weekwise_core = weekwise.loc[:, core_columns]
    metrics_directory = output / "metrics"
    figures_directory = output / "figures"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    headline_case_metrics.to_csv(metrics_directory / "case_metrics.csv", index=False)
    seed_case_metrics.to_csv(metrics_directory / "seed_case_metrics.csv", index=False)
    weekwise.to_csv(metrics_directory / "weekwise_metrics.csv", index=False)
    weekwise_core.to_csv(metrics_directory / "weekwise_core_metrics.csv", index=False)
    pooled.to_csv(metrics_directory / "pooled_metrics.csv", index=False)
    seasonal.to_csv(metrics_directory / "seasonal_weekwise_metrics.csv", index=False)
    seed_weekwise.to_csv(metrics_directory / "seed_weekwise_metrics.csv", index=False)
    seed_variability.to_csv(
        metrics_directory / "seed_variability_by_week.csv", index=False
    )
    selected_comparison.to_csv(
        metrics_directory / "selected_vs_base_weekwise.csv", index=False
    )
    headline_bootstrap.to_csv(
        metrics_directory / "paired_block_bootstrap.csv", index=False
    )
    seed_bootstrap.to_csv(
        metrics_directory / "seed_paired_block_bootstrap.csv", index=False
    )
    matched_bootstrap.to_csv(
        metrics_directory / "matched_seed_selected_vs_base_bootstrap.csv",
        index=False,
    )
    base.write_metric_matrices(weekwise, metrics_directory / "matrices")
    selection_status = {
        "capacity_selection_locked_before_development": True,
        "selected_candidate": receipt.selected_candidate,
        "base_candidate": capacity.BASE_CANDIDATE,
        "selected_distinct_from_base": receipt.selected_distinct_from_base,
        "evaluated_neural_methods": list(methods),
        "duplicate_selected_alias_created": False,
        "selected_vs_base_comparison_status": (
            "distinct_locked_models_compared"
            if receipt.selected_distinct_from_base
            else "selected_is_base_no_distinct_improvement"
        ),
        "development_metrics_used_for_selection": False,
        "sealed_2025_target_opened": False,
    }
    write_json(output / "selection_status.json", selection_status)
    plot_weekwise_core_metrics(
        weekwise,
        seed_weekwise,
        methods,
        receipt.selected_candidate,
        figures_directory / "weekwise_core_metrics",
        smoke=args.smoke,
    )
    plot_bootstrap_effects(
        headline_bootstrap,
        receipt.selected_candidate,
        figures_directory / "bootstrap_effects",
        smoke=args.smoke,
    )
    (output / "README.md").write_text(
        build_readme(
            receipt.selected_candidate,
            methods,
            smoke=args.smoke,
            case_count=len(test_indices),
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test on a 2020-2021 subset"
            if args.smoke
            else "post-selection reused 2020-2021 development evaluation; not independent"
        ),
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "workflow_can_train": False,
            "workflow_can_select": False,
            "capacity_selection_locked_before_development": True,
            "development_metrics_used_for_selection": False,
            "development_years": list(base.TEST_YEARS),
            "evaluated_initializations": len(test_indices),
            "all_reported_ensemble_metrics_use_member_count": 51,
            "sealed_unopened_years": list(base.SEALED_YEARS),
            "sealed_2025_target_opened": False,
            "target": "IMD weekly mean precipitation, mm day-1",
            "region": "39N-0N, 60E-99E, 27x27 India box",
        },
        "capacity_receipt": {
            "path": str(receipt.manifest_path),
            "sha256": receipt.manifest_sha256,
            "experiment": receipt.manifest["experiment"],
            "status": receipt.manifest["status"],
            "mode": receipt.manifest["mode"],
            "artifact_inventory_verified": True,
            "current_training_sources_match_frozen_receipt": True,
        },
        "selection": selection_status,
        "methods": ["raw_fuxi", *methods],
        "seeds": list(SEEDS),
        "seed_handling": {
            "parameters_averaged": False,
            "adjustment_fields_averaged": False,
            "predictions_averaged": False,
            "headline_aggregation": "arithmetic mean of per-seed scores for identical initialization/lead rows",
            "per_seed_metrics_retained": True,
        },
        "input_checkpoints": input_checkpoints,
        "split_counts_archive": split_counts,
        "split_counts_evaluated": {"development": len(test_indices)},
        "evaluation": {
            "metrics": list(CORE_METRICS),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_block_length_initializations": 13,
            "bootstrap_resampling_unit": "initialization with all members and six leads grouped",
            "bootstrap_year_stratification": True,
            "selected_vs_base_bootstrap_performed": receipt.selected_distinct_from_base,
            "scoring_support_cells": int(np.count_nonzero(observations.weights > 0.0)),
        },
        "cache": provenance,
        "observation_stores": list(observations.source_stores),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot_checksums,
    }
    manifest["artifact_sha256"] = output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a locked capacity winner on reused 2020-2021 development data."
    )
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_samples is None:
        args.bootstrap_samples = 100 if args.smoke else 2000
    for name in ("batch_size", "evaluation_batch_size", "bootstrap_samples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    expected_bootstrap = 100 if args.smoke else 2000
    fixed = {
        "batch_size": (args.batch_size, 8),
        "evaluation_batch_size": (args.evaluation_batch_size, 8),
        "bootstrap_samples": (args.bootstrap_samples, expected_bootstrap),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical evaluation settings differ: {mismatch}")
    if args.no_amp:
        raise ValueError("canonical GPU evaluation requires automatic mixed precision")
    if args.device == "cpu":
        raise ValueError("canonical evaluation requires CUDA")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    receipt = validate_capacity_manifest(Path(args.capacity_manifest))
    requested_output = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {requested_output}"
        )
    staging = (
        requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging, receipt)
        os.replace(staging, requested_output)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "capacity_manifest": str(receipt.manifest_path),
                "capacity_manifest_sha256": receipt.manifest_sha256,
                "requested_output": str(requested_output),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} development evaluation "
        f"at {requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
