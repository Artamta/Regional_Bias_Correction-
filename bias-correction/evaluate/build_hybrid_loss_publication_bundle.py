#!/usr/bin/env python3
"""Build a publication bundle from the frozen hybrid-loss V2 experiment.

The source run is immutable.  Before reading metrics or copying figures, this
builder verifies every SHA-256 digest recorded in the frozen manifest and
enforces the canonical full-run, validation-selection, and sealed-2025
contracts.  Figures and primary tables are copied byte-for-byte; only
reliability ECE, rank-edge mass, the narrative README, and bundle manifest are
derived here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPECTED_EXPERIMENT = "fuxi_allseason_hybrid_loss_ablation_v1"
EXPECTED_PROFILES = (
    "crps_only",
    "hybrid_010",
    "hybrid_025",
    "hybrid_050",
    "mse_only",
)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_CACHE_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
)

DEFAULT_RUN = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_hybrid_loss_ablation/"
    "full_final_20260822T141844Z"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "presentation/deliverables/fuxi_hybrid_loss_ablation_20260822"
)

MAIN_FIGURE_STEMS = (
    "training_loss_curves",
    "validation_metric_curves",
    "validation_tradeoff",
    "weekwise_metrics",
    "weekwise_loss_heatmaps",
    "rank_histograms_raw_vs_selected",
    "reliability_diagrams",
    "probabilistic_diagnostics",
    "development_pareto",
)
KEY_TABLES = (
    "validation_checkpoint_metrics_by_seed.csv",
    "validation_profile_summary.csv",
    "pooled_metrics.csv",
    "weekwise_metrics.csv",
    "seed_weekwise_metrics.csv",
    "seed_variability_by_week.csv",
    "seasonal_weekwise_metrics.csv",
    "paired_block_bootstrap_vs_raw.csv",
    "paired_block_bootstrap_vs_crps_only.csv",
    "threshold_reliability_by_week.csv",
    "reliability_bins.csv",
    "seed_reliability_bins.csv",
    "rank_histograms.csv",
    "seed_rank_histograms.csv",
)


class BundleError(RuntimeError):
    """Raised when a frozen input or publication-bundle contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BundleError(f"JSON artifact must contain an object: {path}")
    return payload


def _safe_artifact_path(run: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise BundleError(f"unsafe frozen artifact path: {relative!r}")
    candidate = run / candidate_relative
    try:
        candidate.resolve(strict=False).relative_to(run.resolve())
    except ValueError as error:
        raise BundleError(f"frozen artifact escapes run directory: {relative!r}") from error
    return candidate


def _verify_selection(
    run: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    selection_path = run / "selection.json"
    selection = _read_json(selection_path)
    _require(
        manifest.get("selection_artifact") == "selection.json",
        "V2 manifest must declare selection.json as its selection artifact",
    )
    selection_digest = sha256_file(selection_path)
    _require(
        manifest.get("selection_sha256") == selection_digest,
        "manifest selection_sha256 does not match selection.json",
    )
    _require(
        artifacts.get("selection.json") == selection_digest,
        "selection.json is absent from or inconsistent with the artifact inventory",
    )
    _require(
        manifest.get("selection") == selection,
        "embedded manifest selection differs from the frozen selection.json",
    )
    _require(
        selection.get("status") == "frozen_before_development_evaluation",
        "selection was not frozen before development evaluation",
    )
    _require(
        selection.get("test_metrics_consulted") is False,
        "selection contract says development metrics were consulted",
    )
    _require(
        selection.get("development_evaluation_started") is False,
        "selection was written after development evaluation started",
    )
    _require(
        isinstance(selection.get("reason"), str) and bool(selection["reason"].strip()),
        "selection record has no validation-only reason",
    )
    selected = selection.get("selected_profile")
    _require(
        selected in EXPECTED_PROFILES[:-1],
        f"selected profile is not an eligible V2 arm: {selected!r}",
    )
    profiles = manifest.get("loss_profiles")
    _require(isinstance(profiles, list), "manifest loss_profiles must be a list")
    alpha_by_profile = {
        row.get("name"): float(row.get("alpha_mse"))
        for row in profiles
        if isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and isinstance(row.get("alpha_mse"), (int, float))
    }
    _require(
        set(alpha_by_profile) == set(EXPECTED_PROFILES),
        "manifest loss-profile definitions are incomplete",
    )
    _require(
        np.isclose(
            float(selection.get("selected_alpha_mse", np.nan)),
            alpha_by_profile[selected],
            rtol=0.0,
            atol=0.0,
        ),
        "selected alpha does not agree with the selected loss profile",
    )
    validation_profiles = selection.get("validation_profiles")
    _require(
        isinstance(validation_profiles, list)
        and {row.get("profile") for row in validation_profiles if isinstance(row, dict)}
        == set(EXPECTED_PROFILES),
        "selection lacks the complete validation-only profile comparison",
    )
    return selection


def verify_frozen_run(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every declared artifact and the canonical full V2 contract."""

    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing frozen manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("experiment") == EXPECTED_EXPERIMENT,
        "source is not the frozen hybrid-loss V2 experiment",
    )
    _require(manifest.get("status") == "complete", "source run is not complete")
    _require(manifest.get("mode") == "full", "source run is not full mode")
    _require(manifest.get("smoke") is False, "a smoke run cannot become paper evidence")
    _require(
        isinstance(manifest.get("scientific_status"), str)
        and bool(manifest["scientific_status"].strip()),
        "source run has no scientific-status declaration",
    )
    _require(
        tuple(manifest.get("profiles", ())) == EXPECTED_PROFILES,
        "source run does not contain the canonical V2 loss arms in order",
    )
    _require(
        tuple(manifest.get("seeds", ())) == EXPECTED_SEEDS,
        "source run does not contain the canonical three optimization seeds",
    )

    cache = manifest.get("cache", {})
    _require(isinstance(cache, dict), "manifest cache record is missing")
    _require(cache.get("scope") == "full_archive", "source did not use the full cache")
    _require(
        cache.get("data_sha256") == EXPECTED_CACHE_SHA256,
        "source did not use the canonical full member cache",
    )
    _require(
        cache.get("source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        "source member archive fingerprint is not canonical",
    )

    contract = manifest.get("contract", {})
    _require(isinstance(contract, dict), "manifest experiment contract is missing")
    _require(
        contract.get("development_years") == [2020, 2021],
        "development years differ from the frozen V2 contract",
    )
    _require(
        contract.get("validation_years") == [2018, 2019],
        "validation years differ from the frozen V2 contract",
    )
    _require(
        contract.get("sealed_unopened_years") == [2025]
        and contract.get("sealed_2025_target_opened") is False,
        "2025 is not recorded as sealed and unopened",
    )
    dual_use = contract.get("single_ensemble_dual_use", {})
    _require(
        isinstance(dual_use, dict)
        and dual_use.get("separate_model_blend") is False
        and "51-member" in str(dual_use.get("probabilistic", ""))
        and "same 51" in str(dual_use.get("deterministic", "")),
        "single-ensemble deterministic/probabilistic contract is missing",
    )
    _require(
        not any("2025.zarr" in str(path) for path in manifest.get("observation_stores", [])),
        "manifest observation inventory contains a 2025 store",
    )

    _require(
        manifest.get("split_counts_archive")
        == {"train": 1652, "validation": 196, "test": 208, "embargo": 24},
        "archive split counts differ from canonical full V2",
    )
    _require(
        manifest.get("split_counts_selected")
        == {"train": 1652, "validation": 196, "test_development": 208},
        "effective split counts differ from canonical full V2",
    )
    evaluation = manifest.get("evaluation", {})
    _require(isinstance(evaluation, dict), "manifest evaluation record is missing")
    _require(
        tuple(evaluation.get("methods", ())) == ("raw_fuxi", *EXPECTED_PROFILES),
        "evaluation method order differs from canonical V2",
    )
    _require(
        int(evaluation.get("bootstrap_samples", -1)) == 2000,
        "full V2 must use 2,000 paired block-bootstrap resamples",
    )
    _require(
        int(evaluation.get("support_cells", -1)) == 171,
        "full V2 scoring support must contain 171 cells",
    )
    training = manifest.get("training", {})
    _require(
        isinstance(training, dict)
        and int(training.get("full_members_for_validation_and_evaluation", -1)) == 51
        and int(training.get("member_subsample", -1)) == 16
        and int(training.get("max_epochs", -1)) == 100
        and int(training.get("patience", -1)) == 15,
        "training settings differ from canonical full V2",
    )

    artifacts = manifest.get("artifact_sha256")
    _require(
        isinstance(artifacts, dict) and bool(artifacts),
        "frozen manifest has no artifact SHA-256 inventory",
    )
    failures: list[str] = []
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            failures.append(f"invalid inventory entry {relative!r}")
            continue
        path = _safe_artifact_path(run, relative)
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"checksum mismatch {relative}")
    if failures:
        preview = "; ".join(failures[:8])
        suffix = f"; plus {len(failures) - 8} more" if len(failures) > 8 else ""
        raise BundleError(f"frozen-run verification failed: {preview}{suffix}")

    source_hashes = manifest.get("source_snapshot_sha256")
    _require(isinstance(source_hashes, dict) and bool(source_hashes), "no source hashes")
    _require(
        all(artifacts.get(path) == digest for path, digest in source_hashes.items()),
        "source-snapshot hashes disagree with the verified artifact inventory",
    )
    selection = _verify_selection(run, manifest, artifacts)
    return manifest, selection


def _require_columns(
    frame: pd.DataFrame,
    name: str,
    columns: Sequence[str],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BundleError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise BundleError(f"{name} is empty")


def reliability_ece(reliability: pd.DataFrame) -> pd.DataFrame:
    """Derive W1--W6 pooled area-weighted expected calibration error."""

    _require_columns(
        reliability,
        "reliability_bins.csv",
        (
            "method",
            "threshold_mm_day",
            "probability_bin",
            "area_weight_sum",
            "forecast_probability_weighted_sum",
            "observed_event_weighted_sum",
        ),
    )
    rows: list[dict[str, Any]] = []
    for (method, threshold), selected in reliability.groupby(
        ["method", "threshold_mm_day"], sort=True
    ):
        pooled = selected.groupby("probability_bin", as_index=False)[
            [
                "area_weight_sum",
                "forecast_probability_weighted_sum",
                "observed_event_weighted_sum",
            ]
        ].sum()
        valid = pooled.area_weight_sum.to_numpy(dtype=np.float64) > 0.0
        pooled = pooled.loc[valid]
        weight = pooled.area_weight_sum.to_numpy(dtype=np.float64)
        _require(len(weight) > 0 and np.isfinite(weight).all(), "invalid ECE weights")
        forecast = (
            pooled.forecast_probability_weighted_sum.to_numpy(dtype=np.float64)
            / weight
        )
        observed = (
            pooled.observed_event_weighted_sum.to_numpy(dtype=np.float64) / weight
        )
        _require(
            np.isfinite(forecast).all() and np.isfinite(observed).all(),
            "non-finite reliability values",
        )
        rows.append(
            {
                "method": str(method),
                "threshold_mm_day": float(threshold),
                "ece": float(np.sum(weight * np.abs(forecast - observed)) / weight.sum()),
                "nonempty_bins": int(valid.sum()),
                "definition": "area-weighted absolute reliability gap; W1-W6 pooled",
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "threshold_mm_day"])


def rank_edge_mass(ranks: pd.DataFrame) -> pd.DataFrame:
    """Derive per-week and W1--W6 pooled outer-rank mass."""

    _require_columns(
        ranks,
        "rank_histograms.csv",
        ("method", "lead_week", "rank", "count"),
    )
    rows: list[dict[str, Any]] = []

    def append_row(method: str, lead_week: int, selected: pd.DataFrame) -> None:
        counts = selected.groupby("rank")["count"].sum().sort_index()
        total = float(counts.sum())
        _require(len(counts) >= 2 and total > 0.0, "invalid rank histogram")
        rows.append(
            {
                "method": str(method),
                "lead_week": int(lead_week),
                "edge_mass": float((counts.iloc[0] + counts.iloc[-1]) / total),
                "uniform_expectation": float(2.0 / len(counts)),
            }
        )

    for (method, lead_week), selected in ranks.groupby(
        ["method", "lead_week"], sort=True
    ):
        append_row(str(method), int(lead_week), selected)
    for method, selected in ranks.groupby("method", sort=True):
        append_row(str(method), 0, selected)
    return pd.DataFrame(rows).sort_values(["method", "lead_week"])


def _single_row(frame: pd.DataFrame, context: str, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise BundleError(f"expected one {context} row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _effect_lines(
    bootstrap: pd.DataFrame,
    method: str,
    baseline: str,
) -> list[str]:
    _require_columns(
        bootstrap,
        "paired block bootstrap",
        (
            "method",
            "baseline",
            "lead_scope",
            "metric",
            "effect_name",
            "effect",
            "ci_lower",
            "ci_upper",
        ),
    )
    selected = bootstrap.loc[
        (bootstrap.method == method)
        & (bootstrap.baseline == baseline)
        & (bootstrap.lead_scope == "W1-W6")
    ]
    expected_metrics = ("crps", "rmse", "mae", "acc", "bias")
    if set(selected.metric) != set(expected_metrics) or len(selected) != len(
        expected_metrics
    ):
        raise BundleError(
            f"paired bootstrap lacks one pooled row per metric for {method} vs {baseline}"
        )
    lines = []
    for metric in expected_metrics:
        row = _single_row(selected, "paired-bootstrap", metric=metric)
        lines.append(
            f"- `{row.effect_name}`: {row.effect:+.3f} "
            f"(95% block-bootstrap CI {row.ci_lower:+.3f} to {row.ci_upper:+.3f})."
        )
    return lines


def build_readme(
    pooled: pd.DataFrame,
    selection: Mapping[str, Any],
    raw_bootstrap: pd.DataFrame,
    control_bootstrap: pd.DataFrame,
    ece: pd.DataFrame,
    edge_mass: pd.DataFrame,
    run: Path,
) -> str:
    """Construct the paper narrative only from verified frozen artifacts."""

    _require_columns(
        pooled,
        "pooled_metrics.csv",
        (
            "method",
            "method_label",
            "crps",
            "rmse",
            "mae",
            "bias",
            "acc",
            "spread_skill_ratio",
            "coverage_50",
            "coverage_80",
            "coverage_90",
        ),
    )
    selected = str(selection["selected_profile"])
    selected_alpha = float(selection["selected_alpha_mse"])
    selected_row = _single_row(pooled, "pooled metric", method=selected)
    raw_row = _single_row(pooled, "pooled metric", method="raw_fuxi")
    control_row = _single_row(pooled, "pooled metric", method="crps_only")
    validation_rows = {
        str(row["profile"]): row
        for row in selection["validation_profiles"]
        if isinstance(row, dict)
    }
    validation_selected = validation_rows[selected]
    selected_ece = float(ece.loc[ece.method == selected, "ece"].mean())
    raw_ece = float(ece.loc[ece.method == "raw_fuxi", "ece"].mean())
    selected_edge = _single_row(
        edge_mass, "pooled rank-edge", method=selected, lead_week=0
    )
    raw_edge = _single_row(
        edge_mass, "pooled rank-edge", method="raw_fuxi", lead_week=0
    )
    _require(
        np.isfinite([selected_ece, raw_ece]).all(),
        "selected/raw pooled ECE is unavailable",
    )

    lines = [
        "# FuXi hybrid-loss ablation: publication bundle",
        "",
        "Status: **post-hoc 2020–2021 development ablation; not an untouched final test**.",
        "",
        f"Frozen source run: `{run}`",
        "",
        "## One ensemble, two valid products",
        "",
        "Each arm returns one calibrated 51-member ensemble. The empirical member "
        "distribution supplies probabilities, quantiles, intervals, CRPS, and Brier scores; "
        "the physical mean of those same members supplies RMSE, MAE, bias, and ACC. There is "
        "no separately trained deterministic forecast and no model blend.",
        "",
        "## Validation-frozen selection",
        "",
        "Selection used 2018–2019 only and was frozen before development evaluation. "
        f"It chose **`{selected}` (α={selected_alpha:.2f})**. "
        f"Recorded reason: {selection['reason']}.",
        "",
        f"At its selected checkpoint, validation CRPS/RMSE/coverage error were "
        f"{float(validation_selected['mean_validation_crps']):.4f}/"
        f"{float(validation_selected['mean_validation_rmse']):.4f}/"
        f"{float(validation_selected['mean_validation_coverage_error']):.4f}. "
        "These validation quantities chose the arm; no 2020–2021 score chose it.",
        "",
        "## Reused 2020–2021 development readout",
        "",
        f"The validation-selected arm has pooled CRPS {selected_row.crps:.4f} and its "
        f"deterministic ensemble mean has RMSE {selected_row.rmse:.4f}. For context, raw "
        f"FuXi is {raw_row.crps:.4f}/{raw_row.rmse:.4f}, and the CRPS-only loss control is "
        f"{control_row.crps:.4f}/{control_row.rmse:.4f} (CRPS/RMSE).",
        "",
        f"Its spread/error ratio is {selected_row.spread_skill_ratio:.3f}; central "
        f"50/80/90% coverage is {selected_row.coverage_50:.3f}/"
        f"{selected_row.coverage_80:.3f}/{selected_row.coverage_90:.3f}. The arithmetic "
        f"mean ECE across the fixed 1/5/10/20 mm day⁻¹ thresholds is "
        f"{selected_ece:.4f} ({raw_ece:.4f} raw), and pooled outer-rank mass is "
        f"{selected_edge.edge_mass:.4f} ({raw_edge.edge_mass:.4f} raw; uniform expectation "
        f"{selected_edge.uniform_expectation:.4f}).",
        "",
        "### Validation-selected arm versus raw FuXi",
        "",
        *_effect_lines(raw_bootstrap, selected, "raw_fuxi"),
        "",
    ]
    if selected == "crps_only":
        lines.extend(
            [
                "### Validation-selected arm versus CRPS only",
                "",
                "Not applicable: the validation rule retained the CRPS-only control.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Validation-selected arm versus CRPS only",
                "",
                *_effect_lines(control_bootstrap, selected, "crps_only"),
                "",
            ]
        )

    lines.extend(
        [
            "## Pooled metrics for every fixed arm",
            "",
            "| Method | CRPS | RMSE | MAE | Bias | ACC | Spread/error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("raw_fuxi", *EXPECTED_PROFILES):
        row = _single_row(pooled, "pooled metric", method=method)
        lines.append(
            f"| {row.method_label} | {row.crps:.4f} | {row.rmse:.4f} | "
            f"{row.mae:.4f} | {row.bias:+.4f} | {row.acc:.4f} | "
            f"{row.spread_skill_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This bundle tests only the predeclared loss-weight ablation. The paired "
            "intervals—not whichever isolated point estimate is smallest—are the uncertainty "
            "statement. The 2020–2021 period has prior development exposure. The 2025 control "
            "was not opened, so this bundle cannot support an independent-test or operational "
            "deployment claim.",
            "",
            "## Contents",
            "",
            "- `selection.json`: byte-identical validation-only selection record.",
            "- `figures/`: byte-identical full-run V2 PDF and PNG figures.",
            "- `tables/`: frozen validation, training, pooled, weekwise, seasonal, bootstrap, "
            "reliability, rank, and method-by-week tables.",
            "- `tables/reliability_ece.csv`: derived W1–W6 area-weighted reliability gaps.",
            "- `tables/rank_edge_mass.csv`: derived per-week and pooled outer-rank mass.",
            "- `manifest.json`: source provenance plus SHA-256 for every other bundle file.",
            "",
        ]
    )
    return "\n".join(lines)


def _declared_source(
    run: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> Path:
    artifacts = manifest["artifact_sha256"]
    if relative not in artifacts:
        raise BundleError(f"required source artifact is not hash-declared: {relative}")
    source = _safe_artifact_path(run, relative)
    if not source.is_file():
        raise BundleError(f"required source artifact is missing: {relative}")
    return source


def _copy_declared(
    run: Path,
    manifest: Mapping[str, Any],
    relative: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_declared_source(run, manifest, relative), destination)


def _copy_declared_tree(
    run: Path,
    manifest: Mapping[str, Any],
    relative_directory: str,
    destination: Path,
) -> int:
    source_directory = run / relative_directory
    if not source_directory.is_dir():
        raise BundleError(f"missing required source directory: {relative_directory}")
    count = 0
    for source in sorted(path for path in source_directory.rglob("*") if path.is_file()):
        relative = str(source.relative_to(run))
        target = destination / source.relative_to(source_directory)
        _copy_declared(run, manifest, relative, target)
        count += 1
    if count == 0:
        raise BundleError(f"required source directory is empty: {relative_directory}")
    return count


def _load_verified_frames(
    run: Path,
    manifest: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    names = {
        "pooled": "metrics/pooled_metrics.csv",
        "raw_bootstrap": "metrics/paired_block_bootstrap_vs_raw.csv",
        "control_bootstrap": "metrics/paired_block_bootstrap_vs_crps_only.csv",
        "reliability": "metrics/reliability_bins.csv",
        "ranks": "metrics/rank_histograms.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, relative in names.items():
        source = _declared_source(run, manifest, relative)
        try:
            frames[key] = pd.read_csv(source)
        except (OSError, pd.errors.ParserError) as error:
            raise BundleError(f"cannot read verified table {relative}: {error}") from error
    return frames


def _remove_existing_output(output: Path) -> None:
    if output.is_symlink() or output.is_file():
        output.unlink()
    elif output.is_dir():
        shutil.rmtree(output)


def build_bundle(run: Path, output: Path, *, replace_derived: bool = False) -> None:
    """Verify, derive, checksum, and atomically publish the requested bundle."""

    run = run.resolve()
    output = output.resolve()
    manifest, selection = verify_frozen_run(run)
    frames = _load_verified_frames(run, manifest)
    ece = reliability_ece(frames["reliability"])
    edge_mass = rank_edge_mass(frames["ranks"])
    readme = build_readme(
        frames["pooled"],
        selection,
        frames["raw_bootstrap"],
        frames["control_bootstrap"],
        ece,
        edge_mass,
        run,
    )

    if output.exists() and not replace_derived:
        raise BundleError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise BundleError(f"staging path already exists: {staging}")

    copied_figures = 0
    copied_matrix_tables = 0
    try:
        figures = staging / "figures"
        tables = staging / "tables"
        figures.mkdir(parents=True)
        tables.mkdir(parents=True)

        source_figure_directory = run / "figures"
        if not source_figure_directory.is_dir():
            raise BundleError("full V2 run has no figures directory")
        source_figures = sorted(
            path
            for path in source_figure_directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
        )
        available_names = {path.name for path in source_figures}
        required_names = {
            f"{stem}.{suffix}"
            for stem in MAIN_FIGURE_STEMS
            for suffix in ("pdf", "png")
        }
        missing_figures = sorted(required_names - available_names)
        if missing_figures:
            raise BundleError(
                f"full V2 run is missing main figure files: {missing_figures}"
            )
        for source in source_figures:
            relative = str(source.relative_to(run))
            _copy_declared(run, manifest, relative, figures / source.name)
            copied_figures += 1

        for filename in KEY_TABLES:
            _copy_declared(
                run,
                manifest,
                f"metrics/{filename}",
                tables / filename,
            )
        _copy_declared(
            run,
            manifest,
            "history/training_history.csv",
            tables / "training_history.csv",
        )
        copied_matrix_tables += _copy_declared_tree(
            run, manifest, "metrics/matrices", tables / "matrices"
        )
        copied_matrix_tables += _copy_declared_tree(
            run, manifest, "metrics/seasonal_matrices", tables / "seasonal_matrices"
        )
        _copy_declared(run, manifest, "selection.json", staging / "selection.json")
        _copy_declared(
            run,
            manifest,
            "training_reference.json",
            staging / "training_reference.json",
        )

        ece.to_csv(tables / "reliability_ece.csv", index=False)
        edge_mass.to_csv(tables / "rank_edge_mass.csv", index=False)
        (staging / "README.md").write_text(readme, encoding="utf-8")

        artifact_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "bundle_role": "verified hybrid-loss V2 publication evidence",
                "scientific_status": manifest["scientific_status"],
                "frozen_run": str(run),
                "frozen_experiment": manifest["experiment"],
                "frozen_run_manifest_sha256": sha256_file(run / "manifest.json"),
                "frozen_artifacts_verified": len(manifest["artifact_sha256"]),
                "selection_profile": selection["selected_profile"],
                "selection_alpha_mse": selection["selected_alpha_mse"],
                "selection_sha256": sha256_file(staging / "selection.json"),
                "generator_sha256": sha256_file(Path(__file__).resolve()),
                "copied_figure_files": copied_figures,
                "copied_matrix_table_files": copied_matrix_tables,
                "derived_tables": [
                    "tables/reliability_ece.csv",
                    "tables/rank_edge_mass.csv",
                ],
                "development_status": (
                    "post-hoc 2020-2021 development ablation; not an independent test"
                ),
                "sealed_2025_target_opened": False,
                "checksum_scope": (
                    "every bundle file except manifest.json, which cannot self-hash"
                ),
                "artifact_sha256": artifact_hashes,
            },
        )
        actual_outputs = {
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        _require(
            actual_outputs == set(artifact_hashes),
            "bundle checksum inventory does not cover every non-manifest output",
        )
        for relative, expected in artifact_hashes.items():
            _require(
                sha256_file(staging / relative) == expected,
                f"bundle output changed after hashing: {relative}",
            )

        if output.exists():
            _remove_existing_output(output)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace-derived",
        action="store_true",
        help="replace an existing derived bundle; the frozen source run is untouched",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_bundle(
        args.run,
        args.output,
        replace_derived=bool(args.replace_derived),
    )
    print(f"PASS: hybrid-loss publication bundle written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
