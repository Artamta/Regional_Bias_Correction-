#!/usr/bin/env python3
"""Build a manuscript-scale bundle from a frozen all-season ensemble run.

The source run is treated as immutable.  Every hash recorded in its manifest
is checked before any table is read.  This script then redraws figures at a
7.2-inch manuscript width and adds a clearly labelled post-hoc paired
comparison of the primary model against the summary-only neural ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fuxi_allseason_ensemble_calibration import (
    PRIMARY_CONFIGURATION,
    paired_block_bootstrap,
    plot_probabilistic_diagnostics,
    plot_rank_histograms,
    plot_reliability,
    plot_skill_heatmaps,
    plot_training_loss,
    plot_weekwise_metrics,
)
from project_paths import PROJECT_ROOT


DEFAULT_RUN = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_ensemble_calibration/"
    "full_publication_20260822T115253Z"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "presentation/deliverables/"
    "fuxi_allseason_ensemble_calibration_20260822"
)
SUMMARY_TABLES = (
    "pooled_metrics.csv",
    "weekwise_metrics.csv",
    "seed_weekwise_metrics.csv",
    "seed_variability_by_week.csv",
    "seasonal_weekwise_metrics.csv",
    "paired_block_bootstrap.csv",
    "threshold_reliability_by_week.csv",
)


class BundleError(RuntimeError):
    """Raised when a frozen input or bundle contract is violated."""


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


def verify_frozen_run(run: Path) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing frozen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise BundleError("publication bundle requires a complete non-smoke run")
    expected = manifest.get("artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise BundleError("frozen manifest has no artifact checksums")
    failures = []
    for relative, checksum in expected.items():
        path = run / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif sha256_file(path) != checksum:
            failures.append(f"checksum mismatch {relative}")
    if failures:
        raise BundleError("frozen-run verification failed: " + "; ".join(failures[:8]))
    return manifest


def reliability_ece(reliability: pd.DataFrame) -> pd.DataFrame:
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
        valid = pooled.area_weight_sum > 0.0
        pooled = pooled.loc[valid]
        weight = pooled.area_weight_sum.to_numpy(dtype=np.float64)
        forecast = (
            pooled.forecast_probability_weighted_sum.to_numpy(dtype=np.float64)
            / weight
        )
        observed = (
            pooled.observed_event_weighted_sum.to_numpy(dtype=np.float64) / weight
        )
        rows.append(
            {
                "method": method,
                "threshold_mm_day": float(threshold),
                "ece": float(np.sum(weight * np.abs(forecast - observed)) / weight.sum()),
                "nonempty_bins": int(valid.sum()),
                "definition": "area-weighted absolute reliability gap; W1-W6 pooled",
            }
        )
    return pd.DataFrame(rows)


def rank_edge_mass(ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, lead_week), selected in ranks.groupby(
        ["method", "lead_week"], sort=True
    ):
        counts = selected.groupby("rank")["count"].sum().sort_index()
        total = float(counts.sum())
        rows.append(
            {
                "method": method,
                "lead_week": int(lead_week),
                "edge_mass": float((counts.iloc[0] + counts.iloc[-1]) / total),
                "uniform_expectation": float(2.0 / len(counts)),
            }
        )
    for method, selected in ranks.groupby("method", sort=True):
        counts = selected.groupby("rank")["count"].sum().sort_index()
        total = float(counts.sum())
        rows.append(
            {
                "method": method,
                "lead_week": 0,
                "edge_mass": float((counts.iloc[0] + counts.iloc[-1]) / total),
                "uniform_expectation": float(2.0 / len(counts)),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "lead_week"])


def build_readme(
    pooled: pd.DataFrame,
    raw_bootstrap: pd.DataFrame,
    posthoc: pd.DataFrame,
    run: Path,
) -> str:
    indexed = pooled.set_index("method")
    raw = indexed.loc["raw_fuxi"]
    primary = indexed.loc[PRIMARY_CONFIGURATION]
    uncertainty = raw_bootstrap.loc[
        (raw_bootstrap.method == PRIMARY_CONFIGURATION)
        & (raw_bootstrap.lead_scope == "W1-W6")
    ].set_index("metric")
    comparison = posthoc.loc[posthoc.lead_scope == "W1-W6"].set_index("metric")
    return "\n".join(
        [
            "# All-season FuXi ensemble calibration: publication bundle",
            "",
            "Status: **derived from a verified retrospective 2020–2021 development test; "
            "not an independent final test**.",
            "",
            f"Frozen source run: `{run}`",
            "",
            "The primary location-and-spread adapter keeps all 51 weather members. "
            f"Its pooled CRPS is {primary.crps:.4f} versus {raw.crps:.4f} for raw FuXi, "
            f"a {uncertainty.loc['crps', 'effect']:.2f}% improvement "
            f"(95% block-bootstrap CI {uncertainty.loc['crps', 'ci_lower']:.2f}% to "
            f"{uncertainty.loc['crps', 'ci_upper']:.2f}%). RMSE improves "
            f"{uncertainty.loc['rmse', 'effect']:.2f}% and ACC changes by "
            f"{uncertainty.loc['acc', 'effect']:+.3f}.",
            "",
            "The central paper result is probabilistic calibration: RMS spread / pooled RMS "
            f"error changes from {raw.spread_skill_ratio:.3f} to "
            f"{primary.spread_skill_ratio:.3f}, while 50/80/90% central coverage changes "
            f"from {raw.coverage_50:.3f}/{raw.coverage_80:.3f}/{raw.coverage_90:.3f} to "
            f"{primary.coverage_50:.3f}/{primary.coverage_80:.3f}/{primary.coverage_90:.3f}.",
            "",
            "The learned set encoder adds only a small increment over the summary-only neural "
            f"ablation: {comparison.loc['crps', 'effect']:.3f}% pooled CRPS skill "
            f"(post-hoc 95% CI {comparison.loc['crps', 'ci_lower']:.3f}% to "
            f"{comparison.loc['crps', 'ci_upper']:.3f}%). This comparison was added after the "
            "frozen run and must be labelled post-hoc. The defensible ablation claim is that "
            "learning spread matters; do not claim that the member encoder is essential.",
            "",
            f"Signed bias is not improved overall ({raw.bias:+.3f} raw versus "
            f"{primary.bias:+.3f} primary), and its paired change interval crosses zero. "
            "Residual undercoverage is largest at weeks 1–2. MAM dry bias and JJA wet bias "
            "must be disclosed.",
            "",
            "## Contents",
            "",
            "- `figures/`: manuscript-width PDF and 300-dpi PNG figures with comparable rank axes, "
            "uncertainty on the main CRPSS panel, seed-readable loss curves, and Type 42 PDF fonts.",
            "- `tables/`: unchanged frozen summary tables plus post-hoc primary-vs-summary "
            "bootstrap, ECE, rank-edge, and metric matrices.",
            "- `manifest.json`: verified input provenance and hashes for every derived artifact.",
            "",
            "No 2025 control data were opened. Independent confirmation remains necessary before "
            "an operational or final-test claim.",
            "",
        ]
    )


def build_bundle(run: Path, output: Path, *, replace_derived: bool = False) -> None:
    manifest = verify_frozen_run(run)
    if (output / "manifest.json").is_file() and not replace_derived:
        raise BundleError(f"refusing to overwrite completed output: {output}")
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    metrics = run / "metrics"
    history = pd.read_csv(run / "history/training_history.csv")
    cases = pd.read_csv(metrics / "case_metrics.csv")
    weekwise = pd.read_csv(metrics / "weekwise_metrics.csv")
    variability = pd.read_csv(metrics / "seed_variability_by_week.csv")
    seasonal = pd.read_csv(metrics / "seasonal_weekwise_metrics.csv")
    bootstrap = pd.read_csv(metrics / "paired_block_bootstrap.csv")
    ranks = pd.read_csv(metrics / "rank_histograms.csv")
    reliability = pd.read_csv(metrics / "reliability_bins.csv")
    pooled = pd.read_csv(metrics / "pooled_metrics.csv")
    method_order = tuple(manifest["methods"])

    initializations = np.asarray(
        sorted(pd.to_datetime(cases.init.unique()).to_numpy()),
        dtype="datetime64[ns]",
    )
    posthoc = paired_block_bootstrap(
        cases,
        initializations,
        (PRIMARY_CONFIGURATION,),
        n_resamples=int(manifest["evaluation"]["bootstrap_samples"]),
        block_length=int(
            manifest["evaluation"]["bootstrap_block_length_initializations"]
        ),
        seed=42,
        baseline="summary_only",
    )
    posthoc.insert(0, "analysis_status", "post_hoc_not_predeclared")
    posthoc.to_csv(tables / "primary_vs_summary_block_bootstrap.csv", index=False)
    reliability_ece(reliability).to_csv(tables / "reliability_ece.csv", index=False)
    rank_edge_mass(ranks).to_csv(tables / "rank_edge_mass.csv", index=False)

    for filename in SUMMARY_TABLES:
        shutil.copy2(metrics / filename, tables / filename)
    shutil.copy2(run / "history/training_history.csv", tables / "training_history.csv")
    shutil.copytree(metrics / "matrices", tables / "matrices", dirs_exist_ok=True)
    shutil.copytree(
        metrics / "seasonal_matrices",
        tables / "seasonal_matrices",
        dirs_exist_ok=True,
    )

    plot_training_loss(history, figures / "training_loss_curves", smoke=False)
    plot_weekwise_metrics(
        weekwise,
        figures / "weekwise_metrics",
        method_order,
        smoke=False,
        bootstrap=bootstrap,
        seed_variability=variability,
    )
    plot_skill_heatmaps(
        weekwise,
        figures / "weekwise_ablation_heatmaps",
        method_order,
        smoke=False,
    )
    plot_rank_histograms(
        ranks,
        figures / "rank_histograms_raw_vs_primary",
        PRIMARY_CONFIGURATION,
        smoke=False,
    )
    plot_reliability(
        reliability,
        figures / "reliability_diagrams",
        method_order,
        smoke=False,
    )
    plot_probabilistic_diagnostics(
        weekwise,
        seasonal,
        figures / "probabilistic_diagnostics",
        method_order,
        smoke=False,
    )

    (output / "README.md").write_text(
        build_readme(pooled, bootstrap, posthoc, run.resolve()),
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve()
    plotting_source = PROJECT_ROOT / "src/fuxi_allseason_ensemble_calibration.py"
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_role": "derived manuscript-scale figures and summary tables",
            "scientific_status": manifest["scientific_status"],
            "frozen_run": str(run.resolve()),
            "frozen_run_manifest_sha256": sha256_file(run / "manifest.json"),
            "frozen_artifacts_verified": len(manifest["artifact_sha256"]),
            "posthoc_warning": (
                "primary-vs-summary bootstrap was not predeclared in the frozen run; "
                "label it post-hoc"
            ),
            "generator_sha256": sha256_file(script_path),
            "plotting_source_sha256": sha256_file(plotting_source),
            "artifact_sha256": artifacts,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace-derived",
        action="store_true",
        help="replace an existing bundle at --output; the frozen source run is untouched",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_bundle(
        args.run.resolve(),
        args.output.resolve(),
        replace_derived=args.replace_derived,
    )
    print(f"PASS: publication bundle written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
