#!/usr/bin/env python3
"""Publish IMD homogeneous-region and seasonal Model-v1 comparisons.

This table-only compiler consumes the frozen 2020--2024 probabilistic bridge
case scores.  It never opens forecast, observation, or checkpoint arrays and
never accesses a 2025 target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from india_s2s_block_bootstrap import circular_year_stratified_index_matrix
from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_regional_seasonal_comparison_v1"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge"
    / "scoring_full_20260825T234808Z"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_regional_seasonal_comparison"
SOURCE_MANIFEST_SHA256 = (
    "4ab303a3fa0025bfb2b8654bf34770091f117df18f236c0c724d8e1078450943"
)
SOURCE_CASES_SHA256 = (
    "36e2e1ccadf43e29144c4267e290ca001fc3d1f41658b630f665fd158b1d4f8c"
)
SOURCE_CASE_ROWS = 45_450
SELECTED_CASE_ROWS = 24_240

BASELINE = "raw_fuxi"
CANDIDATE = "location_spread"
METHODS = (BASELINE, CANDIDATE)
METHOD_LABELS = {
    BASELINE: "Model v1",
    CANDIDATE: "Probabilistic Correction v1",
}
REGION_LABELS = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}
REGIONS = tuple(REGION_LABELS)
SEASON_LABELS = {
    "JF": "January–February",
    "MAM": "March–May",
    "JJAS": "June–September",
    "OND": "October–December",
}
SEASONS = tuple(SEASON_LABELS)
LEADS = (1, 2, 3, 4, 5, 6)
METRICS = ("crps", "acc", "rmse", "mae", "bias")
LOSS_METRICS = ("crps", "rmse", "mae")
EXPECTED_CASE_COUNTS = {
    "JF": (84, 82, 80, 78, 76, 74),
    "MAM": (131, 131, 131, 131, 131, 131),
    "JJAS": (169, 169, 169, 169, 169, 169),
    "OND": (121, 123, 125, 127, 129, 131),
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BLOCK_LENGTH = 4
BOOTSTRAP_SEED = 20260902


class RegionalSeasonalError(RuntimeError):
    """Raised when the regional/seasonal comparison contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegionalSeasonalError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_frozen_source(source: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(source).resolve()
    manifest_path = root / "manifest.json"
    cases_path = root / "tables/case_metrics.csv"
    _require(manifest_path.is_file(), f"source manifest is missing: {manifest_path}")
    _require(cases_path.is_file(), f"source case table is missing: {cases_path}")
    _require(
        sha256_file(manifest_path) == SOURCE_MANIFEST_SHA256,
        "source manifest SHA-256 differs",
    )
    _require(
        sha256_file(cases_path) == SOURCE_CASES_SHA256,
        "source case-table SHA-256 differs",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "india_s2s_probabilistic_bridge_scoring_v1",
        "status": "complete",
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
        "opened_observation_years": [2020, 2021, 2022, 2023, 2024],
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"source manifest {key} differs")
    _require(
        manifest.get("cohort", {}).get("maximum_target_label_opened") == "2024-12-30",
        "source maximum target label differs",
    )
    return manifest_path, cases_path, manifest


def _validated_cases(cases: pd.DataFrame, *, frozen: bool) -> pd.DataFrame:
    required = {
        "method",
        "reference",
        "score_status",
        "verification_year",
        "init",
        "valid_period_midpoint",
        "season",
        "region",
        "region_label",
        "lead_week",
        *METRICS,
    }
    missing = sorted(required.difference(cases.columns))
    _require(not missing, f"case table lacks columns: {missing}")
    if frozen:
        _require(len(cases) == SOURCE_CASE_ROWS, "source case-row count differs")
    selected = cases[
        cases.method.isin(METHODS)
        & cases.region.isin(REGIONS)
        & cases.season.isin(SEASONS)
        & cases.lead_week.isin(LEADS)
        & cases.reference.eq("imd")
        & cases.score_status.eq("available")
    ].copy()
    if frozen:
        _require(len(selected) == SELECTED_CASE_ROWS, "selected case-row count differs")
        _require(
            set(selected.verification_year.astype(int)) == {2020, 2021, 2022, 2023, 2024},
            "verification years differ",
        )
        midpoints = pd.to_datetime(selected.valid_period_midpoint)
        _require(midpoints.max() <= pd.Timestamp("2024-12-31"), "comparison crosses 2025")
    labels = selected[["region", "region_label"]].drop_duplicates()
    _require(
        labels.set_index("region")["region_label"].to_dict() == REGION_LABELS,
        "IMD homogeneous-region labels differ",
    )
    _require(
        not selected.duplicated(["method", "region", "season", "lead_week", "init"]).any(),
        "selected cases contain duplicate method/region/season/lead rows",
    )
    numeric = selected[list(METRICS)].to_numpy(dtype=np.float64)
    _require(np.isfinite(numeric).all(), "selected case metrics contain non-finite values")
    if frozen:
        counts = selected.groupby(["method", "region", "season", "lead_week"]).size()
        for season, expected in EXPECTED_CASE_COUNTS.items():
            for lead, count in zip(LEADS, expected, strict=True):
                values = counts.loc[(slice(None), slice(None), season, lead)]
                _require((values == count).all(), f"{season} W{lead} case count differs")
    selected["method_label"] = selected.method.map(METHOD_LABELS)
    selected["season_label"] = selected.season.map(SEASON_LABELS)
    return selected.sort_values(
        ["season", "region", "lead_week", "method", "init"], kind="stable"
    ).reset_index(drop=True)


def descriptive_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    """Return arithmetic case-score means for every method/season/region/lead."""
    rows: list[dict[str, Any]] = []
    keys = ["season", "region", "lead_week", "method"]
    for key, group in cases.groupby(keys, sort=False):
        season, region, lead, method = key
        row: dict[str, Any] = {
            "season": season,
            "season_label": SEASON_LABELS[str(season)],
            "region": region,
            "region_label": REGION_LABELS[str(region)],
            "lead_week": int(lead),
            "method": method,
            "method_label": METHOD_LABELS[str(method)],
            "case_count": int(len(group)),
        }
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(SEASONS) * len(REGIONS) * len(LEADS) * len(METHODS)
    _require(len(result) == expected, "descriptive table is incomplete")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _paired_arrays(group: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wide = group.pivot(index="init", columns="method", values=metric).sort_index()
    _require(set(wide.columns) == set(METHODS), f"{metric} method pairing differs")
    _require(not wide.isna().any().any(), f"{metric} pairing is incomplete")
    return (
        wide.index.to_numpy(dtype="datetime64[D]"),
        wide[CANDIDATE].to_numpy(dtype=np.float64),
        wide[BASELINE].to_numpy(dtype=np.float64),
    )


def paired_intervals(
    cases: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return paired candidate-versus-baseline intervals with shared regional draws."""
    _require(replicates > 0 and block_length > 0, "bootstrap controls must be positive")
    rows: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    for season_index, season in enumerate(SEASONS):
        for lead in LEADS:
            reference_dates: np.ndarray | None = None
            indices: np.ndarray | None = None
            draw_seed = int(seed + 10_000 * season_index + 100 * lead + block_length)
            for region in REGIONS:
                group = cases[
                    cases.season.eq(season)
                    & cases.region.eq(region)
                    & cases.lead_week.eq(lead)
                ]
                dates, _, _ = _paired_arrays(group, "crps")
                if reference_dates is None:
                    reference_dates = dates
                    indices = circular_year_stratified_index_matrix(
                        dates,
                        block_length=block_length,
                        replicates=replicates,
                        seed=draw_seed,
                    )
                    receipts[f"{season}__w{lead}"] = {
                        "case_count": int(len(dates)),
                        "dates_sha256": hashlib.sha256(
                            "".join(
                                f"{np.datetime_as_string(value, unit='D')}\n" for value in dates
                            ).encode("ascii")
                        ).hexdigest(),
                        "index_matrix_sha256": hashlib.sha256(
                            np.ascontiguousarray(indices).tobytes()
                        ).hexdigest(),
                        "shape": list(indices.shape),
                        "seed": draw_seed,
                    }
                else:
                    _require(
                        np.array_equal(dates, reference_dates),
                        f"regional date cohorts differ for {season} W{lead}",
                    )
                assert indices is not None
                for metric in METRICS:
                    metric_dates, candidate, baseline = _paired_arrays(group, metric)
                    _require(np.array_equal(metric_dates, dates), "metric date pairing differs")
                    differences = candidate - baseline
                    samples = differences[indices].mean(axis=1, dtype=np.float64)
                    lower, upper = np.quantile(samples, (0.025, 0.975))
                    favorable = (
                        bool(upper < 0.0)
                        if metric in LOSS_METRICS
                        else bool(lower > 0.0)
                        if metric == "acc"
                        else False
                    )
                    common = {
                        "season": season,
                        "season_label": SEASON_LABELS[season],
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "lead_week": int(lead),
                        "metric": metric,
                        "candidate": CANDIDATE,
                        "candidate_label": METHOD_LABELS[CANDIDATE],
                        "baseline": BASELINE,
                        "baseline_label": METHOD_LABELS[BASELINE],
                        "case_count": int(len(dates)),
                        "confidence": 0.95,
                        "block_length_starts": int(block_length),
                        "replicates": int(replicates),
                        "seed": draw_seed,
                    }
                    rows.append(
                        {
                            **common,
                            "effect": "candidate_minus_baseline",
                            "estimate": float(differences.mean()),
                            "ci_lower": float(lower),
                            "ci_upper": float(upper),
                            "resolved_improvement": favorable,
                        }
                    )
                    if metric in LOSS_METRICS:
                        candidate_means = candidate[indices].mean(axis=1)
                        baseline_means = baseline[indices].mean(axis=1)
                        _require(np.all(baseline_means > 0.0), "bootstrap baseline is non-positive")
                        skill = 100.0 * (1.0 - candidate_means / baseline_means)
                        skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                        rows.append(
                            {
                                **common,
                                "effect": "skill_pct_vs_baseline",
                                "estimate": float(100.0 * (1.0 - candidate.mean() / baseline.mean())),
                                "ci_lower": float(skill_lower),
                                "ci_upper": float(skill_upper),
                                "resolved_improvement": bool(skill_lower > 0.0),
                            }
                        )
    result = pd.DataFrame(rows)
    expected = len(SEASONS) * len(REGIONS) * len(LEADS) * (len(METRICS) + len(LOSS_METRICS))
    _require(len(result) == expected, "paired interval table is incomplete")
    return result, receipts


def comparison_figure(intervals: pd.DataFrame, path: Path) -> None:
    """Render CRPS/RMSE skill and ACC change for every season and region."""
    panels = (
        ("crps", "skill_pct_vs_baseline", "CRPS skill vs Model v1 (%)"),
        ("rmse", "skill_pct_vs_baseline", "RMSE skill vs Model v1 (%)"),
        ("acc", "candidate_minus_baseline", "ACC change vs Model v1"),
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    figure, axes = plt.subplots(3, 4, figsize=(15.5, 10.5), sharex=True)
    for column, season in enumerate(SEASONS):
        for row, (metric, effect, ylabel) in enumerate(panels):
            axis = axes[row, column]
            selected = intervals[
                intervals.season.eq(season)
                & intervals.metric.eq(metric)
                & intervals.effect.eq(effect)
            ]
            for color, region in zip(colors, REGIONS, strict=True):
                values = selected[selected.region.eq(region)].sort_values("lead_week")
                axis.plot(
                    values.lead_week,
                    values.estimate,
                    marker="o",
                    linewidth=1.7,
                    markersize=4,
                    color=color,
                    label=REGION_LABELS[region],
                )
                axis.fill_between(
                    values.lead_week,
                    values.ci_lower,
                    values.ci_upper,
                    color=color,
                    alpha=0.10,
                )
            axis.axhline(0.0, color="#333333", linewidth=0.8)
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_xticks(LEADS, labels=[f"W{lead}" for lead in LEADS])
            if row == 0:
                axis.set_title(f"{season}: {SEASON_LABELS[season]}", fontweight="semibold")
            if column == 0:
                axis.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Probabilistic Correction v1 improvement by IMD homogeneous region and season",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.010,
        "2020–2024 retrospective cases; 95% year-stratified circular-block intervals; valid-period-midpoint seasons.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.105, 1, 0.95))
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": path.stem, "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _write_results(path: Path, descriptive: pd.DataFrame, intervals: pd.DataFrame) -> None:
    rows = [
        "# IMD homogeneous-region and seasonal comparison\n",
        "Model v1 is compared with Probabilistic Correction v1 for W1–W6 using frozen 2020–2024 IMD verification. Seasons are assigned by the midpoint of each seven-day valid period.\n",
        "Each metric cell reports resolved improvement / resolved degradation; remaining cells cross zero.",
        "",
        "| Season | CRPS cells | RMSE cells | ACC cells | Region-lead cells |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    total_cells = len(REGIONS) * len(LEADS)
    for season in SEASONS:
        counts: list[str] = []
        for metric, effect in (
            ("crps", "skill_pct_vs_baseline"),
            ("rmse", "skill_pct_vs_baseline"),
            ("acc", "candidate_minus_baseline"),
        ):
            subset = intervals[
                intervals.season.eq(season)
                & intervals.metric.eq(metric)
                & intervals.effect.eq(effect)
            ]
            improved = int((subset.ci_lower > 0.0).sum())
            degraded = int((subset.ci_upper < 0.0).sum())
            counts.append(f"{improved} / {degraded}")
        rows.append(
            f"| {season} ({SEASON_LABELS[season]}) | {counts[0]} | {counts[1]} | {counts[2]} | {total_cells} |"
        )
    rows.extend(
        [
            "",
            "Intervals are pointwise and descriptive; no multiplicity correction is applied. JF and OND case counts vary by lead because season membership is based on valid-period midpoint and the frozen archive ends in 2024. ERPAS and Deterministic Correction v1 are not included because comparable all-season forecasts are unavailable. No 2025 observation was opened.",
            "",
            f"The descriptive table contains {len(descriptive)} method-region-season-lead rows and the paired table contains {len(intervals)} interval rows.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def publish(
    *,
    source: Path = DEFAULT_SOURCE,
    output: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> Path:
    """Publish a fresh, atomic regional/seasonal comparison package."""
    destination = Path(output).resolve()
    root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(root in destination.parents, "output must be a child of the comparison root")
    _require(not destination.exists(), "output path must be fresh")
    manifest_path, cases_path, source_manifest = _validate_frozen_source(source)
    cases = _validated_cases(pd.read_csv(cases_path), frozen=True)
    descriptive = descriptive_metrics(cases)
    intervals, sampling = paired_intervals(
        cases,
        replicates=replicates,
        block_length=block_length,
        seed=seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        tables = stage / "tables"
        figures = stage / "figures"
        receipts = stage / "receipts"
        code = stage / "code/src"
        for directory in (tables, figures, receipts, code):
            directory.mkdir(parents=True)
        descriptive.to_csv(tables / "descriptive_metrics.csv", index=False, float_format="%.12g")
        intervals.to_csv(tables / "paired_intervals.csv", index=False, float_format="%.12g")
        _write_json(receipts / "bootstrap_sampling.json", sampling)
        comparison_figure(intervals, figures / "regional_seasonal_improvement")
        _write_results(stage / "RESULTS.md", descriptive, intervals)
        source_code = Path(__file__).resolve()
        shutil.copyfile(source_code, code / source_code.name)
        artifact_hashes = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "scientific_status": "retrospective regional and seasonal sensitivity",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "comparison": {"candidate": METHOD_LABELS[CANDIDATE], "baseline": METHOD_LABELS[BASELINE]},
            "reference": "IMD",
            "regions": REGION_LABELS,
            "seasons": SEASON_LABELS,
            "season_assignment": "midpoint of each exact seven-day valid period",
            "verification_years": [2020, 2021, 2022, 2023, 2024],
            "lead_weeks": list(LEADS),
            "source": {
                "manifest": str(manifest_path),
                "manifest_sha256": SOURCE_MANIFEST_SHA256,
                "case_metrics": str(cases_path),
                "case_metrics_sha256": SOURCE_CASES_SHA256,
                "maximum_target_label_opened": source_manifest["cohort"]["maximum_target_label_opened"],
            },
            "uncertainty": {
                "method": "paired year-stratified circular-block bootstrap of initialization starts",
                "replicates": int(replicates),
                "block_length_starts": int(block_length),
                "base_seed": int(seed),
                "confidence": 0.95,
                "multiplicity_adjustment": "none; intervals are pointwise and descriptive",
            },
            "table_rows": {"descriptive_metrics": len(descriptive), "paired_intervals": len(intervals)},
            "opened_2025_observation": False,
            "sealed_2025_target_opened": False,
            "artifacts": artifact_hashes,
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--block-length", type=int, default=BOOTSTRAP_BLOCK_LENGTH)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    publish(
        source=args.source,
        output=args.output,
        replicates=args.replicates,
        block_length=args.block_length,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
