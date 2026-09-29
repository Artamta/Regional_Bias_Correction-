#!/usr/bin/env python3
"""Publish paired MME-versus-FuXi uncertainty from frozen benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from india_s2s_block_bootstrap import (
    paired_year_stratified_circular_block_bootstrap,
)


EXPERIMENT = "india_s2s_benchmark_uncertainty_v2"
EXPECTED_CASE_TABLE_SHA256 = (
    "000544615d004669b41dd87fc647bd979cd92c3638e7deee5ea77992046e1a14"
)
MODELS = ("mme", "fuxi_s2s")
METRICS = ("acc", "rmse", "mae", "bias")
BLOCK_LENGTHS = (16, 13)


class BenchmarkUncertaintyError(RuntimeError):
    """Raised when the frozen benchmark table does not satisfy its receipt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_rows(table: pd.DataFrame, region: str, lead_week: int) -> pd.DataFrame:
    selected = table[
        (table["reference"] == "imd")
        & (table["season"] == "JJAS")
        & (table["region"] == region)
        & (table["lead_week"] == lead_week)
        & (table["model"].isin(MODELS))
        & (table["score_status"] == "available")
    ].copy()
    if selected.duplicated(["model", "init"]).any():
        raise BenchmarkUncertaintyError(
            f"duplicate model/init rows for {region}, W{lead_week}"
        )
    counts = selected.groupby("model")["init"].nunique().to_dict()
    if counts != {"fuxi_s2s": 169, "mme": 169}:
        raise BenchmarkUncertaintyError(
            f"expected 169 paired JJAS cases for {region}, W{lead_week}: {counts}"
        )
    wide = selected.pivot(index="init", columns="model", values=list(METRICS))
    if len(wide) != 169 or wide.isna().any().any():
        raise BenchmarkUncertaintyError(
            f"incomplete MME/FuXi pairing for {region}, W{lead_week}"
        )
    return wide.sort_index()


def compute_intervals(
    table: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    required = {
        "reference",
        "season",
        "region",
        "lead_week",
        "model",
        "score_status",
        "init",
        *METRICS,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise BenchmarkUncertaintyError(f"case table is missing columns: {missing}")
    regions = sorted(table.loc[table["reference"] == "imd", "region"].unique())
    rows: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}
    for region in regions:
        for lead_week in range(1, 7):
            paired = _paired_rows(table, region, lead_week)
            dates = paired.index.to_numpy(dtype="datetime64[D]")
            for metric in METRICS:
                mme = paired[(metric, "mme")].to_numpy(dtype=np.float64)
                fuxi = paired[(metric, "fuxi_s2s")].to_numpy(dtype=np.float64)
                for block_length in BLOCK_LENGTHS:
                    result, samples = paired_year_stratified_circular_block_bootstrap(
                        dates,
                        mme,
                        fuxi,
                        block_length=block_length,
                        replicates=replicates,
                        seed=seed,
                    )
                    key = f"{region}__w{lead_week}__{metric}__block{block_length}"
                    distributions[key] = samples.astype(np.float32)
                    rows.append(
                        {
                            "reference": "imd",
                            "season": "JJAS_valid_midpoint",
                            "region": region,
                            "lead_week": lead_week,
                            "metric": metric,
                            "candidate": "mme",
                            "baseline": "fuxi_s2s",
                            "effect": "mme_minus_fuxi",
                            "candidate_mean": float(mme.mean(dtype=np.float64)),
                            "baseline_mean": float(fuxi.mean(dtype=np.float64)),
                            "estimate": result.estimate,
                            "ci_lower": result.lower,
                            "ci_upper": result.upper,
                            "confidence": result.confidence,
                            "n_cases": result.paired_case_count,
                            "block_length_starts": block_length,
                            "replicates": replicates,
                            "seed": seed,
                            "year_counts": json.dumps(result.year_counts, sort_keys=True),
                        }
                    )
    return pd.DataFrame(rows), distributions


def publish(
    case_metrics_path: Path,
    output: Path,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> Path:
    source = Path(case_metrics_path).resolve()
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"fresh output path required: {destination}")
    observed_hash = sha256_file(source)
    if observed_hash != EXPECTED_CASE_TABLE_SHA256:
        raise BenchmarkUncertaintyError(
            f"case table SHA-256 differs: {observed_hash}"
        )
    staging = destination.with_name(destination.name + f".incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        table = pd.read_csv(case_metrics_path)
        intervals, distributions = compute_intervals(
            table, replicates=replicates, seed=seed
        )
        table_path = staging / "mme_vs_fuxi_paired_intervals.csv"
        samples_path = staging / "bootstrap_distributions.npz"
        intervals.to_csv(table_path, index=False)
        np.savez_compressed(samples_path, **distributions)
        source_paths = {
            "code/src/india_s2s_benchmark_uncertainty.py": Path(__file__).resolve(),
            "code/src/india_s2s_block_bootstrap.py": Path(__file__).resolve().with_name(
                "india_s2s_block_bootstrap.py"
            ),
        }
        source_hashes: dict[str, str] = {}
        for relative, source_path in source_paths.items():
            source_snapshot = staging / relative
            source_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, source_snapshot)
            source_hashes[relative] = sha256_file(source_snapshot)
        artifact_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_case_metrics": str(source),
            "source_case_metrics_sha256": observed_hash,
            "comparison": "mme_minus_fuxi_s2s",
            "cohort": "IMD valid-midpoint JJAS",
            "case_count_per_lead_region": 169,
            "aggregation": "arithmetic mean of paired per-initialization spatial scores",
            "resampling": "paired year-stratified circular initialization blocks",
            "block_lengths_starts": list(BLOCK_LENGTHS),
            "replicates": replicates,
            "seed": seed,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "source_snapshot_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            (staging / "failure.json").write_text(
                json.dumps({"status": "failed"}, indent=2) + "\n", encoding="utf-8"
            )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    manifest = publish(
        arguments.case_metrics,
        arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
    )
    print(f"PASS: {manifest}")


if __name__ == "__main__":
    main()
