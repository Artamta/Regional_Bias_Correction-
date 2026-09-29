#!/usr/bin/env python3
"""Package the frozen 2022--2024 raw-identity versus anchored audit."""

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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / "studies" / "fuxi_imd_adapter_benchmark_v1"
DEFAULT_ANCHORED_AUDIT = STUDY / "results" / "full_context_jjas_2022_2024_job91439"
DEFAULT_RAW_IDENTITY_AUDIT = (
    ROOT
    / "resultsv2"
    / "fuxi_imd_no_log_bias_ablation"
    / "audit_2022_2024_20260822T012837Z"
)
DEFAULT_BOOTSTRAP_ROOT = (
    STUDY / "results" / "full_context_jjas_2022_2024_ccai_figures_v1" / "tables"
)

YEARS = (2022, 2023, 2024)
YEAR_COUNTS = {2022: 35, 2023: 35, 2024: 30}
LEADS = tuple(range(1, 7))
REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
RAW = "raw_fuxi"
LOG_BIAS = "log_bias"
ANCHORED = "log_bias_anchored_adapter"
RAW_IDENTITY = "raw_identity_adapter"
SOURCE_MODEL = "selected_adapter"
METHODS = (RAW, LOG_BIAS, ANCHORED, RAW_IDENTITY)
METHOD_LABELS = {
    RAW: "Raw FuXi",
    LOG_BIAS: "Training-only log-bias",
    ANCHORED: "Neural + log-bias anchor",
    RAW_IDENTITY: "Raw-identity neural (no log-bias anchor)",
}
METRICS = ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc")
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_818


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_audit(run: Path, *, raw_identity: bool) -> dict[str, Any]:
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"audit is not complete: {run}")
    if tuple(manifest.get("audit_initialization_years", ())) != YEARS:
        raise ValueError(f"audit years differ: {run}")
    if manifest.get("audit_counts") != {
        str(year): count for year, count in YEAR_COUNTS.items()
    }:
        raise ValueError(f"audit year counts differ: {run}")
    if manifest.get("audit_case_count") != sum(YEAR_COUNTS.values()):
        raise ValueError(f"audit case count differs: {run}")
    if manifest.get("final_initialization_year_quarantined") != 2025:
        raise ValueError(f"2025 quarantine is missing: {run}")
    if manifest.get("selected_model") != "normal_climo_model" or not np.isclose(
        float(manifest.get("selected_alpha", np.nan)), 1.0
    ):
        raise ValueError(f"frozen matched selection differs: {run}")
    if raw_identity and (
        manifest.get("training_anchor") != RAW
        or manifest.get("uses_fitted_log_bias_in_neural_training") is not False
        or manifest.get("log_bias_role") != "reporting_only"
    ):
        raise ValueError("raw-identity audit does not isolate fitted log-bias")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping) or "case_metrics.csv" not in outputs:
        raise ValueError(f"audit output hashes are missing: {run}")
    if sha256_file(run / "case_metrics.csv") != outputs["case_metrics.csv"]:
        raise ValueError(f"audit case-metric hash differs: {run}")
    return {
        "path": str(run),
        "manifest_sha256": sha256_file(run / "manifest.json"),
        "case_metrics_sha256": outputs["case_metrics.csv"],
        "adapter_run": manifest["adapter_run"],
        "adapter_manifest_sha256": manifest["adapter_manifest_sha256"],
        "adapter_selection_sha256": manifest["adapter_selection_sha256"],
        "selected_alpha": float(manifest["selected_alpha"]),
        "generalization_guards": manifest["generalization_guards"],
    }


def read_cases(run: Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(run) / "case_metrics.csv")
    required = {
        "method",
        "init",
        "year",
        "lead_week",
        "region",
        "valid_cell_count",
        "effective_area_km2",
        *METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"audit case metrics lack columns: {sorted(missing)}")
    if set(frame.method) != {RAW, LOG_BIAS, SOURCE_MODEL}:
        raise ValueError("audit method set differs")
    if set(frame.year.astype(int)) != set(YEARS):
        raise ValueError("audit case years differ")
    if set(frame.lead_week.astype(int)) != set(LEADS):
        raise ValueError("audit lead coverage differs")
    if set(frame.region) != set(REGIONS):
        raise ValueError("audit region coverage differs")
    for year, count in YEAR_COUNTS.items():
        if frame.loc[frame.year.eq(year), "init"].nunique() != count:
            raise ValueError(f"audit initialization count differs for {year}")
    keys = ["method", "init", "lead_week", "region"]
    if frame.duplicated(keys).any():
        raise ValueError("audit contains duplicate method/case/lead/region rows")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("audit metrics contain non-finite values")
    return frame


def combine_cases(anchored: pd.DataFrame, raw_identity: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "init", "year", "lead_week", "region"]
    comparison_columns = [
        *keys,
        "valid_cell_count",
        "effective_area_km2",
        *METRICS,
    ]
    for method in (RAW, LOG_BIAS):
        left = (
            anchored.loc[anchored.method.eq(method), comparison_columns]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        right = (
            raw_identity.loc[raw_identity.method.eq(method), comparison_columns]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    pieces = []
    for source, source_method, method in (
        (anchored, RAW, RAW),
        (anchored, LOG_BIAS, LOG_BIAS),
        (anchored, SOURCE_MODEL, ANCHORED),
        (raw_identity, SOURCE_MODEL, RAW_IDENTITY),
    ):
        selected = source.loc[source.method.eq(source_method)].copy()
        selected["method"] = method
        selected["method_label"] = METHOD_LABELS[method]
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def absolute_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append(frame: pd.DataFrame, scope_type: str, scope: str) -> None:
        for method in METHODS:
            selected = frame.loc[frame.method.eq(method)]
            for metric in METRICS:
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "metric": metric,
                        "value": float(selected[metric].mean()),
                        "n_starts": int(selected.init.nunique()),
                        "n_case_leads": int(len(selected)),
                    }
                )

    india = cases.loc[cases.region.eq("all_india")]
    append(india, "pooled", "W1-W6")
    for year in YEARS:
        append(india.loc[india.year.eq(year)], "year", str(year))
    for lead in LEADS:
        append(india.loc[india.lead_week.eq(lead)], "lead", f"W{lead}")
    for region in REGIONS[1:]:
        append(cases.loc[cases.region.eq(region)], "region", region)
    return pd.DataFrame(rows)


def metric_cube(cases: pd.DataFrame, *, region: str, metric: str) -> np.ndarray:
    dates = sorted(cases.init.unique())
    cube = np.empty((2, len(dates), len(LEADS)), dtype=np.float64)
    for index, method in enumerate((ANCHORED, RAW_IDENTITY)):
        pivot = cases.loc[cases.region.eq(region) & cases.method.eq(method)].pivot(
            index="init", columns="lead_week", values=metric
        )
        cube[index] = pivot.loc[dates, list(LEADS)].to_numpy(dtype=np.float64)
    if not np.isfinite(cube).all():
        raise ValueError(f"paired metric cube is incomplete: {region}/{metric}")
    return cube


def summarize_effect(
    candidate: np.ndarray,
    reference: np.ndarray,
    sample_indices: np.ndarray,
    *,
    metric: str,
) -> tuple[float, float, float, str]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("paired arrays must be [initialization, lead]")
    candidate_draw = candidate[sample_indices].mean(axis=(1, 2))
    reference_draw = reference[sample_indices].mean(axis=(1, 2))
    if metric in {"rmse_mm_day", "mae_mm_day"}:
        point = 100.0 * (reference.mean() - candidate.mean()) / reference.mean()
        draws = 100.0 * (reference_draw - candidate_draw) / reference_draw
        units = "percent reduction relative to anchored adapter"
    else:
        point = candidate.mean() - reference.mean()
        draws = candidate_draw - reference_draw
        units = "raw-identity minus anchored"
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(point), float(lower), float(upper), units


def paired_effects(
    cases: pd.DataFrame,
    two_stage: np.ndarray,
    within_year: np.ndarray,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(cases.init.unique()))
    if tuple(dates.year.value_counts().sort_index().items()) != tuple(
        YEAR_COUNTS.items()
    ):
        raise ValueError("paired audit initialization contract differs")
    if two_stage.shape != (BOOTSTRAP_DRAWS, len(dates)):
        raise ValueError("two-stage bootstrap index shape differs")
    if within_year.shape != two_stage.shape:
        raise ValueError("within-year bootstrap index shape differs")
    if np.any(two_stage < 0) or np.any(two_stage >= len(dates)):
        raise ValueError("two-stage bootstrap index is outside the audit")
    if np.any(within_year < 0) or np.any(within_year >= len(dates)):
        raise ValueError("within-year bootstrap index is outside the audit")

    rows: list[dict[str, Any]] = []

    def append(
        *,
        scope_type: str,
        scope: str,
        region: str,
        lead_indices: Sequence[int],
        positions: np.ndarray,
        sampled: np.ndarray,
    ) -> None:
        remap = np.full(len(dates), -1, dtype=np.int64)
        remap[positions] = np.arange(len(positions))
        local_sample = remap[sampled]
        if np.any(local_sample < 0):
            raise ValueError(f"bootstrap remapping failed for {scope_type}/{scope}")
        for metric in METRICS:
            cube = metric_cube(cases, region=region, metric=metric)
            reference = cube[0, positions][:, lead_indices]
            candidate = cube[1, positions][:, lead_indices]
            effect, lower, upper, units = summarize_effect(
                candidate,
                reference,
                local_sample,
                metric=metric,
            )
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "region": region,
                    "metric": metric,
                    "raw_identity_candidate_skill": effect,
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "effect_units": units,
                    "positive_direction": (
                        "positive favors raw identity for RMSE/MAE/ACC; "
                        "bias is a signed shift, not a skill score"
                    ),
                    "n_starts": len(positions),
                    "n_case_leads": len(positions) * len(lead_indices),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "block_length": BOOTSTRAP_BLOCK_LENGTH,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "interval_excludes_zero_descriptively": bool(
                        lower > 0.0 or upper < 0.0
                    ),
                }
            )

    all_positions = np.arange(len(dates), dtype=np.int64)
    append(
        scope_type="pooled",
        scope="W1-W6",
        region="all_india",
        lead_indices=np.arange(6),
        positions=all_positions,
        sampled=two_stage,
    )
    for lead_index, lead in enumerate(LEADS):
        append(
            scope_type="lead",
            scope=f"W{lead}",
            region="all_india",
            lead_indices=[lead_index],
            positions=all_positions,
            sampled=two_stage,
        )
    cursor = 0
    for year, count in YEAR_COUNTS.items():
        positions = np.arange(cursor, cursor + count, dtype=np.int64)
        append(
            scope_type="year",
            scope=str(year),
            region="all_india",
            lead_indices=np.arange(6),
            positions=positions,
            sampled=within_year[:, positions],
        )
        cursor += count
    for region in REGIONS[1:]:
        append(
            scope_type="region",
            scope=region,
            region=region,
            lead_indices=np.arange(6),
            positions=all_positions,
            sampled=two_stage,
        )
    return pd.DataFrame(rows)


def lookup(
    table: pd.DataFrame,
    *,
    scope_type: str,
    scope: str,
    method: str,
    metric: str,
) -> float:
    selected = table.loc[
        table.scope_type.eq(scope_type)
        & table.scope.eq(scope)
        & table.method.eq(method)
        & table.metric.eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError("ambiguous audit absolute-metric lookup")
    return float(selected.value.iloc[0])


def effect_lookup(
    table: pd.DataFrame, *, scope_type: str, scope: str, metric: str
) -> pd.Series:
    selected = table.loc[
        table.scope_type.eq(scope_type)
        & table.scope.eq(scope)
        & table.metric.eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError("ambiguous audit paired-effect lookup")
    return selected.iloc[0]


def report(absolute: pd.DataFrame, effects: pd.DataFrame) -> str:
    rows = []
    for method in METHODS:
        rows.append(
            "| {label} | {rmse:.3f} | {mae:.3f} | {acc:.3f} | {bias:+.3f} |".format(
                label=METHOD_LABELS[method],
                rmse=lookup(
                    absolute,
                    scope_type="pooled",
                    scope="W1-W6",
                    method=method,
                    metric="rmse_mm_day",
                ),
                mae=lookup(
                    absolute,
                    scope_type="pooled",
                    scope="W1-W6",
                    method=method,
                    metric="mae_mm_day",
                ),
                acc=lookup(
                    absolute,
                    scope_type="pooled",
                    scope="W1-W6",
                    method=method,
                    metric="acc",
                ),
                bias=lookup(
                    absolute,
                    scope_type="pooled",
                    scope="W1-W6",
                    method=method,
                    metric="bias_mm_day",
                ),
            )
        )
    rmse = effect_lookup(
        effects, scope_type="pooled", scope="W1-W6", metric="rmse_mm_day"
    )
    mae = effect_lookup(
        effects, scope_type="pooled", scope="W1-W6", metric="mae_mm_day"
    )
    acc = effect_lookup(effects, scope_type="pooled", scope="W1-W6", metric="acc")
    year_rows = []
    for year in YEARS:
        year_rows.append(
            "| {year} | {anchor:.3f} | {raw:.3f} | {skill:+.3f}% "
            "[{lower:+.3f}%, {upper:+.3f}%] |".format(
                year=year,
                anchor=lookup(
                    absolute,
                    scope_type="year",
                    scope=str(year),
                    method=ANCHORED,
                    metric="rmse_mm_day",
                ),
                raw=lookup(
                    absolute,
                    scope_type="year",
                    scope=str(year),
                    method=RAW_IDENTITY,
                    metric="rmse_mm_day",
                ),
                skill=effect_lookup(
                    effects,
                    scope_type="year",
                    scope=str(year),
                    metric="rmse_mm_day",
                ).raw_identity_candidate_skill,
                lower=effect_lookup(
                    effects,
                    scope_type="year",
                    scope=str(year),
                    metric="rmse_mm_day",
                ).interval_lower,
                upper=effect_lookup(
                    effects,
                    scope_type="year",
                    scope=str(year),
                    metric="rmse_mm_day",
                ).interval_upper,
            )
        )
    return f"""# Raw-identity adapter: frozen 2022–2024 development audit

> Post-hoc exploratory development evidence. The anchored audit outcomes were already known before this no-anchor audit; this is not independent confirmation. The frozen model and α were not changed, and 2025 remains untouched.

## Pooled all-India W1–W6

| Method | RMSE | MAE | Common-IMD ACC | Signed bias |
|---|---:|---:|---:|---:|
{os.linesep.join(rows)}

Against the matched log-bias-anchored neural adapter, raw identity has **{rmse.raw_identity_candidate_skill:+.3f}% RMSE candidate skill** [{rmse.interval_lower:+.3f}%, {rmse.interval_upper:+.3f}%], **{mae.raw_identity_candidate_skill:+.3f}% MAE candidate skill** [{mae.interval_lower:+.3f}%, {mae.interval_upper:+.3f}%], and common-IMD ACC difference **{acc.raw_identity_candidate_skill:+.4f}** [{acc.interval_lower:+.4f}, {acc.interval_upper:+.4f}].

## Year stability

| Year | Anchored RMSE | Raw-identity RMSE | Raw-identity candidate skill |
|---|---:|---:|---:|
{os.linesep.join(year_rows)}

All raw-identity audit guards passed: lower RMSE than raw FuXi at every lead and region, non-worse ACC than raw FuXi at every lead and region, and lower RMSE than fitted log-bias in all three years.

Intervals reuse the saved CCAI audit resamples: 2,000 paired draws, RNG seed {BOOTSTRAP_SEED}, circular block length {BOOTSTRAP_BLOCK_LENGTH}. Pooled, lead, and region intervals use the two-stage year-resampling design; year intervals use within-year circular blocks. They are descriptive and conditional on the realized three-seed ensembles, not p-values or final-test evidence.
"""


def package(
    anchored_audit: Path,
    raw_identity_audit: Path,
    bootstrap_root: Path,
    output: Path,
) -> Path:
    anchored_audit = Path(anchored_audit).resolve()
    raw_identity_audit = Path(raw_identity_audit).resolve()
    bootstrap_root = Path(bootstrap_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    anchored_source = validate_audit(anchored_audit, raw_identity=False)
    raw_source = validate_audit(raw_identity_audit, raw_identity=True)
    cases = combine_cases(read_cases(anchored_audit), read_cases(raw_identity_audit))
    two_stage_path = bootstrap_root / "bootstrap_indices_two_stage.npy"
    within_year_path = bootstrap_root / "bootstrap_indices_within_year.npy"
    two_stage = np.load(two_stage_path, allow_pickle=False)
    within_year = np.load(within_year_path, allow_pickle=False)
    absolute = absolute_metrics(cases)
    effects = paired_effects(cases, two_stage, within_year)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        metrics = temporary / "metrics"
        metrics.mkdir()
        absolute.to_csv(metrics / "absolute_audit_metrics.csv", index=False)
        effects.to_csv(metrics / "paired_audit_effects.csv", index=False)
        (temporary / "REPORT.md").write_text(
            report(absolute, effects), encoding="utf-8"
        )
        artifacts = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_name": "fuxi_imd_no_log_bias_2022_2024_audit_comparison",
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": (
                "post-hoc exploratory development audit; not independent confirmation"
            ),
            "audit_years": list(YEARS),
            "audit_counts": {str(year): count for year, count in YEAR_COUNTS.items()},
            "final_2025_accessed": False,
            "primary_comparison": {
                "candidate": RAW_IDENTITY,
                "reference": ANCHORED,
                "positive_direction": "positive favors raw identity except signed bias",
            },
            "uncertainty": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "circular_block_length_starts": BOOTSTRAP_BLOCK_LENGTH,
                "pooled_design": "resample years, then circular blocks within source year",
                "year_design": "paired circular blocks within year",
                "p_values_computed": False,
                "significance_claimed": False,
            },
            "identity_checks": {
                "raw_and_log_bias_case_metrics_exact_all_regions": True,
            },
            "sources": {
                "anchored_audit": anchored_source,
                "raw_identity_audit": raw_source,
                "two_stage_indices": {
                    "path": str(two_stage_path),
                    "sha256": sha256_file(two_stage_path),
                },
                "within_year_indices": {
                    "path": str(within_year_path),
                    "sha256": sha256_file(within_year_path),
                },
                "postprocessor": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "raw_identity_generalization_guards": raw_source["generalization_guards"],
            "artifacts": dict(sorted(artifacts.items())),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchored-audit", type=Path, default=DEFAULT_ANCHORED_AUDIT)
    parser.add_argument(
        "--raw-identity-audit", type=Path, default=DEFAULT_RAW_IDENTITY_AUDIT
    )
    parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = package(
        args.anchored_audit,
        args.raw_identity_audit,
        args.bootstrap_root,
        args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
