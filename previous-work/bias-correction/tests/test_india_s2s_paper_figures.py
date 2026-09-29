from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from india_s2s_paper_figures import (
    CASE_COUNT_PER_LEAD,
    CASE_ROWS,
    FIXED_MME_COMPONENTS,
    InputHashes,
    FigureInputs,
    SpatialArrayHashes,
    LEADS,
    METHODS,
    MODELS,
    PaperFigureError,
    RAW_IDENTITY_ROWS,
    REGIONS,
    SELECTED_CHECKPOINT_SHA256,
    STORY_COHORT,
    T2M_MME_COMPONENTS,
    T2M_MODELS,
    _annotation_text_color,
    array_sha256,
    generate_figures,
    sha256_file,
)


def test_heatmap_annotation_color_tracks_rendered_cell_luminance() -> None:
    diverging = ScalarMappable(norm=Normalize(-0.045, 0.045), cmap="PuOr_r")
    sequential = ScalarMappable(norm=Normalize(0.015, 0.63), cmap="viridis")

    assert _annotation_text_color(diverging, -0.04) == "white"
    assert _annotation_text_color(diverging, 0.04) == "white"
    assert _annotation_text_color(diverging, 0.0) == "#141414"
    assert _annotation_text_color(sequential, 0.05) == "white"
    assert _annotation_text_color(sequential, 0.50) == "#141414"


def _json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inventory(root: Path, receipt_name: str) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != receipt_name
    }


def _methods(track: str, reference: str, output_group: str) -> dict[str, object]:
    result: dict[str, object] = {
        "output_group": output_group,
        "included_tracks": [track],
        "methods": {
            "season_assignment": (
                "season of the exact seven-day valid-window midpoint "
                "(init + 7*lead_week - 3.5 days)"
            )
        },
        "tracks": {
            track: {
                "reference": reference,
                "models": list(MODELS[:-1]),
                "mme_components": list(FIXED_MME_COMPONENTS),
                "unavailable_lead_weeks": {"ecmwf": [3]},
                "total_paired_initializations": 517,
            }
        },
    }
    if output_group == "imerg_era5_primary":
        result["included_tracks"] = [track, "t2m_era5"]
        result["tracks"]["t2m_era5"] = {
            "track": "t2m_era5",
            "variable": "t2m",
            "reference": "era5",
            "models": list(T2M_MODELS[:-1]),
            "mme_components": list(T2M_MME_COMPONENTS),
            "unavailable_lead_weeks": {},
            "total_paired_initializations": 516,
        }
    return result


def _seasonal(track: str, reference: str, reference_delta: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(REGIONS):
        for model_index, model in enumerate(MODELS):
            for lead in LEADS:
                missing = model == "ecmwf" and lead == 3
                acc = 0.68 - 0.075 * lead - 0.008 * region_index - 0.006 * model_index
                rmse = 3.8 + 0.35 * lead + 0.06 * region_index + 0.08 * model_index
                rows.append(
                    {
                        "track": track,
                        "variable": "tp",
                        "reference": reference,
                        "season": "JJAS",
                        "region": region,
                        "region_label": region.replace("_", " ").title(),
                        "model": model,
                        "lead_week": lead,
                        "case_count": CASE_COUNT_PER_LEAD,
                        "acc_valid_case_count": 0 if missing else CASE_COUNT_PER_LEAD,
                        "acc": np.nan if missing else acc + reference_delta,
                        "rmse_valid_case_count": 0 if missing else CASE_COUNT_PER_LEAD,
                        "rmse": np.nan if missing else rmse + reference_delta,
                    }
                )
    return pd.DataFrame(rows)


def _weekly(
    track: str,
    variable: str,
    reference: str,
    models: tuple[str, ...],
    case_count: int,
    unavailable: set[tuple[str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(REGIONS):
        for model_index, model in enumerate(models):
            for lead in LEADS:
                missing = (model, lead) in unavailable
                rows.append(
                    {
                        "track": track,
                        "variable": variable,
                        "reference": reference,
                        "region": region,
                        "region_label": region.replace("_", " ").title(),
                        "model": model,
                        "lead_week": lead,
                        "case_count": case_count,
                        "acc_valid_case_count": 0 if missing else case_count,
                        "acc": (
                            np.nan
                            if missing
                            else 0.68 - 0.075 * lead - 0.008 * region_index - 0.006 * model_index
                        ),
                        "rmse_valid_case_count": 0 if missing else case_count,
                        "rmse": (
                            np.nan
                            if missing
                            else 3.8 + 0.35 * lead + 0.06 * region_index + 0.08 * model_index
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _case_metrics(track: str, reference: str) -> pd.DataFrame:
    dates = pd.date_range("2020-06-01", "2024-09-30", freq="D")
    dates = dates[dates.month.isin((6, 7, 8, 9))][:CASE_COUNT_PER_LEAD]
    rows: list[dict[str, object]] = []
    for model in ("mme", "fuxi_s2s"):
        for region in REGIONS:
            for lead in LEADS:
                for date in dates:
                    rows.append(
                        {
                            "track": track,
                            "variable": "tp",
                            "reference": reference,
                            "season": "JJAS",
                            "region": region,
                            "model": model,
                            "init": date.strftime("%Y-%m-%d"),
                            "valid_period_midpoint": date.strftime("%Y-%m-%dT12:00:00"),
                            "lead_week": lead,
                            "score_status": "available",
                        }
                    )
    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_adjustment = {
        "raw_fuxi": (0.0, 0.0, 0.0, 0.0),
        "moment_calibration": (-0.35, 0.005, 0.18, 0.28),
        "location_spread": (-0.55, 0.03, 0.28, 0.48),
    }
    for method in METHODS:
        for region in REGIONS:
            for lead in LEADS:
                crps, acc, coverage, spread = method_adjustment[method]
                rows.append(
                    {
                        "method": method,
                        "region": region,
                        "lead_week": lead,
                        "case_count": CASE_COUNT_PER_LEAD,
                        "crps": 3.0 + 0.1 * lead + crps,
                        "acc": 0.67 - 0.08 * lead + acc,
                        "coverage90": 0.2 + 0.08 * lead + coverage,
                        "pooled_spread_skill_ratio": 0.1 + 0.1 * lead + spread,
                    }
                )
    return pd.DataFrame(rows)


INTERVAL_COLUMNS = [
    "reference",
    "season",
    "analysis_cohort",
    "region",
    "lead_week",
    "metric",
    "candidate",
    "baseline",
    "candidate_mean",
    "baseline_mean",
    "n_cases",
    "confidence",
    "block_length_starts",
    "replicates",
    "seed",
    "effect",
    "estimate",
    "ci_lower",
    "ci_upper",
]


def _intervals() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nonstory: list[dict[str, object]] = []
    for lead in LEADS:
        nonstory.extend(
            [
                {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                    "region": "all_india",
                    "lead_week": lead,
                    "metric": "crps",
                    "candidate": "location_spread",
                    "baseline": "raw_fuxi",
                    "candidate_mean": 2.5,
                    "baseline_mean": 3.0,
                    "n_cases": CASE_COUNT_PER_LEAD,
                    "confidence": 0.95,
                    "block_length_starts": 16,
                    "replicates": 10_000,
                    "seed": 1_000 + lead,
                    "effect": "skill_pct_vs_baseline",
                    "estimate": 15.0 - lead,
                    "ci_lower": 13.0 - lead,
                    "ci_upper": 17.0 - lead,
                },
                {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                    "region": "all_india",
                    "lead_week": lead,
                    "metric": "acc",
                    "candidate": "location_spread",
                    "baseline": "raw_fuxi",
                    "candidate_mean": 0.3,
                    "baseline_mean": 0.27,
                    "n_cases": CASE_COUNT_PER_LEAD,
                    "confidence": 0.95,
                    "block_length_starts": 16,
                    "replicates": 10_000,
                    "seed": 1_000 + lead,
                    "effect": "candidate_minus_baseline",
                    "estimate": 0.03,
                    "ci_lower": 0.01,
                    "ci_upper": 0.05,
                },
            ]
        )
    story_values = {
        "crps_skill_pct_vs_raw": (16.2, 14.8, 17.7),
        "crps_skill_pct_vs_moment": (4.3, 3.0, 5.7),
        "acc_delta_vs_raw": (0.026, 0.009, 0.044),
        "bias_delta_vs_raw": (0.020, -0.207, 0.246),
    }
    story_specs = (
        ("crps", "raw_fuxi", "candidate_minus_baseline", -0.5, -0.6, -0.4),
        ("crps", "raw_fuxi", "skill_pct_vs_baseline", *story_values["crps_skill_pct_vs_raw"]),
        ("acc", "raw_fuxi", "candidate_minus_baseline", *story_values["acc_delta_vs_raw"]),
        ("bias", "raw_fuxi", "candidate_minus_baseline", *story_values["bias_delta_vs_raw"]),
        ("crps", "moment_calibration", "candidate_minus_baseline", -0.1, -0.15, -0.06),
        (
            "crps",
            "moment_calibration",
            "skill_pct_vs_baseline",
            *story_values["crps_skill_pct_vs_moment"],
        ),
        ("acc", "moment_calibration", "candidate_minus_baseline", 0.025, 0.005, 0.045),
        ("bias", "moment_calibration", "candidate_minus_baseline", -0.4, -0.5, -0.3),
    )
    story_rows: list[dict[str, object]] = []
    for block in (16, 13):
        for metric, baseline, effect, estimate, lower, upper in story_specs:
            story_rows.append(
                {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": STORY_COHORT,
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline,
                    "candidate_mean": 2.6 if metric == "crps" else 0.3,
                    "baseline_mean": 3.1 if metric == "crps" else 0.27,
                    "n_cases": np.nan,
                    "confidence": 0.95,
                    "block_length_starts": block,
                    "replicates": 10_000,
                    "seed": 9_000 + block,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "lead_aggregation": "synchronized within-year ordinal draws",
                    "n_cases_per_lead": 100,
                    "n_case_lead_rows": 600,
                    "lead_date_intersection_count": 70,
                    "lead_date_union_count": 130,
                }
            )
    scoring = pd.DataFrame(nonstory + story_rows)
    audit_story = pd.DataFrame(story_rows)
    story = {
        "analysis_cohort": STORY_COHORT,
        "cases_per_lead": 100,
        "case_lead_rows": 600,
        "date_intersection_count": 70,
        "date_union_count": 130,
        "primary_block_length_starts": 16,
        "retrospective_2022_2024_component_passes": True,
        "final_headline_gate": "pending_sealed_2025_point-estimate_direction_check",
        "sealed_2025_target_opened": False,
        **{
            key: {"estimate": values[0], "ci_lower": values[1], "ci_upper": values[2]}
            for key, values in story_values.items()
        },
    }
    return scoring, audit_story, story


def _comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows: list[dict[str, object]] = []
    exact_specs = (
        ("crps", "raw_fuxi", "candidate_minus_baseline", -0.50, -0.56, -0.44),
        ("crps", "raw_fuxi", "skill_pct_vs_baseline", 15.86, 14.35, 17.52),
        ("acc", "raw_fuxi", "candidate_minus_baseline", 0.031, 0.012, 0.051),
        ("bias", "raw_fuxi", "candidate_minus_baseline", 0.04, -0.20, 0.28),
        ("crps", "moment_calibration", "candidate_minus_baseline", -0.11, -0.15, -0.07),
        ("crps", "moment_calibration", "skill_pct_vs_baseline", 4.06, 2.61, 5.71),
        ("acc", "moment_calibration", "candidate_minus_baseline", 0.031, 0.010, 0.051),
        ("bias", "moment_calibration", "candidate_minus_baseline", -0.41, -0.49, -0.32),
    )
    for block in (16, 13):
        for metric, baseline, effect, estimate, lower, upper in exact_specs:
            exact_rows.append(
                {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": "2022_2024_exact_common_valid_midpoints_pooled_w1_w6",
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline,
                    "candidate_mean": 2.6,
                    "baseline_mean": 3.1,
                    "n_common_midpoints_per_lead": 85,
                    "n_case_lead_rows": 510,
                    "confidence": 0.95,
                    "block_length_valid_midpoints": block,
                    "replicates": 10_000,
                    "seed": 17_000 + block,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    mme_rows: list[dict[str, object]] = []
    for block in (16, 13):
        for lead in LEADS:
            acc = -0.001 if lead == 2 else 0.03 + 0.005 * lead
            acc_lower = -0.02 if lead == 2 else 0.01
            for metric, effect, estimate, lower, upper in (
                ("acc", "candidate_minus_baseline", acc, acc_lower, 0.06),
                ("rmse", "candidate_minus_baseline", -0.3, -0.45, -0.12),
                ("rmse", "skill_pct_vs_baseline", 5.0, 2.0, 8.0),
                ("mae", "candidate_minus_baseline", -0.2, -0.3, -0.1),
                ("mae", "skill_pct_vs_baseline", 4.0, 1.0, 7.0),
                ("bias", "candidate_minus_baseline", 0.1, -0.1, 0.3),
            ):
                mme_rows.append(
                    {
                        "reference": "imd",
                        "season": "JJAS_valid_midpoint",
                        "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                        "region": "all_india",
                        "lead_week": lead,
                        "metric": metric,
                        "candidate": "location_spread",
                        "baseline": "equal_system_mme_no_ecmwf",
                        "candidate_mean": 4.5,
                        "baseline_mean": 4.8,
                        "n_cases": CASE_COUNT_PER_LEAD,
                        "confidence": 0.95,
                        "block_length_starts": block,
                        "replicates": 10_000,
                        "seed": 23_000 + lead + block,
                        "effect": effect,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
    return pd.DataFrame(exact_rows), pd.DataFrame(mme_rows)


def _frozen_geometry(
    tmp_path: Path,
) -> tuple[Path, Path, Path, SpatialArrayHashes]:
    import xarray as xr

    latitude = np.linspace(39.0, 0.0, 27, dtype=np.float64)
    longitude = np.linspace(60.0, 99.0, 27, dtype=np.float64)
    india = np.zeros((27, 27), dtype=bool)
    india.reshape(-1)[:174] = True
    support = india.copy()
    support.reshape(-1)[171:174] = False
    fractions: dict[str, np.ndarray] = {}
    for region_index, region in enumerate(REGIONS[1:]):
        values = np.zeros((27, 27), dtype=np.float32)
        positions = np.flatnonzero(india.reshape(-1))[region_index::4]
        values.reshape(-1)[positions] = 1.0
        fractions[region] = values
    india_area = india.astype(np.float64) * 25_000.0

    calibration_root = tmp_path / "calibration"
    (calibration_root / "evaluation").mkdir(parents=True)
    adapter_support = calibration_root / "evaluation/scoring_support.npz"
    np.savez(
        adapter_support,
        latitude=latitude,
        longitude=longitude,
        observation_fraction=support.astype(np.float32),
        support_mask=support,
        scoring_weight_km2_fraction=support.astype(np.float64) * 25_000.0,
    )

    spatial_store = tmp_path / "spatial_support.zarr"
    dataset = xr.Dataset(
        data_vars={
            "india_area_weight_km2": (("latitude", "longitude"), india_area),
            **{
                f"{region}_fraction": (("latitude", "longitude"), values)
                for region, values in fractions.items()
            },
        },
        coords={"latitude": latitude, "longitude": longitude},
    )
    dataset.to_zarr(spatial_store, mode="w", consolidated=True)
    spatial_metadata = spatial_store / ".zmetadata"

    calibration_manifest = calibration_root / "manifest.json"
    _json(
        calibration_manifest,
        {
            "experiment": "fuxi_allseason_ensemble_calibration_v2_aligned",
            "status": "complete",
            "evaluation": {
                "scoring_support_artifact": "evaluation/scoring_support.npz",
                "scoring_support_sha256": sha256_file(adapter_support),
                "support_cells": 171,
                "spatial_area_source": str(spatial_store.resolve()),
            },
            "artifact_sha256": {
                "evaluation/scoring_support.npz": sha256_file(adapter_support)
            },
        },
    )
    arrays = {
        "latitude": latitude,
        "longitude": longitude,
        "india_area_weight_km2": india_area,
        **{f"{region}_fraction": values for region, values in fractions.items()},
    }
    spatial_hashes = SpatialArrayHashes(
        latitude=array_sha256(arrays["latitude"], "<f8"),
        longitude=array_sha256(arrays["longitude"], "<f8"),
        india_area_weight_km2=array_sha256(arrays["india_area_weight_km2"], "<f8"),
        northwest_india_fraction=array_sha256(
            arrays["northwest_india_fraction"], "<f4"
        ),
        central_india_fraction=array_sha256(arrays["central_india_fraction"], "<f4"),
        south_peninsula_fraction=array_sha256(
            arrays["south_peninsula_fraction"], "<f4"
        ),
        east_northeast_india_fraction=array_sha256(
            arrays["east_northeast_india_fraction"], "<f4"
        ),
    )
    return calibration_manifest, adapter_support, spatial_metadata, spatial_hashes


def _frozen_fixture(
    tmp_path: Path,
) -> tuple[FigureInputs, InputHashes, SpatialArrayHashes]:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    imd_seasonal = frozen / "imd_seasonal.csv"
    imd_weekly = frozen / "imd_weekly.csv"
    imerg_seasonal = frozen / "imerg_seasonal.csv"
    imerg_weekly = frozen / "imerg_weekly.csv"
    imd_cases = frozen / "imd_cases.csv"
    imerg_cases = frozen / "imerg_cases.csv"
    imd_methods = frozen / "imd_methods.json"
    imerg_methods = frozen / "imerg_methods.json"
    _seasonal("tp_imd", "imd", 0.0).to_csv(imd_seasonal, index=False)
    _seasonal("tp_imerg", "imerg", -0.01).to_csv(imerg_seasonal, index=False)
    _weekly("tp_imd", "tp", "imd", MODELS, 517, {("ecmwf", 3)}).to_csv(
        imd_weekly, index=False
    )
    pd.concat(
        (
            _weekly("tp_imerg", "tp", "imerg", MODELS, 517, {("ecmwf", 3)}),
            _weekly("t2m_era5", "t2m", "era5", T2M_MODELS, 516, set()),
        ),
        ignore_index=True,
    ).to_csv(imerg_weekly, index=False)
    _case_metrics("tp_imd", "imd").to_csv(imd_cases, index=False)
    _case_metrics("tp_imerg", "imerg").to_csv(imerg_cases, index=False)
    _json(imd_methods, _methods("tp_imd", "imd", "imd_tp_sensitivity"))
    _json(imerg_methods, _methods("tp_imerg", "imerg", "imerg_era5_primary"))
    calibration_manifest, adapter_support, spatial_metadata, spatial_hashes = (
        _frozen_geometry(tmp_path)
    )
    calibration_sha = sha256_file(calibration_manifest)

    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    pd.DataFrame({"effect": [0.1]}).to_csv(benchmark_root / "intervals.csv", index=False)
    benchmark = {
        "experiment": "india_s2s_benchmark_uncertainty_v2",
        "status": "complete",
        "cohort": "IMD valid-midpoint JJAS",
        "case_count_per_lead_region": CASE_COUNT_PER_LEAD,
        "source_case_metrics": str(imd_cases.resolve()),
        "source_case_metrics_sha256": sha256_file(imd_cases),
        "artifact_sha256": _inventory(benchmark_root, "manifest.json"),
    }
    benchmark_manifest = benchmark_root / "manifest.json"
    _json(benchmark_manifest, benchmark)

    scoring_root = tmp_path / "scoring"
    (scoring_root / "tables").mkdir(parents=True)
    summary = _summary()
    intervals, story_intervals, story = _intervals()
    summary.to_csv(scoring_root / "tables/jjas_valid_midpoint_summary.csv", index=False)
    intervals.to_csv(scoring_root / "tables/paired_block_intervals.csv", index=False)
    pd.DataFrame(
        {
            "method": ["raw_fuxi"],
            "init": ["2020-01-01"],
            "lead_week": [1],
            "region": ["all_india"],
            "crps": [1.0],
            "acc": [0.5],
        }
    ).to_csv(scoring_root / "tables/case_metrics.csv", index=False)
    scoring = {
        "experiment": "india_s2s_probabilistic_bridge_scoring_v1",
        "status": "complete",
        "cohort": {
            "valid_midpoint_jjas_count_per_lead": CASE_COUNT_PER_LEAD,
            "maximum_target_label_opened": "2024-12-30",
        },
        "opened_observation_years": [2020, 2021, 2022, 2023, 2024],
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
        "selected_seed": 43,
        "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "parent_manifest": str(calibration_manifest.resolve()),
        "parent_manifest_sha256": calibration_sha,
        "preflight_parent_receipt": {
            "selected_seed": 43,
            "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
            "sealed_2025_target_opened": False,
            "parent_manifest": str(calibration_manifest.resolve()),
            "parent_manifest_sha256": calibration_sha,
        },
        "raw_fuxi_identity": {
            "identity": True,
            "complete_bridge_contract": True,
            "matched_case_rows": RAW_IDENTITY_ROWS,
        },
        "table_rows": {"case_metrics": CASE_ROWS},
        "truth_access": {
            "opened_2025": False,
            "sealed_2025_target_opened": False,
            "maximum_target_label": "2024-12-30",
            "spatial_support_store": str(spatial_metadata.resolve().parent),
            "spatial_support_zmetadata_sha256": sha256_file(spatial_metadata),
        },
        "artifact_sha256": _inventory(scoring_root, "manifest.json"),
    }
    scoring_manifest = scoring_root / "manifest.json"
    _json(scoring_manifest, scoring)
    scoring_sha = sha256_file(scoring_manifest)

    audit_root = tmp_path / "audit"
    (audit_root / "reconstructed").mkdir(parents=True)
    summary.to_csv(audit_root / "reconstructed/jjas_valid_midpoint_summary.csv", index=False)
    pd.DataFrame(intervals.iloc[: 2 * len(LEADS)][INTERVAL_COLUMNS]).to_csv(
        audit_root / "reconstructed/paired_block_intervals.csv", index=False
    )
    story_intervals.to_csv(audit_root / "reconstructed/story_gate_intervals.csv", index=False)
    audit = {
        "experiment": "india_s2s_probabilistic_bridge_score_audit_v1",
        "status": "passed",
        "scoring_manifest": str(scoring_manifest.resolve()),
        "scoring_manifest_sha256": scoring_sha,
        "input_hashes_reverified_after_semantic_audit": True,
        "semantic_checks": {
            "selected_seed": 43,
            "raw_identity_rows": RAW_IDENTITY_ROWS,
            "full_case_cartesian_rows": CASE_ROWS,
            "sealed_2025_target_opened": False,
            "forecast_or_truth_arrays_opened_by_auditor": False,
            "valid_midpoint_jjas_summary_reconstructed": True,
            "synchronized_pooled_story_gate_reconstructed": True,
        },
        "story_gate": story,
        "artifact_sha256": _inventory(audit_root, "audit_receipt.json"),
    }
    audit_receipt = audit_root / "audit_receipt.json"
    _json(audit_receipt, audit)
    audit_sha = sha256_file(audit_receipt)

    comparison_root = tmp_path / "comparison"
    (comparison_root / "tables").mkdir(parents=True)
    exact, mme = _comparison_tables()
    exact.to_csv(comparison_root / "tables/exact_common_midpoint_sensitivity.csv", index=False)
    mme.to_csv(comparison_root / "tables/calibrated_fuxi_vs_mme.csv", index=False)
    comparison = {
        "experiment": "india_s2s_paper_comparisons_v1",
        "status": "complete",
        "inputs": {
            "scoring_manifest_sha256": scoring_sha,
            "audit_receipt_sha256": audit_sha,
            "benchmark_cases_sha256": sha256_file(imd_cases),
            "methods_manifest": str(imd_methods.resolve()),
            "methods_manifest_sha256": sha256_file(imd_methods),
        },
        "scoring_safety_contract": {
            "valid_midpoint_jjas_count_per_lead": CASE_COUNT_PER_LEAD,
            "maximum_target_label_opened": "2024-12-30",
        },
        "calibrated_fuxi_vs_mme": {
            "cases_per_lead": CASE_COUNT_PER_LEAD,
            "ecmwf_in_mme": False,
            "mme_members": list(FIXED_MME_COMPONENTS),
        },
        "exact_common_midpoint": {
            "common_midpoints_per_lead": 85,
            "case_lead_rows": 510,
            "sampling_unit": "valid-period midpoint synchronized exactly across six leads",
        },
        "selected_seed": 43,
        "retrospective_only": True,
        "sealed_2025_target_opened": False,
        "latest_target_year": 2024,
        "artifact_sha256": _inventory(comparison_root, "manifest.json"),
    }
    comparison_manifest = comparison_root / "manifest.json"
    _json(comparison_manifest, comparison)

    inputs = FigureInputs(
        benchmark_manifest=benchmark_manifest,
        scoring_manifest=scoring_manifest,
        audit_receipt=audit_receipt,
        comparison_manifest=comparison_manifest,
        imd_methods_manifest=imd_methods,
        imd_seasonal_summary=imd_seasonal,
        imd_weekly_summary=imd_weekly,
        imd_case_metrics=imd_cases,
        imerg_methods_manifest=imerg_methods,
        imerg_seasonal_summary=imerg_seasonal,
        imerg_weekly_summary=imerg_weekly,
        imerg_case_metrics=imerg_cases,
        calibration_manifest=calibration_manifest,
        adapter_support=adapter_support,
        spatial_support_metadata=spatial_metadata,
    )
    hashes = InputHashes(
        **{field: sha256_file(Path(getattr(inputs, field))) for field in inputs.__dataclass_fields__}
    )
    return inputs, hashes, spatial_hashes


def test_generate_figures_is_atomic_manifest_bound_and_reproducible(tmp_path: Path) -> None:
    inputs, hashes, spatial_hashes = _frozen_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    first = output_root / "first"
    second = output_root / "second"
    first_manifest = generate_figures(
        inputs,
        hashes,
        first,
        output_root=output_root,
        spatial_array_hashes=spatial_hashes,
    )
    second_manifest = generate_figures(
        inputs,
        hashes,
        second,
        output_root=output_root,
        spatial_array_hashes=spatial_hashes,
    )

    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["hard_gates"]["valid_midpoint_jjas_cases_per_lead"] == 169
    assert manifest["hard_gates"]["independent_scoring_audit"] == "passed"
    assert manifest["hard_gates"]["selected_seed"] == 43
    assert manifest["hard_gates"]["sealed_2025_target_opened"] is False
    assert manifest["hard_gates"]["matched_imd_imerg_case_ids"] is True
    assert manifest["figures"]["figure1"]["status"] == "rendered"
    assert manifest["figures"]["figure4"]["packaged_case_level_rows"] is False
    assert manifest["visual_inspection"]["status"] == "not_recorded"
    assert manifest["hard_gates"]["frozen_adapter_support_cells"] == 171
    assert manifest["hard_gates"]["fractional_india_geometry_cells"] == 174
    assert len(list((first / "figures").glob("*.png"))) == 6
    assert len(list((first / "figures").glob("*.pdf"))) == 6
    geometry = pd.read_csv(first / manifest["figures"]["figure1"]["data"])
    assert len(geometry) == 27 * 27
    assert int(geometry["adapter_support"].sum()) == 171
    assert int(geometry["india_geometry"].sum()) == 174
    figure4_contract = json.loads(
        (first / manifest["figures"]["figure4"]["contract"]).read_text(encoding="utf-8")
    )
    assert figure4_contract["packaged_case_level_rows"] is False
    assert figure4_contract["upstream_case_level_source"]["row_count"] == CASE_ROWS
    for relative, digest in manifest["artifact_sha256"].items():
        assert sha256_file(first / relative) == digest
    for name in manifest["figures"]:
        for extension in ("png", "pdf"):
            relative = manifest["figures"][name][extension]
            assert sha256_file(first / relative) == sha256_file(second / relative)
    for name in manifest["supplementary_figures"]:
        for extension in ("png", "pdf"):
            relative = manifest["supplementary_figures"][name][extension]
            assert sha256_file(first / relative) == sha256_file(second / relative)
    with pytest.raises(FileExistsError, match="fresh paper-figure output"):
        generate_figures(
            inputs,
            hashes,
            first,
            output_root=output_root,
            spatial_array_hashes=spatial_hashes,
        )


def test_generate_figures_rejects_changed_frozen_table(tmp_path: Path) -> None:
    inputs, hashes, spatial_hashes = _frozen_fixture(tmp_path)
    inputs.imerg_seasonal_summary.write_text(
        inputs.imerg_seasonal_summary.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    with pytest.raises(PaperFigureError, match="SHA-256 differs"):
        generate_figures(
            inputs,
            hashes,
            output_root / "changed",
            output_root=output_root,
            spatial_array_hashes=spatial_hashes,
        )


def test_generate_figures_rejects_changed_spatial_logical_hash(tmp_path: Path) -> None:
    inputs, hashes, spatial_hashes = _frozen_fixture(tmp_path)
    changed = SpatialArrayHashes(
        **{
            **spatial_hashes.__dict__,
            "central_india_fraction": "0" * 64,
        }
    )
    output_root = tmp_path / "outputs"
    with pytest.raises(PaperFigureError, match="spatial logical array differs"):
        generate_figures(
            inputs,
            hashes,
            output_root / "changed-geometry",
            output_root=output_root,
            spatial_array_hashes=changed,
        )
