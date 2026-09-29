"""Contract tests for the table-only calibrated regional sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import india_s2s_calibrated_regional_sensitivity as regional


def _dates() -> list[pd.Timestamp]:
    counts = {2020: 34, 2021: 34, 2022: 34, 2023: 34, 2024: 33}
    return [
        pd.Timestamp(year=year, month=6, day=1) + pd.Timedelta(days=index)
        for year, count in counts.items()
        for index in range(count)
    ]


def _case_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    method_scale = {
        "raw_fuxi": 1.0,
        "moment_calibration": 0.9,
        "location_spread": 0.8,
    }
    region_scale = {
        "northwest_india": 0.8,
        "central_india": 1.1,
        "south_peninsula": 0.9,
        "east_northeast_india": 1.2,
    }
    dates = _dates()
    assert len(dates) == 169
    for method, scale in method_scale.items():
        for region, region_factor in region_scale.items():
            for lead in regional.LEADS:
                for index, midpoint in enumerate(dates):
                    perturbation = 0.001 * index
                    mse = (4.0 + 0.1 * lead + perturbation) * scale * region_factor
                    variance = (1.0 + 0.05 * lead + perturbation) * scale
                    rows.append(
                        {
                            "method": method,
                            "region": region,
                            "region_label": regional.REGION_LABELS[region],
                            "lead_week": lead,
                            "verification_year": midpoint.year,
                            "init": midpoint.strftime("%Y-%m-%d"),
                            "valid_period_midpoint": midpoint.isoformat(),
                            "season": "JJAS",
                            "crps": (2.0 + 0.1 * lead + perturbation) * scale * region_factor,
                            "acc": 0.6 - 0.05 * lead + (1.0 - scale) * 0.1,
                            "rmse": (5.0 + 0.1 * lead + perturbation) * scale * region_factor,
                            "mae": (3.0 + 0.1 * lead + perturbation) * scale * region_factor,
                            "bias": -0.3 + (1.0 - scale) * 0.2,
                            "coverage90": 0.5 + (1.0 - scale),
                            "ensemble_variance": variance,
                            "mean_squared_error": mse,
                            "spread_skill_ratio": np.sqrt(variance / mse),
                        }
                    )
    selected = pd.DataFrame(rows)
    filler_count = regional.EXPECTED_CASE_ROWS - len(selected)
    filler = pd.concat([selected.iloc[[0]]] * filler_count, ignore_index=True)
    filler["region"] = "all_india"
    filler["region_label"] = "All India"
    cases = pd.concat([selected, filler], ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    for (method, region, lead), group in selected.groupby(
        ["method", "region", "lead_week"], sort=False
    ):
        row: dict[str, object] = {
            "method": method,
            "region": region,
            "lead_week": lead,
            "case_count": len(group),
        }
        for metric in ("crps", "acc", "rmse", "mae", "bias", "coverage90"):
            row[metric] = float(group[metric].mean())
        variance = float(group["ensemble_variance"].mean())
        mse = float(group["mean_squared_error"].mean())
        row.update(
            {
                "ensemble_variance": variance,
                "mean_squared_error": mse,
                "ensemble_spread": np.sqrt(variance),
                "pooled_spread_skill_ratio": np.sqrt(variance / mse),
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    score_index = summary.set_index(["region", "method", "lead_week"])
    for index, row in summary.iterrows():
        raw = score_index.loc[(row.region, "raw_fuxi", row.lead_week)]
        for metric in regional.LOSS_METRICS:
            summary.loc[index, f"{metric}_skill_pct_vs_raw"] = 100.0 * (
                1.0 - float(row[metric]) / float(raw[metric])
            )
        for metric, source in (
            ("acc", "acc"),
            ("bias", "bias"),
            ("coverage90", "coverage90"),
            ("spread_skill_ratio", "pooled_spread_skill_ratio"),
        ):
            summary.loc[index, f"delta_{metric}_vs_raw"] = (
                float(row[source]) - float(raw[source])
            )

    score_index = summary.set_index(["region", "method", "lead_week"])
    interval_rows: list[dict[str, object]] = []
    for region in regional.REGIONS:
        for lead in regional.LEADS:
            for metric in regional.METRICS:
                column = (
                    "pooled_spread_skill_ratio"
                    if metric == "spread_skill_ratio"
                    else metric
                )
                candidate = float(score_index.loc[(region, "location_spread", lead), column])
                baseline = float(score_index.loc[(region, "raw_fuxi", lead), column])
                effects = ["candidate_minus_baseline"]
                if metric in regional.LOSS_METRICS:
                    effects.append("skill_pct_vs_baseline")
                for block in regional.BLOCK_LENGTHS:
                    for effect in effects:
                        estimate = (
                            100.0 * (1.0 - candidate / baseline)
                            if effect == "skill_pct_vs_baseline"
                            else candidate - baseline
                        )
                        interval_rows.append(
                            {
                                "reference": "imd",
                                "season": "JJAS_valid_midpoint",
                                "analysis_cohort": regional.ANALYSIS_COHORT,
                                "region": region,
                                "lead_week": lead,
                                "metric": metric,
                                "candidate": "location_spread",
                                "baseline": "raw_fuxi",
                                "candidate_mean": candidate,
                                "baseline_mean": baseline,
                                "n_cases": 169,
                                "confidence": 0.95,
                                "block_length_starts": block,
                                "replicates": 10_000,
                                "seed": 42 + 1000 * lead + block,
                                "effect": effect,
                                "estimate": estimate,
                                "ci_lower": estimate - 0.01,
                                "ci_upper": estimate + 0.01,
                            }
                        )
    return cases, summary, pd.DataFrame(interval_rows)


def test_build_regional_tables_filters_four_regions_and_reuses_intervals() -> None:
    cases, summary, intervals = _case_tables()
    descriptive, all_intervals, primary, receipt = regional.build_regional_tables(
        cases, summary, intervals
    )

    assert len(descriptive) == 72
    assert len(all_intervals) == 480
    assert len(primary) == 240
    assert set(descriptive["region"]) == set(regional.REGIONS)
    assert set(descriptive["method"]) == set(regional.METHODS)
    assert set(descriptive["n_cases"]) == {169}
    assert set(primary["block_length_starts"]) == {16}
    assert receipt["selected_regional_case_rows"] == 12_168
    assert receipt["sealed_2025_target_opened"] is False
    source = intervals[
        intervals["region"].eq("central_india")
        & intervals["lead_week"].eq(3)
        & intervals["metric"].eq("crps")
        & intervals["effect"].eq("skill_pct_vs_baseline")
        & intervals["block_length_starts"].eq(16)
    ].iloc[0]
    published = primary[
        primary["region"].eq("central_india")
        & primary["lead_week"].eq(3)
        & primary["metric"].eq("crps")
        & primary["effect"].eq("skill_pct_vs_baseline")
    ].iloc[0]
    assert published["estimate"] == source["estimate"]
    assert published["ci_lower"] == source["ci_lower"]
    assert published["ci_upper"] == source["ci_upper"]
    assert published["effect_direction_semantics"] == "positive_favors_neural"
    bias = primary[primary["metric"].eq("bias")].iloc[0]
    assert bias["effect_direction_semantics"] == "not_monotone_signed_change"
    assert set(primary["multiplicity_adjustment"]) == {"none"}


def test_build_regional_tables_rejects_missing_case() -> None:
    cases, summary, intervals = _case_tables()
    regional_row = cases.index[
        cases["region"].eq("central_india")
        & cases["method"].eq("raw_fuxi")
        & cases["lead_week"].eq(1)
    ][0]
    filler_row = cases.index[cases["region"].eq("all_india")][0]
    malformed = cases.drop(index=regional_row)
    malformed = pd.concat([malformed, cases.loc[[filler_row]]], ignore_index=True)

    with pytest.raises(regional.RegionalSensitivityError, match="case-row count"):
        regional.build_regional_tables(malformed, summary, intervals)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_publish_is_atomic_hash_bound_and_opens_no_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases, summary, intervals = _case_tables()
    scoring_root = tmp_path / "scoring"
    tables = scoring_root / "tables"
    receipts = scoring_root / "receipts"
    tables.mkdir(parents=True)
    receipts.mkdir(parents=True)
    cases.to_csv(tables / "case_metrics.csv", index=False)
    summary.to_csv(tables / "jjas_valid_midpoint_summary.csv", index=False)
    intervals.to_csv(tables / "paired_block_intervals.csv", index=False)
    sampling = {
        regional.ANALYSIS_COHORT: {
            f"w{lead}__block{block}": {
                "shape": [10_000, 169],
                "seed": 42 + 1000 * lead + block,
                "index_matrix_sha256": f"{lead}{block}".ljust(64, "a")[:64],
            }
            for lead in regional.LEADS
            for block in regional.BLOCK_LENGTHS
        }
    }
    _write_json(receipts / "bootstrap_sampling.json", sampling)
    source_hashes = {
        "tables/case_metrics.csv": regional.sha256_file(tables / "case_metrics.csv"),
        "tables/jjas_valid_midpoint_summary.csv": regional.sha256_file(
            tables / "jjas_valid_midpoint_summary.csv"
        ),
        "tables/paired_block_intervals.csv": regional.sha256_file(
            tables / "paired_block_intervals.csv"
        ),
        "receipts/bootstrap_sampling.json": regional.sha256_file(
            receipts / "bootstrap_sampling.json"
        ),
    }
    manifest = {
        "experiment": "india_s2s_probabilistic_bridge_scoring_v1",
        "status": "complete",
        "scientific_status": "retrospective 2020-2024 benchmark evidence",
        "selected_seed": 43,
        "member_count_each_method": 50,
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
        "opened_observation_years": [2020, 2021, 2022, 2023, 2024],
        "methods": list(regional.METHODS),
        "metrics": list(regional.METRICS),
        "cohort": {
            "valid_midpoint_jjas_count_per_lead": 169,
            "maximum_target_label_opened": "2024-12-30",
        },
        "uncertainty": {
            "block_lengths_starts": [16, 13],
            "replicates": 10_000,
            "cohorts": {regional.ANALYSIS_COHORT: 169},
        },
        "artifact_sha256": source_hashes,
    }
    manifest_path = scoring_root / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        regional, "SCORING_MANIFEST_SHA256", regional.sha256_file(manifest_path)
    )
    monkeypatch.setattr(regional, "SOURCE_ARTIFACT_SHA256", source_hashes)
    output_root = tmp_path / "resultsv3/india_s2s_calibrated_regional_sensitivity"
    monkeypatch.setattr(regional, "DEFAULT_OUTPUT_ROOT", output_root)
    output = output_root / "full_test"

    published = regional.publish(scoring_manifest=manifest_path, output=output)
    payload = json.loads(published.read_text(encoding="utf-8"))

    assert payload["status"] == "complete"
    assert payload["new_resampling_performed"] is False
    assert payload["forecast_arrays_opened"] is False
    assert payload["observation_arrays_opened"] is False
    assert payload["sealed_2025_target_opened"] is False
    assert payload["table_rows"] == {
        "regional_jjas_descriptive_metrics": 72,
        "regional_neural_vs_raw_intervals_all_blocks": 480,
        "regional_neural_vs_raw_primary_block16": 240,
    }
    for relative, expected in payload["artifact_sha256"].items():
        assert regional.sha256_file(output / relative) == expected
    with pytest.raises(regional.RegionalSensitivityError, match="fresh output path"):
        regional.publish(scoring_manifest=manifest_path, output=output)

    source_text = Path(regional.__file__).read_text(encoding="utf-8")
    for forbidden in ("np.load(", "xarray", "open_zarr", "zarr.open"):
        assert forbidden not in source_text
