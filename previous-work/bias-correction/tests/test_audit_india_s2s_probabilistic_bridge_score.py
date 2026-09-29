"""Independent semantic-audit tests for the probabilistic bridge score."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import audit_india_s2s_probabilistic_bridge_score as audit


def _scoring_dates() -> np.ndarray:
    yearly: list[np.ndarray] = []
    for year in audit.FORECAST_YEARS:
        dates = np.arange(
            np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        )
        dates = dates[np.isin(pd.DatetimeIndex(dates).dayofweek, (0, 3))]
        if year == 2024:
            missing = np.asarray(
                (
                    "2024-06-06",
                    "2024-06-10",
                    "2024-06-13",
                    "2024-06-17",
                    "2024-08-05",
                ),
                dtype="datetime64[D]",
            )
            dates = dates[~np.isin(dates, missing)]
        yearly.append(dates)
    forecast = np.concatenate(yearly)
    assert len(forecast) == 517
    assert audit.initialization_dates_sha256(forecast) == audit.FORECAST_DATES_SHA256
    scoreable = forecast[
        forecast + np.timedelta64(42, "D") <= np.datetime64("2024-12-31")
    ]
    assert len(scoreable) == 505
    assert audit.initialization_dates_sha256(scoreable) == audit.SCORING_DATES_SHA256
    return scoreable


def _case_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = _scoring_dates()
    method_effect = {
        "raw_fuxi": (1.0, 0.00, 0.68),
        "moment_calibration": (0.95, 0.03, 0.76),
        "location_spread": (0.90, 0.06, 0.84),
    }
    for date_index, init in enumerate(dates):
        init_text = np.datetime_as_string(init, unit="D")
        for method in audit.METHOD_ORDER:
            scale, acc_gain, coverage = method_effect[method]
            for region_index, region in enumerate(audit.REGION_ORDER):
                for lead in audit.LEAD_WEEKS:
                    raw_crps = 1.5 + 0.08 * lead + 0.01 * region_index + 0.0001 * date_index
                    raw_rmse = 2.5 + 0.10 * lead + 0.02 * region_index + 0.0001 * date_index
                    rmse = raw_rmse * scale
                    variance = (0.75 + 0.03 * lead + 0.02 * region_index) ** 2
                    midpoint = init.astype("datetime64[h]") + np.timedelta64(
                        84 + 168 * (lead - 1), "h"
                    )
                    rows.append(
                        {
                            "track": "tp_imd",
                            "method": method,
                            "variable": "tp",
                            "reference": "imd",
                            "reference_label": "IMD",
                            "units": "mm day-1",
                            "score_status": "available",
                            "verification_year": int(init_text[:4]),
                            "init": init_text,
                            "valid_period_start": np.datetime_as_string(
                                init + np.timedelta64(7 * (lead - 1), "D"), unit="D"
                            ),
                            "valid_period_midpoint": np.datetime_as_string(
                                midpoint, unit="s"
                            ),
                            "valid_period_end_exclusive": np.datetime_as_string(
                                init + np.timedelta64(7 * lead, "D"), unit="D"
                            ),
                            "season": audit._season(pd.Timestamp(midpoint).month),
                            "region": region,
                            "region_label": audit.REGION_LABELS[region],
                            "lead_week": lead,
                            "crps": raw_crps * scale,
                            "acc": 0.15 + 0.01 * lead + 0.005 * region_index + acc_gain,
                            "rmse": rmse,
                            "mae": (1.8 + 0.07 * lead + 0.01 * region_index) * scale,
                            "bias": -0.2 - 0.01 * lead + (1.0 - scale) * 0.1,
                            "valid_cell_count": 171 - region_index,
                            "effective_area_km2": 3_200_000.0 - 100_000.0 * region_index,
                            "coverage90": coverage,
                            "ensemble_variance": variance,
                            "mean_squared_error": rmse**2,
                            "ensemble_spread": np.sqrt(variance),
                            "spread_skill_ratio": np.sqrt(variance) / rmse,
                            "model": method,
                            "model_label": audit.METHOD_LABELS[method],
                            "member_count": 50,
                            "selected_seed": (
                                43 if method == "location_spread" else "not_applicable"
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    assert len(frame) == audit.CASE_ROW_COUNT
    return frame


def _raw_receipt(cases: pd.DataFrame, benchmark_path: Path) -> dict[str, object]:
    return {
        "identity": True,
        "matched_case_rows": audit.RAW_CASE_ROW_COUNT,
        "case_ids_sha256": audit.RAW_CASE_IDS_SHA256,
        "complete_bridge_contract": True,
        "key_columns": ["init", "lead_week", "region"],
        "metric_columns": [
            "acc",
            "rmse",
            "mae",
            "bias",
            "valid_cell_count",
            "effective_area_km2",
        ],
        "atol": 1.0e-6,
        "rtol": 1.0e-7,
        "support_atol": 1.0e-6,
        "support_rtol": 1.0e-12,
        "failure_count_by_metric": {
            metric: 0
            for metric in (
                "acc",
                "rmse",
                "mae",
                "bias",
                "valid_cell_count",
                "effective_area_km2",
            )
        },
        "max_absolute_difference_by_metric": {
            metric: 0.0
            for metric in (
                "acc",
                "rmse",
                "mae",
                "bias",
                "valid_cell_count",
                "effective_area_km2",
            )
        },
        "benchmark_case_metrics": str(benchmark_path.resolve()),
        "benchmark_case_metrics_sha256": audit.BENCHMARK_CASES_SHA256,
    }


@pytest.fixture(scope="module")
def semantic_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("score-audit-bundle")
    (root / "tables").mkdir()
    (root / "receipts").mkdir()
    cases = _case_metrics()
    cases.to_csv(root / "tables/case_metrics.csv", index=False)
    allseason = audit.add_raw_comparisons(audit.aggregate_case_metrics(cases))
    jjas_cases = cases[cases.season.eq("JJAS")]
    jjas = audit.add_raw_comparisons(audit.aggregate_case_metrics(jjas_cases))
    allseason.to_csv(root / "tables/allseason_summary.csv", index=False)
    jjas.to_csv(root / "tables/jjas_valid_midpoint_summary.csv", index=False)

    intervals, receipts = audit.reconstruct_lead_intervals(
        cases, replicates=4, seed=7, block_lengths=(2, 3)
    )
    story, story_receipts, _ = audit.reconstruct_synchronized_story_gate(
        cases, replicates=4, seed=7, block_lengths=(2, 3)
    )
    combined_intervals = pd.concat([intervals, story], ignore_index=True, sort=False)
    combined_intervals.to_csv(root / "tables/paired_block_intervals.csv", index=False)
    receipts[
        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
    ] = story_receipts
    (root / "receipts/bootstrap_sampling.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    benchmark = cases[cases.method.eq("raw_fuxi")].copy()
    benchmark["model"] = "fuxi_s2s"
    benchmark_path = root / "benchmark_case_metrics.csv"
    benchmark.to_csv(benchmark_path, index=False)
    raw_receipt = _raw_receipt(cases, benchmark_path)
    (root / "receipts/raw_fuxi_identity.json").write_text(
        json.dumps(raw_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    truth_receipt = {
        "opened_2025": False,
        "sealed_2025_target_opened": False,
    }
    (root / "receipts/truth_access.json").write_text(
        json.dumps(truth_receipt) + "\n", encoding="utf-8"
    )
    cohort = {
        "scoring": {
            "count": 505,
            "dates_sha256": audit.SCORING_DATES_SHA256,
            "maximum_target_label": "2024-12-30",
        },
        "sealed_2025_target_opened": False,
    }
    (root / "receipts/cohort.json").write_text(
        json.dumps(cohort) + "\n", encoding="utf-8"
    )
    manifest = {
        "table_rows": {
            "case_metrics": len(cases),
            "allseason_summary": len(allseason),
            "jjas_valid_midpoint_summary": len(jjas),
            "paired_block_intervals": len(combined_intervals),
        },
        "uncertainty": {
            "replicates": 4,
            "seed": 7,
            "block_lengths_starts": [2, 3],
        },
        "raw_fuxi_identity": raw_receipt,
    }
    manifest["artifact_sha256"] = {
        str(path.relative_to(root)): audit.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, benchmark_path


def test_complete_case_contract_and_selected_seed() -> None:
    cases = _case_metrics()
    dates = audit.validate_case_metrics(cases)

    assert len(dates) == 505
    assert len(cases[cases.method.eq("raw_fuxi")]) == audit.RAW_CASE_ROW_COUNT
    tampered = cases.copy()
    tampered.loc[tampered.method.eq("location_spread"), "selected_seed"] = "mean_of_seeds"
    with pytest.raises(audit.ScoreAuditError, match="seed 43"):
        audit.validate_case_metrics(tampered)


def test_zero_error_dry_case_allows_only_undefined_spread_ratio() -> None:
    cases = _case_metrics()
    index = cases.index[cases.method.eq("location_spread")][0]
    for column in (
        "crps",
        "rmse",
        "mae",
        "bias",
        "ensemble_variance",
        "mean_squared_error",
        "ensemble_spread",
    ):
        cases.loc[index, column] = 0.0
    cases.loc[index, "spread_skill_ratio"] = np.nan

    audit.validate_case_metrics(cases)
    cases.loc[index, "spread_skill_ratio"] = 0.0
    with pytest.raises(audit.ScoreAuditError, match="zero-MSE"):
        audit.validate_case_metrics(cases)


def test_synchronized_pooled_gate_uses_lead_specific_100_case_cohorts() -> None:
    story, receipts, gate = audit.reconstruct_synchronized_story_gate(
        _case_metrics(), replicates=8, seed=5, block_lengths=(2,)
    )

    assert len(story) == 8
    assert set(story.n_cases_per_lead) == {100}
    assert set(story.n_case_lead_rows) == {600}
    assert gate["date_intersection_count"] == 70
    assert gate["date_union_count"] == 130
    assert gate["retrospective_2022_2024_component_passes"] is True
    receipt = receipts["pooled_w1_w6__block2"]
    assert receipt["shape"] == [8, 100]
    assert receipt["sampling_unit"].startswith("within-verification-year ordinal")
    assert len(set(receipt["lead_dates_sha256"].values())) == 6


def test_complete_saved_semantics_reconstruct_cleanly(
    semantic_bundle: tuple[Path, Path]
) -> None:
    manifest_path, benchmark_path = semantic_bundle
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = audit.validate_output_semantics(
        manifest_path,
        manifest,
        audit.HashLedger(),
        {"benchmark_path": benchmark_path},
    )

    assert result["raw_identity"]["matched_case_rows"] == audit.RAW_CASE_ROW_COUNT
    assert result["story_gate"]["case_lead_rows"] == 600
    assert result["bootstrap_sampling_receipts_checked"] == 3


def test_semantic_tampering_fails_even_when_artifact_hash_is_updated(
    semantic_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    source_manifest, source_benchmark = semantic_bundle
    copied = tmp_path / "copied"
    shutil.copytree(source_manifest.parent, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_fuxi_identity"]["benchmark_case_metrics"] = str(
        (copied / "benchmark_case_metrics.csv").resolve()
    )
    raw_receipt_path = copied / "receipts/raw_fuxi_identity.json"
    raw_receipt = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
    raw_receipt["benchmark_case_metrics"] = str(
        (copied / "benchmark_case_metrics.csv").resolve()
    )
    raw_receipt_path.write_text(
        json.dumps(raw_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["raw_fuxi_identity"] = raw_receipt

    summary_path = copied / "tables/allseason_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[0, "crps"] += 0.25
    summary.to_csv(summary_path, index=False)
    manifest["artifact_sha256"] = {
        str(path.relative_to(copied)): audit.sha256_file(path)
        for path in sorted(copied.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = audit.HashLedger()
    audit._verify_artifacts(manifest_path, manifest, ledger, label="synthetic scoring")

    with pytest.raises(audit.ScoreAuditError, match="all-season summary numeric column differs"):
        audit.validate_output_semantics(
            manifest_path,
            manifest,
            ledger,
            {"benchmark_path": copied / source_benchmark.name},
        )


def test_bootstrap_receipt_tampering_is_detected(
    semantic_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    source_manifest, source_benchmark = semantic_bundle
    copied = tmp_path / "copied"
    shutil.copytree(source_manifest.parent, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = copied / source_benchmark.name
    raw_path = copied / "receipts/raw_fuxi_identity.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["benchmark_case_metrics"] = str(benchmark.resolve())
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["raw_fuxi_identity"] = raw
    receipt_path = copied / "receipts/bootstrap_sampling.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["2022_2024_valid_midpoint_jjas"]["w1__block2"][
        "index_matrix_sha256"
    ] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(audit.ScoreAuditError, match="bootstrap sampling receipt differs"):
        audit.validate_output_semantics(
            manifest_path,
            manifest,
            audit.HashLedger(),
            {"benchmark_path": benchmark},
        )


def test_auditor_has_no_live_forecast_or_truth_loader_dependency() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "import india_s2s_probabilistic_bridge_score" not in source
    assert "open_forecast_dataset" not in source
    assert "open_observation_dataset" not in source
    assert "--scoring-manifest" in source
    assert "--output" in source


def test_no_clobber_audit_publication(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "receipt").write_text("first", encoding="utf-8")
    destination = tmp_path / "published"

    audit.rename_noreplace(source, destination)
    assert (destination / "receipt").read_text(encoding="utf-8") == "first"

    raced = tmp_path / "raced"
    raced.mkdir()
    (raced / "receipt").write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit.rename_noreplace(raced, destination)
    assert (destination / "receipt").read_text(encoding="utf-8") == "first"
