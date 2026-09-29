"""Focused synthetic contracts for the frozen E3 station evaluator."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
for path in (HERE / "src", HERE / "evaluate", HERE / "hpc_compat"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_adapter_station_external_target as station_eval  # noqa: E402


def test_exact_2024_dates_and_windows_never_enter_2025() -> None:
    dates = station_eval.validate_exact_initializations(
        station_eval.EXACT_INITIALIZATIONS.copy()
    )
    required = station_eval.required_verification_dates(dates)

    assert dates.shape == (30,)
    assert required[0] == np.datetime64("2024-06-03")
    assert required[-1] == np.datetime64("2024-11-10")
    assert set(pd.DatetimeIndex(required).year) == {2024}

    changed = dates.copy()
    changed[-1] = np.datetime64("2025-01-01")
    with pytest.raises(
        station_eval.StationEvaluationContractError, match="dates/order"
    ):
        station_eval.validate_exact_initializations(changed)


def test_mixed_station_container_filters_before_rainfall_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write("location_id,rain_day,rain_mm,unused\n")
        stream.write("station_a,2024-06-03,3.5,x\n")
        stream.write("station_b,2024-06-03,NOT_PARSED,x\n")
        stream.write("station_a,2025-01-01,SEALED_VALUE,x\n")

    selected = station_eval.stream_exact_2024_station_truth(
        path,
        ["station_a"],
        np.asarray(["2024-06-03"], dtype="datetime64[D]"),
    )

    assert selected.rows.to_dict("records") == [
        {"location_id": "station_a", "rain_day": "2024-06-03", "rain_mm": 3.5}
    ]
    assert selected.unselected_2025_plus_rows == 1
    assert selected.container_date_max == "2025-01-01"


def test_selected_station_snapshot_is_deterministic_and_2024_only(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "location_id": ["station_a", "station_b"],
            "rain_day": ["2024-06-03", "2024-06-04"],
            "rain_mm": [1.0, 2.0],
        }
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    station_eval.atomic_write_deterministic_gzip_csv(first, frame)
    station_eval.atomic_write_deterministic_gzip_csv(second, frame)

    assert station_eval.sha256_file(first) == station_eval.sha256_file(second)
    restored = pd.read_csv(first)
    assert restored.rain_day.astype(str).str.startswith("2024-").all()


def test_grid_ids_are_parsed_as_native_indices_not_compressed_positions() -> None:
    latitude = 39.0 - 1.5 * np.arange(27)
    longitude = 60.0 + 1.5 * np.arange(27)
    stale_grid_position = np.asarray([999, -4])

    ii, jj = station_eval.parse_grid_cell_ids(
        ["grid_20_12", "grid_00_26"], latitude, longitude
    )

    np.testing.assert_array_equal(ii, [20, 0])
    np.testing.assert_array_equal(jj, [12, 26])
    assert not np.array_equal(ii * 27 + jj, stale_grid_position)
    with pytest.raises(station_eval.StationEvaluationContractError, match="invalid"):
        station_eval.parse_grid_cell_ids(["cell_20_12"], latitude, longitude)


def test_station_week_requires_six_days_and_median_combines_duplicate_gauges() -> None:
    daily = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, np.nan],
            [3, 3, 3, 3, 3, 3, 3],
            [9, 9, 9, 9, 9, np.nan, np.nan],
        ],
        dtype=float,
    )
    climate = np.vstack([np.full(366, 0.5), np.full(366, 1.5), np.full(366, 8.0)])

    target = station_eval.aggregate_station_week_to_cells(
        daily,
        climate,
        np.arange(7),
        np.arange(7),
        np.asarray([0, 0, 1]),
        n_cells=3,
    )

    assert target.station_location_count == 2
    assert target.station_grid_cell_count == 1
    assert target.station_count_by_cell[0] == 2
    assert target.truth[0] == pytest.approx(2.0)
    assert target.climatology[0] == pytest.approx(1.0)
    assert np.isnan(target.truth[1])


def test_weighted_metrics_include_signed_bias_and_enforce_exact_shared_cells() -> None:
    truth = np.linspace(1.0, 20.0, 20)
    prediction = truth + 1.0
    climate = truth * 0.5
    weights = np.linspace(1.0, 2.0, 20)
    common = np.ones(20, dtype=bool)

    metrics = station_eval.weighted_station_metrics(
        truth, prediction, climate, weights, common
    )

    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(1.0)
    assert metrics["absolute_bias"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(1.0)

    changed = prediction.copy()
    changed[0] = np.nan
    with pytest.raises(
        station_eval.StationEvaluationContractError,
        match="method-specific validity",
    ):
        station_eval.weighted_station_metrics(truth, changed, climate, weights, common)


def test_circular_bootstrap_is_shared_deterministic_and_can_wrap() -> None:
    first = station_eval.circular_moving_block_indices(
        30, draws=500, block_length=13, seed=7
    )
    second = station_eval.circular_moving_block_indices(
        30, draws=500, block_length=13, seed=7
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (500, 30)
    assert np.any((first[:, :-1] == 29) & (first[:, 1:] == 0))
    for start in range(0, 30, 13):
        block = first[:, start : min(start + 13, 30)]
        if block.shape[1] > 1:
            assert np.all(np.diff(block, axis=1) % 30 == 1)
    multiplicity = station_eval.mean_date_multiplicity(first, 30)
    assert multiplicity.mean() == pytest.approx(1.0, abs=1.0e-12)
    assert np.max(np.abs(multiplicity - 1.0)) < 0.2
    table = station_eval.bootstrap_multiplicity_table(
        {13: first}, station_eval.EXACT_INITIALIZATIONS
    )
    assert len(table) == 30
    assert table.mean_draw_multiplicity.mean() == pytest.approx(1.0)


def _synthetic_case_metrics() -> pd.DataFrame:
    values = {
        "raw_fuxi": {
            "rmse": 3.0,
            "mae": 2.0,
            "bias": 0.5,
            "absolute_bias": 0.5,
            "acc": 0.4,
        },
        "log_bias": {
            "rmse": 2.5,
            "mae": 1.5,
            "bias": 0.2,
            "absolute_bias": 0.2,
            "acc": 0.5,
        },
        "selected_adapter": {
            "rmse": 2.0,
            "mae": 1.0,
            "bias": -0.1,
            "absolute_bias": 0.1,
            "acc": 0.6,
        },
    }
    rows = []
    for method, metrics in values.items():
        for initialization in station_eval.EXACT_INITIALIZATIONS:
            for lead in station_eval.LEAD_WEEKS:
                rows.append(
                    {
                        "method": method,
                        "initialization": np.datetime_as_string(initialization),
                        "lead_week": lead,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_bootstrap_is_equal_case_raw_minus_adapter_with_leads_attached() -> (
    None
):
    plans = {
        13: station_eval.circular_moving_block_indices(
            30, draws=50, block_length=13, seed=5
        ),
        4: station_eval.circular_moving_block_indices(
            30, draws=50, block_length=4, seed=5
        ),
        8: station_eval.circular_moving_block_indices(
            30, draws=50, block_length=8, seed=5
        ),
    }
    effects = station_eval.paired_bootstrap_effects(
        _synthetic_case_metrics(),
        station_eval.EXACT_INITIALIZATIONS,
        plans,
        station_eval.comparison_specs(station_eval.BASE_METHODS),
    )

    primary = effects.loc[effects.primary_estimand].iloc[0]
    assert primary.comparison == "selected_adapter_vs_raw_fuxi"
    assert primary.effect_definition == "reference_minus_candidate"
    assert primary.point_effect == pytest.approx(1.0)
    assert primary.ci_lower_2p5 == pytest.approx(1.0)
    assert primary.ci_upper_97p5 == pytest.approx(1.0)
    assert primary.case_leads == 180
    assert set(effects.block_length_initializations) == {4, 8, 13}
    acc = effects.loc[
        effects.comparison.eq("selected_adapter_vs_raw_fuxi") & effects.metric.eq("acc")
    ]
    np.testing.assert_allclose(acc.point_effect, 0.2)
    np.testing.assert_allclose(acc.ci_lower_2p5, 0.2)
    np.testing.assert_allclose(acc.ci_upper_97p5, 0.2)
    np.testing.assert_allclose(acc.probability_effect_gt_zero, 1.0)
    assert acc.positive_favors_candidate.eq(True).all()  # noqa: E712


def _extended_manifest(*, uses_weekly_coverage: bool = False) -> dict[str, object]:
    return {
        "status": "complete",
        "canonical": True,
        "scientific_eligible": True,
        "smoke": False,
        "final_2025_store_opened": False,
        "audit_years": [2022, 2023, 2024],
        "audit_counts": {"2022": 35, "2023": 35, "2024": 30},
        "methods": list(station_eval.EXTENDED_STORE_METHODS),
        "bootstrap": {
            "draws": 10_000,
            "block_length_initializations": 13,
            "seed": 20_260_822,
            "canonical_contract": {
                "draws": 10_000,
                "block_length_initializations": 13,
                "seed": 20_260_822,
            },
            "diagnostics": {
                "year_stratified": True,
                "circular_within_year": True,
                "equal_marginal_inclusion_by_design": True,
                "no_year_crossing": True,
                "all_six_leads_retained_per_start": True,
                "mean_multiplicity_across_initializations": 1.0,
            },
        },
        "projection": {
            "weighting_contract": "fixed_forecast_time_india_area_x_frozen_adapter_support",
            "uses_weekly_imd_coverage": uses_weekly_coverage,
            "uses_observed_rainfall_values": False,
            "support_cells": 171,
            "post_hoc": True,
            "maximum_float32_closure_mm_day": 1.0e-6,
        },
        "artifacts": {"predictions.zarr": "a" * 64},
        "array_sha256_contract": (
            "sha256 of contiguous C-order little-endian float32 raw bytes; no header"
        ),
        "array_sha256": {
            "raw_identity": "b" * 64,
            "raw_identity_raw_mean_preserved": "c" * 64,
            "raw_identity_residual": "d" * 64,
        },
        "input_provenance": {"base_tree_sha256": station_eval.BASE_TREE_SHA256},
    }


def test_extended_manifest_forbids_target_coverage_in_projection() -> None:
    station_eval._validate_extended_manifest(_extended_manifest(), "a" * 64)

    with pytest.raises(
        station_eval.StationEvaluationContractError,
        match="uses_weekly_imd_coverage",
    ):
        station_eval._validate_extended_manifest(
            _extended_manifest(uses_weekly_coverage=True), "a" * 64
        )


def test_extended_manifest_rejects_noncanonical_e2_for_production() -> None:
    manifest = _extended_manifest()
    manifest["canonical"] = False
    manifest["scientific_eligible"] = False
    manifest["smoke"] = True
    manifest["status"] = "complete_noncanonical_smoke"

    with pytest.raises(
        station_eval.StationEvaluationContractError,
        match="production gate 'status'",
    ):
        station_eval._validate_extended_manifest(manifest, "a" * 64)
    station_eval._validate_extended_manifest(
        manifest, "a" * 64, require_canonical=False
    )


def test_hashes_bind_file_content_and_tree_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("same", encoding="utf-8")
    (second / "b.txt").write_text("same", encoding="utf-8")

    assert station_eval.sha256_file(first / "a.txt") == station_eval.sha256_file(
        second / "b.txt"
    )
    assert station_eval.sha256_tree(first) != station_eval.sha256_tree(second)


def test_require_extended_fails_before_any_real_source_access(tmp_path: Path) -> None:
    with pytest.raises(
        station_eval.StationEvaluationContractError, match="needs the completed E2"
    ):
        station_eval.run_evaluation(
            tmp_path / "unused", require_extended=True, extended_predictions=None
        )


def test_evaluator_does_not_import_or_run_old_station_cli() -> None:
    source = Path(station_eval.__file__).read_text(encoding="utf-8")
    assert "import piggycast_s2s" not in source
    assert "grid_position" in source  # disclosure/guard text
    assert "usecols=mapping_columns" in source


def test_cpu_launcher_uses_an_account_accessible_safe_partition() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "evaluate_adapter_station_external_target.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu\n" in launcher
    assert "#SBATCH --partition=iiser\n" not in launcher
    assert "#SBATCH --gres=gpu" not in launcher
    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17\n" in launcher
