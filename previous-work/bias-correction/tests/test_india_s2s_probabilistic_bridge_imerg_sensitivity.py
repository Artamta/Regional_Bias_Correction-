from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import india_s2s_benchmark_scoring as benchmark
import india_s2s_probabilistic_bridge_imerg_sensitivity as imerg


def _region_weights(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {name: np.ones(shape, dtype=np.float64) for name in benchmark.REGION_ORDER}


def test_imerg_target_dates_are_period_start_zero_through_41() -> None:
    targets = imerg.imerg_target_dates(
        np.asarray(["2024-06-03", "2024-06-06"], dtype="datetime64[D]")
    )

    assert targets.shape == (2, 6, 7)
    assert targets[0, 0, 0] == np.datetime64("2024-06-03")
    assert targets[0, 0, -1] == np.datetime64("2024-06-09")
    assert targets[0, -1, 0] == np.datetime64("2024-07-08")
    assert targets[0, -1, -1] == np.datetime64("2024-07-14")
    np.testing.assert_array_equal(
        (targets[0] - np.datetime64("2024-06-03")).astype(int).reshape(-1),
        np.arange(42),
    )


def test_weekly_imerg_targets_use_coverage_weighting_minimum_support_and_normal() -> None:
    dates = np.datetime64("2024-01-01") + np.arange(42).astype("timedelta64[D]")
    values = np.broadcast_to(
        np.arange(42, dtype=np.float32)[:, None, None], (42, 1, 2)
    ).copy()
    fractions = np.ones_like(values)
    fractions[0, 0, 0] = 0.5
    fractions[3, 0, 0] = 0.25
    climatology = np.broadcast_to(
        np.arange(366, dtype=np.float32)[:, None, None], (366, 1, 2)
    ).copy()

    weekly = imerg.weekly_imerg_targets(
        [np.datetime64("2024-01-01")],
        dates,
        values,
        fractions,
        climatology,
    )

    expected_w1_cell0 = (0.0 * 0.5 + 1.0 + 2.0 + 3.0 * 0.25 + 4.0 + 5.0 + 6.0) / 5.75
    assert weekly.truth.shape == (1, 6, 1, 2)
    assert weekly.truth[0, 0, 0, 0] == pytest.approx(expected_w1_cell0)
    assert weekly.truth[0, 0, 0, 1] == pytest.approx(3.0)
    assert weekly.observation_support[0, 0, 0, 0] == pytest.approx(0.25)
    assert weekly.observation_support[0, 0, 0, 1] == pytest.approx(1.0)
    assert weekly.climatology[0, 0, 0, 0] == pytest.approx(3.0)
    assert weekly.truth[0, 5, 0, 1] == pytest.approx(38.0)
    assert weekly.target_dates[0, 5, -1] == np.datetime64("2024-02-11")


def test_weekly_imerg_targets_reject_any_2025_source_label() -> None:
    dates = np.asarray(["2024-12-31", "2025-01-01"], dtype="datetime64[D]")
    values = np.zeros((2, 1, 1), dtype=np.float32)
    with pytest.raises(imerg.IMERGSensitivityError, match="2024 firewall"):
        imerg.weekly_imerg_targets(
            [np.datetime64("2024-12-31")],
            dates,
            values,
            np.ones_like(values),
            np.zeros((366, 1, 1), dtype=np.float32),
        )


def test_score_members_relabels_reference_without_changing_forecast_periods() -> None:
    inits = np.asarray(["2024-06-03"], dtype="datetime64[D]")
    members = np.broadcast_to(
        np.linspace(0.0, 2.0, 50, dtype=np.float32).reshape(1, 50, 1, 1, 1),
        (1, 50, 6, 1, 3),
    ).copy()
    weekly = imerg.WeeklyIMERGTargets(
        truth=np.ones((1, 6, 1, 3), dtype=np.float32),
        climatology=np.zeros((1, 6, 1, 3), dtype=np.float32),
        observation_support=np.ones((1, 6, 1, 3), dtype=np.float32),
        target_dates=imerg.imerg_target_dates(inits),
    )

    rows = imerg._score_members(
        method="raw_fuxi",
        initializations=inits,
        members=members,
        truth=weekly,
        truth_indices=np.asarray([0]),
        climatology=np.zeros((1, 6, 1, 3), dtype=np.float32),
        region_weights=_region_weights((1, 3)),
    )

    assert len(rows) == 30
    assert set(rows.track) == {"tp_imerg"}
    assert set(rows.reference) == {"imerg"}
    assert set(rows.reference_label) == {"IMERG"}
    first = rows[(rows.region == "all_india") & (rows.lead_week == 1)].iloc[0]
    last = rows[(rows.region == "all_india") & (rows.lead_week == 6)].iloc[0]
    assert first.valid_period_start == "2024-06-03"
    assert first.valid_period_end_exclusive == "2024-06-10"
    assert last.valid_period_start == "2024-07-08"
    assert last.valid_period_end_exclusive == "2024-07-15"
    assert np.isfinite(rows.crps).all()
    assert np.isfinite(rows.coverage90).all()


def test_common_benchmark_subset_requires_exact_505_by_six_by_five() -> None:
    dates = np.datetime64("2020-01-01") + np.arange(505).astype("timedelta64[D]")
    date_strings = [np.datetime_as_string(value, unit="D") for value in dates]
    rows: list[dict[str, object]] = []
    for init in date_strings:
        for lead in benchmark.LEAD_WEEKS:
            for region in benchmark.REGION_ORDER:
                rows.append(
                    {
                        "model": "fuxi_s2s",
                        "track": "tp_imerg",
                        "reference": "imerg",
                        "score_status": "available",
                        "init": init,
                        "lead_week": lead,
                        "region": region,
                    }
                )
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "model": "fuxi_s2s",
        "track": "tp_imerg",
        "reference": "imerg",
        "score_status": "available",
        "init": "2025-01-01",
        "lead_week": 1,
        "region": "all_india",
    }

    selected = imerg.select_common_benchmark_raw(table, dates)

    assert len(selected) == 15_150
    assert selected.init.nunique() == 505
    assert "2025-01-01" not in set(selected.init)


def test_parser_freezes_real_inputs_and_primary_bootstrap_contract() -> None:
    args = imerg.build_parser().parse_args(
        [
            "--output",
            "resultsv3/india_s2s_probabilistic_bridge/imerg_test",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.inference_manifest == imerg.DEFAULT_INFERENCE_MANIFEST
    assert args.preflight_manifest == imerg.DEFAULT_PREFLIGHT_MANIFEST
    assert args.benchmark_case_metrics == imerg.DEFAULT_BENCHMARK_CASES
    assert args.replicates == 10_000
    assert args.bootstrap_seed == 42
    assert imerg.imd_score.BLOCK_LENGTHS == (16, 13)
    assert 2025 not in benchmark.FORECAST_YEARS
    assert imerg.TARGET_DAY_OFFSETS == tuple(range(42))


def test_expected_real_input_hashes_are_explicit_sha256_values() -> None:
    for value in (
        imerg.EXPECTED_INFERENCE_MANIFEST_SHA256,
        imerg.EXPECTED_PREFLIGHT_MANIFEST_SHA256,
        imerg.EXPECTED_BENCHMARK_CASES_SHA256,
        imerg.EXPECTED_DATA_CONFIG_SHA256,
        imerg.EXPECTED_EVALUATION_CONFIG_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)

    assert Path(imerg.DEFAULT_INFERENCE_MANIFEST).is_file()
    assert Path(imerg.DEFAULT_PREFLIGHT_MANIFEST).is_file()
    assert Path(imerg.DEFAULT_BENCHMARK_CASES).is_file()
