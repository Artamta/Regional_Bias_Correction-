"""Tests for leakage-safe FuXi member-summary features and their raw cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_attention_climatology as experiment  # noqa: E402
import fuxi_imerg_experiment as base  # noqa: E402
import fuxi_imerg_full_archive_latelead as common  # noqa: E402


def _forecast_and_observations() -> tuple[
    base.ForecastData,
    base.ObservationData,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cases, leads, height, width = 3, 6, 27, 27
    case = np.arange(cases, dtype=np.float32)[:, None, None, None]
    lead = np.arange(leads, dtype=np.float32)[None, :, None, None]
    row = np.arange(height, dtype=np.float32)[None, None, :, None]
    column = np.arange(width, dtype=np.float32)[None, None, None, :]
    ensemble_mean = 1.5 + case + 0.2 * lead + 0.03 * row + 0.01 * column
    ensemble_spread = 0.4 + 0.1 * case + 0.02 * lead + 0.004 * row
    ensemble_spread = np.broadcast_to(
        ensemble_spread, (cases, leads, height, width)
    ).copy()
    t2m = 280.0 + case + 0.5 * lead - 0.2 * row + 0.1 * column

    support = np.zeros((height, width), dtype=bool)
    support[7:20, 7:20] = True
    weights = support.astype(np.float64)
    climatology = np.full((cases, leads, height, width), 3.0, dtype=np.float32)
    truth = climatology + 0.25 * case
    climatology[..., ~support] = np.nan
    truth[..., ~support] = np.nan
    climatology_daily = np.full((366, height, width), 3.0, dtype=np.float32)
    climatology_daily[:, ~support] = np.nan

    initializations = np.asarray(
        ["2002-06-01", "2003-06-01", "2018-06-01"], dtype="datetime64[D]"
    )
    forecast = base.ForecastData(
        initializations=initializations,
        valid_dates=base.derive_valid_dates(initializations),
        ensemble_mean=ensemble_mean.astype(np.float32),
        ensemble_spread=ensemble_spread.astype(np.float32),
        latitude=np.linspace(39.0, 0.0, height),
        longitude=np.linspace(60.0, 99.0, width),
        source_files=(),
    )
    observations = base.ObservationData(
        weekly_truth=truth.astype(np.float32),
        weekly_climatology=climatology,
        observation_fraction=support.astype(np.float32),
        source_stores=(),
    )
    return forecast, observations, weights, t2m.astype(np.float32), climatology_daily


def _synthetic_summaries(
    forecast: base.ForecastData,
    t2m: np.ndarray,
) -> common.FuxiMemberSummaries:
    cases, leads, height, width = forecast.ensemble_mean.shape
    case = np.arange(cases, dtype=np.float32)[:, None, None, None]
    lead = np.arange(leads, dtype=np.float32)[None, :, None, None]
    row = np.arange(height, dtype=np.float32)[None, None, :, None]
    column = np.arange(width, dtype=np.float32)[None, None, None, :]
    median = 0.2 + 0.10 * case + 0.03 * lead + 0.002 * row + 0.001 * column
    iqr = 0.1 + 0.02 * case + 0.01 * lead + 0.001 * row
    iqr = np.broadcast_to(iqr, (cases, leads, height, width)).copy()
    wet = 0.2 + 0.05 * case + 0.02 * lead + 0.002 * column
    wet = np.broadcast_to(wet, (cases, leads, height, width)).copy()
    heavy = 0.05 + 0.01 * case + 0.005 * lead + 0.0005 * row
    heavy = np.broadcast_to(heavy, (cases, leads, height, width)).copy()
    t2m_spread = 0.5 + 0.1 * case + 0.02 * lead + 0.001 * row
    t2m_spread = np.broadcast_to(
        t2m_spread, (cases, leads, height, width)
    ).copy()
    return common.FuxiMemberSummaries(
        initializations=forecast.initializations.copy(),
        latitude=forecast.latitude.copy(),
        longitude=forecast.longitude.copy(),
        t2m_weekly_mean=t2m.copy(),
        tp_member_log_median=median.astype(np.float32),
        tp_member_log_iqr=iqr.astype(np.float32),
        tp_member_probability_ge_1mm_day=wet.astype(np.float32),
        tp_member_probability_ge_10mm_day=heavy.astype(np.float32),
        t2m_member_spread_weekly=t2m_spread.astype(np.float32),
        source_fingerprint="synthetic-source-fingerprint",
    )


def test_member_reduction_forms_weeks_before_summarizing_members() -> None:
    member = np.arange(51, dtype=np.float32)[:, None, None, None]
    day = np.arange(42, dtype=np.float32)[None, :, None, None]
    row = np.arange(2, dtype=np.float32)[None, None, :, None]
    column = np.arange(3, dtype=np.float32)[None, None, None, :]
    tp_mm_day = 0.25 * member + 0.1 * day + 0.2 * row + 0.05 * column
    tp_hourly = (tp_mm_day / np.float32(24.0)).astype(np.float32)
    t2m_daily = (280.0 + 0.1 * member + 0.2 * day + row + column).astype(
        np.float32
    )

    result = common._summarize_member_fields(tp_hourly, t2m_daily)

    tp_weekly = tp_mm_day.reshape(51, 6, 7, 2, 3).mean(axis=2)
    t2m_weekly = t2m_daily.reshape(51, 6, 7, 2, 3).mean(
        axis=2, dtype=np.float64
    ).astype(np.float32)
    log_quantiles = np.quantile(
        np.log1p(tp_weekly), [0.25, 0.5, 0.75], axis=0, method="linear"
    )
    np.testing.assert_allclose(
        result["fuxi_tp_member_log_median"], log_quantiles[1], rtol=1e-6
    )
    np.testing.assert_allclose(
        result["fuxi_tp_member_log_iqr"],
        log_quantiles[2] - log_quantiles[0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result["fuxi_tp_member_probability_ge_1mm_day"],
        np.mean(tp_weekly >= 1.0, axis=0),
    )
    np.testing.assert_allclose(
        result["fuxi_tp_member_probability_ge_10mm_day"],
        np.mean(tp_weekly >= 10.0, axis=0),
    )
    legacy_t2m_mean = t2m_daily.reshape(51, 6, 7, 2, 3).mean(
        axis=(0, 2), dtype=np.float64
    ).astype(np.float32)
    np.testing.assert_array_equal(
        result["t2m_weekly_mean"], legacy_t2m_mean
    )
    np.testing.assert_allclose(
        result["fuxi_t2m_member_spread_weekly"],
        np.std(t2m_weekly, axis=0, ddof=0),
        rtol=1e-6,
    )


def test_raw_member_cache_round_trip_and_stale_fingerprint_rejection(
    tmp_path: Path,
) -> None:
    forecast, _, _, t2m, _ = _forecast_and_observations()
    summaries = _synthetic_summaries(forecast, t2m)
    cache = tmp_path / "member_summaries.npz"

    checksum = common._write_member_summary_cache(cache, summaries)
    loaded = common._member_summaries_from_cache(
        cache, forecast, summaries.source_fingerprint
    )

    assert loaded.cache_sha256 == checksum
    assert loaded.cache_path == str(cache.resolve())
    assert tuple(loaded.feature_fields) == common.MEMBER_SUMMARY_FEATURE_NAMES
    np.testing.assert_array_equal(loaded.t2m_weekly_mean, t2m)
    for name in common.MEMBER_SUMMARY_FEATURE_NAMES:
        np.testing.assert_array_equal(
            loaded.feature_fields[name], summaries.feature_fields[name]
        )
    with pytest.raises(base.DataContractError, match="stale"):
        common._member_summaries_from_cache(cache, forecast, "different-source")


def test_optional_member_channels_preserve_control_and_full_domain_contract() -> None:
    forecast, observations, weights, t2m, climatology_daily = (
        _forecast_and_observations()
    )
    summaries = _synthetic_summaries(forecast, t2m)
    train = np.asarray([0, 1], dtype=np.int64)

    control, control_stats, _ = experiment.build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        train,
        t2m,
        preserve_fuxi_context=True,
    )
    member, member_stats, _ = experiment.build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        train,
        summaries.t2m_weekly_mean,
        member_summaries=summaries,
        preserve_fuxi_context=True,
    )

    assert control.shape == (3, 6, 29, 27, 27)
    assert member.shape == (3, 6, 34, 27, 27)
    np.testing.assert_array_equal(member[:, :, :11], control[:, :, :11])
    np.testing.assert_array_equal(member[:, :, 16:], control[:, :, 11:])
    assert member_stats["input_channels"][11:16] == list(
        common.MEMBER_SUMMARY_FEATURE_NAMES
    )
    assert member_stats["fuxi_member_summaries"]["member_count"] == 51
    assert "fuxi_member_summaries" not in control_stats
    support = weights > 0.0
    assert np.isfinite(member[:, :, 11:16, :, :]).all()
    assert np.any(member[:, :, 11:16, ~support] != 0.0)
    assert np.all(member[:, :, 16:, ~support] == 0.0)


def test_member_normalization_ignores_validation_values() -> None:
    forecast, observations, weights, t2m, climatology_daily = (
        _forecast_and_observations()
    )
    summaries = _synthetic_summaries(forecast, t2m)
    train = np.asarray([0, 1], dtype=np.int64)
    original, original_stats, _ = experiment.build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        train,
        t2m,
        member_summaries=summaries,
        preserve_fuxi_context=True,
    )
    changed_fields = {
        name: values.copy() for name, values in summaries.feature_fields.items()
    }
    for name, values in changed_fields.items():
        if "probability" in name:
            values[2] = np.float32(1.0) - values[2]
        else:
            values[2] += np.float32(10_000.0)
    changed = common.FuxiMemberSummaries(
        initializations=summaries.initializations,
        latitude=summaries.latitude,
        longitude=summaries.longitude,
        t2m_weekly_mean=summaries.t2m_weekly_mean,
        tp_member_log_median=changed_fields["fuxi_tp_member_log_median"],
        tp_member_log_iqr=changed_fields["fuxi_tp_member_log_iqr"],
        tp_member_probability_ge_1mm_day=(
            changed_fields["fuxi_tp_member_probability_ge_1mm_day"]
        ),
        tp_member_probability_ge_10mm_day=(
            changed_fields["fuxi_tp_member_probability_ge_10mm_day"]
        ),
        t2m_member_spread_weekly=changed_fields[
            "fuxi_t2m_member_spread_weekly"
        ],
        source_fingerprint=summaries.source_fingerprint,
    )
    rebuilt, rebuilt_stats, _ = experiment.build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        train,
        t2m,
        member_summaries=changed,
        preserve_fuxi_context=True,
    )

    for name in common.MEMBER_SUMMARY_FEATURE_NAMES:
        assert rebuilt_stats[name] == original_stats[name]
    np.testing.assert_array_equal(rebuilt[:2], original[:2])
    assert not np.array_equal(rebuilt[2, :, 11:16], original[2, :, 11:16])
