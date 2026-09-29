"""Regression tests for the IMD-targeted full-domain FuXi feature contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imerg_experiment as base  # noqa: E402
import fuxi_imerg_full_archive_latelead as common  # noqa: E402


def synthetic_fields() -> tuple[
    base.ForecastData,
    base.ObservationData,
    np.ndarray,
    np.ndarray,
]:
    cases, leads, height, width = 3, 6, 27, 27
    case = np.arange(cases, dtype=np.float32)[:, None, None, None]
    lead = np.arange(leads, dtype=np.float32)[None, :, None, None]
    row = np.arange(height, dtype=np.float32)[None, None, :, None]
    column = np.arange(width, dtype=np.float32)[None, None, None, :]
    ensemble_mean = 1.0 + case + 0.2 * lead + 0.03 * row + 0.01 * column
    ensemble_spread = 0.2 + 0.1 * case + 0.02 * lead + 0.004 * row + 0.002 * column
    t2m = 275.0 + case + 0.5 * lead - 0.2 * row + 0.1 * column

    support = np.zeros((height, width), dtype=bool)
    support[7:20, 7:20] = True
    weights = support.astype(np.float64)
    climatology = 3.0 + 0.05 * lead + 0.01 * row + 0.005 * column
    climatology = np.broadcast_to(climatology, (cases, leads, height, width)).copy()
    truth = climatology + 0.3 * case
    climatology[..., ~support] = np.nan
    truth[..., ~support] = np.nan

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
        weekly_climatology=climatology.astype(np.float32),
        observation_fraction=support.astype(np.float32),
        source_stores=(),
    )
    return forecast, observations, weights, t2m.astype(np.float32)


def test_full_domain_fuxi_context_retains_only_forecast_channels() -> None:
    forecast, observations, weights, t2m = synthetic_fields()
    train = np.asarray([0, 1])
    masked, _ = common.make_features(
        forecast, observations, weights, train, t2m
    )
    context, normalization = common.make_features(
        forecast,
        observations,
        weights,
        train,
        t2m,
        preserve_fuxi_context=True,
    )
    support = weights > 0.0

    for channel in (0, 1, 10):
        np.testing.assert_array_equal(
            context[:, :, channel, support], masked[:, :, channel, support]
        )
        assert np.all(masked[:, :, channel, ~support] == 0.0)
        assert np.isfinite(context[:, :, channel, ~support]).all()
        assert np.any(context[:, :, channel, ~support] != 0.0)
    for channel in (2, 9):
        assert np.all(context[:, :, channel, ~support] == 0.0)

    assert normalization["spatial_context"] == {
        "enabled": True,
        "full_domain_channels": [
            "log_fuxi_mean",
            "log_fuxi_spread",
            "fuxi_t2m_weekly",
        ],
        "support_limited_channels": [
            "log_imerg_climatology",
            "explicit_log_fuxi_anomaly",
        ],
        "normalization_fit": "training cases and positive target weights only",
        "target_and_loss_support": "positive target weights only",
    }


def test_inputs_do_not_use_current_imd_truth() -> None:
    forecast, observations, weights, t2m = synthetic_fields()
    train = np.asarray([0, 1])
    original, _ = common.make_features(
        forecast,
        observations,
        weights,
        train,
        t2m,
        preserve_fuxi_context=True,
    )
    changed_truth = observations.weekly_truth.copy()
    changed_truth[..., weights > 0.0] += np.float32(10_000.0)
    changed_observations = base.ObservationData(
        weekly_truth=changed_truth,
        weekly_climatology=observations.weekly_climatology,
        observation_fraction=observations.observation_fraction,
        source_stores=observations.source_stores,
    )
    rebuilt, _ = common.make_features(
        forecast,
        changed_observations,
        weights,
        train,
        t2m,
        preserve_fuxi_context=True,
    )
    np.testing.assert_array_equal(rebuilt, original)


def test_context_statistics_ignore_validation_values() -> None:
    forecast, observations, weights, t2m = synthetic_fields()
    train = np.asarray([0, 1])
    original, original_stats = common.make_features(
        forecast,
        observations,
        weights,
        train,
        t2m,
        preserve_fuxi_context=True,
    )
    changed_mean = forecast.ensemble_mean.copy()
    changed_spread = forecast.ensemble_spread.copy()
    changed_t2m = t2m.copy()
    changed_mean[2] += np.float32(10_000.0)
    changed_spread[2] += np.float32(10_000.0)
    changed_t2m[2] += np.float32(10_000.0)
    changed_forecast = base.ForecastData(
        initializations=forecast.initializations,
        valid_dates=forecast.valid_dates,
        ensemble_mean=changed_mean,
        ensemble_spread=changed_spread,
        latitude=forecast.latitude,
        longitude=forecast.longitude,
        source_files=forecast.source_files,
    )
    rebuilt, rebuilt_stats = common.make_features(
        changed_forecast,
        observations,
        weights,
        train,
        changed_t2m,
        preserve_fuxi_context=True,
    )

    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imerg_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
    ):
        assert rebuilt_stats[name] == original_stats[name]
    np.testing.assert_array_equal(rebuilt[:2], original[:2])
    assert not np.array_equal(rebuilt[2], original[2])
