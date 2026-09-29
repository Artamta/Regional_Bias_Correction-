"""Contracts for the exploratory global FuXi/IMERG cache builder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from prepare_imerg_smoke import (
    CALENDAR_OFFSETS,
    CLIMATOLOGY_YEARS,
    FEATURE_NAMES,
    aggregate_fuxi_tp,
    assemble_npz_payload,
    build_compatible_features,
    fixed_imerg_climatology,
    imerg_category_target,
    imerg_weekly_sums,
    positive_jeffreys_anchor,
    select_blocked_cases,
    training_normalization,
    validate_blocked_years,
    write_npz_atomic,
)
from predict import load_prepared
from train import PreparedCases


def _imerg_dataset(
    time: pd.DatetimeIndex,
    values: np.ndarray,
    fraction: np.ndarray,
) -> xr.Dataset:
    values = np.asarray(values, dtype=np.float32)
    fraction = np.asarray(fraction, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None, None]
    if fraction.ndim == 1:
        fraction = fraction[:, None, None]
    return xr.Dataset(
        {
            "tp": xr.DataArray(
                values,
                dims=("time", "lat", "lon"),
                attrs={"units": "mm day-1"},
            ),
            "observation_fraction": xr.DataArray(
                fraction,
                dims=("time", "lat", "lon"),
                attrs={"units": "1"},
            ),
        },
        coords={"time": time, "lat": np.arange(values.shape[1]), "lon": np.arange(values.shape[2])},
        attrs={"status": "complete"},
    )


def test_blocked_case_selection_is_deterministic_equal_and_whole_case() -> None:
    dates = pd.date_range("2017-01-01", "2020-12-31", freq="7D")

    first = select_blocked_cases(dates, cases_per_year=4)
    second = select_blocked_cases(dates, cases_per_year=4)

    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.splits, second.splits)
    assert len(first.init_dates) == 16
    assert first.init_dates.is_monotonic_increasing
    assert set(first.init_dates.month).issubset({6, 7, 8, 9})
    for year in (2017, 2018, 2019, 2020):
        assert int((first.init_dates.year == year).sum()) == 4
    assert first.splits.tolist().count("train") == 8
    assert first.splits.tolist().count("validation") == 4
    assert first.splits.tolist().count("test") == 4


def test_blocked_years_reject_overlap_and_nonchronological_blocks() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_blocked_years((2017, 2018), (2018,), (2020,))
    with pytest.raises(ValueError, match="chronological"):
        validate_blocked_years((2018,), (2017,), (2020,))


def test_fuxi_conversion_clips_multiplies_and_sums_exact_windows() -> None:
    native = np.zeros((51, 14, 1, 2), dtype=np.float32)
    native[:, :7, 0, 0] = 1.0
    native[:, 7:, 0, 0] = 2.0
    native[:, :, 0, 1] = -1.0

    weekly = aggregate_fuxi_tp(native)

    assert weekly.shape == (2, 51, 1, 2)
    np.testing.assert_allclose(weekly[0, :, 0, 0], 7.0 * 24.0)
    np.testing.assert_allclose(weekly[1, :, 0, 0], 14.0 * 24.0)
    np.testing.assert_array_equal(weekly[:, :, 0, 1], 0.0)


def test_imerg_weekly_sum_uses_coverage_and_requires_exact_dates() -> None:
    time = pd.date_range("2019-06-01", periods=8, freq="D")
    values = np.arange(1.0, 9.0, dtype=np.float32)
    fraction = np.ones(8, dtype=np.float32)
    fraction[1] = 0.0
    dataset = _imerg_dataset(time, values, fraction)

    weekly, coverage = imerg_weekly_sums(dataset, ["2019-06-01"])

    # Day two is unavailable: mean([1,3,4,5,6,7]) * seven.
    assert weekly[0, 0, 0] == pytest.approx(26.0 / 6.0 * 7.0)
    assert coverage[0, 0, 0] == pytest.approx(6.0 / 7.0)
    with pytest.raises(KeyError, match="missing"):
        imerg_weekly_sums(dataset, ["2019-06-04"])


def test_fixed_climatology_uses_only_2002_2016_calendar_offsets() -> None:
    time = pd.date_range("2002-01-01", "2020-12-31", freq="D")
    # Constant within a year makes every seven-day sum exactly 7*year.
    values = np.asarray(time.year, dtype=np.float32)
    values[time.year >= 2017] = 1_000_000.0
    dataset = _imerg_dataset(time, values, np.ones(len(time), dtype=np.float32))

    result = fixed_imerg_climatology(
        dataset,
        "2020-07-15",
        min_weekly_coverage=1.0,
        min_samples=len(CLIMATOLOGY_YEARS) * len(CALENDAR_OFFSETS),
    )

    samples = np.repeat(np.asarray(CLIMATOLOGY_YEARS) * 7.0, len(CALENDAR_OFFSETS))
    expected = np.quantile(samples, (0.2, 0.4, 0.6, 0.8))
    np.testing.assert_allclose(result.bounds[:, 0, 0], expected)
    assert int(result.valid_sample_count[0, 0]) == 75
    assert bool(result.support[0, 0])
    assert float(result.bounds.max()) < 100_000.0


def test_anchor_is_positive_and_target_equality_moves_upward() -> None:
    members = np.arange(51, dtype=np.float32)[:, None, None]
    bounds = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)[:, None, None]
    support = np.ones((1, 1), dtype=bool)

    anchor = positive_jeffreys_anchor(members, bounds, support)
    target = imerg_category_target(np.asarray([[20.0]], dtype=np.float32), bounds, support)

    expected_counts = np.asarray([10, 10, 10, 10, 11], dtype=np.float32)
    np.testing.assert_allclose(anchor[:, 0, 0], (expected_counts + 0.5) / 53.5)
    assert np.all(anchor > 0.0)
    assert anchor[:, 0, 0].sum() == pytest.approx(1.0)
    assert target.item() == 2

    unsupported = positive_jeffreys_anchor(
        members, np.full_like(bounds, np.nan), np.zeros((1, 1), dtype=bool)
    )
    np.testing.assert_allclose(unsupported[:, 0, 0], 0.2)


def test_normalization_ignores_validation_and_test_values() -> None:
    raw = np.empty((4, 2, 5, 1, 1), dtype=np.float32)
    raw[0] = 1.0
    raw[1] = 3.0
    raw[2] = 10_000.0
    raw[3] = 20_000.0
    support = np.ones((4, 2, 1, 1), dtype=bool)
    labels = np.asarray(["train", "train", "validation", "test"])

    mean, std = training_normalization(raw, support, labels)

    np.testing.assert_allclose(mean, 2.0)
    np.testing.assert_allclose(std, 1.0)


def test_feature_builder_rejects_an_extra_anchor_dimension() -> None:
    wrong_rank = np.full((1, 1, 2, 5, 2, 3), 0.2, dtype=np.float32)

    with pytest.raises(ValueError, match="anchors must be"):
        build_compatible_features(
            wrong_rank,
            wrong_rank.copy(),
            ["2017-06-01"],
            np.asarray([1.0, -1.0]),
            np.asarray([0.0, 1.5, 3.0]),
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 5), dtype=np.float32),
            np.ones((2, 5), dtype=np.float32),
        )


def test_payload_is_npz_compatible_and_preserves_exploratory_provenance(tmp_path) -> None:
    dates = pd.to_datetime(["2017-06-01", "2018-06-01", "2019-06-01", "2020-06-01"])
    labels = np.asarray(["train", "train", "validation", "test"])
    height, width = 121, 240
    p0 = np.full((4, 2, 5, height, width), 0.2, dtype=np.float32)
    raw = np.empty_like(p0)
    raw[0] = 1.0
    raw[1] = 3.0
    raw[2] = 100.0
    raw[3] = 200.0
    support = np.ones((4, 2, height, width), dtype=bool)
    target = np.zeros((4, 2, height, width), dtype=np.int8)
    truth = np.ones((4, 2, height, width), dtype=np.float32)
    coverage = np.ones_like(truth)
    thresholds = np.broadcast_to(
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)[None, None, :, None, None],
        (4, 2, 4, height, width),
    ).copy()
    counts = np.full((4, 2, height, width), 75, dtype=np.uint8)

    payload = assemble_npz_payload(
        init_dates=dates,
        splits=labels,
        latitude=np.linspace(90.0, -90.0, height),
        longitude=np.arange(width, dtype=np.float64) * 1.5,
        anchors=p0,
        raw_quantiles=raw,
        targets=target,
        weekly_truth=truth,
        weekly_coverage=coverage,
        support=support,
        thresholds=thresholds,
        climatology_sample_count=counts,
        provenance={"source": "unit-test"},
    )
    output = write_npz_atomic(tmp_path / "imerg_smoke.npz", payload)

    prepared = PreparedCases(output, (2017, 2018))
    assert len(prepared) == 2
    metadata = prepared.metadata()
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    np.testing.assert_allclose(metadata["tp_quantile_mean"], 2.0)
    np.testing.assert_allclose(metadata["tp_quantile_std"], 1.0)
    assert metadata["provenance"]["purpose"] == "exploratory_pretraining"
    assert metadata["provenance"]["official_era5_target"] is False
    features, anchor, observed = prepared[0]
    assert features.shape == (2, 18, height, width)
    assert anchor.shape == (2, 5, height, width)
    assert observed.shape == (2, height, width)
    prediction_batch = load_prepared(output, require_target=True, years=(2019,))
    assert prediction_batch.features.shape == (1, 2, 18, height, width)
    assert prediction_batch.target is not None
    assert prediction_batch.land_fraction is None
    assert prediction_batch.init_dates == ("2019-06-01",)
    with np.load(output, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"].item()))
        assert provenance["purpose"] == "exploratory_pretraining"
        assert provenance["official_ai_weather_quest_validation"] is False
        assert provenance["official_era5_target"] is False
        assert "land_fraction" not in archive
        assert "spatial_support_fraction" in archive
        assert "not a geographic land-sea mask" in provenance[
            "spatial_support_fraction"
        ]
        np.testing.assert_allclose(archive["tp_quantile_mean"], 2.0)
        np.testing.assert_allclose(archive["tp_quantile_std"], 1.0)
        assert archive["features"].dtype == np.float16
        assert archive["p0"].dtype == np.float16
        assert np.all(archive["p0"][:] > 0.0)
        stored_probability_error = np.max(
            np.abs(archive["p0"][:].astype(np.float32).sum(axis=2) - 1.0)
        )
        assert stored_probability_error < 2.0e-3

    with pytest.raises(FileExistsError):
        write_npz_atomic(output, payload)
