"""Synthetic edge tests for the Quest data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from config import EXPERIMENT
from data import (
    FEATURE_NAMES,
    aggregate_weekly,
    build_features,
    convert_tp_to_mm_day,
    member_quintile_probabilities,
    observation_categories,
    previous_20yr_thresholds,
    weekly_valid_dates,
)


def test_official_windows_are_day_one_based_and_inclusive() -> None:
    initialization = pd.Timestamp("2025-07-17")  # Thursday

    first = weekly_valid_dates(initialization, "D19-25")
    second = weekly_valid_dates(initialization, "D26-32")

    assert list(first) == list(pd.date_range("2025-08-04", periods=7))
    assert list(second) == list(pd.date_range("2025-08-11", periods=7))
    assert first[0].day_name() == "Monday"
    assert second[0] - first[-1] == pd.Timedelta(days=1)


def test_weekly_aggregation_uses_exact_dates_and_variable_operation() -> None:
    times = pd.date_range("2025-07-17", periods=40)
    daily = xr.DataArray(
        np.arange(40.0), dims=("time",), coords={"time": times}, attrs={"units": "mm day-1"}
    )

    temperature = aggregate_weekly(daily, "2025-07-17", "D19-25", "tas")
    precipitation = aggregate_weekly(daily, "2025-07-17", (26, 32), "pr")

    assert temperature.item() == pytest.approx(np.arange(18.0, 25.0).mean())
    assert precipitation.item() == pytest.approx(np.arange(25.0, 32.0).sum())
    assert temperature.attrs["valid_start"].startswith("2025-08-04")
    assert precipitation.attrs["weekly_operation"] == "sum"
    assert precipitation.attrs["units"] == "mm"

    with pytest.raises(ValueError, match="missing 1 required"):
        aggregate_weekly(daily.drop_sel(time="2025-08-04"), "2025-07-17", 0, "pr")


def test_tp_conversion_is_metadata_driven_and_fails_closed() -> None:
    hourly_rate = xr.DataArray([0.25, 1.0], dims="x", attrs={"units": "mm h-1"})
    converted = convert_tp_to_mm_day(hourly_rate)
    xr.testing.assert_allclose(converted, xr.DataArray([6.0, 24.0], dims="x"))
    assert converted.attrs["units"] == "mm day-1"
    np.testing.assert_allclose(
        convert_tp_to_mm_day(np.asarray([0.25]), units="mm/h"), [6.0]
    )

    metre_accumulation = convert_tp_to_mm_day(
        np.asarray([0.001, 0.002]), units="m", accumulation_hours=24
    )
    np.testing.assert_allclose(metre_accumulation, [1.0, 2.0])

    with pytest.raises(ValueError, match="explicit precipitation units"):
        convert_tp_to_mm_day(np.asarray([0.001]))
    with pytest.raises(ValueError, match="accumulation_hours"):
        convert_tp_to_mm_day(np.asarray([0.001]), units="m")
    with pytest.raises(ValueError, match="unsupported precipitation units"):
        convert_tp_to_mm_day(np.asarray([1.0]), units="mystery")
    with pytest.raises(ValueError, match="must not be supplied"):
        convert_tp_to_mm_day(
            np.asarray([1.0]), units="mm day-1", accumulation_hours=24
        )


def test_member_probabilities_use_upper_category_at_equal_boundary() -> None:
    members = xr.DataArray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dims="member")
    bounds = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="quantile")

    probabilities = member_quintile_probabilities(members, bounds)

    # Counts are [1, 1, 1, 1, 2]. Exact-boundary values move upward.
    np.testing.assert_allclose(
        probabilities.values, (np.asarray([1, 1, 1, 1, 2]) + 0.5) / 8.5
    )
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities.attrs["boundary_equality"] == "upper_category"


def test_duplicate_boundaries_disable_interior_categories_and_deserts() -> None:
    duplicate = xr.DataArray([0.0, 0.0, 2.0, 3.0], dims="quantile")
    observations = xr.DataArray([0.0, 1.0, 3.0], dims="case")

    result = observation_categories(observations, duplicate)

    # [0, 0) is empty; equality at zero reaches the next non-empty category.
    np.testing.assert_array_equal(result.index.values, [2, 2, 4])
    np.testing.assert_array_equal(result.valid_mask.values, [True, True, True])
    np.testing.assert_array_equal(result.probabilities.sum("quintile").values, 1.0)
    assert not bool(result.probabilities.sel(quintile=1).any())

    all_equal = xr.DataArray([0.0, 0.0, 0.0, 0.0], dims="quantile")
    desert = observation_categories(xr.DataArray([0.0], dims="case"), all_equal)
    assert desert.index.item() == -1
    assert not desert.valid_mask.item()
    assert bool(desert.probabilities.isnull().all())

    # The anchor remains finite at the same cell; target=-1 supplies the mask.
    dry_anchor = member_quintile_probabilities(
        xr.DataArray([0.0, 0.0, 1.0], dims="member"), all_equal
    )
    assert bool(np.isfinite(dry_anchor).all())
    assert dry_anchor.sum().item() == pytest.approx(1.0)


def test_missing_member_or_decreasing_bounds_produces_invalid_probabilities() -> None:
    members = xr.DataArray(
        [[0.0, 1.0], [1.0, np.nan]], dims=("member", "cell")
    )
    bounds = xr.DataArray(
        [[0.2, 0.2], [0.4, 0.4], [0.6, 0.1], [0.8, 0.8]],
        dims=("quantile", "cell"),
    )

    probabilities = member_quintile_probabilities(members, bounds)

    assert bool(probabilities.sel(cell=0).notnull().all())
    assert bool(probabilities.sel(cell=1).isnull().all())


def _feature_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    lead = ["D19-25", "D26-32"]
    member = np.arange(5)
    latitude = [0.0, 90.0]
    longitude = [0.0, 90.0]
    base = np.arange(5.0)[None, :, None, None]
    weekly = np.broadcast_to(base, (2, 5, 2, 2)).copy()
    weekly[1] += 5.0
    weekly_members = xr.DataArray(
        weekly,
        dims=("lead", "member", "latitude", "longitude"),
        coords={
            "lead": lead,
            "member": member,
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={"units": "mm"},
    )
    anchor = xr.DataArray(
        np.full((2, 5, 2, 2), 0.2),
        dims=("lead", "quintile", "latitude", "longitude"),
        coords={
            "lead": lead,
            "quintile": np.arange(5),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    land = xr.DataArray(
        [[1.0, 0.0], [0.75, 0.25]],
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )
    return weekly_members, anchor, land


def test_build_features_exact_contract_values_and_training_stats() -> None:
    weekly_members, anchor, land = _feature_inputs()
    means = {name: 1.0 for name in FEATURE_NAMES[5:10]}
    stds = {name: 2.0 for name in FEATURE_NAMES[5:10]}

    features = build_features(
        weekly_members,
        anchor,
        "2025-07-17",
        weekly_members.latitude,
        weekly_members.longitude,
        land,
        tp_quantile_means=means,
        tp_quantile_stds=stds,
    )

    assert features.dims == ("lead", "feature", "latitude", "longitude")
    assert features.shape == (2, 18, 2, 2)
    assert tuple(features.feature.values) == FEATURE_NAMES == EXPERIMENT.feature_names
    assert features.dtype == np.float32
    assert features.sel(feature="log_p_q1").item(0) == pytest.approx(np.log(0.2))
    # Linear q50 of [0,1,2,3,4] is 2; standardize log1p(2) with mean=1,std=2.
    assert features.sel(lead="D19-25", feature="tp_q50").item(0) == pytest.approx(
        (np.log1p(2.0) - 1.0) / 2.0
    )
    assert features.sel(feature="sin_lat", latitude=90.0).item(0) == pytest.approx(1.0)
    assert features.sel(feature="sin_lon", longitude=90.0).item(0) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        features.sel(feature="lead_flag").isel(latitude=0, longitude=0).values,
        [-1.0, 1.0],
    )
    xr.testing.assert_allclose(
        features.sel(feature="land_fraction", lead="D19-25", drop=True).astype(np.float64),
        land,
    )


def test_build_features_rejects_leaky_or_invalid_implicit_normalization() -> None:
    weekly_members, anchor, land = _feature_inputs()

    with pytest.raises(ValueError, match="supplied together"):
        build_features(
            weekly_members,
            anchor,
            "2025-07-17",
            weekly_members.latitude,
            weekly_members.longitude,
            land,
            tp_quantile_means=np.zeros(5),
        )

    broken_anchor = anchor.copy()
    broken_anchor[0, 0, 0, 0] = 0.3
    with pytest.raises(ValueError, match="sum to one"):
        build_features(
            weekly_members,
            broken_anchor,
            "2025-07-17",
            weekly_members.latitude,
            weekly_members.longitude,
            land,
        )


def test_build_features_accepts_train_stats_per_lead() -> None:
    weekly_members, anchor, land = _feature_inputs()
    means = np.stack([np.zeros(5), np.ones(5)])
    stds = np.full((2, 5), 2.0)

    features = build_features(
        weekly_members,
        anchor,
        "2025-07-17",
        weekly_members.latitude,
        weekly_members.longitude,
        land,
        tp_quantile_means=means,
        tp_quantile_stds=stds,
    )

    # Quantiles are taken after log1p, matching prepare_data.py.
    log_members_first = np.log1p(np.arange(5.0))
    log_members_second = np.log1p(np.arange(5.0, 10.0))
    expected_first = np.quantile(log_members_first, 0.1) / 2.0
    expected_second = (np.quantile(log_members_second, 0.1) - 1.0) / 2.0
    assert features.sel(lead="D19-25", feature="tp_q10").item(0) == pytest.approx(
        expected_first
    )
    assert features.sel(lead="D26-32", feature="tp_q10").item(0) == pytest.approx(
        expected_second
    )

def _daily_year_field() -> xr.DataArray:
    time = pd.date_range("2001-01-01", "2021-12-31", freq="D")
    values = time.year.to_numpy(dtype=np.float64, copy=True)
    values[time.year == 2021] = 999_999.0
    return xr.DataArray(values, dims="time", coords={"time": time}, name="pr")


def test_previous_20yr_thresholds_use_100_past_samples_only() -> None:
    daily = _daily_year_field()
    target = pd.Timestamp("2021-06-07")
    issue = target - pd.Timedelta(days=18)

    thresholds = previous_20yr_thresholds(daily, target, issue, "tas")

    expected_samples = np.repeat(np.arange(2001.0, 2021.0), 5)
    np.testing.assert_allclose(
        thresholds.values, np.quantile(expected_samples, [0.2, 0.4, 0.6, 0.8])
    )
    assert thresholds.attrs["sample_count"] == 100
    assert thresholds.attrs["sample_day_offsets"] == "-4,-2,0,2,4"
    assert np.max(thresholds.values) < 999_999.0

    precipitation = previous_20yr_thresholds(daily, target, issue, "pr")
    np.testing.assert_allclose(precipitation, thresholds * 7.0)


def test_previous_20yr_thresholds_refuse_unavailable_or_missing_history() -> None:
    daily = _daily_year_field()
    target = pd.Timestamp("2021-06-07")

    with pytest.raises(ValueError, match="unavailable at issue_time"):
        previous_20yr_thresholds(daily, target, "2020-06-10", "tas")

    missing = daily.drop_sel(time="2001-06-03")
    with pytest.raises(ValueError, match="missing 1 daily observation"):
        previous_20yr_thresholds(
            missing, target, target - pd.Timedelta(days=18), "tas"
        )
