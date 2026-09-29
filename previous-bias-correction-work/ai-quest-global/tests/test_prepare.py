"""Scientific-contract tests for preprocessing."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import prepare_data
from prepare_data import (
    _official_cache_contract_sha256,
    _static_features,
    _training_filename,
    climatology_thresholds,
    ensemble_probabilities,
    observed_category,
    _validate_land_file,
    _validate_training_file,
)


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("pr", "pr_sevenday_WEEKLYSUM_2020.nc"),
        ("tas", "tas_sevenday_WEEKLYMEAN_2020.nc"),
        ("mslp", "mslp_sevenday_WEEKLYMEAN_2020.nc"),
    ],
)
def test_official_training_filename(variable: str, expected: str) -> None:
    assert _training_filename(variable, 2020) == expected


def test_official_training_filename_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError, match="unsupported Quest training variable"):
        _training_filename("imerg", 2020)


def test_official_download_validation_rejects_partial_annual_file(
    tmp_path, monkeypatch
) -> None:
    latitude = np.asarray([1.0, -1.0])
    longitude = np.asarray([0.0, 120.0, 240.0])
    monkeypatch.setattr(prepare_data, "EXPECTED_LATITUDE", latitude)
    monkeypatch.setattr(prepare_data, "EXPECTED_LONGITUDE", longitude)
    time = pd.date_range("2020-01-01", "2020-12-31")
    field = xr.DataArray(
        np.ones((len(time), len(latitude), len(longitude)), dtype=np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"units": "K"},
        name="tas",
    )
    complete = tmp_path / "tas_complete.nc"
    partial = tmp_path / "tas_partial.nc"
    field.to_dataset().to_netcdf(complete)
    field.isel(time=slice(None, -1)).to_dataset().to_netcdf(partial)

    _validate_training_file(complete, "tas", 2020)
    with pytest.raises(ValueError, match="every daily"):
        _validate_training_file(partial, "tas", 2020)

    land_path = tmp_path / "land.nc"
    xr.DataArray(
        np.full((len(latitude), len(longitude)), 0.5, dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name="land_fraction",
    ).to_dataset().to_netcdf(land_path)
    _validate_land_file(land_path)


def test_official_download_validation_rejects_interior_nan_and_rate_units(
    tmp_path, monkeypatch
) -> None:
    latitude = np.asarray([1.0, -1.0])
    longitude = np.asarray([0.0, 120.0, 240.0])
    monkeypatch.setattr(prepare_data, "EXPECTED_LATITUDE", latitude)
    monkeypatch.setattr(prepare_data, "EXPECTED_LONGITUDE", longitude)
    time = pd.date_range("2020-01-01", "2020-12-31")
    values = np.ones((len(time), len(latitude), len(longitude)), dtype=np.float32)
    values[len(time) // 2, 0, 0] = np.nan
    field = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"units": "mm"},
        name="pr",
    )
    interior_nan = tmp_path / "pr_interior_nan.nc"
    field.to_dataset().to_netcdf(interior_nan)
    with pytest.raises(ValueError, match="non-finite"):
        _validate_training_file(interior_nan, "pr", 2020)

    rate_units = tmp_path / "pr_rate_units.nc"
    field.fillna(1.0).assign_attrs(units="mm day-1").to_dataset().to_netcdf(rate_units)
    with pytest.raises(ValueError, match="incompatible pr units"):
        _validate_training_file(rate_units, "pr", 2020)


def test_official_cache_contract_fingerprint_tracks_split_and_land() -> None:
    class FakeGroup(dict):
        attrs = {
            "feature_names": ["a", "b"],
            "lead_windows": [[19, 25], [26, 32]],
            "source_store": "/source.zarr",
            "tp_conversion": "sum",
            "climatology": "twenty years",
            "tp_quantile_mean": [[0.0]],
            "tp_quantile_std": [[1.0]],
        }

    group = FakeGroup(
        land_fraction=np.asarray([[0.0, 1.0]], dtype=np.float32),
        init_yyyymmdd=np.asarray([20200102], dtype=np.int32),
        valid_start_yyyymmdd=np.asarray([[20200120, 20200127]], dtype=np.int32),
    )
    baseline = _official_cache_contract_sha256(group)
    assert len(baseline) == 64
    assert baseline == _official_cache_contract_sha256(group)

    changed = FakeGroup(group)
    changed["init_yyyymmdd"] = np.asarray([20210107], dtype=np.int32)
    assert _official_cache_contract_sha256(changed) != baseline

    changed = FakeGroup(group)
    changed["land_fraction"] = np.asarray([[1.0, 1.0]], dtype=np.float32)
    assert _official_cache_contract_sha256(changed) != baseline


def test_operational_case_copies_training_cache_fingerprint(
    tmp_path, monkeypatch
) -> None:
    fingerprint = "c" * 64

    class FakeCache(dict):
        attrs = {
            "status": "complete",
            "tp_quantile_mean": np.zeros((2, 5), dtype=np.float32).tolist(),
            "tp_quantile_std": np.ones((2, 5), dtype=np.float32).tolist(),
            "cache_contract_sha256": fingerprint,
        }

    cache = FakeCache(
        land_fraction=np.ones((121, 240), dtype=np.float32),
    )
    monkeypatch.setitem(
        sys.modules,
        "zarr",
        SimpleNamespace(open_group=lambda *args, **kwargs: cache),
    )
    monkeypatch.setattr(
        prepare_data,
        "_raw_run_initialization",
        lambda _: pd.Timestamp("2026-08-13"),
    )
    monkeypatch.setattr(prepare_data, "load_era5_weekly", lambda _: object())
    monkeypatch.setattr(
        prepare_data,
        "load_operational_members",
        lambda _: np.zeros((2, 14, 121, 240), dtype=np.float32),
    )
    monkeypatch.setattr(
        prepare_data,
        "climatology_thresholds",
        lambda *args, **kwargs: np.zeros((4, 121, 240), dtype=np.float32),
    )
    monkeypatch.setattr(
        prepare_data,
        "_static_features",
        lambda *args, **kwargs: np.zeros((2, 18, 121, 240), dtype=np.float32),
    )
    output = tmp_path / "prepared_case.npz"
    args = SimpleNamespace(
        init_date="2026-08-13",
        output=output,
        raw_directory=tmp_path / "raw",
        era5_root=tmp_path / "era5",
        cache=tmp_path / "training.zarr",
    )

    prepare_data.prepare_operational_case(args)

    with np.load(output, allow_pickle=False) as archive:
        assert np.asarray(archive["cache_contract_sha256"]).item() == fingerprint


def test_category_equality_moves_to_upper_nonempty_bin() -> None:
    observation = np.asarray([[1.0, 0.0, 5.0]], dtype=np.float32)
    thresholds = np.asarray(
        [
            [[0.0, 0.0, 5.0]],
            [[1.0, 0.0, 5.0]],
            [[2.0, 1.0, 5.0]],
            [[3.0, 2.0, 5.0]],
        ],
        dtype=np.float32,
    )

    result = observed_category(observation, thresholds)

    assert result.tolist() == [[2, 2, -1]]


def test_smoothed_member_probabilities_are_positive_and_normalized() -> None:
    members = np.arange(10, dtype=np.float32)[:, None, None]
    bounds = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float32)[:, None, None]

    probabilities = ensemble_probabilities(members, bounds)

    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0)
    assert np.all(probabilities > 0.0)
    np.testing.assert_allclose(
        probabilities[:, 0, 0],
        (np.asarray([2, 2, 2, 2, 2]) + 0.5) / 12.5,
    )


def test_climatology_uses_only_previous_twenty_years_and_five_offsets() -> None:
    time = pd.date_range("1999-01-01", "2021-12-31", freq="D")
    values = np.asarray(time.year * 1000 + time.dayofyear, dtype=np.float32)[:, None, None]
    weekly = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )
    target = pd.Timestamp("2021-06-15")
    expected_dates = [
        target.replace(year=year) + pd.Timedelta(days=offset)
        for year in range(2001, 2021)
        for offset in (-4, -2, 0, 2, 4)
    ]
    expected = np.quantile(
        weekly.sel(time=expected_dates).values,
        [0.2, 0.4, 0.6, 0.8],
        axis=0,
    )

    actual = climatology_thresholds(weekly, target)

    np.testing.assert_allclose(actual, expected)


def test_feature_order_and_normalization() -> None:
    p0 = np.full((2, 5, 3, 4), 0.2, dtype=np.float32)
    quantiles = np.arange(10, dtype=np.float32).reshape(2, 5, 1, 1)
    quantiles = np.broadcast_to(quantiles, (2, 5, 3, 4)).copy()
    mean = np.arange(10, dtype=np.float32).reshape(2, 5)
    std = np.full((2, 5), 2.0, dtype=np.float32)
    latitude = np.asarray([90.0, 0.0, -90.0])
    longitude = np.asarray([0.0, 90.0, 180.0, 270.0])
    land = np.ones((3, 4), dtype=np.float32)

    features = _static_features(
        p0,
        quantiles,
        pd.Timestamp("2020-01-02"),
        latitude,
        longitude,
        land,
        mean,
        std,
    )

    assert features.shape == (2, 18, 3, 4)
    np.testing.assert_allclose(features[:, :5], np.log(0.2))
    np.testing.assert_allclose(features[:, 5:10], 0.0)
    np.testing.assert_allclose(features[0, 10, :, 0], [1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(features[0, 12, 0], [0.0, 1.0, 0.0, -1.0], atol=1e-6)
    assert np.all(features[0, 16] == -1.0)
    assert np.all(features[1, 16] == 1.0)
    np.testing.assert_allclose(features[:, 17], 1.0)
