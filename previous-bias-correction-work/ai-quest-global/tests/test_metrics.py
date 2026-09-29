"""Synthetic tests for official-style RPS/RPSS masking and aggregation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from metrics import (
    climatological_probabilities_like,
    quest_rpss,
    ranked_probability_score,
    ranked_probability_skill_score,
    spatial_weights,
    weighted_mean,
    weighted_ranked_probability_score,
)


def _one_hot(indices: np.ndarray, dims: tuple[str, ...], coords: dict) -> xr.DataArray:
    values = np.eye(5, dtype=np.float64)[indices]
    # np.eye appends category last; move it to the official leading position.
    values = np.moveaxis(values, -1, 0)
    return xr.DataArray(
        values,
        dims=("quintile",) + dims,
        coords={"quintile": np.arange(5), **coords},
    )


def test_rps_sums_all_cumulative_terms_and_masks_all_nan_cells() -> None:
    observed = _one_hot(
        np.asarray([[0, 2]]),
        ("latitude", "longitude"),
        {"latitude": [0.0], "longitude": [0.0, 1.0]},
    )
    climatology = climatological_probabilities_like(observed)

    score = ranked_probability_score(climatology, observed)

    # q1 observation: .8^2 + .6^2 + .4^2 + .2^2 + 0 = 1.2.
    assert score.sel(longitude=0.0).item() == pytest.approx(1.2)
    # q3 observation: .2^2 + .4^2 + .4^2 + .2^2 + 0 = 0.4.
    assert score.sel(longitude=1.0).item() == pytest.approx(0.4)

    missing = observed.copy()
    missing.loc[{"longitude": 1.0}] = np.nan
    assert np.isnan(ranked_probability_score(climatology, missing).sel(longitude=1.0))

    partial = observed.copy()
    partial.loc[{"quintile": 2, "longitude": 0.0}] = np.nan
    with pytest.raises(ValueError, match="partially missing"):
        ranked_probability_score(climatology, partial)

    infinite = observed.copy()
    infinite.loc[{"longitude": 0.0}] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        ranked_probability_score(climatology, infinite)


def test_perfect_and_climatological_forecasts_have_one_and_zero_rpss() -> None:
    observed = _one_hot(
        np.asarray([[0, 1], [3, 4]]),
        ("latitude", "longitude"),
        {"latitude": [0.0, 30.0], "longitude": [0.0, 1.0]},
    )
    weights = xr.DataArray(
        np.ones((2, 2)),
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 30.0], "longitude": [0.0, 1.0]},
    )

    perfect = ranked_probability_skill_score(observed, observed, weights)
    climatological = ranked_probability_skill_score(
        climatological_probabilities_like(observed), observed, weights
    )

    assert perfect.item() == pytest.approx(1.0)
    assert climatological.item() == pytest.approx(0.0)
    assert weighted_ranked_probability_score(observed, observed, weights).item() == 0.0


def test_rpss_is_ratio_of_weighted_sums_not_mean_of_cell_skills() -> None:
    # Cell 0 observes q3 and is perfect: RPS=0, climatology RPS=.4.
    # Cell 1 observes q1 and forecasts q5: RPS=4, climatology RPS=1.2.
    observed = _one_hot(
        np.asarray([[2, 0]]),
        ("latitude", "longitude"),
        {"latitude": [0.0], "longitude": [0.0, 1.0]},
    )
    forecast = _one_hot(
        np.asarray([[2, 4]]),
        ("latitude", "longitude"),
        {"latitude": [0.0], "longitude": [0.0, 1.0]},
    )
    weights = xr.DataArray(
        [[1.0, 1.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0], "longitude": [0.0, 1.0]},
    )

    score = ranked_probability_skill_score(forecast, observed, weights)

    assert score.item() == pytest.approx(1.0 - 4.0 / (0.4 + 1.2))
    mean_cell_skill = (1.0 + (1.0 - 4.0 / 1.2)) / 2.0
    assert score.item() != pytest.approx(mean_cell_skill)


def test_cosine_land_and_arid_masks_match_official_rules() -> None:
    latitude = xr.DataArray([0.0, 60.0], dims="latitude", coords={"latitude": [0.0, 60.0]})
    longitude = [0.0, 1.0]
    land = xr.DataArray(
        [[0.5, 0.49], [1.0, 1.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )
    wettest = xr.DataArray(
        [[1.0, 1.0], [0.0, 2.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )

    weights = spatial_weights(
        latitude, "pr", land_fraction=land, wettest_boundary=wettest
    )

    assert weights.sel(latitude=0.0, longitude=0.0).item() == pytest.approx(1.0)
    assert np.isnan(weights.sel(latitude=0.0, longitude=1.0))
    assert np.isnan(weights.sel(latitude=60.0, longitude=0.0))
    assert weights.sel(latitude=60.0, longitude=1.0).item() == pytest.approx(0.5)

    global_weights = spatial_weights(latitude, "mslp")
    np.testing.assert_allclose(global_weights, [1.0, 0.5])
    with pytest.raises(ValueError, match="wettest_boundary is required"):
        spatial_weights(latitude, "pr", land_fraction=land)

    shifted_wettest = wettest.assign_coords(longitude=[10.0, 11.0])
    with pytest.raises(ValueError, match="cannot align"):
        spatial_weights(
            latitude,
            "pr",
            land_fraction=land,
            wettest_boundary=shifted_wettest,
        )


def test_weighted_mean_renormalizes_over_joint_valid_mask() -> None:
    values = xr.DataArray(
        [1.0, np.nan, 5.0], dims="cell", coords={"cell": [0, 1, 2]}
    )
    weights = xr.DataArray(
        [1.0, 100.0, 3.0], dims="cell", coords={"cell": [0, 1, 2]}
    )

    result = weighted_mean(values, weights, dims="cell")

    assert result.item() == pytest.approx((1.0 + 15.0) / 4.0)

    mismatched = weights.assign_coords(cell=[10, 11, 12])
    with pytest.raises(ValueError, match="cannot align"):
        weighted_mean(values, mismatched, dims="cell")


def test_quest_wrapper_applies_land_arid_and_explicit_valid_masks() -> None:
    coords = {"latitude": [0.0], "longitude": [0.0, 1.0, 2.0]}
    observed = _one_hot(
        np.asarray([[0, 0, 0]]), ("latitude", "longitude"), coords
    )
    forecast = observed.copy()
    # Make excluded cells deliberately bad; the retained first cell is perfect.
    forecast.loc[{"longitude": [1.0, 2.0]}] = _one_hot(
        np.asarray([[4, 4]]),
        ("latitude", "longitude"),
        {"latitude": [0.0], "longitude": [1.0, 2.0]},
    )
    land = xr.DataArray(
        [[1.0, 0.2, 1.0]], dims=("latitude", "longitude"), coords=coords
    )
    wettest = xr.DataArray(
        [[1.0, 1.0, 0.0]], dims=("latitude", "longitude"), coords=coords
    )
    valid = xr.DataArray(
        [[True, True, True]], dims=("latitude", "longitude"), coords=coords
    )

    score = quest_rpss(
        forecast,
        observed,
        observed.latitude,
        "pr",
        land_fraction=land,
        wettest_boundary=wettest,
        valid_mask=valid,
    )

    assert score.item() == pytest.approx(1.0)
