"""Synthetic tests for the three-variable temporal prototype contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from multivariable import (
    MULTIVARIABLE_FEATURE_NAMES,
    MULTIVARIABLE_INPUT_CHANNELS,
    QUEST_VARIABLES,
    build_multivariable_anchors,
    build_multivariable_features,
    build_multivariable_spatial_weights,
    build_multivariable_targets,
    multivariable_rps_by_variable,
    multivariable_rps_loss,
)


LEADS = ("D19-25", "D26-32")
LATITUDE = (0.0, 60.0)
LONGITUDE = (0.0, 90.0)


def _weekly_members_and_bounds() -> tuple[
    dict[str, xr.DataArray], dict[str, xr.DataArray]
]:
    offsets = {"pr": 0.0, "tas": 278.0, "mslp": 100_000.0}
    members: dict[str, xr.DataArray] = {}
    bounds: dict[str, xr.DataArray] = {}
    for variable, offset in offsets.items():
        values = np.empty((2, 5, 2, 2), dtype=np.float64)
        values[0] = offset + np.arange(5)[:, None, None]
        values[1] = offset + 1.0 + np.arange(5)[:, None, None]
        members[variable] = xr.DataArray(
            values,
            dims=("lead", "member", "latitude", "longitude"),
            coords={
                "lead": list(LEADS),
                "member": np.arange(5),
                "latitude": list(LATITUDE),
                "longitude": list(LONGITUDE),
            },
        )
        thresholds = np.empty((2, 4, 2, 2), dtype=np.float64)
        thresholds[0] = offset + np.arange(1, 5)[:, None, None]
        thresholds[1] = offset + 1.0 + np.arange(1, 5)[:, None, None]
        bounds[variable] = xr.DataArray(
            thresholds,
            dims=("lead", "boundary", "latitude", "longitude"),
            coords={
                "lead": list(LEADS),
                "boundary": [0.2, 0.4, 0.6, 0.8],
                "latitude": list(LATITUDE),
                "longitude": list(LONGITUDE),
            },
        )
    return members, bounds


def _land_fraction() -> xr.DataArray:
    return xr.DataArray(
        [[1.0, 0.25], [0.75, 1.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": list(LATITUDE), "longitude": list(LONGITUDE)},
    )


def test_anchor_and_target_contracts_use_all_three_variables() -> None:
    members, bounds = _weekly_members_and_bounds()
    anchors = build_multivariable_anchors(
        members, bounds, boundary_dim="boundary"
    )
    observations = {
        variable: variable_bounds.isel(boundary=1, drop=True)
        for variable, variable_bounds in bounds.items()
    }
    targets = build_multivariable_targets(
        observations, bounds, boundary_dim="boundary"
    )

    assert anchors.dims == (
        "variable",
        "lead",
        "quintile",
        "latitude",
        "longitude",
    )
    assert anchors.shape == (3, 2, 5, 2, 2)
    assert tuple(anchors["variable"].values) == QUEST_VARIABLES
    assert bool((anchors > 0.0).all())
    np.testing.assert_allclose(anchors.sum("quintile"), 1.0)

    assert targets.index.dims == ("variable", "lead", "latitude", "longitude")
    assert targets.probabilities.shape == anchors.shape
    # Equality with the second threshold enters the third (zero-based 2) bin.
    np.testing.assert_array_equal(targets.index, 2)
    assert bool(targets.valid_mask.all())

    with pytest.raises(ValueError, match="strictly positive"):
        build_multivariable_anchors(
            members, bounds, boundary_dim="boundary", smoothing=0.0
        )


def test_build_features_has_exact_38_channel_order_and_transforms() -> None:
    members, bounds = _weekly_members_and_bounds()
    anchors = build_multivariable_anchors(
        members, bounds, boundary_dim="boundary"
    )
    means = {
        "pr": np.zeros((2, 5)),
        "tas": np.full((2, 5), 280.0),
        "mslp": np.full((2, 5), 100_002.0),
    }
    stds = {variable: np.ones((2, 5)) for variable in QUEST_VARIABLES}

    features = build_multivariable_features(
        members,
        anchors,
        "2025-07-17",
        anchors.latitude,
        anchors.longitude,
        _land_fraction(),
        member_quantile_means=means,
        member_quantile_stds=stds,
    )

    assert MULTIVARIABLE_INPUT_CHANNELS == 38
    assert features.shape == (2, 38, 2, 2)
    assert tuple(features.feature.values) == MULTIVARIABLE_FEATURE_NAMES
    assert features.attrs["feature_contract"] == "quest_multivariable_38_channel_v1"
    assert features.sel(
        lead="D19-25", feature="pr_q50", latitude=0.0, longitude=0.0
    ).item() == pytest.approx(np.log1p(2.0))
    assert features.sel(
        lead="D19-25", feature="tas_q50", latitude=0.0, longitude=0.0
    ).item() == pytest.approx(0.0)
    assert features.sel(
        lead="D19-25", feature="mslp_q50", latitude=0.0, longitude=0.0
    ).item() == pytest.approx(0.0)
    assert features.sel(
        lead="D19-25", feature="pr_log_p_q1", latitude=0.0, longitude=0.0
    ).item() == pytest.approx(
        np.log(
            anchors.sel(
                variable="pr",
                lead="D19-25",
                quintile=0,
                latitude=0.0,
                longitude=0.0,
            ).item()
        )
    )
    np.testing.assert_array_equal(
        features.sel(feature="lead_flag").isel(latitude=0, longitude=0),
        [-1.0, 1.0],
    )


def test_variable_spatial_masks_are_distinct() -> None:
    members, bounds = _weekly_members_and_bounds()
    # This cell is arid for precipitation but remains eligible for tas/mslp.
    bounds["pr"].loc[{"latitude": 0.0, "longitude": 0.0}] = 0.0
    anchors = build_multivariable_anchors(
        members, bounds, boundary_dim="boundary"
    )
    valid = xr.ones_like(anchors.isel(quintile=0, drop=True), dtype=bool)
    valid.loc[
        {
            "variable": "mslp",
            "lead": "D19-25",
            "latitude": 60.0,
            "longitude": 90.0,
        }
    ] = False

    weights = build_multivariable_spatial_weights(
        anchors.latitude,
        _land_fraction(),
        bounds,
        valid_mask=valid,
        boundary_dim="boundary",
    )

    assert weights.dims == ("variable", "lead", "latitude", "longitude")
    # pr: arid cell and land<0.5 cell are excluded.
    assert weights.sel(variable="pr", latitude=0.0, longitude=0.0).max() == 0.0
    assert weights.sel(variable="pr", latitude=0.0, longitude=90.0).max() == 0.0
    # tas: aridity is irrelevant, but land<0.5 is excluded.
    assert weights.sel(variable="tas", latitude=0.0, longitude=0.0).min() > 0.0
    assert weights.sel(variable="tas", latitude=0.0, longitude=90.0).max() == 0.0
    # mslp: global, apart from the supplied observation-validity mask.
    assert weights.sel(
        variable="mslp", lead="D19-25", latitude=0.0, longitude=90.0
    ).item() > 0.0
    assert weights.sel(
        variable="mslp", lead="D19-25", latitude=60.0, longitude=90.0
    ).item() == 0.0


def test_labeled_normalization_is_reordered_by_lead_and_quantile() -> None:
    members, bounds = _weekly_members_and_bounds()
    anchors = build_multivariable_anchors(
        members, bounds, boundary_dim="boundary"
    )
    # Both axes are deliberately reversed/shuffled. The labels, not raw array
    # position, must determine which statistic is applied to each channel.
    tas_means = (
        members["tas"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9], dim="member")
        .isel(lead=[1, 0], quantile=[4, 2, 0, 3, 1], latitude=0, longitude=0)
        .transpose("lead", "quantile")
    )
    means = {
        "pr": np.zeros((2, 5)),
        "tas": tas_means,
        "mslp": np.zeros((2, 5)),
    }
    stds = {
        "pr": np.ones((2, 5)),
        "tas": xr.ones_like(tas_means),
        "mslp": np.ones((2, 5)),
    }

    features = build_multivariable_features(
        members,
        anchors,
        "2025-07-17",
        anchors.latitude,
        anchors.longitude,
        _land_fraction(),
        member_quantile_means=means,
        member_quantile_stds=stds,
    )

    tas_features = [f"tas_{suffix}" for suffix in ("q10", "q25", "q50", "q75", "q90")]
    np.testing.assert_allclose(
        features.sel(feature=tas_features),
        0.0,
        rtol=0.0,
        atol=1.0e-6,
    )

    means["tas"] = xr.DataArray(
        np.zeros(6),
        dims="quantile",
        coords={"quantile": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
    )
    with pytest.raises(ValueError, match="exactly five"):
        build_multivariable_features(
            members,
            anchors,
            "2025-07-17",
            anchors.latitude,
            anchors.longitude,
            _land_fraction(),
            member_quantile_means=means,
            member_quantile_stds=stds,
        )


def test_rps_normalizes_each_variable_before_equal_averaging() -> None:
    probabilities = torch.zeros(1, 3, 2, 5, 1, 2)
    probabilities[:, 0, :, 0] = 1.0  # perfect when the target is category 0
    probabilities[:, 1] = 0.2  # uniform: RPS = 1.2 for category 0
    probabilities[:, 2, :, 4] = 1.0  # worst category: RPS = 4.0
    target = torch.zeros(1, 3, 2, 1, 2, dtype=torch.long)
    weights = torch.ones(3, 2, 1, 2)
    # Unequal valid-cell counts must not let the global mslp grid dominate.
    weights[0, :, :, 1] = 0.0
    weights[1, 1, :, 1] = 0.0

    by_variable = multivariable_rps_by_variable(probabilities, target, weights)

    torch.testing.assert_close(by_variable, torch.tensor([0.0, 1.2, 4.0]))
    torch.testing.assert_close(
        multivariable_rps_loss(probabilities, target, weights),
        torch.tensor((0.0 + 1.2 + 4.0) / 3.0),
    )


def test_feature_builder_requires_train_stats_as_a_pair() -> None:
    members, bounds = _weekly_members_and_bounds()
    anchors = build_multivariable_anchors(
        members, bounds, boundary_dim="boundary"
    )

    with pytest.raises(ValueError, match="supplied together"):
        build_multivariable_features(
            members,
            anchors,
            "2025-07-17",
            anchors.latitude,
            anchors.longitude,
            _land_fraction(),
            member_quantile_means={
                variable: np.zeros((2, 5)) for variable in QUEST_VARIABLES
            },
        )

    with pytest.raises(ValueError, match="lead order"):
        build_multivariable_features(
            members,
            anchors.isel(lead=[1, 0]),
            "2025-07-17",
            anchors.latitude,
            anchors.longitude,
            _land_fraction(),
        )
