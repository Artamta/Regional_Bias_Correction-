"""Prototype contracts for joint global Quest probability calibration.

This module is deliberately separate from the mature precipitation-only
cache.  It defines one explicit 38-channel contract for the three gridded
Quest variables, builds their ensemble anchors and targets, and provides an
equal-variable RPS loss.  It does not reinterpret an old 18-channel cache as a
multi-variable cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xarray as xr

from data import (
    MEMBER_QUANTILES,
    QUEST_WINDOWS,
    member_quintile_probabilities,
    observation_categories,
    resolve_window,
    weekly_valid_dates,
)
from metrics import spatial_weights


QUEST_VARIABLES: tuple[str, ...] = ("pr", "tas", "mslp")
ANCHOR_FEATURE_SUFFIXES: tuple[str, ...] = tuple(
    f"log_p_q{index}" for index in range(1, 6)
)
MEMBER_QUANTILE_SUFFIXES: tuple[str, ...] = (
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
)
STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "sin_doy",
    "cos_doy",
    "lead_flag",
    "land_fraction",
)
MULTIVARIABLE_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature
    for variable in QUEST_VARIABLES
    for feature in (
        *(f"{variable}_{suffix}" for suffix in ANCHOR_FEATURE_SUFFIXES),
        *(f"{variable}_{suffix}" for suffix in MEMBER_QUANTILE_SUFFIXES),
    )
) + STATIC_FEATURE_NAMES
MULTIVARIABLE_INPUT_CHANNELS = len(MULTIVARIABLE_FEATURE_NAMES)


@dataclass(frozen=True)
class MultivariableTargets:
    """One-hot observations, integer targets, and validity for all variables."""

    probabilities: xr.DataArray
    index: xr.DataArray
    valid_mask: xr.DataArray


def _required_mapping(
    values: Mapping[str, xr.DataArray], *, name: str
) -> Mapping[str, xr.DataArray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by {QUEST_VARIABLES}")
    missing = [variable for variable in QUEST_VARIABLES if variable not in values]
    if missing:
        raise ValueError(f"{name} is missing variable(s): {missing}")
    return values


def _concat_exact(
    arrays: Sequence[xr.DataArray],
    *,
    variable_dim: str,
) -> xr.DataArray:
    aligned = xr.align(*arrays, join="exact")
    return xr.concat(
        aligned,
        dim=xr.IndexVariable(variable_dim, list(QUEST_VARIABLES)),
    )


def _validate_lead_order(array: xr.DataArray, *, lead_dim: str, name: str) -> None:
    if lead_dim not in array.dims or array.sizes[lead_dim] != 2:
        raise ValueError(f"{name} must contain exactly two lead windows")
    try:
        canonical = tuple(
            resolve_window(value.item() if isinstance(value, np.generic) else value)[0]
            for value in array[lead_dim].values
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} lead coordinates must identify D19-25 then D26-32"
        ) from error
    if canonical != tuple(QUEST_WINDOWS):
        raise ValueError(f"{name} lead order must be D19-25 then D26-32")


def build_multivariable_anchors(
    weekly_members: Mapping[str, xr.DataArray],
    climatology_bounds: Mapping[str, xr.DataArray],
    *,
    lead_dim: str = "lead",
    member_dim: str = "member",
    boundary_dim: str | None = None,
    category_dim: str = "quintile",
    variable_dim: str = "variable",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    smoothing: float = 0.5,
) -> xr.DataArray:
    """Build positive ensemble anchors with shape ``[variable,lead,5,lat,lon]``."""

    members_by_variable = _required_mapping(weekly_members, name="weekly_members")
    bounds_by_variable = _required_mapping(
        climatology_bounds, name="climatology_bounds"
    )
    alpha = float(smoothing)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("smoothing must be finite and strictly positive")
    anchors: list[xr.DataArray] = []
    for variable in QUEST_VARIABLES:
        anchor = member_quintile_probabilities(
            members_by_variable[variable],
            bounds_by_variable[variable],
            member_dim=member_dim,
            boundary_dim=boundary_dim,
            category_dim=category_dim,
            smoothing=alpha,
        )
        required = {lead_dim, category_dim, latitude_dim, longitude_dim}
        if not required.issubset(anchor.dims):
            raise ValueError(
                f"{variable} anchor must contain dimensions {sorted(required)}"
            )
        _validate_lead_order(anchor, lead_dim=lead_dim, name=f"{variable} anchor")
        anchors.append(
            anchor.transpose(lead_dim, category_dim, latitude_dim, longitude_dim)
        )

    result = _concat_exact(anchors, variable_dim=variable_dim).transpose(
        variable_dim, lead_dim, category_dim, latitude_dim, longitude_dim
    )
    result.name = "anchor_probability"
    result.attrs.update(
        {
            "variables": ",".join(QUEST_VARIABLES),
            "additive_smoothing": alpha,
            "schema": "quest_multivariable_probability_v1",
        }
    )
    return result.astype(np.float32)


def build_multivariable_targets(
    observations: Mapping[str, xr.DataArray],
    climatology_bounds: Mapping[str, xr.DataArray],
    *,
    lead_dim: str = "lead",
    boundary_dim: str | None = None,
    category_dim: str = "quintile",
    variable_dim: str = "variable",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
) -> MultivariableTargets:
    """Categorize observations for all variables using the upper-bin rule."""

    observations_by_variable = _required_mapping(observations, name="observations")
    bounds_by_variable = _required_mapping(
        climatology_bounds, name="climatology_bounds"
    )
    probabilities: list[xr.DataArray] = []
    indices: list[xr.DataArray] = []
    valid_masks: list[xr.DataArray] = []
    for variable in QUEST_VARIABLES:
        categorized = observation_categories(
            observations_by_variable[variable],
            bounds_by_variable[variable],
            boundary_dim=boundary_dim,
            category_dim=category_dim,
        )
        _validate_lead_order(
            categorized.index,
            lead_dim=lead_dim,
            name=f"{variable} observations",
        )
        probabilities.append(
            categorized.probabilities.transpose(
                lead_dim, category_dim, latitude_dim, longitude_dim
            )
        )
        indices.append(
            categorized.index.transpose(lead_dim, latitude_dim, longitude_dim)
        )
        valid_masks.append(
            categorized.valid_mask.transpose(
                lead_dim, latitude_dim, longitude_dim
            )
        )

    probability_array = _concat_exact(
        probabilities, variable_dim=variable_dim
    ).transpose(
        variable_dim, lead_dim, category_dim, latitude_dim, longitude_dim
    )
    index_array = _concat_exact(indices, variable_dim=variable_dim).transpose(
        variable_dim, lead_dim, latitude_dim, longitude_dim
    )
    valid_array = _concat_exact(valid_masks, variable_dim=variable_dim).transpose(
        variable_dim, lead_dim, latitude_dim, longitude_dim
    )
    probability_array.name = "observation_probability"
    index_array.name = "observation_category"
    valid_array.name = "valid_observation"
    return MultivariableTargets(probability_array, index_array, valid_array)


def _spatial_coordinate(
    coordinate: xr.DataArray | Sequence[float], *, dim: str
) -> xr.DataArray:
    if isinstance(coordinate, xr.DataArray):
        if coordinate.ndim != 1:
            raise ValueError(f"{dim} must be one-dimensional")
        if coordinate.dims != (dim,):
            if coordinate.name == dim:
                coordinate = coordinate.rename({coordinate.dims[0]: dim})
            else:
                raise ValueError(f"{dim} DataArray must use dimension {dim!r}")
        return coordinate.astype(np.float64)
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{dim} must be one-dimensional")
    return xr.DataArray(values, dims=(dim,), coords={dim: values}, name=dim)


def _normalization_arrays(
    variable: str,
    means: Mapping[str, Any] | None,
    stds: Mapping[str, Any] | None,
    *,
    lead_dim: str,
    feature_dim: str,
    lead_coordinate: xr.DataArray,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    if (means is None) != (stds is None):
        raise ValueError("member_quantile_means and member_quantile_stds must be supplied together")
    if means is None or stds is None:
        return None, None
    if variable not in means or variable not in stds:
        raise ValueError(f"normalization mappings must contain {variable!r}")

    names = [f"{variable}_{suffix}" for suffix in MEMBER_QUANTILE_SUFFIXES]

    def coerce(value: Any, name: str, *, positive: bool) -> xr.DataArray:
        if isinstance(value, xr.DataArray):
            if value.ndim not in (1, 2):
                raise ValueError(
                    f"{name}[{variable!r}] DataArray must have one or two dimensions"
                )
            if value.ndim == 2:
                if lead_dim not in value.dims:
                    raise ValueError(
                        f"{name}[{variable!r}] DataArray must use {lead_dim!r}"
                    )
                quantile_dims = [dim for dim in value.dims if dim != lead_dim]
                if len(quantile_dims) != 1:
                    raise ValueError(f"{name}[{variable!r}] has ambiguous dimensions")
                quantile_dim = quantile_dims[0]
                if lead_dim not in value.coords:
                    raise ValueError(
                        f"{name}[{variable!r}] must label its lead coordinate"
                    )
                canonical_leads: list[str] = []
                try:
                    for label in value[lead_dim].values:
                        scalar = label.item() if isinstance(label, np.generic) else label
                        canonical_leads.append(resolve_window(scalar)[0])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name}[{variable!r}] has invalid lead labels"
                    ) from error
                if sorted(canonical_leads) != sorted(QUEST_WINDOWS):
                    raise ValueError(
                        f"{name}[{variable!r}] must label D19-25 and D26-32 once"
                    )
                lead_order = [canonical_leads.index(window) for window in QUEST_WINDOWS]
                value = value.isel({lead_dim: lead_order}).transpose(
                    lead_dim, quantile_dim
                )
            else:
                quantile_dim = value.dims[0]

            if quantile_dim not in value.coords:
                raise ValueError(
                    f"{name}[{variable!r}] must label its member quantiles"
                )
            labels = np.asarray(value[quantile_dim].values)
            if len(labels) != 5:
                raise ValueError(
                    f"{name}[{variable!r}] must contain exactly five member quantiles"
                )
            if np.issubdtype(labels.dtype, np.number):
                quantile_order: list[int] = []
                for requested in MEMBER_QUANTILES:
                    matches = np.flatnonzero(
                        np.isclose(labels.astype(np.float64), requested)
                    )
                    if len(matches) != 1:
                        raise ValueError(
                            f"{name}[{variable!r}] must label q10,q25,q50,q75,q90"
                        )
                    quantile_order.append(int(matches[0]))
            else:
                text_labels = [str(label) for label in labels]
                accepted_orders = (
                    list(MEMBER_QUANTILE_SUFFIXES),
                    names,
                )
                matching = [
                    expected
                    for expected in accepted_orders
                    if len(set(text_labels)) == 5 and set(text_labels) == set(expected)
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"{name}[{variable!r}] must label q10,q25,q50,q75,q90"
                    )
                quantile_order = [text_labels.index(label) for label in matching[0]]
            value = value.isel({quantile_dim: quantile_order})
            if value.ndim == 2:
                value = value.transpose(lead_dim, quantile_dim)
            else:
                value = value.transpose(quantile_dim)
            array = np.asarray(value.values, dtype=np.float64)
        else:
            array = np.asarray(value, dtype=np.float64)
        if array.shape == (5,):
            result = xr.DataArray(
                array,
                dims=(feature_dim,),
                coords={feature_dim: names},
            )
        elif array.shape == (2, 5):
            result = xr.DataArray(
                array,
                dims=(lead_dim, feature_dim),
                coords={lead_dim: lead_coordinate, feature_dim: names},
            )
        else:
            raise ValueError(
                f"{name}[{variable!r}] must have shape [5] or [2, 5]"
            )
        if not bool(np.isfinite(result).all()):
            raise ValueError(f"{name}[{variable!r}] must be finite")
        if positive and bool((result <= 0.0).any()):
            raise ValueError(f"{name}[{variable!r}] must be positive")
        return result

    return (
        coerce(means[variable], "member_quantile_means", positive=False),
        coerce(stds[variable], "member_quantile_stds", positive=True),
    )


def build_multivariable_features(
    weekly_members: Mapping[str, xr.DataArray],
    p0: xr.DataArray,
    init_date: Any,
    latitude: xr.DataArray | Sequence[float],
    longitude: xr.DataArray | Sequence[float],
    land_fraction: xr.DataArray,
    *,
    member_quantile_means: Mapping[str, Any] | None = None,
    member_quantile_stds: Mapping[str, Any] | None = None,
    lead_dim: str = "lead",
    member_dim: str = "member",
    category_dim: str = "quintile",
    variable_dim: str = "variable",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    feature_dim: str = "feature",
    probability_floor: float = 1.0e-8,
) -> xr.DataArray:
    """Build the fixed 38-channel joint feature tensor.

    The five precipitation member quantiles are computed after ``log1p``;
    temperature and pressure quantiles remain in their native K and Pa units.
    Training must supply statistics computed only from the training split.
    """

    members_by_variable = _required_mapping(weekly_members, name="weekly_members")
    if not isinstance(p0, xr.DataArray):
        raise TypeError("p0 must be an xarray.DataArray")
    required_anchor_dims = {
        variable_dim,
        lead_dim,
        category_dim,
        latitude_dim,
        longitude_dim,
    }
    if set(p0.dims) != required_anchor_dims:
        raise ValueError(f"p0 must have exactly dimensions {sorted(required_anchor_dims)}")
    if tuple(str(value) for value in p0[variable_dim].values) != QUEST_VARIABLES:
        raise ValueError(f"p0 variable order must be exactly {QUEST_VARIABLES}")
    if p0.sizes[lead_dim] != 2 or p0.sizes[category_dim] != 5:
        raise ValueError("p0 must contain two leads and five categories")
    _validate_lead_order(p0, lead_dim=lead_dim, name="p0")
    if not bool(np.isfinite(p0).all()) or bool((p0 <= 0.0).any()):
        raise ValueError("p0 must contain finite, strictly positive probabilities")
    if not bool(
        np.isclose(p0.sum(category_dim), 1.0, rtol=1.0e-6, atol=1.0e-7).all()
    ):
        raise ValueError("p0 probabilities must sum to one at every cell")

    lat = _spatial_coordinate(latitude, dim=latitude_dim)
    lon = _spatial_coordinate(longitude, dim=longitude_dim)
    if not p0[latitude_dim].equals(lat) or not p0[longitude_dim].equals(lon):
        raise ValueError("explicit latitude/longitude must exactly match p0")
    if not isinstance(land_fraction, xr.DataArray):
        raise TypeError("land_fraction must be an xarray.DataArray")
    spatial_template = p0.isel(
        {variable_dim: 0, lead_dim: 0, category_dim: 0}, drop=True
    )
    land, spatial_template = xr.align(land_fraction, spatial_template, join="exact")
    land = land.broadcast_like(spatial_template).astype(np.float64)
    if not bool(np.isfinite(land).all()) or bool(
        ((land < 0.0) | (land > 1.0)).any()
    ):
        raise ValueError("land_fraction must be finite and lie in [0, 1]")
    floor = float(probability_floor)
    if not np.isfinite(floor) or not 0.0 < floor < 0.2:
        raise ValueError("probability_floor must be finite and in (0, 0.2)")

    feature_blocks: list[xr.DataArray] = []
    for variable in QUEST_VARIABLES:
        members = members_by_variable[variable]
        if not isinstance(members, xr.DataArray):
            raise TypeError(f"weekly_members[{variable!r}] must be an xarray.DataArray")
        required_member_dims = {lead_dim, member_dim, latitude_dim, longitude_dim}
        if set(members.dims) != required_member_dims:
            raise ValueError(
                f"weekly_members[{variable!r}] must have exactly dimensions "
                f"{sorted(required_member_dims)}"
            )
        if members.sizes[lead_dim] != 2 or members.sizes[member_dim] < 1:
            raise ValueError(f"{variable} members need two leads and at least one member")
        variable_anchor = p0.sel({variable_dim: variable}, drop=True)
        members, variable_anchor = xr.align(
            members,
            variable_anchor,
            join="exact",
            exclude={member_dim, category_dim},
        )
        if not bool(np.isfinite(members).all()):
            raise ValueError(f"weekly_members[{variable!r}] contains non-finite values")
        if variable == "pr" and bool((members < 0.0).any()):
            raise ValueError("weekly precipitation must be non-negative")

        anchor_names = [
            f"{variable}_{suffix}" for suffix in ANCHOR_FEATURE_SUFFIXES
        ]
        log_anchor = np.log(variable_anchor.clip(min=floor))
        log_anchor = log_anchor.assign_coords({category_dim: anchor_names}).rename(
            {category_dim: feature_dim}
        )

        transformed_members = np.log1p(members) if variable == "pr" else members
        quantile_names = [
            f"{variable}_{suffix}" for suffix in MEMBER_QUANTILE_SUFFIXES
        ]
        member_quantiles = transformed_members.quantile(
            MEMBER_QUANTILES, dim=member_dim, skipna=False
        )
        member_quantiles = member_quantiles.assign_coords(
            quantile=quantile_names
        ).rename({"quantile": feature_dim})
        means, stds = _normalization_arrays(
            variable,
            member_quantile_means,
            member_quantile_stds,
            lead_dim=lead_dim,
            feature_dim=feature_dim,
            lead_coordinate=p0[lead_dim],
        )
        if means is not None and stds is not None:
            member_quantiles = (member_quantiles - means) / stds
        feature_blocks.extend((log_anchor, member_quantiles))

    lat_radians = np.deg2rad(lat)
    lon_radians = np.deg2rad(lon)
    sin_lat = np.sin(lat_radians).broadcast_like(spatial_template)
    cos_lat = np.cos(lat_radians).broadcast_like(spatial_template)
    sin_lon = np.sin(lon_radians).broadcast_like(spatial_template)
    cos_lon = np.cos(lon_radians).broadcast_like(spatial_template)
    lead_coordinate = p0[lead_dim]
    starts = [
        weekly_valid_dates(init_date, window)[0] for window in QUEST_WINDOWS
    ]
    phases = np.asarray(
        [2.0 * np.pi * (date.dayofyear - 1) / 365.2425 for date in starts],
        dtype=np.float64,
    )
    lead_template = p0.isel({variable_dim: 0, category_dim: 0}, drop=True)

    def repeat_spatial(field: xr.DataArray) -> xr.DataArray:
        return field.expand_dims({lead_dim: lead_coordinate}).transpose(
            lead_dim, latitude_dim, longitude_dim
        )

    static_channels = [
        repeat_spatial(sin_lat),
        repeat_spatial(cos_lat),
        repeat_spatial(sin_lon),
        repeat_spatial(cos_lon),
        xr.DataArray(
            np.sin(phases), dims=(lead_dim,), coords={lead_dim: lead_coordinate}
        ).broadcast_like(lead_template),
        xr.DataArray(
            np.cos(phases), dims=(lead_dim,), coords={lead_dim: lead_coordinate}
        ).broadcast_like(lead_template),
        xr.DataArray(
            [-1.0, 1.0], dims=(lead_dim,), coords={lead_dim: lead_coordinate}
        ).broadcast_like(lead_template),
        repeat_spatial(land),
    ]
    static = xr.concat(
        static_channels,
        dim=xr.IndexVariable(feature_dim, list(STATIC_FEATURE_NAMES)),
    ).transpose(lead_dim, feature_dim, latitude_dim, longitude_dim)
    feature_blocks.append(static)

    features = xr.concat(feature_blocks, dim=feature_dim).transpose(
        lead_dim, feature_dim, latitude_dim, longitude_dim
    )
    features = features.assign_coords(
        {feature_dim: list(MULTIVARIABLE_FEATURE_NAMES)}
    )
    features.name = "features"
    features.attrs.update(
        {
            "feature_contract": "quest_multivariable_38_channel_v1",
            "variable_order": ",".join(QUEST_VARIABLES),
            "member_transforms": "pr=log1p;tas=identity;mslp=identity",
            "normalization": (
                "training-only supplied statistics"
                if member_quantile_means is not None
                else "not applied"
            ),
            "lead_flag": "-1=D19-25, +1=D26-32",
        }
    )
    return features.astype(np.float32)


def _boundary_dimension(bounds: xr.DataArray, requested: str | None) -> str:
    if requested is not None:
        if requested not in bounds.dims:
            raise ValueError(f"precipitation bounds has no {requested!r} dimension")
        return requested
    candidates = [
        name for name in ("quantile", "quintile", "boundary") if name in bounds.dims
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("could not determine precipitation boundary dimension")


def build_multivariable_spatial_weights(
    latitude: xr.DataArray | Sequence[float],
    land_fraction: xr.DataArray,
    climatology_bounds: Mapping[str, xr.DataArray],
    *,
    valid_mask: xr.DataArray | None = None,
    boundary_dim: str | None = None,
    variable_dim: str = "variable",
    lead_dim: str = "lead",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
) -> xr.DataArray:
    """Return zero-masked cosine weights for each variable and lead.

    Precipitation uses land >= 0.5 and wettest-boundary > 0, temperature uses
    land >= 0.5, and pressure is global.  A supplied target validity mask is
    applied after those domain rules.
    """

    bounds = _required_mapping(climatology_bounds, name="climatology_bounds")
    precipitation_bounds = bounds["pr"]
    if not isinstance(precipitation_bounds, xr.DataArray):
        raise TypeError("climatology_bounds['pr'] must be an xarray.DataArray")
    dim = _boundary_dimension(precipitation_bounds, boundary_dim)
    if precipitation_bounds.sizes[dim] != 4:
        raise ValueError("precipitation bounds must contain four thresholds")
    wettest = precipitation_bounds.isel({dim: -1}, drop=True)
    required_template_dims = {lead_dim, latitude_dim, longitude_dim}
    if set(wettest.dims) != required_template_dims:
        raise ValueError(
            "precipitation wettest boundary must have lead, latitude and longitude"
        )
    _validate_lead_order(wettest, lead_dim=lead_dim, name="precipitation bounds")

    if valid_mask is not None:
        if not isinstance(valid_mask, xr.DataArray):
            raise TypeError("valid_mask must be an xarray.DataArray")
        required_valid_dims = required_template_dims | {variable_dim}
        if set(valid_mask.dims) != required_valid_dims:
            raise ValueError(f"valid_mask must have dimensions {sorted(required_valid_dims)}")
        if tuple(str(value) for value in valid_mask[variable_dim].values) != QUEST_VARIABLES:
            raise ValueError(f"valid_mask variable order must be exactly {QUEST_VARIABLES}")

    per_variable: list[xr.DataArray] = []
    for variable in QUEST_VARIABLES:
        variable_valid = (
            valid_mask.sel({variable_dim: variable}, drop=True)
            if valid_mask is not None
            else None
        )
        weights = spatial_weights(
            latitude,
            variable,
            land_fraction=land_fraction if variable in {"pr", "tas"} else None,
            wettest_boundary=wettest if variable == "pr" else None,
            valid_mask=variable_valid,
            latitude_dim=latitude_dim,
        )
        weights, template = xr.align(weights, wettest, join="exact")
        weights = weights.broadcast_like(template).transpose(
            lead_dim, latitude_dim, longitude_dim
        )
        per_variable.append(weights.fillna(0.0))

    result = _concat_exact(per_variable, variable_dim=variable_dim).transpose(
        variable_dim, lead_dim, latitude_dim, longitude_dim
    )
    result.name = "spatial_weight"
    result.attrs.update(
        {
            "variable_normalization": "normalize each variable before equal averaging",
            "masked_cells": "zero weight",
        }
    )
    return result.astype(np.float32)


def multivariable_rps_by_variable(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return one independently normalized RPS value per target variable."""

    if probabilities.ndim != 6 or probabilities.shape[3] != 5:
        raise ValueError(
            "probabilities must be [batch, variable, period, 5, lat, lon]"
        )
    expected_target = probabilities.shape[:3] + probabilities.shape[-2:]
    if tuple(target.shape) != tuple(expected_target):
        raise ValueError("target does not match probability cases, variables, periods and grid")
    if target.is_floating_point():
        raise TypeError("target must be an integer tensor")
    if target.device != probabilities.device:
        raise ValueError("target and probabilities must be on the same device")
    variables, periods = probabilities.shape[1:3]
    height, width = probabilities.shape[-2:]
    if weights.ndim == 3 and tuple(weights.shape) == (variables, height, width):
        weights = weights[:, None].expand(variables, periods, height, width)
    elif weights.ndim != 4 or tuple(weights.shape) != (
        variables,
        periods,
        height,
        width,
    ):
        raise ValueError("weights must be [variable,period,lat,lon] or [variable,lat,lon]")
    if not probabilities.is_floating_point() or not weights.is_floating_point():
        raise TypeError("probabilities and weights must be floating-point tensors")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool((probabilities < 0.0).any()):
        raise ValueError("probabilities must be non-negative")
    if not bool(
        torch.allclose(
            probabilities.sum(dim=3),
            torch.ones_like(probabilities[:, :, :, 0]),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    ):
        raise ValueError("probabilities must sum to one")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("weights must be finite and non-negative")

    valid = (target >= 0) & (target <= 4)
    categories = torch.arange(5, device=target.device).view(1, 1, 1, 5, 1, 1)
    observed_cdf = (target.unsqueeze(3) <= categories).to(probabilities.dtype)
    forecast_cdf = probabilities.cumsum(dim=3)
    cell_rps = ((forecast_cdf - observed_cdf) ** 2).sum(dim=3)
    combined = weights.to(probabilities.device).unsqueeze(0) * valid.to(
        probabilities.dtype
    )
    reduction_dims = (0, 2, 3, 4)
    numerator = (cell_rps * combined).sum(dim=reduction_dims)
    denominator = combined.sum(dim=reduction_dims)
    if bool((denominator <= 0.0).any()):
        raise ValueError("one or more variables contain no valid scoring cells")
    return numerator / denominator


def multivariable_rps_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return the equal-variable mean of independently normalized RPS values."""

    return multivariable_rps_by_variable(probabilities, target, weights).mean()


__all__ = [
    "MULTIVARIABLE_FEATURE_NAMES",
    "MULTIVARIABLE_INPUT_CHANNELS",
    "MultivariableTargets",
    "QUEST_VARIABLES",
    "STATIC_FEATURE_NAMES",
    "build_multivariable_anchors",
    "build_multivariable_features",
    "build_multivariable_spatial_weights",
    "build_multivariable_targets",
    "multivariable_rps_by_variable",
    "multivariable_rps_loss",
]
