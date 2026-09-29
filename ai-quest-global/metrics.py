"""Official-style ranked-probability metrics for AI Weather Quest fields.

The important aggregation rule is implemented directly: RPSS is one minus the
ratio of weighted RPS sums over one joint mask.  It is not an average of
grid-cell RPSS values, which would give a different answer whenever the
climatological RPS varies between observed categories.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import numpy as np
import xarray as xr


PROBABILITY_DIM_CANDIDATES = ("quintile", "category", "quantile")
LAND_VARIABLES = {
    "tas",
    "t2m",
    "temperature",
    "pr",
    "tp",
    "precip",
    "precipitation",
    "rainfall",
}
PRECIPITATION_VARIABLES = {"pr", "tp", "precip", "precipitation", "rainfall"}
GLOBAL_VARIABLES = {"mslp", "msl", "mean_sea_level_pressure"}


def _probability_dim(array: xr.DataArray, requested: str | None) -> str:
    if requested is not None:
        if requested not in array.dims:
            raise ValueError(f"probability array has no {requested!r} dimension")
        return requested
    candidates = [name for name in PROBABILITY_DIM_CANDIDATES if name in array.dims]
    if len(candidates) != 1:
        raise ValueError(
            "could not uniquely determine probability dimension; pass category_dim"
        )
    return candidates[0]


def _scalar_bool(value: xr.DataArray | np.ndarray | bool) -> bool:
    if isinstance(value, xr.DataArray):
        value = value.compute().values
    return bool(np.asarray(value).item())


def _validate_probability_array(
    probabilities: xr.DataArray,
    *,
    category_dim: str,
    name: str,
    require_five: bool = True,
) -> xr.DataArray:
    if not isinstance(probabilities, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    if require_five and probabilities.sizes[category_dim] != 5:
        raise ValueError(f"{name} must contain exactly five ordered categories")
    if _scalar_bool(np.isinf(probabilities).any()):
        raise ValueError(f"{name} contains infinite values")

    finite = np.isfinite(probabilities)
    any_finite = finite.any(category_dim)
    all_finite = finite.all(category_dim)
    if _scalar_bool((any_finite & ~all_finite).any()):
        raise ValueError(
            f"{name} has partially missing category vectors; use either five "
            "finite probabilities or five NaNs at a cell"
        )
    finite_values = probabilities.where(all_finite)
    if _scalar_bool(((finite_values < -1.0e-10) | (finite_values > 1.0 + 1.0e-10)).any()):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    sums = finite_values.sum(category_dim, skipna=False)
    bad_sum = all_finite & ~xr.apply_ufunc(
        np.isclose,
        sums,
        1.0,
        kwargs={"rtol": 1.0e-6, "atol": 1.0e-7},
        dask="allowed",
    )
    if _scalar_bool(bad_sum.any()):
        raise ValueError(f"{name} probabilities must sum to one at every valid cell")
    return all_finite


def _align_probability_arrays(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    category_dim: str | None,
) -> tuple[xr.DataArray, xr.DataArray, str, xr.DataArray]:
    forecast_dim = _probability_dim(forecast, category_dim)
    observed_dim = _probability_dim(
        observed, forecast_dim if forecast_dim in observed.dims else None
    )
    if observed_dim != forecast_dim:
        observed = observed.rename({observed_dim: forecast_dim})
    if forecast.sizes[forecast_dim] != observed.sizes[forecast_dim]:
        raise ValueError("forecast and observation category counts differ")

    # Category labels are presentation metadata; RPS depends on their order.
    observed = observed.assign_coords({forecast_dim: forecast[forecast_dim]})
    forecast, observed = xr.align(forecast, observed, join="exact")
    forecast_valid = _validate_probability_array(
        forecast, category_dim=forecast_dim, name="forecast_probabilities"
    )
    observed_valid = _validate_probability_array(
        observed, category_dim=forecast_dim, name="observed_probabilities"
    )
    return forecast, observed, forecast_dim, forecast_valid & observed_valid


def ranked_probability_score(
    forecast_probabilities: xr.DataArray,
    observed_probabilities: xr.DataArray,
    *,
    category_dim: str | None = None,
) -> xr.DataArray:
    """Return RPS at every non-category coordinate.

    The squared difference of cumulative probabilities is summed over all five
    ordered categories, matching the official evaluator.  For normalized
    vectors the fifth term is zero, but retaining it makes the implementation
    and audit trail identical to the definition used by the Quest package.
    """

    forecast, observed, dim, valid = _align_probability_arrays(
        forecast_probabilities, observed_probabilities, category_dim
    )
    forecast_cumulative = forecast.cumsum(dim, skipna=False)
    observed_cumulative = observed.cumsum(dim, skipna=False)
    score = ((forecast_cumulative - observed_cumulative) ** 2).sum(
        dim, skipna=False
    )
    score = score.where(valid)
    score.name = "ranked_probability_score"
    return score


def climatological_probabilities_like(
    probabilities: xr.DataArray,
    *,
    category_dim: str | None = None,
) -> xr.DataArray:
    """Return uniform categorical climatology with the same shape/coordinates."""

    dim = _probability_dim(probabilities, category_dim)
    count = probabilities.sizes[dim]
    if count < 2:
        raise ValueError("at least two ordered categories are required")
    result = xr.full_like(probabilities, 1.0 / count, dtype=np.float64)
    result.name = "climatological_probability"
    return result


def _canonical_variable(variable: str) -> str:
    key = variable.strip().lower()
    if key in PRECIPITATION_VARIABLES:
        return "pr"
    if key in LAND_VARIABLES:
        return "tas"
    if key in GLOBAL_VARIABLES:
        return "mslp"
    raise ValueError("variable must identify pr, tas, or mslp")


def _coerce_latitude(
    latitude: xr.DataArray | Sequence[float],
    latitude_dim: str,
) -> xr.DataArray:
    if isinstance(latitude, xr.DataArray):
        if latitude.ndim != 1:
            raise ValueError("latitude must be one-dimensional")
        if latitude.dims != (latitude_dim,):
            if latitude.name == latitude_dim:
                latitude = latitude.rename({latitude.dims[0]: latitude_dim})
            else:
                raise ValueError(
                    f"latitude DataArray must use dimension {latitude_dim!r}"
                )
        result = latitude.astype(np.float64)
    else:
        values = np.asarray(latitude, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("latitude must be one-dimensional")
        result = xr.DataArray(
            values,
            dims=(latitude_dim,),
            coords={latitude_dim: values},
            name=latitude_dim,
        )
    if not _scalar_bool(np.isfinite(result).all()):
        raise ValueError("latitude must be finite")
    if _scalar_bool(((result < -90.0) | (result > 90.0)).any()):
        raise ValueError("latitude must lie in [-90, 90]")
    return result


def spatial_weights(
    latitude: xr.DataArray | Sequence[float],
    variable: str,
    *,
    land_fraction: xr.DataArray | None = None,
    wettest_boundary: xr.DataArray | None = None,
    valid_mask: xr.DataArray | None = None,
    latitude_dim: str = "latitude",
) -> xr.DataArray:
    """Build cosine-latitude weights and the official spatial eligibility mask.

    ``mslp`` is global.  ``tas`` and ``pr`` retain cells with land fraction at
    least 0.5.  For ``pr``, cells whose wettest (0.8) climatological boundary is
    zero are additionally excluded as arid.  Missing masks fail closed.
    """

    canonical = _canonical_variable(variable)
    lat = _coerce_latitude(latitude, latitude_dim)
    weights = np.cos(np.deg2rad(lat)).clip(min=0.0)
    weights.name = "spatial_weight"

    mask: xr.DataArray | bool = True
    if canonical in {"tas", "pr"}:
        if land_fraction is None:
            raise ValueError(f"land_fraction is required for {canonical} scoring")
        if not isinstance(land_fraction, xr.DataArray):
            raise TypeError("land_fraction must be an xarray.DataArray")
        finite_land = np.isfinite(land_fraction)
        if _scalar_bool(
            (((land_fraction < 0.0) | (land_fraction > 1.0)) & finite_land).any()
        ):
            raise ValueError("land_fraction values must lie in [0, 1]")
        mask = finite_land & (land_fraction >= 0.5)

    if canonical == "pr":
        if wettest_boundary is None:
            raise ValueError("wettest_boundary is required for precipitation scoring")
        if not isinstance(wettest_boundary, xr.DataArray):
            raise TypeError("wettest_boundary must be an xarray.DataArray")
        wettest_valid = np.isfinite(wettest_boundary) & (wettest_boundary > 0.0)
        if isinstance(mask, xr.DataArray):
            mask, wettest_valid = xr.align(mask, wettest_valid, join="exact")
        mask = mask & wettest_valid

    if valid_mask is not None:
        if not isinstance(valid_mask, xr.DataArray):
            raise TypeError("valid_mask must be an xarray.DataArray")
        supplied_valid = valid_mask.fillna(False).astype(bool)
        if isinstance(mask, xr.DataArray):
            mask, supplied_valid = xr.align(mask, supplied_valid, join="exact")
        mask = mask & supplied_valid

    if isinstance(mask, xr.DataArray):
        weights, mask = xr.align(weights, mask, join="exact")
        weights, mask = xr.broadcast(weights, mask)
        weights = weights.where(mask)
    weights.attrs.update(
        {
            "area_weighting": "cos(latitude)",
            "variable_mask": canonical,
            "land_fraction_threshold": 0.5 if canonical in {"tas", "pr"} else "not_applied",
            "arid_rule": "wettest_boundary > 0" if canonical == "pr" else "not_applied",
        }
    )
    return weights


def weighted_mean(
    values: xr.DataArray,
    weights: xr.DataArray,
    *,
    dims: str | Iterable[str],
) -> xr.DataArray:
    """Return a true weighted mean over the joint finite mask."""

    if not isinstance(values, xr.DataArray) or not isinstance(weights, xr.DataArray):
        raise TypeError("values and weights must be xarray.DataArray objects")
    dimensions = (dims,) if isinstance(dims, str) else tuple(dims)
    if not dimensions:
        raise ValueError("dims must contain at least one dimension")
    absent = [dim for dim in dimensions if dim not in values.dims]
    if absent:
        raise ValueError(f"values has no reduction dimension(s): {absent}")
    if _scalar_bool(((weights < 0.0) & np.isfinite(weights)).any()):
        raise ValueError("weights must be non-negative")

    values, weights = xr.align(values, weights, join="exact")
    values, weights = xr.broadcast(values, weights)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    numerator = (values.where(valid, 0.0) * weights.where(valid, 0.0)).sum(
        dimensions, skipna=False
    )
    denominator = weights.where(valid, 0.0).sum(dimensions, skipna=False)
    return (numerator / denominator).where(denominator > 0.0)


def weighted_ranked_probability_score(
    forecast_probabilities: xr.DataArray,
    observed_probabilities: xr.DataArray,
    weights: xr.DataArray,
    *,
    reduce_dims: str | Iterable[str] = ("latitude", "longitude"),
    category_dim: str | None = None,
) -> xr.DataArray:
    """Return area/mask-weighted mean RPS over the requested dimensions."""

    cell_score = ranked_probability_score(
        forecast_probabilities,
        observed_probabilities,
        category_dim=category_dim,
    )
    result = weighted_mean(cell_score, weights, dims=reduce_dims)
    result.name = "weighted_ranked_probability_score"
    return result


def ranked_probability_skill_score(
    forecast_probabilities: xr.DataArray,
    observed_probabilities: xr.DataArray,
    weights: xr.DataArray,
    *,
    reduce_dims: str | Iterable[str] = ("latitude", "longitude"),
    category_dim: str | None = None,
    reference_probabilities: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return RPSS as a ratio of weighted RPS sums over one joint mask.

    Extra dimensions not listed in ``reduce_dims`` (for example ``lead`` or
    ``init``) are retained.  Include them explicitly when one aggregate score
    across cases is wanted.
    """

    if not isinstance(weights, xr.DataArray):
        raise TypeError("weights must be an xarray.DataArray")
    dimensions = (reduce_dims,) if isinstance(reduce_dims, str) else tuple(reduce_dims)
    if not dimensions:
        raise ValueError("reduce_dims must contain at least one dimension")

    forecast, observed, dim, _ = _align_probability_arrays(
        forecast_probabilities, observed_probabilities, category_dim
    )
    if reference_probabilities is None:
        reference = climatological_probabilities_like(forecast, category_dim=dim)
    else:
        reference = reference_probabilities
        reference_dim = _probability_dim(reference, dim if dim in reference.dims else None)
        if reference_dim != dim:
            reference = reference.rename({reference_dim: dim})
        reference = reference.assign_coords({dim: forecast[dim]})
        reference, _ = xr.align(reference, forecast, join="exact")
        _validate_probability_array(
            reference, category_dim=dim, name="reference_probabilities"
        )

    forecast_rps = ranked_probability_score(forecast, observed, category_dim=dim)
    reference_rps = ranked_probability_score(reference, observed, category_dim=dim)
    forecast_rps, reference_rps, weights = xr.align(
        forecast_rps, reference_rps, weights, join="exact"
    )
    forecast_rps, reference_rps, weights = xr.broadcast(
        forecast_rps, reference_rps, weights
    )
    absent = [name for name in dimensions if name not in forecast_rps.dims]
    if absent:
        raise ValueError(f"RPS fields have no reduction dimension(s): {absent}")
    if _scalar_bool(((weights < 0.0) & np.isfinite(weights)).any()):
        raise ValueError("weights must be non-negative")

    joint_valid = (
        np.isfinite(forecast_rps)
        & np.isfinite(reference_rps)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    effective_weights = weights.where(joint_valid, 0.0)
    forecast_sum = (forecast_rps.where(joint_valid, 0.0) * effective_weights).sum(
        dimensions, skipna=False
    )
    reference_sum = (reference_rps.where(joint_valid, 0.0) * effective_weights).sum(
        dimensions, skipna=False
    )
    weight_sum = effective_weights.sum(dimensions, skipna=False)
    score = 1.0 - forecast_sum / reference_sum
    score = score.where((weight_sum > 0.0) & (reference_sum > 0.0))
    score.name = "ranked_probability_skill_score"
    score.attrs.update(
        {
            "reference": "uniform climatological probabilities"
            if reference_probabilities is None
            else "provided probabilities",
            "aggregation": "1 - weighted_sum(forecast_RPS) / weighted_sum(reference_RPS)",
        }
    )
    return score


def quest_rpss(
    forecast_probabilities: xr.DataArray,
    observed_probabilities: xr.DataArray,
    latitude: xr.DataArray | Sequence[float],
    variable: str,
    *,
    land_fraction: xr.DataArray | None = None,
    wettest_boundary: xr.DataArray | None = None,
    valid_mask: xr.DataArray | None = None,
    reduce_dims: str | Iterable[str] = ("latitude", "longitude"),
    category_dim: str | None = None,
    latitude_dim: str = "latitude",
) -> xr.DataArray:
    """Convenience wrapper for official masking, weighting, RPS, and RPSS."""

    weights = spatial_weights(
        latitude,
        variable,
        land_fraction=land_fraction,
        wettest_boundary=wettest_boundary,
        valid_mask=valid_mask,
        latitude_dim=latitude_dim,
    )
    return ranked_probability_skill_score(
        forecast_probabilities,
        observed_probabilities,
        weights,
        reduce_dims=reduce_dims,
        category_dim=category_dim,
    )


__all__ = [
    "climatological_probabilities_like",
    "quest_rpss",
    "ranked_probability_score",
    "ranked_probability_skill_score",
    "spatial_weights",
    "weighted_mean",
    "weighted_ranked_probability_score",
]
