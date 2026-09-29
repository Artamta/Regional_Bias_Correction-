"""Data contracts for the global AI Weather Quest precipitation experiment.

The helpers in this module deliberately make the competition's easy-to-miss
conventions explicit:

* forecast day 1 is the initialization date, so D19 starts at ``init + 18d``;
* precipitation is converted from declared metadata, never guessed by scale;
* a value on a climatological boundary belongs to the *upper* category;
* climatological thresholds use only the preceding 20 target years; and
* the model feature order is fixed and checked at construction time.

All spatial operations retain xarray coordinates.  Invalid climatology cells
(including all-equal precipitation boundaries) are represented by NaN and are
therefore available to the metric code as an explicit mask.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta


QUEST_WINDOWS: dict[str, tuple[int, int]] = {
    "D19-25": (19, 25),
    "D26-32": (26, 32),
}
WINDOW_ALIASES: dict[Any, str] = {
    0: "D19-25",
    1: "D26-32",
    "week3": "D19-25",
    "week4": "D26-32",
    "w3": "D19-25",
    "w4": "D26-32",
    "d19-25": "D19-25",
    "d26-32": "D26-32",
}
QUINTILE_LEVELS = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
MEMBER_QUANTILES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
CATEGORY_VALUES = np.arange(5, dtype=np.int8)

FEATURE_NAMES: tuple[str, ...] = (
    "log_p_q1",
    "log_p_q2",
    "log_p_q3",
    "log_p_q4",
    "log_p_q5",
    "tp_q10",
    "tp_q25",
    "tp_q50",
    "tp_q75",
    "tp_q90",
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "sin_doy",
    "cos_doy",
    "lead_flag",
    "land_fraction",
)
TP_QUANTILE_FEATURES = FEATURE_NAMES[5:10]


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    """Convert a scalar datetime to a timezone-naive ``Timestamp``."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid scalar datetime")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def resolve_window(window: str | int | Sequence[int]) -> tuple[str, int, int]:
    """Return the canonical name and inclusive lead-day bounds for a window."""

    if isinstance(window, str):
        canonical = WINDOW_ALIASES.get(window.lower(), window.upper())
        if canonical in QUEST_WINDOWS:
            start, end = QUEST_WINDOWS[canonical]
            return canonical, start, end
    elif isinstance(window, (int, np.integer)) and int(window) in WINDOW_ALIASES:
        canonical = WINDOW_ALIASES[int(window)]
        start, end = QUEST_WINDOWS[canonical]
        return canonical, start, end
    elif not isinstance(window, (bytes, bytearray)):
        values = tuple(int(value) for value in window)
        for canonical, bounds in QUEST_WINDOWS.items():
            if values == bounds:
                return canonical, bounds[0], bounds[1]
    allowed = ", ".join(QUEST_WINDOWS)
    raise ValueError(f"window must be one of {allowed} (or its exact day tuple)")


def weekly_valid_dates(
    init_date: Any,
    window: str | int | Sequence[int],
) -> pd.DatetimeIndex:
    """Return the seven valid dates for an official Quest lead window.

    Day 1 is the initialization date.  Consequently D19--25 corresponds to
    offsets 18--24 and D26--32 corresponds to offsets 25--31.
    """

    initialization = _timestamp(init_date, name="init_date")
    _, first_day, final_day = resolve_window(window)
    return pd.DatetimeIndex(
        initialization + pd.to_timedelta(np.arange(first_day - 1, final_day), unit="D")
    )


def _weekly_operation(variable: str) -> Literal["mean", "sum"]:
    key = variable.strip().lower()
    if key in {
        "pr",
        "tp",
        "precip",
        "precipitation",
        "total_precipitation",
        "rainfall",
    }:
        return "sum"
    if key in {
        "tas",
        "t2m",
        "temperature",
        "mslp",
        "msl",
        "mean_sea_level_pressure",
    }:
        return "mean"
    raise ValueError(
        "variable must identify precipitation (weekly sum), temperature, "
        "or mean-sea-level pressure (weekly mean)"
    )


def aggregate_weekly(
    daily: xr.DataArray,
    init_date: Any,
    window: str | int | Sequence[int],
    variable: str,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Select exactly one official seven-day window and aggregate it.

    Missing or duplicated valid dates are errors rather than silently creating
    a shorter week.  Precipitation is summed; ``tas`` and ``mslp`` are averaged.
    """

    if not isinstance(daily, xr.DataArray):
        raise TypeError("daily must be an xarray.DataArray")
    if time_dim not in daily.dims:
        raise ValueError(f"daily has no {time_dim!r} dimension")

    expected = weekly_valid_dates(init_date, window)
    available = pd.DatetimeIndex(daily[time_dim].values)
    if available.has_duplicates:
        duplicated = available[available.duplicated()].unique()
        raise ValueError(f"{time_dim} contains duplicate dates: {list(duplicated[:3])}")
    missing = expected.difference(available)
    if len(missing):
        raise ValueError(
            f"missing {len(missing)} required daily value(s) for "
            f"{resolve_window(window)[0]}: {list(missing[:3])}"
        )

    selected = daily.sel({time_dim: expected})
    if selected.sizes[time_dim] != 7:
        raise ValueError("an official weekly window must contain exactly seven days")
    operation = _weekly_operation(variable)
    if operation == "sum":
        result = selected.sum(time_dim, skipna=False, keep_attrs=True)
        source_units = daily.attrs.get("units")
        if isinstance(source_units, str) and _normalise_unit_text(source_units) in {
            "mm/day",
            "mmday-1",
            "mmday^-1",
            "mmd-1",
            "mmd^-1",
        }:
            result.attrs["source_daily_units"] = source_units
            result.attrs["units"] = "mm"
    else:
        result = selected.mean(time_dim, skipna=False, keep_attrs=True)

    canonical, first_day, final_day = resolve_window(window)
    result.attrs.update(
        {
            "lead_window": canonical,
            "lead_days_inclusive": f"{first_day}-{final_day}",
            "valid_start": expected[0].isoformat(),
            "valid_end": expected[-1].isoformat(),
            "weekly_operation": operation,
        }
    )
    return result


def _normalise_unit_text(units: str) -> str:
    text = units.strip().lower()
    replacements = {
        "²": "2",
        "³": "3",
        "⁻": "-",
        "¹": "1",
        "·": "",
        "_": "",
        "**": "^",
        "per": "/",
        "hours": "hour",
        "hrs": "hr",
        "days": "day",
        "seconds": "second",
        "secs": "sec",
        "metres": "metre",
        "meters": "meter",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(text.split())


def _tp_scale_to_mm_day(units: str, accumulation_hours: float | None) -> float:
    unit = _normalise_unit_text(units)
    rate_scales = {
        "mm/day": 1.0,
        "mmday-1": 1.0,
        "mmday^-1": 1.0,
        "mmd-1": 1.0,
        "mmd^-1": 1.0,
        "mm/d": 1.0,
        "mm/hour": 24.0,
        "mm/hr": 24.0,
        "mm/h": 24.0,
        "mmh-1": 24.0,
        "mmh^-1": 24.0,
        "mmhr-1": 24.0,
        "mm/hour-1": 24.0,
        "kgm-2day-1": 1.0,
        "kgm^-2day^-1": 1.0,
        "kg/m2/day": 1.0,
        "kgm-2s-1": 86_400.0,
        "kgm^-2s^-1": 86_400.0,
        "kg/m2/s": 86_400.0,
        "m/s": 86_400_000.0,
        "ms-1": 86_400_000.0,
        "ms^-1": 86_400_000.0,
    }
    if unit in rate_scales:
        if accumulation_hours is not None:
            raise ValueError("accumulation_hours must not be supplied for a rate unit")
        return rate_scales[unit]

    accumulated_scales = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimetre": 1.0,
        "kgm-2": 1.0,
        "kgm^-2": 1.0,
        "kg/m2": 1.0,
        "m": 1000.0,
        "meter": 1000.0,
        "metre": 1000.0,
    }
    if unit in accumulated_scales:
        if accumulation_hours is None:
            raise ValueError(
                f"units={units!r} describe an accumulation; supply its exact "
                "accumulation_hours"
            )
        hours = float(accumulation_hours)
        if not np.isfinite(hours) or hours <= 0.0:
            raise ValueError("accumulation_hours must be finite and positive")
        return accumulated_scales[unit] * 24.0 / hours

    raise ValueError(
        f"unsupported precipitation units {units!r}; no magnitude-based unit "
        "inference is performed"
    )


def convert_tp_to_mm_day(
    precipitation: xr.DataArray | np.ndarray,
    *,
    units: str | None = None,
    accumulation_hours: float | None = None,
) -> xr.DataArray | np.ndarray:
    """Convert declared precipitation rate/accumulation units to mm day-1.

    ``units`` defaults to ``DataArray.attrs['units']``.  Bare ``m``, ``mm``,
    and ``kg m-2`` are accumulations and require ``accumulation_hours``.  The
    function intentionally refuses missing or unfamiliar units rather than
    guessing from value magnitude.
    """

    declared = units
    if declared is None and isinstance(precipitation, xr.DataArray):
        declared = precipitation.attrs.get("units")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("explicit precipitation units are required")
    scale = _tp_scale_to_mm_day(declared, accumulation_hours)

    if isinstance(precipitation, xr.DataArray):
        converted = precipitation.astype(np.float64) * scale
        converted.attrs = dict(precipitation.attrs)
        converted.attrs.update(
            {
                "units": "mm day-1",
                "source_units": declared,
                "conversion_scale": scale,
            }
        )
        if accumulation_hours is not None:
            converted.attrs["source_accumulation_hours"] = float(accumulation_hours)
        return converted

    values = np.asarray(precipitation)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("precipitation must contain numeric values")
    return values.astype(np.float64, copy=False) * scale


def _find_boundary_dim(bounds: xr.DataArray, boundary_dim: str | None) -> str:
    if boundary_dim is not None:
        if boundary_dim not in bounds.dims:
            raise ValueError(f"bounds has no {boundary_dim!r} dimension")
        return boundary_dim
    candidates = [name for name in ("quantile", "quintile", "boundary") if name in bounds.dims]
    if len(candidates) == 1:
        return candidates[0]
    if bounds.ndim >= 1 and bounds.sizes[bounds.dims[0]] == 4:
        return bounds.dims[0]
    raise ValueError("could not determine the four-boundary dimension")


def _validate_category_inputs(
    values: xr.DataArray,
    bounds: xr.DataArray,
    boundary_dim: str | None,
) -> tuple[str, xr.DataArray, xr.DataArray]:
    if not isinstance(values, xr.DataArray) or not isinstance(bounds, xr.DataArray):
        raise TypeError("values and bounds must be xarray.DataArray objects")
    dim = _find_boundary_dim(bounds, boundary_dim)
    if bounds.sizes[dim] != 4:
        raise ValueError("quintile bounds must contain exactly four thresholds")
    if dim in values.dims:
        raise ValueError(f"values must not contain the boundary dimension {dim!r}")

    # Boundary labels do not affect categorization; their order does.
    ordered = bounds.assign_coords({dim: np.arange(4)}).astype(np.float64)
    return dim, values, ordered


def _category_criteria(
    values: xr.DataArray,
    bounds: xr.DataArray,
    boundary_dim: str,
    *,
    category_dim: str,
    mask_all_equal: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Implement the evaluator's upper-category equality convention."""

    boundary = [bounds.isel({boundary_dim: index}, drop=True) for index in range(4)]
    criteria: list[xr.DataArray] = [values < boundary[0]]
    for index in range(1, 4):
        lower = boundary[index - 1]
        upper = boundary[index]
        # Repeated boundaries create an empty (disabled) interior category.
        criteria.append((lower <= values) & (values < upper) & (lower != upper))
    criteria.append(values >= boundary[-1])
    one_hot = xr.concat(criteria, dim=pd.Index(CATEGORY_VALUES, name=category_dim))

    finite_bounds = np.isfinite(bounds).all(boundary_dim)
    differences = bounds.diff(boundary_dim)
    nondecreasing = (differences >= 0.0).all(boundary_dim)
    all_equal = bounds.min(boundary_dim) == bounds.max(boundary_dim)
    valid = np.isfinite(values) & finite_bounds & nondecreasing
    if mask_all_equal:
        valid = valid & ~all_equal
    return one_hot.astype(np.float64).where(valid), valid


@dataclass(frozen=True)
class ObservationCategoryResult:
    """One-hot observations, zero-based category index, and their valid mask."""

    probabilities: xr.DataArray
    index: xr.DataArray
    valid_mask: xr.DataArray


def observation_categories(
    observations: xr.DataArray,
    bounds: xr.DataArray,
    *,
    boundary_dim: str | None = None,
    category_dim: str = "quintile",
) -> ObservationCategoryResult:
    """Categorize observations using the official equality and desert rules.

    Equality with a boundary goes to the upper category.  An interior category
    between duplicate bounds is disabled.  Cells where all four bounds are
    equal, bounds decrease/are missing, or the observation is missing are
    invalid: their one-hot values are NaN and their index is ``-1``.
    """

    dim, values, ordered = _validate_category_inputs(observations, bounds, boundary_dim)
    probabilities, valid = _category_criteria(
        values,
        ordered,
        dim,
        category_dim=category_dim,
        mask_all_equal=True,
    )
    index = probabilities.fillna(0.0).argmax(category_dim).astype(np.int8)
    index = index.where(valid, other=np.int8(-1))
    index.name = "observation_category"
    valid.name = "valid_observation"
    probabilities.name = "observation_probability"
    return ObservationCategoryResult(probabilities, index, valid)


def member_quintile_probabilities(
    members: xr.DataArray,
    bounds: xr.DataArray,
    *,
    member_dim: str = "member",
    boundary_dim: str | None = None,
    category_dim: str = "quintile",
    smoothing: float = 0.5,
) -> xr.DataArray:
    """Turn ensemble members into additive-smoothed quintile probabilities.

    The returned value is ``(count + smoothing) / (N + 5*smoothing)``.
    Every member must be finite at a cell; cells with missing members or an
    invalid climatology are NaN.  The default ``smoothing=0.5`` is the
    project's Jeffreys-style smoothing and guarantees positive probabilities.
    All-equal/arid cells retain a finite anchor so the model input remains
    usable; ``observation_categories`` supplies their explicit invalid mask.
    """

    if member_dim not in members.dims:
        raise ValueError(f"members has no {member_dim!r} dimension")
    member_count = members.sizes[member_dim]
    if member_count < 1:
        raise ValueError("at least one ensemble member is required")
    alpha = float(smoothing)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("smoothing must be finite and non-negative")

    dim, values, ordered = _validate_category_inputs(members, bounds, boundary_dim)
    one_hot, boundary_valid = _category_criteria(
        values,
        ordered,
        dim,
        category_dim=category_dim,
        mask_all_equal=False,
    )
    complete_members = np.isfinite(members).all(member_dim)
    counts = one_hot.fillna(0.0).sum(member_dim)
    denominator = float(member_count) + 5.0 * alpha
    probabilities = (counts + alpha) / denominator
    probabilities = probabilities.where(boundary_valid.all(member_dim) & complete_members)
    probabilities.name = "member_quintile_probability"
    probabilities.attrs.update(
        {
            "member_count": member_count,
            "additive_smoothing": alpha,
            "boundary_equality": "upper_category",
        }
    )
    return probabilities


def _as_spatial_coordinate(
    coordinate: xr.DataArray | Sequence[float],
    *,
    dim: str,
) -> xr.DataArray:
    if isinstance(coordinate, xr.DataArray):
        if coordinate.ndim != 1:
            raise ValueError(f"{dim} must be one-dimensional")
        if coordinate.dims != (dim,):
            if coordinate.name == dim and coordinate.dims[0] != dim:
                coordinate = coordinate.rename({coordinate.dims[0]: dim})
            else:
                raise ValueError(f"{dim} DataArray must use dimension {dim!r}")
        return coordinate.astype(np.float64)
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{dim} must be one-dimensional")
    return xr.DataArray(values, dims=(dim,), coords={dim: values}, name=dim)


def _tp_normalization_arrays(
    means: xr.DataArray | Sequence[float] | Mapping[str, Any] | None,
    stds: xr.DataArray | Sequence[float] | Mapping[str, Any] | None,
    *,
    feature_dim: str,
    lead_dim: str,
    lead_coordinate: xr.DataArray,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    if (means is None) != (stds is None):
        raise ValueError("tp_quantile_means and tp_quantile_stds must be supplied together")
    if means is None:
        return None, None

    def coerce(
        value: xr.DataArray | Sequence[float] | Mapping[str, Any], name: str
    ) -> xr.DataArray:
        if isinstance(value, Mapping):
            missing = [key for key in TP_QUANTILE_FEATURES if key not in value]
            if missing:
                raise ValueError(f"{name} is missing feature(s): {missing}")
            array = np.asarray(
                [value[key] for key in TP_QUANTILE_FEATURES], dtype=np.float64
            )
            if array.shape == (5, 2):
                array = array.T
        elif isinstance(value, xr.DataArray):
            if value.ndim == 2 and lead_dim in value.dims:
                other_dims = [dim for dim in value.dims if dim != lead_dim]
                if len(other_dims) != 1:
                    raise ValueError(f"{name} has ambiguous normalization dimensions")
                value = value.sel({lead_dim: lead_coordinate}).transpose(
                    lead_dim, other_dims[0]
                )
            array = np.asarray(value.values, dtype=np.float64)
        else:
            array = np.asarray(value, dtype=np.float64)
        if array.shape == (5,):
            return xr.DataArray(
                array,
                dims=(feature_dim,),
                coords={feature_dim: list(TP_QUANTILE_FEATURES)},
            )
        if array.shape == (2, 5):
            return xr.DataArray(
                array,
                dims=(lead_dim, feature_dim),
                coords={
                    lead_dim: lead_coordinate,
                    feature_dim: list(TP_QUANTILE_FEATURES),
                },
            )
        raise ValueError(
            f"{name} must have shape [5] (shared) or [2, 5] (per lead)"
        )

    mean_array = coerce(means, "tp_quantile_means")
    std_array = coerce(stds, "tp_quantile_stds")
    if not bool(np.isfinite(mean_array).all()) or not bool(np.isfinite(std_array).all()):
        raise ValueError("TP normalization statistics must be finite")
    if bool((std_array <= 0.0).any()):
        raise ValueError("TP normalization standard deviations must be positive")
    return mean_array, std_array


def build_features(
    weekly_members: xr.DataArray,
    p0: xr.DataArray,
    init_date: Any,
    latitude: xr.DataArray | Sequence[float],
    longitude: xr.DataArray | Sequence[float],
    land_fraction: xr.DataArray,
    *,
    lead_dim: str = "lead",
    member_dim: str = "member",
    category_dim: str = "quintile",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    feature_dim: str = "feature",
    tp_quantile_means: xr.DataArray | Sequence[float] | Mapping[str, Any] | None = None,
    tp_quantile_stds: xr.DataArray | Sequence[float] | Mapping[str, Any] | None = None,
    probability_floor: float = 1.0e-8,
) -> xr.DataArray:
    """Construct the fixed 18-channel input tensor for both lead windows.

    Parameters
    ----------
    weekly_members:
        Weekly precipitation accumulations in mm with dimensions
        ``[lead, member, latitude, longitude]``.
    p0:
        Five smoothed anchor probabilities with dimensions
        ``[lead, quintile, latitude, longitude]``.
    tp_quantile_means, tp_quantile_stds:
        Optional *training-only* statistics for the five ``log1p`` member
        quantile channels.  They may have shape ``[5]`` (shared across leads)
        or ``[2, 5]`` (per lead), or be mappings keyed by ``tp_q10`` ...
        ``tp_q90``.  Supplying only one is an error.

    Returns
    -------
    xarray.DataArray
        Dimensions ``[lead, feature, latitude, longitude]`` in exactly
        ``FEATURE_NAMES`` order.
    """

    if not isinstance(weekly_members, xr.DataArray) or not isinstance(p0, xr.DataArray):
        raise TypeError("weekly_members and p0 must be xarray.DataArray objects")
    required_member_dims = {lead_dim, member_dim, latitude_dim, longitude_dim}
    required_anchor_dims = {lead_dim, category_dim, latitude_dim, longitude_dim}
    if not required_member_dims.issubset(weekly_members.dims):
        raise ValueError(f"weekly_members must contain dimensions {sorted(required_member_dims)}")
    if not required_anchor_dims.issubset(p0.dims):
        raise ValueError(f"p0 must contain dimensions {sorted(required_anchor_dims)}")
    if weekly_members.sizes[lead_dim] != 2 or p0.sizes[lead_dim] != 2:
        raise ValueError("weekly_members and p0 must contain exactly two lead windows")
    if p0.sizes[category_dim] != 5:
        raise ValueError("p0 must contain exactly five ordered categories")
    if weekly_members.sizes[member_dim] < 1:
        raise ValueError("weekly_members must contain at least one member")

    members, anchor = xr.align(
        weekly_members,
        p0,
        join="exact",
        exclude={member_dim, category_dim},
    )
    lat = _as_spatial_coordinate(latitude, dim=latitude_dim)
    lon = _as_spatial_coordinate(longitude, dim=longitude_dim)
    if not members[latitude_dim].equals(lat) or not members[longitude_dim].equals(lon):
        raise ValueError("explicit latitude/longitude coordinates must exactly match the data")
    if not isinstance(land_fraction, xr.DataArray):
        raise TypeError("land_fraction must be an xarray.DataArray")
    spatial_template = anchor.isel({lead_dim: 0, category_dim: 0}, drop=True)
    land, spatial_template = xr.align(land_fraction, spatial_template, join="exact")
    land = land.broadcast_like(spatial_template).astype(np.float64)

    if not bool(np.isfinite(members).all()):
        raise ValueError("weekly_members contains missing or non-finite values")
    if bool((members < 0.0).any()):
        raise ValueError("weekly precipitation must be non-negative")
    if not bool(np.isfinite(anchor).all()):
        raise ValueError("p0 contains missing or non-finite values")
    if bool(((anchor < 0.0) | (anchor > 1.0)).any()):
        raise ValueError("p0 probabilities must lie in [0, 1]")
    if not bool(np.isclose(anchor.sum(category_dim), 1.0, rtol=1e-6, atol=1e-7).all()):
        raise ValueError("p0 probabilities must sum to one at every cell")
    if not bool(np.isfinite(land).all()) or bool(((land < 0.0) | (land > 1.0)).any()):
        raise ValueError("land_fraction must be finite and lie in [0, 1]")
    floor = float(probability_floor)
    if not np.isfinite(floor) or not 0.0 < floor < 0.2:
        raise ValueError("probability_floor must be finite and in (0, 0.2)")

    category_labels = list(FEATURE_NAMES[:5])
    log_anchor = np.log(anchor.clip(min=floor))
    log_anchor = log_anchor.assign_coords({category_dim: category_labels}).rename(
        {category_dim: feature_dim}
    )

    member_quantiles = np.log1p(members).quantile(
        MEMBER_QUANTILES, dim=member_dim, skipna=False
    )
    member_quantiles = member_quantiles.assign_coords(
        quantile=list(TP_QUANTILE_FEATURES)
    ).rename({"quantile": feature_dim})
    means, stds = _tp_normalization_arrays(
        tp_quantile_means,
        tp_quantile_stds,
        feature_dim=feature_dim,
        lead_dim=lead_dim,
        lead_coordinate=anchor[lead_dim],
    )
    if means is not None and stds is not None:
        member_quantiles = (member_quantiles - means) / stds

    lat_radians = np.deg2rad(lat)
    lon_radians = np.deg2rad(lon)
    sin_lat = np.sin(lat_radians).broadcast_like(spatial_template)
    cos_lat = np.cos(lat_radians).broadcast_like(spatial_template)
    sin_lon = np.sin(lon_radians).broadcast_like(spatial_template)
    cos_lon = np.cos(lon_radians).broadcast_like(spatial_template)

    starts = [weekly_valid_dates(init_date, canonical)[0] for canonical in QUEST_WINDOWS]
    phases = np.asarray(
        [2.0 * np.pi * (date.dayofyear - 1) / 365.2425 for date in starts],
        dtype=np.float64,
    )
    lead_coord = anchor[lead_dim]

    def repeat_spatial(field: xr.DataArray) -> xr.DataArray:
        return field.expand_dims({lead_dim: lead_coord}).transpose(
            lead_dim, latitude_dim, longitude_dim
        )

    static_channels = [
        repeat_spatial(sin_lat),
        repeat_spatial(cos_lat),
        repeat_spatial(sin_lon),
        repeat_spatial(cos_lon),
        xr.DataArray(
            np.sin(phases), dims=(lead_dim,), coords={lead_dim: lead_coord}
        ).broadcast_like(anchor.isel({category_dim: 0}, drop=True)),
        xr.DataArray(
            np.cos(phases), dims=(lead_dim,), coords={lead_dim: lead_coord}
        ).broadcast_like(anchor.isel({category_dim: 0}, drop=True)),
        xr.DataArray([-1.0, 1.0], dims=(lead_dim,), coords={lead_dim: lead_coord}).broadcast_like(
            anchor.isel({category_dim: 0}, drop=True)
        ),
        repeat_spatial(land),
    ]
    static_names = FEATURE_NAMES[10:]
    static = xr.concat(static_channels, dim=pd.Index(static_names, name=feature_dim))
    static = static.transpose(lead_dim, feature_dim, latitude_dim, longitude_dim)

    features = xr.concat([log_anchor, member_quantiles, static], dim=feature_dim)
    features = features.transpose(lead_dim, feature_dim, latitude_dim, longitude_dim)
    features = features.assign_coords({feature_dim: list(FEATURE_NAMES)})
    features.name = "features"
    features.attrs.update(
        {
            "feature_contract": "global_tp_18_channel_v1",
            "tp_quantiles": "q10,q25,q50,q75,q90 of weekly member accumulation",
            "tp_transform": "log1p then optional training-only standardization",
            "lead_flag": "-1=D19-25, +1=D26-32",
        }
    )
    return features.astype(np.float32)


def previous_20yr_thresholds(
    daily_observations: xr.DataArray,
    target_start: Any,
    issue_time: Any,
    variable: str,
    *,
    time_dim: str = "time",
    years: int = 20,
    day_offsets: Sequence[int] = (-4, -2, 0, 2, 4),
    quantiles: Sequence[float] = tuple(QUINTILE_LEVELS),
) -> xr.DataArray:
    """Compute leakage-safe Quest thresholds from 100 historical weeks.

    For the requested valid-week start, five starts at offsets ``-4,-2,0,2,4``
    are sampled in each of the preceding 20 target years.  Each historical
    start is reduced over seven daily values, followed by the 0.2/0.4/0.6/0.8
    quantiles across the resulting 100 weekly samples.

    ``issue_time`` is mandatory.  Every selected observation, including the
    final day of every historical week, must precede it; otherwise the function
    raises instead of leaking unavailable observations.
    """

    if not isinstance(daily_observations, xr.DataArray):
        raise TypeError("daily_observations must be an xarray.DataArray")
    if time_dim not in daily_observations.dims:
        raise ValueError(f"daily_observations has no {time_dim!r} dimension")
    target = _timestamp(target_start, name="target_start")
    cutoff = _timestamp(issue_time, name="issue_time")
    if target < cutoff:
        raise ValueError("target_start must not precede issue_time")
    if not isinstance(years, (int, np.integer)) or int(years) < 1:
        raise ValueError("years must be a positive integer")
    years = int(years)
    offsets = tuple(int(value) for value in day_offsets)
    if offsets != (-4, -2, 0, 2, 4):
        raise ValueError("official Quest day_offsets must be exactly (-4, -2, 0, 2, 4)")
    q_values = np.asarray(quantiles, dtype=np.float64)
    if q_values.shape != (4,) or not np.array_equal(q_values, QUINTILE_LEVELS):
        raise ValueError("official Quest quantiles must be exactly (0.2, 0.4, 0.6, 0.8)")

    available = pd.DatetimeIndex(daily_observations[time_dim].values)
    if available.has_duplicates:
        raise ValueError(f"{time_dim} contains duplicate timestamps")
    if not available.is_monotonic_increasing:
        raise ValueError(f"{time_dim} must be monotonically increasing")

    weekly_samples: list[xr.DataArray] = []
    sample_starts: list[pd.Timestamp] = []
    missing_dates: list[pd.Timestamp] = []
    operation = _weekly_operation(variable)
    for years_back in range(years, 0, -1):
        anniversary = target + relativedelta(years=-years_back)
        for offset in offsets:
            start = anniversary + pd.Timedelta(days=offset)
            dates = pd.date_range(start, periods=7, freq="D")
            if dates[-1] >= cutoff:
                raise ValueError(
                    "climatology would use observations unavailable at issue_time: "
                    f"historical week {dates[0].date()} to {dates[-1].date()}"
                )
            absent = dates.difference(available)
            if len(absent):
                missing_dates.extend(absent.tolist())
                continue
            week = daily_observations.sel({time_dim: dates})
            if operation == "sum":
                reduced = week.sum(time_dim, skipna=False, keep_attrs=True)
            else:
                reduced = week.mean(time_dim, skipna=False, keep_attrs=True)
            weekly_samples.append(reduced)
            sample_starts.append(start)

    expected_count = years * len(offsets)
    if missing_dates:
        unique_missing = pd.DatetimeIndex(missing_dates).unique().sort_values()
        raise ValueError(
            f"missing {len(unique_missing)} daily observation date(s) needed for "
            f"the {expected_count}-sample climatology: {list(unique_missing[:3])}"
        )
    if len(weekly_samples) != expected_count:
        raise RuntimeError("internal error constructing historical weekly samples")

    sample_index = pd.MultiIndex.from_arrays(
        [
            [date.year for date in sample_starts],
            [offset for _ in range(years) for offset in offsets],
        ],
        names=("sample_year", "day_offset"),
    )
    samples = xr.concat(weekly_samples, dim="sample").assign_coords(sample=sample_index)
    thresholds = samples.quantile(q_values, dim="sample", skipna=False)
    thresholds.name = "climatological_quintile_boundary"
    thresholds.attrs = dict(daily_observations.attrs)
    source_units = daily_observations.attrs.get("units")
    if operation == "sum" and isinstance(source_units, str):
        if _normalise_unit_text(source_units) in {
            "mm/day",
            "mmday-1",
            "mmday^-1",
            "mmd-1",
            "mmd^-1",
        }:
            thresholds.attrs["source_daily_units"] = source_units
            thresholds.attrs["units"] = "mm"
    thresholds.attrs.update(
        {
            "target_start": target.isoformat(),
            "issue_time_cutoff": cutoff.isoformat(),
            "history_years": years,
            "sample_count": expected_count,
            "sample_day_offsets": ",".join(str(value) for value in offsets),
            "weekly_operation": operation,
            "leakage_rule": "all selected observation days strictly precede issue_time",
        }
    )
    return thresholds


__all__ = [
    "CATEGORY_VALUES",
    "FEATURE_NAMES",
    "MEMBER_QUANTILES",
    "ObservationCategoryResult",
    "QUEST_WINDOWS",
    "QUINTILE_LEVELS",
    "TP_QUANTILE_FEATURES",
    "aggregate_weekly",
    "build_features",
    "convert_tp_to_mm_day",
    "member_quintile_probabilities",
    "observation_categories",
    "previous_20yr_thresholds",
    "resolve_window",
    "weekly_valid_dates",
]
