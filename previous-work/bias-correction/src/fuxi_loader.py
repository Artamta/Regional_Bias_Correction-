from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)

SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "JJAS": (6, 7, 8, 9),
    "SON": (9, 10, 11),
}


def _select_inits(inits, date, month, season, year):
    if sum(value is not None for value in (date, month, season)) > 1:
        raise ValueError("use only one of date, month, or season")
    if year is not None and (date is not None or month is not None):
        raise ValueError("year is only used by itself or with season")

    if date is not None:
        date = pd.Timestamp(date).normalize()
        selected = inits[inits == date]
    elif month is not None:
        month = pd.Period(month, freq="M")
        selected = inits[(inits.year == month.year) & (inits.month == month.month)]
    elif season is not None:
        season = season.upper()
        if season not in SEASONS:
            raise ValueError(f"season must be one of {list(SEASONS)}")
        if year is None:
            raise ValueError("year is required with season")
        if season == "DJF":
            selected = inits[
                ((inits.year == year - 1) & (inits.month == 12))
                | ((inits.year == year) & inits.month.isin((1, 2)))
            ]
        else:
            selected = inits[
                (inits.year == year) & inits.month.isin(SEASONS[season])
            ]
    elif year is not None:
        selected = inits[inits.year == year]
    else:
        raise ValueError("provide date, month, season and year, or year")

    if selected.empty:
        raise KeyError("no FuXi initialization found for the requested period")
    return selected


def load_fuxi(
    *,
    date=None,
    month=None,
    season=None,
    year=None,
    channels="tp",
    members=None,
    lead_days=None,
    india_only=True,
    ensemble_mean=False,
    compute=False,
):
    """Load FuXi hindcasts for one date, month, season, or year."""

    ds = xr.open_zarr(STORE, consolidated=True, chunks={})
    inits = pd.DatetimeIndex(ds.init.values)
    selected = _select_inits(inits, date, month, season, year)

    channels = [channels] if isinstance(channels, str) else list(channels)
    data = ds.forecast.sel(init=selected.values, channel=channels)

    if india_only:
        data = data.sel(lat=slice(39.0, 0.0), lon=slice(60.0, 99.0))
    if members is not None:
        data = data.sel(member=list(members))
    if lead_days is not None:
        data = data.sel(lead_day=list(lead_days))

    if "tp" in channels:
        data = xr.where(
            data.channel == "tp",
            data.clip(min=0) * np.float32(24.0),
            data,
            keep_attrs=True,
        )
    if ensemble_mean:
        data = data.mean("member")
    if len(channels) == 1:
        data = data.sel(channel=channels[0], drop=True).rename(channels[0])
    if "tp" in channels:
        data.attrs["tp_units"] = "mm day-1"
        if channels == ["tp"]:
            data.attrs["units"] = "mm day-1"

    return data.load() if compute else data
