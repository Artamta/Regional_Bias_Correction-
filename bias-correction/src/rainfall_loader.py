from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/observations/ground_truth_v1/daily"
)


def load_rainfall(start, end=None, source="imerg", compute=False):
    """Load daily IMERG or IMD rainfall on the FuXi India grid."""

    source = source.lower()
    if source not in {"imerg", "imd"}:
        raise ValueError("source must be 'imerg' or 'imd'")

    start = pd.Timestamp(start).normalize()
    end = start if end is None else pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError("end must be on or after start")

    pieces = []
    fraction = None

    for year in range(start.year, end.year + 1):
        path = ROOT / source / "tp" / "india_1p5_27x27_v1" / f"{year}.zarr"
        if not path.is_dir():
            raise FileNotFoundError(path)

        ds = xr.open_zarr(path, consolidated=True, chunks={})
        part = ds.observation.sel(time=slice(start, end))
        if part.sizes["time"]:
            pieces.append(part)
        if fraction is None:
            fraction = ds.observation_fraction

    if not pieces:
        raise KeyError(f"no {source.upper()} data found from {start.date()} to {end.date()}")

    rainfall = xr.concat(pieces, dim="time")
    expected = pd.date_range(start, end, freq="D")
    actual = pd.DatetimeIndex(rainfall.time.values)
    missing = expected.difference(actual)
    if len(missing):
        dates = ", ".join(str(date.date()) for date in missing[:5])
        raise KeyError(f"{source.upper()} is missing {len(missing)} dates: {dates}")

    result = xr.Dataset(
        {
            "rainfall": rainfall,
            "observation_fraction": fraction,
        }
    ).rename({"latitude": "lat", "longitude": "lon"})

    result.attrs.update(
        source=source.upper(),
        units="mm day-1",
        grid="india_1p5_27x27_v1",
    )
    if source == "imerg":
        result.attrs.update(
            product="GPM_3IMERGDF.07",
            revision="V07B",
            doi="10.5067/GPM/IMERGDF/DAY/07",
        )
    result.rainfall.attrs.update(source=source.upper(), units="mm day-1")
    result.observation_fraction.attrs["units"] = "1"
    return result.load() if compute else result


def load_imerg(start, end=None, compute=False):
    return load_rainfall(start, end, source="imerg", compute=compute)


def load_imd(start, end=None, compute=False):
    return load_rainfall(start, end, source="imd", compute=compute)
