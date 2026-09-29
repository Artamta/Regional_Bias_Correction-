"""Download ERA5 references and build the compact TP-only training cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from config import EXPERIMENT, PATHS


EXPECTED_LATITUDE = np.linspace(90.0, -90.0, 121, dtype=np.float64)
EXPECTED_LONGITUDE = np.arange(0.0, 360.0, 1.5, dtype=np.float64)
TP_QUANTILES = np.asarray((0.10, 0.25, 0.50, 0.75, 0.90))
BOUNDARY_QUANTILES = np.asarray((0.20, 0.40, 0.60, 0.80))
CLIMATOLOGY_OFFSETS = (-4, -2, 0, 2, 4)
EXPECTED_FUXI_CHANNELS = (
    "z850", "z500", "z250", "t850", "t500", "t250",
    "u850", "u500", "u250", "v850", "v500", "v250",
    "q850", "q500", "q250", "t2m", "d2m", "sst", "ttr",
    "10u", "10v", "100u", "100v", "msl", "tcwv", "tp",
)


TRAINING_FILE_STATISTIC = {
    "pr": "WEEKLYSUM",
    "tas": "WEEKLYMEAN",
    "mslp": "WEEKLYMEAN",
}


def _training_filename(variable: str, year: int) -> str:
    try:
        statistic = TRAINING_FILE_STATISTIC[variable]
    except KeyError as error:
        raise ValueError(f"unsupported Quest training variable: {variable!r}") from error
    return f"{variable}_sevenday_{statistic}_{int(year)}.nc"


def _validate_training_file(path: Path, variable: str, year: int) -> None:
    """Reject partial or scientifically incompatible annual Quest files."""

    with xr.open_dataset(path) as dataset:
        field = _normalise_grid(_pick_field(dataset, require_time=True))
        times = pd.DatetimeIndex(field.time.values).normalize()
        expected = pd.date_range(f"{int(year):04d}-01-01", f"{int(year):04d}-12-31")
        if len(times) != len(expected) or not times.equals(expected):
            raise ValueError(
                f"{path.name} must contain every daily seven-day statistic in {year}"
            )
        units = str(field.attrs.get("units", "")).lower().replace(" ", "")
        unit_ok = {
            "pr": units
            in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"},
            "tas": units in {"k", "kelvin"},
            "mslp": units in {"pa", "pascal", "pascals"},
        }[variable]
        if not unit_ok:
            raise ValueError(
                f"{path.name} has incompatible {variable} units {field.attrs.get('units')!r}"
            )
        values = np.asarray(field.values)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} contains non-finite values")


def _validate_land_file(path: Path) -> None:
    with xr.open_dataset(path) as dataset:
        field = _normalise_grid(_pick_field(dataset, require_time=False))
        values = np.asarray(field.values, dtype=np.float32)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("official land fraction must be finite and lie in [0,1]")


def download_era5(args: argparse.Namespace) -> None:
    """Use the official package without putting the ECBox token on the command line."""

    token = os.environ.get(args.password_env)
    if not token:
        raise RuntimeError(
            f"set {args.password_env} to the AI Weather Quest ECBox token; "
            "the CDS token is different"
        )
    try:
        from AI_WQ_package import retrieve_evaluation_data, retrieve_training_data
    except ImportError as error:
        raise RuntimeError(
            "install the current AI-WQ-package and sites-toolkit in the project environment"
        ) from error

    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    for variable in args.variables:
        for year in range(args.start_year, args.end_year + 1):
            path = destination / _training_filename(variable, year)
            if path.exists() and path.stat().st_size > 0:
                try:
                    _validate_training_file(path, variable, year)
                except (OSError, ValueError) as error:
                    print(f"invalid existing file {path}: {error}; retrieving replacement")
                else:
                    print(f"validated {path}")
                    continue
            print(f"retrieving ERA5 weekly {variable} for {year}")
            with tempfile.TemporaryDirectory(
                prefix=f".{variable}-{year}-", dir=destination
            ) as staging:
                dataset = retrieve_training_data.retrieve_annual_training_data(
                    year, variable, token, local_destination=staging
                )
                dataset.close()
                staged_path = Path(staging) / _training_filename(variable, year)
                if not staged_path.is_file():
                    raise RuntimeError(
                        f"Quest package did not create expected file {staged_path.name}"
                    )
                _validate_training_file(staged_path, variable, year)
                os.replace(staged_path, path)

    land_path = destination / "land_sea_mask_1pt5DEG.nc"
    land_valid = False
    if land_path.exists() and land_path.stat().st_size > 0:
        try:
            _validate_land_file(land_path)
        except (OSError, ValueError) as error:
            print(f"invalid existing file {land_path}: {error}; retrieving replacement")
        else:
            print(f"validated {land_path}")
            land_valid = True
    if not land_valid:
        print("retrieving the official 1.5-degree land fraction")
        with tempfile.TemporaryDirectory(prefix=".land-mask-", dir=destination) as staging:
            mask = retrieve_evaluation_data.retrieve_land_sea_mask(
                token, local_destination=staging
            )
            mask.close()
            staged_land = Path(staging) / land_path.name
            if not staged_land.is_file():
                raise RuntimeError(
                    f"Quest package did not create expected file {land_path.name}"
                )
            _validate_land_file(staged_land)
            os.replace(staged_land, land_path)
    print(f"ERA5 reference directory: {destination}")


def _coordinate_name(data: xr.DataArray | xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in data.coords or name in data.dims:
            return name
    raise ValueError(f"none of the expected coordinates {candidates} was found")


def _pick_field(dataset: xr.Dataset, *, require_time: bool) -> xr.DataArray:
    lat_name = _coordinate_name(dataset, ("latitude", "lat"))
    lon_name = _coordinate_name(dataset, ("longitude", "lon"))
    required = {lat_name, lon_name}
    if require_time:
        required.add(_coordinate_name(dataset, ("time", "valid_time")))
    candidates = [
        field
        for field in dataset.data_vars.values()
        if required.issubset(field.dims) and np.issubdtype(field.dtype, np.number)
    ]
    if len(candidates) != 1:
        names = [field.name for field in candidates]
        raise ValueError(f"expected one gridded data variable, found {names}")
    field = candidates[0]
    rename = {
        source: target
        for source, target in ((lat_name, "latitude"), (lon_name, "longitude"))
        if source != target
    }
    if require_time:
        time_name = _coordinate_name(dataset, ("time", "valid_time"))
        if time_name != "time":
            rename[time_name] = "time"
    return (field.rename(rename) if rename else field).squeeze(drop=True)


def _normalise_grid(field: xr.DataArray) -> xr.DataArray:
    longitude = np.mod(np.asarray(field.longitude.values, dtype=np.float64), 360.0)
    field = field.assign_coords(longitude=longitude).sortby("longitude")
    field = field.sortby("latitude", ascending=False)
    if not np.allclose(field.latitude.values, EXPECTED_LATITUDE, atol=1.0e-6):
        raise ValueError("latitude is not the exact 121-point Quest grid")
    if not np.allclose(field.longitude.values, EXPECTED_LONGITUDE, atol=1.0e-6):
        raise ValueError("longitude is not the exact 240-point Quest grid")
    return field


def load_era5_weekly(directory: Path) -> xr.DataArray:
    paths = sorted(Path(directory).glob("pr_sevenday_WEEKLYSUM_*.nc"))
    if not paths:
        raise FileNotFoundError(f"no annual ERA5 weekly precipitation files in {directory}")
    fields = []
    for path in paths:
        dataset = xr.open_dataset(path, chunks=None)
        field = _normalise_grid(_pick_field(dataset, require_time=True))
        units = str(field.attrs.get("units", "")).lower()
        if "mm" not in units:
            raise ValueError(f"{path} must explicitly report precipitation in mm; found {units!r}")
        fields.append(field)
    combined = xr.concat(fields, dim="time").sortby("time")
    times = pd.DatetimeIndex(combined.time.values).normalize()
    keep = ~times.duplicated(keep="first")
    combined = combined.isel(time=np.flatnonzero(keep))
    combined = combined.assign_coords(time=times[keep].values)
    return combined.transpose("time", "latitude", "longitude")


def load_land_fraction(path: Path) -> xr.DataArray:
    dataset = xr.open_dataset(path, chunks=None)
    field = _normalise_grid(_pick_field(dataset, require_time=False))
    field = field.transpose("latitude", "longitude").astype(np.float32)
    values = np.asarray(field.values)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("land fraction must be finite and bounded by 0 and 1")
    return field


def _shift_year(date: pd.Timestamp, years: int) -> pd.Timestamp:
    target_year = date.year + years
    try:
        return date.replace(year=target_year)
    except ValueError:
        return date.replace(year=target_year, day=28)


def climatology_thresholds(weekly: xr.DataArray, valid_start: pd.Timestamp) -> np.ndarray:
    dates = [
        _shift_year(valid_start, years) + pd.Timedelta(days=offset)
        for years in range(-20, 0)
        for offset in CLIMATOLOGY_OFFSETS
    ]
    available = pd.DatetimeIndex(weekly.time.values)
    positions = available.get_indexer(pd.DatetimeIndex(dates))
    if np.any(positions < 0):
        missing = [dates[index] for index in np.flatnonzero(positions < 0)]
        raise KeyError(
            f"ERA5 is missing {len(missing)} climatology dates for {valid_start.date()}; "
            f"first missing date is {missing[0].date()}"
        )
    samples = np.asarray(weekly.isel(time=positions).values, dtype=np.float32)
    if samples.shape[0] != 100:
        raise RuntimeError("a 20-year climatology must contain exactly 100 samples")
    return np.nanquantile(samples, BOUNDARY_QUANTILES, axis=0).astype(np.float32)


def observed_category(observation: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Match AI-WQ: lower inclusive, upper exclusive, equality goes upward."""

    if thresholds.shape[0] != 4 or observation.shape != thresholds.shape[1:]:
        raise ValueError("thresholds must be [4, lat, lon] and match observation")
    valid = np.isfinite(observation) & np.isfinite(thresholds).all(axis=0)
    all_equal = np.ptp(thresholds, axis=0) == 0.0
    category = np.sum(observation[None] >= thresholds, axis=0).astype(np.int8)
    category[~valid | all_equal] = -1
    return category


def ensemble_probabilities(members: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert members to five bins and apply Jeffreys-style 0.5 smoothing."""

    if members.ndim != 3 or thresholds.shape != (4, *members.shape[1:]):
        raise ValueError("members must be [member, lat, lon] with four matching bounds")
    if not np.isfinite(members).all() or not np.isfinite(thresholds).all():
        raise ValueError("ensemble and threshold values must be finite")
    categories = np.sum(members[:, None] >= thresholds[None], axis=1)
    counts = np.stack([(categories == index).sum(axis=0) for index in range(5)])
    probabilities = (counts + 0.5) / (members.shape[0] + 2.5)
    return probabilities.astype(np.float32)


def _static_features(
    p0: np.ndarray,
    tp_quantiles: np.ndarray,
    init_date: pd.Timestamp,
    latitude: np.ndarray,
    longitude: np.ndarray,
    land_fraction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    leads, _, height, width = p0.shape
    features = np.empty((leads, 18, height, width), dtype=np.float32)
    features[:, :5] = np.log(np.maximum(p0, 1.0e-8))
    features[:, 5:10] = (tp_quantiles - mean[:, :, None, None]) / std[:, :, None, None]

    lat_radians = np.deg2rad(latitude)[:, None]
    lon_radians = np.deg2rad(longitude)[None, :]
    features[:, 10] = np.sin(lat_radians)
    features[:, 11] = np.cos(lat_radians)
    features[:, 12] = np.sin(lon_radians)
    features[:, 13] = np.cos(lon_radians)
    for lead_index, (start_day, _) in enumerate(EXPERIMENT.lead_windows):
        valid_start = init_date + pd.Timedelta(days=start_day - 1)
        angle = 2.0 * np.pi * (valid_start.dayofyear - 1) / 365.2425
        features[lead_index, 14] = np.sin(angle)
        features[lead_index, 15] = np.cos(angle)
        features[lead_index, 16] = -1.0 if lead_index == 0 else 1.0
    features[:, 17] = land_fraction
    return features


def _create_cache(path: Path, cases: int, latitude: np.ndarray, longitude: np.ndarray):
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as error:
        raise RuntimeError("zarr and numcodecs are required to build the cache") from error
    if path.exists():
        group = zarr.open_group(str(path), mode="a")
        expected = (cases, 2, 5, len(latitude), len(longitude))
        if tuple(group["p0"].shape) != expected:
            raise ValueError("existing cache has a different case or grid shape")
        return group

    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    height, width = len(latitude), len(longitude)
    group.attrs.update(
        {
            "status": "building",
            "feature_names": list(EXPERIMENT.feature_names),
            "lead_windows": [list(window) for window in EXPERIMENT.lead_windows],
            "source_store": str(PATHS.fuxi_store),
            "tp_source_units": "native FuXi 24-hour mean rate",
            "tp_conversion": "clip at zero, multiply by 24 to mm/day, sum seven days",
            "climatology": "previous 20 years x offsets [-4,-2,0,2,4]",
            "fuxi_competition_use": "written_permission_required",
        }
    )

    def create(name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype, fill):
        return group.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill,
            compressor=compressor,
        )

    create("p0", (cases, 2, 5, height, width), (1, 1, 5, height, width), "f2", np.nan)
    create(
        "raw_tp_quantiles",
        (cases, 2, 5, height, width),
        (1, 1, 5, height, width),
        "f2",
        np.nan,
    )
    create("thresholds", (cases, 2, 4, height, width), (1, 1, 4, height, width), "f4", np.nan)
    create("target", (cases, 2, height, width), (1, 1, height, width), "i1", -1)
    create("features", (cases, 2, 18, height, width), (1, 1, 18, height, width), "f2", np.nan)
    create("case_complete", (cases,), (min(cases, 512),), "bool", False)
    create("feature_complete", (cases,), (min(cases, 512),), "bool", False)
    create("init_yyyymmdd", (cases,), (min(cases, 512),), "i4", 0)
    create("valid_start_yyyymmdd", (cases, 2), (min(cases, 512), 2), "i4", 0)
    create("latitude", (height,), (height,), "f8", np.nan)[:] = latitude
    create("longitude", (width,), (width,), "f8", np.nan)[:] = longitude
    create("land_fraction", (height, width), (height, width), "f4", np.nan)
    return group


def _normalization(group, train_case_indices: np.ndarray, land_fraction: np.ndarray):
    mask = land_fraction >= 0.5
    total = np.zeros((2, 5), dtype=np.float64)
    total_square = np.zeros((2, 5), dtype=np.float64)
    count = np.zeros((2, 5), dtype=np.int64)
    for case_index in train_case_indices:
        fields = np.asarray(group["raw_tp_quantiles"][int(case_index)], dtype=np.float32)
        for lead in range(2):
            values = fields[lead, :, mask]
            finite = np.isfinite(values)
            total[lead] += np.where(finite, values, 0.0).sum(axis=1)
            total_square[lead] += np.where(finite, values * values, 0.0).sum(axis=1)
            count[lead] += finite.sum(axis=1)
    if np.any(count == 0):
        raise RuntimeError("normalization has an empty lead/quantile channel")
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _official_cache_contract_sha256(group) -> str:
    """Fingerprint the exact split and preprocessing contract saved in a cache."""

    land = np.ascontiguousarray(np.asarray(group["land_fraction"][:], dtype=np.float32))
    contract = {
        "schema": "quest-tp-cache-v1",
        "feature_names": list(group.attrs["feature_names"]),
        "lead_windows": list(group.attrs["lead_windows"]),
        "source_store": str(group.attrs["source_store"]),
        "tp_conversion": str(group.attrs["tp_conversion"]),
        "climatology": str(group.attrs["climatology"]),
        "train_years": list(EXPERIMENT.train_years),
        "validation_years": list(EXPERIMENT.validation_years),
        "test_years": list(EXPERIMENT.test_years),
        "init_yyyymmdd": np.asarray(group["init_yyyymmdd"][:], dtype=np.int32).tolist(),
        "valid_start_yyyymmdd": np.asarray(
            group["valid_start_yyyymmdd"][:], dtype=np.int32
        ).tolist(),
        "tp_quantile_mean": list(group.attrs["tp_quantile_mean"]),
        "tp_quantile_std": list(group.attrs["tp_quantile_std"]),
        "land_fraction_sha256": hashlib.sha256(land.tobytes()).hexdigest(),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_fuxi_archive(dataset: xr.Dataset) -> None:
    """Validate the archive and its extraction-time QC without reading 13 TB."""

    if dataset.attrs.get("status") != "complete":
        raise RuntimeError("FuXi source Zarr is not complete")
    expected = {
        "init": 2080,
        "member": 51,
        "lead_day": 42,
        "channel": 26,
        "lat": 121,
        "lon": 240,
    }
    for dimension, size in expected.items():
        if dataset.sizes.get(dimension) != size:
            raise ValueError(f"{dimension} has size {dataset.sizes.get(dimension)}, expected {size}")
    if tuple(str(value) for value in dataset.channel.values) != EXPECTED_FUXI_CHANNELS:
        raise ValueError("FuXi archive does not contain the expected 26 channels in order")
    if not np.allclose(dataset.lat.values, EXPECTED_LATITUDE, atol=1.0e-6):
        raise ValueError("FuXi archive has the wrong latitude grid")
    if not np.allclose(dataset.lon.values, EXPECTED_LONGITUDE, atol=1.0e-6):
        raise ValueError("FuXi archive has the wrong longitude grid")
    if not np.array_equal(dataset.member.values, np.arange(51)):
        raise ValueError("FuXi archive member coordinates must be 0..50")
    if not np.array_equal(dataset.lead_day.values, np.arange(1, 43)):
        raise ValueError("FuXi archive lead coordinates must be 1..42")

    required_qc = ("init_complete", "qc_min", "qc_mean", "qc_max")
    missing_qc = [name for name in required_qc if name not in dataset]
    if missing_qc:
        raise ValueError(f"FuXi archive is missing extraction QC arrays: {missing_qc}")

    init_dates = pd.DatetimeIndex(dataset.init.values)
    if init_dates.has_duplicates or not init_dates.is_monotonic_increasing:
        raise ValueError("FuXi initialization dates must be unique and increasing")
    year_counts = init_dates.year.value_counts().sort_index()
    if tuple(year_counts.index) != tuple(range(2002, 2022)) or not np.all(
        year_counts.values == 104
    ):
        raise ValueError("FuXi archive must contain 104 initializations in every year 2002–2021")

    # These JSON files were written while every full forecast case was scanned.
    # Reading them is much faster than opening 8,320 one-case Zarr QC chunks.
    qc_directory_value = dataset.attrs.get("qc_directory")
    if not qc_directory_value:
        raise ValueError("FuXi archive does not record its extraction QC directory")
    qc_directory = Path(str(qc_directory_value))
    expected_qc_names = {
        f"{date.strftime('%Y%m%d')}.json" for date in init_dates
    }
    actual_qc_paths = sorted(qc_directory.glob("*.json"))
    if {path.name for path in actual_qc_paths} != expected_qc_names:
        raise RuntimeError("FuXi extraction QC files do not match all initialization dates")
    for path in actual_qc_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("finite") is not True:
            raise RuntimeError(f"FuXi extraction QC is not finite: {path}")
        if tuple(record.get("shape", ())) != (51, 42, 26, 121, 240):
            raise RuntimeError(f"FuXi extraction QC has the wrong case shape: {path}")
        if tuple(record.get("channel_names", ())) != EXPECTED_FUXI_CHANNELS:
            raise RuntimeError(f"FuXi extraction QC has the wrong channels: {path}")
        for name in ("channel_min", "channel_mean", "channel_max"):
            values = np.asarray(record.get(name, ()), dtype=np.float64)
            if values.shape != (26,) or not np.isfinite(values).all():
                raise RuntimeError(f"FuXi extraction QC has invalid {name}: {path}")


def inspect_fuxi(args: argparse.Namespace) -> None:
    dataset = xr.open_zarr(args.fuxi_store, consolidated=True, chunks=None)
    validate_fuxi_archive(dataset)
    sample = dataset.forecast.sel(channel="tp", lead_day=list(range(19, 26))).isel(
        init=args.sample_index, member=slice(0, 2), lat=slice(0, 3), lon=slice(0, 3)
    )
    values = np.asarray(sample.values)
    print(dataset)
    print(f"source status: {dataset.attrs['status']}")
    print("metadata/QC: all 2,080 initializations and 26 channels complete")
    print(f"channels: {', '.join(EXPECTED_FUXI_CHANNELS)}")
    print(
        "full_data_verification attribute: "
        f"{dataset.attrs.get('full_data_verification', 'not recorded')}"
    )
    print(f"TP sample native rate: min={values.min():.6g} max={values.max():.6g}")
    print(f"TP sample weekly mm after x24 and sum: {(values.clip(0) * 24).sum(axis=1).mean():.6g}")


def _raw_run_initialization(raw_directory: Path) -> pd.Timestamp | None:
    for metadata_path in (
        raw_directory.parent / "public-v2" / "metadata.json",
        raw_directory.parent / "public" / "metadata.json",
    ):
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = metadata.get("issue", {}).get("initialization")
        if value:
            timestamp = pd.Timestamp(value)
            if timestamp.tz is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            return timestamp.normalize()
    return None


def load_operational_members(raw_directory: Path) -> np.ndarray:
    """Read TP only from FuXi's ``raw/member/NN/DD.nc`` layout."""

    member_root = Path(raw_directory) / "member"
    member_directories = sorted(path for path in member_root.iterdir() if path.is_dir())
    if len(member_directories) < 2:
        raise FileNotFoundError(f"no FuXi ensemble member directories under {member_root}")
    output = np.empty(
        (len(member_directories), 14, len(EXPECTED_LATITUDE), len(EXPECTED_LONGITUDE)),
        dtype=np.float32,
    )
    for member_index, member_directory in enumerate(member_directories):
        for local_lead, lead_day in enumerate(range(19, 33)):
            path = member_directory / f"{lead_day:02d}.nc"
            if not path.exists():
                raise FileNotFoundError(path)
            with xr.open_dataset(path, chunks=None) as dataset:
                if len(dataset.data_vars) != 1:
                    raise ValueError(f"expected one field array in {path}")
                field = next(iter(dataset.data_vars.values()))
                if "channel" not in field.dims or "tp" not in field.channel.values:
                    raise ValueError(f"{path} has no TP channel")
                if "lead_time" in field.coords and int(field.lead_time.item()) != lead_day:
                    raise ValueError(f"{path} contains the wrong lead coordinate")
                latitude = np.asarray(field["lat"].values)
                longitude = np.asarray(field["lon"].values)
                if not np.allclose(latitude, EXPECTED_LATITUDE, atol=1.0e-6):
                    raise ValueError(f"{path} has the wrong latitude grid")
                if not np.allclose(longitude, EXPECTED_LONGITUDE, atol=1.0e-6):
                    raise ValueError(f"{path} has the wrong longitude grid")
                values = np.asarray(field.sel(channel="tp").squeeze(drop=True).values)
                if values.shape != (121, 240) or not np.isfinite(values).all():
                    raise ValueError(f"invalid TP field in {path}: {values.shape}")
                output[member_index, local_lead] = np.clip(values, 0.0, None) * 24.0
        print(
            f"loaded member {member_index + 1}/{len(member_directories)}",
            flush=True,
        )
    return output


def prepare_operational_case(args: argparse.Namespace) -> None:
    """Create the one-case NPZ consumed by predict.py from a global raw run."""

    init_date = pd.Timestamp(args.init_date).normalize()
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    metadata_date = _raw_run_initialization(args.raw_directory)
    if metadata_date is not None and metadata_date != init_date:
        raise ValueError(
            f"requested initialization {init_date.date()} but run metadata says "
            f"{metadata_date.date()}"
        )
    weekly_era5 = load_era5_weekly(args.era5_root)
    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("zarr is required to read training normalization") from error
    cache = zarr.open_group(str(args.cache), mode="r")
    if cache.attrs.get("status") != "complete":
        raise RuntimeError("training cache is not complete")
    cache_fingerprint = str(
        cache.attrs.get("cache_contract_sha256", "")
    ).strip().lower()
    if len(cache_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in cache_fingerprint
    ):
        raise ValueError(
            "training cache must declare a valid cache_contract_sha256"
        )
    mean = np.asarray(cache.attrs["tp_quantile_mean"], dtype=np.float32)
    std = np.asarray(cache.attrs["tp_quantile_std"], dtype=np.float32)
    if mean.shape != (2, 5) or std.shape != (2, 5):
        raise ValueError("cache normalization must have shape [2,5]")
    land = np.asarray(cache["land_fraction"][:], dtype=np.float32)
    daily_mm = load_operational_members(args.raw_directory)

    anchors = np.empty((2, 5, 121, 240), dtype=np.float32)
    raw_quantiles = np.empty_like(anchors)
    thresholds = np.empty((2, 4, 121, 240), dtype=np.float32)
    for lead_index, (start_day, end_day) in enumerate(EXPERIMENT.lead_windows):
        local_start = start_day - 19
        local_end = end_day - 19 + 1
        member_weekly = daily_mm[:, local_start:local_end].sum(axis=1)
        valid_start = init_date + pd.Timedelta(days=start_day - 1)
        bounds = climatology_thresholds(weekly_era5, valid_start)
        anchors[lead_index] = ensemble_probabilities(member_weekly, bounds)
        raw_quantiles[lead_index] = np.quantile(
            np.log1p(member_weekly), TP_QUANTILES, axis=0
        ).astype(np.float32)
        thresholds[lead_index] = bounds
    features = _static_features(
        anchors,
        raw_quantiles,
        init_date,
        EXPECTED_LATITUDE,
        EXPECTED_LONGITUDE,
        land,
        mean,
        std,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features[None].astype(np.float16),
        p0=anchors[None],
        thresholds=thresholds[None],
        latitude=EXPECTED_LATITUDE,
        longitude=EXPECTED_LONGITUDE,
        land_fraction=land,
        init_dates=np.asarray([init_date.strftime("%Y-%m-%d")]),
        cache_contract_sha256=np.asarray(cache_fingerprint),
    )
    print(f"prepared local forecast case: {args.output}")


def build_cache(args: argparse.Namespace) -> None:
    weekly = load_era5_weekly(args.era5_root)
    land = load_land_fraction(args.land_fraction)
    source = xr.open_zarr(args.fuxi_store, consolidated=True, chunks=None)
    validate_fuxi_archive(source)

    init_dates = pd.DatetimeIndex(source.init.values).normalize()
    requested_years = set(
        EXPERIMENT.train_years + EXPERIMENT.validation_years + EXPERIMENT.test_years
    )
    selected = np.flatnonzero(init_dates.year.isin(requested_years))
    if args.limit_cases:
        selected = selected[: args.limit_cases]
    selected_dates = init_dates[selected]
    group = _create_cache(args.output, len(selected), EXPECTED_LATITUDE, EXPECTED_LONGITUDE)
    group["land_fraction"][:] = np.asarray(land.values, dtype=np.float32)
    expected_init_values = np.asarray(
        [int(date.strftime("%Y%m%d")) for date in selected_dates], dtype=np.int32
    )
    existing_init_values = np.asarray(group["init_yyyymmdd"][:], dtype=np.int32)
    populated = existing_init_values != 0
    if np.any(populated) and not np.array_equal(
        existing_init_values[populated], expected_init_values[populated]
    ):
        raise ValueError("existing cache contains different initialization dates")
    group["init_yyyymmdd"][:] = expected_init_values

    for output_index, (source_index, init_date) in enumerate(zip(selected, selected_dates)):
        if bool(group["case_complete"][output_index]):
            print(f"case {output_index + 1}/{len(selected)} exists {init_date.date()}")
            continue
        print(f"case {output_index + 1}/{len(selected)} {init_date.date()}", flush=True)
        daily_rate = source.forecast.sel(channel="tp").isel(init=int(source_index)).sel(
            lead_day=slice(19, 32)
        )
        values = np.asarray(daily_rate.values, dtype=np.float32)
        if values.shape != (51, 14, 121, 240) or not np.isfinite(values).all():
            raise ValueError(f"invalid FuXi TP slice for {init_date.date()}: {values.shape}")
        daily_mm = np.clip(values, 0.0, None) * np.float32(24.0)

        anchors = np.empty((2, 5, 121, 240), dtype=np.float32)
        raw_quantiles = np.empty_like(anchors)
        thresholds = np.empty((2, 4, 121, 240), dtype=np.float32)
        targets = np.empty((2, 121, 240), dtype=np.int8)
        valid_starts = []
        for lead_index, (start_day, end_day) in enumerate(EXPERIMENT.lead_windows):
            local_start = start_day - 19
            local_end = end_day - 19 + 1
            member_weekly = daily_mm[:, local_start:local_end].sum(axis=1)
            valid_start = init_date + pd.Timedelta(days=start_day - 1)
            valid_starts.append(int(valid_start.strftime("%Y%m%d")))
            bounds = climatology_thresholds(weekly, valid_start)
            observation = np.asarray(weekly.sel(time=valid_start).values, dtype=np.float32)
            anchors[lead_index] = ensemble_probabilities(member_weekly, bounds)
            raw_quantiles[lead_index] = np.quantile(
                np.log1p(member_weekly), TP_QUANTILES, axis=0
            ).astype(np.float32)
            thresholds[lead_index] = bounds
            targets[lead_index] = observed_category(observation, bounds)

        group["p0"][output_index] = anchors.astype(np.float16)
        group["raw_tp_quantiles"][output_index] = raw_quantiles.astype(np.float16)
        group["thresholds"][output_index] = thresholds
        group["target"][output_index] = targets
        group["valid_start_yyyymmdd"][output_index] = np.asarray(valid_starts, dtype=np.int32)
        group["case_complete"][output_index] = True

    if not np.asarray(group["case_complete"][:]).all():
        raise RuntimeError("not all source cases were prepared")
    train_indices = np.flatnonzero(selected_dates.year.isin(EXPERIMENT.train_years))
    if not len(train_indices):
        raise RuntimeError("cache contains no configured training cases")
    mean, std = _normalization(group, train_indices, np.asarray(land.values))
    group.attrs["tp_quantile_mean"] = mean.tolist()
    group.attrs["tp_quantile_std"] = std.tolist()

    for index, init_date in enumerate(selected_dates):
        if bool(group["feature_complete"][index]):
            continue
        features = _static_features(
            np.asarray(group["p0"][index], dtype=np.float32),
            np.asarray(group["raw_tp_quantiles"][index], dtype=np.float32),
            init_date,
            EXPECTED_LATITUDE,
            EXPECTED_LONGITUDE,
            np.asarray(land.values, dtype=np.float32),
            mean,
            std,
        )
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite features for {init_date.date()}")
        group["features"][index] = features.astype(np.float16)
        group["feature_complete"][index] = True

    group.attrs["cache_contract_sha256"] = _official_cache_contract_sha256(group)
    group.attrs["status"] = "complete"
    group.attrs["completed_utc"] = datetime.utcnow().isoformat() + "Z"
    print(f"complete cache: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-fuxi", help="validate the real archive contract")
    inspect_parser.add_argument("--fuxi-store", type=Path, default=PATHS.fuxi_store)
    inspect_parser.add_argument("--sample-index", type=int, default=0)
    inspect_parser.set_defaults(function=inspect_fuxi)

    download_parser = subparsers.add_parser("download-era5", help="retrieve official annual references")
    download_parser.add_argument("--start-year", type=int, default=1997)
    download_parser.add_argument("--end-year", type=int, default=2025)
    download_parser.add_argument("--destination", type=Path, default=PATHS.era5_root)
    download_parser.add_argument("--password-env", default="AI_WQ_PASSWORD")
    download_parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(TRAINING_FILE_STATISTIC),
        default=["pr"],
        help="official annual fields to retrieve (default: pr)",
    )
    download_parser.set_defaults(function=download_era5)

    build_parser = subparsers.add_parser("build-cache", help="stream FuXi TP into a small Zarr")
    build_parser.add_argument("--fuxi-store", type=Path, default=PATHS.fuxi_store)
    build_parser.add_argument("--era5-root", type=Path, default=PATHS.era5_root)
    build_parser.add_argument(
        "--land-fraction",
        type=Path,
        default=PATHS.era5_root / "land_sea_mask_1pt5DEG.nc",
    )
    build_parser.add_argument("--output", type=Path, default=PATHS.cache_store)
    build_parser.add_argument("--limit-cases", type=int)
    build_parser.set_defaults(function=build_cache)

    case_parser = subparsers.add_parser(
        "prepare-case", help="prepare one operational global FuXi run for predict.py"
    )
    case_parser.add_argument("--raw-directory", type=Path, required=True)
    case_parser.add_argument("--init-date", required=True, help="YYYY-MM-DD")
    case_parser.add_argument("--cache", type=Path, default=PATHS.cache_store)
    case_parser.add_argument("--era5-root", type=Path, default=PATHS.era5_root)
    case_parser.add_argument("--output", type=Path, required=True)
    case_parser.set_defaults(function=prepare_operational_case)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
