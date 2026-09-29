#!/usr/bin/env python3
"""Build atomic per-case parts and finalize the frozen precipitation cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import xarray as xr

from precip_contract import (
    CONTEXT_DYNAMIC_NAMES,
    GRID_SHAPE,
    MEMBER_QUANTILES,
    N_MEMBERS,
    PHYSICAL_VARIABLES,
    SPLIT,
    TARGET_WINDOWS,
    aggregate_era5_six_hour_week,
    build_model_inputs,
    climatology_boundaries,
    climatology_sample_starts,
    contract_sha256,
    fit_train_land_normalization,
    jeffreys_probabilities,
    member_category_counts,
    observation_categories,
    validate_archive_dates,
    weekly_fuxi_fields,
)


DEFAULT_FUXI = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_global_2002_2021.zarr"
)
DEFAULT_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813"
)
FUXI_CHANNELS = ("tp", "tcwv", "q850", "q500", "q250", "t2m")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_grid(field: xr.DataArray) -> xr.DataArray:
    rename = {}
    for candidates, canonical in (
        (("latitude", "lat"), "lat"),
        (("longitude", "lon"), "lon"),
        (("valid_time", "time"), "time"),
    ):
        found = next((name for name in candidates if name in field.dims), None)
        if found and found != canonical:
            rename[found] = canonical
    if rename:
        field = field.rename(rename)
    if not {"time", "lat", "lon"}.issubset(field.dims):
        raise ValueError("ERA5 field must contain time/lat/lon dimensions")
    field = field.assign_coords(lon=np.mod(field.lon.values, 360.0)).sortby("lon")
    field = field.sortby("lat", ascending=False)
    expected_lat = np.linspace(90.0, -90.0, GRID_SHAPE[0])
    expected_lon = np.arange(GRID_SHAPE[1]) * 1.5
    if not np.allclose(field.lat.values, expected_lat, atol=1.0e-6):
        raise ValueError("ERA5 latitude is not the exact Quest grid")
    if not np.allclose(field.lon.values, expected_lon, atol=1.0e-6):
        raise ValueError("ERA5 longitude is not the exact Quest grid")
    return field.transpose("time", "lat", "lon")


def open_era5(path: Path, variable: str) -> xr.DataArray:
    dataset = xr.open_zarr(path, consolidated=True, chunks=None)
    if variable not in dataset:
        raise KeyError(f"{path} has no {variable!r}; available={list(dataset.data_vars)}")
    field = _normalize_grid(dataset[variable])
    units = str(field.attrs.get("units", "")).strip().lower()
    if units not in {"m", "metre", "metres", "meter", "meters"}:
        raise ValueError(f"ERA5 precipitation must be in metres, got {units!r}")
    return field


def load_land_fraction(path: Path) -> np.ndarray:
    with xr.open_dataset(path, chunks=None) as dataset:
        candidates = [
            value
            for value in dataset.data_vars.values()
            if any(name in value.dims for name in ("lat", "latitude"))
            and any(name in value.dims for name in ("lon", "longitude"))
        ]
        if len(candidates) != 1:
            raise ValueError("land file must contain exactly one gridded field")
        field = _normalize_grid(candidates[0].expand_dims(time=[pd.Timestamp("2000-01-01")]))
        values = np.asarray(field.isel(time=0).values, dtype=np.float32)
    if values.shape != GRID_SHAPE or not np.isfinite(values).all():
        raise ValueError("land fraction is not a finite Quest-grid field")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("land fraction must lie in [0,1]")
    return values


def _week(era5: xr.DataArray, start: pd.Timestamp) -> np.ndarray:
    return np.asarray(aggregate_era5_six_hour_week(era5, start).values, dtype=np.float32)


def _part_path(parts_root: Path, index: int, init_date: pd.Timestamp) -> Path:
    return parts_root / f"{index:04d}_{init_date.strftime('%Y%m%d')}.npz"


def _part_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as part:
        return {
            "index": int(part["source_index"]),
            "init": int(part["init_yyyymmdd"]),
            "contract": str(part["contract_sha256"].item()),
        }


def validate_part(
    path: Path,
    index: int,
    init_date: pd.Timestamp,
    *,
    expected_contract_sha256: str | None = None,
) -> None:
    expected_contract_sha256 = expected_contract_sha256 or contract_sha256()
    with np.load(path, allow_pickle=False) as part:
        expected = {
            "context_dynamic": (6, 15, *GRID_SHAPE),
            "target_quantiles": (2, 5, *GRID_SHAPE),
            "category_counts": (2, 5, *GRID_SHAPE),
            "p0": (2, 5, *GRID_SHAPE),
            "thresholds": (2, 4, *GRID_SHAPE),
            "target": (2, *GRID_SHAPE),
            "target_valid": (2, *GRID_SHAPE),
        }
        missing = [name for name in expected if name not in part]
        if missing:
            raise ValueError(f"{path} is missing arrays {missing}")
        for name, shape in expected.items():
            if part[name].shape != shape:
                raise ValueError(f"{path}:{name} has shape {part[name].shape}, expected {shape}")
        if int(part["source_index"]) != index:
            raise ValueError(f"{path} has the wrong source index")
        if int(part["init_yyyymmdd"]) != int(init_date.strftime("%Y%m%d")):
            raise ValueError(f"{path} has the wrong initialization")
        if str(part["contract_sha256"].item()) != expected_contract_sha256:
            raise ValueError(
                f"{path} has contract {str(part['contract_sha256'].item())!r}, "
                f"expected {expected_contract_sha256!r}"
            )
        counts = np.asarray(part["category_counts"])
        if not np.all(counts.sum(axis=1) == N_MEMBERS):
            raise ValueError(f"{path} category counts do not sum to 51")
        p0 = np.asarray(part["p0"], dtype=np.float32)
        np.testing.assert_allclose(p0, jeffreys_probabilities(counts), atol=2.0e-4)
        if not np.isfinite(part["context_dynamic"]).all():
            raise ValueError(f"{path} has non-finite predictors")


def prepare_part(
    source: xr.Dataset,
    era5: xr.DataArray,
    index: int,
    parts_root: Path,
) -> Path:
    init_date = pd.Timestamp(source.init.values[index]).normalize()
    output = _part_path(parts_root, index, init_date)
    if output.exists():
        validate_part(output, index, init_date)
        print(f"exists {index:04d} {init_date.date()}", flush=True)
        return output

    daily = {}
    for variable in FUXI_CHANNELS:
        values = source.forecast.isel(init=index).sel(channel=variable).values
        daily[variable] = np.asarray(values, dtype=np.float32)
    context, target_members = weekly_fuxi_fields(daily)
    target_quantiles = np.moveaxis(
        np.quantile(np.log1p(target_members), MEMBER_QUANTILES, axis=1), 0, 1
    ).astype(np.float32)
    thresholds = np.empty((2, 4, *GRID_SHAPE), dtype=np.float32)
    target = np.empty((2, *GRID_SHAPE), dtype=np.int8)
    target_valid = np.empty((2, *GRID_SHAPE), dtype=bool)
    counts = np.empty((2, 5, *GRID_SHAPE), dtype=np.uint8)
    for lead, (start_day, _end_day) in enumerate(TARGET_WINDOWS):
        valid_start = init_date + pd.Timedelta(days=start_day - 1)
        samples = np.stack(
            [_week(era5, sample_start) for sample_start in climatology_sample_starts(valid_start)]
        )
        thresholds[lead] = climatology_boundaries(samples)
        observation = _week(era5, valid_start)
        target[lead], target_valid[lead] = observation_categories(
            observation, thresholds[lead]
        )
        counts[lead] = member_category_counts(target_members[lead], thresholds[lead])
    p0 = jeffreys_probabilities(counts)

    parts_root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".npz.part-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_index=np.asarray(index, dtype=np.int32),
            init_yyyymmdd=np.asarray(int(init_date.strftime("%Y%m%d")), dtype=np.int32),
            contract_sha256=np.asarray(contract_sha256()),
            context_dynamic=context.astype(np.float16),
            target_quantiles=target_quantiles.astype(np.float16),
            category_counts=counts,
            p0=p0.astype(np.float32),
            thresholds=thresholds,
            target=target,
            target_valid=target_valid,
        )
    temporary.replace(output)
    validate_part(output, index, init_date)
    print(f"complete {index:04d} {init_date.date()} {output}", flush=True)
    return output


def prepare_task(args: argparse.Namespace) -> None:
    source = xr.open_zarr(args.fuxi_store, consolidated=True, chunks=None)
    validate_archive_dates(source.init.values)
    if source.sizes.get("member") != N_MEMBERS or source.sizes.get("lead_day") != 42:
        raise ValueError("FuXi source must have 51 members and 42 lead days")
    if not set(FUXI_CHANNELS).issubset({str(item) for item in source.channel.values}):
        raise ValueError("FuXi source lacks one or more required channels")
    era5 = open_era5(args.era5_store, args.era5_variable)
    if args.indices:
        indices = [int(item) for item in args.indices.split(",")]
    else:
        indices = list(range(args.task_id, source.sizes["init"], args.task_count))
    for index in indices:
        if not 0 <= index < 2080:
            raise IndexError(f"source index {index} outside 0..2079")
        prepare_part(source, era5, index, args.parts_root)


def _create_array(group, name, shape, chunks, dtype, fill_value):
    from numcodecs import Blosc

    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )


def _fit_target_normalization(parts: list[Path], indices: list[int], land: np.ndarray):
    mask = land >= 0.5
    total = np.zeros((2, 5), dtype=np.float64)
    square = np.zeros_like(total)
    count = np.zeros((2, 5), dtype=np.int64)
    for index in indices:
        with np.load(parts[index], allow_pickle=False) as part:
            values = np.asarray(part["target_quantiles"], dtype=np.float32)[:, :, mask]
        finite = np.isfinite(values)
        total += np.where(finite, values, 0.0).sum(axis=-1)
        square += np.where(finite, values * values, 0.0).sum(axis=-1)
        count += finite.sum(axis=-1)
    if np.any(count == 0):
        raise RuntimeError("target normalization has an empty channel")
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def finalize(args: argparse.Namespace) -> None:
    import zarr

    source = xr.open_zarr(args.fuxi_store, consolidated=True, chunks=None)
    dates = pd.DatetimeIndex(source.init.values).normalize()
    validate_archive_dates(dates)
    parts = [_part_path(args.parts_root, index, date) for index, date in enumerate(dates)]
    missing = [path for path in parts if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot finalize: {len(missing)} of 2,080 parts are missing")
    parts_contract_sha256 = args.parts_contract_sha256 or contract_sha256()
    for index, (path, init_date) in enumerate(zip(parts, dates)):
        validate_part(
            path,
            index,
            init_date,
            expected_contract_sha256=parts_contract_sha256,
        )

    land = load_land_fraction(args.land_fraction)
    train_indices = [index for index, value in enumerate(dates) if value.year in SPLIT.train_years]
    expected_train_cases = 104 * len(SPLIT.train_years)
    if len(train_indices) != expected_train_cases:
        raise RuntimeError(
            f"expected {expected_train_cases:,} training cases, found {len(train_indices)}"
        )

    def training_fields():
        for index in train_indices:
            with np.load(parts[index], allow_pickle=False) as part:
                yield np.asarray(part["context_dynamic"], dtype=np.float32)

    mean, std = fit_train_land_normalization(training_fields(), land)
    target_mean, target_std = _fit_target_normalization(parts, train_indices, land)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite finalized cache {args.output}")
    staging = args.output.parent / f".{args.output.name}.incomplete-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(staging), mode="w")
    group.attrs.update(
        status="building",
        contract_sha256=contract_sha256(),
        parts_contract_sha256=parts_contract_sha256,
        context_dynamic_names=list(CONTEXT_DYNAMIC_NAMES),
        train_years=list(SPLIT.train_years),
        validation_years=list(SPLIT.validation_years),
        test_years=list(SPLIT.test_years),
        normalization_scope=(
            f"{min(SPLIT.train_years)}-{max(SPLIT.train_years)} forecasts over "
            "supplied land_fraction>=0.5 only"
        ),
        land_fraction_source=str(args.land_fraction),
        context_mean=mean.tolist(),
        context_std=std.tolist(),
        target_quantile_mean=target_mean.tolist(),
        target_quantile_std=target_std.tolist(),
        era5_source=str(args.era5_store),
        fuxi_source=str(args.fuxi_store),
        created_utc=utc_now(),
    )
    n = 2080
    arrays = {
        "context_x": _create_array(group, "context_x", (n, 6, 23, *GRID_SHAPE), (1, 1, 23, *GRID_SHAPE), "f2", np.nan),
        "target_x": _create_array(group, "target_x", (n, 2, 18, *GRID_SHAPE), (1, 1, 18, *GRID_SHAPE), "f2", np.nan),
        "category_counts": _create_array(group, "category_counts", (n, 2, 5, *GRID_SHAPE), (1, 1, 5, *GRID_SHAPE), "u1", 0),
        "p0": _create_array(group, "p0", (n, 2, 5, *GRID_SHAPE), (1, 1, 5, *GRID_SHAPE), "f4", np.nan),
        "thresholds": _create_array(group, "thresholds", (n, 2, 4, *GRID_SHAPE), (1, 1, 4, *GRID_SHAPE), "f4", np.nan),
        "target": _create_array(group, "target", (n, 2, *GRID_SHAPE), (1, 1, *GRID_SHAPE), "i1", -1),
        "target_valid": _create_array(group, "target_valid", (n, 2, *GRID_SHAPE), (1, 1, *GRID_SHAPE), "bool", False),
    }
    _create_array(group, "init_yyyymmdd", (n,), (n,), "i4", 0)[:] = np.asarray(
        [int(value.strftime("%Y%m%d")) for value in dates], dtype=np.int32
    )
    _create_array(group, "land_fraction", GRID_SHAPE, GRID_SHAPE, "f4", np.nan)[:] = land
    _create_array(group, "latitude", (121,), (121,), "f8", np.nan)[:] = np.linspace(90, -90, 121)
    _create_array(group, "longitude", (240,), (240,), "f8", np.nan)[:] = np.arange(240) * 1.5
    completed = _create_array(group, "case_complete", (n,), (n,), "bool", False)

    try:
        for index, (path, init_date) in enumerate(zip(parts, dates)):
            with np.load(path, allow_pickle=False) as part:
                p0 = jeffreys_probabilities(part["category_counts"])
                context_x, target_x = build_model_inputs(
                    np.asarray(part["context_dynamic"], dtype=np.float32),
                    np.asarray(part["target_quantiles"], dtype=np.float32),
                    p0,
                    init_date,
                    land,
                    mean,
                    std,
                    target_mean,
                    target_std,
                )
                arrays["context_x"][index] = context_x.astype(np.float16)
                arrays["target_x"][index] = target_x.astype(np.float16)
                for name in ("category_counts", "thresholds", "target", "target_valid"):
                    arrays[name][index] = part[name]
                arrays["p0"][index] = p0.astype(np.float32)
                completed[index] = True
            if (index + 1) % 25 == 0:
                print(f"finalized {index + 1}/2080", flush=True)
        if not np.asarray(completed[:]).all():
            raise RuntimeError("final cache completion bitmap is not full")
        group.attrs["status"] = "complete"
        group.attrs["completed_utc"] = utc_now()
        zarr.consolidate_metadata(str(staging))
        staging.replace(args.output)
    except BaseException:
        # Preserve staging for diagnosis; its leading dot and status=building
        # ensure consumers cannot mistake it for the finalized cache.
        raise

    manifest = {
        "status": "complete",
        "contract_sha256": contract_sha256(),
        "cache": str(args.output),
        "cases": n,
        "split_counts": {
            "train": 104 * len(SPLIT.train_years),
            "validation": 104 * len(SPLIT.validation_years),
            "test": 104 * len(SPLIT.test_years),
        },
        "split_years": {
            "train": list(SPLIT.train_years),
            "validation": list(SPLIT.validation_years),
            "test": list(SPLIT.test_years),
        },
        "parts_contract_sha256": parts_contract_sha256,
        "parts_root": str(args.parts_root),
        "land_fraction_source": str(args.land_fraction),
        "completed_utc": utc_now(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"complete cache {args.output}")


def crosscheck(args: argparse.Namespace) -> None:
    era5 = open_era5(args.era5_store, args.era5_variable)
    with xr.open_dataset(args.official_weekly, chunks=None) as dataset:
        if len(dataset.data_vars) != 1:
            raise ValueError("official weekly file must have one data field")
        official = next(iter(dataset.data_vars.values()))
        if "time" not in official.dims and "valid_time" in official.dims:
            official = official.rename(valid_time="time")
        official = official.rename(
            {name: canonical for name, canonical in (("latitude", "lat"), ("longitude", "lon")) if name in official.dims}
        )
        official = _normalize_grid(official)
        units = str(official.attrs.get("units", "")).lower()
        if "mm" not in units:
            raise ValueError("official Quest weekly precipitation must be in millimetres")
        rows = []
        for value in args.valid_start:
            start = pd.Timestamp(value).normalize()
            wb2 = _week(era5, start)
            reference = np.asarray(official.sel(time=start).values, dtype=np.float32).squeeze()
            difference = wb2 - reference
            rows.append(
                {
                    "valid_start": start.strftime("%Y-%m-%d"),
                    "mean_abs_error_mm": float(np.mean(np.abs(difference))),
                    "max_abs_error_mm": float(np.max(np.abs(difference))),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if any(row["max_abs_error_mm"] > args.max_abs_error_mm for row in rows):
        raise RuntimeError("WeatherBench2 and official Quest weekly fields exceed tolerance")
    print(args.output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    part = subparsers.add_parser("part")
    part.add_argument("--fuxi-store", type=Path, default=DEFAULT_FUXI)
    part.add_argument("--era5-store", type=Path, required=True)
    part.add_argument("--era5-variable", default="total_precipitation_6hr")
    part.add_argument("--parts-root", type=Path, default=DEFAULT_ROOT / "parts")
    part.add_argument("--task-id", type=int, default=0)
    part.add_argument("--task-count", type=int, default=64)
    part.add_argument("--indices", help="comma-separated two-case smoke override")
    part.set_defaults(function=prepare_task)

    final = subparsers.add_parser("finalize")
    final.add_argument("--fuxi-store", type=Path, default=DEFAULT_FUXI)
    final.add_argument("--era5-store", type=Path, required=True)
    final.add_argument("--parts-root", type=Path, default=DEFAULT_ROOT / "parts")
    final.add_argument("--land-fraction", type=Path, required=True)
    final.add_argument(
        "--parts-contract-sha256",
        help="Explicit verified contract hash for reusable split-agnostic atomic parts",
    )
    final.add_argument("--output", type=Path, default=DEFAULT_ROOT / "cache" / "precip_2002_2021.zarr")
    final.set_defaults(function=finalize)

    check = subparsers.add_parser("crosscheck-official")
    check.add_argument("--era5-store", type=Path, required=True)
    check.add_argument("--era5-variable", default="total_precipitation_6hr")
    check.add_argument("--official-weekly", type=Path, required=True)
    check.add_argument("--valid-start", action="append", required=True)
    check.add_argument("--max-abs-error-mm", type=float, default=1.0e-3)
    check.add_argument("--output", type=Path, required=True)
    check.set_defaults(function=crosscheck)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
