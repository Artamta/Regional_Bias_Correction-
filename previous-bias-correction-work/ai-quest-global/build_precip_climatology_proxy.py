#!/usr/bin/env python3
"""Build an explicitly labelled lagged public-ERA5 precipitation climatology.

This is a credential-free operational fallback, not an official AI-WQ
climatology download.  It uses the most recent 20 complete years available in
the local WeatherBench2 ERA5 store and preserves the Quest 100-sample
``20 years x five date offsets`` construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from build_precip_cache import open_era5
from precip_contract import (
    BOUNDARY_QUANTILES,
    CLIMATOLOGY_OFFSETS,
    GRID_SHAPE,
    ISSUE_DATE,
    TARGET_WINDOWS,
    aggregate_era5_six_hour_week,
    climatology_boundaries,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proxy_sample_starts(valid_start: pd.Timestamp, last_year: int) -> tuple[pd.Timestamp, ...]:
    starts = tuple(
        valid_start.replace(year=year) + pd.Timedelta(days=offset)
        for year in range(last_year - 19, last_year + 1)
        for offset in CLIMATOLOGY_OFFSETS
    )
    if len(starts) != 100 or len(set(starts)) != 100:
        raise RuntimeError("proxy climatology must contain 100 unique samples")
    return starts


def nonnegative_precipitation(values: np.ndarray) -> np.ndarray:
    """Remove tiny negative reanalysis accumulation artifacts before quantiles."""

    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("ERA5 weekly precipitation contains non-finite values")
    return np.maximum(array, np.float32(0.0))


def run(args: argparse.Namespace) -> None:
    issue = pd.Timestamp(args.issue_date).normalize()
    if issue.strftime("%Y%m%d") != ISSUE_DATE:
        raise ValueError(f"this fallback is frozen to issue {ISSUE_DATE}")
    if args.last_year >= issue.year:
        raise ValueError("proxy climatology last year must precede the issue year")
    era5 = open_era5(args.era5_store, args.era5_variable)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    latitude = np.linspace(90.0, -90.0, GRID_SHAPE[0], dtype=np.float32)
    longitude = np.arange(GRID_SHAPE[1], dtype=np.float32) * np.float32(1.5)
    for period, (start_day, _end_day) in enumerate(TARGET_WINDOWS, start=1):
        valid_start = issue + pd.Timedelta(days=start_day - 1)
        starts = proxy_sample_starts(valid_start, args.last_year)
        fields = []
        for sample_index, start in enumerate(starts, start=1):
            fields.append(
                nonnegative_precipitation(
                    aggregate_era5_six_hour_week(era5, start).values
                )
            )
            if sample_index % 10 == 0:
                print(
                    f"period {period}: loaded climatology sample "
                    f"{sample_index}/100",
                    flush=True,
                )
        weekly = np.stack(fields)
        bounds = climatology_boundaries(weekly)
        data = xr.DataArray(
            bounds,
            dims=("quintile", "latitude", "longitude"),
            coords={
                "quintile": np.asarray(BOUNDARY_QUANTILES, dtype=np.float32),
                "latitude": latitude,
                "longitude": longitude,
            },
            attrs={
                "units": "mm week-1",
                "status": "public_era5_lagged_proxy_not_official_ai_wq_download",
                "valid_week_start": valid_start.strftime("%Y-%m-%d"),
                "climatology_years": f"{args.last_year - 19}-{args.last_year}",
                "sample_count": 100,
            },
        )
        path = args.output_dir / f"pr_{ISSUE_DATE}_p{period}_climatology_proxy.nc"
        data.to_netcdf(path)
        records.append(
            {
                "period": period,
                "valid_week_start": valid_start.strftime("%Y-%m-%d"),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "status": "complete_public_era5_lagged_proxy",
        "submission_warning": (
            "Replace with AI-WQ retrieve_20yr_quantile_clim outputs when the "
            "registered password is available."
        ),
        "issue_date": ISSUE_DATE,
        "era5_store": str(args.era5_store),
        "climatology_years": [args.last_year - 19, args.last_year],
        "files": records,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "climatology_proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--era5-store", type=Path, required=True)
    result.add_argument("--era5-variable", default="total_precipitation_6hr")
    result.add_argument("--issue-date", default=ISSUE_DATE)
    result.add_argument("--last-year", type=int, default=2023)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
