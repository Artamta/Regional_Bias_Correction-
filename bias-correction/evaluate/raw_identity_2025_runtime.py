#!/usr/bin/env python3
"""Store-capable runtime for one authorized raw-identity 2025 attempt.

This module must only be imported dynamically after the dispatcher has durably
created the one-time access ledger.  It opens only the three explicit locators
bound into the approval receipt; it performs no catalog search or discovery.
"""

from __future__ import annotations

import io
import os
import platform
import stat
import sys
import threading
import traceback
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

import raw_identity_2025_assets as assets
import raw_identity_2025_contract as contract


OFFSETS_DAYS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)


@dataclass(frozen=True)
class IndependentTestData:
    initializations: np.ndarray
    valid_dates: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    support: np.ndarray
    area_weight_km2: np.ndarray
    training_climatology_daily: np.ndarray
    raw_fuxi: np.ndarray
    fuxi_spread: np.ndarray
    fuxi_t2m_kelvin: np.ndarray
    fuxi_member_count: np.ndarray
    fuxi_t2m_member_count: np.ndarray
    selected_daily_imd: np.ndarray
    selected_daily_coverage: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    weekly_coverage: np.ndarray
    source_consumed_key_sha256: Mapping[str, Mapping[str, str]]
    source_store_identity: Mapping[str, Mapping[str, int]]
    evaluated_source_array_sha256: Mapping[str, str]


def _load_frozen_models(path: Path) -> ModuleType:
    content = contract.capture_authenticated_bytes(
        path, contract.EXPECTED_MODEL_SOURCE_SHA256, "sealed model source"
    )
    name = f"_raw_identity_2025_runtime_models_{contract.sha256_bytes(content)[:16]}"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


class DescriptorAnchoredZarrStore(MutableMapping[str, bytes]):
    """Read-only Zarr mapping whose every lookup is rooted at one open fd.

    All path components are opened with ``O_NOFOLLOW``. Every consumed key is
    hashed, repeat reads must be identical, and the consumed keys are reread
    through the same root descriptor before their provenance is released.
    """

    def __init__(self, path: Path):
        self.path = contract.lexical_absolute(path)
        self._root_descriptor = contract._open_directory_no_symlinks(self.path)
        metadata = os.fstat(self._root_descriptor)
        self.root_identity = {
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
        }
        self._consumed: dict[str, str] = {}
        self._lock = threading.Lock()
        self._closed = False
        try:
            self.assert_no_symlinks()
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> int:
        if self._closed:
            raise RuntimeError("descriptor-anchored Zarr store is closed")
        return self._root_descriptor

    @staticmethod
    def _parts(key: str) -> tuple[str, ...]:
        if not isinstance(key, str) or not key or key.startswith("/"):
            raise KeyError(key)
        pure = PurePosixPath(key)
        if str(pure) != key or any(part in ("", ".", "..") for part in pure.parts):
            raise KeyError(key)
        return pure.parts

    def _read_key(self, key: str) -> bytes:
        parts = self._parts(key)
        descriptor = os.dup(self._require_open())
        file_descriptor = -1
        try:
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | contract._O_DIRECTORY | contract._O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            file_descriptor = os.open(
                parts[-1], os.O_RDONLY | contract._O_NOFOLLOW, dir_fd=descriptor
            )
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise contract.SealContractError(
                    f"non-regular Zarr key refused: {self.path}/{key}"
                )
            blocks: list[bytes] = []
            while True:
                block = os.read(file_descriptor, 8 * 1024 * 1024)
                if not block:
                    break
                blocks.append(block)
            after = os.fstat(file_descriptor)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise contract.SealContractError(
                    f"Zarr key changed while being consumed: {self.path}/{key}"
                )
            return b"".join(blocks)
        except FileNotFoundError as exc:
            raise KeyError(key) from exc
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            os.close(descriptor)

    def __getitem__(self, key: str) -> bytes:
        content = self._read_key(key)
        digest = contract.sha256_bytes(content)
        with self._lock:
            previous = self._consumed.get(key)
            if previous is not None and previous != digest:
                raise contract.SealContractError(
                    f"Zarr key changed between reads: {self.path}/{key}"
                )
            self._consumed[key] = digest
        return content

    def _walk(self, descriptor: int, prefix: str = "") -> Iterator[str]:
        for name in sorted(os.listdir(descriptor)):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            key = f"{prefix}/{name}" if prefix else name
            if stat.S_ISLNK(metadata.st_mode):
                raise contract.SealContractError(
                    f"symlink in source Zarr store: {self.path}/{key}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | contract._O_DIRECTORY | contract._O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    yield from self._walk(child, key)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                yield key
            else:
                raise contract.SealContractError(
                    f"non-regular Zarr entry refused: {self.path}/{key}"
                )

    def assert_no_symlinks(self) -> None:
        tuple(self._walk(self._require_open()))

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._walk(self._require_open())))

    def __len__(self) -> int:
        return sum(1 for _ in self._walk(self._require_open()))

    def __setitem__(self, key: str, value: bytes) -> None:
        raise TypeError("descriptor-anchored Zarr store is read-only")

    def __delitem__(self, key: str) -> None:
        raise TypeError("descriptor-anchored Zarr store is read-only")

    def consumed_sha256(self) -> Mapping[str, str]:
        self.assert_no_symlinks()
        with self._lock:
            expected = dict(self._consumed)
        for key, digest in expected.items():
            if contract.sha256_bytes(self._read_key(key)) != digest:
                raise contract.SealContractError(
                    f"consumed Zarr key changed before seal: {self.path}/{key}"
                )
        return dict(sorted(expected.items()))

    def close(self) -> None:
        if not self._closed:
            os.close(self._root_descriptor)
            self._closed = True

    def __enter__(self) -> "DescriptorAnchoredZarrStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextmanager
def _open_bound_zarr(
    path: Path,
    identities: dict[str, Mapping[str, int]],
    consumed: dict[str, Mapping[str, str]],
) -> Iterator[xr.Dataset]:
    with (
        DescriptorAnchoredZarrStore(path) as source_store,
        xr.open_zarr(source_store, consolidated=True) as dataset,
    ):
        identities[str(path)] = dict(source_store.root_identity)
        yield dataset
        consumed[str(path)] = source_store.consumed_sha256()


def _calendar_positions(dates: np.ndarray) -> np.ndarray:
    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    array = np.asarray(dates, dtype="datetime64[D]")
    positions = np.asarray(
        [lookup[pd.Timestamp(value).strftime("%m-%d")] for value in array.reshape(-1)],
        dtype=np.int16,
    )
    return positions.reshape(array.shape)


def _derive_valid_dates(initializations: np.ndarray) -> np.ndarray:
    initializations = np.asarray(initializations, dtype="datetime64[D]")
    offsets = np.arange(42, dtype="timedelta64[D]").reshape(1, 6, 7)
    return initializations[:, None, None] + offsets


def _load_bundle(frozen: contract.FrozenSelection) -> Mapping[str, np.ndarray]:
    expected = str(frozen.payload["support_climatology_bundle"]["sha256"])
    content = contract.capture_authenticated_bytes(
        frozen.bundle_path, expected, "sealed support/climatology bundle"
    )
    with np.load(io.BytesIO(content), allow_pickle=False) as values:
        return {name: np.asarray(values[name]).copy() for name in values.files}


def _collapse_coverage(
    dataset: xr.Dataset,
    spatial_shape: tuple[int, int],
    *,
    selected_times: np.ndarray | None = None,
) -> np.ndarray:
    variable = dataset.observation_fraction
    if variable.ndim == 3 and selected_times is not None:
        variable = variable.sel(time=selected_times)
    values = variable.load().values.astype(np.float32)
    if values.ndim == 2:
        if variable.dims != ("latitude", "longitude") or values.shape != spatial_shape:
            raise contract.SealContractError("IMD coverage grid shape changed")
    elif values.ndim == 3:
        if (
            variable.dims != ("time", "latitude", "longitude")
            or values.shape[1:] != spatial_shape
        ):
            raise contract.SealContractError("IMD coverage grid shape changed")
    else:
        raise contract.SealContractError(
            "unexpected IMD observation_fraction dimensions"
        )
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise contract.SealContractError("IMD coverage must be finite and in [0,1]")
    return values


def _load_selected_imd_arrays(
    dataset: xr.Dataset,
    requested_dates: np.ndarray,
    spatial_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the annual calendar but materialize only evaluated dates."""

    requested = np.asarray(requested_dates, dtype="datetime64[D]")
    if (
        requested.ndim != 1
        or requested.size == 0
        or np.unique(requested).size != requested.size
        or np.any(np.diff(requested) <= np.timedelta64(0, "D"))
    ):
        raise contract.SealContractError(
            "requested IMD dates must be sorted and unique"
        )
    raw_date_index = pd.DatetimeIndex(dataset.time.values)
    if (
        np.any(raw_date_index.hour != 0)
        or np.any(raw_date_index.minute != 0)
        or np.any(raw_date_index.second != 0)
        or np.any(raw_date_index.microsecond != 0)
        or np.any(raw_date_index.nanosecond != 0)
    ):
        raise contract.SealContractError("2025 IMD dates are not daily 00Z")
    annual_dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
    annual_index = pd.DatetimeIndex(annual_dates)
    if (
        set(annual_index.year) != {contract.TEST_YEAR}
        or annual_index.has_duplicates
        or not annual_index.is_monotonic_increasing
        or dataset.observation.shape != (annual_dates.size, *spatial_shape)
    ):
        raise contract.SealContractError("2025 IMD calendar/array contract changed")
    positions = np.searchsorted(annual_dates, requested)
    if np.any(positions >= annual_dates.size) or not np.array_equal(
        annual_dates[positions], requested
    ):
        raise contract.SealContractError(
            "one or more 2025 IMD verification days is missing"
        )
    selected_times = requested.astype("datetime64[ns]")
    values = (
        dataset.observation.sel(time=selected_times).load().values.astype(np.float32)
    )
    coverage = _collapse_coverage(dataset, spatial_shape, selected_times=selected_times)
    if values.shape != (requested.size, *spatial_shape) or (
        coverage.ndim == 3 and coverage.shape != values.shape
    ):
        raise contract.SealContractError("selected IMD array contract changed")
    return requested, values, coverage


def _aggregate_weekly_truth(
    daily: np.ndarray, daily_coverage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    daily = np.asarray(daily, dtype=np.float32)
    daily_coverage = np.asarray(daily_coverage, dtype=np.float32)
    if daily.shape != daily_coverage.shape or daily.ndim != 5 or daily.shape[2] != 7:
        raise contract.SealContractError("daily truth/coverage weekly shape changed")
    if not np.isfinite(daily_coverage).all() or np.any(
        (daily_coverage < 0.0) | (daily_coverage > 1.0)
    ):
        raise contract.SealContractError("daily coverage must be finite and in [0,1]")
    truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    weekly_coverage = np.min(daily_coverage, axis=2).astype(np.float32)
    return truth, weekly_coverage


def load_independent_test_data(
    frozen: contract.FrozenSelection,
) -> IndependentTestData:
    """Open only the three frozen locators after ledger consumption."""

    bundle = _load_bundle(frozen)
    latitude = np.asarray(bundle["latitude"], dtype=np.float64)
    longitude = np.asarray(bundle["longitude"], dtype=np.float64)
    support = np.asarray(bundle["support"], dtype=bool)
    area = np.asarray(bundle["india_area_weight_km2"], dtype=np.float64)
    climatology_daily = np.asarray(
        bundle["training_climatology_daily"], dtype=np.float32
    )
    if (
        contract.array_sha256(climatology_daily, "<f4")
        != contract.EXPECTED_TRAINING_CLIMATOLOGY_SHA256
    ):
        raise contract.SealContractError(
            "daily training climatology changed at runtime consumption"
        )
    data_locators = contract.validate_data_locators(frozen.data_locators)
    tp_path = contract.lexical_absolute(data_locators["forecast_tp_store"])
    t2m_path = contract.lexical_absolute(data_locators["forecast_t2m_store"])
    imd_path = contract.lexical_absolute(data_locators["imd_daily_store"])
    source_consumed: dict[str, Mapping[str, str]] = {}
    source_identities: dict[str, Mapping[str, int]] = {}
    expected_inits = contract.expected_initialization_dates()
    variable_dims = ("init", "lead_week", "latitude", "longitude")

    with _open_bound_zarr(tp_path, source_identities, source_consumed) as dataset:
        for variable in ("ensemble_mean_weekly", "ensemble_std_weekly"):
            if dataset[variable].dims != variable_dims:
                raise contract.SealContractError(
                    f"2025 TP {variable} dimensions changed"
                )
            if (
                dataset[variable].attrs.get("units") != "mm day-1"
                or dataset[variable].attrs.get("temporal_statistic")
                != "mean_of_complete_7_day_block"
            ):
                raise contract.SealContractError(
                    f"2025 TP {variable} units/statistic changed"
                )
        if dataset.ensemble_member_count_weekly.dims != variable_dims:
            raise contract.SealContractError("2025 TP member-count dimensions changed")
        all_initializations = pd.DatetimeIndex(dataset.init.values)
        if (
            set(all_initializations.year) != {contract.TEST_YEAR}
            or all_initializations.has_duplicates
            or not all_initializations.is_monotonic_increasing
            or np.any(all_initializations.hour != 0)
            or np.any(all_initializations.minute != 0)
            or np.any(all_initializations.second != 0)
            or np.any(all_initializations.microsecond != 0)
            or np.any(all_initializations.nanosecond != 0)
        ):
            raise contract.SealContractError(
                "2025 TP initialization coordinate changed"
            )
        selected_mask = (all_initializations.year == contract.TEST_YEAR) & (
            all_initializations.month.isin((6, 7, 8, 9))
        )
        initializations = all_initializations[selected_mask]
        initialization_days = initializations.values.astype("datetime64[D]")
        if not np.array_equal(initialization_days, expected_inits):
            raise contract.SealContractError(
                "2025 starts are not the exact frozen Monday/Thursday JJAS schedule"
            )
        if not np.array_equal(
            dataset.lead_week.values, np.arange(1, 7, dtype=np.int16)
        ):
            raise contract.SealContractError("2025 TP lead-week coordinate changed")
        raw = (
            dataset.ensemble_mean_weekly.sel(init=initializations)
            .load()
            .values.astype(np.float32)
        )
        spread = (
            dataset.ensemble_std_weekly.sel(init=initializations)
            .load()
            .values.astype(np.float32)
        )
        member_count = (
            dataset.ensemble_member_count_weekly.sel(init=initializations).load().values
        )
        tp_latitude = dataset.latitude.values.astype(np.float64)
        tp_longitude = dataset.longitude.values.astype(np.float64)
        if (
            dataset.latitude.dims != ("latitude",)
            or dataset.longitude.dims != ("longitude",)
            or dataset.sizes.get("member") != 50
            or not np.array_equal(dataset.member.values, np.arange(50, dtype=np.int16))
        ):
            raise contract.SealContractError("2025 TP member coordinate changed")

    with _open_bound_zarr(t2m_path, source_identities, source_consumed) as dataset:
        for variable in ("ensemble_mean_weekly", "ensemble_std_weekly"):
            if dataset[variable].dims != variable_dims:
                raise contract.SealContractError(
                    f"2025 T2M {variable} dimensions changed"
                )
            if (
                dataset[variable].attrs.get("units") != "degC"
                or dataset[variable].attrs.get("temporal_statistic")
                != "mean_of_complete_7_day_block"
            ):
                raise contract.SealContractError(
                    f"2025 T2M {variable} units/statistic changed"
                )
        if dataset.ensemble_member_count_weekly.dims != variable_dims:
            raise contract.SealContractError("2025 T2M member-count dimensions changed")
        t2m_inits = pd.DatetimeIndex(dataset.init.values)
        if (
            set(t2m_inits.year) != {contract.TEST_YEAR}
            or t2m_inits.has_duplicates
            or not t2m_inits.is_monotonic_increasing
            or np.any(t2m_inits.hour != 0)
            or np.any(t2m_inits.minute != 0)
            or np.any(t2m_inits.second != 0)
            or np.any(t2m_inits.microsecond != 0)
            or np.any(t2m_inits.nanosecond != 0)
            or not np.array_equal(t2m_inits.values, all_initializations.values)
            or not np.array_equal(
                t2m_inits[
                    (t2m_inits.year == contract.TEST_YEAR)
                    & t2m_inits.month.isin((6, 7, 8, 9))
                ].values.astype("datetime64[D]"),
                expected_inits,
            )
            or not np.array_equal(
                dataset.lead_week.values, np.arange(1, 7, dtype=np.int16)
            )
        ):
            raise contract.SealContractError("2025 T2M schedule/lead contract changed")
        t2m = (
            dataset.ensemble_mean_weekly.sel(init=initializations).load().values
            + np.float32(273.15)
        ).astype(np.float32)
        t2m_member_count = (
            dataset.ensemble_member_count_weekly.sel(init=initializations).load().values
        )
        t2m_latitude = dataset.latitude.values.astype(np.float64)
        t2m_longitude = dataset.longitude.values.astype(np.float64)
        if (
            dataset.latitude.dims != ("latitude",)
            or dataset.longitude.dims != ("longitude",)
            or dataset.sizes.get("member") != 50
            or not np.array_equal(dataset.member.values, np.arange(50, dtype=np.int16))
        ):
            raise contract.SealContractError("2025 T2M member coordinate changed")

    expected_shape = (
        contract.EXPECTED_CASES,
        contract.EXPECTED_LEADS,
        *contract.EXPECTED_GRID,
    )
    if (
        raw.shape != expected_shape
        or spread.shape != raw.shape
        or t2m.shape != raw.shape
        or member_count.shape != raw.shape
        or t2m_member_count.shape != raw.shape
    ):
        raise contract.SealContractError(
            f"unexpected 2025 predictor shapes: {raw.shape}"
        )
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(spread).all()
        or not np.isfinite(t2m).all()
        or np.any(raw < 0.0)
        or np.any(spread < 0.0)
        or not np.all(member_count == 50)
        or not np.all(t2m_member_count == 50)
    ):
        raise contract.SealContractError("2025 predictor fields are invalid")
    if not all(
        np.array_equal(value, latitude) for value in (tp_latitude, t2m_latitude)
    ):
        raise contract.SealContractError("2025 predictor latitude changed")
    if not all(
        np.array_equal(value, longitude) for value in (tp_longitude, t2m_longitude)
    ):
        raise contract.SealContractError("2025 predictor longitude changed")

    valid_dates = _derive_valid_dates(expected_inits)
    if set(pd.DatetimeIndex(valid_dates.reshape(-1)).year) != {contract.TEST_YEAR}:
        raise contract.SealContractError("verification windows leave test year 2025")
    requested = valid_dates.reshape(-1)
    requested_unique = np.unique(requested)
    with _open_bound_zarr(imd_path, source_identities, source_consumed) as dataset:
        if (
            dataset.attrs.get("source") != "imd"
            or dataset.attrs.get("units") != "mm day-1"
            or dataset.observation.dims != ("time", "latitude", "longitude")
        ):
            raise contract.SealContractError(
                "2025 IMD source/dimension contract changed"
            )
        if not np.array_equal(dataset.latitude.values, latitude) or not np.array_equal(
            dataset.longitude.values, longitude
        ):
            raise contract.SealContractError("2025 IMD grid changed")
        dates, values, coverage = _load_selected_imd_arrays(
            dataset, requested_unique, contract.EXPECTED_GRID
        )
    positions = np.searchsorted(dates, requested)
    if np.any(positions >= len(dates)) or not np.array_equal(
        dates[positions], requested
    ):
        raise contract.SealContractError(
            "one or more 2025 IMD verification days is missing"
        )
    daily = values[positions].reshape(
        contract.EXPECTED_CASES, 6, 7, *contract.EXPECTED_GRID
    )
    if coverage.ndim == 2:
        daily_coverage = np.broadcast_to(
            coverage, (contract.EXPECTED_CASES, 6, 7, *contract.EXPECTED_GRID)
        ).copy()
    else:
        daily_coverage = coverage[positions].reshape(
            contract.EXPECTED_CASES, 6, 7, *contract.EXPECTED_GRID
        )
    truth, weekly_coverage = _aggregate_weekly_truth(daily, daily_coverage)
    climatology = np.mean(
        climatology_daily[_calendar_positions(valid_dates)], axis=2, dtype=np.float64
    ).astype(np.float32)
    usable = support[None, None] & (weekly_coverage > 0.0)
    valid_counts = np.sum(usable & np.isfinite(truth), axis=(-2, -1))
    if np.any(valid_counts < 3) or not np.isfinite(climatology[..., support]).all():
        raise contract.SealContractError("2025 IMD truth/climatology is unusable")

    evaluated_hashes = {
        "initializations": contract.array_sha256(
            expected_inits.astype("datetime64[D]").view("<i8"), "<i8"
        ),
        "valid_dates": contract.array_sha256(
            valid_dates.astype("datetime64[D]").view("<i8"), "<i8"
        ),
        "latitude": contract.array_sha256(latitude, "<f8"),
        "longitude": contract.array_sha256(longitude, "<f8"),
        "lead_week": contract.array_sha256(np.arange(1, 7), "<i2"),
        "raw_fuxi": contract.array_sha256(raw, "<f4"),
        "fuxi_spread": contract.array_sha256(spread, "<f4"),
        "fuxi_t2m_kelvin": contract.array_sha256(t2m, "<f4"),
        "fuxi_member_count": contract.array_sha256(member_count, "<i2"),
        "fuxi_t2m_member_count": contract.array_sha256(t2m_member_count, "<i2"),
        "selected_daily_imd": contract.array_sha256(daily, "<f4"),
        "selected_daily_coverage": contract.array_sha256(daily_coverage, "<f4"),
        "truth": contract.array_sha256(truth, "<f4"),
        "weekly_coverage": contract.array_sha256(weekly_coverage, "<f4"),
        "fixed_training_climatology": contract.array_sha256(climatology, "<f4"),
    }
    return IndependentTestData(
        initializations=expected_inits,
        valid_dates=valid_dates,
        latitude=latitude,
        longitude=longitude,
        support=support,
        area_weight_km2=area,
        training_climatology_daily=climatology_daily,
        raw_fuxi=raw,
        fuxi_spread=spread,
        fuxi_t2m_kelvin=t2m,
        fuxi_member_count=np.asarray(member_count, dtype=np.int16),
        fuxi_t2m_member_count=np.asarray(t2m_member_count, dtype=np.int16),
        selected_daily_imd=daily,
        selected_daily_coverage=daily_coverage,
        truth=truth,
        climatology=climatology,
        weekly_coverage=weekly_coverage,
        source_consumed_key_sha256=source_consumed,
        source_store_identity=source_identities,
        evaluated_source_array_sha256=evaluated_hashes,
    )


def _normalize_dynamic(
    values: np.ndarray,
    statistics: Mapping[str, Any],
    support: np.ndarray,
    *,
    preserve_full_domain: bool = False,
) -> np.ndarray:
    mean = np.asarray(statistics["mean_by_lead"], dtype=np.float32)
    std = np.asarray(statistics["std_by_lead"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0.0):
        raise contract.SealContractError("saved normalization statistics changed")
    normalized = (values - mean[None, :, None, None]) / std[None, :, None, None]
    valid = (
        np.isfinite(normalized)
        if preserve_full_domain
        else support[None, None] & np.isfinite(normalized)
    )
    return np.where(valid, normalized, 0.0).astype(np.float32)


def build_frozen_features(
    data: IndependentTestData, normalization: Mapping[str, Any]
) -> np.ndarray:
    context = normalization.get("spatial_context", {})
    if context.get("enabled") is not True or context.get("full_domain_channels") != [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "fuxi_t2m_weekly",
    ]:
        raise contract.SealContractError("saved full-domain context changed")
    channels: list[np.ndarray] = [
        _normalize_dynamic(
            np.log1p(data.raw_fuxi).astype(np.float32),
            normalization["log_fuxi_mean"],
            data.support,
            preserve_full_domain=True,
        ),
        _normalize_dynamic(
            np.log1p(data.fuxi_spread).astype(np.float32),
            normalization["log_fuxi_spread"],
            data.support,
            preserve_full_domain=True,
        ),
        _normalize_dynamic(
            np.log1p(data.climatology).astype(np.float32),
            normalization["log_imd_climatology"],
            data.support,
        ),
    ]
    cases, leads, height, width = data.raw_fuxi.shape
    latitude = data.latitude.astype(np.float32)
    longitude = data.longitude.astype(np.float32)
    lat_scaled = (
        2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    )
    lon_scaled = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    midpoint = pd.DatetimeIndex(data.valid_dates[:, :, 3].reshape(-1))
    angle = 2.0 * np.pi * (midpoint.dayofyear.to_numpy() - 1) / 365.2425
    angle = angle.reshape(cases, leads)
    channels.extend(
        [
            np.broadcast_to(
                lat_scaled[None, None, :, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                lon_scaled[None, None, None, :], (cases, leads, height, width)
            ),
            np.broadcast_to(
                np.sin(angle)[:, :, None, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                np.cos(angle)[:, :, None, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, :, None, None],
                (cases, leads, height, width),
            ),
            np.broadcast_to(
                data.support[None, None], (cases, leads, height, width)
            ).astype(np.float32),
            _normalize_dynamic(
                (np.log1p(data.raw_fuxi) - np.log1p(data.climatology)).astype(
                    np.float32
                ),
                normalization["explicit_log_fuxi_anomaly"],
                data.support,
            ),
            _normalize_dynamic(
                data.fuxi_t2m_kelvin,
                normalization["fuxi_t2m_weekly"],
                data.support,
                preserve_full_domain=True,
            ),
        ]
    )
    candidates = []
    for offset in OFFSETS_DAYS:
        shifted = data.valid_dates + np.timedelta64(offset, "D")
        candidates.append(
            np.mean(
                data.training_climatology_daily[_calendar_positions(shifted)],
                axis=2,
                dtype=np.float64,
            ).astype(np.float32)
        )
    bank = np.stack(candidates, axis=2)
    climo_mean = np.asarray(
        normalization["log_imd_climatology"]["mean_by_lead"], dtype=np.float32
    )
    climo_std = np.asarray(
        normalization["log_imd_climatology"]["std_by_lead"], dtype=np.float32
    )
    normalized_bank = (
        np.log1p(bank).astype(np.float32) - climo_mean[None, :, None, None, None]
    ) / climo_std[None, :, None, None, None]
    anomaly_mean = np.asarray(
        normalization["explicit_log_fuxi_anomaly"]["mean_by_lead"],
        dtype=np.float32,
    )
    anomaly_std = np.asarray(
        normalization["explicit_log_fuxi_anomaly"]["std_by_lead"],
        dtype=np.float32,
    )
    bank_anomaly = np.log1p(data.raw_fuxi)[:, :, None] - np.log1p(bank)
    normalized_bank_anomaly = (
        bank_anomaly - anomaly_mean[None, :, None, None, None]
    ) / anomaly_std[None, :, None, None, None]
    valid = data.support[None, None, None]
    normalized_bank = np.where(valid, normalized_bank, 0.0).astype(np.float32)
    normalized_bank_anomaly = np.where(valid, normalized_bank_anomaly, 0.0).astype(
        np.float32
    )
    features = np.concatenate(
        (
            np.stack(channels, axis=2).astype(np.float32),
            normalized_bank,
            normalized_bank_anomaly,
        ),
        axis=2,
    ).astype(np.float32)
    expected_names = [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_calendar_climatology",
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "support",
        "explicit_log_fuxi_minus_imd_climatology",
        "fuxi_t2m_weekly",
        *[f"imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
        *[f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
    ]
    if list(normalization.get("input_channels", [])) != expected_names:
        raise contract.SealContractError("saved 29-channel ordering changed")
    if features.shape != (cases, 6, 29, 27, 27):
        raise contract.SealContractError(f"unexpected feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise contract.SealContractError("frozen features contain non-finite values")
    for index in (2, 9, *range(11, 29)):
        if np.any(features[:, :, index, ~data.support] != 0.0):
            raise contract.SealContractError(
                "IMD-derived feature leaked outside support"
            )
    return features


def infer_raw_identity(
    frozen: contract.FrozenSelection,
    features: np.ndarray,
    raw_fuxi: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("final independent evaluation requires CUDA")
    models = _load_frozen_models(frozen.model_source_path)
    members: list[np.ndarray] = []
    member_metadata: list[Mapping[str, Any]] = []
    for seed, checkpoint_path, expected_hash in frozen.checkpoints:
        checkpoint_content = contract.read_bytes_no_follow(checkpoint_path)
        if contract.sha256_bytes(checkpoint_content) != expected_hash:
            raise contract.SealContractError(f"checkpoint seed {seed} changed")
        model = models.FixedClimatologyAllLeadUNet(
            input_channels=29, base_channels=16, dropout=0.30
        )
        checkpoint = torch.load(
            io.BytesIO(checkpoint_content), map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, Mapping):
            raise contract.SealContractError(
                f"checkpoint root is not a mapping: {checkpoint_path}"
            )
        if int(checkpoint.get("seed", seed)) != seed:
            raise contract.SealContractError(
                f"checkpoint seed metadata changed: {seed}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to("cuda").eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(features), 32):
                batch = torch.from_numpy(features[start : start + 32]).to("cuda")
                with torch.cuda.amp.autocast(enabled=True):
                    outputs.append(model(batch).float().cpu().numpy())
        member = np.concatenate(outputs).astype(np.float32)
        if member.shape != raw_fuxi.shape or not np.isfinite(member).all():
            raise contract.SealContractError(f"invalid residual from seed {seed}")
        members.append(member)
        member_metadata.append(
            {
                "seed": seed,
                "checkpoint_sha256": expected_hash,
                "standardized_residual_sha256": contract.array_sha256(member, "<f4"),
            }
        )
        del model
    torch.cuda.synchronize()
    residual = np.mean(np.stack(members), axis=0, dtype=np.float64).astype(np.float32)
    anchor_content = contract.capture_authenticated_bytes(
        frozen.raw_anchor_path,
        contract.EXPECTED_RAW_ANCHOR_SHA256,
        "sealed raw training anchor",
    )
    with np.load(io.BytesIO(anchor_content), allow_pickle=False) as anchor:
        anchor_kind = str(np.asarray(anchor["anchor_kind"]).item())
        target_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
        fitted_years = tuple(int(value) for value in anchor["fitted_target_years"])
    if anchor_kind != "raw_fuxi" or fitted_years != contract.TRAIN_YEARS:
        raise contract.SealContractError("raw training anchor changed")
    if (
        target_scale.shape != (6,)
        or not np.isfinite(target_scale).all()
        or np.any(target_scale <= 0.0)
    ):
        raise contract.SealContractError("raw target scale changed")
    valid = np.broadcast_to(support[None, None], raw_fuxi.shape)
    safe_raw = np.where(valid, raw_fuxi, 0.0).astype(np.float64)
    safe_residual = np.where(valid, residual, 0.0).astype(np.float64)
    transformed = np.log1p(safe_raw) + safe_residual * target_scale[None, :, None, None]
    prediction = np.expm1(np.clip(transformed, 0.0, 20.0))
    prediction = np.where(safe_residual == 0.0, safe_raw, prediction)
    prediction[~valid] = np.nan
    prediction = prediction.astype(np.float32)
    if not np.isfinite(prediction[valid]).all() or np.any(prediction[valid] < 0.0):
        raise contract.SealContractError("raw-identity reconstruction is invalid")
    return (
        prediction,
        residual,
        {
            "members": member_metadata,
            "ensemble": "arithmetic mean of standardized residuals",
            "ensemble_residual_sha256": contract.array_sha256(residual, "<f4"),
            "raw_anchor_sha256": contract.EXPECTED_RAW_ANCHOR_SHA256,
            "target_scale": target_scale.tolist(),
            "amp": True,
            "batch_size": 32,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )


def _update_attempt_status(
    frozen: contract.FrozenSelection,
    *,
    status: str,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    path = Path(frozen.canonical_paths["failure_record"])
    previous = contract.read_json(path)
    if (
        previous.get("schema_version") != contract.ATTEMPT_SCHEMA_VERSION
        or previous.get("attempt") != 1
        or previous.get("selection_manifest_sha256") != frozen.sha256
    ):
        raise contract.SealContractError(
            "attempt status is not bound to this selection"
        )
    payload: dict[str, Any] = {
        "schema_version": contract.ATTEMPT_SCHEMA_VERSION,
        "attempt": 1,
        "status": status,
        "stage": stage,
        "updated_utc": contract.utc_now(),
        "selection_manifest_sha256": frozen.sha256,
        "canonical_execution_paths_sha256": frozen.canonical_paths_sha256,
        "access_ledger_committed": True,
        "attempt_consumed": True,
        "retry_permitted": False,
    }
    if details:
        payload["details"] = dict(details)
    contract.atomic_replace_json(path, payload)


def _authenticate_published_output(
    final_output: Path, completed_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    saved = contract.read_json(final_output / "manifest.json")
    if saved != completed_manifest:
        raise contract.SealContractError(
            "published manifest differs from the completed staging manifest"
        )
    actual_artifacts = assets.artifact_hashes(final_output)
    if actual_artifacts != completed_manifest.get("artifacts"):
        raise contract.SealContractError("published artifact hashes do not verify")
    return {
        "final_output": str(final_output),
        "manifest_sha256": contract.sha256_file(final_output / "manifest.json"),
        "artifact_count": len(actual_artifacts),
    }


def _entry_exists_safely(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return contract.entry_exists_no_follow(path)
    except FileNotFoundError:
        return False


def run_evaluation(
    *,
    frozen: contract.FrozenSelection,
    preflight_receipt_sha256: str,
    approval_receipt_sha256: str,
) -> Path:
    """Execute the single attempt after the dispatcher consumes its ledger."""

    data_locators = contract.validate_data_locators(frozen.data_locators)
    data_locator_sha256 = contract.canonical_json_sha256(data_locators)
    final_output = Path(frozen.canonical_paths["final_output"])
    access_ledger = Path(frozen.canonical_paths["access_ledger"])
    ledger, ledger_sha256 = contract.read_json_with_sha256(access_ledger)
    authenticated_code_hashes = {
        filename: contract.sha256_bytes(content)
        for filename, content in frozen.workflow_code_bytes.items()
    }
    if authenticated_code_hashes != frozen.payload.get("code_sha256"):
        raise contract.SealContractError(
            "authenticated workflow bytes differ from the frozen code map"
        )
    ledger_checks = {
        "status": ledger.get("status") == "independent_2025_access_attempt_consumed",
        "attempt": ledger.get("attempt") == 1,
        "selection_manifest_sha256": ledger.get("selection_manifest_sha256")
        == frozen.sha256,
        "preflight_receipt_sha256": ledger.get("preflight_receipt_sha256")
        == preflight_receipt_sha256,
        "approval_receipt_sha256": ledger.get("approval_receipt_sha256")
        == approval_receipt_sha256,
        "test_data_locators_sha256": ledger.get("test_data_locators_sha256")
        == data_locator_sha256,
        "test_data_locators": ledger.get("test_data_locators") == dict(data_locators),
        "canonical_execution_paths": ledger.get("canonical_execution_paths")
        == dict(frozen.canonical_paths),
        "canonical_execution_paths_sha256": ledger.get(
            "canonical_execution_paths_sha256"
        )
        == frozen.canonical_paths_sha256,
        "method_hierarchy": ledger.get("method_hierarchy")
        == list(contract.METHOD_HIERARCHY),
        "runtime_import_state_at_commit": ledger.get("runtime_import_state_at_commit")
        == "not_imported",
    }
    failed_ledger_fields = [
        field for field, valid in ledger_checks.items() if not valid
    ]
    if failed_ledger_fields:
        raise contract.SealContractError(
            "durable access ledger does not match runtime: "
            + ", ".join(failed_ledger_fields)
        )
    running: dict[str, Any] = {
        "status": "running",
        "scientific_status": "one-time untouched independent 2025 test",
        "created_utc": contract.utc_now(),
        "attempt": 1,
        "selection_manifest_sha256": frozen.sha256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "approval_receipt_sha256": approval_receipt_sha256,
        "access_ledger": str(access_ledger),
        "access_ledger_sha256": ledger_sha256,
        "method_hierarchy": list(contract.METHOD_HIERARCHY),
        "no_2025_fitting_tuning_selection_calibration_or_retry": True,
    }
    stage = "pre_staging_validated"
    staging: Path | None = None
    manifest_path: Path | None = None
    published = False
    completed: Mapping[str, Any] | None = None
    contract.assert_secure_directory(final_output.parent, "final output parent")
    if contract.entry_exists_no_follow(final_output):
        raise FileExistsError(f"fresh final output required: {final_output}")
    _update_attempt_status(frozen, status="running", stage=stage)
    try:
        staging = contract.create_secure_staging_directory(
            final_output.parent, f"{final_output.name}.attempt-1-partial"
        )
        manifest_path = staging / "manifest.json"
        stage = "staging_allocated_before_store_open"
        _update_attempt_status(
            frozen,
            status="running",
            stage=stage,
            details={"partial_directory": str(staging)},
        )
        contract.write_json_exclusive(manifest_path, running)
        data = load_independent_test_data(frozen)
        stage = "source_stores_loaded"
        _update_attempt_status(frozen, status="running", stage=stage)
        normalization_content = contract.capture_authenticated_bytes(
            frozen.normalization_path,
            contract.EXPECTED_NORMALIZATION_SHA256,
            "sealed normalization",
        )
        normalization = contract.parse_json_bytes(
            normalization_content, "sealed normalization"
        )
        features = build_frozen_features(data, normalization)
        raw_identity, residual, inference = infer_raw_identity(
            frozen, features, data.raw_fuxi, data.support
        )
        asset_contract = assets.write_assets(
            staging,
            initializations=data.initializations,
            latitude=data.latitude,
            longitude=data.longitude,
            truth=data.truth,
            climatology=data.climatology,
            weekly_coverage=data.weekly_coverage,
            area_weight_km2=data.area_weight_km2,
            support=data.support,
            raw_fuxi=data.raw_fuxi,
            fuxi_spread=data.fuxi_spread,
            fuxi_t2m_kelvin=data.fuxi_t2m_kelvin,
            fuxi_member_count=data.fuxi_member_count,
            fuxi_t2m_member_count=data.fuxi_t2m_member_count,
            valid_dates=data.valid_dates,
            selected_daily_imd=data.selected_daily_imd,
            selected_daily_coverage=data.selected_daily_coverage,
            raw_identity=raw_identity,
            ensemble_standardized_residual=residual,
        )
        stage = "artifacts_staged"
        _update_attempt_status(frozen, status="running", stage=stage)
        code_directory = contract.secure_mkdir(staging / "code")
        for filename in contract.CODE_FILENAMES:
            content = frozen.workflow_code_bytes[filename]
            contract.write_bytes_exclusive(code_directory / filename, content)
        contract.fsync_directory(code_directory)
        completed = {
            **running,
            "status": "complete",
            "completed_utc": contract.utc_now(),
            "test_year": contract.TEST_YEAR,
            "case_count": contract.EXPECTED_CASES,
            "lead_count": contract.EXPECTED_LEADS,
            "initialization_months": [6, 7, 8, 9],
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "primary_comparison": {
                "candidate": "raw_identity",
                "baseline": "raw_fuxi",
            },
            "selected_model": contract.SELECTED_MODEL,
            "selected_alpha": contract.SELECTED_ALPHA,
            "checkpoint_seeds": list(contract.EXPECTED_SEEDS),
            "inference": inference,
            "bootstrap": {
                "draws": contract.BOOTSTRAP_DRAWS,
                "block_length_initializations": contract.BOOTSTRAP_BLOCK_LENGTH,
                "seed": contract.BOOTSTRAP_SEED,
                "method": "paired circular moving blocks by initialization",
                "descriptive_not_null_recentered": True,
            },
            "asset_contract": asset_contract,
            "test_data_locators": dict(data_locators),
            "test_data_locators_sha256": data_locator_sha256,
            "source_zarr_consumed_key_sha256": data.source_consumed_key_sha256,
            "source_zarr_root_identity": data.source_store_identity,
            "evaluated_source_array_sha256": data.evaluated_source_array_sha256,
            "frozen_sha256": dict(frozen.payload["fixed_sha256"]),
            "support_climatology_bundle_sha256": frozen.payload[
                "support_climatology_bundle"
            ]["sha256"],
            "code_sha256": authenticated_code_hashes,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xarray": xr.__version__,
            },
            "atomic_output": {
                "requested_final_directory": str(final_output),
                "staging_directory_name": staging.name,
                "publication": "same-filesystem rename after complete manifest",
            },
        }
        completed["artifacts"] = assets.artifact_hashes(staging)
        contract.atomic_replace_json(manifest_path, completed)
        contract.fsync_directory(staging)
        stage = "complete_staging_fsynced_before_publication"
        _update_attempt_status(
            frozen,
            status="running",
            stage=stage,
            details={"partial_directory": str(staging)},
        )
        contract.rename_noreplace(staging, final_output)
        published = True
        stage = "published_before_parent_fsync"
        publication = _authenticate_published_output(final_output, completed)
        contract.fsync_directory(final_output.parent)
        stage = "published_after_parent_fsync"
        _update_attempt_status(
            frozen,
            status="complete",
            stage=stage,
            details=publication,
        )
        return final_output
    except BaseException as failure:
        failure_traceback = traceback.format_exc()
        if staging is None:
            allocated = getattr(failure, "staging_path", None)
            if allocated is not None:
                staging = Path(allocated)
                manifest_path = staging / "manifest.json"
        staging_exists = _entry_exists_safely(staging)
        final_exists = _entry_exists_safely(final_output)
        publication: Mapping[str, Any] | None = None
        publication_authentication_error: str | None = None
        if final_exists:
            published = True
            if stage == "complete_staging_fsynced_before_publication":
                stage = "published_before_parent_fsync"
            if completed is not None:
                try:
                    publication = _authenticate_published_output(
                        final_output, completed
                    )
                except BaseException:
                    publication_authentication_error = traceback.format_exc()
        partial_directory = str(staging) if staging_exists else None
        failed = {
            **running,
            "status": "failed",
            "failed_utc": contract.utc_now(),
            "failure": failure_traceback,
            "attempt_remains_consumed": True,
            "failure_stage": stage,
            "published_before_failure": published,
            "partial_directory": partial_directory,
            "final_output": str(final_output) if published else None,
            "publication_authentication": publication,
            "publication_authentication_error": publication_authentication_error,
        }
        if not published and staging_exists and _entry_exists_safely(manifest_path):
            contract.atomic_replace_json(manifest_path, failed)
            contract.fsync_directory(staging)
        _update_attempt_status(
            frozen,
            status="failed",
            stage=stage,
            details={
                "traceback": failure_traceback,
                "attempt_remains_consumed": True,
                "published_before_failure": published,
                "partial_directory": partial_directory,
                "final_output": str(final_output) if published else None,
                "publication_authentication": publication,
                "publication_authentication_error": publication_authentication_error,
            },
        )
        raise
