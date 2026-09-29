"""Focused tests for the raw weekly FuXi physical-feature cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fuxi_physical_feature_cache as physical  # noqa: E402


class TrackingArray:
    """NumPy-backed array that records every source selection."""

    def __init__(self, values: np.ndarray):
        self.values = values
        self.calls: list[tuple[object, ...]] = []

    def __getitem__(self, key: tuple[object, ...]) -> np.ndarray:
        self.calls.append(key)
        return self.values[key]


def _weekly_member_then_ensemble(values: np.ndarray) -> np.ndarray:
    height, width = values.shape[-2:]
    member_weekly = values.reshape(51, 6, 7, height, width).mean(
        axis=2, dtype=np.float64
    )
    return member_weekly.mean(axis=0, dtype=np.float64).astype(np.float32)


def test_reduction_reads_each_chunk_once_and_applies_physical_transforms() -> None:
    height, width = 2, 3
    shape = (1, 51, 42, len(physical.SOURCE_CHANNEL_NAMES), height, width)
    source = np.zeros(shape, dtype=np.float32)
    member = np.arange(51, dtype=np.float32)[:, None, None, None]
    day = np.arange(42, dtype=np.float32)[None, :, None, None]
    row = np.arange(height, dtype=np.float32)[None, None, :, None]
    column = np.arange(width, dtype=np.float32)[None, None, None, :]

    q850 = -0.002 + 0.0001 * member + 0.00001 * day + 0.00002 * row
    q850 = np.broadcast_to(q850, (51, 42, height, width)).astype(np.float32)
    u850 = 1.0 + 0.02 * member - 0.01 * day + 0.1 * column
    u850 = np.broadcast_to(u850, q850.shape).astype(np.float32)
    v850 = -2.0 + 0.01 * member + 0.03 * day + 0.2 * row
    v850 = np.broadcast_to(v850, q850.shape).astype(np.float32)
    tcwv = -2.0 + 0.2 * member + 0.03 * day + row + column
    tcwv = np.broadcast_to(tcwv, q850.shape).astype(np.float32)
    z500 = 50_000.0 + 10.0 * member + day + row + column
    z500 = np.broadcast_to(z500, q850.shape).astype(np.float32)
    msl = 100_000.0 + member + 2.0 * day + row
    msl = np.broadcast_to(msl, q850.shape).astype(np.float32)
    ttr = -(220.0 + 0.1 * member + 0.2 * day + row + column)
    ttr = np.broadcast_to(ttr, q850.shape).astype(np.float32)
    raw = {
        "q850": q850,
        "u850": u850,
        "v850": v850,
        "tcwv": tcwv,
        "z500": z500,
        "msl": msl,
        "ttr": ttr,
    }
    for name, values in raw.items():
        source[0, :, :, physical.SOURCE_CHANNEL_NAMES.index(name)] = values

    tracked = TrackingArray(source)
    result = physical._summarize_initialization(
        tracked,
        source_index=0,
        channel_names=physical.SOURCE_CHANNEL_NAMES,
        latitude_slice=slice(None),
        longitude_slice=slice(None),
    )

    assert tuple(result) == physical.PHYSICAL_FEATURE_NAMES
    assert len(tracked.calls) == 6 * 7
    selections = {
        (
            call[2].start,
            call[2].stop,
            call[3].start,
            call[3].stop,
        )
        for call in tracked.calls
    }
    assert len(selections) == 6 * 7
    assert all(call[0] == 0 for call in tracked.calls)

    clipped_q850 = np.maximum(q850, np.float32(0.0))
    clipped_tcwv = np.maximum(tcwv, np.float32(0.0))
    np.testing.assert_allclose(
        result["q850_mean"],
        _weekly_member_then_ensemble(clipped_q850),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result["u850_mean"], _weekly_member_then_ensemble(u850), rtol=2e-6
    )
    np.testing.assert_allclose(
        result["v850_mean"], _weekly_member_then_ensemble(v850), rtol=2e-6
    )
    np.testing.assert_allclose(
        result["z500_mean"], _weekly_member_then_ensemble(z500), rtol=2e-6
    )
    np.testing.assert_allclose(
        result["msl_mean"], _weekly_member_then_ensemble(msl), rtol=2e-6
    )
    np.testing.assert_allclose(
        result["olr_mean"], -_weekly_member_then_ensemble(ttr), rtol=2e-6
    )

    tcwv_member_weekly = clipped_tcwv.reshape(51, 6, 7, height, width).mean(
        axis=2, dtype=np.float64
    )
    np.testing.assert_allclose(
        result["tcwv_mean"],
        tcwv_member_weekly.mean(axis=0).astype(np.float32),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result["tcwv_spread"],
        tcwv_member_weekly.std(axis=0, ddof=0).astype(np.float32),
        rtol=2e-6,
    )
    q_u = (clipped_q850.astype(np.float64) * u850).reshape(51, 6, 7, height, width)
    q_v = (clipped_q850.astype(np.float64) * v850).reshape(51, 6, 7, height, width)
    np.testing.assert_allclose(
        result["q850_u850_flux_mean"],
        q_u.mean(axis=2).mean(axis=0).astype(np.float32),
        rtol=2e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result["q850_v850_flux_mean"],
        q_v.mean(axis=2).mean(axis=0).astype(np.float32),
        rtol=2e-6,
        atol=1e-9,
    )
    assert np.any(q850 < 0.0)
    assert np.all(result["q850_mean"] >= 0.0)
    assert np.all(result["tcwv_mean"] >= 0.0)
    assert np.all(result["tcwv_spread"] >= 0.0)
    assert np.all(result["olr_mean"] > 0.0)


def _synthetic_predictors(
    initializations: np.ndarray,
    source_fingerprint: str = "synthetic-source-fingerprint",
) -> physical.FuxiPhysicalPredictors:
    shape = (len(initializations), 6, 27, 27)
    values = {}
    for index, name in enumerate(physical.PHYSICAL_FEATURE_NAMES, start=1):
        values[name] = np.full(shape, float(index), dtype=np.float32)
    return physical.FuxiPhysicalPredictors(
        initializations=np.asarray(initializations, dtype="datetime64[D]"),
        latitude=physical.EXPECTED_LATITUDE.copy(),
        longitude=physical.EXPECTED_LONGITUDE.copy(),
        source_fingerprint=source_fingerprint,
        source_store="/synthetic/source.zarr",
        **values,
    )


def test_raw_final_cache_round_trip_alignment_and_stale_rejection(
    tmp_path: Path,
) -> None:
    initializations = np.asarray(["2002-06-02", "2002-06-06"], dtype="datetime64[D]")
    predictors = _synthetic_predictors(initializations)
    cache = tmp_path / "physical.npz"

    checksum = physical._write_final_cache(cache, predictors)
    loaded = physical._load_final_cache(
        cache,
        expected_initializations=initializations,
        expected_source_fingerprint=predictors.source_fingerprint,
    )

    assert loaded.cache_path == str(cache.resolve())
    assert loaded.cache_sha256 == checksum
    assert tuple(loaded.feature_fields) == physical.PHYSICAL_FEATURE_NAMES
    for name in physical.PHYSICAL_FEATURE_NAMES:
        np.testing.assert_array_equal(
            loaded.feature_fields[name], predictors.feature_fields[name]
        )
    with np.load(cache, allow_pickle=False) as stored:
        assert np.asarray(stored["normalization"]).item() == "none_raw_native_values"
        assert json.loads(np.asarray(stored["physical_transforms_json"]).item()) == (
            physical.PHYSICAL_TRANSFORMS
        )
        assert np.asarray(stored["olr_conversion"]).item() == "olr=-ttr"
    assert not list(tmp_path.glob("*.temporary"))

    forecast = SimpleNamespace(
        initializations=np.asarray(
            ["2002-06-02", "2002-06-06", "2020-06-02"],
            dtype="datetime64[D]",
        ),
        latitude=physical.EXPECTED_LATITUDE.copy(),
        longitude=physical.EXPECTED_LONGITUDE.copy(),
    )
    physical.validate_fuxi_physical_predictors(loaded, forecast)
    with pytest.raises(physical.PhysicalCacheContractError, match="stale"):
        physical._load_final_cache(
            cache, expected_source_fingerprint="different-source"
        )


def test_cache_rejects_post_2019_initialization() -> None:
    predictors = _synthetic_predictors(
        np.asarray(["2020-06-02"], dtype="datetime64[D]")
    )
    with pytest.raises(physical.PhysicalCacheContractError, match="post-2019"):
        physical._validate_predictors(predictors)


def _screening_dates() -> np.ndarray:
    offsets = np.linspace(0, 120, 35, dtype=np.int64)
    dates = []
    for year in physical.SCREENING_YEARS:
        start = np.datetime64(f"{year}-06-01", "D")
        dates.extend(start + offsets.astype("timedelta64[D]"))
    return np.asarray(dates, dtype="datetime64[D]")


def test_screening_selection_and_parallel_task_partition() -> None:
    selected_dates = _screening_dates()
    all_dates = np.concatenate(
        (
            np.asarray(["2001-06-01"], dtype="datetime64[D]"),
            selected_dates,
            np.asarray(["2020-06-01"], dtype="datetime64[D]"),
        )
    )
    indices, initializations = physical._select_screening_initializations(all_dates)
    assert len(indices) == 630
    np.testing.assert_array_equal(initializations, selected_dates)
    assert np.datetime_as_string(initializations[-1], unit="D") < "2020-01-01"

    contract = physical.SourceContract(
        source_store="/synthetic/source.zarr",
        source_fingerprint="fingerprint",
        all_initializations=all_dates,
        selected_source_indices=indices,
        selected_initializations=initializations,
        latitude=physical.EXPECTED_LATITUDE,
        longitude=physical.EXPECTED_LONGITUDE,
        latitude_slice=slice(0, 27),
        longitude_slice=slice(0, 27),
        channel_names=physical.SOURCE_CHANNEL_NAMES,
    )
    task_records = [
        physical._selected_task_records(
            contract,
            task_index=task_index,
            task_count=64,
            initialization=None,
            max_initializations=None,
        )
        for task_index in range(64)
    ]
    flattened = [record for task in task_records for record in task]
    assert len(flattened) == 630
    assert len({index for index, _ in flattened}) == 630
    assert max(map(len, task_records)) - min(map(len, task_records)) <= 1
    with pytest.raises(physical.PhysicalCacheContractError, match="serial"):
        physical._selected_task_records(
            contract,
            task_index=0,
            task_count=1,
            initialization=None,
            max_initializations=None,
        )

    with pytest.raises(physical.PhysicalCacheContractError, match="35 JJAS"):
        physical._select_screening_initializations(all_dates[1:-2])
