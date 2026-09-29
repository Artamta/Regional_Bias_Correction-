"""Focused synthetic tests for the all-season FuXi member cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fuxi_allseason_member_cache as cache


def _allseason_dates() -> np.ndarray:
    dates: list[np.datetime64] = []
    offsets = np.arange(cache.EXPECTED_INITIALIZATIONS_PER_YEAR) * np.timedelta64(
        3, "D"
    )
    for year in cache.YEARS:
        dates.extend((np.datetime64(f"{year}-01-01", "D") + offsets).tolist())
    return np.asarray(dates, dtype="datetime64[D]")


def _contract(initializations: np.ndarray) -> cache.SourceContract:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    return cache.SourceContract(
        source_store="/synthetic/native.zarr",
        source_fingerprint="synthetic-source-fingerprint",
        source_indices=np.arange(len(dates), dtype=np.int64),
        initializations=dates,
        latitude=cache.EXPECTED_LATITUDE.copy(),
        longitude=cache.EXPECTED_LONGITUDE.copy(),
        latitude_slice=slice(0, 27),
        longitude_slice=slice(0, 27),
        channel_names=cache.SOURCE_CHANNEL_NAMES,
    )


class TrackingForecast:
    """Return a synthetic TP block while retaining the exact source indexer."""

    def __init__(self, tp: np.ndarray):
        self.tp = tp
        self.calls: list[tuple[object, ...]] = []

    def __getitem__(self, key: tuple[object, ...]) -> np.ndarray:
        self.calls.append(key)
        return self.tp


def test_member_weekly_reduction_preserves_members_and_converts_units() -> None:
    member = np.arange(cache.MEMBER_COUNT, dtype=np.float32)[:, None, None, None]
    day = np.arange(cache.LEAD_DAY_COUNT, dtype=np.float32)[None, :, None, None]
    row = np.arange(27, dtype=np.float32)[None, None, :, None]
    column = np.arange(27, dtype=np.float32)[None, None, None, :]
    daily_mm = 1.0 + member * 0.1 + day * 0.2 + row * 0.01 + column * 0.001
    hourly = np.asarray(daily_mm / np.float32(24.0), dtype=np.float32)
    tracked = TrackingForecast(hourly)

    weekly = cache._extract_initialization(
        tracked,
        source_index=7,
        latitude_slice=slice(34, 61),
        longitude_slice=slice(40, 67),
    )

    expected = (
        (hourly * np.float32(24.0))
        .reshape(51, 6, 7, 27, 27)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )
    assert weekly.shape == (51, 6, 27, 27)
    assert weekly.dtype == np.float32
    np.testing.assert_array_equal(weekly, expected)
    assert len(tracked.calls) == 1
    selection = tracked.calls[0]
    assert selection[0] == 7
    assert selection[3] == cache.TP_CHANNEL_INDEX
    assert selection[4] == slice(34, 61)
    assert selection[5] == slice(40, 67)
    assert not np.array_equal(weekly[0], weekly[-1])

    with pytest.raises(cache.MemberCacheContractError, match="negative rainfall"):
        cache._weekly_tp_members(-np.ones_like(hourly))


def test_full_inventory_and_smoke_scope_are_split_safe() -> None:
    dates = _allseason_dates()
    np.testing.assert_array_equal(
        cache._validate_allseason_initializations(dates), dates
    )
    contract = _contract(dates)

    smoke = cache._scope_records(contract, smoke=True)
    smoke_dates = np.asarray([record[1] for record in smoke], dtype="datetime64[D]")
    smoke_years = cache._initialization_years(smoke_dates)
    assert len(smoke) == 64
    assert np.count_nonzero(smoke_years <= 2017) == 32
    assert np.count_nonzero((smoke_years >= 2018) & (smoke_years <= 2019)) == 16
    assert np.count_nonzero(smoke_years >= 2020) == 16
    smoke_ends = smoke_dates + np.timedelta64(41, "D")
    assert np.all(smoke_ends[smoke_years <= 2017] < np.datetime64("2018-01-01"))
    validation = (smoke_years >= 2018) & (smoke_years <= 2019)
    assert np.all(smoke_ends[validation] < np.datetime64("2020-01-01"))
    assert np.all(np.diff([record[0] for record in smoke]) > 0)
    assert [record[0] for record in smoke] != list(range(64))

    tasks = [
        cache._selected_task_records(
            contract,
            task_index=task_index,
            task_count=260,
            initialization=None,
            max_initializations=None,
            smoke=False,
        )
        for task_index in range(260)
    ]
    assert {len(records) for records in tasks} == {8}
    flattened = sorted(record for records in tasks for record in records)
    assert flattened == cache._scope_records(contract)

    smoke_serial = cache._selected_task_records(
        contract,
        task_index=0,
        task_count=1,
        initialization=None,
        max_initializations=None,
        smoke=True,
    )
    assert smoke_serial == smoke
    bounded_tasks = [
        cache._selected_task_records(
            contract,
            task_index=task_index,
            task_count=3,
            initialization=None,
            max_initializations=5,
            smoke=False,
        )
        for task_index in range(3)
    ]
    assert sorted(record for records in bounded_tasks for record in records) == (
        cache._scope_records(contract, max_initializations=5)
    )
    with pytest.raises(cache.MemberCacheContractError, match="serial full-archive"):
        cache._selected_task_records(
            contract,
            task_index=0,
            task_count=1,
            initialization=None,
            max_initializations=None,
            smoke=False,
        )
    with pytest.raises(cache.MemberCacheContractError, match="2080 source initializations"):
        cache._validate_allseason_initializations(dates[:-1])


def test_build_part_is_atomic_idempotent_and_checks_completion(
    tmp_path: Path,
) -> None:
    initialization = np.datetime64("2002-01-01", "D")
    contract = _contract(np.asarray([initialization]))
    raw = np.full(
        (cache.MEMBER_COUNT, cache.LEAD_DAY_COUNT, *cache.GRID_SHAPE),
        np.float32(2.0 / 24.0),
        dtype=np.float32,
    )
    forecast = TrackingForecast(raw)
    group = {
        "forecast": forecast,
        "init_complete": np.asarray([True], dtype=np.bool_),
    }

    part, created = cache.build_part(
        group, contract, 0, initialization, tmp_path / "parts"
    )
    assert created is True
    assert len(forecast.calls) == 1
    np.testing.assert_allclose(
        cache._load_part(part, contract, 0, initialization),
        np.float32(2.0),
    )
    same_part, created = cache.build_part(
        group, contract, 0, initialization, tmp_path / "parts"
    )
    assert same_part == part
    assert created is False
    assert len(forecast.calls) == 1
    assert not list(tmp_path.rglob("*.temporary"))

    incomplete_group = {
        "forecast": TrackingForecast(raw),
        "init_complete": np.asarray([False], dtype=np.bool_),
    }
    with pytest.raises(cache.MemberCacheContractError, match="not marked complete"):
        cache.build_part(
            incomplete_group,
            contract,
            0,
            initialization,
            tmp_path / "different-parts",
        )


def test_atomic_parts_finalize_memmap_and_verify(tmp_path: Path) -> None:
    dates = np.asarray(
        ["2002-01-01", "2002-01-04", "2002-01-07"], dtype="datetime64[D]"
    )
    contract = _contract(dates)
    parts = tmp_path / "parts"
    expected: list[np.ndarray] = []
    for source_index, initialization in enumerate(dates[:2]):
        member = np.arange(51, dtype=np.float32)[:, None, None, None]
        values = np.broadcast_to(
            np.float32(source_index + 1) + member / np.float32(100.0),
            cache.MEMBER_FIELD_SHAPE,
        ).copy()
        expected.append(values)
        path = cache._part_path(parts, initialization)
        cache._atomic_npz(
            path,
            cache._part_payload(
                contract, source_index, initialization, values
            ),
        )
        np.testing.assert_array_equal(
            cache._load_part(path, contract, source_index, initialization), values
        )

    output = tmp_path / "members.npy"
    artifacts = cache.finalize_cache(
        contract, parts, output, max_initializations=2
    )
    assert artifacts.data == output.resolve()
    assert artifacts.metadata.is_file()
    assert artifacts.manifest.is_file()
    assert artifacts.checksums.is_file()
    assert cache.sha256_file(output) == artifacts.data_sha256
    report = cache.verify_cache(output)
    assert report["status"] == "verified"
    assert report["shape"] == [2, 51, 6, 27, 27]
    assert report["full_archive"] is False

    data, metadata = cache.load_member_cache(output)
    assert isinstance(data, np.memmap)
    assert data.flags.writeable is False
    np.testing.assert_array_equal(data, np.stack(expected))
    assert metadata["dims"] == list(cache.OUTPUT_DIMS)
    assert metadata["initializations"] == ["2002-01-01", "2002-01-04"]
    assert metadata["source_init_indices"] == [0, 1]
    assert metadata["member_labels"] == list(range(51))
    assert metadata["lead_week_labels"] == list(range(1, 7))
    assert metadata["normalization"] == "none"
    assert metadata["ensemble_aggregation"] == "none; all 51 members retained"
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert [record["source_init_index"] for record in manifest["records"]] == [0, 1]
    assert not list(tmp_path.rglob("*.temporary"))


def test_part_staleness_and_final_checksum_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    initialization = np.datetime64("2002-01-01", "D")
    contract = _contract(np.asarray([initialization]))
    values = np.ones(cache.MEMBER_FIELD_SHAPE, dtype=np.float32)
    part = cache._part_path(tmp_path / "parts", initialization)
    cache._atomic_npz(
        part, cache._part_payload(contract, 0, initialization, values)
    )
    stale = cache.SourceContract(
        **{
            **contract.__dict__,
            "source_fingerprint": "different-source-fingerprint",
        }
    )
    with pytest.raises(cache.MemberCacheContractError, match="stale"):
        cache._load_part(part, stale, 0, initialization)

    output = tmp_path / "members.npy"
    cache.finalize_cache(contract, part.parent, output)
    with output.open("rb+") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(cache.MemberCacheContractError, match="checksums differ"):
        cache.verify_cache(output)
