"""Focused synthetic tests for the all-season FuXi physical-context cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fuxi_allseason_physical_context_cache as cache


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
        source_fingerprint="synthetic-physical-source-fingerprint",
        source_indices=np.arange(len(dates), dtype=np.int64),
        initializations=dates,
        latitude=cache.EXPECTED_LATITUDE.copy(),
        longitude=cache.EXPECTED_LONGITUDE.copy(),
        latitude_slice=slice(0, 27),
        longitude_slice=slice(0, 27),
        channel_names=cache.SOURCE_CHANNEL_NAMES,
    )


def _valid_context(offset: float = 0.0) -> np.ndarray:
    feature = np.arange(cache.FEATURE_COUNT, dtype=np.float32)[None, :, None, None]
    week = np.arange(cache.LEAD_WEEK_COUNT, dtype=np.float32)[:, None, None, None]
    row = np.arange(27, dtype=np.float32)[None, None, :, None]
    column = np.arange(27, dtype=np.float32)[None, None, None, :]
    values = (
        np.float32(1.0 + offset)
        + feature
        + week * np.float32(0.1)
        + row * np.float32(0.001)
        + column * np.float32(0.0001)
    )
    return np.asarray(values, dtype=np.float32)


class SyntheticForecast:
    """Generate native channel chunks and retain the exact source indexers."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    @staticmethod
    def field(name: str, lead_slice: slice) -> np.ndarray:
        member = np.arange(cache.MEMBER_COUNT, dtype=np.float32)[:, None, None, None]
        day = np.arange(
            int(lead_slice.start), int(lead_slice.stop), dtype=np.float32
        )[None, :, None, None]
        row = np.arange(27, dtype=np.float32)[None, None, :, None]
        column = np.arange(27, dtype=np.float32)[None, None, None, :]
        spatial = row * np.float32(0.001) + column * np.float32(0.0001)
        if name == "q850":
            values = (
                (member - np.float32(20.0)) * np.float32(0.0005)
                + day * np.float32(0.00001)
                + spatial * np.float32(0.001)
            )
        elif name == "tcwv":
            values = (
                member - np.float32(10.0)
                + day * np.float32(0.1)
                + spatial
            )
        elif name == "u850":
            values = (
                np.float32(2.0)
                + member * np.float32(0.1)
                + day * np.float32(0.01)
                + spatial
            )
        elif name == "v850":
            values = (
                -np.float32(1.0)
                + member * np.float32(0.05)
                - day * np.float32(0.02)
                + spatial
            )
        elif name == "t2m":
            values = (
                np.float32(275.0)
                + member * np.float32(0.02)
                + day * np.float32(0.1)
                + spatial
            )
        elif name == "ttr":
            values = -(
                np.float32(200.0)
                + member * np.float32(0.01)
                + day * np.float32(0.05)
                + spatial
            )
        elif name == "z500":
            values = (
                np.float32(50_000.0)
                + member
                + day
                + spatial
            )
        elif name == "msl":
            values = (
                np.float32(100_000.0)
                + member
                + day
                + spatial
            )
        else:
            channel_index = cache.SOURCE_CHANNEL_NAMES.index(name)
            values = (
                np.float32(channel_index + 1)
                + member * np.float32(0.001)
                + day * np.float32(0.001)
                + spatial
            )
        return np.broadcast_to(
            values, (cache.MEMBER_COUNT, cache.DAYS_PER_WEEK, *cache.GRID_SHAPE)
        ).astype(np.float32, copy=False)

    def __getitem__(self, key: tuple[object, ...]) -> np.ndarray:
        self.calls.append(key)
        (
            _,
            member_slice,
            lead_slice,
            channel_slice,
            latitude_slice,
            longitude_slice,
        ) = key
        assert member_slice == slice(None)
        assert latitude_slice == slice(0, 27)
        assert longitude_slice == slice(0, 27)
        channel_indices = range(int(channel_slice.start), int(channel_slice.stop))
        fields = [
            self.field(cache.SOURCE_CHANNEL_NAMES[index], lead_slice)
            for index in channel_indices
        ]
        return np.stack(fields, axis=2)


def _accepted_expected(forecast: SyntheticForecast, lead_week: int) -> np.ndarray:
    lead_slice = slice(lead_week * 7, (lead_week + 1) * 7)
    raw = {
        name: forecast.field(name, lead_slice)
        for name in cache.RAW_CHANNEL_NAMES
    }
    q850 = np.maximum(raw["q850"], np.float32(0.0))
    tcwv = np.maximum(raw["tcwv"], np.float32(0.0))

    def weekly(values: np.ndarray) -> np.ndarray:
        return (
            values.mean(axis=1, dtype=np.float64)
            .mean(axis=0, dtype=np.float64)
            .astype(np.float32)
        )

    expected = {
        "t2m_mean": weekly(raw["t2m"]),
        "tcwv_mean": weekly(tcwv),
        "q850_mean": weekly(q850),
        "u850_mean": weekly(raw["u850"]),
        "v850_mean": weekly(raw["v850"]),
        "q850_u850_flux_mean": (
            q850.astype(np.float64)
            .__mul__(raw["u850"].astype(np.float64))
            .mean(axis=1)
            .mean(axis=0)
            .astype(np.float32)
        ),
        "q850_v850_flux_mean": (
            q850.astype(np.float64)
            .__mul__(raw["v850"].astype(np.float64))
            .mean(axis=1)
            .mean(axis=0)
            .astype(np.float32)
        ),
        "z500_mean": weekly(raw["z500"]),
        "msl_mean": weekly(raw["msl"]),
        "olr_mean": -weekly(raw["ttr"]),
    }
    return np.stack(
        [expected[name] for name in cache.PHYSICAL_CONTEXT_FEATURE_NAMES]
    )


def test_frozen_field_order_and_accepted_weekly_reductions() -> None:
    assert cache.PHYSICAL_CONTEXT_FEATURE_NAMES == (
        "t2m_mean",
        "tcwv_mean",
        "q850_mean",
        "u850_mean",
        "v850_mean",
        "q850_u850_flux_mean",
        "q850_v850_flux_mean",
        "z500_mean",
        "msl_mean",
        "olr_mean",
    )
    assert cache.CONTEXT_FIELD_SHAPE == (6, 10, 27, 27)
    forecast = SyntheticForecast()
    actual = cache._summarize_initialization(
        forecast,
        source_index=9,
        channel_names=cache.SOURCE_CHANNEL_NAMES,
        latitude_slice=slice(0, 27),
        longitude_slice=slice(0, 27),
    )

    assert actual.shape == (6, 10, 27, 27)
    assert actual.dtype == np.float32
    assert len(forecast.calls) == 42
    assert {call[0] for call in forecast.calls} == {9}
    assert [call[3].start for call in forecast.calls[:7]] == [
        0,
        4,
        8,
        12,
        16,
        20,
        24,
    ]
    expected = np.stack(
        [_accepted_expected(forecast, week) for week in range(6)]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    assert np.all(actual[:, cache.FEATURE_NAMES.index("tcwv_mean")] >= 0.0)
    assert np.all(actual[:, cache.FEATURE_NAMES.index("q850_mean")] >= 0.0)
    assert np.all(actual[:, cache.FEATURE_NAMES.index("olr_mean")] >= 0.0)


def test_cache_smoke_is_exactly_one_split_safe_case_per_era() -> None:
    contract = _contract(_allseason_dates())
    records = cache._scope_records(contract, smoke=True)
    dates = np.asarray([record[1] for record in records], dtype="datetime64[D]")
    years = cache._initialization_years(dates)
    assert len(records) == 3
    assert np.count_nonzero(years <= 2017) == 1
    assert np.count_nonzero((years >= 2018) & (years <= 2019)) == 1
    assert np.count_nonzero(years >= 2020) == 1
    forecast_ends = dates + np.timedelta64(41, "D")
    assert forecast_ends[0] < np.datetime64("2018-01-01", "D")
    assert forecast_ends[1] < np.datetime64("2020-01-01", "D")
    assert np.all(np.diff([record[0] for record in records]) > 0)

    parsed = cache.parse_args(
        [
            "build",
            "--parts-dir",
            "/tmp/physical-parts",
            "--task-index",
            "0",
            "--task-count",
            "3",
            "--smoke",
        ]
    )
    assert parsed.command == "build"
    assert parsed.smoke is True


def test_build_part_is_atomic_idempotent_and_checks_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialization = np.datetime64("2002-01-01", "D")
    contract = _contract(np.asarray([initialization]))
    expected = _valid_context()
    calls: list[int] = []

    def summarize(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return expected.copy()

    monkeypatch.setattr(cache, "_summarize_initialization", summarize)
    group = {
        "forecast": object(),
        "init_complete": np.asarray([True], dtype=np.bool_),
    }
    part, created = cache.build_part(
        group, contract, 0, initialization, tmp_path / "parts"
    )
    assert created is True
    np.testing.assert_array_equal(
        cache._load_part(part, contract, 0, initialization), expected
    )
    same_part, created = cache.build_part(
        group, contract, 0, initialization, tmp_path / "parts"
    )
    assert same_part == part
    assert created is False
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.temporary"))

    with pytest.raises(
        cache.PhysicalContextCacheContractError,
        match="not marked complete",
    ):
        cache.build_part(
            {
                "forecast": object(),
                "init_complete": np.asarray([False], dtype=np.bool_),
            },
            contract,
            0,
            initialization,
            tmp_path / "different-parts",
        )


def test_parts_finalize_read_only_memmap_and_alignment_metadata(
    tmp_path: Path,
) -> None:
    dates = np.asarray(
        ["2002-01-01", "2002-01-04", "2002-01-07"],
        dtype="datetime64[D]",
    )
    contract = _contract(dates)
    parts = tmp_path / "parts"
    expected: list[np.ndarray] = []
    for source_index, initialization in enumerate(dates[:2]):
        values = _valid_context(float(source_index))
        expected.append(values)
        path = cache._part_path(parts, initialization)
        cache._atomic_npz(
            path,
            cache._part_payload(
                contract, source_index, initialization, values
            ),
        )

    output = tmp_path / "physical-context.npy"
    artifacts = cache.finalize_cache(
        contract, parts, output, max_initializations=2
    )
    report = cache.verify_cache(
        output,
        expected_source_fingerprint=contract.source_fingerprint,
    )
    assert report["status"] == "verified"
    assert report["shape"] == [2, 6, 10, 27, 27]
    assert report["feature_contract_sha256"] == cache.FEATURE_CONTRACT_SHA256
    assert report["full_archive"] is False
    assert cache.sha256_file(output) == artifacts.data_sha256

    data, metadata = cache.load_physical_context_cache(output)
    assert isinstance(data, np.memmap)
    assert data.flags.writeable is False
    np.testing.assert_array_equal(data, np.stack(expected))
    assert metadata["dims"] == list(cache.OUTPUT_DIMS)
    assert metadata["initializations"] == ["2002-01-01", "2002-01-04"]
    assert metadata["source_init_indices"] == [0, 1]
    assert metadata["latitude"] == cache.EXPECTED_LATITUDE.tolist()
    assert metadata["longitude"] == cache.EXPECTED_LONGITUDE.tolist()
    assert metadata["source_fingerprint"] == contract.source_fingerprint
    assert metadata["feature_names"] == list(cache.FEATURE_NAMES)
    assert metadata["normalization"] == "none"
    assert metadata["data_file"] == output.name
    assert metadata["data_sha256"] == artifacts.data_sha256
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert [record["source_init_index"] for record in manifest["records"]] == [
        0,
        1,
    ]
    assert not list(tmp_path.rglob("*.temporary"))


def test_stale_feature_contract_and_final_checksum_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    initialization = np.datetime64("2002-01-01", "D")
    contract = _contract(np.asarray([initialization]))
    values = _valid_context()
    part = cache._part_path(tmp_path / "parts", initialization)
    payload = dict(cache._part_payload(contract, 0, initialization, values))
    payload["feature_names"] = np.asarray(tuple(reversed(cache.FEATURE_NAMES)))
    cache._atomic_npz(part, payload)
    with pytest.raises(
        cache.PhysicalContextCacheContractError,
        match="feature contract differs",
    ):
        cache._load_part(part, contract, 0, initialization)

    cache._atomic_npz(
        part, cache._part_payload(contract, 0, initialization, values)
    )
    output = tmp_path / "physical-context.npy"
    cache.finalize_cache(contract, part.parent, output)
    with output.open("rb+") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(
        cache.PhysicalContextCacheContractError,
        match="checksums differ",
    ):
        cache.verify_cache(output)
