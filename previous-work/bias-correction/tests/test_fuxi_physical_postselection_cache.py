"""Tests for the isolated 2020--2021 FuXi physical-feature cache."""

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

import fuxi_physical_feature_cache as development  # noqa: E402
import fuxi_physical_postselection_cache as postselection  # noqa: E402


def _postselection_dates() -> np.ndarray:
    offsets = np.linspace(1, 120, 35, dtype=np.int64)
    dates: list[np.datetime64] = []
    for year in postselection.POSTSELECTION_YEARS:
        start = np.datetime64(f"{year}-06-01", "D")
        dates.extend(start + offsets.astype("timedelta64[D]"))
    return np.asarray(dates, dtype="datetime64[D]")


def _contract() -> postselection.PostSelectionSourceContract:
    dates = _postselection_dates()
    indices = np.arange(100, 100 + len(dates), dtype=np.int64)
    return postselection.PostSelectionSourceContract(
        source_store="/synthetic/native_reforecast_global_2002_2021.zarr",
        source_fingerprint="source-fingerprint",
        scope_fingerprint="scope-fingerprint",
        all_initializations=dates.copy(),
        selected_source_indices=indices,
        selected_initializations=dates,
        latitude=postselection.EXPECTED_LATITUDE.copy(),
        longitude=postselection.EXPECTED_LONGITUDE.copy(),
        latitude_slice=slice(0, 27),
        longitude_slice=slice(0, 27),
        channel_names=development.SOURCE_CHANNEL_NAMES,
    )


def _predictors() -> postselection.FuxiPostSelectionPhysicalPredictors:
    dates = _postselection_dates()
    shape = (len(dates), 6, 27, 27)
    fields = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate(postselection.PHYSICAL_FEATURE_NAMES)
    }
    return postselection.FuxiPostSelectionPhysicalPredictors(
        initializations=dates,
        latitude=postselection.EXPECTED_LATITUDE.copy(),
        longitude=postselection.EXPECTED_LONGITUDE.copy(),
        source_fingerprint="source-fingerprint",
        scope_fingerprint="scope-fingerprint",
        source_store="/synthetic/native_reforecast_global_2002_2021.zarr",
        **fields,
    )


def test_scope_is_exactly_70_jjas_initializations_and_array_is_one_to_one() -> None:
    selected = _postselection_dates()
    all_dates = np.concatenate(
        (
            np.asarray(["2019-09-29"], dtype="datetime64[D]"),
            selected,
            np.asarray(["2022-06-02"], dtype="datetime64[D]"),
        )
    )
    indices, dates = postselection._select_postselection_initializations(all_dates)

    assert len(indices) == 70
    np.testing.assert_array_equal(dates, selected)
    years, months = development._date_parts(dates)
    assert set(years.tolist()) == {2020, 2021}
    assert set(months.tolist()).issubset({6, 7, 8, 9})

    contract = _contract()
    tasks = [
        postselection._selected_task_records(
            contract,
            task_index=index,
            task_count=70,
            initialization=None,
        )
        for index in range(70)
    ]
    assert all(len(task) == 1 for task in tasks)
    assert len({task[0][0] for task in tasks}) == 70
    with pytest.raises(
        postselection.PostSelectionCacheContractError, match="exactly 70 tasks"
    ):
        postselection._selected_task_records(
            contract,
            task_index=0,
            task_count=64,
            initialization=None,
        )
    with pytest.raises(
        postselection.PostSelectionCacheContractError, match="exactly 35 JJAS"
    ):
        postselection._select_postselection_initializations(all_dates[:-2])


def test_part_builder_delegates_to_development_reduction_and_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    source_index = int(contract.selected_source_indices[0])
    initialization = contract.selected_initializations[0]
    init_complete = np.ones(source_index + 1, dtype=np.bool_)
    sentinel_forecast = object()
    group = {"init_complete": init_complete, "forecast": sentinel_forecast}
    shape = (6, 27, 27)
    expected_fields = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate(postselection.PHYSICAL_FEATURE_NAMES)
    }
    calls: list[tuple[object, ...]] = []

    def fake_reduction(*args: object) -> dict[str, np.ndarray]:
        calls.append(args)
        return expected_fields

    monkeypatch.setattr(development, "_summarize_initialization", fake_reduction)
    part, created = postselection.build_part(
        group,
        contract,
        source_index,
        initialization,
        tmp_path / "parts",
    )

    assert created
    assert len(calls) == 1
    assert calls[0][0] is sentinel_forecast
    assert calls[0][1] == source_index
    loaded = postselection._load_part(part, contract, source_index, initialization)
    for name in postselection.PHYSICAL_FEATURE_NAMES:
        np.testing.assert_array_equal(loaded[name], expected_fields[name])
    assert not list(tmp_path.rglob("*.temporary"))

    reused_path, reused_created = postselection.build_part(
        group,
        contract,
        source_index,
        initialization,
        tmp_path / "parts",
    )
    assert reused_path == part
    assert not reused_created
    assert len(calls) == 1


def test_final_cache_roundtrip_metadata_alignment_and_stale_rejection(
    tmp_path: Path,
) -> None:
    predictors = _predictors()
    cache = tmp_path / "postselection.npz"
    checksum = postselection._write_final_cache(cache, predictors)
    loaded = postselection._load_final_cache(
        cache,
        expected_initializations=predictors.initializations,
        expected_source_fingerprint=predictors.source_fingerprint,
        expected_scope_fingerprint=predictors.scope_fingerprint,
    )

    assert loaded.cache_path == str(cache.resolve())
    assert loaded.cache_sha256 == checksum
    assert tuple(loaded.feature_fields) == postselection.PHYSICAL_FEATURE_NAMES
    with np.load(cache, allow_pickle=False) as stored:
        assert np.asarray(stored["selection"]).item() == (
            postselection.SELECTION_DESCRIPTION
        )
        assert np.asarray(stored["evaluation_role"]).item() == (
            "exploratory_reused_hindcast_evaluation"
        )
        assert np.asarray(stored["target_data_access"]).item() == "none"
        assert np.asarray(stored["normalization"]).item() == ("none_raw_native_values")
        assert (
            json.loads(np.asarray(stored["physical_transforms_json"]).item())
            == development.PHYSICAL_TRANSFORMS
        )
        assert np.asarray(stored["reduction_implementation"]).item() == (
            "fuxi_physical_feature_cache._summarize_initialization"
        )
    assert not list(tmp_path.glob("*.temporary"))

    forecast = SimpleNamespace(
        initializations=np.concatenate(
            (
                np.asarray(["2019-09-29"], dtype="datetime64[D]"),
                predictors.initializations,
            )
        ),
        latitude=postselection.EXPECTED_LATITUDE.copy(),
        longitude=postselection.EXPECTED_LONGITUDE.copy(),
    )
    aligned = postselection.load_fuxi_postselection_physical_predictors(
        forecast,
        cache,
        expected_source_fingerprint=predictors.source_fingerprint,
        expected_scope_fingerprint=predictors.scope_fingerprint,
    )
    np.testing.assert_array_equal(aligned.initializations, predictors.initializations)
    with pytest.raises(
        postselection.PostSelectionCacheContractError,
        match="source fingerprint is stale",
    ):
        postselection._load_final_cache(
            cache, expected_source_fingerprint="different-source"
        )
    with pytest.raises(
        postselection.PostSelectionCacheContractError,
        match="scope fingerprint is stale",
    ):
        postselection._load_final_cache(
            cache, expected_scope_fingerprint="different-scope"
        )


def test_cache_rejects_development_date_and_scope_fingerprint_changes() -> None:
    predictors = _predictors()
    bad_dates = predictors.initializations.copy()
    bad_dates[0] = np.datetime64("2019-06-02", "D")
    bad_predictors = postselection.FuxiPostSelectionPhysicalPredictors(
        **{
            **predictors.__dict__,
            "initializations": bad_dates,
        }
    )
    with pytest.raises(
        postselection.PostSelectionCacheContractError, match="scope is not exactly"
    ):
        postselection._validate_predictors(bad_predictors)

    dates = _postselection_dates()
    first = postselection._scope_fingerprint(
        "source-a", np.arange(70, dtype=np.int64), dates
    )
    changed_source = postselection._scope_fingerprint(
        "source-b", np.arange(70, dtype=np.int64), dates
    )
    changed_index = postselection._scope_fingerprint(
        "source-a", np.arange(1, 71, dtype=np.int64), dates
    )
    assert first != changed_source
    assert first != changed_index
