"""Focused tests for the independent FuXi training-alignment auditor."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import audit_fuxi_allseason_training_alignment as audit


def test_target_dates_are_exactly_plus_1_through_plus_42() -> None:
    starts = np.asarray(("2017-11-17", "2021-12-29"), dtype="datetime64[D]")
    dates = audit.target_dates(starts)
    assert dates.shape == (2, 6, 7)
    assert dates[0, 0, 0] == np.datetime64("2017-11-18")
    assert dates[0, 0, -1] == np.datetime64("2017-11-24")
    assert dates[0, -1, 0] == np.datetime64("2017-12-23")
    assert dates[0, -1, -1] == np.datetime64("2017-12-29")
    assert dates[-1, -1, -1] == np.datetime64("2022-02-09")
    assert not np.any(dates == starts[:, None, None])


def test_split_reconstruction_purges_every_cross_boundary_window() -> None:
    starts = np.asarray(
        (
            "2017-11-19",  # +42 = 2017-12-31: retained train
            "2017-11-20",  # +42 = 2018-01-01: embargoed train candidate
            "2018-01-03",
            "2019-11-19",  # +42 = 2019-12-31: retained validation
            "2019-11-20",  # +42 = 2020-01-01: embargoed validation candidate
            "2020-01-03",
            "2021-12-29",  # +42 in allowed 2022 observation source
        ),
        dtype="datetime64[D]",
    )
    result = audit.reconstruct_splits(starts)
    np.testing.assert_array_equal(result.indices["train"], np.asarray((0,)))
    np.testing.assert_array_equal(result.indices["validation"], np.asarray((2, 3)))
    np.testing.assert_array_equal(result.indices["test"], np.asarray((5, 6)))
    np.testing.assert_array_equal(result.indices["embargo"], np.asarray((1, 4)))
    assert result.outcome_ends[1] == np.datetime64("2018-01-01")
    assert result.outcome_ends[4] == np.datetime64("2020-01-01")
    assert result.outcome_ends[-1] == np.datetime64("2022-02-09")


def test_cf_day_decode_is_narrow_and_calendar_exact() -> None:
    decoded = audit.decode_cf_days(
        np.asarray((0, 1, 364), dtype=np.int64),
        {
            "units": "days since 2002-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        },
    )
    np.testing.assert_array_equal(
        decoded,
        np.asarray(("2002-01-01", "2002-01-02", "2002-12-31"), dtype="datetime64[D]"),
    )
    with pytest.raises(audit.AlignmentAuditError, match="unsupported IMD time units"):
        audit.decode_cf_days(
            np.asarray((0,), dtype=np.int64),
            {"units": "hours since 2002-01-01", "calendar": "proleptic_gregorian"},
        )


def test_logical_hash_binds_dtype_shape_and_datetime_bytes() -> None:
    first = np.arange(6, dtype=np.int16).reshape(2, 3)
    assert len(audit.array_sha256(first)) == 64
    assert audit.array_sha256(first) != audit.array_sha256(first.astype(np.int32))
    assert audit.array_sha256(first) != audit.array_sha256(first.reshape(3, 2))
    dates = np.asarray(("2017-11-18", "2017-12-29"), dtype="datetime64[D]")
    assert len(audit.array_sha256(dates)) == 64


def test_weekly_target_reconstruction_uses_float64_seven_day_mean() -> None:
    daily = np.arange(2 * 6 * 7 * 2 * 2, dtype=np.float32).reshape(2, 6, 7, 2, 2)
    weekly = audit.weekly_target_mean(daily)
    expected = daily.astype(np.float64).mean(axis=2).astype(np.float32)
    np.testing.assert_array_equal(weekly, expected)
    assert weekly.dtype == np.float32


def test_native_weekly_conversion_preserves_member_and_lead_groups() -> None:
    daily = np.ones((51, 42, 2, 3), dtype=np.float32)
    for day in range(42):
        daily[:, day] *= np.float32(day + 1)
    weekly = audit.weekly_native_tp(daily)
    assert weekly.shape == (51, 6, 2, 3)
    expected = (
        (daily * np.float32(24.0))
        .reshape(51, 6, 7, 2, 3)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )
    np.testing.assert_array_equal(weekly, expected)


def test_rename_noreplace_refuses_an_existing_receipt(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    destination = tmp_path / "receipt"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        audit.rename_noreplace(first, destination)
    assert first.is_dir()
    assert destination.is_dir()


def test_auditor_is_independent_and_observation_years_stop_at_2022() -> None:
    source = inspect.getsource(audit)
    assert "import fuxi_allseason_ensemble_calibration" not in source
    assert "import fuxi_allseason_member_cache" not in source
    assert audit.OBSERVATION_YEARS == tuple(range(2002, 2023))
    assert audit.MAX_OPEN_OBSERVATION_YEAR == 2022
    assert 2025 not in audit.OBSERVATION_YEARS
    assert "retrospective_equality_to_consumed_target_tensors_provable" in source
