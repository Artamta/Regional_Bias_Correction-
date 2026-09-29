"""Unit checks for the clear-story spatial atlas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import plot_clear_story_spatial_atlas as atlas  # noqa: E402


def test_story_fields_use_requested_imd_minus_forecast_sign() -> None:
    observed = np.asarray([[[5.0, 7.0]]])
    raw = np.asarray([[[3.0, 9.0]]])
    corrected = np.asarray([[[4.5, 6.0]]])
    fields = atlas.story_fields(
        {
            "observed_mean": observed,
            "raw_mean": raw,
            "corrected_mean": corrected,
        }
    )
    np.testing.assert_array_equal(fields[3], [[[2.0, -2.0]]])
    np.testing.assert_array_equal(fields[4], [[[0.5, 1.0]]])


def test_display_interpolation_retains_native_support_and_nonnegativity() -> None:
    latitude = np.asarray([3.0, 2.0, 1.0])
    longitude = np.asarray([10.0, 11.0, 12.0])
    field = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]
    )
    support = np.asarray(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    dense_lat, dense_lon, dense = atlas.smooth_display_field(
        field,
        latitude,
        longitude,
        support,
        points_per_axis=60,
        nonnegative=True,
    )
    assert dense.shape == (60, 60)
    assert dense_lat[0] < dense_lat[-1]
    assert dense_lon[0] < dense_lon[-1]
    assert np.nanmin(dense) >= 0.0
    assert np.isnan(dense[0, 0])
    assert np.isfinite(dense[30, 30])


def test_nice_upper_is_stable_and_outward() -> None:
    assert atlas._nice_upper(7.01, step=2.0) == 8.0
    assert atlas._nice_upper(0.2, step=1.0) == 1.0


def test_atomic_json_write_is_complete_and_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    atlas._atomic_write_json(target, {"status": "complete", "lead_weeks": [1, 2]})
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert '"status": "complete"' in target.read_text(encoding="utf-8")
    try:
        atlas._atomic_write_json(target, {"status": "changed"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic JSON writer must not overwrite an artifact")
