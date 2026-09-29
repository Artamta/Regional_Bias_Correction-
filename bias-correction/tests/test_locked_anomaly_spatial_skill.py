"""Focused unit checks for the IMD-referenced anomaly spatial figures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import plot_locked_anomaly_spatial_skill as anomaly  # noqa: E402


def test_anomaly_is_case_matched_before_compositing() -> None:
    climatology = np.asarray(
        [
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            [[[[10.0, 20.0], [30.0, 40.0]]]],
        ]
    ).reshape(2, 1, 2, 2)
    departures = np.asarray(
        [
            [[[[2.0, -2.0], [4.0, -4.0]]]],
            [[[[6.0, -6.0], [8.0, -8.0]]]],
        ]
    ).reshape(2, 1, 2, 2)
    values = {
        "training_climatology": climatology,
        "weights": np.asarray([[1.0, 1.0], [1.0, 0.0]]),
        "truth_imd": climatology + departures,
        "raw_fuxi": climatology + 2.0 * departures,
        "corrected": climatology + 0.5 * departures,
    }
    result = anomaly.anomaly_composites(values)
    expected = np.asarray([[[4.0, -4.0], [6.0, np.nan]]])
    np.testing.assert_allclose(result["truth_imd"], expected, equal_nan=True)
    np.testing.assert_allclose(result["raw_fuxi"], 2.0 * expected, equal_nan=True)
    np.testing.assert_allclose(result["corrected"], 0.5 * expected, equal_nan=True)


def test_weighted_pattern_correlation_is_centered_and_signed() -> None:
    reference = np.asarray([[1.0, 2.0], [4.0, 8.0]])
    weights = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(
        anomaly.area_weighted_pattern_correlation(
            reference, 10.0 + 3.0 * reference, weights
        ),
        1.0,
    )
    assert np.isclose(
        anomaly.area_weighted_pattern_correlation(
            reference, 10.0 - 3.0 * reference, weights
        ),
        -1.0,
    )


def test_displayed_skill_is_per_lead_and_uses_imd_reference() -> None:
    truth = np.asarray(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[2.0, 1.0], [4.0, 7.0]],
        ]
    )
    composites = {
        "truth_imd": truth,
        "raw_fuxi": -truth,
        "corrected": 2.0 * truth + 11.0,
    }
    weights = np.ones((2, 2))
    result = anomaly.displayed_skill(composites, weights)
    np.testing.assert_allclose(result["raw_fuxi"], [-1.0, -1.0])
    np.testing.assert_allclose(result["corrected"], [1.0, 1.0])


def test_color_limit_is_common_outward_quantile() -> None:
    support = np.asarray([[True, True], [True, False]])
    base = np.asarray([[[1.2, -2.2], [3.1, np.nan]]])
    composites = {
        "truth_imd": base,
        "raw_fuxi": 2.0 * base,
        "corrected": 0.5 * base,
    }
    limit = anomaly._color_limit(composites, support)
    assert limit == np.ceil(limit)
    assert limit >= np.quantile(
        np.concatenate(
            [np.abs(composites[key][:, support]).ravel() for key in anomaly.FIELD_KEYS]
        ),
        0.99,
    )


def test_requested_corrected_forecast_label_is_exact() -> None:
    assert anomaly.COLUMN_LABELS[-1].startswith("Corrected Forecast")
