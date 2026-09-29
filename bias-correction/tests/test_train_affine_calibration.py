"""Focused tests for the train-only physical affine diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_train_affine_calibration as affine  # noqa: E402


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    truth = rng.gamma(1.8, 3.0, size=(10, 6, 3, 4)).astype(np.float32)
    prediction = np.maximum(0.0, 0.82 * truth + 0.7).astype(np.float32)
    weights = np.asarray(
        [[1.0, 0.8, 0.0, 0.6], [1.0, 0.9, 0.7, 0.5], [0.8, 0.7, 0.5, 0.0]],
        dtype=np.float64,
    )
    support = weights > 0.0
    spatial = weights[support]
    means = np.asarray(
        [
            np.sum(truth[:, lead, support] * spatial[None, :], dtype=np.float64)
            / (len(truth) * spatial.sum(dtype=np.float64))
            for lead in range(6)
        ],
        dtype=np.float32,
    )
    return prediction, truth, weights, means


def test_fit_respects_tight_physical_bounds_and_nonnegative_output() -> None:
    prediction, truth, weights, means = _fixture()
    fits = affine.fit_leadwise_affine(
        prediction[:8], truth[:8], weights, means
    )
    calibrated = affine.apply_leadwise_affine(prediction[8:], fits)

    assert len(fits) == 6
    assert calibrated.shape == prediction[8:].shape
    assert np.isfinite(calibrated).all()
    assert np.all(calibrated >= 0.0)
    for lead, fit in enumerate(fits):
        assert fit.lead == lead + 1
        assert affine.SLOPE_BOUNDS[0] <= fit.slope <= affine.SLOPE_BOUNDS[1]
        assert (
            affine.INTERCEPT_SCALE_BOUNDS[0] * means[lead]
            <= fit.intercept_mm_day
            <= 0.0
        )


def test_validation_truth_perturbation_cannot_change_train_fit() -> None:
    prediction, truth, weights, means = _fixture()
    train = np.arange(8)
    validation = np.arange(8, 10)
    first = affine.fit_leadwise_affine(
        prediction[train], truth[train], weights, means
    )
    changed = truth.copy()
    changed[validation] = 1_000_000.0
    second = affine.fit_leadwise_affine(
        prediction[train], changed[train], weights, means
    )

    np.testing.assert_array_equal(
        [fit.slope for fit in first], [fit.slope for fit in second]
    )
    np.testing.assert_array_equal(
        [fit.intercept_mm_day for fit in first],
        [fit.intercept_mm_day for fit in second],
    )


def test_apply_rejects_out_of_order_fits() -> None:
    prediction, truth, weights, means = _fixture()
    fits = affine.fit_leadwise_affine(prediction, truth, weights, means)
    with pytest.raises(ValueError, match="lead order"):
        affine.apply_leadwise_affine(prediction, tuple(reversed(fits)))
