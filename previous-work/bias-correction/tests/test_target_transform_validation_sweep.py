"""Contracts for the fixed Box-Cox-1p target-transform screen."""

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

import fuxi_imd_target_transform_validation_sweep as sweep  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    fit_power_target_scale,
    reconstruct_power_precipitation,
    standardize_anchored_target,
    standardize_power_target,
)


def test_candidate_grid_changes_only_target_power() -> None:
    assert len(sweep.CANDIDATES) == 5
    assert [candidate.rain_transform_power for candidate in sweep.CANDIDATES] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert {candidate.anchor_kind for candidate in sweep.CANDIDATES} == {
        "physical_recentered"
    }
    assert all(
        candidate.loss_coefficients == sweep.bias.BIAS_AWARE_LOSS
        for candidate in sweep.CANDIDATES
    )
    assert sweep.REFERENCE_CONFIGURATION == sweep.CANDIDATES[0].name


def test_power_zero_is_exact_legacy_preprocessing_and_reconstruction() -> None:
    truth = np.array([[[[0.0, 2.0], [8.0, 20.0]]]], dtype=np.float32)
    baseline = np.array([[[[0.5, 1.0], [6.0, 12.0]]]], dtype=np.float32)
    weights = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    legacy_scale = fit_anchored_target_scale(
        truth, baseline, weights, split_name="train"
    )
    power_scale = fit_power_target_scale(
        truth,
        baseline,
        weights,
        split_name="train",
        rain_transform_power=0.0,
    )
    assert np.array_equal(power_scale, legacy_scale)
    legacy_target = standardize_anchored_target(truth, baseline, legacy_scale)
    power_target = standardize_power_target(
        truth, baseline, power_scale, rain_transform_power=0.0
    )
    assert np.array_equal(power_target, legacy_target)
    prediction = reconstruct_power_precipitation(
        baseline,
        power_target,
        power_scale,
        rain_transform_power=0.0,
    )
    assert prediction == pytest.approx(truth, rel=2e-6, abs=2e-6)


def test_power_one_is_standardized_physical_residual() -> None:
    truth = np.array([[[[1.0, 5.0]]]], dtype=np.float32)
    baseline = np.array([[[[0.0, 1.0]]]], dtype=np.float32)
    weights = np.ones((1, 2), dtype=np.float32)
    scale = fit_power_target_scale(
        truth,
        baseline,
        weights,
        split_name="train",
        rain_transform_power=1.0,
    )
    assert scale == pytest.approx([np.sqrt((1.0 + 16.0) / 2.0)])
    target = standardize_power_target(
        truth, baseline, scale, rain_transform_power=1.0
    )
    assert target[0, 0, 0] == pytest.approx([1.0 / scale[0], 4.0 / scale[0]])
    prediction = reconstruct_power_precipitation(
        baseline, target, scale, rain_transform_power=1.0
    )
    assert prediction == pytest.approx(truth)


def test_named_subset_requires_exact_log_reference() -> None:
    chosen = sweep.selected_candidates("boxcox_power_000,boxcox_power_050")
    assert [candidate.name for candidate in chosen] == [
        "boxcox_power_000",
        "boxcox_power_050",
    ]
    with pytest.raises(ValueError, match="unknown"):
        sweep.selected_candidates("boxcox_power_000,boxcox_power_999")
