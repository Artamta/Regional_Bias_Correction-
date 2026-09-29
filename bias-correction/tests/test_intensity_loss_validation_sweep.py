"""Contracts for the fixed-anchor intensity-balanced loss screen."""

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

import fuxi_imd_intensity_loss_validation_sweep as sweep  # noqa: E402


def test_candidate_grid_changes_only_loss() -> None:
    assert len(sweep.CANDIDATES) == 5
    assert len(sweep.CANDIDATE_BY_NAME) == len(sweep.CANDIDATES)
    assert {candidate.anchor_kind for candidate in sweep.CANDIDATES} == {
        "physical_recentered"
    }
    assert all(
        np.isclose(candidate.coefficient_sum, 1.0)
        for candidate in sweep.CANDIDATES
    )
    assert sweep.REFERENCE_CONFIGURATION in sweep.CANDIDATE_BY_NAME


def test_intensity_scale_is_training_only_and_regime_specific() -> None:
    # Two cases, one lead, and one cell in every regime.
    truth = np.array(
        [
            [[[0.5, 2.0, 7.0, 20.0]]],
            [[[0.5, 4.0, 9.0, 10.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(truth, dtype=bool)
    weights = np.ones((1, 4), dtype=np.float32)
    dates = np.array(["2002-06-01", "2018-06-01"], dtype="datetime64[D]")

    scale, diagnostics = sweep.fit_training_intensity_bias_scale(
        truth,
        valid,
        weights,
        fit_indices=np.array([0]),
        initializations=dates,
        split_name="train",
    )

    assert scale.shape == (1, 4)
    assert scale[0] == pytest.approx([1.0, 2.0, 7.0, 20.0])
    assert diagnostics.training_cell_case_count.tolist() == [1, 1, 1, 1]


def test_intensity_scale_rejects_validation_year() -> None:
    values = np.ones((1, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="non-training years"):
        sweep.fit_training_intensity_bias_scale(
            values,
            np.ones_like(values, dtype=bool),
            np.ones((1, 1), dtype=np.float32),
            fit_indices=np.array([0]),
            initializations=np.array(["2018-06-01"], dtype="datetime64[D]"),
            split_name="train",
        )


def test_intensity_scale_sanitizes_nan_outside_support() -> None:
    values = np.array([[[[0.5, np.nan, 7.0, 12.0]]]], dtype=np.float32)
    valid = np.array([[[[True, False, True, True]]]])
    # Add one valid light-rain cell so every fixed regime is represented.
    values = np.concatenate(
        (values, np.array([[[[0.5, 3.0, 7.0, 12.0]]]], dtype=np.float32)), axis=0
    )
    valid = np.concatenate((valid, np.ones_like(valid, dtype=bool)), axis=0)
    dates = np.array(["2002-06-01", "2003-06-01"], dtype="datetime64[D]")

    scale, _ = sweep.fit_training_intensity_bias_scale(
        values,
        valid,
        np.ones((1, 4), dtype=np.float32),
        fit_indices=np.array([0, 1]),
        initializations=dates,
        split_name="train",
    )

    assert np.isfinite(scale).all()
    assert scale[0] == pytest.approx([1.0, 3.0, 7.0, 12.0])


def test_named_subset_must_be_known_and_unique() -> None:
    chosen = sweep.selected_candidates(
        f"{sweep.REFERENCE_CONFIGURATION},recentered_anchor_stratified_bias_wet"
    )
    assert [candidate.name for candidate in chosen] == [
        sweep.REFERENCE_CONFIGURATION,
        "recentered_anchor_stratified_bias_wet",
    ]
    with pytest.raises(ValueError, match="unknown"):
        sweep.selected_candidates(f"{sweep.REFERENCE_CONFIGURATION},not_a_loss")
