"""Contract tests for the fixed heavy-rain validation screen."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_tail_weight_validation_sweep as sweep  # noqa: E402


def test_tail_grid_is_fixed_before_screening() -> None:
    assert len(sweep.CANDIDATES) == 7
    reference = sweep.CANDIDATE_BY_NAME[sweep.REFERENCE_CONFIGURATION]
    assert reference.heavy_rain_threshold_mm_day is None
    assert reference.heavy_rain_multiplier == 1.0
    tail = [
        candidate
        for candidate in sweep.CANDIDATES
        if candidate.name != sweep.REFERENCE_CONFIGURATION
    ]
    assert {candidate.anchor_kind for candidate in tail} == {
        "log_anchor",
        "physical_recentered",
    }
    assert {candidate.heavy_rain_multiplier for candidate in tail} == {2.0, 3.0, 5.0}
    assert {candidate.heavy_rain_threshold_mm_day for candidate in tail} == {10.0}


def test_tail_candidates_change_only_anchor_and_smooth_l1_weighting() -> None:
    for candidate in sweep.CANDIDATES:
        assert candidate.loss_coefficients == sweep.bias.CURRENT_LOSS
        assert not candidate.uses_bias_scale
    selected = sweep.selected_candidates(
        f"{sweep.REFERENCE_CONFIGURATION},log_anchor_tail_weight_3"
    )
    assert tuple(candidate.name for candidate in selected) == (
        sweep.REFERENCE_CONFIGURATION,
        "log_anchor_tail_weight_3",
    )
