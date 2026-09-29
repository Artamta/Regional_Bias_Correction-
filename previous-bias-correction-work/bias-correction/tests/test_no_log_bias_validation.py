from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fuxi_imd_attention_climatology as experiment
import fuxi_imd_no_log_bias_validation as no_log_bias
from fuxi_adapter.anchored import reconstruct_anchored_precipitation


def test_raw_training_baseline_is_distinct_from_reporting_log_bias() -> None:
    raw = np.full((2, 6, 2, 2), 3.0, dtype=np.float32)
    corrected = np.full_like(raw, 2.0)

    selected = experiment.select_training_baseline("raw_fuxi", raw, corrected)

    assert np.array_equal(selected, raw)
    assert not np.shares_memory(selected, raw)
    assert not np.array_equal(selected, corrected)
    assert np.array_equal(
        experiment.select_training_baseline("log_bias", raw, corrected), corrected
    )


def test_invalid_training_baseline_contract_is_rejected() -> None:
    raw = np.ones((2, 6, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown training anchor"):
        experiment.select_training_baseline("none", raw, raw)
    with pytest.raises(experiment.base.DataContractError, match="matching"):
        experiment.select_training_baseline("raw_fuxi", raw, raw[:, :, :, :1])
    invalid = raw.copy()
    invalid[0, 0, 0, 0] = -1.0
    with pytest.raises(experiment.base.DataContractError, match="nonnegative"):
        experiment.select_training_baseline("raw_fuxi", invalid, raw)


def test_zero_neural_residual_reconstructs_raw_fuxi_exactly() -> None:
    raw = np.linspace(0.0, 12.0, 48, dtype=np.float32).reshape(2, 6, 2, 2)
    rebuilt = reconstruct_anchored_precipitation(
        raw,
        np.zeros_like(raw),
        np.ones(6, dtype=np.float32),
        valid_mask=np.ones_like(raw, dtype=bool),
    )
    assert np.array_equal(rebuilt, raw)


def test_year_scores_name_the_raw_reference_without_log_bias_aliasing() -> None:
    raw = np.full((4, 6, 2, 2), 2.0, dtype=np.float32)
    truth = np.full_like(raw, 1.0)
    target_scale = np.ones(6, dtype=np.float32)
    exact_residual = np.full_like(raw, np.log1p(1.0) - np.log1p(2.0))
    selection = pd.DataFrame([{"candidate": experiment.NORMAL.name, "alpha": 1.0}])
    initializations = np.asarray(
        ["2018-06-01", "2018-07-01", "2019-06-01", "2019-07-01"],
        dtype="datetime64[D]",
    )

    scored = experiment.engine.add_year_validation_scores(
        selection,
        {experiment.NORMAL.name: exact_residual},
        np.arange(4),
        raw,
        SimpleNamespace(weekly_truth=truth),
        target_scale,
        np.ones((2, 2), dtype=np.float64),
        initializations,
        active_leads=tuple(range(6)),
        baseline_name="raw_fuxi",
    )

    assert "raw_fuxi_rmse_2018" in scored
    assert "rmse_skill_vs_raw_fuxi_2019_pct" in scored
    assert not any("log_bias" in column for column in scored.columns)
    assert bool(scored.loc[0, "improves_every_validation_year"])


def test_matched_ablation_does_not_promote_secondary_attention_screen() -> None:
    selection = pd.DataFrame(
        [
            {
                "candidate": experiment.NORMAL.name,
                "alpha": 0.8,
                "improves_every_validation_year": True,
                "attention_beats_normal_every_year": False,
            },
            {
                "candidate": experiment.ATTENTION.name,
                "alpha": 1.0,
                "improves_every_validation_year": True,
                "attention_beats_normal_every_year": True,
            },
        ]
    )
    selected, alpha, reason = experiment.choose_model(
        selection,
        baseline_method="raw_fuxi",
        baseline_label="raw FuXi",
        matched_normal_only=True,
    )
    assert selected == experiment.NORMAL.name
    assert alpha == pytest.approx(0.8)
    assert "matched-architecture" in reason


def test_dedicated_entrypoint_fixes_raw_identity_contract() -> None:
    assert no_log_bias.FIXED_ARGUMENTS == (
        "--all-weeks",
        "--full-fuxi-context",
        "--training-anchor",
        "raw_fuxi",
    )
    assert "--training-anchor" in no_log_bias.FORBIDDEN_ARGUMENTS
