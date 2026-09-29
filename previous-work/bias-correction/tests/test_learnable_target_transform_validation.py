"""Contracts for the learnable target-transform validation driver."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_learnable_target_transform_validation as sweep  # noqa: E402


def test_three_arms_share_contract_and_only_one_learns() -> None:
    assert [candidate.name for candidate in sweep.CANDIDATES] == [
        sweep.REFERENCE_CONFIGURATION,
        "fixed_power_025_control",
        "learned_global_power_000_050",
    ]
    assert [candidate.learnable for candidate in sweep.CANDIDATES] == [
        False,
        False,
        True,
    ]
    assert [candidate.initial_power for candidate in sweep.CANDIDATES] == [
        0.0,
        0.25,
        0.25,
    ]
    assert {candidate.anchor_kind for candidate in sweep.CANDIDATES} == {
        "physical_recentered"
    }
    assert all(
        candidate.loss_coefficients == sweep.bias.BIAS_AWARE_LOSS
        for candidate in sweep.CANDIDATES
    )
    assert sweep.CHECKPOINT_SELECTION_METRIC == "validation_physical_rmse"
    assert sweep.TRANSFORM_LEARNING_RATE_RATIO == 0.1
    assert sweep.TRANSFORM_WARMUP_EPOCHS == 5


def _intensity_frame(candidate_biases: dict[str, float]) -> pd.DataFrame:
    reference_biases = {
        "dry_lt1": 0.2,
        "light_1_5": -0.4,
        "moderate_5_10": -1.0,
        "heavy_ge10": -5.0,
    }
    rows = []
    for configuration, biases in (
        (sweep.REFERENCE_CONFIGURATION, reference_biases),
        ("learned_global_power_000_050", candidate_biases),
    ):
        for stratum, bias_value in biases.items():
            heavy = stratum == "heavy_ge10"
            rows.append(
                {
                    "configuration": configuration,
                    "lead": "ALL_WEEKS",
                    "stratum": stratum,
                    "bias": bias_value,
                    "rmse": (10.0 if heavy else 2.0)
                    * (0.95 if configuration.startswith("learned") else 1.0),
                    "mae": (8.0 if heavy else 1.5)
                    * (0.95 if configuration.startswith("learned") else 1.0),
                }
            )
    return pd.DataFrame(rows)


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "configuration": sweep.REFERENCE_CONFIGURATION,
                "pooled_rmse": 5.5,
                "pooled_abs_bias": 0.8,
                "qualifies": False,
                "all_lead_abs_bias_not_worse": True,
            },
            {
                "rank": 2,
                "configuration": "learned_global_power_000_050",
                "pooled_rmse": 5.4,
                "pooled_abs_bias": 0.2,
                "qualifies": True,
                "all_lead_abs_bias_not_worse": True,
            },
        ]
    )


def test_intensity_guard_rejects_pooled_bias_cancellation() -> None:
    # Pooled bias could look better while dry/light biases are worse.  The
    # equal-stratum and all-strata guards must reject that candidate.
    intensity = _intensity_frame(
        {
            "dry_lt1": 0.5,
            "light_1_5": 0.8,
            "moderate_5_10": -0.5,
            "heavy_ge10": -4.0,
        }
    )

    ranked, guards = sweep.apply_stratified_qualification_guards(
        _ranking(), intensity
    )
    learned = guards.loc[
        guards.configuration.eq("learned_global_power_000_050")
    ].iloc[0]
    learned_rank = ranked.loc[
        ranked.configuration.eq("learned_global_power_000_050")
    ].iloc[0]

    assert not bool(learned.all_intensity_abs_bias_not_worse)
    assert learned.candidate_equal_stratum_mean_abs_bias > 0.0
    assert learned.equal_stratum_mean_abs_bias_delta != 0.0
    assert not bool(learned.passes_all_intensity_guards)
    assert not bool(learned_rank.qualifies)


def test_learned_power_boundary_guard_rejects_saturation() -> None:
    ranking = _ranking()
    records = pd.DataFrame(
        [
            {
                "configuration": sweep.REFERENCE_CONFIGURATION,
                "best_power": 0.0,
            },
            {
                "configuration": "learned_global_power_000_050",
                "best_power": 0.495,
            },
        ]
    )
    candidates = (
        sweep.CANDIDATE_BY_NAME[sweep.REFERENCE_CONFIGURATION],
        sweep.CANDIDATE_BY_NAME["learned_global_power_000_050"],
    )

    ranked, guards = sweep.apply_learned_power_boundary_guard(
        ranking, records, candidates
    )
    learned = guards.loc[
        guards.configuration.eq("learned_global_power_000_050")
    ].iloc[0]

    assert not bool(learned.learned_power_away_from_bounds)
    assert not bool(
        ranked.loc[
            ranked.configuration.eq("learned_global_power_000_050"), "qualifies"
        ].iloc[0]
    )
