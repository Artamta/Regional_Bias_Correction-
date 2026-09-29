"""Focused contracts for the V2 deterministic/probabilistic loss ablation."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

import fuxi_allseason_hybrid_loss_ablation as ablation
from fuxi_ensemble_calibration_core import (
    EnsembleLocationSpreadCalibrator,
    weighted_ensemble_crps,
)


CANONICAL_ALPHAS = (0.0, 0.10, 0.25, 0.50, 1.0)
CANONICAL_SEEDS = (42, 43, 44)


def _profile(alpha: float) -> ablation.LossProfile:
    matches = [
        profile
        for profile in ablation.LOSS_PROFILES
        if profile.alpha_mse == pytest.approx(alpha)
    ]
    assert len(matches) == 1, f"expected one loss profile for alpha={alpha}"
    return matches[0]


def _validation_summary(
    overrides: dict[float, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Return complete three-seed validation scores for every frozen profile."""

    defaults = {
        0.00: {
            "crps": 1.000,
            "rmse": 2.000,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
        },
        0.10: {
            "crps": 1.004,
            "rmse": 1.900,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
        },
        0.25: {
            "crps": 0.995,
            "rmse": 1.930,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
        },
        0.50: {
            "crps": 1.006,
            "rmse": 1.700,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
        },
        # The deterministic endpoint is a diagnostic and must never be selected.
        1.00: {
            "crps": 0.800,
            "rmse": 1.000,
            "coverage_50": 0.50,
            "coverage_80": 0.80,
            "coverage_90": 0.90,
        },
    }
    for alpha, values in (overrides or {}).items():
        defaults[alpha].update(values)

    rows = []
    for profile in ablation.LOSS_PROFILES:
        for seed in CANONICAL_SEEDS:
            values = defaults[profile.alpha_mse]
            coverage_error = sum(
                abs(values[f"coverage_{level}"] - nominal)
                for level, nominal in ((50, 0.50), (80, 0.80), (90, 0.90))
            ) / 3.0
            rows.append(
                {
                    "split": "validation",
                    "profile": profile.name,
                    "alpha_mse": profile.alpha_mse,
                    "seed": seed,
                    **values,
                    "coverage_error": coverage_error,
                }
            )
    return pd.DataFrame(rows)


def test_loss_profiles_are_the_frozen_one_factor_ablation() -> None:
    assert isinstance(ablation.LOSS_PROFILES, tuple)
    assert tuple(profile.alpha_mse for profile in ablation.LOSS_PROFILES) == CANONICAL_ALPHAS
    assert len({profile.name for profile in ablation.LOSS_PROFILES}) == len(
        ablation.LOSS_PROFILES
    )
    assert all(isinstance(profile, ablation.LossProfile) for profile in ablation.LOSS_PROFILES)
    assert _profile(0.0).selectable
    assert not _profile(1.0).selectable


def test_weighted_ensemble_mean_mse_masks_nan_and_respects_area_weights() -> None:
    members = torch.tensor(
        [
            [
                [[[0.0, 100.0], [2.0, 999.0]]],
                [[[2.0, 200.0], [4.0, 999.0]]],
                [[[4.0, 300.0], [6.0, 999.0]]],
            ]
        ],
        requires_grad=True,
    )
    truth = torch.tensor([[[[1.0, float("nan")], [2.0, 0.0]]]])
    weights = torch.tensor([[1.0, 50.0], [2.0, 0.0]])

    loss = ablation.weighted_ensemble_mean_mse(members, truth, weights)

    # Ensemble means on the two valid cells are 2 and 4, so
    # (1 * (2-1)^2 + 2 * (4-2)^2) / (1 + 2) == 3.
    torch.testing.assert_close(loss, torch.tensor(3.0))
    loss.backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad[..., 0, 1]) == 0
    assert torch.count_nonzero(members.grad[..., 1, 1]) == 0


def test_weighted_ensemble_mean_mse_rejects_nonfinite_valid_forecasts() -> None:
    members = torch.ones(1, 3, 1, 2, 2)
    truth = torch.ones(1, 1, 2, 2)
    weights = torch.ones(2, 2)
    members[0, 0, 0, 0, 0] = float("inf")

    with pytest.raises(FloatingPointError, match="non-finite"):
        ablation.weighted_ensemble_mean_mse(members, truth, weights)


def test_validation_rmse_matches_mean_reported_case_lead_rmse_not_pooled_rms() -> None:
    members = torch.tensor([1.0, 3.0]).reshape(2, 1, 1, 1, 1)
    truth = torch.zeros(2, 1, 1, 1)
    weights = torch.ones(1, 1)

    reported = ablation.weighted_mean_case_lead_rmse(members, truth, weights)
    pooled = torch.sqrt(
        ablation.weighted_ensemble_mean_mse(members, truth, weights)
    )

    # evaluate_ensemble reports RMSE=1 and RMSE=3, whose arithmetic mean is 2.
    torch.testing.assert_close(reported, torch.tensor(2.0))
    torch.testing.assert_close(pooled, torch.sqrt(torch.tensor(5.0)))
    assert reported != pooled


def test_alpha_zero_is_exact_crps_in_value_and_gradient() -> None:
    generator = torch.Generator().manual_seed(811)
    actual_members = torch.rand(
        2, 7, 3, 4, 3, generator=generator, requires_grad=True
    )
    expected_members = actual_members.detach().clone().requires_grad_(True)
    truth = torch.rand(2, 3, 4, 3, generator=generator)
    weights = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)

    components = ablation.hybrid_loss(
        actual_members,
        truth,
        weights,
        _profile(0.0),
        raw_train_crps=2.0,
        raw_train_mse=8.0,
    )
    expected = weighted_ensemble_crps(expected_members, truth, weights)
    (actual_gradient,) = torch.autograd.grad(components.total, actual_members)
    (expected_gradient,) = torch.autograd.grad(expected, expected_members)

    torch.testing.assert_close(components.total, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(components.crps, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)


def test_alpha_one_is_the_train_normalized_ensemble_mean_mse() -> None:
    members = torch.tensor([0.0, 2.0, 4.0]).reshape(1, 3, 1, 1, 1)
    truth = torch.tensor([1.0]).reshape(1, 1, 1, 1)
    weights = torch.ones(1, 1)

    components = ablation.hybrid_loss(
        members,
        truth,
        weights,
        _profile(1.0),
        raw_train_crps=2.0,
        raw_train_mse=8.0,
    )

    expected_mse = torch.tensor(1.0)
    expected_scaled_mse = torch.tensor(0.25)
    torch.testing.assert_close(components.mse, expected_mse)
    torch.testing.assert_close(components.scaled_mse, expected_scaled_mse)
    torch.testing.assert_close(
        components.total, expected_scaled_mse, rtol=0.0, atol=0.0
    )


def test_hybrid_loss_is_the_declared_convex_combination() -> None:
    generator = torch.Generator().manual_seed(918)
    members = torch.rand(2, 5, 2, 3, 4, generator=generator)
    truth = torch.rand(2, 2, 3, 4, generator=generator)
    weights = torch.linspace(0.5, 2.0, 12).reshape(3, 4)
    profile = _profile(0.25)

    components = ablation.hybrid_loss(
        members,
        truth,
        weights,
        profile,
        raw_train_crps=1.5,
        raw_train_mse=3.0,
    )

    torch.testing.assert_close(
        components.total,
        (1.0 - profile.alpha_mse) * components.crps
        + profile.alpha_mse * components.scaled_mse,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(components.scaled_mse, 0.5 * components.mse)


def test_hybrid_loss_backpropagates_into_location_and_spread_heads() -> None:
    generator = torch.Generator().manual_seed(314)
    members = 0.25 + 8.0 * torch.rand(2, 7, 3, 4, 4, generator=generator)
    context = torch.randn(2, 3, 2, 4, 4, generator=generator)
    truth = 0.4 + 0.65 * members.mean(dim=1)
    model = EnsembleLocationSpreadCalibrator(
        context_channels=2,
        member_hidden_channels=4,
        backbone_channels=8,
        mode="location_spread",
    )

    output = model(members, context)
    components = ablation.hybrid_loss(
        output.corrected_members,
        truth,
        torch.ones(4, 4),
        _profile(0.25),
        raw_train_crps=1.0,
        raw_train_mse=2.0,
    )
    components.total.backward()

    gradient = model.parameter_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[0]) > 0, "location head received no gradient"
    assert torch.count_nonzero(gradient[1]) > 0, "spread head received no gradient"


def test_selection_uses_only_validation_and_applies_probabilistic_guards() -> None:
    result = ablation.select_loss_profile(_validation_summary())

    assert result["selected_profile"] == _profile(0.10).name
    assert result["test_metrics_consulted"] is False
    assert len(result["validation_profiles"]) == len(ablation.LOSS_PROFILES)
    assert all("eligible_hybrid" in row for row in result["validation_profiles"])

    contaminated = _validation_summary()
    contaminated.loc[0, "split"] = "test_development"
    with pytest.raises(ValueError, match="validation"):
        ablation.select_loss_profile(contaminated)


def test_selection_falls_back_to_crps_when_no_hybrid_passes() -> None:
    summary = _validation_summary(
        {
            0.10: {"crps": 1.006},
            0.25: {"coverage_80": 0.76},
            0.50: {"crps": 1.020},
        }
    )

    result = ablation.select_loss_profile(summary)

    assert result["selected_profile"] == _profile(0.0).name


def test_selection_never_chooses_a_hybrid_rmse_dominated_by_control() -> None:
    summary = _validation_summary(
        {
            0.10: {"rmse": 2.010},
            0.25: {"rmse": 2.020},
            0.50: {"crps": 1.004, "rmse": 2.030},
        }
    )

    result = ablation.select_loss_profile(summary)

    assert result["selected_profile"] == _profile(0.0).name
    control = next(
        row
        for row in result["validation_profiles"]
        if row["profile"] == _profile(0.0).name
    )
    assert control["eligible_joint_candidate"] is True
    assert result["rules"]["crps_control_competes_in_rmse_selection"] is True


def test_selection_prefers_control_inside_practical_rmse_tie() -> None:
    summary = _validation_summary(
        {
            # This is 0.2% lower than the control RMSE=2.0, inside the frozen
            # 0.25% practical-tie threshold, so the simpler alpha=0 arm wins.
            0.10: {"rmse": 1.996},
            0.25: {"rmse": 2.020},
            0.50: {"crps": 1.004, "rmse": 2.030},
        }
    )

    result = ablation.select_loss_profile(summary)

    assert result["selected_profile"] == _profile(0.0).name


def test_selection_requires_all_three_optimization_seeds() -> None:
    incomplete = _validation_summary().loc[lambda frame: frame.seed != 44]

    with pytest.raises(ValueError, match="seed"):
        ablation.select_loss_profile(incomplete)


def test_selection_prefers_smaller_alpha_inside_rmse_tolerance() -> None:
    summary = _validation_summary(
        {
            0.10: {"rmse": 1.900},
            0.25: {"rmse": 1.898},
        }
    )

    result = ablation.select_loss_profile(summary)

    assert result["selected_profile"] == _profile(0.10).name


def test_validate_args_accepts_only_canonical_profiles_and_seeds() -> None:
    parser = ablation.build_parser()
    canonical_profiles = ",".join(profile.name for profile in ablation.LOSS_PROFILES)

    full = parser.parse_args([])
    ablation.validate_args(full)
    assert full.profiles == canonical_profiles
    assert full.seeds == "42,43,44"

    smoke = parser.parse_args(["--smoke"])
    ablation.validate_args(smoke)
    assert smoke.profiles == canonical_profiles
    assert smoke.seeds == "42"

    subset = parser.parse_args(["--profiles", _profile(0.0).name])
    with pytest.raises(ValueError, match="profiles"):
        ablation.validate_args(subset)

    wrong_full_seeds = parser.parse_args(["--seeds", "42"])
    with pytest.raises(ValueError, match="seeds"):
        ablation.validate_args(wrong_full_seeds)

    wrong_smoke_seeds = parser.parse_args(["--smoke", "--seeds", "42,43,44"])
    with pytest.raises(ValueError, match="seeds"):
        ablation.validate_args(wrong_smoke_seeds)
