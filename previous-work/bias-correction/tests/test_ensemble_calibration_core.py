"""Synthetic contracts for the member-preserving FuXi ensemble calibrator."""

from __future__ import annotations

import pytest
import torch

from fuxi_ensemble_calibration_core import (
    EnsembleLocationSpreadCalibrator,
    apply_location_spread_correction,
    ensemble_crps,
    ensemble_rank_histogram,
    random_member_indices,
    subsample_ensemble_members,
    weighted_ensemble_crps,
    weighted_ensemble_crps_loss,
    weighted_spatial_brier_score,
    weighted_spatial_deterministic_metrics,
    weighted_spatial_interval_metrics,
)


def _fields() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(812)
    members = 20.0 * torch.rand(2, 7, 3, 5, 4, generator=generator)
    context = torch.randn(2, 3, 2, 5, 4, generator=generator)
    return members, context


@pytest.mark.parametrize(
    "mode", ("location_spread", "location_only", "summary_only")
)
def test_every_mode_starts_as_an_exact_physical_noop(mode: str) -> None:
    members, context = _fields()
    model = EnsembleLocationSpreadCalibrator(
        context_channels=2,
        member_hidden_channels=4,
        backbone_channels=8,
        mode=mode,
        dropout=0.2,
    )
    model.train()

    output = model(members, context)

    assert torch.equal(output.corrected_members, members)
    assert torch.count_nonzero(output.delta_log_location) == 0
    assert torch.count_nonzero(output.log_spread) == 0
    assert torch.equal(output.spread_factor, torch.ones_like(output.spread_factor))
    assert torch.equal(output.selected_member_indices, torch.arange(7))
    assert output.corrected_members.shape == (2, 7, 3, 5, 4)
    assert output.delta_log_location.shape == (2, 3, 5, 4)


def test_location_only_and_summary_only_are_real_architecture_ablations() -> None:
    location = EnsembleLocationSpreadCalibrator(
        0, member_hidden_channels=3, backbone_channels=5, mode="location_only"
    )
    summary = EnsembleLocationSpreadCalibrator(
        0, member_hidden_channels=3, backbone_channels=5, mode="summary_only"
    )
    assert location.member_encoder is not None
    assert location.parameter_head.out_channels == 1
    assert summary.member_encoder is None
    assert summary.parameter_head.out_channels == 2

    with torch.no_grad():
        location.parameter_head.bias.fill_(0.7)
    members, _ = _fields()
    parameters = location.predict_parameters(members)
    assert torch.equal(parameters.log_spread, torch.zeros_like(parameters.log_spread))
    assert torch.equal(
        parameters.spread_factor, torch.ones_like(parameters.spread_factor)
    )
    assert torch.count_nonzero(parameters.delta_log_location) > 0


def test_cpu_autocast_preserves_member_dtype_and_exact_epoch_zero_identity() -> None:
    members, context = _fields()
    model = EnsembleLocationSpreadCalibrator(2, 4, 8).eval()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(members, context)

    assert output.delta_log_location.dtype == members.dtype
    assert output.log_spread.dtype == members.dtype
    assert output.corrected_members.dtype == members.dtype
    assert torch.equal(output.corrected_members, members)


def test_model_is_member_permutation_invariant_and_output_is_equivariant() -> None:
    members, context = _fields()
    model = EnsembleLocationSpreadCalibrator(
        2, member_hidden_channels=4, backbone_channels=8, dropout=0.0
    ).eval()
    # A zero head would make invariance trivial.  Activate both learned fields
    # so this test exercises the complete DeepSets/backbone path.
    generator = torch.Generator().manual_seed(44)
    with torch.no_grad():
        model.parameter_head.weight.copy_(
            0.1
            * torch.randn(
                model.parameter_head.weight.shape,
                generator=generator,
            )
        )
        model.parameter_head.bias.copy_(torch.tensor([0.2, -0.15]))

    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    inverse = torch.argsort(permutation)
    original = model(members, context)
    permuted = model(members[:, permutation], context)

    assert torch.equal(original.delta_log_location, permuted.delta_log_location)
    assert torch.equal(original.log_spread, permuted.log_spread)
    assert torch.equal(original.spread_factor, permuted.spread_factor)
    assert torch.equal(
        original.corrected_members,
        permuted.corrected_members[:, inverse],
    )


def test_explicit_parameters_can_be_averaged_and_reapplied() -> None:
    members, context = _fields()
    first = EnsembleLocationSpreadCalibrator(2, 3, 6)
    second = EnsembleLocationSpreadCalibrator(2, 3, 6)
    with torch.no_grad():
        first.parameter_head.bias.copy_(torch.tensor([0.1, 0.2]))
        second.parameter_head.bias.copy_(torch.tensor([0.3, -0.2]))
    first_parameters = first.predict_parameters(members, context)
    second_parameters = second.predict_parameters(members, context)
    mean_delta = 0.5 * (
        first_parameters.delta_log_location
        + second_parameters.delta_log_location
    )
    mean_log_spread = 0.5 * (
        first_parameters.log_spread + second_parameters.log_spread
    )

    corrected = apply_location_spread_correction(
        members, mean_delta, log_spread=mean_log_spread
    )

    assert corrected.shape == members.shape
    assert torch.isfinite(corrected).all()
    assert torch.all(corrected >= 0.0)
    assert torch.count_nonzero(corrected != members) > 0


def test_location_spread_transform_is_rank_preserving_before_zero_clipping() -> None:
    members = torch.tensor([1.0, 2.0, 5.0, 9.0]).reshape(1, 4, 1, 1, 1)
    delta = torch.full((1, 1, 1, 1), 0.2)
    log_spread = torch.full((1, 1, 1, 1), -0.3)

    corrected = apply_location_spread_correction(
        members, delta, log_spread=log_spread
    )

    assert torch.all(torch.diff(corrected, dim=1) > 0.0)
    identity = apply_location_spread_correction(
        members, torch.zeros_like(delta), log_spread=torch.zeros_like(log_spread)
    )
    assert torch.equal(identity, members)


def test_random_member_subsampling_is_reproducible_and_keeps_trajectories() -> None:
    members = torch.arange(2 * 9 * 3 * 2 * 2, dtype=torch.float32).reshape(
        2, 9, 3, 2, 2
    )
    first_generator = torch.Generator().manual_seed(29)
    second_generator = torch.Generator().manual_seed(29)
    first, first_indices = subsample_ensemble_members(
        members, 4, generator=first_generator
    )
    second, second_indices = subsample_ensemble_members(
        members, 4, generator=second_generator
    )

    assert torch.equal(first_indices, second_indices)
    assert first_indices.unique().numel() == 4
    assert torch.equal(first, second)
    assert torch.equal(first, members[:, first_indices])
    assert torch.equal(random_member_indices(9, None), torch.arange(9))

    model = EnsembleLocationSpreadCalibrator(0, 3, 6)
    output = model(members, member_indices=first_indices)
    assert output.corrected_members.shape[1] == 4
    assert torch.equal(output.corrected_members, members[:, first_indices])
    with pytest.raises(ValueError, match="either"):
        model(members, member_indices=first_indices, member_subsample_size=4)
    with pytest.raises(ValueError, match="duplicates"):
        subsample_ensemble_members(members, None, member_indices=torch.tensor([1, 1]))


def test_efficient_crps_matches_quadratic_definition_and_has_gradients() -> None:
    generator = torch.Generator().manual_seed(91)
    members = torch.rand(2, 11, 3, 2, 2, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 2, 2, generator=generator)

    actual = ensemble_crps(members, target)
    reliability = torch.mean(torch.abs(members - target[:, None]), dim=1)
    pairwise = 0.5 * torch.mean(
        torch.abs(members[:, :, None] - members[:, None, :]), dim=(1, 2)
    )
    expected = reliability - pairwise

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    actual.mean().backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad).all()


def test_weighted_crps_trains_the_zero_initialized_parameter_head() -> None:
    members, context = _fields()
    truth = 0.7 * members.mean(dim=1) + 0.5
    model = EnsembleLocationSpreadCalibrator(2, 4, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)

    output = model(members, context, member_subsample_size=5)
    loss = weighted_ensemble_crps_loss(
        output.corrected_members, truth, torch.ones(5, 4)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert model.parameter_head.weight.grad is not None
    assert torch.isfinite(model.parameter_head.weight.grad).all()
    assert torch.count_nonzero(model.parameter_head.weight.grad) > 0
    optimizer.step()
    updated = model(members, context)
    assert torch.count_nonzero(updated.delta_log_location) > 0
    assert torch.all(updated.spread_factor > 0.0)


def test_weighted_crps_masks_missing_targets_and_respects_area_weights() -> None:
    members = torch.tensor(
        [
            [
                [[[0.0, 1.0], [2.0, 3.0]]],
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[2.0, 3.0], [4.0, 5.0]]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor([[[[1.0, float("nan")], [3.0, 5.0]]]])
    weights = torch.tensor([[1.0, 50.0], [2.0, 0.0]])
    pointwise = weighted_ensemble_crps(
        members, target, weights, reduction="none"
    )
    loss = weighted_ensemble_crps_loss(members, target, weights)

    assert torch.isnan(pointwise[0, 0, 0, 1])
    expected = (
        pointwise[0, 0, 0, 0] + 2.0 * pointwise[0, 0, 1, 0]
    ) / 3.0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad[:, :, :, 0, 1]) == 0


def test_nonfinite_forecasts_on_valid_support_fail_instead_of_disappearing() -> None:
    members = torch.ones(1, 3, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2)
    weights = torch.ones(2, 2)
    members[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="non-finite members"):
        weighted_ensemble_crps_loss(members, target, weights)
    prediction = torch.ones_like(target)
    prediction[0, 0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="prediction is non-finite"):
        weighted_spatial_deterministic_metrics(prediction, target, weights)


def test_overflowing_location_correction_fails_loudly() -> None:
    members = torch.ones(1, 3, 1, 1, 1)
    delta = torch.full((1, 1, 1, 1), 1000.0)
    with pytest.raises(FloatingPointError, match="produced non-finite rainfall"):
        apply_location_spread_correction(members, delta)


def test_case_lead_deterministic_metrics_keep_block_structure() -> None:
    target = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        ]
    )
    prediction = target + torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
    weights = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    climatology = torch.zeros_like(target)

    scores = weighted_spatial_deterministic_metrics(
        prediction, target, weights, climatology=climatology
    )

    assert scores["rmse"].shape == (1, 2)
    torch.testing.assert_close(scores["rmse"], torch.ones(1, 2))
    torch.testing.assert_close(scores["mae"], torch.ones(1, 2))
    torch.testing.assert_close(
        scores["bias"], torch.full((1, 2), -1.0 / 2.0)
    )
    assert torch.isfinite(scores["acc"]).all()


def test_probabilistic_metric_primitives_return_weekwise_scores() -> None:
    members = torch.tensor(
        [
            [
                [[[0.0, 2.0], [4.0, 6.0]]],
                [[[1.0, 3.0], [5.0, 7.0]]],
                [[[2.0, 4.0], [6.0, 8.0]]],
            ]
        ]
    )
    target = torch.tensor([[[[1.0, 2.5], [5.0, 9.0]]]])
    weights = torch.ones(2, 2)

    brier = weighted_spatial_brier_score(members, target, weights, threshold=5.0)
    interval = weighted_spatial_interval_metrics(
        members, target, weights, coverage=0.8
    )
    histogram = ensemble_rank_histogram(
        members,
        target,
        randomize_ties=False,
    )

    assert brier.shape == (1, 1)
    assert 0.0 <= brier.item() <= 1.0
    assert interval["coverage"].shape == (1, 1)
    assert interval["width"].shape == (1, 1)
    assert interval["lower"].shape == target.shape
    assert histogram.shape == (4,)
    assert histogram.sum().item() == target.numel()


def test_invalid_physical_and_context_contracts_fail_loudly() -> None:
    members, context = _fields()
    model = EnsembleLocationSpreadCalibrator(2, 3, 6)
    with pytest.raises(ValueError, match="nonnegative"):
        model(-members, context)
    with pytest.raises(ValueError, match="context"):
        model(members, context[:, :, :1])
    with pytest.raises(ValueError, match="sample_size"):
        random_member_indices(7, 8)
    with pytest.raises(ValueError, match="shape"):
        ensemble_crps(members, torch.zeros(2, 3, 5, 5))
