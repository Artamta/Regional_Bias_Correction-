"""Tests for the compact global TP probabilistic residual model."""

import torch

from model import QuestTemporalProbUNet, TPProbUNet


def _random_anchor(batch: int, height: int, width: int) -> torch.Tensor:
    anchor = torch.rand(batch, 2, 5, height, width) + 0.05
    return anchor / anchor.sum(dim=2, keepdim=True)


def _random_multivariable_anchor(
    batch: int, variables: int, height: int, width: int
) -> torch.Tensor:
    anchor = torch.rand(batch, variables, 2, 5, height, width) + 0.05
    return anchor / anchor.sum(dim=3, keepdim=True)


def test_default_parameter_count_and_zero_head() -> None:
    model = TPProbUNet()

    assert sum(parameter.numel() for parameter in model.parameters()) == 111_909
    assert model.period_attention_enabled is False
    assert not hasattr(model, "period_position")
    assert not hasattr(model, "period_attention")
    assert not any("period_attention" in key for key in model.state_dict())
    assert torch.count_nonzero(model.correction_head.weight) == 0
    assert torch.count_nonzero(model.correction_head.bias) == 0


def test_forward_shapes_probability_sum_and_corrections() -> None:
    model = TPProbUNet().eval()
    x = torch.randn(2, 2, 18, 15, 24)
    p0 = _random_anchor(batch=2, height=15, width=24)

    with torch.no_grad():
        probabilities = model(x, p0)
        corrections = model.forward_corrections(x)

    assert probabilities.shape == (2, 2, 5, 15, 24)
    assert corrections.shape == probabilities.shape
    assert torch.count_nonzero(corrections) == 0
    torch.testing.assert_close(
        probabilities.sum(dim=2),
        torch.ones(2, 2, 15, 24),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.all(probabilities > 0.0)


def test_zero_initialized_model_is_anchor_identity() -> None:
    model = TPProbUNet().train()
    x = torch.randn(2, 2, 18, 11, 16)
    p0 = _random_anchor(batch=2, height=11, width=16)

    probabilities = model(x, p0)

    torch.testing.assert_close(probabilities, p0, rtol=1.0e-6, atol=1.0e-7)


def test_gradients_reach_head_anchor_and_features() -> None:
    model = TPProbUNet(dropout=0.0)
    # At the exact identity initialization the zero head intentionally blocks
    # gradients to the backbone. A small nonzero head emulates the second and
    # subsequent optimizer steps and verifies the complete differentiable path.
    with torch.no_grad():
        model.correction_head.weight.normal_(mean=0.0, std=1.0e-3)

    x = torch.randn(1, 2, 18, 8, 12, requires_grad=True)
    p0 = _random_anchor(batch=1, height=8, width=12).requires_grad_()
    probabilities = model(x, p0)
    loss = -torch.log(probabilities[:, :, 0].mean())
    loss.backward()

    assert model.correction_head.weight.grad is not None
    assert torch.isfinite(model.correction_head.weight.grad).all()
    assert torch.count_nonzero(model.correction_head.weight.grad) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert p0.grad is not None and torch.isfinite(p0.grad).all()
    assert torch.count_nonzero(p0.grad) > 0


def test_native_odd_global_grid_size() -> None:
    model = TPProbUNet().eval()
    x = torch.zeros(1, 2, 18, 121, 240)
    p0 = torch.full((1, 2, 5, 121, 240), 0.2)

    with torch.no_grad():
        probabilities = model(x, p0)

    assert probabilities.shape == p0.shape
    torch.testing.assert_close(probabilities, p0, rtol=0.0, atol=1.0e-7)


def test_period_axis_uses_shared_weights() -> None:
    model = TPProbUNet(dropout=0.0).eval()
    with torch.no_grad():
        model.correction_head.weight.normal_(mean=0.0, std=1.0e-3)

    x = torch.randn(1, 2, 18, 12, 16)
    p0 = _random_anchor(batch=1, height=12, width=16)

    with torch.no_grad():
        original = model(x, p0)
        swapped = model(x.flip(1), p0.flip(1))

    torch.testing.assert_close(swapped, original.flip(1), rtol=1.0e-6, atol=1.0e-7)


def test_optional_tp_attention_preserves_identity_and_couples_periods() -> None:
    model = TPProbUNet(
        in_channels=4,
        base_channels=2,
        dropout=0.0,
        period_attention=True,
        attention_heads=2,
        attention_dropout=0.0,
    )
    x = torch.randn(1, 2, 4, 8, 12, requires_grad=True)
    p0 = _random_anchor(batch=1, height=8, width=12)

    identity = model(x, p0)

    assert model.period_attention_enabled is True
    assert hasattr(model, "period_position")
    assert hasattr(model, "period_attention")
    torch.testing.assert_close(identity, p0, rtol=1.0e-6, atol=1.0e-7)

    # The exact zero head intentionally blocks backbone gradients at startup.
    # A small perturbation represents the second optimizer step onward.
    with torch.no_grad():
        model.correction_head.weight.normal_(mean=0.0, std=1.0e-2)
    probabilities = model(x, p0)
    period_zero_loss = -torch.log(probabilities[:, 0, 0].mean())
    period_zero_loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 1]) > 0


def test_optional_tp_attention_validates_head_divisibility_only_when_enabled() -> None:
    # Disabled attention retains the historical constructor behavior.
    TPProbUNet(base_channels=1, period_attention=False, attention_heads=3)

    try:
        TPProbUNet(base_channels=1, period_attention=True, attention_heads=3)
    except ValueError as error:
        assert "divide" in str(error)
    else:  # pragma: no cover - assertion message is clearer than bare failure
        raise AssertionError("invalid attention head count was accepted")


def test_temporal_multivariable_parameter_count_and_zero_heads() -> None:
    model = QuestTemporalProbUNet()

    assert model.in_channels == 38
    assert model.variables == ("pr", "tas", "mslp")
    assert sum(parameter.numel() for parameter in model.parameters()) == 148_559
    for head in model.correction_heads.values():
        assert torch.count_nonzero(head.weight) == 0
        assert torch.count_nonzero(head.bias) == 0


def test_temporal_multivariable_shape_probability_sum_and_anchor_identity() -> None:
    model = QuestTemporalProbUNet(
        base_channels=4,
        dropout=0.2,
        attention_heads=2,
        attention_dropout=0.2,
    ).train()
    x = torch.randn(2, 2, 38, 11, 16)
    p0 = _random_multivariable_anchor(2, 3, 11, 16)

    probabilities = model(x, p0)
    corrections = model.forward_corrections(x)

    assert probabilities.shape == (2, 3, 2, 5, 11, 16)
    assert corrections.shape == probabilities.shape
    assert torch.count_nonzero(corrections) == 0
    torch.testing.assert_close(probabilities, p0, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        probabilities.sum(dim=3),
        torch.ones(2, 3, 2, 11, 16),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_temporal_attention_couples_the_two_period_inputs() -> None:
    model = QuestTemporalProbUNet(
        in_channels=4,
        variables=("pr",),
        base_channels=2,
        dropout=0.0,
        attention_heads=2,
        attention_dropout=0.0,
    )
    # At exact identity the zero head blocks backbone gradients. This small
    # perturbation represents the second optimizer step onward.
    with torch.no_grad():
        model.correction_heads["pr"].weight.normal_(mean=0.0, std=1.0e-2)

    x = torch.randn(1, 2, 4, 8, 12, requires_grad=True)
    p0 = _random_multivariable_anchor(1, 1, 8, 12)
    probabilities = model(x, p0)
    period_zero_loss = -torch.log(probabilities[:, 0, 0, 0].mean())
    period_zero_loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 1]) > 0


def test_temporal_multivariable_native_global_grid() -> None:
    model = QuestTemporalProbUNet(
        base_channels=1,
        dropout=0.0,
        attention_heads=1,
        attention_dropout=0.0,
    ).eval()
    x = torch.zeros(1, 2, 38, 121, 240)
    p0 = torch.full((1, 3, 2, 5, 121, 240), 0.2)

    with torch.no_grad():
        probabilities = model(x, p0)

    assert probabilities.shape == p0.shape
    torch.testing.assert_close(probabilities, p0, rtol=0.0, atol=1.0e-7)


def test_temporal_multivariable_validates_attention_divisibility() -> None:
    try:
        QuestTemporalProbUNet(base_channels=2, attention_heads=3)
    except ValueError as error:
        assert "divide" in str(error)
    else:  # pragma: no cover - assertion message is clearer than bare failure
        raise AssertionError("invalid attention head count was accepted")
