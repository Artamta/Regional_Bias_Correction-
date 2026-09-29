"""Focused correctness tests for the global TP training path."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

import train
from model import TPProbUNet


def _write_explicit_split_cache(path) -> None:
    dates = np.asarray([20120601, 20140601, 20170601, 20190601], dtype=np.int32)
    splits = np.asarray(["train", "train", "validation", "test"], dtype="U10")
    height, width = 17, 24
    features = np.empty((4, 2, 18, height, width), dtype=np.float32)
    for case in range(4):
        features[case] = float(case)
    p0 = np.full((4, 2, 5, height, width), 0.2, dtype=np.float32)
    target = np.zeros((4, 2, height, width), dtype=np.int8)
    provenance = {
        "split_years": {
            "train": [2012, 2014],
            "validation": [2017],
            "test": [2019],
        }
    }
    np.savez_compressed(
        path,
        features=features,
        p0=p0,
        target=target,
        init_dates=dates,
        split=splits,
        latitude=np.linspace(90.0, -90.0, height),
        spatial_support_fraction=np.ones((height, width), dtype=np.float32),
        provenance_json=np.asarray(json.dumps(provenance)),
        cache_contract_sha256=np.asarray("a" * 64),
    )


def test_rps_loss_matches_hand_calculation() -> None:
    # Forecast CDF: [0.1, 0.3, 0.6, 0.8, 1.0]
    # Category-2 observed CDF: [0, 0, 1, 1, 1]
    # RPS = .1^2 + .3^2 + (.6-1)^2 + (.8-1)^2 = 0.30.
    probabilities = torch.tensor(
        [[[[[0.1]], [[0.2]], [[0.3]], [[0.2]], [[0.2]]]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[[[2]]]], dtype=torch.int64)
    weights = torch.ones(1, 1)

    score = train.rps_loss(probabilities, target, weights)

    torch.testing.assert_close(score, torch.tensor(0.30), rtol=0.0, atol=1.0e-7)


def test_rps_loss_masks_invalid_targets() -> None:
    valid_distribution = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
    ignored_distribution = torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025])
    probabilities = torch.stack((valid_distribution, ignored_distribution), dim=-1)
    probabilities = probabilities.reshape(1, 1, 5, 1, 2)
    target = torch.tensor([[[[2, -1]]]], dtype=torch.int64)
    weights = torch.tensor([[1.0, 100.0]])

    score = train.rps_loss(probabilities, target, weights)

    # The invalid second target contributes neither score nor denominator,
    # despite its much larger spatial weight.
    torch.testing.assert_close(score, torch.tensor(0.30), rtol=0.0, atol=1.0e-7)


def test_weighted_modal_quintile_accuracy_is_masked_diagnostic_not_acc() -> None:
    probabilities = torch.zeros(1, 1, 5, 1, 3)
    probabilities[0, 0, 2, 0, 0] = 1.0  # correct, weight 1
    probabilities[0, 0, 4, 0, 1] = 1.0  # incorrect, weight 3
    probabilities[0, 0, 1, 0, 2] = 1.0  # ignored target, weight 100
    target = torch.tensor([[[[2, 0, -1]]]], dtype=torch.int64)
    weights = torch.tensor([[1.0, 3.0, 100.0]])

    score = train.weighted_modal_quintile_accuracy(
        probabilities, target, weights
    )

    torch.testing.assert_close(score, torch.tensor(0.25), rtol=0.0, atol=1.0e-7)


def test_optimizer_excludes_biases_and_norm_parameters_from_decay() -> None:
    model = TPProbUNet()
    optimizer = train.optimizer_for(model, learning_rate=3.0e-4, weight_decay=1.0e-4)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1.0e-4
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    decay_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)

    for name, parameter in model.named_parameters():
        expected = decay_ids if parameter.ndim >= 2 and not name.endswith("bias") else no_decay_ids
        assert id(parameter) in expected, name


def test_spatial_weights_are_nonnegative_at_both_poles() -> None:
    latitude = np.array([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float32)
    land_fraction = np.ones((5, 3), dtype=np.float32)

    weights = train.spatial_weights(latitude, land_fraction)

    assert weights.shape == (5, 3)
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights[2], torch.ones(3))


def test_explicit_npz_splits_are_authoritative_and_leak_free(tmp_path) -> None:
    cache = tmp_path / "arbitrary_blocked_years.npz"
    _write_explicit_split_cache(cache)

    training = train.PreparedCases(cache, split="train")
    validation = train.PreparedCases(cache, split="validation")

    assert training.indices.tolist() == [0, 1]
    assert validation.indices.tolist() == [2]
    assert training.selected_years == (2012, 2014)
    assert validation.selected_years == (2017,)
    assert train.training_year_partitions(training, validation) == {
        "train": [2012, 2014],
        "validation": [2017],
        "test": [2019],
    }
    assert float(training[0][0].mean()) == 0.0
    assert float(training[1][0].mean()) == 1.0
    assert float(validation[0][0].mean()) == 2.0
    assert training.metadata()["cache_contract_sha256"] == "a" * 64

    # Exact-year inference remains available for older callers, but a partial
    # or contradictory year selection cannot bypass an explicit split label.
    assert train.PreparedCases(cache, (2012, 2014)).split_name == "train"
    with pytest.raises(ValueError, match="do not exactly match"):
        train.PreparedCases(cache, (2012,))
    with pytest.raises(ValueError, match="disagree"):
        train.PreparedCases(cache, (2017,), split="train")

    leaking_cache = tmp_path / "same_year_leakage.npz"
    with np.load(cache, allow_pickle=False) as source:
        leaking_payload = {name: source[name] for name in source.files}
    leaking_payload["init_dates"] = np.asarray(
        [20120601, 20140601, 20140608, 20190601], dtype=np.int32
    )
    np.savez_compressed(leaking_cache, **leaking_payload)
    with pytest.raises(ValueError, match="assigned to multiple NPZ splits"):
        train.PreparedCases(leaking_cache, split="train")


def test_training_checkpoint_records_actual_npz_split_years_and_support_key(
    tmp_path, monkeypatch
) -> None:
    cache = tmp_path / "arbitrary_blocked_years.npz"
    run_directory = tmp_path / "explicit_split_run"
    _write_explicit_split_cache(cache)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--cache",
            str(cache),
            "--device",
            "cpu",
            "--run-dir",
            str(run_directory),
            "--max-epochs",
            "0",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--seed",
            "5",
        ],
    )

    train.main()

    checkpoint = torch.load(
        run_directory / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["metadata"]["train_years"] == [2012, 2014]
    assert checkpoint["metadata"]["validation_years"] == [2017]
    assert checkpoint["metadata"]["test_years"] == [2019]
    assert checkpoint["normalization"]["selected_split"] == "train"
    assert checkpoint["normalization"]["cache_contract_sha256"] == "a" * 64


def test_legacy_npz_keeps_year_selection_and_land_fraction_fallback(tmp_path) -> None:
    explicit = tmp_path / "explicit.npz"
    legacy = tmp_path / "legacy.npz"
    _write_explicit_split_cache(explicit)
    with np.load(explicit, allow_pickle=False) as source:
        payload = {
            name: source[name]
            for name in source.files
            if name not in {"split", "provenance_json", "spatial_support_fraction"}
        }
        payload["land_fraction"] = np.ones((17, 24), dtype=np.float32)
        np.savez_compressed(legacy, **payload)

    selected = train.PreparedCases(legacy, (2017,))

    assert selected.indices.tolist() == [2]
    np.testing.assert_array_equal(
        train.prepared_spatial_fraction(selected._npz),
        np.ones((17, 24), dtype=np.float32),
    )


def test_one_epoch_cpu_smoke_writes_complete_run(tmp_path, monkeypatch) -> None:
    run_directory = tmp_path / "smoke_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--smoke",
            "--device",
            "cpu",
            "--run-dir",
            str(run_directory),
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--seed",
            "7",
        ],
    )

    train.main()

    checkpoint_path = run_directory / "best.pt"
    assert checkpoint_path.is_file()
    assert (run_directory / "history.csv").is_file()
    assert (run_directory / "figures" / "loss_curve.png").is_file()
    config_path = run_directory / "config.json"
    assert config_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model_state", "model_config", "normalization", "metadata"}
    assert checkpoint["model_config"] == {
        "in_channels": 18,
        "base_channels": 16,
        "dropout": 0.1,
    }
    assert checkpoint["metadata"]["best_epoch"] == 1
    assert checkpoint["metadata"]["model_name"] == "TPProbUNet"
    assert checkpoint["metadata"]["period_attention_enabled"] is False
    assert checkpoint["metadata"]["fuxi_competition_use"] == "written_permission_required"

    history_header = (run_directory / "history.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "train_weighted_modal_quintile_accuracy_diagnostic_not_acc" in history_header
    assert (
        "validation_weighted_modal_quintile_accuracy_diagnostic_not_acc"
        in history_header
    )

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["command"]["smoke"] is True
    assert saved_config["command"]["device"] == "cpu"
    assert saved_config["result"]["best_epoch"] == 1


def test_attention_smoke_preserves_epoch_zero_anchor_checkpoint(
    tmp_path, monkeypatch
) -> None:
    run_directory = tmp_path / "attention_anchor_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--smoke",
            "--device",
            "cpu",
            "--run-dir",
            str(run_directory),
            "--max-epochs",
            "0",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--seed",
            "11",
            "--period-attention",
            "--attention-heads",
            "4",
            "--attention-dropout",
            "0.0",
        ],
    )

    train.main()

    checkpoint = torch.load(
        run_directory / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["model_config"] == {
        "in_channels": 18,
        "base_channels": 16,
        "dropout": 0.1,
        "period_attention": True,
        "attention_heads": 4,
        "attention_dropout": 0.0,
    }
    assert checkpoint["metadata"]["best_epoch"] == 0
    assert checkpoint["metadata"]["selected_system"] == "raw_p0"
    assert checkpoint["metadata"]["period_attention_enabled"] is True
    assert "period_position" in checkpoint["model_state"]
    assert torch.count_nonzero(
        checkpoint["model_state"]["correction_head.weight"]
    ) == 0
