"""Contracts for restart-safe aligned FuXi bridge inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import india_s2s_probabilistic_bridge as bridge
import india_s2s_probabilistic_bridge_run as runner
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator


def _context_inputs() -> runner.ContextInputs:
    daily = np.empty((366, 27, 27), dtype=np.float32)
    for index in range(366):
        daily[index] = np.float32(index / 100.0)
    return runner.ContextInputs(
        daily_climatology=daily,
        latitude=np.linspace(39.0, 0.0, 27, dtype=np.float64),
        longitude=np.linspace(60.0, 99.0, 27, dtype=np.float64),
        support=np.ones((27, 27), dtype=bool),
        observation_fraction=np.ones((27, 27), dtype=np.float32),
        weights=np.ones((27, 27), dtype=np.float64),
        normalization_mean=np.arange(6, dtype=np.float32) / 10.0,
        normalization_std=np.arange(1, 7, dtype=np.float32),
    )


def test_frozen_context_uses_plus_one_windows_and_parent_normalization() -> None:
    inputs = _context_inputs()
    starts = np.asarray(["2024-01-01"], dtype="datetime64[D]")

    result = runner.build_frozen_context(starts, inputs)

    valid_dates = bridge.target_valid_dates(starts)
    positions = runner.calibration.calendar_positions(valid_dates)
    weekly = inputs.daily_climatology[positions].mean(axis=2)
    expected = (np.log1p(weekly) - inputs.normalization_mean[None, :, None, None]) / inputs.normalization_std[
        None, :, None, None
    ]
    assert result.normalized_climatology.shape == (1, 6, 27, 27)
    assert np.allclose(result.normalized_climatology, expected)
    assert valid_dates[0, 0, 0] == np.datetime64("2024-01-02")
    assert valid_dates[0, -1, -1] == np.datetime64("2024-02-12")
    midpoint_day = int(
        runner.pd.Timestamp(valid_dates[0, 0, 3]).dayofyear - 1
    )
    assert np.isclose(result.season_sin[0, 0], np.sin(2.0 * np.pi * midpoint_day / 365.2425))


def test_operational_50_members_are_exactly_reconstructible() -> None:
    rng = np.random.default_rng(9)
    raw = rng.uniform(0.0, 20.0, size=(2, 50, 6, 27, 27)).astype(np.float32)
    delta = np.zeros((2, 6, 27, 27), dtype=np.float32)
    log_spread = np.zeros_like(delta)

    corrected = runner.reconstruct_corrected_members(raw, delta, log_spread)

    assert corrected.shape == raw.shape
    assert np.allclose(corrected, raw, rtol=2.0e-6, atol=2.0e-6)
    with pytest.raises(bridge.BridgeContractError, match="50"):
        runner.reconstruct_corrected_members(raw[:, :49], delta, log_spread)


def test_model_inference_accepts_50_members_and_writes_only_two_fields() -> None:
    model = EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=8,
        backbone_channels=24,
        mode="location_spread",
        dropout=0.05,
        max_abs_log_spread=2.0,
    ).eval()
    raw = np.zeros((1, 50, 6, 27, 27), dtype=np.float32)
    context = np.zeros((1, 6, 7, 27, 27), dtype=np.float32)

    delta, log_spread = runner.infer_adjustments(
        model,
        raw,
        context,
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert delta.shape == (1, 6, 27, 27)
    assert log_spread.shape == delta.shape
    assert np.array_equal(delta, np.zeros_like(delta))
    assert np.array_equal(log_spread, np.zeros_like(log_spread))


def test_bounded_selection_is_deterministic_and_spans_archive() -> None:
    dates = np.datetime64("2020-01-01") + np.arange(517).astype("timedelta64[D]")

    smoke = runner.select_initializations(dates, smoke=True, max_cases=None)
    bounded = runner.select_initializations(dates, smoke=False, max_cases=17)

    assert smoke.size == runner.SMOKE_CASE_COUNT
    assert smoke[0] == dates[0]
    assert smoke[-1] == dates[-1]
    assert bounded.size == 17
    assert bounded[0] == dates[0]
    assert bounded[-1] == dates[-1]
    with pytest.raises(ValueError, match="either"):
        runner.select_initializations(dates, smoke=True, max_cases=2)
    with pytest.raises(ValueError, match="1..517"):
        runner.select_initializations(dates, smoke=False, max_cases=518)


def _shard_arrays(initializations: np.ndarray) -> dict[str, np.ndarray]:
    count = initializations.size
    fields = np.zeros((count, 6, 27, 27), dtype=np.float32)
    return {
        "initializations": initializations,
        "scoreable_for_truth": bridge.scoring_mask(initializations),
        "delta_log_location": fields.copy(),
        "log_spread": fields.copy(),
        "raw_ensemble_mean": fields.copy(),
        "corrected_ensemble_mean": fields.copy(),
        "latitude": np.linspace(39.0, 0.0, 27),
        "longitude": np.linspace(60.0, 99.0, 27),
        "member_count": np.asarray(50, dtype=np.int16),
        "selected_seed": np.asarray(43, dtype=np.int16),
    }


def test_shard_is_compact_hash_bound_and_carries_exact_truth_firewall(
    tmp_path: Path,
) -> None:
    dates = np.asarray(["2024-11-18", "2024-12-30"], dtype="datetime64[D]")
    shard = tmp_path / "adjustments_2024.npz"
    runner._atomic_savez(shard, **_shard_arrays(dates))
    checkpoint_hash = "a" * 64
    receipt = {
        "year": 2024,
        "artifact_sha256": bridge.sha256_file(shard),
        "checkpoint_sha256": checkpoint_hash,
        "case_count": 2,
        "scoreable_case_count": 1,
        "selected_seed": 43,
        "member_count": 50,
        "raw_source": {
            "selected_initialization_dates_sha256": bridge.initialization_dates_sha256(
                dates
            ),
            "raw_member_shape": [2, 50, 6, 27, 27],
            "raw_weekly_members_sha256": "b" * 64,
            "selection_indices": [87, 99],
        },
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }

    runner.validate_adjustment_shard(
        shard,
        receipt,
        expected_initializations=dates,
        expected_latitude=np.linspace(39.0, 0.0, 27),
        expected_longitude=np.linspace(60.0, 99.0, 27),
        checkpoint_sha256=checkpoint_hash,
    )

    with np.load(shard, allow_pickle=False) as archive:
        assert set(archive.files) == runner.ALLOWED_SHARD_KEYS
        assert archive["scoreable_for_truth"].tolist() == [True, False]
        assert not any(
            token in key.lower()
            for key in set(archive.files) - {"scoreable_for_truth"}
            for token in ("truth", "target", "observation")
        )
    bad_receipt = dict(receipt, artifact_sha256="0" * 64)
    with pytest.raises(bridge.BridgeContractError, match="hash differs"):
        runner.validate_adjustment_shard(
            shard,
            bad_receipt,
            expected_initializations=dates,
            expected_latitude=np.linspace(39.0, 0.0, 27),
            expected_longitude=np.linspace(60.0, 99.0, 27),
            checkpoint_sha256=checkpoint_hash,
        )


def test_selection_receipt_allows_truth_only_through_2024() -> None:
    selected = np.asarray(
        ["2024-11-19", "2024-11-20", "2024-12-30"], dtype="datetime64[D]"
    )
    archive = np.datetime64("2020-01-01") + np.arange(517).astype("timedelta64[D]")

    receipt = runner._selection_receipt(archive, selected)

    assert receipt["scoreable_count"] == 1
    assert receipt["forecast_only_count"] == 2
    assert receipt["verification_truth_opened"] is False
    assert receipt["sealed_2025_target_opened"] is False


def test_parent_validation_gate_runs_before_parent_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = RuntimeError("parent-gate-called")

    def reject(_: Path) -> dict[str, object]:
        raise marker

    monkeypatch.setattr(runner.bridge, "validate_parent_manifest", reject)

    with pytest.raises(RuntimeError, match="parent-gate-called"):
        runner.load_frozen_parent(tmp_path / "manifest.json")


def test_sealed_observation_path_check_catches_2025_zarr_component() -> None:
    assert runner._path_mentions_sealed_2025(Path("/archive/imd/2025.zarr"))
    assert runner._path_mentions_sealed_2025(Path("/archive/imd/imd_2025_daily"))
    assert not runner._path_mentions_sealed_2025(Path("/archive/imd/2024.zarr"))


def test_selected_validation_crps_is_resolved_by_seed_not_position() -> None:
    receipt = {
        "selected_seed": 43,
        "candidate_validation_crps": [
            {"seed": 44, "best_validation_crps": 1.4},
            {"seed": 42, "best_validation_crps": 1.2},
            {"seed": 43, "best_validation_crps": 1.1},
        ],
    }

    assert runner._selected_validation_crps(receipt) == 1.1
    receipt["candidate_validation_crps"].append(
        {"seed": 43, "best_validation_crps": 1.0}
    )
    with pytest.raises(bridge.BridgeContractError, match="unique"):
        runner._selected_validation_crps(receipt)


def test_resume_contract_rejects_mixed_device_or_amp_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    parent = runner.FrozenParent(
        manifest_path=tmp_path / "parent.json",
        run_root=tmp_path,
        manifest={},
        receipt={
            "parent_manifest_sha256": "1" * 64,
            "selected_checkpoint_sha256": "2" * 64,
        },
        checkpoint=tmp_path / "checkpoint.pt",
        support_artifact=tmp_path / "support.npz",
        normalization_mean=np.zeros(6, dtype=np.float32),
        normalization_std=np.ones(6, dtype=np.float32),
    )
    full_cohort = {"forecast_only": {"dates_sha256": "3" * 64}}
    selection = {
        "selected_dates_sha256": "4" * 64,
        "selected_count": 5,
    }
    cpu = {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "automatic_mixed_precision": False,
        "batch_size": 1,
    }
    runner._initialize_or_validate_run(
        output,
        parent=parent,
        full_cohort=full_cohort,
        selection=selection,
        execution_contract=cpu,
    )

    mixed = dict(cpu, resolved_device="cuda", automatic_mixed_precision=True)
    with pytest.raises(bridge.BridgeContractError, match="execution_contract differs"):
        runner._initialize_or_validate_run(
            output,
            parent=parent,
            full_cohort=full_cohort,
            selection=selection,
            execution_contract=mixed,
        )


def test_failed_later_invocation_cannot_mutate_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "published"
    output.mkdir()
    manifest = output / "manifest.json"
    original = b'{"status":"complete"}\n'
    manifest.write_bytes(original)

    def reject(_: argparse.Namespace) -> dict[str, object]:
        raise bridge.BridgeContractError("later invocation rejected")

    monkeypatch.setattr(runner, "run_inference", reject)
    with pytest.raises(bridge.BridgeContractError, match="later invocation"):
        runner.main(
            [
                "inference",
                "--parent-manifest",
                "parent.json",
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )

    assert manifest.read_bytes() == original
    assert not (output / "failure.json").exists()


def test_cli_exposes_restart_gate_and_does_not_offer_scoring() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "inference",
            "--parent-manifest",
            "parent.json",
            "--output",
            "resultsv3/india_s2s_probabilistic_bridge/smoke",
            "--smoke",
            "--device",
            "cpu",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.command == "inference"
    assert args.smoke is True
    assert args.max_cases is None
    assert runner.TRAIN_CONTEXT_YEARS[-1] == 2017
    assert 2025 not in runner.FORECAST_YEARS


def test_source_snapshot_binds_launcher_and_frozen_plan(tmp_path: Path) -> None:
    hashes = runner._source_snapshot(tmp_path)
    assert set(hashes) == {
        "code/src/india_s2s_probabilistic_bridge_run.py",
        "code/src/india_s2s_probabilistic_bridge.py",
        "code/src/fuxi_allseason_ensemble_calibration.py",
        "code/src/fuxi_ensemble_calibration_core.py",
        "code/slurm/run_india_s2s_probabilistic_bridge_inference.sbatch",
        "code/plan/INDIA_S2S_PROBABILISTIC_BRIDGE_20260826.md",
    }
    for relative, digest in hashes.items():
        assert runner.bridge.sha256_file(tmp_path / relative) == digest
