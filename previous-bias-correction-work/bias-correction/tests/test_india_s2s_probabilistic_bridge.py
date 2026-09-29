"""Contracts for the corrected India S2S probabilistic bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import india_s2s_probabilistic_bridge as bridge


def test_imd_targets_are_exact_plus_1_through_plus_42() -> None:
    init = np.asarray(["2024-01-01"], dtype="datetime64[D]")
    dates = bridge.target_valid_dates(init)

    assert dates.shape == (1, 6, 7)
    assert dates[0, 0, 0] == np.datetime64("2024-01-02")
    assert dates[0, 0, -1] == np.datetime64("2024-01-08")
    assert dates[0, -1, 0] == np.datetime64("2024-02-06")
    assert dates[0, -1, -1] == np.datetime64("2024-02-12")
    assert np.array_equal(
        bridge.weekly_means_from_daily_offsets(),
        np.asarray([4.0, 11.0, 18.0, 25.0, 32.0, 39.0]),
    )


def test_firewall_uses_last_end_labelled_imd_day() -> None:
    dates = np.asarray(["2024-11-19", "2024-11-20"], dtype="datetime64[D]")

    assert bridge.scoring_mask(dates).tolist() == [True, False]
    assert bridge.target_valid_dates(dates)[0, -1, -1] == np.datetime64(
        "2024-12-31"
    )
    assert bridge.target_valid_dates(dates)[1, -1, -1] == np.datetime64(
        "2025-01-01"
    )


def test_canonical_cohort_receipts_and_jjas_names_are_exact() -> None:
    initializations = bridge.load_canonical_initializations()
    receipt = bridge.build_cohort_receipt(initializations, strict=True)

    assert receipt["forecast_only"]["count"] == 517
    assert receipt["scoring"]["count"] == 505
    assert receipt["scoring"]["last"] == "2024-11-18"
    assert receipt["scoring"]["maximum_target_label"] == "2024-12-30"
    assert receipt["jjas_initialization"]["count"] == 170
    assert receipt["jjas_initialization"]["not_valid_midpoint_jjas"] is True
    assert set(receipt["jjas_valid_midpoint"]["case_count_by_lead"].values()) == {
        169
    }
    assert receipt["sealed_2025_target_opened"] is False


def _write_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    parent_root = tmp_path / "resultsv3" / "fuxi_allseason_ensemble_calibration"
    run_root = parent_root / "full"
    model_root = run_root / "models" / "location_spread"
    records = []
    artifact_hashes: dict[str, str] = {}
    for seed, score in ((42, 0.8), (43, 0.7), (44, 0.7)):
        checkpoint = model_root / f"seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        relative = str(checkpoint.relative_to(run_root))
        digest = bridge.sha256_file(checkpoint)
        artifact_hashes[relative] = digest
        records.append(
            {
                "seed": seed,
                "best_validation_crps": score,
                "checkpoint": relative,
                "checkpoint_sha256": digest,
            }
        )
    selected = records[1]
    selection = {
        "configuration": "location_spread",
        "criterion": "minimum 2018-2019 validation CRPS; exact ties use lowest seed",
        "selected_seed": selected["seed"],
        "best_validation_crps": selected["best_validation_crps"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "test_metrics_used_for_selection": False,
        "prediction_averaging": False,
        "parameter_averaging": False,
        "candidates": records,
    }
    selection_path = run_root / "models" / "deployable_selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    artifact_hashes[str(selection_path.relative_to(run_root))] = bridge.sha256_file(
        selection_path
    )
    source = run_root / "code" / "source.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_hash = bridge.sha256_file(source)
    artifact_hashes[str(source.relative_to(run_root))] = source_hash
    manifest = {
        "experiment": bridge.PARENT_EXPERIMENT,
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "configurations": ["location_spread"],
        "seeds": [42, 43, 44],
        "contract": {
            "revision": "imd_end_labelled_init_plus_1_through_42_v2",
            "target_day_offsets": list(range(1, 43)),
            "initialization_day_included_in_target": False,
            "sealed_2025_target_opened": False,
            "legacy_v1_zero_offset_checkpoint_loaded": False,
        },
        "deployment_selection": selection,
        "source_snapshot_sha256": {str(source.relative_to(run_root)): source_hash},
        "artifact_sha256": artifact_hashes,
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(bridge, "PARENT_ROOT", parent_root)
    return manifest_path


def test_parent_gate_recomputes_deployable_seed_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_parent(tmp_path, monkeypatch)

    receipt = bridge.validate_parent_manifest(manifest_path)

    assert receipt["selected_seed"] == 43
    assert receipt["target_day_offsets"] == list(range(1, 43))
    assert receipt["sealed_2025_target_opened"] is False


def test_parent_gate_rejects_v1_even_with_self_consistent_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_parent(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"] = "fuxi_allseason_ensemble_calibration_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(bridge.BridgeContractError, match="parent experiment differs"):
        bridge.validate_parent_manifest(manifest_path)


def test_archive_dependency_binary_tag_gate(tmp_path: Path) -> None:
    codec_root = tmp_path / "_deps" / "numcodecs"
    codec_root.mkdir(parents=True)
    assert not bridge.archive_dependencies_match_interpreter(tmp_path)
    suffix = bridge.importlib.machinery.EXTENSION_SUFFIXES[0]
    (codec_root / f"blosc{suffix}").touch()
    assert bridge.archive_dependencies_match_interpreter(tmp_path)
