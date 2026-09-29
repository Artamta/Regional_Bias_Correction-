from __future__ import annotations

import ast
import inspect
import json
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import evaluate_raw_identity_2025 as dispatcher
import freeze_raw_identity_2025_selection as freezer
import preflight_raw_identity_2025 as preflight
import raw_identity_2025_assets as assets
import raw_identity_2025_contract as contract
import raw_identity_2025_runtime as runtime


@pytest.fixture(autouse=True)
def _sandbox_global_2025_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "CANONICAL_EXPERIMENT_ROOT", tmp_path / "frozen")
    monkeypatch.setattr(
        contract,
        "GLOBAL_ACCESS_LEDGER_PATH",
        tmp_path / "global_raw_identity_2025_access_ledger.json",
    )


def _fixed_hashes() -> dict[str, str]:
    return {
        "e2_manifest": contract.EXPECTED_E2_MANIFEST_SHA256,
        "raw_run_manifest": contract.EXPECTED_RAW_MANIFEST_SHA256,
        "raw_selection": contract.EXPECTED_RAW_SELECTION_SHA256,
        "normalization": contract.EXPECTED_NORMALIZATION_SHA256,
        "raw_anchor": contract.EXPECTED_RAW_ANCHOR_SHA256,
        "frozen_model_source": contract.EXPECTED_MODEL_SOURCE_SHA256,
        "training_climatology_daily": contract.EXPECTED_TRAINING_CLIMATOLOGY_SHA256,
        **{
            f"checkpoint_seed_{seed}": contract.EXPECTED_CHECKPOINT_SHA256[seed]
            for seed in contract.EXPECTED_SEEDS
        },
    }


def _schedule_payload() -> dict:
    return {
        "initialization_window": "2025-06-01..2025-09-30",
        "initialization_weekdays_utc": ["Monday", "Thursday"],
        "initialization_hour_utc": 0,
        "initializations": [
            np.datetime_as_string(value, unit="D")
            for value in contract.expected_initialization_dates()
        ],
        "lead_week": [1, 2, 3, 4, 5, 6],
        "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
    }


def _secondary_payload() -> dict:
    return {
        "role": "secondary exploratory within the final untouched evaluation",
        "selection_or_multiplicity_claim": False,
        "stratification_field": "verifying weekly-mean IMD mm/day",
        "metrics": ["rmse_mm_day", "mae_mm_day", "bias_mm_day"],
        "weighting": "pooled India area x exact weekly IMD coverage",
        "empty_stratum": "insufficient_weight/no_estimate; no interval",
        "intensity_strata": [
            {
                "key": key,
                "label": label,
                "lower_mm_day_inclusive": lower,
                "upper_mm_day_exclusive": upper,
            }
            for key, label, lower, upper in contract.INTENSITY_STRATA
        ],
    }


def _selection_payload(root: Path) -> dict:
    canonical = contract.execution_paths_for_root(root)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "frozen",
        "frozen_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "frozen_before_2025_access": True,
        "test_year": 2025,
        "scientific_role": "sole untouched independent-test evaluation",
        "method_hierarchy": list(contract.METHOD_HIERARCHY),
        "primary_comparison": {"candidate": "raw_identity", "baseline": "raw_fuxi"},
        "model": {
            "name": contract.SELECTED_MODEL,
            "alpha": contract.SELECTED_ALPHA,
            "seeds": list(contract.EXPECTED_SEEDS),
            "ensemble": "arithmetic mean of standardized residuals",
            "training_anchor": "raw_fuxi",
            "parameter_count": 144_689,
        },
        "forbidden_methods": [
            "log_bias",
            "legacy_anchored_adapter",
            "raw_identity_raw_mean_preserved",
            "physical_adapter",
            "global_pretraining_adapter",
        ],
        "development_data": {
            "train_years": list(contract.TRAIN_YEARS),
            "validation_years": list(contract.VALIDATION_YEARS),
            "retrospective_audit_end_year": 2024,
        },
        "expected_test_shape": [35, 6, 27, 27],
        "evaluation_schedule": _schedule_payload(),
        "bootstrap": {
            "draws": contract.BOOTSTRAP_DRAWS,
            "block_length_initializations": contract.BOOTSTRAP_BLOCK_LENGTH,
            "seed": contract.BOOTSTRAP_SEED,
            "method": "paired circular moving blocks by initialization",
            "all_six_leads_retained": True,
        },
        "secondary_diagnostics": _secondary_payload(),
        "fixed_sha256": _fixed_hashes(),
        "paths": dict(contract.SEALED_RELATIVE_PATHS),
        "checkpoints": [
            {
                "seed": seed,
                "path": contract.SEALED_CHECKPOINT_RELATIVE_PATHS[seed],
                "sha256": contract.EXPECTED_CHECKPOINT_SHA256[seed],
            }
            for seed in contract.EXPECTED_SEEDS
        ],
        "support_climatology_bundle": {"sha256": "0" * 64, "array_sha256": {}},
        "test_data_locators": dict(contract.DEFAULT_TEST_DATA_LOCATORS),
        "test_data_locators_sha256": contract.canonical_json_sha256(
            contract.DEFAULT_TEST_DATA_LOCATORS
        ),
        "canonical_execution_paths": canonical,
        "canonical_execution_paths_sha256": contract.canonical_json_sha256(canonical),
        "code_sha256": contract.live_code_hashes(),
        "policy": {
            "no_2025_fitting_tuning_selection_calibration_or_retries": True,
            "preflight": "synthetic CUDA only; /storage forbidden",
            "final_access": "exact user receipt then one durable ledger attempt",
            "artifact_consumption": "sealed authenticated local copies only",
        },
    }


def _frozen(tmp_path: Path) -> contract.FrozenSelection:
    root = tmp_path / "frozen"
    root.mkdir()
    path = root / "selection.json"
    payload = _selection_payload(root)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    locators = contract.validate_data_locators(payload["test_data_locators"])
    canonical = contract.execution_paths_for_root(root)
    return contract.FrozenSelection(
        path=path,
        root=root,
        sha256=contract.sha256_file(path),
        payload=payload,
        e2_manifest_path=root / contract.SEALED_RELATIVE_PATHS["e2_manifest"],
        raw_manifest_path=root / contract.SEALED_RELATIVE_PATHS["raw_run_manifest"],
        raw_selection_path=root / contract.SEALED_RELATIVE_PATHS["raw_selection"],
        normalization_path=root / contract.SEALED_RELATIVE_PATHS["normalization"],
        raw_anchor_path=root / contract.SEALED_RELATIVE_PATHS["raw_anchor"],
        model_source_path=root / contract.SEALED_RELATIVE_PATHS["frozen_model_source"],
        bundle_path=root / contract.SEALED_RELATIVE_PATHS["support_climatology_bundle"],
        checkpoints=tuple(
            (
                seed,
                root / contract.SEALED_CHECKPOINT_RELATIVE_PATHS[seed],
                contract.EXPECTED_CHECKPOINT_SHA256[seed],
            )
            for seed in contract.EXPECTED_SEEDS
        ),
        data_locators=locators,
        data_locator_sha256=contract.canonical_json_sha256(locators),
        canonical_paths=canonical,
        canonical_paths_sha256=contract.canonical_json_sha256(canonical),
        workflow_code_bytes=contract.capture_authenticated_workflow_code(
            payload["code_sha256"]
        ),
    )


def _approval_payload(
    frozen: contract.FrozenSelection,
    *,
    preflight_sha256: str,
    approved_utc: str,
) -> dict:
    return {
        "schema_version": contract.APPROVAL_SCHEMA_VERSION,
        "decision": "approve_exactly_one_independent_2025_access",
        "approved_by": "raj.ayush",
        "approved_utc": approved_utc,
        "test_year": 2025,
        "selection_manifest_sha256": frozen.sha256,
        "preflight_receipt_sha256": preflight_sha256,
        "test_data_locators_sha256": frozen.data_locator_sha256,
        "canonical_execution_paths_sha256": frozen.canonical_paths_sha256,
        "final_output": frozen.canonical_paths["final_output"],
        "access_ledger": frozen.canonical_paths["access_ledger"],
        "failure_record": frozen.canonical_paths["failure_record"],
        "allowed_methods": list(contract.METHOD_HIERARCHY),
        "authorization": (
            "I authorize one access attempt for the frozen raw_identity versus "
            "raw_fuxi independent 2025 evaluation."
        ),
    }


def _normalization() -> dict:
    names = [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_calendar_climatology",
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "support",
        "explicit_log_fuxi_minus_imd_climatology",
        "fuxi_t2m_weekly",
        *[f"imd_climatology_offset_{offset:+d}d" for offset in runtime.OFFSETS_DAYS],
        *[
            f"fuxi_minus_imd_climatology_offset_{offset:+d}d"
            for offset in runtime.OFFSETS_DAYS
        ],
    ]
    result = {
        "input_channels": names,
        "spatial_context": {
            "enabled": True,
            "full_domain_channels": [
                "log_fuxi_mean",
                "log_fuxi_spread",
                "fuxi_t2m_weekly",
            ],
        },
    }
    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
    ):
        result[name] = {"mean_by_lead": [0.0] * 6, "std_by_lead": [1.0] * 6}
    return result


def test_pinned_live_development_hashes_are_exact() -> None:
    raw_run = contract.DEFAULT_RAW_RUN
    assert contract.sha256_file(contract.DEFAULT_E2_RUN / "manifest.json") == (
        contract.EXPECTED_E2_MANIFEST_SHA256
    )
    assert contract.sha256_file(raw_run / "manifest.json") == (
        contract.EXPECTED_RAW_MANIFEST_SHA256
    )
    assert contract.sha256_file(raw_run / "selection.json") == (
        contract.EXPECTED_RAW_SELECTION_SHA256
    )
    assert contract.sha256_file(raw_run / "normalization.json") == (
        contract.EXPECTED_NORMALIZATION_SHA256
    )
    assert contract.sha256_file(raw_run / "models/training_anchor_contract.npz") == (
        contract.EXPECTED_RAW_ANCHOR_SHA256
    )
    assert contract.sha256_file(raw_run / "code/models.py") == (
        contract.EXPECTED_MODEL_SOURCE_SHA256
    )


def test_selection_pins_hierarchy_schedule_climatology_and_intensity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    selection = root / "selection.json"
    payload = _selection_payload(root)
    contract.validate_selection_payload(payload, selection)
    dates = contract.expected_initialization_dates()
    assert dates.shape == (35,)
    assert all(pd.Timestamp(value).weekday() in (0, 3) for value in dates)
    assert payload["fixed_sha256"]["training_climatology_daily"] == (
        contract.EXPECTED_TRAINING_CLIMATOLOGY_SHA256
    )
    assert freezer._fixed_hashes() == payload["fixed_sha256"]
    for mutation, message in (
        (("method_hierarchy", ["raw_fuxi", "raw_identity", "projection"]), "only"),
        (("evaluation_schedule", {}), "schedule"),
        (("secondary_diagnostics", {}), "intensity"),
    ):
        changed = _selection_payload(root)
        changed[mutation[0]] = mutation[1]
        with pytest.raises(contract.SealContractError, match=message):
            contract.validate_selection_payload(changed, selection)


def test_selection_rejects_noncanonical_output_contract(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    payload = _selection_payload(root)
    payload["canonical_execution_paths"]["access_ledger"] = str(
        tmp_path / "elsewhere.json"
    )
    with pytest.raises(contract.SealContractError, match="canonical output/ledger"):
        contract.validate_selection_payload(payload, root / "selection.json")


def test_locator_validation_is_lexical_and_never_probes_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("locator validation probed a filesystem path")

    monkeypatch.setattr(contract.os, "stat", forbidden)
    validated = contract.validate_data_locators(contract.DEFAULT_TEST_DATA_LOCATORS)
    assert validated == contract.DEFAULT_TEST_DATA_LOCATORS
    wrong = dict(contract.DEFAULT_TEST_DATA_LOCATORS)
    wrong["forecast_tp_store"] = wrong["forecast_t2m_store"]
    with pytest.raises(contract.SealContractError):
        contract.validate_data_locators(wrong)
    lookalike = {
        key: (
            value.replace(
                "/storage/raj.ayush/s2s_final_data/final_iteration/standardized",
                "/storage/attacker/lookalike",
            )
            if isinstance(value, str)
            else value
        )
        for key, value in contract.DEFAULT_TEST_DATA_LOCATORS.items()
    }
    with pytest.raises(contract.SealContractError, match="canonical stores"):
        contract.validate_data_locators(lookalike)


def test_selection_root_and_global_ledger_are_not_relocatable(tmp_path: Path) -> None:
    canonical = tmp_path / "frozen"
    canonical.mkdir()
    paths = contract.execution_paths_for_root(canonical)
    assert paths["access_ledger"] == str(contract.GLOBAL_ACCESS_LEDGER_PATH)
    assert Path(paths["access_ledger"]).parent == canonical.parent
    assert Path(paths["access_ledger"]).parent != canonical
    moved = tmp_path / "refrozen-elsewhere"
    moved.mkdir()
    with pytest.raises(contract.SealContractError, match="global experiment root"):
        contract.execution_paths_for_root(moved)
    assert (
        "test_data_locators"
        not in inspect.signature(freezer.freeze_selection).parameters
    )


def test_freezer_rejects_alternate_root_before_development_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate = tmp_path / "alternate-freeze"
    monkeypatch.setattr(
        freezer,
        "validate_fixed_development_inputs",
        lambda *args: pytest.fail("development inputs read for alternate root"),
    )
    with pytest.raises(contract.SealContractError, match="canonical experiment root"):
        freezer.freeze_selection(
            tmp_path / "e2",
            tmp_path / "raw",
            alternate,
            attest_no_2025_access=True,
        )


def test_no_follow_reads_writes_and_component_traversal(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    victim = real / "victim.txt"
    victim.write_text("safe", encoding="utf-8")
    file_link = real / "file-link"
    file_link.symlink_to(victim)
    with pytest.raises(OSError):
        contract.read_bytes_no_follow(file_link)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        contract.read_bytes_no_follow(parent_link / "victim.txt")
    output_link = real / "output.json"
    output_link.symlink_to(victim)
    with pytest.raises(FileExistsError):
        contract.write_json_exclusive(output_link, {"unsafe": True})
    assert victim.read_text(encoding="utf-8") == "safe"


def test_descriptor_anchored_zarr_refuses_inner_symlinks(tmp_path: Path) -> None:
    store_path = tmp_path / "store.zarr"
    store_path.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"data")
    (store_path / "unsafe-key").symlink_to(victim)
    with pytest.raises(contract.SealContractError, match="symlink"):
        runtime.DescriptorAnchoredZarrStore(store_path)


def test_descriptor_anchored_zarr_survives_lexical_root_swap(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store.zarr"
    store_path.mkdir()
    (store_path / "chunk").write_bytes(b"authenticated")
    moved = tmp_path / "anchored-store.zarr"
    attacker = tmp_path / "attacker-store.zarr"
    attacker.mkdir()
    (attacker / "chunk").write_bytes(b"attacker")
    with runtime.DescriptorAnchoredZarrStore(store_path) as source:
        store_path.rename(moved)
        store_path.symlink_to(attacker, target_is_directory=True)
        assert source["chunk"] == b"authenticated"
        assert source.consumed_sha256() == {
            "chunk": contract.sha256_bytes(b"authenticated")
        }


def test_descriptor_anchored_zarr_detects_consumed_key_mutation(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store.zarr"
    store_path.mkdir()
    chunk = store_path / "chunk"
    chunk.write_bytes(b"first")
    with runtime.DescriptorAnchoredZarrStore(store_path) as source:
        assert source["chunk"] == b"first"
        chunk.write_bytes(b"second")
        with pytest.raises(contract.SealContractError, match="changed"):
            source.consumed_sha256()


def test_xarray_reads_synthetic_zarr_through_descriptor_store(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "synthetic.zarr"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    xr.Dataset({"value": (("x", "y"), expected)}).to_zarr(
        store_path, mode="w", consolidated=True
    )
    with (
        runtime.DescriptorAnchoredZarrStore(store_path) as source,
        xr.open_zarr(source, consolidated=True) as dataset,
    ):
        assert np.array_equal(dataset.value.load().values, expected)
        consumed = source.consumed_sha256()
    assert ".zmetadata" in consumed
    assert any(key.startswith("value/") for key in consumed)


def test_all_three_bound_stores_record_identity_and_consumed_bytes(
    tmp_path: Path,
) -> None:
    identities: dict[str, dict[str, int]] = {}
    consumed: dict[str, dict[str, str]] = {}
    paths = [tmp_path / f"source-{index}.zarr" for index in range(3)]
    for index, path in enumerate(paths):
        expected = np.asarray([index], dtype=np.float32)
        xr.Dataset({"value": (("x",), expected)}).to_zarr(
            path, mode="w", consolidated=True
        )
        with runtime._open_bound_zarr(path, identities, consumed) as dataset:
            assert np.array_equal(dataset.value.load().values, expected)
    assert set(identities) == {str(path) for path in paths}
    assert set(consumed) == {str(path) for path in paths}
    assert all(set(identity) == {"device", "inode"} for identity in identities.values())
    assert all(".zmetadata" in hashes for hashes in consumed.values())


def test_imd_loader_consumes_only_requested_observation_and_coverage_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic-imd.zarr"
    dates = np.arange(
        np.datetime64("2025-06-01"),
        np.datetime64("2025-06-07"),
        dtype="datetime64[D]",
    )
    values = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    coverage = np.ones_like(values, dtype=np.float32)
    xr.Dataset(
        {
            "observation": (("time", "latitude", "longitude"), values),
            "observation_fraction": (
                ("time", "latitude", "longitude"),
                coverage,
            ),
        },
        coords={
            "time": dates.astype("datetime64[ns]"),
            "latitude": [1.0, 0.0],
            "longitude": [70.0, 71.0],
        },
    ).to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={
            "observation": {"chunks": (2, 2, 2)},
            "observation_fraction": {"chunks": (2, 2, 2)},
        },
    )
    identities: dict[str, dict[str, int]] = {}
    consumed: dict[str, dict[str, str]] = {}
    with runtime._open_bound_zarr(path, identities, consumed) as dataset:
        selected_dates, selected_values, selected_coverage = (
            runtime._load_selected_imd_arrays(dataset, dates[:2], (2, 2))
        )
    assert np.array_equal(selected_dates, dates[:2])
    assert np.array_equal(selected_values, values[:2])
    assert np.array_equal(selected_coverage, coverage[:2])
    keys = consumed[str(path)]
    observation_chunks = [
        key
        for key in keys
        if key.startswith("observation/")
        and "/." not in key
        and not key.endswith(".zarray")
    ]
    coverage_chunks = [
        key
        for key in keys
        if key.startswith("observation_fraction/")
        and "/." not in key
        and not key.endswith(".zarray")
    ]
    assert observation_chunks == ["observation/0.0.0"]
    assert coverage_chunks == ["observation_fraction/0.0.0"]


def test_no_clobber_publication_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "new").write_text("new", encoding="utf-8")
    (destination / "old").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        contract.rename_noreplace(source, destination)
    assert (source / "new").read_text(encoding="utf-8") == "new"
    assert (destination / "old").read_text(encoding="utf-8") == "old"


def test_freezer_refuses_symlink_output_parent_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(contract, "CANONICAL_EXPERIMENT_ROOT", linked / "frozen")
    monkeypatch.setattr(freezer, "validate_fixed_development_inputs", lambda *args: {})
    with pytest.raises(OSError):
        freezer.freeze_selection(
            tmp_path / "e2",
            tmp_path / "raw",
            linked / "frozen",
            attest_no_2025_access=True,
        )


def test_freezer_copies_the_captured_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_tree = tmp_path / "predictions.zarr"
    prediction_tree.mkdir()
    (prediction_tree / ".zmetadata").write_bytes(b"metadata")
    tree_sha = freezer._sha256_tree_no_symlinks(prediction_tree)
    captured = {
        name: f"captured:{name}".encode()
        for name in contract.SEALED_RELATIVE_PATHS
        if name != "support_climatology_bundle"
    }
    checkpoints = {
        seed: f"checkpoint:{seed}".encode() for seed in contract.EXPECTED_SEEDS
    }
    monkeypatch.setattr(
        freezer,
        "validate_fixed_development_inputs",
        lambda *args: {
            "predictions_path": prediction_tree,
            "predictions_tree_sha256": tree_sha,
            "captured": captured,
            "captured_checkpoints": checkpoints,
        },
    )

    def fake_bundle(inputs, output):
        contract.write_bytes_exclusive(output, b"bundle")
        return {"sha256": contract.sha256_bytes(b"bundle"), "array_sha256": {}}

    monkeypatch.setattr(freezer, "build_pre_2025_bundle", fake_bundle)
    output = tmp_path / "frozen"
    selection = freezer.freeze_selection(
        tmp_path / "e2",
        tmp_path / "raw",
        output,
        attest_no_2025_access=True,
    )
    assert selection == output / "selection.json"
    for name, content in captured.items():
        assert (output / contract.SEALED_RELATIVE_PATHS[name]).read_bytes() == content
    for seed, content in checkpoints.items():
        assert (
            output / contract.SEALED_CHECKPOINT_RELATIVE_PATHS[seed]
        ).read_bytes() == content


def test_freezer_parent_fsync_fault_authenticates_renamed_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_tree = tmp_path / "predictions.zarr"
    prediction_tree.mkdir()
    (prediction_tree / ".zmetadata").write_bytes(b"metadata")
    tree_sha = freezer._sha256_tree_no_symlinks(prediction_tree)
    captured = {
        name: f"captured:{name}".encode()
        for name in contract.SEALED_RELATIVE_PATHS
        if name != "support_climatology_bundle"
    }
    checkpoints = {
        seed: f"checkpoint:{seed}".encode() for seed in contract.EXPECTED_SEEDS
    }
    monkeypatch.setattr(
        freezer,
        "validate_fixed_development_inputs",
        lambda *args: {
            "predictions_path": prediction_tree,
            "predictions_tree_sha256": tree_sha,
            "captured": captured,
            "captured_checkpoints": checkpoints,
        },
    )

    def fake_bundle(inputs, output):
        contract.write_bytes_exclusive(output, b"bundle")
        return {"sha256": contract.sha256_bytes(b"bundle"), "array_sha256": {}}

    monkeypatch.setattr(freezer, "build_pre_2025_bundle", fake_bundle)
    output = tmp_path / "frozen"
    renamed = False
    real_rename = contract.rename_noreplace
    real_fsync_directory = contract.fsync_directory

    def tracked_rename(source, destination):
        nonlocal renamed
        real_rename(source, destination)
        renamed = True

    def fail_parent_fsync(path):
        if renamed and Path(path) == output.parent:
            raise OSError("synthetic freezer parent fsync fault")
        real_fsync_directory(path)

    authenticated: list[Path] = []
    monkeypatch.setattr(freezer.contract, "rename_noreplace", tracked_rename)
    monkeypatch.setattr(freezer.contract, "fsync_directory", fail_parent_fsync)
    monkeypatch.setattr(
        freezer.contract,
        "load_frozen_selection",
        lambda path: authenticated.append(Path(path)),
    )
    with pytest.raises(OSError, match="freezer parent fsync fault"):
        freezer.freeze_selection(
            tmp_path / "e2",
            tmp_path / "raw",
            output,
            attest_no_2025_access=True,
        )
    assert (output / "selection.json").is_file()
    assert authenticated == [output / "selection.json"]


def test_preflight_model_executes_only_the_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "models.py"
    original = b"VALUE = 'sealed'\n"
    source.write_bytes(original)
    monkeypatch.setattr(
        contract, "EXPECTED_MODEL_SOURCE_SHA256", contract.sha256_bytes(original)
    )
    real_read = contract.read_bytes_no_follow
    calls = 0

    def read_then_mutate(path):
        nonlocal calls
        calls += 1
        content = real_read(path)
        source.write_bytes(b"VALUE = 'tampered'\n")
        return content

    monkeypatch.setattr(contract, "read_bytes_no_follow", read_then_mutate)
    module = preflight._load_frozen_models(source)
    assert module.VALUE == "sealed"
    assert calls == 1


def test_checkpoint_normalization_and_anchor_consumption_use_in_memory_bytes() -> None:
    preflight_source = inspect.getsource(preflight.synthetic_cuda_proof)
    inference_source = inspect.getsource(runtime.infer_raw_identity)
    run_source = inspect.getsource(runtime.run_evaluation)
    assert "io.BytesIO(checkpoint_content)" in preflight_source
    assert "io.BytesIO(checkpoint_content)" in inference_source
    assert "io.BytesIO(anchor_content)" in inference_source
    assert "parse_json_bytes" in run_source
    assert "capture_authenticated_bytes" in run_source
    assert "spec_from_file_location" not in inspect.getsource(preflight)
    assert "spec_from_file_location" not in inspect.getsource(runtime)


def test_contract_preflight_modules_have_no_store_capable_top_level_imports() -> None:
    evaluate_directory = Path(contract.__file__).parent
    for filename in (
        "raw_identity_2025_contract.py",
        "preflight_raw_identity_2025.py",
        "evaluate_raw_identity_2025.py",
    ):
        tree = ast.parse((evaluate_directory / filename).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert "xarray" not in imports
        assert "zarr" not in imports
        assert "raw_identity_2025_runtime" not in imports


def test_preflight_derives_canonical_receipt_and_creates_no_control_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    monkeypatch.setattr(
        preflight.contract, "load_frozen_selection", lambda path: frozen
    )
    monkeypatch.setattr(
        preflight,
        "synthetic_cuda_proof",
        lambda selection: {
            "synthetic_cuda_inference": True,
            "checkpoint_seeds_loaded": [42, 43, 44],
            "storage_paths_opened": [],
            "access_ledger_created": False,
            "result_created": False,
        },
    )
    receipt = preflight.run_preflight(frozen.path)
    assert receipt == Path(frozen.canonical_paths["preflight_receipt"])
    assert receipt.is_file()
    assert not Path(frozen.canonical_paths["access_ledger"]).exists()
    assert not Path(frozen.canonical_paths["final_output"]).exists()
    assert not Path(frozen.canonical_paths["failure_record"]).exists()
    contract.validate_preflight_receipt(receipt, frozen)


def test_preflight_and_dispatcher_cli_reject_arbitrary_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda *args: pytest.fail("parser accepted override"),
    )
    with pytest.raises(SystemExit):
        preflight.main(
            ["--selection-manifest", "/tmp/selection.json", "--proof-receipt", "/tmp/x"]
        )
    monkeypatch.setattr(
        dispatcher, "dispatch", lambda **kwargs: pytest.fail("parser accepted override")
    )
    with pytest.raises(SystemExit):
        dispatcher.main(
            [
                "--selection-manifest",
                "/tmp/selection.json",
                "--approval-receipt",
                "/tmp/approval.json",
                "--require-cuda",
                "--output",
                "/tmp/result",
            ]
        )


def test_approval_is_sha_and_canonical_path_bound(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    preflight_time = datetime.now(timezone.utc) - timedelta(hours=1)
    approval_time = preflight_time + timedelta(minutes=10)
    approval = _approval_payload(
        frozen,
        preflight_sha256="1" * 64,
        approved_utc=approval_time.isoformat(),
    )
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval) + "\n", encoding="utf-8")
    contract.validate_approval_receipt(
        path,
        frozen,
        preflight_receipt_sha256="1" * 64,
        preflight_created_utc=preflight_time.isoformat(),
    )
    for key in (
        "selection_manifest_sha256",
        "preflight_receipt_sha256",
        "test_data_locators_sha256",
        "canonical_execution_paths_sha256",
        "final_output",
        "access_ledger",
        "failure_record",
    ):
        tampered = dict(approval)
        tampered[key] = "tampered"
        path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
        with pytest.raises(contract.SealContractError):
            contract.validate_approval_receipt(
                path,
                frozen,
                preflight_receipt_sha256="1" * 64,
                preflight_created_utc=preflight_time.isoformat(),
            )


def test_access_ledger_is_canonical_o_excl_fsynced_and_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    fsync_calls: list[int] = []
    real_fsync = contract.os.fsync

    def record_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(contract.os, "fsync", record_fsync)
    ledger = Path(frozen.canonical_paths["access_ledger"])
    contract.create_access_ledger(
        ledger,
        frozen=frozen,
        preflight_receipt_sha256="1" * 64,
        approval_receipt_path=approval,
        approval_receipt_sha256="2" * 64,
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["runtime_import_state_at_commit"] == "not_imported"
    assert payload["canonical_execution_paths"] == frozen.canonical_paths
    assert len(fsync_calls) >= 2
    with pytest.raises(FileExistsError):
        contract.create_access_ledger(
            ledger,
            frozen=frozen,
            preflight_receipt_sha256="1" * 64,
            approval_receipt_path=approval,
            approval_receipt_sha256="2" * 64,
        )
    with pytest.raises(contract.SealContractError, match="canonical"):
        contract.create_access_ledger(
            tmp_path / "other-ledger.json",
            frozen=frozen,
            preflight_receipt_sha256="1" * 64,
            approval_receipt_path=approval,
            approval_receipt_sha256="2" * 64,
        )


def _patch_dispatch_validation(
    monkeypatch: pytest.MonkeyPatch, frozen: contract.FrozenSelection
) -> Path:
    approval = frozen.root.parent / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "raw_identity_2025_runtime", raising=False)
    monkeypatch.setattr(
        dispatcher.contract, "load_frozen_selection", lambda path: frozen
    )
    monkeypatch.setattr(
        dispatcher.contract,
        "validate_preflight_receipt",
        lambda path, selection: (
            {
                "created_utc": (
                    datetime.now(timezone.utc) - timedelta(minutes=2)
                ).isoformat()
            },
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        dispatcher.contract,
        "validate_approval_receipt",
        lambda *args, **kwargs: ({}, "2" * 64),
    )
    return approval


def _prepare_runtime_attempt(
    tmp_path: Path, frozen: contract.FrozenSelection
) -> tuple[str, str]:
    preflight_sha256 = "1" * 64
    approval_sha256 = "2" * 64
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    contract.create_access_ledger(
        frozen.canonical_paths["access_ledger"],
        frozen=frozen,
        preflight_receipt_sha256=preflight_sha256,
        approval_receipt_path=approval,
        approval_receipt_sha256=approval_sha256,
    )
    contract.write_json_exclusive(
        frozen.canonical_paths["failure_record"],
        dispatcher._status_payload(
            frozen,
            status="running",
            stage="runtime_imported_before_store_open",
            ledger_committed=True,
        ),
    )
    return preflight_sha256, approval_sha256


def test_dispatch_imports_runtime_only_after_durable_ledger_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    approval = _patch_dispatch_validation(monkeypatch, frozen)
    events: list[str] = []

    class FakeRuntime:
        @staticmethod
        def run_evaluation(**kwargs):
            events.append("run")
            return Path(frozen.canonical_paths["final_output"])

    def import_runtime(selection):
        assert selection is frozen
        ledger = Path(frozen.canonical_paths["access_ledger"])
        status = Path(frozen.canonical_paths["failure_record"])
        assert ledger.is_file()
        assert json.loads(status.read_text())["stage"] == (
            "ledger_committed_runtime_not_imported"
        )
        events.append("import")
        return FakeRuntime

    monkeypatch.setattr(dispatcher, "_load_authenticated_runtime", import_runtime)
    result = dispatcher.dispatch(
        selection_manifest=frozen.path,
        approval_receipt=approval,
        require_cuda=False,
    )
    assert result == Path(frozen.canonical_paths["final_output"])
    assert events == ["import", "run"]
    ledger = json.loads(Path(frozen.canonical_paths["access_ledger"]).read_text())
    assert ledger["runtime_import_state_at_commit"] == "not_imported"
    with pytest.raises(FileExistsError):
        dispatcher.dispatch(
            selection_manifest=frozen.path,
            approval_receipt=approval,
            require_cuda=False,
        )


def test_runtime_and_assets_execute_from_authenticated_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    code = dict(frozen.workflow_code_bytes)
    code["raw_identity_2025_assets.py"] = b"TOKEN = 'sealed-assets'\n"
    code["raw_identity_2025_runtime.py"] = (
        b"import raw_identity_2025_assets as assets\n"
        b"TOKEN = assets.TOKEN + ':sealed-runtime'\n"
    )
    synthetic = replace(frozen, workflow_code_bytes=code)
    monkeypatch.delitem(sys.modules, "raw_identity_2025_runtime", raising=False)
    monkeypatch.delitem(sys.modules, "raw_identity_2025_assets", raising=False)
    loaded = dispatcher._load_authenticated_runtime(synthetic)
    assert loaded.TOKEN == "sealed-assets:sealed-runtime"


def test_workflow_code_hash_contract_rejects_non_object() -> None:
    with pytest.raises(contract.SealContractError, match="must be an object"):
        contract.capture_authenticated_workflow_code(None)  # type: ignore[arg-type]


def test_post_ledger_failure_is_durable_and_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    approval = _patch_dispatch_validation(monkeypatch, frozen)
    imports = 0

    class FailingRuntime:
        @staticmethod
        def run_evaluation(**kwargs):
            raise RuntimeError("synthetic post-ledger failure")

    def import_runtime(selection):
        assert selection is frozen
        nonlocal imports
        imports += 1
        return FailingRuntime

    monkeypatch.setattr(dispatcher, "_load_authenticated_runtime", import_runtime)
    with pytest.raises(RuntimeError, match="post-ledger failure"):
        dispatcher.dispatch(
            selection_manifest=frozen.path,
            approval_receipt=approval,
            require_cuda=False,
        )
    status = json.loads(Path(frozen.canonical_paths["failure_record"]).read_text())
    assert status["status"] == "failed"
    assert status["attempt_consumed"] is True
    assert status["retry_permitted"] is False
    assert Path(frozen.canonical_paths["access_ledger"]).is_file()
    assert not Path(frozen.canonical_paths["final_output"]).exists()
    with pytest.raises(FileExistsError):
        dispatcher.dispatch(
            selection_manifest=frozen.path,
            approval_receipt=approval,
            require_cuda=False,
        )
    assert imports == 1


def test_ledger_commit_status_failure_keeps_furthest_truthful_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    approval = _patch_dispatch_validation(monkeypatch, frozen)
    real_replace = contract.atomic_replace_json
    calls = 0

    def fail_first_replace(path, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic status fsync failure")
        return real_replace(path, payload)

    monkeypatch.setattr(dispatcher.contract, "atomic_replace_json", fail_first_replace)
    with pytest.raises(OSError, match="status fsync failure"):
        dispatcher.dispatch(
            selection_manifest=frozen.path,
            approval_receipt=approval,
            require_cuda=False,
        )
    status = contract.read_json(frozen.canonical_paths["failure_record"])
    assert status["status"] == "failed"
    assert status["stage"] == "ledger_committed_runtime_not_imported"
    assert status["access_ledger_committed"] is True
    assert status["attempt_consumed"] is True


def test_post_ledger_staging_allocation_failure_records_orphan_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    preflight_sha256, approval_sha256 = _prepare_runtime_attempt(tmp_path, frozen)
    orphan = frozen.root / ".synthetic-orphan"

    def fail_after_mkdir(parent, prefix):
        contract.secure_mkdir(orphan)
        raise contract.StagingAllocationError(orphan, OSError("synthetic fsync fault"))

    monkeypatch.setattr(
        runtime.contract, "create_secure_staging_directory", fail_after_mkdir
    )
    monkeypatch.setattr(
        runtime,
        "load_independent_test_data",
        lambda selection: pytest.fail("store load reached after staging failure"),
    )
    with pytest.raises(contract.StagingAllocationError):
        runtime.run_evaluation(
            frozen=frozen,
            preflight_receipt_sha256=preflight_sha256,
            approval_receipt_sha256=approval_sha256,
        )
    status = contract.read_json(frozen.canonical_paths["failure_record"])
    assert status["status"] == "failed"
    assert status["attempt_consumed"] is True
    assert status["details"]["partial_directory"] == str(orphan)
    assert orphan.is_dir()


def test_parent_fsync_fault_after_rename_records_authenticated_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    preflight_sha256, approval_sha256 = _prepare_runtime_attempt(tmp_path, frozen)
    shape = (35, 6, 27, 27)
    support = np.zeros((27, 27), dtype=bool)
    support.reshape(-1)[:171] = True
    fake_data = SimpleNamespace(
        initializations=contract.expected_initialization_dates(),
        valid_dates=runtime._derive_valid_dates(
            contract.expected_initialization_dates()
        ),
        latitude=np.linspace(39.0, 0.0, 27),
        longitude=np.linspace(60.0, 99.0, 27),
        support=support,
        area_weight_km2=support.astype(np.float64),
        training_climatology_daily=np.zeros((366, 27, 27), dtype=np.float32),
        raw_fuxi=np.zeros(shape, dtype=np.float32),
        fuxi_spread=np.zeros(shape, dtype=np.float32),
        fuxi_t2m_kelvin=np.full(shape, 300.0, dtype=np.float32),
        fuxi_member_count=np.full(shape, 50, dtype=np.int16),
        fuxi_t2m_member_count=np.full(shape, 50, dtype=np.int16),
        selected_daily_imd=np.zeros((35, 6, 7, 27, 27), dtype=np.float32),
        selected_daily_coverage=np.ones((35, 6, 7, 27, 27), dtype=np.float32),
        truth=np.zeros(shape, dtype=np.float32),
        climatology=np.zeros(shape, dtype=np.float32),
        weekly_coverage=np.broadcast_to(support, shape).astype(np.float32),
        source_consumed_key_sha256={"tp": {"chunk": "a" * 64}},
        source_store_identity={"tp": {"device": 1, "inode": 2}},
        evaluated_source_array_sha256={"raw_fuxi": "b" * 64},
    )
    monkeypatch.setattr(
        runtime, "load_independent_test_data", lambda selection: fake_data
    )
    monkeypatch.setattr(
        runtime.contract, "capture_authenticated_bytes", lambda *args, **kwargs: b"{}"
    )
    monkeypatch.setattr(
        runtime, "build_frozen_features", lambda *args: np.zeros((1,), dtype=np.float32)
    )
    monkeypatch.setattr(
        runtime,
        "infer_raw_identity",
        lambda *args: (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            {"synthetic": True},
        ),
    )

    def write_fake_assets(output, **kwargs):
        contract.write_bytes_exclusive(output / "artifact.bin", b"sealed-result")
        return {"synthetic": True}

    monkeypatch.setattr(runtime.assets, "write_assets", write_fake_assets)
    renamed = False
    real_rename = contract.rename_noreplace
    real_fsync_directory = contract.fsync_directory

    def tracked_rename(source, destination):
        nonlocal renamed
        real_rename(source, destination)
        renamed = True

    def fail_parent_fsync(path):
        if renamed and Path(path) == frozen.root:
            raise OSError("synthetic parent fsync fault")
        real_fsync_directory(path)

    monkeypatch.setattr(runtime.contract, "rename_noreplace", tracked_rename)
    monkeypatch.setattr(runtime.contract, "fsync_directory", fail_parent_fsync)
    with pytest.raises(OSError, match="parent fsync fault"):
        runtime.run_evaluation(
            frozen=frozen,
            preflight_receipt_sha256=preflight_sha256,
            approval_receipt_sha256=approval_sha256,
        )
    final_output = Path(frozen.canonical_paths["final_output"])
    status = contract.read_json(frozen.canonical_paths["failure_record"])
    assert final_output.is_dir()
    assert status["status"] == "failed"
    assert status["stage"] == "published_before_parent_fsync"
    assert status["details"]["published_before_failure"] is True
    assert status["details"]["publication_authentication"]["final_output"] == str(
        final_output
    )


def test_invalid_approval_cannot_create_status_ledger_or_import_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    approval = frozen.root.parent / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "raw_identity_2025_runtime", raising=False)
    monkeypatch.setattr(
        dispatcher.contract, "load_frozen_selection", lambda path: frozen
    )
    monkeypatch.setattr(
        dispatcher.contract,
        "validate_preflight_receipt",
        lambda *args: ({"created_utc": contract.utc_now()}, "1" * 64),
    )
    monkeypatch.setattr(
        dispatcher.contract,
        "validate_approval_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            contract.SealContractError("approval missing")
        ),
    )
    monkeypatch.setattr(
        dispatcher,
        "_load_authenticated_runtime",
        lambda selection: pytest.fail("runtime imported before approval"),
    )
    with pytest.raises(contract.SealContractError, match="approval missing"):
        dispatcher.dispatch(
            selection_manifest=frozen.path,
            approval_receipt=approval,
            require_cuda=False,
        )
    assert not Path(frozen.canonical_paths["access_ledger"]).exists()
    assert not Path(frozen.canonical_paths["failure_record"]).exists()


def test_weekly_truth_is_plain_mean_and_coverage_only_sets_score_weight() -> None:
    daily = np.arange(7, dtype=np.float32).reshape(1, 1, 7, 1, 1)
    coverage = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2], dtype=np.float32)
    coverage = coverage.reshape(1, 1, 7, 1, 1)
    truth, weekly = runtime._aggregate_weekly_truth(daily, coverage)
    assert truth.item() == 3.0
    assert weekly.item() == pytest.approx(0.2)
    weighted = float(np.sum(daily.reshape(7) * coverage.reshape(7)) / coverage.sum())
    assert truth.item() != weighted


@pytest.mark.parametrize("bad", [np.nan, -0.01, 1.01])
def test_coverage_contract_rejects_nonfinite_or_out_of_range(bad: float) -> None:
    dataset = xr.Dataset(
        {
            "observation_fraction": (
                ("time", "latitude", "longitude"),
                np.full((2, 3, 4), bad, dtype=np.float32),
            )
        }
    )
    with pytest.raises(contract.SealContractError, match="finite and in"):
        runtime._collapse_coverage(dataset, (3, 4))


def test_primary_metric_contract_matches_e2_float_order_and_rejects_nan() -> None:
    import evaluate_raw_identity_2022_2024_audit as e2

    generator = np.random.default_rng(22)
    shape = (2, 6, 4, 5)
    truth = generator.uniform(0.2, 8.0, size=shape).astype(np.float32)
    prediction = (truth + generator.normal(0.0, 0.8, size=shape)).astype(np.float32)
    climatology = generator.uniform(0.1, 5.0, size=shape).astype(np.float32)
    coverage = generator.uniform(0.2, 1.0, size=shape).astype(np.float32)
    area = generator.uniform(1.0, 3.0, size=shape[-2:]).astype(np.float64)
    actual = assets.weighted_case_metrics(
        prediction, truth, climatology, coverage, area
    )
    for case in range(shape[0]):
        for lead in range(shape[1]):
            weight = area * coverage[case, lead]
            absolute = e2.weighted_metrics(
                prediction[case, lead], truth[case, lead], weight
            )
            anomaly = e2.weighted_metrics(
                prediction[case, lead] - climatology[case, lead],
                truth[case, lead] - climatology[case, lead],
                weight,
            )
            for metric in assets.METRICS:
                expected = anomaly["acc"] if metric == "acc" else absolute[metric]
                assert actual[metric][case, lead] == expected
    coverage[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        assets.weighted_case_metrics(prediction, truth, climatology, coverage, area)


def test_primary_metrics_require_three_cells_and_finite_acc() -> None:
    shape = (1, 1, 2, 2)
    prediction = np.arange(4, dtype=np.float32).reshape(shape)
    truth = (prediction + 1.0).astype(np.float32)
    climatology = np.zeros(shape, dtype=np.float32)
    coverage = np.ones(shape, dtype=np.float32)
    area = np.asarray([[1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="three"):
        assets.weighted_case_metrics(prediction, truth, climatology, coverage, area)
    area[:] = 1.0
    prediction[:] = 2.0
    with pytest.raises(ValueError, match="non-finite"):
        assets.weighted_case_metrics(prediction, truth, climatology, coverage, area)


def test_full_29_channel_feature_builder_is_exactly_e2() -> None:
    import evaluate_raw_identity_2022_2024_audit as e2

    generator = np.random.default_rng(7)
    cases, leads, height, width = 100, 6, 27, 27
    initializations = np.arange(
        np.datetime64("2022-01-01"),
        np.datetime64("2022-01-01") + np.timedelta64(cases, "D"),
    )
    valid_dates = runtime._derive_valid_dates(initializations)
    shape = (cases, leads, height, width)
    support = np.zeros((height, width), dtype=bool)
    support.reshape(-1)[:171] = True
    daily_climatology = generator.uniform(0.0, 12.0, size=(366, height, width)).astype(
        np.float32
    )
    daily_climatology[:, ~support] = np.nan
    raw = generator.uniform(0.0, 20.0, size=shape).astype(np.float32)
    spread = generator.uniform(0.0, 5.0, size=shape).astype(np.float32)
    t2m = generator.uniform(270.0, 315.0, size=shape).astype(np.float32)
    weekly = np.mean(
        daily_climatology[runtime._calendar_positions(valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    latitude = np.linspace(39.0, 0.0, height, dtype=np.float64)
    longitude = np.linspace(60.0, 99.0, width, dtype=np.float64)
    data = runtime.IndependentTestData(
        initializations=initializations,
        valid_dates=valid_dates,
        latitude=latitude,
        longitude=longitude,
        support=support,
        area_weight_km2=support.astype(np.float64),
        training_climatology_daily=daily_climatology,
        raw_fuxi=raw,
        fuxi_spread=spread,
        fuxi_t2m_kelvin=t2m,
        fuxi_member_count=np.full(shape, 50, dtype=np.int16),
        fuxi_t2m_member_count=np.full(shape, 50, dtype=np.int16),
        selected_daily_imd=np.zeros((cases, leads, 7, height, width), dtype=np.float32),
        selected_daily_coverage=np.ones(
            (cases, leads, 7, height, width), dtype=np.float32
        ),
        truth=np.where(support, weekly, np.nan).astype(np.float32),
        climatology=weekly,
        weekly_coverage=np.broadcast_to(support, shape).astype(np.float32),
        source_consumed_key_sha256={},
        source_store_identity={},
        evaluated_source_array_sha256={},
    )
    normalization = _normalization()
    actual = runtime.build_frozen_features(data, normalization)
    forecast = SimpleNamespace(
        ensemble_mean=raw,
        ensemble_spread=spread,
        valid_dates=valid_dates,
        latitude=latitude,
        longitude=longitude,
    )
    expected, expected_weekly = e2.build_features(
        forecast,
        t2m,
        daily_climatology,
        normalization,
        support,
        SimpleNamespace(calendar_positions=runtime._calendar_positions),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(weekly, expected_weekly)


def test_circular_blocks_are_deterministic_wrap_and_keep_35_cases() -> None:
    first = assets.circular_moving_block_indices(35, draws=29)
    second = assets.circular_moving_block_indices(35, draws=29)
    assert first.shape == (29, 35)
    assert np.array_equal(first, second)
    for row in first:
        assert np.all((row[1:13] - row[:12]) % 35 == 1)
        assert np.all((row[14:26] - row[13:25]) % 35 == 1)
    assert assets.circular_moving_block_indices(35).shape == (10_000, 35)


def test_intensity_secondary_has_estimates_for_nonempty_strata() -> None:
    truth_cell = np.asarray([0.5, 2.0, 7.0, 15.0, 25.0], dtype=np.float32)
    truth = np.broadcast_to(truth_cell, (35, 6, 2, 5)).copy()
    raw = truth + np.linspace(0.5, 1.0, 5, dtype=np.float32)[None, None, None]
    raw = np.broadcast_to(raw, truth.shape).copy()
    candidate = truth + np.float32(0.2) * (raw - truth)
    weights = np.ones_like(truth, dtype=np.float64)
    indices = assets.circular_moving_block_indices(35, draws=200)
    table, effects = assets.intensity_diagnostics(
        {"raw_fuxi": raw, "raw_identity": candidate}, truth, weights, indices
    )
    assert len(table) == 10
    assert len(effects) == 15
    assert set(table.estimate_status) == {"estimated"}
    assert set(effects.estimate_status) == {"estimated"}
    assert (effects.improvement > 0.0).all()


def test_empty_intensity_strata_emit_no_estimate_without_failure() -> None:
    truth = np.full((35, 6, 2, 3), 2.0, dtype=np.float32)
    raw = truth + 1.0
    candidate = truth + 0.2
    weights = np.ones_like(truth, dtype=np.float64)
    indices = assets.circular_moving_block_indices(35, draws=50)
    table, effects = assets.intensity_diagnostics(
        {"raw_fuxi": raw, "raw_identity": candidate}, truth, weights, indices
    )
    empty_table = table.loc[~table.stratum.eq("light_1_5")]
    empty_effects = effects.loc[~effects.stratum.eq("light_1_5")]
    assert set(empty_table.estimate_status) == {"insufficient_weight/no_estimate"}
    assert empty_table[["rmse_mm_day", "mae_mm_day", "bias_mm_day"]].isna().all().all()
    assert set(empty_effects.estimate_status) == {"insufficient_weight/no_estimate"}
    assert (
        empty_effects[["improvement", "improvement_ci_low", "improvement_ci_high"]]
        .isna()
        .all()
        .all()
    )


def test_final_field_asset_saves_every_evaluated_source_array(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(19)
    shape = (35, 6, 27, 27)
    daily_shape = (35, 6, 7, 27, 27)
    support = np.zeros((27, 27), dtype=bool)
    support.reshape(-1)[:171] = True
    area = support.astype(np.float64)
    daily = generator.uniform(0.1, 20.0, size=daily_shape).astype(np.float32)
    daily_coverage = np.broadcast_to(support, daily_shape).astype(np.float32).copy()
    truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    coverage = np.min(daily_coverage, axis=2).astype(np.float32)
    raw = generator.uniform(0.1, 25.0, size=shape).astype(np.float32)
    spread = generator.uniform(0.0, 4.0, size=shape).astype(np.float32)
    t2m = generator.uniform(275.0, 315.0, size=shape).astype(np.float32)
    candidate = (0.8 * raw + 0.2 * truth).astype(np.float32)
    climatology = (0.7 * truth + 0.3).astype(np.float32)
    valid_dates = runtime._derive_valid_dates(contract.expected_initialization_dates())
    output = tmp_path / "assets"
    output.mkdir()
    assets.write_assets(
        output,
        initializations=contract.expected_initialization_dates(),
        latitude=np.linspace(39.0, 0.0, 27),
        longitude=np.linspace(60.0, 99.0, 27),
        truth=truth,
        climatology=climatology,
        weekly_coverage=coverage,
        area_weight_km2=area,
        support=support,
        raw_fuxi=raw,
        fuxi_spread=spread,
        fuxi_t2m_kelvin=t2m,
        fuxi_member_count=np.full(shape, 50, dtype=np.int16),
        fuxi_t2m_member_count=np.full(shape, 50, dtype=np.int16),
        valid_dates=valid_dates,
        selected_daily_imd=daily,
        selected_daily_coverage=daily_coverage,
        raw_identity=candidate,
        ensemble_standardized_residual=np.zeros(shape, dtype=np.float32),
    )
    with np.load(output / "independent_2025_fields.npz", allow_pickle=False) as saved:
        required = {
            "initializations",
            "latitude",
            "longitude",
            "lead_week",
            "valid_dates",
            "raw_fuxi",
            "fuxi_spread",
            "fuxi_t2m_kelvin",
            "fuxi_member_count",
            "fuxi_t2m_member_count",
            "selected_daily_imd",
            "selected_daily_coverage",
        }
        assert required.issubset(saved.files)
        assert contract.array_sha256(saved["fuxi_spread"], "<f4") == (
            contract.array_sha256(spread, "<f4")
        )
        assert contract.array_sha256(saved["selected_daily_imd"], "<f4") == (
            contract.array_sha256(daily, "<f4")
        )
        assert contract.array_sha256(
            saved["selected_daily_coverage"], "<f4"
        ) == contract.array_sha256(daily_coverage, "<f4")


def test_slurm_launchers_and_docs_encode_the_sealed_boundary() -> None:
    root = Path(contract.__file__).parents[1]
    preflight_script = (root / "slurm/preflight_raw_identity_2025.sbatch").read_text()
    final_script = (root / "slurm/evaluate_raw_identity_2025.sbatch").read_text()
    exclusion = "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17"
    for script in (preflight_script, final_script):
        assert "#SBATCH --partition=gpu_prio" in script
        assert exclusion in script
        assert "#SBATCH --gres=gpu:1" in script
    assert "--proof-receipt" not in preflight_script
    for option in (
        "--forecast-tp-store",
        "--forecast-t2m-store",
        "--imd-daily-store",
        "--output ",
        "--access-ledger",
    ):
        assert option not in final_script
    new_doc = (root / "docs/RAW_IDENTITY_2025_SEALED_WORKFLOW.md").read_text()
    old_doc = (root / "docs/INDEPENDENT_2025_CONTROL_WORKFLOW.md").read_text()
    assert "No 2025 access has been authorized" in new_doc
    assert "superseded" in old_doc.lower()


def test_legacy_evaluator_remains_separate_and_unimported() -> None:
    root = Path(contract.__file__).parents[1]
    legacy = root / "evaluate/evaluate_independent_2025_control.py"
    assert legacy.is_file()
    for module in (contract, freezer, preflight, dispatcher, runtime, assets):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "import evaluate_independent_2025_control" not in source
