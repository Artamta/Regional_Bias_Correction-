#!/usr/bin/env python3
"""Receipt-gated 2020--2021 persistence-neural/PBC comparison.

This program cannot train or select a model.  It first validates the completed
three-seed persistence experiment, the externally receipted active-mask joint
selection, and the immutable accepted PBC V2 result.  It then rebuilds every
train-only input identity and replays all nine stored validation adjustments.
Only after those gates pass may it index the reused 2020--2021 development
cases, reconstruct each neural ensemble independently, and average proper
scores across optimization seeds.  Forecasts, probabilities, and correction
parameters are never averaged or persisted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import torch

import fuxi_allseason_categorical_comparison as pbc_compare
import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_pbc_baseline as pbc_driver
import fuxi_allseason_persistence_augmented as persistence_driver
import fuxi_allseason_persistence_mask_control as mask_driver
from fuxi_pbc_core import (
    build_daily_issue_time_lags,
    calendar_fields,
    ensemble_cdf,
    is_valid_cdf_for_thresholds,
    observation_cdf,
    project_cdf_for_thresholds,
    ranked_probability_score,
    weighted_spatial_mean,
)
from fuxi_persistence_context import (
    ARMS,
    BASE_ARM,
    PERSISTENCE_LAG_ARM,
    ZERO_LAG_ARM,
    PersistenceContextBundle,
    build_persistence_context_bundle,
)
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_persistence_pbc_comparison_v1"
PERSISTENCE_EXPERIMENT = persistence_driver.EXPERIMENT
PBC_EXPERIMENT = pbc_compare.PBC_EXPERIMENT
JOINT_SELECTION_EXPERIMENT = "fuxi_allseason_persistence_mask_control_v1"
SCORING_CONTRACT_VERSION = persistence_driver.SCORING_CONTRACT_VERSION
PERSISTENCE_RECEIPT_SCHEMA = "fuxi_persistence_slurm_gate_receipt_v1"
JOINT_SELECTION_RECEIPT_SCHEMA = "fuxi_persistence_mask_control_slurm_gate_receipt_v1"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_persistence_pbc_comparison"
)
DEFAULT_PBC_MANIFEST = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_pbc_baseline_v2/"
    "full_20260822T173656Z/manifest.json"
)
ACCEPTED_PBC_MANIFEST_SHA256 = (
    "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6"
)
ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256 = (
    "3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49"
)
ACCEPTED_PBC_PERSISTENCE_LAG_SHA256 = (
    "9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba"
)
ACCEPTED_PBC_FIT_ARTIFACT_SHA256 = (
    "0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36"
)
SEEDS = (42, 43, 44)
EXPECTED_SPLITS = {"train": 1652, "validation": 196, "test": 208, "embargo": 24}
EXPECTED_SELECTED_SPLITS = {"train": 1652, "validation": 196}
COMBINED_PBC = "combined_pbc"
PERSISTENCE_PLUS_PLUS = "persistence_plus_plus"
MASK_ONLY_ARM = "mask_only_45k"
PBC_BASELINES = (COMBINED_PBC, PERSISTENCE_PLUS_PLUS)
PBC_SOURCE_METHODS = {
    COMBINED_PBC: "pbc_combined",
    PERSISTENCE_PLUS_PLUS: "persistence_plus_plus",
}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_SCHEME = "two_stage_year_resample_then_within_year_circular_blocks"
VALIDATION_REPLAY_ATOL = 1.0e-5
VALIDATION_REPLAY_RTOL = 1.0e-6
METHOD_LABELS = {
    BASE_ARM: "Exact base neural adapter",
    ZERO_LAG_ARM: "Zero-lag 45k neural control",
    PERSISTENCE_LAG_ARM: "Persistence lag-1/lag-2 neural adapter",
    COMBINED_PBC: "Combined PBC",
    PERSISTENCE_PLUS_PLUS: "Projected Persistence++",
}


class PersistencePBCComparisonError(RuntimeError):
    """Raised when an input receipt or score-only contract fails."""


@dataclass(frozen=True)
class PersistenceReceipt:
    manifest_path: Path
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    gate_receipt_path: Path
    gate_receipt: Mapping[str, Any]
    gate_receipt_sha256: str
    selection: Mapping[str, Any]
    selected_arm: str
    checkpoint_records: Mapping[tuple[str, int], Mapping[str, Any]]
    cache_path: Path


@dataclass(frozen=True)
class PBCReceipt:
    manifest_path: Path
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    gate_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class JointSelectionReceipt:
    manifest_path: Path
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    gate_receipt_path: Path
    gate_receipt: Mapping[str, Any]
    gate_receipt_sha256: str
    selection: Mapping[str, Any]
    selected_arm: str
    recomputation_evidence: Mapping[str, Any]
    matching_environment_evidence: Mapping[str, Any]
    smoke_parent_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class AcceptedPBCInputs:
    development_indices: np.ndarray
    initializations: np.ndarray
    members: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    support: pbc_compare.SupportBundle
    threshold_bundle: pbc_compare.ThresholdBundle
    quintile_thresholds: np.ndarray
    semidecile_thresholds: np.ndarray
    quintile_observed_cdf: np.ndarray
    semidecile_observed_cdf: np.ndarray
    references: pbc_compare.ReferenceScores
    stored_case_scores: pd.DataFrame
    identity_evidence: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistencePBCComparisonError(message)


def _is_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersistencePBCComparisonError(f"invalid {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return payload


def _safe_receipt_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    _require(
        not candidate.is_absolute() and ".." not in candidate.parts,
        f"unsafe {label} path: {relative!r}",
    )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    _require(
        resolved == resolved_root or resolved_root in resolved.parents,
        f"{label} escapes receipt root: {relative!r}",
    )
    return resolved


def _validate_artifact_inventory(
    root: Path, manifest: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    raw_hashes = manifest.get("artifact_sha256")
    _require(
        isinstance(raw_hashes, dict) and bool(raw_hashes),
        f"{label} artifact map missing",
    )
    assert isinstance(raw_hashes, dict)
    hashes = {str(relative): expected for relative, expected in raw_hashes.items()}
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "slurm_gate_receipt.json"}
    }
    _require(
        set(hashes) == actual, f"{label} artifact inventory is incomplete or stale"
    )
    validated: dict[str, str] = {}
    for relative, expected in sorted(hashes.items()):
        _require(_is_digest(expected), f"{label} artifact digest invalid: {relative}")
        path = _safe_receipt_path(root, str(relative), label=f"{label} artifact")
        _require(path.is_file(), f"{label} artifact missing: {relative}")
        actual_digest = sha256_file(path)
        _require(
            actual_digest == expected,
            f"{label} artifact hash mismatch: {relative}",
        )
        validated[str(relative)] = actual_digest
    return validated


def _json_equivalent(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _json_equivalent(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_equivalent(a, b, tolerance=tolerance)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return bool(left == right)


def _validate_parent_smoke_receipt(full_receipt: Mapping[str, Any]) -> dict[str, Any]:
    smoke_manifest_value = full_receipt.get("smoke_manifest_path")
    smoke_manifest_sha = full_receipt.get("smoke_manifest_sha256")
    _require(
        isinstance(smoke_manifest_value, str)
        and Path(smoke_manifest_value).is_absolute(),
        "full persistence receipt lacks its absolute smoke parent",
    )
    assert isinstance(smoke_manifest_value, str)
    smoke_manifest = Path(smoke_manifest_value).resolve()
    _require(smoke_manifest.is_file(), "persistence smoke parent manifest is missing")
    _require(_is_digest(smoke_manifest_sha), "persistence smoke parent digest invalid")
    _require(
        sha256_file(smoke_manifest) == smoke_manifest_sha,
        "persistence smoke parent manifest changed",
    )
    smoke_payload = _load_json(smoke_manifest, label="persistence smoke manifest")
    _require(
        smoke_payload.get("experiment") == PERSISTENCE_EXPERIMENT
        and smoke_payload.get("status") == "complete"
        and smoke_payload.get("mode") == "smoke"
        and smoke_payload.get("smoke") is True,
        "persistence smoke parent identity is invalid",
    )
    smoke_receipt_path = smoke_manifest.parent / "slurm_gate_receipt.json"
    _require(smoke_receipt_path.is_file(), "persistence smoke parent is unreceipted")
    smoke_receipt = _load_json(smoke_receipt_path, label="persistence smoke receipt")
    _require(
        smoke_receipt.get("schema_version") == PERSISTENCE_RECEIPT_SCHEMA
        and smoke_receipt.get("experiment") == PERSISTENCE_EXPERIMENT
        and smoke_receipt.get("status") == "post_run_audit_passed"
        and smoke_receipt.get("mode") == "smoke"
        and smoke_receipt.get("manifest_sha256") == smoke_manifest_sha,
        "persistence smoke parent receipt is invalid",
    )
    return {
        "manifest_path": str(smoke_manifest),
        "manifest_sha256": smoke_manifest_sha,
        "gate_receipt_path": str(smoke_receipt_path),
        "gate_receipt_sha256": sha256_file(smoke_receipt_path),
        "post_run_audit_passed": True,
    }


def _validate_joint_smoke_parent(
    full_receipt: Mapping[str, Any],
    full_manifest: Mapping[str, Any],
    persistence: PersistenceReceipt,
    pbc: PBCReceipt,
) -> dict[str, Any]:
    """Require the full mask addendum to descend from an audited smoke run."""

    value = full_receipt.get("mask_smoke_manifest_path")
    digest = full_receipt.get("mask_smoke_manifest_sha256")
    receipt_value = full_receipt.get("mask_smoke_receipt_path")
    receipt_digest = full_receipt.get("mask_smoke_receipt_sha256")
    _require(
        isinstance(value, str) and Path(value).is_absolute(),
        "joint full receipt lacks its absolute smoke parent",
    )
    assert isinstance(value, str)
    smoke_manifest_path = Path(value).resolve()
    _require(
        smoke_manifest_path.is_file() and _is_digest(digest),
        "joint smoke parent manifest/digest is invalid",
    )
    _require(
        sha256_file(smoke_manifest_path) == digest,
        "joint smoke parent manifest changed",
    )
    smoke_manifest = _load_json(
        smoke_manifest_path, label="joint mask-control smoke manifest"
    )
    _require(
        smoke_manifest.get("experiment") == JOINT_SELECTION_EXPERIMENT
        and smoke_manifest.get("status") == "complete"
        and smoke_manifest.get("mode") == "smoke"
        and smoke_manifest.get("smoke") is True,
        "joint smoke parent identity is invalid",
    )
    smoke_selection = smoke_manifest.get("joint_selection", {})
    _require(
        isinstance(smoke_selection, Mapping)
        and smoke_selection.get("status") == "joint_validation_selection_locked"
        and smoke_selection.get("scientific_selection") is False
        and smoke_selection.get("test_metrics_consulted") is False,
        "joint smoke parent selection boundary is invalid",
    )
    smoke_receipt_path = smoke_manifest_path.parent / "slurm_gate_receipt.json"
    _require(
        smoke_receipt_path.is_file()
        and isinstance(receipt_value, str)
        and Path(receipt_value).resolve() == smoke_receipt_path
        and _is_digest(receipt_digest)
        and sha256_file(smoke_receipt_path) == receipt_digest,
        "joint mask-control smoke parent is unreceipted",
    )
    smoke_receipt = _load_json(
        smoke_receipt_path, label="joint mask-control smoke receipt"
    )
    _require(
        smoke_receipt.get("schema_version") == JOINT_SELECTION_RECEIPT_SCHEMA
        and smoke_receipt.get("experiment") == JOINT_SELECTION_EXPERIMENT
        and smoke_receipt.get("status") == "post_run_audit_passed"
        and smoke_receipt.get("mode") == "smoke"
        and smoke_receipt.get("manifest_sha256") == digest,
        "joint mask-control smoke receipt is invalid",
    )
    _require(
        Path(str(smoke_receipt.get("manifest_path", ""))).resolve()
        == smoke_manifest_path
        and Path(str(smoke_receipt.get("output_path", ""))).resolve()
        == smoke_manifest_path.parent,
        "joint mask-control smoke receipt path binding is invalid",
    )
    smoke_artifacts = _validate_artifact_inventory(
        smoke_manifest_path.parent,
        smoke_manifest,
        label="joint mask-control smoke",
    )
    _require(
        smoke_manifest.get("source_snapshot_sha256")
        == full_manifest.get("source_snapshot_sha256")
        == smoke_receipt.get("source_snapshot_sha256"),
        "joint full/smoke source snapshots differ",
    )
    _require(
        smoke_manifest.get("addendum_plan") == full_manifest.get("addendum_plan"),
        "joint full/smoke frozen plan bindings differ",
    )
    _require(
        smoke_manifest.get("accepted_pbc") == full_manifest.get("accepted_pbc")
        and smoke_manifest.get("accepted_pbc", {}).get("manifest_sha256")
        == pbc.manifest_sha256,
        "joint full/smoke accepted-PBC bindings differ",
    )
    _require(
        smoke_manifest.get("cache", {}).get("data_sha256")
        == persistence.manifest.get("cache", {}).get("data_sha256")
        and smoke_manifest.get("observation_bundle_provenance", {}).get("sha256")
        == ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256
        and smoke_manifest.get("persistence_lag_provenance", {}).get("sha256")
        == ACCEPTED_PBC_PERSISTENCE_LAG_SHA256,
        "joint mask smoke cache/observation/lag binding changed",
    )
    parent_smoke = _validate_parent_smoke_receipt(persistence.gate_receipt)
    _require(
        smoke_manifest.get("original_v1", {}).get("manifest_sha256")
        == parent_smoke["manifest_sha256"]
        and smoke_manifest.get("original_v1", {}).get("receipt_sha256")
        == parent_smoke["gate_receipt_sha256"],
        "joint mask smoke has the wrong original-v1 smoke ancestry",
    )
    binding = full_manifest.get("mask_smoke_parent", {})
    _require(
        isinstance(binding, Mapping)
        and binding.get("manifest_path") == str(smoke_manifest_path)
        and binding.get("manifest_sha256") == digest
        and binding.get("receipt_path") == str(smoke_receipt_path)
        and binding.get("receipt_sha256") == receipt_digest
        and binding.get("output_path") == str(smoke_manifest_path.parent)
        and binding.get("source_snapshot_sha256")
        == smoke_manifest.get("source_snapshot_sha256")
        and binding.get("source_binding_sha256")
        == smoke_manifest.get("source_binding_sha256")
        and binding.get("artifact_sha256") == smoke_artifacts
        and binding.get("artifact_binding_sha256")
        == smoke_manifest.get("artifact_binding_sha256")
        and binding.get("external_receipt") == smoke_receipt
        and binding.get("original_v1_manifest_sha256")
        == parent_smoke["manifest_sha256"]
        and binding.get("original_v1_receipt_sha256")
        == parent_smoke["gate_receipt_sha256"]
        and binding.get("accepted_pbc") == smoke_manifest.get("accepted_pbc"),
        "joint full manifest lost its mask-smoke parent binding",
    )
    return {
        "manifest_path": str(smoke_manifest_path),
        "manifest_sha256": digest,
        "gate_receipt_path": str(smoke_receipt_path),
        "gate_receipt_sha256": receipt_digest,
        "source_snapshot_exact_vs_full": True,
        "artifact_inventory_bound": True,
        "original_v1_smoke_ancestry_exact": True,
        "accepted_pbc_binding_exact": True,
        "post_run_audit_passed": True,
    }


def _validate_joint_matching_environment(
    manifest: Mapping[str, Any],
    persistence: PersistenceReceipt,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Independently enforce the addendum's matched training environment."""

    parent_software = persistence.manifest.get("software", {})
    joint_software = manifest.get("software", {})
    _require(
        isinstance(parent_software, Mapping) and isinstance(joint_software, Mapping),
        "joint/parent software receipt is missing",
    )
    matched_software_keys = (
        "python",
        "platform",
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "cuda_device",
    )
    for key in matched_software_keys:
        _require(
            joint_software.get(key) == parent_software.get(key),
            f"joint/parent software or GPU mismatch: {key}",
        )
    hardware_match = manifest.get("hardware_software_match", {})
    _require(
        joint_software.get("cuda_available") is True
        and parent_software.get("cuda_available") is True
        and joint_software.get("cudnn_deterministic") is True
        and joint_software.get("cudnn_benchmark") is False
        and isinstance(hardware_match, Mapping)
        and hardware_match.get("same_physical_node") is True
        and hardware_match.get("same_python_executable") is True
        and hardware_match.get("same_gpu_model") is True
        and hardware_match.get("same_python_and_recorded_library_versions") is True
        and hardware_match.get("same_amp_mode") is True
        and hardware_match.get("same_deterministic_recipe") is True
        and hardware_match.get("matches_original") is True
        and hardware_match.get("parent_node") == persistence.gate_receipt.get("node")
        and hardware_match.get("current_node") == persistence.gate_receipt.get("node")
        and hardware_match.get("parent_cuda_device")
        == parent_software.get("cuda_device"),
        "joint CUDA/AMP/determinism environment is not canonical",
    )

    parent_training = persistence.manifest.get("training", {})
    joint_training = manifest.get("training", {})
    _require(
        isinstance(parent_training, Mapping) and isinstance(joint_training, Mapping),
        "joint/parent training receipt is missing",
    )
    matched_training_keys = (
        "batch_size",
        "member_subsample",
        "full_members_for_validation",
        "max_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "automatic_mixed_precision",
        "device",
        "optimizer",
        "scheduler",
        "gradient_clip_max_norm",
    )
    for key in matched_training_keys:
        _require(
            joint_training.get(key) == parent_training.get(key),
            f"joint/parent training contract mismatch: {key}",
        )
    _require(
        joint_training.get("automatic_mixed_precision") is True,
        "joint mask-control training did not use canonical AMP",
    )

    raw_runs = joint_training.get("runs", [])
    _require(
        isinstance(raw_runs, list) and len(raw_runs) == len(SEEDS),
        "joint mask-control run grid is incomplete",
    )
    joint_runs: dict[int, Mapping[str, Any]] = {}
    for raw_record in raw_runs:
        _require(isinstance(raw_record, Mapping), "invalid joint mask run record")
        seed = int(raw_record.get("seed", -1))
        _require(
            raw_record.get("arm") == MASK_ONLY_ARM
            and seed in SEEDS
            and seed not in joint_runs
            and int(raw_record.get("parameter_count", -1)) == 45_026,
            f"invalid joint mask run identity for seed {seed}",
        )
        expected_checkpoint = f"models/{MASK_ONLY_ARM}/seed_{seed}/checkpoints/best.pt"
        expected_adjustment = (
            f"models/{MASK_ONLY_ARM}/seed_{seed}/validation_adjustments.npz"
        )
        _require(
            raw_record.get("checkpoint") == expected_checkpoint
            and raw_record.get("validation_adjustment") == expected_adjustment,
            f"joint mask checkpoint/adjustment path mismatch for seed {seed}",
        )
        for relative_key, digest_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("validation_adjustment", "validation_adjustment_sha256"),
        ):
            relative = str(raw_record[relative_key])
            digest = raw_record.get(digest_key)
            _require(
                _is_digest(digest) and artifact_hashes.get(relative) == digest,
                f"joint mask {relative_key} is not artifact-bound for seed {seed}",
            )
        parent_record = persistence.checkpoint_records[(PERSISTENCE_LAG_ARM, seed)]
        _require(
            raw_record.get("initial_state_sha256")
            == parent_record.get("initial_state_sha256"),
            f"joint mask initial state differs from v1 candidate for seed {seed}",
        )
        _require(
            raw_record.get("training_rng_state_sha256")
            == parent_record.get("training_rng_state_sha256"),
            f"joint mask stochastic stream differs from v1 for seed {seed}",
        )
        joint_runs[seed] = raw_record
    _require(set(joint_runs) == set(SEEDS), "joint mask run seeds are incomplete")
    return {
        "matched_software_keys": list(matched_software_keys),
        "matched_training_keys": list(matched_training_keys),
        "same_physical_node_exact": True,
        "same_gpu_model_exact": True,
        "same_python_and_library_stack_exact": True,
        "same_amp_and_determinism_contract_exact": True,
        "same_seed_initial_states_exact_vs_v1_candidate": True,
        "same_seed_stochastic_streams_exact_vs_v1": True,
        "mask_checkpoint_count": len(joint_runs),
    }


def validate_joint_selection_reproduction(
    locked_selection: Mapping[str, Any],
    parent_metrics: pd.DataFrame,
    mask_metrics: pd.DataFrame,
    expected_initializations: Sequence[Any],
    *,
    expected_seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Recompute and compare every frozen joint-selector output field."""

    try:
        recomputed = mask_driver.select_joint_arm(
            parent_metrics,
            mask_metrics,
            expected_seeds=expected_seeds,
            expected_initializations=expected_initializations,
        )
    except (ValueError, KeyError, mask_driver.MaskControlError) as error:
        raise PersistencePBCComparisonError(
            f"cannot independently reproduce joint selection: {error}"
        ) from error
    for selection_key, recomputed_value in recomputed.items():
        _require(
            selection_key in locked_selection
            and _json_equivalent(
                recomputed_value,
                locked_selection[selection_key],
                tolerance=1.0e-12,
            ),
            f"joint selection is not independently reproducible: {selection_key}",
        )
    return {
        "selector": "fuxi_allseason_persistence_mask_control.select_joint_arm",
        "validation_initialization_count": len(expected_initializations),
        "seeds": [int(seed) for seed in expected_seeds],
        "all_recomputed_gate_fields_exact": True,
        "selected_arm_exact": recomputed.get("selected_arm")
        == locked_selection.get("selected_arm"),
        "development_indices_accessed": False,
    }


def validate_persistence_receipt(path: Path) -> PersistenceReceipt:
    """Validate the full selection receipt before any development access."""

    manifest_path = Path(path).resolve()
    _require(
        manifest_path.name == "manifest.json" and manifest_path.is_file(),
        "persistence receipt must be an existing manifest.json",
    )
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="persistence manifest")
    manifest_sha = sha256_file(manifest_path)
    _require(
        manifest.get("experiment") == PERSISTENCE_EXPERIMENT,
        "wrong persistence experiment",
    )
    _require(manifest.get("status") == "complete", "persistence run is incomplete")
    _require(
        manifest.get("mode") == "full" and manifest.get("smoke") is False,
        "comparison refuses smoke or reduced persistence runs",
    )
    _require(
        Path(str(manifest.get("output_path", ""))).resolve() == root,
        "persistence output identity mismatch",
    )
    _require(
        manifest.get("seeds") == list(SEEDS), "persistence full seed grid mismatch"
    )
    _require(
        manifest.get("split_counts_archive") == EXPECTED_SPLITS,
        "persistence archive splits mismatch",
    )
    _require(
        manifest.get("split_counts_selected") == EXPECTED_SELECTED_SPLITS,
        "persistence train/validation split is reduced",
    )
    contract = manifest.get("contract", {})
    _require(
        contract.get("train_years") == list(base.TRAIN_YEARS),
        "persistence training years changed",
    )
    _require(
        contract.get("validation_years") == list(base.VALIDATION_YEARS),
        "persistence validation years changed",
    )
    _require(
        contract.get("selection_data") == "2018-2019 validation only",
        "persistence selection boundary changed",
    )
    _require(
        contract.get("test_metrics_consulted") is False,
        "persistence selection consulted development",
    )
    _require(
        contract.get("development_years_not_scored") == list(base.TEST_YEARS),
        "persistence full run already scored development",
    )
    _require(
        contract.get("sealed_2025_target_opened") is False,
        "persistence run opened sealed 2025",
    )

    artifact_hashes = _validate_artifact_inventory(root, manifest, label="persistence")
    selection_path = _safe_receipt_path(root, "selection.json", label="selection")
    _require(
        "selection.json" in artifact_hashes,
        "persistence selection is not artifact-bound",
    )
    selection = _load_json(selection_path, label="persistence selection")
    _require(
        selection == manifest.get("selection"),
        "persistence selection artifact disagrees with manifest",
    )
    _require(
        selection.get("status") == "validation_selection_locked",
        "persistence selection is not durably locked",
    )
    _require(
        selection.get("scientific_selection") is True,
        "persistence full selection is not scientific",
    )
    _require(
        selection.get("test_metrics_consulted") is False,
        "locked persistence selection consulted development",
    )
    _require(
        selection.get("zero_lag_selectable") is False,
        "zero-lag control became selectable",
    )
    selected_arm = str(selection.get("selected_arm", ""))
    _require(
        selected_arm in {BASE_ARM, PERSISTENCE_LAG_ARM},
        "locked persistence arm is invalid",
    )

    validation_dates = manifest.get("retained_initializations", {}).get("validation")
    _require(
        isinstance(validation_dates, list) and len(validation_dates) == 196,
        "persistence validation inventory is incomplete",
    )
    validation_initializations = np.asarray(validation_dates, dtype="datetime64[D]")
    _require(
        np.unique(validation_initializations).size == 196
        and set(pd.DatetimeIndex(validation_initializations).year)
        == set(base.VALIDATION_YEARS),
        "persistence selection inventory is not exact 2018--2019",
    )
    metrics_relative = "metrics/validation_case_metrics.csv"
    _require(
        metrics_relative in artifact_hashes,
        "persistence validation metrics are unbound",
    )
    validation_metrics = pd.read_csv(root / metrics_relative)
    try:
        recomputed = persistence_driver.select_persistence_arm(
            validation_metrics,
            expected_seeds=SEEDS,
            expected_initializations=validation_initializations.tolist(),
        )
    except (ValueError, KeyError, base.DataContractError) as error:
        raise PersistencePBCComparisonError(
            f"cannot reproduce locked persistence selection: {error}"
        ) from error
    for selection_key, value in recomputed.items():
        _require(
            selection_key in selection
            and _json_equivalent(value, selection[selection_key]),
            f"locked persistence selection is not reproducible: {selection_key}",
        )

    training = manifest.get("training", {})
    _require(
        training.get("member_subsample") == 16,
        "persistence training member subsample changed",
    )
    _require(
        training.get("full_members_for_validation") == 51,
        "persistence checkpoint selection was not full-member",
    )
    _require(
        training.get("same_seed_45k_initial_states_exact") is True,
        "persistence 45k initialization receipt missing",
    )
    _require(
        training.get("same_seed_stochastic_training_streams_exact_across_all_arms")
        is True,
        "persistence stochastic-stream receipt missing",
    )
    runs = training.get("runs", [])
    _require(
        len(runs) == len(ARMS) * len(SEEDS), "persistence checkpoint grid is incomplete"
    )
    checkpoints: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in runs:
        arm = str(record.get("arm", ""))
        seed = int(record.get("seed", -1))
        run_key = (arm, seed)
        _require(
            arm in ARMS and seed in SEEDS and run_key not in checkpoints,
            f"invalid persistence run: {run_key}",
        )
        _require(
            int(record.get("parameter_count", -1))
            == persistence_driver.EXPECTED_PARAMETER_COUNTS[arm],
            f"persistence parameter count mismatch: {run_key}",
        )
        relative = str(record.get("checkpoint", ""))
        expected_relative = f"models/{arm}/seed_{seed}/checkpoints/best.pt"
        _require(
            relative == expected_relative,
            f"persistence checkpoint path mismatch: {run_key}",
        )
        checkpoint = _safe_receipt_path(root, relative, label="checkpoint")
        _require(
            relative in artifact_hashes and checkpoint.is_file(),
            f"persistence checkpoint is unbound: {run_key}",
        )
        _require(
            artifact_hashes[relative] == record.get("checkpoint_sha256"),
            f"persistence checkpoint receipt mismatch: {run_key}",
        )
        adjustment_relative = str(record.get("validation_adjustment", ""))
        expected_adjustment_relative = (
            f"models/{arm}/seed_{seed}/validation_adjustments.npz"
        )
        _require(
            adjustment_relative == expected_adjustment_relative,
            f"persistence validation-adjustment path mismatch: {run_key}",
        )
        adjustment = _safe_receipt_path(
            root,
            adjustment_relative,
            label="validation adjustment",
        )
        _require(
            adjustment_relative in artifact_hashes and adjustment.is_file(),
            f"persistence validation adjustment is unbound: {run_key}",
        )
        _require(
            artifact_hashes[adjustment_relative]
            == record.get("validation_adjustment_sha256"),
            f"persistence validation-adjustment receipt mismatch: {run_key}",
        )
        _require(
            _is_digest(record.get("training_rng_state_sha256")),
            f"persistence RNG receipt invalid: {run_key}",
        )
        checkpoints[run_key] = record
    _require(
        set(checkpoints) == {(arm, seed) for arm in ARMS for seed in SEEDS},
        "persistence checkpoint grid does not contain all nine models",
    )
    for seed in SEEDS:
        _require(
            checkpoints[(ZERO_LAG_ARM, seed)].get("initial_state_sha256")
            == checkpoints[(PERSISTENCE_LAG_ARM, seed)].get("initial_state_sha256"),
            f"persistence 45k initial states differ for seed {seed}",
        )
        _require(
            len(
                {
                    checkpoints[(arm, seed)].get("training_rng_state_sha256")
                    for arm in ARMS
                }
            )
            == 1,
            f"persistence stochastic streams differ for seed {seed}",
        )

    cdf_evidence = manifest.get("categorical_validation", {}).get(
        "finite_and_cdf_validity_evidence", {}
    )
    for key in (
        "threshold_and_observed_shapes_equal",
        "thresholds_finite_on_scoring_support",
        "observed_cdf_finite_on_scoring_support",
        "observed_cdf_valid_for_thresholds",
        "every_scoring_chunk_shape_matched_thresholds",
        "every_forecast_cdf_finite_where_threshold_defined",
        "every_observed_cdf_finite_where_threshold_defined",
        "every_forecast_cdf_valid_for_thresholds",
        "every_observed_cdf_valid_for_thresholds",
        "every_validation_score_finite",
    ):
        _require(
            cdf_evidence.get(key) is True,
            f"persistence validation CDF gate failed: {key}",
        )

    source_hashes = manifest.get("source_snapshot_sha256", {})
    source_map = {
        "code/src/fuxi_allseason_persistence_augmented.py": Path(
            persistence_driver.__file__
        ).resolve(),
        "code/src/fuxi_persistence_context.py": persistence_driver.HELPER_PATH,
        "code/src/fuxi_allseason_ensemble_calibration.py": Path(
            base.__file__
        ).resolve(),
        "code/src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "code/src/fuxi_allseason_pbc_baseline.py": Path(pbc_driver.__file__).resolve(),
        "code/src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "code/src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "code/slurm/run_allseason_persistence_augmented.sbatch": persistence_driver.SLURM_PATH,
        "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": persistence_driver.PLAN_PATH,
    }
    _require(
        set(source_hashes) == set(source_map),
        "persistence source/plan snapshot inventory changed",
    )
    for relative, live in source_map.items():
        digest = source_hashes.get(relative)
        _require(_is_digest(digest), f"persistence source digest invalid: {relative}")
        _require(
            artifact_hashes.get(relative) == digest,
            f"persistence source is not artifact-bound: {relative}",
        )
        _require(
            sha256_file(root / relative) == digest,
            f"persistence frozen source drift: {relative}",
        )
        _require(
            live.is_file() and sha256_file(live) == digest,
            f"live source differs from persistence receipt: {live}",
        )

    gate_path = root / "slurm_gate_receipt.json"
    _require(gate_path.is_file(), "completed persistence full is unreceipted")
    gate = _load_json(gate_path, label="persistence Slurm receipt")
    _require(
        gate.get("schema_version") == PERSISTENCE_RECEIPT_SCHEMA,
        "persistence receipt schema mismatch",
    )
    _require(
        gate.get("experiment") == PERSISTENCE_EXPERIMENT,
        "persistence receipt experiment mismatch",
    )
    _require(
        gate.get("status") == "post_run_audit_passed",
        "persistence post-run audit did not pass",
    )
    _require(gate.get("mode") == "full", "persistence gate receipt is not full")
    _require(
        gate.get("manifest_sha256") == manifest_sha,
        "persistence gate does not bind manifest",
    )
    _require(
        Path(str(gate.get("manifest_path", ""))).resolve() == manifest_path,
        "persistence receipt manifest path mismatch",
    )
    _require(
        Path(str(gate.get("output_path", ""))).resolve() == root,
        "persistence receipt output path mismatch",
    )
    _require(
        gate.get("source_snapshot_sha256") == source_hashes,
        "persistence receipt source map mismatch",
    )
    expected_source_binding = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(
        gate.get("source_binding_sha256") == expected_source_binding,
        "persistence receipt source binding mismatch",
    )
    _require(
        gate.get("cache_sha256") == persistence_driver.EXPECTED_CACHE_SHA256,
        "persistence receipt cache mismatch",
    )
    _require(
        gate.get("persistence_lag_sha256")
        == manifest.get("persistence_lag_provenance", {}).get("sha256"),
        "persistence receipt lag identity mismatch",
    )
    _require(
        gate.get("observation_bundle_sha256")
        == manifest.get("observation_bundle_provenance", {}).get("sha256"),
        "persistence receipt observation identity mismatch",
    )
    plan_digest = source_hashes["code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"]
    plan_binding = gate.get("plan_binding", {})
    _require(
        plan_binding.get("expected_sha256") == plan_digest
        and plan_binding.get("snapshot_sha256") == plan_digest
        and plan_binding.get("live_sha256") == plan_digest,
        "persistence frozen-plan binding mismatch",
    )
    _validate_parent_smoke_receipt(gate)

    cache_info = manifest.get("cache", {})
    cache_path = Path(str(cache_info.get("data_file", ""))).resolve()
    _require(cache_path.is_file(), "persistence member cache is unavailable")
    _require(
        cache_info.get("data_sha256") == persistence_driver.EXPECTED_CACHE_SHA256,
        "persistence cache SHA is noncanonical",
    )
    _require(
        cache_info.get("source_fingerprint")
        == persistence_driver.EXPECTED_SOURCE_FINGERPRINT,
        "persistence cache source fingerprint changed",
    )
    return PersistenceReceipt(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        gate_receipt_path=gate_path,
        gate_receipt=gate,
        gate_receipt_sha256=sha256_file(gate_path),
        selection=selection,
        selected_arm=selected_arm,
        checkpoint_records=checkpoints,
        cache_path=cache_path,
    )


def validate_accepted_pbc_receipt(path: Path) -> PBCReceipt:
    """Accept only the immutable, post-run-audited PBC V2 result."""

    manifest_path = Path(path).resolve()
    _require(
        manifest_path.name == "manifest.json" and manifest_path.is_file(),
        "PBC receipt must be an existing manifest.json",
    )
    manifest_sha = sha256_file(manifest_path)
    _require(
        manifest_sha == ACCEPTED_PBC_MANIFEST_SHA256,
        "PBC manifest is not the immutable accepted V2 receipt",
    )
    manifest = _load_json(manifest_path, label="accepted PBC manifest")
    _require(
        manifest.get("experiment") == PBC_EXPERIMENT, "accepted PBC experiment mismatch"
    )
    _require(manifest.get("status") == "complete", "accepted PBC result is incomplete")
    _require(
        manifest.get("mode") == "full" and manifest.get("smoke") is False,
        "accepted PBC result is not full",
    )
    root = manifest_path.parent
    _validate_artifact_inventory(root, manifest, label="accepted PBC")
    try:
        gate = pbc_compare.validate_pbc_full_receipt(root, manifest, manifest_sha)
        dates = pbc_compare._manifest_test_dates(manifest)
    except pbc_compare.ComparisonContractError as error:
        raise PersistencePBCComparisonError(
            f"accepted PBC receipt failed: {error}"
        ) from error
    _require(
        tuple(manifest.get("methods", ())) == pbc_compare.PBC_METHODS,
        "accepted PBC methods changed",
    )
    _require(
        manifest.get("observation_bundle_provenance", {}).get("sha256")
        == ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256,
        "accepted PBC observation bundle identity changed",
    )
    _require(
        manifest.get("persistence_lag_provenance", {}).get("sha256")
        == ACCEPTED_PBC_PERSISTENCE_LAG_SHA256,
        "accepted PBC persistence-lag identity changed",
    )
    _require(
        manifest.get("artifact_sha256", {}).get("models/pbc_fit.npz")
        == ACCEPTED_PBC_FIT_ARTIFACT_SHA256,
        "accepted PBC fit artifact identity changed",
    )
    _require(
        len(dates) == 208 and set(pd.DatetimeIndex(dates).year) == set(base.TEST_YEARS),
        "accepted PBC cases are not exact 2020--2021 development",
    )
    frozen_core = root / "code/src/fuxi_pbc_core.py"
    _require(
        frozen_core.is_file()
        and sha256_file(frozen_core)
        == sha256_file(PROJECT_ROOT / "src/fuxi_pbc_core.py"),
        "live PBC scoring core differs from accepted PBC V2",
    )
    return PBCReceipt(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        gate_receipt=gate,
    )


def validate_joint_selection_receipt(
    path: Path,
    persistence: PersistenceReceipt,
    pbc: PBCReceipt,
) -> JointSelectionReceipt:
    """Validate the externally receipted mask-control joint selector.

    The addendum is a validation-only attribution gate.  Its mask-only arm is
    never development-scored here, but its jointly locked selection supersedes
    the original three-arm selection for every PBC claim label.
    """

    manifest_path = Path(path).resolve()
    _require(
        manifest_path.name == "manifest.json" and manifest_path.is_file(),
        "joint selection receipt must be an existing manifest.json",
    )
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="joint mask-control manifest")
    manifest_sha = sha256_file(manifest_path)
    _require(
        manifest.get("schema_version") == "fuxi_persistence_mask_control_manifest_v1",
        "joint selection manifest schema mismatch",
    )
    _require(
        manifest.get("experiment") == JOINT_SELECTION_EXPERIMENT,
        "joint selection experiment identity mismatch",
    )
    _require(manifest.get("status") == "complete", "joint selection is incomplete")
    _require(
        manifest.get("mode") == "full" and manifest.get("smoke") is False,
        "comparison refuses smoke or reduced joint selections",
    )
    _require(
        Path(str(manifest.get("output_path", ""))).resolve() == root,
        "joint selection output identity mismatch",
    )
    artifact_hashes = _validate_artifact_inventory(
        root, manifest, label="joint mask-control"
    )

    selection_relative = "joint_selection.json"
    _require(
        selection_relative in artifact_hashes,
        "joint selection artifact is not hash-bound",
    )
    _require(
        manifest.get("joint_selection_artifact") == selection_relative
        and manifest.get("joint_selection_sha256")
        == artifact_hashes.get(selection_relative),
        "joint selection artifact receipt changed",
    )
    selection_path = _safe_receipt_path(
        root, selection_relative, label="joint selection"
    )
    selection = _load_json(selection_path, label="joint selection")
    _require(
        selection == manifest.get("joint_selection"),
        "joint selection artifact disagrees with its manifest",
    )
    _require(
        selection.get("status") == "joint_validation_selection_locked",
        "joint validation selection is not durably locked",
    )
    _require(
        selection.get("scientific_selection") is True,
        "joint validation selection is not scientific",
    )
    _require(
        selection.get("test_metrics_consulted") is False
        and selection.get("development_indices_accessed") is False,
        "joint selector consulted development metrics",
    )
    _require(
        selection.get("candidate_arm") == PERSISTENCE_LAG_ARM,
        "joint selector candidate changed",
    )
    _require(
        selection.get("mask_only_selectable") is False,
        "mask-only attribution control became selectable",
    )
    selected_arm = str(selection.get("selected_arm", ""))
    _require(
        selected_arm in {BASE_ARM, PERSISTENCE_LAG_ARM},
        "joint selected arm is invalid",
    )
    promoted = selection.get("candidate_promoted")
    _require(
        isinstance(promoted, bool)
        and promoted == (selected_arm == PERSISTENCE_LAG_ARM),
        "joint promotion flag disagrees with its selected arm",
    )
    _require(
        isinstance(selection.get("original_three_arm_gates"), Mapping)
        and isinstance(selection.get("mask_only_comparison"), Mapping),
        "joint selection lacks its frozen gate evidence",
    )

    original = manifest.get("original_v1", {})
    _require(isinstance(original, Mapping), "joint selection lacks original-v1 binding")
    _require(
        original.get("experiment") == PERSISTENCE_EXPERIMENT
        and original.get("mode") == "full"
        and Path(str(original.get("manifest_path", ""))).resolve()
        == persistence.manifest_path
        and original.get("manifest_sha256") == persistence.manifest_sha256
        and Path(str(original.get("output_path", ""))).resolve() == persistence.root,
        "joint selection does not bind the accepted original-v1 run",
    )
    _require(
        Path(str(original.get("receipt_path", ""))).resolve()
        == persistence.gate_receipt_path
        and original.get("receipt_sha256") == persistence.gate_receipt_sha256,
        "joint selection does not bind the original-v1 external receipt",
    )
    _require(
        original.get("receipt_schema") == PERSISTENCE_RECEIPT_SCHEMA
        and original.get("external_receipt") == persistence.gate_receipt,
        "joint selection copied an invalid original-v1 receipt",
    )
    original_selection_path = persistence.root / "selection.json"
    _require(
        Path(str(original.get("selection_path", ""))).resolve()
        == original_selection_path
        and original.get("selection_sha256") == sha256_file(original_selection_path),
        "joint selection does not bind the original-v1 selection artifact",
    )
    parent_metrics_path = persistence.root / "metrics/validation_case_metrics.csv"
    _require(
        Path(str(original.get("validation_case_metrics_path", ""))).resolve()
        == parent_metrics_path
        and original.get("validation_case_metrics_sha256")
        == sha256_file(parent_metrics_path),
        "joint selection does not bind the original validation scores",
    )
    parent_source_hashes = persistence.manifest.get("source_snapshot_sha256", {})
    parent_artifact_hashes = persistence.manifest.get("artifact_sha256", {})
    _require(
        original.get("source_snapshot_sha256") == parent_source_hashes
        and original.get("source_binding_sha256")
        == hashlib.sha256(
            json.dumps(
                parent_source_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "joint selection original-v1 source binding changed",
    )
    _require(
        original.get("artifact_sha256") == parent_artifact_hashes
        and original.get("artifact_binding_sha256")
        == hashlib.sha256(
            json.dumps(
                parent_artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "joint selection original-v1 artifact binding changed",
    )
    original_smoke = original.get("smoke_parent", {})
    parent_smoke = _validate_parent_smoke_receipt(persistence.gate_receipt)
    _require(
        isinstance(original_smoke, Mapping)
        and original_smoke.get("manifest_path") == parent_smoke["manifest_path"]
        and original_smoke.get("manifest_sha256") == parent_smoke["manifest_sha256"]
        and original_smoke.get("receipt_path") == parent_smoke["gate_receipt_path"]
        and original_smoke.get("receipt_sha256") == parent_smoke["gate_receipt_sha256"],
        "joint selection lost the original-v1 smoke-parent binding",
    )

    accepted_pbc = manifest.get("accepted_pbc", {})
    _require(
        isinstance(accepted_pbc, Mapping),
        "joint selection lacks accepted-PBC binding",
    )
    _require(
        Path(str(accepted_pbc.get("manifest_path", ""))).resolve() == pbc.manifest_path
        and accepted_pbc.get("manifest_sha256")
        == pbc.manifest_sha256
        == ACCEPTED_PBC_MANIFEST_SHA256
        and Path(str(accepted_pbc.get("fit_path", ""))).resolve()
        == pbc.root / "models/pbc_fit.npz",
        "joint selection does not bind the immutable accepted PBC manifest",
    )
    _require(
        accepted_pbc.get("observation_bundle_sha256")
        == ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256
        and accepted_pbc.get("persistence_lag_sha256")
        == ACCEPTED_PBC_PERSISTENCE_LAG_SHA256
        and accepted_pbc.get("fit_sha256") == ACCEPTED_PBC_FIT_ARTIFACT_SHA256,
        "joint selection accepted-PBC identities changed",
    )
    _require(
        manifest.get("seeds") == list(SEEDS)
        and manifest.get("split_counts_archive") == EXPECTED_SPLITS
        and manifest.get("split_counts_selected") == EXPECTED_SELECTED_SPLITS,
        "joint mask-control full inventory is reduced",
    )
    _require(
        manifest.get("retained_initializations")
        == persistence.manifest.get("retained_initializations"),
        "joint mask-control train/validation inventories changed",
    )
    contract = manifest.get("contract", {})
    _require(
        isinstance(contract, Mapping)
        and contract.get("train_years") == list(base.TRAIN_YEARS)
        and contract.get("validation_years") == list(base.VALIDATION_YEARS)
        and contract.get("selection_data") == "2018-2019 validation only"
        and contract.get("test_metrics_consulted") is False
        and contract.get("development_years_not_scored") == list(base.TEST_YEARS)
        and contract.get("sealed_2025_target_opened") is False
        and contract.get("mask_only_selectable") is False,
        "joint mask-control temporal/selection contract changed",
    )
    _require(
        manifest.get("persistence_lag_provenance")
        == persistence.manifest.get("persistence_lag_provenance")
        == pbc.manifest.get("persistence_lag_provenance"),
        "joint persistence-lag provenance changed",
    )
    _require(
        manifest.get("observation_bundle_provenance")
        == persistence.manifest.get("observation_bundle_provenance")
        == pbc.manifest.get("observation_bundle_provenance"),
        "joint observation provenance changed",
    )
    _require(
        manifest.get("persistence_context_provenance")
        == persistence.manifest.get("persistence_context_provenance"),
        "joint normalized persistence context changed",
    )
    parent_cache = persistence.manifest.get("cache", {})
    joint_cache = manifest.get("cache", {})
    _require(
        isinstance(parent_cache, Mapping)
        and isinstance(joint_cache, Mapping)
        and all(joint_cache.get(key) == value for key, value in parent_cache.items())
        and joint_cache.get("actual_data_sha256")
        == persistence_driver.EXPECTED_CACHE_SHA256,
        "joint cache provenance changed",
    )
    temporal = manifest.get("temporal_evidence", {})
    _require(
        isinstance(temporal, Mapping)
        and temporal.get("development_indices_accessed") is False,
        "joint addendum accessed development indices",
    )
    reproduction = manifest.get("validation_reproduction", {})
    _require(
        isinstance(reproduction, Mapping)
        and reproduction.get("status") == "all_original_validation_artifacts_reproduced"
        and reproduction.get("all_available_parent_checkpoints_reproduced") is True
        and reproduction.get("all_adjustment_arrays_reproduced_byte_exact") is True
        and reproduction.get("original_selection_recomputed_exact") is True
        and reproduction.get("persisted_score_reproduction", {}).get("passes") is True,
        "joint addendum did not reproduce every original validation artifact",
    )
    mask_cdf_evidence = manifest.get("categorical_validation", {}).get(
        "finite_and_cdf_validity_evidence", {}
    )
    joint_categorical = manifest.get("categorical_validation", {})
    parent_categorical = persistence.manifest.get("categorical_validation", {})
    _require(
        joint_categorical.get("threshold_fit_indices")
        == parent_categorical.get("threshold_fit_indices")
        and joint_categorical.get("threshold_fit_sha256")
        == parent_categorical.get("threshold_fit_sha256"),
        "joint validation threshold fit changed",
    )
    for evidence_key in (
        "every_scoring_chunk_shape_matched_thresholds",
        "every_forecast_cdf_finite_where_threshold_defined",
        "every_observed_cdf_finite_where_threshold_defined",
        "every_forecast_cdf_valid_for_thresholds",
        "every_observed_cdf_valid_for_thresholds",
        "every_validation_score_finite",
    ):
        _require(
            mask_cdf_evidence.get(evidence_key) is True,
            f"joint mask validation CDF gate failed: {evidence_key}",
        )
    _require(
        manifest.get("scoring_support_rebuild", {}).get("all_arrays_rebuilt_byte_exact")
        is True
        and manifest.get("persistence_normalization_rebuild", {}).get(
            "all_arrays_rebuilt_byte_exact"
        )
        is True,
        "joint addendum did not byte-rebuild support/normalization artifacts",
    )

    evaluation = manifest.get("evaluation", {})
    _require(
        isinstance(evaluation, Mapping),
        "joint selector evaluation receipt is missing",
    )
    mask_metrics_relative = str(
        evaluation.get("mask_validation_case_metrics_artifact", "")
    )
    _require(
        mask_metrics_relative == "metrics/mask_validation_case_metrics.csv"
        and evaluation.get("mask_validation_case_metrics_sha256")
        == artifact_hashes.get(mask_metrics_relative),
        "joint selector mask validation-score receipt changed",
    )
    _require(
        mask_metrics_relative in artifact_hashes,
        "joint selector mask validation scores are not hash-bound",
    )
    mask_metrics_path = _safe_receipt_path(
        root,
        mask_metrics_relative,
        label="mask validation metrics",
    )
    expected_dates = persistence.manifest.get("retained_initializations", {}).get(
        "validation"
    )
    _require(
        isinstance(expected_dates, list) and len(expected_dates) == 196,
        "joint selector lacks the exact validation inventory",
    )
    recomputation_evidence = validate_joint_selection_reproduction(
        selection,
        pd.read_csv(parent_metrics_path),
        pd.read_csv(mask_metrics_path),
        expected_dates,
        expected_seeds=SEEDS,
    )
    recomputation_evidence.update(
        {
            "parent_validation_case_metrics_path": str(parent_metrics_path),
            "parent_validation_case_metrics_sha256": sha256_file(parent_metrics_path),
            "mask_validation_case_metrics_path": str(mask_metrics_path),
            "mask_validation_case_metrics_sha256": sha256_file(mask_metrics_path),
        }
    )

    source_hashes = manifest.get("source_snapshot_sha256", {})
    _require(
        isinstance(source_hashes, Mapping) and bool(source_hashes),
        "joint selection source snapshot is missing",
    )
    expected_source_hashes = {
        f"code/{relative}": source
        for relative, source in mask_driver.SOURCE_PATHS.items()
    }
    _require(
        set(source_hashes) == set(expected_source_hashes),
        "joint selection source/plan/launcher inventory changed",
    )
    expected_joint_source_binding = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_joint_artifact_binding = hashlib.sha256(
        json.dumps(
            artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _require(
        manifest.get("source_binding_sha256") == expected_joint_source_binding
        and manifest.get("artifact_binding_sha256") == expected_joint_artifact_binding,
        "joint source/artifact aggregate binding changed",
    )
    required_source_paths = {
        "src/fuxi_allseason_persistence_mask_control.py",
        "src/fuxi_persistence_mask_control.py",
        "src/fuxi_allseason_persistence_augmented.py",
        "src/fuxi_persistence_context.py",
        "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py",
        "src/fuxi_allseason_pbc_baseline.py",
        "src/fuxi_pbc_core.py",
        "src/project_paths.py",
        "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md",
        "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
        "slurm/run_allseason_persistence_mask_control.sbatch",
    }
    _require(
        required_source_paths <= set(mask_driver.SOURCE_PATHS),
        "joint source contract omits a required driver/helper/plan/launcher",
    )
    plan = manifest.get("addendum_plan", {})
    plan_snapshot = "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
    _require(
        isinstance(plan, Mapping)
        and Path(str(plan.get("path", ""))).resolve()
        == mask_driver.ADDENDUM_PLAN_PATH.resolve()
        and plan.get("expected_sha256") == mask_driver.ADDENDUM_PLAN_SHA256
        and plan.get("live_sha256") == mask_driver.ADDENDUM_PLAN_SHA256
        and plan.get("snapshot_path") == plan_snapshot
        and plan.get("snapshot_sha256")
        == source_hashes.get(plan_snapshot)
        == mask_driver.ADDENDUM_PLAN_SHA256,
        "joint frozen addendum-plan binding changed",
    )
    for relative_value, digest in source_hashes.items():
        relative = str(relative_value)
        _require(_is_digest(digest), f"joint source digest invalid: {relative}")
        _require(
            artifact_hashes.get(relative) == digest
            and sha256_file(root / relative) == digest,
            f"joint source snapshot is not artifact-bound: {relative}",
        )
        live = expected_source_hashes[relative]
        _require(
            live.is_file() and sha256_file(live) == digest,
            f"live source differs from joint selection receipt: {live}",
        )

    gate_path = root / "slurm_gate_receipt.json"
    _require(gate_path.is_file(), "joint selection lacks an external Slurm receipt")
    gate = _load_json(gate_path, label="joint mask-control Slurm receipt")
    _require(
        gate.get("schema_version") == JOINT_SELECTION_RECEIPT_SCHEMA
        and gate.get("experiment") == JOINT_SELECTION_EXPERIMENT
        and gate.get("status") == "post_run_audit_passed"
        and gate.get("mode") == "full",
        "joint selection external receipt did not pass its full audit",
    )
    _require(
        gate.get("manifest_sha256") == manifest_sha
        and Path(str(gate.get("manifest_path", ""))).resolve() == manifest_path
        and Path(str(gate.get("output_path", ""))).resolve() == root,
        "joint external receipt does not bind its manifest/output",
    )
    _require(
        gate.get("source_snapshot_sha256") == source_hashes,
        "joint external receipt source map mismatch",
    )
    _require(
        gate.get("source_binding_sha256") == expected_joint_source_binding,
        "joint external receipt source binding mismatch",
    )
    _require(
        gate.get("original_v1_manifest_sha256") == persistence.manifest_sha256
        and gate.get("original_v1_receipt_sha256") == persistence.gate_receipt_sha256
        and gate.get("accepted_pbc_manifest_sha256") == pbc.manifest_sha256
        and gate.get("accepted_pbc_slurm_receipt_sha256")
        == sha256_file(pbc.root / "slurm_gate_receipt.json")
        and gate.get("accepted_pbc_fit_sha256") == ACCEPTED_PBC_FIT_ARTIFACT_SHA256,
        "joint external receipt lost an immutable parent binding",
    )
    _require(
        gate.get("cache_sha256") == persistence_driver.EXPECTED_CACHE_SHA256
        and gate.get("observation_bundle_sha256")
        == ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256
        and gate.get("persistence_lag_sha256") == ACCEPTED_PBC_PERSISTENCE_LAG_SHA256
        and gate.get("persistence_context_sha256")
        == manifest.get("persistence_context_provenance", {}).get("sha256")
        and gate.get("base_context_and_weekly_climatology_sha256")
        == manifest.get("base_context_and_weekly_climatology_provenance", {}).get(
            "sha256"
        ),
        "joint external receipt lost cache/observation/context bindings",
    )
    _require(
        gate.get("artifact_sha256") == artifact_hashes
        and gate.get("artifact_binding_sha256") == expected_joint_artifact_binding,
        "joint external receipt artifact binding mismatch",
    )
    _require(
        gate.get("current_gpu_receipt") == gate.get("gpu")
        and gate.get("parent_gpu_receipt") == persistence.gate_receipt.get("gpu")
        and gate.get("gpu_model_memory_driver_exact") is True,
        "joint external receipt lost exact current/parent GPU evidence",
    )
    expected_mask_context_contract = {
        "normalized_rainfall_channels_forced_exact_zero": True,
        "availability_channels_are_issue_safe_support_masked_true_masks": True,
        "added_channel_order": list(mask_driver.MASK_ONLY_ADDED_CHANNEL_NAMES),
    }
    _require(
        gate.get("mask_only_context_provenance_sha256")
        == manifest.get("mask_only_context_provenance", {}).get("sha256")
        and gate.get("mask_only_context_contract") == expected_mask_context_contract,
        "joint external receipt lost mask-only context provenance/contract",
    )
    launcher = mask_driver.SOURCE_PATHS[
        "slurm/run_allseason_persistence_mask_control.sbatch"
    ].resolve()
    _require(
        Path(str(gate.get("launcher_path", ""))).resolve() == launcher
        and gate.get("launcher_sha256")
        == sha256_file(launcher)
        == source_hashes.get(
            "code/slurm/run_allseason_persistence_mask_control.sbatch"
        ),
        "joint external receipt launcher binding mismatch",
    )
    _require(
        gate.get("addendum_plan_binding") == manifest.get("addendum_plan"),
        "joint external receipt addendum-plan binding mismatch",
    )
    firewalls = gate.get("firewalls", {})
    for firewall in (
        "same_physical_parent_node",
        "nvidia_a30",
        "automatic_mixed_precision",
        "deterministic_recipe",
    ):
        _require(
            isinstance(firewalls, Mapping) and firewalls.get(firewall) is True,
            f"joint external receipt firewall failed: {firewall}",
        )
    _require(
        firewalls.get("test_metrics_consulted") is False
        and firewalls.get("development_indices_accessed") is False
        and firewalls.get("mask_only_selectable") is False
        and firewalls.get("sealed_2025_target_opened") is False,
        "joint external receipt temporal/selectability firewall failed",
    )
    matching_environment_evidence = _validate_joint_matching_environment(
        manifest,
        persistence,
        artifact_hashes,
    )
    smoke_parent_evidence = _validate_joint_smoke_parent(
        gate,
        manifest,
        persistence,
        pbc,
    )
    return JointSelectionReceipt(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        gate_receipt_path=gate_path,
        gate_receipt=gate,
        gate_receipt_sha256=sha256_file(gate_path),
        selection=selection,
        selected_arm=selected_arm,
        recomputation_evidence=recomputation_evidence,
        matching_environment_evidence=matching_environment_evidence,
        smoke_parent_evidence=smoke_parent_evidence,
    )


def average_seed_scores(
    frame: pd.DataFrame,
    value_columns: Sequence[str] = ("rps",),
    *,
    expected_methods: Sequence[str] = ARMS,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    """Average scores only after proving a complete method/seed/case grid."""

    required = {"method", "seed", "initialization", "lead_week", *value_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed score table lacks columns: {missing}")
    methods = tuple(expected_methods)
    expected_seed_set = {int(seed) for seed in seeds}
    if set(frame.method) != set(methods):
        raise ValueError("seed score table has the wrong method inventory")
    values = frame.loc[:, value_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("seed score table contains non-finite values")
    selected = frame.copy()
    selected["seed"] = pd.to_numeric(selected.seed, errors="raise").astype(int)
    selected["lead_week"] = pd.to_numeric(selected.lead_week, errors="raise").astype(
        int
    )
    if set(selected.seed) != expected_seed_set or set(selected.lead_week) != set(
        range(1, 7)
    ):
        raise ValueError("seed score table has wrong seeds or leads")
    identity = ["method", "seed", "initialization", "lead_week"]
    if selected.duplicated(identity).any():
        raise ValueError("seed score table contains duplicate rows")
    reference_keys: pd.DataFrame | None = None
    for method in methods:
        rows = selected.loc[selected.method == method]
        for seed in sorted(expected_seed_set):
            keys = (
                rows.loc[rows.seed == seed, ["initialization", "lead_week"]]
                .sort_values(["initialization", "lead_week"])
                .reset_index(drop=True)
            )
            if reference_keys is None:
                reference_keys = keys
            elif not keys.equals(reference_keys):
                raise ValueError(
                    "seed scores are not paired on one case/lead inventory"
                )
    counts = selected.groupby(["method", "initialization", "lead_week"]).seed.agg(
        ["size", "nunique"]
    )
    if not (
        (counts["size"] == len(expected_seed_set))
        & (counts["nunique"] == len(expected_seed_set))
    ).all():
        raise ValueError("one or more case scores lacks every optimization seed")
    result = (
        selected.groupby(["method", "initialization", "lead_week"], as_index=False)[
            list(value_columns)
        ]
        .mean()
        .sort_values(["method", "initialization", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(1, "seed", "mean_of_seed_scores_42_43_44")
    result.insert(
        3,
        "year",
        pd.DatetimeIndex(result.initialization).year.to_numpy(dtype=np.int64),
    )
    return result


def paired_pbc_bootstrap(
    case_scores: pd.DataFrame,
    methods: Sequence[str],
    baselines: Sequence[str] = PBC_BASELINES,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Flexible paired RPS intervals with the frozen two-stage resampler."""

    required = {"method", "initialization", "lead_week", "rps"}
    if not required.issubset(case_scores):
        raise ValueError("bootstrap case scores lack required columns")
    if samples < 1 or block_length < 1:
        raise ValueError("bootstrap samples and block length must be positive")
    method_names = tuple(dict.fromkeys(str(value) for value in methods))
    baseline_names = tuple(dict.fromkeys(str(value) for value in baselines))
    if (
        not method_names
        or not baseline_names
        or set(method_names) & set(baseline_names)
    ):
        raise ValueError("bootstrap methods/baselines must be nonempty and disjoint")
    needed = set(method_names) | set(baseline_names)
    selected = case_scores.loc[case_scores.method.isin(needed)].copy()
    if set(selected.method) != needed:
        raise ValueError("bootstrap method inventory is incomplete")
    selected["lead_week"] = pd.to_numeric(selected.lead_week, errors="raise").astype(
        int
    )
    if set(selected.lead_week) != set(range(1, 7)):
        raise ValueError("bootstrap requires exact leads 1..6")
    if selected.duplicated(["method", "initialization", "lead_week"]).any():
        raise ValueError("bootstrap case scores are not uniquely paired")
    if not np.isfinite(selected.rps.to_numpy(dtype=np.float64)).all() or np.any(
        selected.rps.to_numpy(dtype=np.float64) < 0.0
    ):
        raise ValueError("bootstrap RPS values must be finite and nonnegative")
    case_order = np.asarray(
        sorted(selected.initialization.unique()), dtype="datetime64[D]"
    )
    years = pd.DatetimeIndex(case_order).year.to_numpy()
    unique_years = np.unique(years)
    year_counts = np.asarray(
        [np.count_nonzero(years == year) for year in unique_years], dtype=np.int64
    )
    if (
        len(case_order) != 208
        or unique_years.tolist() != list(base.TEST_YEARS)
        or np.unique(year_counts).size != 1
    ):
        raise ValueError("bootstrap requires 208 equally split 2020--2021 cases")
    labels = [np.datetime_as_string(value, unit="D") for value in case_order]
    expected_rows = len(case_order) * 6
    for method in needed:
        if len(selected.loc[selected.method == method]) != expected_rows:
            raise ValueError(f"bootstrap method {method!r} lacks a complete case grid")
    draws = base._block_bootstrap_indices(
        case_order,
        n_resamples=samples,
        block_length=block_length,
        seed=seed,
    )
    _require(
        draws.shape == (samples, len(case_order)),
        "bootstrap draw geometry is invalid",
    )
    scopes: list[tuple[str, tuple[int, ...]]] = [("W1-W6", tuple(range(1, 7)))] + [
        (f"W{lead}", (lead,)) for lead in range(1, 7)
    ]
    records: list[dict[str, Any]] = []
    for lead_scope, leads in scopes:
        rows = selected.loc[selected.lead_week.isin(leads)]
        per_case = (
            rows.groupby(["initialization", "method"], as_index=False)
            .rps.mean()
            .pivot(index="initialization", columns="method", values="rps")
            .reindex(labels)
        )
        if per_case.loc[:, list(needed)].isna().any().any():
            raise ValueError("bootstrap comparison is not completely paired")
        for method in method_names:
            method_values = per_case[method].to_numpy(dtype=np.float64)
            for baseline in baseline_names:
                baseline_values = per_case[baseline].to_numpy(dtype=np.float64)
                baseline_point = float(np.mean(baseline_values))
                method_point = float(np.mean(method_values))
                if baseline_point <= 0.0:
                    raise ValueError("bootstrap baseline RPS must be positive")
                baseline_draw = np.mean(baseline_values[draws], axis=1)
                method_draw = np.mean(method_values[draws], axis=1)
                if np.any(baseline_draw <= 0.0):
                    raise ValueError("bootstrap produced a zero baseline denominator")
                reductions = 1.0 - method_draw / baseline_draw
                records.append(
                    {
                        "score_contract": SCORING_CONTRACT_VERSION,
                        "method": method,
                        "baseline": baseline,
                        "lead_scope": lead_scope,
                        "lead_week": 0 if lead_scope == "W1-W6" else int(leads[0]),
                        "method_rps": method_point,
                        "baseline_rps": baseline_point,
                        "rps_reduction_fraction": float(
                            1.0 - method_point / baseline_point
                        ),
                        "rps_reduction_pct": float(
                            100.0 * (1.0 - method_point / baseline_point)
                        ),
                        "ci_lower_95": float(np.quantile(reductions, 0.025)),
                        "ci_upper_95": float(np.quantile(reductions, 0.975)),
                        "bootstrap_probability_improvement": float(
                            np.mean(reductions > 0.0)
                        ),
                        "paired_initializations": int(len(case_order)),
                        "bootstrap_samples": int(samples),
                        "block_length_initializations": int(block_length),
                        "bootstrap_seed": int(seed),
                        "bootstrap_scheme": BOOTSTRAP_SCHEME,
                        "resampling_unit": "initialization with all six leads grouped",
                        "development_years": list(base.TEST_YEARS),
                        "initializations_per_year": int(year_counts[0]),
                    }
                )
    return pd.DataFrame.from_records(records)


def classify_pbc_claims(
    bootstrap: pd.DataFrame,
    candidate_methods: Mapping[str, str] | Sequence[str],
) -> dict[str, Any]:
    """Apply the frozen interval-only language gates to pooled comparisons."""

    roles = (
        {str(role): str(method) for role, method in candidate_methods.items()}
        if isinstance(candidate_methods, Mapping)
        else {str(method): str(method) for method in candidate_methods}
    )
    if not roles:
        raise ValueError("claim classification requires at least one candidate")
    required = {
        "method",
        "baseline",
        "lead_scope",
        "rps_reduction_fraction",
        "ci_lower_95",
        "ci_upper_95",
    }
    if not required.issubset(bootstrap):
        raise ValueError("bootstrap table lacks claim columns")
    output: dict[str, Any] = {}
    for role, method in roles.items():
        method_rows = bootstrap.loc[bootstrap.method == method]
        comparison_records = []
        for _, row in method_rows.sort_values(["lead_week", "baseline"]).iterrows():
            lower = float(row.ci_lower_95)
            upper = float(row.ci_upper_95)
            if lower > 0.0:
                label = "better"
            elif upper < 0.0:
                label = "loss"
            else:
                label = "unresolved"
            comparison_records.append(
                {
                    "baseline": str(row.baseline),
                    "lead_scope": str(row.lead_scope),
                    "rps_reduction_fraction": float(row.rps_reduction_fraction),
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "claim_label": label,
                }
            )
        pooled = {
            row["baseline"]: row
            for row in comparison_records
            if row["lead_scope"] == "W1-W6"
        }
        if set(pooled) != set(PBC_BASELINES):
            raise ValueError(f"candidate {method!r} lacks both pooled PBC comparisons")
        better_combined = float(cast(float, pooled[COMBINED_PBC]["ci_lower_95"])) > 0.0
        better_both = all(
            float(cast(float, pooled[name]["ci_lower_95"])) > 0.0
            for name in PBC_BASELINES
        )
        output[role] = {
            "method": method,
            "better_than_combined_pbc_pooled": bool(better_combined),
            "better_than_strongest_classical_baseline_pooled": bool(better_both),
            "pooled_language": (
                "better_than_strongest_classical_baseline"
                if better_both
                else (
                    "better_than_combined_pbc_only"
                    if better_combined
                    else "tie_loss_or_unresolved"
                )
            ),
            "comparisons": comparison_records,
        }
    return output


def _categorical_case_frame(
    method: str,
    seed: int,
    rps: np.ndarray,
    initializations: np.ndarray,
) -> pd.DataFrame:
    if rps.shape != (len(initializations), 6) or not np.isfinite(rps).all():
        raise PersistencePBCComparisonError(
            "neural categorical RPS has invalid geometry"
        )
    rows = []
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    for case, initialization in enumerate(initializations):
        for lead in range(6):
            rows.append(
                {
                    "split": "test_development",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": int(seed),
                    "score_contract": SCORING_CONTRACT_VERSION,
                    "initialization": np.datetime_as_string(initialization, unit="D"),
                    "year": int(years[case]),
                    "lead_week": lead + 1,
                    "rps": float(rps[case, lead]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _checkpoint_path(receipt: PersistenceReceipt, arm: str, seed: int) -> Path:
    record = receipt.checkpoint_records[(arm, seed)]
    return _safe_receipt_path(
        receipt.root, str(record["checkpoint"]), label="persistence checkpoint"
    )


def reconstruct_and_score_arm_seed(
    receipt: PersistenceReceipt,
    arm: str,
    seed: int,
    cache: base.MemberCache,
    weekly_truth: np.ndarray,
    context: PersistenceContextBundle,
    pbc_inputs: AcceptedPBCInputs,
    *,
    device: torch.device,
    batch_size: int,
    evaluation_batch_size: int,
    cdf_chunk_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct one checkpoint independently and retain only its scores."""

    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unknown persistence arm or seed")
    base.METHOD_LABELS[arm] = METHOD_LABELS[arm]
    checkpoint = _checkpoint_path(receipt, arm, seed)
    checkpoint_sha = sha256_file(checkpoint)
    expected = receipt.checkpoint_records[(arm, seed)]
    _require(
        checkpoint_sha == expected.get("checkpoint_sha256"),
        "checkpoint changed after receipt validation",
    )
    model = persistence_driver.load_checkpoint_model(checkpoint, arm, seed, device)
    delta, log_spread = persistence_driver.predict_adjustments(
        model,
        arm,
        cache.members,
        weekly_truth,
        context,
        pbc_inputs.development_indices,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )
    del model
    if not np.isfinite(delta).all() or not np.isfinite(log_spread).all():
        raise PersistencePBCComparisonError(
            f"non-finite development adjustment for {arm}, seed {seed}"
        )
    spread = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
    corrected = base.apply_affine_log_calibration(pbc_inputs.members, delta, spread)
    if not np.isfinite(corrected).all() or np.any(corrected < 0.0):
        raise PersistencePBCComparisonError(
            f"invalid reconstructed ensemble for {arm}, seed {seed}"
        )
    continuous, _ = base.evaluate_ensemble(
        arm,
        corrected,
        pbc_inputs.truth,
        pbc_inputs.climatology,
        pbc_inputs.initializations,
        pbc_inputs.support.weights,
        chunk_size=evaluation_batch_size,
        seed_label=seed,
    )
    forecast_cdf = ensemble_cdf(
        corrected,
        pbc_inputs.quintile_thresholds,
        chunk_size=cdf_chunk_size,
    )
    cdf_valid = is_valid_cdf_for_thresholds(
        forecast_cdf, pbc_inputs.quintile_thresholds, axis=2
    )
    if not cdf_valid:
        raise PersistencePBCComparisonError(
            f"invalid equality-aware CDF for {arm}, seed {seed}"
        )
    rps = weighted_spatial_mean(
        ranked_probability_score(
            forecast_cdf,
            pbc_inputs.quintile_observed_cdf,
            pbc_inputs.quintile_thresholds,
        ),
        pbc_inputs.support.weights,
    )
    categorical = _categorical_case_frame(arm, seed, rps, pbc_inputs.initializations)
    reconstruction = {
        "arm": arm,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "parameter_count": int(expected["parameter_count"]),
        "development_initializations": int(len(pbc_inputs.initializations)),
        "members_per_ensemble": int(corrected.shape[1]),
        "adjustments_finite": True,
        "reconstructed_members_finite_nonnegative": True,
        "quintile_cdf_valid_for_exact_pbc_thresholds": True,
        "continuous_scores_finite": bool(
            np.isfinite(
                continuous.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
            ).all()
        ),
        "categorical_scores_finite": bool(np.isfinite(rps).all()),
        "forecasts_or_parameters_written": False,
        "forecasts_or_parameters_averaged_across_seeds": False,
    }
    _require(
        bool(reconstruction["continuous_scores_finite"]),
        "continuous scores are non-finite",
    )
    del delta, log_spread, spread, corrected, forecast_cdf, rps
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return continuous, categorical, reconstruction


def load_accepted_pbc_inputs(
    receipt: PBCReceipt,
    cache: base.MemberCache,
    observations: base.ObservationBundle,
    development_indices: np.ndarray,
    support: pbc_compare.SupportBundle,
    thresholds: pbc_compare.ThresholdBundle,
) -> AcceptedPBCInputs:
    """Materialize the exact accepted PBC 208-case scoring geometry."""

    selected = np.asarray(development_indices, dtype=np.int64)
    _require(selected.shape == (208,), "accepted PBC comparison needs 208 cases")
    initializations = cache.initializations[selected]
    expected_dates = pbc_compare._manifest_test_dates(receipt.manifest)
    _require(
        np.array_equal(initializations, expected_dates),
        "cache development dates differ from accepted PBC",
    )
    _require(
        set(pd.DatetimeIndex(initializations).year) == set(base.TEST_YEARS)
        and not np.any(pd.DatetimeIndex(initializations).year >= 2022),
        "comparison attempted a 2022+ initialization cohort",
    )
    quintile_thresholds = calendar_fields(thresholds.quintile, initializations, 6)
    semidecile_thresholds = calendar_fields(thresholds.semidecile, initializations, 6)
    quintile_climatology = project_cdf_for_thresholds(
        calendar_fields(thresholds.quintile, initializations, 6, empirical=True),
        quintile_thresholds,
        axis=2,
    )
    semidecile_climatology = project_cdf_for_thresholds(
        calendar_fields(thresholds.semidecile, initializations, 6, empirical=True),
        semidecile_thresholds,
        axis=2,
    )
    truth = np.asarray(observations.weekly_truth[selected], dtype=np.float32)
    climatology = np.asarray(
        observations.weekly_climatology[selected], dtype=np.float32
    )
    members = base.materialize_cases(cache.members, selected)
    quintile_observed = observation_cdf(truth, quintile_thresholds)
    semidecile_observed = observation_cdf(truth, semidecile_thresholds)
    _require(
        is_valid_cdf_for_thresholds(quintile_observed, quintile_thresholds, axis=2)
        and is_valid_cdf_for_thresholds(
            semidecile_observed, semidecile_thresholds, axis=2
        ),
        "development observations violate accepted PBC threshold geometry",
    )
    references = pbc_compare.reference_scores(
        thresholds.quintile,
        thresholds.semidecile,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        quintile_climatology,
        semidecile_climatology,
        support.weights,
    )
    case_path, case_sha = pbc_compare._artifact(
        receipt.root, receipt.manifest, "metrics/quintile_case_scores.csv"
    )
    stored_case_scores = pd.read_csv(case_path)
    return AcceptedPBCInputs(
        development_indices=selected.copy(),
        initializations=initializations.copy(),
        members=members,
        truth=truth,
        climatology=climatology,
        support=support,
        threshold_bundle=thresholds,
        quintile_thresholds=quintile_thresholds,
        semidecile_thresholds=semidecile_thresholds,
        quintile_observed_cdf=quintile_observed,
        semidecile_observed_cdf=semidecile_observed,
        references=references,
        stored_case_scores=stored_case_scores,
        identity_evidence={
            "accepted_pbc_manifest_sha256": receipt.manifest_sha256,
            "case_score_artifact_sha256": case_sha,
            "development_case_count": 208,
            "development_years": list(base.TEST_YEARS),
            "all_six_leads_grouped": True,
            "quintile_observed_cdf_valid_for_thresholds": True,
            "semidecile_observed_cdf_valid_for_thresholds": True,
            "threshold_fit_indices_equal_exact_training_split": True,
            "no_2022_plus_initialization_cohort": True,
            "sealed_2025_opened": False,
        },
    )


def _fresh_context_provenance(
    context: PersistenceContextBundle,
    observations: base.ObservationBundle,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Content-bind derived inputs omitted by the original v1 manifest."""

    base_context = context.base_context
    base_context_provenance = pbc_driver.array_bundle_provenance(
        {
            "normalized_climatology": base_context.normalized_climatology,
            "climatology_mean_by_lead": base_context.climatology_mean_by_lead,
            "climatology_std_by_lead": base_context.climatology_std_by_lead,
            "latitude_scaled": base_context.latitude_scaled,
            "longitude_scaled": base_context.longitude_scaled,
            "season_sin": base_context.season_sin,
            "season_cos": base_context.season_cos,
            "lead_scaled": base_context.lead_scaled,
            "support": base_context.support,
        },
        bundle_name="fresh_full_base_neural_context",
    )
    climatology_provenance = pbc_driver.array_bundle_provenance(
        {"weekly_climatology": observations.weekly_climatology},
        bundle_name="fresh_full_weekly_training_climatology",
    )
    return base_context_provenance, climatology_provenance


def validate_validation_adjustment_replay(
    receipt: PersistenceReceipt,
    cache: base.MemberCache,
    observations: base.ObservationBundle,
    context: PersistenceContextBundle,
    validation_indices: np.ndarray,
    *,
    device: torch.device,
    evaluation_batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> Mapping[str, Any]:
    """Replay all nine validation predictions before development is unlocked."""

    selected = np.asarray(validation_indices, dtype=np.int64)
    _require(selected.shape == (196,), "validation replay requires all 196 cases")
    expected_dates = np.asarray(cache.initializations[selected], dtype="datetime64[D]")
    _require(
        set(pd.DatetimeIndex(expected_dates).year) == set(base.VALIDATION_YEARS),
        "validation replay attempted the wrong years",
    )
    expected_shape = (196, 6, 27, 27)
    expected_archive_fields = {
        "initializations",
        "delta_log_location",
        "log_spread",
        "spread_factor",
        "arm",
        "seed",
    }
    cells: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            record = receipt.checkpoint_records[(arm, seed)]
            relative = str(record["validation_adjustment"])
            archive_path = _safe_receipt_path(
                receipt.root,
                relative,
                label="validation replay artifact",
            )
            _require(
                sha256_file(archive_path) == record.get("validation_adjustment_sha256"),
                f"validation replay artifact changed for {arm}, seed {seed}",
            )
            with np.load(archive_path, allow_pickle=False) as archive:
                _require(
                    set(archive.files) == expected_archive_fields,
                    f"validation adjustment schema changed for {arm}, seed {seed}",
                )
                stored_dates = np.asarray(
                    archive["initializations"], dtype="datetime64[D]"
                )
                stored_delta = np.asarray(archive["delta_log_location"])
                stored_log_spread = np.asarray(archive["log_spread"])
                stored_spread = np.asarray(archive["spread_factor"])
                stored_arm = str(np.asarray(archive["arm"]).item())
                stored_seed = int(np.asarray(archive["seed"]).item())
            _require(
                np.array_equal(stored_dates, expected_dates)
                and stored_arm == arm
                and stored_seed == seed,
                f"validation adjustment identity mismatch for {arm}, seed {seed}",
            )
            for label, values in (
                ("delta_log_location", stored_delta),
                ("log_spread", stored_log_spread),
                ("spread_factor", stored_spread),
            ):
                _require(
                    values.shape == expected_shape
                    and values.dtype == np.dtype(np.float32)
                    and np.isfinite(values).all(),
                    f"stored validation {label} invalid for {arm}, seed {seed}",
                )
            stored_spread_from_log = np.exp(
                np.clip(stored_log_spread, -2.0, 2.0)
            ).astype(np.float32)
            _require(
                np.array_equal(stored_spread, stored_spread_from_log),
                f"stored spread/log-spread disagree for {arm}, seed {seed}",
            )

            checkpoint = _checkpoint_path(receipt, arm, seed)
            model = persistence_driver.load_checkpoint_model(
                checkpoint, arm, seed, device
            )
            replay_delta, replay_log_spread = persistence_driver.predict_adjustments(
                model,
                arm,
                cache.members,
                observations.weekly_truth,
                context,
                selected,
                device=device,
                batch_size=evaluation_batch_size,
                num_workers=num_workers,
                use_amp=use_amp,
            )
            del model
            replay_spread = np.exp(np.clip(replay_log_spread, -2.0, 2.0)).astype(
                np.float32
            )
            comparisons: dict[str, dict[str, Any]] = {}
            for label, replayed, stored in (
                ("delta_log_location", replay_delta, stored_delta),
                ("log_spread", replay_log_spread, stored_log_spread),
                ("spread_factor", replay_spread, stored_spread),
            ):
                byte_equal = bool(np.array_equal(replayed, stored))
                max_abs = float(
                    np.max(
                        np.abs(replayed.astype(np.float64) - stored.astype(np.float64))
                    )
                )
                tolerance_equal = bool(
                    np.allclose(
                        replayed,
                        stored,
                        rtol=VALIDATION_REPLAY_RTOL,
                        atol=VALIDATION_REPLAY_ATOL,
                        equal_nan=False,
                    )
                )
                _require(
                    tolerance_equal,
                    f"validation replay differs for {arm}, seed {seed}, {label}: "
                    f"max_abs={max_abs:.9g}",
                )
                comparisons[label] = {
                    "byte_equal": byte_equal,
                    "within_tight_tolerance": tolerance_equal,
                    "max_absolute_difference": max_abs,
                }
            cells.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "validation_adjustment_path": str(archive_path),
                    "validation_adjustment_sha256": sha256_file(archive_path),
                    "initializations_exact": True,
                    "comparisons": comparisons,
                    "all_arrays_byte_equal": all(
                        bool(item["byte_equal"]) for item in comparisons.values()
                    ),
                    "all_arrays_within_tight_tolerance": True,
                }
            )
            del replay_delta, replay_log_spread, replay_spread
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _require(
        len(cells) == len(ARMS) * len(SEEDS), "validation replay grid is incomplete"
    )
    return {
        "validation_initialization_count": int(len(selected)),
        "validation_years": list(base.VALIDATION_YEARS),
        "checkpoint_count": len(cells),
        "all_nine_stored_adjustments_reconstructed": True,
        "all_initialization_arrays_exact": True,
        "all_arrays_within_tight_tolerance": True,
        "all_arrays_byte_equal": all(
            bool(cell["all_arrays_byte_equal"]) for cell in cells
        ),
        "comparison_rtol": VALIDATION_REPLAY_RTOL,
        "comparison_atol": VALIDATION_REPLAY_ATOL,
        "development_indices_accessed": False,
        "cells": cells,
    }


def _fresh_identity_gates(
    persistence: PersistenceReceipt,
    pbc: PBCReceipt,
    joint: JointSelectionReceipt,
    *,
    device: torch.device,
    evaluation_batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[
    base.MemberCache,
    base.SplitIndices,
    base.ObservationBundle,
    PersistenceContextBundle,
    pbc_compare.SupportBundle,
    pbc_compare.ThresholdBundle,
    dict[str, Any],
]:
    """Rebuild every shared identity without indexing development cases."""

    try:
        source_bindings = pbc_compare.validate_source_bindings(
            pbc.root, pbc.manifest, persistence.root, persistence.manifest
        )
        cache = base.load_member_cache(persistence.cache_path, allow_partial=False)
        cache_receipt = pbc_compare.validate_cache_provenance(
            cache, pbc.manifest, persistence.manifest
        )
    except (pbc_compare.ComparisonContractError, base.DataContractError) as error:
        raise PersistencePBCComparisonError(
            f"source/cache identity failed: {error}"
        ) from error
    splits = base.make_split_indices(cache.initializations)
    _require(
        len(splits.train) == EXPECTED_SPLITS["train"]
        and len(splits.validation) == EXPECTED_SPLITS["validation"]
        and len(splits.embargo) == EXPECTED_SPLITS["embargo"],
        "fresh pre-development archive splits differ from both receipts",
    )
    try:
        pbc_support, pbc_support_sha = pbc_compare.load_support(pbc.root, pbc.manifest)
        persistence_support, persistence_support_sha = pbc_compare.load_support(
            persistence.root, persistence.manifest
        )
        pbc_compare.assert_same_support(pbc_support, persistence_support)
        thresholds = pbc_compare.load_thresholds(
            pbc.root, pbc.manifest, pbc_support.support, splits.train
        )
    except pbc_compare.ComparisonContractError as error:
        raise PersistencePBCComparisonError(
            f"PBC support/threshold identity failed: {error}"
        ) from error
    _require(
        np.array_equal(cache.latitude, pbc_support.latitude)
        and np.array_equal(cache.longitude, pbc_support.longitude),
        "cache grid differs from accepted PBC support",
    )

    observations = base.load_imd_observations(cache)
    _require(
        all("2025" not in str(path) for path in observations.source_stores),
        "fresh observation load opened sealed 2025",
    )
    try:
        observation_receipt = pbc_compare.validate_observation_provenance(
            observations, pbc.manifest
        )
    except pbc_compare.ComparisonContractError as error:
        raise PersistencePBCComparisonError(
            f"observation identity failed: {error}"
        ) from error
    expected_observation = persistence.manifest.get("observation_bundle_provenance", {})
    _require(
        observation_receipt.get("sha256") == expected_observation.get("sha256")
        and observation_receipt.get("arrays") == expected_observation.get("arrays"),
        "fresh observations differ from persistence receipt",
    )
    _require(
        np.array_equal(observations.weights, pbc_support.weights),
        "fresh observation weights differ from accepted PBC support",
    )

    daily_dates, daily_values = pbc_driver.load_daily_imd_for_lags(
        cache, observations.source_stores
    )
    lags = build_daily_issue_time_lags(
        cache.initializations, daily_dates, daily_values, lag_weeks=(1, 2)
    )
    del daily_dates, daily_values
    lag_provenance = dict(pbc_driver.issue_time_lag_provenance(lags))
    supported_available = np.asarray(lags.values)[lags.source_indices >= 0][
        :, pbc_support.support
    ]
    _require(
        bool(
            np.isfinite(supported_available).all()
            and np.all(supported_available >= 0.0)
        ),
        "fresh persistence lags are invalid on support",
    )
    lag_provenance["available_supported_values_finite_nonnegative"] = True
    _require(
        lag_provenance
        == persistence.manifest.get("persistence_lag_provenance")
        == pbc.manifest.get("persistence_lag_provenance"),
        "fresh/PBC/persistence lag identities differ",
    )
    persistence_driver.assert_validation_temporal_contract(
        cache.initializations, splits.train, splits.validation, lags
    )
    base_context = base.build_context_bundle(cache, observations, splits.train)
    context = build_persistence_context_bundle(
        base_context,
        cache.initializations,
        lags,
        splits.train,
        observations.weights,
    )
    context_provenance = pbc_driver.array_bundle_provenance(
        {
            "normalized_lag_log1p": context.normalized_lag_log1p,
            "lag_available": context.lag_available,
            "lag_log1p_mean_by_lag": context.lag_log1p_mean_by_lag,
            "lag_log1p_std_by_lag": context.lag_log1p_std_by_lag,
            "normalization_fit_indices": context.normalization_fit_indices,
        },
        bundle_name="train_normalized_persistence_context",
    )
    _require(
        context_provenance
        == persistence.manifest.get("persistence_context_provenance"),
        "fresh persistence context differs from full selection receipt",
    )
    base_context_provenance, weekly_climatology_provenance = _fresh_context_provenance(
        context, observations
    )
    combined_base_context_provenance = mask_driver.base_context_provenance(
        context.base_context,
        observations.weekly_climatology,
    )
    _require(
        combined_base_context_provenance
        == joint.manifest.get("base_context_and_weekly_climatology_provenance"),
        "fresh base context/climatology differs from joint receipt",
    )
    normalization_path = persistence.root / "evaluation/persistence_normalization.npz"
    with np.load(normalization_path, allow_pickle=False) as archive:
        normalization_identity = bool(
            np.array_equal(
                archive["lag_log1p_mean_by_lag"],
                context.lag_log1p_mean_by_lag,
            )
            and np.array_equal(
                archive["lag_log1p_std_by_lag"], context.lag_log1p_std_by_lag
            )
            and np.array_equal(
                archive["normalization_fit_indices"],
                context.normalization_fit_indices,
            )
        )
    _require(
        normalization_identity,
        "persistence normalization artifact differs from fresh context",
    )
    validation_replay = validate_validation_adjustment_replay(
        persistence,
        cache,
        observations,
        context,
        splits.validation,
        device=device,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )
    evidence = {
        "all_required_gates_passed_before_development_indexing": True,
        "source_bindings": source_bindings,
        "cache": cache_receipt,
        "pbc_support_artifact_sha256": pbc_support_sha,
        "persistence_support_artifact_sha256": persistence_support_sha,
        "support_exactly_shared": True,
        "threshold_fit_artifact_sha256": thresholds.artifact_sha256,
        "thresholds_fit_on_exact_1652_training_indices": True,
        "observation_bundle": observation_receipt,
        "lag_provenance": lag_provenance,
        "persistence_context_provenance": context_provenance,
        "fresh_full_base_context_provenance": base_context_provenance,
        "fresh_full_weekly_climatology_provenance": weekly_climatology_provenance,
        "joint_combined_base_context_and_weekly_climatology_provenance": (
            combined_base_context_provenance
        ),
        "normalization_artifact_exact": True,
        "validation_adjustment_replay": validation_replay,
        "development_indices_accessed": False,
        "sealed_2025_opened": False,
    }
    return cache, splits, observations, context, pbc_support, thresholds, evidence


def _pbc_baseline_case_scores(
    stored: pd.DataFrame,
    initializations: np.ndarray,
) -> pd.DataFrame:
    expected_dates = {
        np.datetime_as_string(value, unit="D") for value in initializations
    }
    frames = []
    for output_name, source_name in PBC_SOURCE_METHODS.items():
        selected = stored.loc[stored.method == source_name].copy()
        _require(
            len(selected) == len(initializations) * 6,
            f"accepted PBC {source_name} case scores are incomplete",
        )
        _require(
            set(selected.initialization) == expected_dates
            and set(pd.to_numeric(selected.lead_week).astype(int)) == set(range(1, 7)),
            f"accepted PBC {source_name} cases differ from development inventory",
        )
        _require(
            set(selected.score_contract) == {SCORING_CONTRACT_VERSION}
            and np.isfinite(selected.rps.to_numpy(dtype=np.float64)).all(),
            f"accepted PBC {source_name} scores are invalid",
        )
        selected["method"] = output_name
        selected["method_label"] = METHOD_LABELS[output_name]
        selected["seed"] = "not_applicable"
        selected["split"] = "test_development"
        selected["year"] = pd.DatetimeIndex(selected.initialization).year.to_numpy()
        frames.append(
            selected[
                [
                    "split",
                    "method",
                    "method_label",
                    "seed",
                    "score_contract",
                    "initialization",
                    "year",
                    "lead_week",
                    "rps",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_persistence_pbc_comparison.py": Path(__file__).resolve(),
        "src/fuxi_allseason_persistence_augmented.py": Path(
            persistence_driver.__file__
        ).resolve(),
        "src/fuxi_persistence_context.py": persistence_driver.HELPER_PATH,
        "src/fuxi_allseason_persistence_mask_control.py": Path(
            mask_driver.__file__
        ).resolve(),
        "src/fuxi_persistence_mask_control.py": PROJECT_ROOT
        / "src/fuxi_persistence_mask_control.py",
        "src/fuxi_allseason_categorical_comparison.py": Path(
            pbc_compare.__file__
        ).resolve(),
        "src/fuxi_allseason_pbc_baseline.py": Path(pbc_driver.__file__).resolve(),
        "src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "src/fuxi_allseason_ensemble_calibration.py": Path(base.__file__).resolve(),
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "src/project_paths.py": PROJECT_ROOT / "src/project_paths.py",
        "tests/test_allseason_persistence_pbc_comparison.py": PROJECT_ROOT
        / "tests/test_allseason_persistence_pbc_comparison.py",
        "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch": PROJECT_ROOT
        / "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch",
        "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": persistence_driver.PLAN_PATH,
        "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md": (
            mask_driver.ADDENDUM_PLAN_PATH
        ),
    }
    checksums = {}
    for relative, source in sources.items():
        _require(source.is_file(), f"comparison source missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def _output_checksums(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(item for item in output.rglob("*") if item.is_file())
        if path.name not in {"manifest.json", "failure.json"}
    }


def _build_readme(
    original_selected_arm: str,
    joint_selected_arm: str,
    categorical_pooled: pd.DataFrame,
    claims: Mapping[str, Any],
) -> str:
    pooled = categorical_pooled[["method", "mean_rps"]].to_dict(orient="records")
    return "\n".join(
        [
            "# Persistence neural vs accepted PBC V2",
            "",
            "**Status:** retrospective 2020–2021 development comparison; not an independent test.",
            "",
            f"The attribution-controlled joint validation selector locked `{joint_selected_arm}`. The historical three-arm v1 selector had locked `{original_selected_arm}`; it is not used for PBC claim eligibility. Development scores did not select or change either result.",
            "",
            "All nine checkpoints were reconstructed independently. Only per-case proper scores were averaged across seeds; forecasts, probabilities, and parameters were neither averaged nor written.",
            "",
            "Pooled equality-aware quintile RPS:",
            "",
            "```json",
            json.dumps(_json_safe(pooled), indent=2, sort_keys=True),
            "```",
            "",
            "Frozen interval-language decisions:",
            "",
            "```json",
            json.dumps(_json_safe(claims), indent=2, sort_keys=True),
            "```",
            "",
            "Only the jointly selected arm is claim-eligible. Every other interval is descriptive/exploratory and cannot support superiority language.",
            "",
            "Joint promotion means the frozen validation RPS-improvement gates passed with CRPS no-harm. Development CRPS values are point estimates only: this report makes no CRPS significance claim, and describes a lower value only when the reported mean CRPS is strictly lower.",
            "",
            "The paired intervals resample the two years and circular 13-initialization blocks with all six leads grouped. No 2022+ initialization cohort or sealed 2025 target was used.",
            "",
        ]
    )


def _evaluate_validated(
    args: argparse.Namespace,
    output: Path,
    persistence: PersistenceReceipt,
    pbc: PBCReceipt,
    joint: JointSelectionReceipt,
) -> Mapping[str, Any]:
    started = time.monotonic()
    snapshot = _source_snapshot(output)
    device = base.resolve_device(args.device)
    _require(device.type == "cuda", f"canonical comparison requires CUDA, got {device}")
    for arm in ARMS:
        base.METHOD_LABELS[arm] = METHOD_LABELS[arm]
        base.PLOT_METHOD_LABELS[arm] = METHOD_LABELS[arm]

    cache, splits, observations, context, support, thresholds, gate_evidence = (
        _fresh_identity_gates(
            persistence,
            pbc,
            joint,
            device=device,
            evaluation_batch_size=args.evaluation_batch_size,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
        )
    )
    # This is the sole development unlock. Every receipt and freshly rebuilt
    # identity above has passed before the test/development indices are read.
    development_indices = np.asarray(splits.test, dtype=np.int64)
    _require(
        development_indices.shape == (EXPECTED_SPLITS["test"],),
        "fresh development split has the wrong case count",
    )
    gate_evidence["development_indices_accessed"] = True
    gate_evidence["development_unlocked_after_all_required_gates"] = True
    pbc_inputs = load_accepted_pbc_inputs(
        pbc,
        cache,
        observations,
        development_indices,
        support,
        thresholds,
    )

    inputs = output / "inputs"
    metrics = output / "metrics"
    inputs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    shutil.copy2(persistence.manifest_path, inputs / "persistence_manifest.json")
    shutil.copy2(
        persistence.gate_receipt_path, inputs / "persistence_slurm_gate_receipt.json"
    )
    shutil.copy2(
        persistence.root / "selection.json", inputs / "persistence_selection.json"
    )
    shutil.copy2(pbc.manifest_path, inputs / "accepted_pbc_manifest.json")
    shutil.copy2(
        pbc.root / "slurm_gate_receipt.json",
        inputs / "accepted_pbc_slurm_gate_receipt.json",
    )
    shutil.copy2(joint.manifest_path, inputs / "joint_selection_manifest.json")
    shutil.copy2(
        joint.gate_receipt_path,
        inputs / "joint_selection_slurm_gate_receipt.json",
    )
    shutil.copy2(
        joint.root / "joint_selection.json",
        inputs / "joint_selection.json",
    )

    raw_continuous, _ = base.evaluate_ensemble(
        "raw_fuxi",
        pbc_inputs.members,
        pbc_inputs.truth,
        pbc_inputs.climatology,
        pbc_inputs.initializations,
        support.weights,
        chunk_size=args.evaluation_batch_size,
    )
    raw_scores = pbc_compare.score_ensemble(
        pbc_inputs.members,
        pbc_inputs.quintile_thresholds,
        pbc_inputs.semidecile_thresholds,
        pbc_inputs.quintile_observed_cdf,
        pbc_inputs.semidecile_observed_cdf,
        support.weights,
        chunk_size=args.cdf_chunk_size,
    )
    raw_case = pbc_compare.case_score_frame(
        "raw_fuxi_categorical",
        "not_applicable",
        raw_scores,
        pbc_inputs.references,
        pbc_inputs.initializations,
        pbc_inputs.quintile_thresholds,
        support.weights,
    )
    try:
        raw_identity = pbc_compare.validate_pbc_case_scores(
            pbc_inputs.stored_case_scores,
            pbc_inputs.initializations,
            raw_case,
        )
    except pbc_compare.ComparisonContractError as error:
        raise PersistencePBCComparisonError(
            f"accepted PBC raw identity failed: {error}"
        ) from error
    del raw_scores, raw_case

    seed_continuous_frames: list[pd.DataFrame] = []
    seed_categorical_frames: list[pd.DataFrame] = []
    reconstruction_receipts: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            print(f"Reconstructing {arm}, seed {seed}...", flush=True)
            continuous, categorical, reconstruction = reconstruct_and_score_arm_seed(
                persistence,
                arm,
                seed,
                cache,
                observations.weekly_truth,
                context,
                pbc_inputs,
                device=device,
                batch_size=args.batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                cdf_chunk_size=args.cdf_chunk_size,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            seed_continuous_frames.append(continuous)
            seed_categorical_frames.append(categorical)
            reconstruction_receipts.append(reconstruction)

    seed_continuous = pd.concat(seed_continuous_frames, ignore_index=True)
    averaged_continuous = base.mean_seed_case_metrics(seed_continuous, SEEDS)
    continuous_case = pd.concat(
        (raw_continuous, averaged_continuous), ignore_index=True
    )
    continuous_weekwise, continuous_pooled, continuous_seasonal, _ = (
        base.summarize_metrics(continuous_case)
    )

    seed_categorical = pd.concat(seed_categorical_frames, ignore_index=True)
    averaged_categorical = average_seed_scores(seed_categorical)
    averaged_categorical["split"] = "test_development"
    averaged_categorical["method_label"] = averaged_categorical.method.map(
        METHOD_LABELS
    )
    averaged_categorical["score_contract"] = SCORING_CONTRACT_VERSION
    pbc_baselines = _pbc_baseline_case_scores(
        pbc_inputs.stored_case_scores, pbc_inputs.initializations
    )
    categorical_case = pd.concat(
        (
            averaged_categorical[
                [
                    "split",
                    "method",
                    "method_label",
                    "seed",
                    "score_contract",
                    "initialization",
                    "year",
                    "lead_week",
                    "rps",
                ]
            ],
            pbc_baselines,
        ),
        ignore_index=True,
    )
    expected_categorical_methods = set(ARMS) | set(PBC_BASELINES)
    _require(
        set(categorical_case.method) == expected_categorical_methods
        and not categorical_case.duplicated(
            ["method", "initialization", "lead_week"]
        ).any(),
        "merged neural/PBC categorical case scores are incomplete",
    )
    categorical_weekwise = (
        categorical_case.groupby(
            ["method", "method_label", "seed", "lead_week"], as_index=False
        )
        .agg(mean_rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values(["method", "lead_week"])
    )
    categorical_pooled = (
        categorical_case.groupby(["method", "method_label", "seed"], as_index=False)
        .agg(mean_rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values("method")
    )

    # Retain the fully frozen three-arm ablation in the interval artifact. The
    # claim table below uses only the attribution-controlled joint selection,
    # but all original v1 controls remain visible and exactly paired.
    target_methods = ARMS
    bootstrap = paired_pbc_bootstrap(
        categorical_case,
        target_methods,
        PBC_BASELINES,
        samples=args.bootstrap_samples,
        block_length=args.block_length,
        seed=args.bootstrap_seed,
    )
    # Superiority language belongs only to the joint mask-control selection.
    # Other complete paired intervals remain descriptive/exploratory evidence.
    claims = classify_pbc_claims(
        bootstrap,
        {"joint_validation_selected_arm": joint.selected_arm},
    )

    seed_continuous.to_csv(metrics / "continuous_seed_case_metrics.csv", index=False)
    continuous_case.to_csv(metrics / "continuous_case_metrics.csv", index=False)
    continuous_weekwise.to_csv(metrics / "continuous_weekwise_metrics.csv", index=False)
    continuous_pooled.to_csv(metrics / "continuous_pooled_metrics.csv", index=False)
    continuous_seasonal.to_csv(metrics / "continuous_seasonal_metrics.csv", index=False)
    seed_categorical.to_csv(metrics / "categorical_seed_case_scores.csv", index=False)
    categorical_case.to_csv(metrics / "categorical_case_scores.csv", index=False)
    categorical_weekwise.to_csv(
        metrics / "categorical_weekwise_metrics.csv", index=False
    )
    categorical_pooled.to_csv(metrics / "categorical_pooled_metrics.csv", index=False)
    bootstrap.to_csv(metrics / "paired_pbc_bootstrap.csv", index=False)
    write_json(output / "claims.json", claims)
    (output / "README.md").write_text(
        _build_readme(
            persistence.selected_arm,
            joint.selected_arm,
            categorical_pooled,
            claims,
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "evidence_status": "reused 2020-2021 retrospective development comparison; not independent confirmation",
        "contract": {
            "training_performed": False,
            "selection_performed": False,
            "original_v1_validation_locked_selected_arm": persistence.selected_arm,
            "joint_validation_locked_selected_arm": joint.selected_arm,
            "selection_data": "2018-2019 validation only in the externally receipted joint mask-control addendum",
            "evaluation_initialization_years": list(base.TEST_YEARS),
            "evaluation_initialization_count": 208,
            "development_scores_can_change_selection": False,
            "superiority_claims_limited_to_joint_selected_arm": True,
            "original_v1_selection_is_not_claim_authority": True,
            "nonselected_persistence_candidate_is_descriptive_only": bool(
                joint.selected_arm != PERSISTENCE_LAG_ARM
            ),
            "promotion_interpretation": "validation RPS improvement with CRPS no-harm",
            "superiority_claim_metric": "equality-aware quintile RPS only",
            "crps_improvement_language_requires_strictly_lower_mean": True,
            "crps_significance_claimed": False,
            "no_2022_plus_initialization_cohort": True,
            "sealed_2025_target_opened": False,
            "all_51_members_evaluated": True,
            "all_six_leads_grouped": True,
            "forecasts_probabilities_or_parameters_averaged_across_seeds": False,
            "forecasts_probabilities_or_parameters_written": False,
            "only_per_seed_scores_averaged": True,
            "categorical_score_contract": SCORING_CONTRACT_VERSION,
        },
        "input_receipts": {
            "persistence": {
                "manifest_path": str(persistence.manifest_path),
                "manifest_sha256": persistence.manifest_sha256,
                "gate_receipt_path": str(persistence.gate_receipt_path),
                "gate_receipt_sha256": persistence.gate_receipt_sha256,
                "full_post_run_audit_passed": True,
                "selected_arm": persistence.selected_arm,
            },
            "joint_mask_control_selection": {
                "manifest_path": str(joint.manifest_path),
                "manifest_sha256": joint.manifest_sha256,
                "gate_receipt_path": str(joint.gate_receipt_path),
                "gate_receipt_sha256": joint.gate_receipt_sha256,
                "full_post_run_audit_passed": True,
                "selected_arm": joint.selected_arm,
                "claim_authority": True,
                "joint_selector_recomputation": joint.recomputation_evidence,
                "matching_environment": joint.matching_environment_evidence,
                "smoke_parent": joint.smoke_parent_evidence,
            },
            "accepted_pbc_v2": {
                "manifest_path": str(pbc.manifest_path),
                "manifest_sha256": pbc.manifest_sha256,
                "pinned_manifest_sha256": ACCEPTED_PBC_MANIFEST_SHA256,
                "pinned_observation_bundle_sha256": (
                    ACCEPTED_PBC_OBSERVATION_BUNDLE_SHA256
                ),
                "pinned_persistence_lag_sha256": (ACCEPTED_PBC_PERSISTENCE_LAG_SHA256),
                "pinned_pbc_fit_artifact_sha256": (ACCEPTED_PBC_FIT_ARTIFACT_SHA256),
                "gate_receipt": pbc.gate_receipt,
                "immutable_accepted_identity": True,
            },
        },
        "identity_gates": gate_evidence,
        "fresh_derived_input_provenance": {
            "full_base_context": gate_evidence["fresh_full_base_context_provenance"],
            "full_weekly_climatology": gate_evidence[
                "fresh_full_weekly_climatology_provenance"
            ],
            "joint_combined_base_context_and_weekly_climatology": gate_evidence[
                "joint_combined_base_context_and_weekly_climatology_provenance"
            ],
        },
        "accepted_pbc_inputs": pbc_inputs.identity_evidence,
        "pbc_raw_identity": raw_identity,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "all_nine_checkpoints_loaded": len(reconstruction_receipts) == 9,
        "reconstruction_receipts": reconstruction_receipts,
        "score_aggregation": {
            "neural": "mean of three independently reconstructed per-seed proper scores",
            "pbc": "stored accepted case scores after exact raw FuXi identity verification",
            "pbc_methods_reused": list(PBC_BASELINES),
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "block_length_initializations": args.block_length,
            "scheme": BOOTSTRAP_SCHEME,
            "year_resampling": True,
            "all_six_leads_grouped": True,
            "reported_scopes": ["W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"],
            "comparison_methods": list(target_methods),
            "claim_eligible_methods": [joint.selected_arm],
            "nonselected_methods_descriptive_only": [
                arm for arm in ARMS if arm != joint.selected_arm
            ],
            "baselines": list(PBC_BASELINES),
        },
        "claims": claims,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot,
    }
    _require(
        manifest["all_nine_checkpoints_loaded"],
        "not all nine checkpoints were evaluated",
    )
    manifest["artifact_sha256"] = _output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def run_evaluation(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    """Validate all immutable receipts, then run the score-only evaluator."""

    persistence = validate_persistence_receipt(Path(args.persistence_manifest))
    pbc = validate_accepted_pbc_receipt(Path(args.pbc_manifest))
    joint = validate_joint_selection_receipt(
        Path(args.joint_selection_manifest), persistence, pbc
    )
    return _evaluate_validated(args, output, persistence, pbc, joint)


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a locked full persistence run with accepted PBC V2."
    )
    parser.add_argument("--persistence-manifest", type=Path, required=True)
    parser.add_argument("--joint-selection-manifest", type=Path, required=True)
    parser.add_argument("--pbc-manifest", type=Path, default=DEFAULT_PBC_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--cdf-chunk-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--block-length", type=int, default=BOOTSTRAP_BLOCK_LENGTH)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "batch_size": (args.batch_size, 8),
        "evaluation_batch_size": (args.evaluation_batch_size, 8),
        "cdf_chunk_size": (args.cdf_chunk_size, 8),
        "num_workers": (args.num_workers, 0),
        "bootstrap_samples": (args.bootstrap_samples, BOOTSTRAP_SAMPLES),
        "block_length": (args.block_length, BOOTSTRAP_BLOCK_LENGTH),
        "bootstrap_seed": (args.bootstrap_seed, BOOTSTRAP_SEED),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical comparison settings differ: {mismatch}")
    if args.device == "cpu":
        raise ValueError("canonical comparison requires CUDA")
    if args.no_amp:
        raise ValueError("canonical comparison requires automatic mixed precision")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {requested}")
    staging = requested.parent / f".{requested.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_evaluation(args, staging)
        os.replace(staging, requested)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "requested_output": str(requested),
                "persistence_manifest": str(Path(args.persistence_manifest).resolve()),
                "joint_selection_manifest": str(
                    Path(args.joint_selection_manifest).resolve()
                ),
                "pbc_manifest": str(Path(args.pbc_manifest).resolve()),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(f"Complete: {requested}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
