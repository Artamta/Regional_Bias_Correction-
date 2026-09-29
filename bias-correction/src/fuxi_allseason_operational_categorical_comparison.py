#!/usr/bin/env python3
"""Frozen PBC/neural categorical comparison on the 2022--2024 audit cases.

This evaluator cannot train, tune, select, or refit anything.  It reloads the
exact operational-era members and observations audited by the completed V3
full run, reconstructs each neural seed from its receipted adjustment fields,
and applies the already persisted PBC V2 parameters.  All headline neural
values are arithmetic means of independently computed scores; members,
probabilities, parameters, and adjustment fields are never averaged.

Only literal 2022--2024 forecast/verification stores are used.  The sole
additional observation access is the exact late-2021 daily slice required to
construct complete issue-time Persistence++ lags for the earliest 2022 cases.
No 2025 forecast or observation store is constructed or opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

import fuxi_allseason_ensemble_calibration as neural
import fuxi_allseason_operational_era_audit as operational
import fuxi_pbc_core as pbc
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_operational_categorical_comparison_v1"
AUDIT_CONTRACT_VERSION = "operational_categorical_contract_v1"
POSTRUN_AUDIT_VERSION = "operational_categorical_postrun_v1"
SCORING_CONTRACT_VERSION = "normalized_informative_positive_cut_dynamic_v1"
NOMINAL_REFERENCE_CONTRACT = "threshold_projected_zero_anchored_nominal_dynamic_v1"
BOOTSTRAP_SCHEME = "two_stage_year_resample_then_within_year_circular_13_init_blocks"
EVIDENCE_LABEL = (
    "post-hoc 2022-2024 operational-era categorical retrospective; "
    "no retraining, selection, or fitted blending"
)

PBC_EXPERIMENT = "fuxi_allseason_pbc_baseline_v2"
PBC_POSTRUN_AUDIT_VERSION = "pbc_postrun_v2"
PBC_SCORING_CONTRACT_VERSION = "normalized_informative_positive_cut_v2"
PBC_MANIFEST_SHA256 = "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6"
PBC_RECEIPT_SHA256 = "1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2"
PBC_FIT_SHA256 = "0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36"
OPERATIONAL_MANIFEST_SHA256 = "7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad"
OPERATIONAL_RECEIPT_SHA256 = "65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd"
DEFAULT_PBC_MANIFEST = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_pbc_baseline_v2/"
    "full_20260822T173656Z/manifest.json"
)
DEFAULT_OPERATIONAL_ROOT = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z"
)
DEFAULT_OPERATIONAL_MANIFEST = DEFAULT_OPERATIONAL_ROOT / "manifest.json"
DEFAULT_OPERATIONAL_RECEIPT = DEFAULT_OPERATIONAL_ROOT / "slurm_gate_receipt.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_operational_categorical_comparison"
)
PROTOCOL_PATH = (
    PROJECT_ROOT / "plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md"
)
SLURM_PATH = (
    PROJECT_ROOT
    / "slurm/evaluate_allseason_operational_categorical_comparison.sbatch"
)
TEST_PATH = (
    PROJECT_ROOT / "tests/test_allseason_operational_categorical_comparison.py"
)

SEEDS = (42, 43, 44)
FAMILIES = ("quintile", "semidecile")
LEVELS = {
    "quintile": np.asarray(np.arange(0.2, 1.0, 0.2), dtype=np.float32),
    "semidecile": np.asarray(np.arange(0.05, 1.0, 0.05), dtype=np.float32),
}
PBC_METHODS = ("debias_plus_plus", "persistence_plus_plus", "pbc_combined")
METHODS = ("raw_fuxi_categorical", "base_42k", *PBC_METHODS)
METHOD_LABELS = {
    "raw_fuxi_categorical": "Raw FuXi categorical",
    "base_42k": "Locked neural adapter (42,434 parameters)",
    "debias_plus_plus": "Frozen projected Debias++",
    "persistence_plus_plus": "Frozen projected Persistence++",
    "pbc_combined": "Frozen equal-weight PBC",
}
COMPARISONS = (
    ("base_42k", "raw_fuxi_categorical"),
    ("debias_plus_plus", "raw_fuxi_categorical"),
    ("persistence_plus_plus", "raw_fuxi_categorical"),
    ("pbc_combined", "raw_fuxi_categorical"),
    ("base_42k", "debias_plus_plus"),
    ("base_42k", "persistence_plus_plus"),
    ("base_42k", "pbc_combined"),
    ("pbc_combined", "debias_plus_plus"),
    ("pbc_combined", "persistence_plus_plus"),
)

EXPECTED_CASES = 296
EXPECTED_YEAR_COUNTS = {2022: 104, 2023: 104, 2024: 88}
FORECAST_MEMBERS = 50
LEAD_COUNT = 6
GRID_SHAPE = (27, 27)
TRAIN_CASES = 1652
PERSISTENCE_USABLE_CASES = 1648
DEBIAS_SPANS = (14, 28, 35)
PERSISTENCE_RIDGE = 1.0e-3
BLEND_DEBIAS_WEIGHT = 0.5
BOOTSTRAP_DRAWS = 10_000
SMOKE_BOOTSTRAP_DRAWS = 200
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_823
CDF_CHUNK_SIZE = 8
SMOKE_CASES_PER_YEAR = 4
LAG_WEEKS = (1, 2)
PERSISTENCE_FEATURE_NAMES = (
    "intercept",
    "training_empirical_climatology_cdf",
    "lag_1week_indicator",
    "lag_2week_indicator",
    "raw_fuxi_cdf",
)
EXPECTED_FULL_LAG_COVERAGE_BLOCKS = (
    ("2023-10-16", 1, 4, "2023-10-12"),
    ("2023-10-19", 1, 1, "2023-10-12"),
    ("2023-10-23", 2, 4, "2023-10-12"),
    ("2023-10-26", 2, 1, "2023-10-12"),
)
MEMBER_CACHE_YEARS = tuple(range(2002, 2022))
EXPECTED_MEMBER_CACHE_CASES = 2080
MEMBER_CACHE_DATA_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
MEMBER_CACHE_CHECKSUMS_SHA256 = (
    "c4bb28ea67c650910280a9ec6ed340de01e5d54b82848cd4c592e277a6732603"
)


class OperationalCategoricalError(RuntimeError):
    """Raised when a frozen input, temporal firewall, or score invariant moves."""


@dataclass(frozen=True)
class FrozenFamily:
    name: str
    model: pbc.CalendarQuantiles
    debias_fits: tuple[pbc.DebiasFit, ...]
    selected_debias_spans: np.ndarray
    persistence_fit: pbc.PersistenceFit


@dataclass(frozen=True)
class FrozenPBC:
    quintile: FrozenFamily
    semidecile: FrozenFamily
    fit_path: Path
    fit_sha256: str


@dataclass(frozen=True)
class DynamicSupport:
    latitude: np.ndarray
    longitude: np.ndarray
    frozen_support: np.ndarray
    frozen_weights: np.ndarray
    dynamic_weights: np.ndarray


@dataclass(frozen=True)
class FamilyReference:
    projected_nominal_rps: np.ndarray
    empirical_rps: np.ndarray
    paper_fixed_nominal_rps: np.ndarray | None
    projected_nominal_upper_brier: np.ndarray | None
    empirical_upper_brier: np.ndarray | None


@dataclass(frozen=True)
class FamilyScore:
    rps: np.ndarray
    probability_bias: np.ndarray
    paper_q80_rps: np.ndarray | None
    upper_brier: np.ndarray | None


@dataclass(frozen=True)
class LagInputs:
    lags: pbc.IssueTimeLags
    provenance: Mapping[str, Any]
    opened_store: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalCategoricalError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(
    root: Path, manifest: Mapping[str, Any], relative: str
) -> tuple[Path, str]:
    path = Path(root) / relative
    expected = manifest.get("artifact_sha256", {}).get(relative)
    _require(path.is_file(), f"required artifact is missing: {path}")
    _require(_valid_sha256(expected), f"artifact is not receipted: {relative}")
    actual = sha256_file(path)
    _require(actual == expected, f"artifact hash differs: {relative}")
    return path, actual


def output_checksums(output: Path) -> dict[str, str]:
    excluded = {"manifest.json", "failure.json", "slurm_gate_receipt.json"}
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(item for item in output.rglob("*") if item.is_file())
        if str(path.relative_to(output)) not in excluded
    }


def source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_operational_categorical_comparison.py": Path(__file__).resolve(),
        "src/fuxi_allseason_operational_era_audit.py": Path(operational.__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": Path(neural.__file__).resolve(),
        "src/fuxi_pbc_core.py": Path(pbc.__file__).resolve(),
        "src/fuxi_allseason_capacity_ablation.py": PROJECT_ROOT / "src/fuxi_allseason_capacity_ablation.py",
        "src/fuxi_allseason_capacity_development_evaluation.py": PROJECT_ROOT / "src/fuxi_allseason_capacity_development_evaluation.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT / "src/fuxi_allseason_member_cache.py",
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT / "src/fuxi_ensemble_calibration_core.py",
        "plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md": PROTOCOL_PATH,
        "slurm/evaluate_allseason_operational_categorical_comparison.sbatch": SLURM_PATH,
        "tests/test_allseason_operational_categorical_comparison.py": TEST_PATH,
    }
    destination_root = output / "code"
    hashes: dict[str, str] = {}
    for relative, source in sources.items():
        _require(source.is_file(), f"source snapshot input is missing: {source}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[f"code/{relative}"] = sha256_file(destination)
    return hashes


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON input is not an object: {path}")
    return payload


def _reject_forbidden_year_store(path: Path) -> None:
    _require(
        all(part != "2025.zarr" for part in Path(path).parts),
        f"forbidden year store: {path}",
    )


def _verify_declared_artifacts(root: Path, manifest: Mapping[str, Any]) -> int:
    inventory = manifest.get("artifact_sha256")
    _require(isinstance(inventory, dict) and inventory, "input artifact inventory is empty")
    for relative, expected in sorted(inventory.items()):
        _require(_valid_sha256(expected), f"invalid artifact digest: {relative}")
        path = root / relative
        _require(path.is_file(), f"input artifact is missing: {relative}")
        _require(sha256_file(path) == expected, f"input artifact changed: {relative}")
    return len(inventory)


def validate_pbc_input(manifest_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    _require(path.is_file() and path.name == "manifest.json", "PBC manifest is unavailable")
    _require(sha256_file(path) == PBC_MANIFEST_SHA256, "PBC manifest hash differs")
    manifest = _read_json(path)
    _require(manifest.get("experiment") == PBC_EXPERIMENT, "PBC experiment differs")
    _require(
        manifest.get("status") == "complete"
        and manifest.get("mode") == "full"
        and manifest.get("smoke") is False,
        "PBC input is not a completed full run",
    )
    _require(
        manifest.get("scoring_contract_version") == PBC_SCORING_CONTRACT_VERSION,
        "PBC scoring contract differs",
    )
    _require(
        manifest.get("split_counts_selected")
        == {"train": 1652, "validation": 196, "test": 208},
        "PBC selected split counts differ",
    )
    _require(
        manifest.get("contract", {}).get("sealed_2025_target_opened") is False,
        "PBC input does not prove the sealed-year firewall",
    )
    root = path.parent
    receipt_path = root / "slurm_gate_receipt.json"
    _require(receipt_path.is_file(), "PBC Slurm receipt is absent")
    _require(sha256_file(receipt_path) == PBC_RECEIPT_SHA256, "PBC receipt hash differs")
    receipt = _read_json(receipt_path)
    for key, expected in {
        "experiment": PBC_EXPERIMENT,
        "gate_status": "passed",
        "mode": "full",
        "post_run_audit_version": PBC_POSTRUN_AUDIT_VERSION,
        "scoring_contract_version": PBC_SCORING_CONTRACT_VERSION,
        "manifest_sha256": PBC_MANIFEST_SHA256,
    }.items():
        _require(receipt.get(key) == expected, f"PBC receipt differs for {key}")
    fit_path, fit_hash = _artifact(root, manifest, "models/pbc_fit.npz")
    _require(fit_hash == PBC_FIT_SHA256, "frozen PBC fit hash differs")
    inventory_count = _verify_declared_artifacts(root, manifest)
    return root, manifest, {
        "manifest_path": str(path),
        "manifest_sha256": PBC_MANIFEST_SHA256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": PBC_RECEIPT_SHA256,
        "pbc_fit_path": str(fit_path),
        "pbc_fit_sha256": PBC_FIT_SHA256,
        "declared_artifacts_verified": inventory_count,
    }


def validate_operational_input() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate the one immutable accepted operational-era V3 parent."""

    path = DEFAULT_OPERATIONAL_MANIFEST.resolve()
    receipt_path = DEFAULT_OPERATIONAL_RECEIPT.resolve()
    _require(path.is_file() and path.name == "manifest.json", "operational manifest is unavailable")
    _require(receipt_path.is_file(), "operational receipt is unavailable")
    actual_manifest_sha = sha256_file(path)
    actual_receipt_sha = sha256_file(receipt_path)
    _require(actual_manifest_sha == OPERATIONAL_MANIFEST_SHA256, "operational manifest hash differs")
    _require(actual_receipt_sha == OPERATIONAL_RECEIPT_SHA256, "operational receipt hash differs")
    manifest = _read_json(path)
    receipt = _read_json(receipt_path)
    _require(
        manifest.get("experiment") == operational.EXPERIMENT
        and manifest.get("audit_contract_version") == operational.AUDIT_CONTRACT_VERSION,
        "operational experiment/contract differs",
    )
    _require(
        manifest.get("status") == "complete"
        and manifest.get("mode") == "full"
        and manifest.get("smoke") is False
        and manifest.get("canonical") is True
        and manifest.get("scientific_eligible") is True,
        "operational input is not the canonical completed full",
    )
    for key, expected in {
        "experiment": operational.EXPERIMENT,
        "gate_status": "passed",
        "mode": "full",
        "audit_contract_version": operational.AUDIT_CONTRACT_VERSION,
        "post_run_audit_version": operational.POSTRUN_AUDIT_VERSION,
        "manifest_sha256": OPERATIONAL_MANIFEST_SHA256,
    }.items():
        _require(receipt.get(key) == expected, f"operational receipt differs for {key}")
    _require(receipt_path.parent == path.parent, "operational manifest/receipt roots differ")
    cases = manifest.get("cases", {})
    _require(
        cases.get("full_eligible_counts") == {"2022": 104, "2023": 104, "2024": 88}
        and cases.get("evaluated_counts") == {"2022": 104, "2023": 104, "2024": 88}
        and cases.get("evaluated_total") == EXPECTED_CASES,
        "operational full case contract differs",
    )
    _require(manifest.get("methods") == ["raw_fuxi", "base_42k"], "operational methods differ")
    _require(manifest.get("seeds") == list(SEEDS), "operational seeds differ")
    contract = manifest.get("contract", {})
    for key in (
        "workflow_can_train",
        "workflow_can_finetune",
        "workflow_can_select",
        "audit_metrics_used_for_selection",
        "final_2025_store_opened",
        "sealed_2025_target_opened",
    ):
        _require(contract.get(key) is False, f"operational firewall differs for {key}")
    _require(
        contract.get("forecast_year_store_whitelist") == [2022, 2023, 2024]
        and contract.get("verification_year_store_whitelist") == [2022, 2023, 2024],
        "operational year whitelist differs",
    )
    _require(
        contract.get("maximum_verification_date") <= "2024-12-31",
        "operational verification enters the forbidden year",
    )
    for raw_path in manifest.get("opened_store_paths_exact", []):
        _reject_forbidden_year_store(Path(raw_path))
    semantic_path = path.parent / "evaluation/postflight_semantic_audit.json"
    _require(semantic_path.is_file(), "operational semantic audit is absent")
    _require(
        sha256_file(semantic_path) == receipt.get("postflight_semantic_audit_sha256"),
        "operational semantic audit hash differs",
    )
    semantic = _read_json(semantic_path)
    _require(
        semantic.get("status") == "passed"
        and semantic.get("mode") == "full"
        and semantic.get("post_run_audit_version") == operational.POSTRUN_AUDIT_VERSION,
        "operational semantic audit did not pass",
    )
    inventory_count = _verify_declared_artifacts(path.parent, manifest)
    return path.parent, manifest, {
        "manifest_path": str(path),
        "manifest_sha256": actual_manifest_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": actual_receipt_sha,
        "semantic_audit_path": str(semantic_path),
        "semantic_audit_sha256": sha256_file(semantic_path),
        "declared_artifacts_verified": inventory_count,
    }


def validate_source_bindings(
    pbc_root: Path,
    pbc_manifest: Mapping[str, Any],
    operational_root: Path,
    operational_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require reused live code to be byte-identical to accepted parent runs."""

    pbc_snapshot, pbc_sha = _artifact(
        pbc_root, pbc_manifest, "code/src/fuxi_pbc_core.py"
    )
    live_pbc = Path(pbc.__file__).resolve()
    _require(sha256_file(live_pbc) == pbc_sha, "live PBC core differs from frozen PBC V2")

    runtime_relatives = (
        "code/src/fuxi_allseason_operational_era_audit.py",
        "code/src/fuxi_allseason_ensemble_calibration.py",
        "code/src/fuxi_allseason_capacity_ablation.py",
        "code/src/fuxi_allseason_capacity_development_evaluation.py",
        "code/src/fuxi_allseason_member_cache.py",
        "code/src/fuxi_ensemble_calibration_core.py",
    )
    live_lookup = {
        "code/src/fuxi_allseason_operational_era_audit.py": Path(operational.__file__).resolve(),
        "code/src/fuxi_allseason_ensemble_calibration.py": Path(neural.__file__).resolve(),
        "code/src/fuxi_allseason_capacity_ablation.py": PROJECT_ROOT / "src/fuxi_allseason_capacity_ablation.py",
        "code/src/fuxi_allseason_capacity_development_evaluation.py": PROJECT_ROOT / "src/fuxi_allseason_capacity_development_evaluation.py",
        "code/src/fuxi_allseason_member_cache.py": PROJECT_ROOT / "src/fuxi_allseason_member_cache.py",
        "code/src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT / "src/fuxi_ensemble_calibration_core.py",
    }
    operational_bindings: dict[str, Any] = {}
    source_inventory = operational_manifest.get("source_snapshot_sha256", {})
    for relative in runtime_relatives:
        expected = source_inventory.get(relative)
        snapshot = operational_root / relative
        live = live_lookup[relative]
        _require(_valid_sha256(expected), f"operational source receipt absent: {relative}")
        _require(snapshot.is_file(), f"operational source snapshot absent: {relative}")
        _require(sha256_file(snapshot) == expected, f"operational source snapshot differs: {relative}")
        _require(sha256_file(live) == expected, f"live operational dependency differs: {relative}")
        operational_bindings[relative] = {
            "live_path": str(live),
            "snapshot_path": str(snapshot),
            "sha256": expected,
            "exact_identity": True,
        }
    return {
        "pbc_core": {
            "live_path": str(live_pbc),
            "snapshot_path": str(pbc_snapshot),
            "sha256": pbc_sha,
            "exact_identity": True,
        },
        "operational_runtime": operational_bindings,
    }


def neural_member_cache_access_receipt(
    cache: neural.MemberCache,
    capacity_receipt: Any,
) -> dict[str, Any]:
    """Directly receipt the parent-bound 2002--2021 neural member cache."""

    cache_info = capacity_receipt.manifest.get("cache", {})
    members_path = Path(cache.members_path).resolve()
    metadata_path = Path(cache.metadata_path).resolve()
    _require(cache.manifest_path is not None, "member cache manifest is absent")
    manifest_path = Path(cache.manifest_path).resolve()
    cache_stem = members_path.with_suffix("")
    checksums_path = cache_stem.with_name(cache_stem.name + ".sha256")
    expected_paths = {
        "data_file": members_path,
        "metadata_file": metadata_path,
        "manifest_file": manifest_path,
    }
    expected_hash_keys = {
        "data_file": "data_sha256",
        "metadata_file": "metadata_sha256",
        "manifest_file": "manifest_sha256",
    }
    for key, path in expected_paths.items():
        _require(
            Path(str(cache_info.get(key, ""))).resolve() == path and path.is_file(),
            f"parent-bound member cache path differs: {key}",
        )
        digest = cache_info.get(expected_hash_keys[key])
        _require(_valid_sha256(digest), f"member cache hash is absent: {key}")
        if key != "data_file":
            _require(sha256_file(path) == digest, f"member cache bytes differ: {key}")
    _require(
        cache_info.get("data_sha256") == MEMBER_CACHE_DATA_SHA256
        and MEMBER_CACHE_DATA_SHA256 == operational.capacity.EXPECTED_CACHE_SHA256,
        "member cache data identity differs from the parent gate",
    )
    _require(
        checksums_path.is_file()
        and sha256_file(checksums_path) == MEMBER_CACHE_CHECKSUMS_SHA256,
        "member cache checksum sidecar differs",
    )
    starts = np.asarray(cache.initializations, dtype="datetime64[D]")
    years = sorted(set(pd.DatetimeIndex(starts).year.tolist()))
    _require(
        starts.shape == (EXPECTED_MEMBER_CACHE_CASES,)
        and years == list(MEMBER_CACHE_YEARS),
        "member cache 2002-2021 temporal scope differs",
    )
    return {
        "access_role": (
            "read-only neural hindcast member cache used to reconstruct the "
            "frozen split and operational audit context"
        ),
        "years": list(MEMBER_CACHE_YEARS),
        "initialization_count": int(starts.size),
        "data_file": str(members_path),
        "data_sha256": MEMBER_CACHE_DATA_SHA256,
        "data_bytes_reverified_by_parent_capacity_gate": True,
        "canonical_loader_verify_true": True,
        "whole_npy_integrity_scanned": True,
        "hindcast_member_values_scored_in_this_comparison": False,
        "metadata_used_for_frozen_split_and_grid_context": True,
        "metadata_file": str(metadata_path),
        "metadata_sha256": cache_info["metadata_sha256"],
        "manifest_file": str(manifest_path),
        "manifest_sha256": cache_info["manifest_sha256"],
        "checksums_file": str(checksums_path),
        "checksums_sha256": MEMBER_CACHE_CHECKSUMS_SHA256,
        "capacity_manifest_path": str(capacity_receipt.manifest_path),
        "capacity_manifest_sha256": capacity_receipt.manifest_sha256,
        "parent_bound": True,
        "read_only": True,
    }


def _calendar_sample_counts(
    initializations: np.ndarray,
    fit_indices: np.ndarray,
    half_window_days: int,
    *,
    minimum_samples: int,
) -> np.ndarray:
    """Reconstruct the complete DebiasFit count metadata without fitting."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    selected = np.asarray(fit_indices, dtype=np.int64)
    _require(
        dates.ndim == 1
        and np.array_equal(selected, np.arange(TRAIN_CASES, dtype=np.int64)),
        "Debias metadata needs the exact 1,652-case training split",
    )
    positions = pbc._calendar_positions(  # exact frozen-core calendar geometry
        pbc.verification_midpoints(dates, LEAD_COUNT)
    )
    counts = np.empty((366, LEAD_COUNT), dtype=np.int32)
    for lead in range(LEAD_COUNT):
        selected_positions = positions[selected, lead]
        daily_count = np.bincount(selected_positions, minlength=366).astype(np.int32)
        repeated = np.tile(daily_count, 3)
        prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(repeated, dtype=np.int64))
        )
        centres = np.arange(366, dtype=np.int64) + 366
        window = (
            prefix[centres + half_window_days + 1]
            - prefix[centres - half_window_days]
        )
        for day in range(366):
            if window[day] >= minimum_samples:
                counts[day, lead] = int(window[day])
            else:
                distance = np.minimum(
                    (selected_positions - day) % 366,
                    (day - selected_positions) % 366,
                )
                counts[day, lead] = min(
                    minimum_samples, int(np.count_nonzero(np.isfinite(distance)))
                )
    return counts


def _load_calendar_model(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    support: np.ndarray,
    pbc_manifest: Mapping[str, Any],
) -> pbc.CalendarQuantiles:
    levels = np.asarray(archive[f"{prefix}_levels"], dtype=np.float32)
    _require(np.array_equal(levels, LEVELS[prefix]), f"{prefix} levels differ")
    thresholds = np.asarray(
        archive[f"{prefix}_thresholds_mm_day"], dtype=np.float32
    )
    empirical = np.asarray(
        archive[f"{prefix}_training_empirical_strict_cdf"], dtype=np.float32
    )
    expected_shape = (366, levels.size, *GRID_SHAPE)
    _require(
        thresholds.shape == expected_shape and empirical.shape == expected_shape,
        f"{prefix} calendar-field shape differs",
    )
    mask = np.asarray(support, dtype=bool)
    _require(mask.shape == GRID_SHAPE and np.count_nonzero(mask) == 171, "PBC support differs")
    _require(
        np.isfinite(thresholds[..., mask]).all()
        and np.all(thresholds[..., mask] >= 0.0)
        and np.isfinite(empirical[..., mask]).all()
        and np.all((empirical[..., mask] >= 0.0) & (empirical[..., mask] <= 1.0)),
        f"{prefix} supported calendar fields are invalid",
    )
    _require(
        np.isnan(thresholds[..., ~mask]).all()
        and np.isnan(empirical[..., ~mask]).all(),
        f"{prefix} unsupported calendar fields are not NaN",
    )
    fit_indices = np.asarray(archive[f"{prefix}_fit_indices"], dtype=np.int64)
    _require(
        np.array_equal(fit_indices, np.arange(TRAIN_CASES, dtype=np.int64)),
        f"{prefix} fit indices differ",
    )
    sample_count = np.asarray(
        archive[f"{prefix}_calendar_sample_count"], dtype=np.int32
    )
    _require(sample_count.shape == (366,) and np.all(sample_count > 0), f"{prefix} sample counts differ")
    definitions = pbc_manifest.get("quantile_definitions", {})
    window_days = int(definitions.get("calendar_window_days", -1))
    minimum_samples = int(
        definitions.get("minimum_samples_with_nearest_calendar_fallback", -1)
    )
    _require(window_days == 31 and minimum_samples == 8, "PBC calendar contract differs")
    return pbc.CalendarQuantiles(
        levels=levels,
        thresholds=thresholds,
        empirical_cdf=empirical,
        support=mask.copy(),
        window_radius_days=(window_days - 1) // 2,
        minimum_samples=minimum_samples,
        fit_indices=fit_indices,
        sample_count_by_day=sample_count,
        unique_fit_window_count=int(
            np.asarray(archive[f"{prefix}_unique_fit_window_count"]).item()
        ),
        duplicate_fit_window_count=int(
            np.asarray(archive[f"{prefix}_duplicate_fit_window_count"]).item()
        ),
    )


def _load_frozen_family(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    support: np.ndarray,
    pbc_manifest: Mapping[str, Any],
    training_initializations: np.ndarray,
) -> FrozenFamily:
    model = _load_calendar_model(archive, prefix, support, pbc_manifest)
    fitting = pbc_manifest.get("fitting", {})
    _require(
        fitting.get("debias_candidate_half_spans_days") == list(DEBIAS_SPANS),
        "frozen Debias++ candidate set differs",
    )
    selected = np.asarray(
        archive[f"{prefix}_selected_debias_span_by_lead"], dtype=np.int16
    )
    expected_selected = np.full(LEAD_COUNT, 14, dtype=np.int16)
    _require(np.array_equal(selected, expected_selected), f"{prefix} selected spans differ")
    _require(
        fitting.get(f"{prefix}_selected_half_span_by_lead") == expected_selected.tolist(),
        f"{prefix} selected-span manifest differs",
    )
    debias_fits: list[pbc.DebiasFit] = []
    for span in DEBIAS_SPANS:
        correction = np.asarray(
            archive[f"{prefix}_debias_correction_span_{span}"], dtype=np.float32
        )
        expected_shape = (366, LEAD_COUNT, model.levels.size, *GRID_SHAPE)
        _require(correction.shape == expected_shape, f"{prefix} span-{span} shape differs")
        _require(
            np.isfinite(correction[..., support]).all()
            and np.isnan(correction[..., ~support]).all(),
            f"{prefix} span-{span} support differs",
        )
        counts = _calendar_sample_counts(
            training_initializations,
            model.fit_indices,
            span,
            minimum_samples=model.minimum_samples,
        )
        debias_fits.append(
            pbc.DebiasFit(
                correction=correction,
                half_window_days=span,
                fit_indices=model.fit_indices.copy(),
                sample_count_by_day_lead=counts,
            )
        )

    coefficients = np.asarray(
        archive[f"{prefix}_persistence_coefficients"], dtype=np.float32
    )
    expected_coefficients = (LEAD_COUNT, model.levels.size, *GRID_SHAPE, 5)
    _require(coefficients.shape == expected_coefficients, f"{prefix} persistence shape differs")
    _require(
        np.isfinite(coefficients[..., support, :]).all()
        and np.isnan(coefficients[..., ~support, :]).all(),
        f"{prefix} persistence support differs",
    )
    usable = np.asarray(
        archive[f"{prefix}_persistence_usable_fit_indices"], dtype=np.int64
    )
    _require(
        np.array_equal(usable, np.arange(4, TRAIN_CASES, dtype=np.int64))
        and usable.size == PERSISTENCE_USABLE_CASES,
        f"{prefix} persistence usable indices differ",
    )
    _require(
        math.isclose(
            float(fitting.get("persistence_ridge", np.nan)),
            PERSISTENCE_RIDGE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and tuple(fitting.get("persistence_feature_names", ()))
        == PERSISTENCE_FEATURE_NAMES
        and int(fitting.get("persistence_usable_training_cases", -1))
        == PERSISTENCE_USABLE_CASES,
        "frozen Persistence++ metadata differs",
    )
    persistence = pbc.PersistenceFit(
        coefficients=coefficients,
        ridge=PERSISTENCE_RIDGE,
        fit_indices=model.fit_indices.copy(),
        usable_fit_indices=usable,
        feature_names=PERSISTENCE_FEATURE_NAMES,
    )
    return FrozenFamily(
        name=prefix,
        model=model,
        debias_fits=tuple(debias_fits),
        selected_debias_spans=selected,
        persistence_fit=persistence,
    )


def load_frozen_pbc(
    pbc_root: Path,
    pbc_manifest: Mapping[str, Any],
    support: np.ndarray,
    training_initializations: np.ndarray,
) -> FrozenPBC:
    fit_path, digest = _artifact(pbc_root, pbc_manifest, "models/pbc_fit.npz")
    _require(digest == PBC_FIT_SHA256, "PBC fit changed after input gate")
    with np.load(fit_path, allow_pickle=False) as archive:
        quintile = _load_frozen_family(
            archive, "quintile", support, pbc_manifest, training_initializations
        )
        semidecile = _load_frozen_family(
            archive, "semidecile", support, pbc_manifest, training_initializations
        )
    return FrozenPBC(quintile, semidecile, fit_path, digest)


def load_pbc_static_support(
    pbc_root: Path, pbc_manifest: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    path, digest = _artifact(pbc_root, pbc_manifest, "evaluation/scoring_support.npz")
    with np.load(path, allow_pickle=False) as archive:
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
        support = np.asarray(archive["support_mask"], dtype=bool)
        weight_key = (
            "scoring_weight"
            if "scoring_weight" in archive
            else "scoring_weight_km2_fraction"
        )
        weights = np.asarray(archive[weight_key], dtype=np.float64)
    _require(
        latitude.shape == (27,)
        and longitude.shape == (27,)
        and support.shape == GRID_SHAPE
        and weights.shape == GRID_SHAPE,
        "PBC support geometry differs",
    )
    _require(np.array_equal(support, weights > 0.0), "PBC support/weight identity differs")
    return latitude, longitude, support, weights, digest


def _daily_fraction_array(
    dataset: xr.Dataset, positions: np.ndarray, expected_days: int
) -> tuple[np.ndarray, np.ndarray]:
    fraction = dataset["observation_fraction"]
    if fraction.dims == ("latitude", "longitude"):
        stored = np.asarray(fraction.load().values, dtype=np.float32)
        _require(stored.shape == GRID_SHAPE, "late-2021 static fraction shape differs")
        selected = np.broadcast_to(stored[None], (expected_days, *GRID_SHAPE)).copy()
    elif fraction.dims == ("time", "latitude", "longitude"):
        stored = np.asarray(fraction.isel(time=positions).load().values, dtype=np.float32)
        _require(stored.shape == (expected_days, *GRID_SHAPE), "late-2021 dynamic fraction shape differs")
        selected = stored.copy()
    else:
        raise OperationalCategoricalError("late-2021 observation_fraction dimensions differ")
    return selected, stored


def load_late_2021_lag_slice(
    initializations: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    reference_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    """Load only the late-2021 days explicitly required by 2022 issue lags."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    requested = np.unique(
        np.concatenate(
            [
                issue
                - np.timedelta64(7 * lag, "D")
                + np.arange(7).astype("timedelta64[D]")
                for issue in starts
                for lag in LAG_WEEKS
            ]
        )
    )
    late = requested[pd.DatetimeIndex(requested).year.to_numpy() == 2021]
    _require(late.size > 0, "operational cases do not request a late-2021 lag slice")
    _require(
        np.min(late) >= np.datetime64("2021-12-01")
        and np.max(late) <= np.datetime64("2021-12-31")
        and np.all(np.diff(late) == np.timedelta64(1, "D")),
        "requested lag-only 2021 dates are not one narrow late-December slice",
    )
    store = operational.IMD_ROOT / "2021.zarr"
    _reject_forbidden_year_store(store)
    _require(store.name == "2021.zarr" and (store / ".zmetadata").is_file(), "late-2021 store is unavailable")
    with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
        _require(
            dataset.attrs.get("source") == "imd"
            and dataset.attrs.get("variable") == "tp"
            and dataset.attrs.get("units") == "mm day-1",
            "late-2021 IMD metadata differs",
        )
        _require(
            np.array_equal(np.asarray(dataset.latitude.values, dtype=np.float64), latitude)
            and np.array_equal(np.asarray(dataset.longitude.values, dtype=np.float64), longitude),
            "late-2021 IMD grid differs",
        )
        calendar = np.asarray(dataset.time.values, dtype="datetime64[D]")
        _require(
            calendar.size == 365
            and np.all(np.diff(calendar) == np.timedelta64(1, "D"))
            and set(pd.DatetimeIndex(calendar).year) == {2021},
            "late-2021 IMD calendar is incomplete",
        )
        positions = np.searchsorted(calendar, late)
        _require(
            np.all(positions < calendar.size)
            and np.array_equal(calendar[positions], late),
            "one or more requested late-2021 lag days is absent",
        )
        values = np.asarray(
            dataset["observation"].isel(time=positions).load().values,
            dtype=np.float32,
        )
        fractions, stored_fraction = _daily_fraction_array(dataset, positions, late.size)
    _require(values.shape == (late.size, *GRID_SHAPE), "late-2021 IMD slice shape differs")
    _require(
        np.allclose(
            fractions[0], reference_fraction, rtol=0.0, atol=1.0e-7, equal_nan=True
        )
        and np.allclose(
            fractions, reference_fraction[None], rtol=0.0, atol=1.0e-7, equal_nan=True
        ),
        "late-2021 lag-only observation coverage differs from frozen support",
    )
    support = fractions > 0.0
    _require(
        np.isfinite(fractions).all()
        and np.all(fractions >= 0.0)
        and np.isfinite(values[support]).all()
        and np.all(values[support] >= 0.0),
        "late-2021 lag-only values/fractions are invalid",
    )
    receipt = {
        "access_contract": "literal 2021.zarr; only exact requested late-December positions loaded",
        "store": str(store),
        "zmetadata_sha256": sha256_file(store / ".zmetadata"),
        "selected_first_date": np.datetime_as_string(late[0], unit="D"),
        "selected_last_date": np.datetime_as_string(late[-1], unit="D"),
        "selected_day_count": int(late.size),
        "selected_dates": operational.array_receipt(late.astype("<i8")),
        "selected_observation": operational.array_receipt(values.astype("<f4", copy=False)),
        "selected_observation_fraction": operational.array_receipt(
            fractions.astype("<f4", copy=False)
        ),
        "stored_fraction_payload": operational.array_receipt(
            stored_fraction.astype("<f4", copy=False)
        ),
        "no_other_2018_2021_observation_values_loaded": True,
    }
    return late, values, fractions, receipt, str(store)


def build_fraction_weighted_issue_lags(
    initializations: np.ndarray,
    daily_dates: np.ndarray,
    daily_values: np.ndarray,
    daily_fraction: np.ndarray,
    reference_fraction: np.ndarray,
    support: np.ndarray,
    *,
    smoke: bool,
) -> tuple[pbc.IssueTimeLags, dict[str, Any]]:
    """Build complete issue-14..issue-1 lags with daily coverage weighting."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    dates = np.asarray(daily_dates, dtype="datetime64[D]")
    values = np.asarray(daily_values, dtype=np.float32)
    fractions = np.asarray(daily_fraction, dtype=np.float32)
    frozen = np.asarray(reference_fraction, dtype=np.float32)
    mask = np.asarray(support, dtype=bool)
    _require(
        starts.ndim == 1
        and np.unique(starts).size == starts.size
        and dates.ndim == 1
        and values.shape == fractions.shape == (dates.size, *GRID_SHAPE)
        and frozen.shape == mask.shape == GRID_SHAPE,
        "lag construction arrays are misaligned",
    )
    _require(
        np.unique(dates).size == dates.size
        and np.all(np.diff(dates) == np.timedelta64(1, "D")),
        "lag daily calendar is not unique and continuous",
    )
    lag_values = np.full((starts.size, len(LAG_WEEKS), *GRID_SHAPE), np.nan, dtype=np.float32)
    source_indices = np.full((starts.size, len(LAG_WEEKS)), -1, dtype=np.int64)
    window_start = np.full((starts.size, len(LAG_WEEKS)), np.datetime64("NaT"), dtype="datetime64[D]")
    window_end = window_start.copy()
    fraction_sums = np.zeros((starts.size, len(LAG_WEEKS), *GRID_SHAPE), dtype=np.float64)
    affected_blocks: list[tuple[str, int, int, str]] = []
    for case, issue in enumerate(starts):
        for column, lag in enumerate(LAG_WEEKS):
            requested_start = issue - np.timedelta64(7 * lag, "D")
            requested = requested_start + np.arange(7).astype("timedelta64[D]")
            position = int(np.searchsorted(dates, requested_start))
            stop = position + 7
            _require(
                position >= 0
                and stop <= dates.size
                and np.array_equal(dates[position:stop], requested),
                f"incomplete issue-time lag for {issue} lag {lag}",
            )
            _require(requested[-1] < issue, "lag window reaches forecast issue time")
            block_values = values[position:stop]
            block_fraction = fractions[position:stop]
            valid = (
                (block_fraction > 0.0)
                & np.isfinite(block_fraction)
                & np.isfinite(block_values)
            )
            denominator = np.sum(
                np.where(valid, block_fraction, 0.0), axis=0, dtype=np.float64
            )
            numerator = np.sum(
                np.where(valid, block_values, 0.0).astype(np.float64)
                * np.where(valid, block_fraction, 0.0).astype(np.float64),
                axis=0,
                dtype=np.float64,
            )
            _require(
                np.all(denominator[mask] > 0.0),
                f"lag window has no supported coverage for {issue} lag {lag}",
            )
            field = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=field, where=denominator > 0.0)
            _require(
                np.isfinite(field[mask]).all() and np.all(field[mask] >= 0.0),
                "weighted lag contains invalid supported precipitation",
            )
            lag_values[case, column] = field.astype(np.float32)
            source_indices[case, column] = position
            window_start[case, column] = requested_start
            window_end[case, column] = requested[-1]
            fraction_sums[case, column] = denominator
            changed_days = np.flatnonzero(
                np.any(
                    ~np.isclose(
                        block_fraction,
                        frozen[None],
                        rtol=0.0,
                        atol=1.0e-7,
                        equal_nan=False,
                    ),
                    axis=(-2, -1),
                )
            )
            for day in changed_days:
                affected_blocks.append(
                    (
                        np.datetime_as_string(issue, unit="D"),
                        int(lag),
                        int(day) + 1,
                        np.datetime_as_string(requested[day], unit="D"),
                    )
                )
    expected_affected = () if smoke else EXPECTED_FULL_LAG_COVERAGE_BLOCKS
    _require(
        tuple(affected_blocks) == expected_affected,
        f"lag coverage-anomaly blocks differ: {affected_blocks}",
    )
    lags = pbc.IssueTimeLags(
        values=lag_values,
        source_indices=source_indices,
        window_start=window_start,
        window_end=window_end,
    )
    _require(lags.available.all(), "one or more operational Persistence++ lags is unavailable")
    provenance = {
        "schema_version": "fraction_weighted_issue_time_lags_v1",
        "lag_weeks": list(LAG_WEEKS),
        "window_contract": {
            "lag_1": "issue-7 through issue-1",
            "lag_2": "issue-14 through issue-8",
            "all_fourteen_days_required": True,
            "fallback_or_imputation": False,
        },
        "aggregation": "sum(observation x observation_fraction) / sum(observation_fraction) independently per cell",
        "all_lags_available": True,
        "case_count": int(starts.size),
        "affected_coverage_blocks": [
            {
                "initialization": init,
                "lag_week": lag,
                "day_within_lag": day,
                "date": date,
            }
            for init, lag, day, date in affected_blocks
        ],
        "affected_coverage_block_count": len(affected_blocks),
        "lag_values": operational.array_receipt(lag_values.astype("<f4", copy=False)),
        "source_indices": operational.array_receipt(source_indices.astype("<i8", copy=False)),
        "window_start": operational.array_receipt(window_start.astype("<i8", copy=False)),
        "window_end": operational.array_receipt(window_end.astype("<i8", copy=False)),
        "fraction_sum": operational.array_receipt(fraction_sums.astype("<f8", copy=False)),
    }
    return lags, provenance


def dynamic_weighted_spatial_mean(
    values: np.ndarray, dynamic_weights: np.ndarray
) -> np.ndarray:
    """Reduce the final grid dimensions using case/lead-specific weights."""

    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(dynamic_weights, dtype=np.float64)
    _require(
        array.ndim >= 4
        and array.shape[:2] == weights.shape[:2]
        and array.shape[-2:] == weights.shape[-2:]
        and weights.ndim == 4,
        "dynamic weights do not align with scored fields",
    )
    expanded = weights
    for _ in range(array.ndim - weights.ndim):
        expanded = np.expand_dims(expanded, axis=2)
    expanded = np.broadcast_to(expanded, array.shape)
    valid = np.isfinite(array) & np.isfinite(expanded) & (expanded > 0.0)
    numerator = np.sum(
        np.where(valid, array * expanded, 0.0), axis=(-2, -1), dtype=np.float64
    )
    denominator = np.sum(
        np.where(valid, expanded, 0.0), axis=(-2, -1), dtype=np.float64
    )
    _require(np.all(denominator > 0.0), "one or more categorical fields has no scoring weight")
    return numerator / denominator


def _upper_brier_dynamic(
    cdf: np.ndarray,
    observed: np.ndarray,
    thresholds: np.ndarray,
    dynamic_weights: np.ndarray,
) -> np.ndarray:
    _require(
        pbc.is_valid_cdf_for_thresholds(cdf, thresholds, axis=2)
        and pbc.is_valid_cdf_for_thresholds(observed, thresholds, axis=2),
        "q95 CDF inputs violate equality-aware geometry",
    )
    score = np.where(
        thresholds[:, :, -1] > 0.0,
        ((1.0 - cdf[:, :, -1]) - (1.0 - observed[:, :, -1])) ** 2,
        np.nan,
    )
    return dynamic_weighted_spatial_mean(score, dynamic_weights)


def build_family_reference(
    family: FrozenFamily,
    initializations: np.ndarray,
    truth: np.ndarray,
    dynamic_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FamilyReference]:
    thresholds = pbc.calendar_fields(family.model, initializations, LEAD_COUNT)
    observed = pbc.observation_cdf(truth, thresholds)
    empirical = pbc.project_cdf_for_thresholds(
        pbc.calendar_fields(
            family.model, initializations, LEAD_COUNT, empirical=True
        ),
        thresholds,
        axis=2,
    )
    nominal = pbc.project_cdf_for_thresholds(
        np.where(
            np.isfinite(thresholds),
            np.broadcast_to(
                family.model.levels[None, None, :, None, None],
                thresholds.shape,
            ),
            np.nan,
        ),
        thresholds,
        axis=2,
    )
    for name, values in (("observed", observed), ("empirical", empirical), ("nominal", nominal)):
        _require(
            pbc.is_valid_cdf_for_thresholds(values, thresholds, axis=2),
            f"{family.name} {name} CDF violates equality-aware geometry",
        )
    nominal_rps = dynamic_weighted_spatial_mean(
        pbc.ranked_probability_score(nominal, observed, thresholds), dynamic_weights
    )
    empirical_rps = dynamic_weighted_spatial_mean(
        pbc.ranked_probability_score(empirical, observed, thresholds), dynamic_weights
    )
    paper_reference: np.ndarray | None = None
    nominal_upper: np.ndarray | None = None
    empirical_upper: np.ndarray | None = None
    if family.name == "quintile":
        paper_reference = dynamic_weighted_spatial_mean(
            pbc.paper_fixed_nominal_reference_rps(observed, thresholds),
            dynamic_weights,
        )
    else:
        nominal_upper = _upper_brier_dynamic(
            nominal, observed, thresholds, dynamic_weights
        )
        empirical_upper = _upper_brier_dynamic(
            empirical, observed, thresholds, dynamic_weights
        )
    return thresholds, observed, empirical, FamilyReference(
        projected_nominal_rps=nominal_rps,
        empirical_rps=empirical_rps,
        paper_fixed_nominal_rps=paper_reference,
        projected_nominal_upper_brier=nominal_upper,
        empirical_upper_brier=empirical_upper,
    )


def score_cdf(
    family_name: str,
    cdf: np.ndarray,
    observed: np.ndarray,
    thresholds: np.ndarray,
    dynamic_weights: np.ndarray,
) -> FamilyScore:
    _require(
        pbc.is_valid_cdf_for_thresholds(cdf, thresholds, axis=2),
        f"{family_name} forecast CDF violates equality-aware geometry",
    )
    rps = dynamic_weighted_spatial_mean(
        pbc.ranked_probability_score(cdf, observed, thresholds), dynamic_weights
    )
    probability_bias = dynamic_weighted_spatial_mean(
        np.asarray(cdf, dtype=np.float64) - np.asarray(observed, dtype=np.float64),
        dynamic_weights,
    )
    paper: np.ndarray | None = None
    upper: np.ndarray | None = None
    if family_name == "quintile":
        paper = dynamic_weighted_spatial_mean(
            pbc.paper_nominal_cut_rps(cdf, observed, thresholds), dynamic_weights
        )
    else:
        upper = _upper_brier_dynamic(cdf, observed, thresholds, dynamic_weights)
    _require(
        np.isfinite(rps).all()
        and np.all((rps >= 0.0) & (rps <= 1.0))
        and np.isfinite(probability_bias).all(),
        f"{family_name} categorical scores are invalid",
    )
    return FamilyScore(rps, probability_bias, paper, upper)


def score_ensemble(
    members: np.ndarray,
    family_name: str,
    thresholds: np.ndarray,
    observed: np.ndarray,
    dynamic_weights: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, FamilyScore]:
    cdf = pbc.ensemble_cdf(members, thresholds, chunk_size=chunk_size)
    return cdf, score_cdf(
        family_name, cdf, observed, thresholds, dynamic_weights
    )


def apply_frozen_pbc(
    raw_cdf: np.ndarray,
    family: FrozenFamily,
    initializations: np.ndarray,
    thresholds: np.ndarray,
    empirical_climatology: np.ndarray,
    lag_cdf: np.ndarray,
) -> dict[str, np.ndarray]:
    available = np.ones(len(initializations), dtype=bool)
    debias = pbc.apply_selected_debias(
        raw_cdf,
        initializations,
        family.debias_fits,
        family.selected_debias_spans,
        thresholds,
        project=True,
    )
    persistence = pbc.apply_persistence(
        raw_cdf,
        empirical_climatology,
        lag_cdf,
        available,
        family.persistence_fit,
        thresholds,
        project=True,
    )
    combined = pbc.combine_projected_components(
        debias,
        persistence,
        thresholds,
        debias_weight=BLEND_DEBIAS_WEIGHT,
    )
    result = {
        "debias_plus_plus": debias,
        "persistence_plus_plus": persistence,
        "pbc_combined": combined,
    }
    for method, values in result.items():
        _require(
            pbc.is_valid_cdf_for_thresholds(values, thresholds, axis=2),
            f"{family.name}/{method} output CDF is invalid",
        )
    return result


def _verification_dates(initializations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = initializations[:, None] + (
        7 * np.arange(LEAD_COUNT, dtype=np.int64)
    ).astype("timedelta64[D]")[None]
    midpoints = starts + np.timedelta64(3, "D")
    ends = starts + np.timedelta64(6, "D")
    return starts, midpoints, ends


def rps_case_frame(
    family_name: str,
    method: str,
    seed: int | str,
    scores: FamilyScore,
    reference: FamilyReference,
    initializations: np.ndarray,
    thresholds: np.ndarray,
    dynamic_weights: np.ndarray,
) -> pd.DataFrame:
    starts, midpoints, ends = _verification_dates(initializations)
    seasons = operational.verification_seasons(initializations)
    total_weight = np.sum(dynamic_weights, axis=(-2, -1), dtype=np.float64)
    q80_weight: np.ndarray | None = None
    if family_name == "quintile":
        q80_weight = np.sum(
            np.where(thresholds[:, :, -1] > 0.0, dynamic_weights, 0.0),
            axis=(-2, -1),
            dtype=np.float64,
        )
        _require(np.all(q80_weight > 0.0), "one or more case/leads has no positive-q80 weight")
        _require(
            scores.paper_q80_rps is not None
            and reference.paper_fixed_nominal_rps is not None,
            "quintile score lacks q80 sensitivity",
        )
    records: list[dict[str, Any]] = []
    for case, initialization in enumerate(initializations):
        for lead in range(LEAD_COUNT):
            score = float(scores.rps[case, lead])
            nominal = float(reference.projected_nominal_rps[case, lead])
            empirical = float(reference.empirical_rps[case, lead])
            row: dict[str, Any] = {
                "family": family_name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "score_contract": SCORING_CONTRACT_VERSION,
                "nominal_reference_contract": NOMINAL_REFERENCE_CONTRACT,
                "initialization": np.datetime_as_string(initialization, unit="D"),
                "year": int(pd.Timestamp(initialization).year),
                "lead_week": lead + 1,
                "verification_start": np.datetime_as_string(starts[case, lead], unit="D"),
                "verification_midpoint": np.datetime_as_string(midpoints[case, lead], unit="D"),
                "verification_end": np.datetime_as_string(ends[case, lead], unit="D"),
                "season": str(seasons[case, lead]),
                "dynamic_scoring_weight": float(total_weight[case, lead]),
                "rps": score,
                "nominal_climatology_rps": nominal,
                "training_empirical_climatology_rps": empirical,
                "rpss_vs_nominal_climatology_case": (
                    1.0 - score / nominal if nominal > 0.0 else np.nan
                ),
                "rpss_vs_training_empirical_climatology_case": (
                    1.0 - score / empirical if empirical > 0.0 else np.nan
                ),
            }
            if family_name == "quintile":
                assert q80_weight is not None
                assert scores.paper_q80_rps is not None
                assert reference.paper_fixed_nominal_rps is not None
                paper = float(scores.paper_q80_rps[case, lead])
                paper_reference = float(reference.paper_fixed_nominal_rps[case, lead])
                row.update(
                    {
                        "paper_nominal_cut_q80_positive_rps": paper,
                        "paper_fixed_nominal_climatology_rps": paper_reference,
                        "paper_q80_retained_scoring_weight": float(q80_weight[case, lead]),
                        "paper_q80_total_supported_scoring_weight": float(total_weight[case, lead]),
                        "paper_q80_retained_scoring_weight_fraction": float(
                            q80_weight[case, lead] / total_weight[case, lead]
                        ),
                        "paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case": (
                            1.0 - paper / paper_reference
                            if paper_reference > 0.0
                            else np.nan
                        ),
                    }
                )
            records.append(row)
    return pd.DataFrame.from_records(records)


def upper_case_frame(
    method: str,
    seed: int | str,
    scores: FamilyScore,
    reference: FamilyReference,
    initializations: np.ndarray,
    thresholds: np.ndarray,
    dynamic_weights: np.ndarray,
) -> pd.DataFrame:
    _require(
        scores.upper_brier is not None
        and reference.projected_nominal_upper_brier is not None
        and reference.empirical_upper_brier is not None,
        "semidecile score lacks q95 Brier fields",
    )
    _, midpoints, _ = _verification_dates(initializations)
    seasons = operational.verification_seasons(initializations)
    total_weight = np.sum(dynamic_weights, axis=(-2, -1), dtype=np.float64)
    retained = np.sum(
        np.where(thresholds[:, :, -1] > 0.0, dynamic_weights, 0.0),
        axis=(-2, -1),
        dtype=np.float64,
    )
    _require(np.all(retained > 0.0), "one or more case/leads has no positive-q95 weight")
    records: list[dict[str, Any]] = []
    for case, initialization in enumerate(initializations):
        for lead in range(LEAD_COUNT):
            records.append(
                {
                    "event": "upper_5pct_informative_positive_q95",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "initialization": np.datetime_as_string(initialization, unit="D"),
                    "year": int(pd.Timestamp(initialization).year),
                    "lead_week": lead + 1,
                    "verification_midpoint": np.datetime_as_string(midpoints[case, lead], unit="D"),
                    "season": str(seasons[case, lead]),
                    "brier_score": float(scores.upper_brier[case, lead]),
                    "nominal_climatology_brier_score": float(
                        reference.projected_nominal_upper_brier[case, lead]
                    ),
                    "training_empirical_climatology_brier_score": float(
                        reference.empirical_upper_brier[case, lead]
                    ),
                    "q95_retained_scoring_weight": float(retained[case, lead]),
                    "q95_total_supported_scoring_weight": float(total_weight[case, lead]),
                    "q95_retained_scoring_weight_fraction": float(
                        retained[case, lead] / total_weight[case, lead]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def probability_bias_frame(
    family_name: str,
    method: str,
    seed: int | str,
    bias: np.ndarray,
    initializations: np.ndarray,
) -> pd.DataFrame:
    seasons = operational.verification_seasons(initializations)
    levels = LEVELS[family_name]
    _require(
        bias.shape == (len(initializations), LEAD_COUNT, levels.size),
        f"{family_name} probability-bias shape differs",
    )
    records: list[dict[str, Any]] = []
    for lead in range(LEAD_COUNT):
        groups: list[tuple[str, np.ndarray]] = [
            ("ALL", np.ones(len(initializations), dtype=bool))
        ]
        groups.extend(
            (season, seasons[:, lead] == season)
            for season in ("DJF", "MAM", "JJA", "SON")
        )
        for season, selected in groups:
            if not np.any(selected):
                continue
            for cut, nominal in enumerate(levels):
                values = bias[selected, lead, cut]
                records.append(
                    {
                        "family": family_name,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": seed,
                        "lead_week": lead + 1,
                        "season": season,
                        "nominal_cumulative_probability": float(nominal),
                        "probability_bias": float(np.mean(values)),
                        "mean_absolute_case_probability_bias": float(
                            np.mean(np.abs(values))
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def average_neural_rps_scores(frame: pd.DataFrame, *, quintile: bool) -> pd.DataFrame:
    """Average per-seed proper scores; never average forecast probabilities."""

    _require(set(frame.method) == {"base_42k"}, "neural seed RPS frame has another method")
    identity = ["family", "method", "initialization", "lead_week", "seed"]
    _require(not frame.duplicated(identity).any(), "duplicate neural seed RPS row")
    counts = frame.groupby(["family", "method", "initialization", "lead_week"]).seed.nunique()
    _require(np.all(counts.to_numpy() == len(SEEDS)), "neural RPS cases lack three seeds")
    _require(set(pd.to_numeric(frame.seed).astype(int)) == set(SEEDS), "neural seed identities differ")
    keys = [
        "family",
        "method",
        "method_label",
        "score_contract",
        "nominal_reference_contract",
        "initialization",
        "year",
        "lead_week",
        "verification_start",
        "verification_midpoint",
        "verification_end",
        "season",
    ]
    reference_columns = [
        "dynamic_scoring_weight",
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
    ]
    if quintile:
        reference_columns.extend(
            [
                "paper_fixed_nominal_climatology_rps",
                "paper_q80_retained_scoring_weight",
                "paper_q80_total_supported_scoring_weight",
                "paper_q80_retained_scoring_weight_fraction",
            ]
        )
    spread = frame.groupby(["family", "method", "initialization", "lead_week"])[
        reference_columns
    ].agg(lambda values: float(values.max() - values.min()))
    _require(
        np.allclose(spread.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12),
        "neural climatology/support references differ across seeds",
    )
    aggregations: dict[str, tuple[str, str]] = {
        "dynamic_scoring_weight": ("dynamic_scoring_weight", "first"),
        "rps": ("rps", "mean"),
        "nominal_climatology_rps": ("nominal_climatology_rps", "first"),
        "training_empirical_climatology_rps": (
            "training_empirical_climatology_rps",
            "first",
        ),
    }
    if quintile:
        aggregations.update(
            {
                "paper_nominal_cut_q80_positive_rps": (
                    "paper_nominal_cut_q80_positive_rps",
                    "mean",
                ),
                "paper_fixed_nominal_climatology_rps": (
                    "paper_fixed_nominal_climatology_rps",
                    "first",
                ),
                "paper_q80_retained_scoring_weight": (
                    "paper_q80_retained_scoring_weight",
                    "first",
                ),
                "paper_q80_total_supported_scoring_weight": (
                    "paper_q80_total_supported_scoring_weight",
                    "first",
                ),
                "paper_q80_retained_scoring_weight_fraction": (
                    "paper_q80_retained_scoring_weight_fraction",
                    "first",
                ),
            }
        )
    result = (
        frame.groupby(keys, as_index=False)
        .agg(**aggregations)
        .sort_values(["method", "initialization", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(3, "seed", "mean_of_seed_scores_42_43_44")
    result["rpss_vs_nominal_climatology_case"] = (
        1.0 - result.rps / result.nominal_climatology_rps
    )
    result["rpss_vs_training_empirical_climatology_case"] = (
        1.0 - result.rps / result.training_empirical_climatology_rps
    )
    if quintile:
        result["paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case"] = (
            1.0
            - result.paper_nominal_cut_q80_positive_rps
            / result.paper_fixed_nominal_climatology_rps
        )
    return result


def average_neural_upper_scores(frame: pd.DataFrame) -> pd.DataFrame:
    _require(set(frame.method) == {"base_42k"}, "neural upper frame has another method")
    identity = ["method", "initialization", "lead_week", "seed"]
    _require(not frame.duplicated(identity).any(), "duplicate neural upper row")
    counts = frame.groupby(["method", "initialization", "lead_week"]).seed.nunique()
    _require(np.all(counts.to_numpy() == len(SEEDS)), "neural upper cases lack three seeds")
    _require(
        set(pd.to_numeric(frame.seed).astype(int)) == set(SEEDS),
        "neural upper seed identities differ",
    )
    keys = [
        "event",
        "method",
        "method_label",
        "initialization",
        "year",
        "lead_week",
        "verification_midpoint",
        "season",
    ]
    reference_columns = [
        "nominal_climatology_brier_score",
        "training_empirical_climatology_brier_score",
        "q95_retained_scoring_weight",
        "q95_total_supported_scoring_weight",
        "q95_retained_scoring_weight_fraction",
    ]
    spread = frame.groupby(["method", "initialization", "lead_week"])[
        reference_columns
    ].agg(lambda values: float(values.max() - values.min()))
    _require(
        np.allclose(spread.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12),
        "neural q95 references differ across seeds",
    )
    result = (
        frame.groupby(keys, as_index=False)
        .agg(
            brier_score=("brier_score", "mean"),
            nominal_climatology_brier_score=(
                "nominal_climatology_brier_score",
                "first",
            ),
            training_empirical_climatology_brier_score=(
                "training_empirical_climatology_brier_score",
                "first",
            ),
            q95_retained_scoring_weight=("q95_retained_scoring_weight", "first"),
            q95_total_supported_scoring_weight=(
                "q95_total_supported_scoring_weight",
                "first",
            ),
            q95_retained_scoring_weight_fraction=(
                "q95_retained_scoring_weight_fraction",
                "first",
            ),
        )
        .sort_values(["method", "initialization", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(3, "seed", "mean_of_seed_scores_42_43_44")
    return result


def average_neural_probability_bias(frame: pd.DataFrame) -> pd.DataFrame:
    _require(set(frame.method) == {"base_42k"}, "neural bias frame has another method")
    keys = [
        "family",
        "method",
        "method_label",
        "lead_week",
        "season",
        "nominal_cumulative_probability",
    ]
    _require(not frame.duplicated([*keys, "seed"]).any(), "duplicate neural bias row")
    counts = frame.groupby(keys).seed.nunique()
    _require(np.all(counts.to_numpy() == len(SEEDS)), "neural bias rows lack three seeds")
    _require(
        set(pd.to_numeric(frame.seed).astype(int)) == set(SEEDS),
        "neural bias seed identities differ",
    )
    result = (
        frame.groupby(keys, as_index=False)
        .agg(
            probability_bias=("probability_bias", "mean"),
            mean_absolute_case_probability_bias=(
                "mean_absolute_case_probability_bias",
                "mean",
            ),
        )
        .sort_values(["family", "method", "season", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(3, "seed", "mean_of_seed_scores_42_43_44")
    return result


def aggregate_rps_scores(
    frame: pd.DataFrame, group_columns: Sequence[str], *, quintile: bool
) -> pd.DataFrame:
    prepared = frame.copy()
    aggregations: dict[str, tuple[str, str]] = {
        "score_contract": ("score_contract", "first"),
        "nominal_reference_contract": ("nominal_reference_contract", "first"),
        "rps": ("rps", "mean"),
        "nominal_climatology_rps": ("nominal_climatology_rps", "mean"),
        "training_empirical_climatology_rps": (
            "training_empirical_climatology_rps",
            "mean",
        ),
        "n_initializations": ("initialization", "nunique"),
    }
    if quintile:
        prepared["_paper_score_sum"] = (
            prepared.paper_nominal_cut_q80_positive_rps
            * prepared.paper_q80_retained_scoring_weight
        )
        prepared["_paper_reference_sum"] = (
            prepared.paper_fixed_nominal_climatology_rps
            * prepared.paper_q80_retained_scoring_weight
        )
        aggregations.update(
            {
                "_paper_score_sum": ("_paper_score_sum", "sum"),
                "_paper_reference_sum": ("_paper_reference_sum", "sum"),
                "paper_q80_retained_scoring_weight": (
                    "paper_q80_retained_scoring_weight",
                    "sum",
                ),
            }
        )
    result = prepared.groupby(
        ["family", *group_columns, "method", "method_label"], as_index=False
    ).agg(**aggregations)
    result["rpss_vs_nominal_climatology"] = (
        1.0 - result.rps / result.nominal_climatology_rps
    )
    result["rpss_vs_training_empirical_climatology"] = (
        1.0 - result.rps / result.training_empirical_climatology_rps
    )
    if quintile:
        result["paper_nominal_cut_q80_positive_rps"] = (
            result._paper_score_sum / result.paper_q80_retained_scoring_weight
        )
        result["paper_fixed_nominal_climatology_rps"] = (
            result._paper_reference_sum / result.paper_q80_retained_scoring_weight
        )
        result["paper_nominal_cut_rpss_vs_fixed_nominal_climatology"] = (
            1.0
            - result.paper_nominal_cut_q80_positive_rps
            / result.paper_fixed_nominal_climatology_rps
        )
        result = result.drop(columns=["_paper_score_sum", "_paper_reference_sum"])
    return result


def aggregate_upper_scores(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, label, lead), rows in frame.groupby(
        ["method", "method_label", "lead_week"]
    ):
        for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
            selected = rows if season == "ALL" else rows.loc[rows.season.eq(season)]
            if selected.empty:
                continue
            score = float(selected.brier_score.mean())
            nominal = float(selected.nominal_climatology_brier_score.mean())
            empirical = float(selected.training_empirical_climatology_brier_score.mean())
            records.append(
                {
                    "event": "upper_5pct_informative_positive_q95",
                    "method": method,
                    "method_label": label,
                    "lead_week": int(lead),
                    "season": season,
                    "brier_score": score,
                    "nominal_climatology_brier_score": nominal,
                    "training_empirical_climatology_brier_score": empirical,
                    "bss_vs_nominal_climatology": 1.0 - score / nominal,
                    "bss_vs_training_empirical_climatology": 1.0 - score / empirical,
                    "n_initializations": int(selected.initialization.nunique()),
                }
            )
    return pd.DataFrame.from_records(records)


def _metric_scope_table(
    frame: pd.DataFrame,
    value_column: str,
    leads: Sequence[int],
    *,
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    selected = frame.loc[frame.lead_week.isin(leads)].copy()
    labels = sorted(selected.initialization.unique().tolist())
    if weight_column is None:
        pooled = selected.groupby(["initialization", "method"], as_index=False).agg(
            score=(value_column, "mean")
        )
        pivot = pooled.pivot(index="initialization", columns="method", values="score").reindex(labels)
        _require(set(pivot.columns) == set(METHODS) and not pivot.isna().any().any(), "bootstrap metric is not fully paired")
        return pivot.loc[:, METHODS].to_numpy(dtype=np.float64), None, labels
    selected["_numerator"] = selected[value_column] * selected[weight_column]
    pooled = selected.groupby(["initialization", "method"], as_index=False).agg(
        numerator=("_numerator", "sum"), weight=(weight_column, "sum")
    )
    score = pooled.assign(score=pooled.numerator / pooled.weight)
    score_pivot = score.pivot(index="initialization", columns="method", values="score").reindex(labels)
    weight_pivot = score.pivot(index="initialization", columns="method", values="weight").reindex(labels)
    _require(
        set(score_pivot.columns) == set(METHODS)
        and not score_pivot.isna().any().any()
        and not weight_pivot.isna().any().any(),
        "weighted bootstrap metric is not fully paired",
    )
    weights = weight_pivot.loc[:, METHODS].to_numpy(dtype=np.float64)
    _require(
        np.allclose(weights, weights[:, :1], rtol=0.0, atol=1.0e-8),
        "threshold-only bootstrap weights differ between methods",
    )
    return score_pivot.loc[:, METHODS].to_numpy(dtype=np.float64), weights, labels


def _draw_means(
    values: np.ndarray,
    plan: operational.TwoStageBootstrap,
    weights: np.ndarray | None,
) -> np.ndarray:
    output = np.empty((len(plan.draws), values.shape[1]), dtype=np.float64)
    draw_lengths = np.asarray([len(draw) for draw in plan.draws], dtype=np.int64)
    for length in np.unique(draw_lengths):
        rows = np.flatnonzero(draw_lengths == length)
        indices = np.stack([plan.draws[int(row)] for row in rows], axis=0)
        if weights is None:
            output[rows] = np.mean(values[indices], axis=1, dtype=np.float64)
        else:
            sampled_weights = weights[indices]
            output[rows] = np.sum(
                values[indices] * sampled_weights, axis=1, dtype=np.float64
            ) / np.sum(sampled_weights, axis=1, dtype=np.float64)
    return output


def paired_two_stage_bootstrap(
    quintile_case: pd.DataFrame,
    semidecile_case: pd.DataFrame,
    upper_case: pd.DataFrame,
    initializations: np.ndarray,
    plan: operational.TwoStageBootstrap,
) -> pd.DataFrame:
    metric_inputs = (
        (
            "quintile_effective_positive_cut_rps",
            quintile_case,
            "rps",
            None,
            SCORING_CONTRACT_VERSION,
        ),
        (
            "semidecile_effective_positive_cut_rps",
            semidecile_case,
            "rps",
            None,
            SCORING_CONTRACT_VERSION,
        ),
        (
            "quintile_q80_nominal_sum_rps",
            quintile_case,
            "paper_nominal_cut_q80_positive_rps",
            "paper_q80_retained_scoring_weight",
            "guan_four_nominal_cut_sum_q80_positive_dynamic_v1",
        ),
        (
            "upper_q95_positive_brier",
            upper_case,
            "brier_score",
            None,
            "informative_positive_q95_dynamic_brier_v1",
        ),
    )
    expected_labels = [np.datetime_as_string(value, unit="D") for value in initializations]
    records: list[dict[str, Any]] = []
    lead_scopes = [("W1-W6", tuple(range(1, 7)))] + [
        (f"W{lead}", (lead,)) for lead in range(1, 7)
    ]
    for metric, frame, value_column, weight_column, metric_contract in metric_inputs:
        _require(set(frame.method) == set(METHODS), f"{metric} methods are incomplete")
        _require(
            not frame.duplicated(["method", "initialization", "lead_week"]).any(),
            f"{metric} rows are not uniquely paired",
        )
        for lead_scope, leads in lead_scopes:
            values, weights, labels = _metric_scope_table(
                frame, value_column, leads, weight_column=weight_column
            )
            _require(labels == expected_labels, f"{metric}/{lead_scope} case order differs")
            draws = _draw_means(values, plan, weights)
            if weights is None:
                points = np.mean(values, axis=0, dtype=np.float64)
                retained_weight = np.nan
            else:
                points = np.sum(values * weights, axis=0, dtype=np.float64) / np.sum(
                    weights, axis=0, dtype=np.float64
                )
                retained_weight = float(np.sum(weights[:, 0], dtype=np.float64))
            method_index = {name: index for index, name in enumerate(METHODS)}
            for method, baseline in COMPARISONS:
                left = method_index[method]
                right = method_index[baseline]
                effects = 1.0 - draws[:, left] / draws[:, right]
                point = 1.0 - points[left] / points[right]
                records.append(
                    {
                        "score_contract": SCORING_CONTRACT_VERSION,
                        "metric": metric,
                        "metric_contract": metric_contract,
                        "lead_scope": lead_scope,
                        "lead_week": 0 if lead_scope == "W1-W6" else int(leads[0]),
                        "method": method,
                        "baseline": baseline,
                        "method_score": float(points[left]),
                        "baseline_score": float(points[right]),
                        "score_reduction_fraction": float(point),
                        "ci_lower_95": float(np.quantile(effects, 0.025)),
                        "ci_upper_95": float(np.quantile(effects, 0.975)),
                        "bootstrap_probability_improvement": float(np.mean(effects > 0.0)),
                        "threshold_retained_scoring_weight": retained_weight,
                        "bootstrap_samples": len(plan.draws),
                        "block_length_initializations": plan.block_length,
                        "bootstrap_seed": plan.seed,
                        "bootstrap_scheme": BOOTSTRAP_SCHEME,
                        "source_year_clusters": len(plan.source_years),
                        "small_year_cluster_limitation": True,
                    }
                )
    return pd.DataFrame.from_records(records)


def _read_verified_csv(
    root: Path, manifest: Mapping[str, Any], relative: str
) -> tuple[pd.DataFrame, str]:
    path, digest = _artifact(root, manifest, relative)
    frame = pd.read_csv(path)
    _require(not frame.empty, f"verified CSV is empty: {relative}")
    _require(
        not any(str(column).startswith("Unnamed:") for column in frame.columns),
        f"verified CSV has an unnamed column: {relative}",
    )
    return frame, digest


def _smoke_indices(initializations: np.ndarray) -> np.ndarray:
    starts = np.asarray(initializations, dtype="datetime64[D]")
    years = pd.DatetimeIndex(starts).year.to_numpy()
    selected: list[np.ndarray] = []
    for year in operational.AUDIT_YEARS:
        group = np.flatnonzero(years == year)
        _require(group.size == EXPECTED_YEAR_COUNTS[year], f"{year} full case count differs")
        local = operational._select_smoke_indices(
            starts[group],
            np.arange(group.size, dtype=np.int64),
            year,
            SMOKE_CASES_PER_YEAR,
        )
        selected.append(group[local])
    result = np.concatenate(selected).astype(np.int64)
    _require(result.size == 12 and np.unique(result).size == 12, "smoke selection differs")
    return result


def validate_operational_reconstruction(
    root: Path,
    manifest: Mapping[str, Any],
    operational_members: operational.OperationalMembers,
    daily: operational.DailyObservations,
    targets: operational.AuditTargets,
) -> dict[str, Any]:
    expected_dates = np.asarray(
        manifest.get("cases", {}).get("evaluated_initializations", ()),
        dtype="datetime64[D]",
    )
    _require(
        expected_dates.shape == (EXPECTED_CASES,)
        and np.array_equal(expected_dates, operational_members.initializations),
        "reloaded operational initialization identities differ",
    )
    expected_content_path, expected_content_sha = _artifact(
        root, manifest, "evaluation/input_content_receipts.json"
    )
    expected_content = _read_json(expected_content_path)
    actual_content = _json_safe(
        {
            "forecast_inputs": dict(operational_members.content_inventory),
            "observation_inputs": dict(daily.content_inventory),
            "derived_inputs": dict(targets.content_inventory),
        }
    )
    _require(actual_content == expected_content, "reloaded operational content receipts differ")

    support_path, support_sha = _artifact(
        root, manifest, "evaluation/scoring_support.npz"
    )
    denominator = targets.weights.sum(axis=(-2, -1), keepdims=True, dtype=np.float64)
    expected_normalized = targets.weights / denominator
    with np.load(support_path, allow_pickle=False) as archive:
        checks = {
            "latitude": np.array_equal(archive["latitude"], operational_members.latitude),
            "longitude": np.array_equal(archive["longitude"], operational_members.longitude),
            "frozen_observation_fraction": np.array_equal(
                archive["frozen_observation_fraction"], daily.observation_fraction
            ),
            "frozen_support_mask": np.array_equal(
                archive["frozen_support_mask"], targets.frozen_spatial_weights > 0.0
            ),
            "frozen_scoring_weight_km2_fraction": np.array_equal(
                archive["frozen_scoring_weight_km2_fraction"], targets.frozen_spatial_weights
            ),
            "dynamic_support_mask": np.array_equal(
                archive["dynamic_support_mask"], targets.weights > 0.0
            ),
            "dynamic_scoring_weight_km2_fraction": np.array_equal(
                archive["dynamic_scoring_weight_km2_fraction"], targets.weights
            ),
            "normalized_scoring_weight": np.allclose(
                archive["normalized_scoring_weight"],
                expected_normalized,
                rtol=0.0,
                atol=1.0e-15,
            ),
        }
    _require(all(checks.values()), f"reloaded operational support differs: {checks}")
    return {
        "input_content_receipt_path": str(expected_content_path),
        "input_content_receipt_sha256": expected_content_sha,
        "scoring_support_path": str(support_path),
        "scoring_support_sha256": support_sha,
        "exact_initialization_identity": True,
        "exact_input_content_identity": True,
        "exact_scoring_support_identity": True,
        "array_identity_checks": checks,
    }


def load_operational_adjustment(
    root: Path,
    manifest: Mapping[str, Any],
    seed: int,
    initializations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    relative = f"models/base_42k/seed_{seed}/audit_adjustments.npz"
    path, digest = _artifact(root, manifest, relative)
    checkpoint_records = {
        int(record["seed"]): record for record in manifest.get("input_checkpoints", ())
    }
    _require(seed in checkpoint_records, f"operational checkpoint receipt lacks seed {seed}")
    checkpoint = checkpoint_records[seed]
    with np.load(path, allow_pickle=False) as archive:
        dates = np.asarray(archive["initializations"], dtype="datetime64[D]")
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        log_spread = np.asarray(archive["log_spread"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        stored_seed = int(np.asarray(archive["seed"]).item())
        member_count = int(np.asarray(archive["source_member_count"]).item())
        checkpoint_hash = str(np.asarray(archive["source_checkpoint_sha256"]).item())
    expected_shape = (EXPECTED_CASES, LEAD_COUNT, *GRID_SHAPE)
    _require(
        np.array_equal(dates, initializations)
        and delta.shape == log_spread.shape == spread.shape == expected_shape
        and stored_seed == seed
        and member_count == FORECAST_MEMBERS,
        f"seed {seed} adjustment identity/shape differs",
    )
    expected_spread = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
    _require(
        np.isfinite(delta).all()
        and np.isfinite(log_spread).all()
        and np.isfinite(spread).all()
        and np.all(spread > 0.0)
        and np.array_equal(spread, expected_spread),
        f"seed {seed} adjustment values/log-spread identity differ",
    )
    _require(
        checkpoint_hash == checkpoint.get("sha256"),
        f"seed {seed} adjustment checkpoint binding differs",
    )
    return delta, spread, {
        "path": str(path),
        "sha256": digest,
        "seed": seed,
        "shape": list(delta.shape),
        "source_member_count": member_count,
        "source_checkpoint_sha256": checkpoint_hash,
        "initialization_first": np.datetime_as_string(dates[0], unit="D"),
        "initialization_last": np.datetime_as_string(dates[-1], unit="D"),
        "spread_is_exact_exp_clipped_log_spread_float32": True,
    }


def continuous_crps_identity(
    members: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    initializations: np.ndarray,
    expected: pd.DataFrame,
    *,
    label: str,
) -> dict[str, Any]:
    actual = operational._dynamic_weighted_field_mean(
        neural.numpy_ensemble_crps(members, truth), weights
    )
    expected_dates = [np.datetime_as_string(value, unit="D") for value in initializations]
    selected = expected.loc[expected.init.astype(str).isin(expected_dates)].copy()
    selected = selected.sort_values(["init", "lead_week"])
    _require(
        len(selected) == len(initializations) * LEAD_COUNT
        and selected.init.astype(str).drop_duplicates().tolist() == expected_dates,
        f"{label} continuous receipt rows differ",
    )
    stored = selected.crps.to_numpy(dtype=np.float64).reshape(actual.shape)
    difference = np.abs(actual - stored)
    maximum = float(np.max(difference))
    _require(maximum <= 2.0e-6, f"{label} continuous CRPS identity differs: {maximum}")
    support_cells = np.count_nonzero(weights > 0.0, axis=(-2, -1)).reshape(-1)
    stored_cells = pd.to_numeric(selected.support_cells).astype(int).to_numpy()
    stored_weights = selected.scoring_weight_sum_km2_fraction.to_numpy(dtype=np.float64)
    actual_weights = weights.sum(axis=(-2, -1), dtype=np.float64).reshape(-1)
    _require(
        np.array_equal(support_cells, stored_cells)
        and np.allclose(actual_weights, stored_weights, rtol=2.0e-10, atol=2.0e-10),
        f"{label} continuous support identity differs",
    )
    return {
        "maximum_absolute_crps_difference": maximum,
        "case_lead_rows": int(actual.size),
        "dynamic_support_cells_exact": True,
        "dynamic_weight_sum_csv_identity": True,
    }


def build_readme(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lag_provenance: Mapping[str, Any],
    *,
    smoke: bool,
) -> str:
    lines = [
        "# Operational-era categorical comparison",
        "",
        EVIDENCE_LABEL + ".",
        "",
        (
            "This is a non-scientific fixed 12-case plumbing smoke."
            if smoke
            else "This full result scores all 296 eligible 2022--2024 cases."
        ),
        "No model was trained, fine-tuned, selected, or refit. No 2025 store was opened.",
        "Neural headline rows average scores from seeds 42/43/44; no forecasts or parameters are averaged.",
        "The parent-bound 2002--2021 neural member cache was opened read-only and is directly receipted.",
        "",
        "## Pooled equality-aware RPS",
        "",
        "| Family | Method | RPS | RPSS vs training empirical climatology |",
        "|---|---|---:|---:|",
    ]
    for row in pooled.sort_values(["family", "rps"]).itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.method_label} | {row.rps:.6f} | "
            f"{100.0 * row.rpss_vs_training_empirical_climatology:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired uncertainty",
            "",
            f"The {'smoke' if smoke else 'full'} run uses "
            f"{SMOKE_BOOTSTRAP_DRAWS if smoke else BOOTSTRAP_DRAWS:,} paired "
            "two-stage resamples: years first, then circular 13-initialization blocks.",
            "Only three year clusters are available, so intervals are exploratory.",
            "",
            f"Lag coverage blocks affected by the 2023-10-12 anomaly: "
            f"{lag_provenance['affected_coverage_block_count']}.",
            "Lag weeks are observation-fraction weighted and require all fourteen pre-issue days.",
            "",
            f"Bootstrap rows: {len(bootstrap)}.",
            "",
            "See `manifest.json`, `receipts/input_receipts.json`, and the per-case CSV files for auditable details.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    started = time.monotonic()
    snapshot_hashes = source_snapshot(output)
    pbc_root, pbc_manifest, pbc_receipt = validate_pbc_input(Path(args.pbc_manifest))
    operational_root, operational_manifest, operational_receipt = (
        validate_operational_input()
    )
    source_bindings = validate_source_bindings(
        pbc_root, pbc_manifest, operational_root, operational_manifest
    )

    capacity_path = Path(
        operational_manifest.get("capacity_receipt", {}).get("path", "")
    ).resolve()
    capacity_receipt = operational.validate_locked_model_receipt(capacity_path)
    _require(
        capacity_receipt.manifest_sha256
        == operational_manifest.get("capacity_receipt", {}).get("sha256"),
        "operational capacity receipt changed",
    )
    canonical_cache = neural.load_member_cache(
        capacity_receipt.cache_path, allow_partial=False
    )
    member_cache_receipt = neural_member_cache_access_receipt(
        canonical_cache, capacity_receipt
    )
    splits = neural.make_split_indices(canonical_cache.initializations)
    _require(
        len(splits.train) == TRAIN_CASES
        and np.array_equal(splits.train, np.arange(TRAIN_CASES, dtype=np.int64)),
        "canonical training split differs",
    )

    print("Reloading and verifying all 296 operational members...", flush=True)
    full_members = operational.load_operational_members(canonical_cache, smoke=False)
    print("Reloading exact 2002-2017 and 2022-2024 IMD inputs...", flush=True)
    daily = operational.load_daily_observations(
        full_members.latitude, full_members.longitude
    )
    targets = operational.build_audit_targets(
        canonical_cache,
        full_members,
        daily,
        capacity_receipt,
        smoke=False,
    )
    operational_identity = validate_operational_reconstruction(
        operational_root, operational_manifest, full_members, daily, targets
    )

    pbc_latitude, pbc_longitude, pbc_support, pbc_weights, pbc_support_sha = (
        load_pbc_static_support(pbc_root, pbc_manifest)
    )
    _require(
        np.array_equal(pbc_latitude, full_members.latitude)
        and np.array_equal(pbc_longitude, full_members.longitude)
        and np.array_equal(pbc_support, targets.frozen_spatial_weights > 0.0)
        and np.array_equal(pbc_weights, targets.frozen_spatial_weights),
        "PBC and operational frozen support are not exactly identical",
    )
    frozen_pbc = load_frozen_pbc(
        pbc_root,
        pbc_manifest,
        pbc_support,
        canonical_cache.initializations[splits.train],
    )

    selected = (
        _smoke_indices(full_members.initializations)
        if args.smoke
        else np.arange(EXPECTED_CASES, dtype=np.int64)
    )
    initializations = full_members.initializations[selected]
    members = np.asarray(full_members.members[selected], dtype=np.float32)
    truth = np.asarray(targets.truth[selected], dtype=np.float32)
    weights = np.asarray(targets.weights[selected], dtype=np.float64)
    _require(
        members.shape
        == (len(selected), FORECAST_MEMBERS, LEAD_COUNT, *GRID_SHAPE)
        and truth.shape == weights.shape == (len(selected), LEAD_COUNT, *GRID_SHAPE),
        "selected operational scoring arrays differ",
    )
    dynamic_support = DynamicSupport(
        latitude=full_members.latitude.copy(),
        longitude=full_members.longitude.copy(),
        frozen_support=pbc_support.copy(),
        frozen_weights=pbc_weights.copy(),
        dynamic_weights=weights.copy(),
    )

    late_dates, late_values, late_fraction, late_receipt, late_store = (
        load_late_2021_lag_slice(
            initializations,
            full_members.latitude,
            full_members.longitude,
            daily.observation_fraction,
        )
    )
    lag_dates = np.concatenate((late_dates, daily.verification_dates))
    lag_values = np.concatenate((late_values, daily.verification_values))
    lag_fraction = np.concatenate((late_fraction, daily.verification_fraction))
    lags, lag_provenance = build_fraction_weighted_issue_lags(
        initializations,
        lag_dates,
        lag_values,
        lag_fraction,
        daily.observation_fraction,
        pbc_support,
        smoke=args.smoke,
    )
    lag_receipt = LagInputs(lags=lags, provenance=lag_provenance, opened_store=late_store)

    family_context: dict[
        str,
        tuple[
            FrozenFamily,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            FamilyReference,
            np.ndarray,
        ],
    ] = {}
    for family in (frozen_pbc.quintile, frozen_pbc.semidecile):
        thresholds, observed, empirical, reference = build_family_reference(
            family, initializations, truth, weights
        )
        lag_cdf = pbc.lag_observation_cdf(lags, family.model)
        _require(
            lag_cdf.shape
            == (len(initializations), len(LAG_WEEKS), family.model.levels.size, *GRID_SHAPE)
            and np.isfinite(lag_cdf[..., pbc_support]).all(),
            f"{family.name} lag CDF shape/support differs",
        )
        family_context[family.name] = (
            family,
            thresholds,
            observed,
            empirical,
            reference,
            lag_cdf,
        )

    operational_case_metrics, operational_case_sha = _read_verified_csv(
        operational_root, operational_manifest, "metrics/case_metrics.csv"
    )
    operational_seed_metrics, operational_seed_sha = _read_verified_csv(
        operational_root, operational_manifest, "metrics/seed_case_metrics.csv"
    )
    raw_expected = operational_case_metrics.loc[
        operational_case_metrics.method.eq("raw_fuxi")
    ]
    reconstruction: dict[str, Any] = {
        "raw_continuous_crps_identity": continuous_crps_identity(
            members,
            truth,
            weights,
            initializations,
            raw_expected,
            label="raw FuXi",
        ),
        "forecast_probability_arrays_written": False,
        "forecast_member_arrays_written": False,
        "parameters_or_adjustments_written": False,
        "neural_forecasts_or_probabilities_averaged": False,
    }

    non_neural_rps: dict[str, list[pd.DataFrame]] = {name: [] for name in FAMILIES}
    non_neural_upper: list[pd.DataFrame] = []
    non_neural_bias: list[pd.DataFrame] = []
    print("Scoring raw FuXi and applying frozen PBC components...", flush=True)
    for family_name in FAMILIES:
        family, thresholds, observed, empirical, reference, lag_cdf = family_context[
            family_name
        ]
        raw_cdf, raw_score = score_ensemble(
            members,
            family_name,
            thresholds,
            observed,
            weights,
            chunk_size=args.cdf_chunk_size,
        )
        non_neural_rps[family_name].append(
            rps_case_frame(
                family_name,
                "raw_fuxi_categorical",
                "not_applicable",
                raw_score,
                reference,
                initializations,
                thresholds,
                weights,
            )
        )
        non_neural_bias.append(
            probability_bias_frame(
                family_name,
                "raw_fuxi_categorical",
                "not_applicable",
                raw_score.probability_bias,
                initializations,
            )
        )
        if family_name == "semidecile":
            non_neural_upper.append(
                upper_case_frame(
                    "raw_fuxi_categorical",
                    "not_applicable",
                    raw_score,
                    reference,
                    initializations,
                    thresholds,
                    weights,
                )
            )
        pbc_cdfs = apply_frozen_pbc(
            raw_cdf,
            family,
            initializations,
            thresholds,
            empirical,
            lag_cdf,
        )
        del raw_cdf, raw_score
        for method in PBC_METHODS:
            method_score = score_cdf(
                family_name,
                pbc_cdfs[method],
                observed,
                thresholds,
                weights,
            )
            non_neural_rps[family_name].append(
                rps_case_frame(
                    family_name,
                    method,
                    "not_applicable",
                    method_score,
                    reference,
                    initializations,
                    thresholds,
                    weights,
                )
            )
            non_neural_bias.append(
                probability_bias_frame(
                    family_name,
                    method,
                    "not_applicable",
                    method_score.probability_bias,
                    initializations,
                )
            )
            if family_name == "semidecile":
                non_neural_upper.append(
                    upper_case_frame(
                        method,
                        "not_applicable",
                        method_score,
                        reference,
                        initializations,
                        thresholds,
                        weights,
                    )
                )
            del method_score
        del pbc_cdfs
        gc.collect()

    prediction_receipt_path, prediction_receipt_sha = _artifact(
        operational_root,
        operational_manifest,
        "evaluation/output_prediction_receipts.json",
    )
    expected_prediction_receipts = _read_json(prediction_receipt_path)
    neural_seed_rps: dict[str, list[pd.DataFrame]] = {name: [] for name in FAMILIES}
    neural_seed_upper: list[pd.DataFrame] = []
    neural_seed_bias: list[pd.DataFrame] = []
    adjustment_receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"Reconstructing and scoring locked neural seed {seed}...", flush=True)
        delta, spread, adjustment_receipt = load_operational_adjustment(
            operational_root,
            operational_manifest,
            seed,
            full_members.initializations,
        )
        corrected_full = neural.apply_affine_log_calibration(
            full_members.members, delta, spread
        )
        _require(
            corrected_full.shape == full_members.members.shape
            and np.isfinite(corrected_full).all()
            and np.all(corrected_full >= 0.0),
            f"seed {seed} reconstructed members are invalid",
        )
        stored_prediction = expected_prediction_receipts.get(f"seed_{seed}", {})
        actual_prediction = {
            "delta_log_location": operational.array_receipt(
                delta.astype("<f4", copy=False)
            ),
            "log_spread": operational.array_receipt(
                np.log(spread).astype("<f4", copy=False)
            ),
            "corrected_members": operational.array_receipt(
                corrected_full.astype("<f4", copy=False)
            ),
        }
        # The stored log-spread is checked exactly in the adjustment artifact;
        # recomputing log(spread) can round differently, so bind its receipt
        # directly from that artifact below before comparing.
        adjustment_path = Path(adjustment_receipt["path"])
        with np.load(adjustment_path, allow_pickle=False) as archive:
            stored_log_spread = np.asarray(archive["log_spread"], dtype=np.float32)
        actual_prediction["log_spread"] = operational.array_receipt(
            stored_log_spread.astype("<f4", copy=False)
        )
        _require(
            actual_prediction == stored_prediction,
            f"seed {seed} reconstructed prediction receipts differ",
        )
        corrected = np.asarray(corrected_full[selected], dtype=np.float32)
        expected_seed = operational_seed_metrics.loc[
            pd.to_numeric(operational_seed_metrics.seed).astype(int).eq(seed)
        ]
        adjustment_receipt["continuous_crps_identity"] = continuous_crps_identity(
            corrected,
            truth,
            weights,
            initializations,
            expected_seed,
            label=f"neural seed {seed}",
        )
        adjustment_receipt["corrected_member_receipt_matches_operational_full"] = True
        adjustment_receipts.append(adjustment_receipt)
        del corrected_full, delta, spread, stored_log_spread
        for family_name in FAMILIES:
            _, thresholds, observed, _, reference, _ = family_context[family_name]
            cdf, score = score_ensemble(
                corrected,
                family_name,
                thresholds,
                observed,
                weights,
                chunk_size=args.cdf_chunk_size,
            )
            neural_seed_rps[family_name].append(
                rps_case_frame(
                    family_name,
                    "base_42k",
                    seed,
                    score,
                    reference,
                    initializations,
                    thresholds,
                    weights,
                )
            )
            neural_seed_bias.append(
                probability_bias_frame(
                    family_name,
                    "base_42k",
                    seed,
                    score.probability_bias,
                    initializations,
                )
            )
            if family_name == "semidecile":
                neural_seed_upper.append(
                    upper_case_frame(
                        "base_42k",
                        seed,
                        score,
                        reference,
                        initializations,
                        thresholds,
                        weights,
                    )
                )
            del cdf, score
        del corrected
        gc.collect()

    seed_quintile = pd.concat(neural_seed_rps["quintile"], ignore_index=True)
    seed_semidecile = pd.concat(neural_seed_rps["semidecile"], ignore_index=True)
    seed_upper = pd.concat(neural_seed_upper, ignore_index=True)
    seed_bias = pd.concat(neural_seed_bias, ignore_index=True)
    neural_quintile = average_neural_rps_scores(seed_quintile, quintile=True)
    neural_semidecile = average_neural_rps_scores(seed_semidecile, quintile=False)
    neural_upper = average_neural_upper_scores(seed_upper)
    neural_bias = average_neural_probability_bias(seed_bias)

    quintile_case = pd.concat(
        (*non_neural_rps["quintile"], neural_quintile),
        ignore_index=True,
        sort=False,
    )
    semidecile_case = pd.concat(
        (*non_neural_rps["semidecile"], neural_semidecile),
        ignore_index=True,
        sort=False,
    )
    upper_case = pd.concat(
        (*non_neural_upper, neural_upper), ignore_index=True, sort=False
    )
    probability_bias = pd.concat(
        (*non_neural_bias, neural_bias), ignore_index=True, sort=False
    )
    for label, frame in (
        ("quintile", quintile_case),
        ("semidecile", semidecile_case),
        ("upper", upper_case),
    ):
        _require(
            set(frame.method) == set(METHODS)
            and not frame.duplicated(["method", "initialization", "lead_week"]).any(),
            f"merged {label} case scores are incomplete",
        )

    weekwise = pd.concat(
        (
            aggregate_rps_scores(quintile_case, ("lead_week",), quintile=True),
            aggregate_rps_scores(semidecile_case, ("lead_week",), quintile=False),
        ),
        ignore_index=True,
        sort=False,
    )
    seasonal = pd.concat(
        (
            aggregate_rps_scores(
                quintile_case, ("season", "lead_week"), quintile=True
            ),
            aggregate_rps_scores(
                semidecile_case, ("season", "lead_week"), quintile=False
            ),
        ),
        ignore_index=True,
        sort=False,
    )
    pooled = pd.concat(
        (
            aggregate_rps_scores(quintile_case, (), quintile=True),
            aggregate_rps_scores(semidecile_case, (), quintile=False),
        ),
        ignore_index=True,
        sort=False,
    )
    upper_summary = aggregate_upper_scores(upper_case)
    bootstrap_draws = SMOKE_BOOTSTRAP_DRAWS if args.smoke else BOOTSTRAP_DRAWS
    bootstrap_plan = operational.two_stage_year_block_indices(
        initializations,
        n_resamples=bootstrap_draws,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )
    bootstrap = paired_two_stage_bootstrap(
        quintile_case,
        semidecile_case,
        upper_case,
        initializations,
        bootstrap_plan,
    )
    bootstrap_design = operational.bootstrap_diagnostics(bootstrap_plan)

    metrics = output / "metrics"
    evaluation = output / "evaluation"
    receipts = output / "receipts"
    for directory in (metrics, evaluation, receipts):
        directory.mkdir(parents=True, exist_ok=True)
    seed_quintile.to_csv(metrics / "neural_seed_quintile_case_scores.csv", index=False)
    seed_semidecile.to_csv(metrics / "neural_seed_semidecile_case_scores.csv", index=False)
    seed_upper.to_csv(metrics / "neural_seed_upper_case_scores.csv", index=False)
    seed_bias.to_csv(metrics / "neural_seed_probability_bias.csv", index=False)
    quintile_case.to_csv(metrics / "quintile_case_scores.csv", index=False)
    semidecile_case.to_csv(metrics / "semidecile_case_scores.csv", index=False)
    upper_case.to_csv(metrics / "upper_q95_case_scores.csv", index=False)
    probability_bias.to_csv(metrics / "probability_bias.csv", index=False)
    weekwise.to_csv(metrics / "weekwise_rps.csv", index=False)
    seasonal.to_csv(metrics / "seasonal_weekwise_rps.csv", index=False)
    pooled.to_csv(metrics / "pooled_rps.csv", index=False)
    upper_summary.to_csv(metrics / "upper_q95_summary.csv", index=False)
    bootstrap.to_csv(metrics / "paired_two_stage_bootstrap.csv", index=False)
    np.savez_compressed(
        evaluation / "scoring_support.npz",
        latitude=dynamic_support.latitude.astype("<f8", copy=False),
        longitude=dynamic_support.longitude.astype("<f8", copy=False),
        frozen_support_mask=dynamic_support.frozen_support,
        frozen_scoring_weight=dynamic_support.frozen_weights.astype("<f8", copy=False),
        dynamic_scoring_weight=dynamic_support.dynamic_weights.astype("<f8", copy=False),
        dynamic_support_mask=dynamic_support.dynamic_weights > 0.0,
        truth=truth.astype("<f4", copy=False),
        initializations=initializations.astype("<i8"),
    )
    np.savez_compressed(
        evaluation / "persistence_lags.npz",
        initializations=initializations.astype("<i8"),
        lag_values=lags.values.astype("<f4", copy=False),
        source_indices=lags.source_indices.astype("<i8", copy=False),
        window_start=lags.window_start.astype("<i8", copy=False),
        window_end=lags.window_end.astype("<i8", copy=False),
    )
    write_json(evaluation / "persistence_lag_provenance.json", lag_provenance)
    write_json(evaluation / "late_2021_lag_only_receipt.json", late_receipt)
    write_json(evaluation / "bootstrap_design.json", bootstrap_design)

    reconstruction["all_three_seed_continuous_receipts_passed"] = True
    reconstruction["all_cdfs_equality_aware_and_positive_cut_scored"] = True
    input_receipts = {
        "pbc": pbc_receipt,
        "operational": operational_receipt,
        "source_bindings": source_bindings,
        "operational_reconstruction": operational_identity,
        "pbc_scoring_support_sha256": pbc_support_sha,
        "operational_case_metrics_sha256": operational_case_sha,
        "operational_seed_case_metrics_sha256": operational_seed_sha,
        "operational_output_prediction_receipts": {
            "path": str(prediction_receipt_path),
            "sha256": prediction_receipt_sha,
        },
        "pbc_fit": {
            "path": str(frozen_pbc.fit_path),
            "sha256": frozen_pbc.fit_sha256,
        },
        "neural_member_cache": member_cache_receipt,
        "late_2021_lag_only": late_receipt,
        "persistence_lags": lag_provenance,
        "neural_adjustments": adjustment_receipts,
        "reconstruction": reconstruction,
    }
    write_json(receipts / "input_receipts.json", input_receipts)
    (output / "README.md").write_text(
        build_readme(
            pooled,
            bootstrap,
            lag_receipt.provenance,
            smoke=args.smoke,
        ),
        encoding="utf-8",
    )

    opened_store_paths = [
        *full_members.store_paths,
        *daily.training_stores,
        *daily.verification_stores,
        late_store,
        str(neural.SPATIAL_STORE),
    ]
    opened_file_paths = [
        member_cache_receipt["data_file"],
        member_cache_receipt["metadata_file"],
        member_cache_receipt["manifest_file"],
        member_cache_receipt["checksums_file"],
    ]
    opened_input_paths = [*opened_store_paths, *opened_file_paths]
    for raw_path in opened_input_paths:
        _reject_forbidden_year_store(Path(raw_path))
    mode = "smoke" if args.smoke else "full"
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "status": "complete",
        "mode": mode,
        "smoke": bool(args.smoke),
        "canonical": not args.smoke,
        "scientific_eligible": not args.smoke,
        "scientific_status": (
            "non-scientific fixed year-balanced plumbing smoke"
            if args.smoke
            else EVIDENCE_LABEL
        ),
        "evidence_label": EVIDENCE_LABEL,
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started),
        "output_path": str(Path(args.publication_output).resolve()),
        "publication": {
            "requested_output_path": str(Path(args.publication_output).resolve()),
            "driver_output_path": str(Path(args.output).resolve()),
            "launcher_semantic_and_receipt_gate_required_before_publication": True,
        },
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "workflow_can_train": False,
            "workflow_can_finetune": False,
            "workflow_can_select": False,
            "workflow_can_fit_pbc": False,
            "workflow_can_select_debias_span": False,
            "workflow_can_select_blend_weight": False,
            "fixed_frozen_half_weight_combination_applied": True,
            "audit_metrics_used_for_selection": False,
            "forecast_year_store_whitelist": [2022, 2023, 2024],
            "verification_year_store_whitelist": [2022, 2023, 2024],
            "lag_only_observation_year_whitelist": [2021],
            "lag_only_2021_access": (
                "only exact late-December daily positions required by earliest 2022 issue lags"
            ),
            "neural_member_cache_years": list(MEMBER_CACHE_YEARS),
            "neural_member_cache_parent_bound": True,
            "neural_member_cache_read_only": True,
            "year_store_discovery_or_globbing_used": False,
            "final_2025_store_opened": False,
            "sealed_2025_target_opened": False,
            "maximum_verification_date": np.datetime_as_string(
                np.max(initializations + np.timedelta64(41, "D")), unit="D"
            ),
            "region": "39N-0N, 60E-99E, 27x27 India box",
            "target": "IMD weekly mean precipitation, mm day-1",
            "dynamic_weight": "India area x seven-day mean observation fraction",
            "cdf_definition": "strict empirical P(Y < persisted threshold)",
            "threshold_geometry": (
                "exact ties share one probability; q=0 CDF forced to zero and excluded"
            ),
            "persistence_lags": (
                "observation-fraction weighted issue-7..issue-1 and issue-14..issue-8; all days required"
            ),
        },
        "methods": list(METHODS),
        "pbc_methods": list(PBC_METHODS),
        "seeds": list(SEEDS),
        "seed_handling": {
            "parameters_averaged": False,
            "adjustment_fields_averaged": False,
            "members_averaged_across_seeds": False,
            "probabilities_averaged_across_seeds": False,
            "headline_aggregation": "arithmetic mean of per-seed scores on identical case/lead rows",
            "per_seed_scores_retained": True,
        },
        "cases": {
            "full_parent_case_count": EXPECTED_CASES,
            "evaluated_total": len(initializations),
            "evaluated_counts": {
                str(year): int(
                    np.count_nonzero(pd.DatetimeIndex(initializations).year == year)
                )
                for year in operational.AUDIT_YEARS
            },
            "evaluated_initializations": initializations.astype(str).tolist(),
            "all_eligible_cases_retained_in_full": not args.smoke,
        },
        "frozen_pbc": {
            "manifest_sha256": PBC_MANIFEST_SHA256,
            "receipt_sha256": PBC_RECEIPT_SHA256,
            "fit_sha256": PBC_FIT_SHA256,
            "calendar_families": {"quintile": 4, "semidecile": 19},
            "debias_candidate_spans_days": list(DEBIAS_SPANS),
            "selected_debias_span_by_lead": [14] * LEAD_COUNT,
            "persistence_ridge": PERSISTENCE_RIDGE,
            "persistence_feature_names": list(PERSISTENCE_FEATURE_NAMES),
            "persistence_usable_fit_indices": [4, TRAIN_CASES - 1],
            "persistence_usable_fit_count": PERSISTENCE_USABLE_CASES,
            "fixed_debias_weight": BLEND_DEBIAS_WEIGHT,
            "parameters_refit": False,
        },
        "operational_parent": {
            "manifest_path": str(DEFAULT_OPERATIONAL_MANIFEST.resolve()),
            "manifest_sha256": OPERATIONAL_MANIFEST_SHA256,
            "receipt_path": str(DEFAULT_OPERATIONAL_RECEIPT.resolve()),
            "receipt_sha256": OPERATIONAL_RECEIPT_SHA256,
            "audit_contract_version": operational.AUDIT_CONTRACT_VERSION,
            "post_run_audit_version": operational.POSTRUN_AUDIT_VERSION,
        },
        "evaluation": {
            "primary_metrics": [
                "K=4 normalized distinct-positive-cut RPS",
                "K=19 normalized distinct-positive-cut RPS",
            ],
            "secondary_metrics": [
                "summed four-slot q80-positive formula RPS",
                "q95-positive upper-tail Brier score",
                "CDF probability bias by cut",
            ],
            "common_dynamic_denominator_across_methods": True,
            "q80_mask_is_threshold_only": True,
            "q95_mask_is_threshold_only": True,
            "bootstrap": bootstrap_design,
            "bootstrap_full_draws": BOOTSTRAP_DRAWS,
            "bootstrap_scheme": BOOTSTRAP_SCHEME,
            "small_year_cluster_limitation": True,
        },
        "persistence_lag_provenance": lag_provenance,
        "input_receipts": input_receipts,
        "opened_store_paths_exact": opened_store_paths,
        "opened_file_paths_exact": opened_file_paths,
        "opened_input_paths_exact": opened_input_paths,
        "source_snapshot_sha256": snapshot_hashes,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
        },
    }
    manifest["artifact_sha256"] = output_checksums(output)
    for relative, digest in snapshot_hashes.items():
        _require(
            manifest["artifact_sha256"].get(relative) == digest
            and sha256_file(output / relative) == digest,
            f"source snapshot is not artifact-gated: {relative}",
        )
    write_json(output / "manifest.json", manifest)
    return manifest


def _semantic_sort(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="stable").reset_index(drop=True)


def _frames_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
    atol: float = 2.0e-10,
) -> None:
    _require(set(actual.columns) == set(expected.columns), f"{label} columns differ")
    _require(len(actual) == len(expected), f"{label} row count differs")
    _require(not actual.duplicated(list(keys)).any(), f"{label} has duplicate keys")
    _require(not expected.duplicated(list(keys)).any(), f"regenerated {label} has duplicate keys")
    left = _semantic_sort(actual, keys)
    right = _semantic_sort(expected, keys)
    for column in sorted(left.columns):
        left_column = left[column]
        right_column = right[column]
        if pd.api.types.is_numeric_dtype(left_column) and pd.api.types.is_numeric_dtype(
            right_column
        ):
            _require(
                np.allclose(
                    left_column.to_numpy(dtype=np.float64),
                    right_column.to_numpy(dtype=np.float64),
                    rtol=2.0e-10,
                    atol=atol,
                    equal_nan=True,
                ),
                f"{label} numeric column differs: {column}",
            )
        else:
            _require(
                np.array_equal(
                    left_column.fillna("<NA>").astype(str).to_numpy(),
                    right_column.fillna("<NA>").astype(str).to_numpy(),
                ),
                f"{label} categorical column differs: {column}",
            )


def _output_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    _require(path.is_file(), f"semantic artifact is missing: {relative}")
    frame = pd.read_csv(path)
    _require(not frame.empty, f"semantic artifact is empty: {relative}")
    return frame


def validate_output_semantics(
    root: Path,
    mode: str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild score-table relationships before the Slurm receipt is issued."""

    root = Path(root).resolve()
    if manifest is None:
        manifest = _read_json(root / "manifest.json")
    smoke = mode == "smoke"
    expected_cases = 12 if smoke else EXPECTED_CASES
    _require(mode in {"smoke", "full"}, "semantic mode differs")
    _require(
        manifest.get("experiment") == EXPERIMENT
        and manifest.get("audit_contract_version") == AUDIT_CONTRACT_VERSION
        and manifest.get("scoring_contract_version") == SCORING_CONTRACT_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("mode") == mode
        and manifest.get("smoke") is smoke
        and manifest.get("canonical") is (not smoke)
        and manifest.get("scientific_eligible") is (not smoke),
        "semantic manifest identity differs",
    )
    contract = manifest.get("contract", {})
    for key in (
        "workflow_can_train",
        "workflow_can_finetune",
        "workflow_can_select",
        "workflow_can_fit_pbc",
        "workflow_can_select_debias_span",
        "workflow_can_select_blend_weight",
        "audit_metrics_used_for_selection",
        "final_2025_store_opened",
        "sealed_2025_target_opened",
    ):
        _require(contract.get(key) is False, f"semantic firewall differs: {key}")
    _require(
        contract.get("fixed_frozen_half_weight_combination_applied") is True
        and contract.get("forecast_year_store_whitelist") == [2022, 2023, 2024]
        and contract.get("verification_year_store_whitelist") == [2022, 2023, 2024]
        and contract.get("lag_only_observation_year_whitelist") == [2021],
        "semantic frozen application/year whitelist differs",
    )
    opened_store_paths = manifest.get("opened_store_paths_exact", [])
    opened_file_paths = manifest.get("opened_file_paths_exact", [])
    opened_input_paths = manifest.get("opened_input_paths_exact", [])
    _require(
        isinstance(opened_store_paths, list)
        and isinstance(opened_file_paths, list)
        and opened_input_paths == [*opened_store_paths, *opened_file_paths],
        "semantic opened-input inventory differs",
    )
    for raw_path in opened_input_paths:
        _reject_forbidden_year_store(Path(raw_path))
    cache_receipt = manifest.get("input_receipts", {}).get(
        "neural_member_cache", {}
    )
    _require(
        cache_receipt.get("years") == list(MEMBER_CACHE_YEARS)
        and cache_receipt.get("initialization_count")
        == EXPECTED_MEMBER_CACHE_CASES
        and cache_receipt.get("data_sha256") == MEMBER_CACHE_DATA_SHA256
        and cache_receipt.get("checksums_sha256")
        == MEMBER_CACHE_CHECKSUMS_SHA256
        and cache_receipt.get("whole_npy_integrity_scanned") is True
        and cache_receipt.get("hindcast_member_values_scored_in_this_comparison")
        is False
        and cache_receipt.get("parent_bound") is True
        and cache_receipt.get("read_only") is True
        and opened_file_paths
        == [
            cache_receipt.get("data_file"),
            cache_receipt.get("metadata_file"),
            cache_receipt.get("manifest_file"),
            cache_receipt.get("checksums_file"),
        ],
        "semantic neural member-cache receipt differs",
    )
    _require(
        manifest.get("methods") == list(METHODS)
        and manifest.get("pbc_methods") == list(PBC_METHODS)
        and manifest.get("seeds") == list(SEEDS),
        "semantic method/seed identities differ",
    )
    cases = manifest.get("cases", {})
    initializations = np.asarray(
        cases.get("evaluated_initializations", ()), dtype="datetime64[D]"
    )
    _require(
        len(initializations) == expected_cases
        and np.unique(initializations).size == expected_cases
        and np.all(np.diff(initializations) > np.timedelta64(0, "D"))
        and np.max(initializations + np.timedelta64(41, "D"))
        <= np.datetime64("2024-12-31"),
        "semantic initialization contract differs",
    )
    expected_year_counts = (
        {"2022": 4, "2023": 4, "2024": 4}
        if smoke
        else {str(key): value for key, value in EXPECTED_YEAR_COUNTS.items()}
    )
    _require(
        cases.get("evaluated_total") == expected_cases
        and cases.get("evaluated_counts") == expected_year_counts,
        "semantic year counts differ",
    )

    quintile = _output_csv(root, "metrics/quintile_case_scores.csv")
    semidecile = _output_csv(root, "metrics/semidecile_case_scores.csv")
    upper = _output_csv(root, "metrics/upper_q95_case_scores.csv")
    seed_quintile = _output_csv(
        root, "metrics/neural_seed_quintile_case_scores.csv"
    )
    seed_semidecile = _output_csv(
        root, "metrics/neural_seed_semidecile_case_scores.csv"
    )
    seed_upper = _output_csv(root, "metrics/neural_seed_upper_case_scores.csv")
    seed_bias = _output_csv(root, "metrics/neural_seed_probability_bias.csv")
    bias = _output_csv(root, "metrics/probability_bias.csv")

    expected_case_rows = expected_cases * LEAD_COUNT * len(METHODS)
    expected_seed_rows = expected_cases * LEAD_COUNT * len(SEEDS)
    for family, frame in (("quintile", quintile), ("semidecile", semidecile)):
        _require(
            len(frame) == expected_case_rows
            and set(frame.family) == {family}
            and set(frame.method) == set(METHODS)
            and set(frame.score_contract) == {SCORING_CONTRACT_VERSION}
            and set(frame.nominal_reference_contract)
            == {NOMINAL_REFERENCE_CONTRACT}
            and not frame.duplicated(
                ["method", "initialization", "lead_week"]
            ).any(),
            f"semantic {family} case geometry differs",
        )
        _require(
            np.isfinite(
                frame[
                    [
                        "rps",
                        "nominal_climatology_rps",
                        "training_empirical_climatology_rps",
                        "dynamic_scoring_weight",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
            and frame.rps.between(0.0, 1.0).all(),
            f"semantic {family} score domains differ",
        )
        references = [
            "dynamic_scoring_weight",
            "nominal_climatology_rps",
            "training_empirical_climatology_rps",
        ]
        if family == "quintile":
            references.extend(
                [
                    "paper_fixed_nominal_climatology_rps",
                    "paper_q80_retained_scoring_weight",
                    "paper_q80_total_supported_scoring_weight",
                    "paper_q80_retained_scoring_weight_fraction",
                ]
            )
            _require(
                np.isfinite(
                    frame[
                        [
                            "paper_nominal_cut_q80_positive_rps",
                            *references[3:],
                        ]
                    ].to_numpy(dtype=np.float64)
                ).all(),
                "semantic q80 fields are nonfinite",
            )
        spread = frame.groupby(["initialization", "lead_week"])[references].agg(
            lambda values: float(values.max() - values.min())
        )
        _require(
            np.allclose(spread.to_numpy(), 0.0, rtol=0.0, atol=2.0e-10),
            f"semantic {family} references/support differ across methods",
        )
    _require(
        len(upper) == expected_case_rows
        and set(upper.method) == set(METHODS)
        and not upper.duplicated(["method", "initialization", "lead_week"]).any()
        and upper.brier_score.between(0.0, 1.0).all(),
        "semantic q95 case geometry/domains differ",
    )
    upper_spread = upper.groupby(["initialization", "lead_week"])[
        [
            "nominal_climatology_brier_score",
            "training_empirical_climatology_brier_score",
            "q95_retained_scoring_weight",
            "q95_total_supported_scoring_weight",
            "q95_retained_scoring_weight_fraction",
        ]
    ].agg(lambda values: float(values.max() - values.min()))
    _require(
        np.allclose(upper_spread.to_numpy(), 0.0, rtol=0.0, atol=2.0e-10),
        "semantic q95 references/support differ across methods",
    )
    _require(
        len(seed_quintile) == len(seed_semidecile) == len(seed_upper) == expected_seed_rows,
        "semantic per-seed case row counts differ",
    )

    expected_neural_quintile = average_neural_rps_scores(
        seed_quintile, quintile=True
    )
    expected_neural_semidecile = average_neural_rps_scores(
        seed_semidecile, quintile=False
    )
    expected_neural_upper = average_neural_upper_scores(seed_upper)
    _frames_equivalent(
        quintile.loc[quintile.method.eq("base_42k")],
        expected_neural_quintile,
        keys=("method", "initialization", "lead_week"),
        label="headline neural quintile score average",
    )
    _frames_equivalent(
        semidecile.loc[semidecile.method.eq("base_42k")],
        expected_neural_semidecile,
        keys=("method", "initialization", "lead_week"),
        label="headline neural semidecile score average",
    )
    _frames_equivalent(
        upper.loc[upper.method.eq("base_42k")],
        expected_neural_upper,
        keys=("method", "initialization", "lead_week"),
        label="headline neural q95 score average",
    )
    expected_neural_bias = average_neural_probability_bias(seed_bias)
    _frames_equivalent(
        bias.loc[bias.method.eq("base_42k")],
        expected_neural_bias,
        keys=(
            "family",
            "method",
            "lead_week",
            "season",
            "nominal_cumulative_probability",
        ),
        label="headline neural probability-bias score average",
    )
    _require(
        set(bias.family) == set(FAMILIES)
        and set(bias.method) == set(METHODS)
        and np.isfinite(
            bias[["probability_bias", "mean_absolute_case_probability_bias"]].to_numpy(
                dtype=np.float64
            )
        ).all()
        and np.all(np.abs(bias.probability_bias.to_numpy(dtype=np.float64)) <= 1.0)
        and bias.mean_absolute_case_probability_bias.between(0.0, 1.0).all(),
        "semantic probability-bias domains differ",
    )

    expected_weekwise = pd.concat(
        (
            aggregate_rps_scores(quintile, ("lead_week",), quintile=True),
            aggregate_rps_scores(semidecile, ("lead_week",), quintile=False),
        ),
        ignore_index=True,
        sort=False,
    )
    expected_seasonal = pd.concat(
        (
            aggregate_rps_scores(
                quintile, ("season", "lead_week"), quintile=True
            ),
            aggregate_rps_scores(
                semidecile, ("season", "lead_week"), quintile=False
            ),
        ),
        ignore_index=True,
        sort=False,
    )
    expected_pooled = pd.concat(
        (
            aggregate_rps_scores(quintile, (), quintile=True),
            aggregate_rps_scores(semidecile, (), quintile=False),
        ),
        ignore_index=True,
        sort=False,
    )
    expected_upper_summary = aggregate_upper_scores(upper)
    _frames_equivalent(
        _output_csv(root, "metrics/weekwise_rps.csv"),
        expected_weekwise,
        keys=("family", "method", "lead_week"),
        label="weekwise RPS aggregation",
    )
    _frames_equivalent(
        _output_csv(root, "metrics/seasonal_weekwise_rps.csv"),
        expected_seasonal,
        keys=("family", "method", "season", "lead_week"),
        label="seasonal RPS aggregation",
    )
    _frames_equivalent(
        _output_csv(root, "metrics/pooled_rps.csv"),
        expected_pooled,
        keys=("family", "method"),
        label="pooled RPS aggregation",
    )
    _frames_equivalent(
        _output_csv(root, "metrics/upper_q95_summary.csv"),
        expected_upper_summary,
        keys=("method", "lead_week", "season"),
        label="q95 summary aggregation",
    )

    expected_draws = SMOKE_BOOTSTRAP_DRAWS if smoke else BOOTSTRAP_DRAWS
    plan = operational.two_stage_year_block_indices(
        initializations,
        n_resamples=expected_draws,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )
    expected_bootstrap = paired_two_stage_bootstrap(
        quintile, semidecile, upper, initializations, plan
    )
    actual_bootstrap = _output_csv(
        root, "metrics/paired_two_stage_bootstrap.csv"
    )
    _require(
        len(actual_bootstrap) == 4 * 7 * len(COMPARISONS),
        "semantic bootstrap row count differs",
    )
    _frames_equivalent(
        actual_bootstrap,
        expected_bootstrap,
        keys=("metric", "lead_scope", "method", "baseline"),
        label="paired two-stage bootstrap",
        atol=3.0e-10,
    )
    stored_design = _read_json(root / "evaluation/bootstrap_design.json")
    _require(
        stored_design == _json_safe(operational.bootstrap_diagnostics(plan))
        and manifest.get("evaluation", {}).get("bootstrap") == stored_design,
        "semantic bootstrap design differs",
    )

    lag_provenance = _read_json(root / "evaluation/persistence_lag_provenance.json")
    _require(
        lag_provenance == manifest.get("persistence_lag_provenance"),
        "semantic lag provenance differs",
    )
    expected_affected = 0 if smoke else len(EXPECTED_FULL_LAG_COVERAGE_BLOCKS)
    _require(
        lag_provenance.get("all_lags_available") is True
        and lag_provenance.get("case_count") == expected_cases
        and lag_provenance.get("affected_coverage_block_count") == expected_affected,
        "semantic lag completeness/anomaly receipt differs",
    )
    with np.load(root / "evaluation/persistence_lags.npz", allow_pickle=False) as archive:
        _require(
            archive["lag_values"].shape
            == (expected_cases, len(LAG_WEEKS), *GRID_SHAPE)
            and np.isfinite(archive["lag_values"][:, :, pbc_support_mask(root)]).all()
            and np.all(archive["source_indices"] >= 0),
            "semantic lag artifact differs",
        )

    return {
        "experiment": EXPERIMENT,
        "status": "passed",
        "mode": mode,
        "post_run_audit_version": POSTRUN_AUDIT_VERSION,
        "case_count": expected_cases,
        "quintile_case_rows": len(quintile),
        "semidecile_case_rows": len(semidecile),
        "upper_case_rows": len(upper),
        "neural_seed_case_rows_per_family": len(seed_quintile),
        "bootstrap_rows_recomputed": len(actual_bootstrap),
        "bootstrap_draws_recomputed": expected_draws,
        "headline_neural_score_average_recomputed": True,
        "aggregate_tables_recomputed": True,
        "lag_receipts_rechecked": True,
        "neural_member_cache_receipt_rechecked": True,
        "no_2025_year_store_component": True,
    }


def pbc_support_mask(root: Path) -> np.ndarray:
    with np.load(root / "evaluation/scoring_support.npz", allow_pickle=False) as archive:
        return np.asarray(archive["frozen_support_mask"], dtype=bool)


def preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    pbc_root, pbc_manifest, pbc_receipt = validate_pbc_input(Path(args.pbc_manifest))
    operational_root, operational_manifest, operational_receipt = (
        validate_operational_input()
    )
    source_bindings = validate_source_bindings(
        pbc_root, pbc_manifest, operational_root, operational_manifest
    )
    return {
        "experiment": EXPERIMENT,
        "status": "preflight_passed",
        "mode": "smoke" if args.smoke else "full",
        "pbc": pbc_receipt,
        "operational": operational_receipt,
        "source_bindings": source_bindings,
        "forecast_year_stores": [2022, 2023, 2024],
        "verification_year_stores": [2022, 2023, 2024],
        "lag_only_year_store": 2021,
        "sealed_2025_target_opened": False,
        "workflow_can_train": False,
        "workflow_can_select": False,
    }


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen neural and PBC categorical calibration on a completed "
            "2022-2024 operational-era full audit."
        )
    )
    parser.add_argument("--pbc-manifest", type=Path, default=DEFAULT_PBC_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--publication-output", type=Path, default=None)
    parser.add_argument("--cdf-chunk-size", type=int, default=CDF_CHUNK_SIZE)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    _require(
        args.cdf_chunk_size == CDF_CHUNK_SIZE,
        f"--cdf-chunk-size must remain the frozen {CDF_CHUNK_SIZE}",
    )
    expected_draws = SMOKE_BOOTSTRAP_DRAWS if args.smoke else BOOTSTRAP_DRAWS
    if args.bootstrap_draws is None:
        args.bootstrap_draws = expected_draws
    _require(
        args.bootstrap_draws == expected_draws,
        f"--bootstrap-draws must be {expected_draws} in this mode",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.preflight_only:
        print(json.dumps(_json_safe(preflight(args)), indent=2, sort_keys=True))
        return 0
    _require(
        args.publication_output is not None,
        "--publication-output is required outside read-only preflight",
    )
    requested = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    publication_output = Path(args.publication_output).resolve()
    _require(
        requested != publication_output
        and requested.parent == publication_output.parent,
        "driver output must be a distinct hidden sibling of publication output",
    )
    args.output = requested
    args.publication_output = publication_output
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.exists():
        raise FileExistsError(f"refusing to overwrite output: {requested}")
    staging = requested.parent / f".{requested.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
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
                "traceback": traceback.format_exc(),
                "requested_output": str(requested),
                "sealed_2025_target_opened": False,
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} operational categorical comparison at {requested}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
