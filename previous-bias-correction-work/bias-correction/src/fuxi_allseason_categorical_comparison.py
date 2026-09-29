#!/usr/bin/env python3
"""Fair categorical comparison of frozen neural calibration and PBC.

This post-hoc evaluator never trains a model.  It reconstructs each frozen
neural test ensemble independently from its stored adjustment artifact,
scores that ensemble against the exact PBC thresholds/support/cases, then
discards it.  Across random seeds, only proper scores and probability-bias
statistics are averaged; forecasts and parameters are never averaged.

The evidence is retrospective 2020--2021 development evidence.  No 2025
forecast or observation is opened by this program.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT

import fuxi_allseason_ensemble_calibration as frozen
from fuxi_pbc_core import (
    CalendarQuantiles,
    calendar_fields,
    ensemble_cdf,
    is_valid_cdf_for_thresholds,
    observation_cdf,
    paper_fixed_nominal_reference_rps,
    paper_nominal_cut_rps,
    probability_bias,
    project_cdf_for_thresholds,
    ranked_probability_score,
    upper_tail_brier_score,
    verification_midpoints,
    weighted_spatial_mean,
)


EXPERIMENT = "fuxi_allseason_categorical_comparison_v2"
PBC_EXPERIMENT = "fuxi_allseason_pbc_baseline_v2"
NEURAL_EXPERIMENT = "fuxi_allseason_ensemble_calibration_v1"
SCORING_CONTRACT_VERSION = "normalized_informative_positive_cut_v2"
NOMINAL_REFERENCE_CONTRACT = "threshold_projected_zero_anchored_nominal_sensitivity"
PBC_POSTRUN_AUDIT_VERSION = "pbc_postrun_v2"
BOOTSTRAP_SCHEME = "two_stage_year_resample_then_within_year_circular_blocks"
UPPER_EVENT = "upper_5pct_informative_positive_q95"
DEFAULT_NEURAL_MANIFEST = (
    PROJECT_ROOT / "resultsv2/fuxi_allseason_ensemble_calibration/"
    "full_publication_20260822T115253Z/manifest.json"
)
FROZEN_NEURAL_MANIFEST_SHA256 = (
    "94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "resultsv2" / "fuxi_allseason_categorical_comparison_v2"
)
NEURAL_CONFIGURATIONS = ("summary_only", "location_spread")
SEEDS = (42, 43, 44)
PBC_METHODS = (
    "raw_fuxi_categorical",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
)
COMPARISON_METHODS = (
    "raw_fuxi_categorical",
    "moment_calibration",
    "summary_only",
    "location_spread",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
)
METHOD_LABELS = {
    "raw_fuxi_categorical": "Raw FuXi categorical",
    "moment_calibration": "Train-only moment calibration",
    "summary_only": "Summary-only neural calibration",
    "location_spread": "Set neural location + spread",
    "debias_plus_plus": "Projected Debias++",
    "persistence_plus_plus": "Projected Persistence++",
    "pbc_combined": "Combined PBC",
}
# Match the frozen PBC threshold artifact bit-for-bit: its probability levels
# are generated in float64 and only then persisted as float32.  Generating an
# arange directly in float32 accumulates one-ULP differences at several cuts,
# which makes otherwise identical categorical rows fail their receipt join.
QUINTILE_LEVELS = np.asarray(
    np.arange(0.2, 1.0, 0.2, dtype=np.float64), dtype=np.float32
)
SEMIDECILE_LEVELS = np.asarray(
    np.arange(0.05, 1.0, 0.05, dtype=np.float64), dtype=np.float32
)
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20260822


class ComparisonContractError(ValueError):
    """Raised when an input receipt or comparison invariant fails."""


@dataclass(frozen=True)
class SupportBundle:
    latitude: np.ndarray
    longitude: np.ndarray
    support: np.ndarray
    weights: np.ndarray
    normalized_weights: np.ndarray


@dataclass(frozen=True)
class ThresholdBundle:
    quintile: CalendarQuantiles
    semidecile: CalendarQuantiles
    artifact_path: Path
    artifact_sha256: str


@dataclass(frozen=True)
class ReferenceScores:
    projected_nominal_rps: np.ndarray
    training_empirical_rps: np.ndarray
    paper_fixed_nominal_rps: np.ndarray
    projected_nominal_upper_brier: np.ndarray
    training_empirical_upper_brier: np.ndarray


@dataclass(frozen=True)
class ScoreArrays:
    quintile_rps: np.ndarray
    paper_nominal_cut_rps: np.ndarray
    upper_brier: np.ndarray
    quintile_probability_bias: np.ndarray
    semidecile_probability_bias: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def array_bundle_provenance(
    arrays: Mapping[str, np.ndarray], *, bundle_name: str
) -> dict[str, Any]:
    """Content-bind named arrays exactly as the PBC V2 provenance contract."""

    if not bundle_name or not arrays:
        raise ComparisonContractError("array provenance needs a name and arrays")
    digest = hashlib.sha256()
    metadata: dict[str, Any] = {}
    for name in sorted(arrays):
        values = np.ascontiguousarray(np.asarray(arrays[name]))
        dtype = values.dtype.str
        shape = [int(value) for value in values.shape]
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(dtype.encode("ascii") + b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(values.view(np.uint8)))
        metadata[str(name)] = {"dtype": dtype, "shape": shape}
    return {
        "schema_version": "named_c_order_array_bundle_sha256_v1",
        "bundle_name": bundle_name,
        "sha256": digest.hexdigest(),
        "arrays": metadata,
    }


def _function_ast_bundle_sha256(path: Path, names: Sequence[str]) -> str:
    """Hash selected helper definitions independent of unrelated file drift."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(set(names) - set(definitions))
    if missing:
        raise ComparisonContractError(
            f"source {path} lacks required helpers: {', '.join(missing)}"
        )
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(
            ast.dump(
                definitions[name], annotate_fields=True, include_attributes=False
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_categorical_comparison.py": Path(__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "src/fuxi_allseason_pbc_baseline.py": PROJECT_ROOT
        / "src/fuxi_allseason_pbc_baseline.py",
        "plan/CATEGORICAL_COMPARISON_20260822.md": PROJECT_ROOT
        / "plan/CATEGORICAL_COMPARISON_20260822.md",
    }
    launcher = PROJECT_ROOT / "slurm/evaluate_allseason_categorical_comparison.sbatch"
    if launcher.is_file():
        sources["slurm/evaluate_allseason_categorical_comparison.sbatch"] = launcher
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = frozen.sha256_file(
            destination
        )
    return checksums


def _output_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative in {"manifest.json", "failure.json"}:
            continue
        checksums[relative] = frozen.sha256_file(path)
    return checksums


def _load_manifest(path: Path, *, experiment: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.name != "manifest.json":
        raise ComparisonContractError(f"manifest is unavailable: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("experiment") != experiment:
        raise ComparisonContractError(
            f"unexpected experiment in {resolved}: {payload.get('experiment')!r}"
        )
    if payload.get("status") != "complete" or payload.get("mode") != "full":
        raise ComparisonContractError(f"input is not a completed full run: {resolved}")
    if payload.get("smoke") is not False:
        raise ComparisonContractError(
            f"smoke input cannot enter comparison: {resolved}"
        )
    contract = payload.get("contract", {})
    if contract.get("sealed_2025_target_opened") is not False:
        raise ComparisonContractError(f"input does not prove sealed 2025: {resolved}")
    return resolved, payload, frozen.sha256_file(resolved)


def validate_pbc_full_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Require the audited, daily-lag, full PBC result—not a stale predecessor."""

    receipt_path = root / "slurm_gate_receipt.json"
    if not receipt_path.is_file():
        raise ComparisonContractError(
            "completed PBC full lacks its successful Slurm gate receipt"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        "experiment": PBC_EXPERIMENT,
        "gate_status": "passed",
        "mode": "full",
        "post_run_audit_version": PBC_POSTRUN_AUDIT_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "manifest_sha256": manifest_sha256,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ComparisonContractError(
                f"PBC Slurm receipt differs for {key}: {receipt.get(key)!r}"
            )

    canonical_counts = {
        "train": 1652,
        "validation": 196,
        "test": 208,
        "embargo": 24,
    }
    if manifest.get("split_counts_archive") != canonical_counts:
        raise ComparisonContractError("PBC full uses noncanonical archive splits")
    if manifest.get("split_counts_selected") != {
        "train": 1652,
        "validation": 196,
        "test": 208,
    }:
        raise ComparisonContractError("PBC full uses a reduced selected split")

    temporal = manifest.get("temporal_evidence", {})
    if temporal.get("persistence_lag_source") != (
        "complete daily IMD calendar; exact seven-day means"
    ):
        raise ComparisonContractError(
            "PBC full predates the complete-daily-IMD Persistence++ correction"
        )
    expected_usable = {
        "persistence_usable_training_cases": 1648,
        "persistence_usable_validation_cases": 196,
        "persistence_usable_development_cases": 208,
    }
    for key, expected in expected_usable.items():
        if temporal.get(key) != expected:
            raise ComparisonContractError(
                f"PBC full temporal evidence differs for {key}: {temporal.get(key)!r}"
            )
    if manifest.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ComparisonContractError("PBC full has the wrong scoring contract")
    evaluation = manifest.get("evaluation", {})
    if not str(evaluation.get("primary_metric", "")).startswith(
        "normalized informative-positive-cut quintile RPSS"
    ):
        raise ComparisonContractError("PBC full has the wrong primary metric")
    for gate in (
        evaluation.get("effective_cut_support", {})
        .get("quintile", {})
        .get("all_case_leads_have_positive_scoring_weight"),
        evaluation.get("paper_nominal_cut_support", {}).get(
            "all_case_leads_have_positive_q80_scoring_weight"
        ),
        evaluation.get("informative_upper_tail_support", {}).get(
            "all_case_leads_have_positive_q95_scoring_weight"
        ),
    ):
        if gate is not True:
            raise ComparisonContractError("PBC full fails an informative-support gate")

    validity = manifest.get("fitting", {}).get("cdf_validity", {})
    for family in ("quintile", "semidecile"):
        if set(validity.get(family, {})) != set(PBC_METHODS):
            raise ComparisonContractError(f"PBC {family} CDF validity is incomplete")
        for method, evidence in validity[family].items():
            for key in (
                "valid_probability_cdf",
                "valid_effective_threshold_cdf",
                "all_supported_values_finite",
                "all_supported_values_bounded_0_1",
                "all_supported_rows_nondecreasing",
                "all_identical_threshold_cdfs_exactly_equal",
                "all_zero_threshold_cdfs_exactly_zero",
                "all_unsupported_values_nan",
            ):
                if evidence.get(key) is not True:
                    raise ComparisonContractError(f"PBC {family}/{method} fails {key}")

    provenance_receipts: dict[str, Any] = {}
    for name, relative, receipt_key in (
        (
            "persistence_lag",
            "evaluation/persistence_lag_provenance.json",
            "persistence_lag_sha256",
        ),
        (
            "observation_bundle",
            "evaluation/observation_bundle_provenance.json",
            "observation_bundle_sha256",
        ),
    ):
        path, artifact_sha = _artifact(root, manifest, relative)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload != manifest.get(f"{name}_provenance"):
            raise ComparisonContractError(f"PBC {name} provenance artifact drift")
        if payload.get("schema_version") != (
            "named_c_order_array_bundle_sha256_v1"
        ) or receipt.get(receipt_key) != payload.get("sha256"):
            raise ComparisonContractError(f"PBC {name} provenance is unbound")
        provenance_receipts[name] = {
            "path": str(path.resolve()),
            "artifact_sha256": artifact_sha,
            "content_sha256": payload["sha256"],
        }

    return {
        "path": str(receipt_path.resolve()),
        "sha256": frozen.sha256_file(receipt_path),
        "gate_status": "passed",
        "mode": "full",
        "manifest_sha256": manifest_sha256,
        "post_run_audit_version": PBC_POSTRUN_AUDIT_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "daily_lag_contract_verified": True,
        "provenance": provenance_receipts,
    }


def _artifact(
    root: Path, manifest: Mapping[str, Any], relative: str
) -> tuple[Path, str]:
    path = root / relative
    expected = manifest.get("artifact_sha256", {}).get(relative)
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ComparisonContractError(f"unreceipted input artifact: {relative}")
    actual = frozen.sha256_file(path)
    if actual != expected:
        raise ComparisonContractError(
            f"artifact hash mismatch for {path}: {actual} != {expected}"
        )
    return path, actual


def validate_source_bindings(
    pbc_root: Path,
    pbc_manifest: Mapping[str, Any],
    neural_root: Path,
    neural_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the live scoring core and reused neural helpers to frozen sources."""

    pbc_snapshot, pbc_snapshot_sha = _artifact(
        pbc_root, pbc_manifest, "code/src/fuxi_pbc_core.py"
    )
    live_pbc_core = PROJECT_ROOT / "src/fuxi_pbc_core.py"
    live_pbc_sha = frozen.sha256_file(live_pbc_core)
    if live_pbc_sha != pbc_snapshot_sha:
        raise ComparisonContractError(
            "live categorical scoring core differs from completed PBC V2 source"
        )

    neural_snapshot, neural_snapshot_sha = _artifact(
        neural_root,
        neural_manifest,
        "code/src/fuxi_allseason_ensemble_calibration.py",
    )
    live_neural = PROJECT_ROOT / "src/fuxi_allseason_ensemble_calibration.py"
    helper_names = (
        "_block_bootstrap_indices",
        "apply_affine_log_calibration",
        "apply_moment_fit",
        "load_imd_observations",
        "load_member_cache",
        "make_split_indices",
        "materialize_cases",
        "numpy_ensemble_crps",
    )
    frozen_helper_sha = _function_ast_bundle_sha256(neural_snapshot, helper_names)
    live_helper_sha = _function_ast_bundle_sha256(live_neural, helper_names)
    if frozen_helper_sha != live_helper_sha:
        raise ComparisonContractError(
            "live neural reconstruction helpers differ from the frozen neural run"
        )
    return {
        "pbc_core": {
            "live_path": str(live_pbc_core.resolve()),
            "live_sha256": live_pbc_sha,
            "frozen_snapshot_path": str(pbc_snapshot.resolve()),
            "frozen_snapshot_sha256": pbc_snapshot_sha,
            "exact_file_identity": True,
        },
        "neural_helpers": {
            "live_path": str(live_neural.resolve()),
            "live_file_sha256": frozen.sha256_file(live_neural),
            "frozen_snapshot_path": str(neural_snapshot.resolve()),
            "frozen_snapshot_sha256": neural_snapshot_sha,
            "helper_names": list(helper_names),
            "helper_ast_bundle_sha256": live_helper_sha,
            "selected_helper_identity": True,
        },
    }


def validate_cache_provenance(
    cache: frozen.MemberCache,
    pbc_manifest: Mapping[str, Any],
    neural_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream the shared cache and bind its metadata to both input manifests."""

    pbc_cache = pbc_manifest.get("cache", {})
    neural_cache = neural_manifest.get("cache", {})
    for key in (
        "data_sha256",
        "metadata_sha256",
        "manifest_sha256",
        "source_fingerprint",
        "scope",
    ):
        if pbc_cache.get(key) != neural_cache.get(key):
            raise ComparisonContractError(f"PBC/neural cache differs for {key}")
    if Path(pbc_cache.get("data_file", "")).resolve() != cache.members_path.resolve():
        raise ComparisonContractError("loaded member cache differs from PBC receipt")
    actual = {
        "data_sha256": frozen.sha256_file(cache.members_path),
        "metadata_sha256": frozen.sha256_file(cache.metadata_path),
        "manifest_sha256": (
            None
            if cache.manifest_path is None
            else frozen.sha256_file(cache.manifest_path)
        ),
    }
    for key, digest in actual.items():
        if digest != pbc_cache.get(key):
            raise ComparisonContractError(f"live cache hash differs for {key}")
    return {
        "data_file": str(cache.members_path.resolve()),
        "metadata_file": str(cache.metadata_path.resolve()),
        "manifest_file": (
            None if cache.manifest_path is None else str(cache.manifest_path.resolve())
        ),
        **actual,
        "source_fingerprint": pbc_cache.get("source_fingerprint"),
        "scope": pbc_cache.get("scope"),
        "bound_to_both_input_manifests": True,
    }


def validate_observation_provenance(
    observations: frozen.ObservationBundle,
    pbc_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the exact PBC observation bundle from freshly loaded IMD."""

    actual = array_bundle_provenance(
        {
            "weekly_truth": observations.weekly_truth,
            "observation_fraction": observations.observation_fraction,
            "weights": observations.weights,
        },
        bundle_name="imd_weekly_truth_fraction_and_scoring_weights",
    )
    expected = pbc_manifest.get("observation_bundle_provenance", {})
    if actual != {
        key: expected.get(key)
        for key in ("schema_version", "bundle_name", "sha256", "arrays")
    }:
        raise ComparisonContractError(
            "fresh IMD truth/fraction/weights differ from the completed PBC V2 run"
        )
    return {**actual, "exact_pbc_identity": True}


def _first_npz_array(
    archive: Mapping[str, np.ndarray], names: Sequence[str]
) -> np.ndarray:
    for name in names:
        if name in archive:
            return np.asarray(archive[name])
    raise ComparisonContractError(f"support artifact lacks every key in {tuple(names)}")


def load_support(root: Path, manifest: Mapping[str, Any]) -> tuple[SupportBundle, str]:
    path, digest = _artifact(root, manifest, "evaluation/scoring_support.npz")
    with np.load(path, allow_pickle=False) as archive:
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
        support = np.asarray(archive["support_mask"], dtype=bool)
        weights = _first_npz_array(
            archive,
            ("scoring_weight", "scoring_weight_km2_fraction"),
        ).astype(np.float64)
        normalized = _first_npz_array(
            archive,
            ("normalized_scoring_weight", "normalized_weight"),
        ).astype(np.float64)
    if support.shape != weights.shape or normalized.shape != weights.shape:
        raise ComparisonContractError("scoring-support arrays have incompatible shapes")
    if latitude.shape != (weights.shape[0],) or longitude.shape != (weights.shape[1],):
        raise ComparisonContractError("scoring coordinates do not match weights")
    if not np.array_equal(support, weights > 0.0):
        raise ComparisonContractError(
            "support mask differs from positive scoring weights"
        )
    expected_normalized = weights / weights.sum(dtype=np.float64)
    if not np.allclose(normalized, expected_normalized, rtol=0.0, atol=1.0e-15):
        raise ComparisonContractError("normalized scoring weights are inconsistent")
    return SupportBundle(latitude, longitude, support, weights, normalized), digest


def assert_same_support(left: SupportBundle, right: SupportBundle) -> None:
    for name in ("latitude", "longitude", "support", "weights", "normalized_weights"):
        first = getattr(left, name)
        second = getattr(right, name)
        if first.dtype == bool:
            equal = np.array_equal(first, second)
        else:
            equal = np.allclose(first, second, rtol=0.0, atol=0.0)
        if not equal:
            raise ComparisonContractError(
                f"PBC and neural scoring support differ: {name}"
            )


def _calendar_model(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    expected_levels: np.ndarray,
    support: np.ndarray,
    manifest: Mapping[str, Any],
) -> CalendarQuantiles:
    levels = np.asarray(archive[f"{prefix}_levels"], dtype=np.float32)
    if not np.array_equal(levels, expected_levels):
        raise ComparisonContractError(f"unexpected {prefix} probability levels")
    thresholds = np.asarray(archive[f"{prefix}_thresholds_mm_day"], dtype=np.float32)
    empirical = np.asarray(
        archive[f"{prefix}_training_empirical_strict_cdf"], dtype=np.float32
    )
    expected_shape = (366, len(levels), *support.shape)
    if thresholds.shape != expected_shape or empirical.shape != expected_shape:
        raise ComparisonContractError(f"unexpected {prefix} threshold shape")
    if not np.isfinite(thresholds[..., support]).all() or np.any(
        thresholds[..., support] < 0.0
    ):
        raise ComparisonContractError(f"invalid supported {prefix} thresholds")
    if not np.isfinite(empirical[..., support]).all() or np.any(
        (empirical[..., support] < 0.0) | (empirical[..., support] > 1.0)
    ):
        raise ComparisonContractError(f"invalid supported {prefix} climatology CDF")
    if (
        not np.isnan(thresholds[..., ~support]).all()
        or not np.isnan(empirical[..., ~support]).all()
    ):
        raise ComparisonContractError(f"unsupported {prefix} thresholds must be NaN")
    quantiles = manifest.get("quantile_definitions", {})
    window_days = int(quantiles.get("calendar_window_days", -1))
    fit_indices = np.asarray(archive[f"{prefix}_fit_indices"], dtype=np.int64)
    sample_count = np.asarray(
        archive[f"{prefix}_calendar_sample_count"], dtype=np.int32
    )
    unique_count = int(np.asarray(archive[f"{prefix}_unique_fit_window_count"]).item())
    duplicate_count = int(
        np.asarray(archive[f"{prefix}_duplicate_fit_window_count"]).item()
    )
    return CalendarQuantiles(
        levels=levels,
        thresholds=thresholds,
        empirical_cdf=empirical,
        support=support.copy(),
        window_radius_days=(window_days - 1) // 2,
        minimum_samples=int(
            quantiles.get("minimum_samples_with_nearest_calendar_fallback", -1)
        ),
        fit_indices=fit_indices,
        sample_count_by_day=sample_count,
        unique_fit_window_count=unique_count,
        duplicate_fit_window_count=duplicate_count,
    )


def load_thresholds(
    root: Path,
    manifest: Mapping[str, Any],
    support: np.ndarray,
    train_indices: np.ndarray,
) -> ThresholdBundle:
    path, digest = _artifact(root, manifest, "models/pbc_fit.npz")
    with np.load(path, allow_pickle=False) as archive:
        quintile = _calendar_model(
            archive, "quintile", QUINTILE_LEVELS, support, manifest
        )
        semidecile = _calendar_model(
            archive, "semidecile", SEMIDECILE_LEVELS, support, manifest
        )
    for model in (quintile, semidecile):
        if not np.array_equal(model.fit_indices, train_indices):
            raise ComparisonContractError(
                "PBC thresholds were not fit on exact train split"
            )
    return ThresholdBundle(quintile, semidecile, path, digest)


def score_ensemble(
    members: np.ndarray,
    quintile_thresholds: np.ndarray,
    semidecile_thresholds: np.ndarray,
    quintile_observed: np.ndarray,
    semidecile_observed: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> ScoreArrays:
    """Score one ensemble and retain no forecast probability arrays."""

    quintile_cdf = ensemble_cdf(members, quintile_thresholds, chunk_size=chunk_size)
    semidecile_cdf = ensemble_cdf(members, semidecile_thresholds, chunk_size=chunk_size)
    if not is_valid_cdf_for_thresholds(
        quintile_cdf, quintile_thresholds, axis=2
    ) or not is_valid_cdf_for_thresholds(semidecile_cdf, semidecile_thresholds, axis=2):
        raise ComparisonContractError(
            "reconstructed ensemble violates equality-aware threshold geometry"
        )
    result = ScoreArrays(
        quintile_rps=weighted_spatial_mean(
            ranked_probability_score(
                quintile_cdf, quintile_observed, quintile_thresholds
            ),
            weights,
        ),
        paper_nominal_cut_rps=weighted_spatial_mean(
            paper_nominal_cut_rps(quintile_cdf, quintile_observed, quintile_thresholds),
            weights,
        ),
        upper_brier=upper_tail_brier_score(
            semidecile_cdf,
            semidecile_observed,
            semidecile_thresholds,
            weights,
        ),
        quintile_probability_bias=probability_bias(
            quintile_cdf, quintile_observed, weights
        ),
        semidecile_probability_bias=probability_bias(
            semidecile_cdf, semidecile_observed, weights
        ),
    )
    del quintile_cdf, semidecile_cdf
    return result


def reference_scores(
    quintile_model: CalendarQuantiles,
    semidecile_model: CalendarQuantiles,
    quintile_thresholds: np.ndarray,
    semidecile_thresholds: np.ndarray,
    quintile_observed: np.ndarray,
    semidecile_observed: np.ndarray,
    quintile_climatology: np.ndarray,
    semidecile_climatology: np.ndarray,
    weights: np.ndarray,
) -> ReferenceScores:
    # Nominal levels are a representation sensitivity only. They must be
    # masked, zero-anchored, tie-collapsed, and projected before being treated
    # as a forecast CDF. The literal [0.2, ..., 0.8] vector is used separately
    # as the Guan-formula denominator and is never called a valid tied CDF.
    quintile_thresholds = np.asarray(quintile_thresholds, dtype=np.float32)
    semidecile_thresholds = np.asarray(semidecile_thresholds, dtype=np.float32)
    if quintile_thresholds.shape != quintile_observed.shape or (
        semidecile_thresholds.shape != semidecile_observed.shape
    ):
        raise ComparisonContractError("reference scoring lacks exact test thresholds")
    if not is_valid_cdf_for_thresholds(
        quintile_climatology, quintile_thresholds, axis=2
    ) or not is_valid_cdf_for_thresholds(
        semidecile_climatology, semidecile_thresholds, axis=2
    ):
        raise ComparisonContractError(
            "training empirical climatology violates threshold geometry"
        )
    nominal_quintile = project_cdf_for_thresholds(
        np.where(
            np.isfinite(quintile_thresholds),
            np.broadcast_to(
                quintile_model.levels[None, None, :, None, None],
                quintile_observed.shape,
            ),
            np.nan,
        ),
        quintile_thresholds,
        axis=2,
    )
    nominal_semidecile = project_cdf_for_thresholds(
        np.where(
            np.isfinite(semidecile_thresholds),
            np.broadcast_to(
                semidecile_model.levels[None, None, :, None, None],
                semidecile_observed.shape,
            ),
            np.nan,
        ),
        semidecile_thresholds,
        axis=2,
    )
    return ReferenceScores(
        projected_nominal_rps=weighted_spatial_mean(
            ranked_probability_score(
                nominal_quintile, quintile_observed, quintile_thresholds
            ),
            weights,
        ),
        training_empirical_rps=weighted_spatial_mean(
            ranked_probability_score(
                quintile_climatology,
                quintile_observed,
                quintile_thresholds,
            ),
            weights,
        ),
        paper_fixed_nominal_rps=weighted_spatial_mean(
            paper_fixed_nominal_reference_rps(quintile_observed, quintile_thresholds),
            weights,
        ),
        projected_nominal_upper_brier=upper_tail_brier_score(
            nominal_semidecile,
            semidecile_observed,
            semidecile_thresholds,
            weights,
        ),
        training_empirical_upper_brier=upper_tail_brier_score(
            semidecile_climatology,
            semidecile_observed,
            semidecile_thresholds,
            weights,
        ),
    )


def _seasons(midpoints: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(midpoints.reshape(-1)).month.to_numpy()
    labels = np.asarray(
        [
            (
                "DJF"
                if month in (12, 1, 2)
                else (
                    "MAM"
                    if month in (3, 4, 5)
                    else "JJA" if month in (6, 7, 8) else "SON"
                )
            )
            for month in months
        ],
        dtype=object,
    )
    return labels.reshape(midpoints.shape)


def case_score_frame(
    method: str,
    seed: int | str,
    scores: ScoreArrays,
    references: ReferenceScores,
    initializations: np.ndarray,
    quintile_thresholds: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    midpoints = verification_midpoints(initializations, scores.quintile_rps.shape[1])
    starts = (
        initializations[:, None]
        + (7 * np.arange(scores.quintile_rps.shape[1])).astype("timedelta64[D]")[None]
    )
    ends = starts + np.timedelta64(6, "D")
    seasons = _seasons(midpoints)
    supported_weight = float(np.sum(np.asarray(weights)[np.asarray(weights) > 0.0]))
    expanded_weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64),
        quintile_thresholds.shape[:2] + np.asarray(weights).shape,
    )
    paper_weight = np.sum(
        np.where(quintile_thresholds[:, :, -1] > 0.0, expanded_weights, 0.0),
        axis=(-2, -1),
        dtype=np.float64,
    )
    if paper_weight.shape != scores.quintile_rps.shape or np.any(paper_weight <= 0.0):
        raise ComparisonContractError(
            "one or more case/leads has no positive-q80 weight"
        )
    records: list[dict[str, Any]] = []
    for case in range(len(initializations)):
        for lead in range(scores.quintile_rps.shape[1]):
            score = float(scores.quintile_rps[case, lead])
            nominal = float(references.projected_nominal_rps[case, lead])
            empirical = float(references.training_empirical_rps[case, lead])
            paper_score = float(scores.paper_nominal_cut_rps[case, lead])
            paper_reference = float(references.paper_fixed_nominal_rps[case, lead])
            records.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "score_contract": SCORING_CONTRACT_VERSION,
                    "nominal_reference_contract": NOMINAL_REFERENCE_CONTRACT,
                    "initialization": np.datetime_as_string(
                        initializations[case], unit="D"
                    ),
                    "lead_week": lead + 1,
                    "verification_start": np.datetime_as_string(
                        starts[case, lead], unit="D"
                    ),
                    "verification_midpoint": np.datetime_as_string(
                        midpoints[case, lead], unit="D"
                    ),
                    "verification_end": np.datetime_as_string(
                        ends[case, lead], unit="D"
                    ),
                    "season": seasons[case, lead],
                    "rps": score,
                    "paper_nominal_cut_q80_positive_rps": paper_score,
                    "paper_fixed_nominal_climatology_rps": paper_reference,
                    "paper_q80_retained_scoring_weight": float(
                        paper_weight[case, lead]
                    ),
                    "paper_q80_total_supported_scoring_weight": supported_weight,
                    "paper_q80_retained_scoring_weight_fraction": float(
                        paper_weight[case, lead] / supported_weight
                    ),
                    "nominal_climatology_rps": nominal,
                    "training_empirical_climatology_rps": empirical,
                    "rpss_vs_nominal_climatology_case": (
                        1.0 - score / nominal if nominal > 0.0 else np.nan
                    ),
                    "rpss_vs_training_empirical_climatology_case": (
                        1.0 - score / empirical if empirical > 0.0 else np.nan
                    ),
                    "paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case": (
                        1.0 - paper_score / paper_reference
                        if paper_reference > 0.0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def upper_case_frame(
    method: str,
    seed: int | str,
    scores: ScoreArrays,
    references: ReferenceScores,
    initializations: np.ndarray,
) -> pd.DataFrame:
    midpoints = verification_midpoints(initializations, scores.upper_brier.shape[1])
    seasons = _seasons(midpoints)
    records: list[dict[str, Any]] = []
    for case in range(len(initializations)):
        for lead in range(scores.upper_brier.shape[1]):
            records.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "initialization": np.datetime_as_string(
                        initializations[case], unit="D"
                    ),
                    "lead_week": lead + 1,
                    "verification_midpoint": np.datetime_as_string(
                        midpoints[case, lead], unit="D"
                    ),
                    "season": seasons[case, lead],
                    "brier_score": float(scores.upper_brier[case, lead]),
                    "nominal_climatology_brier_score": float(
                        references.projected_nominal_upper_brier[case, lead]
                    ),
                    "training_empirical_climatology_brier_score": float(
                        references.training_empirical_upper_brier[case, lead]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def probability_bias_frame(
    method: str,
    seed: int | str,
    scores: ScoreArrays,
    initializations: np.ndarray,
) -> pd.DataFrame:
    midpoints = verification_midpoints(initializations, scores.quintile_rps.shape[1])
    seasons = _seasons(midpoints)
    records: list[dict[str, Any]] = []
    for family, levels, bias in (
        ("quintile", QUINTILE_LEVELS, scores.quintile_probability_bias),
        ("semidecile", SEMIDECILE_LEVELS, scores.semidecile_probability_bias),
    ):
        for lead in range(bias.shape[1]):
            groups = [("ALL", np.ones(len(initializations), dtype=bool))]
            groups.extend(
                (season, seasons[:, lead] == season)
                for season in ("DJF", "MAM", "JJA", "SON")
            )
            for season, chosen in groups:
                if not np.any(chosen):
                    continue
                for cut, nominal in enumerate(levels):
                    values = bias[chosen, lead, cut]
                    records.append(
                        {
                            "family": family,
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


def average_seed_case_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Average proper scores across seeds without averaging forecasts."""

    if set(frame.method.unique()) != set(NEURAL_CONFIGURATIONS):
        raise ComparisonContractError("seed-score table lacks a neural configuration")
    expected_rows = len(SEEDS)
    keys = [
        "method",
        "method_label",
        "score_contract",
        "nominal_reference_contract",
        "initialization",
        "lead_week",
        "verification_start",
        "verification_midpoint",
        "verification_end",
        "season",
    ]
    identity_keys = ["method", "initialization", "lead_week", "seed"]
    if frame.duplicated(identity_keys).any():
        raise ComparisonContractError("seed-score table contains duplicate case rows")
    counts = frame.groupby(["method", "initialization", "lead_week"]).seed.nunique()
    if not np.all(counts.to_numpy() == expected_rows):
        raise ComparisonContractError("not every neural case has all three seed scores")
    if set(pd.to_numeric(frame.seed).astype(int).unique()) != set(SEEDS):
        raise ComparisonContractError("unexpected neural seed set")
    if set(frame.score_contract) != {SCORING_CONTRACT_VERSION} or set(
        frame.nominal_reference_contract
    ) != {NOMINAL_REFERENCE_CONTRACT}:
        raise ComparisonContractError("neural seed scores use an invalid contract")
    reference_columns = [
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
        "paper_fixed_nominal_climatology_rps",
        "paper_q80_retained_scoring_weight",
        "paper_q80_total_supported_scoring_weight",
        "paper_q80_retained_scoring_weight_fraction",
    ]
    reference_spread = frame.groupby(["method", "initialization", "lead_week"])[
        reference_columns
    ].agg(lambda values: float(values.max() - values.min()))
    if not np.allclose(reference_spread.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12):
        raise ComparisonContractError("climatology references differ across seeds")
    result = (
        frame.groupby(keys, as_index=False)
        .agg(
            rps=("rps", "mean"),
            paper_nominal_cut_q80_positive_rps=(
                "paper_nominal_cut_q80_positive_rps",
                "mean",
            ),
            nominal_climatology_rps=("nominal_climatology_rps", "first"),
            training_empirical_climatology_rps=(
                "training_empirical_climatology_rps",
                "first",
            ),
            paper_fixed_nominal_climatology_rps=(
                "paper_fixed_nominal_climatology_rps",
                "first",
            ),
            paper_q80_retained_scoring_weight=(
                "paper_q80_retained_scoring_weight",
                "first",
            ),
            paper_q80_total_supported_scoring_weight=(
                "paper_q80_total_supported_scoring_weight",
                "first",
            ),
            paper_q80_retained_scoring_weight_fraction=(
                "paper_q80_retained_scoring_weight_fraction",
                "first",
            ),
        )
        .sort_values(["method", "initialization", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(2, "seed", "mean_of_seed_scores_42_43_44")
    result["rpss_vs_nominal_climatology_case"] = (
        1.0 - result.rps / result.nominal_climatology_rps
    )
    result["rpss_vs_training_empirical_climatology_case"] = (
        1.0 - result.rps / result.training_empirical_climatology_rps
    )
    result["paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case"] = (
        1.0
        - result.paper_nominal_cut_q80_positive_rps
        / result.paper_fixed_nominal_climatology_rps
    )
    return result


def average_seed_upper_scores(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method",
        "method_label",
        "initialization",
        "lead_week",
        "verification_midpoint",
        "season",
    ]
    identity_keys = ["method", "initialization", "lead_week", "seed"]
    if frame.duplicated(identity_keys).any():
        raise ComparisonContractError("upper-tail table contains duplicate seed rows")
    counts = frame.groupby(["method", "initialization", "lead_week"]).seed.nunique()
    if not np.all(counts.to_numpy() == len(SEEDS)):
        raise ComparisonContractError("upper-tail rows lack three seed scores")
    reference_spread = frame.groupby(["method", "initialization", "lead_week"])[
        [
            "nominal_climatology_brier_score",
            "training_empirical_climatology_brier_score",
        ]
    ].agg(lambda values: float(values.max() - values.min()))
    if not np.allclose(reference_spread.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12):
        raise ComparisonContractError("upper-tail references differ across seeds")
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
        )
        .sort_values(["method", "initialization", "lead_week"])
        .reset_index(drop=True)
    )
    result.insert(2, "seed", "mean_of_seed_scores_42_43_44")
    return result


def average_seed_probability_bias(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "family",
        "method",
        "method_label",
        "lead_week",
        "season",
        "nominal_cumulative_probability",
    ]
    if frame.duplicated([*keys, "seed"]).any():
        raise ComparisonContractError(
            "probability-bias table contains duplicate seed rows"
        )
    counts = frame.groupby(keys).seed.nunique()
    if not np.all(counts.to_numpy() == len(SEEDS)):
        raise ComparisonContractError("probability-bias rows lack three seed scores")
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


def aggregate_quintile_scores(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["_paper_score_weighted_sum"] = (
        prepared.paper_nominal_cut_q80_positive_rps
        * prepared.paper_q80_retained_scoring_weight
    )
    prepared["_paper_reference_weighted_sum"] = (
        prepared.paper_fixed_nominal_climatology_rps
        * prepared.paper_q80_retained_scoring_weight
    )
    result = prepared.groupby(
        [*group_columns, "method", "method_label"], as_index=False
    ).agg(
        score_contract=("score_contract", "first"),
        nominal_reference_contract=("nominal_reference_contract", "first"),
        rps=("rps", "mean"),
        _paper_score_weighted_sum=("_paper_score_weighted_sum", "sum"),
        _paper_reference_weighted_sum=("_paper_reference_weighted_sum", "sum"),
        paper_q80_retained_scoring_weight=(
            "paper_q80_retained_scoring_weight",
            "sum",
        ),
        nominal_climatology_rps=("nominal_climatology_rps", "mean"),
        training_empirical_climatology_rps=(
            "training_empirical_climatology_rps",
            "mean",
        ),
        n_initializations=("initialization", "nunique"),
    )
    result["rpss_vs_nominal_climatology"] = (
        1.0 - result.rps / result.nominal_climatology_rps
    )
    result["rpss_vs_training_empirical_climatology"] = (
        1.0 - result.rps / result.training_empirical_climatology_rps
    )
    result["paper_nominal_cut_q80_positive_rps"] = (
        result._paper_score_weighted_sum / result.paper_q80_retained_scoring_weight
    )
    result["paper_fixed_nominal_climatology_rps"] = (
        result._paper_reference_weighted_sum / result.paper_q80_retained_scoring_weight
    )
    result["paper_nominal_cut_rpss_vs_fixed_nominal_climatology"] = (
        1.0
        - result.paper_nominal_cut_q80_positive_rps
        / result.paper_fixed_nominal_climatology_rps
    )
    return result.drop(
        columns=["_paper_score_weighted_sum", "_paper_reference_weighted_sum"]
    )


def aggregate_upper_scores(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, label, lead), rows in frame.groupby(
        ["method", "method_label", "lead_week"]
    ):
        for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
            selected = rows if season == "ALL" else rows.loc[rows.season == season]
            if selected.empty:
                continue
            score = float(selected.brier_score.mean())
            nominal = float(selected.nominal_climatology_brier_score.mean())
            tie = float(selected.training_empirical_climatology_brier_score.mean())
            records.append(
                {
                    "event": UPPER_EVENT,
                    "status": "scored_on_positive_q95_cells_only",
                    "method": method,
                    "method_label": label,
                    "lead_week": int(lead),
                    "season": season,
                    "brier_score": score,
                    "nominal_climatology_brier_score": nominal,
                    "training_empirical_climatology_brier_score": tie,
                    "bss_vs_nominal_climatology": 1.0 - score / nominal,
                    "bss_vs_training_empirical_climatology": 1.0 - score / tie,
                    "n_initializations": int(selected.initialization.nunique()),
                }
            )
    return pd.DataFrame.from_records(records)


def paired_block_bootstrap(
    case_scores: pd.DataFrame,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    """Paired two-stage year/block CIs for primary and paper-formula RPS."""

    required = {
        "method",
        "initialization",
        "lead_week",
        "rps",
        "paper_nominal_cut_q80_positive_rps",
        "paper_q80_retained_scoring_weight",
    }
    if not required.issubset(case_scores):
        raise ComparisonContractError("bootstrap case scores lack required columns")
    if set(case_scores.method) != set(COMPARISON_METHODS):
        raise ComparisonContractError("bootstrap methods are incomplete")
    if case_scores.duplicated(["method", "initialization", "lead_week"]).any():
        raise ComparisonContractError("bootstrap rows are not uniquely paired")

    case_order = np.asarray(
        sorted(case_scores.initialization.unique()), dtype="datetime64[D]"
    )
    years = pd.DatetimeIndex(case_order).year.to_numpy()
    unique_years = np.unique(years)
    year_counts = np.asarray(
        [np.count_nonzero(years == year) for year in unique_years], dtype=np.int64
    )
    if unique_years.size != 2 or np.unique(year_counts).size != 1:
        raise ComparisonContractError(
            "comparison bootstrap needs two test years with equal initialization counts"
        )
    draws = frozen._block_bootstrap_indices(
        case_order,
        n_resamples=samples,
        block_length=block_length,
        seed=seed,
    )
    if draws.shape != (samples, len(case_order)):
        raise ComparisonContractError("unexpected bootstrap draw geometry")
    labels = [np.datetime_as_string(value, unit="D") for value in case_order]

    comparisons = tuple(
        (method, baseline)
        for method in ("moment_calibration", "summary_only", "location_spread")
        for baseline in PBC_METHODS
    ) + (
        ("pbc_combined", "raw_fuxi_categorical"),
        ("pbc_combined", "debias_plus_plus"),
        ("pbc_combined", "persistence_plus_plus"),
    )

    primary_pooled = case_scores.groupby(
        ["initialization", "method"], as_index=False
    ).agg(score=("rps", "mean"))
    paper = case_scores.assign(
        _numerator=(
            case_scores.paper_nominal_cut_q80_positive_rps
            * case_scores.paper_q80_retained_scoring_weight
        )
    )
    paper_pooled = paper.groupby(["initialization", "method"], as_index=False).agg(
        numerator=("_numerator", "sum"),
        weight=("paper_q80_retained_scoring_weight", "sum"),
    )
    paper_pooled["score"] = paper_pooled.numerator / paper_pooled.weight

    scopes: list[tuple[str, int, pd.DataFrame, pd.DataFrame]] = [
        ("W1-W6", 0, primary_pooled, paper_pooled)
    ]
    for lead in sorted(case_scores.lead_week.unique()):
        primary_week = case_scores.loc[case_scores.lead_week == lead].rename(
            columns={"rps": "score"}
        )
        paper_week = paper.loc[paper.lead_week == lead].rename(
            columns={
                "paper_nominal_cut_q80_positive_rps": "score",
                "paper_q80_retained_scoring_weight": "weight",
                "_numerator": "numerator",
            }
        )
        scopes.append((f"W{int(lead)}", int(lead), primary_week, paper_week))

    records: list[dict[str, Any]] = []
    for lead_scope, lead_week, primary_rows, paper_rows in scopes:
        primary_pivot = primary_rows.pivot(
            index="initialization", columns="method", values="score"
        ).reindex(labels)
        paper_pivot = paper_rows.pivot(
            index="initialization", columns="method", values="score"
        ).reindex(labels)
        weight_pivot = paper_rows.pivot(
            index="initialization", columns="method", values="weight"
        ).reindex(labels)
        if (
            primary_pivot.isna().any().any()
            or paper_pivot.isna().any().any()
            or weight_pivot.isna().any().any()
        ):
            raise ComparisonContractError("bootstrap comparison is not fully paired")
        weights_matrix = weight_pivot.to_numpy(dtype=np.float64)
        if not np.allclose(
            weights_matrix,
            weights_matrix[:, :1],
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ComparisonContractError(
                "paper q80 support differs between comparison methods"
            )

        for metric, pivot, metric_contract in (
            (
                "primary_normalized_informative_positive_cut_rps",
                primary_pivot,
                SCORING_CONTRACT_VERSION,
            ),
            (
                "secondary_guan_nominal_cut_q80_positive_rps",
                paper_pivot,
                "guan_four_nominal_cut_sum_q80_positive_weighted_v1",
            ),
        ):
            for method, baseline in comparisons:
                method_values = pivot[method].to_numpy(dtype=np.float64)
                baseline_values = pivot[baseline].to_numpy(dtype=np.float64)
                if metric.startswith("primary"):
                    method_point = float(np.mean(method_values))
                    baseline_point = float(np.mean(baseline_values))
                    method_draw = np.mean(method_values[draws], axis=1)
                    baseline_draw = np.mean(baseline_values[draws], axis=1)
                    retained_weight = np.nan
                else:
                    method_weight = weight_pivot[method].to_numpy(dtype=np.float64)
                    baseline_weight = weight_pivot[baseline].to_numpy(dtype=np.float64)
                    method_point = float(
                        np.sum(method_values * method_weight) / np.sum(method_weight)
                    )
                    baseline_point = float(
                        np.sum(baseline_values * baseline_weight)
                        / np.sum(baseline_weight)
                    )
                    method_draw = np.sum(
                        method_values[draws] * method_weight[draws], axis=1
                    ) / np.sum(method_weight[draws], axis=1)
                    baseline_draw = np.sum(
                        baseline_values[draws] * baseline_weight[draws], axis=1
                    ) / np.sum(baseline_weight[draws], axis=1)
                    retained_weight = float(np.sum(method_weight))
                reductions = 1.0 - method_draw / baseline_draw
                records.append(
                    {
                        "score_contract": SCORING_CONTRACT_VERSION,
                        "metric": metric,
                        "metric_contract": metric_contract,
                        "lead_scope": lead_scope,
                        "lead_week": int(lead_week),
                        "method": method,
                        "baseline": baseline,
                        "method_score": method_point,
                        "baseline_score": baseline_point,
                        "rps_reduction_fraction": float(
                            1.0 - method_point / baseline_point
                        ),
                        "ci_lower_95": float(np.quantile(reductions, 0.025)),
                        "ci_upper_95": float(np.quantile(reductions, 0.975)),
                        "bootstrap_probability_improvement": float(
                            np.mean(reductions > 0.0)
                        ),
                        "paper_retained_scoring_weight": retained_weight,
                        "bootstrap_samples": int(samples),
                        "block_length_initializations": int(block_length),
                        "bootstrap_seed": int(seed),
                        "bootstrap_scheme": BOOTSTRAP_SCHEME,
                        "test_year_count": int(unique_years.size),
                        "initializations_per_test_year": int(year_counts[0]),
                    }
                )
    return pd.DataFrame.from_records(records)


def load_adjustment(
    neural_root: Path,
    manifest: Mapping[str, Any],
    configuration: str,
    seed: int,
    initializations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    relative = f"models/{configuration}/seed_{seed}/test_adjustments.npz"
    path, digest = _artifact(neural_root, manifest, relative)
    with np.load(path, allow_pickle=False) as archive:
        dates = np.asarray(archive["initializations"], dtype="datetime64[D]")
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        log_spread = np.asarray(archive["log_spread"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        stored_seed = int(np.asarray(archive["seed"]).item())
    expected_shape = (len(initializations), 6, 27, 27)
    if not np.array_equal(dates, initializations) or stored_seed != seed:
        raise ComparisonContractError(f"adjustment identity mismatch: {path}")
    if delta.shape != expected_shape or spread.shape != expected_shape:
        raise ComparisonContractError(f"adjustment shape mismatch: {path}")
    if log_spread.shape != expected_shape:
        raise ComparisonContractError(f"log-spread shape mismatch: {path}")
    if (
        not np.isfinite(delta).all()
        or not np.isfinite(spread).all()
        or np.any(spread <= 0.0)
    ):
        raise ComparisonContractError(f"invalid adjustment values: {path}")
    expected_spread = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
    if not np.allclose(spread, expected_spread, rtol=1.0e-6, atol=1.0e-7):
        raise ComparisonContractError(f"spread/log-spread mismatch: {path}")
    receipt = {
        "path": str(path),
        "sha256": digest,
        "configuration": configuration,
        "seed": seed,
        "shape": list(delta.shape),
        "initialization_first": np.datetime_as_string(dates[0], unit="D"),
        "initialization_last": np.datetime_as_string(dates[-1], unit="D"),
    }
    return delta, spread, receipt


def reconstruct_moment(
    neural_root: Path,
    manifest: Mapping[str, Any],
    members: np.ndarray,
    initializations: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    relative = "models/moment_calibration_fit.npz"
    path, digest = _artifact(neural_root, manifest, relative)
    with np.load(path, allow_pickle=False) as archive:
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        shrinkage = float(np.asarray(archive["shrinkage"]).item())
    if delta.shape != (6, 12, 27, 27) or spread.shape != (6, 12):
        raise ComparisonContractError("moment calibration fit has wrong shape")
    if (
        not np.isfinite(delta).all()
        or not np.isfinite(spread).all()
        or np.any(spread <= 0.0)
    ):
        raise ComparisonContractError("moment calibration fit contains invalid values")
    expected_shrinkage = float(
        manifest.get("moment_calibration", {}).get("location_shrinkage", np.nan)
    )
    if not np.isclose(shrinkage, expected_shrinkage, rtol=0.0, atol=1.0e-6):
        raise ComparisonContractError("moment shrinkage differs from manifest")
    fit = frozen.MomentFit(delta, spread, shrinkage)
    corrected = frozen.apply_moment_fit(members, initializations, fit)
    return corrected, {
        "path": str(path),
        "sha256": digest,
        "delta_shape": list(delta.shape),
        "spread_shape": list(spread.shape),
        "parameters_retained_in_comparison_output": False,
    }


def continuous_crps_receipt(
    members: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    expected: pd.DataFrame,
) -> float:
    actual = weighted_spatial_mean(frozen.numpy_ensemble_crps(members, truth), weights)
    ordered = expected.sort_values(["init", "lead_week"]).crps.to_numpy(
        dtype=np.float64
    )
    if ordered.shape != (actual.size,):
        raise ComparisonContractError("continuous CRPS receipt has wrong row count")
    difference = np.abs(actual.reshape(-1) - ordered)
    maximum = float(np.max(difference))
    if maximum > 2.0e-6:
        raise ComparisonContractError(
            f"reconstructed ensemble differs from frozen CRPS evidence: {maximum}"
        )
    return maximum


def validate_pbc_case_scores(
    frame: pd.DataFrame,
    initializations: np.ndarray,
    raw_frame: pd.DataFrame,
    *,
    tolerance: float = 2.0e-7,
) -> dict[str, float]:
    required = {
        "method",
        "method_label",
        "score_contract",
        "nominal_reference_contract",
        "initialization",
        "lead_week",
        "rps",
        "paper_nominal_cut_q80_positive_rps",
        "paper_fixed_nominal_climatology_rps",
        "paper_q80_retained_scoring_weight",
        "paper_q80_total_supported_scoring_weight",
        "paper_q80_retained_scoring_weight_fraction",
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
    }
    if not required.issubset(frame.columns):
        raise ComparisonContractError("PBC case-score table lacks required columns")
    if set(frame.method.unique()) != set(PBC_METHODS):
        raise ComparisonContractError("unexpected PBC methods")
    if set(frame.score_contract) != {SCORING_CONTRACT_VERSION} or set(
        frame.nominal_reference_contract
    ) != {NOMINAL_REFERENCE_CONTRACT}:
        raise ComparisonContractError("PBC case scores have the wrong contract")
    if len(frame) != len(initializations) * 6 * len(PBC_METHODS):
        raise ComparisonContractError("unexpected PBC case-score row count")
    if frame.duplicated(["method", "initialization", "lead_week"]).any():
        raise ComparisonContractError("duplicate PBC case-score keys")
    expected_dates = {
        np.datetime_as_string(value, unit="D") for value in initializations
    }
    if set(frame.initialization) != expected_dates:
        raise ComparisonContractError("PBC case-score dates differ from neural test")
    numeric = (
        "rps",
        "paper_nominal_cut_q80_positive_rps",
        "paper_fixed_nominal_climatology_rps",
        "paper_q80_retained_scoring_weight",
        "paper_q80_total_supported_scoring_weight",
        "paper_q80_retained_scoring_weight_fraction",
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
    )
    if not np.isfinite(frame.loc[:, numeric].to_numpy(dtype=np.float64)).all():
        raise ComparisonContractError("PBC case-score table contains nonfinite values")
    if (
        not frame.rps.between(0.0, 1.0).all()
        or not frame["paper_nominal_cut_q80_positive_rps"].between(0.0, 4.0).all()
    ):
        raise ComparisonContractError("PBC case-score bounds are invalid")
    support_spread = frame.groupby(["initialization", "lead_week"])[
        [
            "paper_q80_retained_scoring_weight",
            "paper_q80_total_supported_scoring_weight",
            "paper_q80_retained_scoring_weight_fraction",
        ]
    ].agg(lambda values: float(values.max() - values.min()))
    if not np.allclose(support_spread, 0.0, rtol=0.0, atol=1.0e-8):
        raise ComparisonContractError("PBC method rows use different q80 support")
    stored_raw = frame.loc[frame.method == "raw_fuxi_categorical"].sort_values(
        ["initialization", "lead_week"]
    )
    recomputed_raw = raw_frame.sort_values(["initialization", "lead_week"])
    receipts: dict[str, float] = {}
    for column in (
        "rps",
        "paper_nominal_cut_q80_positive_rps",
        "paper_fixed_nominal_climatology_rps",
        "paper_q80_retained_scoring_weight",
        "paper_q80_total_supported_scoring_weight",
        "paper_q80_retained_scoring_weight_fraction",
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
    ):
        maximum = float(
            np.max(
                np.abs(
                    stored_raw[column].to_numpy(dtype=np.float64)
                    - recomputed_raw[column].to_numpy(dtype=np.float64)
                )
            )
        )
        if maximum > tolerance:
            raise ComparisonContractError(
                f"PBC raw identity differs for {column}: {maximum}"
            )
        receipts[f"maximum_absolute_{column}_difference"] = maximum
    return receipts


def validate_pbc_aggregate_identity(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    key_columns: Sequence[str],
    tolerance: float = 2.0e-7,
) -> dict[str, float]:
    left = stored.sort_values(list(key_columns)).reset_index(drop=True)
    right = recomputed.sort_values(list(key_columns)).reset_index(drop=True)
    if len(left) != len(right):
        raise ComparisonContractError("PBC aggregate identity keys differ")
    for column in key_columns:
        left_key = left[column]
        right_key = right[column]
        if pd.api.types.is_numeric_dtype(left_key) and pd.api.types.is_numeric_dtype(
            right_key
        ):
            equal = np.allclose(
                left_key.to_numpy(dtype=np.float64),
                right_key.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
                equal_nan=True,
            )
        else:
            equal = np.array_equal(
                left_key.astype(str).to_numpy(), right_key.astype(str).to_numpy()
            )
        if not equal:
            raise ComparisonContractError(
                f"PBC aggregate identity keys differ for {column}"
            )
    receipts: dict[str, float] = {}
    for column in value_columns:
        maximum = float(
            np.nanmax(
                np.abs(
                    left[column].to_numpy(dtype=np.float64)
                    - right[column].to_numpy(dtype=np.float64)
                )
            )
        )
        if maximum > tolerance:
            raise ComparisonContractError(
                f"PBC aggregate identity differs for {column}: {maximum}"
            )
        receipts[f"maximum_absolute_{column}_difference"] = maximum
    return receipts


def _manifest_test_dates(manifest: Mapping[str, Any]) -> np.ndarray:
    values = manifest.get("retained_initializations", {}).get("test")
    if not isinstance(values, list) or not values:
        raise ComparisonContractError("manifest lacks retained test initializations")
    result = np.asarray(values, dtype="datetime64[D]")
    if np.unique(result).size != result.size or np.any(result[1:] <= result[:-1]):
        raise ComparisonContractError("manifest test initializations are invalid")
    if np.any(pd.DatetimeIndex(result).year == 2025):
        raise ComparisonContractError("2025 appears in retained test cases")
    return result


def _read_verified_csv(
    root: Path, manifest: Mapping[str, Any], relative: str
) -> tuple[pd.DataFrame, str]:
    path, digest = _artifact(root, manifest, relative)
    return pd.read_csv(path), digest


def build_readme(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    reconstruction: Mapping[str, Any],
) -> str:
    display = pooled[
        [
            "method",
            "rps",
            "rpss_vs_training_empirical_climatology",
            "rpss_vs_nominal_climatology",
            "paper_nominal_cut_q80_positive_rps",
            "paper_nominal_cut_rpss_vs_fixed_nominal_climatology",
        ]
    ].sort_values("rps")
    pooled_ci = bootstrap.loc[bootstrap.lead_scope == "W1-W6"]
    lines = [
        "# Frozen neural versus categorical PBC comparison",
        "",
        (
            "**Scientific status:** post-hoc 2020–2021 reused-development evidence. "
            "It is neither an untouched test nor 2025 evidence."
        ),
        "",
        (
            "Every method uses the exact same FuXi members, IMD truth, 208 test "
            "initializations, six leads, PBC training-only thresholds, and area "
            "weights. Neural ensembles are reconstructed independently for seeds "
            "42/43/44. Only their scores are averaged; forecasts and parameters "
            "are never averaged."
        ),
        "",
        (
            "The headline is normalized informative-positive-cut RPSS against the "
            "training empirical strict-CDF climatology. It counts each distinct "
            "positive physical cut once, excludes deterministic q=0 cuts, and "
            "reweights the grid on the common threshold-only mask."
        ),
        "",
        (
            "Two secondary sensitivities are explicit: a threshold-projected nominal "
            "reference, and the Guan-style sum over four nominal quintile slots on "
            "persisted q80>0 support with retained-weight pooling and literal "
            "[0.2,0.4,0.6,0.8] reference. This is formula-aligned post-hoc evidence, "
            "not a reproduction of the paper's rolling protocol."
        ),
        "",
        (
            "Upper-tail Brier/BSS uses only positive q95 cells. The partially "
            "zero-tied lower 5% tail is not scored or promoted."
        ),
        (
            "The separate quintile and semidecile probability-bias curves are generic "
            "CDF-bias diagnostics on their ordinary supported geometry; they are not "
            "positive-q95 event-probability bias."
        ),
        "",
        "## Pooled W1–W6 quintile scores",
        "",
        "```text",
        display.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "## Paired pooled comparisons",
        "",
        "```text",
        pooled_ci.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "## Integrity",
        "",
        (
            "Stored continuous CRPS is reproduced from every reconstructed neural "
            "ensemble before categorical scoring. PBC raw categorical scores, "
            "upper-tail metrics, probability bias, and both quintile score contracts "
            "are independently reproduced and checked before component rows are merged."
        ),
        "",
        (
            "Confidence intervals use paired two-stage test-year resampling followed "
            "by circular 13-initialization blocks within each sampled year. The pooled "
            "primary score first averages W1–W6 within initialization; the paper-formula "
            "sensitivity resamples retained-weight numerators and denominators. The "
            f"fixed bootstrap seed is {BOOTSTRAP_SEED}. Because only two year clusters "
            "are available, these intervals are exploratory and weak for interannual "
            "uncertainty; they are not independent-test confirmation."
        ),
        "",
        "```json",
        json.dumps(_json_safe(reconstruction), indent=2, sort_keys=True),
        "```",
        "",
        "No reconstructed ensemble or forecast-probability field is written to disk.",
    ]
    return "\n".join(lines) + "\n"


def run_comparison(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    """Run one immutable, post-hoc, score-only categorical comparison."""

    snapshot = _source_snapshot(output)
    pbc_manifest_path, pbc_manifest, pbc_manifest_sha = _load_manifest(
        Path(args.pbc_manifest), experiment=PBC_EXPERIMENT
    )
    neural_manifest_path, neural_manifest, neural_manifest_sha = _load_manifest(
        Path(args.neural_manifest), experiment=NEURAL_EXPERIMENT
    )
    pbc_root = pbc_manifest_path.parent
    neural_root = neural_manifest_path.parent
    if neural_manifest_sha != FROZEN_NEURAL_MANIFEST_SHA256:
        raise ComparisonContractError(
            "neural manifest is not the immutable full_publication_20260822T115253Z "
            f"receipt: {neural_manifest_sha}"
        )
    pbc_slurm_receipt = validate_pbc_full_receipt(
        pbc_root, pbc_manifest, pbc_manifest_sha
    )
    source_bindings = validate_source_bindings(
        pbc_root, pbc_manifest, neural_root, neural_manifest
    )
    if tuple(neural_manifest.get("seeds", ())) != SEEDS:
        raise ComparisonContractError("frozen neural run has an unexpected seed set")
    if not set(NEURAL_CONFIGURATIONS).issubset(
        neural_manifest.get("configurations", ())
    ):
        raise ComparisonContractError("frozen neural run lacks required configurations")
    if tuple(pbc_manifest.get("methods", ())) != PBC_METHODS:
        raise ComparisonContractError("completed PBC run has unexpected methods")

    pbc_dates = _manifest_test_dates(pbc_manifest)
    neural_dates = _manifest_test_dates(neural_manifest)
    if not np.array_equal(pbc_dates, neural_dates):
        raise ComparisonContractError("PBC and neural test dates differ")
    if len(neural_dates) != 208 or set(pd.DatetimeIndex(neural_dates).year) != {
        2020,
        2021,
    }:
        raise ComparisonContractError(
            "canonical comparison requires 208 2020-2021 cases"
        )
    expected_selected = {"train": 1652, "validation": 196, "test": 208}
    if pbc_manifest.get("split_counts_selected") != expected_selected or (
        neural_manifest.get("split_counts_selected") != expected_selected
    ):
        raise ComparisonContractError("input selected splits are not canonical")

    cache_path = Path(neural_manifest.get("cache", {}).get("data_file", ""))
    cache = frozen.load_member_cache(cache_path, allow_partial=False)
    cache_receipt = validate_cache_provenance(cache, pbc_manifest, neural_manifest)
    splits = frozen.make_split_indices(cache.initializations)
    if not np.array_equal(cache.initializations[splits.test], neural_dates):
        raise ComparisonContractError("cache test cases differ from both manifests")

    pbc_support, pbc_support_sha = load_support(pbc_root, pbc_manifest)
    neural_support, neural_support_sha = load_support(neural_root, neural_manifest)
    assert_same_support(pbc_support, neural_support)
    if not np.array_equal(cache.latitude, pbc_support.latitude) or not np.array_equal(
        cache.longitude, pbc_support.longitude
    ):
        raise ComparisonContractError("cache grid differs from scoring support")

    thresholds = load_thresholds(
        pbc_root, pbc_manifest, pbc_support.support, splits.train
    )
    test_initializations = cache.initializations[splits.test]
    quintile_thresholds = calendar_fields(thresholds.quintile, test_initializations, 6)
    semidecile_thresholds = calendar_fields(
        thresholds.semidecile, test_initializations, 6
    )
    quintile_climatology = project_cdf_for_thresholds(
        calendar_fields(thresholds.quintile, test_initializations, 6, empirical=True),
        quintile_thresholds,
        axis=2,
    )
    semidecile_climatology = project_cdf_for_thresholds(
        calendar_fields(thresholds.semidecile, test_initializations, 6, empirical=True),
        semidecile_thresholds,
        axis=2,
    )

    observations = frozen.load_imd_observations(cache)
    if any("2025" in str(path) for path in observations.source_stores):
        raise ComparisonContractError("observation loader opened sealed 2025")
    observation_receipt = validate_observation_provenance(observations, pbc_manifest)
    if not np.allclose(observations.weights, pbc_support.weights, rtol=0.0, atol=0.0):
        raise ComparisonContractError("fresh IMD weights differ from frozen support")
    test_truth = np.asarray(observations.weekly_truth[splits.test], dtype=np.float32)
    quintile_observed = observation_cdf(test_truth, quintile_thresholds)
    semidecile_observed = observation_cdf(test_truth, semidecile_thresholds)
    if not is_valid_cdf_for_thresholds(
        quintile_observed, quintile_thresholds, axis=2
    ) or not is_valid_cdf_for_thresholds(
        semidecile_observed, semidecile_thresholds, axis=2
    ):
        raise ComparisonContractError("IMD observations violate threshold geometry")
    references = reference_scores(
        thresholds.quintile,
        thresholds.semidecile,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        quintile_climatology,
        semidecile_climatology,
        pbc_support.weights,
    )
    test_members = frozen.materialize_cases(cache.members, splits.test)

    neural_seed_continuous, seed_metric_sha = _read_verified_csv(
        neural_root, neural_manifest, "metrics/seed_case_metrics.csv"
    )
    neural_case_continuous, case_metric_sha = _read_verified_csv(
        neural_root, neural_manifest, "metrics/case_metrics.csv"
    )
    reconstruction: dict[str, Any] = {
        "all_reconstructed_cdfs_valid_under_threshold_geometry": True,
        "forecast_probability_arrays_written": False,
    }
    adjustment_receipts: list[dict[str, Any]] = []
    seed_quintile_frames: list[pd.DataFrame] = []
    seed_upper_frames: list[pd.DataFrame] = []
    seed_bias_frames: list[pd.DataFrame] = []

    raw_scores = score_ensemble(
        test_members,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        pbc_support.weights,
        chunk_size=args.cdf_chunk_size,
    )
    raw_case = case_score_frame(
        "raw_fuxi_categorical",
        "not_applicable",
        raw_scores,
        references,
        test_initializations,
        quintile_thresholds,
        pbc_support.weights,
    )
    raw_upper_case = upper_case_frame(
        "raw_fuxi_categorical",
        "not_applicable",
        raw_scores,
        references,
        test_initializations,
    )
    raw_bias = probability_bias_frame(
        "raw_fuxi_categorical",
        "not_applicable",
        raw_scores,
        test_initializations,
    )
    raw_expected = neural_case_continuous.loc[
        neural_case_continuous.method == "raw_fuxi"
    ]
    reconstruction["raw_fuxi_maximum_absolute_continuous_crps_difference"] = (
        continuous_crps_receipt(
            test_members, test_truth, pbc_support.weights, raw_expected
        )
    )
    del raw_scores

    moment_members, moment_receipt = reconstruct_moment(
        neural_root, neural_manifest, test_members, test_initializations
    )
    moment_expected = neural_case_continuous.loc[
        neural_case_continuous.method == "moment_calibration"
    ]
    moment_receipt["maximum_absolute_continuous_crps_difference"] = (
        continuous_crps_receipt(
            moment_members, test_truth, pbc_support.weights, moment_expected
        )
    )
    reconstruction["moment_calibration"] = moment_receipt
    moment_scores = score_ensemble(
        moment_members,
        quintile_thresholds,
        semidecile_thresholds,
        quintile_observed,
        semidecile_observed,
        pbc_support.weights,
        chunk_size=args.cdf_chunk_size,
    )
    del moment_members
    moment_case = case_score_frame(
        "moment_calibration",
        "not_applicable",
        moment_scores,
        references,
        test_initializations,
        quintile_thresholds,
        pbc_support.weights,
    )
    moment_upper_case = upper_case_frame(
        "moment_calibration",
        "not_applicable",
        moment_scores,
        references,
        test_initializations,
    )
    moment_bias = probability_bias_frame(
        "moment_calibration",
        "not_applicable",
        moment_scores,
        test_initializations,
    )
    del moment_scores

    for configuration in NEURAL_CONFIGURATIONS:
        for seed in SEEDS:
            delta, spread, receipt = load_adjustment(
                neural_root,
                neural_manifest,
                configuration,
                seed,
                test_initializations,
            )
            corrected = frozen.apply_affine_log_calibration(test_members, delta, spread)
            del delta, spread
            expected = neural_seed_continuous.loc[
                (neural_seed_continuous.method == configuration)
                & (pd.to_numeric(neural_seed_continuous.seed) == seed)
            ]
            receipt["maximum_absolute_continuous_crps_difference"] = (
                continuous_crps_receipt(
                    corrected, test_truth, pbc_support.weights, expected
                )
            )
            adjustment_receipts.append(receipt)
            scores = score_ensemble(
                corrected,
                quintile_thresholds,
                semidecile_thresholds,
                quintile_observed,
                semidecile_observed,
                pbc_support.weights,
                chunk_size=args.cdf_chunk_size,
            )
            del corrected
            seed_quintile_frames.append(
                case_score_frame(
                    configuration,
                    seed,
                    scores,
                    references,
                    test_initializations,
                    quintile_thresholds,
                    pbc_support.weights,
                )
            )
            seed_upper_frames.append(
                upper_case_frame(
                    configuration, seed, scores, references, test_initializations
                )
            )
            seed_bias_frames.append(
                probability_bias_frame(
                    configuration, seed, scores, test_initializations
                )
            )
            del scores
            gc.collect()

    seed_quintile = pd.concat(seed_quintile_frames, ignore_index=True)
    seed_upper = pd.concat(seed_upper_frames, ignore_index=True)
    seed_bias = pd.concat(seed_bias_frames, ignore_index=True)
    averaged_neural_case = average_seed_case_scores(seed_quintile)
    averaged_neural_upper_case = average_seed_upper_scores(seed_upper)
    averaged_neural_bias = average_seed_probability_bias(seed_bias)

    pbc_case_scores, pbc_case_sha = _read_verified_csv(
        pbc_root, pbc_manifest, "metrics/quintile_case_scores.csv"
    )
    reconstruction["pbc_raw_quintile_identity"] = validate_pbc_case_scores(
        pbc_case_scores, test_initializations, raw_case
    )
    pbc_weekwise, pbc_weekwise_sha = _read_verified_csv(
        pbc_root, pbc_manifest, "metrics/weekwise_metrics.csv"
    )
    recomputed_pbc_weekwise = aggregate_quintile_scores(pbc_case_scores, ("lead_week",))
    reconstruction["pbc_component_weekwise_identity"] = validate_pbc_aggregate_identity(
        pbc_weekwise,
        recomputed_pbc_weekwise,
        value_columns=(
            "rps",
            "nominal_climatology_rps",
            "training_empirical_climatology_rps",
            "paper_nominal_cut_q80_positive_rps",
            "paper_fixed_nominal_climatology_rps",
            "paper_q80_retained_scoring_weight",
        ),
        key_columns=("method", "lead_week"),
    )
    pbc_case_scores = pbc_case_scores.copy()
    pbc_case_scores["seed"] = "not_applicable"
    comparison_case = pd.concat(
        (pbc_case_scores, moment_case, averaged_neural_case),
        ignore_index=True,
        sort=False,
    )
    if set(comparison_case.method) != set(COMPARISON_METHODS) or (
        comparison_case.duplicated(["method", "initialization", "lead_week"]).any()
    ):
        raise ComparisonContractError("merged comparison case scores are incomplete")
    if set(comparison_case.score_contract) != {SCORING_CONTRACT_VERSION}:
        raise ComparisonContractError("merged case scores mix scoring contracts")

    pbc_extreme, pbc_extreme_sha = _read_verified_csv(
        pbc_root, pbc_manifest, "metrics/semidecile_extreme_metrics.csv"
    )
    recomputed_raw_upper = aggregate_upper_scores(raw_upper_case)
    stored_raw_upper = pbc_extreme.loc[
        (pbc_extreme.event == UPPER_EVENT)
        & (pbc_extreme.method == "raw_fuxi_categorical")
    ]
    reconstruction["pbc_raw_upper_identity"] = validate_pbc_aggregate_identity(
        stored_raw_upper,
        recomputed_raw_upper,
        value_columns=(
            "brier_score",
            "nominal_climatology_brier_score",
            "training_empirical_climatology_brier_score",
        ),
        key_columns=("method", "lead_week", "season"),
    )
    non_pbc_upper = aggregate_upper_scores(
        pd.concat((moment_upper_case, averaged_neural_upper_case), ignore_index=True)
    )
    pbc_upper = pbc_extreme.loc[pbc_extreme.event == UPPER_EVENT].copy()
    pbc_upper["method_label"] = pbc_upper.method.map(METHOD_LABELS)
    comparison_upper = pd.concat(
        (pbc_upper, non_pbc_upper), ignore_index=True, sort=False
    )
    if set(comparison_upper.method) != set(COMPARISON_METHODS) or (
        comparison_upper.duplicated(["method", "lead_week", "season"]).any()
    ):
        raise ComparisonContractError("upper-tail comparison is incomplete")

    pbc_bias, pbc_bias_sha = _read_verified_csv(
        pbc_root, pbc_manifest, "metrics/probability_bias.csv"
    )
    reconstruction["pbc_raw_probability_bias_identity"] = (
        validate_pbc_aggregate_identity(
            pbc_bias.loc[pbc_bias.method == "raw_fuxi_categorical"],
            raw_bias,
            value_columns=(
                "probability_bias",
                "mean_absolute_case_probability_bias",
            ),
            key_columns=(
                "family",
                "method",
                "lead_week",
                "season",
                "nominal_cumulative_probability",
            ),
        )
    )
    pbc_bias = pbc_bias.copy()
    pbc_bias["method_label"] = pbc_bias.method.map(METHOD_LABELS)
    pbc_bias["seed"] = "not_applicable"
    comparison_bias = pd.concat(
        (pbc_bias, moment_bias, averaged_neural_bias), ignore_index=True, sort=False
    )
    bias_keys = [
        "family",
        "method",
        "lead_week",
        "season",
        "nominal_cumulative_probability",
    ]
    if set(comparison_bias.method) != set(COMPARISON_METHODS) or (
        comparison_bias.duplicated(bias_keys).any()
    ):
        raise ComparisonContractError("probability-bias comparison is incomplete")

    weekwise = aggregate_quintile_scores(comparison_case, ("lead_week",))
    seasonal = aggregate_quintile_scores(comparison_case, ("season", "lead_week"))
    pooled = aggregate_quintile_scores(comparison_case, ())
    bootstrap = paired_block_bootstrap(
        comparison_case,
        samples=args.bootstrap_samples,
        block_length=args.block_length,
        seed=BOOTSTRAP_SEED,
    )

    metrics = output / "metrics"
    receipts_directory = output / "receipts"
    metrics.mkdir(parents=True, exist_ok=True)
    receipts_directory.mkdir(parents=True, exist_ok=True)
    seed_quintile.to_csv(metrics / "neural_seed_quintile_case_scores.csv", index=False)
    seed_upper.to_csv(
        metrics / "neural_seed_upper_extreme_case_scores.csv", index=False
    )
    seed_bias.to_csv(metrics / "neural_seed_probability_bias.csv", index=False)
    comparison_case.to_csv(metrics / "quintile_case_scores.csv", index=False)
    weekwise.to_csv(metrics / "weekwise_metrics.csv", index=False)
    seasonal.to_csv(metrics / "seasonal_weekwise_metrics.csv", index=False)
    pooled.to_csv(metrics / "pooled_metrics.csv", index=False)
    comparison_upper.to_csv(metrics / "upper_extreme_metrics.csv", index=False)
    comparison_bias.to_csv(metrics / "probability_bias.csv", index=False)
    bootstrap.to_csv(metrics / "paired_block_bootstrap.csv", index=False)

    input_receipts: dict[str, Any] = {
        "pbc_manifest": {
            "path": str(pbc_manifest_path),
            "sha256": pbc_manifest_sha,
        },
        "pbc_slurm_gate_receipt": pbc_slurm_receipt,
        "neural_manifest": {
            "path": str(neural_manifest_path),
            "sha256": neural_manifest_sha,
            "expected_sha256": FROZEN_NEURAL_MANIFEST_SHA256,
        },
        "source_bindings": source_bindings,
        "cache": cache_receipt,
        "observation_bundle": observation_receipt,
        "pbc_thresholds": {
            "path": str(thresholds.artifact_path.resolve()),
            "sha256": thresholds.artifact_sha256,
        },
        "scoring_support": {
            "pbc_sha256": pbc_support_sha,
            "neural_sha256": neural_support_sha,
            "exact_array_identity": True,
        },
        "pbc_metrics": {
            "quintile_case_scores_sha256": pbc_case_sha,
            "weekwise_metrics_sha256": pbc_weekwise_sha,
            "semidecile_extreme_metrics_sha256": pbc_extreme_sha,
            "probability_bias_sha256": pbc_bias_sha,
        },
        "neural_metrics": {
            "seed_case_metrics_sha256": seed_metric_sha,
            "case_metrics_sha256": case_metric_sha,
        },
        "adjustments": adjustment_receipts,
        "moment_calibration": moment_receipt,
        "reconstruction_identity": reconstruction,
    }
    _atomic_json(receipts_directory / "input_receipts.json", input_receipts)
    (output / "README.md").write_text(
        build_readme(pooled, bootstrap, reconstruction), encoding="utf-8"
    )

    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "scientific_status": (
            "post-hoc retrospective 2020-2021 reused-development categorical "
            "comparison; not an untouched test or a retraining experiment"
        ),
        "created_utc": utc_now(),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "forecast": "same cached 51-member FuXi test ensembles",
            "target": "same IMD weekly mean precipitation",
            "test_years": [2020, 2021],
            "test_initializations": 208,
            "lead_weeks": [1, 2, 3, 4, 5, 6],
            "thresholds": (
                "exact persisted training-only PBC V2 quintiles and semideciles"
            ),
            "support": "exact byte-equivalent PBC and neural scoring arrays",
            "cdf_definition": "strict empirical P(Y < q)",
            "zero_threshold_contract": (
                "q=0 is deterministic, forced to F(0)=0, and excluded from primary scoring"
            ),
            "sealed_unopened_years": [2025],
            "sealed_2025_target_opened": False,
            "test_reuse_warning": (
                "2020-2021 was previously used for development and is not independent"
            ),
            "lower_tail_status": (
                "not scored or promoted because zero-tied q05 thresholds are partially "
                "degenerate and heterogeneous"
            ),
            "retraining_performed": False,
        },
        "methods": list(COMPARISON_METHODS),
        "pbc_methods": list(PBC_METHODS),
        "neural_configurations": list(NEURAL_CONFIGURATIONS),
        "seeds": list(SEEDS),
        "split_counts_selected": expected_selected,
        "retained_initializations": {
            "test": [
                np.datetime_as_string(value, unit="D") for value in test_initializations
            ]
        },
        "seed_aggregation": {
            "rule": (
                "arithmetic mean of per-seed proper scores for each identical case/lead; "
                "retained-q80 weights and references are threshold-derived and identical"
            ),
            "forecast_averaging": False,
            "parameter_averaging": False,
            "forecast_arrays_written": False,
            "learned_parameters_written": False,
            "per_seed_artifacts": [
                "metrics/neural_seed_quintile_case_scores.csv",
                "metrics/neural_seed_upper_extreme_case_scores.csv",
                "metrics/neural_seed_probability_bias.csv",
            ],
        },
        "evaluation": {
            "primary_metric": (
                "normalized informative-positive-cut quintile RPSS vs training "
                "empirical strict-CDF climatology"
            ),
            "projected_nominal_representation_sensitivity": (
                "normalized informative-positive-cut RPSS vs threshold-projected, "
                "zero-anchored nominal levels"
            ),
            "paper_formula_sensitivity": (
                "summed squared error over all four nominal quintile slots on common "
                "persisted-q80>0 support; retained-weight pooled against literal "
                "[0.2,0.4,0.6,0.8] formula reference"
            ),
            "paper_formula_status": (
                "secondary formula-aligned sensitivity under equality-aware forecasts; "
                "not a reproduction of the rolling paper protocol"
            ),
            "rps_definition": (
                "mean squared CDF error over distinct strictly positive persisted cuts; "
                "all-q0 rows are NaN and spatially reweighted on a common mask"
            ),
            "upper_extreme_event": UPPER_EVENT,
            "upper_extreme_metric": (
                "Brier/BSS for Y >= persisted training q95 on positive-q95 cells only"
            ),
            "probability_bias_metric": (
                "generic quintile and semidecile CDF-bias curves on ordinary supported "
                "CDF geometry; separate from and not labeled positive-q95 event bias"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_block_length_initializations": args.block_length,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_scheme": BOOTSTRAP_SCHEME,
            "bootstrap_pairing": (
                "two-stage year and within-year initialization blocks; primary W1-W6 "
                "first averages leads within initialization; paper sensitivity "
                "resamples retained-weight numerators and denominators"
            ),
            "bootstrap_inference_status": (
                "exploratory paired uncertainty only; two year clusters provide weak "
                "interannual uncertainty evidence and do not create an independent test"
            ),
        },
        "input_hash_gates": {
            "frozen_neural_manifest_expected_sha256": FROZEN_NEURAL_MANIFEST_SHA256,
            "frozen_neural_manifest_match": True,
            "pbc_manifest_bound_to_successful_v2_full_slurm_receipt": True,
            "pbc_daily_issue_time_lag_contract_verified": True,
            "pbc_observation_bundle_recomputed_exactly": True,
            "member_cache_streamed_sha256_verified": True,
            "live_pbc_core_matches_completed_pbc_snapshot": True,
            "neural_reconstruction_helpers_match_frozen_snapshot": True,
        },
        "reconstruction_validity": reconstruction,
        "input_receipts": input_receipts,
        "source_snapshot_sha256": snapshot,
        "observation_stores": list(observations.source_stores),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest["artifact_sha256"] = _output_checksums(output)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen neural ensembles with a completed full categorical PBC run."
        )
    )
    parser.add_argument("--pbc-manifest", type=Path, required=True)
    parser.add_argument("--neural-manifest", type=Path, default=DEFAULT_NEURAL_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--cdf-chunk-size", type=int, default=8)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("bootstrap_samples", "block_length", "cdf_chunk_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.block_length != DEFAULT_BLOCK_LENGTH:
        raise ValueError(
            f"--block-length must remain the predeclared {DEFAULT_BLOCK_LENGTH} initializations"
        )


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
        run_comparison(args, staging)
        os.replace(staging, requested)
    except Exception as error:
        _atomic_json(
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
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(f"PASS: completed categorical comparison at {requested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
