#!/usr/bin/env python3
"""Post-hoc 2018--2019 diagnostics for the global-pretraining comparison.

This evaluator never opens the raw FuXi or IMD archives.  It verifies and reads
only a completed, full, validation-only comparison artifact.  Two explicitly
post-hoc questions are evaluated:

1. Does constraining the neural correction to preserve the training-only
   log-bias anchor's India-area mean improve the spatial forecast?
2. How uncertain are paired effects when initialization-date dependence and
   the 2018/2019 season boundary are retained in the resampling design?

The projection fits no target-derived parameter.  For each case and lead it
adds one spatially uniform offset in log-rainfall space, chosen so reconstructed
physical precipitation has the same weighted area mean as the anchor.  The
bootstrap samples circular blocks independently within each year and uses one
shared draw matrix for every model, seed, lead, and metric.  Circular starts
give season-edge and interior dates equal marginal inclusion probability.

Results are exploratory development evidence.  They cannot amend the source
run's predeclared promotion decision, and no 2020+ field may be present.
"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import NEURAL_ADAPTER_SRC, PROJECT_ROOT, SOURCE_ROOT


for _path in (SOURCE_ROOT, NEURAL_ADAPTER_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fuxi_imd_bias_aware_validation_sweep as bias_aware  # noqa: E402
import fuxi_adapter.anchored as anchored_module  # noqa: E402
import fuxi_adapter.metrics as metrics_module  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    reconstruct_anchored_precipitation,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402


RESULTS_ROOT = PROJECT_ROOT / "results" / "fuxi_imd_global_pretraining_followups"
EXPECTED_TRAIN_YEARS = tuple(range(2002, 2018))
EXPECTED_VALIDATION_YEARS = (2018, 2019)
SEALED_YEARS = tuple(range(2020, 2026))
SEEDS = (42, 43, 44)
N_CASES = 70
CASES_PER_YEAR = 35
N_LEADS = 6
GRID_SHAPE = (27, 27)
SUPPORT_CELLS = 171
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BLOCK_LENGTH = 13
DEFAULT_BOOTSTRAP_SEED = 20_260_822
MAXIMUM_LOG_RAIN = 20.0
MEAN_CLOSURE_TOLERANCE = 5.0e-6

OUTPUT_ARRAY_KEYS = {
    "initializations",
    "truth",
    "climatology",
    "area_weights",
    "raw_fuxi",
    "log_bias",
    "scratch",
    "global_pretrained",
    "scratch_standardized_residual",
    "global_pretrained_standardized_residual",
}


class FollowupContractError(RuntimeError):
    """Raised when a source or derived artifact violates the evidence contract."""


@dataclass(frozen=True)
class ProjectionResult:
    prediction: np.ndarray
    adjusted_standardized_residual: np.ndarray
    log_offset: np.ndarray
    anchor_mean: np.ndarray
    unprojected_mean: np.ndarray
    projected_mean: np.ndarray
    absolute_closure_error: np.ndarray
    zero_residual_identity: np.ndarray


@dataclass(frozen=True)
class BootstrapPlan:
    indices: np.ndarray
    year_slices: Mapping[int, slice]
    year_positions: Mapping[int, np.ndarray]
    draws: int
    block_length: int
    seed: int
    mean_multiplicity: np.ndarray
    maximum_absolute_multiplicity_deviation: float


@dataclass(frozen=True)
class SourceBundle:
    run: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    verified_artifacts: Mapping[str, str]
    initializations: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    area_weights: np.ndarray
    raw_fuxi: np.ndarray
    log_bias: np.ndarray
    stored_predictions: Mapping[str, np.ndarray]
    ensemble_residuals: Mapping[str, np.ndarray]
    member_residuals: Mapping[str, Mapping[str, np.ndarray]]
    target_scale: np.ndarray


@dataclass(frozen=True)
class ComparisonSpec:
    candidate: str
    reference: str
    candidate_member: str
    reference_member: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def resolve_source_run(source_root: Path) -> Path:
    """Resolve either one completed run or a parent containing exactly one."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root / "manifest.json").is_file():
        return root
    candidates = []
    for path in sorted(root.iterdir()):
        manifest_path = path / "manifest.json"
        if (
            not path.is_dir()
            or path.name.startswith(".")
            or not manifest_path.is_file()
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(manifest, Mapping)
            and manifest.get("status") == "complete_validation_only"
            and manifest.get("mode") == "full"
            and manifest.get("smoke") is False
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise FollowupContractError(
            f"expected exactly one completed comparison below {root}; "
            f"found {len(candidates)}"
        )
    return candidates[0].resolve()


def _artifact_path(run: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise FollowupContractError(f"unsafe artifact path in manifest: {relative}")
    path = run / candidate
    if path.is_symlink():
        raise FollowupContractError(f"source artifact may not be a symlink: {relative}")
    resolved = path.resolve()
    try:
        resolved.relative_to(run)
    except ValueError as exc:
        raise FollowupContractError(
            f"artifact escapes completed source run: {relative}"
        ) from exc
    if not resolved.is_file():
        raise FollowupContractError(f"missing source artifact: {relative}")
    return resolved


def _validate_manifest(run: Path, manifest: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "status": "complete_validation_only",
        "mode": "full",
        "smoke": False,
        "scientific_eligible": True,
        "test_predictions_created": False,
        "experiment": "matched_scratch_vs_global_pretraining",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FollowupContractError(
                f"source manifest {key!r} must be {value!r}; "
                f"found {manifest.get(key)!r}"
            )
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise FollowupContractError("source manifest lacks the split contract")
    split_expected = {
        "train_initialization_years": list(EXPECTED_TRAIN_YEARS),
        "validation_initialization_years": list(EXPECTED_VALIDATION_YEARS),
        "sealed_initialization_years": list(SEALED_YEARS),
        "later_year_predictions_created": False,
        "later_year_metrics_computed": False,
    }
    for key, value in split_expected.items():
        if split.get(key) != value:
            raise FollowupContractError(
                f"source split {key!r} must be {value!r}; found {split.get(key)!r}"
            )
    training = manifest.get("training")
    if not isinstance(training, Mapping) or training.get("seeds") != list(SEEDS):
        raise FollowupContractError("source run is not the matched three-seed full run")
    data = manifest.get("data")
    if not isinstance(data, Mapping):
        raise FollowupContractError("source manifest lacks data provenance")
    if data.get("loaded_initialization_years") != list(
        EXPECTED_TRAIN_YEARS + EXPECTED_VALIDATION_YEARS
    ):
        raise FollowupContractError("source loaded initialization years changed")
    if int(data.get("maximum_loaded_initialization_year", -1)) != 2019:
        raise FollowupContractError("source manifest reaches a 2020+ initialization")
    if data.get("sealed_initialization_years_opened") is not False:
        raise FollowupContractError(
            "source manifest does not prove sealed years stayed closed"
        )
    counts = data.get("full_split_counts")
    if counts != {"train": 560, "validation": 70, "test": 0}:
        raise FollowupContractError(f"unexpected full split counts: {counts!r}")
    if int(data.get("effective_validation_cases", -1)) != N_CASES:
        raise FollowupContractError(
            "source full run does not contain 70 validation cases"
        )
    if int(data.get("support_cells", -1)) != SUPPORT_CELLS:
        raise FollowupContractError(
            "source weighted support is not the fixed 171 cells"
        )
    for key in (
        "initialization_date_max",
        "verification_target_date_max",
        "observation_date_max",
    ):
        try:
            maximum = np.datetime64(str(data[key]), "D")
        except (KeyError, TypeError, ValueError) as exc:
            raise FollowupContractError(f"invalid source date bound: {key}") from exc
        if maximum >= np.datetime64("2020-01-01", "D"):
            raise FollowupContractError(f"source {key} reaches the sealed 2020+ tier")
    if run.name.endswith(".failed") or run.name.startswith("."):
        raise FollowupContractError(
            "failed or staging source directory is not admissible"
        )


def _verify_manifest_artifacts(
    run: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise FollowupContractError("source manifest has no artifact hash inventory")
    verified: dict[str, str] = {}
    for relative, expected_hash in sorted(artifacts.items()):
        if not isinstance(relative, str) or not _is_sha256(expected_hash):
            raise FollowupContractError("invalid source artifact hash inventory")
        path = _artifact_path(run, relative)
        observed = sha256_file(path)
        if observed != expected_hash:
            raise FollowupContractError(
                f"source artifact hash mismatch for {relative}: "
                f"expected {expected_hash}, found {observed}"
            )
        verified[relative] = observed
    required = {
        "metrics/validation_outputs.npz",
        "models/log_bias_anchor.npz",
        *{
            f"models/{configuration}/seed_{seed}/validation_residual.npy"
            for configuration in ("scratch", "global_pretrained")
            for seed in SEEDS
        },
    }
    missing = sorted(required - set(verified))
    if missing:
        raise FollowupContractError(
            "source manifest does not hash required artifacts: " + ", ".join(missing)
        )
    return verified


def _load_npz_exact(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        found = set(payload.files)
        if found != expected_keys:
            raise FollowupContractError(
                f"unexpected arrays in {path}: missing={sorted(expected_keys-found)}, "
                f"extra={sorted(found-expected_keys)}"
            )
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_dates(initializations: np.ndarray) -> None:
    dates = np.asarray(initializations)
    if dates.shape != (N_CASES,) or dates.dtype.kind != "M":
        raise FollowupContractError("initializations must be 70 datetime64 values")
    dates = dates.astype("datetime64[D]")
    if np.isnat(dates).any() or len(np.unique(dates)) != N_CASES:
        raise FollowupContractError("initializations contain NaT or duplicate dates")
    if not np.array_equal(dates, np.sort(dates)):
        raise FollowupContractError("initializations are not in chronological order")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    counts = {
        int(year): int(np.count_nonzero(years == year)) for year in np.unique(years)
    }
    if counts != {2018: CASES_PER_YEAR, 2019: CASES_PER_YEAR}:
        raise FollowupContractError(
            f"validation dates must contain 35 starts in each of 2018 and 2019: {counts}"
        )
    if np.any(years >= 2020):
        raise FollowupContractError("validation arrays contain a sealed 2020+ date")


def load_source_bundle(source_root: Path) -> SourceBundle:
    """Verify every declared artifact, then load the exact validation arrays."""

    run = resolve_source_run(source_root)
    manifest_path = run / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FollowupContractError(
            f"cannot read source manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise FollowupContractError("source manifest must be a JSON object")
    _validate_manifest(run, manifest)
    verified = _verify_manifest_artifacts(run, manifest)

    arrays = _load_npz_exact(
        run / "metrics" / "validation_outputs.npz", OUTPUT_ARRAY_KEYS
    )
    _validate_dates(arrays["initializations"])
    field_shape = (N_CASES, N_LEADS, *GRID_SHAPE)
    for name in OUTPUT_ARRAY_KEYS - {"initializations", "area_weights"}:
        if arrays[name].shape != field_shape:
            raise FollowupContractError(
                f"source array {name} has shape {arrays[name].shape}; "
                f"expected {field_shape}"
            )
    weights = np.asarray(arrays["area_weights"], dtype=np.float64)
    if weights.shape != GRID_SHAPE or not np.isfinite(weights).all():
        raise FollowupContractError("area weights are not a finite 27x27 array")
    if np.any(weights < 0.0) or np.count_nonzero(weights > 0.0) != SUPPORT_CELLS:
        raise FollowupContractError(
            "area weights changed from the fixed 171-cell support"
        )
    support = weights > 0.0
    for name in (
        "truth",
        "climatology",
        "raw_fuxi",
        "log_bias",
        "scratch",
        "global_pretrained",
    ):
        values = np.asarray(arrays[name])
        if not np.isfinite(values[..., support]).all():
            raise FollowupContractError(f"{name} is nonfinite on weighted support")
        if np.any(values[..., support] < 0.0):
            raise FollowupContractError(f"{name} contains negative precipitation")
    for name in (
        "scratch_standardized_residual",
        "global_pretrained_standardized_residual",
    ):
        if not np.isfinite(arrays[name]).all():
            raise FollowupContractError(f"{name} contains nonfinite residuals")

    with np.load(run / "models" / "log_bias_anchor.npz", allow_pickle=False) as anchor:
        if set(anchor.files) != {"lead_month_residual", "shrinkage", "target_scale"}:
            raise FollowupContractError("log-bias anchor artifact keys changed")
        target_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
    if target_scale.shape != (N_LEADS,) or not np.isfinite(target_scale).all():
        raise FollowupContractError("target scale must be a finite six-lead vector")
    if np.any(target_scale <= 0.0):
        raise FollowupContractError("target scale must be strictly positive")

    members: dict[str, dict[str, np.ndarray]] = {
        "scratch": {},
        "global_pretrained": {},
    }
    ensemble_residuals: dict[str, np.ndarray] = {}
    stored_predictions: dict[str, np.ndarray] = {}
    valid_mask = np.broadcast_to(support, field_shape)
    for configuration in ("scratch", "global_pretrained"):
        residuals = []
        for seed in SEEDS:
            residual = np.asarray(
                np.load(
                    run
                    / "models"
                    / configuration
                    / f"seed_{seed}"
                    / "validation_residual.npy",
                    allow_pickle=False,
                ),
                dtype=np.float32,
            )
            if residual.shape != field_shape or not np.isfinite(residual).all():
                raise FollowupContractError(
                    f"invalid {configuration} seed {seed} validation residual"
                )
            members[configuration][f"seed_{seed}"] = residual
            residuals.append(residual)
        ensemble = np.mean(residuals, axis=0, dtype=np.float64).astype(np.float32)
        stored_ensemble = np.asarray(
            arrays[f"{configuration}_standardized_residual"], dtype=np.float32
        )
        if not np.array_equal(ensemble, stored_ensemble):
            raise FollowupContractError(
                f"stored {configuration} ensemble residual is not the exact "
                "float64 mean of seeds 42, 43, and 44"
            )
        reconstructed = reconstruct_anchored_precipitation(
            arrays["log_bias"],
            ensemble,
            target_scale,
            valid_mask=valid_mask,
            maximum_log_rain=MAXIMUM_LOG_RAIN,
        )
        stored = np.asarray(arrays[configuration], dtype=np.float32)
        if not np.array_equal(reconstructed[..., support], stored[..., support]):
            raise FollowupContractError(
                f"stored {configuration} physical ensemble cannot be reconstructed"
            )
        members[configuration]["ensemble"] = ensemble
        ensemble_residuals[configuration] = ensemble
        stored_predictions[configuration] = stored

    return SourceBundle(
        run=run,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        verified_artifacts=verified,
        initializations=np.asarray(arrays["initializations"], dtype="datetime64[D]"),
        truth=np.asarray(arrays["truth"], dtype=np.float32),
        climatology=np.asarray(arrays["climatology"], dtype=np.float32),
        area_weights=weights,
        raw_fuxi=np.asarray(arrays["raw_fuxi"], dtype=np.float32),
        log_bias=np.asarray(arrays["log_bias"], dtype=np.float32),
        stored_predictions=stored_predictions,
        ensemble_residuals=ensemble_residuals,
        member_residuals=members,
        target_scale=target_scale,
    )


def _weighted_area_mean(
    values: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> float:
    selected_weights = np.asarray(weights, dtype=np.float64)[mask]
    return float(
        np.sum(np.asarray(values, dtype=np.float64)[mask] * selected_weights)
        / np.sum(selected_weights, dtype=np.float64)
    )


def preserve_anchor_area_mean(
    baseline: np.ndarray,
    standardized_residual: np.ndarray,
    target_scale: np.ndarray,
    area_weights: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    mean_closure_tolerance: float = MEAN_CLOSURE_TOLERANCE,
) -> ProjectionResult:
    """Project a residual onto the anchor-area-mean constraint.

    A single log-rainfall offset is solved independently for every case and
    lead.  It is added uniformly to the standardized residual after division
    by that lead's fixed train-only target scale.  No observation or
    climatology enters this function.
    """

    anchor = np.asarray(baseline, dtype=np.float32)
    residual = np.asarray(standardized_residual, dtype=np.float32)
    scale = np.asarray(target_scale, dtype=np.float32)
    weights = np.asarray(area_weights, dtype=np.float64)
    if anchor.ndim != 4 or residual.shape != anchor.shape:
        raise ValueError("baseline and residual must share [case, lead, lat, lon]")
    if scale.shape != (anchor.shape[1],) or not np.isfinite(scale).all():
        raise ValueError("target_scale must be a finite vector matching the lead axis")
    if np.any(scale <= 0.0):
        raise ValueError("target_scale must be strictly positive")
    if weights.shape != anchor.shape[-2:] or not np.isfinite(weights).all():
        raise ValueError("area_weights do not match the spatial grid")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("area_weights must be nonnegative with positive support")
    if not np.isfinite(mean_closure_tolerance) or mean_closure_tolerance <= 0.0:
        raise ValueError("mean_closure_tolerance must be finite and positive")
    if valid_mask is None:
        mask = np.broadcast_to(weights > 0.0, anchor.shape).copy()
    else:
        try:
            mask = np.broadcast_to(
                np.asarray(valid_mask, dtype=bool), anchor.shape
            ).copy()
        except ValueError as exc:
            raise ValueError(
                "valid_mask cannot be broadcast to the rainfall fields"
            ) from exc
        mask &= weights[None, None] > 0.0
    if not np.all(np.any(mask, axis=(-2, -1))):
        raise ValueError("every case and lead must have positive valid support")
    if not np.isfinite(anchor[mask]).all() or np.any(anchor[mask] < 0.0):
        raise ValueError("baseline must be finite and nonnegative on valid support")
    if not np.isfinite(residual[mask]).all():
        raise ValueError("residual must be finite on valid support")

    adjusted = np.zeros_like(residual, dtype=np.float32)
    adjusted[mask] = residual[mask]
    offsets = np.zeros(anchor.shape[:2], dtype=np.float64)
    anchor_means = np.empty(anchor.shape[:2], dtype=np.float64)
    before_means = np.empty(anchor.shape[:2], dtype=np.float64)
    zero_identity = np.zeros(anchor.shape[:2], dtype=bool)
    for case_index in range(anchor.shape[0]):
        for lead_index in range(anchor.shape[1]):
            selected = mask[case_index, lead_index]
            spatial_weights = weights[selected]
            baseline_values = anchor[case_index, lead_index, selected].astype(
                np.float64
            )
            residual_values = residual[case_index, lead_index, selected].astype(
                np.float64
            )
            target_mean = float(
                np.sum(baseline_values * spatial_weights, dtype=np.float64)
                / np.sum(spatial_weights, dtype=np.float64)
            )
            anchor_means[case_index, lead_index] = target_mean
            base_log = np.log1p(baseline_values) + residual_values * float(
                scale[lead_index]
            )
            before_rain = np.expm1(np.clip(base_log, 0.0, MAXIMUM_LOG_RAIN))
            before_means[case_index, lead_index] = float(
                np.sum(before_rain * spatial_weights, dtype=np.float64)
                / np.sum(spatial_weights, dtype=np.float64)
            )
            if np.array_equal(residual_values, np.zeros_like(residual_values)):
                zero_identity[case_index, lead_index] = True
                continue
            scalar, _ = bias_aware.solve_physical_mean_recenter_scalar(
                base_log[None, :], spatial_weights, target_mean
            )
            if float(np.max(base_log + scalar)) > MAXIMUM_LOG_RAIN:
                raise FollowupContractError(
                    "area-mean projection reached the reconstruction overflow guard"
                )
            offsets[case_index, lead_index] = scalar
            adjusted[case_index, lead_index, selected] = (
                residual_values + scalar / float(scale[lead_index])
            ).astype(np.float32)

    prediction = reconstruct_anchored_precipitation(
        anchor,
        adjusted,
        scale,
        valid_mask=mask,
        maximum_log_rain=MAXIMUM_LOG_RAIN,
    )
    if np.any(prediction[mask] < 0.0) or not np.isfinite(prediction[mask]).all():
        raise FollowupContractError("projection produced invalid physical rainfall")
    projected_means = np.empty(anchor.shape[:2], dtype=np.float64)
    for case_index in range(anchor.shape[0]):
        for lead_index in range(anchor.shape[1]):
            selected = mask[case_index, lead_index]
            projected_means[case_index, lead_index] = _weighted_area_mean(
                prediction[case_index, lead_index], weights, selected
            )
            if zero_identity[case_index, lead_index] and not np.array_equal(
                prediction[case_index, lead_index, selected],
                anchor[case_index, lead_index, selected],
            ):
                raise FollowupContractError(
                    "zero neural residual did not preserve the anchor bit-for-bit"
                )
    closure = np.abs(projected_means - anchor_means)
    if float(np.max(closure)) > mean_closure_tolerance:
        raise FollowupContractError(
            "area-mean projection did not close in float32 reconstruction: "
            f"maximum error={float(np.max(closure)):.9g}"
        )
    return ProjectionResult(
        prediction=prediction,
        adjusted_standardized_residual=adjusted,
        log_offset=offsets,
        anchor_mean=anchor_means,
        unprojected_mean=before_means,
        projected_mean=projected_means,
        absolute_closure_error=closure,
        zero_residual_identity=zero_identity,
    )


def year_stratified_circular_block_indices(
    initializations: np.ndarray,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_years: Sequence[int] = EXPECTED_VALIDATION_YEARS,
    expected_cases_per_year: int | None = CASES_PER_YEAR,
) -> BootstrapPlan:
    """Create one shared circular block plan, stratified by year."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size < 1 or np.isnat(dates).any():
        raise ValueError(
            "initializations must be a nonempty one-dimensional date array"
        )
    if len(np.unique(dates)) != len(dates) or not np.array_equal(dates, np.sort(dates)):
        raise ValueError("initializations must be unique and chronological")
    if draws <= 0 or block_length <= 0:
        raise ValueError("bootstrap draws and block_length must be positive")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    expected = tuple(int(year) for year in expected_years)
    if tuple(sorted(np.unique(years).tolist())) != expected:
        raise ValueError("bootstrap initialization years differ from the contract")
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_length, dtype=np.int64)
    segments: list[np.ndarray] = []
    year_slices: dict[int, slice] = {}
    positions_by_year: dict[int, np.ndarray] = {}
    cursor = 0
    for year in expected:
        positions = np.flatnonzero(years == year).astype(np.int64)
        n_year = len(positions)
        if expected_cases_per_year is not None and n_year != expected_cases_per_year:
            raise ValueError(f"unexpected {year} bootstrap case count: {n_year}")
        if block_length > n_year:
            raise ValueError(f"block length {block_length} exceeds {year} case count")
        blocks_per_draw = math.ceil(n_year / block_length)
        starts = rng.integers(0, n_year, size=(draws, blocks_per_draw))
        local = ((starts[:, :, None] + offsets[None, None, :]) % n_year).reshape(
            draws, -1
        )
        local = local[:, :n_year]
        segment = positions[local]
        segments.append(segment)
        positions_by_year[year] = positions
        year_slices[year] = slice(cursor, cursor + n_year)
        cursor += n_year
    indices = np.concatenate(segments, axis=1)
    if indices.shape != (draws, len(dates)):
        raise FollowupContractError("bootstrap index matrix has the wrong shape")
    multiplicity = np.bincount(indices.reshape(-1), minlength=len(dates)).astype(
        np.float64
    ) / float(draws)
    maximum_deviation = float(np.max(np.abs(multiplicity - 1.0)))
    # Six standard-error units is deliberately conservative for small synthetic
    # tests while still detecting the old season-edge under-inclusion at 10k.
    allowed_deviation = max(0.05, 6.0 / math.sqrt(draws))
    if not np.isclose(multiplicity.mean(), 1.0, rtol=0.0, atol=1.0e-12):
        raise FollowupContractError("bootstrap mean date multiplicity is not one")
    if maximum_deviation > allowed_deviation:
        raise FollowupContractError(
            "bootstrap date multiplicity is inconsistent with equal marginal "
            f"inclusion: maximum deviation={maximum_deviation:.6g}, "
            f"allowed={allowed_deviation:.6g}"
        )
    plan = BootstrapPlan(
        indices=indices,
        year_slices=year_slices,
        year_positions=positions_by_year,
        draws=int(draws),
        block_length=int(block_length),
        seed=int(seed),
        mean_multiplicity=multiplicity,
        maximum_absolute_multiplicity_deviation=maximum_deviation,
    )
    validate_bootstrap_plan(plan, initializations)
    return plan


def validate_bootstrap_plan(plan: BootstrapPlan, initializations: np.ndarray) -> None:
    """Prove blocks are modulo-consecutive and stay inside their year."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    if plan.indices.shape != (plan.draws, len(dates)):
        raise FollowupContractError("bootstrap plan shape changed")
    if np.any(plan.indices < 0) or np.any(plan.indices >= len(dates)):
        raise FollowupContractError("bootstrap plan contains an out-of-range case")
    for year, positions in plan.year_positions.items():
        segment = plan.indices[:, plan.year_slices[year]]
        if not np.isin(segment, positions).all() or not np.all(years[segment] == year):
            raise FollowupContractError(f"bootstrap blocks cross the {year} stratum")
        local_lookup = np.full(len(dates), -1, dtype=np.int64)
        local_lookup[positions] = np.arange(len(positions), dtype=np.int64)
        local = local_lookup[segment]
        for start in range(0, local.shape[1], plan.block_length):
            block = local[:, start : min(start + plan.block_length, local.shape[1])]
            if block.shape[1] > 1 and not np.all(
                np.diff(block, axis=1) % len(positions) == 1
            ):
                raise FollowupContractError(
                    f"bootstrap contains a block that is not circular-consecutive in {year}"
                )
    observed_multiplicity = np.bincount(
        plan.indices.reshape(-1), minlength=len(dates)
    ).astype(np.float64) / float(plan.draws)
    if not np.array_equal(observed_multiplicity, plan.mean_multiplicity):
        raise FollowupContractError("bootstrap multiplicity diagnostic changed")


def bootstrap_date_multiplicity(
    plan: BootstrapPlan, initializations: np.ndarray
) -> pd.DataFrame:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.shape != plan.mean_multiplicity.shape:
        raise FollowupContractError("bootstrap multiplicity/date shape changed")
    return pd.DataFrame(
        {
            "initialization": dates,
            "year": pd.DatetimeIndex(dates).year.to_numpy(dtype=int),
            "mean_draw_multiplicity": plan.mean_multiplicity,
            "deviation_from_one": plan.mean_multiplicity - 1.0,
        }
    )


def _case_metrics(
    bundle: SourceBundle,
    prediction: np.ndarray,
    configuration: str,
    member: str,
) -> pd.DataFrame:
    support = bundle.area_weights > 0.0
    valid_mask = np.broadcast_to(support, bundle.truth.shape)
    metrics = compute_case_metrics(
        bundle.truth,
        prediction,
        bundle.truth - bundle.climatology,
        prediction - bundle.climatology,
        bundle.area_weights,
        predictor=f"{configuration}_{member}",
        case_ids=bundle.initializations,
        leads=np.arange(1, N_LEADS + 1),
        valid_mask=valid_mask,
    )
    metrics.insert(0, "member", member)
    metrics.insert(0, "configuration", configuration)
    return metrics


def summarize_case_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = case_metrics.copy()
    frame["case_id"] = pd.to_datetime(frame["case_id"])
    frame["year"] = frame.case_id.dt.year
    rows: list[dict[str, Any]] = []

    def append_scope(
        group: pd.DataFrame,
        configuration: str,
        member: str,
        scope_type: str,
        scope: str,
    ) -> None:
        rows.append(
            {
                "configuration": configuration,
                "member": member,
                "scope_type": scope_type,
                "scope": scope,
                "initializations": int(group.case_id.nunique()),
                "case_leads": int(len(group)),
                **{
                    metric: float(group[metric].mean())
                    for metric in ("rmse", "mae", "bias", "acc")
                },
            }
        )

    for (configuration, member), group in frame.groupby(
        ["configuration", "member"], sort=True
    ):
        append_scope(group, configuration, member, "pooled", "W1-W6")
        for year in EXPECTED_VALIDATION_YEARS:
            append_scope(
                group.loc[group.year.eq(year)],
                configuration,
                member,
                "year",
                str(year),
            )
        for lead in range(1, N_LEADS + 1):
            append_scope(
                group.loc[group.lead.eq(lead)],
                configuration,
                member,
                "lead",
                f"W{lead}",
            )
    return pd.DataFrame(rows).sort_values(
        ["configuration", "member", "scope_type", "scope"]
    )


def _metric_cube(
    case_metrics: pd.DataFrame,
    configuration: str,
    member: str,
    metric: str,
    initializations: np.ndarray,
) -> np.ndarray:
    selected = case_metrics.loc[
        case_metrics.configuration.eq(configuration)
        & case_metrics.member.eq(member)
        & case_metrics.region.eq("india")
        & case_metrics.season.eq("ALL")
    ].copy()
    selected["case_id"] = pd.to_datetime(selected.case_id)
    if selected.duplicated(["case_id", "lead"]).any():
        raise FollowupContractError(
            f"duplicate metric rows for {configuration}/{member}"
        )
    pivot = selected.pivot(index="case_id", columns="lead", values=metric)
    order = pd.DatetimeIndex(np.asarray(initializations, dtype="datetime64[D]"))
    try:
        values = pivot.loc[order, list(range(1, N_LEADS + 1))].to_numpy(
            dtype=np.float64
        )
    except KeyError as exc:
        raise FollowupContractError(
            f"incomplete metric cube for {configuration}/{member}/{metric}"
        ) from exc
    if values.shape != (len(initializations), N_LEADS) or not np.isfinite(values).all():
        raise FollowupContractError(
            f"invalid metric cube for {configuration}/{member}/{metric}"
        )
    return values


def _favourable_effect(
    metric: str, candidate_mean: np.ndarray, reference_mean: np.ndarray
) -> np.ndarray:
    if metric in ("rmse", "mae"):
        return reference_mean - candidate_mean
    if metric == "acc":
        return candidate_mean - reference_mean
    if metric == "bias":
        return np.abs(reference_mean) - np.abs(candidate_mean)
    raise ValueError(metric)


def paired_bootstrap_effects(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    plan: BootstrapPlan,
    comparisons: Iterable[ComparisonSpec],
) -> pd.DataFrame:
    """Return paired favourable-direction effects using one shared case plan."""

    scopes: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        (
            "pooled",
            "W1-W6",
            np.arange(len(initializations), dtype=np.int64),
            np.arange(N_LEADS, dtype=np.int64),
        )
    ]
    scopes.extend(
        (
            "lead",
            f"W{lead + 1}",
            np.arange(len(initializations), dtype=np.int64),
            np.asarray([lead], dtype=np.int64),
        )
        for lead in range(N_LEADS)
    )
    scopes.extend(
        (
            "year",
            str(year),
            plan.year_positions[year],
            np.arange(N_LEADS, dtype=np.int64),
        )
        for year in EXPECTED_VALIDATION_YEARS
    )
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric in ("rmse", "mae", "bias", "acc"):
            candidate = _metric_cube(
                case_metrics,
                comparison.candidate,
                comparison.candidate_member,
                metric,
                initializations,
            )
            reference = _metric_cube(
                case_metrics,
                comparison.reference,
                comparison.reference_member,
                metric,
                initializations,
            )
            for scope_type, scope, observed_positions, lead_positions in scopes:
                if scope_type == "year":
                    sampled = plan.indices[:, plan.year_slices[int(scope)]]
                else:
                    sampled = plan.indices
                candidate_observed = candidate[
                    np.ix_(observed_positions, lead_positions)
                ]
                reference_observed = reference[
                    np.ix_(observed_positions, lead_positions)
                ]
                candidate_mean = float(candidate_observed.mean())
                reference_mean = float(reference_observed.mean())
                observed_effect = float(
                    _favourable_effect(metric, candidate_mean, reference_mean)
                )
                candidate_draw = candidate[sampled][:, :, lead_positions].mean(
                    axis=(1, 2)
                )
                reference_draw = reference[sampled][:, :, lead_positions].mean(
                    axis=(1, 2)
                )
                effects = _favourable_effect(metric, candidate_draw, reference_draw)
                lower, upper = np.quantile(effects, (0.025, 0.975))
                if metric in ("rmse", "mae"):
                    relative = 100.0 * observed_effect / reference_mean
                    relative_draws = 100.0 * effects / reference_draw
                    relative_lower, relative_upper = np.quantile(
                        relative_draws, (0.025, 0.975)
                    )
                else:
                    relative = np.nan
                    relative_lower = np.nan
                    relative_upper = np.nan
                rows.append(
                    {
                        "candidate": comparison.candidate,
                        "candidate_member": comparison.candidate_member,
                        "reference": comparison.reference,
                        "reference_member": comparison.reference_member,
                        "scope_type": scope_type,
                        "scope": scope,
                        "metric": metric,
                        "effect_definition": (
                            "positive_favours_candidate; reference_minus_candidate"
                            if metric in ("rmse", "mae")
                            else (
                                "positive_favours_candidate; candidate_minus_reference"
                                if metric == "acc"
                                else "positive_favours_candidate; absolute_reference_bias_minus_absolute_candidate_bias"
                            )
                        ),
                        "candidate_mean": candidate_mean,
                        "reference_mean": reference_mean,
                        "effect": observed_effect,
                        "bootstrap_mean_effect": float(np.mean(effects)),
                        "ci_lower_2p5": float(lower),
                        "ci_upper_97p5": float(upper),
                        "probability_favourable": float(np.mean(effects > 0.0)),
                        "relative_effect_pct": float(relative),
                        "relative_ci_lower_2p5": float(relative_lower),
                        "relative_ci_upper_97p5": float(relative_upper),
                        "initializations": int(len(observed_positions)),
                        "case_leads": int(
                            len(observed_positions) * len(lead_positions)
                        ),
                        "bootstrap_draws": plan.draws,
                        "block_length_initializations": plan.block_length,
                        "bootstrap_seed": plan.seed,
                        "bootstrap_method": (
                            "paired year-stratified circular moving blocks; "
                            "one shared initialization draw matrix retains all six leads"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _comparison_specs() -> tuple[ComparisonSpec, ...]:
    matched_pairs = (
        ("global_pretrained", "scratch"),
        ("scratch_anchor_mean_preserved", "scratch"),
        ("global_pretrained_anchor_mean_preserved", "global_pretrained"),
        (
            "global_pretrained_anchor_mean_preserved",
            "scratch_anchor_mean_preserved",
        ),
    )
    members = tuple(f"seed_{seed}" for seed in SEEDS) + ("ensemble",)
    specs = [
        ComparisonSpec(candidate, reference, member, member)
        for candidate, reference in matched_pairs
        for member in members
    ]
    specs.extend(
        (
            ComparisonSpec(
                "scratch_anchor_mean_preserved",
                "log_bias",
                "ensemble",
                "deterministic",
            ),
            ComparisonSpec(
                "global_pretrained_anchor_mean_preserved",
                "log_bias",
                "ensemble",
                "deterministic",
            ),
        )
    )
    return tuple(specs)


def _summary_row(
    summary: pd.DataFrame,
    configuration: str,
    member: str,
    scope_type: str,
    scope: str,
) -> pd.Series:
    selected = summary.loc[
        summary.configuration.eq(configuration)
        & summary.member.eq(member)
        & summary.scope_type.eq(scope_type)
        & summary.scope.eq(scope)
    ]
    if len(selected) != 1:
        raise FollowupContractError(
            f"expected one summary row for {configuration}/{member}/{scope_type}/{scope}"
        )
    return selected.iloc[0]


def exploratory_projection_gate(
    summary: pd.DataFrame, projection_diagnostics: pd.DataFrame
) -> Mapping[str, Any]:
    results: dict[str, Any] = {}
    for configuration in ("scratch", "global_pretrained"):
        projected_name = f"{configuration}_anchor_mean_preserved"
        original = _summary_row(summary, configuration, "ensemble", "pooled", "W1-W6")
        projected = _summary_row(summary, projected_name, "ensemble", "pooled", "W1-W6")
        seed_wins = 0
        for seed in SEEDS:
            member = f"seed_{seed}"
            seed_original = _summary_row(
                summary, configuration, member, "pooled", "W1-W6"
            )
            seed_projected = _summary_row(
                summary, projected_name, member, "pooled", "W1-W6"
            )
            seed_wins += int(seed_projected.rmse < seed_original.rmse)
        year_wins = sum(
            _summary_row(summary, projected_name, "ensemble", "year", str(year)).rmse
            < _summary_row(summary, configuration, "ensemble", "year", str(year)).rmse
            for year in EXPECTED_VALIDATION_YEARS
        )
        lead_wins = sum(
            _summary_row(summary, projected_name, "ensemble", "lead", f"W{lead}").rmse
            < _summary_row(summary, configuration, "ensemble", "lead", f"W{lead}").rmse
            for lead in range(1, N_LEADS + 1)
        )
        diagnostics = projection_diagnostics.loc[
            projection_diagnostics.configuration.eq(projected_name)
        ]
        conditions = {
            "mean_closure_verified": bool(
                diagnostics.absolute_closure_error.max() <= MEAN_CLOSURE_TOLERANCE
            ),
            "nonnegative_verified": bool(diagnostics.minimum_prediction.min() >= 0.0),
            "pooled_rmse_improves": bool(projected.rmse < original.rmse),
            "rmse_improves_in_both_years": bool(year_wins == 2),
            "at_least_four_leads_improve_rmse": bool(lead_wins >= 4),
            "at_least_two_seeds_improve_rmse": bool(seed_wins >= 2),
            "pooled_mae_not_worse": bool(projected.mae <= original.mae),
            "pooled_acc_within_guard": bool(projected.acc - original.acc >= -0.002),
        }
        results[configuration] = {
            "passes_exploratory_guards": bool(all(conditions.values())),
            "conditions": conditions,
            "diagnostics": {
                "pooled_rmse_change_projected_minus_original": float(
                    projected.rmse - original.rmse
                ),
                "pooled_mae_change_projected_minus_original": float(
                    projected.mae - original.mae
                ),
                "pooled_bias_change_projected_minus_original": float(
                    projected.bias - original.bias
                ),
                "pooled_acc_change_projected_minus_original": float(
                    projected.acc - original.acc
                ),
                "years_improving_rmse": int(year_wins),
                "leads_improving_rmse": int(lead_wins),
                "seeds_improving_rmse": int(seed_wins),
            },
        }
    return {
        "status": "posthoc_2018_2019_development_diagnostic",
        "scientific_eligible_for_source_promotion": False,
        "disposition": "hypothesis_only_requires_predeclared_later_confirmation",
        "results": results,
    }


def _projection_diagnostics(
    result: ProjectionResult,
    configuration: str,
    member: str,
    initializations: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for case_index, initialization in enumerate(initializations):
        for lead_index in range(N_LEADS):
            rows.append(
                {
                    "configuration": configuration,
                    "member": member,
                    "initialization": np.datetime_as_string(initialization, unit="D"),
                    "lead": lead_index + 1,
                    "log_offset": float(result.log_offset[case_index, lead_index]),
                    "anchor_area_mean": float(
                        result.anchor_mean[case_index, lead_index]
                    ),
                    "unprojected_area_mean": float(
                        result.unprojected_mean[case_index, lead_index]
                    ),
                    "projected_area_mean": float(
                        result.projected_mean[case_index, lead_index]
                    ),
                    "absolute_closure_error": float(
                        result.absolute_closure_error[case_index, lead_index]
                    ),
                    "zero_residual_identity": bool(
                        result.zero_residual_identity[case_index, lead_index]
                    ),
                    "minimum_prediction": float(
                        np.min(result.prediction[case_index, lead_index])
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    source_root: Path,
    output: Path,
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    noncanonical_smoke: bool = False,
) -> Path:
    """Run and atomically publish the artifact-only post-hoc diagnostics."""

    started = time.monotonic()
    destination = Path(output).expanduser().resolve()
    canonical_settings = (
        bootstrap_draws == DEFAULT_BOOTSTRAP_DRAWS
        and block_length == DEFAULT_BLOCK_LENGTH
        and bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
    )
    canonical_artifact = canonical_settings and not noncanonical_smoke
    if not canonical_settings and not noncanonical_smoke:
        raise FollowupContractError(
            "alternate bootstrap settings require explicit --noncanonical-smoke"
        )
    if noncanonical_smoke and not any(
        label in destination.name.lower() for label in ("noncanonical", "smoke")
    ):
        raise FollowupContractError(
            "noncanonical smoke output name must contain 'noncanonical' or 'smoke'"
        )
    bundle = load_source_bundle(source_root)
    if destination.exists():
        raise FileExistsError(f"fresh output directory required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    for directory in (
        staging,
        staging / "metrics",
        staging / "predictions",
        staging / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    try:
        support = bundle.area_weights > 0.0
        valid_mask = np.broadcast_to(support, bundle.truth.shape)
        case_frames = [
            _case_metrics(bundle, bundle.raw_fuxi, "raw_fuxi", "deterministic"),
            _case_metrics(bundle, bundle.log_bias, "log_bias", "deterministic"),
        ]
        diagnostics_frames = []
        derived_arrays: dict[str, np.ndarray] = {
            "initializations": bundle.initializations,
            "area_weights": bundle.area_weights,
        }
        for configuration in ("scratch", "global_pretrained"):
            projected_configuration = f"{configuration}_anchor_mean_preserved"
            for member, residual in bundle.member_residuals[configuration].items():
                original = reconstruct_anchored_precipitation(
                    bundle.log_bias,
                    residual,
                    bundle.target_scale,
                    valid_mask=valid_mask,
                    maximum_log_rain=MAXIMUM_LOG_RAIN,
                )
                if member == "ensemble" and not np.array_equal(
                    original[..., support],
                    bundle.stored_predictions[configuration][..., support],
                ):
                    raise FollowupContractError(
                        f"{configuration} ensemble changed after source validation"
                    )
                case_frames.append(
                    _case_metrics(bundle, original, configuration, member)
                )
                projected = preserve_anchor_area_mean(
                    bundle.log_bias,
                    residual,
                    bundle.target_scale,
                    bundle.area_weights,
                    valid_mask=valid_mask,
                )
                case_frames.append(
                    _case_metrics(
                        bundle,
                        projected.prediction,
                        projected_configuration,
                        member,
                    )
                )
                diagnostics_frames.append(
                    _projection_diagnostics(
                        projected,
                        projected_configuration,
                        member,
                        bundle.initializations,
                    )
                )
                key = f"{configuration}_{member}"
                stored_prediction = projected.prediction.copy()
                stored_prediction[..., ~support] = np.nan
                derived_arrays[f"{key}_prediction"] = stored_prediction
                derived_arrays[f"{key}_adjusted_standardized_residual"] = (
                    projected.adjusted_standardized_residual
                )
                derived_arrays[f"{key}_log_offset"] = projected.log_offset

        case_metrics = pd.concat(case_frames, ignore_index=True)
        projection_diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
        summary = summarize_case_metrics(case_metrics)
        plan = year_stratified_circular_block_indices(
            bundle.initializations,
            draws=bootstrap_draws,
            block_length=block_length,
            seed=bootstrap_seed,
        )
        bootstrap = paired_bootstrap_effects(
            case_metrics,
            bundle.initializations,
            plan,
            _comparison_specs(),
        )
        gate = exploratory_projection_gate(summary, projection_diagnostics)
        multiplicity = bootstrap_date_multiplicity(plan, bundle.initializations)

        case_metrics.to_csv(
            staging / "metrics" / "validation_case_metrics.csv", index=False
        )
        summary.to_csv(staging / "metrics" / "validation_summary.csv", index=False)
        projection_diagnostics.to_csv(
            staging / "metrics" / "area_mean_projection_diagnostics.csv",
            index=False,
        )
        bootstrap.to_csv(
            staging / "metrics" / "paired_year_stratified_circular_block_bootstrap.csv",
            index=False,
        )
        multiplicity.to_csv(
            staging / "metrics" / "bootstrap_date_multiplicity.csv", index=False
        )
        np.save(staging / "metrics" / "bootstrap_indices.npy", plan.indices)
        np.savez_compressed(
            staging / "predictions" / "anchor_mean_preserved_outputs.npz",
            **derived_arrays,
        )
        atomic_write_json(
            staging / "metrics" / "exploratory_projection_gate.json", gate
        )
        atomic_write_json(
            staging / "source_verification.json",
            {
                "source_run": str(bundle.run),
                "source_manifest_sha256": bundle.manifest_sha256,
                "verified_artifact_count": len(bundle.verified_artifacts),
                "verified_artifacts": bundle.verified_artifacts,
                "source_tree_modified": False,
                "raw_archives_opened": False,
                "later_year_data_opened": False,
            },
        )

        source_files = {
            "evaluate_global_pretraining_followups": Path(__file__).resolve(),
            "fuxi_imd_bias_aware_validation_sweep": Path(bias_aware.__file__).resolve(),
            "fuxi_adapter_anchored": Path(anchored_module.__file__).resolve(),
            "fuxi_adapter_metrics": Path(metrics_module.__file__).resolve(),
        }
        code_sources: dict[str, Mapping[str, str]] = {}
        for label, source in source_files.items():
            snapshot = staging / "code" / f"{label}.py"
            shutil.copy2(source, snapshot)
            code_sources[label] = {
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "snapshot_path": str(snapshot.relative_to(staging)),
                "snapshot_sha256": sha256_file(snapshot),
            }

        artifacts = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema_version": 1,
            "status": (
                "complete_posthoc_validation_only"
                if canonical_artifact
                else "complete_posthoc_validation_only_noncanonical_smoke"
            ),
            "created_utc": utc_now(),
            "canonical_artifact": canonical_artifact,
            "execution_tier": (
                "canonical_10000draw_block13_seed20260822"
                if canonical_artifact
                else "explicit_noncanonical_smoke"
            ),
            "scientific_eligible": False,
            "evidence_role": "posthoc_2018_2019_development_diagnostic",
            "source_promotion_decision_modified": False,
            "test_predictions_created": False,
            "raw_archives_opened": False,
            "later_year_predictions_created": False,
            "later_year_metrics_computed": False,
            "source": {
                "run": str(bundle.run),
                "manifest_sha256": bundle.manifest_sha256,
                "verified_artifact_count": len(bundle.verified_artifacts),
            },
            "data": {
                "initialization_years": list(EXPECTED_VALIDATION_YEARS),
                "initialization_count": N_CASES,
                "cases_per_year": CASES_PER_YEAR,
                "leads": N_LEADS,
                "grid_shape": list(GRID_SHAPE),
                "support_cells": SUPPORT_CELLS,
                "sealed_years": list(SEALED_YEARS),
                "sealed_years_opened": False,
            },
            "projection": {
                "name": "anchor_physical_area_mean_log_offset",
                "fit_target": "none; target mean is the forecast-time log-bias anchor",
                "offset_scope": "one spatially uniform log-rain offset per initialization and lead",
                "nonnegative_reconstruction": True,
                "exact_zero_residual_identity": True,
                "mean_closure_tolerance_mm_day": MEAN_CLOSURE_TOLERANCE,
                "maximum_observed_closure_error_mm_day": float(
                    projection_diagnostics.absolute_closure_error.max()
                ),
            },
            "bootstrap": {
                "method": "paired year-stratified circular moving blocks",
                "draws": plan.draws,
                "block_length_initializations": plan.block_length,
                "seed": plan.seed,
                "same_draw_matrix_all_models_seeds_leads_metrics": True,
                "all_six_leads_retained_per_initialization": True,
                "year_boundary_crossed": False,
                "equal_marginal_date_inclusion": True,
                "mean_date_multiplicity": float(plan.mean_multiplicity.mean()),
                "minimum_date_multiplicity": float(plan.mean_multiplicity.min()),
                "maximum_date_multiplicity": float(plan.mean_multiplicity.max()),
                "maximum_absolute_multiplicity_deviation": (
                    plan.maximum_absolute_multiplicity_deviation
                ),
                "interpretation": (
                    "descriptive percentile intervals conditional on the selected "
                    "2018 and 2019 development seasons; not p-values"
                ),
            },
            "exploratory_gate": gate,
            "elapsed_seconds": float(time.monotonic() - started),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "execution_environment": {
                "node": platform.node(),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "code_sources": code_sources,
            "artifacts": artifacts,
        }
        atomic_write_json(staging / "manifest.json", _json_safe(manifest))
        os.replace(staging, destination)
        print(destination, flush=True)
        return destination
    except Exception:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "failed_utc": utc_now(),
            "source_run": str(bundle.run),
            "traceback": traceback.format_exc(),
            "later_year_data_opened": False,
        }
        if staging.exists():
            atomic_write_json(staging / "failure.json", failure)
            failed = destination.parent / f"{destination.name}.failed"
            if failed.exists():
                failed = destination.parent / f"{destination.name}.failed-{os.getpid()}"
            os.replace(staging, failed)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="completed comparison run or parent containing exactly one full run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="fresh atomic output directory",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--noncanonical-smoke",
        action="store_true",
        help=(
            "explicitly label an ad-hoc/synthetic run as noncanonical; required "
            "when draws, block length, or seed differ from the scheduled contract"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.bootstrap_draws <= 0:
        raise ValueError("bootstrap-draws must be positive")
    if arguments.block_length <= 0:
        raise ValueError("block-length must be positive")
    if arguments.bootstrap_seed < 0:
        raise ValueError("bootstrap-seed must be nonnegative")
    run_analysis(
        arguments.source_root,
        arguments.output,
        bootstrap_draws=arguments.bootstrap_draws,
        block_length=arguments.block_length,
        bootstrap_seed=arguments.bootstrap_seed,
        noncanonical_smoke=arguments.noncanonical_smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
