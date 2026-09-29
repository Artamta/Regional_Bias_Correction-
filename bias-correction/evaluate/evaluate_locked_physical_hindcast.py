#!/usr/bin/env python3
"""Freeze a validation-only decision and evaluate it once on 2020--2021.

This module deliberately separates model selection from held-out evaluation:

``freeze``
    Reads only a completed 2018--2019 physical validation sweep and writes an
    immutable selection document.  It never loads forecasts or observations.

``evaluate``
    Requires that selection document, verifies every selected checkpoint and
    frozen preprocessing artifact, and only then opens the 2020--2021 data.
    No parameter, normalization statistic, climatological correction, or model
    choice is fitted on the held-out period.

The 2020--2021 period is labelled *exploratory/reused* throughout.  It is not
presented as a fresh independent confirmation set.
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
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch


EVALUATE_ROOT = Path(__file__).resolve().parent
HERE = EVALUATE_ROOT.parent
SOURCE_ROOT = HERE / "src"
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for source in (SOURCE_ROOT, EVALUATE_ROOT, NEURAL_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import fuxi_imd_attention_climatology as experiment  # noqa: E402
import fuxi_imd_compact_validation_sweep as sweep  # noqa: E402
from fuxi_adapter.anchored import reconstruct_anchored_precipitation  # noqa: E402
from fuxi_adapter.baselines import (  # noqa: E402
    LogBiasCorrection,
    apply_log_bias_correction,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.training import predict  # noqa: E402
from plot_physical_validation_results import (  # noqa: E402
    DEFAULT_INDIA_BOUNDARY,
    load_india_boundary,
)


SCHEMA_NAME = "fuxi_imd_locked_hindcast_evaluation"
SCHEMA_VERSION = 1
SELECTION_SCHEMA_NAME = "fuxi_imd_frozen_validation_selection"
SELECTION_SCHEMA_VERSION = 1
TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
TEST_YEARS = (2020, 2021)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_CONFIGURATIONS = (
    "physical_control",
    "physical_tcwv",
    "physical_moisture_circulation",
    "physical_full_compact",
)
EXPECTED_REFERENCE = "physical_control"
METHODS = ("raw_fuxi", "log_bias", "corrected")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "log_bias": "Training-only log-bias",
    "corrected": "Frozen neural correction (3-seed)",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#0072B2",
    "corrected": "#D55E00",
}
PHYSICAL_NAMES = tuple(sweep.PHYSICAL_PREDICTOR_NAMES)
EXPECTED_LATITUDE = np.arange(39.0, -0.01, -1.5, dtype=np.float64)
EXPECTED_LONGITUDE = np.arange(60.0, 99.01, 1.5, dtype=np.float64)
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BLOCK_LENGTH = 13
DEFAULT_BOOTSTRAP_SEED = 20260812
DEFAULT_FDR_Q = 0.05
# The completed three-seed confirmation explicitly overrode the sweep default
# with ``--minimum-loss-improvement 0.001``.  This is part of the frozen
# predeclared selection rule and must match the recorded run, not the CLI
# default used by unrelated sweeps.
EXPECTED_MINIMUM_LOSS_IMPROVEMENT = 0.001
EXPECTED_LOG_BIAS_SHRINKAGE = 10.0
EXPECTED_SURVEY_OF_INDIA_SOURCE_SHA256 = (
    "2b786fe7338f2b1b6d0d0721aeada910c027dad4a237a58d905a2f77586a90b8"
)
EXPECTED_POSTSELECTION_PHYSICAL_CACHE = (
    HERE / "cache" / "fuxi_physical_weekly_2020_2021_jjas_exploratory_v2.npz"
).resolve()
EXPECTED_POSTSELECTION_PHYSICAL_CACHE_SHA256 = (
    "1474b504bd72d155ae5876c34e7f376a1ed3e5880c8da5ceb6ca9f49458f764b"
)
EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_NAME = (
    "fuxi-physical-weekly-jjas-exploratory"
)
EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_VERSION = 2
EVALUATION_ROLE = "exploratory_reused_hindcast_evaluation"


class EvaluationContractError(ValueError):
    """Raised when the frozen-selection or held-out-data contract changes."""


@dataclass(frozen=True)
class FrozenTestData:
    """All arrays needed after the selection gate has been passed."""

    features: np.ndarray
    raw_fuxi: np.ndarray
    log_bias: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    weights: np.ndarray
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    target_scale: np.ndarray
    source_files: tuple[str, ...]
    observation_stores: tuple[str, ...]
    spatial_store: str
    physical_cache_path: str
    physical_cache_sha256: str


@dataclass(frozen=True)
class ForecastOnlyPreflight:
    """Forecast-only artifacts validated before any held-out IMD target is opened."""

    forecast: Any
    test_forecast: Any
    test_indices: np.ndarray
    t2m_weekly: np.ndarray
    physical_predictors: Any
    normalization: Mapping[str, Any]
    lead_month_residual: np.ndarray
    shrinkage: float
    target_scale: np.ndarray
    models: tuple[torch.nn.Module, ...]
    device: str
    boundary_segments: tuple[np.ndarray, ...]
    boundary_provenance: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    """Publish text atomically in the destination directory."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish one CSV atomically."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Publish one NPY atomically without NumPy changing the filename."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path: Path, **values: Any) -> None:
    """Publish one compressed NPZ atomically."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy2(source: Path, destination: Path) -> None:
    """Copy one provenance source and publish it atomically."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".temporary", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise EvaluationContractError(f"JSON root must be an object: {path}")
    return payload


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin(("true", "false")).all():
        raise EvaluationContractError("ranking qualifies column is not boolean")
    return lowered.eq("true")


def canonical_candidate_payload(name: str) -> Mapping[str, Any]:
    """Return the exact predeclared candidate definition used by this evaluator."""

    if name not in EXPECTED_CONFIGURATIONS:
        raise EvaluationContractError(f"unexpected physical candidate: {name!r}")
    return _json_safe(asdict(sweep.CANDIDATE_BY_NAME[name]))


def frozen_live_code_contract(root: Path) -> tuple[tuple[Path, Path], ...]:
    """Map training-time source snapshots to the live inference implementation."""

    return (
        (
            root / "code" / "fuxi_imd_compact_validation_sweep.py",
            SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
        ),
        (
            root / "code" / "validation_sweep_models.py",
            NEURAL_SRC / "fuxi_adapter" / "validation_sweep_models.py",
        ),
        (
            root / "code" / "v3_training.py",
            NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
        ),
        (
            root / "code" / "anchored.py",
            NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        ),
        (
            root / "code" / "fuxi_imd_attention_climatology.py",
            SOURCE_ROOT / "fuxi_imd_attention_climatology.py",
        ),
        (
            root / "code" / "fuxi_physical_feature_cache.py",
            SOURCE_ROOT / "fuxi_physical_feature_cache.py",
        ),
    )


def validate_frozen_live_code(root: Path, manifest: Mapping[str, Any]) -> None:
    """Reject semantic implementation drift despite state-dict compatibility."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationContractError("confirmation manifest lacks artifact hashes")
    for frozen, live in frozen_live_code_contract(root):
        if not frozen.is_file() or not live.is_file():
            raise FileNotFoundError(frozen if not frozen.is_file() else live)
        relative = str(frozen.relative_to(root))
        frozen_hash = sha256_file(frozen)
        if artifacts.get(relative) != frozen_hash:
            raise EvaluationContractError(
                f"training-time code artifact hash changed: {relative}"
            )
        if sha256_file(live) != frozen_hash:
            raise EvaluationContractError(
                f"live implementation differs from training-time snapshot: {live}"
            )


def validate_completed_confirmation_sweep(sweep_directory: Path) -> Mapping[str, Any]:
    """Validate the physical confirmation without opening any forecast target."""

    root = Path(sweep_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise EvaluationContractError("confirmation sweep is not complete")
    if bool(manifest.get("smoke")):
        raise EvaluationContractError("a smoke run cannot be frozen for test evaluation")
    if manifest.get("training_mode") != "full":
        raise EvaluationContractError("confirmation must be the full three-seed run")
    if manifest.get("test_predictions_created") is not False:
        raise EvaluationContractError("source sweep already accessed the held-out test")
    if tuple(manifest.get("train_years", ())) != TRAIN_YEARS:
        raise EvaluationContractError("training years differ from 2002--2017")
    if tuple(manifest.get("validation_years", ())) != VALIDATION_YEARS:
        raise EvaluationContractError("validation years differ from 2018--2019")
    if manifest.get("split_counts") != {
        "train": 560,
        "validation": 70,
        "test": 70,
    }:
        raise EvaluationContractError("source split counts differ")
    if tuple(int(seed) for seed in manifest.get("seeds", ())) != EXPECTED_SEEDS:
        raise EvaluationContractError("confirmation must contain seeds 42, 43, and 44")
    if manifest.get("reference_configuration") != EXPECTED_REFERENCE:
        raise EvaluationContractError("physical-control reference changed")
    if manifest.get("loss_coefficients") != sweep.LOSS_COEFFICIENTS:
        raise EvaluationContractError("confirmation loss coefficients changed")
    if not np.array_equal(
        np.asarray(manifest.get("lead_weights"), dtype=np.float64),
        np.asarray(sweep.LEAD_WEIGHTS, dtype=np.float64),
    ):
        raise EvaluationContractError("confirmation lead weights changed")
    if not np.isclose(
        float(manifest.get("minimum_loss_improvement", np.nan)),
        EXPECTED_MINIMUM_LOSS_IMPROVEMENT,
        rtol=0.0,
        atol=0.0,
    ):
        raise EvaluationContractError("confirmation selection threshold changed")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise EvaluationContractError("source manifest lacks candidate definitions")
    candidate_names = tuple(str(item.get("name")) for item in candidates)
    if candidate_names != EXPECTED_CONFIGURATIONS:
        raise EvaluationContractError(
            "confirmation candidates/order differ from the frozen four-model screen"
        )
    for item in candidates:
        name = str(item.get("name"))
        if _json_safe(item) != canonical_candidate_payload(name):
            raise EvaluationContractError(
                f"confirmation candidate definition changed: {name}"
            )
    if manifest.get("physical_predictors_loaded") is not True:
        raise EvaluationContractError("source run did not load physical predictors")
    for relative in (
        "normalization.json",
        "models/log_bias_anchor.npz",
        "metrics/ranked_configurations.csv",
        "metrics/validation_case_metrics.csv",
    ):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        recorded = manifest.get("artifacts", {}).get(relative)
        if recorded != sha256_file(path):
            raise EvaluationContractError(f"source artifact hash changed: {relative}")
    validate_frozen_live_code(root, manifest)
    return manifest


def freeze_validation_selection(
    sweep_directory: Path,
    output_path: Path,
    *,
    expected_configuration: str | None = None,
) -> Path:
    """Write a decision fixed entirely by blocked-validation evidence."""

    root = Path(sweep_directory).expanduser().resolve()
    manifest = validate_completed_confirmation_sweep(root)
    ranking_path = root / "metrics" / "ranked_configurations.csv"
    ranking = pd.read_csv(ranking_path)
    required = {"configuration", "rank", "qualifies"}
    if not required.issubset(ranking.columns):
        raise EvaluationContractError(
            f"ranking lacks columns: {sorted(required - set(ranking.columns))}"
        )
    if tuple(ranking.configuration) != tuple(
        ranking.sort_values("rank").configuration
    ):
        ranking = ranking.sort_values("rank").reset_index(drop=True)
    if set(ranking.configuration) != set(EXPECTED_CONFIGURATIONS):
        raise EvaluationContractError("ranking configurations differ from confirmation")
    if ranking.configuration.duplicated().any() or set(ranking["rank"]) != set(
        range(1, len(EXPECTED_CONFIGURATIONS) + 1)
    ):
        raise EvaluationContractError("ranking must contain unique ranks 1--4")
    qualifies = _as_bool(ranking.qualifies)
    qualified = ranking.loc[qualifies].sort_values("rank")
    if len(qualified):
        selected = str(qualified.iloc[0].configuration)
        decision = "top validation-ranked configuration passing all predeclared guards"
    else:
        selected = EXPECTED_REFERENCE
        decision = "no candidate passed all guards; retain predeclared compact control"
    if expected_configuration is not None and selected != expected_configuration:
        raise EvaluationContractError(
            f"automatic validation decision is {selected!r}, not asserted "
            f"{expected_configuration!r}"
        )
    candidate = next(
        item for item in manifest["candidates"] if item.get("name") == selected
    )
    if _json_safe(candidate) != canonical_candidate_payload(selected):
        raise EvaluationContractError("selected candidate is not the canonical definition")
    checkpoints = []
    for seed in EXPECTED_SEEDS:
        record_path = root / "models" / selected / f"seed_{seed}" / "run_record.json"
        record = _read_json(record_path)
        if record.get("status") != "complete" or int(record.get("seed", -1)) != seed:
            raise EvaluationContractError(f"invalid selected run record: {record_path}")
        if _json_safe(record.get("candidate")) != canonical_candidate_payload(selected):
            raise EvaluationContractError(
                "run-record candidate differs from the canonical selection"
            )
        checkpoint = root / str(record.get("checkpoint"))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checksum = sha256_file(checkpoint)
        if checksum != record.get("checkpoint_sha256"):
            raise EvaluationContractError(f"checkpoint hash differs: {checkpoint}")
        checkpoints.append(
            {
                "seed": seed,
                "relative_path": str(checkpoint.relative_to(root)),
                "sha256": checksum,
                "run_record_relative_path": str(record_path.relative_to(root)),
                "run_record_sha256": sha256_file(record_path),
                "best_epoch_zero_based": int(record["best_epoch_zero_based"]),
                "best_validation_loss": float(record["best_validation_loss"]),
            }
        )
    selection = {
        "schema_name": SELECTION_SCHEMA_NAME,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "created_utc": utc_now(),
        "source_sweep": str(root),
        "source_sweep_manifest_sha256": sha256_file(root / "manifest.json"),
        "ranking_relative_path": str(ranking_path.relative_to(root)),
        "ranking_sha256": sha256_file(ranking_path),
        "normalization_relative_path": "normalization.json",
        "normalization_sha256": sha256_file(root / "normalization.json"),
        "anchor_relative_path": "models/log_bias_anchor.npz",
        "anchor_sha256": sha256_file(root / "models/log_bias_anchor.npz"),
        "selection_basis": "blocked validation 2018-2019 only",
        "selection_rule": (
            "choose the best-ranked qualifying configuration; otherwise retain "
            "physical_control"
        ),
        "decision": decision,
        "selected_configuration": selected,
        "selected_candidate": _json_safe(candidate),
        "seeds": list(EXPECTED_SEEDS),
        "checkpoints": checkpoints,
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "held_out_years": list(TEST_YEARS),
        "test_data_accessed_by_freeze": False,
        "test_metrics_used_for_selection": False,
    }
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path, json.dumps(selection, indent=2) + "\n")
    return output_path


def validate_frozen_selection(selection_path: Path) -> Mapping[str, Any]:
    """Verify that a selection remains tied to untouched validation artifacts."""

    path = Path(selection_path).expanduser().resolve()
    selection = _read_json(path)
    if (
        selection.get("schema_name") != SELECTION_SCHEMA_NAME
        or int(selection.get("schema_version", -1)) != SELECTION_SCHEMA_VERSION
        or selection.get("status") != "frozen"
    ):
        raise EvaluationContractError("selection document schema/status differs")
    if selection.get("selection_basis") != "blocked validation 2018-2019 only":
        raise EvaluationContractError("selection is not validation-only")
    if selection.get("test_data_accessed_by_freeze") is not False:
        raise EvaluationContractError("selection freeze accessed held-out data")
    if selection.get("test_metrics_used_for_selection") is not False:
        raise EvaluationContractError("selection used held-out metrics")
    if tuple(selection.get("seeds", ())) != EXPECTED_SEEDS:
        raise EvaluationContractError("selection does not freeze all three seeds")
    if tuple(selection.get("train_years", ())) != TRAIN_YEARS:
        raise EvaluationContractError("selection training years differ")
    if tuple(selection.get("validation_years", ())) != VALIDATION_YEARS:
        raise EvaluationContractError("selection validation years differ")
    if tuple(selection.get("held_out_years", ())) != TEST_YEARS:
        raise EvaluationContractError("selection held-out years differ")
    root = Path(str(selection.get("source_sweep"))).expanduser().resolve()
    validate_completed_confirmation_sweep(root)
    checks = (
        ("manifest.json", selection.get("source_sweep_manifest_sha256")),
        (selection.get("ranking_relative_path"), selection.get("ranking_sha256")),
        (
            selection.get("normalization_relative_path"),
            selection.get("normalization_sha256"),
        ),
        (selection.get("anchor_relative_path"), selection.get("anchor_sha256")),
    )
    for relative, expected in checks:
        artifact = root / str(relative)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise EvaluationContractError(f"frozen source artifact changed: {artifact}")
    selected = str(selection.get("selected_configuration"))
    if selected not in EXPECTED_CONFIGURATIONS:
        raise EvaluationContractError("selected configuration is not in confirmation")
    ranking = pd.read_csv(root / str(selection.get("ranking_relative_path")))
    required = {"configuration", "rank", "qualifies"}
    if not required.issubset(ranking.columns):
        raise EvaluationContractError("frozen ranking no longer supports selection")
    qualified = ranking.loc[_as_bool(ranking.qualifies)].sort_values("rank")
    automatic = (
        str(qualified.iloc[0].configuration)
        if len(qualified)
        else EXPECTED_REFERENCE
    )
    if selected != automatic:
        raise EvaluationContractError(
            "frozen selection differs from the predeclared validation rule"
        )
    candidate = selection.get("selected_candidate")
    if (
        not isinstance(candidate, Mapping)
        or _json_safe(candidate) != canonical_candidate_payload(selected)
    ):
        raise EvaluationContractError("frozen candidate definition differs")
    checkpoints = selection.get("checkpoints")
    if not isinstance(checkpoints, list) or tuple(
        int(item.get("seed", -1)) for item in checkpoints
    ) != EXPECTED_SEEDS:
        raise EvaluationContractError("frozen checkpoint list differs")
    for item in checkpoints:
        checkpoint = root / str(item.get("relative_path"))
        record = root / str(item.get("run_record_relative_path"))
        if not checkpoint.is_file() or sha256_file(checkpoint) != item.get("sha256"):
            raise EvaluationContractError(f"frozen checkpoint changed: {checkpoint}")
        if not record.is_file() or sha256_file(record) != item.get("run_record_sha256"):
            raise EvaluationContractError(f"frozen run record changed: {record}")
        record_payload = _read_json(record)
        if (
            int(record_payload.get("seed", -1)) != int(item.get("seed", -2))
            or _json_safe(record_payload.get("candidate"))
            != canonical_candidate_payload(selected)
            or str(record_payload.get("checkpoint")) != str(item.get("relative_path"))
            or record_payload.get("checkpoint_sha256") != item.get("sha256")
            or int(record_payload.get("best_epoch_zero_based", -1))
            != int(item.get("best_epoch_zero_based", -2))
            or not np.isclose(
                float(record_payload.get("best_validation_loss", np.nan)),
                float(item.get("best_validation_loss", np.nan)),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise EvaluationContractError(
                f"frozen run-record contract differs: {record}"
            )
    return selection


def _stats(normalization: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    item = normalization.get(name)
    if not isinstance(item, Mapping):
        raise EvaluationContractError(f"frozen normalization lacks {name}")
    mean = np.asarray(item.get("mean_by_lead"), dtype=np.float32)
    std = np.asarray(item.get("std_by_lead"), dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or not np.isfinite(mean).all():
        raise EvaluationContractError(f"invalid frozen statistics for {name}")
    if not np.isfinite(std).all() or np.any(std <= 0.0):
        raise EvaluationContractError(f"invalid frozen standard deviation for {name}")
    return mean, std


def normalize_with_frozen_stats(
    values: np.ndarray,
    normalization: Mapping[str, Any],
    name: str,
    support: np.ndarray,
    *,
    full_domain: bool,
) -> np.ndarray:
    """Apply stored lead moments; this function has no fitting code path."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (6, 27, 27):
        raise EvaluationContractError(f"invalid field shape for {name}: {values.shape}")
    mean, std = _stats(normalization, name)
    result = (values - mean[None, :, None, None]) / std[None, :, None, None]
    valid = np.isfinite(result)
    if not full_domain:
        valid &= np.asarray(support, dtype=bool)[None, None]
    result = np.where(valid, result, 0.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise EvaluationContractError(f"frozen normalization failed for {name}")
    return result


def build_frozen_test_features(
    forecast: Any,
    weekly_climatology: np.ndarray,
    climatology_daily: np.ndarray,
    t2m_weekly: np.ndarray,
    physical_fields: Mapping[str, np.ndarray],
    weights: np.ndarray,
    normalization: Mapping[str, Any],
) -> np.ndarray:
    """Build the exact 38-channel test tensor using only stored moments."""

    n_case = len(forecast.initializations)
    expected_field = (n_case, 6, 27, 27)
    for name, values in (
        ("raw_fuxi", forecast.ensemble_mean),
        ("fuxi_spread", forecast.ensemble_spread),
        ("weekly_climatology", weekly_climatology),
        ("t2m_weekly", t2m_weekly),
    ):
        if np.asarray(values).shape != expected_field:
            raise EvaluationContractError(f"{name} has shape {np.asarray(values).shape}")
    if not np.array_equal(np.asarray(forecast.latitude), EXPECTED_LATITUDE):
        raise EvaluationContractError("test latitude differs from the trained grid")
    if not np.array_equal(np.asarray(forecast.longitude), EXPECTED_LONGITUDE):
        raise EvaluationContractError("test longitude differs from the trained grid")
    support = np.asarray(weights, dtype=np.float64) > 0.0
    if support.shape != (27, 27) or np.count_nonzero(support) != 171:
        raise EvaluationContractError("IMD verification support must contain 171 cells")

    log_fuxi = np.log1p(np.maximum(forecast.ensemble_mean, 0.0)).astype(np.float32)
    log_spread = np.log1p(np.maximum(forecast.ensemble_spread, 0.0)).astype(np.float32)
    log_climatology = np.log1p(np.maximum(weekly_climatology, 0.0)).astype(np.float32)
    channels: list[np.ndarray] = [
        normalize_with_frozen_stats(
            log_fuxi, normalization, "log_fuxi_mean", support, full_domain=True
        ),
        normalize_with_frozen_stats(
            log_spread, normalization, "log_fuxi_spread", support, full_domain=True
        ),
        normalize_with_frozen_stats(
            log_climatology,
            normalization,
            "log_imd_climatology",
            support,
            full_domain=False,
        ),
    ]
    latitude = np.asarray(forecast.latitude, dtype=np.float32)
    longitude = np.asarray(forecast.longitude, dtype=np.float32)
    lat_scaled = 2.0 * (latitude - latitude.min()) / (
        latitude.max() - latitude.min()
    ) - 1.0
    lon_scaled = 2.0 * (longitude - longitude.min()) / (
        longitude.max() - longitude.min()
    ) - 1.0
    channels.extend(
        [
            np.broadcast_to(
                lat_scaled[None, None, :, None], expected_field
            ).astype(np.float32),
            np.broadcast_to(
                lon_scaled[None, None, None, :], expected_field
            ).astype(np.float32),
        ]
    )
    midpoints = np.asarray(forecast.valid_dates)[:, :, 3]
    midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
    day = (midpoint_index.dayofyear.to_numpy() - 1).reshape(n_case, 6)
    angle = 2.0 * np.pi * day / 365.2425
    channels.extend(
        [
            np.broadcast_to(np.sin(angle)[:, :, None, None], expected_field).astype(
                np.float32
            ),
            np.broadcast_to(np.cos(angle)[:, :, None, None], expected_field).astype(
                np.float32
            ),
            np.broadcast_to(
                np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, :, None, None],
                expected_field,
            ).astype(np.float32),
            np.broadcast_to(support[None, None], expected_field).astype(np.float32),
        ]
    )
    anomaly = log_fuxi - log_climatology
    channels.append(
        normalize_with_frozen_stats(
            anomaly,
            normalization,
            "explicit_log_fuxi_anomaly",
            support,
            full_domain=False,
        )
    )
    channels.append(
        normalize_with_frozen_stats(
            np.asarray(t2m_weekly, dtype=np.float32),
            normalization,
            "fuxi_t2m_weekly",
            support,
            full_domain=True,
        )
    )

    climatology_daily = np.asarray(climatology_daily, dtype=np.float32)
    if climatology_daily.shape != (366, 27, 27):
        raise EvaluationContractError("training climatology bank has changed shape")
    bank = []
    for offset in experiment.OFFSETS_DAYS:
        dates = np.asarray(forecast.valid_dates) + np.timedelta64(offset, "D")
        positions = sweep.base.calendar_positions(dates)
        weekly = np.mean(
            climatology_daily[positions], axis=2, dtype=np.float64
        ).astype(np.float32)
        bank.append(weekly)
    bank_array = np.stack(bank, axis=2)
    centre = bank_array[:, :, experiment.OFFSETS_DAYS.index(0)]
    if not np.allclose(
        centre[..., support],
        np.asarray(weekly_climatology)[..., support],
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise EvaluationContractError("zero-offset frozen climatology differs")
    climo_mean, climo_std = _stats(normalization, "log_imd_climatology")
    normalized_bank = (
        np.log1p(np.maximum(bank_array, 0.0))
        - climo_mean[None, :, None, None, None]
    ) / climo_std[None, :, None, None, None]
    normalized_bank = np.where(
        support[None, None, None], normalized_bank, 0.0
    ).astype(np.float32)
    anomaly_mean, anomaly_std = _stats(
        normalization, "explicit_log_fuxi_anomaly"
    )
    normalized_anomaly_bank = (
        log_fuxi[:, :, None] - np.log1p(np.maximum(bank_array, 0.0))
        - anomaly_mean[None, :, None, None, None]
    ) / anomaly_std[None, :, None, None, None]
    normalized_anomaly_bank = np.where(
        support[None, None, None], normalized_anomaly_bank, 0.0
    ).astype(np.float32)
    for offset_index in range(len(experiment.OFFSETS_DAYS)):
        channels.append(normalized_bank[:, :, offset_index])
    for offset_index in range(len(experiment.OFFSETS_DAYS)):
        channels.append(normalized_anomaly_bank[:, :, offset_index])

    missing = sorted(set(PHYSICAL_NAMES) - set(physical_fields))
    if missing:
        raise EvaluationContractError(f"test physical cache lacks {missing}")
    for name in PHYSICAL_NAMES:
        values = np.asarray(physical_fields[name], dtype=np.float32)
        if values.shape != expected_field or not np.isfinite(values).all():
            raise EvaluationContractError(f"test physical field is invalid: {name}")
        channels.append(
            normalize_with_frozen_stats(
                values, normalization, name, support, full_domain=True
            )
        )
    features = np.stack(channels, axis=2).astype(np.float32)
    expected_names = [
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
        *[f"imd_climatology_offset_{offset:+d}d" for offset in experiment.OFFSETS_DAYS],
        *[
            f"fuxi_minus_imd_climatology_offset_{offset:+d}d"
            for offset in experiment.OFFSETS_DAYS
        ],
        *PHYSICAL_NAMES,
    ]
    if list(normalization.get("input_channels", ())) != expected_names:
        raise EvaluationContractError("frozen feature order differs from 38-channel test")
    if features.shape != (n_case, 6, 38, 27, 27) or not np.isfinite(features).all():
        raise EvaluationContractError(f"invalid frozen test tensor: {features.shape}")
    return features


def _test_forecast_subset(forecast: Any, indices: np.ndarray) -> Any:
    selected_initializations = np.asarray(forecast.initializations)[indices]
    selected_set = set(
        np.datetime_as_string(selected_initializations, unit="D").tolist()
    )
    sources = tuple(
        source
        for source in forecast.source_files
        if Path(source).stem[:4] in {"2020", "2021"}
    )
    if len(sources) != 70:
        raise EvaluationContractError("expected 70 held-out FuXi source shards")
    source_dates = {
        f"{Path(source).stem[:4]}-{Path(source).stem[4:6]}-{Path(source).stem[6:8]}"
        for source in sources
    }
    if source_dates != selected_set:
        raise EvaluationContractError("held-out source shards differ from split dates")
    return sweep.base.ForecastData(
        initializations=selected_initializations.copy(),
        valid_dates=np.asarray(forecast.valid_dates)[indices].copy(),
        ensemble_mean=np.asarray(forecast.ensemble_mean)[indices].copy(),
        ensemble_spread=np.asarray(forecast.ensemble_spread)[indices].copy(),
        latitude=np.asarray(forecast.latitude).copy(),
        longitude=np.asarray(forecast.longitude).copy(),
        source_files=sources,
    )


def _load_frozen_preprocessing(
    selection: Mapping[str, Any],
) -> tuple[Mapping[str, Any], np.ndarray, float, np.ndarray]:
    root = Path(str(selection["source_sweep"])).resolve()
    normalization = _read_json(root / str(selection["normalization_relative_path"]))
    with np.load(
        root / str(selection["anchor_relative_path"]), allow_pickle=False
    ) as anchor:
        lead_month_residual = np.asarray(
            anchor["lead_month_residual"], dtype=np.float32
        )
        shrinkage = float(np.asarray(anchor["shrinkage"]).item())
        target_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
    if lead_month_residual.shape != (6, 12, 27, 27):
        raise EvaluationContractError("frozen log-bias correction shape differs")
    if not np.isclose(
        shrinkage, EXPECTED_LOG_BIAS_SHRINKAGE, rtol=0.0, atol=0.0
    ):
        raise EvaluationContractError("frozen log-bias shrinkage changed")
    if (
        target_scale.shape != (6,)
        or not np.isfinite(target_scale).all()
        or np.any(target_scale <= 0.0)
    ):
        raise EvaluationContractError("frozen anchored target scale differs")
    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
        *PHYSICAL_NAMES,
    ):
        _stats(normalization, name)
    return normalization, lead_month_residual, shrinkage, target_scale


def _explicit_cuda_device(value: str) -> str:
    requested = str(value).strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda:"):
        raise EvaluationContractError(
            "locked evaluation requires an explicit CUDA device such as cuda:0"
        )
    if not torch.cuda.is_available():
        raise EvaluationContractError("CUDA was requested but is not visible")
    device = torch.device(requested)
    index = 0 if device.index is None else int(device.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise EvaluationContractError(f"CUDA device is unavailable: {requested}")
    torch.cuda.set_device(device)
    probe = torch.zeros(1, device=device)
    if not torch.isfinite(probe).all():
        raise EvaluationContractError("CUDA allocation probe failed")
    del probe
    return f"cuda:{index}"


def _load_and_smoke_frozen_models(
    selection: Mapping[str, Any],
    normalization: Mapping[str, Any],
    target_scale: np.ndarray,
    *,
    device: str,
) -> tuple[torch.nn.Module, ...]:
    """Strict-load and execute every frozen checkpoint without target data."""

    candidate_payload = dict(canonical_candidate_payload(str(selection["selected_configuration"])))
    candidate_payload["physical_predictors"] = tuple(
        candidate_payload.get("physical_predictors", ())
    )
    candidate = sweep.SweepCandidate(**candidate_payload)
    _, mean_std = _stats(normalization, "log_fuxi_mean")
    _, anomaly_std = _stats(normalization, "explicit_log_fuxi_anomaly")
    ratio = mean_std / anomaly_std
    root = Path(str(selection["source_sweep"])).resolve()
    models: list[torch.nn.Module] = []
    dummy = np.zeros((1, 6, 38, 27, 27), dtype=np.float32)
    for checkpoint_item in selection["checkpoints"]:
        seed = int(checkpoint_item["seed"])
        checkpoint_path = root / str(checkpoint_item["relative_path"])
        if sha256_file(checkpoint_path) != checkpoint_item["sha256"]:
            raise EvaluationContractError("checkpoint changed after selection validation")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or int(checkpoint.get("seed", -1)) != seed:
            raise EvaluationContractError("checkpoint seed/schema differs")
        if not np.array_equal(
            np.asarray(checkpoint.get("target_scale"), dtype=np.float32), target_scale
        ):
            raise EvaluationContractError("checkpoint target scale differs from anchor")
        if not np.array_equal(
            np.asarray(checkpoint.get("lead_weights"), dtype=np.float32),
            np.asarray(sweep.LEAD_WEIGHTS, dtype=np.float32),
        ):
            raise EvaluationContractError("checkpoint lead weights differ")
        if checkpoint.get("loss_coefficients") != sweep.LOSS_COEFFICIENTS:
            raise EvaluationContractError("checkpoint loss coefficients differ")
        if int(checkpoint.get("best_epoch", -1)) != int(
            checkpoint_item["best_epoch_zero_based"]
        ) or not np.isclose(
            float(checkpoint.get("best_validation_loss", np.nan)),
            float(checkpoint_item["best_validation_loss"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise EvaluationContractError("checkpoint validation selection differs")
        model = sweep.build_model(candidate, 38, ratio)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise EvaluationContractError(
                f"checkpoint state is incompatible: {checkpoint_path}"
            ) from exc
        smoke = predict(
            model,
            dummy,
            device=device,
            batch_size=1,
            use_amp=True,
        )
        if smoke.shape != (1, 6, 27, 27) or not np.isfinite(smoke).all():
            raise EvaluationContractError("checkpoint CUDA dummy forward failed")
        models.append(model)
        del checkpoint, smoke
    return tuple(models)


def forecast_only_preflight(
    selection: Mapping[str, Any],
    *,
    physical_cache: Path | None,
    device: str,
    batch_size: int,
    bootstrap_replicates: int,
    bootstrap_block_length: int,
    bootstrap_seed: int,
    fdr_q: float,
    india_boundary: Path,
) -> ForecastOnlyPreflight:
    """Validate all failure-prone forecast/model inputs before opening IMD test data."""

    if batch_size < 1:
        raise EvaluationContractError("batch size must be positive")
    if bootstrap_replicates < 1000:
        raise EvaluationContractError(
            "at least 1000 bootstrap replicates are required for final inference"
        )
    if not 1 <= bootstrap_block_length <= 35:
        raise EvaluationContractError("bootstrap block length must lie in 1..35")
    if not 0 <= bootstrap_seed < 2**63:
        raise EvaluationContractError("bootstrap seed must lie in 0..2^63-1")
    if not 0.0 < fdr_q < 1.0:
        raise EvaluationContractError("FDR q must lie strictly between zero and one")
    resolved_device = _explicit_cuda_device(device)
    boundary_segments, boundary_provenance = load_india_boundary(india_boundary)
    source = boundary_provenance.get("source", {})
    if source.get("source_sha256") != EXPECTED_SURVEY_OF_INDIA_SOURCE_SHA256:
        raise EvaluationContractError("India boundary source checksum differs")

    experiment.set_experiment_scope(
        all_weeks=True,
        large_model=False,
        regularized_large=False,
        full_fuxi_context=True,
    )
    experiment.configure_contract()
    if tuple(experiment.TRAIN_YEARS) != TRAIN_YEARS:
        raise EvaluationContractError("runtime training-climatology years changed")
    if tuple(sweep.base.TRAIN_YEARS) != TRAIN_YEARS:
        raise EvaluationContractError("base runtime training years changed")
    forecast = sweep.base.load_fuxi()
    split = sweep.base.split_indices(forecast.initializations)
    indices = np.asarray(split["test"], dtype=np.int64)
    test_forecast = _test_forecast_subset(forecast, indices)
    years = pd.DatetimeIndex(test_forecast.initializations).year.to_numpy()
    if indices.shape != (70,) or tuple(sorted(np.unique(years))) != TEST_YEARS:
        raise EvaluationContractError("held-out split is not exactly 2020--2021")
    if pd.Series(years).value_counts().sort_index().to_dict() != {2020: 35, 2021: 35}:
        raise EvaluationContractError("held-out years must contain 35 starts each")
    dates = np.asarray(test_forecast.initializations, dtype="datetime64[D]")
    if np.any(dates[1:] <= dates[:-1]):
        raise EvaluationContractError("held-out initialization order is not chronological")

    normalization, lead_month_residual, shrinkage, target_scale = (
        _load_frozen_preprocessing(selection)
    )
    physical_metadata = normalization.get("fuxi_physical_predictors")
    if not isinstance(physical_metadata, Mapping):
        raise EvaluationContractError("frozen physical-predictor provenance is missing")
    import fuxi_physical_feature_cache as development_cache
    import fuxi_physical_postselection_cache as postselection_cache

    if (
        postselection_cache.CACHE_SCHEMA_NAME
        != EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_NAME
        or int(postselection_cache.CACHE_SCHEMA_VERSION)
        != EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_VERSION
        or postselection_cache.EVALUATION_ROLE != EVALUATION_ROLE
    ):
        raise EvaluationContractError(
            "post-selection physical-cache schema or evaluation role changed"
        )

    _, development_contract = development_cache.inspect_source()
    if physical_metadata.get("source_fingerprint") != development_contract.source_fingerprint:
        raise EvaluationContractError("development physical source fingerprint changed")
    development_path = Path(str(physical_metadata.get("cache_path"))).resolve()
    expected_development_hash = physical_metadata.get("cache_sha256")
    if (
        not development_path.is_file()
        or not expected_development_hash
        or sha256_file(development_path) != expected_development_hash
    ):
        raise EvaluationContractError("development physical cache hash changed")

    requested_physical_cache = (
        EXPECTED_POSTSELECTION_PHYSICAL_CACHE
        if physical_cache is None
        else Path(physical_cache).expanduser().resolve()
    )
    if requested_physical_cache != EXPECTED_POSTSELECTION_PHYSICAL_CACHE:
        raise EvaluationContractError(
            "the locked exploratory-v2 post-selection physical cache is required"
        )
    if (
        not requested_physical_cache.is_file()
        or sha256_file(requested_physical_cache)
        != EXPECTED_POSTSELECTION_PHYSICAL_CACHE_SHA256
    ):
        raise EvaluationContractError(
            "locked exploratory-v2 post-selection physical cache checksum differs"
        )

    _, live_postselection_contract = postselection_cache.inspect_source()
    loader_kwargs: dict[str, Any] = {
        "expected_source_fingerprint": live_postselection_contract.source_fingerprint,
        "expected_scope_fingerprint": live_postselection_contract.scope_fingerprint,
        "cache_path": requested_physical_cache,
    }
    predictors = postselection_cache.load_fuxi_postselection_physical_predictors(
        test_forecast, **loader_kwargs
    )
    if Path(str(predictors.source_store)).resolve() != Path(
        development_contract.source_store
    ).resolve():
        raise EvaluationContractError(
            "post-selection physical fields come from a different source archive"
        )
    if not predictors.cache_path or not predictors.cache_sha256:
        raise EvaluationContractError("post-selection physical cache lacks a checksum")
    t2m_weekly = sweep.common.load_t2m_weekly(test_forecast)
    models = _load_and_smoke_frozen_models(
        selection, normalization, target_scale, device=resolved_device
    )
    return ForecastOnlyPreflight(
        forecast=forecast,
        test_forecast=test_forecast,
        test_indices=indices,
        t2m_weekly=t2m_weekly,
        physical_predictors=predictors,
        normalization=normalization,
        lead_month_residual=lead_month_residual,
        shrinkage=shrinkage,
        target_scale=target_scale,
        models=models,
        device=resolved_device,
        boundary_segments=boundary_segments,
        boundary_provenance=boundary_provenance,
    )


def load_frozen_test_data(
    selection: Mapping[str, Any], preflight: ForecastOnlyPreflight
) -> FrozenTestData:
    """Open IMD only after every forecast/model preflight check has passed."""

    observations, climatology_daily, _, observation_stores = experiment.load_imd(
        preflight.forecast
    )
    weights = experiment.load_imd_weights(
        preflight.forecast, observations.observation_fraction
    )
    support = weights > 0.0
    indices = preflight.test_indices
    truth = np.asarray(observations.weekly_truth, dtype=np.float32)[indices]
    climatology = np.asarray(observations.weekly_climatology, dtype=np.float32)[indices]
    if (
        truth.shape != (70, 6, 27, 27)
        or climatology.shape != truth.shape
        or not np.isfinite(truth[..., support]).all()
        or not np.isfinite(climatology[..., support]).all()
        or np.any(truth[..., support] < 0.0)
        or np.any(climatology[..., support] < 0.0)
    ):
        raise EvaluationContractError("held-out IMD truth/climatology is incomplete")
    if not np.isfinite(preflight.lead_month_residual[..., support]).all():
        raise EvaluationContractError("log-bias anchor is non-finite on IMD support")
    correction = LogBiasCorrection(
        preflight.lead_month_residual, preflight.shrinkage
    )
    log_bias = apply_log_bias_correction(
        preflight.test_forecast.ensemble_mean,
        preflight.test_forecast.initializations,
        correction,
    )
    if not np.isfinite(log_bias[..., support]).all():
        raise EvaluationContractError("frozen log-bias prediction is incomplete")
    features = build_frozen_test_features(
        preflight.test_forecast,
        climatology,
        climatology_daily,
        preflight.t2m_weekly,
        preflight.physical_predictors.feature_fields,
        weights,
        preflight.normalization,
    )
    return FrozenTestData(
        features=features,
        raw_fuxi=np.asarray(preflight.test_forecast.ensemble_mean, dtype=np.float32),
        log_bias=np.asarray(log_bias, dtype=np.float32),
        truth=truth,
        climatology=climatology,
        weights=np.asarray(weights, dtype=np.float64),
        initializations=np.asarray(
            preflight.test_forecast.initializations, dtype="datetime64[D]"
        ),
        latitude=np.asarray(preflight.test_forecast.latitude, dtype=np.float64),
        longitude=np.asarray(preflight.test_forecast.longitude, dtype=np.float64),
        target_scale=preflight.target_scale,
        source_files=tuple(preflight.test_forecast.source_files),
        observation_stores=tuple(str(store) for store in observation_stores),
        spatial_store=str(Path(sweep.base.SPATIAL_STORE).resolve()),
        physical_cache_path=str(preflight.physical_predictors.cache_path),
        physical_cache_sha256=str(preflight.physical_predictors.cache_sha256),
    )


def predict_frozen_ensemble(
    models: Sequence[torch.nn.Module],
    data: FrozenTestData,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run only the checkpoint models already strict-loaded during preflight."""

    residual_members = []
    for model in models:
        residual = predict(
            model,
            data.features,
            device=device,
            batch_size=batch_size,
            use_amp=True,
        )
        if residual.shape != data.truth.shape or not np.isfinite(residual).all():
            raise EvaluationContractError("checkpoint produced an invalid residual")
        residual_members.append(np.asarray(residual, dtype=np.float32))
        model.to("cpu")
        torch.cuda.empty_cache()
    member_array = np.stack(residual_members)
    ensemble_residual = np.mean(member_array, axis=0, dtype=np.float64).astype(
        np.float32
    )
    valid = np.broadcast_to(
        (data.weights > 0.0)[None, None], data.log_bias.shape
    ).copy()
    corrected = reconstruct_anchored_precipitation(
        data.log_bias,
        ensemble_residual,
        data.target_scale,
        valid_mask=valid,
    )
    corrected[..., ~(data.weights > 0.0)] = np.nan
    return corrected.astype(np.float32), member_array


def build_case_metrics(
    data: FrozenTestData, predictions: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    valid = np.broadcast_to(
        (data.weights > 0.0)[None, None], data.truth.shape
    ).copy()
    frames = []
    for method in METHODS:
        prediction = np.asarray(predictions[method], dtype=np.float32)
        frame = compute_case_metrics(
            data.truth,
            prediction,
            data.truth - data.climatology,
            prediction - data.climatology,
            data.weights,
            predictor=method,
            case_ids=data.initializations,
            leads=np.arange(1, 7),
            valid_mask=valid,
        )
        frame.insert(0, "method", method)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != len(METHODS) * 70 * 6:
        raise EvaluationContractError("case-metric row count differs")
    return result


def circular_moving_block_indices(
    group_indices: np.ndarray,
    target_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample chronological circular blocks from one initialization year."""

    group = np.asarray(group_indices, dtype=np.int64)
    if group.ndim != 1 or not len(group) or target_count < 1:
        raise ValueError("moving-block group/target must be nonempty")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    blocks = math.ceil(target_count / block_length)
    starts = rng.integers(0, len(group), size=blocks)
    offsets = np.arange(block_length, dtype=np.int64)
    sampled = np.concatenate([group[(start + offsets) % len(group)] for start in starts])
    return sampled[:target_count]


def two_stage_moving_block_draws(
    initializations: np.ndarray,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample years, then blocks within years, retaining every lead together."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    years = pd.DatetimeIndex(dates).year.to_numpy()
    unique_years = np.asarray(sorted(np.unique(years)), dtype=np.int64)
    if tuple(unique_years) != TEST_YEARS:
        raise ValueError("bootstrap requires exactly the 2020 and 2021 strata")
    groups = {year: np.flatnonzero(years == year) for year in unique_years}
    if any(len(group) != 35 for group in groups.values()):
        raise ValueError("each bootstrap year must have 35 initializations")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, len(dates)), dtype=np.int16)
    for replicate in range(replicates):
        sampled_years = rng.choice(unique_years, size=len(unique_years), replace=True)
        pieces = [
            circular_moving_block_indices(
                groups[int(year)], 35, block_length, rng
            )
            for year in sampled_years
        ]
        result[replicate] = np.concatenate(pieces).astype(np.int16)
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values while preserving NaNs."""

    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values.reshape(-1)))
    if not len(finite_positions):
        return result
    finite = values.reshape(-1)[finite_positions]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    flat = result.reshape(-1)
    flat[finite_positions] = restored
    return result


def _paired_effect(
    baseline: np.ndarray, candidate: np.ndarray, metric: str
) -> float:
    baseline_mean = float(np.nanmean(baseline))
    candidate_mean = float(np.nanmean(candidate))
    if metric == "rmse":
        return 100.0 * (baseline_mean - candidate_mean) / baseline_mean
    if metric == "acc":
        return candidate_mean - baseline_mean
    if metric == "bias":
        return abs(baseline_mean) - abs(candidate_mean)
    raise ValueError(metric)


def paired_metric_bootstrap(
    case_metrics: pd.DataFrame,
    bootstrap_draws: np.ndarray,
    *,
    fdr_q: float = DEFAULT_FDR_Q,
) -> pd.DataFrame:
    """Paired uncertainty for corrected-vs-baseline effects by lead and overall."""

    if not 0.0 < float(fdr_q) < 1.0:
        raise ValueError("fdr_q must lie strictly between zero and one")
    dates = np.asarray(
        sorted(pd.to_datetime(case_metrics.case_id).dt.normalize().unique()),
        dtype="datetime64[ns]",
    )
    rows = []
    for baseline_method in ("raw_fuxi", "log_bias"):
        for metric in ("rmse", "acc", "bias"):
            matrices = {}
            for method in (baseline_method, "corrected"):
                selected = case_metrics.loc[
                    case_metrics.method.eq(method), ["case_id", "lead", metric]
                ].copy()
                selected["case_id"] = pd.to_datetime(selected.case_id).dt.normalize()
                pivot = selected.pivot(index="case_id", columns="lead", values=metric)
                pivot = pivot.reindex(pd.to_datetime(dates))
                values = pivot.to_numpy(dtype=np.float64)
                if values.shape != (70, 6):
                    raise EvaluationContractError("bootstrap metric matrix differs")
                if metric != "acc" and not np.isfinite(values).all():
                    raise EvaluationContractError(
                        f"non-finite held-out {metric} prevents paired inference"
                    )
                matrices[method] = values
            baseline = matrices[baseline_method]
            corrected = matrices["corrected"]
            if metric == "acc":
                paired_finite = np.isfinite(baseline) & np.isfinite(corrected)
                if np.any(np.count_nonzero(paired_finite, axis=0) < 60):
                    raise EvaluationContractError(
                        "too few paired finite ACC cases for block-bootstrap inference"
                    )
                baseline = np.where(paired_finite, baseline, np.nan)
                corrected = np.where(paired_finite, corrected, np.nan)
            for lead_index in (*range(6), None):
                if lead_index is None:
                    baseline_scope = baseline
                    corrected_scope = corrected
                    scope = "ALL_WEEKS"
                else:
                    baseline_scope = baseline[:, lead_index]
                    corrected_scope = corrected[:, lead_index]
                    scope = f"W{lead_index + 1}"
                point = _paired_effect(baseline_scope, corrected_scope, metric)
                samples = np.empty(len(bootstrap_draws), dtype=np.float64)
                for replicate, draw in enumerate(bootstrap_draws):
                    samples[replicate] = _paired_effect(
                        baseline_scope[np.asarray(draw, dtype=np.int64)],
                        corrected_scope[np.asarray(draw, dtype=np.int64)],
                        metric,
                    )
                lower, upper = np.quantile(samples, (0.025, 0.975))
                probability = float(np.mean(samples > 0.0))
                # A formal null reference must be centred at zero.  Tail
                # fractions of the uncentred effect bootstrap are confidence
                # summaries, not null-test p-values.
                null_samples = samples - point
                p_value = min(
                    1.0,
                    2.0
                    * min(
                        (np.count_nonzero(null_samples <= -abs(point)) + 1)
                        / (len(null_samples) + 1),
                        (np.count_nonzero(null_samples >= abs(point)) + 1)
                        / (len(null_samples) + 1),
                    ),
                )
                units = {
                    "rmse": "percent reduction",
                    "acc": "ACC difference",
                    "bias": "absolute-bias reduction (mm day-1)",
                }[metric]
                rows.append(
                    {
                        "candidate": "corrected",
                        "baseline": baseline_method,
                        "scope": scope,
                        "metric": metric,
                        "effect_positive_is_better": point,
                        "effect_units": units,
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                        "probability_improved": probability,
                        "centered_block_null_two_sided_p": float(p_value),
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q_across_six_leads"] = np.nan
    for (baseline, metric), group in result.loc[
        result.scope.ne("ALL_WEEKS")
    ].groupby(["baseline", "metric"]):
        q_values = benjamini_hochberg(
            group.centered_block_null_two_sided_p.to_numpy()
        )
        result.loc[group.index, "bh_q_across_six_leads"] = q_values
    result["bootstrap_supported_improvement"] = (
        (result.ci95_lower > 0.0)
        & (
            result.scope.eq("ALL_WEEKS")
            | (result.bh_q_across_six_leads <= float(fdr_q))
        )
    )
    return result


def spatial_statistics(
    truth: np.ndarray,
    raw: np.ndarray,
    corrected: np.ndarray,
    weights: np.ndarray,
    bootstrap_draws: np.ndarray,
    *,
    fdr_q: float,
) -> Mapping[str, np.ndarray]:
    """Local RMSE skill and cell-wise paired block-bootstrap field control."""

    support = np.asarray(weights) > 0.0
    raw_error2 = (np.asarray(raw, dtype=np.float64) - truth) ** 2
    corrected_error2 = (np.asarray(corrected, dtype=np.float64) - truth) ** 2
    raw_rmse = np.sqrt(np.mean(raw_error2, axis=0))
    corrected_rmse = np.sqrt(np.mean(corrected_error2, axis=0))
    skill = 100.0 * (raw_rmse - corrected_rmse) / np.maximum(raw_rmse, 1.0e-8)
    differences = raw_error2 - corrected_error2
    bootstrap = np.empty(
        (len(bootstrap_draws), 6, 27, 27), dtype=np.float32
    )
    for replicate, draw in enumerate(bootstrap_draws):
        bootstrap[replicate] = np.mean(
            differences[np.asarray(draw, dtype=np.int64)], axis=0, dtype=np.float64
        ).astype(np.float32)
    lower = np.quantile(bootstrap, 0.025, axis=0)
    upper = np.quantile(bootstrap, 0.975, axis=0)
    p_values = np.full((6, 27, 27), np.nan, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan)
    supported = np.zeros((6, 27, 27), dtype=bool)
    point_difference = np.mean(differences, axis=0)
    for lead in range(6):
        samples = bootstrap[:, lead, support]
        null_samples = samples - point_difference[lead, support][None]
        observed = np.abs(point_difference[lead, support])
        p = np.minimum(
            1.0,
            2.0
            * np.minimum(
                (np.count_nonzero(null_samples <= -observed[None], axis=0) + 1)
                / (len(null_samples) + 1),
                (np.count_nonzero(null_samples >= observed[None], axis=0) + 1)
                / (len(null_samples) + 1),
            ),
        )
        q = benjamini_hochberg(p)
        p_values[lead, support] = p
        q_values[lead, support] = q
        supported[lead, support] = (
            (q <= fdr_q)
            & (lower[lead, support] > 0.0)
            & (point_difference[lead, support] > 0.0)
        )
    for array in (raw_rmse, corrected_rmse, skill, lower, upper, point_difference):
        array[:, ~support] = np.nan
    return {
        "raw_local_rmse": raw_rmse.astype(np.float32),
        "corrected_local_rmse": corrected_rmse.astype(np.float32),
        "rmse_skill_pct": skill.astype(np.float32),
        "mse_difference_ci95_lower": lower.astype(np.float32),
        "mse_difference_ci95_upper": upper.astype(np.float32),
        "mse_difference_centered_block_null_p": p_values.astype(np.float32),
        "mse_difference_bh_q": q_values.astype(np.float32),
        "bootstrap_supported_rmse_improvement_fdr": supported,
    }


def _decorate_map(
    axis: Any,
    boundary_segments: Sequence[np.ndarray],
    *,
    left: bool,
    bottom: bool,
) -> None:
    axis.set_xlim(60.0, 99.0)
    axis.set_ylim(0.0, 39.0)
    axis.set_aspect("equal", adjustable="box")
    if left:
        axis.set_ylabel("Latitude (°N)")
    if bottom:
        axis.set_xlabel("Longitude (°E)")
    axis.tick_params(labelleft=left, labelbottom=bottom)
    axis.grid(linewidth=0.35, color="0.5", alpha=0.22)
    axis.add_collection(
        LineCollection(
            boundary_segments,
            colors="#111111",
            linewidths=0.32,
            alpha=0.95,
            zorder=6,
        ),
        autolim=False,
    )


def _robust_limit(values: np.ndarray, minimum: float) -> float:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    return max(minimum, float(np.quantile(finite, 0.98))) if len(finite) else minimum


def save_figure(figure: plt.Figure, stem: Path, *, svg: bool = False) -> None:
    targets = ((stem.with_suffix(".png"), {"dpi": 360}), (stem.with_suffix(".pdf"), {}))
    if svg:
        targets = (*targets, (stem.with_suffix(".svg"), {}))
    try:
        for target, options in targets:
            target = Path(target).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                figure.savefig(
                    temporary, bbox_inches="tight", format=target.suffix.lstrip("."), **options
                )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        plt.close(figure)


def plot_lead_metrics(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_stem: Path,
    *,
    fdr_q: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.1))
    for method in METHODS:
        group = summary.loc[summary.method.eq(method)].sort_values("lead")
        for axis, metric in zip(axes, ("rmse", "acc", "bias")):
            axis.plot(
                group.lead,
                group[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
    axes[0].set_ylabel("RMSE (mm day$^{-1}$) ↓")
    axes[1].set_ylabel("Spatial ACC ↑")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].set_ylabel("Bias (mm day$^{-1}$; zero is best)")
    axes[2].axhline(0.0, color="0.2", linewidth=0.9)
    corrected = summary.loc[summary.method.eq("corrected")].sort_values("lead")
    for axis, metric in zip(axes, ("rmse", "acc", "bias")):
        support_rows = bootstrap.loc[
            bootstrap.baseline.eq("raw_fuxi")
            & bootstrap.metric.eq(metric)
            & bootstrap.scope.ne("ALL_WEEKS")
        ].sort_values("scope", key=lambda values: values.str.extract(r"(\d+)")[0].astype(int))
        for lead, (_, row) in enumerate(support_rows.iterrows(), start=1):
            if bool(row.bootstrap_supported_improvement):
                y_value = float(
                    corrected.loc[corrected.lead.eq(lead), metric].iloc[0]
                )
                axis.annotate(
                    "★",
                    (lead, y_value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=11,
                    color=METHOD_COLORS["corrected"],
                )
    for axis in axes:
        axis.set_xlabel("Lead week")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "FuXi-S2S rainfall skill over India · exploratory reused hindcast test (2020–2021)",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
    )
    figure.text(
        0.5,
        0.01,
        "★ approximate block-bootstrap support for corrected improvement vs raw FuXi: paired "
        "two-stage moving-block percentile CI > 0 plus centred block-null "
        f"BH q≤{fdr_q:.3g} across six leads. "
        "This period is not fresh independent confirmation.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.84))
    save_figure(figure, output_stem, svg=True)


def plot_spatial_week_panels(
    data: FrozenTestData,
    corrected: np.ndarray,
    spatial: Mapping[str, np.ndarray],
    boundary_segments: Sequence[np.ndarray],
    output_directory: Path,
    *,
    fdr_q: float,
) -> None:
    support = data.weights > 0.0
    observed_mean = np.mean(data.truth, axis=0)
    raw_mean = np.mean(data.raw_fuxi, axis=0)
    corrected_mean = np.mean(corrected, axis=0)
    raw_bias = np.mean(data.raw_fuxi - data.truth, axis=0)
    corrected_bias = np.mean(corrected - data.truth, axis=0)
    for array in (observed_mean, raw_mean, corrected_mean, raw_bias, corrected_bias):
        array[:, ~support] = np.nan
    rain_limit = float(
        np.quantile(
            np.concatenate(
                [
                    observed_mean[np.isfinite(observed_mean)],
                    raw_mean[np.isfinite(raw_mean)],
                    corrected_mean[np.isfinite(corrected_mean)],
                ]
            ),
            0.98,
        )
    )
    bias_limit = _robust_limit(np.stack((raw_bias, corrected_bias)), 0.5)
    skill_limit = _robust_limit(spatial["rmse_skill_pct"], 2.0)
    lon2d, lat2d = np.meshgrid(data.longitude, data.latitude)
    for lead in range(6):
        figure, axes = plt.subplots(2, 3, figsize=(14.7, 9.1))
        top_fields = (observed_mean[lead], raw_mean[lead], corrected_mean[lead])
        top_titles = ("IMD observed", "Raw FuXi-S2S", "Frozen corrected")
        rain_image = None
        for column, (field, title) in enumerate(zip(top_fields, top_titles)):
            axis = axes[0, column]
            rain_image = axis.pcolormesh(
                data.longitude,
                data.latitude,
                np.ma.masked_invalid(field),
                cmap="YlGnBu",
                vmin=0.0,
                vmax=rain_limit,
                shading="nearest",
            )
            _decorate_map(axis, boundary_segments, left=column == 0, bottom=False)
            axis.set_title(title, fontweight="semibold")
        bias_image = None
        for column, (field, title) in enumerate(
            zip(
                (raw_bias[lead], corrected_bias[lead]),
                ("Raw error (mean bias)", "Corrected error (mean bias)"),
            )
        ):
            axis = axes[1, column]
            bias_image = axis.pcolormesh(
                data.longitude,
                data.latitude,
                np.ma.masked_invalid(field),
                cmap="RdBu_r",
                vmin=-bias_limit,
                vmax=bias_limit,
                shading="nearest",
            )
            _decorate_map(axis, boundary_segments, left=column == 0, bottom=True)
            axis.set_title(title, fontweight="semibold")
        skill_axis = axes[1, 2]
        skill_image = skill_axis.pcolormesh(
            data.longitude,
            data.latitude,
            np.ma.masked_invalid(spatial["rmse_skill_pct"][lead]),
            cmap="RdBu",
            vmin=-skill_limit,
            vmax=skill_limit,
            shading="nearest",
        )
        _decorate_map(skill_axis, boundary_segments, left=False, bottom=True)
        supported = spatial["bootstrap_supported_rmse_improvement_fdr"][lead]
        skill_axis.scatter(
            lon2d[supported],
            lat2d[supported],
            s=7,
            marker=".",
            color="black",
            linewidths=0,
            zorder=8,
        )
        denominator = data.weights[support].sum()
        improved_area = (
            100.0
            * data.weights[
                support & (spatial["rmse_skill_pct"][lead] > 0.0)
            ].sum()
            / denominator
        )
        skill_axis.set_title(
            "RMSE improvement vs raw (%)\n"
            f"{improved_area:.1f}% area improved · "
            f"{supported[support].sum()}/171 bootstrap-supported cells",
            fontweight="semibold",
        )
        assert rain_image is not None and bias_image is not None
        figure.subplots_adjust(
            left=0.055,
            right=0.985,
            top=0.89,
            bottom=0.22,
            wspace=0.06,
            hspace=0.18,
        )
        rain_bar = figure.colorbar(
            rain_image,
            cax=figure.add_axes((0.06, 0.105, 0.27, 0.018)),
            orientation="horizontal",
        )
        rain_bar.set_label("JJAS weekly-mean rainfall (mm day$^{-1}$)")
        bias_bar = figure.colorbar(
            bias_image,
            cax=figure.add_axes((0.37, 0.105, 0.27, 0.018)),
            orientation="horizontal",
            extend="both",
        )
        bias_bar.set_label("Mean bias: forecast − IMD (mm day$^{-1}$)")
        skill_bar = figure.colorbar(
            skill_image,
            cax=figure.add_axes((0.69, 0.105, 0.27, 0.018)),
            orientation="horizontal",
            extend="both",
        )
        skill_bar.set_label("Local RMSE reduction (%) · positive is better")
        figure.suptitle(
            f"India rainfall bias correction · lead week {lead + 1}\n"
            "Exploratory reused FuXi hindcast test, JJAS 2020–2021",
            fontsize=15,
            fontweight="semibold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.012,
            "Official Survey of India ABDB-derived state/UT boundary. "
            "Stippling: approximate paired two-stage moving-block support; percentile CI > 0 plus "
            f"centred block-null BH q≤{fdr_q:.3g} across 171 supported cells. "
            "Not independent confirmation.",
            ha="center",
            fontsize=8.2,
        )
        save_figure(
            figure,
            output_directory / f"02_spatial_week_{lead + 1}_exploratory_test",
        )


def sha256_tree(path: Path) -> str:
    """Hash relative names and contents of a directory deterministically."""

    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_inventory(data: FrozenTestData) -> pd.DataFrame:
    paths: list[tuple[str, Path]] = [
        ("fuxi_shard", Path(path)) for path in data.source_files
    ]
    paths.extend(("imd_zarr", Path(store)) for store in data.observation_stores)
    paths.append(("spatial_support_zarr", Path(data.spatial_store)))
    paths.append(("physical_cache", Path(data.physical_cache_path)))
    rows = []
    for kind, path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            size = int(path.stat().st_size)
            checksum = sha256_file(path)
            checksum_scope = "full file"
        elif path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
            size = int(sum(item.stat().st_size for item in files))
            checksum = sha256_tree(path)
            checksum_scope = "full directory tree"
        else:
            raise EvaluationContractError(f"unsupported source artifact: {path}")
        rows.append(
            {
                "kind": kind,
                "path": str(path.resolve()),
                "size_bytes": size,
                "sha256": checksum,
                "checksum_scope": checksum_scope,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(args: argparse.Namespace) -> Path:
    """Gate on the selection, then perform one immutable held-out evaluation."""

    selection_path = Path(args.selection_manifest).expanduser().resolve()
    selection = validate_frozen_selection(selection_path)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh output is required: {output}")
    (output / "figures").mkdir(parents=True, exist_ok=False)
    (output / "metrics").mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir(parents=True, exist_ok=False)
    (output / "code").mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_utc": utc_now(),
        "evaluation_role": EVALUATION_ROLE,
        "evaluation_scope": (
            "2020-2021 exploratory/reused hindcast test; not independent confirmation"
        ),
        "test_years": list(TEST_YEARS),
        "reused_test_period": True,
        "genuine_independent_test": False,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selected_configuration": selection["selected_configuration"],
        "selection_locked_before_target_access": True,
        "selection_locked_before_test": True,
        "model_selection_locked": True,
        "parameter_updates": 0,
        "normalization_fit_on_test": False,
        "bias_fit_on_test": False,
        "test_used_for_selection": False,
        "forecast_only_preflight_complete": False,
        "test_target_accessed": False,
    }
    atomic_write_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    try:
        preflight = forecast_only_preflight(
            selection,
            physical_cache=args.physical_cache,
            device=args.device,
            batch_size=int(args.batch_size),
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_block_length=int(args.bootstrap_block_length),
            bootstrap_seed=int(args.bootstrap_seed),
            fdr_q=float(args.fdr_q),
            india_boundary=args.india_boundary,
        )
        manifest.update(
            {
                "forecast_only_preflight_complete": True,
                "preflight_completed_utc": utc_now(),
                "resolved_device": preflight.device,
                "cuda_device_name": torch.cuda.get_device_name(
                    torch.device(preflight.device)
                ),
                "postselection_physical_cache": str(
                    preflight.physical_predictors.cache_path
                ),
                "postselection_physical_cache_sha256": str(
                    preflight.physical_predictors.cache_sha256
                ),
                "postselection_physical_cache_schema_name": (
                    EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_NAME
                ),
                "postselection_physical_cache_schema_version": (
                    EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_VERSION
                ),
                "postselection_physical_cache_evaluation_role": EVALUATION_ROLE,
                "india_boundary": preflight.boundary_provenance,
            }
        )
        atomic_write_text(
            output / "manifest.json",
            json.dumps(_json_safe(manifest), indent=2) + "\n",
        )
        # This conservative marker is published immediately before the only
        # function permitted to open held-out IMD targets.
        manifest["test_target_accessed"] = True
        manifest["test_target_access_started_utc"] = utc_now()
        atomic_write_text(
            output / "manifest.json",
            json.dumps(_json_safe(manifest), indent=2) + "\n",
        )
        data = load_frozen_test_data(selection, preflight)
        corrected, residual_members = predict_frozen_ensemble(
            preflight.models,
            data,
            device=preflight.device,
            batch_size=int(args.batch_size),
        )
        predictions = {
            "raw_fuxi": data.raw_fuxi,
            "log_bias": data.log_bias,
            "corrected": corrected,
        }
        case_metrics = build_case_metrics(data, predictions)
        atomic_to_csv(case_metrics, output / "metrics" / "test_case_metrics.csv")
        lead_summary = (
            case_metrics.groupby(["method", "lead"], as_index=False)[
                ["rmse", "mae", "bias", "acc"]
            ]
            .mean()
            .sort_values(["method", "lead"])
        )
        atomic_to_csv(lead_summary, output / "metrics" / "test_metrics_by_lead.csv")
        draws = two_stage_moving_block_draws(
            data.initializations,
            replicates=int(args.bootstrap_replicates),
            block_length=int(args.bootstrap_block_length),
            seed=int(args.bootstrap_seed),
        )
        atomic_save_npy(output / "metrics" / "bootstrap_initialization_draws.npy", draws)
        bootstrap = paired_metric_bootstrap(
            case_metrics,
            draws,
            fdr_q=float(args.fdr_q),
        )
        atomic_to_csv(
            bootstrap,
            output / "metrics" / "paired_two_stage_moving_block_bootstrap.csv",
        )
        spatial = spatial_statistics(
            data.truth,
            data.raw_fuxi,
            corrected,
            data.weights,
            draws,
            fdr_q=float(args.fdr_q),
        )
        support = data.weights > 0.0
        observed_mean = np.mean(data.truth, axis=0)
        raw_mean = np.mean(data.raw_fuxi, axis=0)
        corrected_mean = np.mean(corrected, axis=0)
        raw_bias = np.mean(data.raw_fuxi - data.truth, axis=0)
        corrected_bias = np.mean(corrected - data.truth, axis=0)
        for array in (observed_mean, raw_mean, corrected_mean, raw_bias, corrected_bias):
            array[:, ~support] = np.nan
        atomic_savez(
            output / "metrics" / "spatial_test_fields.npz",
            latitude=data.latitude,
            longitude=data.longitude,
            weights=data.weights,
            observed_mean=observed_mean.astype(np.float32),
            raw_mean=raw_mean.astype(np.float32),
            corrected_mean=corrected_mean.astype(np.float32),
            raw_bias=raw_bias.astype(np.float32),
            corrected_bias=corrected_bias.astype(np.float32),
            **spatial,
        )
        atomic_savez(
            output / "predictions" / "exploratory_test_predictions.npz",
            initializations=data.initializations,
            lead_week=np.arange(1, 7, dtype=np.int16),
            latitude=data.latitude,
            longitude=data.longitude,
            weights=data.weights,
            truth_imd=data.truth,
            training_climatology=data.climatology,
            raw_fuxi=data.raw_fuxi,
            training_log_bias=data.log_bias,
            corrected=corrected,
            standardized_residual_by_seed=residual_members,
            seeds=np.asarray(EXPECTED_SEEDS, dtype=np.int32),
        )
        plot_lead_metrics(
            lead_summary,
            bootstrap,
            output / "figures" / "01_acc_rmse_bias_by_lead_exploratory_test",
            fdr_q=float(args.fdr_q),
        )
        plot_spatial_week_panels(
            data,
            corrected,
            spatial,
            preflight.boundary_segments,
            output / "figures",
            fdr_q=float(args.fdr_q),
        )
        inventory = _source_inventory(data)
        atomic_to_csv(inventory, output / "metrics" / "input_source_inventory.csv")
        for source in (
            Path(__file__),
            SOURCE_ROOT / "fuxi_physical_postselection_cache.py",
            SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
            SOURCE_ROOT / "fuxi_imd_attention_climatology.py",
            SOURCE_ROOT / "fuxi_physical_feature_cache.py",
            NEURAL_SRC / "fuxi_adapter" / "validation_sweep_models.py",
            NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
            NEURAL_SRC / "fuxi_adapter" / "anchored.py",
            NEURAL_SRC / "fuxi_adapter" / "baselines.py",
            NEURAL_SRC / "fuxi_adapter" / "metrics.py",
            NEURAL_SRC / "fuxi_adapter" / "training.py",
        ):
            atomic_copy2(source, output / "code" / source.name)
        artifacts = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        primary_all = bootstrap.loc[
            bootstrap.baseline.eq("raw_fuxi")
            & bootstrap.scope.eq("ALL_WEEKS")
        ]
        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "test_years": list(TEST_YEARS),
                "test_initialization_count": 70,
                "lead_weeks": 6,
                "support_cells": int(np.count_nonzero(support)),
                "bootstrap": {
                    "design": (
                        "paired two-stage: resample years, then circular moving "
                        "blocks of initialization cases; retain all six leads together"
                    ),
                    "replicates": int(args.bootstrap_replicates),
                    "block_length_initializations": int(args.bootstrap_block_length),
                    "seed": int(args.bootstrap_seed),
                },
                "spatial_inference": {
                    "paired_statistic": "raw squared error minus corrected squared error",
                    "confidence_interval": "two-sided percentile 95%",
                    "null_test": (
                        "approximate two-sided centred paired two-stage "
                        "moving-block bootstrap"
                    ),
                    "multiplicity": "Benjamini-Hochberg across 171 cells separately per lead",
                    "fdr_q": float(args.fdr_q),
                    "wording": (
                        "approximate bootstrap-supported evidence with only two "
                        "years; not independent confirmation"
                    ),
                    "stipple_requires_positive_point_effect_and_ci": True,
                },
                "india_boundary": preflight.boundary_provenance,
                "all_week_effects_vs_raw": primary_all.to_dict(orient="records"),
                "software": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "resolved_device": preflight.device,
                    "cuda_device_name": torch.cuda.get_device_name(
                        torch.device(preflight.device)
                    ),
                    "mixed_precision_inference": True,
                },
                "artifacts": artifacts,
            }
        )
        atomic_write_text(
            output / "manifest.json",
            json.dumps(_json_safe(manifest), indent=2) + "\n",
        )
        print(f"PASS: locked exploratory test evaluation: {output}", flush=True)
        return output
    except Exception:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "failure": traceback.format_exc(),
            }
        )
        atomic_write_text(
            output / "manifest.json",
            json.dumps(_json_safe(manifest), indent=2) + "\n",
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="freeze the automatic blocked-validation decision"
    )
    freeze.add_argument("sweep_directory", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.add_argument(
        "--expected-configuration",
        default=None,
        help="assert the automatic decision; never overrides it",
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate the already-frozen decision once"
    )
    evaluate.add_argument("selection_manifest", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--physical-cache", type=Path, default=None)
    evaluate.add_argument(
        "--device",
        default="cuda:0",
        help="explicit CUDA device; CPU/auto fallbacks are refused",
    )
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    evaluate.add_argument(
        "--bootstrap-block-length", type=int, default=DEFAULT_BLOCK_LENGTH
    )
    evaluate.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    evaluate.add_argument("--fdr-q", type=float, default=DEFAULT_FDR_Q)
    evaluate.add_argument(
        "--india-boundary", type=Path, default=DEFAULT_INDIA_BOUNDARY
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_validation_selection(
            args.sweep_directory,
            args.output,
            expected_configuration=args.expected_configuration,
        )
        print(f"PASS: frozen validation selection: {result}", flush=True)
        return 0
    if not (0.0 < float(args.fdr_q) < 1.0):
        raise ValueError("--fdr-q must lie strictly between zero and one")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    if int(args.bootstrap_replicates) < 1000:
        raise ValueError("--bootstrap-replicates must be at least 1000")
    if not 1 <= int(args.bootstrap_block_length) <= 35:
        raise ValueError("--bootstrap-block-length must lie in 1..35")
    if not 0 <= int(args.bootstrap_seed) < 2**63:
        raise ValueError("--bootstrap-seed must lie in 0..2^63-1")
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
