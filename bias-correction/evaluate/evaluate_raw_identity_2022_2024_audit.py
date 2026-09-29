#!/usr/bin/env python3
"""Matched 2022--2024 audit of the frozen raw-identity FuXi--IMD adapter.

This is a post-hoc development audit, not a selection or tuning stage.  It
binds itself to the completed no-log-bias run and to the existing frozen
100-start audit, rebuilds the predictors from the same 2002--2017 training
observations, and never reads a 2025 store.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = PROJECT_ROOT.parent
DEFAULT_AUDIT_RUN = (
    CLEAN_ROOT
    / "studies"
    / "fuxi_imd_adapter_benchmark_v1"
    / "results"
    / "full_context_jjas_2022_2024_job91439"
)
DEFAULT_RAW_IDENTITY_RUN = (
    PROJECT_ROOT
    / "resultsv2"
    / "fuxi_imd_no_log_bias_ablation"
    / "full_20260822T010749Z"
)
SPATIAL_SUPPORT_STORE = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/spatial/spatial_support.zarr"
)

AUDIT_YEARS = (2022, 2023, 2024)
EXPECTED_YEAR_COUNTS = {2022: 35, 2023: 35, 2024: 30}
TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
OFFSETS_DAYS = (-28, -21, -14, -7, 0, 7, 14, 21, 28)
EXPECTED_REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
REGION_LABELS = {
    "all_india": "All India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}
METHODS = (
    "raw_fuxi",
    "log_bias",
    "legacy_anchored_adapter",
    "raw_identity",
    "raw_identity_raw_mean_preserved",
)
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Training-only log-bias",
    "legacy_anchored_adapter": "Frozen anchored adapter",
    "raw_identity": "Frozen raw-identity adapter",
    "raw_identity_raw_mean_preserved": "Raw-mean-preserving raw-identity",
}
COMPARISONS = (
    ("log_bias", "raw_fuxi"),
    ("legacy_anchored_adapter", "raw_fuxi"),
    ("legacy_anchored_adapter", "log_bias"),
    ("raw_identity", "raw_fuxi"),
    ("raw_identity", "log_bias"),
    ("raw_identity", "legacy_anchored_adapter"),
    ("raw_identity_raw_mean_preserved", "raw_fuxi"),
    ("raw_identity_raw_mean_preserved", "log_bias"),
    ("raw_identity_raw_mean_preserved", "legacy_anchored_adapter"),
    ("raw_identity_raw_mean_preserved", "raw_identity"),
)
INTENSITY_STRATA = (
    ("dry_lt1", "<1", 0.0, 1.0),
    ("light_1_5", "1-5", 1.0, 5.0),
    ("moderate_5_10", "5-10", 5.0, 10.0),
    ("heavy_10_20", "10-20", 10.0, 20.0),
    ("extreme_ge20", ">=20", 20.0, np.inf),
)
THRESHOLDS_MM_DAY = (1.0, 5.0, 10.0, 20.0)
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_BLOCK_LENGTH = 13
DEFAULT_BOOTSTRAP_SEED = 20_260_822

# These hashes deliberately make this evaluator refuse a different audit or
# a retrained candidate.  Updating one is a new scientific evaluation version.
EXPECTED_AUDIT_MANIFEST_SHA256 = (
    "260fd5e344dead5359c482e74479ccbe1e3775fc5f0d040a4bbffdd412ac83e9"
)
EXPECTED_RAW_MANIFEST_SHA256 = (
    "09317899c7d8c1d21952a23586499f195cf47f6b24ee3ca733580a38dd8d5463"
)
EXPECTED_RAW_SELECTION_SHA256 = (
    "705721f64d517194be7fa002c3ad6a7de6534b24e6ef215beb6c25ba43aa911c"
)
EXPECTED_SPATIAL_METADATA_SHA256 = (
    "07bb0e60a396a6056df0cea9c3b96861aeb5fe0f1db9640173b9e166306cfbe4"
)
EXPECTED_ARRAY_SHA256 = {
    "initializations": "6d7f1728308c8c233f276703f08882140eb37180a6aadb5f1384b0205b0a53b7",
    "latitude": "a1ca9eb14bbab26c6ac5f911e32bb5ebe22458a49820e7a12daa17a6dc800ce7",
    "longitude": "f18a6f780547a636f81300d5f360c7691b375705d2ce046d743b858ad8262e0c",
    "support": "fe883aad0eaa22eef8cb24c0e95ae4ae311e188d16086d42539fc2acbdc3263a",
    "audit_prediction": "285fe9fde39ac82a37796d460edd96443725e3517cc7186aecb0ff9e3582c99f",
    "forecast_mean": "18fad4378f89130f1920411ce768b48862f5cd614d02f8d7565989eb46d7a3ab",
    "forecast_spread": "400902a3be7287f4738601df5579d87d74a05b518ef1ac2ddc3e026e7e609c4a",
    "forecast_t2m": "a5113d0f2e3ebfdc8734c692586c9cf3046178096c3647f93eaf0cd5cad11a3a",
    "training_climatology": "eefdbe00a6e5f7be7cc417005bfdda897884fd31a3701cd6d3f36c5518e37127",
    "truth": "dc367e79ff6dfae24e45c0107880569c454e6a0f19243b2bc126afc9c8088931",
    "weekly_climatology": "8d5b994297b7e94ca4b8cb220328273ae647334570b5fc96366b536b010219b2",
    "weekly_coverage": "155fc8b92c56bbd917d7f8e7dde6c960fc71fcec77cc14a90bb842dc0b7f7e74",
}
EXPECTED_SPATIAL_ARRAY_SHA256 = {
    "cell_area_km2": "a8ca9f637caa255fcd56b4739f40cf8c4f1ef65196b5cf68697813b0206fd6fa",
    "india_area_weight_km2": "147ccd9bbff94449ee62da84813f2834d04abe9e507e53b5a34ef1b6b10d9e3b",
    "northwest_india_fraction": "606d3058d7164ac02d0cb15252f10c209e50bd36b81c7316949ae964bb5ca92a",
    "central_india_fraction": "40e5ca9e3905d22d4e42bad470cffdc2409a3a2fda5be2fd2cd98f3bbbdfe2b1",
    "south_peninsula_fraction": "a6aecd239b7066530f4674345863bf788dbbbbd43c64bb244fb6602e5f33be25",
    "east_northeast_india_fraction": "3fe8d6ded4146c05a5220ec4a31ecbf94af4fb43ee61b87400ec240e2356a785",
}
ADDITIONAL_INFERENCE_SOURCE_SHA256 = {
    "src/project_paths.py": "b07ad15e602751d2bb45f00b511eb53494d0a4fdcf8cb7da580f0edc82594d4d",
    "src/fuxi_imerg_spatiotemporal.py": "7c9745a3c01c41b2722a5e2fa31ae2907e641547c031fd555533c3f2d0049a22",
    "../neural_adapter/src/fuxi_adapter/__init__.py": "371c173f90bf69b86179760acdfdb4296886ba9591b1bfab34f215cad18c509d",
    "../neural_adapter/src/fuxi_adapter/training.py": "354702584abb9b3f22f944f164bd0fb16ba87b5d67f926a9dea9c06d9d9d4bcf",
    "../neural_adapter/src/fuxi_adapter/metrics.py": "2f6429c81f4ebe3aebad25ec35513f5fdd242e92c37bf421bfc10c53a6c3dbe3",
    "../neural_adapter/src/fuxi_adapter/config.py": "46e741cc970ade2184647c93410482dfb722768f0fb2e520b1f5182fc91a1029",
    "../neural_adapter/src/fuxi_adapter/artifacts.py": "5e64793a85bdf5c9fd1ced473916aecdb22d98b577e692ce98a9b96750582ab1",
}


class AuditContractError(ValueError):
    """Raised when a frozen input or evaluation invariant changes."""


@dataclass(frozen=True)
class FrozenAuditFields:
    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    support: np.ndarray
    raw_fuxi: np.ndarray
    log_bias: np.ndarray
    legacy_adapter: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise AuditContractError(f"{label} hash changed: {actual} != {expected}")
    return actual


def _require_semantically_equivalent_python(
    snapshot: Path, live: Path, label: str
) -> str:
    """Allow formatting drift while rejecting any executable-source drift.

    The archived snapshot remains byte-pinned separately.  This comparison is
    only for the checked-out convenience copy, which is not imported by the
    frozen evaluator and may have been reformatted after the scientific run.
    """

    if not live.is_file():
        raise FileNotFoundError(live)
    try:
        snapshot_tree = ast.parse(
            snapshot.read_text(encoding="utf-8"),
            filename=str(snapshot),
            type_comments=True,
        )
        live_tree = ast.parse(
            live.read_text(encoding="utf-8"),
            filename=str(live),
            type_comments=True,
        )
    except SyntaxError as error:
        raise AuditContractError(f"{label} is not valid Python: {error}") from error
    if ast.dump(snapshot_tree, include_attributes=False) != ast.dump(
        live_tree, include_attributes=False
    ):
        raise AuditContractError(f"{label} executable semantics changed")
    return sha256_file(live)


def _require_array_hash(values: np.ndarray, dtype: str, key: str) -> str:
    actual = array_sha256(values, dtype)
    expected = EXPECTED_ARRAY_SHA256[key]
    if actual != expected:
        raise AuditContractError(f"{key} array changed: {actual} != {expected}")
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_checkpoint_set(
    run: Path,
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = str(selection.get("selected_model"))
    if (
        selected != "normal_climo_model"
        or float(selection.get("selected_alpha", -1)) != 1.0
    ):
        raise AuditContractError(
            "frozen selection is not normal_climo_model at alpha=1"
        )
    training = manifest.get("training", {}).get(selected)
    if not isinstance(training, Mapping):
        raise AuditContractError("selected training metadata is missing")
    records = list(training.get("runs", []))
    if [int(record.get("seed", -1)) for record in records] != [42, 43, 44]:
        raise AuditContractError("selected checkpoint seed set changed")
    selected_hashes = list(selection.get("checkpoint_sha256", {}).get(selected, []))
    record_hashes = [str(record.get("checkpoint_sha256")) for record in records]
    if selected_hashes != record_hashes:
        raise AuditContractError("selection and manifest checkpoint hashes differ")
    checked = []
    for record, expected in zip(records, record_hashes, strict=True):
        relative = Path(str(record["checkpoint"]))
        actual = _require_hash(run / relative, expected, f"checkpoint {relative}")
        artifact_hash = manifest.get("artifacts", {}).get(relative.as_posix())
        if artifact_hash != actual:
            raise AuditContractError(f"checkpoint artifact hash differs: {relative}")
        checked.append({"path": str(run / relative), "sha256": actual})
    return checked


def validate_input_artifacts(audit_run: Path, raw_run: Path) -> dict[str, Any]:
    """Validate immutable audit, model, checkpoint, and inference-source inputs."""

    audit_run = audit_run.resolve()
    raw_run = raw_run.resolve()
    audit_hash = _require_hash(
        audit_run / "manifest.json",
        EXPECTED_AUDIT_MANIFEST_SHA256,
        "frozen 2022-2024 audit manifest",
    )
    raw_hash = _require_hash(
        raw_run / "manifest.json",
        EXPECTED_RAW_MANIFEST_SHA256,
        "raw-identity manifest",
    )
    raw_selection_hash = _require_hash(
        raw_run / "selection.json",
        EXPECTED_RAW_SELECTION_SHA256,
        "raw-identity selection",
    )
    audit_manifest = _load_json(audit_run / "manifest.json")
    raw_manifest = _load_json(raw_run / "manifest.json")
    raw_selection = _load_json(raw_run / "selection.json")

    if audit_manifest.get("status") != "complete":
        raise AuditContractError("frozen audit is incomplete")
    if tuple(audit_manifest.get("audit_initialization_years", ())) != AUDIT_YEARS:
        raise AuditContractError("frozen audit years changed")
    if audit_manifest.get("audit_counts") != {
        str(k): v for k, v in EXPECTED_YEAR_COUNTS.items()
    }:
        raise AuditContractError("frozen audit year counts changed")
    if int(audit_manifest.get("audit_case_count", -1)) != 100:
        raise AuditContractError("frozen audit does not contain 100 starts")
    if int(audit_manifest.get("final_initialization_year_quarantined", -1)) != 2025:
        raise AuditContractError("2025 quarantine declaration changed")
    operational_stores = [
        Path(value) for value in audit_manifest["operational_fuxi_source_stores"]
    ]
    verification_stores = [
        Path(value) for value in audit_manifest["verification_imd_source_stores"]
    ]
    training_stores = [
        Path(value) for value in audit_manifest["training_imd_source_stores"]
    ]
    if sorted(int(path.stem) for path in operational_stores) != [
        2022,
        2022,
        2023,
        2023,
        2024,
        2024,
    ]:
        raise AuditContractError("operational source years changed")
    if tuple(sorted(int(path.stem) for path in verification_stores)) != AUDIT_YEARS:
        raise AuditContractError("verification source years changed")
    if tuple(sorted(int(path.stem) for path in training_stores)) != TRAIN_YEARS:
        raise AuditContractError("training IMD source years changed")

    expected_tree = str(audit_manifest.get("outputs", {}).get("predictions.zarr"))
    actual_tree = sha256_tree(audit_run / "predictions.zarr")
    if actual_tree != expected_tree:
        raise AuditContractError("frozen audit prediction tree changed")
    for relative, expected in audit_manifest.get("outputs", {}).items():
        if relative == "predictions.zarr":
            continue
        if relative.startswith("code/"):
            _require_hash(
                audit_run / relative, str(expected), f"audit snapshot {relative}"
            )

    if (
        raw_manifest.get("status") != "complete"
        or raw_manifest.get("smoke") is not False
    ):
        raise AuditContractError("raw-identity input is not a completed full run")
    if raw_manifest.get("training_anchor") != "raw_fuxi":
        raise AuditContractError("raw-identity run used a different training anchor")
    if raw_manifest.get("uses_fitted_log_bias_in_neural_training") is not False:
        raise AuditContractError("raw-identity run used fitted log-bias in training")
    if raw_manifest.get("log_bias_role") != "reporting_only":
        raise AuditContractError("log-bias role changed in raw-identity run")
    if (
        raw_selection.get("status") != "frozen"
        or raw_selection.get("smoke") is not False
    ):
        raise AuditContractError("raw-identity selection is not frozen")
    if raw_selection.get("selection_scope") != "validation_only":
        raise AuditContractError("raw-identity selection was not validation-only")
    if tuple(raw_selection.get("train_years", ())) != TRAIN_YEARS:
        raise AuditContractError("raw-identity training years changed")
    if tuple(raw_selection.get("validation_years", ())) != VALIDATION_YEARS:
        raise AuditContractError("raw-identity validation years changed")
    if raw_manifest.get("selection_sha256") != raw_selection_hash:
        raise AuditContractError("raw-identity manifest does not bind the selection")
    raw_checkpoints = _validate_checkpoint_set(raw_run, raw_manifest, raw_selection)

    required_raw_artifacts = (
        "models/log_bias_anchor.npz",
        "models/training_anchor_contract.npz",
        "normalization.json",
        "selection.json",
    )
    for relative in required_raw_artifacts:
        expected = str(raw_manifest.get("artifacts", {}).get(relative, ""))
        _require_hash(raw_run / relative, expected, f"raw-identity artifact {relative}")

    raw_code_hashes = raw_manifest.get("code_sha256", {})
    raw_live_map = {
        "fuxi_imd_attention_climatology.py": PROJECT_ROOT
        / "src/fuxi_imd_attention_climatology.py",
        "fuxi_imd_no_log_bias_validation.py": PROJECT_ROOT
        / "src/fuxi_imd_no_log_bias_validation.py",
        "fuxi_imerg_a100_big_temporal.py": PROJECT_ROOT
        / "src/fuxi_imerg_a100_big_temporal.py",
        "fuxi_imerg_experiment.py": PROJECT_ROOT / "src/fuxi_imerg_experiment.py",
        "fuxi_imerg_full_archive_latelead.py": PROJECT_ROOT
        / "src/fuxi_imerg_full_archive_latelead.py",
        "models.py": CLEAN_ROOT / "neural_adapter/src/fuxi_adapter/models.py",
        "anchored.py": CLEAN_ROOT / "neural_adapter/src/fuxi_adapter/anchored.py",
        "baselines.py": CLEAN_ROOT / "neural_adapter/src/fuxi_adapter/baselines.py",
        "v3_training.py": CLEAN_ROOT / "neural_adapter/src/fuxi_adapter/v3_training.py",
    }
    checked_sources: dict[str, str] = {}
    for name, expected in raw_code_hashes.items():
        snapshot = raw_run / "code" / name
        _require_hash(snapshot, str(expected), f"raw-identity source snapshot {name}")
        live = raw_live_map[name]
        checked_sources[str(live)] = _require_semantically_equivalent_python(
            snapshot, live, f"live source {name}"
        )
    for relative, expected in ADDITIONAL_INFERENCE_SOURCE_SHA256.items():
        path = (PROJECT_ROOT / relative).resolve()
        checked_sources[str(path)] = _require_hash(
            path, expected, f"inference dependency {relative}"
        )

    legacy_run = Path(str(audit_manifest["adapter_run"])).resolve()
    legacy_manifest_hash = _require_hash(
        legacy_run / "manifest.json",
        str(audit_manifest["adapter_manifest_sha256"]),
        "legacy adapter manifest",
    )
    legacy_selection_hash = _require_hash(
        legacy_run / "selection.json",
        str(audit_manifest["adapter_selection_sha256"]),
        "legacy adapter selection",
    )
    legacy_manifest = _load_json(legacy_run / "manifest.json")
    legacy_selection = _load_json(legacy_run / "selection.json")
    if (
        legacy_manifest.get("status") != "complete"
        or legacy_manifest.get("smoke") is not False
    ):
        raise AuditContractError("legacy adapter is not a completed full run")
    legacy_checkpoints = _validate_checkpoint_set(
        legacy_run, legacy_manifest, legacy_selection
    )

    return {
        "audit_run": str(audit_run),
        "audit_manifest_sha256": audit_hash,
        "audit_predictions_tree_sha256": actual_tree,
        "raw_identity_run": str(raw_run),
        "raw_identity_manifest_sha256": raw_hash,
        "raw_identity_selection_sha256": raw_selection_hash,
        "raw_identity_checkpoints": raw_checkpoints,
        "legacy_run": str(legacy_run),
        "legacy_manifest_sha256": legacy_manifest_hash,
        "legacy_selection_sha256": legacy_selection_hash,
        "legacy_checkpoints": legacy_checkpoints,
        "checked_inference_sources": checked_sources,
    }


def validate_initializations(initializations: np.ndarray) -> pd.DatetimeIndex:
    values = np.asarray(initializations, dtype="datetime64[D]")
    dates = pd.DatetimeIndex(values)
    if len(dates) != 100 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise AuditContractError("audit initialization ordering changed")
    if tuple(sorted(dates.year.unique().tolist())) != AUDIT_YEARS:
        raise AuditContractError("audit initialization years changed")
    counts = dates.year.value_counts().sort_index().to_dict()
    if counts != EXPECTED_YEAR_COUNTS:
        raise AuditContractError(f"audit initialization counts changed: {counts}")
    if not dates.month.isin((6, 7, 8, 9)).all():
        raise AuditContractError("non-JJAS initialization entered the audit")
    _require_array_hash(values.astype("<i8"), "<i8", "initializations")
    return dates


def load_frozen_audit_fields(audit_run: Path) -> FrozenAuditFields:
    with xr.open_zarr(audit_run / "predictions.zarr", consolidated=True) as dataset:
        methods = tuple(str(value) for value in dataset.method.values.tolist())
        if methods != ("raw_fuxi", "log_bias", "selected_adapter"):
            raise AuditContractError(f"frozen audit methods changed: {methods}")
        initializations = dataset.init.values.astype("datetime64[D]")
        validate_initializations(initializations)
        latitude = dataset.latitude.values.astype(np.float64)
        longitude = dataset.longitude.values.astype(np.float64)
        support = dataset.adapter_support.load().values.astype(bool)
        prediction = dataset.prediction.load().values.astype(np.float32)
        truth = dataset.truth_imd.load().values.astype(np.float32)
        climatology = dataset.fixed_imd_climatology.load().values.astype(np.float32)
    _require_array_hash(latitude, "<f8", "latitude")
    _require_array_hash(longitude, "<f8", "longitude")
    _require_array_hash(support, "u1", "support")
    _require_array_hash(prediction, "<f4", "audit_prediction")
    _require_array_hash(truth, "<f4", "truth")
    _require_array_hash(climatology, "<f4", "weekly_climatology")
    if prediction.shape != (3, 100, 6, 27, 27):
        raise AuditContractError(
            f"unexpected frozen prediction shape: {prediction.shape}"
        )
    if truth.shape != (100, 6, 27, 27) or climatology.shape != truth.shape:
        raise AuditContractError("frozen truth/climatology shape changed")
    if int(support.sum()) != 171:
        raise AuditContractError("frozen IMD support changed")
    return FrozenAuditFields(
        initializations=initializations,
        latitude=latitude,
        longitude=longitude,
        support=support,
        raw_fuxi=prediction[0],
        log_bias=prediction[1],
        legacy_adapter=prediction[2],
        truth=truth,
        climatology=climatology,
    )


def load_frozen_model_modules(raw_run: Path) -> tuple[Any, Any, Any]:
    """Import the hash-validated source snapshot used by the raw-identity run."""

    snapshot_root = (raw_run / "code").resolve()
    for name in (
        "fuxi_imerg_experiment",
        "fuxi_imerg_full_archive_latelead",
        "fuxi_imerg_a100_big_temporal",
    ):
        loaded = sys.modules.get(name)
        if loaded is not None:
            source = Path(str(getattr(loaded, "__file__", ""))).resolve()
            if snapshot_root not in source.parents:
                raise AuditContractError(
                    f"{name} was already imported from non-frozen source"
                )
    sys.path.insert(0, str(snapshot_root))
    base = importlib.import_module("fuxi_imerg_experiment")
    common = importlib.import_module("fuxi_imerg_full_archive_latelead")
    engine = importlib.import_module("fuxi_imerg_a100_big_temporal")
    return base, common, engine


def _store_for_year(stores: Sequence[Path], year: int, variable: str) -> Path:
    matches = [
        path
        for path in stores
        if path.stem == str(year) and f"/{variable}/common_1p5/" in path.as_posix()
    ]
    if len(matches) != 1:
        raise AuditContractError(
            f"source lookup failed for {variable}/{year}: {len(matches)}"
        )
    return matches[0]


def load_operational_predictors(
    audit_manifest: Mapping[str, Any], frozen: FrozenAuditFields, base: Any
) -> tuple[Any, np.ndarray, tuple[str, ...]]:
    stores = [Path(value) for value in audit_manifest["operational_fuxi_source_stores"]]
    means: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    initializations: list[np.ndarray] = []
    opened: list[str] = []
    for year in AUDIT_YEARS:
        tp_store = _store_for_year(stores, year, "tp")
        t2m_store = _store_for_year(stores, year, "t2m")
        with xr.open_zarr(tp_store, consolidated=True) as dataset:
            if dataset.ensemble_mean_weekly.attrs.get("units") != "mm day-1":
                raise AuditContractError(f"unexpected TP units: {tp_store}")
            if not np.array_equal(
                dataset.latitude.values, frozen.latitude
            ) or not np.array_equal(dataset.longitude.values, frozen.longitude):
                raise AuditContractError(f"operational TP grid changed: {tp_store}")
            all_inits = pd.DatetimeIndex(dataset.init.values)
            inits = all_inits[all_inits.month.isin((6, 7, 8, 9))]
            if len(inits) != EXPECTED_YEAR_COUNTS[year]:
                raise AuditContractError(f"{year}: operational JJAS count changed")
            means.append(
                dataset.ensemble_mean_weekly.sel(init=inits)
                .load()
                .values.astype(np.float32)
            )
            spreads.append(
                dataset.ensemble_std_weekly.sel(init=inits)
                .load()
                .values.astype(np.float32)
            )
        with xr.open_zarr(t2m_store, consolidated=True) as dataset:
            if dataset.ensemble_mean_weekly.attrs.get("units") != "degC":
                raise AuditContractError(f"unexpected T2M units: {t2m_store}")
            if not np.array_equal(
                dataset.latitude.values, frozen.latitude
            ) or not np.array_equal(dataset.longitude.values, frozen.longitude):
                raise AuditContractError(f"operational T2M grid changed: {t2m_store}")
            temperatures.append(
                (
                    dataset.ensemble_mean_weekly.sel(init=inits).load().values
                    + np.float32(273.15)
                ).astype(np.float32)
            )
        initializations.append(inits.values.astype("datetime64[D]"))
        opened.extend((str(tp_store), str(t2m_store)))
    inits = np.concatenate(initializations)
    mean = np.concatenate(means)
    spread = np.concatenate(spreads)
    t2m = np.concatenate(temperatures)
    validate_initializations(inits)
    if not np.array_equal(inits, frozen.initializations):
        raise AuditContractError("operational and frozen initialization dates differ")
    for key, values in (
        ("forecast_mean", mean),
        ("forecast_spread", spread),
        ("forecast_t2m", t2m),
    ):
        _require_array_hash(values, "<f4", key)
    if not np.array_equal(mean, frozen.raw_fuxi):
        raise AuditContractError("operational raw FuXi differs from the frozen audit")
    if (
        mean.shape != (100, 6, 27, 27)
        or spread.shape != mean.shape
        or t2m.shape != mean.shape
    ):
        raise AuditContractError("operational predictor shapes changed")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(spread).all()
        or not np.isfinite(t2m).all()
    ):
        raise AuditContractError("operational predictors contain non-finite values")
    if np.any(mean < 0.0) or np.any(spread < 0.0):
        raise AuditContractError(
            "operational rainfall predictors contain negative values"
        )
    forecast = base.ForecastData(
        initializations=inits,
        valid_dates=base.derive_valid_dates(inits),
        ensemble_mean=mean,
        ensemble_spread=spread,
        latitude=frozen.latitude,
        longitude=frozen.longitude,
        source_files=tuple(opened),
    )
    return forecast, t2m, tuple(opened)


def _load_imd_store(
    path: Path, latitude: np.ndarray, longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_zarr(path, consolidated=True) as dataset:
        if (
            dataset.attrs.get("source") != "imd"
            or dataset.attrs.get("units") != "mm day-1"
        ):
            raise AuditContractError(f"unexpected IMD metadata: {path}")
        if not np.array_equal(dataset.latitude.values, latitude) or not np.array_equal(
            dataset.longitude.values, longitude
        ):
            raise AuditContractError(f"IMD grid changed: {path}")
        dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
        values = dataset.observation.load().values.astype(np.float32)
        coverage = dataset.observation_fraction.load().values.astype(np.float32)
    if coverage.ndim == 2:
        coverage = np.broadcast_to(coverage, values.shape).copy()
    if coverage.shape != values.shape or values.shape != (dates.size, 27, 27):
        raise AuditContractError(f"invalid IMD array shape: {path}")
    if not np.isfinite(coverage).all() or np.any((coverage < 0.0) | (coverage > 1.0)):
        raise AuditContractError(f"invalid IMD coverage: {path}")
    return dates, values, coverage


def load_training_climatology(
    audit_manifest: Mapping[str, Any], frozen: FrozenAuditFields, base: Any
) -> tuple[np.ndarray, tuple[str, ...]]:
    stores = [Path(value) for value in audit_manifest["training_imd_source_stores"]]
    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for path in stores:
        year_dates, year_values, _ = _load_imd_store(
            path, frozen.latitude, frozen.longitude
        )
        if set(pd.DatetimeIndex(year_dates).year) != {int(path.stem)}:
            raise AuditContractError(f"training IMD store contains wrong year: {path}")
        dates.append(year_dates)
        values.append(year_values)
    base.TRAIN_YEARS = TRAIN_YEARS
    climatology = base.build_training_climatology(
        np.concatenate(dates), np.concatenate(values), frozen.support
    )
    _require_array_hash(climatology, "<f4", "training_climatology")
    return climatology, tuple(str(path) for path in stores)


def load_verification_fields(
    audit_manifest: Mapping[str, Any],
    frozen: FrozenAuditFields,
    forecast: Any,
    training_climatology: np.ndarray,
    base: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    stores = [Path(value) for value in audit_manifest["verification_imd_source_stores"]]
    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    for path in stores:
        year_dates, year_values, year_coverage = _load_imd_store(
            path, frozen.latitude, frozen.longitude
        )
        if int(path.stem) not in AUDIT_YEARS or set(
            pd.DatetimeIndex(year_dates).year
        ) != {int(path.stem)}:
            raise AuditContractError(
                f"verification IMD store contains wrong year: {path}"
            )
        dates.append(year_dates)
        values.append(year_values)
        coverage.append(year_coverage)
    all_dates = np.concatenate(dates)
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_values = np.concatenate(values)[order]
    all_coverage = np.concatenate(coverage)[order]
    requested = forecast.valid_dates.reshape(-1)
    positions = np.searchsorted(all_dates, requested)
    if np.any(positions >= len(all_dates)) or not np.array_equal(
        all_dates[positions], requested
    ):
        raise AuditContractError(
            "one or more 2022-2024 IMD verification days are missing"
        )
    daily = all_values[positions].reshape(100, 6, 7, 27, 27)
    daily_coverage = all_coverage[positions].reshape(100, 6, 7, 27, 27)
    truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    weekly_coverage = np.min(daily_coverage, axis=2).astype(np.float32)
    climatology = np.mean(
        training_climatology[base.calendar_positions(forecast.valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    _require_array_hash(truth, "<f4", "truth")
    _require_array_hash(climatology, "<f4", "weekly_climatology")
    _require_array_hash(weekly_coverage, "<f4", "weekly_coverage")
    if not np.array_equal(truth, frozen.truth, equal_nan=True):
        raise AuditContractError("rebuilt IMD truth differs from frozen audit")
    if not np.allclose(
        climatology, frozen.climatology, rtol=0.0, atol=2.0e-6, equal_nan=True
    ):
        raise AuditContractError("rebuilt IMD climatology differs from frozen audit")
    return truth, climatology, weekly_coverage, tuple(str(path) for path in stores)


def normalize_dynamic(
    values: np.ndarray,
    statistics: Mapping[str, Any],
    support: np.ndarray,
    *,
    preserve_full_domain: bool = False,
) -> np.ndarray:
    mean = np.asarray(statistics["mean_by_lead"], dtype=np.float32)
    std = np.asarray(statistics["std_by_lead"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0.0):
        raise AuditContractError("saved normalization statistics changed")
    normalized = (values - mean[None, :, None, None]) / std[None, :, None, None]
    valid = (
        np.isfinite(normalized)
        if preserve_full_domain
        else support[None, None] & np.isfinite(normalized)
    )
    return np.where(valid, normalized, 0.0).astype(np.float32)


def build_features(
    forecast: Any,
    t2m_weekly: np.ndarray,
    climatology_daily: np.ndarray,
    normalization: Mapping[str, Any],
    support: np.ndarray,
    base: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the exact 29-channel full-context model input."""

    context = normalization.get("spatial_context", {})
    preserve_context = bool(context.get("enabled", False))
    if context.get("full_domain_channels") != [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "fuxi_t2m_weekly",
    ]:
        raise AuditContractError("saved full-domain context contract changed")
    weekly_climatology = np.mean(
        climatology_daily[base.calendar_positions(forecast.valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    channels: list[np.ndarray] = [
        normalize_dynamic(
            np.log1p(forecast.ensemble_mean).astype(np.float32),
            normalization["log_fuxi_mean"],
            support,
            preserve_full_domain=preserve_context,
        ),
        normalize_dynamic(
            np.log1p(forecast.ensemble_spread).astype(np.float32),
            normalization["log_fuxi_spread"],
            support,
            preserve_full_domain=preserve_context,
        ),
        normalize_dynamic(
            np.log1p(weekly_climatology).astype(np.float32),
            normalization["log_imd_climatology"],
            support,
        ),
    ]
    cases, leads, height, width = forecast.ensemble_mean.shape
    latitude = forecast.latitude.astype(np.float32)
    longitude = forecast.longitude.astype(np.float32)
    lat_scaled = (
        2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    )
    lon_scaled = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    channels.extend(
        [
            np.broadcast_to(
                lat_scaled[None, None, :, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                lon_scaled[None, None, None, :], (cases, leads, height, width)
            ),
        ]
    )
    midpoint = pd.DatetimeIndex(forecast.valid_dates[:, :, 3].reshape(-1))
    angle = 2.0 * np.pi * (midpoint.dayofyear.to_numpy() - 1) / 365.2425
    angle = angle.reshape(cases, leads)
    channels.extend(
        [
            np.broadcast_to(
                np.sin(angle)[:, :, None, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                np.cos(angle)[:, :, None, None], (cases, leads, height, width)
            ),
            np.broadcast_to(
                np.linspace(-1.0, 1.0, leads, dtype=np.float32)[None, :, None, None],
                (cases, leads, height, width),
            ),
            np.broadcast_to(support[None, None], (cases, leads, height, width)).astype(
                np.float32
            ),
        ]
    )
    raw_anomaly = np.log1p(forecast.ensemble_mean) - np.log1p(weekly_climatology)
    channels.append(
        normalize_dynamic(
            raw_anomaly.astype(np.float32),
            normalization["explicit_log_fuxi_anomaly"],
            support,
        )
    )
    channels.append(
        normalize_dynamic(
            t2m_weekly,
            normalization["fuxi_t2m_weekly"],
            support,
            preserve_full_domain=preserve_context,
        )
    )
    standard = np.stack(channels, axis=2).astype(np.float32)

    candidates = []
    for offset in OFFSETS_DAYS:
        shifted = forecast.valid_dates + np.timedelta64(offset, "D")
        candidate = np.mean(
            climatology_daily[base.calendar_positions(shifted)],
            axis=2,
            dtype=np.float64,
        )
        candidates.append(candidate.astype(np.float32))
    bank = np.stack(candidates, axis=2)
    climo_stats = normalization["log_imd_climatology"]
    climo_mean = np.asarray(climo_stats["mean_by_lead"], dtype=np.float32)
    climo_std = np.asarray(climo_stats["std_by_lead"], dtype=np.float32)
    normalized_bank = (
        np.log1p(bank).astype(np.float32) - climo_mean[None, :, None, None, None]
    ) / climo_std[None, :, None, None, None]
    anomaly_stats = normalization["explicit_log_fuxi_anomaly"]
    anomaly_mean = np.asarray(anomaly_stats["mean_by_lead"], dtype=np.float32)
    anomaly_std = np.asarray(anomaly_stats["std_by_lead"], dtype=np.float32)
    bank_anomaly = np.log1p(forecast.ensemble_mean)[:, :, None] - np.log1p(bank)
    normalized_bank_anomaly = (
        bank_anomaly - anomaly_mean[None, :, None, None, None]
    ) / anomaly_std[None, :, None, None, None]
    valid = support[None, None, None]
    normalized_bank = np.where(valid, normalized_bank, 0.0).astype(np.float32)
    normalized_bank_anomaly = np.where(valid, normalized_bank_anomaly, 0.0).astype(
        np.float32
    )
    features = np.concatenate(
        (standard, normalized_bank, normalized_bank_anomaly), axis=2
    ).astype(np.float32)
    expected_channels = [
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
        *[f"imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
        *[f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in OFFSETS_DAYS],
    ]
    if features.shape != (100, 6, 29, 27, 27) or not np.isfinite(features).all():
        raise AuditContractError(f"invalid adapter feature array: {features.shape}")
    if list(normalization.get("input_channels", [])) != expected_channels:
        raise AuditContractError("saved input-channel contract changed")
    for index in (2, 9, *range(11, 29)):
        if np.any(features[:, :, index, ~support] != 0.0):
            raise AuditContractError("IMD-derived feature leaked outside support")
    return features, weekly_climatology


def infer_raw_identity(
    raw_run: Path,
    raw_manifest: Mapping[str, Any],
    raw_selection: Mapping[str, Any],
    forecast: Any,
    features: np.ndarray,
    support: np.ndarray,
    common: Any,
    engine: Any,
) -> tuple[np.ndarray, np.ndarray]:
    selected = str(raw_selection["selected_model"])
    alpha = float(raw_selection["selected_alpha"])
    metadata = raw_manifest["training"][selected]
    candidate = engine.Candidate(
        name=str(metadata["name"]),
        label=str(metadata["label"]),
        architecture=str(metadata["architecture"]),
        batch_size=int(metadata["batch_size"]),
        learning_rate=float(metadata["learning_rate"]),
        weight_decay=float(metadata["weight_decay"]),
        dropout=float(metadata["dropout"]),
        color=str(metadata["color"]),
    )
    residual = engine.predict_candidate(
        candidate,
        metadata,
        features,
        np.arange(len(features), dtype=np.int64),
        raw_run,
        inactive_lead_count=0,
    )
    with np.load(raw_run / "models/training_anchor_contract.npz") as anchor:
        anchor_kind = str(np.asarray(anchor["anchor_kind"]).item())
        target_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
        fitted_years = tuple(int(value) for value in anchor["fitted_target_years"])
    if anchor_kind != "raw_fuxi" or fitted_years != TRAIN_YEARS:
        raise AuditContractError("raw-identity target contract changed")
    if (
        target_scale.shape != (6,)
        or not np.isfinite(target_scale).all()
        or np.any(target_scale <= 0.0)
    ):
        raise AuditContractError("raw-identity target scale is invalid")
    prediction = common.reconstruct(
        forecast.ensemble_mean,
        (alpha * residual).astype(np.float32),
        target_scale,
        support,
    )
    if prediction.shape != forecast.ensemble_mean.shape:
        raise AuditContractError("raw-identity prediction shape changed")
    if not np.isfinite(prediction[..., support]).all() or np.any(
        prediction[..., support] < 0.0
    ):
        raise AuditContractError("raw-identity prediction is invalid")
    return prediction.astype(np.float32), residual.astype(np.float32)


def verify_log_bias_identity(
    raw_run: Path,
    legacy_run: Path,
    raw_fuxi: np.ndarray,
    initializations: np.ndarray,
    stored_log_bias: np.ndarray,
) -> None:
    from fuxi_adapter.baselines import LogBiasCorrection, apply_log_bias_correction

    with np.load(raw_run / "models/log_bias_anchor.npz") as raw_anchor, np.load(
        legacy_run / "models/log_bias_anchor.npz"
    ) as legacy_anchor:
        if not np.array_equal(
            raw_anchor["lead_month_residual"],
            legacy_anchor["lead_month_residual"],
            equal_nan=True,
        ) or float(raw_anchor["shrinkage"]) != float(legacy_anchor["shrinkage"]):
            raise AuditContractError(
                "raw and legacy training-only log-bias fits differ"
            )
        correction = LogBiasCorrection(
            raw_anchor["lead_month_residual"], float(raw_anchor["shrinkage"])
        )
    rebuilt = apply_log_bias_correction(raw_fuxi, initializations, correction)
    valid = np.isfinite(stored_log_bias)
    if not np.allclose(rebuilt[valid], stored_log_bias[valid], rtol=0.0, atol=2.0e-6):
        raise AuditContractError(
            "stored audit log-bias differs from the training-only fit"
        )


def load_spatial_areas(
    latitude: np.ndarray, longitude: np.ndarray, support: np.ndarray
) -> dict[str, np.ndarray]:
    _require_hash(
        SPATIAL_SUPPORT_STORE / ".zmetadata",
        EXPECTED_SPATIAL_METADATA_SHA256,
        "spatial support metadata",
    )
    with xr.open_zarr(SPATIAL_SUPPORT_STORE, consolidated=True) as dataset:
        if not np.array_equal(dataset.latitude.values, latitude) or not np.array_equal(
            dataset.longitude.values, longitude
        ):
            raise AuditContractError("spatial support grid changed")
        arrays = {
            name: dataset[name].load().values for name in EXPECTED_SPATIAL_ARRAY_SHA256
        }
    for name, expected in EXPECTED_SPATIAL_ARRAY_SHA256.items():
        dtype = "<f8" if name.endswith("km2") else "<f4"
        actual = array_sha256(arrays[name], dtype)
        if actual != expected:
            raise AuditContractError(f"spatial array changed: {name}")
    cell_area = arrays["cell_area_km2"].astype(np.float64)
    areas = {"all_india": arrays["india_area_weight_km2"].astype(np.float64)}
    for region in EXPECTED_REGIONS[1:]:
        areas[region] = cell_area * arrays[f"{region}_fraction"].astype(np.float64)
    return {name: np.where(support, values, 0.0) for name, values in areas.items()}


def fixed_projection_weights(
    india_area_weight_km2: np.ndarray,
    support: np.ndarray,
    prediction_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Build forecast-time projection weights independent of truth/coverage."""

    area = np.asarray(india_area_weight_km2, dtype=np.float64)
    if area.shape != support.shape or prediction_shape[-2:] != support.shape:
        raise ValueError("fixed projection grid shape differs")
    fixed = np.where(support & np.isfinite(area) & (area > 0.0), area, 0.0)
    if int(np.count_nonzero(fixed)) != 171:
        raise AuditContractError("fixed projection support is not the frozen 171 cells")
    return np.broadcast_to(fixed[None, None], prediction_shape)


def project_log_offset_to_reference_mean(
    candidate: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    support: np.ndarray,
    *,
    iterations: int = 80,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply a nonnegative spatially uniform log offset without using truth.

    For each initialization and lead, solve for ``scale`` in
    ``max(scale * (1 + candidate) - 1, 0)`` so its fixed India-area mean
    equals the raw FuXi mean under the same forecast-time weights.  The caller
    must supply fixed weights; truth and observation coverage are not inputs.
    """

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape != weights.shape:
        raise ValueError("candidate, reference, and weights must have identical shapes")
    if candidate.ndim != 4 or support.shape != candidate.shape[-2:]:
        raise ValueError("projection expects [case, lead, latitude, longitude]")
    if iterations < 40:
        raise ValueError("at least 40 bisection iterations are required")
    result = np.full(candidate.shape, np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for case in range(candidate.shape[0]):
        for lead in range(candidate.shape[1]):
            valid = (
                support
                & np.isfinite(candidate[case, lead])
                & np.isfinite(reference[case, lead])
                & np.isfinite(weights[case, lead])
                & (weights[case, lead] > 0.0)
            )
            if int(valid.sum()) < 3:
                raise AuditContractError(
                    "projection has fewer than three weighted cells"
                )
            values = candidate[case, lead].copy()
            if np.any(values[valid] < -1.0e-7) or np.any(
                reference[case, lead][valid] < 0.0
            ):
                raise AuditContractError("projection received negative rainfall")
            values = np.maximum(values, 0.0)
            weight = weights[case, lead][valid]
            total = float(weight.sum(dtype=np.float64))
            target = float(
                np.sum(weight * reference[case, lead][valid], dtype=np.float64) / total
            )
            before = float(np.sum(weight * values[valid], dtype=np.float64) / total)
            if target <= 0.0:
                scale = 0.0
            else:
                low = 0.0
                high = 1.0

                def projected_mean(value: float) -> float:
                    projected = np.maximum(value * (1.0 + values[valid]) - 1.0, 0.0)
                    return float(np.sum(weight * projected, dtype=np.float64) / total)

                while projected_mean(high) < target:
                    high *= 2.0
                    if high > 1.0e6:
                        raise AuditContractError(
                            "projection scale could not bracket raw mean"
                        )
                for _ in range(iterations):
                    middle = 0.5 * (low + high)
                    if projected_mean(middle) < target:
                        low = middle
                    else:
                        high = middle
                scale = 0.5 * (low + high)
            projected = np.maximum(scale * (1.0 + values) - 1.0, 0.0)
            projected[~support] = np.nan
            result[case, lead] = projected
            after = float(np.sum(weight * projected[valid], dtype=np.float64) / total)
            rows.append(
                {
                    "case_index": case,
                    "lead_week": lead + 1,
                    "raw_target_mean_mm_day": target,
                    "raw_identity_mean_before_mm_day": before,
                    "projected_mean_mm_day": after,
                    "log_space_scale": scale,
                    "log_offset": float(np.log(scale)) if scale > 0.0 else -np.inf,
                    "absolute_mean_closure_mm_day": abs(after - target),
                    "zero_fraction_on_support": float(
                        np.mean(projected[support] == 0.0)
                    ),
                }
            )
    output = result.astype(np.float32)
    diagnostics = pd.DataFrame(rows)
    # Recheck closure after the float32 storage conversion.
    closures = []
    for case in range(output.shape[0]):
        for lead in range(output.shape[1]):
            valid = (
                support & (weights[case, lead] > 0.0) & np.isfinite(weights[case, lead])
            )
            weight = weights[case, lead][valid]
            target = np.sum(weight * reference[case, lead][valid]) / np.sum(weight)
            actual = np.sum(weight * output[case, lead][valid]) / np.sum(weight)
            closures.append(abs(float(actual - target)))
    diagnostics["float32_absolute_mean_closure_mm_day"] = closures
    if float(np.max(closures)) > 2.0e-6:
        raise AuditContractError(
            "float32 projected fields do not preserve the raw area mean"
        )
    if not np.isfinite(output[..., support]).all() or np.any(
        output[..., support] < 0.0
    ):
        raise AuditContractError("projected prediction is not finite and nonnegative")
    return output, diagnostics


def weighted_metrics(
    forecast: np.ndarray, observation: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    valid = (
        np.isfinite(forecast)
        & np.isfinite(observation)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    if int(valid.sum()) < 3:
        raise AuditContractError("fewer than three weighted cells")
    predicted = np.asarray(forecast[valid], dtype=np.float64)
    observed = np.asarray(observation[valid], dtype=np.float64)
    weights = np.asarray(weight[valid], dtype=np.float64)
    total = float(weights.sum(dtype=np.float64))
    error = predicted - observed
    p_centered = predicted - float(np.sum(weights * predicted) / total)
    o_centered = observed - float(np.sum(weights * observed) / total)
    denominator = float(
        np.sqrt(np.sum(weights * p_centered**2) * np.sum(weights * o_centered**2))
    )
    return {
        "acc": (
            float(np.sum(weights * p_centered * o_centered) / denominator)
            if denominator > 0.0
            else np.nan
        ),
        "rmse_mm_day": float(np.sqrt(np.sum(weights * error**2) / total)),
        "mae_mm_day": float(np.sum(weights * np.abs(error)) / total),
        "bias_mm_day": float(np.sum(weights * error) / total),
        "valid_cell_count": int(valid.sum()),
        "effective_area_km2": total,
    }


def score_predictions(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    climatology: np.ndarray,
    weekly_coverage: np.ndarray,
    areas: Mapping[str, np.ndarray],
    initializations: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.DatetimeIndex(initializations)
    for method in METHODS:
        prediction = predictions[method]
        for case, init in enumerate(dates):
            for lead in range(6):
                for region in EXPECTED_REGIONS:
                    weight = areas[region] * weekly_coverage[case, lead]
                    absolute = weighted_metrics(
                        prediction[case, lead], truth[case, lead], weight
                    )
                    anomaly = weighted_metrics(
                        prediction[case, lead] - climatology[case, lead],
                        truth[case, lead] - climatology[case, lead],
                        weight,
                    )
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "case_index": case,
                            "init": str(init.date()),
                            "year": int(init.year),
                            "lead_week": lead + 1,
                            "region": region,
                            "region_label": REGION_LABELS[region],
                            "rmse_mm_day": absolute["rmse_mm_day"],
                            "mae_mm_day": absolute["mae_mm_day"],
                            "bias_mm_day": absolute["bias_mm_day"],
                            "acc": anomaly["acc"],
                            "valid_cell_count": absolute["valid_cell_count"],
                            "effective_area_km2": absolute["effective_area_km2"],
                        }
                    )
    result = pd.DataFrame(rows)
    expected = len(METHODS) * 100 * 6 * len(EXPECTED_REGIONS)
    if (
        len(result) != expected
        or result.duplicated(["method", "init", "lead_week", "region"]).any()
    ):
        raise AuditContractError("case-score table is incomplete or duplicated")
    if result[["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]].isna().any().any():
        raise AuditContractError("case-score table contains non-finite metrics")
    return result


def summarize_metrics(cases: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = ["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]

    def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        result = frame.groupby(groups, as_index=False)[metrics].mean()
        counts = (
            frame.groupby(groups, as_index=False)
            .size()
            .rename(columns={"size": "case_lead_count"})
        )
        return result.merge(counts, on=groups, validate="one_to_one")

    india = cases.loc[cases.region.eq("all_india")]
    return {
        "summary_pooled.csv": aggregate(india, ["method", "method_label"]),
        "summary_by_year.csv": aggregate(india, ["year", "method", "method_label"]),
        "summary_by_lead.csv": aggregate(
            india, ["lead_week", "method", "method_label"]
        ),
        "summary_by_region.csv": aggregate(
            cases, ["region", "region_label", "method", "method_label"]
        ),
    }


def intensity_metrics(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    dynamic_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    forecast = np.stack([predictions[method] for method in METHODS]).astype(np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(dynamic_weights, dtype=np.float64)
    common_valid = (
        (weights > 0.0) & np.isfinite(truth64) & np.all(np.isfinite(forecast), axis=0)
    )
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, np.ndarray]] = {}
    for key, label, lower, upper in INTENSITY_STRATA:
        stratum = common_valid & (truth64 >= lower) & (truth64 < upper)
        aggregates[key] = {}
        denominator_by_case = np.sum(
            np.where(stratum, weights, 0.0), axis=(1, 2, 3), dtype=np.float64
        )
        if float(denominator_by_case.sum()) <= 0.0:
            raise AuditContractError(f"empty intensity stratum: {key}")
        aggregates[key]["denominator"] = denominator_by_case
        for method_index, method in enumerate(METHODS):
            error = forecast[method_index] - truth64
            squared_by_case = np.sum(
                np.where(stratum, weights * error**2, 0.0),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            absolute_by_case = np.sum(
                np.where(stratum, weights * np.abs(error), 0.0),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            bias_by_case = np.sum(
                np.where(stratum, weights * error, 0.0),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            truth_by_case = np.sum(
                np.where(stratum, weights * truth64, 0.0),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            prediction_by_case = np.sum(
                np.where(stratum, weights * forecast[method_index], 0.0),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            aggregates[key][f"{method}:squared"] = squared_by_case
            aggregates[key][f"{method}:absolute"] = absolute_by_case
            aggregates[key][f"{method}:bias"] = bias_by_case
            denominator = float(denominator_by_case.sum(dtype=np.float64))
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "stratum": key,
                    "stratum_label": label,
                    "lower_mm_day": lower,
                    "upper_mm_day": upper,
                    "cell_case_lead_count": int(np.count_nonzero(stratum)),
                    "dynamic_area_weight_sum_km2_case_lead": denominator,
                    "rmse_mm_day": float(np.sqrt(squared_by_case.sum() / denominator)),
                    "mae_mm_day": float(absolute_by_case.sum() / denominator),
                    "bias_mm_day": float(bias_by_case.sum() / denominator),
                    "truth_mean_mm_day": float(truth_by_case.sum() / denominator),
                    "prediction_mean_mm_day": float(
                        prediction_by_case.sum() / denominator
                    ),
                    "definition": (
                        "pooled cell-lead error stratified by verifying weekly-mean IMD; "
                        "all-India cell area x exact weekly IMD coverage weighted"
                    ),
                }
            )

    threshold_rows: list[dict[str, Any]] = []
    valid_weight = weights[common_valid]
    observed = truth64[common_valid]
    total = float(valid_weight.sum(dtype=np.float64))
    for method_index, method in enumerate(METHODS):
        predicted = forecast[method_index][common_valid]
        for threshold in THRESHOLDS_MM_DAY:
            observed_event = observed >= threshold
            predicted_event = predicted >= threshold
            hits = float(valid_weight[observed_event & predicted_event].sum())
            misses = float(valid_weight[observed_event & ~predicted_event].sum())
            false_alarms = float(valid_weight[~observed_event & predicted_event].sum())
            correct_negatives = float(
                valid_weight[~observed_event & ~predicted_event].sum()
            )
            random_hits = (hits + misses) * (hits + false_alarms) / total
            ets_denominator = hits + misses + false_alarms - random_hits
            threshold_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "threshold_mm_day": threshold,
                    "ets": (hits - random_hits) / ets_denominator,
                    "csi": hits / (hits + misses + false_alarms),
                    "pod": hits / (hits + misses),
                    "far": false_alarms / (hits + false_alarms),
                    "frequency_bias": (hits + false_alarms) / (hits + misses),
                    "deterministic_brier_score": (misses + false_alarms) / total,
                    "hits_weight": hits,
                    "misses_weight": misses,
                    "false_alarms_weight": false_alarms,
                    "correct_negatives_weight": correct_negatives,
                    "definition": (
                        "pooled deterministic weekly-mean threshold contingency; "
                        "all-India cell area x exact weekly IMD coverage weighted"
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows), aggregates


def year_stratified_circular_block_indices(
    initializations: np.ndarray,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    block_length: int = DEFAULT_BOOTSTRAP_BLOCK_LENGTH,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, dict[int, slice]]:
    """Sample circular blocks within each year, retaining all six leads.

    Uniform starts on a circular year make every initialization have the same
    marginal inclusion probability, including dates near the JJAS boundaries.
    The final block is truncated only after wrapping, so each year contributes
    exactly its original number of initialization dates to every draw.
    """

    if draws < 1_000:
        raise ValueError("at least 1,000 bootstrap draws are required")
    if block_length <= 0:
        raise ValueError("bootstrap block length must be positive")
    dates = validate_initializations(initializations)
    years = dates.year.to_numpy()
    generator = np.random.default_rng(seed)
    offsets = np.arange(block_length, dtype=np.int64)[None, None, :]
    pieces: list[np.ndarray] = []
    slices: dict[int, slice] = {}
    cursor = 0
    for year in AUDIT_YEARS:
        positions = np.flatnonzero(years == year)
        count = EXPECTED_YEAR_COUNTS[year]
        if len(positions) != count or block_length > count:
            raise AuditContractError(f"invalid bootstrap contract for {year}")
        block_count = (count + block_length - 1) // block_length
        starts = generator.integers(0, count, size=(draws, block_count), endpoint=False)
        local = ((starts[:, :, None] + offsets) % count).reshape(draws, -1)[:, :count]
        pieces.append(positions[local])
        slices[year] = slice(cursor, cursor + count)
        cursor += count
    indices = np.concatenate(pieces, axis=1).astype(np.int16)
    if indices.shape != (draws, 100):
        raise AuditContractError("bootstrap index matrix shape changed")
    return indices, slices


def bootstrap_index_diagnostics(
    initializations: np.ndarray,
    indices: np.ndarray,
    year_slices: Mapping[int, slice],
    block_length: int,
) -> dict[str, Any]:
    dates = validate_initializations(initializations)
    years = dates.year.to_numpy()
    diagnostics: dict[str, Any] = {
        "draws": int(indices.shape[0]),
        "starts_per_draw": int(indices.shape[1]),
        "block_length_initializations": int(block_length),
        "year_stratified": True,
        "circular_within_year": True,
        "equal_marginal_inclusion_by_design": True,
        "no_year_crossing": True,
        "all_six_leads_retained_per_start": True,
        "year_draw_counts": {},
        "mean_multiplicity_per_initialization": {},
        "maximum_absolute_mean_multiplicity_deviation_from_one": 0.0,
    }
    all_mean_multiplicities = np.empty(len(initializations), dtype=np.float64)
    for year in AUDIT_YEARS:
        positions = np.flatnonzero(years == year)
        segment = indices[:, year_slices[year]]
        if not np.isin(segment, positions).all():
            raise AuditContractError(f"bootstrap draws cross the {year} stratum")
        diagnostics["year_draw_counts"][str(year)] = int(segment.shape[1])
        local_lookup = np.full(len(initializations), -1, dtype=np.int16)
        local_lookup[positions] = np.arange(len(positions), dtype=np.int16)
        local_segment = local_lookup[segment]
        for start in range(0, segment.shape[1], block_length):
            block = local_segment[
                :, start : min(start + block_length, segment.shape[1])
            ]
            if block.shape[1] > 1 and not np.all(
                np.mod(np.diff(block, axis=1), len(positions)) == 1
            ):
                raise AuditContractError(
                    f"bootstrap contains a nonconsecutive circular {year} block"
                )
        multiplicity = np.bincount(segment.reshape(-1), minlength=len(initializations))[
            positions
        ] / float(indices.shape[0])
        all_mean_multiplicities[positions] = multiplicity
        diagnostics["mean_multiplicity_per_initialization"].update(
            {
                str(dates[index].date()): float(multiplicity[local_index])
                for local_index, index in enumerate(positions)
            }
        )
    diagnostics["mean_multiplicity_across_initializations"] = float(
        np.mean(all_mean_multiplicities)
    )
    diagnostics["minimum_mean_multiplicity"] = float(np.min(all_mean_multiplicities))
    diagnostics["maximum_mean_multiplicity"] = float(np.max(all_mean_multiplicities))
    diagnostics["maximum_absolute_mean_multiplicity_deviation_from_one"] = float(
        np.max(np.abs(all_mean_multiplicities - 1.0))
    )
    if not np.isclose(
        diagnostics["mean_multiplicity_across_initializations"],
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AuditContractError("bootstrap mean inclusion multiplicity is not one")
    return diagnostics


def _bootstrap_counts(indices: np.ndarray, case_count: int) -> np.ndarray:
    counts = np.zeros((indices.shape[0], case_count), dtype=np.int16)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    np.add.at(counts, (rows, indices.reshape(-1)), 1)
    return counts


def _metric_cube(cases: pd.DataFrame, region: str, metric: str) -> np.ndarray:
    cube = np.empty((len(METHODS), 100, 6), dtype=np.float64)
    selected = cases.loc[cases.region.eq(region)]
    for method_index, method in enumerate(METHODS):
        pivot = selected.loc[selected.method.eq(method)].pivot(
            index="case_index", columns="lead_week", values=metric
        )
        cube[method_index] = pivot.loc[np.arange(100), np.arange(1, 7)].to_numpy(
            dtype=np.float64
        )
    if not np.isfinite(cube).all():
        raise AuditContractError(f"non-finite bootstrap cube: {region}/{metric}")
    return cube


def paired_block_effects(
    cases: pd.DataFrame,
    indices: np.ndarray,
    year_slices: Mapping[int, slice],
    block_length: int,
) -> pd.DataFrame:
    method_index = {method: index for index, method in enumerate(METHODS)}
    scopes: list[tuple[str, str, str, np.ndarray, np.ndarray]] = [
        ("pooled", "W1-W6", "all_india", np.arange(6), indices)
    ]
    for lead in range(6):
        scopes.append(
            ("lead", f"W{lead + 1}", "all_india", np.asarray([lead]), indices)
        )
    for year in AUDIT_YEARS:
        scopes.append(
            (
                "year",
                str(year),
                "all_india",
                np.arange(6),
                indices[:, year_slices[year]],
            )
        )
    for region in EXPECTED_REGIONS[1:]:
        scopes.append(("region", REGION_LABELS[region], region, np.arange(6), indices))

    rows: list[dict[str, Any]] = []
    for scope_type, scope, region, leads, sample in scopes:
        counts = _bootstrap_counts(sample, 100)
        sampled_starts = sample.shape[1]
        for metric in ("rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"):
            cube = _metric_cube(cases, region, metric)
            per_case = cube[:, :, leads].mean(axis=2)
            point_means = (
                per_case[:, np.unique(sample)].mean(axis=1)
                if scope_type == "year"
                else per_case.mean(axis=1)
            )
            draw_means = (counts @ per_case.T) / float(sampled_starts)
            for candidate, baseline in COMPARISONS:
                candidate_index = method_index[candidate]
                baseline_index = method_index[baseline]
                if metric in ("rmse_mm_day", "mae_mm_day"):
                    effect = point_means[baseline_index] - point_means[candidate_index]
                    distribution = (
                        draw_means[:, baseline_index] - draw_means[:, candidate_index]
                    )
                    relative = 100.0 * effect / point_means[baseline_index]
                    relative_distribution = (
                        100.0 * distribution / draw_means[:, baseline_index]
                    )
                    definition = (
                        f"baseline minus candidate {metric}; positive favors candidate"
                    )
                elif metric == "acc":
                    effect = point_means[candidate_index] - point_means[baseline_index]
                    distribution = (
                        draw_means[:, candidate_index] - draw_means[:, baseline_index]
                    )
                    relative = np.nan
                    relative_distribution = np.full(distribution.shape, np.nan)
                    definition = (
                        "candidate minus baseline ACC; positive favors candidate"
                    )
                else:
                    effect = abs(point_means[baseline_index]) - abs(
                        point_means[candidate_index]
                    )
                    distribution = np.abs(draw_means[:, baseline_index]) - np.abs(
                        draw_means[:, candidate_index]
                    )
                    relative = np.nan
                    relative_distribution = np.full(distribution.shape, np.nan)
                    definition = "absolute pooled bias baseline minus candidate; positive favors candidate"
                definition = (
                    f"{definition}; paired year-stratified circular moving-block "
                    f"bootstrap with actual block length {block_length} initializations"
                )
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "region": region,
                        "candidate": candidate,
                        "baseline": baseline,
                        "source_metric": metric,
                        "effect": float(effect),
                        "ci_lower_2p5": float(np.quantile(distribution, 0.025)),
                        "ci_upper_97p5": float(np.quantile(distribution, 0.975)),
                        "bootstrap_probability_improved": float(
                            np.mean(distribution > 0.0)
                        ),
                        "relative_effect_pct": float(relative),
                        "relative_ci_lower_2p5": (
                            float(np.nanquantile(relative_distribution, 0.025))
                            if np.isfinite(relative_distribution).any()
                            else np.nan
                        ),
                        "relative_ci_upper_97p5": (
                            float(np.nanquantile(relative_distribution, 0.975))
                            if np.isfinite(relative_distribution).any()
                            else np.nan
                        ),
                        "n_starts": sampled_starts,
                        "n_leads_per_start": len(leads),
                        "definition": definition,
                    }
                )
    return pd.DataFrame(rows)


def intensity_block_effects(
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    indices: np.ndarray,
    block_length: int,
) -> pd.DataFrame:
    counts = _bootstrap_counts(indices, 100)
    rows: list[dict[str, Any]] = []
    for stratum, values in aggregates.items():
        denominator = np.asarray(values["denominator"], dtype=np.float64)
        point_denominator = float(denominator.sum())
        draw_denominator = counts @ denominator
        for candidate, baseline in COMPARISONS:
            for metric, numerator_kind in (
                ("rmse_mm_day", "squared"),
                ("mae_mm_day", "absolute"),
                ("bias_mm_day", "bias"),
            ):
                candidate_num = np.asarray(
                    values[f"{candidate}:{numerator_kind}"], dtype=np.float64
                )
                baseline_num = np.asarray(
                    values[f"{baseline}:{numerator_kind}"], dtype=np.float64
                )
                if metric == "rmse_mm_day":
                    candidate_point = float(
                        np.sqrt(candidate_num.sum() / point_denominator)
                    )
                    baseline_point = float(
                        np.sqrt(baseline_num.sum() / point_denominator)
                    )
                    candidate_draw = np.sqrt(
                        (counts @ candidate_num) / draw_denominator
                    )
                    baseline_draw = np.sqrt((counts @ baseline_num) / draw_denominator)
                    effect = baseline_point - candidate_point
                    distribution = baseline_draw - candidate_draw
                elif metric == "mae_mm_day":
                    candidate_point = float(candidate_num.sum() / point_denominator)
                    baseline_point = float(baseline_num.sum() / point_denominator)
                    candidate_draw = (counts @ candidate_num) / draw_denominator
                    baseline_draw = (counts @ baseline_num) / draw_denominator
                    effect = baseline_point - candidate_point
                    distribution = baseline_draw - candidate_draw
                else:
                    candidate_point = float(candidate_num.sum() / point_denominator)
                    baseline_point = float(baseline_num.sum() / point_denominator)
                    candidate_draw = (counts @ candidate_num) / draw_denominator
                    baseline_draw = (counts @ baseline_num) / draw_denominator
                    effect = abs(baseline_point) - abs(candidate_point)
                    distribution = np.abs(baseline_draw) - np.abs(candidate_draw)
                rows.append(
                    {
                        "stratum": stratum,
                        "candidate": candidate,
                        "baseline": baseline,
                        "source_metric": metric,
                        "effect": float(effect),
                        "ci_lower_2p5": float(np.quantile(distribution, 0.025)),
                        "ci_upper_97p5": float(np.quantile(distribution, 0.975)),
                        "bootstrap_probability_improved": float(
                            np.mean(distribution > 0.0)
                        ),
                        "n_starts": 100,
                        "n_leads_per_start": 6,
                        "definition": (
                            "paired year-stratified circular moving-block bootstrap with "
                            f"actual block length {block_length} initializations; "
                            "all six leads and dynamically weighted cells retained"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def save_predictions(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    frozen: FrozenAuditFields,
    truth: np.ndarray,
    climatology: np.ndarray,
    weekly_coverage: np.ndarray,
    india_area: np.ndarray,
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack([predictions[method] for method in METHODS]).astype(
                    np.float32
                ),
            ),
            "truth_imd": (
                ("init", "lead_week", "latitude", "longitude"),
                truth.astype(np.float32),
            ),
            "fixed_imd_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                climatology.astype(np.float32),
            ),
            "weekly_imd_coverage": (
                ("init", "lead_week", "latitude", "longitude"),
                weekly_coverage.astype(np.float32),
            ),
            "india_area_weight_km2": (
                ("latitude", "longitude"),
                india_area.astype(np.float64),
            ),
            "adapter_support": (("latitude", "longitude"), frozen.support),
        },
        coords={
            "method": list(METHODS),
            "init": frozen.initializations.astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": frozen.latitude,
            "longitude": frozen.longitude,
        },
        attrs={
            "scientific_status": "post-hoc 2022-2024 development audit; no retraining or retuning",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "projection": (
                "nonnegative spatially uniform log offset preserving each raw FuXi "
                "fixed all-India-area mean on frozen adapter support; independent of "
                "rainfall truth and weekly observation coverage"
            ),
            "final_2025_initialization_year": "quarantined and not read",
            "units": "mm day-1",
        },
    )
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={"prediction": {"chunks": (1, 25, 1, 27, 27)}},
    )


def verify_saved_prediction_store(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    frozen: FrozenAuditFields,
    truth: np.ndarray,
    climatology: np.ndarray,
    weekly_coverage: np.ndarray,
    india_area: np.ndarray,
) -> dict[str, Any]:
    """Prove coordinates, legacy fields, and new methods survived Zarr round-trip."""

    with xr.open_zarr(path, consolidated=True) as dataset:
        methods = tuple(str(value) for value in dataset.method.values.tolist())
        if methods != METHODS:
            raise AuditContractError("saved prediction method coordinate changed")
        if not np.array_equal(
            dataset.init.values.astype("datetime64[D]"), frozen.initializations
        ):
            raise AuditContractError(
                "saved prediction initialization coordinate changed"
            )
        if not np.array_equal(
            dataset.lead_week.values, np.arange(1, 7, dtype=np.int16)
        ):
            raise AuditContractError("saved prediction lead coordinate changed")
        if not np.array_equal(
            dataset.latitude.values, frozen.latitude
        ) or not np.array_equal(dataset.longitude.values, frozen.longitude):
            raise AuditContractError("saved prediction grid changed")
        if not np.array_equal(
            dataset.adapter_support.values.astype(bool), frozen.support
        ):
            raise AuditContractError("saved adapter support changed")
        saved_truth = dataset.truth_imd.load().values.astype(np.float32)
        saved_climatology = dataset.fixed_imd_climatology.load().values.astype(
            np.float32
        )
        saved_coverage = dataset.weekly_imd_coverage.load().values.astype(np.float32)
        saved_area = dataset.india_area_weight_km2.load().values.astype(np.float64)
        saved_predictions = {
            method: dataset.prediction.sel({"method": method})
            .load()
            .values.astype(np.float32)
            for method in METHODS
        }
    for method in METHODS:
        if not np.array_equal(
            saved_predictions[method], predictions[method], equal_nan=True
        ):
            raise AuditContractError(
                f"saved prediction changed on round-trip: {method}"
            )
    for name, saved, expected in (
        ("truth", saved_truth, truth),
        ("climatology", saved_climatology, climatology),
        ("weekly coverage", saved_coverage, weekly_coverage),
        ("India area", saved_area, india_area),
    ):
        if not np.array_equal(saved, expected, equal_nan=True):
            raise AuditContractError(f"saved {name} changed on round-trip")
    return {
        "verified": True,
        "coordinates_exact": True,
        "legacy_fields_exact": True,
        "all_methods_exact": True,
        "method_array_sha256": {
            method: array_sha256(saved_predictions[method], "<f4") for method in METHODS
        },
    }


def write_results(
    output: Path,
    summaries: Mapping[str, pd.DataFrame],
    effects: pd.DataFrame,
    bootstrap_draws: int,
    bootstrap_block_length: int,
    noncanonical_smoke: bool,
) -> None:
    pooled = summaries["summary_pooled.csv"].set_index("method")
    status_notice = (
        "> **NONCANONICAL SMOKE:** integration evidence only; scientifically ineligible. "
        "No retraining, retuning, or 2025 data access occurred."
        if noncanonical_smoke
        else "> Canonical post-hoc E2 development evidence. No retraining, retuning, or "
        "2025 data access occurred; this is not an untouched final test."
    )
    lines = [
        "# Frozen raw-identity adapter: matched 2022-2024 development audit",
        "",
        status_notice,
        "",
        "## Pooled all-India W1-W6",
        "",
        "| Method | RMSE | MAE | Bias | ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = pooled.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.rmse_mm_day:.3f} | {row.mae_mm_day:.3f} | "
            f"{row.bias_mm_day:.3f} | {row.acc:.3f} |"
        )
    headline = effects.loc[
        effects.scope_type.eq("pooled")
        & effects.candidate.eq("raw_identity_raw_mean_preserved")
        & effects.baseline.eq("raw_fuxi")
        & effects.source_metric.eq("rmse_mm_day")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Paired descriptive uncertainty",
            "",
            f"Projected raw-identity RMSE skill versus raw FuXi: {headline.relative_effect_pct:+.2f}% "
            f"(95% block interval {headline.relative_ci_lower_2p5:+.2f}% to "
            f"{headline.relative_ci_upper_97p5:+.2f}%).",
            "",
            f"Intervals use {bootstrap_draws:,} shared paired, year-stratified, circular "
            f"moving-block resamples with actual block length {bootstrap_block_length} "
            "initialization dates. They are "
            "descriptive conditional on these "
            "three audit seasons, not p-values or an untouched final test.",
            "",
            "The raw-mean projection is a post-hoc diagnostic fixed after the original audit. "
            "Its weights are forecast-time India area and the frozen adapter support only; "
            "neither rainfall truth nor weekly observation coverage enters the prediction.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_hashes(output: Path, prediction_store: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if prediction_store in path.parents:
            continue
        hashes[str(path.relative_to(output))] = sha256_file(path)
    hashes["predictions.zarr"] = sha256_tree(prediction_store)
    return hashes


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    noncanonical_smoke = bool(args.noncanonical_smoke)
    canonical = not noncanonical_smoke
    audit_run = args.audit_run.resolve()
    raw_run = args.raw_identity_run.resolve()
    provenance = validate_input_artifacts(audit_run, raw_run)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "requested_canonical": canonical,
                    "requested_noncanonical_smoke": noncanonical_smoke,
                    "bootstrap": {
                        "draws": args.bootstrap_draws,
                        "block_length_initializations": args.bootstrap_block_length,
                        "seed": args.bootstrap_seed,
                    },
                    **provenance,
                },
                indent=2,
            ),
            flush=True,
        )
        return Path()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    for directory in (staging, staging / "metrics", staging / "code"):
        directory.mkdir(parents=True, exist_ok=False)

    audit_manifest = _load_json(audit_run / "manifest.json")
    raw_manifest = _load_json(raw_run / "manifest.json")
    raw_selection = _load_json(raw_run / "selection.json")
    frozen = load_frozen_audit_fields(audit_run)
    base, common, engine = load_frozen_model_modules(raw_run)
    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    print("Loading only the pinned 2022-2024 operational predictors...", flush=True)
    forecast, t2m, fuxi_stores = load_operational_predictors(
        audit_manifest, frozen, base
    )
    print(
        "Rebuilding the pinned 2002-2017 training-only IMD climatology...", flush=True
    )
    training_climatology, training_stores = load_training_climatology(
        audit_manifest, frozen, base
    )
    truth, climatology, weekly_coverage, verification_stores = load_verification_fields(
        audit_manifest, frozen, forecast, training_climatology, base
    )
    normalization = _load_json(raw_run / "normalization.json")
    features, rebuilt_climatology = build_features(
        forecast, t2m, training_climatology, normalization, frozen.support, base
    )
    if not np.allclose(
        rebuilt_climatology, climatology, rtol=0.0, atol=2.0e-6, equal_nan=True
    ):
        raise AuditContractError("feature and verification climatologies differ")

    legacy_run = Path(provenance["legacy_run"])
    verify_log_bias_identity(
        raw_run,
        legacy_run,
        frozen.raw_fuxi,
        frozen.initializations,
        frozen.log_bias,
    )
    print(
        "Applying the frozen three-seed raw-identity adapter without retraining...",
        flush=True,
    )
    raw_identity, residual = infer_raw_identity(
        raw_run,
        raw_manifest,
        raw_selection,
        forecast,
        features,
        frozen.support,
        common,
        engine,
    )
    del features
    areas = load_spatial_areas(frozen.latitude, frozen.longitude, frozen.support)
    projection_weights = fixed_projection_weights(
        areas["all_india"], frozen.support, raw_identity.shape
    )
    projected, projection_diagnostics = project_log_offset_to_reference_mean(
        raw_identity,
        frozen.raw_fuxi,
        projection_weights,
        frozen.support,
    )
    projection_diagnostics.insert(
        1,
        "init",
        np.repeat(frozen.initializations.astype(str), 6),
    )
    projection_diagnostics.insert(
        2, "year", pd.DatetimeIndex(projection_diagnostics.init).year
    )

    predictions = {
        "raw_fuxi": frozen.raw_fuxi,
        "log_bias": frozen.log_bias,
        "legacy_anchored_adapter": frozen.legacy_adapter,
        "raw_identity": raw_identity,
        "raw_identity_raw_mean_preserved": projected,
    }
    print(
        "Scoring pooled, year, lead, region, and intensity diagnostics...", flush=True
    )
    cases = score_predictions(
        predictions,
        truth,
        climatology,
        weekly_coverage,
        areas,
        frozen.initializations,
    )
    summaries = summarize_metrics(cases)
    dynamic_scoring_weights = areas["all_india"][None, None] * weekly_coverage
    intensity, thresholds, intensity_aggregates = intensity_metrics(
        predictions, truth, dynamic_scoring_weights
    )
    bootstrap_indices, year_slices = year_stratified_circular_block_indices(
        frozen.initializations,
        draws=args.bootstrap_draws,
        block_length=args.bootstrap_block_length,
        seed=args.bootstrap_seed,
    )
    bootstrap_diagnostics = bootstrap_index_diagnostics(
        frozen.initializations,
        bootstrap_indices,
        year_slices,
        args.bootstrap_block_length,
    )
    effects = paired_block_effects(
        cases,
        bootstrap_indices,
        year_slices,
        args.bootstrap_block_length,
    )
    intensity_effects = intensity_block_effects(
        intensity_aggregates,
        bootstrap_indices,
        args.bootstrap_block_length,
    )

    cases.to_csv(staging / "metrics/case_metrics.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(staging / "metrics" / name, index=False)
    intensity.to_csv(staging / "metrics/intensity_strata_metrics.csv", index=False)
    thresholds.to_csv(staging / "metrics/threshold_metrics.csv", index=False)
    effects.to_csv(staging / "metrics/paired_block_bootstrap_effects.csv", index=False)
    intensity_effects.to_csv(
        staging / "metrics/intensity_paired_block_bootstrap_effects.csv", index=False
    )
    projection_diagnostics.to_csv(
        staging / "metrics/projection_mean_closure.csv", index=False
    )
    np.save(
        staging / "metrics/bootstrap_indices.npy", bootstrap_indices, allow_pickle=False
    )
    np.save(staging / "metrics/raw_identity_residual.npy", residual, allow_pickle=False)
    (staging / "metrics/bootstrap_diagnostics.json").write_text(
        json.dumps(bootstrap_diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    prediction_store = staging / "predictions.zarr"
    save_predictions(
        prediction_store,
        predictions,
        frozen,
        truth,
        climatology,
        weekly_coverage,
        areas["all_india"],
    )
    prediction_roundtrip = verify_saved_prediction_store(
        prediction_store,
        predictions,
        frozen,
        truth,
        climatology,
        weekly_coverage,
        areas["all_india"],
    )
    write_results(
        staging,
        summaries,
        effects,
        args.bootstrap_draws,
        args.bootstrap_block_length,
        noncanonical_smoke,
    )
    shutil.copy2(Path(__file__), staging / "code" / Path(__file__).name)

    manifest = {
        "status": "complete" if canonical else "complete_noncanonical_smoke",
        "canonical": canonical,
        "scientific_eligible": canonical,
        "smoke": noncanonical_smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "scientific_status": (
            "post-hoc matched 2022-2024 canonical E2 development audit; no retraining, "
            "retuning, or untouched-final-test claim"
            if canonical
            else "noncanonical smoke/integration audit; scientifically ineligible; no "
            "retraining, retuning, or untouched-final-test claim"
        ),
        "experiment_role": "E2 frozen raw-identity matched audit",
        "audit_years": list(AUDIT_YEARS),
        "audit_counts": {
            str(key): value for key, value in EXPECTED_YEAR_COUNTS.items()
        },
        "final_initialization_year_quarantined": 2025,
        "final_2025_store_opened": False,
        "methods": list(METHODS),
        "raw_identity_selection": {
            "model": raw_selection["selected_model"],
            "alpha": raw_selection["selected_alpha"],
            "training_anchor": raw_selection["training_anchor"],
            "uses_fitted_log_bias_in_neural_training": False,
            "train_years": list(TRAIN_YEARS),
            "validation_years": list(VALIDATION_YEARS),
            "retrained_for_audit": False,
            "retuned_on_audit": False,
        },
        "projection": {
            "kind": "nonnegative spatially uniform log-space offset",
            "target": "raw FuXi per-case/per-lead all-India weighted mean",
            "weighting_contract": "fixed_forecast_time_india_area_x_frozen_adapter_support",
            "weights": "fixed india_area_weight_km2 masked by frozen adapter_support",
            "support_cells": 171,
            "uses_weekly_imd_coverage": False,
            "uses_observed_rainfall_values": False,
            "post_hoc": True,
            "operational_claim": False,
            "maximum_float32_closure_mm_day": float(
                projection_diagnostics.float32_absolute_mean_closure_mm_day.max()
            ),
        },
        "metric_contract": {
            "errors": "mean case-wise area x exact-weekly-coverage weighted RMSE, MAE, bias",
            "acc": "case-wise weighted spatial ACC after common fixed 2002-2017 IMD climatology",
            "intensity": "pooled weekly cell-lead metrics using area x exact weekly IMD coverage",
            "weekly_imd_coverage_role": "scoring only; never used to construct a forecast",
        },
        "bootstrap": {
            "method": (
                "paired year-stratified circular moving blocks with actual block length "
                f"{args.bootstrap_block_length} initializations"
            ),
            "draws": args.bootstrap_draws,
            "block_length_initializations": args.bootstrap_block_length,
            "seed": args.bootstrap_seed,
            "canonical_contract": {
                "draws": DEFAULT_BOOTSTRAP_DRAWS,
                "block_length_initializations": DEFAULT_BOOTSTRAP_BLOCK_LENGTH,
                "seed": DEFAULT_BOOTSTRAP_SEED,
            },
            "all_six_leads_retained": True,
            "shared_indices_across_all_methods_metrics_leads_regions_and_intensities": True,
            "interpretation": "descriptive percentile intervals conditional on three audit seasons",
            "diagnostics": bootstrap_diagnostics,
        },
        "global_pretraining": {
            "included": False,
            "reason": (
                "E2 is artifact-bound to the frozen raw-identity and legacy adapters. A global "
                "candidate is forbidden here unless its separate E0 promotion gate passes and a "
                "new evaluation version explicitly binds that artifact."
            ),
        },
        "prediction_store_roundtrip": prediction_roundtrip,
        "atomic_output": {
            "enabled": True,
            "contract": "write to sibling dot-staging directory; rename to final path only after all checks",
            "failed_staging_is_noncanonical": True,
        },
        "input_provenance": provenance,
        "source_stores_opened": {
            "operational_fuxi_2022_2024": list(fuxi_stores),
            "training_imd_2002_2017": list(training_stores),
            "verification_imd_2022_2024": list(verification_stores),
            "spatial_support": str(SPATIAL_SUPPORT_STORE),
        },
        "array_sha256": {
            "raw_identity": array_sha256(raw_identity, "<f4"),
            "raw_identity_raw_mean_preserved": array_sha256(projected, "<f4"),
            "raw_identity_residual": array_sha256(residual, "<f4"),
        },
        "array_sha256_contract": (
            "sha256 of contiguous C-order little-endian float32 raw bytes; no header"
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "artifacts": {},
    }
    manifest["artifacts"] = _artifact_hashes(staging, prediction_store)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(output)
    print((output / "RESULTS.md").read_text(encoding="utf-8"), flush=True)
    mode_label = "CANONICAL" if canonical else "NONCANONICAL_SMOKE"
    print(
        f"PASS [{mode_label}]: frozen raw-identity 2022-2024 audit: {output}",
        flush=True,
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", type=Path, default=DEFAULT_AUDIT_RUN)
    parser.add_argument(
        "--raw-identity-run", type=Path, default=DEFAULT_RAW_IDENTITY_RUN
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--bootstrap-block-length", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_LENGTH
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--noncanonical-smoke",
        action="store_true",
        help=(
            "permit noncanonical bootstrap settings for integration testing; outputs are "
            "marked smoke=true and scientific_eligible=false"
        ),
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    if args.bootstrap_draws < 1_000:
        parser.error("--bootstrap-draws must be at least 1000")
    if args.bootstrap_block_length <= 0:
        parser.error("--bootstrap-block-length must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be nonnegative")
    canonical_bootstrap = (
        args.bootstrap_draws == DEFAULT_BOOTSTRAP_DRAWS
        and args.bootstrap_block_length == DEFAULT_BOOTSTRAP_BLOCK_LENGTH
        and args.bootstrap_seed == DEFAULT_BOOTSTRAP_SEED
    )
    if not canonical_bootstrap and not args.noncanonical_smoke:
        parser.error(
            "alternate bootstrap settings require --noncanonical-smoke; the canonical "
            f"contract is draws={DEFAULT_BOOTSTRAP_DRAWS}, "
            f"block={DEFAULT_BOOTSTRAP_BLOCK_LENGTH}, seed={DEFAULT_BOOTSTRAP_SEED}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
