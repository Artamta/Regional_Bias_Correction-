#!/usr/bin/env python3
"""Pretrain the India FuXi residual adapter on deterministic global patches.

This module deliberately separates *representation pretraining* from the India
experiment.  It reads the existing global FuXi/IMERG annual caches, fits every
target-derived quantity on an early training split, and trains the exact
``FixedClimatologyAllLeadUNet`` backbone used by the India adapter on 27x27
patches.  The resulting checkpoint is a transfer artifact: India fine-tuning
must refit its own IMD normalization, climatology, anchor, and target scale.

The scientific safety boundary is enforced in code:

* no cache from 2018 or later can be opened;
* annual metadata and initialization dates are read before target arrays;
* D1--D42 overlap is purged before any excluded target chunk is indexed;
* all preprocessing is fitted on the purged global training references only;
* patch schedules are deterministic, longitude wraps, and latitude never does;
* epoch zero is the exact global log-bias anchor and remains a valid fallback;
* checkpoints and manifests are published atomically.

The global cache has no T2M field.  Channel 11 is therefore an explicit zero
placeholder during pretraining.  Its first-convolution weights are also zero,
so India fine-tuning can learn T2M without inheriting arbitrary random weights.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import zarr

from project_paths import NEURAL_ADAPTER_SRC, RESULTS_ROOT


if str(NEURAL_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_ADAPTER_SRC))

from fuxi_adapter.models import FixedClimatologyAllLeadUNet  # noqa: E402
import fuxi_adapter.models as adapter_models  # noqa: E402


SCHEMA_VERSION = 1
MAXIMUM_ALLOWED_SOURCE_YEAR = 2017
N_WEEK = 6
N_STAT = 8
N_LAT = 121
N_LON = 240
PATCH_SIZE = 27
BACKBONE_CHANNELS = 11
STAT_NAMES = (
    "mean",
    "std",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "wet_fraction",
)
FEATURE_NAMES = (
    "log_fuxi_mean",
    "log_fuxi_spread",
    "log_imerg_calendar_climatology",
    "patch_relative_latitude",
    "patch_relative_longitude",
    "season_sin",
    "season_cos",
    "lead_week",
    "training_observation_support",
    "explicit_log_fuxi_minus_imerg_climatology",
    "fuxi_t2m_weekly_zero_placeholder",
)
LEAD_WINDOWS = (
    "D1-7",
    "D8-14",
    "D15-21",
    "D22-28",
    "D29-35",
    "D36-42",
)
NORMALIZED_FEATURE_NAMES = (
    "log_fuxi_mean",
    "log_fuxi_spread",
    "log_imerg_calendar_climatology",
    "explicit_log_fuxi_minus_imerg_climatology",
)
TRANSFER_RESET_KEYS = (
    "backbone.residual_head.weight",
    "backbone.residual_head.bias",
)
GLOBAL_CACHE_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/global_tp_adapter/cache/annual"
)
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "fuxi_global_patch_pretraining"


class DataContractError(RuntimeError):
    """Raised when a cache or split violates the frozen scientific contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GlobalPatchContract:
    """Immutable, hashable contract for one global pretraining run."""

    mode: str
    cache_root: str
    fit_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    patch_size: int = PATCH_SIZE
    patches_per_case: int = 1
    patch_seed: int = 20260820
    seed: int = 42
    fit_case_limit: int = 0
    validation_case_limit: int = 0
    minimum_observation_fraction: float = 0.95
    anchor_shrinkage: float = 10.0
    base_channels: int = 16
    dropout: float = 0.30
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 2.0e-3
    smooth_l1_beta: float = 1.0
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'")
        if not self.fit_years or not self.validation_years:
            raise ValueError("fit and validation years must both be non-empty")
        if len(set(self.fit_years)) != len(self.fit_years) or len(
            set(self.validation_years)
        ) != len(self.validation_years):
            raise ValueError("split years must be unique")
        if set(self.fit_years) & set(self.validation_years):
            raise ValueError("fit and validation years must be disjoint")
        all_years = self.fit_years + self.validation_years
        if max(all_years) > MAXIMUM_ALLOWED_SOURCE_YEAR:
            raise ValueError(
                "global pretraining is hard-blocked from opening 2018+ caches"
            )
        if min(all_years) < 2002:
            raise ValueError("the prepared global cache starts in 2002")
        if max(self.fit_years) >= min(self.validation_years):
            raise ValueError("all fit years must precede all validation years")
        if self.patch_size != PATCH_SIZE:
            raise ValueError("the India-compatible patch size is fixed at 27")
        if self.patches_per_case < 1:
            raise ValueError("patches_per_case must be positive")
        if self.fit_case_limit < 0 or self.validation_case_limit < 0:
            raise ValueError("case limits must be nonnegative")
        if not 0.0 < self.minimum_observation_fraction <= 1.0:
            raise ValueError("minimum_observation_fraction must be in (0, 1]")
        if not np.isfinite(self.anchor_shrinkage) or self.anchor_shrinkage < 0.0:
            raise ValueError("anchor_shrinkage must be finite and nonnegative")
        if self.base_channels != 16:
            raise ValueError("exact India compatibility requires base_channels=16")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size < 1 or self.epochs < 0 or self.patience < 1:
            raise ValueError("invalid batch/epoch/patience configuration")
        positive = (
            self.learning_rate,
            self.weight_decay,
            self.smooth_l1_beta,
            self.gradient_clip,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("optimizer and loss scales must be finite and positive")

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "maximum_allowed_source_year": MAXIMUM_ALLOWED_SOURCE_YEAR,
                "feature_names": list(FEATURE_NAMES),
                "normalized_feature_names": list(NORMALIZED_FEATURE_NAMES),
                "target_units": "standardized log1p rainfall residual",
                "physical_units": "mm day-1 weekly mean rate",
                "lead_windows": list(LEAD_WINDOWS),
            }
        )
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class YearMetadata:
    year: int
    path: str
    init_yyyymmdd: tuple[int, ...]
    latitude: tuple[float, ...]
    longitude: tuple[float, ...]
    attrs: Mapping[str, Any]
    metadata_sha256: str


@dataclass(frozen=True)
class CaseReference:
    split: str
    year: int
    path: str
    cache_index: int
    init_yyyymmdd: int


@dataclass(frozen=True)
class PatchReference:
    split: str
    case: CaseReference
    patch_number: int
    latitude_start: int
    longitude_start: int


@dataclass(frozen=True)
class GlobalPreprocessing:
    log_bias: np.ndarray  # [lead, month, latitude, longitude]
    climatology: np.ndarray  # [lead, month, latitude, longitude]
    support: np.ndarray  # [latitude, longitude]
    feature_mean: np.ndarray  # [4, lead]
    feature_std: np.ndarray  # [4, lead]
    target_scale: np.ndarray  # [lead]
    fit_content_sha256: str

    def normalization_payload(self) -> dict[str, Any]:
        return {
            name: {
                "mean_by_lead": self.feature_mean[index].tolist(),
                "std_by_lead": self.feature_std[index].tolist(),
            }
            for index, name in enumerate(NORMALIZED_FEATURE_NAMES)
        }


@dataclass(frozen=True)
class CaseFields:
    mean: np.ndarray
    spread: np.ndarray
    truth: np.ndarray
    fraction: np.ndarray


def _update_case_content_digest(
    digest: Any, reference: CaseReference, fields: CaseFields
) -> None:
    digest.update(
        canonical_json(
            {
                "year": reference.year,
                "cache_index": reference.cache_index,
                "init_yyyymmdd": reference.init_yyyymmdd,
            }
        ).encode("utf-8")
    )
    for name in ("mean", "spread", "truth", "fraction"):
        values = np.ascontiguousarray(getattr(fields, name), dtype=np.float32)
        digest.update(name.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(memoryview(values).cast("B"))


def fingerprint_case_content(references: Sequence[CaseReference]) -> str:
    """Hash logical arrays for selected cases without touching purged chunks."""

    if not references:
        raise ValueError("cannot fingerprint an empty case sequence")
    reader = AnnualCacheReader()
    digest = hashlib.sha256()
    for reference in references:
        _update_case_content_digest(digest, reference, reader.read_case(reference))
    return digest.hexdigest()


def _metadata_payload(group: zarr.hierarchy.Group, year: int, path: Path) -> dict[str, Any]:
    return {
        "year": year,
        "path": str(path.resolve()),
        "attrs": dict(group.attrs),
        "arrays": {
            name: {
                "shape": list(group[name].shape),
                "chunks": list(group[name].chunks),
                "dtype": str(group[name].dtype),
            }
            for name in (
                "dynamic",
                "truth",
                "observation_fraction",
                "case_complete",
                "init_yyyymmdd",
                "lat",
                "lon",
            )
        },
    }


def read_year_metadata(cache_root: Path, year: int) -> YearMetadata:
    """Read and validate only allowed non-target metadata for one annual cache."""

    if year > MAXIMUM_ALLOWED_SOURCE_YEAR:
        raise DataContractError("refusing to open a 2018+ global cache")
    path = cache_root / f"{year}.zarr"
    if not path.is_dir():
        raise FileNotFoundError(path)
    group = zarr.open_group(str(path), mode="r")
    if group.attrs.get("status") != "complete":
        raise DataContractError(f"cache is not complete: {path}")
    if tuple(group.attrs.get("stat_names", ())) != STAT_NAMES:
        raise DataContractError(f"feature contract differs: {path}")
    if tuple(group.attrs.get("lead_windows", ())) != LEAD_WINDOWS:
        raise DataContractError(f"lead-window contract differs: {path}")
    if group.attrs.get("stored_units") != "mm day-1 weekly mean rate":
        raise DataContractError(f"stored rainfall units differ: {path}")
    expected_tail = (N_WEEK, N_STAT, N_LAT, N_LON)
    if tuple(group["dynamic"].shape[1:]) != expected_tail:
        raise DataContractError(f"unexpected dynamic shape: {path}")
    case_count = int(group["dynamic"].shape[0])
    if case_count != 104:
        raise DataContractError(f"expected 104 annual cases in {path}, got {case_count}")
    expected_fields = {
        "truth": (case_count, N_WEEK, N_LAT, N_LON),
        "observation_fraction": (case_count, N_WEEK, N_LAT, N_LON),
        "case_complete": (case_count,),
        "init_yyyymmdd": (case_count,),
        "lat": (N_LAT,),
        "lon": (N_LON,),
    }
    for name, shape in expected_fields.items():
        if tuple(group[name].shape) != shape:
            raise DataContractError(f"unexpected {name} shape in {path}")
    expected_dtypes = {
        "dynamic": np.dtype("float16"),
        "truth": np.dtype("float32"),
        "observation_fraction": np.dtype("float16"),
        "case_complete": np.dtype("bool"),
        "init_yyyymmdd": np.dtype("int32"),
        "lat": np.dtype("float64"),
        "lon": np.dtype("float64"),
    }
    for name, dtype in expected_dtypes.items():
        if np.dtype(group[name].dtype) != dtype:
            raise DataContractError(
                f"unexpected {name} dtype in {path}: {group[name].dtype}"
            )

    # These arrays are small contract metadata.  Dynamic and target arrays are
    # intentionally not indexed until the purge and optional limit are frozen.
    complete = np.asarray(group["case_complete"][:], dtype=bool)
    if not complete.all():
        raise DataContractError(f"annual cache contains incomplete cases: {path}")
    initializations = np.asarray(group["init_yyyymmdd"][:], dtype=np.int32)
    latitude = np.asarray(group["lat"][:], dtype=np.float64)
    longitude = np.asarray(group["lon"][:], dtype=np.float64)
    if len(np.unique(initializations)) != case_count:
        raise DataContractError(f"duplicate initializations in {path}")
    if np.any(np.diff(initializations.astype(np.int64)) <= 0):
        raise DataContractError(f"initializations are not strictly sorted in {path}")
    parsed_years = initializations // 10000
    if not np.all(parsed_years == year):
        raise DataContractError(f"initialization year mismatch in {path}")
    if not np.array_equal(latitude, np.linspace(90.0, -90.0, N_LAT)):
        raise DataContractError(f"unexpected latitude grid in {path}")
    if not np.array_equal(longitude, np.arange(0.0, 360.0, 1.5)):
        raise DataContractError(f"unexpected longitude grid in {path}")
    payload = _metadata_payload(group, year, path)
    return YearMetadata(
        year=year,
        path=str(path.resolve()),
        init_yyyymmdd=tuple(int(value) for value in initializations),
        latitude=tuple(float(value) for value in latitude),
        longitude=tuple(float(value) for value in longitude),
        attrs=dict(group.attrs),
        metadata_sha256=canonical_sha256(payload),
    )


def load_allowed_metadata(contract: GlobalPatchContract) -> dict[int, YearMetadata]:
    """Load metadata for exactly the contract years, never by directory scan."""

    root = Path(contract.cache_root)
    years = tuple(sorted(set(contract.fit_years + contract.validation_years)))
    if any(year > MAXIMUM_ALLOWED_SOURCE_YEAR for year in years):
        raise DataContractError("refusing to inspect a 2018+ source")
    metadata = {year: read_year_metadata(root, year) for year in years}
    reference_latitude = metadata[years[0]].latitude
    reference_longitude = metadata[years[0]].longitude
    for year in years[1:]:
        if metadata[year].latitude != reference_latitude:
            raise DataContractError("annual latitude grids differ")
        if metadata[year].longitude != reference_longitude:
            raise DataContractError("annual longitude grids differ")
    return metadata


def _yyyymmdd_to_date(value: int) -> datetime:
    try:
        return datetime.strptime(str(int(value)), "%Y%m%d")
    except ValueError as exc:
        raise DataContractError(f"invalid initialization date {value}") from exc


def midpoint_month_indices(init_yyyymmdd: int) -> np.ndarray:
    init = _yyyymmdd_to_date(init_yyyymmdd)
    return np.asarray(
        [(init + timedelta(days=3 + 7 * lead)).month - 1 for lead in range(N_WEEK)],
        dtype=np.int8,
    )


def midpoint_phases(init_yyyymmdd: int) -> np.ndarray:
    init = _yyyymmdd_to_date(init_yyyymmdd)
    result = np.empty((N_WEEK, 2), dtype=np.float32)
    for lead in range(N_WEEK):
        midpoint = init + timedelta(days=3 + 7 * lead)
        angle = 2.0 * math.pi * (midpoint.timetuple().tm_yday - 1) / 365.2425
        result[lead] = (math.sin(angle), math.cos(angle))
    return result


def build_case_references(
    contract: GlobalPatchContract,
    metadata: Mapping[int, YearMetadata],
) -> tuple[
    tuple[CaseReference, ...],
    tuple[CaseReference, ...],
    int,
    int,
]:
    """Freeze split/purge/limit decisions before any target array is indexed."""

    validation_metadata_candidates = [
        CaseReference("validation", year, metadata[year].path, index, init)
        for year in contract.validation_years
        for index, init in enumerate(metadata[year].init_yyyymmdd)
    ]
    target_boundary = datetime(MAXIMUM_ALLOWED_SOURCE_YEAR + 1, 1, 1)
    validation_candidates = [
        item
        for item in validation_metadata_candidates
        if _yyyymmdd_to_date(item.init_yyyymmdd) + timedelta(days=41)
        < target_boundary
    ]
    validation_boundary_purged = len(validation_metadata_candidates) - len(
        validation_candidates
    )
    validation_candidates.sort(key=lambda item: item.init_yyyymmdd)
    if not validation_candidates:
        raise DataContractError("validation split is empty")
    first_validation = _yyyymmdd_to_date(validation_candidates[0].init_yyyymmdd)

    fit_candidates = [
        CaseReference("fit", year, metadata[year].path, index, init)
        for year in contract.fit_years
        for index, init in enumerate(metadata[year].init_yyyymmdd)
    ]
    fit_candidates.sort(key=lambda item: item.init_yyyymmdd)
    fit_references = tuple(
        item
        for item in fit_candidates
        if _yyyymmdd_to_date(item.init_yyyymmdd) + timedelta(days=41)
        < first_validation
    )
    purged = len(fit_candidates) - len(fit_references)
    if contract.fit_case_limit:
        fit_references = fit_references[: contract.fit_case_limit]
    validation_references = tuple(validation_candidates)
    if contract.validation_case_limit:
        validation_references = validation_references[
            : contract.validation_case_limit
        ]
    if not fit_references or not validation_references:
        raise DataContractError("case limit or temporal purge emptied a split")
    if max(item.init_yyyymmdd for item in fit_references) >= min(
        item.init_yyyymmdd for item in validation_references
    ):
        raise DataContractError("fit and validation initialization order is invalid")
    return (
        fit_references,
        validation_references,
        purged,
        validation_boundary_purged,
    )


def case_date_bounds(references: Sequence[CaseReference]) -> dict[str, str]:
    """Return exact initialization and D1--D42 target bounds for provenance."""

    if not references:
        raise ValueError("cannot summarize an empty case-reference sequence")
    initializations = [
        _yyyymmdd_to_date(reference.init_yyyymmdd) for reference in references
    ]
    return {
        "initialization_date_min": min(initializations).date().isoformat(),
        "initialization_date_max": max(initializations).date().isoformat(),
        "target_date_min": min(initializations).date().isoformat(),
        "target_date_max": (
            max(initializations) + timedelta(days=41)
        ).date().isoformat(),
    }


class AnnualCacheReader:
    """Small read-only Zarr group cache used after split references are frozen."""

    def __init__(self) -> None:
        self._groups: dict[str, zarr.hierarchy.Group] = {}

    def group(self, path: str) -> zarr.hierarchy.Group:
        if path not in self._groups:
            if int(Path(path).stem) > MAXIMUM_ALLOWED_SOURCE_YEAR:
                raise DataContractError("refusing to open a 2018+ source")
            self._groups[path] = zarr.open_group(path, mode="r")
        return self._groups[path]

    def read_case(self, reference: CaseReference) -> CaseFields:
        if reference.year > MAXIMUM_ALLOWED_SOURCE_YEAR:
            raise DataContractError("refusing to index a 2018+ source")
        group = self.group(reference.path)
        # The split and target-overlap purge have already selected this exact
        # case reference.  No excluded case is indexed here.
        dynamic = np.asarray(
            group["dynamic"][reference.cache_index, :, :2, :, :], dtype=np.float32
        )
        truth = np.asarray(
            group["truth"][reference.cache_index, :, :, :], dtype=np.float32
        )
        fraction = np.asarray(
            group["observation_fraction"][reference.cache_index, :, :, :],
            dtype=np.float32,
        )
        if dynamic.shape != (N_WEEK, 2, N_LAT, N_LON):
            raise DataContractError("unexpected dynamic case shape")
        if truth.shape != (N_WEEK, N_LAT, N_LON):
            raise DataContractError("unexpected truth case shape")
        if fraction.shape != truth.shape:
            raise DataContractError("observation fraction shape differs")
        mean = dynamic[:, 0]
        spread = dynamic[:, 1]
        if not np.isfinite(mean).all() or not np.isfinite(spread).all():
            raise DataContractError("FuXi mean/spread contains non-finite values")
        if np.any(mean < 0.0) or np.any(spread < 0.0):
            raise DataContractError("FuXi mean/spread must be nonnegative")
        return CaseFields(mean, spread, truth, fraction)


def _selected_month_field(bank: np.ndarray, months: np.ndarray) -> np.ndarray:
    if bank.shape != (N_WEEK, 12, N_LAT, N_LON):
        raise ValueError("calendar bank has an invalid shape")
    return bank[np.arange(N_WEEK), months]


def _case_valid_mask(
    fields: CaseFields, minimum_observation_fraction: float
) -> np.ndarray:
    return (
        np.isfinite(fields.truth)
        & np.isfinite(fields.fraction)
        & (fields.fraction >= minimum_observation_fraction)
        & (fields.truth >= 0.0)
    )


def fit_global_preprocessing(
    fit_references: Sequence[CaseReference],
    latitude: np.ndarray,
    contract: GlobalPatchContract,
) -> GlobalPreprocessing:
    """Fit all target-derived preprocessing on purged fit references only."""

    if not fit_references or any(item.split != "fit" for item in fit_references):
        raise ValueError("preprocessing requires non-empty fit references")
    if any(item.year not in contract.fit_years for item in fit_references):
        raise DataContractError("non-fit reference reached preprocessing")
    reader = AnnualCacheReader()
    lead_count = np.zeros((N_WEEK, N_LAT, N_LON), dtype=np.int64)
    lead_residual_sum = np.zeros((N_WEEK, N_LAT, N_LON), dtype=np.float64)
    lead_truth_sum = np.zeros_like(lead_residual_sum)
    month_count = np.zeros((N_WEEK, 12, N_LAT, N_LON), dtype=np.int32)
    month_residual_sum = np.zeros((N_WEEK, 12, N_LAT, N_LON), dtype=np.float64)
    month_truth_sum = np.zeros_like(month_residual_sum)
    fit_content_digest = hashlib.sha256()

    for position, reference in enumerate(fit_references, start=1):
        fields = reader.read_case(reference)
        _update_case_content_digest(fit_content_digest, reference, fields)
        valid = _case_valid_mask(fields, contract.minimum_observation_fraction)
        residual = np.zeros_like(fields.truth, dtype=np.float64)
        residual[valid] = np.log1p(fields.truth[valid]) - np.log1p(
            fields.mean[valid]
        )
        lead_count += valid
        lead_residual_sum += np.where(valid, residual, 0.0)
        lead_truth_sum += np.where(valid, fields.truth, 0.0)
        months = midpoint_month_indices(reference.init_yyyymmdd)
        for lead, month in enumerate(months):
            month_count[lead, month] += valid[lead]
            month_residual_sum[lead, month] += np.where(
                valid[lead], residual[lead], 0.0
            )
            month_truth_sum[lead, month] += np.where(
                valid[lead], fields.truth[lead], 0.0
            )
        if position % 100 == 0 or position == len(fit_references):
            print(
                f"preprocessing pass 1: {position}/{len(fit_references)} fit cases",
                flush=True,
            )

    support = np.all(lead_count > 0, axis=0)
    if not support.any():
        raise DataContractError("global training observation support is empty")
    lead_residual_mean = np.divide(
        lead_residual_sum,
        lead_count,
        out=np.zeros_like(lead_residual_sum),
        where=lead_count > 0,
    )
    lead_truth_mean = np.divide(
        lead_truth_sum,
        lead_count,
        out=np.zeros_like(lead_truth_sum),
        where=lead_count > 0,
    )
    shrinkage = float(contract.anchor_shrinkage)
    log_bias = np.divide(
        month_residual_sum + shrinkage * lead_residual_mean[:, None],
        month_count + shrinkage,
        out=np.broadcast_to(lead_residual_mean[:, None], month_count.shape).copy(),
        where=(month_count + shrinkage) > 0,
    )
    climatology = np.divide(
        month_truth_sum + shrinkage * lead_truth_mean[:, None],
        month_count + shrinkage,
        out=np.broadcast_to(lead_truth_mean[:, None], month_count.shape).copy(),
        where=(month_count + shrinkage) > 0,
    )
    log_bias[..., ~support] = 0.0
    climatology[..., ~support] = 0.0
    if not np.isfinite(log_bias).all() or not np.isfinite(climatology).all():
        raise DataContractError("fitted calendar banks are non-finite")
    if np.any(climatology < 0.0):
        raise DataContractError("fitted climatology is negative")

    # Statistics for four normalized channels, indexed in
    # NORMALIZED_FEATURE_NAMES order.  Global area weighting avoids giving polar
    # grid rows the same influence as tropical rows.
    feature_weight = np.clip(np.cos(np.deg2rad(latitude)), 0.0, None)[:, None]
    feature_weight = np.broadcast_to(feature_weight, (N_LAT, N_LON)) * support
    sum_weight = np.zeros((4, N_WEEK), dtype=np.float64)
    sum_value = np.zeros_like(sum_weight)
    sum_square = np.zeros_like(sum_weight)
    target_weight = np.zeros(N_WEEK, dtype=np.float64)
    target_square = np.zeros(N_WEEK, dtype=np.float64)

    for position, reference in enumerate(fit_references, start=1):
        fields = reader.read_case(reference)
        months = midpoint_month_indices(reference.init_yyyymmdd)
        selected_bias = _selected_month_field(log_bias, months)
        selected_climatology = _selected_month_field(climatology, months)
        log_mean = np.log1p(fields.mean).astype(np.float64)
        log_spread = np.log1p(fields.spread).astype(np.float64)
        log_climatology = np.log1p(selected_climatology).astype(np.float64)
        anomaly = log_mean - log_climatology
        normalized_fields = (log_mean, log_spread, log_climatology, anomaly)
        for field_index, values in enumerate(normalized_fields):
            for lead in range(N_WEEK):
                valid_feature = np.isfinite(values[lead]) & support
                weight = np.where(valid_feature, feature_weight, 0.0)
                sum_weight[field_index, lead] += weight.sum(dtype=np.float64)
                sum_value[field_index, lead] += np.sum(
                    weight * np.where(valid_feature, values[lead], 0.0),
                    dtype=np.float64,
                )
                sum_square[field_index, lead] += np.sum(
                    weight * np.where(valid_feature, values[lead] ** 2, 0.0),
                    dtype=np.float64,
                )

        baseline_log = np.maximum(0.0, log_mean + selected_bias)
        valid_target = _case_valid_mask(
            fields, contract.minimum_observation_fraction
        ) & support[None]
        residual = np.zeros_like(log_mean, dtype=np.float64)
        residual[valid_target] = np.log1p(fields.truth[valid_target]) - baseline_log[
            valid_target
        ]
        for lead in range(N_WEEK):
            valid = valid_target[lead]
            weight = np.where(
                valid,
                feature_weight
                * np.clip(fields.fraction[lead].astype(np.float64), 0.0, 1.0),
                0.0,
            )
            target_weight[lead] += weight.sum(dtype=np.float64)
            target_square[lead] += np.sum(
                weight * np.where(valid, residual[lead] ** 2, 0.0),
                dtype=np.float64,
            )
        if position % 100 == 0 or position == len(fit_references):
            print(
                f"preprocessing pass 2: {position}/{len(fit_references)} fit cases",
                flush=True,
            )

    if np.any(sum_weight <= 0.0) or np.any(target_weight <= 0.0):
        raise DataContractError("one or more preprocessing channels have no weight")
    feature_mean = sum_value / sum_weight
    feature_variance = np.maximum(sum_square / sum_weight - feature_mean**2, 1.0e-6)
    feature_std = np.sqrt(feature_variance)
    target_scale = np.sqrt(np.maximum(target_square / target_weight, 1.0e-12))
    arrays = (feature_mean, feature_std, target_scale)
    if not all(np.isfinite(value).all() for value in arrays):
        raise DataContractError("normalization contains non-finite values")
    if np.any(feature_std <= 0.0) or np.any(target_scale <= 0.0):
        raise DataContractError("normalization scales must be positive")
    return GlobalPreprocessing(
        log_bias=log_bias.astype(np.float32),
        climatology=climatology.astype(np.float32),
        support=support.astype(bool),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
        fit_content_sha256=fit_content_digest.hexdigest(),
    )


def build_patch_schedule(
    cases: Sequence[CaseReference],
    latitude: np.ndarray,
    contract: GlobalPatchContract,
    *,
    split: str,
) -> tuple[PatchReference, ...]:
    """Create a deterministic area-aware patch schedule for one split."""

    if split not in {"fit", "validation"}:
        raise ValueError("split must be fit or validation")
    if not cases or any(item.split != split for item in cases):
        raise ValueError("case references do not match the requested split")
    starts = np.arange(0, N_LAT - contract.patch_size + 1, dtype=np.int64)
    centres = starts + contract.patch_size // 2
    probability = np.clip(np.cos(np.deg2rad(latitude[centres])), 0.0, None)
    probability /= probability.sum()
    split_offset = 0 if split == "fit" else 1_000_003
    generator = np.random.default_rng(contract.patch_seed + split_offset)
    schedule: list[PatchReference] = []
    for case in cases:
        latitude_starts = generator.choice(
            starts,
            size=contract.patches_per_case,
            replace=True,
            p=probability,
        )
        longitude_starts = generator.integers(
            0, N_LON, size=contract.patches_per_case
        )
        for number, (latitude_start, longitude_start) in enumerate(
            zip(latitude_starts, longitude_starts)
        ):
            schedule.append(
                PatchReference(
                    split=split,
                    case=case,
                    patch_number=number,
                    latitude_start=int(latitude_start),
                    longitude_start=int(longitude_start),
                )
            )
    if contract.mode == "smoke":
        # Boundary sentinels make every normal smoke exercise non-wrapping
        # latitude edges and circular longitude extraction, independent of the
        # pseudo-random schedule.  They remain ordinary training examples.
        sentinels = (
            (0, 230),
            (N_LAT - contract.patch_size, 230),
        )
        for index, (latitude_start, longitude_start) in enumerate(sentinels):
            if index >= len(schedule):
                break
            original = schedule[index]
            schedule[index] = PatchReference(
                split=original.split,
                case=original.case,
                patch_number=original.patch_number,
                latitude_start=latitude_start,
                longitude_start=longitude_start,
            )
    return tuple(schedule)


def patch_schedule_array(schedule: Sequence[PatchReference]) -> np.ndarray:
    return np.asarray(
        [
            (
                item.case.init_yyyymmdd,
                item.case.cache_index,
                item.patch_number,
                item.latitude_start,
                item.longitude_start,
            )
            for item in schedule
        ],
        dtype=np.int64,
    )


def patch_schedule_diagnostics(
    schedule: Sequence[PatchReference], patch_size: int = PATCH_SIZE
) -> dict[str, int]:
    return {
        "patches": len(schedule),
        "longitude_wraps": sum(
            item.longitude_start + patch_size > N_LON for item in schedule
        ),
        "north_edge_patches": sum(item.latitude_start == 0 for item in schedule),
        "south_edge_patches": sum(
            item.latitude_start == N_LAT - patch_size for item in schedule
        ),
    }


def _extract_patch(values: np.ndarray, latitude_start: int, longitude_start: int) -> np.ndarray:
    if not 0 <= latitude_start <= N_LAT - PATCH_SIZE:
        raise ValueError("latitude patch would wrap or leave the global grid")
    if not 0 <= longitude_start < N_LON:
        raise ValueError("longitude start is outside the global grid")
    latitude_slice = slice(latitude_start, latitude_start + PATCH_SIZE)
    longitude_indices = (
        np.arange(longitude_start, longitude_start + PATCH_SIZE) % N_LON
    )
    selected = values[..., latitude_slice, :]
    return np.take(selected, longitude_indices, axis=-1)


class GlobalPatchDataset(Dataset):
    """Lazy global patches with exact India-backbone feature semantics."""

    def __init__(
        self,
        schedule: Sequence[PatchReference],
        preprocessing: GlobalPreprocessing,
        latitude: np.ndarray,
        contract: GlobalPatchContract,
    ) -> None:
        if not schedule:
            raise ValueError("patch schedule is empty")
        self.schedule = tuple(schedule)
        self.preprocessing = preprocessing
        self.latitude = np.asarray(latitude, dtype=np.float32)
        self.minimum_fraction = float(contract.minimum_observation_fraction)
        self.reader: AnnualCacheReader | None = None
        self.latitude_ramp = np.broadcast_to(
            np.linspace(1.0, -1.0, PATCH_SIZE, dtype=np.float32)[:, None],
            (PATCH_SIZE, PATCH_SIZE),
        ).copy()
        self.longitude_ramp = np.broadcast_to(
            np.linspace(-1.0, 1.0, PATCH_SIZE, dtype=np.float32)[None, :],
            (PATCH_SIZE, PATCH_SIZE),
        ).copy()
        self.lead_ramp = np.linspace(-1.0, 1.0, N_WEEK, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["reader"] = None
        return state

    def __len__(self) -> int:
        return len(self.schedule)

    def _reader(self) -> AnnualCacheReader:
        if self.reader is None:
            self.reader = AnnualCacheReader()
        return self.reader

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        patch = self.schedule[index]
        fields = self._reader().read_case(patch.case)
        mean = _extract_patch(fields.mean, patch.latitude_start, patch.longitude_start)
        spread = _extract_patch(
            fields.spread, patch.latitude_start, patch.longitude_start
        )
        truth = _extract_patch(fields.truth, patch.latitude_start, patch.longitude_start)
        fraction = _extract_patch(
            fields.fraction, patch.latitude_start, patch.longitude_start
        )
        support = _extract_patch(
            self.preprocessing.support,
            patch.latitude_start,
            patch.longitude_start,
        ).astype(bool)
        months = midpoint_month_indices(patch.case.init_yyyymmdd)
        selected_bias = _extract_patch(
            _selected_month_field(self.preprocessing.log_bias, months),
            patch.latitude_start,
            patch.longitude_start,
        )
        climatology = _extract_patch(
            _selected_month_field(self.preprocessing.climatology, months),
            patch.latitude_start,
            patch.longitude_start,
        )

        log_mean = np.log1p(mean).astype(np.float32)
        log_spread = np.log1p(spread).astype(np.float32)
        log_climatology = np.log1p(climatology).astype(np.float32)
        anomaly = log_mean - log_climatology
        normalized_fields = []
        for field_index, values in enumerate(
            (log_mean, log_spread, log_climatology, anomaly)
        ):
            normalized = (
                values - self.preprocessing.feature_mean[field_index, :, None, None]
            ) / self.preprocessing.feature_std[field_index, :, None, None]
            if field_index in (2, 3):
                normalized = np.where(support[None], normalized, 0.0)
            normalized_fields.append(normalized.astype(np.float32))

        phases = midpoint_phases(patch.case.init_yyyymmdd)
        features = np.empty(
            (N_WEEK, BACKBONE_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32
        )
        features[:, 0] = normalized_fields[0]
        features[:, 1] = normalized_fields[1]
        features[:, 2] = normalized_fields[2]
        features[:, 3] = self.latitude_ramp[None]
        features[:, 4] = self.longitude_ramp[None]
        features[:, 5] = phases[:, 0, None, None]
        features[:, 6] = phases[:, 1, None, None]
        features[:, 7] = self.lead_ramp[:, None, None]
        features[:, 8] = support[None].astype(np.float32)
        features[:, 9] = normalized_fields[3]
        features[:, 10] = 0.0

        baseline_log = np.maximum(0.0, log_mean + selected_bias)
        valid = (
            support[None]
            & np.isfinite(truth)
            & np.isfinite(fraction)
            & (fraction >= self.minimum_fraction)
            & (truth >= 0.0)
        )
        target = np.zeros_like(truth, dtype=np.float32)
        target[valid] = (
            np.log1p(truth[valid]) - baseline_log[valid]
        ) / np.broadcast_to(
            self.preprocessing.target_scale[:, None, None], truth.shape
        )[valid]

        latitude_patch = self.latitude[
            patch.latitude_start : patch.latitude_start + PATCH_SIZE
        ]
        area = np.clip(np.cos(np.deg2rad(latitude_patch)), 0.0, None)[:, None]
        area = np.broadcast_to(area, (PATCH_SIZE, PATCH_SIZE))
        weights = np.where(
            valid,
            area[None] * np.clip(fraction.astype(np.float64), 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise DataContractError("patch features or target are non-finite")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise DataContractError("patch weights are invalid")
        if np.any(weights.sum(axis=(-2, -1)) <= 0.0):
            raise DataContractError("one or more patch leads have empty support")
        return (
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.from_numpy(weights),
        )


def build_model(contract: GlobalPatchContract) -> FixedClimatologyAllLeadUNet:
    """Build the exact 144,689-parameter India adapter architecture."""

    model = FixedClimatologyAllLeadUNet(
        input_channels=BACKBONE_CHANNELS,
        backbone_channels=BACKBONE_CHANNELS,
        base_channels=contract.base_channels,
        dropout=contract.dropout,
    )
    with torch.no_grad():
        # This source has no T2M.  A zero input plus zero kernel slice prevents
        # arbitrary random T2M behavior when the representation is transferred.
        first = model.backbone.encoder_1.block[0]
        if not isinstance(first, nn.Conv2d) or first.weight.shape[1] != BACKBONE_CHANNELS:
            raise RuntimeError("unexpected India adapter first convolution")
        first.weight[:, 10].zero_()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 144_689:
        raise RuntimeError(f"unexpected adapter parameter count {parameter_count}")
    return model


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def weighted_patch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Average area/support-weighted Smooth-L1 over active case-lead pairs."""

    if prediction.shape != target.shape or spatial_weights.shape != target.shape:
        raise ValueError("prediction, target, and weights must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1:] != (
        N_WEEK,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise ValueError("unexpected patch prediction shape")
    field = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    denominator = spatial_weights.sum(dim=(-2, -1))
    active = denominator > 0.0
    if not active.any():
        raise ValueError("batch has no active case-lead pairs")
    pair_loss = (field * spatial_weights).sum(dim=(-2, -1)) / denominator.clamp_min(
        1.0e-12
    )
    return pair_loss[active].mean()


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    contract: GlobalPatchContract,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    cases = 0
    gradient_context = torch.enable_grad if training else torch.no_grad
    with gradient_context():
        for features, target, weights in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, enabled=True):
                prediction = model(features)
                loss = weighted_patch_loss(
                    prediction,
                    target,
                    weights,
                    beta=contract.smooth_l1_beta,
                )
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), contract.gradient_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), contract.gradient_clip
                    )
                    optimizer.step()
            batch_size = int(features.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            cases += batch_size
    if cases == 0:
        raise RuntimeError("data loader produced no cases")
    return total / cases


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def train_model(
    model: FixedClimatologyAllLeadUNet,
    fit_dataset: GlobalPatchDataset,
    validation_dataset: GlobalPatchDataset,
    contract: GlobalPatchContract,
    *,
    device: torch.device,
    num_workers: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float, float]:
    """Train with validation selection while retaining exact epoch-zero fallback."""

    set_deterministic_seed(contract.seed)
    model.to(device)
    generator = torch.Generator().manual_seed(contract.seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=contract.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=contract.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1.0e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # The model head is zero-initialized by construction.  This validation loss
    # is therefore the exact global log-bias-anchor fallback.
    epoch_zero_validation = run_epoch(
        model, validation_loader, device, contract, None, None
    )
    best_loss = epoch_zero_validation
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "validation_loss": epoch_zero_validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selected": True,
            "fallback_anchor": True,
        }
    ]
    print(
        f"epoch=0 validation={epoch_zero_validation:.6f} exact_anchor_fallback=true",
        flush=True,
    )
    stale = 0
    for epoch in range(1, contract.epochs + 1):
        train_loss = run_epoch(
            model, fit_loader, device, contract, optimizer, scaler
        )
        validation_loss = run_epoch(
            model, validation_loader, device, contract, None, None
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1.0e-7
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selected": improved,
                "fallback_anchor": False,
            }
        )
        print(
            f"epoch={epoch} train={train_loss:.6f} validation={validation_loss:.6f} "
            f"selected={str(improved).lower()}",
            flush=True,
        )
        if stale >= contract.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state, strict=True)
    first_weight = best_state["backbone.encoder_1.block.0.weight"]
    if torch.count_nonzero(first_weight[:, 10]).item() != 0:
        raise RuntimeError("zero T2M placeholder acquired nonzero kernel weights")
    if best_loss > epoch_zero_validation + 1.0e-12:
        raise RuntimeError("selected checkpoint is worse than epoch-zero fallback")
    return best_state, history, best_epoch, best_loss, epoch_zero_validation


def source_provenance(
    metadata: Mapping[int, YearMetadata], contract: GlobalPatchContract
) -> dict[str, Any]:
    years = sorted(metadata)
    return {
        "cache_root": str(Path(contract.cache_root).resolve()),
        "opened_years": years,
        "hard_blocked_years": "2018 and later",
        "metadata_only_before_purge": True,
        "target_chunks_indexed_only_after_purge_and_case_limit": True,
        "annual": {
            str(year): {
                "path": metadata[year].path,
                "case_count": len(metadata[year].init_yyyymmdd),
                "first_initialization": metadata[year].init_yyyymmdd[0],
                "last_initialization": metadata[year].init_yyyymmdd[-1],
                "metadata_sha256": metadata[year].metadata_sha256,
                "fuxi_source": metadata[year].attrs.get("fuxi_source"),
                "imerg_source": metadata[year].attrs.get("imerg_source"),
                "completed_utc": metadata[year].attrs.get("completed_utc"),
            }
            for year in years
        },
    }


def _architecture_payload(contract: GlobalPatchContract) -> dict[str, Any]:
    return {
        "class": "FixedClimatologyAllLeadUNet",
        "module": "fuxi_adapter.models",
        "input_channels": BACKBONE_CHANNELS,
        "backbone_channels": BACKBONE_CHANNELS,
        "base_channels": contract.base_channels,
        "dropout": contract.dropout,
        "parameter_count": 144_689,
        "input_shape": ["batch", N_WEEK, BACKBONE_CHANNELS, PATCH_SIZE, PATCH_SIZE],
        "output_shape": ["batch", N_WEEK, PATCH_SIZE, PATCH_SIZE],
    }


def run_pretraining(
    contract: GlobalPatchContract,
    *,
    output_root: Path,
    run_name: str,
    device_name: str,
    num_workers: int,
) -> Path:
    """Execute one complete run and publish its directory atomically."""

    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one safe path component")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    output_root.mkdir(parents=True, exist_ok=True)
    final_output = output_root / run_name
    if final_output.exists():
        raise FileExistsError(f"run output already exists: {final_output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_name}.partial-", dir=output_root)
    )
    started = time.monotonic()
    try:
        _atomic_write_json(staging / "contract.json", contract.payload())
        metadata = load_allowed_metadata(contract)
        (
            fit_cases,
            validation_cases,
            purged_cases,
            validation_boundary_purged,
        ) = build_case_references(contract, metadata)
        latitude = np.asarray(
            metadata[min(metadata)].latitude, dtype=np.float64
        )
        preprocessing = fit_global_preprocessing(fit_cases, latitude, contract)
        validation_content_sha256 = fingerprint_case_content(validation_cases)
        fit_schedule = build_patch_schedule(
            fit_cases, latitude, contract, split="fit"
        )
        validation_schedule = build_patch_schedule(
            validation_cases, latitude, contract, split="validation"
        )
        fit_schedule_values = patch_schedule_array(fit_schedule)
        validation_schedule_values = patch_schedule_array(validation_schedule)
        _atomic_save_npz(
            staging / "artifacts" / "training_preprocessing.npz",
            log_bias=preprocessing.log_bias,
            climatology=preprocessing.climatology,
            support=preprocessing.support,
            feature_mean=preprocessing.feature_mean,
            feature_std=preprocessing.feature_std,
            target_scale=preprocessing.target_scale,
            fit_patch_schedule=fit_schedule_values,
            validation_patch_schedule=validation_schedule_values,
        )
        fit_dataset = GlobalPatchDataset(
            fit_schedule, preprocessing, latitude, contract
        )
        validation_dataset = GlobalPatchDataset(
            validation_schedule, preprocessing, latitude, contract
        )
        device = torch.device(
            device_name
            if device_name != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not visible")
        # Model initialization is part of the seed contract, so seed before
        # constructing the network (the training loop seeds again for loaders).
        set_deterministic_seed(contract.seed)
        model = build_model(contract)
        best_state, history, best_epoch, best_loss, epoch_zero_loss = train_model(
            model,
            fit_dataset,
            validation_dataset,
            contract,
            device=device,
            num_workers=num_workers,
        )
        _atomic_write_csv(staging / "history.csv", history)
        source_snapshots: dict[str, dict[str, str]] = {}
        for label, source in {
            "fuxi_global_patch_pretraining": Path(__file__).resolve(),
            "fuxi_adapter_models": Path(adapter_models.__file__).resolve(),
        }.items():
            destination = staging / "code" / f"{label}.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_snapshots[label] = {
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "snapshot_path": str(destination.relative_to(staging)),
                "snapshot_sha256": sha256_file(destination),
            }
        checkpoint = staging / "checkpoints" / "best.pt"
        normalization = preprocessing.normalization_payload()
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "global_patch_pretraining",
            "model_state_dict": best_state,
            "architecture": _architecture_payload(contract),
            "feature_names": list(FEATURE_NAMES),
            "contract": contract.payload(),
            "contract_sha256": contract.sha256,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epoch_zero_validation_loss": epoch_zero_loss,
            "seed": contract.seed,
            "source_snapshots": source_snapshots,
            "selected_case_content_sha256": {
                "fit": preprocessing.fit_content_sha256,
                "validation": validation_content_sha256,
            },
            "target_scale": preprocessing.target_scale,
            "normalization": normalization,
            "transfer": {
                "policy": "load representation then reset output head",
                "reset_keys": list(TRANSFER_RESET_KEYS),
                "zero_pretraining_channels": [FEATURE_NAMES[10]],
                "india_must_refit": [
                    "IMD normalization",
                    "IMD climatology",
                    "IMD log-bias anchor",
                    "IMD target scale",
                ],
            },
        }
        _atomic_torch_save(checkpoint, checkpoint_payload)
        elapsed = time.monotonic() - started
        preprocessing_path = staging / "artifacts" / "training_preprocessing.npz"
        history_path = staging / "history.csv"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_utc": utc_now(),
            "smoke": contract.mode == "smoke",
            "scientific_eligible": False,
            "test_predictions_created": False,
            "evidence_role": (
                "execution smoke only"
                if contract.mode == "smoke"
                else "global pretraining validation; not India test evidence"
            ),
            "contract": contract.payload(),
            "contract_sha256": contract.sha256,
            "architecture": _architecture_payload(contract),
            "feature_contract": {
                "names": list(FEATURE_NAMES),
                "normalized": list(NORMALIZED_FEATURE_NAMES),
                "t2m_policy": "explicit zero placeholder; zero first-convolution slice",
                "coordinates": "patch-relative; latitude does not wrap; longitude wraps",
                "forecast_context": "FuXi mean/spread remain defined on full patch",
                "target_context": "IMERG-derived fields and loss use train support only",
            },
            "target_contract": {
                "truth": "IMERG Final V07B weekly mean rainfall rate",
                "anchor": "training-only lead/verification-month/grid log-bias",
                "anchor_shrinkage": contract.anchor_shrinkage,
                "target": "(log1p(truth) - log1p(anchor)) / train-only lead RMS",
                "loss": "per-case/lead area-, support-, and observation-fraction-weighted Smooth-L1",
                "epoch_zero": "exact log-bias anchor",
            },
            "splits": {
                "fit_years": list(contract.fit_years),
                "validation_years": list(contract.validation_years),
                "fit_cases_after_purge_and_limit": len(fit_cases),
                "validation_cases_after_limit": len(validation_cases),
                "fit_date_bounds": case_date_bounds(fit_cases),
                "validation_date_bounds": case_date_bounds(validation_cases),
                "purged_fit_cases": purged_cases,
                "purged_validation_cases_crossing_2018_target_boundary": (
                    validation_boundary_purged
                ),
                "fit_patches": len(fit_schedule),
                "validation_patches": len(validation_schedule),
                "fit_schedule_sha256": sha256_array(fit_schedule_values),
                "validation_schedule_sha256": sha256_array(
                    validation_schedule_values
                ),
                "fit_schedule_diagnostics": patch_schedule_diagnostics(
                    fit_schedule, contract.patch_size
                ),
                "validation_schedule_diagnostics": patch_schedule_diagnostics(
                    validation_schedule, contract.patch_size
                ),
            },
            "selection": {
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "epoch_zero_validation_loss": epoch_zero_loss,
                "epoch_zero_retained": best_epoch == 0,
                "selected_not_worse_than_epoch_zero": best_loss <= epoch_zero_loss,
            },
            "source_provenance": {
                **source_provenance(metadata, contract),
                "selected_case_content_sha256": {
                    "fit": preprocessing.fit_content_sha256,
                    "validation": validation_content_sha256,
                },
            },
            "transfer": checkpoint_payload["transfer"],
            "elapsed_seconds": elapsed,
            "device": str(device),
            "runtime": {
                "hostname": os.uname().nodename,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_node": os.environ.get("SLURMD_NODENAME")
                or os.environ.get("SLURM_NODELIST"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "cuda_total_memory_bytes": (
                    int(torch.cuda.get_device_properties(device).total_memory)
                    if device.type == "cuda"
                    else None
                ),
                "torch_version": torch.__version__,
            },
            "artifacts": {
                "checkpoints/best.pt": sha256_file(checkpoint),
                "artifacts/training_preprocessing.npz": sha256_file(
                    preprocessing_path
                ),
                "history.csv": sha256_file(history_path),
                "contract.json": sha256_file(staging / "contract.json"),
                **{
                    details["snapshot_path"]: details["snapshot_sha256"]
                    for details in source_snapshots.values()
                },
            },
            "code": source_snapshots,
            "atomic_publication": {
                "staging_directory": True,
                "publication": "same-filesystem os.replace after complete manifest",
            },
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, final_output)
        print(f"complete: {final_output}", flush=True)
        return final_output
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_utc": utc_now(),
            "run_name": run_name,
            "contract": contract.payload(),
            "contract_sha256": contract.sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "test_predictions_created": False,
        }
        _atomic_write_json(staging / "manifest.json", failure)
        failed_output = output_root / f"{run_name}.failed"
        if failed_output.exists():
            failed_output = output_root / f"{run_name}.failed-{os.getpid()}"
        os.replace(staging, failed_output)
        print(f"failed run record: {failed_output}", flush=True)
        raise


def _parse_years(values: Sequence[int] | None, default: Iterable[int]) -> tuple[int, ...]:
    selected = tuple(int(value) for value in (values if values is not None else default))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cache-root", type=Path, default=GLOBAL_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--fit-years", nargs="+", type=int)
    parser.add_argument("--validation-years", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-seed", type=int, default=20260820)
    parser.add_argument("--patches-per-case", type=int)
    parser.add_argument("--fit-case-limit", type=int)
    parser.add_argument("--validation-case-limit", type=int)
    parser.add_argument("--minimum-observation-fraction", type=float, default=0.95)
    parser.add_argument("--anchor-shrinkage", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-3)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def contract_from_args(args: argparse.Namespace) -> GlobalPatchContract:
    smoke = args.mode == "smoke"
    fit_years = _parse_years(
        args.fit_years,
        (2002,) if smoke else range(2002, 2016),
    )
    validation_years = _parse_years(
        args.validation_years,
        (2003,) if smoke else (2016, 2017),
    )
    return GlobalPatchContract(
        mode=args.mode,
        cache_root=str(args.cache_root.resolve()),
        fit_years=fit_years,
        validation_years=validation_years,
        patches_per_case=(
            args.patches_per_case if args.patches_per_case is not None else 1
        ),
        patch_seed=args.patch_seed,
        seed=args.seed,
        fit_case_limit=(
            args.fit_case_limit
            if args.fit_case_limit is not None
            else (8 if smoke else 0)
        ),
        validation_case_limit=(
            args.validation_case_limit
            if args.validation_case_limit is not None
            else (4 if smoke else 0)
        ),
        minimum_observation_fraction=args.minimum_observation_fraction,
        anchor_shrinkage=args.anchor_shrinkage,
        dropout=args.dropout,
        batch_size=(args.batch_size if args.batch_size is not None else (2 if smoke else 32)),
        epochs=(args.epochs if args.epochs is not None else (1 if smoke else 40)),
        patience=(args.patience if args.patience is not None else (1 if smoke else 8)),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        smooth_l1_beta=args.smooth_l1_beta,
        gradient_clip=args.gradient_clip,
    )


def main() -> None:
    args = parse_args()
    contract = contract_from_args(args)
    run_name = args.run_name or f"{contract.mode}_seed{contract.seed}"
    print(
        json.dumps(
            {
                "contract_sha256": contract.sha256,
                "mode": contract.mode,
                "fit_years": contract.fit_years,
                "validation_years": contract.validation_years,
                "hard_blocked_from_2018_plus": True,
                "run_name": run_name,
            },
            indent=2,
        ),
        flush=True,
    )
    run_pretraining(
        contract,
        output_root=args.output_root.resolve(),
        run_name=run_name,
        device_name=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
