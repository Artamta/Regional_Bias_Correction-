#!/usr/bin/env python3
"""Sealed contract and hardened filesystem boundary for the 2025 test.

This module intentionally imports no xarray, Zarr, catalog, model, or runtime
code. Preflight-safe files are accessed through component-by-component
``O_NOFOLLOW`` directory traversal. Final data locators are validated only as
strings before the durable access ledger exists.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[1]
DEFAULT_E2_RUN = (
    PROJECT_ROOT
    / "resultsv2/fuxi_imd_raw_identity_2022_2024_audit"
    / "canonical_circular_20260822T0225Z"
)
DEFAULT_RAW_RUN = (
    PROJECT_ROOT / "resultsv2/fuxi_imd_no_log_bias_ablation/full_20260822T010749Z"
)
CANONICAL_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "resultsv2/raw_identity_independent_2025_sealed"
)
# This sentinel is deliberately a sibling of the experiment directory. Moving
# or deleting a selection directory cannot create a fresh one-attempt ledger.
GLOBAL_ACCESS_LEDGER_PATH = (
    PROJECT_ROOT / "resultsv2/raw_identity_independent_2025_access_ledger.json"
)

SCHEMA_VERSION = "raw_identity_independent_2025_selection_v2"
PREFLIGHT_SCHEMA_VERSION = "raw_identity_independent_2025_preflight_v2"
APPROVAL_SCHEMA_VERSION = "raw_identity_independent_2025_user_approval_v2"
LEDGER_SCHEMA_VERSION = "raw_identity_independent_2025_access_ledger_v2"
ATTEMPT_SCHEMA_VERSION = "raw_identity_independent_2025_attempt_status_v1"

TEST_YEAR = 2025
TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
EXPECTED_CASES = 35
EXPECTED_LEADS = 6
EXPECTED_GRID = (27, 27)
EXPECTED_SUPPORT_CELLS = 171
EXPECTED_SEEDS = (42, 43, 44)
SELECTED_MODEL = "normal_climo_model"
SELECTED_ALPHA = 1.0
METHOD_HIERARCHY = ("raw_fuxi", "raw_identity")
INTENSITY_STRATA = (
    ("dry_lt1", "<1", 0.0, 1.0),
    ("light_1_5", "1-5", 1.0, 5.0),
    ("moderate_5_10", "5-10", 5.0, 10.0),
    ("heavy_10_20", "10-20", 10.0, 20.0),
    ("extreme_ge20", ">=20", 20.0, None),
)

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_822

EXPECTED_E2_MANIFEST_SHA256 = (
    "bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec"
)
EXPECTED_RAW_MANIFEST_SHA256 = (
    "09317899c7d8c1d21952a23586499f195cf47f6b24ee3ca733580a38dd8d5463"
)
EXPECTED_RAW_SELECTION_SHA256 = (
    "705721f64d517194be7fa002c3ad6a7de6534b24e6ef215beb6c25ba43aa911c"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "4d5cdac09748cbac2bbaa9d3887bc7da1c6de012524c29d05dfcbb6d1bd0f15c"
)
EXPECTED_RAW_ANCHOR_SHA256 = (
    "a30cd309738dc3aec1351c7b924d08d9476bc5c6919e125247d7c619c531d0f3"
)
EXPECTED_MODEL_SOURCE_SHA256 = (
    "35a70b0e05043841c7e5b62793da05819f4f2b7e5e3f7a8e375f3bd76941f569"
)
EXPECTED_TRAINING_CLIMATOLOGY_SHA256 = (
    "eefdbe00a6e5f7be7cc417005bfdda897884fd31a3701cd6d3f36c5518e37127"
)
EXPECTED_CHECKPOINT_SHA256 = {
    42: "05739bb81a26694ccf5946daee9e4d0fc2bcacbdbfc976e5fc5c4c437f19cdd0",
    43: "90914d67cd807d118b57f295917a66662a59fa94157eb471fd18b988b629e193",
    44: "c7c346a12dd781e7b043c84887dc4a947f95f888e410a272d334e6f517b37f52",
}

DEFAULT_FORECAST_ROOT = (
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1/forecasts/fuxi_s2s/"
    "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
)
DEFAULT_TEST_DATA_LOCATORS = {
    "test_year": TEST_YEAR,
    "forecast_tp_store": f"{DEFAULT_FORECAST_ROOT}/tp/common_1p5/2025.zarr",
    "forecast_t2m_store": f"{DEFAULT_FORECAST_ROOT}/t2m/common_1p5/2025.zarr",
    "imd_daily_store": (
        "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
        "india_s2s_benchmark_v1/observations/ground_truth_v1/daily/imd/tp/"
        "india_1p5_27x27_v1/2025.zarr"
    ),
}

SEALED_RELATIVE_PATHS = {
    "e2_manifest": "sealed/e2_manifest.json",
    "raw_run_manifest": "sealed/raw_run_manifest.json",
    "raw_selection": "sealed/raw_selection.json",
    "normalization": "sealed/normalization.json",
    "raw_anchor": "sealed/raw_anchor.npz",
    "frozen_model_source": "sealed/frozen_models.py",
    "support_climatology_bundle": "sealed/pre_2025_support_climatology.npz",
}
SEALED_CHECKPOINT_RELATIVE_PATHS = {
    seed: f"sealed/checkpoint_seed_{seed}.pt" for seed in EXPECTED_SEEDS
}

CODE_FILENAMES = (
    "raw_identity_2025_contract.py",
    "freeze_raw_identity_2025_selection.py",
    "preflight_raw_identity_2025.py",
    "evaluate_raw_identity_2025.py",
    "raw_identity_2025_runtime.py",
    "raw_identity_2025_assets.py",
)

_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_RENAME_NOREPLACE = 1


class SealContractError(ValueError):
    """Raised before test access when a sealed invariant differs."""


class StagingAllocationError(OSError):
    """A staging directory exists, but its allocation could not complete."""

    def __init__(self, path: Path, cause: BaseException):
        super().__init__(f"staging allocation failed after mkdir: {path}: {cause}")
        self.staging_path = path


@dataclass(frozen=True)
class FrozenSelection:
    path: Path
    root: Path
    sha256: str
    payload: Mapping[str, Any]
    e2_manifest_path: Path
    raw_manifest_path: Path
    raw_selection_path: Path
    normalization_path: Path
    raw_anchor_path: Path
    model_source_path: Path
    bundle_path: Path
    checkpoints: tuple[tuple[int, Path, str], ...]
    data_locators: Mapping[str, Any]
    data_locator_sha256: str
    canonical_paths: Mapping[str, str]
    canonical_paths_sha256: str
    workflow_code_bytes: Mapping[str, bytes]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_initialization_dates() -> np.ndarray:
    current = date(TEST_YEAR, 6, 1)
    final = date(TEST_YEAR, 9, 30)
    values: list[np.datetime64] = []
    while current <= final:
        if current.weekday() in (0, 3):
            values.append(np.datetime64(current.isoformat(), "D"))
        current += timedelta(days=1)
    result = np.asarray(values, dtype="datetime64[D]")
    if result.shape != (EXPECTED_CASES,):  # pragma: no cover - constant guard
        raise RuntimeError("frozen Monday/Thursday schedule no longer has 35 starts")
    return result


def lexical_absolute(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def is_storage_path(path: os.PathLike[str] | str) -> bool:
    value = os.fspath(lexical_absolute(path))
    return value == "/storage" or value.startswith("/storage/")


def require_preflight_safe_path(path: os.PathLike[str] | str, label: str) -> Path:
    result = lexical_absolute(path)
    if is_storage_path(result):
        raise SealContractError(f"{label} is forbidden before access: {result}")
    return result


def _open_directory_no_symlinks(path: os.PathLike[str] | str) -> int:
    directory = lexical_absolute(path)
    descriptor = os.open("/", os.O_RDONLY | _O_DIRECTORY)
    try:
        for component in directory.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise SealContractError(f"not a directory: {directory}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def assert_secure_directory(path: os.PathLike[str] | str, label: str) -> Path:
    result = require_preflight_safe_path(path, label)
    descriptor = _open_directory_no_symlinks(result)
    os.close(descriptor)
    return result


def _open_file_read_no_follow(
    path: os.PathLike[str] | str, *, preflight_safe: bool = True
) -> tuple[int, Path]:
    source = (
        require_preflight_safe_path(path, "file input")
        if preflight_safe
        else lexical_absolute(path)
    )
    parent_descriptor = _open_directory_no_symlinks(source.parent)
    try:
        descriptor = os.open(
            source.name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_descriptor
        )
    finally:
        os.close(parent_descriptor)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SealContractError(f"input is not a regular file: {source}")
    return descriptor, source


def read_bytes_no_follow(path: os.PathLike[str] | str) -> bytes:
    descriptor, _ = _open_file_read_no_follow(path)
    blocks: list[bytes] = []
    try:
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            blocks.append(block)
    finally:
        os.close(descriptor)
    return b"".join(blocks)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    descriptor, _ = _open_file_read_no_follow(path)
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def sha256_file_anywhere_no_follow(path: os.PathLike[str] | str) -> str:
    """Hash a freezer-authorized pre-2025 file with no symlink traversal."""

    descriptor, _ = _open_file_read_no_follow(path, preflight_safe=False)
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def capture_authenticated_bytes(
    path: os.PathLike[str] | str, expected_sha256: str, label: str
) -> bytes:
    content = read_bytes_no_follow(path)
    actual = sha256_bytes(content)
    if actual != expected_sha256:
        raise SealContractError(f"{label} hash changed: {actual} != {expected_sha256}")
    return content


def array_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def parse_json_bytes(content: bytes, label: str) -> Mapping[str, Any]:
    value = json.loads(content)
    if not isinstance(value, Mapping):
        raise SealContractError(f"JSON root must be an object: {label}")
    return value


def read_json_with_sha256(
    path: os.PathLike[str] | str,
) -> tuple[Mapping[str, Any], str]:
    content = read_bytes_no_follow(path)
    return parse_json_bytes(content, os.fspath(path)), sha256_bytes(content)


def read_json(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    return read_json_with_sha256(path)[0]


def _write_bytes_exclusive(
    path: os.PathLike[str] | str, content: bytes, mode: int = 0o644
) -> Path:
    target = require_preflight_safe_path(path, "exclusive file output")
    parent_descriptor = _open_directory_no_symlinks(target.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    try:
        descriptor = os.open(target.name, flags, mode, dir_fd=parent_descriptor)
        try:
            offset = 0
            while offset < len(content):
                offset += os.write(descriptor, content[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return target


def write_bytes_exclusive(
    path: os.PathLike[str] | str, content: bytes, mode: int = 0o644
) -> Path:
    return _write_bytes_exclusive(path, content, mode)


def write_json_exclusive(
    path: os.PathLike[str] | str, payload: Mapping[str, Any]
) -> Path:
    content = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    return _write_bytes_exclusive(path, content)


def atomic_replace_json(
    path: os.PathLike[str] | str, payload: Mapping[str, Any]
) -> Path:
    target = require_preflight_safe_path(path, "attempt-status output")
    content = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    parent_descriptor = _open_directory_no_symlinks(target.parent)
    temporary_name = f".{target.name}.replace-{secrets.token_hex(12)}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
            0o644,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            target_stat = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            target_stat = None
        if target_stat is None or not stat.S_ISREG(target_stat.st_mode):
            raise SealContractError(
                f"attempt-status target must remain a regular file: {target}"
            )
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)
    return target


def secure_mkdir(path: os.PathLike[str] | str, mode: int = 0o755) -> Path:
    target = require_preflight_safe_path(path, "directory output")
    parent_descriptor = _open_directory_no_symlinks(target.parent)
    try:
        os.mkdir(target.name, mode=mode, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return target


def create_secure_staging_directory(parent: Path, prefix: str) -> Path:
    parent = assert_secure_directory(parent, "staging parent")
    descriptor = _open_directory_no_symlinks(parent)
    try:
        for _ in range(100):
            name = f".{prefix}-{secrets.token_hex(12)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
                path = parent / name
                try:
                    os.fsync(descriptor)
                except BaseException as exc:
                    raise StagingAllocationError(path, exc) from exc
                return path
            except FileExistsError:
                continue
    finally:
        os.close(descriptor)
    raise FileExistsError("could not allocate a unique secure staging directory")


def entry_exists_no_follow(path: os.PathLike[str] | str) -> bool:
    target = require_preflight_safe_path(path, "local path check")
    parent_descriptor = _open_directory_no_symlinks(target.parent)
    try:
        try:
            metadata = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode):
            raise SealContractError(f"refusing symlink path: {target}")
        return True
    finally:
        os.close(parent_descriptor)


def fsync_file(path: os.PathLike[str] | str) -> None:
    descriptor, _ = _open_file_read_no_follow(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: os.PathLike[str] | str) -> None:
    descriptor = _open_directory_no_symlinks(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    source = require_preflight_safe_path(source, "rename source")
    destination = require_preflight_safe_path(destination, "rename destination")
    if source.parent != destination.parent:
        raise SealContractError("no-clobber publication requires one parent directory")
    parent_descriptor = _open_directory_no_symlinks(source.parent)
    try:
        source_metadata = os.stat(
            source.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISDIR(source_metadata.st_mode):
            raise SealContractError("publication source is not a real directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("Linux renameat2 is required for no-clobber publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_descriptor,
            os.fsencode(source.name),
            parent_descriptor,
            os.fsencode(destination.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(parent_descriptor)


def live_code_hashes() -> dict[str, str]:
    directory = Path(os.path.abspath(__file__)).parent
    return {filename: sha256_file(directory / filename) for filename in CODE_FILENAMES}


def capture_live_workflow_code() -> dict[str, bytes]:
    directory = Path(os.path.abspath(__file__)).parent
    return {
        filename: read_bytes_no_follow(directory / filename)
        for filename in CODE_FILENAMES
    }


def workflow_code_hashes(content: Mapping[str, bytes]) -> dict[str, str]:
    if set(content) != set(CODE_FILENAMES):
        raise SealContractError("workflow code-byte fields changed")
    return {filename: sha256_bytes(content[filename]) for filename in CODE_FILENAMES}


def capture_authenticated_workflow_code(
    expected_sha256: Mapping[str, Any],
) -> dict[str, bytes]:
    """Capture each executable workflow source once and verify those bytes."""

    if not isinstance(expected_sha256, Mapping):
        raise SealContractError("frozen workflow code hashes must be an object")
    if set(expected_sha256) != set(CODE_FILENAMES):
        raise SealContractError("frozen workflow code-hash fields changed")
    captured = capture_live_workflow_code()
    for filename, content in captured.items():
        actual = sha256_bytes(content)
        if actual != expected_sha256.get(filename):
            raise SealContractError(f"workflow source changed: {filename}: {actual}")
    return captured


def execution_paths_for_root(root: os.PathLike[str] | str) -> dict[str, str]:
    root = require_preflight_safe_path(root, "frozen selection root")
    canonical_root = require_preflight_safe_path(
        CANONICAL_EXPERIMENT_ROOT, "canonical experiment root"
    )
    if root != canonical_root:
        raise SealContractError(
            f"selection root must be the one global experiment root: {canonical_root}"
        )
    return {
        "final_output": str(root / "independent_2025_result"),
        "access_ledger": str(
            require_preflight_safe_path(
                GLOBAL_ACCESS_LEDGER_PATH, "global access ledger"
            )
        ),
        "failure_record": str(root / "independent_2025_attempt_status.json"),
        "preflight_receipt": str(root / "preflight_receipt.json"),
    }


def canonical_execution_paths(selection_path: os.PathLike[str] | str) -> dict[str, str]:
    selection = require_preflight_safe_path(selection_path, "selection manifest")
    if selection.name != "selection.json":
        raise SealContractError("frozen selection must be named selection.json")
    root = selection.parent
    if root != require_preflight_safe_path(
        CANONICAL_EXPERIMENT_ROOT, "canonical experiment root"
    ):
        raise SealContractError("selection manifest is outside the canonical root")
    assert_secure_directory(root, "frozen selection root")
    return execution_paths_for_root(root)


def _validate_locator_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise SealContractError(f"{label} must be an absolute POSIX path")
    pure = PurePosixPath(value)
    if any(part in (".", "..") for part in pure.parts) or str(pure) != value:
        raise SealContractError(f"{label} is not a canonical lexical path")
    if not value.startswith("/storage/"):
        raise SealContractError(f"{label} must be under /storage")
    return value


def validate_data_locators(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {
        "test_year",
        "forecast_tp_store",
        "forecast_t2m_store",
        "imd_daily_store",
    }:
        raise SealContractError("test data locator fields changed")
    if payload.get("test_year") != TEST_YEAR:
        raise SealContractError("test data locator year changed")
    tp = _validate_locator_string(payload.get("forecast_tp_store"), "TP locator")
    t2m = _validate_locator_string(payload.get("forecast_t2m_store"), "T2M locator")
    imd = _validate_locator_string(payload.get("imd_daily_store"), "IMD locator")
    strict = "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
    if (
        strict not in tp
        or strict not in t2m
        or not tp.endswith("/tp/common_1p5/2025.zarr")
        or not t2m.endswith("/t2m/common_1p5/2025.zarr")
        or not imd.endswith("/daily/imd/tp/india_1p5_27x27_v1/2025.zarr")
    ):
        raise SealContractError(
            "test locator structure is not the frozen data contract"
        )
    validated = {
        "test_year": TEST_YEAR,
        "forecast_tp_store": tp,
        "forecast_t2m_store": t2m,
        "imd_daily_store": imd,
    }
    if validated != DEFAULT_TEST_DATA_LOCATORS:
        raise SealContractError("test data locators are not the three canonical stores")
    paths = (tp, t2m, imd)
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            common = os.path.commonpath((first, second))
            if common in (first, second):
                raise SealContractError("test data locators overlap")
    return validated


def _expected_fixed_hashes() -> dict[str, str]:
    return {
        "e2_manifest": EXPECTED_E2_MANIFEST_SHA256,
        "raw_run_manifest": EXPECTED_RAW_MANIFEST_SHA256,
        "raw_selection": EXPECTED_RAW_SELECTION_SHA256,
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "raw_anchor": EXPECTED_RAW_ANCHOR_SHA256,
        "frozen_model_source": EXPECTED_MODEL_SOURCE_SHA256,
        "training_climatology_daily": EXPECTED_TRAINING_CLIMATOLOGY_SHA256,
        **{
            f"checkpoint_seed_{seed}": digest
            for seed, digest in EXPECTED_CHECKPOINT_SHA256.items()
        },
    }


def validate_selection_payload(
    payload: Mapping[str, Any], selection_path: Path | None = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SealContractError("selection schema version changed")
    if (
        payload.get("status") != "frozen"
        or payload.get("frozen_before_2025_access") is not True
    ):
        raise SealContractError("selection is not frozen before 2025 access")
    if payload.get("test_year") != TEST_YEAR:
        raise SealContractError("selection test year changed")
    if payload.get("scientific_role") != "sole untouched independent-test evaluation":
        raise SealContractError("selection scientific role changed")
    if tuple(payload.get("method_hierarchy", ())) != METHOD_HIERARCHY:
        raise SealContractError("only raw_identity versus raw_fuxi is permitted")
    if payload.get("primary_comparison") != {
        "candidate": "raw_identity",
        "baseline": "raw_fuxi",
    }:
        raise SealContractError("primary comparison hierarchy changed")
    model = payload.get("model", {})
    if (
        model.get("name") != SELECTED_MODEL
        or float(model.get("alpha", -1.0)) != SELECTED_ALPHA
        or tuple(model.get("seeds", ())) != EXPECTED_SEEDS
        or model.get("ensemble") != "arithmetic mean of standardized residuals"
        or model.get("training_anchor") != "raw_fuxi"
        or model.get("parameter_count") != 144_689
    ):
        raise SealContractError("frozen raw-identity model changed")
    if payload.get("forbidden_methods") != [
        "log_bias",
        "legacy_anchored_adapter",
        "raw_identity_raw_mean_preserved",
        "physical_adapter",
        "global_pretraining_adapter",
    ]:
        raise SealContractError("forbidden-method declaration changed")
    if payload.get("bootstrap") != {
        "draws": BOOTSTRAP_DRAWS,
        "block_length_initializations": BOOTSTRAP_BLOCK_LENGTH,
        "seed": BOOTSTRAP_SEED,
        "method": "paired circular moving blocks by initialization",
        "all_six_leads_retained": True,
    }:
        raise SealContractError("bootstrap contract changed")
    development = payload.get("development_data", {})
    if (
        tuple(development.get("train_years", ())) != TRAIN_YEARS
        or tuple(development.get("validation_years", ())) != VALIDATION_YEARS
        or development.get("retrospective_audit_end_year") != 2024
    ):
        raise SealContractError("development split changed")
    if payload.get("expected_test_shape") != [35, 6, 27, 27]:
        raise SealContractError("expected independent-test shape changed")
    schedule = payload.get("evaluation_schedule")
    expected_schedule = {
        "initialization_window": "2025-06-01..2025-09-30",
        "initialization_weekdays_utc": ["Monday", "Thursday"],
        "initialization_hour_utc": 0,
        "initializations": [
            np.datetime_as_string(value, unit="D")
            for value in expected_initialization_dates()
        ],
        "lead_week": [1, 2, 3, 4, 5, 6],
        "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
    }
    if schedule != expected_schedule:
        raise SealContractError("frozen initialization/lead schedule changed")
    expected_secondary = {
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
            for key, label, lower, upper in INTENSITY_STRATA
        ],
    }
    if payload.get("secondary_diagnostics") != expected_secondary:
        raise SealContractError("secondary intensity diagnostic contract changed")
    if payload.get("policy") != {
        "no_2025_fitting_tuning_selection_calibration_or_retries": True,
        "preflight": "synthetic CUDA only; /storage forbidden",
        "final_access": "exact user receipt then one durable ledger attempt",
        "artifact_consumption": "sealed authenticated local copies only",
    }:
        raise SealContractError("sealed access policy changed")
    if payload.get("fixed_sha256") != _expected_fixed_hashes():
        raise SealContractError("fixed E2/model hashes changed")
    if payload.get("paths") != SEALED_RELATIVE_PATHS:
        raise SealContractError("sealed artifact path contract changed")
    expected_records = [
        {
            "seed": seed,
            "path": SEALED_CHECKPOINT_RELATIVE_PATHS[seed],
            "sha256": EXPECTED_CHECKPOINT_SHA256[seed],
        }
        for seed in EXPECTED_SEEDS
    ]
    if payload.get("checkpoints") != expected_records:
        raise SealContractError("sealed checkpoint path/hash contract changed")
    locators = validate_data_locators(payload.get("test_data_locators", {}))
    if payload.get("test_data_locators_sha256") != canonical_json_sha256(locators):
        raise SealContractError("test data locator digest changed")
    if selection_path is not None:
        canonical_paths = canonical_execution_paths(selection_path)
        if payload.get("canonical_execution_paths") != canonical_paths:
            raise SealContractError("canonical output/ledger paths changed")
        if payload.get("canonical_execution_paths_sha256") != canonical_json_sha256(
            canonical_paths
        ):
            raise SealContractError("canonical path-contract digest changed")


def _validate_bundle(path: Path, payload: Mapping[str, Any]) -> None:
    bundle = payload.get("support_climatology_bundle", {})
    reconstruction_error = bundle.get("audit_climatology_max_abs_reconstruction_error")
    if (
        bundle.get("source")
        != "canonical E2 support/area and exact IMD 2002-2017 training stores"
        or bundle.get("contains_2025_values") is not False
        or not isinstance(reconstruction_error, (int, float))
        or not np.isfinite(reconstruction_error)
        or reconstruction_error > 2.0e-6
        or set(bundle.get("source_training_year_metadata_sha256", {}))
        != {str(year) for year in TRAIN_YEARS}
    ):
        raise SealContractError("support/climatology bundle provenance changed")
    content = read_bytes_no_follow(path)
    if sha256_bytes(content) != bundle.get("sha256"):
        raise SealContractError("sealed support/climatology bundle hash changed")
    with np.load(io.BytesIO(content), allow_pickle=False) as values:
        required = {
            "latitude",
            "longitude",
            "support",
            "india_area_weight_km2",
            "training_climatology_daily",
            "training_years",
        }
        if set(values.files) != required:
            raise SealContractError("support/climatology bundle members changed")
        arrays = {name: np.asarray(values[name]).copy() for name in required}
    latitude = arrays["latitude"].astype(np.float64)
    longitude = arrays["longitude"].astype(np.float64)
    support = arrays["support"].astype(bool)
    area = arrays["india_area_weight_km2"].astype(np.float64)
    climatology = arrays["training_climatology_daily"].astype(np.float32)
    years = arrays["training_years"].astype(np.int16)
    if not np.array_equal(latitude, np.linspace(39.0, 0.0, 27)) or not np.array_equal(
        longitude, np.linspace(60.0, 99.0, 27)
    ):
        raise SealContractError("bundle coordinates changed")
    if support.shape != EXPECTED_GRID or int(support.sum()) != EXPECTED_SUPPORT_CELLS:
        raise SealContractError("bundle support changed")
    if (
        area.shape != EXPECTED_GRID
        or not np.isfinite(area).all()
        or np.any(area < 0.0)
        or not np.array_equal(area > 0.0, support)
    ):
        raise SealContractError("bundle area/support changed")
    if (
        climatology.shape != (366, 27, 27)
        or not np.isfinite(climatology[:, support]).all()
        or array_sha256(climatology, "<f4") != EXPECTED_TRAINING_CLIMATOLOGY_SHA256
    ):
        raise SealContractError("bundle climatology changed")
    if tuple(int(value) for value in years) != TRAIN_YEARS:
        raise SealContractError("bundle training years changed")
    dtypes = {
        "latitude": "<f8",
        "longitude": "<f8",
        "support": "u1",
        "india_area_weight_km2": "<f8",
        "training_climatology_daily": "<f4",
        "training_years": "<i2",
    }
    actual_hashes = {
        name: array_sha256(arrays[name], dtype) for name, dtype in dtypes.items()
    }
    if actual_hashes != bundle.get("array_sha256"):
        raise SealContractError("bundle array hashes changed")


def load_frozen_selection(path: os.PathLike[str] | str) -> FrozenSelection:
    selection_path = require_preflight_safe_path(path, "selection manifest")
    payload, selection_sha256 = read_json_with_sha256(selection_path)
    validate_selection_payload(payload, selection_path)
    root = assert_secure_directory(selection_path.parent, "selection root")
    resolved = {
        name: root / relative for name, relative in SEALED_RELATIVE_PATHS.items()
    }
    locks = payload["fixed_sha256"]
    for name in (
        "e2_manifest",
        "raw_run_manifest",
        "raw_selection",
        "normalization",
        "raw_anchor",
        "frozen_model_source",
    ):
        if sha256_file(resolved[name]) != locks[name]:
            raise SealContractError(f"sealed {name} hash changed")
    checkpoints: list[tuple[int, Path, str]] = []
    for seed in EXPECTED_SEEDS:
        checkpoint = root / SEALED_CHECKPOINT_RELATIVE_PATHS[seed]
        expected = EXPECTED_CHECKPOINT_SHA256[seed]
        if sha256_file(checkpoint) != expected:
            raise SealContractError(f"sealed checkpoint seed {seed} changed")
        checkpoints.append((seed, checkpoint, expected))
    workflow_code_bytes = capture_authenticated_workflow_code(
        payload.get("code_sha256", {})
    )
    _validate_bundle(resolved["support_climatology_bundle"], payload)
    locators = validate_data_locators(payload["test_data_locators"])
    canonical_paths = canonical_execution_paths(selection_path)
    return FrozenSelection(
        path=selection_path,
        root=root,
        sha256=selection_sha256,
        payload=payload,
        e2_manifest_path=resolved["e2_manifest"],
        raw_manifest_path=resolved["raw_run_manifest"],
        raw_selection_path=resolved["raw_selection"],
        normalization_path=resolved["normalization"],
        raw_anchor_path=resolved["raw_anchor"],
        model_source_path=resolved["frozen_model_source"],
        bundle_path=resolved["support_climatology_bundle"],
        checkpoints=tuple(checkpoints),
        data_locators=locators,
        data_locator_sha256=canonical_json_sha256(locators),
        canonical_paths=canonical_paths,
        canonical_paths_sha256=canonical_json_sha256(canonical_paths),
        workflow_code_bytes=workflow_code_bytes,
    )


def validate_preflight_receipt(
    path: os.PathLike[str] | str, frozen: FrozenSelection
) -> tuple[Mapping[str, Any], str]:
    receipt_path = require_preflight_safe_path(path, "preflight receipt")
    if str(receipt_path) != frozen.canonical_paths["preflight_receipt"]:
        raise SealContractError("preflight receipt is not at its canonical path")
    payload, receipt_sha256 = read_json_with_sha256(receipt_path)
    if (
        payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or payload.get("status") != "passed"
    ):
        raise SealContractError("preflight receipt is not a passed v2 receipt")
    if payload.get("selection_manifest_sha256") != frozen.sha256:
        raise SealContractError("preflight is bound to another selection")
    if payload.get("method_hierarchy") != list(METHOD_HIERARCHY):
        raise SealContractError("preflight method hierarchy changed")
    if payload.get("code_sha256") != frozen.payload.get("code_sha256"):
        raise SealContractError("preflight used non-frozen workflow code")
    proof = payload.get("proof", {})
    if (
        proof.get("storage_paths_opened") != []
        or proof.get("access_ledger_created") is not False
        or proof.get("result_created") is not False
        or proof.get("synthetic_cuda_inference") is not True
        or tuple(proof.get("checkpoint_seeds_loaded", ())) != EXPECTED_SEEDS
    ):
        raise SealContractError("preflight proof boundary changed")
    without_digest = dict(payload)
    embedded = without_digest.pop("proof_payload_sha256", None)
    if embedded != canonical_json_sha256(without_digest):
        raise SealContractError("preflight proof digest does not verify")
    return payload, receipt_sha256


def validate_approval_receipt(
    path: os.PathLike[str] | str,
    frozen: FrozenSelection,
    *,
    preflight_receipt_sha256: str,
    preflight_created_utc: str,
) -> tuple[Mapping[str, Any], str]:
    approval_path = require_preflight_safe_path(path, "user approval receipt")
    payload, approval_sha256 = read_json_with_sha256(approval_path)
    expected_keys = {
        "schema_version",
        "decision",
        "approved_by",
        "approved_utc",
        "test_year",
        "selection_manifest_sha256",
        "preflight_receipt_sha256",
        "test_data_locators_sha256",
        "canonical_execution_paths_sha256",
        "final_output",
        "access_ledger",
        "failure_record",
        "allowed_methods",
        "authorization",
    }
    if set(payload) != expected_keys:
        raise SealContractError("approval receipt fields changed")
    if (
        payload.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or payload.get("decision") != "approve_exactly_one_independent_2025_access"
        or payload.get("approved_by") != "raj.ayush"
    ):
        raise SealContractError("exact raj.ayush one-attempt approval is absent")
    try:
        approved = datetime.fromisoformat(str(payload["approved_utc"]))
        frozen_time = datetime.fromisoformat(str(frozen.payload["frozen_utc"]))
        preflight_time = datetime.fromisoformat(preflight_created_utc)
    except ValueError as exc:
        raise SealContractError(
            "approval/freeze/preflight timestamp is invalid"
        ) from exc
    if any(value.tzinfo is None for value in (approved, frozen_time, preflight_time)):
        raise SealContractError("approval/freeze/preflight timestamps need timezones")
    if approved <= max(frozen_time, preflight_time) or approved > datetime.now(
        timezone.utc
    ):
        raise SealContractError("approval timestamp is outside the allowed interval")
    paths = frozen.canonical_paths
    if (
        payload.get("test_year") != TEST_YEAR
        or payload.get("selection_manifest_sha256") != frozen.sha256
        or payload.get("preflight_receipt_sha256") != preflight_receipt_sha256
        or payload.get("test_data_locators_sha256") != frozen.data_locator_sha256
        or payload.get("canonical_execution_paths_sha256")
        != frozen.canonical_paths_sha256
        or payload.get("final_output") != paths["final_output"]
        or payload.get("access_ledger") != paths["access_ledger"]
        or payload.get("failure_record") != paths["failure_record"]
        or payload.get("allowed_methods") != list(METHOD_HIERARCHY)
    ):
        raise SealContractError("approval is not bound to the exact frozen execution")
    expected_sentence = (
        "I authorize one access attempt for the frozen raw_identity versus "
        "raw_fuxi independent 2025 evaluation."
    )
    if payload.get("authorization") != expected_sentence:
        raise SealContractError("approval authorization sentence is not exact")
    return payload, approval_sha256


def create_access_ledger(
    path: os.PathLike[str] | str,
    *,
    frozen: FrozenSelection,
    preflight_receipt_sha256: str,
    approval_receipt_path: os.PathLike[str] | str,
    approval_receipt_sha256: str,
) -> Path:
    target = require_preflight_safe_path(path, "access ledger")
    if str(target) != frozen.canonical_paths["access_ledger"]:
        raise SealContractError("access ledger path is not canonical")
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "independent_2025_access_attempt_consumed",
        "attempt": 1,
        "created_utc": utc_now(),
        "test_year": TEST_YEAR,
        "selection_manifest": str(frozen.path),
        "selection_manifest_sha256": frozen.sha256,
        "preflight_receipt": frozen.canonical_paths["preflight_receipt"],
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "approval_receipt": str(
            require_preflight_safe_path(approval_receipt_path, "approval receipt")
        ),
        "approval_receipt_sha256": approval_receipt_sha256,
        "test_data_locators": dict(frozen.data_locators),
        "test_data_locators_sha256": frozen.data_locator_sha256,
        "canonical_execution_paths": dict(frozen.canonical_paths),
        "canonical_execution_paths_sha256": frozen.canonical_paths_sha256,
        "method_hierarchy": list(METHOD_HIERARCHY),
        "runtime_import_state_at_commit": "not_imported",
        "policy": "one attempt remains consumed after success, failure, or interruption",
    }
    return write_json_exclusive(target, payload)
