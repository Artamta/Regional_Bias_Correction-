#!/usr/bin/env python3
"""Freeze a self-contained raw-identity-vs-raw 2025 evaluation.

Only canonical E2 artifacts and IMD 2002--2017 training stores are read. Every
small executable/model artifact is captured once, authenticated from those
exact bytes, and copied into the frozen directory. No 2025 path is probed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import raw_identity_2025_contract as contract


SELECTION_NAME = "selection.json"


def _sha256_tree_no_symlinks(path: Path) -> str:
    root = contract.assert_secure_directory(path, "canonical E2 prediction tree")
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise contract.SealContractError(
                        f"symlink in canonical E2 tree: {entry.path}"
                    )
                item = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(item)
                elif entry.is_file(follow_symlinks=False):
                    files.append(item)
                else:
                    raise contract.SealContractError(
                        f"non-regular E2 tree entry: {entry.path}"
                    )
    digest = hashlib.sha256()
    for item in sorted(files):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        descriptor, _ = contract._open_file_read_no_follow(item)
        try:
            while True:
                block = os.read(descriptor, 8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        finally:
            os.close(descriptor)
    if not files:
        raise contract.SealContractError("canonical E2 prediction tree is empty")
    return digest.hexdigest()


def _captured_json(content: bytes, label: str) -> Mapping[str, Any]:
    return contract.parse_json_bytes(content, label)


def validate_fixed_development_inputs(e2_run: Path, raw_run: Path) -> Mapping[str, Any]:
    """Capture and authenticate selected artifacts before store access."""

    e2_run = contract.assert_secure_directory(e2_run, "canonical E2 run")
    raw_run = contract.assert_secure_directory(raw_run, "raw-identity run")
    source_paths = {
        "e2_manifest": e2_run / "manifest.json",
        "raw_run_manifest": raw_run / "manifest.json",
        "raw_selection": raw_run / "selection.json",
        "normalization": raw_run / "normalization.json",
        "raw_anchor": raw_run / "models/training_anchor_contract.npz",
        "frozen_model_source": raw_run / "code/models.py",
    }
    expected = {
        "e2_manifest": contract.EXPECTED_E2_MANIFEST_SHA256,
        "raw_run_manifest": contract.EXPECTED_RAW_MANIFEST_SHA256,
        "raw_selection": contract.EXPECTED_RAW_SELECTION_SHA256,
        "normalization": contract.EXPECTED_NORMALIZATION_SHA256,
        "raw_anchor": contract.EXPECTED_RAW_ANCHOR_SHA256,
        "frozen_model_source": contract.EXPECTED_MODEL_SOURCE_SHA256,
    }
    captured = {
        name: contract.capture_authenticated_bytes(source_paths[name], digest, name)
        for name, digest in expected.items()
    }
    e2 = _captured_json(captured["e2_manifest"], "canonical E2 manifest")
    raw_manifest = _captured_json(captured["raw_run_manifest"], "raw-identity manifest")
    raw_selection = _captured_json(captured["raw_selection"], "raw-identity selection")
    if not (
        e2.get("status") == "complete"
        and e2.get("canonical") is True
        and e2.get("scientific_eligible") is True
        and e2.get("smoke") is False
        and e2.get("final_2025_store_opened") is False
        and e2.get("final_initialization_year_quarantined") == 2025
        and tuple(e2.get("audit_years", ())) == (2022, 2023, 2024)
    ):
        raise contract.SealContractError("canonical E2 quarantine contract changed")
    if not (
        raw_manifest.get("status") == "complete"
        and raw_manifest.get("smoke") is False
        and raw_manifest.get("training_anchor") == "raw_fuxi"
        and raw_manifest.get("uses_fitted_log_bias_in_neural_training") is False
        and raw_manifest.get("selected_model") == contract.SELECTED_MODEL
        and float(raw_manifest.get("selected_alpha", -1.0)) == contract.SELECTED_ALPHA
    ):
        raise contract.SealContractError("raw-identity run contract changed")
    if not (
        raw_selection.get("status") == "frozen"
        and raw_selection.get("selection_scope") == "validation_only"
        and raw_selection.get("test_predictions_created") is False
        and raw_selection.get("training_anchor") == "raw_fuxi"
        and raw_selection.get("selected_model") == contract.SELECTED_MODEL
        and float(raw_selection.get("selected_alpha", -1.0)) == contract.SELECTED_ALPHA
        and tuple(raw_selection.get("train_years", ())) == contract.TRAIN_YEARS
        and tuple(raw_selection.get("validation_years", ()))
        == contract.VALIDATION_YEARS
    ):
        raise contract.SealContractError("validation-only selection changed")
    provenance = e2.get("input_provenance", {})
    if (
        provenance.get("raw_identity_manifest_sha256")
        != contract.EXPECTED_RAW_MANIFEST_SHA256
        or provenance.get("raw_identity_selection_sha256")
        != contract.EXPECTED_RAW_SELECTION_SHA256
    ):
        raise contract.SealContractError("E2 is not bound to the raw-identity run")

    records = (
        raw_manifest.get("training", {})
        .get(contract.SELECTED_MODEL, {})
        .get("runs", [])
    )
    if len(records) != 3:
        raise contract.SealContractError("exactly three selected runs are required")
    captured_checkpoints: dict[int, bytes] = {}
    for expected_seed, record in zip(contract.EXPECTED_SEEDS, records, strict=True):
        seed = int(record.get("seed", -1))
        digest = contract.EXPECTED_CHECKPOINT_SHA256[expected_seed]
        if seed != expected_seed or record.get("checkpoint_sha256") != digest:
            raise contract.SealContractError("selected checkpoint set changed")
        expected_relative = Path(
            f"models/{contract.SELECTED_MODEL}/seed_{seed}/checkpoints/best.pt"
        )
        if Path(str(record.get("checkpoint", ""))) != expected_relative:
            raise contract.SealContractError("selected checkpoint locator changed")
        checkpoint_path = raw_run / expected_relative
        captured_checkpoints[seed] = contract.capture_authenticated_bytes(
            checkpoint_path, digest, f"checkpoint seed {seed}"
        )
    if [
        item.get("sha256") for item in provenance.get("raw_identity_checkpoints", [])
    ] != [
        contract.EXPECTED_CHECKPOINT_SHA256[seed] for seed in contract.EXPECTED_SEEDS
    ]:
        raise contract.SealContractError("E2 checkpoint binding changed")

    predictions_path = e2_run / "predictions.zarr"
    predictions_expected = str(e2.get("artifacts", {}).get("predictions.zarr", ""))
    if _sha256_tree_no_symlinks(predictions_path) != predictions_expected:
        raise contract.SealContractError("canonical E2 prediction tree changed")

    stores = list(e2.get("source_stores_opened", {}).get("training_imd_2002_2017", []))
    if len(stores) != len(contract.TRAIN_YEARS):
        raise contract.SealContractError("E2 training-store count changed")
    training_stores: list[tuple[int, Path]] = []
    for expected_year, value in zip(contract.TRAIN_YEARS, stores, strict=True):
        path = contract.lexical_absolute(str(value))
        if (
            path.name != f"{expected_year}.zarr"
            or path.parent.name != "india_1p5_27x27_v1"
            or "/daily/imd/tp/" not in path.as_posix()
        ):
            raise contract.SealContractError(
                f"non-training locator refused before access: {value}"
            )
        descriptor = contract._open_directory_no_symlinks(path)
        os.close(descriptor)
        training_stores.append((expected_year, path))

    return {
        "e2": e2,
        "predictions_path": predictions_path,
        "predictions_tree_sha256": predictions_expected,
        "training_stores": tuple(training_stores),
        "captured": captured,
        "captured_checkpoints": captured_checkpoints,
    }


def _calendar_positions(dates: np.ndarray) -> np.ndarray:
    import pandas as pd

    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    array = np.asarray(dates, dtype="datetime64[D]")
    values = np.asarray(
        [lookup[pd.Timestamp(value).strftime("%m-%d")] for value in array.reshape(-1)],
        dtype=np.int16,
    )
    return values.reshape(array.shape)


def _build_training_climatology(
    dates: np.ndarray, values: np.ndarray, support: np.ndarray
) -> np.ndarray:
    import pandas as pd

    date_index = pd.DatetimeIndex(dates)
    if (
        tuple(sorted(int(value) for value in date_index.year.unique()))
        != contract.TRAIN_YEARS
    ):
        raise contract.SealContractError("climatology input years changed")
    positions = _calendar_positions(dates)
    years = date_index.year.to_numpy()
    climatology = np.full((366, *support.shape), np.nan, dtype=np.float32)
    for day in range(366):
        distance = np.minimum((positions - day) % 366, (day - positions) % 366)
        means = []
        for year in contract.TRAIN_YEARS:
            selected = (years == year) & (distance <= 15)
            if not np.any(selected):
                raise contract.SealContractError(
                    f"missing climatology window: day {day}, year {year}"
                )
            mean = np.mean(values[selected], axis=0, dtype=np.float64)
            if not np.isfinite(mean[support]).all():
                raise contract.SealContractError("training climatology is non-finite")
            means.append(mean)
        climatology[day] = np.mean(means, axis=0, dtype=np.float64).astype(np.float32)
    climatology[:, ~support] = np.nan
    if (
        contract.array_sha256(climatology, "<f4")
        != contract.EXPECTED_TRAINING_CLIMATOLOGY_SHA256
    ):
        raise contract.SealContractError("daily training climatology hash changed")
    return climatology


def build_pre_2025_bundle(inputs: Mapping[str, Any], output: Path) -> Mapping[str, Any]:
    """Build the portable inference bundle from exact pre-2025 inputs."""

    import pandas as pd
    import xarray as xr

    predictions_path = Path(inputs["predictions_path"])
    with xr.open_zarr(predictions_path, consolidated=True) as dataset:
        latitude = dataset.latitude.values.astype(np.float64)
        longitude = dataset.longitude.values.astype(np.float64)
        support = dataset.adapter_support.load().values.astype(bool)
        area = dataset.india_area_weight_km2.load().values.astype(np.float64)
        audit_initializations = dataset.init.values.astype("datetime64[D]")
        audit_climatology = dataset.fixed_imd_climatology.load().values.astype(
            np.float32
        )
    if (
        not np.array_equal(latitude, np.linspace(39.0, 0.0, 27))
        or not np.array_equal(longitude, np.linspace(60.0, 99.0, 27))
        or support.shape != contract.EXPECTED_GRID
        or int(support.sum()) != contract.EXPECTED_SUPPORT_CELLS
        or area.shape != contract.EXPECTED_GRID
        or not np.array_equal(area > 0.0, support)
    ):
        raise contract.SealContractError("canonical E2 support/area changed")

    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    metadata_hashes: dict[str, str] = {}
    for expected_year, store in inputs["training_stores"]:
        before = contract.sha256_file_anywhere_no_follow(store / ".zmetadata")
        with xr.open_zarr(store, consolidated=True) as dataset:
            if (
                dataset.attrs.get("source") != "imd"
                or dataset.attrs.get("units") != "mm day-1"
                or dataset.observation.dims != ("time", "latitude", "longitude")
            ):
                raise contract.SealContractError(
                    f"training IMD metadata/dimensions changed: {expected_year}"
                )
            if not np.array_equal(
                dataset.latitude.values, latitude
            ) or not np.array_equal(dataset.longitude.values, longitude):
                raise contract.SealContractError(
                    f"training IMD grid changed: {expected_year}"
                )
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = dataset.observation.load().values.astype(np.float32)
        after = contract.sha256_file_anywhere_no_follow(store / ".zmetadata")
        if before != after:
            raise contract.SealContractError(
                f"training metadata changed while freezing: {expected_year}"
            )
        index = pd.DatetimeIndex(dates)
        if (
            set(index.year) != {expected_year}
            or index.has_duplicates
            or not index.is_monotonic_increasing
            or values.shape != (len(dates), 27, 27)
            or not np.isfinite(values[:, support]).all()
            or np.any(values[:, support] < 0.0)
        ):
            raise contract.SealContractError(
                f"training IMD values/calendar changed: {expected_year}"
            )
        all_dates.append(dates)
        all_values.append(values)
        metadata_hashes[str(expected_year)] = before

    dates = np.concatenate(all_dates)
    values = np.concatenate(all_values)
    if np.unique(dates).size != dates.size or np.any(
        np.diff(dates) <= np.timedelta64(0, "D")
    ):
        raise contract.SealContractError("training dates are not sorted and unique")
    climatology = _build_training_climatology(dates, values, support)
    valid_dates = audit_initializations[:, None, None] + np.arange(
        42, dtype="timedelta64[D]"
    ).reshape(1, 6, 7)
    rebuilt = np.mean(
        climatology[_calendar_positions(valid_dates)], axis=2, dtype=np.float64
    ).astype(np.float32)
    if not np.allclose(
        rebuilt, audit_climatology, rtol=0.0, atol=2.0e-6, equal_nan=True
    ):
        raise contract.SealContractError("bundle does not replay E2 climatology")

    arrays = {
        "latitude": latitude,
        "longitude": longitude,
        "support": support,
        "india_area_weight_km2": area,
        "training_climatology_daily": climatology,
        "training_years": np.asarray(contract.TRAIN_YEARS, dtype=np.int16),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    content = buffer.getvalue()
    contract.write_bytes_exclusive(output, content)
    array_hashes = {
        "latitude": contract.array_sha256(latitude, "<f8"),
        "longitude": contract.array_sha256(longitude, "<f8"),
        "support": contract.array_sha256(support, "u1"),
        "india_area_weight_km2": contract.array_sha256(area, "<f8"),
        "training_climatology_daily": contract.array_sha256(climatology, "<f4"),
        "training_years": contract.array_sha256(arrays["training_years"], "<i2"),
    }
    return {
        "sha256": contract.sha256_bytes(content),
        "array_sha256": array_hashes,
        "source": "canonical E2 support/area and exact IMD 2002-2017 training stores",
        "source_training_year_metadata_sha256": metadata_hashes,
        "audit_climatology_max_abs_reconstruction_error": float(
            np.nanmax(np.abs(rebuilt.astype(np.float64) - audit_climatology))
        ),
        "contains_2025_values": False,
    }


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
            f"checkpoint_seed_{seed}": digest
            for seed, digest in contract.EXPECTED_CHECKPOINT_SHA256.items()
        },
    }


def freeze_selection(
    e2_run: Path,
    raw_run: Path,
    output_directory: Path,
    *,
    attest_no_2025_access: bool,
) -> Path:
    if not attest_no_2025_access:
        raise contract.SealContractError(
            "refusing to freeze without explicit no-2025-access attestation"
        )
    workflow_code_bytes = contract.capture_live_workflow_code()
    workflow_code_sha256 = contract.workflow_code_hashes(workflow_code_bytes)
    output_directory = contract.require_preflight_safe_path(
        output_directory, "frozen selection directory"
    )
    if output_directory != contract.require_preflight_safe_path(
        contract.CANONICAL_EXPERIMENT_ROOT, "canonical experiment root"
    ):
        raise contract.SealContractError(
            "the selection may be frozen only at the one canonical experiment root"
        )
    locators = contract.validate_data_locators(contract.DEFAULT_TEST_DATA_LOCATORS)
    inputs = validate_fixed_development_inputs(e2_run, raw_run)
    contract.assert_secure_directory(output_directory.parent, "freeze output parent")
    if contract.entry_exists_no_follow(output_directory):
        raise FileExistsError(f"frozen selection directory exists: {output_directory}")
    try:
        staging = contract.create_secure_staging_directory(
            output_directory.parent, f"{output_directory.name}.partial"
        )
    except contract.StagingAllocationError as exc:
        if contract.entry_exists_no_follow(exc.staging_path):
            shutil.rmtree(exc.staging_path)
            contract.fsync_directory(output_directory.parent)
        raise
    published = False
    try:
        sealed = contract.secure_mkdir(staging / "sealed", mode=0o755)
        for name, relative in contract.SEALED_RELATIVE_PATHS.items():
            if name == "support_climatology_bundle":
                continue
            contract.write_bytes_exclusive(staging / relative, inputs["captured"][name])
        for seed, relative in contract.SEALED_CHECKPOINT_RELATIVE_PATHS.items():
            contract.write_bytes_exclusive(
                staging / relative, inputs["captured_checkpoints"][seed]
            )
        bundle = build_pre_2025_bundle(
            inputs,
            staging / contract.SEALED_RELATIVE_PATHS["support_climatology_bundle"],
        )
        if (
            _sha256_tree_no_symlinks(Path(inputs["predictions_path"]))
            != inputs["predictions_tree_sha256"]
        ):
            raise contract.SealContractError("E2 tree changed while freezing")

        final_selection = output_directory / SELECTION_NAME
        canonical_paths = contract.execution_paths_for_root(output_directory)
        selection = {
            "schema_version": contract.SCHEMA_VERSION,
            "status": "frozen",
            "frozen_utc": contract.utc_now(),
            "frozen_before_2025_access": True,
            "test_year": contract.TEST_YEAR,
            "scientific_role": "sole untouched independent-test evaluation",
            "method_hierarchy": list(contract.METHOD_HIERARCHY),
            "primary_comparison": {
                "candidate": "raw_identity",
                "baseline": "raw_fuxi",
            },
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
            "evaluation_schedule": {
                "initialization_window": "2025-06-01..2025-09-30",
                "initialization_weekdays_utc": ["Monday", "Thursday"],
                "initialization_hour_utc": 0,
                "initializations": [
                    np.datetime_as_string(value, unit="D")
                    for value in contract.expected_initialization_dates()
                ],
                "lead_week": [1, 2, 3, 4, 5, 6],
                "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            },
            "bootstrap": {
                "draws": contract.BOOTSTRAP_DRAWS,
                "block_length_initializations": contract.BOOTSTRAP_BLOCK_LENGTH,
                "seed": contract.BOOTSTRAP_SEED,
                "method": "paired circular moving blocks by initialization",
                "all_six_leads_retained": True,
            },
            "secondary_diagnostics": {
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
            },
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
            "support_climatology_bundle": bundle,
            "test_data_locators": locators,
            "test_data_locators_sha256": contract.canonical_json_sha256(locators),
            "canonical_execution_paths": canonical_paths,
            "canonical_execution_paths_sha256": contract.canonical_json_sha256(
                canonical_paths
            ),
            "code_sha256": workflow_code_sha256,
            "policy": {
                "no_2025_fitting_tuning_selection_calibration_or_retries": True,
                "preflight": "synthetic CUDA only; /storage forbidden",
                "final_access": "exact user receipt then one durable ledger attempt",
                "artifact_consumption": "sealed authenticated local copies only",
            },
        }
        contract.write_bytes_exclusive(
            staging / SELECTION_NAME,
            json.dumps(selection, indent=2, allow_nan=False).encode() + b"\n",
        )
        if (
            contract.capture_authenticated_workflow_code(workflow_code_sha256)
            != workflow_code_bytes
        ):
            raise contract.SealContractError(
                "workflow source changed between freeze capture and publication"
            )
        contract.fsync_directory(sealed)
        contract.fsync_directory(staging)
        contract.rename_noreplace(staging, output_directory)
        published = True
        contract.fsync_directory(output_directory.parent)
        return final_selection
    except BaseException:
        # A rename syscall can succeed even if the subsequent parent fsync
        # fails. Authenticate that canonical directory and never mistake it
        # for an orphan staging tree or delete it.
        if not published and contract.entry_exists_no_follow(output_directory):
            contract.load_frozen_selection(final_selection)
            published = True
        elif published:
            contract.load_frozen_selection(final_selection)
        if not published and contract.entry_exists_no_follow(staging):
            shutil.rmtree(staging)
            contract.fsync_directory(output_directory.parent)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-run", type=Path, default=contract.DEFAULT_E2_RUN)
    parser.add_argument("--raw-run", type=Path, default=contract.DEFAULT_RAW_RUN)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=contract.CANONICAL_EXPERIMENT_ROOT,
    )
    parser.add_argument("--attest-no-2025-access", action="store_true", required=True)
    args = parser.parse_args(argv)
    result = freeze_selection(
        args.e2_run,
        args.raw_run,
        args.output_directory,
        attest_no_2025_access=args.attest_no_2025_access,
    )
    print(f"PASS: frozen raw-identity 2025 selection: {result}", flush=True)


if __name__ == "__main__":
    main()
