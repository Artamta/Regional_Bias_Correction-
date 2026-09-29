#!/usr/bin/env python3
"""One-time independent 2025 evaluation of a frozen TP/T2M-only adapter.

This workflow is intentionally narrower than the physical-variable sweep.  It
accepts only a validation-frozen compact control whose effective backbone uses
the established FuXi precipitation/temperature and training-only IMD feature
contract.  A physical-feature or member-feature selection is rejected *before*
any 2025 forecast or observation store is opened.

The script never trains, selects, calibrates, or fits normalization on 2025.
It applies three hash-locked pre-2020 checkpoints, averages their standardized
residuals, and evaluates the ensemble once on 35 strict-00Z JJAS starts.  Paired
uncertainty uses a deterministic one-year circular moving-block bootstrap over
initializations; every draw resamples complete six-lead vectors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.collections import LineCollection


EVALUATE_ROOT = Path(__file__).resolve().parent
HERE = EVALUATE_ROOT.parent
CLEAN = HERE.parent
SOURCE_ROOT = HERE / "src"
NEURAL_SRC = CLEAN / "neural_adapter" / "src"
BENCHMARK_DIR = CLEAN / "studies" / "fuxi_imd_adapter_benchmark_v1"
for source in (SOURCE_ROOT, EVALUATE_ROOT, NEURAL_SRC, BENCHMARK_DIR):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import fuxi_imd_compact_validation_sweep as sweep  # noqa: E402
import plot_physical_validation_results as validation_plots  # noqa: E402
import run_benchmark as benchmark  # noqa: E402
from fuxi_adapter.anchored import reconstruct_anchored_precipitation  # noqa: E402
from fuxi_adapter.baselines import (  # noqa: E402
    LogBiasCorrection,
    apply_log_bias_correction,
)
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402


base = benchmark.base
diagnostic = benchmark.diagnostic
TEST_YEAR = 2025
TEST_MONTHS = (6, 7, 8, 9)
EXPECTED_CASES = 35
EXPECTED_LEADS = 6
EXPECTED_GRID = (27, 27)
EXPECTED_SUPPORT_CELLS = 171
EXPECTED_CONTROL_CONTRACT = "tp_t2m_only_compact_v1"
DEFAULT_BOOTSTRAP_DRAWS = 2000
DEFAULT_BLOCK_LENGTH = 13
DEFAULT_BOOTSTRAP_SEED = 20250812
METHODS = ("raw_fuxi", "corrected")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "corrected": "Frozen IMD-corrected",
}
METHOD_COLORS = {"raw_fuxi": "#3C5488", "corrected": "#00A087"}
METRICS = ("rmse_mm_day", "acc", "bias_mm_day")


@dataclass(frozen=True)
class FrozenControl:
    """Validated files needed for the untouched-test inference path."""

    validation_run: Path
    selection_path: Path
    selection: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    candidate: sweep.SweepCandidate
    checkpoint_paths: tuple[Path, ...]
    checkpoint_seeds: tuple[int, ...]
    normalization_path: Path
    normalization: Mapping[str, Any]
    control_normalization: Mapping[str, Any]
    anchor_path: Path


@dataclass(frozen=True)
class TestData:
    forecast: Any
    t2m_weekly: np.ndarray
    truth: np.ndarray
    climatology: np.ndarray
    weekly_coverage: np.ndarray
    training_climatology_daily: np.ndarray
    support: np.ndarray
    area_weight_km2: np.ndarray
    source_stores: tuple[str, ...]
    training_imd_stores: tuple[str, ...]
    verification_imd_store: str
    source_metadata_hashes: Mapping[str, Mapping[str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON data contract using stable key/order/number encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_code_files() -> Mapping[str, Path]:
    """Return every live source file that can alter the frozen inference path."""

    return {
        "evaluation": Path(__file__).resolve(),
        "validation_sweep": Path(sweep.__file__).resolve(),
        "validation_model": (
            NEURAL_SRC / "fuxi_adapter" / "validation_sweep_models.py"
        ).resolve(),
        "backbone_model": (NEURAL_SRC / "fuxi_adapter" / "models.py").resolve(),
        "anchored_reconstruction": (
            NEURAL_SRC / "fuxi_adapter" / "anchored.py"
        ).resolve(),
        "log_bias_baseline": (
            NEURAL_SRC / "fuxi_adapter" / "baselines.py"
        ).resolve(),
        "prediction_loop": (
            NEURAL_SRC / "fuxi_adapter" / "training.py"
        ).resolve(),
        "benchmark_loader": Path(benchmark.__file__).resolve(),
        "boundary_loader": Path(validation_plots.__file__).resolve(),
    }


def live_code_hashes() -> Mapping[str, str]:
    files = evaluation_code_files()
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"live evaluation code is incomplete: {missing}")
    return {name: sha256_file(path) for name, path in files.items()}


def build_control_data_contract(
    candidate: sweep.SweepCandidate,
    control_normalization: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    *,
    normalization_sha256: str,
    anchor_sha256: str,
) -> Mapping[str, Any]:
    """Build the predictor/target/statistics contract frozen before test access."""

    return {
        "contract_version": "independent_2025_tp_t2m_control_v2",
        "selected_configuration": candidate.name,
        "candidate": asdict(candidate),
        "forecast_dynamic_inputs": [
            "fuxi_tp_ensemble_mean_weekly",
            "fuxi_tp_ensemble_std_weekly",
            "fuxi_t2m_ensemble_mean_weekly",
        ],
        "forbidden_missing_inputs": [
            *sweep.PHYSICAL_PREDICTOR_NAMES,
            "fuxi_member_summary_channels",
        ],
        "missing_input_policy": "fail before 2025 access; never synthesize zeros",
        "input_channels": list(_expected_input_channels()),
        "effective_backbone_channels": sweep.STANDARD_BACKBONE_CHANNELS,
        "projected_training_normalization_sha256": canonical_json_sha256(
            control_normalization
        ),
        "normalization_file_sha256": normalization_sha256,
        "log_bias_anchor_sha256": anchor_sha256,
        "training_years": list(run_manifest.get("train_years", ())),
        "validation_years": list(run_manifest.get("validation_years", ())),
        "test_year": TEST_YEAR,
        "test_initialization_months": list(TEST_MONTHS),
        "expected_test_cases": EXPECTED_CASES,
        "lead_weeks": EXPECTED_LEADS,
        "grid_shape": list(EXPECTED_GRID),
        "india_support_cells": EXPECTED_SUPPORT_CELLS,
        "strict_archive_token": (
            "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
        ),
        "units": {"tp": "mm day-1", "t2m_saved": "degC", "t2m_model": "K"},
        "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
        "acc_reference": "fixed training-only 2002-2017 IMD climatology",
        "bootstrap": {
            "method": "paired one-year circular moving blocks by initialization",
            "draws": DEFAULT_BOOTSTRAP_DRAWS,
            "block_length_starts": DEFAULT_BLOCK_LENGTH,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "all_six_leads_kept_together": True,
            "inference_language": (
                "percentile interval support only; no bootstrap p-values, "
                "null test, multiplicity-adjusted q-values, or significance claim"
            ),
        },
    }


def torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a hash-verified local checkpoint across supported Torch versions."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Torch releases before the weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint root is not a mapping: {path}")
    return payload


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"frozen artifact escapes validation run: {value}") from exc
    return resolved


def _expected_input_channels() -> tuple[str, ...]:
    offsets = benchmark.imd_experiment.OFFSETS_DAYS
    return (
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
        *(f"imd_climatology_offset_{offset:+d}d" for offset in offsets),
        *(f"fuxi_minus_imd_climatology_offset_{offset:+d}d" for offset in offsets),
    )


def project_control_normalization(
    normalization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the saved training statistics' TP/T2M-only control view.

    Physical sweep runs append nine training-normalized physical channels to
    every in-memory feature tensor, including the control.  The control wrapper
    nevertheless consumed only its first eleven channels.  For 2025 we retain
    the original saved statistics and deterministically discard only unused
    trailing physical metadata; no value is re-estimated.
    """

    expected = _expected_input_channels()
    names = tuple(str(value) for value in normalization.get("input_channels", ()))
    if len(names) < len(expected) or names[: len(expected)] != expected:
        raise ValueError("saved normalization does not begin with the frozen 29-channel contract")
    trailing = names[len(expected) :]
    allowed_trailing = set(sweep.PHYSICAL_PREDICTOR_NAMES)
    if trailing and (set(trailing) - allowed_trailing or len(trailing) != len(set(trailing))):
        raise ValueError(
            "saved normalization contains non-control member/unknown channels; "
            "2025 has no compatible input contract"
        )
    projected = copy.deepcopy(dict(normalization))
    projected["input_channels"] = list(expected)
    context = dict(projected.get("spatial_context", {}))
    if context.get("normalization_fit") != "training cases and positive target weights only":
        raise ValueError("normalization is not explicitly marked training-only")
    if int(context.get("target_support_cells", -1)) != EXPECTED_SUPPORT_CELLS:
        raise ValueError("saved normalization target support is not the frozen 171-cell mask")
    context["full_domain_channels"] = [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "fuxi_t2m_weekly",
    ]
    projected["spatial_context"] = context
    projected.pop("fuxi_physical_predictors", None)
    for name in trailing:
        projected.pop(name, None)

    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
    ):
        statistics = projected.get(name)
        if not isinstance(statistics, Mapping):
            raise ValueError(f"saved normalization lacks {name!r} training statistics")
        mean = np.asarray(statistics.get("mean_by_lead"), dtype=np.float64)
        std = np.asarray(statistics.get("std_by_lead"), dtype=np.float64)
        if (
            mean.shape != (EXPECTED_LEADS,)
            or std.shape != (EXPECTED_LEADS,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0.0)
        ):
            raise ValueError(f"invalid frozen lead-wise statistics for {name}")
    attention = projected.get("climatology_attention", {})
    if "2002-2017 only" not in str(attention.get("source", "")):
        raise ValueError("IMD climatology metadata is not restricted to 2002-2017")
    return projected


def _candidate_from_manifest(
    run_manifest: Mapping[str, Any], selected_configuration: str
) -> sweep.SweepCandidate:
    matches = [
        candidate
        for candidate in run_manifest.get("candidates", [])
        if str(candidate.get("name")) == selected_configuration
    ]
    if len(matches) != 1:
        raise ValueError(
            f"validation manifest has {len(matches)} definitions for "
            f"{selected_configuration!r}"
        )
    metadata = dict(matches[0])
    metadata["physical_predictors"] = tuple(metadata.get("physical_predictors", ()))
    candidate = sweep.SweepCandidate(**metadata)
    return candidate


def _validate_control_candidate(
    candidate: sweep.SweepCandidate,
    selection: Mapping[str, Any],
) -> None:
    explicit_contract = str(selection.get("independent_input_contract", ""))
    name_is_control = candidate.name == "physical_control"
    if not name_is_control and explicit_contract != EXPECTED_CONTROL_CONTRACT:
        raise ValueError(
            "independent 2025 evaluation refused before data access: the frozen "
            "selection is not `physical_control` and does not explicitly declare "
            f"`{EXPECTED_CONTROL_CONTRACT}`"
        )
    physical = (
        candidate.model_kind == sweep.PHYSICAL_MODEL_KIND
        or bool(candidate.physical_predictors)
    )
    incompatible = (
        physical
        or candidate.model_kind != "temporal"
        or candidate.use_member_summaries
        or candidate.backbone_channels != sweep.STANDARD_BACKBONE_CHANNELS
    )
    if incompatible:
        raise ValueError(
            "independent 2025 evaluation refused before data access: selected "
            "architecture requires physical/member inputs that are absent from "
            "the strict00z 2025 TP/T2M archive; missing channels will never be "
            "replaced with zeros"
        )
    if candidate.noise_std != 0.0 or candidate.noise_probability != 0.0:
        raise ValueError("the independent control must not depend on stochastic input noise")


def validate_frozen_control(
    validation_run: Path,
    selection_path: Path,
) -> FrozenControl:
    """Validate every frozen artifact without opening a 2025 data store."""

    validation_run = Path(validation_run).expanduser().resolve()
    selection_path = Path(selection_path).expanduser().resolve()
    run_manifest_path = validation_run / "manifest.json"
    normalization_path = validation_run / "normalization.json"
    anchor_path = validation_run / "models" / "log_bias_anchor.npz"
    for path in (run_manifest_path, normalization_path, anchor_path, selection_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required frozen artifact is missing: {path}")
    run_manifest = _read_json(run_manifest_path)
    selection = _read_json(selection_path)
    if run_manifest.get("status") != "complete" or run_manifest.get("smoke") is not False:
        raise ValueError("validation run is not a complete non-smoke experiment")
    if run_manifest.get("test_predictions_created") is not False:
        raise ValueError("validation run does not attest that test predictions were absent")
    if selection.get("status") != "frozen":
        raise ValueError("selection manifest is not frozen")
    if selection.get("frozen_before_2025_access") is not True:
        raise ValueError("selection does not attest that it was frozen before 2025 access")
    selected_configuration = str(
        selection.get("selected_configuration", selection.get("selected_model", ""))
    )
    if not selected_configuration:
        raise ValueError("selection manifest lacks selected_configuration")

    selected_run = selection.get("validation_run", selection.get("source_validation_run"))
    if selected_run is None or Path(str(selected_run)).expanduser().resolve() != validation_run:
        raise ValueError("selection manifest points to a different validation run")
    required_hashes = {
        "validation_run_manifest_sha256": sha256_file(run_manifest_path),
        "normalization_sha256": sha256_file(normalization_path),
        "log_bias_anchor_sha256": sha256_file(anchor_path),
    }
    for key, actual in required_hashes.items():
        recorded = selection.get(key)
        if key == "log_bias_anchor_sha256" and recorded is None:
            recorded = selection.get("anchor_sha256")
        if recorded != actual:
            raise ValueError(f"frozen artifact hash mismatch for {key}")

    train_years = tuple(int(value) for value in run_manifest.get("train_years", ()))
    validation_years = tuple(
        int(value) for value in run_manifest.get("validation_years", ())
    )
    if train_years != tuple(range(2002, 2018)) or validation_years != (2018, 2019):
        raise ValueError("checkpoint chronology is not the frozen 2002-2019 contract")
    if max((*train_years, *validation_years), default=9999) >= 2020:
        raise ValueError("a frozen fitting or selection year reaches the test era")
    if TEST_YEAR not in set(int(v) for v in run_manifest.get("quarantined_years", ())):
        raise ValueError("2025 was not quarantined by the validation experiment")
    if int(selection.get("selection_data_end_year", 2019)) != 2019:
        raise ValueError("selection manifest was not restricted through 2019")

    candidate = _candidate_from_manifest(run_manifest, selected_configuration)
    _validate_control_candidate(candidate, selection)
    normalization = _read_json(normalization_path)
    control_normalization = project_control_normalization(normalization)

    frozen_code_hashes = selection.get("code_sha256")
    if not isinstance(frozen_code_hashes, Mapping):
        raise ValueError("selection manifest lacks frozen live-code hashes")
    current_code_hashes = live_code_hashes()
    if dict(frozen_code_hashes) != dict(current_code_hashes):
        changed = sorted(
            name
            for name in set(frozen_code_hashes) | set(current_code_hashes)
            if frozen_code_hashes.get(name) != current_code_hashes.get(name)
        )
        raise ValueError(
            "live evaluation code differs from the pre-2025 freeze: "
            f"{changed}; create no test output"
        )

    recorded_contract = selection.get("data_contract")
    recorded_contract_hash = selection.get("data_contract_sha256")
    if not isinstance(recorded_contract, Mapping) or not isinstance(
        recorded_contract_hash, str
    ):
        raise ValueError("selection manifest lacks the frozen data-contract hash")
    if canonical_json_sha256(recorded_contract) != recorded_contract_hash:
        raise ValueError("frozen data-contract payload does not match its hash")
    live_contract = build_control_data_contract(
        candidate,
        control_normalization,
        run_manifest,
        normalization_sha256=sha256_file(normalization_path),
        anchor_sha256=sha256_file(anchor_path),
    )
    live_contract_hash = canonical_json_sha256(live_contract)
    # Canonical JSON is the contract representation. In-memory dataclass
    # tuples become JSON arrays and reload as lists, so the canonical hashes
    # are the exact, type-stable equality test.
    if live_contract_hash != recorded_contract_hash:
        raise ValueError(
            "live predictor/target/data contract differs from the pre-2025 freeze"
        )

    with np.load(anchor_path) as anchor:
        required = {"lead_month_residual", "shrinkage", "target_scale"}
        if set(anchor.files) != required:
            raise ValueError("frozen log-bias anchor has an unexpected key contract")
        anchor_residual = np.asarray(anchor["lead_month_residual"])
        anchor_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
        if anchor_residual.shape != (6, 12, 27, 27) or anchor_scale.shape != (6,):
            raise ValueError("frozen anchor has an unexpected shape")
        if not np.isfinite(anchor_scale).all() or np.any(anchor_scale <= 0.0):
            raise ValueError("frozen anchor target scale is invalid")
    mean_std = np.asarray(
        control_normalization["log_fuxi_mean"]["std_by_lead"], dtype=np.float32
    )
    anomaly_std = np.asarray(
        control_normalization["explicit_log_fuxi_anomaly"]["std_by_lead"],
        dtype=np.float32,
    )
    ratio = mean_std / anomaly_std

    checkpoint_entries = selection.get("checkpoints")
    if not isinstance(checkpoint_entries, list) or len(checkpoint_entries) != 3:
        raise ValueError("frozen selection must enumerate exactly three checkpoints")
    paths: list[Path] = []
    seeds: list[int] = []
    for entry in checkpoint_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("checkpoint entries must be objects")
        seed = int(entry["seed"])
        path = _resolve_inside(validation_run, str(entry["path"]))
        expected_path = (
            validation_run
            / "models"
            / selected_configuration
            / f"seed_{seed}"
            / "checkpoints"
            / "best.pt"
        ).resolve()
        if path != expected_path or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"unexpected or missing checkpoint for seed {seed}: {path}")
        actual_hash = sha256_file(path)
        if entry.get("sha256") != actual_hash:
            raise ValueError(f"checkpoint hash mismatch for seed {seed}")
        record_path = path.parents[1] / "run_record.json"
        record = _read_json(record_path)
        if (
            record.get("status") != "complete"
            or int(record.get("seed", -1)) != seed
            or record.get("checkpoint_sha256") != actual_hash
            or str(record.get("candidate", {}).get("name")) != selected_configuration
        ):
            raise ValueError(f"checkpoint run record is inconsistent for seed {seed}")
        payload = torch_load_checkpoint(path)
        if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
            raise ValueError(f"checkpoint payload contract is invalid for seed {seed}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"checkpoint payload seed differs for seed {seed}")
        if int(payload.get("best_epoch", -1)) != int(
            record.get("best_epoch_zero_based", -2)
        ):
            raise ValueError(f"checkpoint best epoch differs for seed {seed}")
        if not np.isclose(
            float(payload.get("best_validation_loss", np.nan)),
            float(record.get("best_validation_loss", np.nan)),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(f"checkpoint validation loss differs for seed {seed}")
        if not np.array_equal(
            np.asarray(payload.get("target_scale"), dtype=np.float32), anchor_scale
        ):
            raise ValueError(f"checkpoint target scale differs from anchor for seed {seed}")
        if not np.allclose(
            np.asarray(payload.get("lead_weights"), dtype=np.float64),
            np.full(6, 1.0 / 6.0),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(f"checkpoint lead weights differ for seed {seed}")
        if dict(payload.get("loss_coefficients", {})) != sweep.LOSS_COEFFICIENTS:
            raise ValueError(f"checkpoint loss coefficients differ for seed {seed}")
        probe_model = sweep.build_model(candidate, len(_expected_input_channels()), ratio)
        probe_model.load_state_dict(payload["model_state_dict"], strict=True)
        checkpoint_ratio = probe_model.mean_to_anomaly_ratio.detach().cpu().numpy()
        if not np.allclose(checkpoint_ratio, ratio, rtol=0.0, atol=1.0e-7):
            raise ValueError(
                f"checkpoint normalization ratio differs from saved statistics for seed {seed}"
            )
        del probe_model, payload
        paths.append(path)
        seeds.append(seed)
    if len(set(seeds)) != 3 or tuple(sorted(seeds)) != tuple(seeds):
        raise ValueError("checkpoint seeds must be three unique values in sorted order")

    return FrozenControl(
        validation_run=validation_run,
        selection_path=selection_path,
        selection=selection,
        run_manifest=run_manifest,
        candidate=candidate,
        checkpoint_paths=tuple(paths),
        checkpoint_seeds=tuple(seeds),
        normalization_path=normalization_path,
        normalization=normalization,
        control_normalization=control_normalization,
        anchor_path=anchor_path,
    )


def audit_live_against_frozen(
    frozen: FrozenControl,
    *,
    expected_selection_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Recheck immutable inference inputs and return explicit live/frozen hashes."""

    selection_sha256 = sha256_file(frozen.selection_path)
    if (
        expected_selection_sha256 is not None
        and selection_sha256 != expected_selection_sha256
    ):
        raise ValueError("selection manifest changed after the one-time access lock")

    frozen_code = dict(frozen.selection.get("code_sha256", {}))
    live_code = dict(live_code_hashes())
    if frozen_code != live_code:
        raise ValueError("live code changed after the pre-2025 freeze")

    frozen_checkpoints = {
        str(entry["path"]): str(entry["sha256"])
        for entry in frozen.selection.get("checkpoints", ())
    }
    live_checkpoints = {
        str(path.relative_to(frozen.validation_run)): sha256_file(path)
        for path in frozen.checkpoint_paths
    }
    if frozen_checkpoints != live_checkpoints:
        raise ValueError("one or more live checkpoints changed after the freeze")

    live_run_manifest = _read_json(frozen.validation_run / "manifest.json")
    live_normalization = _read_json(frozen.normalization_path)
    live_control_normalization = project_control_normalization(live_normalization)
    live_candidate = _candidate_from_manifest(
        live_run_manifest, frozen.candidate.name
    )
    live_contract = build_control_data_contract(
        live_candidate,
        live_control_normalization,
        live_run_manifest,
        normalization_sha256=sha256_file(frozen.normalization_path),
        anchor_sha256=sha256_file(frozen.anchor_path),
    )
    frozen_contract = frozen.selection.get("data_contract")
    frozen_contract_hash = str(frozen.selection.get("data_contract_sha256", ""))
    live_contract_hash = canonical_json_sha256(live_contract)
    if (
        not isinstance(frozen_contract, Mapping)
        or canonical_json_sha256(frozen_contract) != frozen_contract_hash
        or live_contract_hash != frozen_contract_hash
    ):
        raise ValueError("live data contract changed after the pre-2025 freeze")

    frozen_files = {
        "validation_run_manifest_sha256": str(
            frozen.selection.get("validation_run_manifest_sha256", "")
        ),
        "normalization_sha256": str(
            frozen.selection.get("normalization_sha256", "")
        ),
        "log_bias_anchor_sha256": str(
            frozen.selection.get("log_bias_anchor_sha256", "")
        ),
    }
    live_files = {
        "validation_run_manifest_sha256": sha256_file(
            frozen.validation_run / "manifest.json"
        ),
        "normalization_sha256": sha256_file(frozen.normalization_path),
        "log_bias_anchor_sha256": sha256_file(frozen.anchor_path),
    }
    if frozen_files != live_files:
        raise ValueError("a frozen validation/data-contract file changed after access")

    return {
        "all_live_vs_frozen_hashes_match": True,
        "selection_manifest_sha256": selection_sha256,
        "code": {"frozen": frozen_code, "live": live_code, "match": True},
        "checkpoints": {
            "frozen": frozen_checkpoints,
            "live": live_checkpoints,
            "match": True,
        },
        "data_contract": {
            "frozen_sha256": frozen_contract_hash,
            "live_sha256": live_contract_hash,
            "match": True,
        },
        "validation_contract_files": {
            "frozen": frozen_files,
            "live": live_files,
            "match": True,
        },
    }


def create_access_ledger(path: Path, frozen: FrozenControl, output: Path) -> None:
    """Atomically consume the declared one-time independent-test access."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "2025_access_started",
        "created_utc": utc_now(),
        "test_year": TEST_YEAR,
        "selection_manifest": str(frozen.selection_path),
        "selection_manifest_sha256": sha256_file(frozen.selection_path),
        "validation_run": str(frozen.validation_run),
        "output": str(Path(output).resolve()),
        "policy": "single frozen evaluation; no fitting, tuning, or reselection on 2025",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"independent 2025 access was already consumed: {path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def zarr_metadata_hashes(path: Path) -> Mapping[str, str]:
    """Hash only Zarr metadata files, not large chunk payloads."""

    path = Path(path)
    names = {".zmetadata", ".zgroup", ".zattrs", ".zarray", "zarr.json"}
    files = sorted(item for item in path.rglob("*") if item.is_file() and item.name in names)
    if not files:
        raise ValueError(f"Zarr store contains no recognizable metadata files: {path}")
    return {str(item.relative_to(path)): sha256_file(item) for item in files}


def load_spatial_support() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the established India mask/area grid used by the benchmark."""

    path = Path(diagnostic.SPATIAL_SUPPORT)
    with xr.open_zarr(path, consolidated=True) as dataset:
        area = dataset.india_area_weight_km2.load().values.astype(np.float64)
        latitude = dataset.latitude.values.astype(np.float64)
        longitude = dataset.longitude.values.astype(np.float64)
    support = np.isfinite(area) & (area > 0.0)
    if area.shape != EXPECTED_GRID or int(support.sum()) != EXPECTED_SUPPORT_CELLS:
        raise ValueError("standardized India spatial support contract changed")
    return support, area, latitude, longitude, str(path)


def load_operational_2025(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[Any, np.ndarray, tuple[str, ...], Mapping[str, Mapping[str, str]]]:
    """Load exactly 35 strict-00Z operational FuXi JJAS starts for 2025."""

    tp_records, _ = diagnostic.load_catalog_records()
    t2m_records = benchmark.catalog_records("t2m")
    tp_record = benchmark.find_record(tp_records, "fuxi_s2s", TEST_YEAR)
    t2m_record = benchmark.find_record(t2m_records, "fuxi_s2s", TEST_YEAR)
    stores = (str(tp_record["store"]), str(t2m_record["store"]))
    required_store_token = "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
    if any(required_store_token not in store for store in stores):
        raise ValueError("2025 FuXi catalog does not point to the strict00z ens50 archive")

    with xr.open_zarr(stores[0], consolidated=True) as dataset:
        all_inits = pd.DatetimeIndex(dataset.init.values)
        inits = all_inits[all_inits.month.isin(TEST_MONTHS)]
        if len(inits) != EXPECTED_CASES or not inits.is_monotonic_increasing:
            raise ValueError(f"expected 35 sorted 2025 JJAS starts, found {len(inits)}")
        if (
            np.any(inits.year != TEST_YEAR)
            or np.any(inits.hour != 0)
            or inits.has_duplicates
        ):
            raise ValueError("strict00z 2025 initialization contract changed")
        if dataset.ensemble_mean_weekly.attrs.get("units") != "mm day-1":
            raise ValueError("2025 TP units are not mm day-1")
        mean = dataset.ensemble_mean_weekly.sel(init=inits).load().values.astype(np.float32)
        spread = dataset.ensemble_std_weekly.sel(init=inits).load().values.astype(np.float32)
        current_latitude = dataset.latitude.values.astype(np.float64)
        current_longitude = dataset.longitude.values.astype(np.float64)
    with xr.open_zarr(stores[1], consolidated=True) as dataset:
        if dataset.ensemble_mean_weekly.attrs.get("units") != "degC":
            raise ValueError("2025 T2M units are not degC")
        t2m = (
            dataset.ensemble_mean_weekly.sel(init=inits).load().values
            + np.float32(273.15)
        ).astype(np.float32)
        t2m_latitude = dataset.latitude.values.astype(np.float64)
        t2m_longitude = dataset.longitude.values.astype(np.float64)
    expected_shape = (EXPECTED_CASES, EXPECTED_LEADS, *EXPECTED_GRID)
    if mean.shape != expected_shape or spread.shape != mean.shape or t2m.shape != mean.shape:
        raise ValueError(f"unexpected 2025 FuXi weekly shapes: {mean.shape}")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(spread).all()
        or not np.isfinite(t2m).all()
        or np.any(mean < 0.0)
        or np.any(spread < 0.0)
    ):
        raise ValueError("2025 FuXi fields contain invalid values")
    for current in (current_latitude, t2m_latitude):
        if not np.array_equal(current, latitude):
            raise ValueError("2025 FuXi latitude differs from frozen support")
    for current in (current_longitude, t2m_longitude):
        if not np.array_equal(current, longitude):
            raise ValueError("2025 FuXi longitude differs from frozen support")
    initializations = inits.values.astype("datetime64[D]")
    forecast = base.ForecastData(
        initializations=initializations,
        valid_dates=base.derive_valid_dates(initializations),
        ensemble_mean=mean,
        ensemble_spread=spread,
        latitude=latitude,
        longitude=longitude,
        source_files=stores,
    )
    hashes = {store: zarr_metadata_hashes(Path(store)) for store in stores}
    return forecast, t2m, stores, hashes


def load_imd_2025(
    forecast: Any,
    training_climatology: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, Mapping[str, str]]:
    """Align daily IMD to the six fixed seven-day verification windows."""

    dates, values, coverage = diagnostic.annual_observation(TEST_YEAR)
    all_dates = dates.values.astype("datetime64[D]")
    requested = forecast.valid_dates.reshape(-1)
    if set(pd.DatetimeIndex(requested).year) != {TEST_YEAR}:
        raise ValueError("2025 weekly verification unexpectedly crosses a calendar year")
    positions = np.searchsorted(all_dates, requested)
    if np.any(positions >= len(all_dates)) or not np.array_equal(
        all_dates[positions], requested
    ):
        raise ValueError("one or more 2025 IMD verification dates are missing")
    daily = values[positions].reshape(EXPECTED_CASES, 6, 7, *EXPECTED_GRID)
    daily_coverage = coverage[positions].reshape(EXPECTED_CASES, 6, 7, *EXPECTED_GRID)
    daily_denominator = daily_coverage.sum(axis=2, dtype=np.float64)
    truth = np.divide(
        (
            np.nan_to_num(daily, nan=0.0) * daily_coverage
        ).sum(axis=2, dtype=np.float64),
        daily_denominator,
        out=np.full((EXPECTED_CASES, 6, *EXPECTED_GRID), np.nan, dtype=np.float64),
        where=daily_denominator > 0.0,
    ).astype(np.float32)
    weekly_coverage = np.min(daily_coverage, axis=2).astype(np.float32)
    climatology = np.mean(
        training_climatology[base.calendar_positions(forecast.valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    usable = support[None, None] & (weekly_coverage > 0.0)
    if not np.any(usable) or not np.isfinite(truth[usable]).all():
        raise ValueError("2025 IMD truth is incomplete on all supported cases")
    store = str(
        diagnostic.OBS_ROOT
        / "daily/imd/tp/india_1p5_27x27_v1"
        / f"{TEST_YEAR}.zarr"
    )
    return truth, climatology, weekly_coverage, store, zarr_metadata_hashes(Path(store))


def load_test_data() -> TestData:
    """Load test predictors/truth only after the frozen preflight and access lock."""

    support, area, latitude, longitude, spatial_store = load_spatial_support()
    forecast, t2m, forecast_stores, source_hashes = load_operational_2025(
        latitude, longitude
    )
    training_climatology, training_stores = benchmark.build_training_climatology(support)
    truth, climatology, coverage, verification_store, verification_hashes = load_imd_2025(
        forecast, training_climatology, support
    )
    hashes = dict(source_hashes)
    hashes[verification_store] = verification_hashes
    hashes[spatial_store] = zarr_metadata_hashes(Path(spatial_store))
    for store in training_stores:
        hashes[str(store)] = zarr_metadata_hashes(Path(store))
    return TestData(
        forecast=forecast,
        t2m_weekly=t2m,
        truth=truth,
        climatology=climatology,
        weekly_coverage=coverage,
        training_climatology_daily=training_climatology,
        support=support,
        area_weight_km2=area,
        source_stores=forecast_stores,
        training_imd_stores=training_stores,
        verification_imd_store=verification_store,
        source_metadata_hashes=hashes,
    )


def build_frozen_features(
    data: TestData,
    frozen: FrozenControl,
) -> np.ndarray:
    features, rebuilt_climatology = benchmark.build_features(
        data.forecast,
        data.t2m_weekly,
        data.training_climatology_daily,
        frozen.control_normalization,
        data.support,
    )
    if features.shape != (EXPECTED_CASES, 6, 29, *EXPECTED_GRID):
        raise ValueError(f"unexpected frozen 2025 control feature shape: {features.shape}")
    if not np.allclose(
        rebuilt_climatology,
        data.climatology,
        rtol=0.0,
        atol=2.0e-6,
        equal_nan=True,
    ):
        raise ValueError("feature and verification training-only climatologies differ")
    return features


def infer_frozen_ensemble(
    features: np.ndarray,
    data: TestData,
    frozen: FrozenControl,
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact three frozen residual checkpoints without fitting."""

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean_std = np.asarray(
        frozen.control_normalization["log_fuxi_mean"]["std_by_lead"],
        dtype=np.float32,
    )
    anomaly_std = np.asarray(
        frozen.control_normalization["explicit_log_fuxi_anomaly"]["std_by_lead"],
        dtype=np.float32,
    )
    ratio = mean_std / anomaly_std
    members: list[np.ndarray] = []
    for seed, checkpoint_path in zip(
        frozen.checkpoint_seeds, frozen.checkpoint_paths, strict=True
    ):
        set_deterministic_seed(seed)
        model = sweep.build_model(frozen.candidate, features.shape[2], ratio)
        checkpoint = torch_load_checkpoint(checkpoint_path)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"checkpoint lacks model_state_dict: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        member = predict(
            model,
            features,
            device=target_device,
            batch_size=32,
            use_amp=False,
        )
        if member.shape != data.truth.shape or not np.isfinite(member).all():
            raise ValueError(f"invalid residual from seed {seed}: {member.shape}")
        members.append(member)
    residual = np.mean(np.stack(members), axis=0, dtype=np.float64).astype(np.float32)
    with np.load(frozen.anchor_path) as anchor:
        correction = LogBiasCorrection(
            np.asarray(anchor["lead_month_residual"], dtype=np.float32),
            float(anchor["shrinkage"]),
        )
        target_scale = np.asarray(anchor["target_scale"], dtype=np.float32)
    log_bias = apply_log_bias_correction(
        data.forecast.ensemble_mean,
        data.forecast.initializations,
        correction,
    )
    valid = np.broadcast_to(data.support[None, None], data.truth.shape)
    corrected = reconstruct_anchored_precipitation(
        log_bias,
        residual,
        target_scale,
        valid_mask=valid,
    )
    return corrected, log_bias, residual


def weighted_case_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    coverage: np.ndarray,
    area_weight_km2: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Return case/lead India-area scores under the common IMD reference."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    climatology = np.asarray(climatology, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    area = np.asarray(area_weight_km2, dtype=np.float64)
    if prediction.shape != truth.shape or truth.shape != climatology.shape:
        raise ValueError("prediction/truth/climatology shapes differ")
    if coverage.shape != truth.shape or area.shape != truth.shape[-2:]:
        raise ValueError("verification weights do not match fields")
    weight = area[None, None] * np.where(np.isfinite(coverage), coverage, 0.0)
    finite = np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(climatology)
    weight = np.where(finite & (weight > 0.0), weight, 0.0)
    denominator = weight.sum(axis=(-2, -1), dtype=np.float64)
    if np.any(denominator <= 0.0):
        raise ValueError("one or more 2025 case/leads has no verification weight")
    error = prediction - truth
    rmse = np.sqrt(
        (weight * error**2).sum(axis=(-2, -1), dtype=np.float64) / denominator
    )
    bias = (weight * error).sum(axis=(-2, -1), dtype=np.float64) / denominator

    forecast_anomaly = prediction - climatology
    truth_anomaly = truth - climatology
    forecast_mean = (
        weight * forecast_anomaly
    ).sum(axis=(-2, -1), dtype=np.float64) / denominator
    truth_mean = (
        weight * truth_anomaly
    ).sum(axis=(-2, -1), dtype=np.float64) / denominator
    forecast_centered = forecast_anomaly - forecast_mean[..., None, None]
    truth_centered = truth_anomaly - truth_mean[..., None, None]
    covariance = (
        weight * forecast_centered * truth_centered
    ).sum(axis=(-2, -1), dtype=np.float64)
    variance_forecast = (
        weight * forecast_centered**2
    ).sum(axis=(-2, -1), dtype=np.float64)
    variance_truth = (
        weight * truth_centered**2
    ).sum(axis=(-2, -1), dtype=np.float64)
    acc_denominator = np.sqrt(variance_forecast * variance_truth)
    acc = np.divide(
        covariance,
        acc_denominator,
        out=np.full_like(covariance, np.nan),
        where=acc_denominator > 0.0,
    )
    return {
        "rmse_mm_day": rmse,
        "bias_mm_day": bias,
        "acc": acc,
    }


def circular_moving_block_indices(
    case_count: int,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Draw circular initialization blocks, retaining full six-lead cases."""

    if case_count < 2 or draws < 1:
        raise ValueError("case_count and draws must be positive")
    if not 1 <= block_length <= case_count:
        raise ValueError("block_length must be between one and case_count")
    generator = np.random.default_rng(seed)
    block_count = math.ceil(case_count / block_length)
    starts = generator.integers(0, case_count, size=(draws, block_count))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % case_count
    return indices.reshape(draws, -1)[:, :case_count].astype(np.int16)


def bootstrap_metric_summary(
    raw: Mapping[str, np.ndarray],
    corrected: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> pd.DataFrame:
    """Summarize paired lead metrics and circular-block uncertainty."""

    indices = np.asarray(indices, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        raw_values = np.asarray(raw[metric], dtype=np.float64)
        corrected_values = np.asarray(corrected[metric], dtype=np.float64)
        if raw_values.shape != corrected_values.shape or raw_values.ndim != 2:
            raise ValueError(f"{metric} must share [case, lead] arrays")
        for lead_index in range(raw_values.shape[1]):
            sampled_raw = np.nanmean(raw_values[indices, lead_index], axis=1)
            sampled_corrected = np.nanmean(
                corrected_values[indices, lead_index], axis=1
            )
            raw_point = float(np.nanmean(raw_values[:, lead_index]))
            corrected_point = float(np.nanmean(corrected_values[:, lead_index]))
            if metric == "rmse_mm_day":
                improvement_draws = sampled_raw - sampled_corrected
                improvement = raw_point - corrected_point
                improvement_pct_draws = 100.0 * improvement_draws / sampled_raw
                improvement_pct = 100.0 * improvement / raw_point
                direction = "positive means lower corrected RMSE"
            elif metric == "acc":
                improvement_draws = sampled_corrected - sampled_raw
                improvement = corrected_point - raw_point
                improvement_pct_draws = np.full_like(improvement_draws, np.nan)
                improvement_pct = float("nan")
                direction = "positive means higher corrected ACC"
            else:
                improvement_draws = np.abs(sampled_raw) - np.abs(sampled_corrected)
                improvement = abs(raw_point) - abs(corrected_point)
                improvement_pct_draws = np.full_like(improvement_draws, np.nan)
                improvement_pct = float("nan")
                direction = "positive means smaller absolute corrected bias"
            improvement_ci_low = float(
                np.nanquantile(improvement_draws, 0.025)
            )
            improvement_ci_high = float(
                np.nanquantile(improvement_draws, 0.975)
            )
            row = {
                "metric": metric,
                "lead_week": lead_index + 1,
                "raw": raw_point,
                "raw_ci_low": float(np.nanquantile(sampled_raw, 0.025)),
                "raw_ci_high": float(np.nanquantile(sampled_raw, 0.975)),
                "corrected": corrected_point,
                "corrected_ci_low": float(np.nanquantile(sampled_corrected, 0.025)),
                "corrected_ci_high": float(np.nanquantile(sampled_corrected, 0.975)),
                "improvement": float(improvement),
                "improvement_ci_low": improvement_ci_low,
                "improvement_ci_high": improvement_ci_high,
                "improvement_pct": float(improvement_pct),
                "improvement_pct_ci_low": float(
                    np.nanquantile(improvement_pct_draws, 0.025)
                ) if metric == "rmse_mm_day" else float("nan"),
                "improvement_pct_ci_high": float(
                    np.nanquantile(improvement_pct_draws, 0.975)
                ) if metric == "rmse_mm_day" else float("nan"),
                "bootstrap_supported_improvement_95": bool(
                    improvement > 0.0 and improvement_ci_low > 0.0
                ),
                "bootstrap_interval": (
                    "paired circular moving-block percentile 95%; descriptive "
                    "effect interval, not a null-hypothesis test"
                ),
                "improvement_direction": direction,
            }
            rows.append(row)

        sampled_raw = np.nanmean(raw_values[indices], axis=(1, 2))
        sampled_corrected = np.nanmean(corrected_values[indices], axis=(1, 2))
        raw_point = float(np.nanmean(raw_values))
        corrected_point = float(np.nanmean(corrected_values))
        if metric == "rmse_mm_day":
            improvement_draws = sampled_raw - sampled_corrected
            improvement = raw_point - corrected_point
            percent_draws = 100.0 * improvement_draws / sampled_raw
            percent = 100.0 * improvement / raw_point
        elif metric == "acc":
            improvement_draws = sampled_corrected - sampled_raw
            improvement = corrected_point - raw_point
            percent_draws = np.full_like(improvement_draws, np.nan)
            percent = float("nan")
        else:
            improvement_draws = np.abs(sampled_raw) - np.abs(sampled_corrected)
            improvement = abs(raw_point) - abs(corrected_point)
            percent_draws = np.full_like(improvement_draws, np.nan)
            percent = float("nan")
        improvement_ci_low = float(np.nanquantile(improvement_draws, 0.025))
        improvement_ci_high = float(np.nanquantile(improvement_draws, 0.975))
        rows.append(
            {
                "metric": metric,
                "lead_week": "all",
                "raw": raw_point,
                "raw_ci_low": float(np.nanquantile(sampled_raw, 0.025)),
                "raw_ci_high": float(np.nanquantile(sampled_raw, 0.975)),
                "corrected": corrected_point,
                "corrected_ci_low": float(np.nanquantile(sampled_corrected, 0.025)),
                "corrected_ci_high": float(np.nanquantile(sampled_corrected, 0.975)),
                "improvement": float(improvement),
                "improvement_ci_low": improvement_ci_low,
                "improvement_ci_high": improvement_ci_high,
                "improvement_pct": float(percent),
                "improvement_pct_ci_low": float(np.nanquantile(percent_draws, 0.025))
                if metric == "rmse_mm_day"
                else float("nan"),
                "improvement_pct_ci_high": float(np.nanquantile(percent_draws, 0.975))
                if metric == "rmse_mm_day"
                else float("nan"),
                "bootstrap_supported_improvement_95": bool(
                    improvement > 0.0 and improvement_ci_low > 0.0
                ),
                "bootstrap_interval": (
                    "paired circular moving-block percentile 95%; descriptive "
                    "effect interval, not a null-hypothesis test"
                ),
                "improvement_direction": rows[-1]["improvement_direction"],
            }
        )
    return pd.DataFrame(rows)


def cellwise_rmse_skill_interval(
    raw: np.ndarray,
    corrected: np.ndarray,
    truth: np.ndarray,
    coverage: np.ndarray,
    support: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local skill and an unadjusted paired percentile-effect interval.

    This is descriptive spatial uncertainty, not a field significance test.
    The bootstrap distribution is not recentered under a null, and intervals
    are not multiplicity-adjusted across cells.
    """

    raw = np.asarray(raw, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    coverage = np.asarray(coverage, dtype=np.float64)
    if coverage.shape != truth.shape:
        raise ValueError("cellwise coverage does not match verification fields")
    valid_case = (
        support[None, None]
        & np.isfinite(raw)
        & np.isfinite(corrected)
        & np.isfinite(truth)
        & np.isfinite(coverage)
        & (coverage > 0.0)
    )
    case_weight = np.where(valid_case, coverage, 0.0)
    raw_squared = np.where(valid_case, (raw - truth) ** 2, 0.0)
    corrected_squared = np.where(valid_case, (corrected - truth) ** 2, 0.0)
    denominator = case_weight.sum(axis=0, dtype=np.float64)
    raw_mse = np.divide(
        (case_weight * raw_squared).sum(axis=0, dtype=np.float64),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    corrected_mse = np.divide(
        (case_weight * corrected_squared).sum(axis=0, dtype=np.float64),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    raw_rmse = np.sqrt(raw_mse)
    corrected_rmse = np.sqrt(corrected_mse)
    skill = np.full(raw_rmse.shape, np.nan, dtype=np.float64)
    valid = support[None] & np.isfinite(raw_rmse) & (raw_rmse > 1.0e-10)
    skill[valid] = 100.0 * (raw_rmse[valid] - corrected_rmse[valid]) / raw_rmse[valid]

    case_count = raw.shape[0]
    counts = np.zeros((len(bootstrap_indices), case_count), dtype=np.float32)
    for draw, indices in enumerate(np.asarray(bootstrap_indices, dtype=np.int64)):
        counts[draw] = np.bincount(indices, minlength=case_count) / float(case_count)
    weighted_difference = case_weight * (raw_squared - corrected_squared)
    bootstrap_numerator = np.einsum(
        "dc,clhw->dlhw", counts, weighted_difference, optimize=True
    )
    bootstrap_denominator = np.einsum(
        "dc,clhw->dlhw", counts, case_weight, optimize=True
    )
    bootstrap_difference = np.divide(
        bootstrap_numerator,
        bootstrap_denominator,
        out=np.full_like(bootstrap_numerator, np.nan),
        where=bootstrap_denominator > 0.0,
    )
    interval_low = np.nanquantile(bootstrap_difference, 0.025, axis=0)
    interval_high = np.nanquantile(bootstrap_difference, 0.975, axis=0)
    interval_low[:, ~support] = np.nan
    interval_high[:, ~support] = np.nan
    bootstrap_supported = support[None] & (skill > 0.0) & (interval_low > 0.0)
    return skill, interval_low, interval_high, bootstrap_supported


def case_metrics_table(
    initializations: np.ndarray,
    metric_values: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.DatetimeIndex(initializations)
    for method in METHODS:
        for case_index, initialization in enumerate(dates):
            for lead in range(EXPECTED_LEADS):
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "initialization": str(initialization.date()),
                        "lead_week": lead + 1,
                        **{
                            metric: float(metric_values[method][metric][case_index, lead])
                            for metric in METRICS
                        },
                    }
                )
    return pd.DataFrame(rows)


def _add_boundary(axis: plt.Axes, segments: Sequence[np.ndarray]) -> None:
    axis.add_collection(
        LineCollection(
            segments,
            colors="#202020",
            linewidths=0.32,
            alpha=0.78,
            zorder=5,
        )
    )


def _masked(field: np.ndarray, support: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~support | ~np.isfinite(field), field)


def _map_axis(
    axis: plt.Axes,
    latitude: np.ndarray,
    longitude: np.ndarray,
    field: np.ndarray,
    support: np.ndarray,
    segments: Sequence[np.ndarray],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
) -> Any:
    image = axis.pcolormesh(
        longitude,
        latitude,
        _masked(field, support),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    _add_boundary(axis, segments)
    axis.set_xlim(66.0, 99.0)
    axis.set_ylim(6.0, 38.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=10.2, weight="semibold", pad=6)
    axis.tick_params(labelsize=7, length=2)
    axis.grid(color="white", alpha=0.20, linewidth=0.35)
    return image


def plot_spatial_lead(
    output: Path,
    *,
    lead: int,
    truth: np.ndarray,
    raw: np.ndarray,
    corrected: np.ndarray,
    coverage: np.ndarray,
    skill: np.ndarray,
    bootstrap_supported: np.ndarray,
    support: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    boundary_segments: Sequence[np.ndarray],
    lead_summary: pd.DataFrame,
) -> None:
    """Write the requested observation/raw/corrected 2x3 panel for one lead."""

    index = lead - 1
    current_weight = np.where(
        np.isfinite(coverage[:, index]) & (coverage[:, index] > 0.0),
        coverage[:, index],
        0.0,
    )
    def weighted_case_mean(values: np.ndarray) -> np.ndarray:
        finite_weight = np.where(np.isfinite(values), current_weight, 0.0)
        current_denominator = finite_weight.sum(axis=0, dtype=np.float64)
        return np.divide(
            (finite_weight * np.where(np.isfinite(values), values, 0.0)).sum(
                axis=0, dtype=np.float64
            ),
            current_denominator,
            out=np.full(EXPECTED_GRID, np.nan, dtype=np.float64),
            where=current_denominator > 0.0,
        )

    truth_mean = weighted_case_mean(truth[:, index])
    raw_mean = weighted_case_mean(raw[:, index])
    corrected_mean = weighted_case_mean(corrected[:, index])
    raw_error = weighted_case_mean(raw[:, index] - truth[:, index])
    corrected_error = weighted_case_mean(corrected[:, index] - truth[:, index])
    rain_values = np.concatenate(
        (truth_mean[support], raw_mean[support], corrected_mean[support])
    )
    rain_max = max(1.0, float(np.nanquantile(rain_values, 0.99)))
    error_values = np.concatenate((raw_error[support], corrected_error[support]))
    error_limit = max(0.5, float(np.nanquantile(np.abs(error_values), 0.98)))
    finite_skill = np.abs(skill[index, support])
    skill_limit = min(100.0, max(10.0, float(np.nanquantile(finite_skill, 0.95))))

    figure = plt.figure(figsize=(13.2, 9.6))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(1.0, 0.055, 1.0, 0.055),
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.885,
        wspace=0.16,
        hspace=0.30,
    )
    axes = np.empty((2, 3), dtype=object)
    for column in range(3):
        axes[0, column] = figure.add_subplot(grid[0, column])
        axes[1, column] = figure.add_subplot(grid[2, column])
    top_color_axis = figure.add_subplot(grid[1, :])
    error_color_axis = figure.add_subplot(grid[3, :2])
    skill_color_axis = figure.add_subplot(grid[3, 2])
    top_images = []
    for axis, field, title in zip(
        axes[0],
        (truth_mean, raw_mean, corrected_mean),
        ("IMD observation", "Raw FuXi-S2S", "Frozen corrected"),
        strict=True,
    ):
        top_images.append(
            _map_axis(
                axis,
                latitude,
                longitude,
                field,
                support,
                boundary_segments,
                cmap="YlGnBu",
                vmin=0.0,
                vmax=rain_max,
                title=title,
            )
        )
    bottom_images = []
    for axis, field, title in zip(
        axes[1, :2],
        (raw_error, corrected_error),
        ("Raw error (forecast − IMD)", "Corrected error (forecast − IMD)"),
        strict=True,
    ):
        bottom_images.append(
            _map_axis(
                axis,
                latitude,
                longitude,
                field,
                support,
                boundary_segments,
                cmap="RdBu_r",
                vmin=-error_limit,
                vmax=error_limit,
                title=title,
            )
        )
    metric_row = lead_summary.loc[
        lead_summary.metric.eq("rmse_mm_day")
        & lead_summary.lead_week.astype(str).eq(str(lead))
    ].iloc[0]
    skill_title = (
        "Local RMSE improvement (%)\n"
        f"India W{lead}: {metric_row.improvement_pct:+.1f}% "
        f"[{metric_row.improvement_pct_ci_low:+.1f}, "
        f"{metric_row.improvement_pct_ci_high:+.1f}]"
    )
    skill_image = _map_axis(
        axes[1, 2],
        latitude,
        longitude,
        skill[index],
        support,
        boundary_segments,
        cmap="RdYlGn",
        vmin=-skill_limit,
        vmax=skill_limit,
        title=skill_title,
    )
    rows, columns = np.where(bootstrap_supported[index])
    if len(rows):
        axes[1, 2].scatter(
            longitude[columns],
            latitude[rows],
            s=5,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.32,
            zorder=7,
        )

    top_colorbar = figure.colorbar(
        top_images[0],
        cax=top_color_axis,
        orientation="horizontal",
    )
    top_colorbar.set_label("Weekly mean rainfall (mm day$^{-1}$)", fontsize=9, labelpad=3)
    top_colorbar.ax.xaxis.set_label_position("top")
    error_colorbar = figure.colorbar(
        bottom_images[0],
        cax=error_color_axis,
        orientation="horizontal",
    )
    error_colorbar.set_label("Mean error (mm day$^{-1}$)", fontsize=9)
    skill_colorbar = figure.colorbar(
        skill_image,
        cax=skill_color_axis,
        orientation="horizontal",
    )
    skill_colorbar.set_label("RMSE improvement vs raw FuXi (%)", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude (°N)", fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Longitude (°E)", fontsize=8)
    figure.suptitle(
        f"Independent operational 2025 · Lead week {lead}",
        fontsize=17,
        weight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.930,
        "35 strict-00Z JJAS starts · frozen pre-2020 three-seed ensemble · "
        "IMD verification · no 2025 fitting or tuning",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.text(
        0.5,
        0.012,
        "Open circles: local MSE improvement with an unadjusted paired 95% "
        "moving-block percentile interval wholly above zero. Descriptive "
        "bootstrap support only—no field-significance or multiplicity claim. "
        "Boundary: Survey of India ABDB derivative.",
        ha="center",
        fontsize=7.6,
        color="#555555",
    )
    stem = output / "figures" / f"spatial_independent_2025_W{lead}"
    figure.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), dpi=320, facecolor="white")
    plt.close(figure)


def plot_metric_curves(output: Path, summary: pd.DataFrame) -> None:
    """Plot metrics with paired percentile intervals and support markers."""

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    specifications = (
        ("acc", "Anomaly correlation", None),
        ("rmse_mm_day", "RMSE (mm day$^{-1}$)", None),
        ("bias_mm_day", "Bias (mm day$^{-1}$)", 0.0),
    )
    weeks = np.arange(1, 7)
    for axis, (metric, ylabel, zero_line) in zip(axes, specifications, strict=True):
        current = summary.loc[
            summary.metric.eq(metric) & summary.lead_week.ne("all")
        ].copy()
        current["lead_week"] = current.lead_week.astype(int)
        current = current.sort_values("lead_week")
        for method in METHODS:
            column = "raw" if method == "raw_fuxi" else "corrected"
            values = current[column].to_numpy(dtype=float)
            low = current[f"{column}_ci_low"].to_numpy(dtype=float)
            high = current[f"{column}_ci_high"].to_numpy(dtype=float)
            axis.plot(
                weeks,
                values,
                color=METHOD_COLORS[method],
                marker="o" if method == "raw_fuxi" else "D",
                markersize=5.2,
                linewidth=2.2,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(weeks, low, high, color=METHOD_COLORS[method], alpha=0.14)
        supported = current.bootstrap_supported_improvement_95.to_numpy(dtype=bool)
        corrected = current.corrected.to_numpy(dtype=float)
        span = max(1.0e-6, float(np.nanmax(current[["raw_ci_high", "corrected_ci_high"]]) - np.nanmin(current[["raw_ci_low", "corrected_ci_low"]])))
        for week, value, passed in zip(weeks, corrected, supported, strict=True):
            if passed:
                axis.text(
                    week,
                    value + 0.045 * span,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#7A1F5C",
                )
        if metric == "rmse_mm_day":
            for _, row in current.iterrows():
                axis.annotate(
                    f"{row.improvement_pct:+.1f}%",
                    (int(row.lead_week), float(row.corrected)),
                    xytext=(0, -17),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=METHOD_COLORS["corrected"],
                )
        if zero_line is not None:
            axis.axhline(zero_line, color="#333333", linewidth=0.8, alpha=0.7)
        axis.set_xticks(weeks, [f"W{week}" for week in weeks])
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.margins(y=0.14)
        axis.grid(True, axis="y", alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(ylabel, fontsize=12, weight="semibold")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Independent operational 2025: frozen FuXi–IMD correction",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        "Bands: 95% circular moving-block bootstrap CI (2,000 draws; 13-start "
        "blocks; all six leads kept together). ★ the descriptive paired "
        "percentile interval for improvement is wholly above zero; no p/q or "
        "multiplicity-adjusted significance claim.",
        ha="center",
        fontsize=8,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.94))
    stem = output / "figures" / "metrics_independent_2025_by_lead"
    figure.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), dpi=320, facecolor="white")
    plt.close(figure)


def plot_skill_contact_sheet(
    output: Path,
    skill: np.ndarray,
    bootstrap_supported: np.ndarray,
    support: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    boundary_segments: Sequence[np.ndarray],
) -> None:
    finite = np.abs(skill[:, support])
    limit = min(100.0, max(10.0, float(np.nanquantile(finite, 0.95))))
    figure = plt.figure(figsize=(12.2, 8.6))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.055),
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.91,
        wspace=0.13,
        hspace=0.18,
    )
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            axes[row, column] = figure.add_subplot(grid[row, column])
    color_axis = figure.add_subplot(grid[2, :])
    image = None
    for lead, axis in enumerate(axes.flat, start=1):
        image = _map_axis(
            axis,
            latitude,
            longitude,
            skill[lead - 1],
            support,
            boundary_segments,
            cmap="RdYlGn",
            vmin=-limit,
            vmax=limit,
            title=f"Lead week {lead}",
        )
        rows, columns = np.where(bootstrap_supported[lead - 1])
        if len(rows):
            axis.scatter(
                longitude[columns],
                latitude[rows],
                s=4,
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.30,
                zorder=7,
            )
    assert image is not None
    figure.colorbar(
        image,
        cax=color_axis,
        orientation="horizontal",
        label="Local RMSE improvement vs raw FuXi (%)",
    )
    figure.suptitle(
        "Independent operational 2025 · spatial RMSE improvement",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Open circles denote an unadjusted paired 95% percentile interval wholly "
        "above zero (descriptive bootstrap support; not field significance).",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    stem = output / "figures" / "spatial_rmse_skill_contact_sheet_2025"
    figure.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), dpi=320, facecolor="white")
    plt.close(figure)


def write_results(output: Path, summary: pd.DataFrame, frozen: FrozenControl) -> None:
    all_rows = summary.loc[summary.lead_week.eq("all")].set_index("metric")
    lines = [
        "# Independent operational 2025 FuXi–IMD evaluation",
        "",
        "> One-time untouched test. The architecture, three checkpoints, anchor, "
        "and all normalization statistics were frozen using data through 2019 "
        "before 2025 stores were opened.",
        "",
        "## Contract",
        "",
        f"- Frozen configuration: `{frozen.candidate.name}`",
        f"- Input contract: `{EXPECTED_CONTROL_CONTRACT}` (no physical/member inputs)",
        "- 35 strict-00Z JJAS 2025 starts; six seven-day leads kept together",
        "- IMD target; 171-cell India support; training-only 2002–2017 IMD ACC reference",
        "- 2,000 deterministic circular moving-block draws; block length 13 starts",
        "- No fitting, selection, alpha tuning, threshold tuning, or normalization on 2025",
        "",
        "## Pooled W1–W6",
        "",
        "| Metric | Raw FuXi | Corrected | Paired improvement | 95% percentile interval | Bootstrap-supported |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for metric, label in (
        ("rmse_mm_day", "RMSE (mm/day)"),
        ("acc", "ACC"),
        ("bias_mm_day", "Bias (mm/day)"),
    ):
        row = all_rows.loc[metric]
        if metric == "rmse_mm_day":
            improvement = f"{row.improvement_pct:+.2f}%"
            interval = (
                f"[{row.improvement_pct_ci_low:+.2f}%, "
                f"{row.improvement_pct_ci_high:+.2f}%]"
            )
        else:
            improvement = f"{row.improvement:+.4f}"
            interval = (
                f"[{row.improvement_ci_low:+.4f}, "
                f"{row.improvement_ci_high:+.4f}]"
            )
        lines.append(
            f"| {label} | {row.raw:.4f} | {row.corrected:.4f} | {improvement} | "
            f"{interval} | "
            f"{'yes' if bool(row.bootstrap_supported_improvement_95) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Lead-week presentation summary",
            "",
            "| Week | Raw RMSE | Corrected RMSE | RMSE improvement (95% percentile interval) | Raw ACC | Corrected ACC | Corrected bias | Bootstrap-supported RMSE gain |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for lead in range(1, 7):
        current = summary.loc[summary.lead_week.astype(str).eq(str(lead))].set_index(
            "metric"
        )
        rmse = current.loc["rmse_mm_day"]
        acc = current.loc["acc"]
        bias = current.loc["bias_mm_day"]
        supported = (
            "yes" if bool(rmse.bootstrap_supported_improvement_95) else "no"
        )
        lines.append(
            f"| W{lead} | {rmse.raw:.3f} | {rmse.corrected:.3f} | "
            f"{rmse.improvement_pct:+.1f}% "
            f"[{rmse.improvement_pct_ci_low:+.1f}, {rmse.improvement_pct_ci_high:+.1f}] | "
            f"{acc.raw:.3f} | {acc.corrected:.3f} | {bias.corrected:+.3f} | "
            f"{supported} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "`Bootstrap-supported` means only that the paired circular moving-"
            "block 95% percentile interval for the descriptive improvement is "
            "wholly above zero. The bootstrap distribution is not recentered "
            "under a null, so this workflow emits no p-values, q-values, FDR, or "
            "statistical-significance claim. Spatial circles are unadjusted and "
            "exploratory because one year provides limited independent blocks.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_fields(
    output: Path,
    data: TestData,
    corrected: np.ndarray,
    log_bias: np.ndarray,
    residual: np.ndarray,
    skill: np.ndarray,
    local_mse_improvement_ci_low: np.ndarray,
    local_mse_improvement_ci_high: np.ndarray,
    local_bootstrap_supported: np.ndarray,
) -> None:
    np.savez_compressed(
        output / "independent_2025_fields.npz",
        initializations=data.forecast.initializations,
        latitude=data.forecast.latitude,
        longitude=data.forecast.longitude,
        imd_truth=data.truth,
        raw_fuxi=data.forecast.ensemble_mean,
        log_bias=log_bias,
        corrected=corrected,
        ensemble_standardized_residual=residual,
        fixed_training_climatology=data.climatology,
        weekly_coverage=data.weekly_coverage,
        india_area_weight_km2=data.area_weight_km2,
        support=data.support,
        local_rmse_improvement_pct=skill,
        local_mse_improvement_percentile_ci_low=local_mse_improvement_ci_low,
        local_mse_improvement_percentile_ci_high=local_mse_improvement_ci_high,
        local_bootstrap_supported_95_unadjusted=local_bootstrap_supported,
    )


def _artifact_hashes(output: Path) -> Mapping[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def run_evaluation(
    frozen: FrozenControl,
    output: Path,
    *,
    boundary_path: Path,
    access_ledger: Path | None = None,
    require_cuda: bool = False,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> None:
    """Execute the already-authorized, one-time frozen test evaluation."""

    if bootstrap_draws != DEFAULT_BOOTSTRAP_DRAWS or block_length != DEFAULT_BLOCK_LENGTH:
        raise ValueError("final 2025 contract requires exactly 2,000 draws and block length 13")
    if access_ledger is None or not Path(access_ledger).is_file():
        raise ValueError("final 2025 evaluation requires a created one-time access ledger")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this final evaluation but is unavailable")
    final_output = Path(output).expanduser().resolve()
    access_ledger = Path(access_ledger).expanduser().resolve()
    ledger_payload = _read_json(access_ledger)
    selection_sha256_at_access = sha256_file(frozen.selection_path)
    if (
        ledger_payload.get("status") != "2025_access_started"
        or ledger_payload.get("selection_manifest_sha256")
        != selection_sha256_at_access
        or Path(str(ledger_payload.get("output", ""))).expanduser().resolve()
        != final_output
    ):
        raise ValueError("one-time access ledger does not match this frozen evaluation")
    if final_output.exists():
        raise FileExistsError(f"fresh final output required: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.partial-",
            dir=final_output.parent,
        )
    ).resolve()
    (output / "figures").mkdir()
    manifest_path = output / "manifest.json"
    running = {
        "status": "running",
        "scientific_status": "one-time independent operational 2025 test",
        "created_utc": utc_now(),
        "selection_manifest": str(frozen.selection_path),
        "selection_manifest_sha256": selection_sha256_at_access,
        "access_ledger": None if access_ledger is None else str(access_ledger),
        "access_ledger_sha256": (
            None if access_ledger is None else sha256_file(access_ledger)
        ),
        "no_2025_fitting_or_tuning": True,
    }
    manifest_path.write_text(json.dumps(running, indent=2) + "\n", encoding="utf-8")
    try:
        data = load_test_data()
        features = build_frozen_features(data, frozen)
        corrected, log_bias, residual = infer_frozen_ensemble(features, data, frozen)
        predictions = {
            "raw_fuxi": data.forecast.ensemble_mean,
            "corrected": corrected,
        }
        metric_values = {
            method: weighted_case_metrics(
                prediction,
                data.truth,
                data.climatology,
                data.weekly_coverage,
                data.area_weight_km2,
            )
            for method, prediction in predictions.items()
        }
        bootstrap_indices = circular_moving_block_indices(
            EXPECTED_CASES,
            draws=bootstrap_draws,
            block_length=block_length,
            seed=bootstrap_seed,
        )
        summary = bootstrap_metric_summary(
            metric_values["raw_fuxi"], metric_values["corrected"], bootstrap_indices
        )
        (
            skill,
            spatial_interval_low,
            spatial_interval_high,
            spatial_bootstrap_supported,
        ) = cellwise_rmse_skill_interval(
            predictions["raw_fuxi"],
            predictions["corrected"],
            data.truth,
            data.weekly_coverage,
            data.support,
            bootstrap_indices,
        )
        cases = case_metrics_table(data.forecast.initializations, metric_values)
        cases.to_csv(output / "case_metrics.csv", index=False)
        summary.to_csv(output / "metric_summary_block_bootstrap.csv", index=False)
        np.save(output / "bootstrap_initialization_indices.npy", bootstrap_indices)
        save_fields(
            output,
            data,
            corrected,
            log_bias,
            residual,
            skill,
            spatial_interval_low,
            spatial_interval_high,
            spatial_bootstrap_supported,
        )
        boundary_segments, boundary_provenance = validation_plots.load_india_boundary(
            boundary_path
        )
        plot_metric_curves(output, summary)
        for lead in range(1, 7):
            plot_spatial_lead(
                output,
                lead=lead,
                truth=data.truth,
                raw=predictions["raw_fuxi"],
                corrected=predictions["corrected"],
                coverage=data.weekly_coverage,
                skill=skill,
                bootstrap_supported=spatial_bootstrap_supported,
                support=data.support,
                latitude=data.forecast.latitude,
                longitude=data.forecast.longitude,
                boundary_segments=boundary_segments,
                lead_summary=summary,
            )
        plot_skill_contact_sheet(
            output,
            skill,
            spatial_bootstrap_supported,
            data.support,
            data.forecast.latitude,
            data.forecast.longitude,
            boundary_segments,
        )
        write_results(output, summary, frozen)

        projected_path = output / "frozen_control_normalization_view.json"
        projected_path.write_text(
            json.dumps(frozen.control_normalization, indent=2) + "\n",
            encoding="utf-8",
        )
        provenance_lock = audit_live_against_frozen(
            frozen,
            expected_selection_sha256=selection_sha256_at_access,
        )
        result_manifest: dict[str, Any] = {
            "status": "complete",
            "scientific_status": "one-time independent operational 2025 test",
            "completed_utc": utc_now(),
            "atomic_output": {
                "requested_final_directory": str(final_output),
                "staging_directory_name": output.name,
                "publication": "same-filesystem os.replace after complete manifest",
            },
            "test_year": TEST_YEAR,
            "initialization_months": list(TEST_MONTHS),
            "case_count": EXPECTED_CASES,
            "lead_count": EXPECTED_LEADS,
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "selected_configuration": frozen.candidate.name,
            "independent_input_contract": EXPECTED_CONTROL_CONTRACT,
            "candidate": asdict(frozen.candidate),
            "checkpoint_seeds": list(frozen.checkpoint_seeds),
            "checkpoint_sha256": {
                str(path.relative_to(frozen.validation_run)): sha256_file(path)
                for path in frozen.checkpoint_paths
            },
            "selection_manifest": str(frozen.selection_path),
            "selection_manifest_sha256": selection_sha256_at_access,
            "access_ledger": None if access_ledger is None else str(access_ledger),
            "access_ledger_sha256": (
                None if access_ledger is None else sha256_file(access_ledger)
            ),
            "validation_run": str(frozen.validation_run),
            "validation_run_manifest_sha256": sha256_file(
                frozen.validation_run / "manifest.json"
            ),
            "original_normalization_sha256": sha256_file(frozen.normalization_path),
            "projected_control_normalization_sha256": sha256_file(projected_path),
            "log_bias_anchor_sha256": sha256_file(frozen.anchor_path),
            "normalization_policy": (
                "saved training-only values; deterministic removal of unused "
                "trailing physical metadata; no refitting"
            ),
            "no_2025_fitting_tuning_selection_or_calibration": True,
            "cuda_required": bool(require_cuda),
            "support_cells": int(data.support.sum()),
            "bootstrap": {
                "method": "paired one-year circular moving blocks by initialization",
                "draws": bootstrap_draws,
                "block_length_starts": block_length,
                "seed": bootstrap_seed,
                "all_six_leads_kept_together": True,
                "indices_sha256": sha256_array(bootstrap_indices),
                "inference_language": (
                    "paired percentile-effect intervals and bootstrap-supported "
                    "labels only; no p-values, q-values, FDR, or null-test claim"
                ),
                "spatial_caveat": (
                    "unadjusted exploratory percentile intervals; one year and "
                    "approximately three 13-start blocks"
                ),
            },
            "metric_contract": {
                "rmse_bias": "mean of case-wise India-area x weekly-coverage weighted spatial scores",
                "acc": "case-wise weighted spatial ACC after subtracting fixed 2002-2017 IMD climatology",
                "bias_improvement": "reduction in absolute aggregate signed bias",
                "uncertainty": (
                    "paired circular moving-block percentile intervals; "
                    "bootstrap-supported means the descriptive improvement "
                    "interval is wholly above zero, not statistical significance"
                ),
            },
            "source_stores": {
                "operational_fuxi_tp_t2m": list(data.source_stores),
                "training_imd_climatology": list(data.training_imd_stores),
                "verification_imd": data.verification_imd_store,
                "spatial_support": str(diagnostic.SPATIAL_SUPPORT),
            },
            "source_zarr_metadata_sha256": data.source_metadata_hashes,
            "india_boundary": boundary_provenance,
            "live_vs_frozen_provenance_lock": provenance_lock,
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
        }
        result_manifest["artifacts"] = _artifact_hashes(output)
        manifest_path.write_text(
            json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8"
        )
        results_text = (output / "RESULTS.md").read_text(encoding="utf-8")
        os.replace(output, final_output)
        print(results_text, flush=True)
        print(
            f"PASS: atomically published one-time independent 2025 evaluation: "
            f"{final_output}",
            flush=True,
        )
    except Exception:
        running.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "failure": traceback.format_exc(),
            }
        )
        if manifest_path.parent.exists():
            manifest_path.write_text(
                json.dumps(running, indent=2) + "\n", encoding="utf-8"
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate frozen artifacts and boundary, then exit before ledger/data access",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail preflight unless PyTorch exposes a CUDA device",
    )
    parser.add_argument(
        "--access-ledger",
        type=Path,
        help=(
            "atomic one-time access marker; defaults beside the selection "
            "manifest and makes repeat test access fail"
        ),
    )
    parser.add_argument(
        "--india-boundary",
        type=Path,
        default=validation_plots.DEFAULT_INDIA_BOUNDARY,
    )
    args = parser.parse_args()

    # This complete preflight must stay before create_access_ledger and every
    # function that can open an operational 2025 or IMD 2025 store.
    frozen = validate_frozen_control(args.validation_run, args.selection_manifest)
    boundary = args.india_boundary.expanduser().resolve()
    # Validate the complete presentation boundary before consuming the one-time
    # test access. This reads only the checked local Survey of India derivative.
    validation_plots.load_india_boundary(boundary)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but PyTorch exposes no CUDA device")
    if args.preflight_only:
        print(
            "PASS: frozen independent-2025 control preflight; no access ledger "
            "created; metadata/checkpoint-only validation opened neither 2025 "
            "predictors nor IMD targets",
            flush=True,
        )
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"fresh output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger = (
        args.access_ledger.expanduser().resolve()
        if args.access_ledger
        else args.selection_manifest.expanduser().resolve().with_name(
            args.selection_manifest.name + ".independent_2025_access.json"
        )
    )
    create_access_ledger(ledger, frozen, output)
    run_evaluation(
        frozen,
        output,
        boundary_path=boundary,
        access_ledger=ledger,
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
    main()
