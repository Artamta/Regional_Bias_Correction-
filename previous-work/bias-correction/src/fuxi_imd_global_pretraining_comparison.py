#!/usr/bin/env python3
"""Matched India-only comparison of scratch and globally pretrained adapters.

This validation-only driver keeps the frozen FuXi--IMD feature, anchor, target,
model, optimizer, and verification contracts fixed.  Its sole experimental
factor is initialization: random weights versus a compatible global-patch
pretraining checkpoint.  It loads initialization years 2002--2019 only and
never constructs predictions or metrics for 2020 onward.

The global pretraining cache currently has no T2M predictor.  Consequently the
global model's channel-10 first-convolution slice is not learned.  After strict
checkpoint loading, this driver restores that slice from the exactly matched
scratch initialization and resets the residual head.  Both India candidates
therefore start as exact no-ops around the same training-only log-bias anchor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE

if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imd_attention_climatology as experiment  # noqa: E402
import fuxi_imd_compact_validation_sweep as sweep  # noqa: E402
import fuxi_adapter.anchored as adapter_anchored  # noqa: E402
import fuxi_adapter.baselines as adapter_baselines  # noqa: E402
import fuxi_adapter.metrics as adapter_metrics  # noqa: E402
import fuxi_adapter.models as adapter_models  # noqa: E402
import fuxi_adapter.training as adapter_training  # noqa: E402
import fuxi_adapter.v3_training as adapter_v3_training  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.metrics import paired_moving_block_bootstrap  # noqa: E402
from fuxi_adapter.models import FixedClimatologyAllLeadUNet  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


base = sweep.base
common = sweep.common
engine = sweep.engine

RESULTS_ROOT = HERE / "results" / "fuxi_imd_global_pretraining_comparison"
TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
SEALED_INITIALIZATION_YEARS = tuple(range(2020, 2026))
FULL_SEEDS = (42, 43, 44)

INPUT_CHANNELS = 29
BACKBONE_CHANNELS = 11
BASE_CHANNELS = 16
DROPOUT = 0.30
PARAMETER_COUNT = 144_689
T2M_CHANNEL_INDEX = 10
FIRST_CONV_KEY = "backbone.encoder_1.block.0.weight"
HEAD_KEYS = (
    "backbone.residual_head.weight",
    "backbone.residual_head.bias",
)
GLOBAL_FEATURE_NAMES = (
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
INDIA_BACKBONE_FEATURE_NAMES = (
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
)

BATCH_SIZE = 32
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 2.0e-3
MAX_EPOCHS = 100
PATIENCE = 15
LEAD_WEIGHTS = (1.0 / 6.0,) * 6
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}

MINIMUM_LOSS_IMPROVEMENT = 0.0025
MINIMUM_SEEDS_BEATING_SCRATCH = 2
MINIMUM_IMPROVING_LEADS = 4
MAXIMUM_LEAD_RMSE_REGRESSION_PCT = 0.5
MINIMUM_LEAD_ACC_DELTA = -0.01
MAXIMUM_POOLED_ACC_REGRESSION = 0.002
MAXIMUM_ABSOLUTE_BIAS_REGRESSION = 0.05

CANONICAL_GLOBAL_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "smoke": {
        "fit_years": (2002,),
        "validation_years": (2003,),
        "patch_size": 27,
        "patches_per_case": 1,
        "patch_seed": 20260820,
        "fit_case_limit": 8,
        "validation_case_limit": 4,
        "minimum_observation_fraction": 0.95,
        "anchor_shrinkage": 10.0,
        "base_channels": 16,
        "dropout": 0.30,
        "batch_size": 2,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 2.0e-4,
        "weight_decay": 2.0e-3,
        "smooth_l1_beta": 1.0,
        "gradient_clip": 5.0,
    },
    "full": {
        "fit_years": tuple(range(2002, 2016)),
        "validation_years": (2016, 2017),
        "patch_size": 27,
        "patches_per_case": 1,
        "patch_seed": 20260820,
        "fit_case_limit": 0,
        "validation_case_limit": 0,
        "minimum_observation_fraction": 0.95,
        "anchor_shrinkage": 10.0,
        "base_channels": 16,
        "dropout": 0.30,
        "batch_size": 32,
        "epochs": 40,
        "patience": 8,
        "learning_rate": 2.0e-4,
        "weight_decay": 2.0e-3,
        "smooth_l1_beta": 1.0,
        "gradient_clip": 5.0,
    },
}


@dataclass(frozen=True)
class PreparedBundle:
    prepared: sweep.PreparedData
    normalization: Mapping[str, Any]
    anchor: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointSource:
    path: Path
    seed: int
    sha256: str
    manifest_path: Path | None
    manifest_sha256: str | None
    source_fingerprint: str | None
    contract_sha256: str
    feature_names: tuple[str, ...]
    payload: Mapping[str, Any]


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
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    # ``memoryview.cast`` rejects NumPy datetime/timedelta dtype codes even
    # though their storage is a stable integer byte representation.  Hash the
    # contiguous C-order bytes directly so provenance covers those arrays too.
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a checkpoint replacement on the same filesystem."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@contextmanager
def validation_only_contract() -> Iterator[None]:
    """Temporarily configure existing loaders for 2002--2019 only."""

    touched = {
        experiment: (
            "TRAIN_YEARS",
            "VALIDATION_YEARS",
            "TEST_YEARS",
            "ACTIVE_LEADS",
            "ACTIVE_WEEKS",
            "ACTIVE_SCOPE",
            "INACTIVE_LEAD_COUNT",
            "LEAD_WEIGHTS",
            "VALIDATION_SCORE_COLUMN",
            "RESULTS_ROOT",
            "NORMAL",
            "ATTENTION",
            "CANDIDATES",
            "METHOD_LABELS",
        ),
        base: ("TRAIN_YEARS", "VALIDATION_YEARS", "TEST_YEARS", "ALL_YEARS"),
        engine: (
            "CANDIDATES",
            "CANDIDATE_BY_NAME",
            "METHOD_ORDER",
            "METHOD_LABELS",
            "METHOD_COLORS",
            "METHOD_MARKERS",
            "ACTIVE_LEADS",
            "LEAD_WEIGHTS",
            "LOSS_COEFFICIENTS",
        ),
        engine.diagnostics: (
            "METHOD_ORDER",
            "METHOD_LABELS",
            "METHOD_COLORS",
            "METHOD_MARKERS",
            "PLOT_METHODS",
            "LEAD_SCOPES",
            "ALL_COMPARISONS",
        ),
    }
    snapshot = {
        (module, name): getattr(module, name)
        for module, names in touched.items()
        for name in names
    }
    try:
        experiment.TRAIN_YEARS = TRAIN_YEARS
        experiment.VALIDATION_YEARS = VALIDATION_YEARS
        experiment.TEST_YEARS = ()
        experiment.set_experiment_scope(
            all_weeks=True,
            large_model=False,
            regularized_large=False,
            full_fuxi_context=True,
        )
        experiment.configure_contract()
        if tuple(base.ALL_YEARS) != TRAIN_YEARS + VALIDATION_YEARS:
            raise RuntimeError("validation-only loader contract includes sealed years")
        yield
    finally:
        for (module, name), value in snapshot.items():
            setattr(module, name, value)


def smoke_validation_indices(
    initializations: np.ndarray,
    validation_indices: np.ndarray,
    *,
    cases_per_year: int = 8,
) -> np.ndarray:
    """Select a deterministic, year-balanced validation execution subset."""

    indices = np.asarray(validation_indices, dtype=np.int64)
    years = pd.DatetimeIndex(
        np.asarray(initializations, dtype="datetime64[D]")[indices]
    ).year.to_numpy()
    selected = []
    for year in VALIDATION_YEARS:
        positions = indices[years == year]
        if len(positions) < cases_per_year:
            raise base.DataContractError(
                f"smoke validation needs {cases_per_year} cases in {year}, "
                f"found {len(positions)}"
            )
        selected.append(positions[:cases_per_year])
    result = np.concatenate(selected).astype(np.int64, copy=False)
    if len(result) != cases_per_year * len(VALIDATION_YEARS):
        raise RuntimeError("smoke validation selection count changed")
    return result


def prepare_validation_only_data(*, smoke: bool) -> PreparedBundle:
    """Build the established India arrays without opening later-year stores."""

    with validation_only_contract():
        forecast = base.load_fuxi()
        loaded_years = tuple(
            sorted(set(pd.DatetimeIndex(forecast.initializations).year.to_numpy()))
        )
        if loaded_years != TRAIN_YEARS + VALIDATION_YEARS:
            raise base.DataContractError(
                f"loaded initialization years differ from contract: {loaded_years}"
            )
        if np.any(
            np.isin(
                pd.DatetimeIndex(forecast.initializations).year.to_numpy(),
                SEALED_INITIALIZATION_YEARS,
            )
        ):
            raise base.DataContractError("a sealed initialization year was loaded")

        observations, climatology_daily, observation_dates, source_stores = (
            experiment.load_imd(forecast)
        )
        initialization_dates = np.asarray(
            forecast.initializations, dtype="datetime64[D]"
        )
        verification_dates = np.asarray(forecast.valid_dates, dtype="datetime64[D]")
        observation_dates = np.asarray(observation_dates, dtype="datetime64[D]")
        sealed_boundary = np.datetime64("2020-01-01", "D")
        if verification_dates.max() >= sealed_boundary:
            raise base.DataContractError(
                "a forecast target date reaches the sealed 2020+ boundary"
            )
        if observation_dates.max() >= sealed_boundary:
            raise base.DataContractError(
                "an IMD source contains a sealed 2020+ observation date"
            )
        expected_observation_dates = np.arange(
            np.datetime64("2002-01-01", "D"),
            sealed_boundary,
            np.timedelta64(1, "D"),
        )
        if not np.array_equal(observation_dates, expected_observation_dates):
            raise base.DataContractError(
                "IMD validation-only calendar is not the exact unique daily "
                "2002-01-01 through 2019-12-31 sequence"
            )
        observation_years = tuple(
            sorted(set(pd.DatetimeIndex(observation_dates).year.to_numpy()))
        )
        if observation_years != TRAIN_YEARS + VALIDATION_YEARS:
            raise base.DataContractError(
                "loaded IMD calendar years differ from the validation-only "
                f"contract: {observation_years}"
            )
        weights = experiment.load_imd_weights(
            forecast, observations.observation_fraction
        )
        support = weights > 0.0
        splits = base.split_indices(forecast.initializations)
        counts = {name: int(len(indices)) for name, indices in splits.items()}
        if counts != {"train": 560, "validation": 70, "test": 0}:
            raise base.DataContractError(f"unexpected split counts: {counts}")

        t2m_weekly = common.load_t2m_weekly(forecast)
        features, normalization, centre_difference = (
            experiment.build_climatology_features(
                forecast,
                observations,
                climatology_daily,
                weights,
                splits["train"],
                t2m_weekly,
                preserve_fuxi_context=True,
            )
        )
        if features.shape != (630, 6, INPUT_CHANNELS, 27, 27):
            raise base.DataContractError(
                f"unexpected India feature shape: {features.shape}"
            )
        india_feature_names = tuple(normalization["input_channels"])
        if india_feature_names[:BACKBONE_CHANNELS] != INDIA_BACKBONE_FEATURE_NAMES:
            raise base.DataContractError(
                "the established India backbone feature order changed: "
                f"{india_feature_names[:BACKBONE_CHANNELS]}"
            )

        correction = fit_log_bias_correction(
            forecast.ensemble_mean[splits["train"]],
            observations.weekly_truth[splits["train"]],
            forecast.initializations[splits["train"]],
            support,
            shrinkage=10.0,
        )
        bias_baseline = apply_log_bias_correction(
            forecast.ensemble_mean, forecast.initializations, correction
        )
        valid_mask = np.broadcast_to(
            support[None, None], bias_baseline.shape
        ).copy()
        target_scale = fit_anchored_target_scale(
            observations.weekly_truth[splits["train"]],
            bias_baseline[splits["train"]],
            weights,
            split_name="train",
            valid_mask=valid_mask[splits["train"]],
        )
        target = standardize_anchored_target(
            observations.weekly_truth,
            bias_baseline,
            target_scale,
            valid_mask=valid_mask,
        )
        mean_std = np.asarray(
            normalization["log_fuxi_mean"]["std_by_lead"], dtype=np.float32
        )
        anomaly_std = np.asarray(
            normalization["explicit_log_fuxi_anomaly"]["std_by_lead"],
            dtype=np.float32,
        )
        ratio = mean_std / anomaly_std
        if ratio.shape != (6,) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
            raise base.DataContractError("invalid mean-to-anomaly scale ratio")

        train_indices = np.asarray(splits["train"], dtype=np.int64)
        validation_indices = np.asarray(splits["validation"], dtype=np.int64)
        if smoke:
            train_indices = train_indices[:64]
            validation_indices = smoke_validation_indices(
                forecast.initializations,
                validation_indices,
                cases_per_year=8,
            )
        prepared = sweep.PreparedData(
            features=np.asarray(features, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            target_scale=np.asarray(target_scale, dtype=np.float32),
            raw_fuxi=np.asarray(forecast.ensemble_mean, dtype=np.float32),
            bias_baseline=np.asarray(bias_baseline, dtype=np.float32),
            truth=np.asarray(observations.weekly_truth, dtype=np.float32),
            climatology=np.asarray(observations.weekly_climatology, dtype=np.float32),
            valid_mask=valid_mask,
            weights=np.asarray(weights, dtype=np.float64),
            initializations=np.asarray(
                forecast.initializations, dtype="datetime64[D]"
            ),
            train_indices=train_indices,
            validation_indices=validation_indices,
            mean_to_anomaly_ratio=ratio,
        )
        metadata = {
            "loaded_initialization_years": list(loaded_years),
            "maximum_loaded_initialization_year": max(loaded_years),
            "initialization_date_min": np.datetime_as_string(
                initialization_dates.min(), unit="D"
            ),
            "initialization_date_max": np.datetime_as_string(
                initialization_dates.max(), unit="D"
            ),
            "verification_target_date_min": np.datetime_as_string(
                verification_dates.min(), unit="D"
            ),
            "verification_target_date_max": np.datetime_as_string(
                verification_dates.max(), unit="D"
            ),
            "observation_date_min": np.datetime_as_string(
                observation_dates.min(), unit="D"
            ),
            "observation_date_max": np.datetime_as_string(
                observation_dates.max(), unit="D"
            ),
            "sealed_initialization_years": list(SEALED_INITIALIZATION_YEARS),
            "sealed_initialization_years_opened": False,
            "full_split_counts": counts,
            "effective_train_cases": int(len(train_indices)),
            "effective_validation_cases": int(len(validation_indices)),
            "support_cells": int(support.sum()),
            "feature_shape": list(features.shape),
            "feature_names": list(normalization["input_channels"]),
            "climatology_centre_max_abs_difference": float(centre_difference),
            "forecast_sources": list(forecast.source_files),
            "observation_sources": list(source_stores),
            "logical_array_sha256": {
                "features": sha256_array(features),
                "target": sha256_array(target),
                "raw_fuxi": sha256_array(forecast.ensemble_mean),
                "bias_baseline": sha256_array(bias_baseline),
                "truth": sha256_array(observations.weekly_truth),
                "climatology": sha256_array(observations.weekly_climatology),
                "area_weights": sha256_array(weights),
                "initializations": sha256_array(initialization_dates),
            },
        }
        anchor = {
            "lead_month_residual": correction.lead_month_residual,
            "shrinkage": np.asarray(correction.shrinkage, dtype=np.float32),
            "target_scale": target_scale,
        }
        return PreparedBundle(prepared, normalization, anchor, metadata)


def build_model() -> FixedClimatologyAllLeadUNet:
    model = FixedClimatologyAllLeadUNet(
        input_channels=INPUT_CHANNELS,
        backbone_channels=BACKBONE_CHANNELS,
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNT:
        raise RuntimeError(f"model parameter count changed: {count}")
    return model


def load_checkpoint_source(
    path: Path, expected_seed: int, *, expected_mode: str | None = None
) -> CheckpointSource:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"global checkpoint is not a mapping: {path}")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported global checkpoint schema: {path}")
    if payload.get("stage") != "global_patch_pretraining":
        raise ValueError(f"checkpoint is not global patch pretraining: {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"checkpoint seed {payload.get('seed')} does not match {expected_seed}: {path}"
        )
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(f"checkpoint lacks architecture contract: {path}")
    expected_architecture = {
        "class": "FixedClimatologyAllLeadUNet",
        "input_channels": BACKBONE_CHANNELS,
        "backbone_channels": BACKBONE_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "parameter_count": PARAMETER_COUNT,
    }
    for name, expected in expected_architecture.items():
        if architecture.get(name) != expected:
            raise ValueError(
                f"global architecture {name}={architecture.get(name)!r}; "
                f"expected {expected!r}"
            )
    if not np.isclose(float(architecture.get("dropout", np.nan)), DROPOUT):
        raise ValueError("global and India dropout contracts differ")

    feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
    if feature_names != GLOBAL_FEATURE_NAMES:
        raise ValueError(
            "global feature contract differs; "
            f"received={feature_names}, expected={GLOBAL_FEATURE_NAMES}"
        )
    transfer = payload.get("transfer")
    if not isinstance(transfer, Mapping):
        raise ValueError("global checkpoint lacks transfer metadata")
    reset_keys = tuple(str(value) for value in transfer.get("reset_keys", ()))
    if reset_keys != HEAD_KEYS:
        raise ValueError(f"global reset-key contract differs: {reset_keys}")
    zero_channels = tuple(
        str(value).lower() for value in transfer.get("zero_pretraining_channels", ())
    )
    if not any("t2m" in value for value in zero_channels):
        raise ValueError("global checkpoint does not disclose zero-placeholder T2M")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("global checkpoint lacks model_state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError("global model_state_dict must map names to tensors")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("global checkpoint lacks its immutable data contract")
    contract_sha256 = str(payload.get("contract_sha256", ""))
    calculated_contract_sha256 = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if contract_sha256 != calculated_contract_sha256:
        raise ValueError("global checkpoint contract SHA-256 does not verify")
    if int(contract.get("seed", -1)) != expected_seed:
        raise ValueError("global checkpoint payload and contract seeds differ")
    if expected_mode is not None and contract.get("mode") != expected_mode:
        raise ValueError(
            f"global checkpoint mode {contract.get('mode')!r}; "
            f"expected {expected_mode!r}"
        )
    if expected_mode is not None:
        expected_recipe = CANONICAL_GLOBAL_RECIPES[expected_mode]
        mismatches = {
            key: {"received": contract.get(key), "expected": expected}
            for key, expected in expected_recipe.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "global checkpoint does not use the canonical "
                f"{expected_mode} recipe: {mismatches}"
            )
    source_years = tuple(contract.get("fit_years", ())) + tuple(
        contract.get("validation_years", ())
    )
    if not source_years or max(int(year) for year in source_years) > 2017:
        raise ValueError("global checkpoint is not hard-blocked from 2018+ data")
    try:
        source_best_epoch = int(payload["best_epoch"])
        source_best_loss = float(payload["best_validation_loss"])
        source_epoch_zero_loss = float(payload["epoch_zero_validation_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("global checkpoint lacks valid epoch-zero selection metadata") from exc
    if source_best_epoch < 0 or not np.isfinite(source_best_loss):
        raise ValueError("global checkpoint selection metadata is invalid")
    if (
        not np.isfinite(source_epoch_zero_loss)
        or source_best_loss > source_epoch_zero_loss + 1.0e-12
    ):
        raise ValueError("global checkpoint is worse than its epoch-zero fallback")
    if torch.count_nonzero(state[FIRST_CONV_KEY][:, T2M_CHANNEL_INDEX]).item() != 0:
        raise ValueError("global checkpoint T2M placeholder kernel is not zero")
    checkpoint_sha256 = sha256_file(path)
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    source_fingerprint: str | None = None
    if expected_mode is not None:
        candidate_manifest = path.parent.parent / "manifest.json"
        if not candidate_manifest.is_file():
            raise ValueError(
                f"canonical global checkpoint lacks its run manifest: {path}"
            )
        manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
        expected_smoke = expected_mode == "smoke"
        if manifest.get("status") != "complete":
            raise ValueError("global source manifest is not complete")
        if manifest.get("smoke") is not expected_smoke:
            raise ValueError("global source manifest mode differs from checkpoint")
        if manifest.get("contract_sha256") != contract_sha256:
            raise ValueError("global source manifest contract hash differs")
        artifact_hash = manifest.get("artifacts", {}).get("checkpoints/best.pt")
        if artifact_hash != checkpoint_sha256:
            raise ValueError("global checkpoint SHA-256 differs from its manifest")
        if manifest.get("test_predictions_created") is not False:
            raise ValueError(
                "global source manifest does not prove test predictions stayed closed"
            )
        if manifest.get("scientific_eligible") is not False:
            raise ValueError("global source is mislabeled as standalone scientific evidence")
        if (
            manifest.get("selection", {}).get(
                "selected_not_worse_than_epoch_zero"
            )
            is not True
        ):
            raise ValueError("global source did not preserve the epoch-zero fallback")
        if int(manifest["selection"].get("best_epoch", -1)) != source_best_epoch:
            raise ValueError("global checkpoint and manifest selected epochs differ")
        expected_opened_years = sorted(
            int(year)
            for year in tuple(contract["fit_years"])
            + tuple(contract["validation_years"])
        )
        provenance = manifest.get("source_provenance", {})
        if provenance.get("opened_years") != expected_opened_years:
            raise ValueError("global source manifest opened-year provenance differs")
        if provenance.get("hard_blocked_years") != "2018 and later":
            raise ValueError("global source manifest lacks the 2018+ hard boundary")
        checkpoint_code = payload.get("source_snapshots")
        if not isinstance(checkpoint_code, Mapping) or not checkpoint_code:
            raise ValueError("global checkpoint lacks source-code fingerprints")
        if manifest.get("code") != checkpoint_code:
            raise ValueError("global checkpoint and manifest source-code hashes differ")
        selected_content = provenance.get("selected_case_content_sha256")
        if (
            not isinstance(selected_content, Mapping)
            or not selected_content.get("fit")
            or not selected_content.get("validation")
            or payload.get("selected_case_content_sha256") != selected_content
        ):
            raise ValueError("global checkpoint lacks bound selected-data fingerprints")
        split_provenance = manifest.get("splits", {})
        source_identity = {
            "opened_years": provenance.get("opened_years"),
            "annual_metadata": {
                year: details.get("metadata_sha256")
                for year, details in provenance.get("annual", {}).items()
            },
            "fit_schedule_sha256": split_provenance.get("fit_schedule_sha256"),
            "validation_schedule_sha256": split_provenance.get(
                "validation_schedule_sha256"
            ),
            "fit_date_bounds": split_provenance.get("fit_date_bounds"),
            "validation_date_bounds": split_provenance.get(
                "validation_date_bounds"
            ),
            "selected_case_content_sha256": selected_content,
            "source_code": checkpoint_code,
        }
        if (
            not source_identity["annual_metadata"]
            or not source_identity["fit_schedule_sha256"]
            or not source_identity["validation_schedule_sha256"]
        ):
            raise ValueError("global source manifest lacks source/schedule fingerprints")
        source_fingerprint = hashlib.sha256(
            json.dumps(
                source_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = candidate_manifest.resolve()
        manifest_sha256 = sha256_file(candidate_manifest)
    return CheckpointSource(
        path=path,
        seed=expected_seed,
        sha256=checkpoint_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        source_fingerprint=source_fingerprint,
        contract_sha256=contract_sha256,
        feature_names=feature_names,
        payload=payload,
    )


def transfer_global_initialization(
    model: FixedClimatologyAllLeadUNet,
    source: CheckpointSource,
    scratch_initial_state: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    """Strictly load global weights and restore India-safe no-op parameters."""

    state = source.payload["model_state_dict"]
    expected = model.state_dict()
    if tuple(state) != tuple(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"global state-key order differs; missing={missing}, unexpected={unexpected}"
        )
    for key in expected:
        if tuple(state[key].shape) != tuple(expected[key].shape):
            raise ValueError(
                f"global tensor shape differs for {key}: "
                f"{tuple(state[key].shape)} versus {tuple(expected[key].shape)}"
            )
    model.load_state_dict(state, strict=True)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key in HEAD_KEYS:
            named_parameters[key].zero_()
        named_parameters[FIRST_CONV_KEY][:, T2M_CHANNEL_INDEX].copy_(
            scratch_initial_state[FIRST_CONV_KEY][:, T2M_CHANNEL_INDEX]
        )

    transferred = model.state_dict()
    for key in transferred:
        if key in HEAD_KEYS:
            if torch.count_nonzero(transferred[key]).item() != 0:
                raise RuntimeError(f"residual head reset failed for {key}")
        elif key == FIRST_CONV_KEY:
            keep = [index for index in range(BACKBONE_CHANNELS) if index != T2M_CHANNEL_INDEX]
            if not torch.equal(transferred[key][:, keep], state[key][:, keep]):
                raise RuntimeError("non-T2M first-convolution weights changed in transfer")
            if not torch.equal(
                transferred[key][:, T2M_CHANNEL_INDEX],
                scratch_initial_state[key][:, T2M_CHANNEL_INDEX],
            ):
                raise RuntimeError("T2M first-convolution slice was not restored")
        elif not torch.equal(transferred[key], state[key]):
            raise RuntimeError(f"transferred tensor changed unexpectedly: {key}")
    return {
        "strict_state_key_count": len(transferred),
        "reset_keys": list(HEAD_KEYS),
        "restored_scratch_slice": {
            "key": FIRST_CONV_KEY,
            "input_channel_index": T2M_CHANNEL_INDEX,
            "feature": "fuxi_t2m_weekly",
        },
        "retained_pretrained_parameters": PARAMETER_COUNT - 17 - (
            BASE_CHANNELS * 3 * 3
        ),
        "reset_head_parameters": 17,
        "restored_scratch_parameters": BASE_CHANNELS * 3 * 3,
    }


def assert_initial_noop(
    model: FixedClimatologyAllLeadUNet,
    prepared: sweep.PreparedData,
) -> Mapping[str, Any]:
    indices = prepared.validation_indices[:1]
    model.eval()
    with torch.inference_mode():
        residual = model(torch.from_numpy(prepared.features[indices])).cpu().numpy()
    if not np.array_equal(residual, np.zeros_like(residual)):
        raise RuntimeError("reset model is not an exact zero-residual no-op")
    reconstructed = reconstruct_anchored_precipitation(
        prepared.bias_baseline[indices],
        residual,
        prepared.target_scale,
        valid_mask=prepared.valid_mask[indices],
    )
    valid = prepared.valid_mask[indices]
    if not np.array_equal(
        reconstructed[valid], prepared.bias_baseline[indices][valid]
    ):
        raise RuntimeError("zero residual does not exactly reproduce the anchor")
    return {
        "maximum_absolute_initial_residual": float(np.max(np.abs(residual))),
        "anchor_identity_exact_on_valid_support": True,
        "checked_cases": int(len(indices)),
    }


def make_datasets(prepared: sweep.PreparedData):
    observations = base.ObservationData(
        weekly_truth=prepared.truth,
        weekly_climatology=prepared.climatology,
        observation_fraction=(prepared.weights > 0.0).astype(np.float32),
        source_stores=(),
    )
    support = prepared.weights > 0.0
    train = common.make_dataset(
        prepared.train_indices,
        prepared.features,
        prepared.target,
        prepared.bias_baseline,
        observations,
        support,
    )
    validation = common.make_dataset(
        prepared.validation_indices,
        prepared.features,
        prepared.target,
        prepared.bias_baseline,
        observations,
        support,
    )
    return train, validation


def fit_india_model(
    name: str,
    seed: int,
    model: FixedClimatologyAllLeadUNet,
    prepared: sweep.PreparedData,
    train_dataset,
    validation_dataset,
    output: Path,
    *,
    device: str,
    smoke: bool,
    initialization: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    run_directory = output / "models" / name / f"seed_{seed}"
    (run_directory / "logs").mkdir(parents=True, exist_ok=False)
    (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
    expected_shape = (len(prepared.validation_indices), 6, 27, 27)
    epoch_zero_residual = np.zeros(expected_shape, dtype=np.float32)
    epoch_zero_score, _ = sweep.residual_metrics(
        epoch_zero_residual,
        prepared,
        predictor=f"{name}_seed_{seed}_epoch_zero",
    )
    epoch_zero_loss = float(epoch_zero_score["composite_loss"])
    epoch_zero_state = copy.deepcopy(model.state_dict())
    result = train_anchored_model(
        model,
        train_dataset,
        validation_dataset,
        prepared.weights,
        prepared.target_scale,
        LEAD_WEIGHTS,
        LOSS_COEFFICIENTS,
        run_directory,
        seed=seed,
        device=device,
        batch_size=BATCH_SIZE,
        max_epochs=2 if smoke else MAX_EPOCHS,
        patience=1 if smoke else PATIENCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        smooth_l1_beta=1.0,
        num_workers=0,
        use_amp=True,
    )
    trained_residual = predict(
        model,
        prepared.features[prepared.validation_indices],
        device=device,
        batch_size=BATCH_SIZE,
        use_amp=True,
    )
    if trained_residual.shape != expected_shape or not np.isfinite(
        trained_residual
    ).all():
        raise RuntimeError(
            f"invalid {name} validation residual: {trained_residual.shape}"
        )
    trained_score, _ = sweep.residual_metrics(
        trained_residual,
        prepared,
        predictor=f"{name}_seed_{seed}_trained",
    )
    trained_selection_loss = float(trained_score["composite_loss"])
    epoch_zero_retained = not (
        trained_selection_loss < epoch_zero_loss - 1.0e-7
    )
    checkpoint = run_directory / "checkpoints" / "best.pt"
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload.update(
        {
            "trainer_best_epoch": int(result.best_epoch),
            "trainer_best_validation_loss": float(result.best_validation_loss),
            "selection_metric": "full-validation composite_loss",
            "epoch_zero_validation_loss": epoch_zero_loss,
            "trained_selection_validation_loss": trained_selection_loss,
            "epoch_zero_retained": epoch_zero_retained,
        }
    )
    if epoch_zero_retained:
        model.load_state_dict(epoch_zero_state, strict=True)
        residual = epoch_zero_residual
        selected_epoch = -1
        selected_loss = epoch_zero_loss
        checkpoint_payload["model_state_dict"] = epoch_zero_state
    else:
        residual = trained_residual
        selected_epoch = int(result.best_epoch)
        selected_loss = trained_selection_loss
    checkpoint_payload["best_epoch"] = selected_epoch
    checkpoint_payload["best_validation_loss"] = selected_loss
    atomic_torch_save(checkpoint, checkpoint_payload)
    residual_path = run_directory / "validation_residual.npy"
    np.save(residual_path, residual)
    record = {
        "status": "complete",
        "configuration": name,
        "seed": seed,
        "device": device,
        "parameter_count": PARAMETER_COUNT,
        "best_epoch_zero_based": selected_epoch,
        "best_validation_loss": selected_loss,
        "trainer_best_epoch_zero_based": int(result.best_epoch),
        "trainer_best_validation_loss": float(result.best_validation_loss),
        "epoch_zero_validation_loss": epoch_zero_loss,
        "trained_selection_validation_loss": trained_selection_loss,
        "epoch_zero_retained": epoch_zero_retained,
        "elapsed_seconds": float(result.elapsed_seconds),
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_residual": str(residual_path.relative_to(output)),
        "initialization": _json_safe(initialization),
    }
    atomic_write_json(run_directory / "run_record.json", record)
    return residual, record


def _case_summary(case_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = case_metrics.copy()
    frame["year"] = pd.DatetimeIndex(frame.case_id).year
    rows = []
    for keys, group in frame.groupby(
        ["configuration", "member", "year", "lead"], dropna=False
    ):
        configuration, member, year, lead = keys
        rows.append(
            {
                "configuration": configuration,
                "member": member,
                "year": int(year),
                "lead": int(lead),
                "cases": int(group.case_id.nunique()),
                **{
                    metric: float(group[metric].mean())
                    for metric in ("rmse", "mae", "bias", "acc")
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["configuration", "member", "year", "lead"]
    )


def _pooled_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = case_metrics.copy()
    frame["year"] = pd.DatetimeIndex(frame.case_id).year
    rows = []
    for (configuration, member), group in frame.groupby(
        ["configuration", "member"], dropna=False
    ):
        row: dict[str, Any] = {
            "configuration": configuration,
            "member": member,
            "cases": int(group.case_id.nunique()),
        }
        for metric in ("rmse", "mae", "bias", "acc"):
            row[f"pooled_{metric}"] = float(group[metric].mean())
            for year in VALIDATION_YEARS:
                row[f"{year}_{metric}"] = float(
                    group.loc[group.year.eq(year), metric].mean()
                )
            for lead in range(1, 7):
                row[f"W{lead}_{metric}"] = float(
                    group.loc[group.lead.eq(lead), metric].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["configuration", "member"])


def aggregate_validation(
    prepared: sweep.PreparedData,
    residuals: Mapping[str, Mapping[int, np.ndarray]],
    records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any], Mapping[str, np.ndarray]]:
    case_frames = []
    score_rows = []
    ensemble_residuals: dict[str, np.ndarray] = {}
    for name in ("scratch", "global_pretrained"):
        members = []
        for seed, residual in sorted(residuals[name].items()):
            score, metrics = sweep.residual_metrics(
                residual, prepared, predictor=f"{name}_seed_{seed}"
            )
            metrics.insert(0, "member", f"seed_{seed}")
            metrics.insert(0, "configuration", name)
            case_frames.append(metrics)
            score_rows.append(
                {
                    "configuration": name,
                    "member": f"seed_{seed}",
                    **score,
                }
            )
            members.append(residual)
        ensemble = np.mean(members, axis=0, dtype=np.float64).astype(np.float32)
        ensemble_residuals[name] = ensemble
        score, metrics = sweep.residual_metrics(
            ensemble, prepared, predictor=name
        )
        metrics.insert(0, "member", "ensemble")
        metrics.insert(0, "configuration", name)
        case_frames.append(metrics)
        score_rows.append(
            {"configuration": name, "member": "ensemble", **score}
        )

    for baseline_name in ("raw_fuxi", "log_bias"):
        metrics = sweep.baseline_case_metrics(prepared, baseline_name)
        metrics["predictor"] = baseline_name
        metrics.insert(0, "member", "deterministic")
        metrics.insert(0, "configuration", baseline_name)
        case_frames.append(metrics)

    case_metrics = pd.concat(case_frames, ignore_index=True)
    composite_scores = pd.DataFrame(score_rows)
    case_metrics.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    composite_scores.to_csv(
        output / "metrics" / "validation_composite_scores.csv", index=False
    )
    _case_summary(case_metrics).to_csv(
        output / "metrics" / "validation_year_lead_metrics.csv", index=False
    )
    pooled = _pooled_metrics(case_metrics)
    pooled.to_csv(output / "metrics" / "validation_pooled_metrics.csv", index=False)

    histories = []
    for record in records:
        history_path = (
            output
            / "models"
            / str(record["configuration"])
            / f"seed_{int(record['seed'])}"
            / "logs"
            / "training_history.csv"
        )
        history = pd.read_csv(history_path)
        history.insert(0, "seed", int(record["seed"]))
        history.insert(0, "configuration", str(record["configuration"]))
        histories.append(history)
    pd.concat(histories, ignore_index=True).to_csv(
        output / "metrics" / "training_history_tidy.csv", index=False
    )
    pd.DataFrame([_json_safe(record) for record in records]).to_csv(
        output / "metrics" / "run_records.csv", index=False
    )

    paired_source = case_metrics.loc[
        ((case_metrics.member == "ensemble") & case_metrics.configuration.isin(
            ("scratch", "global_pretrained")
        ))
        | (
            (case_metrics.member == "deterministic")
            & case_metrics.configuration.isin(("raw_fuxi", "log_bias"))
        )
    ].copy()
    paired_frames = []
    comparisons = (
        ("global_pretrained", "scratch"),
        ("scratch", "log_bias"),
        ("global_pretrained", "log_bias"),
    )
    for candidate, reference in comparisons:
        interval = paired_moving_block_bootstrap(
            paired_source,
            candidate,
            reference,
            metric_columns=("acc", "rmse", "mae", "bias"),
            block_length=2 if smoke else 13,
            n_resamples=50 if smoke else 2000,
            seed=42,
        )
        interval.insert(0, "comparison", f"{candidate}_minus_{reference}")
        paired_frames.append(interval)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        output / "metrics" / "paired_skill_by_lead.csv", index=False
    )

    promotion = build_promotion_gate(
        pooled, composite_scores, records, scientific_eligible=not smoke
    )
    atomic_write_json(output / "metrics" / "promotion_gate.json", promotion)

    indices = prepared.validation_indices
    support = prepared.weights > 0.0
    physical_predictions: dict[str, np.ndarray] = {}
    for name, residual in ensemble_residuals.items():
        physical_predictions[name] = common.reconstruct(
            prepared.bias_baseline[indices], residual, prepared.target_scale, support
        )
    raw = prepared.raw_fuxi[indices].copy()
    raw[..., ~support] = np.nan
    log_bias = prepared.bias_baseline[indices].copy()
    log_bias[..., ~support] = np.nan
    np.savez_compressed(
        output / "metrics" / "validation_outputs.npz",
        initializations=prepared.initializations[indices],
        truth=prepared.truth[indices],
        climatology=prepared.climatology[indices],
        area_weights=prepared.weights,
        raw_fuxi=raw,
        log_bias=log_bias,
        scratch=physical_predictions["scratch"],
        global_pretrained=physical_predictions["global_pretrained"],
        scratch_standardized_residual=ensemble_residuals["scratch"],
        global_pretrained_standardized_residual=(
            ensemble_residuals["global_pretrained"]
        ),
    )
    return case_metrics, composite_scores, promotion, ensemble_residuals


def _row(
    table: pd.DataFrame, configuration: str, member: str = "ensemble"
) -> pd.Series:
    selected = table.loc[
        table.configuration.eq(configuration) & table.member.eq(member)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one pooled row for {configuration}/{member}, found {len(selected)}"
        )
    return selected.iloc[0]


def build_promotion_gate(
    pooled: pd.DataFrame,
    scores: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    *,
    scientific_eligible: bool,
) -> Mapping[str, Any]:
    scratch = _row(pooled, "scratch")
    pretrained = _row(pooled, "global_pretrained")
    anchor = _row(pooled, "log_bias", "deterministic")
    score_scratch = _row(scores, "scratch")
    score_pretrained = _row(scores, "global_pretrained")

    record_frame = pd.DataFrame(records)
    scratch_seed = record_frame.loc[
        record_frame.configuration.eq("scratch"), ["seed", "best_validation_loss"]
    ].set_index("seed")
    pretrained_seed = record_frame.loc[
        record_frame.configuration.eq("global_pretrained"),
        ["seed", "best_validation_loss"],
    ].set_index("seed")
    paired_seed = pretrained_seed.join(
        scratch_seed, lsuffix="_pretrained", rsuffix="_scratch", validate="one_to_one"
    )

    seed_count = len(paired_seed)
    required_seed_wins = min(MINIMUM_SEEDS_BEATING_SCRATCH, seed_count)
    seeds_beating = int(
        np.count_nonzero(
            paired_seed.best_validation_loss_pretrained
            < paired_seed.best_validation_loss_scratch
        )
    )
    mean_pretrained_loss = float(
        paired_seed.best_validation_loss_pretrained.mean()
    )
    mean_scratch_loss = float(paired_seed.best_validation_loss_scratch.mean())
    improving_leads = sum(
        float(score_pretrained[f"W{lead}_composite_loss"])
        < float(score_scratch[f"W{lead}_composite_loss"])
        for lead in range(1, 7)
    )
    maximum_lead_rmse_regression = max(
        100.0
        * (float(pretrained[f"W{lead}_rmse"]) - float(scratch[f"W{lead}_rmse"]))
        / float(scratch[f"W{lead}_rmse"])
        for lead in range(1, 7)
    )
    minimum_lead_acc_delta = min(
        float(pretrained[f"W{lead}_acc"]) - float(scratch[f"W{lead}_acc"])
        for lead in range(1, 7)
    )
    source_best_epochs = [
        int(record["initialization"]["source_best_epoch"])
        for record in records
        if record["configuration"] == "global_pretrained"
    ]
    conditions = {
        "every_global_source_learned_beyond_epoch_zero": (
            len(source_best_epochs) == seed_count
            and all(epoch > 0 for epoch in source_best_epochs)
        ),
        "mean_seed_validation_loss_improves": (
            mean_pretrained_loss
            <= mean_scratch_loss - MINIMUM_LOSS_IMPROVEMENT
        ),
        "ensemble_validation_loss_improves": (
            float(score_pretrained.composite_loss)
            <= float(score_scratch.composite_loss) - MINIMUM_LOSS_IMPROVEMENT
        ),
        "enough_seeds_improve": seeds_beating >= required_seed_wins,
        "rmse_improves_in_2018": (
            float(pretrained["2018_rmse"]) < float(scratch["2018_rmse"])
        ),
        "rmse_improves_in_2019": (
            float(pretrained["2019_rmse"]) < float(scratch["2019_rmse"])
        ),
        "pooled_mae_not_worse": (
            float(pretrained.pooled_mae) <= float(scratch.pooled_mae)
        ),
        "pooled_acc_within_guard": (
            float(pretrained.pooled_acc) - float(scratch.pooled_acc)
            >= -MAXIMUM_POOLED_ACC_REGRESSION
        ),
        "absolute_bias_within_guard": (
            abs(float(pretrained.pooled_bias)) - abs(float(scratch.pooled_bias))
            <= MAXIMUM_ABSOLUTE_BIAS_REGRESSION
        ),
        "at_least_four_leads_improve_composite": (
            improving_leads >= MINIMUM_IMPROVING_LEADS
        ),
        "maximum_lead_rmse_regression_within_guard": (
            maximum_lead_rmse_regression
            <= MAXIMUM_LEAD_RMSE_REGRESSION_PCT
        ),
        "minimum_lead_acc_delta_within_guard": (
            minimum_lead_acc_delta >= MINIMUM_LEAD_ACC_DELTA
        ),
    }
    pretrained_anchor_conditions = {
        "pooled_rmse_improves_anchor": (
            float(pretrained.pooled_rmse) < float(anchor.pooled_rmse)
        ),
        "rmse_improves_anchor_in_2018": (
            float(pretrained["2018_rmse"]) < float(anchor["2018_rmse"])
        ),
        "rmse_improves_anchor_in_2019": (
            float(pretrained["2019_rmse"]) < float(anchor["2019_rmse"])
        ),
        "pooled_mae_not_worse_than_anchor": (
            float(pretrained.pooled_mae) <= float(anchor.pooled_mae)
        ),
        "pooled_acc_within_anchor_guard": (
            float(pretrained.pooled_acc) - float(anchor.pooled_acc)
            >= -MAXIMUM_POOLED_ACC_REGRESSION
        ),
        "absolute_bias_within_anchor_guard": (
            abs(float(pretrained.pooled_bias)) - abs(float(anchor.pooled_bias))
            <= MAXIMUM_ABSOLUTE_BIAS_REGRESSION
        ),
    }
    qualifies = bool(
        scientific_eligible
        and all(conditions.values())
        and all(pretrained_anchor_conditions.values())
    )

    scratch_anchor_conditions = {
        "pooled_rmse_improves_anchor": (
            float(scratch.pooled_rmse) < float(anchor.pooled_rmse)
        ),
        "rmse_improves_anchor_in_2018": (
            float(scratch["2018_rmse"]) < float(anchor["2018_rmse"])
        ),
        "rmse_improves_anchor_in_2019": (
            float(scratch["2019_rmse"]) < float(anchor["2019_rmse"])
        ),
        "pooled_mae_not_worse_than_anchor": (
            float(scratch.pooled_mae) <= float(anchor.pooled_mae)
        ),
        "pooled_acc_within_anchor_guard": (
            float(scratch.pooled_acc) - float(anchor.pooled_acc)
            >= -MAXIMUM_POOLED_ACC_REGRESSION
        ),
        "absolute_bias_within_anchor_guard": (
            abs(float(scratch.pooled_bias)) - abs(float(anchor.pooled_bias))
            <= MAXIMUM_ABSOLUTE_BIAS_REGRESSION
        ),
    }
    scratch_passes_anchor = bool(all(scratch_anchor_conditions.values()))
    if not scientific_eligible:
        selected_system = "none_smoke_is_non_scientific"
        reason = "Smoke runs verify execution only and cannot promote a model."
    elif qualifies:
        selected_system = "global_pretrained"
        reason = "Global pretraining passed every predeclared matched guard."
    elif scratch_passes_anchor:
        selected_system = "scratch"
        reason = "Global pretraining was not promoted; scratch passed the anchor guards."
    else:
        selected_system = "log_bias"
        reason = "Neither neural candidate passed its predeclared guards; retain no-op anchor."
    return {
        "status": "validation_only",
        "scientific_eligible": scientific_eligible,
        "global_pretraining_qualifies": qualifies,
        "selected_system": selected_system,
        "selection_reason": reason,
        "conditions": _json_safe(conditions),
        "global_pretrained_anchor_conditions": _json_safe(
            pretrained_anchor_conditions
        ),
        "scratch_anchor_conditions": _json_safe(scratch_anchor_conditions),
        "scratch_passes_anchor": scratch_passes_anchor,
        "diagnostics": {
            "paired_seed_count": seed_count,
            "seeds_beating_scratch": seeds_beating,
            "required_seed_wins": required_seed_wins,
            "mean_pretrained_validation_loss": mean_pretrained_loss,
            "mean_scratch_validation_loss": mean_scratch_loss,
            "ensemble_loss_delta_pretrained_minus_scratch": float(
                score_pretrained.composite_loss - score_scratch.composite_loss
            ),
            "pooled_rmse_skill_vs_scratch_pct": float(
                100.0
                * (scratch.pooled_rmse - pretrained.pooled_rmse)
                / scratch.pooled_rmse
            ),
            "pooled_acc_delta_vs_scratch": float(
                pretrained.pooled_acc - scratch.pooled_acc
            ),
            "pooled_abs_bias_delta_vs_scratch": float(
                abs(pretrained.pooled_bias) - abs(scratch.pooled_bias)
            ),
            "leads_improving_composite": improving_leads,
            "maximum_lead_rmse_regression_pct": float(
                maximum_lead_rmse_regression
            ),
            "minimum_lead_acc_delta": float(minimum_lead_acc_delta),
            "global_source_best_epochs": source_best_epochs,
        },
        "thresholds": {
            "minimum_loss_improvement": MINIMUM_LOSS_IMPROVEMENT,
            "minimum_seeds_beating_scratch": MINIMUM_SEEDS_BEATING_SCRATCH,
            "minimum_improving_leads": MINIMUM_IMPROVING_LEADS,
            "maximum_lead_rmse_regression_pct": (
                MAXIMUM_LEAD_RMSE_REGRESSION_PCT
            ),
            "minimum_lead_acc_delta": MINIMUM_LEAD_ACC_DELTA,
            "maximum_pooled_acc_regression": MAXIMUM_POOLED_ACC_REGRESSION,
            "maximum_absolute_bias_regression": (
                MAXIMUM_ABSOLUTE_BIAS_REGRESSION
            ),
        },
    }


def resolve_checkpoint_sources(
    paths: Sequence[Path], seeds: Sequence[int], *, expected_mode: str | None = None
) -> Mapping[int, CheckpointSource]:
    if len(paths) != len(seeds):
        raise ValueError(
            f"expected {len(seeds)} global checkpoints, received {len(paths)}"
        )
    by_seed: dict[int, CheckpointSource] = {}
    for path in paths:
        payload = torch.load(
            Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping) or "seed" not in payload:
            raise ValueError(f"checkpoint does not declare its seed: {path}")
        seed = int(payload["seed"])
        if seed not in seeds:
            raise ValueError(f"unexpected checkpoint seed {seed}: {path}")
        if seed in by_seed:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        by_seed[seed] = load_checkpoint_source(
            Path(path), seed, expected_mode=expected_mode
        )
    if tuple(sorted(by_seed)) != tuple(sorted(seeds)):
        raise ValueError("global checkpoints do not cover the required seeds")
    reference_source = by_seed[min(by_seed)]
    reference_contract = dict(reference_source.payload["contract"])
    reference_contract.pop("seed", None)
    reference_normalization = reference_source.payload.get("normalization")
    reference_target_scale = np.asarray(
        reference_source.payload.get("target_scale"), dtype=np.float32
    )
    if reference_normalization is None or reference_target_scale.shape != (6,):
        raise ValueError("global checkpoint lacks transferable preprocessing provenance")
    for seed, source in sorted(by_seed.items()):
        comparable_contract = dict(source.payload["contract"])
        comparable_contract.pop("seed", None)
        if comparable_contract != reference_contract:
            raise ValueError(
                "global checkpoint contracts differ across seeds after removing "
                f"the model seed; first mismatch is seed {seed}"
            )
        if source.payload.get("normalization") != reference_normalization:
            raise ValueError(
                f"global preprocessing normalization differs for seed {seed}"
            )
        target_scale = np.asarray(
            source.payload.get("target_scale"), dtype=np.float32
        )
        if not np.array_equal(target_scale, reference_target_scale):
            raise ValueError(f"global target scale differs for seed {seed}")
        if source.source_fingerprint != reference_source.source_fingerprint:
            raise ValueError(
                f"global cache or patch-schedule fingerprint differs for seed {seed}"
            )
    return by_seed


def run(args: argparse.Namespace) -> Path:
    smoke = bool(args.smoke)
    seeds = FULL_SEEDS[:1] if smoke else FULL_SEEDS
    sources = resolve_checkpoint_sources(
        args.pretrained_checkpoint, seeds, expected_mode="smoke" if smoke else "full"
    )
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    mode = "smoke" if smoke else "full"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{mode}_{timestamp}"
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / run_name
    staging = output_root / f".{run_name}.staging-{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(final if final.exists() else staging)
    for directory in (
        staging,
        staging / "models",
        staging / "metrics",
        staging / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    try:
        bundle = prepare_validation_only_data(smoke=smoke)
        prepared = bundle.prepared
        atomic_write_json(
            staging / "normalization.json", _json_safe(bundle.normalization)
        )
        np.savez_compressed(staging / "models" / "log_bias_anchor.npz", **bundle.anchor)
        train_dataset, validation_dataset = make_datasets(prepared)

        all_residuals: dict[str, dict[int, np.ndarray]] = {
            "scratch": {},
            "global_pretrained": {},
        }
        records: list[Mapping[str, Any]] = []
        initialization_audits: dict[str, Any] = {}
        for seed in seeds:
            set_deterministic_seed(seed)
            scratch_model = build_model()
            scratch_initial_state = copy.deepcopy(scratch_model.state_dict())
            scratch_noop = assert_initial_noop(scratch_model, prepared)
            scratch_residual, scratch_record = fit_india_model(
                "scratch",
                seed,
                scratch_model,
                prepared,
                train_dataset,
                validation_dataset,
                staging,
                device=device,
                smoke=smoke,
                initialization={
                    "kind": "matched_random_scratch",
                    "seed": seed,
                    "no_op": scratch_noop,
                },
            )
            all_residuals["scratch"][seed] = scratch_residual
            records.append(scratch_record)

            set_deterministic_seed(seed)
            pretrained_model = build_model()
            transfer = transfer_global_initialization(
                pretrained_model, sources[seed], scratch_initial_state
            )
            pretrained_noop = assert_initial_noop(pretrained_model, prepared)
            pretrained_residual, pretrained_record = fit_india_model(
                "global_pretrained",
                seed,
                pretrained_model,
                prepared,
                train_dataset,
                validation_dataset,
                staging,
                device=device,
                smoke=smoke,
                initialization={
                    "kind": "global_patch_pretrained",
                    "source_checkpoint": str(sources[seed].path),
                    "source_checkpoint_sha256": sources[seed].sha256,
                    "source_manifest": (
                        None
                        if sources[seed].manifest_path is None
                        else str(sources[seed].manifest_path)
                    ),
                    "source_manifest_sha256": sources[seed].manifest_sha256,
                    "source_fingerprint": sources[seed].source_fingerprint,
                    "source_contract_sha256": sources[seed].contract_sha256,
                    "source_feature_names": list(sources[seed].feature_names),
                    "source_best_epoch": int(
                        sources[seed].payload["best_epoch"]
                    ),
                    "transfer": transfer,
                    "no_op": pretrained_noop,
                },
            )
            all_residuals["global_pretrained"][seed] = pretrained_residual
            records.append(pretrained_record)
            initialization_audits[str(seed)] = {
                "scratch": scratch_noop,
                "global_pretrained": pretrained_noop,
                "transfer": transfer,
            }

        atomic_write_json(
            staging / "metrics" / "initialization_audit.json",
            initialization_audits,
        )
        _, _, promotion, _ = aggregate_validation(
            prepared,
            all_residuals,
            records,
            staging,
            smoke=smoke,
        )

        implementation_sources = {
            "fuxi_imd_global_pretraining_comparison": Path(__file__).resolve(),
            "fuxi_imd_compact_validation_sweep": Path(sweep.__file__).resolve(),
            "fuxi_imd_attention_climatology": Path(experiment.__file__).resolve(),
            "india_engine": Path(engine.__file__).resolve(),
            "india_common": Path(common.__file__).resolve(),
            "india_base": Path(base.__file__).resolve(),
            "fuxi_adapter_anchored": Path(adapter_anchored.__file__).resolve(),
            "fuxi_adapter_baselines": Path(adapter_baselines.__file__).resolve(),
            "fuxi_adapter_metrics": Path(adapter_metrics.__file__).resolve(),
            "fuxi_adapter_models": Path(adapter_models.__file__).resolve(),
            "fuxi_adapter_training": Path(adapter_training.__file__).resolve(),
            "fuxi_adapter_v3_training": Path(
                adapter_v3_training.__file__
            ).resolve(),
        }
        code_sources: dict[str, Mapping[str, str]] = {}
        for label, source in implementation_sources.items():
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
            "status": "complete_smoke" if smoke else "complete_validation_only",
            "created_utc": utc_now(),
            "mode": mode,
            "smoke": smoke,
            "scientific_eligible": not smoke,
            "test_predictions_created": False,
            "experiment": "matched_scratch_vs_global_pretraining",
            "sole_experimental_factor": "model initialization",
            "data": _json_safe(bundle.metadata),
            "split": {
                "train_initialization_years": list(TRAIN_YEARS),
                "validation_initialization_years": list(VALIDATION_YEARS),
                "sealed_initialization_years": list(SEALED_INITIALIZATION_YEARS),
                "later_year_predictions_created": False,
                "later_year_metrics_computed": False,
            },
            "model": {
                "class": "FixedClimatologyAllLeadUNet",
                "input_channels": INPUT_CHANNELS,
                "backbone_channels": BACKBONE_CHANNELS,
                "base_channels": BASE_CHANNELS,
                "dropout": DROPOUT,
                "parameter_count": PARAMETER_COUNT,
            },
            "training": {
                "seeds": list(seeds),
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "maximum_epochs": 2 if smoke else MAX_EPOCHS,
                "patience": 1 if smoke else PATIENCE,
                "lead_weights": list(LEAD_WEIGHTS),
                "loss_coefficients": LOSS_COEFFICIENTS,
                "same_datasets_and_order_for_both_initializations": True,
                "fixed_residual_scale_alpha": 1.0,
                "epoch_zero_checkpoint_fallback": True,
            },
            "global_sources": {
                str(seed): {
                    "path": str(sources[seed].path),
                    "sha256": sources[seed].sha256,
                    "manifest": (
                        None
                        if sources[seed].manifest_path is None
                        else str(sources[seed].manifest_path)
                    ),
                    "manifest_sha256": sources[seed].manifest_sha256,
                    "source_fingerprint": sources[seed].source_fingerprint,
                    "contract_sha256": sources[seed].contract_sha256,
                }
                for seed in seeds
            },
            "anchor_fallback": {
                "type": "training-only lead/month/grid log-bias correction",
                "exact_epoch_zero_identity_verified": True,
                "retained_when_neural_guards_fail": True,
            },
            "promotion": promotion,
            "elapsed_seconds": float(time.monotonic() - started),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "execution_environment": {
                "node": platform.node(),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "software": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": device,
            },
            "code_sources": code_sources,
            "artifacts": artifacts,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, final)
        print(final, flush=True)
        return final
    except Exception:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "failed_utc": utc_now(),
            "traceback": traceback.format_exc(),
            "sealed_initialization_years": list(SEALED_INITIALIZATION_YEARS),
        }
        if staging.exists():
            atomic_write_json(staging / "failure.json", failure)
            atomic_write_json(staging / "manifest.json", failure)
            failed = output_root / f"{run_name}.failed"
            if failed.exists():
                failed = output_root / f"{run_name}.failed-{os.getpid()}"
            os.replace(staging, failed)
            print(f"failed run record: {failed}", flush=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="one-seed execution check")
    mode.add_argument(
        "--full", action="store_true", help="three-seed validation-only comparison"
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        action="append",
        required=True,
        type=Path,
        help=(
            "global patch-pretraining best.pt; pass once for smoke and once per "
            "seed (42, 43, 44) for full"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="atomic run-directory parent",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="training device such as auto, cpu, cuda, or cuda:0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
