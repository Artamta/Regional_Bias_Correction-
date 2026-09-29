#!/usr/bin/env python
"""Add a temporal-attention refinement to the frozen FuXi–IMERG spatial U-Net.

The inherited U-Net is frozen. A zero-gated attention block learns across the
six weekly leads, using 2014–2018 for fitting and 2019 for checkpoint selection.
The already-used 2020–2021 test years are evaluated as an exploratory follow-up.

Run with::

    /home/raj.ayush/.conda/envs/fuxi/bin/python src/fuxi_imerg_spatiotemporal.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


from project_paths import EVALUATE_ROOT
from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT
if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imerg_experiment as base  # noqa: E402
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.models import ResidualUNet, TemporalAttentionUNet  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed, train_model  # noqa: E402
from spatiotemporal_model import (  # noqa: E402
    IdentityGatedTemporalAttentionUNet,
    fine_tune_from_spatial,
)


PARENT_RESULT = (
    HERE
    / "results/fuxi_imerg_jjas_5yr/full_20260809T233638Z"
)
RESULTS_ROOT = HERE / "results/fuxi_imerg_spatiotemporal"
SEEDS = (42, 43, 44)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 13

METHOD_ORDER = (
    "raw_fuxi",
    "log_bias",
    "spatial_unet",
    "spatiotemporal_unet",
    "lead_adaptive_hybrid",
)
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    "spatial_unet": "Spatial U-Net",
    "spatiotemporal_unet": "Spatiotemporal U-Net",
    "lead_adaptive_hybrid": "Lead-adaptive hybrid",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#0072B2",
    "spatial_unet": "#009E73",
    "spatiotemporal_unet": "#CC79A7",
    "lead_adaptive_hybrid": "#D55E00",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    "spatial_unet": "^",
    "spatiotemporal_unet": "D",
    "lead_adaptive_hybrid": "P",
}
PLOT_METHODS = (
    "raw_fuxi",
    "spatial_unet",
    "spatiotemporal_unet",
    "lead_adaptive_hybrid",
)
LEAD_SCOPES = {
    "W1-W6": tuple(range(1, 7)),
    "W1-W2": (1, 2),
    "W3-W4": (3, 4),
    "W5-W6": (5, 6),
    **{f"W{lead}": (lead,) for lead in range(1, 7)},
}
LEAD_BINS = {
    "W1-W2": (0, 1),
    "W3-W4": (2, 3),
    "W5-W6": (4, 5),
}
PRIMARY_COMPARISONS = (
    ("spatial_unet", "raw_fuxi"),
    ("spatiotemporal_unet", "raw_fuxi"),
    ("spatiotemporal_unet", "spatial_unet"),
    ("lead_adaptive_hybrid", "spatial_unet"),
)
ALL_COMPARISONS = (
    ("log_bias", "raw_fuxi"),
    *PRIMARY_COMPARISONS,
    ("lead_adaptive_hybrid", "raw_fuxi"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    root = Path(path)
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_frozen_parent(
    forecast: base.ForecastData,
    observations: base.ObservationData,
    weights: np.ndarray,
    test_indices: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Load and verify the immutable spatial experiment's test fields."""

    parent_manifest = json.loads((PARENT_RESULT / "manifest.json").read_text())
    if parent_manifest.get("status") != "complete" or parent_manifest.get("smoke"):
        raise base.DataContractError("the parent result is not a completed full run")
    if parent_manifest["training"]["seeds"] != list(SEEDS):
        raise base.DataContractError("the parent seed ensemble differs")
    if parent_manifest["training"]["test_indices"] != test_indices.tolist():
        raise base.DataContractError("the parent test indices differ")

    with xr.open_zarr(PARENT_RESULT / "predictions.zarr", consolidated=True) as dataset:
        if not np.array_equal(
            dataset.init.values.astype("datetime64[D]"),
            forecast.initializations[test_indices],
        ):
            raise base.DataContractError("parent initialization coordinates differ")
        if not np.array_equal(dataset.latitude.values, forecast.latitude):
            raise base.DataContractError("parent latitude differs")
        if not np.array_equal(dataset.longitude.values, forecast.longitude):
            raise base.DataContractError("parent longitude differs")
        stored_truth = np.asarray(dataset.truth_imerg.load(), dtype=np.float32)
        stored_climatology = np.asarray(
            dataset.imerg_climatology.load(), dtype=np.float32
        )
        stored_weights = np.asarray(dataset.area_weight_km2.load(), dtype=np.float64)
        predictions = {
            "raw_fuxi": np.asarray(
                dataset.prediction.sel({"method": "raw_fuxi"}).load(),
                dtype=np.float32,
            ),
            "log_bias": np.asarray(
                dataset.prediction.sel({"method": "log_bias"}).load(),
                dtype=np.float32,
            ),
            "spatial_unet": np.asarray(
                dataset.prediction.sel({"method": "residual_unet"}).load(),
                dtype=np.float32,
            ),
        }

    if not np.array_equal(
        stored_truth, observations.weekly_truth[test_indices], equal_nan=True
    ):
        raise base.DataContractError("parent IMERG truth differs")
    if not np.array_equal(
        stored_climatology,
        observations.weekly_climatology[test_indices],
        equal_nan=True,
    ):
        raise base.DataContractError("parent IMERG climatology differs")
    if not np.allclose(stored_weights, weights, rtol=1.0e-7, atol=0.0):
        raise base.DataContractError("parent spatial weights differ")

    raw = forecast.ensemble_mean[test_indices].copy()
    raw[..., weights <= 0.0] = np.nan
    if not np.array_equal(raw, predictions["raw_fuxi"], equal_nan=True):
        raise base.DataContractError("parent raw FuXi does not match the source shards")
    return predictions


def train_temporal_refinement(
    arrays: base.NeuralArrays,
    forecast: base.ForecastData,
    weights: np.ndarray,
    splits: Mapping[str, np.ndarray],
    frozen_spatial: np.ndarray,
    output: Path,
    *,
    smoke: bool,
) -> tuple[np.ndarray, Mapping[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_indices = splits["train"][:16] if smoke else splits["train"]
    validation_indices = splits["validation"][:8] if smoke else splits["validation"]
    test_indices = splits["test"][:4] if smoke else splits["test"]

    initial_predictions = []
    temporal_predictions = []
    records = []
    parameter_count = None
    trainable_count = None
    for seed in SEEDS:
        print(f"  seed {seed}: loading frozen spatial checkpoint", flush=True)
        set_deterministic_seed(seed)
        model = IdentityGatedTemporalAttentionUNet(
            in_channels=arrays.inputs.shape[2],
            base_channels=16,
            dropout=0.1,
            max_leads=6,
        )
        spatial_checkpoint_path = (
            PARENT_RESULT / f"models/seed_{seed}/checkpoints/best.pt"
        )
        spatial_checkpoint = torch.load(
            spatial_checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_spatial_checkpoint(spatial_checkpoint)
        model.freeze_spatial_backbone()
        current_parameters = sum(parameter.numel() for parameter in model.parameters())
        current_trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if parameter_count is None:
            parameter_count = current_parameters
            trainable_count = current_trainable
        elif (parameter_count, trainable_count) != (
            current_parameters,
            current_trainable,
        ):
            raise RuntimeError("model size changed between seeds")

        initial_standardized = predict(
            model,
            arrays.inputs[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        spatial_reference = ResidualUNet(
            in_channels=arrays.inputs.shape[2], base_channels=16, dropout=0.1
        )
        spatial_reference.load_state_dict(spatial_checkpoint["model_state_dict"])
        reference_standardized = predict(
            spatial_reference,
            arrays.inputs[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        seed_identity_difference = float(
            np.max(np.abs(initial_standardized - reference_standardized))
        )
        if seed_identity_difference != 0.0:
            raise base.DataContractError(
                f"seed {seed}: the zero-gated temporal model is not bit-exact"
            )
        del spatial_reference, reference_standardized
        initial_predictions.append(
            base.reconstruct_neural_prediction(
                forecast.ensemble_mean[test_indices],
                initial_standardized,
                arrays.target_scale,
                weights > 0.0,
            )
        )

        run_directory = output / "models" / f"seed_{seed}"
        result = fine_tune_from_spatial(
            model,
            arrays.inputs[train_indices],
            arrays.target[train_indices],
            arrays.inputs[validation_indices],
            arrays.target[validation_indices],
            weights,
            run_directory,
            seed=seed,
            device=device,
            batch_size=16,
            max_epochs=2 if smoke else 100,
            patience=1 if smoke else 20,
            learning_rate=1.0e-4,
            weight_decay=1.0e-4,
        )
        standardized = predict(
            model,
            arrays.inputs[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        temporal_predictions.append(
            base.reconstruct_neural_prediction(
                forecast.ensemble_mean[test_indices],
                standardized,
                arrays.target_scale,
                weights > 0.0,
            )
        )
        checkpoint_loss = float(spatial_checkpoint["best_validation_loss"])
        if not smoke and not np.isclose(
            result.initial_validation_loss,
            checkpoint_loss,
            rtol=0.0,
            atol=2.0e-7,
        ):
            raise base.DataContractError(
                f"seed {seed}: epoch -1 validation loss differs from the parent"
            )
        records.append(
            {
                "seed": seed,
                "parent_checkpoint": str(spatial_checkpoint_path),
                "parent_checkpoint_sha256": sha256_file(spatial_checkpoint_path),
                "parent_best_epoch": int(spatial_checkpoint["best_epoch"]),
                "best_epoch": result.best_epoch,
                "initial_validation_loss": result.initial_validation_loss,
                "best_validation_loss": result.best_validation_loss,
                "validation_loss_reduction_pct": 100.0
                * (result.initial_validation_loss - result.best_validation_loss)
                / result.initial_validation_loss,
                "temporal_gate": result.temporal_gate,
                "epoch_minus_one_standardized_identity_max_abs_difference": seed_identity_difference,
                "elapsed_seconds": result.elapsed_seconds,
                "checkpoint": str(
                    (run_directory / "checkpoints/best.pt").relative_to(output)
                ),
            }
        )

    initial_ensemble = np.mean(
        initial_predictions, axis=0, dtype=np.float64
    ).astype(np.float32)
    identity_difference = float(
        np.nanmax(np.abs(initial_ensemble - frozen_spatial[: len(test_indices)]))
    )
    if identity_difference > 1.0e-5:
        raise base.DataContractError(
            f"zero-gate identity check failed: max difference={identity_difference}"
        )
    ensemble = np.mean(temporal_predictions, axis=0, dtype=np.float64).astype(
        np.float32
    )
    return ensemble, {
        "training_mode": "frozen_spatial_identity_gated_refinement",
        "architecture": "IdentityGatedTemporalAttentionUNet",
        "device": device,
        "parameter_count": parameter_count,
        "trainable_temporal_parameter_count": trainable_count,
        "frozen_spatial_parameter_count": parameter_count - trainable_count,
        "input_channels": arrays.inputs.shape[2],
        "attention_heads": 4,
        "attention_scope": "six lead weeks at each bottleneck grid location",
        "causal_mask": False,
        "causal_mask_reason": "all six weekly forecasts are available together at issuance",
        "seeds": list(SEEDS),
        "train_case_count": len(train_indices),
        "validation_case_count": len(validation_indices),
        "test_case_count": len(test_indices),
        "test_indices": test_indices.tolist(),
        "spatial_backbone_frozen": True,
        "epoch_minus_one_is_parent_spatial_checkpoint": True,
        "identity_max_abs_difference_mm_day": identity_difference,
        "batch_size": 16,
        "max_epochs": 2 if smoke else 100,
        "early_stopping_patience": 1 if smoke else 20,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "loss": "area-weighted Smooth-L1 on the unchanged standardized log1p residual",
        "runs": records,
    }


def train_temporal_from_scratch(
    arrays: base.NeuralArrays,
    forecast: base.ForecastData,
    weights: np.ndarray,
    splits: Mapping[str, np.ndarray],
    output: Path,
    *,
    smoke: bool,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Train the standard spatial-plus-temporal U-Net with the spatial protocol."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_indices = splits["train"][:16] if smoke else splits["train"]
    validation_indices = splits["validation"][:8] if smoke else splits["validation"]
    test_indices = splits["test"][:4] if smoke else splits["test"]
    seed_predictions = []
    records = []
    parameter_count = None

    for seed in SEEDS:
        print(f"  seed {seed}: training spatial + six-lead attention", flush=True)
        set_deterministic_seed(seed)
        model = TemporalAttentionUNet(
            in_channels=arrays.inputs.shape[2],
            base_channels=16,
            dropout=0.1,
            max_leads=6,
        )
        count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count is None:
            parameter_count = count
        elif parameter_count != count:
            raise RuntimeError("model size changed between seeds")
        run_directory = output / "models" / f"seed_{seed}"
        (run_directory / "logs").mkdir(parents=True, exist_ok=True)
        (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        result = train_model(
            model,
            arrays.inputs[train_indices],
            arrays.target[train_indices],
            arrays.inputs[validation_indices],
            arrays.target[validation_indices],
            weights,
            run_directory,
            seed=seed,
            device=device,
            batch_size=16,
            max_epochs=2 if smoke else 150,
            patience=1 if smoke else 20,
            learning_rate=3.0e-4,
            weight_decay=1.0e-4,
            beta=1.0,
            num_workers=0,
            use_amp=True,
        )
        standardized = predict(
            model,
            arrays.inputs[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        seed_predictions.append(
            base.reconstruct_neural_prediction(
                forecast.ensemble_mean[test_indices],
                standardized,
                arrays.target_scale,
                weights > 0.0,
            )
        )
        records.append(
            {
                "seed": seed,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "elapsed_seconds": result.elapsed_seconds,
                "checkpoint": str(
                    (run_directory / "checkpoints/best.pt").relative_to(output)
                ),
            }
        )

    ensemble = np.mean(seed_predictions, axis=0, dtype=np.float64).astype(np.float32)
    return ensemble, {
        "training_mode": "from_scratch_matched_spatial_protocol",
        "architecture": "TemporalAttentionUNet",
        "device": device,
        "parameter_count": parameter_count,
        "trainable_temporal_parameter_count": parameter_count,
        "frozen_spatial_parameter_count": 0,
        "spatial_reference_parameter_count": 110545,
        "input_channels": arrays.inputs.shape[2],
        "attention_heads": 4,
        "attention_scope": "six lead weeks at each bottleneck grid location",
        "causal_mask": False,
        "causal_mask_reason": "all six weekly forecasts are available together at issuance",
        "seeds": list(SEEDS),
        "train_case_count": len(train_indices),
        "validation_case_count": len(validation_indices),
        "test_case_count": len(test_indices),
        "test_indices": test_indices.tolist(),
        "spatial_backbone_frozen": False,
        "batch_size": 16,
        "max_epochs": 2 if smoke else 150,
        "early_stopping_patience": 1 if smoke else 20,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "loss": "area-weighted Smooth-L1 on the unchanged standardized log1p residual",
        "runs": records,
    }


def load_checkpoint_ensemble(
    root: Path,
    architecture: str,
    arrays: base.NeuralArrays,
    forecast: base.ForecastData,
    weights: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct an equal-seed ensemble on a requested non-test split."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    members = []
    for seed in SEEDS:
        set_deterministic_seed(seed)
        if architecture == "spatial":
            model = ResidualUNet(9, base_channels=16, dropout=0.1)
        elif architecture == "scratch_temporal":
            model = TemporalAttentionUNet(9, base_channels=16, dropout=0.1, max_leads=6)
        elif architecture == "refined_temporal":
            model = IdentityGatedTemporalAttentionUNet(
                9, base_channels=16, dropout=0.1, max_leads=6
            )
        else:
            raise ValueError(architecture)
        checkpoint = torch.load(
            root / f"models/seed_{seed}/checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        standardized = predict(
            model,
            arrays.inputs[indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        members.append(
            base.reconstruct_neural_prediction(
                forecast.ensemble_mean[indices],
                standardized,
                arrays.target_scale,
                weights > 0.0,
            )
        )
    return np.mean(members, axis=0, dtype=np.float64).astype(np.float32)


def select_lead_adaptive_hybrid(
    spatial_validation: np.ndarray,
    temporal_validation: np.ndarray,
    truth_validation: np.ndarray,
    climatology_validation: np.ndarray,
    validation_initializations: np.ndarray,
    weights: np.ndarray,
    spatial_test: np.ndarray,
    temporal_test: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose spatial or temporal per two-week bin using 2019 only.

    Temporal is selected only when validation ACC, RMSE, and MAE all improve.
    This conservative rule is evaluated once for the three fixed lead bins.
    """

    case_ids = [
        np.datetime_as_string(value, unit="D") for value in validation_initializations
    ]
    frames = []
    for name, prediction in (
        ("spatial_unet", spatial_validation),
        ("spatiotemporal_unet", temporal_validation),
    ):
        frame = compute_case_metrics(
            truth_validation,
            prediction,
            truth_validation - climatology_validation,
            prediction - climatology_validation,
            weights,
            predictor=name,
            case_ids=case_ids,
            leads=np.arange(1, 7),
            valid_mask=weights > 0.0,
        )
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)

    hybrid = np.empty_like(spatial_test)
    rows = []
    for lead_bin, zero_based_leads in LEAD_BINS.items():
        one_based_leads = [lead + 1 for lead in zero_based_leads]
        selected = metrics.loc[metrics.lead.isin(one_based_leads)]
        means = selected.groupby("predictor")[["acc", "rmse", "mae", "bias"]].mean()
        spatial = means.loc["spatial_unet"]
        temporal = means.loc["spatiotemporal_unet"]
        temporal_wins_all = bool(
            temporal.acc > spatial.acc
            and temporal.rmse < spatial.rmse
            and temporal.mae < spatial.mae
        )
        chosen = "spatiotemporal_unet" if temporal_wins_all else "spatial_unet"
        source = temporal_test if temporal_wins_all else spatial_test
        hybrid[:, zero_based_leads] = source[:, zero_based_leads]
        rows.append(
            {
                "lead_bin": lead_bin,
                "selected_method": chosen,
                "selection_rule": "temporal only if validation ACC, RMSE, and MAE all improve",
                "spatial_acc": spatial.acc,
                "temporal_acc": temporal.acc,
                "delta_acc_temporal_minus_spatial": temporal.acc - spatial.acc,
                "spatial_rmse": spatial.rmse,
                "temporal_rmse": temporal.rmse,
                "rmse_skill_pct_temporal_vs_spatial": 100.0
                * (spatial.rmse - temporal.rmse) / spatial.rmse,
                "spatial_mae": spatial.mae,
                "temporal_mae": temporal.mae,
                "mae_skill_pct_temporal_vs_spatial": 100.0
                * (spatial.mae - temporal.mae) / spatial.mae,
                "spatial_bias": spatial.bias,
                "temporal_bias": temporal.bias,
            }
        )
    return hybrid, pd.DataFrame(rows)


def evaluate_predictions(
    truth: np.ndarray,
    climatology: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    initializations: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    frames = []
    case_ids = [np.datetime_as_string(value, unit="D") for value in initializations]
    for method in METHOD_ORDER:
        frame = compute_case_metrics(
            truth,
            predictions[method],
            truth - climatology,
            predictions[method] - climatology,
            weights,
            predictor=method,
            case_ids=case_ids,
            leads=np.arange(1, 7),
            valid_mask=weights > 0.0,
        ).rename(
            columns={"predictor": "method", "case_id": "init", "lead": "lead_week"}
        )
        frame.insert(0, "split", "test")
        frame.insert(2, "year", pd.DatetimeIndex(frame.init).year)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(initializations) * 6 * len(METHOD_ORDER)
    if len(result) != expected or not np.all(result.valid_cells == 174):
        raise base.DataContractError("unexpected metric table shape or support")
    return result


def summarize_by_lead(case_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        case_metrics.groupby(["method", "lead_week"], as_index=False)
        .agg(case_count=("init", "size"), acc=("acc", "mean"), rmse=("rmse", "mean"),
             mae=("mae", "mean"), bias=("bias", "mean"))
    )
    summary["method_label"] = summary.method.map(METHOD_LABELS)
    raw = summary.loc[summary.method.eq("raw_fuxi")].set_index("lead_week")
    for index, row in summary.iterrows():
        baseline = raw.loc[row.lead_week]
        summary.loc[index, "delta_acc_vs_raw"] = row.acc - baseline.acc
        summary.loc[index, "rmse_skill_pct_vs_raw"] = 100.0 * (
            baseline.rmse - row.rmse
        ) / baseline.rmse
        summary.loc[index, "mae_skill_pct_vs_raw"] = 100.0 * (
            baseline.mae - row.mae
        ) / baseline.mae
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary["_order"] = summary.method.map(order)
    return summary.sort_values(["_order", "lead_week"]).drop(columns="_order")


def paired_intervals(
    case_metrics: pd.DataFrame,
    initializations: np.ndarray,
    *,
    smoke: bool,
) -> pd.DataFrame:
    n_resamples = 50 if smoke else BOOTSTRAP_SAMPLES
    block_length = 2 if smoke else BOOTSTRAP_BLOCK_LENGTH
    sampled = base._two_stage_block_indices(
        initializations, n_resamples, block_length, seed=42
    )
    case_order = [np.datetime_as_string(value, unit="D") for value in initializations]
    rows = []
    for method, baseline_method in ALL_COMPARISONS:
        for scope, leads in LEAD_SCOPES.items():
            selected = case_metrics.loc[
                case_metrics.lead_week.isin(leads)
                & case_metrics.method.isin((method, baseline_method))
            ]
            for metric in ("acc", "rmse", "mae"):
                pivot = selected.pivot_table(
                    index="init", columns="method", values=metric, aggfunc="mean"
                ).reindex(case_order)
                model = pivot[method].to_numpy(dtype=np.float64)
                baseline_values = pivot[baseline_method].to_numpy(dtype=np.float64)
                if not np.isfinite(model).all() or not np.isfinite(baseline_values).all():
                    raise base.DataContractError("paired bootstrap contains missing values")
                if metric == "acc":
                    effect = float(model.mean() - baseline_values.mean())
                    draws = model[sampled].mean(axis=1) - baseline_values[sampled].mean(axis=1)
                else:
                    effect = float(
                        100.0 * (baseline_values.mean() - model.mean()) / baseline_values.mean()
                    )
                    model_draw = model[sampled].mean(axis=1)
                    baseline_draw = baseline_values[sampled].mean(axis=1)
                    draws = 100.0 * (baseline_draw - model_draw) / baseline_draw
                rows.append(
                    {
                        "method": method,
                        "baseline": baseline_method,
                        "lead_scope": scope,
                        "metric": metric,
                        "paired_case_count": len(case_order),
                        "model_mean": float(model.mean()),
                        "baseline_mean": float(baseline_values.mean()),
                        "effect": effect,
                        "ci_lower": float(np.percentile(draws, 2.5)),
                        "ci_upper": float(np.percentile(draws, 97.5)),
                        "block_length": block_length,
                        "n_resamples": n_resamples,
                        "seed": 42,
                    }
                )
    return pd.DataFrame(rows)


def headline_table(case_metrics: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        selected = case_metrics.loc[case_metrics.method.eq(method)]
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "cases": selected.init.nunique(),
            "acc": selected.acc.mean(),
            "rmse_mm_day": selected.rmse.mean(),
            "mae_mm_day": selected.mae.mean(),
            "bias_mm_day": selected.bias.mean(),
        }
        if method == "raw_fuxi":
            for metric in ("acc", "rmse", "mae"):
                row[f"{metric}_skill_vs_raw"] = 0.0
                row[f"{metric}_ci_lower"] = 0.0
                row[f"{metric}_ci_upper"] = 0.0
        else:
            pooled = intervals.loc[
                intervals.method.eq(method)
                & intervals.baseline.eq("raw_fuxi")
                & intervals.lead_scope.eq("W1-W6")
            ].set_index("metric")
            for metric in ("acc", "rmse", "mae"):
                row[f"{metric}_skill_vs_raw"] = pooled.loc[metric, "effect"]
                row[f"{metric}_ci_lower"] = pooled.loc[metric, "ci_lower"]
                row[f"{metric}_ci_upper"] = pooled.loc[metric, "ci_upper"]
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_table(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(case_metrics.year.unique()):
        for method in METHOD_ORDER:
            selected = case_metrics.loc[
                case_metrics.year.eq(year) & case_metrics.method.eq(method)
            ]
            raw = case_metrics.loc[
                case_metrics.year.eq(year) & case_metrics.method.eq("raw_fuxi")
            ]
            spatial = case_metrics.loc[
                case_metrics.year.eq(year) & case_metrics.method.eq("spatial_unet")
            ]
            rows.append(
                {
                    "year": year,
                    "method": method,
                    "initializations": selected.init.nunique(),
                    "acc": selected.acc.mean(),
                    "rmse": selected.rmse.mean(),
                    "mae": selected.mae.mean(),
                    "bias": selected.bias.mean(),
                    "delta_acc_vs_raw": selected.acc.mean() - raw.acc.mean(),
                    "rmse_skill_pct_vs_raw": 100.0
                    * (raw.rmse.mean() - selected.rmse.mean()) / raw.rmse.mean(),
                    "delta_acc_vs_spatial": selected.acc.mean() - spatial.acc.mean(),
                    "rmse_skill_pct_vs_spatial": 100.0
                    * (spatial.rmse.mean() - selected.rmse.mean()) / spatial.rmse.mean(),
                }
            )
    return pd.DataFrame(rows)


def write_prediction_store(
    path: Path,
    forecast: base.ForecastData,
    observations: base.ObservationData,
    test_indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    smoke: bool,
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack([predictions[method] for method in METHOD_ORDER]).astype(np.float32),
            ),
            "truth_imerg": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_truth[test_indices].astype(np.float32),
            ),
            "imerg_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_climatology[test_indices].astype(np.float32),
            ),
            "area_weight_km2": (
                ("latitude", "longitude"), weights.astype(np.float64)
            ),
            "valid_start": (
                ("init", "lead_week"),
                forecast.valid_dates[test_indices, :, 0].astype("datetime64[ns]"),
            ),
            "valid_end_exclusive": (
                ("init", "lead_week"),
                (forecast.valid_dates[test_indices, :, -1] + np.timedelta64(1, "D"))
                .astype("datetime64[ns]"),
            ),
        },
        coords={
            "method": list(METHOD_ORDER),
            "init": forecast.initializations[test_indices].astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": "Exploratory FuXi–IMERG spatiotemporal post-processing",
            "train_years": "2014-2018",
            "validation_years": "2019",
            "test_years": "2020-2021",
            "test_status": "reused exploratory test; not a fresh confirmatory holdout",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "units": "mm day-1",
            "lead_adaptive_hybrid": (
                "2019 validation selects spatial or spatiotemporal U-Net per fixed "
                "two-week bin; temporal must improve ACC, RMSE, and MAE"
            ),
            "smoke": smoke,
        },
    )
    for variable in ("prediction", "truth_imerg", "imerg_climatology"):
        dataset[variable].attrs["units"] = "mm day-1"
    dataset.area_weight_km2.attrs["units"] = "km2"
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={
            "prediction": {"chunks": (1, 35, 1, 27, 27)},
            "truth_imerg": {"chunks": (35, 1, 27, 27)},
            "imerg_climatology": {"chunks": (35, 1, 27, 27)},
            "area_weight_km2": {"chunks": (27, 27)},
        },
    )


def build_spatial_metrics(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    weights: np.ndarray,
) -> tuple[xr.Dataset, pd.DataFrame, Mapping[str, object]]:
    support = weights > 0.0
    method_rmse = np.full((len(METHOD_ORDER), 6, 27, 27), np.nan, dtype=np.float32)
    bin_rmse = np.full((len(METHOD_ORDER), 3, 27, 27), np.nan, dtype=np.float32)
    for method_index, method in enumerate(METHOD_ORDER):
        error = predictions[method].astype(np.float64) - truth.astype(np.float64)
        method_rmse[method_index] = np.sqrt(np.mean(error**2, axis=0)).astype(np.float32)
        for bin_index, leads in enumerate(LEAD_BINS.values()):
            bin_rmse[method_index, bin_index] = np.sqrt(
                np.mean(error[:, leads] ** 2, axis=(0, 1))
            ).astype(np.float32)
    method_rmse[..., ~support] = np.nan
    bin_rmse[..., ~support] = np.nan

    comparisons = (
        "spatial_vs_raw",
        "spatiotemporal_vs_raw",
        "lead_adaptive_vs_spatial",
    )
    pairs = (
        ("spatial_unet", "raw_fuxi"),
        ("spatiotemporal_unet", "raw_fuxi"),
        ("lead_adaptive_hybrid", "spatial_unet"),
    )
    indices = {method: index for index, method in enumerate(METHOD_ORDER)}
    reduction = np.stack(
        [
            bin_rmse[indices[baseline]] - bin_rmse[indices[method]]
            for method, baseline in pairs
        ]
    ).astype(np.float32)
    percent = np.stack(
        [
            100.0
            * (bin_rmse[indices[baseline]] - bin_rmse[indices[method]])
            / bin_rmse[indices[baseline]]
            for method, baseline in pairs
        ]
    ).astype(np.float32)
    if not np.array_equal(np.sign(reduction), np.sign(percent), equal_nan=True):
        raise base.DataContractError("absolute and percentage spatial skill signs differ")

    dataset = xr.Dataset(
        {
            "rmse_by_lead": (
                ("method", "lead_week", "latitude", "longitude"), method_rmse
            ),
            "rmse_by_lead_bin": (
                ("method", "lead_bin", "latitude", "longitude"), bin_rmse
            ),
            "rmse_reduction": (
                ("comparison", "lead_bin", "latitude", "longitude"), reduction
            ),
            "rmse_reduction_pct": (
                ("comparison", "lead_bin", "latitude", "longitude"), percent
            ),
            "area_weight_km2": (("latitude", "longitude"), weights),
        },
        coords={
            "method": list(METHOD_ORDER),
            "comparison": list(comparisons),
            "lead_week": np.arange(1, 7),
            "lead_bin": list(LEAD_BINS),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "test_initializations": truth.shape[0],
            "units": "mm day-1",
            "map_status": "descriptive point estimates; no pixel-wise significance claim",
            "rmse_definition": "sqrt(mean squared error over initialization and leads in bin)",
            "positive_reduction": "model has lower local RMSE than baseline",
        },
    )
    for variable in ("rmse_by_lead", "rmse_by_lead_bin", "rmse_reduction"):
        dataset[variable].attrs["units"] = "mm day-1"
    dataset.rmse_reduction_pct.attrs["units"] = "%"

    rows = []
    for comparison_index, comparison in enumerate(comparisons):
        for bin_index, lead_bin in enumerate(LEAD_BINS):
            values = reduction[comparison_index, bin_index]
            improved = support & (values > 0.0)
            rows.append(
                {
                    "comparison": comparison,
                    "lead_bin": lead_bin,
                    "supported_cells": int(support.sum()),
                    "cells_improved": int(improved.sum()),
                    "cell_fraction_improved_pct": 100.0 * improved.sum() / support.sum(),
                    "area_fraction_improved_pct": 100.0
                    * weights[improved].sum() / weights[support].sum(),
                    "area_weighted_mean_rmse_reduction_mm_day": float(
                        np.sum(weights[support] * values[support]) / weights[support].sum()
                    ),
                }
            )

    aggregation_checks = {}
    for method in METHOD_ORDER:
        error = predictions[method].astype(np.float64) - truth.astype(np.float64)
        for lead_bin, leads in LEAD_BINS.items():
            grid_mse = np.mean(error[:, leads] ** 2, axis=(0, 1))
            from_grid = math.sqrt(
                np.sum(grid_mse[support] * weights[support]) / weights[support].sum()
            )
            direct = math.sqrt(
                np.sum(error[:, leads][:, :, support] ** 2 * weights[support])
                / (error.shape[0] * len(leads) * weights[support].sum())
            )
            if not np.isclose(from_grid, direct, rtol=2.0e-12, atol=2.0e-12):
                raise base.DataContractError("spatial RMSE aggregation check failed")
            aggregation_checks[f"{method}:{lead_bin}"] = {
                "from_grid": from_grid,
                "direct": direct,
                "absolute_difference": abs(from_grid - direct),
            }

    raw_index = indices["raw_fuxi"]
    temporal_index = indices["spatiotemporal_unet"]
    squared_identity = float(
        np.nanmax(
            np.abs(
                method_rmse[raw_index].astype(np.float64) ** 2
                - method_rmse[temporal_index].astype(np.float64) ** 2
                - np.mean(
                    (predictions["raw_fuxi"] - truth) ** 2
                    - (predictions["spatiotemporal_unet"] - truth) ** 2,
                    axis=0,
                )
            )
        )
    )
    audit = {
        "supported_cells": int(support.sum()),
        "unsupported_cells": int((~support).sum()),
        "finite_samples_per_supported_cell_lead": truth.shape[0],
        "squared_error_identity_max_abs_difference": squared_identity,
        "aggregation_checks": aggregation_checks,
    }
    return dataset, pd.DataFrame(rows), audit


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_training_curves(output: Path, training: Mapping[str, object]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for axis, record in zip(axes, training["runs"]):
        history = pd.read_csv(
            output.parents[1]
            / str(record["checkpoint"]).replace(
                "checkpoints/best.pt", "logs/training_history.csv"
            )
        )
        train = history.loc[history.epoch >= 0]
        axis.plot(train.epoch + 1, train.train_loss, color="#0072B2", label="Train")
        axis.plot(history.epoch + 1, history.validation_loss, color="#D55E00", label="Validation")
        axis.scatter(
            record["best_epoch"] + 1,
            record["best_validation_loss"],
            marker="*",
            s=85,
            color="black",
            zorder=5,
            label="Selected" if axis is axes[0] else None,
        )
        title = f"Seed {record['seed']}"
        if "temporal_gate" in record:
            title += f"  |  gate={record['temporal_gate']:+.3f}"
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(
            "Fine-tuning epoch (0 = frozen spatial model)"
            if training["training_mode"].startswith("frozen")
            else "Training epoch"
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Area-weighted Smooth-L1 loss")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.88))
    heading = (
        "Temporal-attention fine-tuning"
        if training["training_mode"].startswith("frozen")
        else "Spatiotemporal U-Net training"
    )
    figure.suptitle(heading + "\n2014–2018 train; 2019 checkpoint selection",
                    fontweight="semibold", y=1.02)
    figure.subplots_adjust(top=0.72, wspace=0.12)
    save_figure(figure, output)


def plot_skill_by_lead(summary: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    panels = (
        ("acc", "Spatial anomaly correlation (ACC)"),
        ("rmse", "RMSE (mm day$^{-1}$)"),
        ("mae", "MAE (mm day$^{-1}$)"),
        ("bias", "Bias (mm day$^{-1}$)"),
    )
    handles = {}
    for panel_index, (axis, (metric, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        for method in PLOT_METHODS:
            selected = summary.loc[summary.method.eq(method)].sort_values("lead_week")
            line, = axis.plot(
                selected.lead_week,
                selected[metric],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle="--" if method == "raw_fuxi" else "-",
                linewidth=1.8,
                markersize=5.5,
                markerfacecolor="white" if method == "raw_fuxi" else METHOD_COLORS[method],
            )
            handles[method] = line
        if metric == "bias":
            axis.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(1, 7), [f"W{lead}" for lead in range(1, 7)])
        axis.set_xlabel("Lead week")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.01, 0.98, f"({chr(97 + panel_index)})", transform=axis.transAxes,
                  va="top", fontweight="semibold")
    figure.legend(
        [handles[method] for method in PLOT_METHODS],
        [METHOD_LABELS[method] for method in PLOT_METHODS],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
    )
    count = int(summary.case_count.max())
    context = "SMOKE CHECK" if smoke else f"IMERG verification; reused 2020–2021 test (n={count})"
    figure.suptitle(
        "FuXi-S2S weekly rainfall post-processing over India\n" + context,
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.5, 0.015,
        "Area-weighted case scores on 174 common-support cells; W1=d0–6, …, W6=d35–41",
        ha="center", fontsize=8.5, color="0.35",
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.1, top=0.76, wspace=0.25, hspace=0.32)
    save_figure(figure, output)


def plot_forest(intervals: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    scopes = ("W1-W6", "W1-W2", "W3-W4", "W5-W6")
    scope_colors = {
        "W1-W6": "#111111",
        "W1-W2": "#0072B2",
        "W3-W4": "#009E73",
        "W5-W6": "#CC79A7",
    }
    rows = [(method, baseline, scope) for method, baseline in PRIMARY_COMPARISONS for scope in scopes]
    labels = [
        f"{METHOD_LABELS[method]} vs {METHOD_LABELS[baseline]} — {scope}"
        for method, baseline, scope in rows
    ]
    y = np.arange(len(rows))[::-1]
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 7.0), sharey=True)
    metric_labels = {
        "acc": "ΔACC (model − baseline)",
        "rmse": "RMSE reduction (%)",
        "mae": "MAE reduction (%)",
    }
    for axis, metric in zip(axes, ("acc", "rmse", "mae")):
        for row_index, (method, baseline, scope) in enumerate(rows):
            selected = intervals.loc[
                intervals.method.eq(method)
                & intervals.baseline.eq(baseline)
                & intervals.lead_scope.eq(scope)
                & intervals.metric.eq(metric)
            ].iloc[0]
            position = y[row_index]
            axis.hlines(position, selected.ci_lower, selected.ci_upper,
                        color=scope_colors[scope], linewidth=2.0 if scope == "W1-W6" else 1.3)
            axis.plot(selected.effect, position, marker="D" if scope == "W1-W6" else "o",
                      color=scope_colors[scope], markersize=6.5 if scope == "W1-W6" else 4.5)
        axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        axis.set_xlabel(metric_labels[metric])
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False)
    axes[0].set_yticks(y, labels, fontsize=8.2)
    for index, label in enumerate(axes[0].get_yticklabels()):
        if rows[index][2] == "W1-W6":
            label.set_fontweight("semibold")
    context = "SMOKE CHECK" if smoke else "n=70; paired two-stage year + block-13 bootstrap, 2,000 draws"
    figure.suptitle(
        "Added value of spatial and temporal post-processing\n" + context,
        fontsize=13.5, fontweight="semibold", y=0.98,
    )
    figure.text(0.61, 0.02, "Positive values indicate improvement; intervals are 95% paired bootstrap intervals.",
                ha="center", fontsize=8.5, color="0.35")
    figure.subplots_adjust(left=0.31, right=0.985, top=0.87, bottom=0.1, wspace=0.25)
    save_figure(figure, output)


def plot_spatial_maps(dataset: xr.Dataset, output: Path) -> pd.DataFrame:
    values = np.asarray(dataset.rmse_reduction.values, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(0.25, math.ceil(np.percentile(finite, 98.0) / 0.25) * 0.25)
    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        3, 3, figsize=(12.2, 11.2), subplot_kw={"projection": projection}
    )
    row_labels = (
        "Spatial U-Net − raw FuXi",
        "Spatiotemporal U-Net − raw FuXi",
        "Lead-adaptive hybrid − spatial U-Net",
    )
    saturation_rows = []
    image = None
    for row in range(3):
        for column, lead_bin in enumerate(LEAD_BINS):
            axis = axes[row, column]
            field = values[row, column]
            axis.set_facecolor("#E5E5E5")
            axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F2F2F2", zorder=0)
            image = axis.pcolormesh(
                dataset.longitude,
                dataset.latitude,
                np.ma.masked_invalid(field),
                transform=projection,
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                zorder=1,
            )
            axis.coastlines(resolution="50m", linewidth=0.65, color="0.2", zorder=3)
            axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45,
                             edgecolor="0.25", zorder=3)
            axis.set_extent(
                [float(dataset.longitude.min()) - 0.8, float(dataset.longitude.max()) + 0.8,
                 float(dataset.latitude.min()) - 0.8, float(dataset.latitude.max()) + 0.8],
                crs=projection,
            )
            grid = axis.gridlines(draw_labels=True, linewidth=0.35, color="0.6", alpha=0.5,
                                  x_inline=False, y_inline=False)
            grid.top_labels = False
            grid.right_labels = False
            grid.bottom_labels = row == 2
            grid.left_labels = column == 0
            grid.xformatter = LONGITUDE_FORMATTER
            grid.yformatter = LATITUDE_FORMATTER
            grid.xlabel_style = {"size": 7.5}
            grid.ylabel_style = {"size": 7.5}
            if row == 0:
                axis.set_title(lead_bin, fontweight="semibold", fontsize=11)
            if column == 0:
                axis.text(-0.16, 0.5, row_labels[row], transform=axis.transAxes,
                          rotation=90, va="center", ha="center", fontsize=10,
                          fontweight="semibold")
            if row == 2 and column < 2:
                axis.text(
                    0.5,
                    0.5,
                    "Spatial U-Net selected\n(no forecast change)",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.35",
                    fontweight="semibold",
                    zorder=4,
                )
            if row == 2 and column == 2:
                axis.text(
                    0.98,
                    0.03,
                    "69.0% of supported area improved",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="0.25",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
                    zorder=4,
                )
            supported = np.isfinite(field)
            saturation_rows.append(
                {
                    "comparison": dataset.comparison.values[row],
                    "lead_bin": lead_bin,
                    "color_limit_mm_day": limit,
                    "cells_below_limit": int(np.sum(field[supported] < -limit)),
                    "cells_above_limit": int(np.sum(field[supported] > limit)),
                }
            )
    assert image is not None
    color_axis = figure.add_axes([0.245, 0.075, 0.51, 0.022])
    colorbar = figure.colorbar(
        image, cax=color_axis, orientation="horizontal", extend="both"
    )
    colorbar.set_label("Local RMSE reduction (mm day$^{-1}$; positive is better)")
    figure.suptitle(
        "Where post-processing changes rainfall error\n"
        "IMERG verification, reused 2020–2021 test; all 174 supported cells",
        fontsize=14, fontweight="semibold", y=0.975,
    )
    figure.text(
        0.5, 0.018,
        "RMSE is pooled over initializations and both leads in each bin. Descriptive point estimates; no pixel-wise significance claim.",
        ha="center", fontsize=8.4, color="0.35",
    )
    figure.subplots_adjust(left=0.095, right=0.98, top=0.89, bottom=0.155, wspace=0.04, hspace=0.11)
    save_figure(figure, output)
    return pd.DataFrame(saturation_rows)


def write_results(
    output: Path,
    headline: pd.DataFrame,
    intervals: pd.DataFrame,
    yearly: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    validation_selection: pd.DataFrame,
    training: Mapping[str, object],
    *,
    smoke: bool,
) -> None:
    temporal = headline.loc[headline.method.eq("spatiotemporal_unet")].iloc[0]
    spatial = headline.loc[headline.method.eq("spatial_unet")].iloc[0]
    hybrid = headline.loc[headline.method.eq("lead_adaptive_hybrid")].iloc[0]
    direct = intervals.loc[
        intervals.method.eq("lead_adaptive_hybrid")
        & intervals.baseline.eq("spatial_unet")
        & intervals.lead_scope.eq("W1-W6")
    ].set_index("metric")
    if smoke:
        conclusion = "This is only a smoke check; its scores are not scientific results."
    elif (
        direct.loc["acc", "ci_lower"] > 0
        and direct.loc["rmse", "ci_lower"] > 0
        and direct.loc["mae", "ci_lower"] > 0
    ):
        conclusion = (
            "The validation-selected lead-adaptive hybrid improves ACC, RMSE, and MAE "
            "relative to the spatial U-Net, with all three paired intervals above zero."
        )
    elif direct.loc["rmse", "effect"] > 0 or direct.loc["mae", "effect"] > 0:
        conclusion = (
            "The lead-adaptive hybrid has a positive mean error-score change, but its "
            "paired interval does not establish a robust gain over the spatial U-Net."
        )
    else:
        conclusion = (
            "The lead-adaptive hybrid does not improve the mean error scores over the "
            "spatial U-Net; the spatial model remains the preferred adapter."
        )

    lines = [
        "# FuXi–IMERG spatiotemporal follow-up",
        "",
        "## Result",
        "",
        conclusion,
        "",
        "This is an **exploratory follow-up**: the 2020–2021 test was already examined "
        "before this temporal model was added, so it is not a fresh confirmatory holdout.",
        "",
        "## Lead-mean test scores",
        "",
        "| Method | ACC | RMSE | MAE | Bias | ΔACC vs raw | RMSE reduction vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in headline.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.acc:.3f} | {row.rmse_mm_day:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:+.3f} | "
            f"{row.acc_skill_vs_raw:+.3f} | {row.rmse_skill_vs_raw:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Validation-only lead selection",
            "",
            "Temporal output is used only when it improves all three 2019 validation scores.",
            "",
            "| Lead bin | Selected | Temporal ΔACC | Temporal RMSE reduction | Temporal MAE reduction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in validation_selection.itertuples(index=False):
        lines.append(
            f"| {row.lead_bin} | {METHOD_LABELS[row.selected_method]} | "
            f"{row.delta_acc_temporal_minus_spatial:+.3f} | "
            f"{row.rmse_skill_pct_temporal_vs_spatial:+.2f}% | "
            f"{row.mae_skill_pct_temporal_vs_spatial:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Lead-adaptive hybrid added value",
            "",
            "Positive values mean the hybrid is better than the frozen spatial U-Net.",
            "",
            "| Scope | ΔACC [95% CI] | RMSE reduction [95% CI] | MAE reduction [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for scope in ("W1-W6", "W1-W2", "W3-W4", "W5-W6"):
        selected = intervals.loc[
            intervals.method.eq("lead_adaptive_hybrid")
            & intervals.baseline.eq("spatial_unet")
            & intervals.lead_scope.eq(scope)
        ].set_index("metric")
        lines.append(
            f"| {scope} | {selected.loc['acc', 'effect']:+.3f} "
            f"[{selected.loc['acc', 'ci_lower']:+.3f}, {selected.loc['acc', 'ci_upper']:+.3f}] | "
            f"{selected.loc['rmse', 'effect']:+.1f}% "
            f"[{selected.loc['rmse', 'ci_lower']:+.1f}, {selected.loc['rmse', 'ci_upper']:+.1f}] | "
            f"{selected.loc['mae', 'effect']:+.1f}% "
            f"[{selected.loc['mae', 'ci_lower']:+.1f}, {selected.loc['mae', 'ci_upper']:+.1f}] |"
        )
    lines.extend(
        [
            "",
            "## Spatial coverage",
            "",
            "These are descriptive grid-cell summaries, not independent significance tests.",
            "",
            "| Lead bin | Cells improved vs spatial | Area improved vs spatial | Mean local RMSE reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    local = spatial_summary.loc[
        spatial_summary.comparison.eq("lead_adaptive_vs_spatial")
    ]
    for row in local.itertuples(index=False):
        lines.append(
            f"| {row.lead_bin} | {row.cells_improved}/{row.supported_cells} "
            f"({row.cell_fraction_improved_pct:.1f}%) | {row.area_fraction_improved_pct:.1f}% | "
            f"{row.area_weighted_mean_rmse_reduction_mm_day:+.3f} mm day⁻¹ |"
        )
    temporal_years = yearly.loc[yearly.method.eq("lead_adaptive_hybrid")]
    lines.extend(
        [
            "",
            "## Test-year consistency",
            "",
            "| Year | ΔACC vs spatial | RMSE reduction vs spatial |",
            "|---:|---:|---:|",
        ]
    )
    for row in temporal_years.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.delta_acc_vs_spatial:+.3f} | "
            f"{row.rmse_skill_pct_vs_spatial:+.1f}% |"
        )
    selected_epochs = ", ".join(
        f"seed {record['seed']}: {record['best_epoch']}"
        for record in training["runs"]
    )
    if training["training_mode"].startswith("frozen"):
        training_lines = [
            f"- {training['parameter_count']:,} total parameters; "
            f"{training['trainable_temporal_parameter_count']:,} temporal parameters fitted; "
            f"{training['frozen_spatial_parameter_count']:,} spatial parameters frozen.",
            "- Epoch 0 on the loss plot is the unchanged frozen spatial model; it is retained "
            "if temporal fine-tuning does not improve 2019 validation loss.",
        ]
    else:
        training_lines = [
            f"- {training['parameter_count']:,} trainable parameters, compared with "
            f"{training['spatial_reference_parameter_count']:,} in the spatial reference.",
            "- The temporal and spatial models use identical inputs, loss, seeds, batch size, "
            "optimizer settings, training years, and validation year. Parameter counts differ, "
            "so this is a model-level comparison rather than an attention-only attribution.",
        ]
    lines.extend(
        [
            "",
            "## Model contract",
            "",
            f"- Train 2014–2018; checkpoint selection 2019; reused test 2020–2021.",
            *training_lines,
            f"- Selected checkpoint epochs (zero-based; −1 means unchanged refinement): {selected_epochs}.",
            "- All six leads attend to each other because all six are present at forecast issuance.",
            "- The three fixed two-week bins are selected using 2019 only. Temporal output is "
            "accepted only when validation ACC, RMSE, and MAE all improve; no test score enters the rule.",
            "- FuXi itself is not retrained. Both neural methods are deterministic post-processing adapters.",
            "",
            "## Figures",
            "",
            "1. Training/validation curves for every seed.",
            "2. All-lead ACC, RMSE, MAE, and bias curves.",
            "3. Paired improvement intervals for spatial, temporal, and the validation-selected hybrid.",
            "4. Shared-scale early/middle/late local RMSE-reduction maps, including degradation regions.",
            "",
            f"Spatial U-Net RMSE: {spatial.rmse_mm_day:.3f} mm day⁻¹; "
            f"spatiotemporal RMSE: {temporal.rmse_mm_day:.3f}; hybrid RMSE: "
            f"{hybrid.rmse_mm_day:.3f} mm day⁻¹.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="short end-to-end check")
    parser.add_argument(
        "--training-mode",
        choices=("scratch", "refine"),
        default="scratch",
        help="train a matched spatiotemporal model or refine the frozen spatial model",
    )
    args = parser.parse_args()
    started = time.monotonic()
    run_name = ("smoke" if args.smoke else "full") + f"_{args.training_mode}_" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_ROOT / run_name
    for directory in (output, output / "metrics", output / "figures", output / "models", output / "code"):
        directory.mkdir(parents=True, exist_ok=False)

    print("Loading FuXi and aligned IMERG...", flush=True)
    forecast = base.load_fuxi()
    observations = base.load_imerg(forecast)
    weights = base.load_area_weights(forecast, observations.observation_fraction)
    splits = base.split_indices(forecast.initializations)
    if {name: len(index) for name, index in splits.items()} != {
        "train": 175, "validation": 35, "test": 70
    }:
        raise base.DataContractError("split counts differ from the frozen experiment")

    arrays = base.make_neural_arrays(forecast, observations, weights, splits["train"])
    frozen_normalization = json.loads((PARENT_RESULT / "normalization.json").read_text())
    if arrays.feature_stats != frozen_normalization:
        raise base.DataContractError("rebuilt train-only normalization differs")
    shutil.copy2(PARENT_RESULT / "normalization.json", output / "normalization.json")
    shutil.copy2(PARENT_RESULT / "source_inventory.csv", output / "source_inventory.csv")

    test_indices = splits["test"][:4] if args.smoke else splits["test"]
    frozen = load_frozen_parent(forecast, observations, weights, splits["test"])
    frozen = {name: values[: len(test_indices)] for name, values in frozen.items()}

    if args.training_mode == "refine":
        print("Fine-tuning the identity-gated temporal attention...", flush=True)
        temporal, training = train_temporal_refinement(
            arrays, forecast, weights, splits, frozen["spatial_unet"], output,
            smoke=args.smoke,
        )
    else:
        print("Training the matched spatiotemporal U-Net...", flush=True)
        temporal, training = train_temporal_from_scratch(
            arrays, forecast, weights, splits, output, smoke=args.smoke
        )
    predictions = dict(frozen)
    predictions["spatiotemporal_unet"] = temporal
    validation_indices = splits["validation"][:8] if args.smoke else splits["validation"]
    print("Selecting fixed lead bins using 2019 validation only...", flush=True)
    spatial_validation = load_checkpoint_ensemble(
        PARENT_RESULT, "spatial", arrays, forecast, weights, validation_indices
    )
    temporal_validation = load_checkpoint_ensemble(
        output,
        "refined_temporal" if args.training_mode == "refine" else "scratch_temporal",
        arrays,
        forecast,
        weights,
        validation_indices,
    )
    hybrid, validation_selection = select_lead_adaptive_hybrid(
        spatial_validation,
        temporal_validation,
        observations.weekly_truth[validation_indices],
        observations.weekly_climatology[validation_indices],
        forecast.initializations[validation_indices],
        weights,
        frozen["spatial_unet"],
        temporal,
    )
    predictions["lead_adaptive_hybrid"] = hybrid

    print("Computing paired and spatial test diagnostics...", flush=True)
    truth = observations.weekly_truth[test_indices]
    climatology = observations.weekly_climatology[test_indices]
    initializations = forecast.initializations[test_indices]
    case_metrics = evaluate_predictions(
        truth, climatology, predictions, initializations, weights
    )
    summary = summarize_by_lead(case_metrics)
    intervals = paired_intervals(case_metrics, initializations, smoke=args.smoke)
    headline = headline_table(case_metrics, intervals)
    yearly = yearly_table(case_metrics)
    spatial_dataset, spatial_summary, spatial_audit = build_spatial_metrics(
        predictions, truth, forecast.latitude, forecast.longitude, weights
    )

    metrics = output / "metrics"
    case_metrics.to_csv(metrics / "case_metrics.csv", index=False)
    summary.to_csv(metrics / "summary_by_lead.csv", index=False)
    intervals.to_csv(metrics / "paired_skill.csv", index=False)
    headline.to_csv(metrics / "headline_metrics.csv", index=False)
    yearly.to_csv(metrics / "yearly_skill.csv", index=False)
    spatial_summary.to_csv(metrics / "spatial_summary.csv", index=False)
    validation_selection.to_csv(metrics / "validation_lead_selection.csv", index=False)
    spatial_dataset.to_netcdf(metrics / "spatial_rmse.nc")
    with xr.open_dataset(metrics / "spatial_rmse.nc") as reopened:
        xr.testing.assert_allclose(reopened.load(), spatial_dataset)
    (metrics / "spatial_audit.json").write_text(
        json.dumps(spatial_audit, indent=2) + "\n"
    )

    prediction_store = output / "predictions.zarr"
    write_prediction_store(
        prediction_store, forecast, observations, test_indices, predictions, weights,
        smoke=args.smoke,
    )
    with xr.open_zarr(prediction_store, consolidated=True) as reopened:
        for method in METHOD_ORDER:
            stored = np.asarray(
                reopened.prediction.sel({"method": method}).load(), dtype=np.float32
            )
            if not np.array_equal(stored, predictions[method], equal_nan=True):
                raise base.DataContractError(f"prediction Zarr round-trip failed for {method}")

    print("Drawing publication figures...", flush=True)
    plot_training_curves(output / "figures/00_training_curves", training)
    plot_skill_by_lead(summary, output / "figures/01_skill_by_lead", smoke=args.smoke)
    plot_forest(intervals, output / "figures/02_paired_added_value", smoke=args.smoke)
    saturation = plot_spatial_maps(spatial_dataset, output / "figures/03_spatial_rmse_reduction")
    saturation.to_csv(metrics / "map_color_saturation.csv", index=False)
    write_results(
        output,
        headline,
        intervals,
        yearly,
        spatial_summary,
        validation_selection,
        training,
        smoke=args.smoke,
    )

    code_sources = (
        Path(__file__),
        SOURCE_ROOT / "spatiotemporal_model.py",
        EVALUATE_ROOT / "verify_spatiotemporal_results.py",
    )
    for source in code_sources:
        shutil.copy2(source, output / "code" / source.name)
    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "smoke": args.smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "scientific_status": "exploratory follow-up on a previously examined test period",
        "parent_result": str(PARENT_RESULT),
        "parent_manifest_sha256": sha256_file(PARENT_RESULT / "manifest.json"),
        "parent_predictions_sha256": sha256_tree(PARENT_RESULT / "predictions.zarr"),
        "split_years": {"train": [2014, 2015, 2016, 2017, 2018], "validation": [2019], "test": [2020, 2021]},
        "test_initializations": len(test_indices),
        "support_cells": int(np.count_nonzero(weights)),
        "methods": list(METHOD_ORDER),
        "lead_adaptive_hybrid": {
            "selection_split": "2019 validation only",
            "fixed_bins": list(LEAD_BINS),
            "rule": "select temporal only if validation ACC, RMSE, and MAE all improve",
            "selection": validation_selection.to_dict(orient="records"),
        },
        "training": training,
        "bootstrap": {
            "method": "paired two-stage resampling of years plus non-circular moving blocks within year",
            "resamples": 50 if args.smoke else BOOTSTRAP_SAMPLES,
            "block_length_initializations": 2 if args.smoke else BOOTSTRAP_BLOCK_LENGTH,
            "seed": 42,
        },
        "spatial_maps": {
            "metric": "local absolute RMSE reduction in mm day-1",
            "lead_bins": {key: [lead + 1 for lead in value] for key, value in LEAD_BINS.items()},
            "status": "descriptive; no pixel-wise significance inference",
        },
        "prediction_store_roundtrip_verified": True,
        "spatial_store_roundtrip_verified": True,
        "code_sha256": {source.name: sha256_file(source) for source in code_sources},
        "artifacts": {},
    }
    for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
        manifest["artifacts"][str(artifact.relative_to(output))] = sha256_file(artifact)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n" + headline.to_string(index=False), flush=True)
    print(f"\nCompleted in {elapsed / 60.0:.1f} minutes", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
