#!/usr/bin/env python
"""A100 experiment for a larger FuXi-to-IMERG Week 5--6 adapter.

The experiment uses 2002--2017 for fitting, 2018--2019 for validation, and
does not create 2020--2021 predictions until a validation-only selection file
has been written. Three fixed-seed ensembles separate model size from temporal
mixing: the original small temporal model, a large spatial control, and a
large multi-scale temporal model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imerg_full_archive_latelead as common  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    anchored_composite_loss,
    fit_anchored_target_scale,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.models import (  # noqa: E402
    AttentiveClimatologyAllLeadMultiScaleUNet,
    AttentiveClimatologyAllLeadUNet,
    AttentiveClimatologyMultiScaleUNet,
    AttentiveClimatologyLateLeadUNet,
    FixedClimatologyAllLeadMultiScaleUNet,
    FixedClimatologyAllLeadUNet,
    FixedClimatologyMultiScaleUNet,
    FixedClimatologyLateLeadUNet,
    MultiScaleLateLeadTemporalUNet,
)
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


base = common.base
diagnostics = common.diagnostics

TRAIN_YEARS = tuple(range(2002, 2018))
VALIDATION_YEARS = (2018, 2019)
TEST_YEARS = (2020, 2021)
SEEDS = (42, 43, 44)
ACTIVE_LEADS = (4, 5)
LEAD_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 0.5, 0.5)
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
ALPHA_GRID = np.linspace(0.0, 1.0, 41, dtype=np.float64)
RESULTS_ROOT = HERE / "results" / "fuxi_imerg_a100_big_temporal"


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    architecture: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    color: str


CANDIDATES = (
    Candidate(
        "small_temporal",
        "Small temporal (control)",
        "small_temporal",
        16,
        3.0e-4,
        3.0e-4,
        0.20,
        "#0072B2",
    ),
    Candidate(
        "big_spatial",
        "Large spatial control",
        "big_spatial",
        32,
        1.5e-4,
        1.0e-3,
        0.25,
        "#E69F00",
    ),
    Candidate(
        "big_temporal",
        "Large multi-scale temporal",
        "big_temporal",
        32,
        1.5e-4,
        1.0e-3,
        0.25,
        "#009E73",
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}

METHOD_ORDER = (
    "raw_fuxi",
    "log_bias",
    "small_temporal",
    "big_spatial",
    "big_temporal",
    "selected_model",
)
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Log-bias",
    **{candidate.name: candidate.label for candidate in CANDIDATES},
    "selected_model": "Validation-selected",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#CC79A7",
    **{candidate.name: candidate.color for candidate in CANDIDATES},
    "selected_model": "#D55E00",
}
METHOD_MARKERS = {
    "raw_fuxi": "o",
    "log_bias": "s",
    "small_temporal": "^",
    "big_spatial": "D",
    "big_temporal": "P",
    "selected_model": "*",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_contract() -> None:
    base.TRAIN_YEARS = TRAIN_YEARS
    base.VALIDATION_YEARS = VALIDATION_YEARS
    base.TEST_YEARS = TEST_YEARS
    base.ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

    diagnostics.METHOD_ORDER = METHOD_ORDER
    diagnostics.METHOD_LABELS = METHOD_LABELS
    diagnostics.METHOD_COLORS = METHOD_COLORS
    diagnostics.METHOD_MARKERS = METHOD_MARKERS
    diagnostics.PLOT_METHODS = METHOD_ORDER[:-1]
    comparisons = [(method, "raw_fuxi") for method in METHOD_ORDER[1:]]
    comparisons.extend(
        [
            ("small_temporal", "log_bias"),
            ("big_spatial", "log_bias"),
            ("big_temporal", "log_bias"),
            ("big_spatial", "small_temporal"),
            ("big_temporal", "small_temporal"),
            ("big_temporal", "big_spatial"),
            ("selected_model", "log_bias"),
            ("selected_model", "small_temporal"),
            ("selected_model", "big_temporal"),
        ]
    )
    diagnostics.ALL_COMPARISONS = tuple(dict.fromkeys(comparisons))


def build_candidate(candidate: Candidate, in_channels: int) -> torch.nn.Module:
    if candidate.architecture == "small_temporal":
        return common.LateLeadTemporalUNet(
            in_channels=in_channels,
            base_channels=16,
            dropout=candidate.dropout,
            max_leads=6,
        )
    if candidate.architecture == "big_spatial":
        return common.LateLeadSpatialUNet(
            in_channels=in_channels,
            base_channels=48,
            dropout=candidate.dropout,
        )
    if candidate.architecture == "big_temporal":
        return MultiScaleLateLeadTemporalUNet(
            in_channels=in_channels,
            base_channels=48,
            spatial_dropout=0.25,
            temporal_dropout=0.20,
            skip_layers=2,
            bottleneck_layers=3,
            attention_heads=8,
            layer_scale=1.0e-3,
            context_dropout=0.10,
        )
    if candidate.architecture == "fixed_climatology_temporal":
        return FixedClimatologyMultiScaleUNet(input_channels=in_channels)
    if candidate.architecture == "attention_climatology_temporal":
        return AttentiveClimatologyMultiScaleUNet(input_channels=in_channels)
    if candidate.architecture == "fixed_climatology_week36":
        return FixedClimatologyLateLeadUNet(
            input_channels=in_channels,
            base_channels=16,
            dropout=candidate.dropout,
        )
    if candidate.architecture == "attention_climatology_week36":
        return AttentiveClimatologyLateLeadUNet(
            input_channels=in_channels,
            base_channels=16,
            dropout=candidate.dropout,
        )
    if candidate.architecture == "fixed_climatology_allweeks":
        return FixedClimatologyAllLeadUNet(
            input_channels=in_channels,
            base_channels=16,
            dropout=candidate.dropout,
        )
    if candidate.architecture == "attention_climatology_allweeks":
        return AttentiveClimatologyAllLeadUNet(
            input_channels=in_channels,
            base_channels=16,
            dropout=candidate.dropout,
        )
    if candidate.architecture == "fixed_climatology_big_allweeks":
        return FixedClimatologyAllLeadMultiScaleUNet(input_channels=in_channels)
    if candidate.architecture == "attention_climatology_big_allweeks":
        return AttentiveClimatologyAllLeadMultiScaleUNet(input_channels=in_channels)
    if candidate.architecture == "fixed_climatology_big_allweeks_regularized":
        return FixedClimatologyAllLeadMultiScaleUNet(
            input_channels=in_channels,
            spatial_dropout=0.30,
            temporal_dropout=0.25,
            context_dropout=0.15,
        )
    if candidate.architecture == "attention_climatology_big_allweeks_regularized":
        return AttentiveClimatologyAllLeadMultiScaleUNet(
            input_channels=in_channels,
            spatial_dropout=0.30,
            temporal_dropout=0.25,
            context_dropout=0.15,
        )
    raise ValueError(candidate.architecture)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def train_candidate(
    candidate: Candidate,
    features: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    weights: np.ndarray,
    splits: Mapping[str, np.ndarray],
    output: Path,
    *,
    smoke: bool,
    lead_weights: tuple[float, ...] = LEAD_WEIGHTS,
    inactive_lead_count: int = 4,
    loss_coefficients: Mapping[str, float] = LOSS_COEFFICIENTS,
) -> tuple[np.ndarray, Mapping[str, object]]:
    train_indices = splits["train"][:64] if smoke else splits["train"]
    validation_indices = splits["validation"][:16] if smoke else splits["validation"]
    seeds = SEEDS[:1] if smoke else SEEDS
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parallel = len(seeds) > 1 and gpu_count >= len(seeds)
    arguments = [
        (
            candidate,
            seed,
            features,
            target,
            target_scale,
            bias_baseline,
            observations,
            weights,
            train_indices,
            validation_indices,
            output,
            smoke,
            f"cuda:{index if parallel else 0}" if gpu_count else "cpu",
            lead_weights,
            inactive_lead_count,
            loss_coefficients,
        )
        for index, seed in enumerate(seeds)
    ]
    if parallel:
        print(f"  running {len(seeds)} seeds on {len(seeds)} GPUs", flush=True)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(seeds), mp_context=mp.get_context("spawn")
        ) as executor:
            results = list(executor.map(_train_seed_worker_from_tuple, arguments))
    else:
        results = [_train_seed_worker_from_tuple(values) for values in arguments]

    validation_members = [values[0] for values in results]
    records = [values[1] for values in results]
    parameter_counts = {int(values[2]) for values in results}
    if len(parameter_counts) != 1:
        raise RuntimeError("parameter count changed between seeds")
    parameter_count = parameter_counts.pop()
    devices = sorted({str(record["device"]) for record in records})
    ensemble = np.mean(validation_members, axis=0, dtype=np.float64).astype(np.float32)
    return ensemble, {
        **asdict(candidate),
        "devices": devices,
        "parallel_seeds": parallel,
        "parameter_count": parameter_count,
        "seeds": list(seeds),
        "train_case_count": len(train_indices),
        "validation_case_count": len(validation_indices),
        "max_epochs": 2 if smoke else 100,
        "patience": 1 if smoke else 15,
        "lead_weights": list(lead_weights),
        "inactive_lead_count": inactive_lead_count,
        "loss_coefficients": dict(loss_coefficients),
        "runs": records,
    }


def _train_seed_worker_from_tuple(arguments: tuple) -> tuple[np.ndarray, dict, int]:
    return _train_seed_worker(*arguments)


def _train_seed_worker(
    candidate: Candidate,
    seed: int,
    features: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    weights: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output: Path,
    smoke: bool,
    device: str,
    lead_weights: tuple[float, ...],
    inactive_lead_count: int,
    loss_coefficients: Mapping[str, float],
) -> tuple[np.ndarray, dict, int]:
    print(f"  {candidate.label}, seed {seed}, {device}", flush=True)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    torch.set_num_threads(max(1, allocated_cpus // max(len(SEEDS), 1)))
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    support = weights > 0.0
    train_data = common.make_dataset(
        train_indices, features, target, bias_baseline, observations, support
    )
    validation_data = common.make_dataset(
        validation_indices, features, target, bias_baseline, observations, support
    )
    set_deterministic_seed(seed)
    model = build_candidate(candidate, features.shape[2])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_directory = output / "models" / candidate.name / f"seed_{seed}"
    (run_directory / "logs").mkdir(parents=True, exist_ok=True)
    (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    result = train_anchored_model(
        model,
        train_data,
        validation_data,
        weights,
        target_scale,
        lead_weights,
        loss_coefficients,
        run_directory,
        seed=seed,
        device=device,
        batch_size=candidate.batch_size,
        max_epochs=2 if smoke else 100,
        patience=1 if smoke else 15,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        smooth_l1_beta=1.0,
        num_workers=0,
        use_amp=True,
    )
    residual = predict(
        model,
        features[validation_indices],
        device=device,
        batch_size=32,
        use_amp=True,
    )
    if not np.array_equal(
        residual[:, :inactive_lead_count],
        np.zeros_like(residual[:, :inactive_lead_count]),
    ):
        raise base.DataContractError(
            f"{candidate.name} changed an inactive lead before W{inactive_lead_count + 1}"
        )
    checkpoint = run_directory / "checkpoints" / "best.pt"
    record = {
        "seed": seed,
        "device": device,
        "best_epoch": int(result.best_epoch),
        "best_validation_loss": float(result.best_validation_loss),
        "elapsed_seconds": float(result.elapsed_seconds),
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str(
            (run_directory / "logs" / "training_history.csv").relative_to(output)
        ),
    }
    return residual, record, parameter_count


def predict_candidate(
    candidate: Candidate,
    training: Mapping[str, object],
    features: np.ndarray,
    indices: np.ndarray,
    output: Path,
    *,
    inactive_lead_count: int = 4,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    members = []
    for record in training["runs"]:
        model = build_candidate(candidate, features.shape[2])
        checkpoint_path = output / str(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        residual = predict(
            model,
            features[indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        if not np.array_equal(
            residual[:, :inactive_lead_count],
            np.zeros_like(residual[:, :inactive_lead_count]),
        ):
            raise base.DataContractError(
                f"{candidate.name} changed an inactive lead before "
                f"W{inactive_lead_count + 1}"
            )
        members.append(residual)
    return np.mean(members, axis=0, dtype=np.float64).astype(np.float32)


def composite_score(
    residual: np.ndarray,
    indices: np.ndarray,
    target: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    target_scale: np.ndarray,
    weights: np.ndarray,
    *,
    lead_weights: tuple[float, ...] = LEAD_WEIGHTS,
    loss_coefficients: Mapping[str, float] = LOSS_COEFFICIENTS,
) -> Mapping[str, float]:
    shape = residual.shape
    valid = np.broadcast_to((weights > 0.0)[None, None], shape).copy()
    with torch.no_grad():
        total, components = anchored_composite_loss(
            torch.from_numpy(residual),
            torch.from_numpy(target[indices]),
            torch.from_numpy(bias_baseline[indices]),
            torch.from_numpy(observations.weekly_truth[indices]),
            torch.from_numpy(observations.weekly_climatology[indices]),
            torch.from_numpy(target_scale),
            torch.from_numpy(weights.astype(np.float32)),
            torch.tensor(lead_weights, dtype=torch.float32),
            valid_mask=torch.from_numpy(valid),
            smooth_l1_coefficient=loss_coefficients["smooth_l1"],
            acc_coefficient=loss_coefficients["acc"],
            bias_coefficient=loss_coefficients["bias"],
            smooth_l1_beta=1.0,
            return_components=True,
        )
    return {
        "composite_loss": float(total),
        **{name: float(value) for name, value in components.items()},
    }


def case_rmse(
    prediction: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    support = weights > 0.0
    error = prediction[..., support].astype(np.float64) - truth[..., support]
    normalized = weights[support] / weights[support].sum()
    return np.sqrt(np.sum(error**2 * normalized[None, None], axis=-1))


def scan_validation_alpha(
    residuals: Mapping[str, np.ndarray],
    indices: np.ndarray,
    target: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    target_scale: np.ndarray,
    weights: np.ndarray,
    *,
    active_leads: tuple[int, ...] = ACTIVE_LEADS,
    lead_weights: tuple[float, ...] = LEAD_WEIGHTS,
    loss_coefficients: Mapping[str, float] = LOSS_COEFFICIENTS,
    score_column: str = "w5_w6_case_mean_rmse",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = weights > 0.0
    truth = observations.weekly_truth[indices]
    rows = []
    for name, residual in residuals.items():
        for alpha in ALPHA_GRID:
            scaled = (float(alpha) * residual).astype(np.float32)
            prediction = common.reconstruct(
                bias_baseline[indices], scaled, target_scale, support
            )
            rmses = case_rmse(prediction, truth, weights)[:, active_leads]
            score = composite_score(
                scaled,
                indices,
                target,
                bias_baseline,
                observations,
                target_scale,
                weights,
                lead_weights=lead_weights,
                loss_coefficients=loss_coefficients,
            )
            rows.append(
                {
                    "candidate": name,
                    "alpha": float(alpha),
                    score_column: float(rmses.mean()),
                    **score,
                }
            )
    scan = pd.DataFrame(rows)
    selected_rows = []
    for name in residuals:
        subset = scan.loc[scan.candidate.eq(name)]
        selected_rows.append(subset.loc[subset[score_column].idxmin()])
    return scan, pd.DataFrame(selected_rows).reset_index(drop=True)


def add_year_validation_scores(
    selection: pd.DataFrame,
    residuals: Mapping[str, np.ndarray],
    indices: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    target_scale: np.ndarray,
    weights: np.ndarray,
    initializations: np.ndarray,
    *,
    active_leads: tuple[int, ...] = ACTIVE_LEADS,
    baseline_name: str = "log_bias",
) -> pd.DataFrame:
    if not baseline_name or not baseline_name.replace("_", "").isalnum():
        raise ValueError("baseline_name must be a nonempty identifier")
    support = weights > 0.0
    truth = observations.weekly_truth[indices]
    years = pd.DatetimeIndex(initializations[indices]).year.to_numpy()
    baseline_rmse = case_rmse(bias_baseline[indices], truth, weights)[:, active_leads]
    result = selection.copy()
    for year in sorted(np.unique(years)):
        year_mask = years == year
        baseline_score = float(baseline_rmse[year_mask].mean())
        result[f"{baseline_name}_rmse_{year}"] = baseline_score
        values = []
        skills = []
        for row in result.itertuples(index=False):
            prediction = common.reconstruct(
                bias_baseline[indices],
                (float(row.alpha) * residuals[row.candidate]).astype(np.float32),
                target_scale,
                support,
            )
            model_score = float(
                case_rmse(prediction, truth, weights)[year_mask][:, active_leads].mean()
            )
            values.append(model_score)
            skills.append(100.0 * (baseline_score - model_score) / baseline_score)
        result[f"model_rmse_{year}"] = values
        result[f"rmse_skill_vs_{baseline_name}_{year}_pct"] = skills
    skill_columns = [
        column
        for column in result
        if column.startswith(f"rmse_skill_vs_{baseline_name}_")
        and column.endswith("_pct")
    ]
    result["improves_every_validation_year"] = result[skill_columns].gt(0.0).all(axis=1)
    return result


def choose_model(selection: pd.DataFrame) -> tuple[str, float, str]:
    robust = selection.loc[selection.improves_every_validation_year]
    if robust.empty:
        return (
            "log_bias",
            0.0,
            "No neural candidate reduced W5--W6 RMSE in every validation year.",
        )
    chosen = robust.loc[robust.w5_w6_case_mean_rmse.idxmin()]
    return (
        str(chosen.candidate),
        float(chosen.alpha),
        "Lowest pooled W5--W6 RMSE among candidates improving both 2018 and 2019.",
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    diagnostics.save_figure(figure, stem)


def tidy_training_history(
    output: Path, training: Mapping[str, Mapping[str, object]]
) -> pd.DataFrame:
    rows = []
    components = (
        ("loss", "composite"),
        ("smooth_l1", "smooth_l1"),
        ("mean_spatial_acc", "spatial_acc"),
        ("mean_bias_squared", "bias_squared"),
    )
    for name, metadata in training.items():
        for record in metadata["runs"]:
            history = pd.read_csv(output / str(record["history"]))
            for values in history.itertuples(index=False):
                for source, label in components:
                    for split in ("train", "validation"):
                        column = f"{split}_{source}"
                        rows.append(
                            {
                                "candidate": name,
                                "seed": int(record["seed"]),
                                "epoch": int(values.epoch) + 1,
                                "split": split,
                                "component": label,
                                "value": float(getattr(values, column)),
                                "selected_checkpoint": int(values.epoch)
                                == int(record["best_epoch"]),
                            }
                        )
    return pd.DataFrame(rows)


def plot_training_components(
    tidy: pd.DataFrame,
    training: Mapping[str, Mapping[str, object]],
    output: Path,
) -> None:
    panels = (
        ("smooth_l1", "Smooth-L1", "lower is better"),
        ("spatial_acc", "Spatial ACC", "higher is better"),
        ("bias_squared", "Bias penalty$^2$", "lower is better"),
        ("composite", "Composite objective", "lower is better"),
    )
    figure, axes = plt.subplots(
        len(CANDIDATES), len(panels), figsize=(15.0, 10.0), sharey="col"
    )
    split_colors = {"train": "#0072B2", "validation": "#D55E00"}
    for row_index, candidate in enumerate(CANDIDATES):
        for column_index, (component, title, direction) in enumerate(panels):
            axis = axes[row_index, column_index]
            selected = tidy.loc[
                tidy.candidate.eq(candidate.name) & tidy.component.eq(component)
            ]
            for split in ("train", "validation"):
                split_data = selected.loc[selected.split.eq(split)]
                for _, seed_data in split_data.groupby("seed"):
                    axis.plot(
                        seed_data.epoch,
                        seed_data.value,
                        color=split_colors[split],
                        alpha=0.22,
                        linewidth=0.8,
                    )
                median = split_data.groupby("epoch", as_index=False).value.median()
                axis.plot(
                    median.epoch,
                    median.value,
                    color=split_colors[split],
                    linewidth=2.0,
                    label=(
                        split.title() if row_index == 0 and column_index == 0 else None
                    ),
                )
            if component == "composite":
                records = training[candidate.name]["runs"]
                for record in records:
                    axis.scatter(
                        int(record["best_epoch"]) + 1,
                        float(record["best_validation_loss"]),
                        marker="*",
                        color="black",
                        s=45,
                        zorder=5,
                    )
            if row_index == 0:
                axis.set_title(f"{title}\n{direction}", fontsize=10.5)
            if column_index == 0:
                axis.set_ylabel(candidate.label)
            if row_index == len(CANDIDATES) - 1:
                axis.set_xlabel("Epoch")
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        handles=[
            plt.Line2D([], [], color=split_colors["train"], linewidth=2, label="Train"),
            plt.Line2D(
                [],
                [],
                color=split_colors["validation"],
                linewidth=2,
                label="Validation",
            ),
            plt.Line2D(
                [], [], color="black", marker="*", linestyle="", label="Checkpoint"
            ),
        ],
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
    )
    figure.suptitle(
        "Where the generalization gap comes from\n"
        "2002–2017 train · 2018–2019 validation · three fixed seeds",
        y=0.995,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(figure, output)


def plot_validation_selection(
    scan: pd.DataFrame,
    selected: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    for candidate in CANDIDATES:
        values = scan.loc[scan.candidate.eq(candidate.name)].sort_values("alpha")
        chosen = selected.loc[selected.candidate.eq(candidate.name)].iloc[0]
        axes[0].plot(
            values.alpha,
            values.w5_w6_case_mean_rmse,
            color=candidate.color,
            label=candidate.label,
        )
        axes[0].scatter(
            chosen.alpha,
            chosen.w5_w6_case_mean_rmse,
            color=candidate.color,
            edgecolor="black",
            zorder=4,
        )
        axes[1].plot(
            values.alpha,
            values.composite_loss,
            color=candidate.color,
        )
        axes[1].scatter(
            chosen.alpha,
            chosen.composite_loss,
            color=candidate.color,
            edgecolor="black",
            zorder=4,
        )
    axes[0].set_ylabel("Validation RMSE (mm day$^{-1}$)")
    axes[0].set_title("Primary: Weeks 5–6 RMSE", fontweight="semibold")
    axes[1].set_ylabel("Composite objective")
    axes[1].set_title("Loss at the same residual gate", fontweight="semibold")
    for axis in axes[:2]:
        axis.set_xlabel("Residual gate α (0 = log-bias, 1 = full model)")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    years = sorted(
        int(column.removeprefix("rmse_skill_vs_log_bias_").removesuffix("_pct"))
        for column in selected
        if column.startswith("rmse_skill_vs_log_bias_") and column.endswith("_pct")
    )
    positions = np.arange(len(CANDIDATES), dtype=float)
    width = 0.34 if len(years) == 2 else 0.7 / max(len(years), 1)
    offsets = (np.arange(len(years)) - (len(years) - 1) / 2.0) * width
    for offset, year in zip(offsets, years):
        values = [
            float(
                selected.loc[
                    selected.candidate.eq(candidate.name),
                    f"rmse_skill_vs_log_bias_{year}_pct",
                ].iloc[0]
            )
            for candidate in CANDIDATES
        ]
        axes[2].bar(positions + offset, values, width=width, label=str(year))
    axes[2].axhline(0.0, color="0.4", linestyle="--", linewidth=0.9)
    axes[2].set_xticks(
        positions, ["Small\ntemporal", "Large\nspatial", "Large\ntemporal"]
    )
    axes[2].set_ylabel("RMSE skill vs log-bias (%)")
    axes[2].set_title("Must improve each validation year", fontweight="semibold")
    axes[2].legend(frameon=False, title="Validation year")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
    )
    figure.suptitle(
        "Validation-only model selection\n"
        "A single shrinkage gate controls over-correction",
        y=1.02,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(figure, output)


def late_headline(case_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = case_metrics.loc[case_metrics.lead_week.isin((5, 6))]
    table = selected.groupby("method", as_index=False).agg(
        cases=("init", "nunique"),
        acc=("acc", "mean"),
        rmse_mm_day=("rmse", "mean"),
        mae_mm_day=("mae", "mean"),
        bias_mm_day=("bias", "mean"),
    )
    raw = table.loc[table.method.eq("raw_fuxi")].iloc[0]
    anchor = table.loc[table.method.eq("log_bias")].iloc[0]
    table["rmse_skill_vs_raw_pct"] = (
        100.0 * (raw.rmse_mm_day - table.rmse_mm_day) / raw.rmse_mm_day
    )
    table["rmse_skill_vs_log_bias_pct"] = (
        100.0 * (anchor.rmse_mm_day - table.rmse_mm_day) / anchor.rmse_mm_day
    )
    table["method_label"] = table.method.map(METHOD_LABELS)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    table["_order"] = table.method.map(order)
    return table.sort_values("_order").drop(columns="_order")


def build_spatial_reductions(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    weights: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[xr.Dataset, pd.DataFrame]:
    support = weights > 0.0
    pairs = (
        ("big_temporal_vs_log_bias", "big_temporal", "log_bias"),
        ("big_temporal_vs_small", "big_temporal", "small_temporal"),
        ("selected_vs_log_bias", "selected_model", "log_bias"),
    )
    method_rmse = {}
    for method in {
        value for _, method, baseline in pairs for value in (method, baseline)
    }:
        error = predictions[method][:, ACTIVE_LEADS] - truth[:, ACTIVE_LEADS]
        field = np.sqrt(np.mean(error.astype(np.float64) ** 2, axis=(0, 1)))
        field[~support] = np.nan
        method_rmse[method] = field
    reductions = np.stack(
        [method_rmse[baseline] - method_rmse[method] for _, method, baseline in pairs]
    )
    dataset = xr.Dataset(
        {
            "rmse_reduction": (
                ("comparison", "latitude", "longitude"),
                reductions.astype(np.float32),
            ),
            "area_weight_km2": (("latitude", "longitude"), weights),
        },
        coords={
            "comparison": [name for name, _, _ in pairs],
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "lead_scope": "W5-W6",
            "positive_reduction": "first method has lower local RMSE",
            "map_status": "descriptive; no pixel-wise significance inference",
        },
    )
    dataset.rmse_reduction.attrs["units"] = "mm day-1"
    rows = []
    for index, (name, method, baseline) in enumerate(pairs):
        values = reductions[index]
        improved = support & (values > 0.0)
        rows.append(
            {
                "comparison": name,
                "method": method,
                "baseline": baseline,
                "area_fraction_improved_pct": float(
                    100.0 * weights[improved].sum() / weights[support].sum()
                ),
                "area_weighted_mean_rmse_reduction_mm_day": float(
                    np.sum(weights[support] * values[support]) / weights[support].sum()
                ),
            }
        )
    return dataset, pd.DataFrame(rows)


def plot_spatial_reductions(
    dataset: xr.Dataset, summary: pd.DataFrame, output: Path
) -> None:
    fields = np.asarray(dataset.rmse_reduction.values, dtype=np.float64)
    finite = np.abs(fields[np.isfinite(fields)])
    limit = max(0.1, float(np.ceil(np.percentile(finite, 98) / 0.1) * 0.1))
    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1, 3, figsize=(12.2, 4.3), subplot_kw={"projection": projection}
    )
    titles = (
        "Large temporal vs log-bias",
        "Large temporal vs small temporal",
        "Selected model vs log-bias",
    )
    image = None
    for index, (axis, title) in enumerate(zip(axes, titles)):
        image = axis.pcolormesh(
            dataset.longitude,
            dataset.latitude,
            np.ma.masked_invalid(fields[index]),
            transform=projection,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
        )
        axis.coastlines(resolution="50m", linewidth=0.65, color="0.2")
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.25"
        )
        axis.set_extent(
            [
                float(dataset.longitude.min()) - 0.8,
                float(dataset.longitude.max()) + 0.8,
                float(dataset.latitude.min()) - 0.8,
                float(dataset.latitude.max()) + 0.8,
            ],
            crs=projection,
        )
        grid = axis.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="0.6",
            alpha=0.5,
            x_inline=False,
            y_inline=False,
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.left_labels = index == 0
        axis.set_title(title, fontsize=10.2, fontweight="semibold")
        row = summary.iloc[index]
        axis.text(
            0.98,
            0.03,
            f"{row.area_fraction_improved_pct:.1f}% area improved",
            transform=axis.transAxes,
            ha="right",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
    assert image is not None
    color_axis = figure.add_axes([0.27, 0.15, 0.46, 0.035])
    colorbar = figure.colorbar(
        image, cax=color_axis, orientation="horizontal", extend="both"
    )
    colorbar.set_label("Local RMSE reduction (mm day$^{-1}$; positive is better)")
    figure.suptitle(
        "Where post-processing changes late-lead skill\n"
        "Weeks 5–6 · exploratory 2020–2021 test",
        y=0.98,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.025,
        "Pooled point estimates over 70 initializations; no cell-wise significance claim.",
        ha="center",
        fontsize=8.4,
        color="0.35",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.27, wspace=0.08)
    save_figure(figure, output)


def write_prediction_store(
    path: Path,
    forecast: base.ForecastData,
    observations: base.ObservationData,
    indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    selection: Mapping[str, object],
    *,
    smoke: bool,
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack([predictions[name] for name in METHOD_ORDER]).astype(
                    np.float32
                ),
            ),
            "truth_imerg": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_truth[indices].astype(np.float32),
            ),
            "imerg_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_climatology[indices].astype(np.float32),
            ),
            "area_weight_km2": (("latitude", "longitude"), weights.astype(np.float64)),
        },
        coords={
            "method": list(METHOD_ORDER),
            "init": forecast.initializations[indices].astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": "A100 large FuXi-to-IMERG late-lead experiment",
            "train_years": "2002-2017",
            "validation_years": "2018-2019",
            "test_years": "2020-2021",
            "test_status": "exploratory; this period had already been examined",
            "selection_scope": selection["selection_scope"],
            "selected_model": selection["selected_model"],
            "selected_alpha": float(selection["selected_alpha"]),
            "early_lead_contract": "all learned methods equal log-bias at W1-W4",
            "units": "mm day-1",
            "smoke": smoke,
        },
    )
    chunk_cases = min(35, len(indices))
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={
            "prediction": {"chunks": (1, chunk_cases, 1, 27, 27)},
            "truth_imerg": {"chunks": (chunk_cases, 1, 27, 27)},
            "imerg_climatology": {"chunks": (chunk_cases, 1, 27, 27)},
            "area_weight_km2": {"chunks": (27, 27)},
        },
    )


def write_results(
    output: Path,
    training: Mapping[str, Mapping[str, object]],
    validation_selection: pd.DataFrame,
    selection: Mapping[str, object],
    late: pd.DataFrame,
    *,
    smoke: bool,
) -> None:
    lines = [
        "# A100 large temporal FuXi–IMERG experiment",
        "",
        (
            "> Smoke check only; do not interpret scores."
            if smoke
            else "> Exploratory test: 2020–2021 was viewed in earlier development."
        ),
        "",
        "## Why the earlier validation loss looked high",
        "",
        "The earlier temporal checkpoint loss (~0.562) was a composite, not RMSE: "
        "about 0.378 came from weighted Smooth-L1, 0.178 from 1−ACC, and 0.005 "
        "from the bias penalty. Its train–validation gap was real, so capacity alone "
        "was not the fix.",
        "",
        "## Protocol",
        "",
        "- Train: 2002–2017 (560 initializations)",
        "- Validation: 2018–2019 (70 initializations; two seasonal blocks)",
        "- Exploratory test: 2020–2021 (70 initializations)",
        "- Corrections: Weeks 5–6 only; Weeks 1–4 exactly equal log-bias",
        "- Fixed ensemble: seeds 42, 43, and 44; no best-seed selection",
        "- Primary selection: validation W5–W6 RMSE, with improvement required in both years",
        "- Residual gate: one validation-fitted scalar α per candidate, constrained to [0, 1]",
        "",
        "## Model capacity",
        "",
        "| Candidate | Parameters | Median best epoch | Mean best validation loss |",
        "|---|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        metadata = training[candidate.name]
        records = metadata["runs"]
        lines.append(
            f"| {candidate.label} | {metadata['parameter_count']:,} | "
            f"{np.median([record['best_epoch'] + 1 for record in records]):.0f} | "
            f"{np.mean([record['best_validation_loss'] for record in records]):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation decision",
            "",
            f"Selected: **{METHOD_LABELS[str(selection['selected_model'])]}** "
            f"with α={float(selection['selected_alpha']):.3f}.",
            "",
            str(selection["selection_reason"]),
            "",
            "| Candidate | α | W5–W6 RMSE | Composite loss | Better in every year? |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in validation_selection.itertuples(index=False):
        lines.append(
            f"| {METHOD_LABELS[row.candidate]} | {row.alpha:.3f} | "
            f"{row.w5_w6_case_mean_rmse:.3f} | {row.composite_loss:.4f} | "
            f"{'yes' if row.improves_every_validation_year else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory test — Weeks 5–6",
            "",
            "| Method | ACC | RMSE | MAE | Bias | RMSE skill vs raw | RMSE skill vs log-bias |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in late.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.acc:.3f} | {row.rmse_mm_day:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:+.3f} | "
            f"{row.rmse_skill_vs_raw_pct:+.2f}% | "
            f"{row.rmse_skill_vs_log_bias_pct:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "The large temporal model is a deterministic post-processor; FuXi itself "
            "was not retrained. Use paired confidence intervals in "
            "`metrics/paired_skill.csv` before making a significance claim.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="short A100 integration check"
    )
    args = parser.parse_args()
    configure_contract()
    started = time.monotonic()
    prefix = "smoke" if args.smoke else "full"
    output = RESULTS_ROOT / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    for directory in (
        output,
        output / "models",
        output / "metrics",
        output / "figures",
        output / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    print("Loading FuXi 2002–2021...", flush=True)
    forecast = base.load_fuxi()
    print("Loading IMERG and fitting the 2002–2017 climatology...", flush=True)
    observations = base.load_imerg(forecast)
    weights = base.load_area_weights(forecast, observations.observation_fraction)
    splits = base.split_indices(forecast.initializations)
    counts = {name: len(indices) for name, indices in splits.items()}
    if counts != {"train": 560, "validation": 70, "test": 70}:
        raise base.DataContractError(f"unexpected split counts: {counts}")
    support = weights > 0.0

    print("Loading FuXi T2M and building eleven-channel arrays...", flush=True)
    t2m_weekly = common.load_t2m_weekly(forecast)
    features, normalization = common.make_features(
        forecast, observations, weights, splits["train"], t2m_weekly
    )
    (output / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
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
    valid = np.broadcast_to(support[None, None], bias_baseline.shape)
    target_scale = fit_anchored_target_scale(
        observations.weekly_truth[splits["train"]],
        bias_baseline[splits["train"]],
        weights,
        split_name="train",
        valid_mask=valid[splits["train"]],
    )
    target = standardize_anchored_target(
        observations.weekly_truth,
        bias_baseline,
        target_scale,
        valid_mask=valid,
    )
    np.savez_compressed(
        output / "models" / "log_bias_anchor.npz",
        lead_month_residual=correction.lead_month_residual,
        shrinkage=np.float32(correction.shrinkage),
        target_scale=target_scale,
    )

    validation_indices = (
        splits["validation"][:16] if args.smoke else splits["validation"]
    )
    training = {}
    validation_residuals = {}
    for candidate in CANDIDATES:
        print(f"Training {candidate.label}...", flush=True)
        residual, metadata = train_candidate(
            candidate,
            features,
            target,
            target_scale,
            bias_baseline,
            observations,
            weights,
            splits,
            output,
            smoke=args.smoke,
        )
        validation_residuals[candidate.name] = residual
        training[candidate.name] = metadata

    print("Selecting residual gates using validation only...", flush=True)
    alpha_scan, validation_selection = scan_validation_alpha(
        validation_residuals,
        validation_indices,
        target,
        bias_baseline,
        observations,
        target_scale,
        weights,
    )
    validation_selection = add_year_validation_scores(
        validation_selection,
        validation_residuals,
        validation_indices,
        bias_baseline,
        observations,
        target_scale,
        weights,
        forecast.initializations,
    )
    selected_model, selected_alpha, selection_reason = choose_model(
        validation_selection
    )
    best_neural_row = validation_selection.loc[
        validation_selection.w5_w6_case_mean_rmse.idxmin()
    ]
    frozen_utc = utc_now()
    selection_record = {
        "status": "frozen",
        "smoke": args.smoke,
        "selection_scope": "validation_only",
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "test_years_quarantined_during_selection": list(TEST_YEARS),
        "primary_metric": "equal-case W5-W6 area-weighted RMSE",
        "robustness_rule": "positive RMSE skill vs log-bias in every validation year",
        "candidate_set": [candidate.name for candidate in CANDIDATES],
        "fixed_seeds": list(SEEDS[:1] if args.smoke else SEEDS),
        "selected_model": selected_model,
        "selected_alpha": selected_alpha,
        "selection_reason": selection_reason,
        "best_neural_candidate": str(best_neural_row.candidate),
        "best_neural_alpha": float(best_neural_row.alpha),
        "frozen_utc": frozen_utc,
        "test_predictions_created": False,
        "checkpoint_sha256": {
            candidate.name: [
                record["checkpoint_sha256"]
                for record in training[candidate.name]["runs"]
            ]
            for candidate in CANDIDATES
        },
    }
    selection_path = output / "selection.json"
    selection_path.write_text(
        json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
    )
    selection_hash = sha256_file(selection_path)
    alpha_scan.to_csv(output / "metrics" / "validation_alpha_scan.csv", index=False)
    validation_selection.to_csv(
        output / "metrics" / "validation_selection.csv", index=False
    )

    gated_validation = {}
    for candidate in CANDIDATES:
        alpha = float(
            validation_selection.loc[
                validation_selection.candidate.eq(candidate.name), "alpha"
            ].iloc[0]
        )
        gated_validation[candidate.name] = common.reconstruct(
            bias_baseline[validation_indices],
            (alpha * validation_residuals[candidate.name]).astype(np.float32),
            target_scale,
            support,
        )
    raw_validation = forecast.ensemble_mean[validation_indices].copy()
    raw_validation[..., ~support] = np.nan
    log_validation = bias_baseline[validation_indices].copy()
    log_validation[..., ~support] = np.nan
    selected_validation = (
        log_validation
        if selected_model == "log_bias"
        else gated_validation[selected_model]
    )
    validation_predictions = {
        "raw_fuxi": raw_validation,
        "log_bias": log_validation,
        **gated_validation,
        "selected_model": selected_validation,
    }
    validation_metrics = diagnostics.evaluate_predictions(
        observations.weekly_truth[validation_indices],
        observations.weekly_climatology[validation_indices],
        validation_predictions,
        forecast.initializations[validation_indices],
        weights,
    )
    validation_metrics["split"] = "validation"
    validation_metrics.to_csv(
        output / "metrics" / "validation_case_metrics.csv", index=False
    )

    tidy = tidy_training_history(output, training)
    tidy.to_csv(output / "metrics" / "training_history_tidy.csv", index=False)
    plot_training_components(
        tidy, training, output / "figures" / "00_training_components"
    )
    plot_validation_selection(
        alpha_scan,
        validation_selection,
        output / "figures" / "01_validation_selection",
    )

    # This timestamp is recorded only after the immutable validation selection exists.
    test_evaluation_started_utc = utc_now()
    if sha256_file(selection_path) != selection_hash:
        raise RuntimeError("selection.json changed before test evaluation")
    print(
        f"Selection frozen: {selected_model}, alpha={selected_alpha:.3f}. "
        "Now creating exploratory test predictions...",
        flush=True,
    )
    test_indices = splits["test"][:8] if args.smoke else splits["test"]
    test_predictions = {}
    for candidate in CANDIDATES:
        residual = predict_candidate(
            candidate, training[candidate.name], features, test_indices, output
        )
        alpha = float(
            validation_selection.loc[
                validation_selection.candidate.eq(candidate.name), "alpha"
            ].iloc[0]
        )
        test_predictions[candidate.name] = common.reconstruct(
            bias_baseline[test_indices],
            (alpha * residual).astype(np.float32),
            target_scale,
            support,
        )
    raw_test = forecast.ensemble_mean[test_indices].copy()
    raw_test[..., ~support] = np.nan
    log_test = bias_baseline[test_indices].copy()
    log_test[..., ~support] = np.nan
    selected_test = (
        log_test if selected_model == "log_bias" else test_predictions[selected_model]
    )
    predictions = {
        "raw_fuxi": raw_test,
        "log_bias": log_test,
        **test_predictions,
        "selected_model": selected_test,
    }
    for method in METHOD_ORDER[2:]:
        if not np.array_equal(
            predictions[method][:, :4], log_test[:, :4], equal_nan=True
        ):
            raise base.DataContractError(f"{method} differs from log-bias at W1--W4")

    truth = observations.weekly_truth[test_indices]
    climatology = observations.weekly_climatology[test_indices]
    initializations = forecast.initializations[test_indices]
    case_metrics = diagnostics.evaluate_predictions(
        truth, climatology, predictions, initializations, weights
    )
    summary = diagnostics.summarize_by_lead(case_metrics)
    intervals = diagnostics.paired_intervals(
        case_metrics, initializations, smoke=args.smoke
    )
    headline = diagnostics.headline_table(case_metrics, intervals)
    late = late_headline(case_metrics)
    case_metrics.to_csv(output / "metrics" / "case_metrics.csv", index=False)
    summary.to_csv(output / "metrics" / "summary_by_lead.csv", index=False)
    intervals.to_csv(output / "metrics" / "paired_skill.csv", index=False)
    headline.to_csv(output / "metrics" / "headline_metrics.csv", index=False)
    late.to_csv(output / "metrics" / "late_headline_metrics.csv", index=False)

    spatial_dataset, spatial_summary = build_spatial_reductions(
        predictions, truth, weights, forecast.latitude, forecast.longitude
    )
    spatial_dataset.to_netcdf(output / "metrics" / "late_spatial_rmse.nc")
    spatial_summary.to_csv(output / "metrics" / "late_spatial_summary.csv", index=False)
    diagnostics.plot_skill_by_lead(
        summary, output / "figures" / "02_test_skill_by_lead", smoke=args.smoke
    )
    plot_spatial_reductions(
        spatial_dataset,
        spatial_summary,
        output / "figures" / "03_test_spatial_improvement",
    )

    prediction_store = output / "predictions.zarr"
    write_prediction_store(
        prediction_store,
        forecast,
        observations,
        test_indices,
        predictions,
        weights,
        selection_record,
        smoke=args.smoke,
    )
    with xr.open_zarr(prediction_store, consolidated=True) as stored:
        for method in METHOD_ORDER:
            rebuilt = np.asarray(stored.prediction.sel({"method": method}).load())
            if not np.array_equal(rebuilt, predictions[method], equal_nan=True):
                raise base.DataContractError(f"Zarr round-trip failed for {method}")

    write_results(
        output,
        training,
        validation_selection,
        selection_record,
        late,
        smoke=args.smoke,
    )
    sources = (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imerg_full_archive_latelead.py",
        NEURAL_SRC / "fuxi_adapter" / "models.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
    )
    code_hashes = {}
    for source in sources:
        destination = output / "code" / source.name
        shutil.copy2(source, destination)
        code_hashes[source.name] = sha256_file(destination)

    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "smoke": args.smoke,
        "created_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "scientific_status": "exploratory; reused 2020-2021 test",
        "split_years": {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": list(TEST_YEARS),
        },
        "split_counts": counts,
        "test_count_used": len(test_indices),
        "active_leads": [5, 6],
        "inactive_lead_identity_verified": True,
        "support_cells": int(support.sum()),
        "loss_coefficients": LOSS_COEFFICIENTS,
        "lead_weights": list(LEAD_WEIGHTS),
        "selection_file": "selection.json",
        "selection_sha256": selection_hash,
        "selection_frozen_utc": frozen_utc,
        "test_evaluation_started_utc": test_evaluation_started_utc,
        "selected_model": selected_model,
        "selected_alpha": selected_alpha,
        "training": training,
        "features": normalization["input_channels"],
        "prediction_store_roundtrip_verified": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code_sha256": code_hashes,
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "artifacts": {},
    }
    for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
        if prediction_store in artifact.parents:
            continue
        manifest["artifacts"][str(artifact.relative_to(output))] = sha256_file(artifact)
    manifest["artifacts"]["predictions.zarr"] = sha256_tree(prediction_store)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print("\n" + late.to_string(index=False), flush=True)
    print(
        "\nPASS: validation frozen before test, W1-W4 identity, and Zarr round-trip",
        flush=True,
    )
    print(f"Completed in {elapsed / 60.0:.1f} minutes", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
