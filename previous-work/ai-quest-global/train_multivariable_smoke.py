"""Optimize the joint temporal model on deterministic synthetic Quest fields.

This executable verifies that the zero-anchor model can learn a fixed rule on
independent synthetic train/validation cases.  It writes real optimization
curves, but the results are not evidence of meteorological forecast skill.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import QuestTemporalProbUNet
from multivariable import (
    MULTIVARIABLE_FEATURE_NAMES,
    MULTIVARIABLE_INPUT_CHANNELS,
    QUEST_VARIABLES,
    multivariable_rps_loss,
)
from predict import FUXI_PERMISSION_WARNING, load_checkpoint_model
from prototype_multivariable import synthetic_anchor, synthetic_features
from train import choose_device, optimizer_for, set_seed


SYNTHETIC_NOTICE = "Synthetic optimization smoke only — not scientific forecast skill"
CATEGORY_CUTS = np.asarray([-0.8, -0.25, 0.25, 0.8], dtype=np.float32)
QUANTILE_OFFSETS = np.asarray([-1.2, -0.5, 0.0, 0.5, 1.2], dtype=np.float32)
VARIABLE_LABELS = {"pr": "Precipitation", "tas": "Temperature", "mslp": "MSLP"}


@dataclass(frozen=True)
class SyntheticSplit:
    """One independent collection of complete synthetic forecast cases."""

    features: torch.Tensor
    anchor: torch.Tensor
    target: torch.Tensor

    @property
    def cases(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class SplitMetrics:
    """Exact whole-split scores after independent normalization by variable."""

    objective: float
    rps_mean: float
    rps_by_variable: tuple[float, float, float]
    modal_accuracy_mean: float
    modal_accuracy_by_variable: tuple[float, float, float]
    correction_rms: float


def synthetic_spatial_weights(height: int, width: int) -> torch.Tensor:
    """Return Quest-like ``[variable,period,lat,lon]`` synthetic weights."""

    if height < 4 or width < 4:
        raise ValueError("synthetic grid dimensions must be at least four")
    latitude = np.linspace(75.0, -75.0, height, dtype=np.float64)
    cosine = np.clip(np.cos(np.deg2rad(latitude)), 0.0, None)[:, None]
    rows, columns = np.indices((height, width))
    land = ((2 * rows + columns) % 5) != 0
    non_arid = ((rows + 3 * columns) % 7) != 0
    domains = np.stack((land & non_arid, land, np.ones_like(land)), axis=0)
    weights = domains[:, None] * cosine[None, None]
    weights = np.broadcast_to(weights, (3, 2, height, width)).copy()
    if np.any(weights.reshape(3, -1).sum(axis=1) <= 0.0):
        raise ValueError("synthetic domains must retain cells for every variable")
    return torch.from_numpy(weights.astype(np.float32))


def make_synthetic_split(
    cases: int,
    seed: int,
    *,
    height: int = 8,
    width: int = 12,
) -> SyntheticSplit:
    """Generate cases from one fixed rule without fitting anything to validation."""

    if cases < 1:
        raise ValueError("cases must be positive")
    if height < 4 or width < 4:
        raise ValueError("synthetic grid dimensions must be at least four")
    rng = np.random.default_rng(seed)
    features = rng.normal(
        0.0,
        0.03,
        size=(cases, 2, MULTIVARIABLE_INPUT_CHANNELS, height, width),
    ).astype(np.float32)
    anchors = np.empty((cases, 3, 2, 5, height, width), dtype=np.float32)
    targets = np.empty((cases, 3, 2, height, width), dtype=np.int64)
    latitude = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    longitude = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    land_fraction = (((2 * np.arange(height)[:, None] + np.arange(width)) % 5) != 0).astype(
        np.float32
    )
    feature_index = {
        name: index for index, name in enumerate(MULTIVARIABLE_FEATURE_NAMES)
    }

    case_phases = rng.uniform(-np.pi, np.pi, size=(cases, 3)).astype(np.float32)
    for case_index in range(cases):
        for variable_index, variable in enumerate(QUEST_VARIABLES):
            phase = float(case_phases[case_index, variable_index])
            for period in range(2):
                latent = (
                    np.sin(
                        (variable_index + 1) * np.pi * longitude
                        + phase
                        + 0.45 * period
                    )
                    + 0.65 * np.cos(np.pi * latitude - 0.3 * phase)
                    + 0.10 * rng.normal(size=(height, width))
                ).astype(np.float32)
                targets[case_index, variable_index, period] = np.digitize(
                    latent, CATEGORY_CUTS
                )
                quantile_start = feature_index[f"{variable}_q10"]
                features[
                    case_index,
                    period,
                    quantile_start : quantile_start + 5,
                ] = latent[None] + QUANTILE_OFFSETS[:, None, None]

                biased_proxy = np.digitize(
                    latent + 0.55 + 0.40 * rng.normal(size=(height, width)),
                    CATEGORY_CUTS,
                )
                anchor = np.full((5, height, width), 0.075, dtype=np.float32)
                for category in range(5):
                    anchor[category, biased_proxy == category] = 0.70
                anchors[case_index, variable_index, period] = anchor
                anchor_start = feature_index[f"{variable}_log_p_q1"]
                features[
                    case_index,
                    period,
                    anchor_start : anchor_start + 5,
                ] = np.log(anchor)

        features[case_index, :, feature_index["sin_lat"]] = np.sin(
            np.pi * latitude / 2.0
        )[None]
        features[case_index, :, feature_index["cos_lat"]] = np.cos(
            np.pi * latitude / 2.0
        )[None]
        features[case_index, :, feature_index["sin_lon"]] = np.sin(
            np.pi * longitude
        )[None]
        features[case_index, :, feature_index["cos_lon"]] = np.cos(
            np.pi * longitude
        )[None]
        issue_phase = float(case_phases[case_index].mean())
        features[case_index, :, feature_index["sin_doy"]] = np.sin(issue_phase)
        features[case_index, :, feature_index["cos_doy"]] = np.cos(issue_phase)
        features[case_index, :, feature_index["lead_flag"]] = np.asarray(
            [-1.0, 1.0], dtype=np.float32
        )[:, None, None]
        features[case_index, :, feature_index["land_fraction"]] = land_fraction

    if not np.isfinite(features).all() or not np.isfinite(anchors).all():
        raise RuntimeError("synthetic generator produced non-finite inputs")
    if not np.allclose(anchors.sum(axis=3), 1.0, atol=1.0e-7):
        raise RuntimeError("synthetic anchors do not sum to one")
    return SyntheticSplit(
        torch.from_numpy(features),
        torch.from_numpy(anchors),
        torch.from_numpy(targets),
    )


def _score_components(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    categories = torch.arange(5, device=target.device).view(1, 1, 1, 5, 1, 1)
    valid = (target >= 0) & (target <= 4)
    observed_cdf = (target.unsqueeze(3) <= categories).to(probabilities.dtype)
    cell_rps = ((probabilities.cumsum(dim=3) - observed_cdf) ** 2).sum(dim=3)
    effective = weights.to(probabilities.device).unsqueeze(0) * valid.to(
        probabilities.dtype
    )
    reduction = (0, 2, 3, 4)
    rps_numerator = (cell_rps * effective).sum(dim=reduction)
    denominator = effective.sum(dim=reduction)
    predicted = probabilities.argmax(dim=3)
    accuracy_numerator = (
        (predicted == target).to(probabilities.dtype) * effective
    ).sum(dim=reduction)
    if bool((denominator <= 0.0).any()):
        raise ValueError("a synthetic variable has no scoring cells")
    return rps_numerator, accuracy_numerator, denominator, cell_rps


def _metrics_from_tensors(
    probabilities: torch.Tensor,
    corrections: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    correction_penalty: float,
) -> SplitMetrics:
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("model probabilities are not finite")
    if not bool(
        torch.allclose(
            probabilities.sum(dim=3),
            torch.ones_like(probabilities[:, :, :, 0]),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    ):
        raise RuntimeError("model probabilities do not sum to one")
    rps_numerator, accuracy_numerator, denominator, _ = _score_components(
        probabilities, target, weights
    )
    rps = rps_numerator / denominator
    accuracy = accuracy_numerator / denominator
    correction_rms = corrections.square().mean().sqrt()
    objective = rps.mean() + float(correction_penalty) * corrections.square().mean()
    return SplitMetrics(
        objective=float(objective.detach().cpu()),
        rps_mean=float(rps.mean().detach().cpu()),
        rps_by_variable=tuple(float(value) for value in rps.detach().cpu()),  # type: ignore[arg-type]
        modal_accuracy_mean=float(accuracy.mean().detach().cpu()),
        modal_accuracy_by_variable=tuple(
            float(value) for value in accuracy.detach().cpu()
        ),  # type: ignore[arg-type]
        correction_rms=float(correction_rms.detach().cpu()),
    )


def evaluate_split(
    model: QuestTemporalProbUNet,
    split: SyntheticSplit,
    weights: torch.Tensor,
    device: torch.device,
    *,
    correction_penalty: float,
    return_probabilities: bool = False,
) -> tuple[SplitMetrics, np.ndarray | None]:
    """Evaluate a whole split at once so scores are aggregated exactly."""

    model.eval()
    with torch.inference_mode():
        features = split.features.to(device)
        anchor = split.anchor.to(device)
        target = split.target.to(device)
        corrections = model.forward_corrections(features)
        probabilities = torch.softmax(
            torch.log(anchor.clamp_min(1.0e-8)) + corrections,
            dim=3,
        )
        metrics = _metrics_from_tensors(
            probabilities,
            corrections,
            target,
            weights,
            correction_penalty,
        )
        output = probabilities.detach().cpu().numpy() if return_probabilities else None
    return metrics, output


def train_one_epoch(
    model: QuestTemporalProbUNet,
    split: SyntheticSplit,
    weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch_size: int,
    correction_penalty: float,
    gradient_clip: float,
    generator: torch.Generator,
) -> None:
    """Run one shuffled optimization pass over complete forecast cases."""

    model.train()
    permutation = torch.randperm(split.cases, generator=generator)
    for start in range(0, split.cases, batch_size):
        indices = permutation[start : start + batch_size]
        features = split.features[indices].to(device)
        anchor = split.anchor[indices].to(device)
        target = split.target[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        corrections = model.forward_corrections(features)
        probabilities = torch.softmax(
            torch.log(anchor.clamp_min(1.0e-8)) + corrections,
            dim=3,
        )
        loss = multivariable_rps_loss(probabilities, target, weights) + float(
            correction_penalty
        ) * corrections.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()


def _history_row(
    epoch: int,
    train_metrics: SplitMetrics,
    validation_metrics: SplitMetrics,
    validation_anchor: SplitMetrics,
    validation_uniform: SplitMetrics,
    learning_rate: float,
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": float(epoch),
        "train_objective": train_metrics.objective,
        "validation_objective": validation_metrics.objective,
        "train_rps_mean": train_metrics.rps_mean,
        "validation_rps_mean": validation_metrics.rps_mean,
        "train_modal_accuracy": train_metrics.modal_accuracy_mean,
        "validation_modal_accuracy": validation_metrics.modal_accuracy_mean,
        "validation_skill_vs_p0": 1.0
        - validation_metrics.rps_mean / validation_anchor.rps_mean,
        "validation_rpss_vs_uniform": 1.0
        - validation_metrics.rps_mean / validation_uniform.rps_mean,
        "learning_rate": float(learning_rate),
        "correction_rms": validation_metrics.correction_rms,
    }
    for index, variable in enumerate(QUEST_VARIABLES):
        row[f"train_rps_{variable}"] = train_metrics.rps_by_variable[index]
        row[f"validation_rps_{variable}"] = validation_metrics.rps_by_variable[index]
        row[f"validation_skill_vs_p0_{variable}"] = 1.0 - (
            validation_metrics.rps_by_variable[index]
            / validation_anchor.rps_by_variable[index]
        )
    return row


def write_history(history: list[dict[str, float]], path: Path) -> None:
    """Write the complete numeric evidence behind every learning curve."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(
    history: list[dict[str, float]],
    destination: Path,
    *,
    best_epoch: int,
    anchor_rps: float,
) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["train_objective"] for row in history], label="Train")
    axes[0, 0].plot(
        epochs, [row["validation_objective"] for row in history], label="Validation"
    )
    axes[0, 0].set(title="Objective", ylabel="RPS + correction penalty")

    axes[0, 1].plot(epochs, [row["train_rps_mean"] for row in history], label="Train")
    axes[0, 1].plot(
        epochs, [row["validation_rps_mean"] for row in history], label="Validation"
    )
    axes[0, 1].axhline(anchor_rps, color="0.35", linestyle="--", label="Validation p0")
    axes[0, 1].set(title="Equal-variable mean RPS", ylabel="RPS (lower is better)")

    colors = {"pr": "#2b83ba", "tas": "#d7191c", "mslp": "#7b3294"}
    for variable in QUEST_VARIABLES:
        axes[1, 0].plot(
            epochs,
            [row[f"validation_rps_{variable}"] for row in history],
            label=VARIABLE_LABELS[variable],
            color=colors[variable],
        )
    axes[1, 0].set(title="Validation RPS by variable", ylabel="RPS")

    axes[1, 1].plot(
        epochs,
        [row["train_modal_accuracy"] for row in history],
        label="Train",
    )
    axes[1, 1].plot(
        epochs,
        [row["validation_modal_accuracy"] for row in history],
        label="Validation",
    )
    axes[1, 1].set(
        title="Weighted modal-quintile accuracy (diagnostic; not ACC)",
        ylabel="Hit rate",
        ylim=(0.0, 1.0),
    )

    for axis in axes.flat:
        axis.axvline(best_epoch, color="0.55", linestyle=":", linewidth=1.0)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(SYNTHETIC_NOTICE)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy_curve(
    history: list[dict[str, float]], destination: Path, *, best_epoch: int
) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(epochs, [row["train_modal_accuracy"] for row in history], label="Train")
    axis.plot(
        epochs,
        [row["validation_modal_accuracy"] for row in history],
        label="Validation",
    )
    axis.axvline(best_epoch, color="0.45", linestyle=":", label="Selected epoch")
    axis.set(
        title="Weighted modal-quintile accuracy — synthetic diagnostic, not ACC",
        xlabel="Epoch",
        ylabel="Hit rate",
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_variable_rps(
    anchor: SplitMetrics,
    selected: SplitMetrics,
    uniform: SplitMetrics,
    destination: Path,
) -> None:
    positions = np.arange(3)
    width = 0.25
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    axis.bar(positions - width, uniform.rps_by_variable, width, label="Uniform")
    axis.bar(positions, anchor.rps_by_variable, width, label="p0 anchor")
    axis.bar(positions + width, selected.rps_by_variable, width, label="Selected model")
    axis.set(
        title=f"Validation RPS by variable — {SYNTHETIC_NOTICE.lower()}",
        ylabel="RPS (lower is better)",
        xticks=positions,
        xticklabels=[VARIABLE_LABELS[variable] for variable in QUEST_VARIABLES],
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _reliability_points(
    probabilities: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    observed = target[:, None] == np.arange(5)[None, :, None, None]
    effective = np.broadcast_to(weights[None, None], probabilities.shape)
    forecast_values = probabilities.reshape(-1)
    observed_values = observed.reshape(-1).astype(np.float64)
    weight_values = effective.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.clip(np.digitize(forecast_values, edges[1:-1]), 0, bins - 1)
    mean_forecast: list[float] = []
    observed_frequency: list[float] = []
    for index in range(bins):
        selected = bin_index == index
        total_weight = float(weight_values[selected].sum())
        if total_weight <= 0.0:
            continue
        mean_forecast.append(
            float(np.sum(forecast_values[selected] * weight_values[selected]) / total_weight)
        )
        observed_frequency.append(
            float(np.sum(observed_values[selected] * weight_values[selected]) / total_weight)
        )
    return np.asarray(mean_forecast), np.asarray(observed_frequency)


def plot_reliability(
    anchor_probabilities: np.ndarray,
    model_probabilities: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(9.0, 10.5), constrained_layout=True)
    periods = ("D19-25", "D26-32")
    for variable_index, variable in enumerate(QUEST_VARIABLES):
        for period in range(2):
            axis = axes[variable_index, period]
            axis.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--")
            for label, probabilities, color in (
                ("p0 anchor", anchor_probabilities, "#777777"),
                ("Selected model", model_probabilities, "#1b9e77"),
            ):
                forecast, observed = _reliability_points(
                    probabilities[:, variable_index, period],
                    target[:, variable_index, period],
                    weights[variable_index, period],
                )
                axis.plot(forecast, observed, marker="o", label=label, color=color)
            axis.set(
                title=f"{VARIABLE_LABELS[variable]} · {periods[period]}",
                xlabel="Mean forecast probability",
                ylabel="Observed category frequency",
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            )
            axis.grid(alpha=0.18)
            if variable_index == 0 and period == 0:
                axis.legend(frameon=False)
    figure.suptitle(f"Reliability by variable and lead — {SYNTHETIC_NOTICE.lower()}")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def native_grid_backward_check(
    model: QuestTemporalProbUNet, device: torch.device
) -> dict[str, float | list[int] | bool]:
    """Run one native-grid forward/backward pass and report concrete evidence."""

    started = time.monotonic()
    anchor_np = synthetic_anchor()
    features_np = synthetic_features(anchor_np, "2026-08-15")
    features = torch.from_numpy(features_np[None]).to(device).requires_grad_()
    anchor = torch.from_numpy(anchor_np[None]).to(device)
    target = anchor.argmax(dim=3)
    weights = torch.ones((3, 2, 121, 240), dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)
    model.train()
    probabilities = model(features, anchor)
    loss = multivariable_rps_loss(probabilities, target, weights)
    loss.backward()
    finite_gradient = features.grad is not None and bool(torch.isfinite(features.grad).all())
    if not finite_gradient:
        raise RuntimeError("native-grid backward pass produced an invalid feature gradient")
    return {
        "shape": list(probabilities.shape),
        "probabilities_finite": bool(torch.isfinite(probabilities).all()),
        "feature_gradient_finite": finite_gradient,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_smoke(args: argparse.Namespace) -> Path:
    """Execute training, validation selection, plots, and round-trip checks."""

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.train_cases < 1 or args.validation_cases < 1:
        raise ValueError("train and validation cases must be positive")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    device = choose_device(args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        args.run_dir
        if args.run_dir is not None
        else Path(__file__).resolve().parent / "runs" / f"multivariable_smoke_{stamp}"
    ).expanduser().resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    figure_directory = run_directory / "figures"
    figure_directory.mkdir()

    train_split = make_synthetic_split(
        args.train_cases,
        args.seed + 101,
        height=args.height,
        width=args.width,
    )
    validation_split = make_synthetic_split(
        args.validation_cases,
        args.seed + 202,
        height=args.height,
        width=args.width,
    )
    weights = synthetic_spatial_weights(args.height, args.width).to(device)
    model_config: dict[str, Any] = {
        "in_channels": MULTIVARIABLE_INPUT_CHANNELS,
        "variables": list(QUEST_VARIABLES),
        "base_channels": args.base_channels,
        "dropout": args.dropout,
        "attention_heads": args.attention_heads,
        "attention_dropout": args.attention_dropout,
    }
    model = QuestTemporalProbUNet(
        in_channels=model_config["in_channels"],
        variables=tuple(model_config["variables"]),
        base_channels=model_config["base_channels"],
        dropout=model_config["dropout"],
        attention_heads=model_config["attention_heads"],
        attention_dropout=model_config["attention_dropout"],
    ).to(device)
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
        min_lr=1.0e-5,
    )
    generator = torch.Generator().manual_seed(args.seed + 303)

    train_anchor, _ = evaluate_split(
        model,
        train_split,
        weights,
        device,
        correction_penalty=args.correction_penalty,
    )
    validation_anchor, anchor_probabilities = evaluate_split(
        model,
        validation_split,
        weights,
        device,
        correction_penalty=args.correction_penalty,
        return_probabilities=True,
    )
    assert anchor_probabilities is not None
    initial_identity_error = float(
        np.max(np.abs(anchor_probabilities - validation_split.anchor.numpy()))
    )
    if initial_identity_error > 1.0e-6:
        raise RuntimeError(
            f"epoch-zero model does not reproduce p0 (error={initial_identity_error})"
        )
    uniform_probabilities = torch.full_like(validation_split.anchor, 0.2)
    zero_corrections = torch.zeros_like(uniform_probabilities)
    validation_uniform = _metrics_from_tensors(
        uniform_probabilities.to(device),
        zero_corrections.to(device),
        validation_split.target.to(device),
        weights,
        0.0,
    )

    history = [
        _history_row(
            0,
            train_anchor,
            validation_anchor,
            validation_anchor,
            validation_uniform,
            optimizer.param_groups[0]["lr"],
        )
    ]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = validation_anchor.rps_mean
    stale = 0
    started = time.monotonic()
    print(SYNTHETIC_NOTICE)
    print(f"device={device} train={train_split.cases} validation={validation_split.cases}")
    print(f"epoch=000 validation_p0_rps={best_validation:.6f}")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            train_split,
            weights,
            optimizer,
            device,
            batch_size=args.batch_size,
            correction_penalty=args.correction_penalty,
            gradient_clip=args.gradient_clip,
            generator=generator,
        )
        train_metrics, _ = evaluate_split(
            model,
            train_split,
            weights,
            device,
            correction_penalty=args.correction_penalty,
        )
        validation_metrics, _ = evaluate_split(
            model,
            validation_split,
            weights,
            device,
            correction_penalty=args.correction_penalty,
        )
        scheduler.step(validation_metrics.rps_mean)
        history.append(
            _history_row(
                epoch,
                train_metrics,
                validation_metrics,
                validation_anchor,
                validation_uniform,
                optimizer.param_groups[0]["lr"],
            )
        )
        print(
            f"epoch={epoch:03d} train_rps={train_metrics.rps_mean:.6f} "
            f"validation_rps={validation_metrics.rps_mean:.6f} "
            f"validation_hit_rate={validation_metrics.modal_accuracy_mean:.4f}"
        )
        if validation_metrics.rps_mean < best_validation - 1.0e-7:
            best_validation = validation_metrics.rps_mean
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    selected_metrics, selected_probabilities = evaluate_split(
        model,
        validation_split,
        weights,
        device,
        correction_penalty=args.correction_penalty,
        return_probabilities=True,
    )
    assert selected_probabilities is not None
    improvement = validation_anchor.rps_mean - selected_metrics.rps_mean
    minimum_improvement_passed = improvement >= args.minimum_improvement

    native_evidence: dict[str, Any] | None = None
    if not args.skip_native_grid_check:
        native_evidence = native_grid_backward_check(model, device)
    model.eval()

    elapsed = time.monotonic() - started
    checkpoint_path = run_directory / "best.pt"
    checkpoint = {
        "schema_version": 1,
        "model_name": "QuestTemporalProbUNet",
        "model_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "model_config": model_config,
        "feature_contract": {
            "name": "quest_multivariable_38_channel_v1",
            "feature_names": list(MULTIVARIABLE_FEATURE_NAMES),
            "variable_order": list(QUEST_VARIABLES),
        },
        "metadata": {
            "data": "synthetic",
            "trained": best_epoch > 0,
            "optimization_attempted": True,
            "submission_ready": False,
            "selected_system": "p0_anchor" if best_epoch == 0 else "optimized_model",
            "minimum_improvement_required": args.minimum_improvement,
            "minimum_improvement_passed": minimum_improvement_passed,
            "best_epoch": best_epoch,
            "validation_p0_rps": validation_anchor.rps_mean,
            "best_validation_rps": selected_metrics.rps_mean,
            "validation_rps_improvement": improvement,
            "validation_skill_vs_p0": 1.0
            - selected_metrics.rps_mean / validation_anchor.rps_mean,
            "elapsed_seconds": elapsed,
            "permission_notice": FUXI_PERMISSION_WARNING,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    restored, _ = load_checkpoint_model(
        checkpoint_path,
        device=str(device),
        in_channels=MULTIVARIABLE_INPUT_CHANNELS,
    )
    with torch.inference_mode():
        restored_output = restored(
            validation_split.features[:1].to(device),
            validation_split.anchor[:1].to(device),
        ).cpu().numpy()
        selected_output = model(
            validation_split.features[:1].to(device),
            validation_split.anchor[:1].to(device),
        ).cpu().numpy()
    checkpoint_roundtrip_error = float(np.max(np.abs(restored_output - selected_output)))
    if checkpoint_roundtrip_error > 1.0e-7:
        raise RuntimeError(
            f"checkpoint round trip changed output by {checkpoint_roundtrip_error}"
        )

    history_path = run_directory / "history.csv"
    write_history(history, history_path)
    plot_training_history(
        history,
        figure_directory / "training_history.png",
        best_epoch=best_epoch,
        anchor_rps=validation_anchor.rps_mean,
    )
    plot_accuracy_curve(
        history,
        figure_directory / "modal_quintile_accuracy.png",
        best_epoch=best_epoch,
    )
    plot_variable_rps(
        validation_anchor,
        selected_metrics,
        validation_uniform,
        figure_directory / "validation_rps_by_variable.png",
    )
    plot_reliability(
        anchor_probabilities,
        selected_probabilities,
        validation_split.target.numpy(),
        synthetic_spatial_weights(args.height, args.width).numpy(),
        figure_directory / "validation_reliability.png",
    )

    summary = {
        "status": (
            "verified_synthetic_optimization_smoke"
            if minimum_improvement_passed
            else "synthetic_optimization_below_required_improvement"
        ),
        "notice": SYNTHETIC_NOTICE,
        "submission_ready": False,
        "model_config": model_config,
        "training_config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"run_dir"}
        }
        | {"run_dir": str(run_directory)},
        "best_epoch": best_epoch,
        "selected_system": "p0_anchor" if best_epoch == 0 else "optimized_model",
        "minimum_improvement_required": args.minimum_improvement,
        "minimum_improvement_passed": minimum_improvement_passed,
        "initial_anchor_identity_error": initial_identity_error,
        "validation_anchor": asdict(validation_anchor),
        "validation_uniform": asdict(validation_uniform),
        "selected_validation": asdict(selected_metrics),
        "validation_rps_improvement": improvement,
        "validation_skill_vs_p0": 1.0
        - selected_metrics.rps_mean / validation_anchor.rps_mean,
        "validation_rpss_vs_uniform": 1.0
        - selected_metrics.rps_mean / validation_uniform.rps_mean,
        "checkpoint_roundtrip_max_error": checkpoint_roundtrip_error,
        "native_grid_backward_check": native_evidence,
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "history": history_path.name,
            "figures": sorted(path.name for path in figure_directory.glob("*.png")),
        },
        "limitations": [
            "synthetic train and validation cases, not ERA5/FuXi skill",
            "modal-quintile hit rate is diagnostic and is not meteorological ACC",
            "real ACC requires continuous weekly forecasts, observations, and climatology",
            FUXI_PERMISSION_WARNING,
        ],
    }
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"best_epoch={best_epoch} validation_rps={selected_metrics.rps_mean:.6f} "
        f"improvement={improvement:.6f}"
    )
    if not minimum_improvement_passed:
        print(
            "verification=below_required_improvement; artifacts retain the "
            "selected p0/model fallback"
        )
    print(f"saved {run_directory}")
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--validation-cases", type=int, default=12)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--correction-penalty", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-native-grid-check", action="store_true")
    return parser


def main() -> None:
    run_smoke(build_parser().parse_args())


if __name__ == "__main__":
    main()
