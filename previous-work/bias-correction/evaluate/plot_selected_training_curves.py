#!/usr/bin/env python3
"""Render a slide-ready learning-curve figure for one frozen configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_COLOR = "#2878B5"
VALIDATION_COLOR = "#D95319"
SEED_COLORS = ("#3B75AF", "#7A5195", "#2A9D8F")


def _load_complete_sweep(sweep_directory: Path) -> tuple[dict, pd.DataFrame]:
    manifest_path = sweep_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or bool(manifest.get("smoke")):
        raise ValueError("learning curves require a complete, non-smoke sweep")
    if manifest.get("test_predictions_created") is not False:
        raise ValueError("source sweep does not declare the test period untouched")
    history = pd.read_csv(
        sweep_directory / "metrics" / "training_history_tidy.csv"
    )
    required = {
        "configuration",
        "seed",
        "epoch",
        "train_loss",
        "validation_loss",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"training history lacks columns: {sorted(missing)}")
    return manifest, history


def _default_configuration(sweep_directory: Path, manifest: dict) -> str:
    ranking = pd.read_csv(
        sweep_directory / "metrics" / "ranked_configurations.csv"
    )
    reference = str(manifest.get("reference_configuration", "physical_control"))
    qualifying = ranking.loc[
        ranking.configuration.ne(reference)
        & ranking.get("qualifies", pd.Series(False, index=ranking.index)).astype(bool)
    ]
    if not qualifying.empty:
        return str(qualifying.sort_values("rank", kind="stable").iloc[0].configuration)
    if reference not in set(ranking.configuration.astype(str)):
        raise ValueError(f"reference configuration {reference!r} is absent")
    return reference


def _epoch_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for epoch, group in selected.groupby("epoch", sort=True):
        row = {"epoch": int(epoch)}
        for metric in ("train_loss", "validation_loss"):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_minimum"] = float(values.min())
            row[f"{metric}_maximum"] = float(values.max())
            row[f"{metric}_seed_count"] = int(values.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _shared_epoch_history(selected: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    """Keep only epochs represented by every seed for an honest mean curve."""

    counts = selected.groupby("epoch").seed.nunique()
    shared_epochs = counts.index[counts.eq(len(seeds))]
    if shared_epochs.empty:
        raise ValueError("training histories have no epoch shared by every seed")
    return selected.loc[selected.epoch.isin(shared_epochs)].copy()


def render(
    sweep_directory: Path,
    output_stem: Path,
    *,
    configuration: str | None = None,
) -> tuple[Path, Path, Path]:
    sweep_directory = Path(sweep_directory).expanduser().resolve()
    output_stem = Path(output_stem).expanduser().resolve()
    manifest, history = _load_complete_sweep(sweep_directory)
    configuration = configuration or _default_configuration(
        sweep_directory, manifest
    )
    selected = history.loc[history.configuration.eq(configuration)].copy()
    seeds = tuple(sorted(int(value) for value in selected.seed.unique()))
    if len(seeds) != 3:
        raise ValueError(
            f"presentation confirmation requires exactly three seeds; got {seeds}"
        )
    if selected.duplicated(["seed", "epoch"]).any():
        raise ValueError("training history contains duplicate seed/epoch rows")
    if not np.isfinite(
        selected[["train_loss", "validation_loss"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("training history contains non-finite objective values")

    ranking = pd.read_csv(
        sweep_directory / "metrics" / "ranked_configurations.csv"
    )
    match = ranking.loc[ranking.configuration.eq(configuration)]
    label = str(match.iloc[0].get("label", configuration)) if len(match) else configuration
    shared = _shared_epoch_history(selected, seeds)
    summary = _epoch_summary(shared)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#9AA8B3",
            "axes.labelcolor": "#243746",
            "xtick.color": "#425563",
            "ytick.color": "#425563",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    axis = axes[0]
    epochs = summary.epoch.to_numpy(dtype=np.int64) + 1
    for metric, color, name in (
        ("train_loss", TRAIN_COLOR, "Training"),
        ("validation_loss", VALIDATION_COLOR, "Blocked validation"),
    ):
        mean = summary[f"{metric}_mean"].to_numpy(dtype=np.float64)
        low = summary[f"{metric}_minimum"].to_numpy(dtype=np.float64)
        high = summary[f"{metric}_maximum"].to_numpy(dtype=np.float64)
        axis.fill_between(epochs, low, high, color=color, alpha=0.13, linewidth=0)
        axis.plot(epochs, mean, color=color, linewidth=2.5, label=f"{name} mean")
    axis.set_title("Mean learning trajectory across three seeds")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Composite objective")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="best")

    axis = axes[1]
    best_rows = []
    for color, seed in zip(SEED_COLORS, seeds):
        group = selected.loc[selected.seed.eq(seed)].sort_values("epoch")
        display_epoch = group.epoch.to_numpy(dtype=np.int64) + 1
        values = group.validation_loss.to_numpy(dtype=np.float64)
        best_position = int(np.argmin(values))
        axis.plot(
            display_epoch,
            values,
            color=color,
            linewidth=1.9,
            alpha=0.92,
            label=f"Seed {seed}",
        )
        axis.scatter(
            display_epoch[best_position],
            values[best_position],
            marker="*",
            s=105,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        best_rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "best_epoch_display": int(display_epoch[best_position]),
                "best_validation_loss": float(values[best_position]),
            }
        )
    axis.set_title("Seed stability and selected checkpoints")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Blocked-validation objective")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="best")
    for current in axes:
        current.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"Learning curves · {label}\n"
        "Train 2002–2017 · architecture/epoch selection only on 2018–2019",
        fontsize=15.0,
        fontweight="semibold",
        color="#172B3A",
        y=1.035,
    )
    figure.text(
        0.5,
        0.015,
        "Left: only epochs shared by all three seeds; shading spans seeds. Right: complete seed histories; stars mark minima.",
        ha="center",
        fontsize=8.8,
        color="#5D7180",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.92), w_pad=2.4)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".csv"),
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to replace an existing output: {output_stem}")
    figure.savefig(outputs[0], dpi=340, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    pd.DataFrame(best_rows).to_csv(outputs[2], index=False)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_directory", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--configuration", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    render(
        args.sweep_directory,
        args.output_stem,
        configuration=args.configuration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
