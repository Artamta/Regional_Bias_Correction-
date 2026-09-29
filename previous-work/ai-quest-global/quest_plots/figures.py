"""Small, reusable Matplotlib figures for the AI Quest experiment.

The module deliberately depends only on NumPy and Matplotlib.  Heavy model
inference and artifact discovery live in :mod:`quest_plots.report` so these
plotters remain easy to test with small synthetic arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


INK = "#17212b"
MUTED = "#607080"
GRID = "#d9e1e8"
BLUE = "#277da1"
GREEN = "#2a9d8f"
ORANGE = "#f4a261"
RED = "#e76f51"
PURPLE = "#7b6fd0"
PINK = "#cc79a7"
GREY = "#7a858f"
CREAM = "#f7f4ed"
EXPLORATORY_NOTE = "Exploratory FuXi–IMERG smoke · not official ERA5 / competition validation"
LEAD_LABELS = ("D19–25", "D26–32")


def apply_style() -> None:
    """Apply the package's restrained, presentation-friendly visual style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: Figure, destination: Path, formats: Sequence[str]) -> list[Path]:
    """Save a figure under one stem in each requested format."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = destination.with_suffix(f".{extension}")
        figure.savefig(path, dpi=220 if extension == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def _box(
    axis: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    title_color: str = INK,
    title_size: float = 10.2,
    body_size: float = 8.8,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=facecolor,
        facecolor=facecolor,
        alpha=0.13,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.68,
        title,
        ha="center",
        va="center",
        weight="bold",
        color=title_color,
        fontsize=title_size,
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.34,
        body,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=body_size,
        linespacing=1.3,
    )


def _arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=MUTED,
            connectionstyle="arc3,rad=0",
        )
    )


def plot_architecture(destination: Path, formats: Sequence[str]) -> list[Path]:
    """Draw the evaluated 18-channel residual U-Net architecture."""

    apply_style()
    figure, axis = plt.subplots(figsize=(13.2, 6.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle(
        "Global precipitation probability adapter — one spatial ensemble member",
        fontsize=17,
        weight="bold",
        y=0.98,
    )
    axis.text(
        0.5,
        0.925,
        "Shared weights for both lead windows · 111,909 trainable parameters · global 1.5° grid",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )

    _box(
        axis,
        (0.02, 0.57),
        0.17,
        0.22,
        "18 input maps / lead",
        "5 log FuXi probabilities\n5 rainfall quantiles\n8 space/time/static maps",
        facecolor=BLUE,
    )
    _box(
        axis,
        (0.245, 0.59),
        0.12,
        0.18,
        "Encoder 1",
        "16 channels\nConv ×2 + GroupNorm\nSiLU + dropout",
        facecolor=GREEN,
        body_size=8.2,
    )
    _box(
        axis,
        (0.41, 0.59),
        0.12,
        0.18,
        "Encoder 2",
        "32 channels\n2× downsample\nperiodic longitude",
        facecolor=GREEN,
        body_size=8.2,
    )
    _box(
        axis,
        (0.575, 0.59),
        0.12,
        0.18,
        "Bottleneck",
        "64 channels\n2× downsample\nspatial control",
        facecolor=PURPLE,
        body_size=8.2,
    )
    _box(
        axis,
        (0.74, 0.59),
        0.12,
        0.18,
        "Decoder",
        "bilinear upsample\nskip connections\n32 → 16 channels",
        facecolor=ORANGE,
        body_size=8.2,
    )
    _box(
        axis,
        (0.885, 0.57),
        0.095,
        0.22,
        "5 corrections",
        "zero-initialized\n1×1 head\nper lead/cell",
        facecolor=RED,
        body_size=8.0,
    )
    for start, end in (
        ((0.19, 0.68), (0.245, 0.68)),
        ((0.365, 0.68), (0.41, 0.68)),
        ((0.53, 0.68), (0.575, 0.68)),
        ((0.695, 0.68), (0.74, 0.68)),
        ((0.86, 0.68), (0.885, 0.68)),
    ):
        _arrow(axis, start, end)

    _box(
        axis,
        (0.09, 0.18),
        0.22,
        0.19,
        "FuXi anchor p₀",
        "51-member weekly totals →\n5 climatological category probabilities\nJeffreys smoothing keeps p > 0",
        facecolor=PINK,
    )
    _box(
        axis,
        (0.39, 0.18),
        0.22,
        0.19,
        "Residual probability update",
        "softmax[ log(p₀) + correction logits ]\nzero head means epoch 0 = raw FuXi",
        facecolor=PURPLE,
    )
    _box(
        axis,
        (0.69, 0.18),
        0.22,
        0.19,
        "Final probabilistic field",
        "[2 leads, 5 categories, 121, 240]\nprobabilities sum to 1 at every cell",
        facecolor=GREEN,
    )
    _arrow(axis, (0.31, 0.275), (0.39, 0.275))
    _arrow(axis, (0.61, 0.275), (0.69, 0.275))
    _arrow(axis, (0.932, 0.57), (0.58, 0.37))
    axis.text(
        0.5,
        0.075,
        "The recommended exploratory candidate is a 3-seed average of this simple spatial model, followed by a validation-fitted uniform blend.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )
    return save_figure(figure, destination, formats)


def plot_training_workflow(destination: Path, formats: Sequence[str]) -> list[Path]:
    """Draw data, normalization, objective, and selection as one workflow."""

    apply_style()
    figure, axis = plt.subplots(figsize=(13.2, 7.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle("Training and validation workflow", fontsize=17, weight="bold", y=0.98)
    axis.text(0.5, 0.925, EXPLORATORY_NOTE, ha="center", color=RED, fontsize=10.2)

    boxes = [
        (
            (0.02, 0.59),
            "1 · Build weekly data",
            "FuXi: 51 members\nD19–25 / D26–32 sums\nIMERG truth\n2002–2016 calendar climatology",
            BLUE,
        ),
        (
            (0.215, 0.59),
            "2 · Make targets + anchor",
            "Five climatological categories\np₀ from member frequencies\nJeffreys smoothing\ninvalid/support cells masked",
            PINK,
        ),
        (
            (0.41, 0.59),
            "3 · Normalize inputs",
            "TP q10/q25/q50/q75/q90\nmean/std from train only\nlog(p₀), coordinates + season\nkept on fixed scales",
            ORANGE,
        ),
        (
            (0.605, 0.59),
            "4 · Optimize",
            "AdamW · learning rate 3×10⁻⁴\narea-weighted RPS\n+ 10⁻⁴ × mean(correction²)\nvalidation-based LR schedule",
            PURPLE,
        ),
        (
            (0.80, 0.59),
            "5 · Select + calibrate",
            "lowest 2019 validation RPS\nepoch 0 = p₀ fallback\n3-seed model average\nper-lead uniform blend",
            GREEN,
        ),
    ]
    for xy, title, body, color in boxes:
        _box(
            axis,
            (xy[0], 0.57),
            0.175,
            0.27,
            title,
            body,
            facecolor=color,
            title_size=8.4,
            body_size=7.25,
        )
    for left in (0.195, 0.39, 0.585, 0.78):
        _arrow(axis, (left, 0.71), (left + 0.02, 0.71))

    _box(
        axis,
        (0.05, 0.19),
        0.24,
        0.20,
        "Train · 2017–2018",
        "8 initialization dates\nupdates model weights\nprovides normalization statistics",
        facecolor=BLUE,
    )
    _box(
        axis,
        (0.38, 0.19),
        0.24,
        0.20,
        "Validation / calibration · 2019",
        "4 initialization dates\ncheckpoint, architecture, ensemble\nand blend coefficients selected here",
        facecolor=ORANGE,
    )
    _box(
        axis,
        (0.71, 0.19),
        0.24,
        0.20,
        "Retrospective evaluation · 2020",
        "4 already-opened initialization dates\nno refitting in the score calculation\nsmall-sample exploratory evidence",
        facecolor=GREEN,
    )
    _arrow(axis, (0.29, 0.29), (0.38, 0.29))
    _arrow(axis, (0.62, 0.29), (0.71, 0.29))
    axis.text(
        0.5,
        0.08,
        "Primary metric: Ranked Probability Score (RPS; lower is better).  RPSS is measured against uniform 20% quintile probabilities.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    return save_figure(figure, destination, formats)


def plot_training_history(
    histories: Mapping[str, Mapping[str, np.ndarray]],
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot objective loss and RPS curves for one or more random seeds."""

    if not histories:
        raise ValueError("at least one training history is required")
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    colors = (BLUE, GREEN, PURPLE, ORANGE)
    for index, (name, history) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        epoch = np.asarray(history["epoch"], dtype=float)
        train_loss = np.asarray(history["train_loss"], dtype=float)
        validation_loss = np.asarray(history["validation_loss"], dtype=float)
        train_rps = np.asarray(history["train_rps"], dtype=float)
        validation_rps = np.asarray(history["validation_rps"], dtype=float)
        alpha = 1.0 if index == 0 else 0.62
        axes[0].plot(epoch, train_loss, color=color, linewidth=1.6, alpha=alpha, label=f"{name} train")
        axes[0].plot(epoch, validation_loss, color=color, linewidth=2.0, linestyle="--", alpha=alpha, label=f"{name} validation")
        axes[1].plot(epoch, train_rps, color=color, linewidth=1.6, alpha=alpha, label=f"{name} train")
        axes[1].plot(epoch, validation_rps, color=color, linewidth=2.0, linestyle="--", alpha=alpha, label=f"{name} validation")
        best = int(np.nanargmin(validation_rps))
        axes[1].scatter(epoch[best], validation_rps[best], s=34, color=color, zorder=4)
    raw_anchor = float(next(iter(histories.values()))["validation_rps"][0])
    axes[1].axhline(raw_anchor, color=PINK, linewidth=1.3, linestyle=":", label="raw FuXi p₀")
    for axis in axes:
        axis.grid(axis="y", color=GRID, alpha=0.65)
        axis.set_xlabel("Epoch")
    axes[0].set_title("Optimization objective")
    axes[0].set_ylabel("RPS + correction penalty (lower is better)")
    axes[1].set_title("Ranked Probability Score")
    axes[1].set_ylabel("Area-weighted RPS (lower is better)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Training curves · three simple spatial U-Net seeds", fontsize=15, weight="bold")
    figure.text(0.5, -0.015, EXPLORATORY_NOTE, ha="center", color=RED, fontsize=9)
    return save_figure(figure, destination, formats)


def plot_scores(
    score_rows: Sequence[Mapping[str, object]],
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot pooled validation and retrospective RPS/RPSS for named systems."""

    if not score_rows:
        raise ValueError("score rows cannot be empty")
    apply_style()
    systems = list(dict.fromkeys(str(row["label"]) for row in score_rows))
    splits = ("Validation / calibration 2019", "Retrospective evaluation 2020")
    lookup = {(str(row["split_label"]), str(row["label"])): row for row in score_rows}
    colors = {
        "Uniform": GREY,
        "Raw FuXi p₀": PINK,
        "Calibrated p₀": ORANGE,
        "Raw spatial ensemble": BLUE,
        "Selected calibrated ensemble": GREEN,
    }
    x = np.arange(len(splits), dtype=float)
    width = min(0.16, 0.78 / max(len(systems), 1))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for index, system in enumerate(systems):
        offset = (index - (len(systems) - 1) / 2) * width
        rps = [float(lookup[(split, system)]["rps"]) for split in splits]
        rpss = [100.0 * float(lookup[(split, system)]["rpss"]) for split in splits]
        bars = axes[0].bar(x + offset, rps, width * 0.92, color=colors.get(system, GREY), label=system)
        axes[1].bar(x + offset, rpss, width * 0.92, color=colors.get(system, GREY), label=system)
        for bar, value in zip(bars, rps, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
    axes[0].set_ylabel("Pooled RPS (lower is better)")
    axes[0].set_ylim(0.75, 1.075)
    axes[0].set_title("Absolute probabilistic error")
    axes[1].axhline(0, color=INK, linewidth=0.9)
    axes[1].set_ylabel("RPSS vs uniform (%) · higher is better")
    axes[1].set_title("Skill relative to uniform climatology")
    for axis in axes:
        axis.set_xticks(x, splits)
        axis.grid(axis="y", color=GRID, alpha=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncol=len(systems), fontsize=8.4)
    figure.suptitle(
        "Validation and retrospective evaluation scores",
        fontsize=15,
        weight="bold",
        y=1.07,
    )
    figure.text(0.5, 1.012, EXPLORATORY_NOTE, ha="center", color=RED, fontsize=9)
    return save_figure(figure, destination, formats)


def plot_probabilistic_metrics(
    metric_rows: Sequence[Mapping[str, object]],
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot RPS, multicategory Brier, and log scores by lead and pooled."""

    if not metric_rows:
        raise ValueError("metric rows cannot be empty")
    apply_style()
    systems = list(dict.fromkeys(str(row["system"]) for row in metric_rows))
    leads = ("D19–25", "D26–32", "Pooled")
    lookup = {
        (str(row["lead"]), str(row["system"])): row for row in metric_rows
    }
    missing = [
        (lead, system)
        for lead in leads
        for system in systems
        if (lead, system) not in lookup
    ]
    if missing:
        raise ValueError(f"metric rows are incomplete: {missing}")
    colors = {
        "Uniform": GREY,
        "Raw FuXi p₀": PINK,
        "Selected calibrated ensemble": GREEN,
    }
    specifications = (
        ("rps", "Ranked Probability Score", "RPS"),
        ("brier", "Multicategory Brier score", "Σₖ (pₖ − oₖ)²"),
        ("log", "Log score", "−log probability of observed category"),
    )
    x = np.arange(len(leads), dtype=float)
    width = 0.23
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    for axis, (field, title, ylabel) in zip(axes, specifications, strict=True):
        for index, system in enumerate(systems):
            offset = (index - (len(systems) - 1) / 2) * width
            values = [float(lookup[(lead, system)][field]) for lead in leads]
            bars = axis.bar(
                x + offset,
                values,
                width * 0.92,
                color=colors.get(system, GREY),
                label=system,
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    rotation=90,
                )
        axis.set_title(title)
        axis.set_ylabel(f"{ylabel} · lower is better")
        axis.set_xticks(x, leads)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color=GRID, alpha=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
        ncol=len(systems),
        fontsize=8.8,
    )
    figure.suptitle(
        "Probabilistic scorecard · 2020 retrospective cases",
        fontsize=15,
        weight="bold",
        y=1.07,
    )
    figure.text(
        0.5,
        1.012,
        "RPS is the ordered-category analogue of CRPS; exact rainfall-unit CRPS is not identifiable from five category probabilities",
        ha="center",
        color=RED,
        fontsize=8.8,
    )
    return save_figure(figure, destination, formats)


def plot_prediction_probabilities(
    probability: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    init_date: str,
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot the ten maps that constitute one two-lead, five-category forecast."""

    probability = np.asarray(probability, dtype=float)
    if probability.shape != (2, 5, len(latitude), len(longitude)):
        raise ValueError("probability must have shape [2, 5, latitude, longitude]")
    apply_style()
    figure, axes = plt.subplots(2, 5, figsize=(15.0, 6.4), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    category_labels = ("Q1 · driest", "Q2", "Q3", "Q4", "Q5 · wettest")
    for lead in range(2):
        for category in range(5):
            axis = axes[lead, category]
            mesh = axis.pcolormesh(longitude, latitude, probability[lead, category], cmap="YlGnBu", vmin=0.0, vmax=0.5, shading="auto")
            if lead == 0:
                axis.set_title(category_labels[category])
            if category == 0:
                axis.set_ylabel(f"{LEAD_LABELS[lead]}\nLatitude")
            if lead == 1:
                axis.set_xlabel("Longitude")
            axis.set_xlim(float(np.min(longitude)), float(np.max(longitude)))
            axis.set_ylim(float(np.min(latitude)), float(np.max(latitude)))
            axis.grid(alpha=0.12, linewidth=0.5)
    assert mesh is not None
    colorbar = figure.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.82, pad=0.015)
    colorbar.set_label("Forecast probability")
    max_sum_error = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
    figure.suptitle(
        f"What the model predicts · initialization {init_date}",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        f"Two weekly windows × five rainfall categories; max probability-sum error {max_sum_error:.1e} · {EXPLORATORY_NOTE}",
        ha="center",
        color=RED,
        fontsize=8.8,
    )
    return save_figure(figure, destination, formats)


def plot_category_comparison(
    baseline_probability: np.ndarray,
    selected_probability: np.ndarray,
    target: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    init_date: str,
    lead_index: int,
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Compare all five FuXi, selected, and observed category fields for one lead."""

    expected_probability = (2, 5, len(latitude), len(longitude))
    expected_target = (2, len(latitude), len(longitude))
    baseline_probability = np.asarray(baseline_probability, dtype=float)
    selected_probability = np.asarray(selected_probability, dtype=float)
    target = np.asarray(target)
    if baseline_probability.shape != expected_probability:
        raise ValueError(f"baseline_probability must have shape {expected_probability}")
    if selected_probability.shape != expected_probability:
        raise ValueError(f"selected_probability must have shape {expected_probability}")
    if target.shape != expected_target:
        raise ValueError(f"target must have shape {expected_target}")
    if lead_index not in (0, 1):
        raise ValueError("lead_index must be 0 or 1")

    apply_style()
    probability_limit = min(
        1.0,
        max(
            0.5,
            float(
                np.nanpercentile(
                    np.concatenate(
                        (baseline_probability.ravel(), selected_probability.ravel())
                    ),
                    99.5,
                )
            ),
        ),
    )
    figure, axes = plt.subplots(
        3,
        5,
        figsize=(15.0, 8.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    category_labels = ("Q1 · driest", "Q2", "Q3", "Q4", "Q5 · wettest")
    probability_mesh = truth_mesh = None
    valid = (target[lead_index] >= 0) & (target[lead_index] <= 4)
    for category in range(5):
        probability_mesh = axes[0, category].pcolormesh(
            longitude,
            latitude,
            baseline_probability[lead_index, category],
            cmap="YlGnBu",
            vmin=0.0,
            vmax=probability_limit,
            shading="auto",
        )
        axes[1, category].pcolormesh(
            longitude,
            latitude,
            selected_probability[lead_index, category],
            cmap="YlGnBu",
            vmin=0.0,
            vmax=probability_limit,
            shading="auto",
        )
        observed = np.where(valid, target[lead_index] == category, np.nan)
        truth_mesh = axes[2, category].pcolormesh(
            longitude,
            latitude,
            observed,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        axes[0, category].set_title(category_labels[category])
        axes[2, category].set_xlabel("Longitude")
        for row in range(3):
            axes[row, category].set_xlim(float(np.min(longitude)), float(np.max(longitude)))
            axes[row, category].set_ylim(float(np.min(latitude)), float(np.max(latitude)))
            axes[row, category].grid(alpha=0.10, linewidth=0.45)
    axes[0, 0].set_ylabel("Raw FuXi p₀\nprobability\nLatitude")
    axes[1, 0].set_ylabel("Selected model\nprobability\nLatitude")
    axes[2, 0].set_ylabel("Ground truth\none-hot category\nLatitude")
    assert probability_mesh is not None and truth_mesh is not None
    probability_bar = figure.colorbar(
        probability_mesh,
        ax=axes[:2].ravel().tolist(),
        shrink=0.82,
        pad=0.012,
    )
    probability_bar.set_label("Forecast probability")
    truth_bar = figure.colorbar(
        truth_mesh,
        ax=axes[2].ravel().tolist(),
        shrink=0.82,
        pad=0.012,
        ticks=(0, 1),
    )
    truth_bar.set_label("Observed category membership")
    figure.suptitle(
        f"All five categories with FuXi baseline and ground truth · {LEAD_LABELS[lead_index]} · {init_date}",
        fontsize=14.5,
        weight="bold",
    )
    figure.text(
        0.5,
        -0.012,
        "Q1–Q5 are mutually exclusive climatological rainfall categories · "
        + EXPLORATORY_NOTE,
        ha="center",
        color=RED,
        fontsize=8.8,
    )
    return save_figure(figure, destination, formats)


def plot_spatial_scores(
    p0_rps: np.ndarray,
    model_rps: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    destination: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Compare case-mean raw FuXi and selected-model RPS at every grid cell."""

    p0_rps = np.asarray(p0_rps, dtype=float)
    model_rps = np.asarray(model_rps, dtype=float)
    expected = (2, len(latitude), len(longitude))
    if p0_rps.shape != expected or model_rps.shape != expected:
        raise ValueError(f"spatial RPS fields must have shape {expected}")
    improvement = p0_rps - model_rps
    finite = np.isfinite(improvement)
    limit = float(np.nanpercentile(np.abs(improvement[finite]), 98)) if finite.any() else 1.0
    limit = max(limit, 1.0e-6)
    error_max = float(np.nanpercentile(np.concatenate([p0_rps[np.isfinite(p0_rps)], model_rps[np.isfinite(model_rps)]]), 98))
    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True, sharey=True, constrained_layout=True)
    error_mesh = improvement_mesh = None
    for lead in range(2):
        error_mesh = axes[lead, 0].pcolormesh(longitude, latitude, p0_rps[lead], cmap="magma_r", vmin=0, vmax=error_max, shading="auto")
        axes[lead, 1].pcolormesh(longitude, latitude, model_rps[lead], cmap="magma_r", vmin=0, vmax=error_max, shading="auto")
        improvement_mesh = axes[lead, 2].pcolormesh(longitude, latitude, improvement[lead], cmap="PiYG", vmin=-limit, vmax=limit, shading="auto")
        axes[lead, 0].set_ylabel(f"{LEAD_LABELS[lead]}\nLatitude")
        for column in range(3):
            axes[lead, column].set_xlim(float(np.min(longitude)), float(np.max(longitude)))
            axes[lead, column].set_ylim(float(np.min(latitude)), float(np.max(latitude)))
            axes[lead, column].grid(alpha=0.12, linewidth=0.5)
            if lead == 1:
                axes[lead, column].set_xlabel("Longitude")
    axes[0, 0].set_title("Raw FuXi p₀ RPS")
    axes[0, 1].set_title("Selected calibrated ensemble RPS")
    axes[0, 2].set_title("Improvement: p₀ − selected")
    assert error_mesh is not None and improvement_mesh is not None
    error_bar = figure.colorbar(error_mesh, ax=axes[:, :2].ravel().tolist(), shrink=0.82, pad=0.015)
    error_bar.set_label("Case-mean grid-cell RPS")
    improvement_bar = figure.colorbar(improvement_mesh, ax=axes[:, 2].ravel().tolist(), shrink=0.82, pad=0.02)
    improvement_bar.set_label("RPS reduction (positive = improvement)")
    figure.suptitle("Where the selected adapter improves raw FuXi · 2020 retrospective cases", fontsize=15, weight="bold")
    figure.text(0.5, -0.012, EXPLORATORY_NOTE, ha="center", color=RED, fontsize=9)
    return save_figure(figure, destination, formats)
