"""One-page, share-ready PDF summary of the exploratory AI Quest experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from .figures import (
    BLUE,
    CREAM,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    PINK,
    PURPLE,
    RED,
    apply_style,
    save_figure,
)


def _panel(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.009,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
        )
    )
    axis.text(
        x + 0.018,
        y + height - 0.028,
        title,
        fontsize=11.0,
        weight="bold",
        color=INK,
        va="top",
    )


def _wrapped(
    axis: Axes,
    x: float,
    y: float,
    text: str,
    *,
    width: int,
    fontsize: float = 8.6,
    color: str = MUTED,
    weight: str = "normal",
    linespacing: float = 1.25,
) -> None:
    axis.text(
        x,
        y,
        textwrap.fill(text, width=width),
        fontsize=fontsize,
        color=color,
        weight=weight,
        va="top",
        linespacing=linespacing,
    )


def _find(
    rows: Sequence[Mapping[str, object]],
    **criteria: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one report row for {criteria}; found {len(matches)}")
    return matches[0]


def plot_group_report(
    score_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    spatial: Sequence[Mapping[str, float]],
    destination: Path,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """Render a concise single-page report suitable for group sharing."""

    if len(spatial) != 2:
        raise ValueError("spatial summary must contain exactly two leads")
    selected_validation = _find(
        score_rows, split="fit", label="Selected calibrated ensemble"
    )
    selected_test = _find(
        score_rows, split="evaluation", label="Selected calibrated ensemble"
    )
    calibrated_p0 = _find(score_rows, split="evaluation", label="Calibrated p₀")
    metric_systems = ("Uniform", "Raw FuXi p₀", "Selected calibrated ensemble")
    pooled_metrics = {
        system: _find(metric_rows, lead="Pooled", system=system)
        for system in metric_systems
    }
    neural_added_value = 100 * (
        float(calibrated_p0["rps"]) - float(selected_test["rps"])
    ) / float(calibrated_p0["rps"])

    apply_style()
    figure, axis = plt.subplots(figsize=(13.6, 9.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.035,
        0.965,
        "AI Quest global precipitation adapter",
        fontsize=21,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.035,
        0.921,
        "Short methodology and results note · 16 August 2026",
        fontsize=10.0,
        color=MUTED,
        va="top",
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.035, 0.858),
            0.93,
            0.043,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            linewidth=0,
            facecolor=RED,
            alpha=0.12,
        )
    )
    axis.text(
        0.5,
        0.879,
        "EXPLORATORY FuXi–IMERG SMOKE TEST · NOT OFFICIAL ERA5 OR COMPETITION VALIDATION",
        ha="center",
        va="center",
        fontsize=9.2,
        color=RED,
        weight="bold",
    )

    left_x, left_w = 0.035, 0.445
    right_x, right_w = 0.515, 0.45

    _panel(axis, left_x, 0.724, left_w, 0.105, title="Purpose", color=BLUE)
    _wrapped(
        axis,
        left_x + 0.018,
        0.780,
        "Convert 51-member FuXi subseasonal rainfall forecasts into better-calibrated probabilities for five climatological rainfall categories at D19–25 and D26–32.",
        width=72,
        fontsize=8.8,
    )

    _panel(axis, left_x, 0.408, left_w, 0.286, title="Methodology", color=PURPLE)
    methods = (
        ("1", "Data", "FuXi weekly totals + IMERG Final V07B; 2002–2016 IMERG calendar climatology."),
        ("2", "Inputs", "5 log(p₀), 5 TP quantiles, and 8 space/time/static maps; train-only normalization."),
        ("3", "Model", "111,909-parameter spatial residual U-Net; 3-seed mean. Output [2, 5, 121, 240]."),
        ("4", "Training", "Area-weighted RPS + 10⁻⁴ correction penalty; epoch 0 retains raw FuXi p₀ as fallback."),
        ("5", "Selection", "Lowest 2019 validation RPS; per-lead validation-fitted blend with uniform probabilities."),
    )
    start_y = 0.641
    for index, (number, title, body) in enumerate(methods):
        y = start_y - index * 0.051
        axis.text(
            left_x + 0.020,
            y,
            number,
            fontsize=8.3,
            weight="bold",
            color="white",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": PURPLE, "edgecolor": "none"},
        )
        axis.text(
            left_x + 0.048,
            y + 0.010,
            title,
            fontsize=8.5,
            weight="bold",
            color=INK,
            va="top",
        )
        _wrapped(
            axis,
            left_x + 0.125,
            y + 0.010,
            body,
            width=52,
            fontsize=7.55,
            linespacing=1.15,
        )

    _panel(axis, left_x, 0.257, left_w, 0.122, title="Data split and prediction", color=ORANGE)
    axis.text(
        left_x + 0.020,
        0.324,
        "TRAIN\n2017–18\n8 cases",
        fontsize=8.3,
        weight="bold",
        color=BLUE,
        va="top",
        ha="left",
    )
    axis.text(
        left_x + 0.151,
        0.324,
        "VALIDATE / CALIBRATE\n2019\n4 cases",
        fontsize=8.3,
        weight="bold",
        color=ORANGE,
        va="top",
        ha="left",
    )
    axis.text(
        left_x + 0.330,
        0.324,
        "RETROSPECTIVE\n2020\n4 cases",
        fontsize=8.3,
        weight="bold",
        color=GREEN,
        va="top",
        ha="left",
    )
    axis.text(
        left_x + 0.020,
        0.269,
        "Each forecast is a global probability field for Q1 (driest) through Q5 (wettest); probabilities sum to one at every cell.",
        fontsize=7.7,
        color=MUTED,
        va="top",
    )

    _panel(axis, left_x, 0.086, left_w, 0.142, title="Metric contract", color=GREEN)
    _wrapped(
        axis,
        left_x + 0.018,
        0.176,
        "RPS is the proper ordered-category analogue of CRPS and remains the primary score. Multicategory Brier and log scores are supporting proper scores; lower is better.",
        width=73,
        fontsize=8.15,
    )
    _wrapped(
        axis,
        left_x + 0.018,
        0.125,
        "Exact CRPS in millimetres is unavailable: the output is categorical, not continuous. We do not fabricate a bin-midpoint CRPS.",
        width=73,
        fontsize=7.8,
        color=RED,
    )

    _panel(axis, right_x, 0.610, right_w, 0.219, title="Headline results", color=GREEN)
    axis.text(
        right_x + 0.018,
        0.765,
        f"2019 selection RPS  {float(selected_validation['rps']):.3f}     |     2020 retrospective RPS  {float(selected_test['rps']):.3f}",
        fontsize=9.2,
        weight="bold",
        color=GREEN,
        va="top",
    )
    headers = ("System", "RPS", "Brier", "Log")
    col_x = (right_x + 0.020, right_x + 0.258, right_x + 0.326, right_x + 0.394)
    for x, header in zip(col_x, headers, strict=True):
        axis.text(x, 0.718, header, fontsize=8.1, weight="bold", color=INK, va="top")
    display_names = {
        "Uniform": "Uniform",
        "Raw FuXi p₀": "Raw FuXi p₀",
        "Selected calibrated ensemble": "Selected system",
    }
    row_colors = {"Uniform": GREY, "Raw FuXi p₀": PINK, "Selected calibrated ensemble": GREEN}
    for row_index, system in enumerate(metric_systems):
        y = 0.688 - row_index * 0.032
        row = pooled_metrics[system]
        values = (
            display_names[system],
            f"{float(row['rps']):.3f}",
            f"{float(row['brier']):.3f}",
            f"{float(row['log']):.3f}",
        )
        for x, value in zip(col_x, values, strict=True):
            axis.text(
                x,
                y,
                value,
                fontsize=8.3,
                weight="bold" if system == "Selected calibrated ensemble" else "normal",
                color=row_colors[system],
                va="top",
            )
    axis.text(
        right_x + 0.018,
        0.585,
        f"Selected vs raw FuXi: {float(selected_test['improvement_vs_p0_percent']):.2f}% lower RPS · RPSS {100 * float(selected_test['rpss']):+.2f}% vs uniform · neural added value beyond calibrated p₀: {neural_added_value:.2f}%",
        fontsize=7.7,
        color=MUTED,
        va="top",
    )

    _panel(axis, right_x, 0.432, right_w, 0.149, title="Spatial interpretation", color=BLUE)
    _wrapped(
        axis,
        right_x + 0.018,
        0.516,
        f"Improves {100 * spatial[0]['area_fraction_improved']:.1f}% of valid cosine-area-weighted area at D19–25 and {100 * spatial[1]['area_fraction_improved']:.1f}% at D26–32. About 35% worsens—broad, not universal.",
        width=74,
        fontsize=8.2,
    )
    _wrapped(
        axis,
        right_x + 0.018,
        0.464,
        "Green means raw-FuXi RPS minus selected RPS is positive. Lower aggregate RPS establishes improvement; visual smoothness does not.",
        width=74,
        fontsize=7.55,
    )

    _panel(axis, right_x, 0.285, right_w, 0.132, title="Evidence strength", color=GREEN)
    _wrapped(
        axis,
        right_x + 0.018,
        0.354,
        "Positive: the selected system beats raw FuXi and uniform probabilities on RPS, Brier, and log score; both lead windows have positive RPSS.",
        width=74,
        fontsize=7.85,
        color=GREEN,
        weight="bold",
    )
    _wrapped(
        axis,
        right_x + 0.018,
        0.317,
        f"Measured gain is modest: neural added value beyond calibrated p₀ is {neural_added_value:.2f}%. Treat this as calibration evidence until ERA5 testing.",
        width=74,
        fontsize=7.55,
    )

    _panel(axis, right_x, 0.086, right_w, 0.185, title="Limitations and next steps", color=ORANGE)
    steps = (
        "Only 16 initializations; 2020 contains four already-opened retrospective cases.",
        "IMERG and all-grid support are not the official ERA5 target or land/aridity mask.",
        "Finalize the completed ERA5 parts with the official mask, then run frozen validation and open the untouched test once.",
        "Scale case count before regional claims or promotion of a multivariable/context model.",
    )
    for index, text in enumerate(steps, start=1):
        y = 0.215 - (index - 1) * 0.034
        axis.text(right_x + 0.021, y, f"{index}.", fontsize=8.0, weight="bold", color=ORANGE, va="top")
        _wrapped(
            axis,
            right_x + 0.045,
            y,
            text,
            width=68,
            fontsize=7.7,
            linespacing=1.12,
        )

    axis.text(
        0.5,
        0.027,
        "Current conclusion: technically promising calibration evidence, not competition-performance evidence.",
        ha="center",
        fontsize=8.6,
        weight="bold",
        color=INK,
    )
    return save_figure(figure, destination, formats)
