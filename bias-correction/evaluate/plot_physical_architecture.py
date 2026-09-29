#!/usr/bin/env python3
"""Render the FuXi--IMD physical temporal U-Net architecture.

The diagram mirrors the implementation in
``fuxi_adapter.validation_sweep_models.FixedCapacityPhysicalTemporalUNet``
and the data/loss contract in ``src/fuxi_imd_compact_validation_sweep.py``.
It intentionally distinguishes six weekly outputs from six independent
output heads: the physical model uses one shared zero-initialized head.

Outputs are written as editable SVG, vector PDF, and a high-resolution PNG.
No trained values are needed, so the rendering is deterministic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "presentation" / "figures" / "physical_temporal_unet_architecture_v2"
)

# Color-blind-safe, print-friendly palette.
INK = "#172B3A"
MUTED = "#526777"
GRID = "#D8E1E8"
NAVY = "#255C85"
NAVY_DARK = "#153F60"
BLUE_FILL = "#E8F2F8"
TEAL = "#198C7A"
TEAL_FILL = "#E4F4EF"
PURPLE = "#7656A8"
PURPLE_FILL = "#F0EAF8"
ORANGE = "#C85B2B"
ORANGE_FILL = "#FBECE5"
GOLD = "#B67A12"
GOLD_FILL = "#FFF4D6"
GREEN = "#2E7D5B"
GREEN_FILL = "#E6F3EC"
GREY_FILL = "#F4F7F9"


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = "white",
    edge: str = GRID,
    linewidth: float = 1.2,
    radius: float = 0.10,
    shadow: bool = False,
    zorder: float = 2.0,
) -> FancyBboxPatch:
    """Add a rounded rectangular panel in data coordinates."""

    if shadow:
        axis.add_patch(
            FancyBboxPatch(
                (x + 0.045, y - 0.045),
                width,
                height,
                boxstyle=f"round,pad=0.015,rounding_size={radius}",
                facecolor="#8293A0",
                edgecolor="none",
                alpha=0.14,
                zorder=zorder - 0.5,
            )
        )
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.35,
    style: str = "-|>",
    connection: str = "arc3,rad=0",
    dashed: bool = False,
    zorder: float = 4.0,
) -> FancyArrowPatch:
    """Add a clean directional connector."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=width,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    axis.add_patch(arrow)
    return arrow


def add_chip(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    edge: str,
    width: float,
) -> None:
    add_box(axis, x, y, width, 0.28, face=face, edge=edge, linewidth=0.9, radius=0.14)
    axis.text(
        x + width / 2,
        y + 0.14,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        color=edge,
        weight="bold",
        zorder=5,
    )


def add_module(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    face: str,
    edge: str,
    title_size: float = 8.0,
    detail_size: float = 6.7,
) -> None:
    add_box(
        axis,
        x,
        y,
        width,
        height,
        face=face,
        edge=edge,
        linewidth=1.25,
        radius=0.08,
        shadow=True,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=INK,
        weight="bold",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=MUTED,
        linespacing=1.15,
        zorder=5,
    )


def draw_input_panel(axis: plt.Axes) -> None:
    add_box(axis, 0.25, 3.02, 3.05, 4.90, face="white", edge=GRID, shadow=True)
    axis.text(0.48, 7.63, "1  Weekly FuXi + static context", fontsize=10.5, weight="bold", color=INK)
    axis.text(
        0.48,
        7.35,
        "51 members · 42 daily leads → 6 non-overlapping weeks",
        fontsize=7.25,
        color=MUTED,
    )

    add_box(axis, 0.46, 5.30, 2.63, 1.86, face=BLUE_FILL, edge=NAVY, linewidth=1.1)
    axis.text(0.64, 6.89, "Fixed backbone channels  C = 11", fontsize=8.5, weight="bold", color=NAVY_DARK)
    axis.text(
        0.64,
        6.62,
        "1  log₁₊ FuXi TP mean     2  log₁₊ TP spread\n"
        "3  log₁₊ IMD calendar climo     4  latitude     5  longitude\n"
        "6  seasonal sin     7  seasonal cos     8  lead week\n"
        "9  IMD support     10  log FuXi−IMD climo anomaly\n"
        "11  FuXi T2m weekly mean",
        fontsize=5.85,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    axis.text(
        0.64,
        5.50,
        "Train-only, lead-wise normalization\nfull 27 × 27 regional context retained",
        fontsize=5.75,
        color=NAVY_DARK,
        style="italic",
    )

    add_box(axis, 0.46, 3.42, 2.63, 1.56, face=TEAL_FILL, edge=TEAL, linewidth=1.1)
    axis.text(0.64, 4.72, "Fixed physical bank  P = 9", fontsize=8.5, weight="bold", color=TEAL)
    axis.text(
        0.64,
        4.46,
        "TCWV · q850 · u850 · v850\n"
        "q850×u850 · q850×v850 · z500\n"
        "mean-sea-level pressure · outgoing longwave radiation",
        fontsize=5.85,
        color=INK,
        va="top",
        linespacing=1.3,
    )
    axis.text(
        0.64,
        3.59,
        "7-day × 51-member means\nmoisture flux formed before averaging",
        fontsize=5.65,
        color=TEAL,
        style="italic",
    )
    axis.text(
        0.47,
        3.14,
        "Input tensor: B × 6 × 38 × 27 × 27. The matched loader also carries\n"
        "18 climatology-bank channels (C12–C29), intentionally not routed here.",
        fontsize=6.05,
        color=MUTED,
        linespacing=1.25,
    )


def draw_physical_adapter(axis: plt.Axes) -> None:
    add_box(axis, 3.50, 3.50, 1.80, 3.98, face="white", edge=TEAL, linewidth=1.25, shadow=True)
    axis.text(3.69, 7.19, "2  Fixed-capacity adapter", fontsize=9.3, weight="bold", color=INK)
    axis.text(3.69, 6.89, "Ablation mask  m ∈ {0,1}⁹", fontsize=7.3, color=MUTED)

    slots_x = [3.75 + 0.15 * index for index in range(9)]
    for index, slot_x in enumerate(slots_x):
        face = TEAL if index in (0, 1, 2, 3, 4, 5, 6, 7, 8) else "white"
        axis.add_patch(Circle((slot_x, 6.56), 0.052, facecolor=face, edgecolor=TEAL, linewidth=0.7, zorder=5))
    axis.text(3.70, 6.25, "+TCWV: 1   ·   moisture/circ.: 6   ·   full: 9", fontsize=5.55, color=TEAL)

    add_module(
        axis,
        3.77,
        5.20,
        1.25,
        0.77,
        "mask × P",
        "inactive slots = 0",
        face=TEAL_FILL,
        edge=TEAL,
        title_size=8.1,
        detail_size=6.0,
    )
    add_arrow(axis, (4.395, 5.18), (4.395, 4.92), color=TEAL)
    add_module(
        axis,
        3.72,
        4.03,
        1.36,
        0.87,
        "1 × 1 Conv",
        "9 → 11 · no bias\nweights initialized to 0",
        face=TEAL_FILL,
        edge=TEAL,
        title_size=8.1,
        detail_size=5.9,
    )

    plus = Circle((4.40, 3.73), 0.13, facecolor="white", edgecolor=TEAL, linewidth=1.25, zorder=5)
    axis.add_patch(plus)
    axis.text(4.40, 3.73, "+", ha="center", va="center", fontsize=12, color=TEAL, weight="bold", zorder=6)
    add_arrow(axis, (4.40, 4.01), (4.40, 3.88), color=TEAL)
    axis.text(3.70, 3.54, "effective 11-ch input = X₁₁ + Conv(m ⊙ P)", fontsize=6.15, color=INK)

    # Backbone route into the addition node and physical route from the input panel.
    add_arrow(axis, (3.10, 6.12), (3.69, 3.75), color=NAVY, connection="arc3,rad=0.18")
    add_arrow(axis, (3.10, 4.24), (3.76, 5.59), color=TEAL, connection="arc3,rad=-0.14")


def draw_unet(axis: plt.Axes) -> None:
    add_box(axis, 5.53, 3.02, 7.35, 4.90, face="white", edge=GRID, shadow=True)
    axis.text(5.76, 7.63, "3  Shared spatial U-Net + bottleneck temporal attention", fontsize=10.5, weight="bold", color=INK)
    axis.text(
        5.76,
        7.34,
        "Six lead maps share every convolution; lead weeks mix only at matched bottleneck grid locations",
        fontsize=7.15,
        color=MUTED,
    )

    # Main U-shape modules.
    add_module(axis, 5.82, 5.68, 1.20, 0.90, "Encoder 1", "11 → 24\n27 × 27", face=BLUE_FILL, edge=NAVY)
    add_module(axis, 7.20, 4.79, 1.18, 0.90, "Encoder 2", "24 → 48\n13 × 13", face=BLUE_FILL, edge=NAVY)
    add_module(axis, 8.63, 3.76, 1.54, 1.22, "Temporal mixer", "96-d tokens · 6 × 6\n1 Transformer layer\n4-head MHSA · FFN 192", face=PURPLE_FILL, edge=PURPLE, title_size=8.25, detail_size=6.25)
    add_module(axis, 10.43, 4.79, 1.18, 0.90, "Decoder 2", "96 ⊕ 48 → 48\n13 × 13", face=BLUE_FILL, edge=NAVY)
    add_module(axis, 11.77, 5.68, 0.87, 0.90, "Decoder 1", "48 ⊕ 24 → 24\n27 × 27", face=BLUE_FILL, edge=NAVY, title_size=7.5, detail_size=5.9)

    add_arrow(axis, (5.30, 3.73), (5.82, 6.13), color=NAVY, connection="arc3,rad=-0.16")
    add_arrow(axis, (7.02, 6.05), (7.20, 5.24), color=NAVY)
    axis.text(7.03, 5.60, "2×2\nmax pool", fontsize=5.7, color=MUTED, ha="center")
    add_arrow(axis, (8.38, 5.14), (8.63, 4.37), color=NAVY)
    axis.text(8.40, 4.77, "2×2\nmax pool", fontsize=5.7, color=MUTED, ha="center")
    add_arrow(axis, (10.17, 4.37), (10.43, 5.22), color=NAVY)
    add_arrow(axis, (11.61, 5.22), (11.77, 6.11), color=NAVY)

    # Skip connections.
    add_arrow(axis, (6.55, 6.59), (12.20, 6.59), color=TEAL, width=1.15, connection="arc3,rad=-0.16")
    add_arrow(axis, (7.78, 5.70), (11.02, 5.70), color=TEAL, width=1.15, connection="arc3,rad=-0.14")
    axis.text(9.36, 6.98, "skip E1", fontsize=6.25, color=TEAL, weight="bold", ha="center")
    axis.text(9.40, 5.92, "skip E2", fontsize=6.25, color=TEAL, weight="bold", ha="center")

    # Exact block/attention contract callouts.
    add_box(axis, 5.83, 3.29, 2.31, 0.92, face=GREY_FILL, edge=GRID, linewidth=0.9)
    axis.text(5.98, 4.00, "Each ConvBlock", fontsize=7.3, color=INK, weight="bold")
    axis.text(
        5.98,
        3.77,
        "2 × [3×3 Conv → GroupNorm → SiLU]\nthen Dropout2d p = 0.30",
        fontsize=6.25,
        color=MUTED,
        va="top",
        linespacing=1.25,
    )
    add_box(axis, 10.41, 3.29, 2.22, 0.92, face=GREY_FILL, edge=GRID, linewidth=0.9)
    axis.text(10.57, 4.00, "Temporal tokenization", fontsize=7.3, color=INK, weight="bold")
    axis.text(
        10.57,
        3.77,
        "B·6·6 spatial sequences × length 6\nlearned lead position · pre-LayerNorm",
        fontsize=6.15,
        color=MUTED,
        va="top",
        linespacing=1.25,
    )


def draw_output_panel(axis: plt.Axes) -> None:
    add_box(axis, 13.10, 3.02, 2.65, 4.90, face="white", edge=GRID, shadow=True)
    axis.text(13.31, 7.63, "4  Anchored weekly rainfall", fontsize=10.0, weight="bold", color=INK)

    add_module(
        axis,
        13.37,
        6.38,
        2.10,
        0.75,
        "Shared residual head",
        "1 × 1 Conv  24 → 1 · zero initialized",
        face=ORANGE_FILL,
        edge=ORANGE,
        title_size=8.6,
        detail_size=6.25,
    )
    add_arrow(axis, (12.64, 6.13), (13.36, 6.75), color=ORANGE, connection="arc3,rad=-0.10")
    axis.text(
        14.42,
        6.08,
        "Δz = [Δz₁, …, Δz₆]\nB × 6 × 27 × 27",
        ha="center",
        fontsize=7.0,
        color=ORANGE,
        weight="bold",
        linespacing=1.25,
    )
    axis.plot(
        (15.47, 15.62, 15.62),
        (6.75, 6.75, 4.14),
        color=ORANGE,
        linewidth=1.0,
        linestyle="--",
        zorder=4,
    )
    add_arrow(axis, (15.62, 4.14), (15.53, 4.14), color=ORANGE, width=1.0)

    add_box(axis, 13.36, 4.87, 2.12, 0.89, face=GOLD_FILL, edge=GOLD, linewidth=1.0)
    axis.text(14.42, 5.52, "Training-only log-bias anchor  B", ha="center", fontsize=7.5, color=INK, weight="bold")
    axis.text(14.42, 5.22, "FuXi TP mean · lead × calendar-month\nshrinkage = 10", ha="center", fontsize=6.2, color=MUTED, linespacing=1.2)
    add_arrow(axis, (14.42, 4.86), (14.42, 4.61), color=GOLD)

    add_box(axis, 13.31, 3.70, 2.22, 0.87, face=ORANGE_FILL, edge=ORANGE, linewidth=1.1)
    axis.text(14.42, 4.33, "Physical reconstruction", ha="center", fontsize=8.0, color=INK, weight="bold")
    axis.text(
        14.42,
        4.00,
        "ŷₗ = expm1[ log1p(Bₗ) + sₗ Δzₗ ]\nnonnegative weekly mean · mm day⁻¹",
        ha="center",
        fontsize=6.35,
        color=MUTED,
        linespacing=1.2,
    )

    axis.text(13.32, 3.36, "One map per lead:  W1   W2   W3   W4   W5   W6", fontsize=6.65, color=INK, weight="bold")
    for index in range(6):
        x = 13.47 + index * 0.34
        shade = 0.24 + 0.09 * index
        axis.add_patch(
            FancyBboxPatch(
                (x, 3.12),
                0.26,
                0.16,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                facecolor=ORANGE,
                edgecolor=ORANGE,
                alpha=shade,
                linewidth=0.6,
                zorder=4,
            )
        )


def draw_training_contract(axis: plt.Axes) -> None:
    add_box(axis, 0.25, 0.23, 15.50, 2.48, face="white", edge=GRID, shadow=True)
    axis.text(0.48, 2.42, "Leakage-safe learning and evaluation contract", fontsize=10.3, weight="bold", color=INK)

    # Chronological split.
    axis.text(0.50, 2.08, "Chronological split", fontsize=7.8, weight="bold", color=INK)
    split_y = 1.27
    split_h = 0.58
    widths = (4.10, 1.70, 1.72)
    starts = (0.50, 4.63, 6.36)
    colors = ((BLUE_FILL, NAVY), (GOLD_FILL, GOLD), (GREEN_FILL, GREEN))
    titles = (
        "TRAIN · 2002–2017 · 560 inits",
        "VALIDATE · 2018–2019 · 70",
        "REUSED CHECK · 2020–21 · 70",
    )
    details = (
        "fit normalization, climatology, anchor, target scale, weights",
        "select architecture / epoch only",
        "locked exploratory evaluation",
    )
    for start, width, (face, edge), title, detail in zip(starts, widths, colors, titles, details):
        add_box(axis, start, split_y, width, split_h, face=face, edge=edge, linewidth=1.0, radius=0.07)
        axis.text(start + 0.10, split_y + 0.39, title, fontsize=6.85, color=edge, weight="bold")
        axis.text(start + 0.10, split_y + 0.15, detail, fontsize=5.55, color=MUTED)
    axis.text(
        0.50,
        0.88,
        "JJAS starts · 35 initializations year⁻¹ · 2020–21 excluded during selection, but not a fresh independent period",
        fontsize=6.25,
        color=MUTED,
        style="italic",
    )

    # Target and objective.
    add_box(axis, 8.38, 0.63, 7.08, 1.38, face=GREY_FILL, edge=GRID, linewidth=1.0)
    axis.text(8.61, 1.78, "IMD target + area-weighted composite objective", fontsize=8.2, weight="bold", color=INK)
    axis.text(
        8.61,
        1.47,
        "Target: standardized { log1p(IMD weekly mean) − log1p(B) } on 171 supported India cells",
        fontsize=6.55,
        color=MUTED,
    )
    axis.text(
        8.61,
        1.13,
        "L = 0.75 · SmoothL1(Δz, z*)   +   0.20 · [1 − spatial ACC]   +   0.05 · normalized mean-bias²",
        fontsize=7.0,
        color=INK,
        weight="bold",
    )
    axis.text(
        8.61,
        0.82,
        "Equal lead weights (1/6 each) · IMD/cos-latitude spatial weights · all W1–W6 optimized jointly",
        fontsize=6.3,
        color=MUTED,
    )


def build_figure() -> plt.Figure:
    """Construct the complete 16:9 architecture diagram."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "fuxi-imd-physical-architecture-v1",
        }
    )
    figure = plt.figure(figsize=(16, 9), facecolor="#FAFCFD")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    axis.text(
        0.30,
        8.67,
        "Physics-guided temporal residual correction of FuXi-S2S rainfall over India",
        fontsize=18.5,
        weight="bold",
        color=INK,
        va="center",
    )
    axis.text(
        0.31,
        8.31,
        "Fixed-capacity physical ablation · 323,116 parameters · six weekly leads · spatial encoder–decoder · temporal attention · IMD supervision",
        fontsize=9.0,
        color=MUTED,
        va="center",
    )
    add_chip(axis, 12.60, 8.47, "27 × 27 regional grid", face=BLUE_FILL, edge=NAVY, width=1.45)
    add_chip(axis, 14.18, 8.47, "W1–W6 jointly", face=PURPLE_FILL, edge=PURPLE, width=1.30)

    draw_input_panel(axis)
    draw_physical_adapter(axis)
    draw_unet(axis)
    draw_output_panel(axis)
    draw_training_contract(axis)

    # Figure-level provenance note: compact enough for a slide but explicit for paper use.
    axis.text(
        15.72,
        0.10,
        "Architecture reflects current implementation; shared residual head, not six independent heads.",
        fontsize=5.2,
        color="#70818E",
        ha="right",
        va="bottom",
    )
    return figure


def save_figure(figure: plt.Figure, output_stem: Path, dpi: int) -> tuple[Path, ...]:
    """Write SVG, PDF, and high-DPI PNG outputs."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg = output_stem.with_suffix(".svg")
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    common = {"Creator": "plot_physical_architecture.py", "Title": "FuXi-IMD physical temporal U-Net"}
    figure.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.08, metadata={**common, "Date": None})
    figure.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={**common, "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        png,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=figure.get_facecolor(),
        metadata={"Software": "plot_physical_architecture.py"},
    )
    return svg, pdf, png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path without extension",
    )
    parser.add_argument("--dpi", type=int, default=360, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 150:
        raise SystemExit("--dpi must be at least 150 for presentation output")
    # Avoid time-varying PDF metadata in Matplotlib versions that honor this.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    figure = build_figure()
    outputs = save_figure(figure, args.output_stem.resolve(), args.dpi)
    plt.close(figure)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
