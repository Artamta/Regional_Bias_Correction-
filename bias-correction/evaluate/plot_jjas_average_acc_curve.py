#!/usr/bin/env python3
"""Create a clean JJAS-average ACC curve from locked result CSVs only.

The completed 2020--2021 evaluation is exploratory/reused.  IMD is the
verification reference, not a forecast method, so an IMD-versus-IMD curve is
deliberately omitted (it would be identically one and would not measure
forecast skill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_locked_exploratory_acc_figures as locked_acc  # noqa: E402


DEFAULT_RESULT_DIR = locked_acc.DEFAULT_RESULT_DIR
DEFAULT_OUTPUT_DIR = (
    REPOSITORY
    / "presentation"
    / "generated"
    / "jjas_average_acc_curve_exploratory_2020_2021_v1"
)

FIGURE_STEM = "jjas_average_acc_against_imd_by_lead"
METHOD_ORDER = ("raw_fuxi", "log_bias", "corrected")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "log_bias": "Training-only log-bias",
    "corrected": "Corrected Forecast",
}
METHOD_COLORS = {
    "raw_fuxi": "#4B5563",
    "log_bias": "#1976B9",
    "corrected": "#D95F02",
}
METHOD_MARKERS = {"raw_fuxi": "o", "log_bias": "s", "corrected": "D"}

SCOPE = "India · JJAS 2020–2021 · 70 initialization dates per lead"
STATUS = "EXPLORATORY / REUSED · NOT INDEPENDENT CONFIRMATION"
REFERENCE_NOTE = (
    "IMD is the verification reference. No IMD self-curve is drawn because "
    "IMD correlated with itself is exactly 1 and is not forecast skill."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#263442", labelsize=10)
    axis.grid(axis="y", color="#AAB5C0", alpha=0.28, linewidth=0.8)


def plot_jjas_average_acc(data: locked_acc.AccFigureData) -> plt.Figure:
    """Plot mean spatial ACC curves and saved paired corrected-minus-raw CIs."""

    weeks = np.asarray(locked_acc.EXPECTED_LEADS, dtype=int)
    figure = plt.figure(figsize=(13.2, 6.7), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.55, 1.0),
        left=0.075,
        right=0.975,
        bottom=0.22,
        top=0.80,
        wspace=0.19,
    )
    curve_axis = figure.add_subplot(grid[0])
    gain_axis = figure.add_subplot(grid[1])

    for method in METHOD_ORDER:
        values = (
            data.lead_metrics.loc[data.lead_metrics["method"].eq(method)]
            .sort_values("lead")
            .reset_index(drop=True)
        )
        if values["lead"].tolist() != weeks.tolist():
            raise locked_acc.AccFigureContractError(
                f"{method} does not contain the complete W1--W6 ACC curve"
            )
        curve_axis.plot(
            values["lead"],
            values["acc"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=7.3 if method == "corrected" else 6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=3.0 if method == "corrected" else 2.25,
            label=METHOD_LABELS[method],
            zorder=5 if method == "corrected" else 4,
        )

    curve_axis.set_xticks(weeks, [f"W{week}" for week in weeks])
    curve_axis.set_xlim(0.75, 6.25)
    curve_axis.set_ylim(0.0, 0.70)
    curve_axis.set_yticks(np.arange(0.0, 0.71, 0.1))
    curve_axis.set_xlabel("Lead week", fontsize=11)
    curve_axis.set_ylabel("Mean spatial ACC against IMD", fontsize=11)
    curve_axis.set_title(
        "a  JJAS-average forecast skill",
        loc="left",
        fontsize=13,
        weight="bold",
        color="#17232E",
        pad=12,
    )
    curve_axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        borderaxespad=0.3,
    )
    _style_axis(curve_axis)

    improvement = data.corrected_vs_raw.sort_values("lead").reset_index(drop=True)
    if improvement["lead"].tolist() != weeks.tolist():
        raise locked_acc.AccFigureContractError(
            "paired corrected-minus-raw ACC summary is not W1--W6"
        )
    effect = improvement["effect_positive_is_better"].to_numpy(dtype=float)
    lower = improvement["ci95_lower"].to_numpy(dtype=float)
    upper = improvement["ci95_upper"].to_numpy(dtype=float)
    yerr = np.vstack((effect - lower, upper - effect))

    gain_axis.axhspan(0.0, 0.25, color="#2C9C69", alpha=0.045, zorder=0)
    gain_axis.axhline(0.0, color="#475569", linewidth=1.1, linestyle="--", zorder=2)
    gain_axis.errorbar(
        weeks,
        effect,
        yerr=yerr,
        color=METHOD_COLORS["corrected"],
        marker="D",
        markersize=7.0,
        markeredgecolor="white",
        markeredgewidth=0.75,
        linewidth=2.2,
        elinewidth=1.8,
        capsize=5,
        capthick=1.4,
        zorder=5,
    )
    for week, value in zip(weeks, effect, strict=True):
        gain_axis.annotate(
            f"{value:+.3f}",
            (week, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#8A360E",
            weight="semibold",
        )
    gain_axis.set_xticks(weeks, [f"W{week}" for week in weeks])
    gain_axis.set_xlim(0.75, 6.25)
    gain_axis.set_ylim(-0.085, 0.235)
    gain_axis.set_yticks(np.arange(-0.05, 0.201, 0.05))
    gain_axis.set_xlabel("Lead week", fontsize=11)
    gain_axis.set_ylabel("Corrected Forecast − Raw FuXi ACC", fontsize=11)
    gain_axis.set_title(
        "b  Paired ACC improvement",
        loc="left",
        fontsize=13,
        weight="bold",
        color="#17232E",
        pad=12,
    )
    _style_axis(gain_axis)

    pooled = data.pooled_corrected_vs_raw
    gain_axis.text(
        0.97,
        0.06,
        (
            "All-week mean gain\n"
            f"ΔACC = {pooled['effect']:+.3f}\n"
            f"95% percentile interval\n[{pooled['lower']:+.3f}, {pooled['upper']:+.3f}]"
        ),
        transform=gain_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        linespacing=1.35,
        color="#263442",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#C8D0D8",
            "alpha": 0.95,
        },
    )

    figure.suptitle(
        "JJAS-average spatial ACC against IMD",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=20,
        weight="bold",
        color="#17232E",
    )
    figure.text(0.075, 0.905, SCOPE, ha="left", fontsize=10.7, color="#52616E")
    figure.text(
        0.975,
        0.905,
        STATUS,
        ha="right",
        fontsize=8.4,
        weight="bold",
        color="#8C2D2D",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#FFF3F1",
            "edgecolor": "#D9A29C",
            "linewidth": 0.8,
        },
    )
    figure.text(
        0.5,
        0.105,
        REFERENCE_NOTE,
        ha="center",
        va="center",
        fontsize=9.1,
        color="#3E4C59",
    )
    figure.text(
        0.5,
        0.048,
        (
            "Error bars use the saved paired two-stage moving-block 95% percentile "
            f"intervals ({locked_acc.BOOTSTRAP_REPLICATES:,} replicates; "
            f"block length {locked_acc.BOOTSTRAP_BLOCK_LENGTH} starts). "
            "Exploratory descriptive uncertainty only."
        ),
        ha="center",
        va="center",
        fontsize=8.3,
        color="#66727E",
    )
    return figure


def _save_figure(figure: plt.Figure, staging: Path) -> list[Path]:
    outputs: list[Path] = []
    try:
        for suffix, extra in ((".png", {"dpi": 360}), (".pdf", {})):
            target = staging / f"{FIGURE_STEM}{suffix}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{FIGURE_STEM}.", suffix=suffix, dir=staging
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                figure.savefig(
                    temporary,
                    format=suffix[1:],
                    facecolor="white",
                    bbox_inches="tight",
                    metadata={
                        "Title": "JJAS-average spatial ACC against IMD",
                        "Creator": "locked CSV-only presentation post-processing",
                    },
                    **extra,
                )
                os.replace(temporary, target)
                target.chmod(0o644)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(target)
    finally:
        plt.close(figure)
    return outputs


def _write_manifest(
    staging: Path, data: locked_acc.AccFigureData, outputs: list[Path]
) -> None:
    lead_values: dict[str, Any] = {}
    for method in METHOD_ORDER:
        group = data.lead_metrics.loc[
            data.lead_metrics["method"].eq(method)
        ].sort_values("lead")
        lead_values[METHOD_LABELS[method]] = [float(value) for value in group["acc"]]
    improvement = data.corrected_vs_raw.sort_values("lead")
    payload = {
        "schema_name": "jjas_average_acc_presentation_figure",
        "schema_version": 1,
        "evaluation_scope": (
            "India JJAS 2020-2021 exploratory/reused locked hindcasts; "
            "not independent confirmation"
        ),
        "imd_role": "verification reference; IMD self-curve deliberately omitted",
        "method_labels": METHOD_LABELS,
        "source_arrays_opened": False,
        "metrics_recomputed": False,
        "model_refit": False,
        "input_contract": "three completed locked CSV result tables only",
        "lead_weeks": list(locked_acc.EXPECTED_LEADS),
        "initializations_per_lead": locked_acc.EXPECTED_CASES,
        "acc_by_lead": lead_values,
        "corrected_minus_raw": {
            "effect": [float(value) for value in improvement["effect_positive_is_better"]],
            "ci95_lower": [float(value) for value in improvement["ci95_lower"]],
            "ci95_upper": [float(value) for value in improvement["ci95_upper"]],
            "all_week_effect": float(data.pooled_corrected_vs_raw["effect"]),
            "all_week_ci95": [
                float(data.pooled_corrected_vs_raw["lower"]),
                float(data.pooled_corrected_vs_raw["upper"]),
            ],
            "interval_language": "saved paired descriptive percentile interval",
        },
        "unused_inference_fields": ["probability", "p-value", "q-value", "FDR"],
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in data.source_paths.items()
        },
        "figures": {path.name: sha256_file(path) for path in outputs},
    }
    (staging / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def generate_figure(
    data: locked_acc.AccFigureData, output_dir: Path
) -> Path:
    """Atomically publish PNG/PDF and provenance to a fresh directory."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh JJAS ACC output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.partial-", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        with plt.rc_context(
            {
                "font.family": "DejaVu Sans",
                "axes.labelcolor": "#263442",
                "text.color": "#263442",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.transparent": False,
            }
        ):
            outputs = _save_figure(plot_jjas_average_acc(data), staging)
        _write_manifest(staging, data, outputs)
        os.replace(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data = locked_acc.load_acc_figure_data(args.result_dir)
    output = generate_figure(data, args.output_dir)
    print(f"PASS: wrote clean JJAS-average ACC figure: {output}")


if __name__ == "__main__":
    main()
