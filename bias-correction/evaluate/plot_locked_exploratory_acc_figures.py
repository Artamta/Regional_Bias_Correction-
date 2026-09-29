#!/usr/bin/env python3
"""Plot honest ACC diagnostics from the locked 2020--2021 result tables.

This is deliberately a read-only post-processing workflow. It reads only the
three completed CSV tables written by the locked evaluation and never imports
or opens forecast, observation, prediction, Zarr, or NPZ arrays. New figures
are published to a separate fresh directory.

The 2020--2021 period is exploratory/reused. These figures must not be
described as an independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = (
    HERE
    / "results"
    / "fuxi_imd_locked_hindcast_evaluation"
    / "physical_full_compact_exploratory_2020_2021_20260812T010224Z"
)
DEFAULT_OUTPUT_DIR = (
    HERE
    / "presentation"
    / "generated"
    / "locked_exploratory_acc_2020_2021_v4"
)

EXPECTED_METHODS = ("raw_fuxi", "log_bias", "corrected")
EXPECTED_LEADS = tuple(range(1, 7))
EXPECTED_YEARS = (2020, 2021)
EXPECTED_STARTS_PER_YEAR = 35
EXPECTED_CASES = EXPECTED_STARTS_PER_YEAR * len(EXPECTED_YEARS)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 13

METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "log_bias": "Training-only log-bias",
    "corrected": "Corrected Forecast",
}
METHOD_COLORS = {
    "raw_fuxi": "#4D4D4D",
    "log_bias": "#0072B2",
    "corrected": "#D55E00",
}
LEAD_COLORS = {
    1: "#0072B2",
    2: "#56B4E9",
    3: "#009E73",
    4: "#E69F00",
    5: "#D55E00",
    6: "#CC79A7",
}

SCOPE_LINE = (
    "India · JJAS 2020–2021 locked exploratory/reused hindcasts · IMD verification"
)
GUARD_LINE = "EXPLORATORY / REUSED · NOT INDEPENDENT CONFIRMATION"

CASE_FILENAME = "test_case_metrics.csv"
LEAD_FILENAME = "test_metrics_by_lead.csv"
BOOTSTRAP_FILENAME = "paired_two_stage_moving_block_bootstrap.csv"

CASE_READ_COLUMNS = ("method", "case_id", "lead", "region", "season", "acc")
LEAD_READ_COLUMNS = ("method", "lead", "acc")
BOOTSTRAP_READ_COLUMNS = (
    "candidate",
    "baseline",
    "scope",
    "metric",
    "effect_positive_is_better",
    "effect_units",
    "ci95_lower",
    "ci95_upper",
)

FIGURE_A_STEM = "01_paired_case_acc_raw_vs_corrected_exploratory_2020_2021"
FIGURE_B_STEM = "02_acc_by_lead_and_paired_gain_exploratory_2020_2021"


class AccFigureContractError(ValueError):
    """Raised when a completed metric table no longer has the locked contract."""


@dataclass(frozen=True)
class AccFigureData:
    """Validated ACC-only views of the three immutable result CSVs."""

    paired_cases: pd.DataFrame
    lead_metrics: pd.DataFrame
    corrected_vs_raw: pd.DataFrame
    pooled_corrected_vs_raw: Mapping[str, float]
    source_paths: Mapping[str, Path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AccFigureContractError(f"{name} lacks required columns: {missing}")


def _metric_paths(result_dir: Path) -> Mapping[str, Path]:
    root = Path(result_dir).expanduser().resolve()
    metrics = root / "metrics" if (root / "metrics").is_dir() else root
    paths = {
        "case_metrics": metrics / CASE_FILENAME,
        "lead_metrics": metrics / LEAD_FILENAME,
        "bootstrap_summary": metrics / BOOTSTRAP_FILENAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked ACC result tables are missing: {missing}")
    return paths


def _validate_case_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "case_id", "lead", "region", "season", "acc"}
    _require_columns(frame, required, CASE_FILENAME)
    case = frame.loc[:, sorted(required)].copy()
    case["method"] = case["method"].astype(str)
    case["case_id"] = pd.to_datetime(case["case_id"], errors="raise").dt.normalize()
    case["lead"] = pd.to_numeric(case["lead"], errors="raise").astype(int)
    case["acc"] = pd.to_numeric(case["acc"], errors="raise").astype(float)

    if set(case["method"]) != set(EXPECTED_METHODS):
        raise AccFigureContractError(
            "case metrics must contain exactly raw_fuxi, log_bias, and corrected"
        )
    if set(case["lead"]) != set(EXPECTED_LEADS):
        raise AccFigureContractError("case metrics must contain exactly W1--W6")
    if set(case["region"].astype(str)) != {"india"}:
        raise AccFigureContractError("case metrics are not exclusively the India region")
    if set(case["season"].astype(str)) != {"ALL"}:
        raise AccFigureContractError("case metrics are not exclusively the locked ALL scope")
    if case.duplicated(["method", "case_id", "lead"]).any():
        raise AccFigureContractError("case metrics contain duplicate method/date/lead rows")
    if not np.isfinite(case["acc"]).all() or not case["acc"].between(-1.0, 1.0).all():
        raise AccFigureContractError("case ACC values must be finite and lie in [-1, 1]")

    dates = pd.DatetimeIndex(sorted(case["case_id"].unique()))
    years = pd.Series(dates.year).value_counts().sort_index().to_dict()
    expected_years = {
        year: EXPECTED_STARTS_PER_YEAR for year in EXPECTED_YEARS
    }
    if len(dates) != EXPECTED_CASES or years != expected_years:
        raise AccFigureContractError(
            "case metrics must contain 35 unique starts in each of 2020 and 2021"
        )
    counts = case.groupby(["method", "lead"], observed=True).size()
    if len(counts) != len(EXPECTED_METHODS) * len(EXPECTED_LEADS) or not (
        counts == EXPECTED_CASES
    ).all():
        raise AccFigureContractError("each method/lead must contain exactly 70 cases")

    paired = case.pivot(
        index=["case_id", "lead"], columns="method", values="acc"
    ).reset_index()
    paired.columns.name = None
    if paired[list(EXPECTED_METHODS)].isna().any().any():
        raise AccFigureContractError("ACC case pairs are incomplete")
    paired["year"] = paired["case_id"].dt.year.astype(int)
    paired["delta_corrected_raw"] = paired["corrected"] - paired["raw_fuxi"]
    return paired.sort_values(["case_id", "lead"]).reset_index(drop=True)


def _validate_lead_metrics(
    frame: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    required = {"method", "lead", "acc"}
    _require_columns(frame, required, LEAD_FILENAME)
    lead = frame.loc[:, sorted(required)].copy()
    lead["method"] = lead["method"].astype(str)
    lead["lead"] = pd.to_numeric(lead["lead"], errors="raise").astype(int)
    lead["acc"] = pd.to_numeric(lead["acc"], errors="raise").astype(float)
    if (
        set(lead["method"]) != set(EXPECTED_METHODS)
        or set(lead["lead"]) != set(EXPECTED_LEADS)
        or lead.duplicated(["method", "lead"]).any()
        or len(lead) != len(EXPECTED_METHODS) * len(EXPECTED_LEADS)
    ):
        raise AccFigureContractError("lead summary is not the complete 3-method × 6-lead table")
    if not np.isfinite(lead["acc"]).all() or not lead["acc"].between(-1.0, 1.0).all():
        raise AccFigureContractError("lead-summary ACC values are invalid")

    expected = (
        paired.melt(
            id_vars=["case_id", "lead"],
            value_vars=list(EXPECTED_METHODS),
            var_name="method",
            value_name="acc",
        )
        .groupby(["method", "lead"], as_index=False, observed=True)["acc"]
        .mean()
    )
    joined = lead.merge(
        expected,
        on=["method", "lead"],
        how="outer",
        suffixes=("_saved", "_from_cases"),
        validate="one_to_one",
    )
    if len(joined) != len(lead) or not np.allclose(
        joined["acc_saved"], joined["acc_from_cases"], rtol=0.0, atol=1.0e-12
    ):
        raise AccFigureContractError(
            "saved lead ACC does not equal the paired initialization-level mean"
        )
    return lead.sort_values(["method", "lead"]).reset_index(drop=True)


def _scope_to_lead(scope: str) -> int:
    value = str(scope)
    if not value.startswith("W") or not value[1:].isdigit():
        raise AccFigureContractError(f"unexpected bootstrap lead scope: {scope!r}")
    return int(value[1:])


def _validate_bootstrap_summary(
    frame: pd.DataFrame, lead_metrics: pd.DataFrame, paired: pd.DataFrame
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    # Deliberately omit probability/p/q/FDR columns. Figure B uses only the
    # saved point effect and paired percentile interval.
    required = {
        "candidate",
        "baseline",
        "scope",
        "metric",
        "effect_positive_is_better",
        "effect_units",
        "ci95_lower",
        "ci95_upper",
    }
    _require_columns(frame, required, BOOTSTRAP_FILENAME)
    selected = frame.loc[
        frame["candidate"].astype(str).eq("corrected")
        & frame["baseline"].astype(str).eq("raw_fuxi")
        & frame["metric"].astype(str).eq("acc"),
        list(required),
    ].copy()
    if len(selected) != 7 or selected["scope"].duplicated().any():
        raise AccFigureContractError(
            "bootstrap summary must have W1--W6 plus ALL_WEEKS ACC rows"
        )
    if set(selected["scope"].astype(str)) != {
        *(f"W{lead}" for lead in EXPECTED_LEADS),
        "ALL_WEEKS",
    }:
        raise AccFigureContractError("bootstrap ACC scopes differ from W1--W6/ALL_WEEKS")
    if set(selected["effect_units"].astype(str)) != {"ACC difference"}:
        raise AccFigureContractError("bootstrap ACC effect is not an ACC difference")
    for column in ("effect_positive_is_better", "ci95_lower", "ci95_upper"):
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
    if not np.isfinite(
        selected[["effect_positive_is_better", "ci95_lower", "ci95_upper"]]
    ).all().all():
        raise AccFigureContractError("bootstrap ACC effects/intervals must be finite")
    if (selected["ci95_lower"] > selected["ci95_upper"]).any():
        raise AccFigureContractError("bootstrap ACC interval bounds are reversed")
    if (
        (selected["effect_positive_is_better"] < selected["ci95_lower"])
        | (selected["effect_positive_is_better"] > selected["ci95_upper"])
    ).any():
        raise AccFigureContractError("bootstrap ACC point effect lies outside its interval")

    lead_rows = selected.loc[selected["scope"].ne("ALL_WEEKS")].copy()
    lead_rows["lead"] = lead_rows["scope"].map(_scope_to_lead).astype(int)
    lead_rows = lead_rows.sort_values("lead").reset_index(drop=True)
    corrected = lead_metrics.loc[
        lead_metrics["method"].eq("corrected"), ["lead", "acc"]
    ].rename(columns={"acc": "corrected_acc"})
    raw = lead_metrics.loc[
        lead_metrics["method"].eq("raw_fuxi"), ["lead", "acc"]
    ].rename(columns={"acc": "raw_acc"})
    lead_rows = lead_rows.merge(corrected, on="lead", validate="one_to_one").merge(
        raw, on="lead", validate="one_to_one"
    )
    expected_effect = lead_rows["corrected_acc"] - lead_rows["raw_acc"]
    if not np.allclose(
        lead_rows["effect_positive_is_better"],
        expected_effect,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AccFigureContractError(
            "bootstrap corrected-minus-raw point effect differs from lead metrics"
        )

    pooled_row = selected.loc[selected["scope"].eq("ALL_WEEKS")].iloc[0]
    pooled_effect = float(paired["delta_corrected_raw"].mean())
    if not np.isclose(
        float(pooled_row["effect_positive_is_better"]),
        pooled_effect,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AccFigureContractError(
            "pooled bootstrap ACC effect differs from paired case metrics"
        )
    pooled = {
        "effect": float(pooled_row["effect_positive_is_better"]),
        "lower": float(pooled_row["ci95_lower"]),
        "upper": float(pooled_row["ci95_upper"]),
    }
    return lead_rows, pooled


def load_acc_figure_data(result_dir: Path) -> AccFigureData:
    """Read and strictly cross-check only the three locked result CSVs."""

    paths = _metric_paths(result_dir)
    case_frame = pd.read_csv(paths["case_metrics"], usecols=CASE_READ_COLUMNS)
    lead_frame = pd.read_csv(paths["lead_metrics"], usecols=LEAD_READ_COLUMNS)
    bootstrap_frame = pd.read_csv(
        paths["bootstrap_summary"], usecols=BOOTSTRAP_READ_COLUMNS
    )
    paired = _validate_case_metrics(case_frame)
    lead_metrics = _validate_lead_metrics(lead_frame, paired)
    improvement, pooled = _validate_bootstrap_summary(
        bootstrap_frame, lead_metrics, paired
    )
    return AccFigureData(
        paired_cases=paired,
        lead_metrics=lead_metrics,
        corrected_vs_raw=improvement,
        pooled_corrected_vs_raw=pooled,
        source_paths=paths,
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#B8C2CC", alpha=0.28, linewidth=0.7)
    axis.tick_params(labelsize=9, colors="#25313C")


def _joint_limits(paired: pd.DataFrame) -> tuple[float, float]:
    values = paired[["raw_fuxi", "corrected"]].to_numpy(dtype=float)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = max(0.2, maximum - minimum)
    lower = max(-1.0, np.floor((minimum - 0.07 * span) * 10.0) / 10.0)
    upper = min(1.0, np.ceil((maximum + 0.07 * span) * 10.0) / 10.0)
    return lower, upper


def _scope_badge(figure: plt.Figure) -> None:
    figure.text(
        0.985,
        0.925,
        GUARD_LINE,
        ha="right",
        va="center",
        fontsize=8.2,
        weight="bold",
        color="#8C2D2D",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "#FFF3F1",
            "edgecolor": "#D9A29C",
            "linewidth": 0.8,
        },
    )


def plot_paired_acc_scatter(data: AccFigureData) -> plt.Figure:
    """Figure A: paired case-level ACC scatter with descriptive marginals."""

    paired = data.paired_cases
    lower, upper = _joint_limits(paired)
    bins = np.linspace(lower, upper, 23)
    figure = plt.figure(figsize=(11.6, 8.7), facecolor="white")
    grid = figure.add_gridspec(
        5,
        6,
        height_ratios=(0.78, 1.0, 1.0, 1.0, 1.0),
        width_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 0.92),
        left=0.085,
        right=0.965,
        bottom=0.18,
        top=0.86,
        hspace=0.06,
        wspace=0.08,
    )
    marginal_raw = figure.add_subplot(grid[0, :5])
    main = figure.add_subplot(grid[1:, :5])
    marginal_corrected = figure.add_subplot(grid[1:, 5])

    raw_values = paired["raw_fuxi"].to_numpy(dtype=float)
    corrected_values = paired["corrected"].to_numpy(dtype=float)
    marginal_raw.hist(
        raw_values,
        bins=bins,
        color=METHOD_COLORS["raw_fuxi"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    marginal_raw.axvline(
        np.median(raw_values), color="#111111", linestyle=":", linewidth=1.2
    )
    marginal_raw.set_xlim(lower, upper)
    marginal_raw.set_ylabel("Count", fontsize=8.5)
    marginal_raw.set_title(
        "Raw ACC marginal · dotted line = median",
        loc="left",
        fontsize=9,
        color="#43505C",
        pad=3,
    )
    marginal_raw.tick_params(axis="x", labelbottom=False)
    _style_axis(marginal_raw)

    marginal_corrected.hist(
        corrected_values,
        bins=bins,
        orientation="horizontal",
        color=METHOD_COLORS["corrected"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    marginal_corrected.axhline(
        np.median(corrected_values), color="#111111", linestyle=":", linewidth=1.2
    )
    marginal_corrected.set_ylim(lower, upper)
    marginal_corrected.set_xlabel("Count", fontsize=8.5)
    marginal_corrected.set_title(
        "Corrected Forecast\nACC marginal",
        fontsize=9,
        color="#43505C",
        pad=5,
    )
    marginal_corrected.tick_params(axis="y", labelleft=False)
    marginal_corrected.spines[["top", "right"]].set_visible(False)
    marginal_corrected.grid(axis="x", color="#B8C2CC", alpha=0.28, linewidth=0.7)

    label_offsets = {
        1: (7, 7),
        2: (7, 7),
        3: (5, 8),
        4: (8, 7),
        5: (-29, -3),
        6: (8, -11),
    }
    for lead in EXPECTED_LEADS:
        group = paired.loc[paired["lead"].eq(lead)]
        main.scatter(
            group["raw_fuxi"],
            group["corrected"],
            s=28,
            color=LEAD_COLORS[lead],
            alpha=0.58,
            edgecolors="white",
            linewidths=0.35,
            label=f"W{lead}",
            zorder=3,
        )
        mean_raw = float(group["raw_fuxi"].mean())
        mean_corrected = float(group["corrected"].mean())
        main.scatter(
            [mean_raw],
            [mean_corrected],
            marker="D",
            s=88,
            color=LEAD_COLORS[lead],
            edgecolors="#1A1A1A",
            linewidths=0.8,
            zorder=6,
        )
        main.annotate(
            f"W{lead}",
            (mean_raw, mean_corrected),
            xytext=label_offsets[lead],
            textcoords="offset points",
            fontsize=7.8,
            weight="bold",
            color="#222222",
            zorder=7,
        )
    main.plot(
        [lower, upper],
        [lower, upper],
        color="#202020",
        linestyle="--",
        linewidth=1.25,
        zorder=2,
        label="1:1",
    )
    main.fill_between(
        [lower, upper],
        [lower, upper],
        [upper, upper],
        color="#009E73",
        alpha=0.025,
        zorder=1,
    )
    main.set_xlim(lower, upper)
    main.set_ylim(lower, upper)
    main.set_aspect("equal", adjustable="box")
    main.set_xlabel("Raw FuXi-S2S spatial ACC against IMD", fontsize=11)
    main.set_ylabel("Corrected Forecast spatial ACC against IMD", fontsize=11)
    main.grid(color="#AAB4BE", alpha=0.27, linewidth=0.7)
    main.spines[["top", "right"]].set_visible(False)
    main.tick_params(labelsize=9.5)
    main.text(
        lower + 0.05 * (upper - lower),
        upper - 0.07 * (upper - lower),
        "Above 1:1 → Corrected Forecast ACC is higher",
        ha="left",
        va="center",
        fontsize=8.7,
        color="#287A5C",
        weight="semibold",
    )

    fraction = float(np.mean(paired["delta_corrected_raw"] > 0.0))
    pooled = data.pooled_corrected_vs_raw
    main.text(
        0.975,
        0.035,
        (
            f"{len(paired):,} paired initialization × lead cases\n"
            f"{fraction:.0%} lie above 1:1 (descriptive)\n"
            f"Pooled mean ΔACC = {pooled['effect']:+.3f}"
        ),
        transform=main.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        linespacing=1.45,
        color="#26333F",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#C7D0D9",
            "alpha": 0.94,
        },
    )
    handles, labels = main.get_legend_handles_labels()
    ordered = [(handle, label) for handle, label in zip(handles, labels) if label != "1:1"]
    figure.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="lower center",
        bbox_to_anchor=(0.48, 0.072),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.3,
        columnspacing=1.15,
    )

    figure.suptitle(
        "Case-level ACC: Corrected Forecast versus raw FuXi-S2S",
        x=0.085,
        y=0.968,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#18222C",
    )
    figure.text(
        0.085,
        0.925,
        SCOPE_LINE,
        ha="left",
        fontsize=10.3,
        color="#52616E",
    )
    figure.text(
        0.5,
        0.018,
        (
            "Each point is one India-area-weighted spatial ACC against IMD "
            "(70 starts per lead); large diamonds are lead means. Scatter, marginals, "
            "fractions, and means are descriptive; temporal/lead dependence is not "
            "treated as independent sampling."
        ),
        ha="center",
        fontsize=8.1,
        color="#52616E",
    )
    _scope_badge(figure)
    return figure


def plot_acc_by_lead(data: AccFigureData) -> plt.Figure:
    """Figure B: ACC curves plus saved paired corrected-minus-raw intervals."""

    figure = plt.figure(figsize=(11.8, 8.2), facecolor="white")
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.55, 1.0),
        left=0.09,
        right=0.97,
        bottom=0.14,
        top=0.86,
        hspace=0.18,
    )
    curve_axis = figure.add_subplot(grid[0])
    gain_axis = figure.add_subplot(grid[1], sharex=curve_axis)
    weeks = np.asarray(EXPECTED_LEADS)

    for method in EXPECTED_METHODS:
        group = data.lead_metrics.loc[
            data.lead_metrics["method"].eq(method)
        ].sort_values("lead")
        curve_axis.plot(
            group["lead"],
            group["acc"],
            color=METHOD_COLORS[method],
            linewidth=2.6,
            marker="o" if method != "corrected" else "D",
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=METHOD_LABELS[method],
            zorder=4 if method == "corrected" else 3,
        )
    curve_axis.set_ylabel("Mean spatial ACC against IMD", fontsize=10.5)
    curve_axis.set_ylim(0.0, 0.72)
    curve_axis.set_yticks(np.arange(0.0, 0.71, 0.1))
    curve_axis.tick_params(axis="x", labelbottom=False)
    curve_axis.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.5,
        columnspacing=1.5,
    )
    _style_axis(curve_axis)
    curve_axis.text(
        0.012,
        0.94,
        "a",
        transform=curve_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
    )
    curve_axis.text(
        0.995,
        0.055,
        (
            "Pooled W1–W6\n"
            f"ΔACC {data.pooled_corrected_vs_raw['effect']:+.3f}  "
            f"[{data.pooled_corrected_vs_raw['lower']:+.3f}, "
            f"{data.pooled_corrected_vs_raw['upper']:+.3f}]"
        ),
        transform=curve_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
        color="#26333F",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#C7D0D9",
            "alpha": 0.95,
        },
    )

    improvement = data.corrected_vs_raw.sort_values("lead")
    effects = improvement["effect_positive_is_better"].to_numpy(dtype=float)
    lower = improvement["ci95_lower"].to_numpy(dtype=float)
    upper = improvement["ci95_upper"].to_numpy(dtype=float)
    gain_axis.axhspan(0.0, 1.0, color="#009E73", alpha=0.035, zorder=0)
    gain_axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=2)
    gain_axis.plot(
        weeks,
        effects,
        color=METHOD_COLORS["corrected"],
        linewidth=1.5,
        alpha=0.72,
        zorder=3,
    )
    for week, effect, low, high in zip(weeks, effects, lower, upper, strict=True):
        wholly_positive = low > 0.0
        gain_axis.errorbar(
            [week],
            [effect],
            yerr=np.asarray([[effect - low], [high - effect]]),
            fmt="o",
            markersize=7.5,
            markerfacecolor=(
                METHOD_COLORS["corrected"] if wholly_positive else "white"
            ),
            markeredgecolor=METHOD_COLORS["corrected"],
            markeredgewidth=1.5,
            ecolor=METHOD_COLORS["corrected"],
            elinewidth=2.0,
            capsize=5,
            capthick=1.4,
            zorder=5,
        )
        gain_axis.annotate(
            f"{effect:+.3f}",
            (week, high),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#6C2C17",
        )
    gain_axis.set_xticks(weeks, [f"W{week}" for week in weeks])
    gain_axis.set_xlabel("Lead week", fontsize=10.5)
    gain_axis.set_ylabel("Corrected Forecast − Raw FuXi-S2S ACC", fontsize=10.5)
    gain_axis.set_ylim(-0.08, 0.235)
    gain_axis.set_yticks(np.arange(-0.05, 0.201, 0.05))
    _style_axis(gain_axis)
    gain_axis.text(
        0.012,
        0.94,
        "b",
        transform=gain_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
    )
    interval_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=METHOD_COLORS["corrected"],
            markerfacecolor=METHOD_COLORS["corrected"],
            linewidth=1.6,
            label="Bootstrap-supported: 95% interval wholly above 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=METHOD_COLORS["corrected"],
            markerfacecolor="white",
            linewidth=1.6,
            label="95% interval includes 0",
        ),
    ]
    gain_axis.legend(
        handles=interval_handles,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.5,
        handlelength=2.0,
    )

    figure.suptitle(
        "ACC by lead and paired gain from Corrected Forecast",
        x=0.09,
        y=0.968,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#18222C",
    )
    figure.text(
        0.09,
        0.925,
        SCOPE_LINE,
        ha="left",
        fontsize=10.3,
        color="#52616E",
    )
    figure.text(
        0.5,
        0.025,
        (
            f"Error bars are the saved paired two-stage moving-block 95% percentile "
            f"intervals ({BOOTSTRAP_REPLICATES:,} replicates; resample years, then "
            f"circular blocks of {BOOTSTRAP_BLOCK_LENGTH} starts; all six leads retained "
            "together). Descriptive exploratory uncertainty only; no independent-confirmation claim."
        ),
        ha="center",
        fontsize=8.15,
        color="#52616E",
    )
    _scope_badge(figure)
    return figure


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    try:
        for suffix, options in ((".png", {"dpi": 360}), (".pdf", {})):
            target = stem.with_suffix(suffix)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=suffix, dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                figure.savefig(
                    temporary,
                    format=suffix.lstrip("."),
                    facecolor="white",
                    bbox_inches="tight",
                    metadata={
                        "Title": stem.name,
                        "Creator": "locked exploratory ACC CSV post-processing",
                    },
                    **options,
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
    output: Path, data: AccFigureData, figures: list[Path]
) -> None:
    payload: dict[str, Any] = {
        "schema_name": "locked_exploratory_acc_presentation_figures",
        "schema_version": 1,
        "evaluation_scope": (
            "2020-2021 exploratory/reused locked hindcasts; not independent confirmation"
        ),
        "genuine_independent_confirmation": False,
        "source_arrays_opened": False,
        "input_contract": "three completed CSV result tables only",
        "cases": EXPECTED_CASES,
        "lead_weeks": list(EXPECTED_LEADS),
        "paired_points": int(len(data.paired_cases)),
        "uncertainty_contract": {
            "quantity": "Corrected Forecast minus Raw FuXi-S2S spatial ACC",
            "interval": "saved paired two-stage moving-block percentile 95%",
            "replicates": BOOTSTRAP_REPLICATES,
            "block_length_initializations": BOOTSTRAP_BLOCK_LENGTH,
            "all_six_leads_retained_together": True,
            "language": (
                "descriptive exploratory uncertainty; interval inclusion/exclusion "
                "of zero is shown without an independent-confirmation claim"
            ),
        },
        "unused_inference_fields": (
            "probability, p-value, q-value, FDR, and binary support columns in the "
            "legacy bootstrap CSV are intentionally not read or used"
        ),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in data.source_paths.items()
        },
        "figures": {
            path.name: sha256_file(path) for path in sorted(figures)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def generate_acc_figure_set(data: AccFigureData, output_dir: Path) -> Path:
    """Atomically publish both figure pairs to a fresh non-source directory."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh ACC figure output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.partial-", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        with plt.rc_context(
            {
                "font.family": "DejaVu Sans",
                "axes.labelcolor": "#26333F",
                "axes.titlecolor": "#18222C",
                "text.color": "#26333F",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.transparent": False,
            }
        ):
            figures = []
            figures.extend(
                _save_figure(
                    plot_paired_acc_scatter(data), staging / FIGURE_A_STEM
                )
            )
            figures.extend(
                _save_figure(plot_acc_by_lead(data), staging / FIGURE_B_STEM)
            )
        _write_manifest(staging, data, figures)
        os.replace(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="completed locked result directory, or its metrics directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="fresh directory for the new PNG/PDF figures",
    )
    args = parser.parse_args()
    data = load_acc_figure_data(args.result_dir)
    output = generate_acc_figure_set(data, args.output_dir)
    print(f"PASS: wrote exploratory/reused ACC figures (not independent): {output}")


if __name__ == "__main__":
    main()
