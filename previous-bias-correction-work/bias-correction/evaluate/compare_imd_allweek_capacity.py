#!/usr/bin/env python
"""Compare the selected 145k and 2.54M FuXi-to-IMD W1-W6 models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def two_stage_indices(
    initializations: np.ndarray,
    *,
    n_resamples: int = 2_000,
    block_length: int = 13,
    seed: int = 42,
) -> np.ndarray:
    """Resample years, then contiguous initialization blocks within each year."""
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    groups = [np.flatnonzero(years == year) for year in np.sort(np.unique(years))]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("each test year must contain the same number of cases")
    cases_per_year = len(groups[0])
    if block_length > cases_per_year:
        raise ValueError("block length exceeds cases per test year")

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(cases_per_year / block_length))
    offsets = np.arange(block_length, dtype=np.int64)
    sampled_years = rng.integers(0, len(groups), (n_resamples, len(groups)))
    sampled = np.empty((n_resamples, len(groups) * cases_per_year), dtype=np.int64)
    for draw in range(n_resamples):
        for slot, year_index in enumerate(sampled_years[draw]):
            starts = rng.integers(
                0, cases_per_year - block_length + 1, size=blocks
            )
            local = (starts[:, None] + offsets).reshape(-1)[:cases_per_year]
            begin = slot * cases_per_year
            sampled[draw, begin : begin + cases_per_year] = groups[year_index][local]
    return sampled


def selected_cases(run: Path, label: str) -> pd.DataFrame:
    path = run / "metrics" / "case_metrics.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[frame.method.eq("selected_model")].copy()
    selected["model"] = label
    return selected


def paired_capacity(
    small: pd.DataFrame,
    large: pd.DataFrame,
) -> pd.DataFrame:
    case_order = sorted(set(small.init) & set(large.init))
    initializations = np.asarray(case_order, dtype="datetime64[D]")
    sampled = two_stage_indices(initializations)
    scopes = {
        "W1-W6": tuple(range(1, 7)),
        **{f"W{week}": (week,) for week in range(1, 7)},
    }
    rows: list[dict[str, object]] = []
    for scope, weeks in scopes.items():
        for metric in ("rmse", "mae", "acc", "pcc"):
            small_values = (
                small.loc[small.lead_week.isin(weeks)]
                .pivot_table(index="init", values=metric, aggfunc="mean")
                .reindex(case_order)[metric]
                .to_numpy(dtype=np.float64)
            )
            large_values = (
                large.loc[large.lead_week.isin(weeks)]
                .pivot_table(index="init", values=metric, aggfunc="mean")
                .reindex(case_order)[metric]
                .to_numpy(dtype=np.float64)
            )
            if not np.isfinite(small_values).all() or not np.isfinite(large_values).all():
                raise ValueError(f"incomplete pairing for {scope} {metric}")
            small_draw = small_values[sampled].mean(axis=1)
            large_draw = large_values[sampled].mean(axis=1)
            if metric in ("acc", "pcc"):
                effect = large_values.mean() - small_values.mean()
                draws = large_draw - small_draw
                units = "difference"
            else:
                effect = 100.0 * (
                    small_values.mean() - large_values.mean()
                ) / small_values.mean()
                draws = 100.0 * (small_draw - large_draw) / small_draw
                units = "percent reduction"
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "small_mean": small_values.mean(),
                    "large_mean": large_values.mean(),
                    "large_minus_small_effect": effect,
                    "ci_lower": np.percentile(draws, 2.5),
                    "ci_upper": np.percentile(draws, 97.5),
                    "effect_units": units,
                    "paired_cases": len(case_order),
                    "bootstrap_draws": len(sampled),
                    "block_length": 13,
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(
    small_run: Path,
    large_run: Path,
    small: pd.DataFrame,
    large: pd.DataFrame,
    paired: pd.DataFrame,
    output: Path,
) -> None:
    large_all = pd.read_csv(large_run / "metrics" / "case_metrics.csv")
    methods = {
        "Raw FuXi": large_all.loc[large_all.method.eq("raw_fuxi")],
        "Log-bias": large_all.loc[large_all.method.eq("log_bias")],
        "145k temporal": small,
        "2.54M temporal": large,
    }
    colors = {
        "Raw FuXi": "#555555",
        "Log-bias": "#CC79A7",
        "145k temporal": "#0072B2",
        "2.54M temporal": "#D55E00",
    }
    markers = {"Raw FuXi": "o", "Log-bias": "s", "145k temporal": "^", "2.54M temporal": "P"}
    weeks = np.arange(1, 7)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for label, frame in methods.items():
        values = frame.groupby("lead_week").rmse.mean().reindex(weeks)
        axes[0].plot(
            weeks,
            values,
            marker=markers[label],
            linewidth=2.3,
            markersize=7,
            color=colors[label],
            label=label,
        )
    axes[0].set_title("(a) RMSE by lead week")
    axes[0].set_xlabel("Lead week")
    axes[0].set_ylabel("RMSE (mm day$^{-1}$)")
    axes[0].set_xticks(weeks, [f"W{week}" for week in weeks])
    axes[0].grid(alpha=0.25)

    log_rmse = methods["Log-bias"].groupby("lead_week").rmse.mean().reindex(weeks)
    x = np.arange(6)
    width = 0.36
    for offset, label in ((-width / 2, "145k temporal"), (width / 2, "2.54M temporal")):
        model_rmse = methods[label].groupby("lead_week").rmse.mean().reindex(weeks)
        skill = 100.0 * (log_rmse - model_rmse) / log_rmse
        axes[1].bar(x + offset, skill, width, color=colors[label], label=label)
    axes[1].axhline(0.0, color="#777777", linewidth=1.0)
    axes[1].set_title("(b) RMSE improvement over log-bias")
    axes[1].set_xlabel("Lead week")
    axes[1].set_ylabel("RMSE reduction (%)")
    axes[1].set_xticks(x, [f"W{week}" for week in weeks])
    axes[1].grid(axis="y", alpha=0.25)

    overall = paired.loc[
        paired.scope.eq("W1-W6") & paired.metric.eq("rmse")
    ].iloc[0]
    fig.suptitle(
        "FuXi-S2S all-week capacity comparison over India\n"
        "IMD verification; reused 2020–2021 test (n=70)",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.84),
    )
    fig.text(
        0.5,
        0.055,
        "Large vs 145k pooled W1–W6 RMSE reduction: "
        f"{overall.large_minus_small_effect:+.2f}% "
        f"(paired 95% CI {overall.ci_lower:+.2f}% to {overall.ci_upper:+.2f}%)",
        ha="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.74, wspace=0.20)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-run", required=True, type=Path)
    parser.add_argument("--large-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    small = selected_cases(args.small_run, "145k temporal")
    large = selected_cases(args.large_run, "2.54M temporal")
    paired = paired_capacity(small, large)
    paired.to_csv(args.output / "paired_capacity_comparison.csv", index=False)
    plot_comparison(
        args.small_run,
        args.large_run,
        small,
        large,
        paired,
        args.output / "capacity_comparison",
    )
    print(paired.loc[paired.scope.eq("W1-W6")].to_string(index=False))


if __name__ == "__main__":
    main()
