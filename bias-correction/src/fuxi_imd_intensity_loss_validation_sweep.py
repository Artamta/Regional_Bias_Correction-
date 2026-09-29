#!/usr/bin/env python3
"""Validation-only intensity-balanced loss ablation for FuXi-to-IMD rainfall.

The experiment locks the width-24 temporal U-Net, physical-recentered anchor,
features, normalization, train years (2002--2017), and blocked validation years
(2018--2019).  Only the loss changes.  Years 2020 onward remain quarantined.

The screen addresses a diagnosed failure mode: a global bias penalty reduces
heavy-rain underprediction by making every cell wetter, thereby increasing
dry/light-rain MAE.  It compares the established bias-aware objective with a
moderate global-bias coefficient, equal-weight rainfall-regime bias, a soft
wet-occurrence Brier term, and robust physical MAE.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

for path in (SOURCE_ROOT, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_bias_aware_validation_sweep as bias  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


RESULTS_ROOT = HERE / "results" / "fuxi_imd_intensity_loss_validation_sweep"
REFERENCE_CONFIGURATION = "recentered_anchor_bias_aware_loss"
INTENSITY_BOUNDS = ((0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, None))
INTENSITY_LABELS = ("dry_lt1", "light_1_5", "moderate_5_10", "heavy_ge10")
WET_THRESHOLD_MM_DAY = 1.0
WET_TEMPERATURE_MM_DAY = 0.5


@dataclass(frozen=True)
class LossCandidate:
    name: str
    label: str
    anchor_kind: str
    loss_kind: str
    loss_coefficients: Mapping[str, float]
    stratified_bias_coefficient: float = 0.0
    wet_brier_coefficient: float = 0.0
    physical_mae_coefficient: float = 0.0
    heavy_rain_threshold_mm_day: float | None = None
    heavy_rain_multiplier: float = 1.0

    @property
    def uses_bias_scale(self) -> bool:
        return True

    @property
    def coefficient_sum(self) -> float:
        return float(sum(self.loss_coefficients.values())) + float(
            self.stratified_bias_coefficient
            + self.wet_brier_coefficient
            + self.physical_mae_coefficient
        )


CANDIDATES = (
    LossCandidate(
        REFERENCE_CONFIGURATION,
        "Recentered anchor + global bias 0.25",
        "physical_recentered",
        "global_bias_025",
        {"smooth_l1": 0.55, "acc": 0.20, "bias": 0.25},
    ),
    LossCandidate(
        "recentered_anchor_moderate_global_bias",
        "Recentered anchor + global bias 0.15",
        "physical_recentered",
        "global_bias_015",
        {"smooth_l1": 0.65, "acc": 0.20, "bias": 0.15},
    ),
    LossCandidate(
        "recentered_anchor_stratified_bias",
        "Recentered anchor + rainfall-regime bias",
        "physical_recentered",
        "stratified_bias",
        {"smooth_l1": 0.60, "acc": 0.20, "bias": 0.05},
        stratified_bias_coefficient=0.15,
    ),
    LossCandidate(
        "recentered_anchor_stratified_bias_wet",
        "Regime bias + wet-occurrence Brier",
        "physical_recentered",
        "stratified_bias_wet",
        {"smooth_l1": 0.55, "acc": 0.20, "bias": 0.05},
        stratified_bias_coefficient=0.15,
        wet_brier_coefficient=0.05,
    ),
    LossCandidate(
        "recentered_anchor_balanced_physical",
        "Regime bias + wet Brier + physical MAE",
        "physical_recentered",
        "balanced_physical",
        {"smooth_l1": 0.45, "acc": 0.20, "bias": 0.05},
        stratified_bias_coefficient=0.15,
        wet_brier_coefficient=0.05,
        physical_mae_coefficient=0.10,
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_candidates(names: str | None) -> tuple[LossCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(item.strip() for item in names.split(",") if item.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def fit_training_intensity_bias_scale(
    truth: np.ndarray,
    valid_mask: np.ndarray,
    area_weights: np.ndarray,
    fit_indices: np.ndarray,
    initializations: np.ndarray,
    *,
    split_name: str,
    minimum_scale: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit lead/IMD-regime mean-rain scales using training years only."""

    if split_name != "train":
        raise ValueError("intensity scales must be fitted on the train split")
    indices = np.asarray(fit_indices, dtype=np.int64)
    years = bias.compact._calendar_years(
        np.asarray(initializations, dtype="datetime64[D]")[indices]
    )
    if not np.all(np.isin(years, bias.TRAIN_YEARS)):
        raise ValueError("intensity scales received non-training years")
    values = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    weights = np.asarray(area_weights, dtype=np.float64)
    if values.shape != valid.shape or weights.shape != values.shape[-2:]:
        raise ValueError("truth, valid mask, and area weights have incompatible shapes")
    scales = np.empty((values.shape[1], len(INTENSITY_BOUNDS)), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for lead in range(values.shape[1]):
        lead_truth = values[indices, lead]
        lead_valid = valid[indices, lead]
        for stratum, (label, bounds) in enumerate(
            zip(INTENSITY_LABELS, INTENSITY_BOUNDS)
        ):
            lower, upper = bounds
            member = lead_valid & (lead_truth >= lower)
            if upper is not None:
                member &= lead_truth < upper
            stratum_weights = weights[None] * member
            denominator = float(stratum_weights.sum(dtype=np.float64))
            if denominator <= 0.0:
                raise ValueError(f"lead {lead + 1} stratum {label} has no training data")
            # IMD is allowed to contain NaN outside valid support.  Mask the
            # values before multiplication because IEEE NaN * 0 remains NaN.
            safe_truth = np.where(member, lead_truth, 0.0)
            mean_truth = float(
                np.sum(safe_truth * weights[None], dtype=np.float64) / denominator
            )
            scale = max(mean_truth, float(minimum_scale))
            scales[lead, stratum] = np.float32(scale)
            rows.append(
                {
                    "lead": lead + 1,
                    "stratum": label,
                    "lower_mm_day": lower,
                    "upper_mm_day": np.inf if upper is None else upper,
                    "training_weight_sum": denominator,
                    "training_cell_case_count": int(member.sum()),
                    "training_mean_truth_mm_day": mean_truth,
                    "normalization_scale_mm_day": scale,
                }
            )
    return scales, pd.DataFrame(rows)


def _worker(
    worker_index: int,
    device: str,
    tasks: Sequence[tuple[LossCandidate, int]],
    prepared: bias.PreparedBiasExperiment,
    intensity_scale: np.ndarray,
    output: Path,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> None:
    try:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        torch.set_num_threads(max(1, allocated // 2))
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        shared = prepared.shared
        observations = bias._observation_data(shared)
        support = shared.weights > 0.0
        anchor = prepared.anchors["physical_recentered"]
        train_data = bias.common.make_dataset(
            shared.train_indices,
            shared.features,
            anchor.target,
            anchor.bias_baseline,
            observations,
            support,
        )
        validation_data = bias.common.make_dataset(
            shared.validation_indices,
            shared.features,
            anchor.target,
            anchor.bias_baseline,
            observations,
            support,
        )
        for candidate, seed in tasks:
            print(
                f"[{device}] {candidate.name}, seed={seed}, loss={candidate.loss_kind}",
                flush=True,
            )
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            (run_directory / "logs").mkdir(parents=True, exist_ok=False)
            (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
            set_deterministic_seed(seed)
            model = bias.compact.build_model(
                bias.MODEL_SPEC,
                shared.features.shape[2],
                shared.mean_to_anomaly_ratio,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result = train_anchored_model(
                model,
                train_data,
                validation_data,
                shared.weights,
                anchor.target_scale,
                bias.LEAD_WEIGHTS,
                candidate.loss_coefficients,
                run_directory,
                seed=seed,
                device=device,
                batch_size=bias.MODEL_SPEC.batch_size,
                max_epochs=2 if smoke else max_epochs,
                patience=1 if smoke else patience,
                learning_rate=bias.MODEL_SPEC.learning_rate,
                weight_decay=bias.MODEL_SPEC.weight_decay,
                smooth_l1_beta=1.0,
                bias_scale=prepared.bias_scale,
                intensity_bias_scale=intensity_scale,
                stratified_bias_coefficient=candidate.stratified_bias_coefficient,
                wet_brier_coefficient=candidate.wet_brier_coefficient,
                physical_mae_coefficient=candidate.physical_mae_coefficient,
                wet_threshold_mm_day=WET_THRESHOLD_MM_DAY,
                wet_temperature_mm_day=WET_TEMPERATURE_MM_DAY,
                num_workers=0,
                use_amp=True,
            )
            residual = predict(
                model,
                shared.features[shared.validation_indices],
                device=device,
                batch_size=32,
                use_amp=True,
            )
            expected_shape = (len(shared.validation_indices), 6, 27, 27)
            if residual.shape != expected_shape or not np.isfinite(residual).all():
                raise ValueError(f"unexpected validation residual {residual.shape}")
            np.save(run_directory / "validation_residual.npy", residual)
            checkpoint = run_directory / "checkpoints" / "best.pt"
            record = {
                "status": "complete",
                "configuration": candidate.name,
                "candidate": asdict(candidate),
                "seed": int(seed),
                "device": device,
                "worker_index": int(worker_index),
                "parameter_count": int(parameter_count),
                "best_epoch_zero_based": int(result.best_epoch),
                "best_epoch_display": int(result.best_epoch + 1),
                "best_validation_objective": float(result.best_validation_loss),
                "elapsed_seconds": float(result.elapsed_seconds),
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": bias.compact.sha256_file(checkpoint),
                "history": str(
                    (run_directory / "logs" / "training_history.csv").relative_to(output)
                ),
                "validation_residual": str(
                    (run_directory / "validation_residual.npy").relative_to(output)
                ),
                "objective_values_comparable_across_configurations": False,
            }
            (run_directory / "run_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
    except Exception:
        (output / f"worker_{worker_index}_failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


def run_parallel(
    candidates: Sequence[LossCandidate],
    seeds: Sequence[int],
    prepared: bias.PreparedBiasExperiment,
    intensity_scale: np.ndarray,
    output: Path,
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    workers: int,
) -> None:
    tasks = [(candidate, int(seed)) for candidate in candidates for seed in seeds]
    if not tasks or workers < 1:
        raise ValueError("sweep must contain tasks and at least one worker")
    if workers == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _worker(
            0, device, tasks, prepared, intensity_scale, output, max_epochs, patience, smoke
        )
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
        raise RuntimeError(
            f"requested {workers} GPU workers but only {torch.cuda.device_count()} visible"
        )
    groups = [tasks[index::workers] for index in range(workers)]
    context = mp.get_context("spawn")
    processes = []
    for index, group in enumerate(groups):
        process = context.Process(
            target=_worker,
            args=(
                index,
                f"cuda:{index}",
                group,
                prepared,
                intensity_scale,
                output,
                max_epochs,
                patience,
                smoke,
            ),
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append((process.pid, process.exitcode))
    if failures:
        raise RuntimeError(f"one or more sweep workers failed: {failures}")


def save_plots(
    output: Path,
    ranking: pd.DataFrame,
    history: pd.DataFrame,
    case_metrics: pd.DataFrame,
    candidates: Sequence[LossCandidate],
) -> None:
    names = [candidate.name for candidate in candidates]
    selected = case_metrics.loc[
        case_metrics.configuration.isin(names + ["raw_fuxi"])
        & case_metrics.member.isin(["ensemble", "deterministic"])
    ]
    summary = selected.groupby(["configuration", "lead"], as_index=False).agg(
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        bias=("bias", "mean"),
        acc=("acc", "mean"),
    )
    labels = {"raw_fuxi": "Raw FuXi", **dict(zip(ranking.configuration, ranking.label))}
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)
    for name, group in summary.groupby("configuration"):
        style = "--" if name == "raw_fuxi" else "-"
        for axis, metric in zip(axes.flat, ("rmse", "mae", "acc", "bias")):
            axis.plot(
                group.lead,
                group[metric],
                marker="o",
                linewidth=1.5,
                linestyle=style,
                label=labels[name],
            )
    for axis, ylabel in zip(
        axes.flat,
        ("RMSE (mm/day)", "MAE (mm/day)", "Spatial ACC", "Bias (mm/day)"),
    ):
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Lead week")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.22)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    handles, plot_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, plot_labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Intensity-balanced loss screen · blocked 2018–2019 validation")
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.png", dpi=240)
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.pdf")
    plt.close(figure)

    columns = 2
    rows = math.ceil(len(candidates) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.7 * rows), squeeze=False)
    for axis, candidate in zip(axes.flat, candidates):
        subset = history.loc[history.configuration.eq(candidate.name)]
        for seed, frame in subset.groupby("seed"):
            epoch = frame.epoch + 1
            axis.plot(epoch, frame.train_loss, "--", alpha=0.7, label=f"train {seed}")
            axis.plot(epoch, frame.validation_loss, alpha=0.9, label=f"val {seed}")
        axis.set_title(candidate.label, fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Own composite objective")
        axis.grid(alpha=0.22)
    for axis in axes.flat[len(candidates) :]:
        axis.set_visible(False)
    handles, curve_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, curve_labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Training curves; objective magnitudes are not compared across losses")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "figures" / "02_training_validation_curves.png", dpi=240)
    figure.savefig(output / "figures" / "02_training_validation_curves.pdf")
    plt.close(figure)


def output_files(output: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def copy_sources(output: Path) -> None:
    sources = (
        Path(__file__),
        HERE / "docs" / "experiments" / "LOSS_EXPERIMENTS.md",
        SOURCE_ROOT / "fuxi_imd_bias_aware_validation_sweep.py",
        SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
        HERE / "slurm" / "run_imd_intensity_loss_validation_sweep.sbatch",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
    )
    for source in sources:
        if source.exists():
            shutil.copy2(source, output / "code" / source.name)


def write_readme(
    output: Path,
    ranking: pd.DataFrame,
    selection_status: str,
    selected: pd.Series,
    seeds: Sequence[int],
) -> None:
    table_columns = [
        "configuration",
        "pooled_rmse",
        "pooled_mae",
        "pooled_bias",
        "pooled_acc",
        "qualifies",
    ]
    table_frame = ranking[table_columns]
    table_lines = [
        "| Configuration | RMSE | MAE | Bias | ACC | Qualifies |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in table_frame.itertuples(index=False):
        table_lines.append(
            f"| `{row.configuration}` | {row.pooled_rmse:.4f} | "
            f"{row.pooled_mae:.4f} | {row.pooled_bias:.4f} | "
            f"{row.pooled_acc:.4f} | {bool(row.qualifies)} |"
        )
    table = "\n".join(table_lines)
    lines = [
        "# FuXi–IMD intensity-balanced loss validation screen",
        "",
        "Everything except the training loss is fixed: width-24 temporal U-Net, "
        "physical-recentered anchor, features, normalization, optimizer, and split.",
        "Train: 2002–2017. Blocked validation: 2018–2019. No 2020+ predictions or "
        "metrics were created.",
        "",
        f"Seeds: `{list(seeds)}`.",
        f"Selection status: **{selection_status}**.",
        f"Selected configuration: **{selected.label}** "
        f"(`{selected.configuration}`).",
        "",
        table,
        "",
        "Training objective values are not comparable across loss definitions. "
        "Selection uses physical IMD RMSE, MAE, bias and ACC plus the predeclared "
        "year/lead/seed guards in `manifest.json`.",
        "",
        "The one-seed screen is hypothesis generation. A non-reference winner must "
        "be rerun with seeds 42, 43 and 44 before promotion. The independent test "
        "years remain untouched until the loss and stopping policy are frozen.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", help="comma-separated configuration names")
    parser.add_argument("--seeds", help="comma-separated integer seeds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    try:
        candidates = selected_candidates(args.configs)
        seeds = bias.compact.selected_seeds(args.seeds, smoke=args.smoke)
    except ValueError as exc:
        parser.error(str(exc))
    if REFERENCE_CONFIGURATION not in {candidate.name for candidate in candidates}:
        parser.error(f"--configs must include {REFERENCE_CONFIGURATION}")
    if any(not np.isclose(candidate.coefficient_sum, 1.0) for candidate in candidates):
        parser.error("every candidate's loss coefficients must sum to one")

    output = (
        args.output.resolve()
        if args.output
        else (
            RESULTS_ROOT
            / f"{'smoke' if args.smoke else 'screen'}_"
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        ).resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "metrics", "figures", "code"):
        (output / name).mkdir()
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "status": "running",
        "created_utc": utc_now(),
        "purpose": "validation-only fixed-anchor intensity-balanced loss ablation",
        "train_years": list(bias.TRAIN_YEARS),
        "validation_years": list(bias.VALIDATION_YEARS),
        "quarantined_years": list(bias.QUARANTINED_YEARS),
        "test_predictions_created": False,
        "fixed_anchor": "physical_recentered",
        "model": asdict(bias.MODEL_SPEC),
        "candidates": [asdict(candidate) for candidate in candidates],
        "seeds": list(seeds),
        "reference_configuration": REFERENCE_CONFIGURATION,
        "smoke": bool(args.smoke),
        "screening_stage": "smoke" if args.smoke else (
            "one_seed" if len(seeds) == 1 else "multi_seed_confirmation"
        ),
        "ranking_uses_physical_metrics_only": True,
        "objective_values_comparable_across_configurations": False,
        "wet_occurrence": {
            "threshold_mm_day": WET_THRESHOLD_MM_DAY,
            "sigmoid_temperature_mm_day": WET_TEMPERATURE_MM_DAY,
        },
        "intensity_strata_mm_day": [
            [lower, upper] for lower, upper in INTENSITY_BOUNDS
        ],
        "promotion_rule": (
            "a one-seed non-reference winner requires 3-seed confirmation; "
            "independent test remains quarantined until configuration and stopping freeze"
        ),
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(
        json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        prepared, normalization, preparation = bias.prepare_data()
        shared = prepared.shared
        intensity_scale, scale_diagnostics = fit_training_intensity_bias_scale(
            shared.truth,
            shared.valid_mask,
            shared.weights,
            shared.train_indices,
            shared.initializations,
            split_name="train",
        )
        scale_diagnostics.to_csv(
            output / "metrics" / "training_intensity_normalization.csv", index=False
        )
        (output / "normalization.json").write_text(
            json.dumps(bias.compact._json_safe(normalization), indent=2) + "\n",
            encoding="utf-8",
        )
        current = preparation["current_correction"]
        recentered = preparation["recentered_fit"]
        np.savez_compressed(
            output / "models" / "loss_normalization_and_anchor.npz",
            current_lead_month_residual=current.lead_month_residual,
            recentered_lead_month_residual=recentered.correction.lead_month_residual,
            recenter_scalar_by_lead_month=recentered.scalar_by_lead_month,
            target_scale=prepared.anchors["physical_recentered"].target_scale,
            global_bias_scale=prepared.bias_scale,
            intensity_bias_scale=intensity_scale,
        )
        preparation["recenter_diagnostics"].to_csv(
            output / "metrics" / "anchor_training_recenter_diagnostics.csv",
            index=False,
        )
        manifest.update(preparation["metadata"])
        manifest["global_bias_scale_by_lead_mm_day"] = prepared.bias_scale.tolist()
        manifest["intensity_bias_scale_by_lead_and_stratum_mm_day"] = (
            intensity_scale.tolist()
        )
        workers = args.workers
        if workers <= 0:
            workers = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        manifest["workers"] = int(workers)
        (output / "manifest.json").write_text(
            json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )

        run_parallel(
            candidates,
            seeds,
            prepared,
            intensity_scale,
            output,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            workers=workers,
        )
        records, history, case_metrics, _ = bias.aggregate_results(
            output, candidates, seeds, prepared
        )
        seed_guards = bias.build_seed_physical_guards(
            case_metrics,
            candidates,
            reference_configuration=REFERENCE_CONFIGURATION,
        )
        seed_guards.to_csv(output / "metrics" / "seed_physical_guards.csv", index=False)
        ranking = bias.build_physical_ranking(
            records,
            case_metrics,
            candidates,
            reference_configuration=REFERENCE_CONFIGURATION,
            seed_guards=seed_guards,
        )
        ranking.to_csv(output / "metrics" / "ranked_configurations.csv", index=False)
        bias.candidate_vs_own_anchor(case_metrics, candidates).to_csv(
            output / "metrics" / "candidate_vs_own_anchor.csv", index=False
        )
        bias.paired_physical_deltas(
            case_metrics, candidates, REFERENCE_CONFIGURATION
        ).to_csv(
            output / "metrics" / "paired_physical_deltas_vs_reference.csv", index=False
        )
        selection_status, selected = bias.select_configuration(
            ranking, reference_configuration=REFERENCE_CONFIGURATION
        )
        selection = {
            "selection_status": selection_status,
            "selected_configuration": str(selected.configuration),
            "selected_label": str(selected.label),
            "selected_qualifies": bool(selected.qualifies),
            "confirmation_required": len(seeds) == 1
            and str(selected.configuration) != REFERENCE_CONFIGURATION,
        }
        (output / "metrics" / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        save_plots(output, ranking, history, case_metrics, candidates)
        write_readme(output, ranking, selection_status, selected, seeds)
        copy_sources(output)
        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                **selection,
                "software": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda_visible_devices": torch.cuda.device_count(),
                },
            }
        )
        manifest["artifacts"] = {
            str(path.relative_to(output)): bias.compact.sha256_file(path)
            for path in output_files(output)
        }
        (output / "manifest.json").write_text(
            json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        print(ranking.to_string(index=False), flush=True)
        print(f"PASS: intensity loss validation screen complete: {output}", flush=True)
    except Exception:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "failure": traceback.format_exc(),
            }
        )
        (output / "manifest.json").write_text(
            json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
