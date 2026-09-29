#!/usr/bin/env python3
"""Fixed Box-Cox-1p target-transform screen for FuXi-to-IMD correction.

This is a one-factor validation experiment.  It keeps the recentered anchor,
features, width-24 temporal U-Net, bias-aware loss, optimizer, and year blocks
fixed.  Only the monotone target/reconstruction transform power changes from
exact log1p (power zero) through physical rainfall (power one).  No prediction
or metric is produced for 2020 onward.
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
from fuxi_adapter.anchored import (  # noqa: E402
    fit_power_target_scale,
    reconstruct_power_precipitation,
    standardize_power_target,
)
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


RESULTS_ROOT = HERE / "results" / "fuxi_imd_target_transform_validation_sweep"
REFERENCE_CONFIGURATION = "boxcox_power_000"
POWERS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class TransformCandidate:
    name: str
    label: str
    anchor_kind: str
    loss_kind: str
    loss_coefficients: Mapping[str, float]
    rain_transform_power: float
    heavy_rain_threshold_mm_day: float | None = None
    heavy_rain_multiplier: float = 1.0

    @property
    def uses_bias_scale(self) -> bool:
        return True


def _candidate(power: float) -> TransformCandidate:
    code = f"{int(round(100 * power)):03d}"
    if power == 0.0:
        description = "Exact log1p target"
    elif power == 1.0:
        description = "Physical-linear target"
    else:
        description = f"Box-Cox-1p target, power {power:.2f}"
    return TransformCandidate(
        name=f"boxcox_power_{code}",
        label=description,
        anchor_kind="physical_recentered",
        loss_kind=f"boxcox_power_{power:.2f}",
        loss_coefficients=bias.BIAS_AWARE_LOSS,
        rain_transform_power=float(power),
    )


CANDIDATES = tuple(_candidate(power) for power in POWERS)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


@dataclass(frozen=True)
class TransformTarget:
    target: np.ndarray
    target_scale: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_candidates(names: str | None) -> tuple[TransformCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(item.strip() for item in names.split(",") if item.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def prepare_transform_targets(
    prepared: bias.PreparedBiasExperiment,
    candidates: Sequence[TransformCandidate],
) -> tuple[dict[str, TransformTarget], pd.DataFrame]:
    shared = prepared.shared
    baseline = prepared.anchors["physical_recentered"].bias_baseline
    targets: dict[str, TransformTarget] = {}
    rows = []
    for candidate in candidates:
        power = candidate.rain_transform_power
        scale = fit_power_target_scale(
            shared.truth[shared.train_indices],
            baseline[shared.train_indices],
            shared.weights,
            split_name="train",
            valid_mask=shared.valid_mask[shared.train_indices],
            rain_transform_power=power,
        )
        target = standardize_power_target(
            shared.truth,
            baseline,
            scale,
            valid_mask=shared.valid_mask,
            rain_transform_power=power,
        )
        targets[candidate.name] = TransformTarget(target, scale)
        for lead, value in enumerate(scale, start=1):
            rows.append(
                {
                    "configuration": candidate.name,
                    "rain_transform_power": power,
                    "lead": lead,
                    "training_only_residual_rms": float(value),
                }
            )
    return targets, pd.DataFrame(rows)


def _worker(
    worker_index: int,
    device: str,
    tasks: Sequence[tuple[TransformCandidate, int]],
    prepared: bias.PreparedBiasExperiment,
    transform_targets: Mapping[str, TransformTarget],
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
        baseline = prepared.anchors["physical_recentered"].bias_baseline
        datasets = {}
        for candidate in {task[0].name: task[0] for task in tasks}.values():
            transformed = transform_targets[candidate.name]
            datasets[candidate.name] = (
                bias.common.make_dataset(
                    shared.train_indices,
                    shared.features,
                    transformed.target,
                    baseline,
                    observations,
                    support,
                ),
                bias.common.make_dataset(
                    shared.validation_indices,
                    shared.features,
                    transformed.target,
                    baseline,
                    observations,
                    support,
                ),
            )
        for candidate, seed in tasks:
            print(
                f"[{device}] {candidate.name}, seed={seed}, "
                f"power={candidate.rain_transform_power}",
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
            train_data, validation_data = datasets[candidate.name]
            transformed = transform_targets[candidate.name]
            result = train_anchored_model(
                model,
                train_data,
                validation_data,
                shared.weights,
                transformed.target_scale,
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
                rain_transform_power=candidate.rain_transform_power,
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
    candidates: Sequence[TransformCandidate],
    seeds: Sequence[int],
    prepared: bias.PreparedBiasExperiment,
    transform_targets: Mapping[str, TransformTarget],
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
            0,
            device,
            tasks,
            prepared,
            transform_targets,
            output,
            max_epochs,
            patience,
            smoke,
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
                transform_targets,
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
        raise RuntimeError(f"one or more transform workers failed: {failures}")


def _prediction_from_residual(
    residual: np.ndarray,
    candidate: TransformCandidate,
    prepared: bias.PreparedBiasExperiment,
    transform_targets: Mapping[str, TransformTarget],
) -> np.ndarray:
    shared = prepared.shared
    baseline = prepared.anchors["physical_recentered"].bias_baseline[
        shared.validation_indices
    ]
    return reconstruct_power_precipitation(
        baseline,
        residual,
        transform_targets[candidate.name].target_scale,
        valid_mask=shared.valid_mask[shared.validation_indices],
        rain_transform_power=candidate.rain_transform_power,
    )


def aggregate_results(
    output: Path,
    candidates: Sequence[TransformCandidate],
    seeds: Sequence[int],
    prepared: bias.PreparedBiasExperiment,
    transform_targets: Mapping[str, TransformTarget],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    histories = []
    case_frames = []
    intensity_frames = []
    shared = prepared.shared
    validation_truth = shared.truth[shared.validation_indices]
    for candidate in candidates:
        residuals = []
        for seed in seeds:
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            record = json.loads(
                (run_directory / "run_record.json").read_text(encoding="utf-8")
            )
            records.append(record)
            history = pd.read_csv(run_directory / "logs" / "training_history.csv")
            history.insert(0, "loss_kind", candidate.loss_kind)
            history.insert(0, "seed", int(seed))
            history.insert(0, "configuration", candidate.name)
            histories.append(history)
            residual = np.load(run_directory / "validation_residual.npy")
            residuals.append(residual)
            prediction = _prediction_from_residual(
                residual, candidate, prepared, transform_targets
            )
            frame = bias._case_metrics(
                prediction, prepared, predictor=f"{candidate.name}_seed_{seed}"
            )
            frame.insert(0, "member", f"seed_{seed}")
            frame.insert(0, "configuration", candidate.name)
            case_frames.append(frame)
        ensemble = np.mean(residuals, axis=0, dtype=np.float64).astype(np.float32)
        np.save(
            output / "models" / candidate.name / "validation_residual_ensemble.npy",
            ensemble,
        )
        prediction = _prediction_from_residual(
            ensemble, candidate, prepared, transform_targets
        )
        frame = bias._case_metrics(
            prediction, prepared, predictor=f"{candidate.name}_ensemble"
        )
        frame.insert(0, "member", "ensemble")
        frame.insert(0, "configuration", candidate.name)
        case_frames.append(frame)
        intensity_frames.append(
            bias.intensity_strata_metrics(
                prediction,
                validation_truth,
                shared.weights,
                configuration=candidate.name,
            )
        )

    baselines = {
        "raw_fuxi": shared.raw_fuxi[shared.validation_indices],
        "log_anchor": prepared.anchors["log_anchor"].bias_baseline[
            shared.validation_indices
        ],
        "physical_recentered_anchor": prepared.anchors[
            "physical_recentered"
        ].bias_baseline[shared.validation_indices],
    }
    for name, prediction in baselines.items():
        frame = bias._case_metrics(prediction, prepared, predictor=name)
        frame.insert(0, "member", "deterministic")
        frame.insert(0, "configuration", name)
        case_frames.append(frame)
        intensity_frames.append(
            bias.intensity_strata_metrics(
                prediction, validation_truth, shared.weights, configuration=name
            )
        )
    records_frame = pd.json_normalize(records, sep="_")
    history_frame = pd.concat(histories, ignore_index=True)
    case_frame = pd.concat(case_frames, ignore_index=True)
    intensity_frame = pd.concat(intensity_frames, ignore_index=True)
    records_frame.to_csv(output / "metrics" / "run_records.csv", index=False)
    history_frame.to_csv(output / "metrics" / "training_history_tidy.csv", index=False)
    case_frame.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    intensity_frame.to_csv(
        output / "metrics" / "validation_intensity_strata.csv", index=False
    )
    bias.compact.summarize_case_metrics(case_frame).to_csv(
        output / "metrics" / "validation_year_lead_matrix.csv", index=False
    )
    return records_frame, history_frame, case_frame, intensity_frame


def save_plots(
    output: Path,
    ranking: pd.DataFrame,
    history: pd.DataFrame,
    case_metrics: pd.DataFrame,
    candidates: Sequence[TransformCandidate],
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
        for axis, metric in zip(axes.flat, ("rmse", "mae", "acc", "bias")):
            axis.plot(
                group.lead,
                group[metric],
                marker="o",
                linewidth=1.5,
                linestyle="--" if name == "raw_fuxi" else "-",
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
    handles, labels_plot = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels_plot, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Target-transform screen · blocked 2018–2019 validation")
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
    figure.suptitle("Training curves; objectives are not compared across transforms")
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
        HERE / "docs" / "experiments" / "TARGET_TRANSFORM_EXPERIMENTS.md",
        SOURCE_ROOT / "fuxi_imd_bias_aware_validation_sweep.py",
        SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
        HERE / "slurm" / "run_imd_target_transform_validation_sweep.sbatch",
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
    lines = [
        "# FuXi–IMD target-transform validation screen",
        "",
        "Only Box-Cox-1p target curvature changes. Train: 2002–2017; blocked "
        "validation: 2018–2019; no prediction or metric for 2020 onward.",
        "",
        f"Seeds: `{list(seeds)}`.",
        f"Selection status: **{selection_status}**.",
        f"Selected: **{selected.label}** (`{selected.configuration}`).",
        "",
        "| Configuration | RMSE | MAE | Bias | ACC | Qualifies |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| `{row.configuration}` | {row.pooled_rmse:.4f} | "
            f"{row.pooled_mae:.4f} | {row.pooled_bias:.4f} | "
            f"{row.pooled_acc:.4f} | {bool(row.qualifies)} |"
        )
    lines.extend(
        [
            "",
            "Objective magnitudes are not compared across transforms. Ranking "
            "uses reconstructed physical IMD metrics and predeclared guards.",
            "",
            "A one-seed non-log result is hypothesis generation and requires "
            "three-seed confirmation before independent evaluation.",
        ]
    )
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
        "purpose": "validation-only fixed Box-Cox-1p target-transform screen",
        "train_years": list(bias.TRAIN_YEARS),
        "validation_years": list(bias.VALIDATION_YEARS),
        "quarantined_years": list(bias.QUARANTINED_YEARS),
        "test_predictions_created": False,
        "fixed_anchor": "physical_recentered",
        "fixed_loss": dict(bias.BIAS_AWARE_LOSS),
        "powers": list(POWERS),
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
        "learnable_transform_policy": (
            "deferred unless an intermediate fixed power passes every physical guard"
        ),
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(
        json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        prepared, normalization, preparation = bias.prepare_data()
        transform_targets, target_diagnostics = prepare_transform_targets(
            prepared, candidates
        )
        target_diagnostics.to_csv(
            output / "metrics" / "training_transform_target_scales.csv", index=False
        )
        (output / "normalization.json").write_text(
            json.dumps(bias.compact._json_safe(normalization), indent=2) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            output / "models" / "transform_target_scales.npz",
            **{
                f"{candidate.name}_scale": transform_targets[
                    candidate.name
                ].target_scale
                for candidate in candidates
            },
        )
        preparation["recenter_diagnostics"].to_csv(
            output / "metrics" / "anchor_training_recenter_diagnostics.csv",
            index=False,
        )
        manifest.update(preparation["metadata"])
        manifest["transform_target_scale_by_configuration"] = {
            candidate.name: transform_targets[candidate.name].target_scale.tolist()
            for candidate in candidates
        }
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
            transform_targets,
            output,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            workers=workers,
        )
        records, history, case_metrics, _ = aggregate_results(
            output, candidates, seeds, prepared, transform_targets
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
        bias.paired_physical_deltas(
            case_metrics, candidates, REFERENCE_CONFIGURATION
        ).to_csv(
            output / "metrics" / "paired_physical_deltas_vs_log_transform.csv",
            index=False,
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
        print(f"PASS: target-transform screen complete: {output}", flush=True)
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
