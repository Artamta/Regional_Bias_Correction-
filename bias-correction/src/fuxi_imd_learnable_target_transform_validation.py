#!/usr/bin/env python3
"""Apples-to-apples learned Box--Cox target experiment for FuXi-to-IMD.

Three arms use the identical training loop: exact log1p, fixed power 0.25, and
one bounded learned global power initialized at 0.25.  During optimization the
full 2002--2017 scale is recomputed for every batch with its lambda derivative;
validation refits that scale from training data with gradients off. Checkpoints
are selected by physical IMD-space RMSE on blocked 2018--2019 validation. Years
2020 onward remain quarantined and receive no prediction or metric.
"""

from __future__ import annotations

import argparse
import json
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
from fuxi_adapter.anchored import reconstruct_power_precipitation  # noqa: E402
from fuxi_adapter.learnable_boxcox import (  # noqa: E402
    GlobalBoxCoxTransform,
    LearnableTransformSequenceDataset,
    train_learnable_boxcox_model,
)
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402


RESULTS_ROOT = HERE / "results" / "fuxi_imd_learnable_target_transform_validation"
REFERENCE_CONFIGURATION = "exact_log_control"
CHECKPOINT_SELECTION_METRIC = "validation_physical_rmse"
TRANSFORM_LEARNING_RATE_RATIO = 0.1
TRANSFORM_WARMUP_EPOCHS = 5
LEARNED_POWER_BOUNDARY_MARGIN = 0.02


@dataclass(frozen=True)
class LearnableTransformCandidate:
    name: str
    label: str
    initial_power: float
    learnable: bool
    minimum_power: float = 0.0
    maximum_power: float = 0.5
    anchor_kind: str = "physical_recentered"
    loss_kind: str = "bias_aware_boxcox"
    loss_coefficients: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.loss_coefficients is None:
            object.__setattr__(self, "loss_coefficients", dict(bias.BIAS_AWARE_LOSS))


CANDIDATES = (
    LearnableTransformCandidate(
        name=REFERENCE_CONFIGURATION,
        label="Exact log1p control",
        initial_power=0.0,
        learnable=False,
        loss_kind="fixed_boxcox_power_0.00",
    ),
    LearnableTransformCandidate(
        name="fixed_power_025_control",
        label="Fixed Box-Cox power 0.25",
        initial_power=0.25,
        learnable=False,
        loss_kind="fixed_boxcox_power_0.25",
    ),
    LearnableTransformCandidate(
        name="learned_global_power_000_050",
        label="Learned global Box-Cox power",
        initial_power=0.25,
        learnable=True,
        loss_kind="learned_boxcox_global_0.00_0.50",
    ),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_candidates(names: str | None) -> tuple[LearnableTransformCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(part.strip() for part in names.split(",") if part.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def _datasets(
    prepared: bias.PreparedBiasExperiment,
) -> tuple[LearnableTransformSequenceDataset, LearnableTransformSequenceDataset]:
    shared = prepared.shared
    baseline = prepared.anchors["physical_recentered"].bias_baseline

    def make(indices: np.ndarray) -> LearnableTransformSequenceDataset:
        return LearnableTransformSequenceDataset(
            shared.features[indices],
            baseline[indices],
            shared.truth[indices],
            shared.climatology[indices],
            shared.valid_mask[indices],
        )

    return make(shared.train_indices), make(shared.validation_indices)


def _worker(
    worker_index: int,
    device: str,
    tasks: Sequence[tuple[LearnableTransformCandidate, int]],
    prepared: bias.PreparedBiasExperiment,
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
        train_dataset, validation_dataset = _datasets(prepared)
        shared = prepared.shared
        validation_indices = shared.validation_indices
        validation_baseline = prepared.anchors[
            "physical_recentered"
        ].bias_baseline[validation_indices]
        for candidate, seed in tasks:
            print(f"[{device}] {candidate.name}, seed={seed}", flush=True)
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            run_directory.mkdir(parents=True, exist_ok=False)
            set_deterministic_seed(seed)
            model = bias.compact.build_model(
                bias.MODEL_SPEC,
                shared.features.shape[2],
                shared.mean_to_anomaly_ratio,
            )
            transform = GlobalBoxCoxTransform(
                candidate.initial_power,
                minimum_power=candidate.minimum_power,
                maximum_power=candidate.maximum_power,
                learnable=candidate.learnable,
            )
            model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
            transform_parameter_count = sum(
                parameter.numel() for parameter in transform.parameters()
            )
            result = train_learnable_boxcox_model(
                model,
                transform,
                train_dataset,
                validation_dataset,
                shared.weights,
                bias.LEAD_WEIGHTS,
                candidate.loss_coefficients or bias.BIAS_AWARE_LOSS,
                run_directory,
                seed=seed,
                device=device,
                batch_size=bias.MODEL_SPEC.batch_size,
                max_epochs=2 if smoke else max_epochs,
                patience=1 if smoke else patience,
                learning_rate=bias.MODEL_SPEC.learning_rate,
                transform_learning_rate=(
                    TRANSFORM_LEARNING_RATE_RATIO * bias.MODEL_SPEC.learning_rate
                ),
                transform_warmup_epochs=TRANSFORM_WARMUP_EPOCHS,
                weight_decay=bias.MODEL_SPEC.weight_decay,
                bias_scale=prepared.bias_scale,
                num_workers=0,
                use_amp=True,
            )
            residual = predict(
                model,
                shared.features[validation_indices],
                device=device,
                batch_size=32,
                use_amp=True,
            )
            expected_shape = (len(validation_indices), 6, 27, 27)
            if residual.shape != expected_shape or not np.isfinite(residual).all():
                raise ValueError(f"unexpected validation residual {residual.shape}")
            physical_prediction = reconstruct_power_precipitation(
                validation_baseline,
                residual,
                result.best_target_scale,
                rain_transform_power=result.best_power,
                valid_mask=shared.valid_mask[validation_indices],
            )
            np.save(run_directory / "validation_residual.npy", residual)
            np.save(run_directory / "validation_physical_prediction.npy", physical_prediction)
            model_checkpoint = run_directory / "checkpoints" / "best.pt"
            transform_checkpoint = run_directory / "checkpoints" / "best_transform.pt"
            record = {
                "status": "complete",
                "configuration": candidate.name,
                "candidate": asdict(candidate),
                "seed": int(seed),
                "device": device,
                "worker_index": int(worker_index),
                "parameter_count": int(model_parameter_count),
                "transform_parameter_count": int(transform_parameter_count),
                "total_parameter_count": int(
                    model_parameter_count + transform_parameter_count
                ),
                "best_epoch_zero_based": int(result.best_epoch),
                "best_epoch_display": int(result.best_epoch + 1),
                "best_validation_objective": float(
                    result.best_validation_physical_rmse
                ),
                "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
                "best_power": float(result.best_power),
                "best_target_scale": result.best_target_scale.tolist(),
                "elapsed_seconds": float(result.elapsed_seconds),
                "checkpoint": str(model_checkpoint.relative_to(output)),
                "transform_checkpoint": str(transform_checkpoint.relative_to(output)),
                "checkpoint_sha256": bias.compact.sha256_file(model_checkpoint),
                "transform_checkpoint_sha256": bias.compact.sha256_file(
                    transform_checkpoint
                ),
                "history": str(
                    (run_directory / "logs" / "training_history.csv").relative_to(output)
                ),
                "validation_residual": str(
                    (run_directory / "validation_residual.npy").relative_to(output)
                ),
                "validation_physical_prediction": str(
                    (run_directory / "validation_physical_prediction.npy").relative_to(
                        output
                    )
                ),
                "objective_values_comparable_across_configurations": True,
                "transform_scale_fit_split": "train",
                "validation_used_for_transform_scale": False,
                "training_scale_gradient": "full_ds_dlambda",
                "training_scale_refresh": "every_optimization_batch",
                "transform_warmup_epochs": TRANSFORM_WARMUP_EPOCHS,
                "transform_learning_rate_ratio": TRANSFORM_LEARNING_RATE_RATIO,
            }
            (run_directory / "run_record.json").write_text(
                json.dumps(bias.compact._json_safe(record), indent=2) + "\n",
                encoding="utf-8",
            )
    except Exception:
        (output / f"worker_{worker_index}_failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


def run_parallel(
    candidates: Sequence[LearnableTransformCandidate],
    seeds: Sequence[int],
    prepared: bias.PreparedBiasExperiment,
    output: Path,
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    workers: int,
) -> None:
    tasks = [(candidate, int(seed)) for candidate in candidates for seed in seeds]
    if not tasks or workers < 1:
        raise ValueError("experiment requires tasks and at least one worker")
    if workers == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _worker(
            0,
            device,
            tasks,
            prepared,
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


def aggregate_results(
    output: Path,
    candidates: Sequence[LearnableTransformCandidate],
    seeds: Sequence[int],
    prepared: bias.PreparedBiasExperiment,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, histories, case_frames, intensity_frames = [], [], [], []
    shared = prepared.shared
    validation_truth = shared.truth[shared.validation_indices]
    for candidate in candidates:
        predictions = []
        for seed in seeds:
            run_directory = output / "models" / candidate.name / f"seed_{seed}"
            record = json.loads(
                (run_directory / "run_record.json").read_text(encoding="utf-8")
            )
            records.append(record)
            history = pd.read_csv(run_directory / "logs" / "training_history.csv")
            history.insert(0, "seed", int(seed))
            history.insert(0, "configuration", candidate.name)
            histories.append(history)
            prediction = np.load(run_directory / "validation_physical_prediction.npy")
            predictions.append(prediction)
            frame = bias._case_metrics(
                prediction, prepared, predictor=f"{candidate.name}_seed_{seed}"
            )
            frame = frame.assign(
                member=f"seed_{seed}", configuration=candidate.name
            )
            case_frames.append(frame)
        # Ensemble in physical space because learned seeds may have different
        # powers/scales; averaging standardized residuals would be invalid.
        ensemble = np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)
        np.save(
            output / "models" / candidate.name / "validation_prediction_ensemble.npy",
            ensemble,
        )
        frame = bias._case_metrics(
            ensemble, prepared, predictor=f"{candidate.name}_ensemble"
        )
        frame = frame.assign(member="ensemble", configuration=candidate.name)
        case_frames.append(frame)
        intensity_frames.append(
            bias.intensity_strata_metrics(
                ensemble,
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
        frame = frame.assign(member="deterministic", configuration=name)
        case_frames.append(frame)
        intensity_frames.append(
            bias.intensity_strata_metrics(
                prediction,
                validation_truth,
                shared.weights,
                configuration=name,
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


def apply_stratified_qualification_guards(
    ranking: pd.DataFrame,
    intensity_metrics: pd.DataFrame,
    *,
    reference_configuration: str = REFERENCE_CONFIGURATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prevent pooled-bias cancellation from qualifying a transform.

    Besides the existing per-year and per-lead guards, every IMD intensity
    stratum must have non-worse absolute bias.  Dry/light MAE may regress by at
    most 1%, and heavy-rain RMSE and MAE must each improve by at least 2%.
    """

    result = ranking.copy()
    all_weeks = intensity_metrics.loc[intensity_metrics.lead.eq("ALL_WEEKS")]
    reference = all_weeks.loc[
        all_weeks.configuration.eq(reference_configuration)
    ].set_index("stratum")
    expected_strata = ("dry_lt1", "light_1_5", "moderate_5_10", "heavy_ge10")
    if (
        reference.empty
        or not reference.index.is_unique
        or any(stratum not in reference.index for stratum in expected_strata)
    ):
        raise ValueError("missing exact-log intensity reference")
    rows = []
    for configuration in result.configuration:
        candidate = all_weeks.loc[
            all_weeks.configuration.eq(configuration)
        ].set_index("stratum")
        if any(stratum not in candidate.index for stratum in expected_strata):
            raise ValueError(f"missing intensity strata for {configuration}")
        bias_not_worse = {
            stratum: abs(float(candidate.loc[stratum, "bias"]))
            <= abs(float(reference.loc[stratum, "bias"])) + 1.0e-12
            for stratum in expected_strata
        }
        dry_light_mae_guard = all(
            float(candidate.loc[stratum, "mae"])
            <= float(reference.loc[stratum, "mae"]) * 1.01
            for stratum in ("dry_lt1", "light_1_5")
        )
        heavy_rmse_improvement = 100.0 * (
            float(reference.loc["heavy_ge10", "rmse"])
            - float(candidate.loc["heavy_ge10", "rmse"])
        ) / float(reference.loc["heavy_ge10", "rmse"])
        heavy_mae_improvement = 100.0 * (
            float(reference.loc["heavy_ge10", "mae"])
            - float(candidate.loc["heavy_ge10", "mae"])
        ) / float(reference.loc["heavy_ge10", "mae"])
        candidate_equal_stratum_mean_abs_bias = float(
            np.mean(
                [abs(float(candidate.loc[stratum, "bias"])) for stratum in expected_strata]
            )
        )
        reference_equal_stratum_mean_abs_bias = float(
            np.mean(
                [abs(float(reference.loc[stratum, "bias"])) for stratum in expected_strata]
            )
        )
        equal_stratum_bias_guard = (
            candidate_equal_stratum_mean_abs_bias
            < reference_equal_stratum_mean_abs_bias - 1.0e-12
        )
        row = {
            "configuration": configuration,
            **{
                f"{stratum}_absolute_bias_not_worse": passed
                for stratum, passed in bias_not_worse.items()
            },
            "intensity_abs_bias_not_worse_count": int(sum(bias_not_worse.values())),
            "all_intensity_abs_bias_not_worse": bool(all(bias_not_worse.values())),
            "candidate_equal_stratum_mean_abs_bias": (
                candidate_equal_stratum_mean_abs_bias
            ),
            "reference_equal_stratum_mean_abs_bias": (
                reference_equal_stratum_mean_abs_bias
            ),
            "equal_stratum_mean_abs_bias_delta": (
                candidate_equal_stratum_mean_abs_bias
                - reference_equal_stratum_mean_abs_bias
            ),
            "equal_stratum_mean_abs_bias_guard": bool(equal_stratum_bias_guard),
            "dry_light_mae_guard": bool(dry_light_mae_guard),
            "heavy_rmse_improvement_pct": heavy_rmse_improvement,
            "heavy_mae_improvement_pct": heavy_mae_improvement,
            "heavy_rmse_improves_at_least_2pct": bool(
                heavy_rmse_improvement >= 2.0
            ),
            "heavy_mae_improves_at_least_2pct": bool(
                heavy_mae_improvement >= 2.0
            ),
        }
        row["passes_all_intensity_guards"] = bool(
            row["all_intensity_abs_bias_not_worse"]
            and row["equal_stratum_mean_abs_bias_guard"]
            and row["dry_light_mae_guard"]
            and row["heavy_rmse_improves_at_least_2pct"]
            and row["heavy_mae_improves_at_least_2pct"]
        )
        rows.append(row)
    guards = pd.DataFrame(rows)
    result = result.merge(guards, on="configuration", validate="one_to_one")
    result["qualification_before_intensity_guards"] = result.qualifies.astype(bool)
    result["qualifies"] = (
        result.qualification_before_intensity_guards
        & result.passes_all_intensity_guards
        & result.all_lead_abs_bias_not_worse
    )
    result = result.sort_values(
        ["qualifies", "pooled_rmse", "pooled_abs_bias"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result, guards


def apply_learned_power_boundary_guard(
    ranking: pd.DataFrame,
    records: pd.DataFrame,
    candidates: Sequence[LearnableTransformCandidate],
    *,
    margin: float = LEARNED_POWER_BOUNDARY_MARGIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject a learned transform that saturates near either power bound."""

    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("power boundary margin must be finite and positive")
    rows = []
    for candidate in candidates:
        values = records.loc[
            records.configuration.eq(candidate.name), "best_power"
        ].to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"missing finite fitted powers for {candidate.name}")
        lower = candidate.minimum_power + margin
        upper = candidate.maximum_power - margin
        passes = bool(
            not candidate.learnable
            or (np.all(values > lower) and np.all(values < upper))
        )
        rows.append(
            {
                "configuration": candidate.name,
                "power_is_learned": bool(candidate.learnable),
                "fitted_power_min": float(values.min()),
                "fitted_power_mean": float(values.mean()),
                "fitted_power_max": float(values.max()),
                "power_boundary_margin": float(margin),
                "learned_power_away_from_bounds": passes,
            }
        )
    guards = pd.DataFrame(rows)
    result = ranking.merge(guards, on="configuration", validate="one_to_one")
    result["qualification_before_power_boundary_guard"] = result.qualifies.astype(bool)
    result["qualifies"] = (
        result.qualification_before_power_boundary_guard
        & result.learned_power_away_from_bounds
    )
    result = result.sort_values(
        ["qualifies", "pooled_rmse", "pooled_abs_bias"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result, guards


def save_plots(
    output: Path,
    history: pd.DataFrame,
    case_metrics: pd.DataFrame,
    candidates: Sequence[LearnableTransformCandidate],
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
    labels = {"raw_fuxi": "Raw FuXi", **{c.name: c.label for c in candidates}}
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), sharex=True)
    for name, frame in summary.groupby("configuration"):
        for axis, metric in zip(axes.flat, ("rmse", "mae", "acc", "bias")):
            axis.plot(
                frame.lead,
                frame[metric],
                marker="o",
                linewidth=1.6,
                linestyle="--" if name == "raw_fuxi" else "-",
                label=labels[name],
            )
    for axis, ylabel in zip(
        axes.flat, ("RMSE (mm/day)", "MAE (mm/day)", "Spatial ACC", "Bias (mm/day)")
    ):
        axis.set(ylabel=ylabel, xlabel="Lead week")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.22)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Learnable target transform · blocked 2018–2019 validation")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.png", dpi=240)
    figure.savefig(output / "figures" / "01_physical_metrics_by_lead.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for (configuration, seed), frame in history.groupby(["configuration", "seed"]):
        label = f"{labels[configuration]} · {seed}"
        axes[0].plot(
            frame.epoch + 1,
            frame.validation_physical_rmse,
            label=label,
        )
        axes[1].plot(frame.epoch + 1, frame.power_epoch_end, label=label)
    axes[0].set(ylabel="Validation physical RMSE (mm/day)", xlabel="Epoch")
    axes[1].set(ylabel="Global Box-Cox power", xlabel="Epoch")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Physical checkpoint criterion and learned curvature")
    figure.tight_layout()
    figure.savefig(output / "figures" / "02_physical_rmse_and_power.png", dpi=240)
    figure.savefig(output / "figures" / "02_physical_rmse_and_power.pdf")
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
        HERE / "slurm" / "run_imd_learnable_target_transform_validation.sbatch",
        NEURAL_SRC / "fuxi_adapter" / "learnable_boxcox.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
    )
    for source in sources:
        shutil.copy2(source, output / "code" / source.name)


def write_readme(
    output: Path,
    ranking: pd.DataFrame,
    selection_status: str,
    selected: pd.Series,
    seeds: Sequence[int],
) -> None:
    lines = [
        "# FuXi–IMD learnable target-transform validation",
        "",
        "Train: 2002–2017. Blocked validation: 2018–2019. No dynamic target, "
        "prediction, or metric is created for 2020 onward.",
        "",
        "All three arms use one training loop. The training-scale derivative "
        "with respect to the global power is retained and recomputed every batch. "
        "The first five epochs keep power fixed at 0.25; its later learning rate "
        "is 0.1 times the forecast-model learning rate.",
        "",
        "Checkpoint selection: physical validation RMSE in mm/day. Transform "
        "state and train-fitted scale are stored separately from model weights.",
        "",
        f"Seeds: `{list(seeds)}`. Selection status: **{selection_status}**. "
        f"Selected: **{selected.label}** (`{selected.configuration}`).",
        "",
        "| Configuration | RMSE | MAE | Bias | ACC | fitted λ | Qualifies |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| `{row.configuration}` | {row.pooled_rmse:.4f} | "
            f"{row.pooled_mae:.4f} | {row.pooled_bias:.4f} | "
            f"{row.pooled_acc:.4f} | {row.fitted_power_mean:.4f} | "
            f"{bool(row.qualifies)} |"
        )
    lines.extend(
        [
            "",
            "Qualification requires the existing year/lead/raw-FuXi guards, "
            "per-intensity no-cancellation guards, equal-stratum mean absolute "
            "bias, heavy-rain improvement, and a non-saturated learned power.",
            "",
            "A one-seed result is exploratory and requires three-seed confirmation "
            "before any independent evaluation.",
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
        "purpose": "validation-only learned global target-transform experiment",
        "train_years": list(bias.TRAIN_YEARS),
        "validation_years": list(bias.VALIDATION_YEARS),
        "quarantined_years": list(bias.QUARANTINED_YEARS),
        "test_predictions_created": False,
        "fixed_anchor": "physical_recentered",
        "fixed_loss": dict(bias.BIAS_AWARE_LOSS),
        "model": asdict(bias.MODEL_SPEC),
        "candidates": [asdict(candidate) for candidate in candidates],
        "seeds": list(seeds),
        "reference_configuration": REFERENCE_CONFIGURATION,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "checkpoint_selection_uses_physical_units": True,
        "transform_scale_fit_split": "train",
        "training_transform_scale_gradient": "full_ds_dlambda",
        "training_transform_scale_refresh": "every_optimization_batch",
        "validation_transform_scale_gradient": "off",
        "transform_learning_rate_ratio": TRANSFORM_LEARNING_RATE_RATIO,
        "transform_warmup_epochs": TRANSFORM_WARMUP_EPOCHS,
        "validation_used_for_transform_scale": False,
        "learned_parameter_scope": "one_global_scalar",
        "ensemble_space": "physical_mm_day",
        "smoke": bool(args.smoke),
        "screening_stage": "smoke" if args.smoke else (
            "one_seed" if len(seeds) == 1 else "multi_seed_confirmation"
        ),
        "ranking_uses_physical_metrics_only": True,
        "intensity_and_lead_no_cancellation_guards": True,
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(
        json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        prepared, normalization, preparation = bias.prepare_data()
        quarantined = bias.validate_quarantined_splits(
            prepared.shared.initializations,
            prepared.shared.train_indices,
            prepared.shared.validation_indices,
        )
        quarantined_anchor = prepared.anchors["physical_recentered"]
        if np.count_nonzero(quarantined_anchor.target[quarantined]) != 0:
            raise ValueError("quarantined 2020+ target buffer is not zero")
        # The learnable target is constructed only inside the two sliced
        # datasets, so no dynamic target exists for quarantined cases.
        manifest.update(
            {
                "quarantined_dynamic_targets_constructed": False,
                "quarantined_reference_target_values_nonzero": False,
                "quarantined_case_count": int(len(quarantined)),
            }
        )
        (output / "normalization.json").write_text(
            json.dumps(bias.compact._json_safe(normalization), indent=2) + "\n",
            encoding="utf-8",
        )
        preparation["recenter_diagnostics"].to_csv(
            output / "metrics" / "anchor_training_recenter_diagnostics.csv",
            index=False,
        )
        manifest.update(preparation["metadata"])
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
            output,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            workers=workers,
        )
        records, history, case_metrics, intensity_metrics = aggregate_results(
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
        ranking, intensity_guards = apply_stratified_qualification_guards(
            ranking,
            intensity_metrics,
            reference_configuration=REFERENCE_CONFIGURATION,
        )
        ranking, power_boundary_guards = apply_learned_power_boundary_guard(
            ranking, records, candidates
        )
        ranking.to_csv(output / "metrics" / "ranked_configurations.csv", index=False)
        intensity_guards.to_csv(
            output / "metrics" / "stratified_qualification_guards.csv", index=False
        )
        power_boundary_guards.to_csv(
            output / "metrics" / "learned_power_boundary_guards.csv", index=False
        )
        bias.paired_physical_deltas(
            case_metrics, candidates, REFERENCE_CONFIGURATION
        ).to_csv(
            output / "metrics" / "paired_physical_deltas_vs_exact_log.csv",
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
        save_plots(output, history, case_metrics, candidates)
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
        print(f"PASS: learnable target-transform experiment complete: {output}")
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
