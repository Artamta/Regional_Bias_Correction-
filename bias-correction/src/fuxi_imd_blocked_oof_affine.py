#!/usr/bin/env python3
"""Blocked out-of-fold affine calibration for the locked FuXi--IMD adapter.

The four outer folds cover 2002--2017 exactly once.  Every target-derived
quantity is independently fitted on the corresponding ten base-training years;
two separate years are used only for neural early stopping.  Six lead-wise
physical affine maps are fitted after pooling the outer-fold predictions.  The
coefficients are written and frozen before the 2018--2019 development block is
loaded or scored.  No 2020+ prediction or metric is produced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
import fuxi_imd_train_affine_calibration as affine  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.metrics import compute_case_metrics  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import train_anchored_model  # noqa: E402


compact = bias.compact
base = bias.base
common = bias.common
experiment = compact.experiment

RESULTS_ROOT = HERE / "results" / "fuxi_imd_blocked_oof_affine"
DEFAULT_SOURCE = bias.RESULTS_ROOT / "full_20260813T135600Z"
ALL_CALIBRATION_YEARS = tuple(range(2002, 2018))
FINAL_DEVELOPMENT_YEARS = (2018, 2019)
QUARANTINED_YEARS = tuple(range(2020, 2026))
SOURCE_CONFIGURATION = affine.SOURCE_CONFIGURATION
REFERENCE_CONFIGURATION = affine.REFERENCE_CONFIGURATION
CALIBRATED_CONFIGURATION = "recentered_bias_aware_blocked_oof_affine"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_years: tuple[int, ...]
    inner_validation_years: tuple[int, ...]
    outer_years: tuple[int, ...]


FOLDS = (
    FoldSpec("fold1_outer_2002_2005", tuple(range(2008, 2018)), (2006, 2007), tuple(range(2002, 2006))),
    FoldSpec("fold2_outer_2006_2009", (*range(2002, 2006), *range(2012, 2018)), (2010, 2011), tuple(range(2006, 2010))),
    FoldSpec("fold3_outer_2010_2013", (*range(2002, 2010), 2016, 2017), (2014, 2015), tuple(range(2010, 2014))),
    FoldSpec("fold4_outer_2014_2017", tuple(range(2004, 2014)), (2002, 2003), tuple(range(2014, 2018))),
)


@dataclass
class FoldPrepared:
    fold: FoldSpec
    forecast: Any
    observations: Any
    features: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    target_scale: np.ndarray
    bias_scale: np.ndarray
    weights: np.ndarray
    valid_mask: np.ndarray
    train_indices: np.ndarray
    inner_indices: np.ndarray
    outer_indices: np.ndarray
    mean_to_anomaly_ratio: np.ndarray
    normalization: Mapping[str, Any]
    recenter_diagnostics: tuple[Mapping[str, Any], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calendar_years(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[Y]").astype(np.int64) + 1970


def validate_fold_design(folds: Sequence[FoldSpec] = FOLDS) -> None:
    outer = []
    for fold in folds:
        train = set(fold.train_years)
        inner = set(fold.inner_validation_years)
        held = set(fold.outer_years)
        if len(train) != 10 or len(inner) != 2 or len(held) != 4:
            raise ValueError(f"invalid 10/2/4 year counts in {fold.name}")
        if train & inner or train & held or inner & held:
            raise ValueError(f"year overlap in {fold.name}")
        if train | inner | held != set(ALL_CALIBRATION_YEARS):
            raise ValueError(f"{fold.name} does not partition 2002--2017")
        outer.extend(fold.outer_years)
    if sorted(outer) != list(ALL_CALIBRATION_YEARS):
        raise ValueError("outer folds do not cover every calibration year exactly once")


def configure_calibration_archive(train_years: Sequence[int]) -> None:
    """Configure a structurally sealed archive containing only 2002--2017."""

    experiment.set_experiment_scope(
        all_weeks=True,
        large_model=False,
        regularized_large=False,
        full_fuxi_context=True,
    )
    experiment.configure_contract()
    base.TRAIN_YEARS = tuple(int(year) for year in train_years)
    base.VALIDATION_YEARS = ()
    base.TEST_YEARS = ()
    base.ALL_YEARS = ALL_CALIBRATION_YEARS
    if set(base.ALL_YEARS) & set(FINAL_DEVELOPMENT_YEARS + QUARANTINED_YEARS):
        raise RuntimeError("sealed calibration archive includes a post-2017 year")


def load_sealed_sources() -> tuple[Any, np.ndarray]:
    configure_calibration_archive(FOLDS[0].train_years)
    print("Loading sealed 2002--2017 FuXi archive once...", flush=True)
    forecast = base.load_fuxi()
    years = calendar_years(forecast.initializations)
    if tuple(sorted(np.unique(years))) != ALL_CALIBRATION_YEARS:
        raise base.DataContractError("sealed FuXi archive years differ from 2002--2017")
    if len(forecast.initializations) != 560:
        raise base.DataContractError("sealed FuXi archive must contain 560 cases")
    print("Loading sealed 2002--2017 T2M context once...", flush=True)
    t2m = common.load_t2m_weekly(forecast)
    return forecast, t2m


def prepare_fold(fold: FoldSpec, forecast: Any, t2m_weekly: np.ndarray) -> FoldPrepared:
    configure_calibration_archive(fold.train_years)
    years = calendar_years(forecast.initializations)
    train_indices = np.flatnonzero(np.isin(years, fold.train_years)).astype(np.int64)
    inner_indices = np.flatnonzero(np.isin(years, fold.inner_validation_years)).astype(np.int64)
    outer_indices = np.flatnonzero(np.isin(years, fold.outer_years)).astype(np.int64)
    if (len(train_indices), len(inner_indices), len(outer_indices)) != (350, 70, 140):
        raise base.DataContractError(f"unexpected case counts for {fold.name}")
    if np.intersect1d(train_indices, np.concatenate((inner_indices, outer_indices))).size:
        raise base.DataContractError(f"overlapping case indices in {fold.name}")

    print(f"[{fold.name}] Loading IMD and fitting fold-local climatology...", flush=True)
    observations, climatology_daily, _, _ = experiment.load_imd(forecast)
    weights = experiment.load_imd_weights(forecast, observations.observation_fraction)
    support = weights > 0.0
    features, normalization, centre_difference = experiment.build_climatology_features(
        forecast,
        observations,
        climatology_daily,
        weights,
        train_indices,
        t2m_weekly,
        preserve_fuxi_context=True,
    )
    if centre_difference > 2.0e-6:
        raise base.DataContractError("fold climatology centre mismatch")
    log_correction = fit_log_bias_correction(
        forecast.ensemble_mean[train_indices],
        observations.weekly_truth[train_indices],
        forecast.initializations[train_indices],
        support,
        shrinkage=10.0,
    )
    recentered = bias.fit_physical_recentered_log_correction(
        forecast.ensemble_mean,
        observations.weekly_truth,
        forecast.initializations,
        train_indices,
        weights,
        log_correction.lead_month_residual,
        split_name="train",
        shrinkage=log_correction.shrinkage,
    )
    baseline = apply_log_bias_correction(
        forecast.ensemble_mean, forecast.initializations, recentered.correction
    )
    valid_mask = np.broadcast_to(support[None, None], baseline.shape).copy()
    target_scale = fit_anchored_target_scale(
        observations.weekly_truth[train_indices],
        baseline[train_indices],
        weights,
        split_name="train",
        valid_mask=valid_mask[train_indices],
    )
    target = standardize_anchored_target(
        observations.weekly_truth,
        baseline,
        target_scale,
        valid_mask=valid_mask,
    )
    bias_scale = bias.fit_training_bias_scale(
        observations.weekly_truth,
        weights,
        train_indices,
        forecast.initializations,
        split_name="train",
    )
    mean_std = np.asarray(normalization["log_fuxi_mean"]["std_by_lead"], dtype=np.float32)
    anomaly_std = np.asarray(
        normalization["explicit_log_fuxi_anomaly"]["std_by_lead"], dtype=np.float32
    )
    ratio = mean_std / anomaly_std
    if ratio.shape != (6,) or not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise ValueError("invalid fold-local mean/anomaly noise ratio")
    return FoldPrepared(
        fold=fold,
        forecast=forecast,
        observations=observations,
        features=np.asarray(features, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        baseline=np.asarray(baseline, dtype=np.float32),
        target_scale=np.asarray(target_scale, dtype=np.float32),
        bias_scale=np.asarray(bias_scale, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float64),
        valid_mask=valid_mask,
        train_indices=train_indices,
        inner_indices=inner_indices,
        outer_indices=outer_indices,
        mean_to_anomaly_ratio=ratio,
        normalization=normalization,
        recenter_diagnostics=recentered.diagnostics,
    )


def train_fold_seed(
    prepared: FoldPrepared,
    seed: int,
    output: Path,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    run_directory = output / "folds" / prepared.fold.name / f"seed_{seed}"
    (run_directory / "logs").mkdir(parents=True, exist_ok=False)
    (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
    support = prepared.weights > 0.0
    train_data = common.make_dataset(
        prepared.train_indices,
        prepared.features,
        prepared.target,
        prepared.baseline,
        prepared.observations,
        support,
    )
    inner_data = common.make_dataset(
        prepared.inner_indices,
        prepared.features,
        prepared.target,
        prepared.baseline,
        prepared.observations,
        support,
    )
    set_deterministic_seed(seed)
    model = compact.build_model(
        bias.MODEL_SPEC,
        prepared.features.shape[2],
        prepared.mean_to_anomaly_ratio,
    )
    started = time.monotonic()
    result = train_anchored_model(
        model,
        train_data,
        inner_data,
        prepared.weights,
        prepared.target_scale,
        bias.LEAD_WEIGHTS,
        bias.BIAS_AWARE_LOSS,
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
        num_workers=0,
        use_amp=device.startswith("cuda"),
    )
    residual = predict(
        model,
        prepared.features[prepared.outer_indices],
        device=device,
        batch_size=32,
        use_amp=device.startswith("cuda"),
    )
    prediction = reconstruct_anchored_precipitation(
        prepared.baseline[prepared.outer_indices],
        residual,
        prepared.target_scale,
        valid_mask=prepared.valid_mask[prepared.outer_indices],
    )
    np.save(run_directory / "outer_prediction.npy", prediction)
    np.save(run_directory / "outer_indices.npy", prepared.outer_indices)
    checkpoint = run_directory / "checkpoints" / "best.pt"
    record = {
        "status": "complete",
        "fold": asdict(prepared.fold),
        "seed": int(seed),
        "device": device,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch_zero_based": int(result.best_epoch),
        "best_epoch_display": int(result.best_epoch + 1),
        "best_inner_validation_objective": float(result.best_validation_loss),
        "elapsed_seconds": float(time.monotonic() - started),
        "checkpoint_sha256": compact.sha256_file(checkpoint),
        "target_preprocessing_fit_years": list(prepared.fold.train_years),
        "outer_targets_used_during_model_or_preprocessing_fit": False,
    }
    (run_directory / "run_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return prediction, record


def paired_case_bootstrap(
    case_metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    repetitions: int = 10000,
    seed: int = 20260813,
) -> pd.DataFrame:
    """Paired initialization-date bootstrap, retaining all six leads."""

    selected = case_metrics.loc[
        case_metrics.member.eq("ensemble")
        & case_metrics.configuration.isin((candidate, reference))
    ]
    tables = {
        name: frame.set_index(["case_id", "lead"]).sort_index()
        for name, frame in selected.groupby("configuration")
    }
    if set(tables) != {candidate, reference} or not tables[candidate].index.equals(
        tables[reference].index
    ):
        raise ValueError("paired bootstrap requires matching candidate/reference cases")
    dates = np.asarray(sorted(selected.case_id.unique()), dtype="datetime64[ns]")
    rng = np.random.default_rng(seed)
    differences = {metric: np.empty(repetitions, dtype=np.float64) for metric in ("rmse", "mae", "bias_abs", "acc")}
    by_date = {
        name: {
            np.datetime64(date, "ns"): frame.loc[
                pd.to_datetime(frame.index.get_level_values("case_id")) == pd.Timestamp(date)
            ]
            for date in dates
        }
        for name, frame in tables.items()
    }
    for repetition in range(repetitions):
        sampled = dates[rng.integers(0, len(dates), size=len(dates))]
        values: dict[str, dict[str, float]] = {}
        for name in (candidate, reference):
            blocks = [by_date[name][date] for date in sampled]
            frame = pd.concat(blocks, ignore_index=True)
            values[name] = {
                "rmse": float(frame.rmse.mean()),
                "mae": float(frame.mae.mean()),
                "bias_abs": abs(float(frame.bias.mean())),
                "acc": float(frame.acc.mean()),
            }
        for metric in differences:
            differences[metric][repetition] = values[candidate][metric] - values[reference][metric]
    rows = []
    for metric, samples in differences.items():
        lower, median, upper = np.quantile(samples, (0.025, 0.5, 0.975))
        desired_negative = metric != "acc"
        probability_improved = float(np.mean(samples < 0.0 if desired_negative else samples > 0.0))
        rows.append(
            {
                "metric": metric,
                "delta_definition": "candidate_minus_reference",
                "median_delta": float(median),
                "ci_lower_2p5": float(lower),
                "ci_upper_97p5": float(upper),
                "probability_improved": probability_improved,
                "bootstrap_unit": "initialization date with all six leads retained",
                "repetitions": repetitions,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def final_evaluation(
    source: Path,
    output: Path,
    fits_by_seed: Mapping[int, Sequence[affine.AffineFit]],
    *,
    smoke: bool,
) -> Mapping[str, Any]:
    """Load and score 2018--2019 only after OOF coefficients are frozen."""

    frozen_path = output / "models" / "frozen_oof_affine_coefficients.json"
    if not frozen_path.is_file():
        raise RuntimeError("OOF coefficients were not frozen before final evaluation")
    frozen_hash = compact.sha256_file(frozen_path)
    print("OOF coefficients frozen; now loading the 2018--2019 development block...", flush=True)
    prepared, _, preparation = bias.prepare_data()
    years = calendar_years(
        prepared.shared.initializations[prepared.shared.validation_indices]
    )
    if tuple(sorted(np.unique(years))) != FINAL_DEVELOPMENT_YEARS:
        raise RuntimeError("final development indices differ from 2018--2019")

    frames = []
    calibrated_members = []
    source_members = []
    reference_members = []
    for seed, fits in fits_by_seed.items():
        source_prediction = affine._existing_validation_prediction(
            source, prepared, SOURCE_CONFIGURATION, seed
        )
        reference_prediction = affine._existing_validation_prediction(
            source, prepared, REFERENCE_CONFIGURATION, seed
        )
        calibrated = affine.apply_leadwise_affine(source_prediction, fits)
        calibrated_members.append(calibrated)
        source_members.append(source_prediction)
        reference_members.append(reference_prediction)
        member = f"seed_{seed}"
        frames.extend(
            (
                affine._validation_metrics(calibrated, prepared, CALIBRATED_CONFIGURATION, member),
                affine._validation_metrics(source_prediction, prepared, SOURCE_CONFIGURATION, member),
                affine._validation_metrics(reference_prediction, prepared, REFERENCE_CONFIGURATION, member),
            )
        )
    ensembles = {
        CALIBRATED_CONFIGURATION: np.mean(calibrated_members, axis=0, dtype=np.float64).astype(np.float32),
        SOURCE_CONFIGURATION: affine._existing_validation_ensemble_prediction(source, prepared, SOURCE_CONFIGURATION),
        REFERENCE_CONFIGURATION: affine._existing_validation_ensemble_prediction(source, prepared, REFERENCE_CONFIGURATION),
    }
    for name, prediction in ensembles.items():
        np.save(output / "models" / f"{name}_development_ensemble.npy", prediction)
        frames.append(affine._validation_metrics(prediction, prepared, name, "ensemble"))
    raw = prepared.shared.raw_fuxi[prepared.shared.validation_indices]
    frames.append(affine._validation_metrics(raw, prepared, "raw_fuxi", "deterministic"))
    case_metrics = pd.concat(frames, ignore_index=True)
    case_metrics.to_csv(output / "metrics" / "development_2018_2019_case_metrics.csv", index=False)
    affine._summary(case_metrics).to_csv(output / "metrics" / "development_2018_2019_summary.csv", index=False)

    calibrated_candidate = bias.BiasCandidate(
        CALIBRATED_CONFIGURATION,
        "Recentered bias-aware + blocked-OOF affine",
        "physical_recentered",
        "bias_aware",
        bias.BIAS_AWARE_LOSS,
    )
    ranking_candidates = (
        bias.CANDIDATE_BY_NAME[REFERENCE_CONFIGURATION],
        calibrated_candidate,
    )
    records = pd.DataFrame(
        [
            {
                "configuration": candidate.name,
                "parameter_count": 323017 + (12 if candidate.name == CALIBRATED_CONFIGURATION else 0),
                "best_validation_objective": np.nan,
                "elapsed_seconds": np.nan,
            }
            for candidate in ranking_candidates
        ]
    )
    seed_guards = bias.build_seed_physical_guards(
        case_metrics, ranking_candidates, reference_configuration=REFERENCE_CONFIGURATION
    )
    ranking = bias.build_physical_ranking(
        records,
        case_metrics,
        ranking_candidates,
        reference_configuration=REFERENCE_CONFIGURATION,
        seed_guards=seed_guards,
    )
    ranking.to_csv(output / "metrics" / "strict_development_ranking.csv", index=False)
    seed_guards.to_csv(output / "metrics" / "seed_physical_guards.csv", index=False)
    paired_case_bootstrap(
        case_metrics, CALIBRATED_CONFIGURATION, REFERENCE_CONFIGURATION
    ).to_csv(output / "metrics" / "paired_initialization_bootstrap.csv", index=False)
    candidate_row = ranking.loc[
        ranking.configuration.eq(CALIBRATED_CONFIGURATION)
    ].iloc[0]
    return {
        "development_evaluation_completed_utc": utc_now(),
        "frozen_coefficients_sha256": frozen_hash,
        "candidate_passed_strict_guards": bool(candidate_row.qualifies),
        "disposition": (
            "candidate_passed_development_guards_requires_future_untouched_confirmation"
            if bool(candidate_row.qualifies) and not smoke
            else "not_promoted"
        ),
        "development_years": list(FINAL_DEVELOPMENT_YEARS),
        "development_block_is_independent_test": False,
        "quarantined_predictions_or_metrics_produced": False,
        "preparation_metadata": preparation["metadata"],
    }


def run(
    source: Path,
    output: Path,
    seeds: Sequence[int],
    *,
    device: str,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> Mapping[str, Any]:
    validate_fold_design()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    (output / "folds").mkdir(parents=True)
    (output / "metrics").mkdir()
    (output / "models").mkdir()
    (output / "code").mkdir()
    for path in (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imd_train_affine_calibration.py",
        SOURCE_ROOT / "fuxi_imd_bias_aware_validation_sweep.py",
        SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
    ):
        shutil.copy2(path, output / "code" / path.name)

    forecast, t2m = load_sealed_sources()
    folds = FOLDS[:1] if smoke else FOLDS
    selected_seeds = tuple(seeds[:1] if smoke else seeds)
    oof_by_seed: dict[int, dict[int, np.ndarray]] = {
        int(seed): {} for seed in selected_seeds
    }
    records = []
    preprocessing_rows = []
    canonical_truth = None
    canonical_weights = None
    for fold in folds:
        prepared = prepare_fold(fold, forecast, t2m)
        if canonical_truth is None:
            canonical_truth = prepared.observations.weekly_truth.copy()
            canonical_weights = prepared.weights.copy()
        elif not np.array_equal(
            canonical_truth,
            prepared.observations.weekly_truth,
            equal_nan=True,
        ):
            raise base.DataContractError("weekly IMD truth changed between folds")
        for diagnostic in prepared.recenter_diagnostics:
            preprocessing_rows.append({"fold": fold.name, **diagnostic})
        (output / "folds" / fold.name).mkdir(parents=True, exist_ok=True)
        (output / "folds" / fold.name / "normalization.json").write_text(
            json.dumps(prepared.normalization, indent=2) + "\n", encoding="utf-8"
        )
        for seed in selected_seeds:
            print(f"[{fold.name}] training seed {seed}...", flush=True)
            prediction, record = train_fold_seed(
                prepared,
                int(seed),
                output,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                smoke=smoke,
            )
            records.append(record)
            for index, value in zip(prepared.outer_indices, prediction):
                if int(index) in oof_by_seed[int(seed)]:
                    raise RuntimeError("duplicate outer-fold prediction")
                oof_by_seed[int(seed)][int(index)] = value
        del prepared

    assert canonical_truth is not None and canonical_weights is not None
    pd.DataFrame(records).to_csv(output / "metrics" / "fold_run_records.csv", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(
        output / "metrics" / "fold_recenter_diagnostics.csv", index=False
    )
    expected_indices = np.arange(140, dtype=np.int64) if smoke else np.arange(560, dtype=np.int64)
    # Smoke uses only fold 1 outer indices, not positions 0..139 necessarily.
    if smoke:
        years = calendar_years(forecast.initializations)
        expected_indices = np.flatnonzero(np.isin(years, folds[0].outer_years))
    fits_by_seed: dict[int, tuple[affine.AffineFit, ...]] = {}
    coefficient_rows = []
    all_train_scale = bias.fit_training_bias_scale(
        canonical_truth,
        canonical_weights,
        np.arange(len(forecast.initializations), dtype=np.int64),
        forecast.initializations,
        split_name="train",
    )
    for seed in selected_seeds:
        present = np.asarray(sorted(oof_by_seed[int(seed)]), dtype=np.int64)
        if not np.array_equal(present, expected_indices):
            raise RuntimeError(f"OOF coverage mismatch for seed {seed}")
        oof = np.stack([oof_by_seed[int(seed)][int(index)] for index in present])
        np.save(output / "models" / f"seed_{seed}_oof_prediction.npy", oof)
        fits = affine.fit_leadwise_affine(
            oof,
            canonical_truth[present],
            canonical_weights,
            all_train_scale,
        )
        fits_by_seed[int(seed)] = fits
        coefficient_rows.extend({"seed": int(seed), **fit.__dict__} for fit in fits)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(output / "metrics" / "oof_affine_coefficients.csv", index=False)
    frozen_payload = {
        "frozen_utc": utc_now(),
        "fit_source": "blocked outer-fold predictions from 2002-2017 only",
        "folds": [asdict(fold) for fold in folds],
        "seeds": list(selected_seeds),
        "fits": coefficient_rows,
        "post_2017_truth_loaded_before_freeze": False,
    }
    frozen_path = output / "models" / "frozen_oof_affine_coefficients.json"
    frozen_path.write_text(json.dumps(frozen_payload, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "created_utc": utc_now(),
        "status": "coefficients_frozen",
        "scientific_status": "blocked out-of-fold calibration experiment",
        "source": str(source),
        "seeds": list(selected_seeds),
        "smoke": smoke,
        "folds": [asdict(fold) for fold in folds],
        "calibration_archive_years": list(ALL_CALIBRATION_YEARS),
        "post_2017_truth_loaded_before_coefficients_frozen": False,
        "quarantined_years": list(QUARANTINED_YEARS),
        "quarantined_predictions_or_metrics_produced": False,
        "frozen_coefficients_sha256": compact.sha256_file(frozen_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    evaluation = final_evaluation(
        source, output, fits_by_seed, smoke=smoke
    )
    manifest.update(evaluation)
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    summary = pd.read_csv(output / "metrics" / "development_2018_2019_summary.csv")
    ranking = pd.read_csv(output / "metrics" / "strict_development_ranking.csv")
    print(summary.to_string(index=False), flush=True)
    print(
        ranking[["configuration", "pooled_rmse", "pooled_mae", "pooled_bias", "pooled_acc", "qualifies"]].to_string(index=False),
        flush=True,
    )
    print(f"disposition={manifest['disposition']}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    seeds = affine.parse_seeds(arguments.seeds)
    if arguments.max_epochs < 1 or arguments.patience < 1:
        raise ValueError("max-epochs and patience must be positive")
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run(
        arguments.source.resolve(),
        arguments.output.resolve(),
        seeds,
        device=arguments.device,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        smoke=arguments.smoke,
    )


if __name__ == "__main__":
    main()
