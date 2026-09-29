#!/usr/bin/env python3
"""Train-only physical affine calibration diagnostic for FuXi-to-IMD.

This script deliberately reuses the already-trained three-seed
``recentered_anchor_bias_aware_loss`` models.  For every seed and lead week it
fits only two physical parameters on 2002--2017 model predictions::

    calibrated = max(0, slope * prediction + intercept)

The slope and intercept are tightly bounded.  The coefficients never see
2018--2019 targets; 2020 onward is neither predicted nor scored.  Because the
base models generated their calibration predictions in sample, a successful
result is only a screening diagnostic and must be repeated with blocked
out-of-fold predictions before it can be treated as a model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

for path in (SOURCE_ROOT, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_bias_aware_validation_sweep as bias  # noqa: E402
from fuxi_adapter.anchored import reconstruct_anchored_precipitation  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402


RESULTS_ROOT = HERE / "results" / "fuxi_imd_train_affine_calibration"
DEFAULT_SOURCE = (
    bias.RESULTS_ROOT / "full_20260813T135600Z"
)
SOURCE_CONFIGURATION = "recentered_anchor_bias_aware_loss"
CALIBRATED_CONFIGURATION = "recentered_bias_aware_train_affine"
REFERENCE_CONFIGURATION = bias.REFERENCE_CONFIGURATION
DEFAULT_SEEDS = (42, 43, 44)
SLOPE_BOUNDS = (0.90, 1.20)
INTERCEPT_SCALE_BOUNDS = (-0.15, 0.0)
SMOOTH_L1_BETA = 0.25
BIAS_PENALTY = 0.15
REGULARIZATION = 0.02


@dataclass(frozen=True)
class AffineFit:
    lead: int
    slope: float
    intercept_mm_day: float
    target_mean_mm_day: float
    objective: float
    smooth_l1_component: float
    bias_component: float
    regularization_component: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique comma-separated list")
    return seeds


def _validate_calibration_inputs(
    prediction: np.ndarray,
    truth: np.ndarray,
    area_weights: np.ndarray,
    target_mean_by_lead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forecast = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    scales = np.asarray(target_mean_by_lead, dtype=np.float64)
    if forecast.shape != observed.shape or forecast.ndim != 4:
        raise ValueError("prediction and truth must match [case, lead, lat, lon]")
    if weights.shape != forecast.shape[-2:]:
        raise ValueError("area_weights do not match the spatial grid")
    if scales.shape != (forecast.shape[1],):
        raise ValueError("target_mean_by_lead does not match the lead dimension")
    support = weights > 0.0
    if not np.any(support) or not np.isfinite(weights[support]).all():
        raise ValueError("area_weights need finite positive support")
    if not np.isfinite(forecast[:, :, support]).all():
        raise ValueError("prediction is nonfinite on weighted support")
    if not np.isfinite(observed[:, :, support]).all():
        raise ValueError("truth is nonfinite on weighted support")
    if np.any(forecast[:, :, support] < 0.0) or np.any(
        observed[:, :, support] < 0.0
    ):
        raise ValueError("precipitation must be nonnegative")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("target lead means must be finite and positive")
    return forecast, observed, weights, scales


def _lead_objective_components(
    parameters: Sequence[float],
    prediction: np.ndarray,
    truth: np.ndarray,
    spatial_weights: np.ndarray,
    target_mean: float,
) -> tuple[float, float, float, float]:
    slope, intercept = (float(parameters[0]), float(parameters[1]))
    calibrated = np.maximum(0.0, slope * prediction + intercept)
    normalized_error = (calibrated - truth) / target_mean
    absolute_error = np.abs(normalized_error)
    smooth_l1 = np.where(
        absolute_error < SMOOTH_L1_BETA,
        0.5 * normalized_error**2 / SMOOTH_L1_BETA,
        absolute_error - 0.5 * SMOOTH_L1_BETA,
    )
    denominator = prediction.shape[0] * float(spatial_weights.sum(dtype=np.float64))
    smooth_component = float(
        np.sum(smooth_l1 * spatial_weights[None, :], dtype=np.float64)
        / denominator
    )
    mean_error = float(
        np.sum(
            (calibrated - truth) * spatial_weights[None, :], dtype=np.float64
        )
        / denominator
    )
    bias_component = BIAS_PENALTY * (mean_error / target_mean) ** 2
    regularization_component = REGULARIZATION * (
        (slope - 1.0) ** 2 + (intercept / target_mean) ** 2
    )
    total = smooth_component + bias_component + regularization_component
    return total, smooth_component, bias_component, regularization_component


def fit_leadwise_affine(
    prediction: np.ndarray,
    truth: np.ndarray,
    area_weights: np.ndarray,
    target_mean_by_lead: np.ndarray,
) -> tuple[AffineFit, ...]:
    """Fit six tightly constrained transforms using only supplied cases."""

    forecast, observed, weights, scales = _validate_calibration_inputs(
        prediction, truth, area_weights, target_mean_by_lead
    )
    support = weights > 0.0
    spatial = weights[support]
    fits: list[AffineFit] = []
    for lead_index, target_mean in enumerate(scales):
        lead_prediction = forecast[:, lead_index, support]
        lead_truth = observed[:, lead_index, support]
        bounds = (
            SLOPE_BOUNDS,
            (
                INTERCEPT_SCALE_BOUNDS[0] * float(target_mean),
                INTERCEPT_SCALE_BOUNDS[1] * float(target_mean),
            ),
        )

        def objective(parameters: np.ndarray) -> float:
            return _lead_objective_components(
                parameters,
                lead_prediction,
                lead_truth,
                spatial,
                float(target_mean),
            )[0]

        result = minimize(
            objective,
            x0=np.asarray([1.0, -0.01 * float(target_mean)], dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 300},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f"nonfinite affine optimizer result for lead {lead_index + 1}")
        # L-BFGS-B can report a line-search warning at a valid bounded optimum;
        # retain it transparently but require an objective no worse than identity.
        identity_objective = objective(np.asarray([1.0, 0.0]))
        if float(result.fun) > identity_objective + 1.0e-10:
            raise RuntimeError(
                f"affine fit for lead {lead_index + 1} is worse than identity"
            )
        total, smooth, bias_component, regularization = _lead_objective_components(
            result.x,
            lead_prediction,
            lead_truth,
            spatial,
            float(target_mean),
        )
        fits.append(
            AffineFit(
                lead=lead_index + 1,
                slope=float(result.x[0]),
                intercept_mm_day=float(result.x[1]),
                target_mean_mm_day=float(target_mean),
                objective=float(total),
                smooth_l1_component=float(smooth),
                bias_component=float(bias_component),
                regularization_component=float(regularization),
                optimizer_success=bool(result.success),
                optimizer_message=str(result.message),
                optimizer_iterations=int(result.nit),
            )
        )
    return tuple(fits)


def apply_leadwise_affine(
    prediction: np.ndarray, fits: Sequence[AffineFit]
) -> np.ndarray:
    forecast = np.asarray(prediction, dtype=np.float64)
    if forecast.ndim != 4 or len(fits) != forecast.shape[1]:
        raise ValueError("prediction/fits lead dimensions do not match")
    corrected = np.empty_like(forecast)
    for lead_index, fit in enumerate(fits):
        if fit.lead != lead_index + 1:
            raise ValueError("affine fits are not in lead order")
        corrected[:, lead_index] = np.maximum(
            0.0,
            fit.slope * forecast[:, lead_index] + fit.intercept_mm_day,
        )
    if not np.isfinite(corrected).all() or np.any(corrected < 0.0):
        raise FloatingPointError("affine calibration produced invalid precipitation")
    return corrected.astype(np.float32)


def _reconstruct(
    residual: np.ndarray,
    prepared: bias.PreparedBiasExperiment,
    indices: np.ndarray,
) -> np.ndarray:
    anchor = prepared.anchors["physical_recentered"]
    return reconstruct_anchored_precipitation(
        anchor.bias_baseline[indices],
        residual,
        anchor.target_scale,
        valid_mask=prepared.shared.valid_mask[indices],
    )


def _validation_metrics(
    prediction: np.ndarray,
    prepared: bias.PreparedBiasExperiment,
    configuration: str,
    member: str,
) -> pd.DataFrame:
    frame = bias._case_metrics(prediction, prepared, predictor=configuration)
    frame.insert(0, "member", member)
    frame.insert(0, "configuration", configuration)
    return frame


def _load_model_prediction(
    source: Path,
    prepared: bias.PreparedBiasExperiment,
    seed: int,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    checkpoint_path = (
        source
        / "models"
        / SOURCE_CONFIGURATION
        / f"seed_{seed}"
        / "checkpoints"
        / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    set_deterministic_seed(seed)
    shared = prepared.shared
    model = bias.compact.build_model(
        bias.MODEL_SPEC,
        shared.features.shape[2],
        shared.mean_to_anomaly_ratio,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    residual = predict(
        model,
        shared.features[indices],
        device=device,
        batch_size=batch_size,
        use_amp=device.startswith("cuda"),
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    expected = (len(indices), 6, 27, 27)
    if residual.shape != expected or not np.isfinite(residual).all():
        raise ValueError(f"unexpected predicted residual shape {residual.shape}")
    return residual


def _existing_validation_prediction(
    source: Path,
    prepared: bias.PreparedBiasExperiment,
    configuration: str,
    seed: int,
) -> np.ndarray:
    residual_path = (
        source
        / "models"
        / configuration
        / f"seed_{seed}"
        / "validation_residual.npy"
    )
    residual = np.load(residual_path)
    candidate = bias.CANDIDATE_BY_NAME[configuration]
    anchor = prepared.anchors[candidate.anchor_kind]
    indices = prepared.shared.validation_indices
    return reconstruct_anchored_precipitation(
        anchor.bias_baseline[indices],
        residual,
        anchor.target_scale,
        valid_mask=prepared.shared.valid_mask[indices],
    )


def _existing_validation_ensemble_prediction(
    source: Path,
    prepared: bias.PreparedBiasExperiment,
    configuration: str,
) -> np.ndarray:
    residual = np.load(
        source / "models" / configuration / "validation_residual_ensemble.npy"
    )
    candidate = bias.CANDIDATE_BY_NAME[configuration]
    anchor = prepared.anchors[candidate.anchor_kind]
    indices = prepared.shared.validation_indices
    return reconstruct_anchored_precipitation(
        anchor.bias_baseline[indices],
        residual,
        anchor.target_scale,
        valid_mask=prepared.shared.valid_mask[indices],
    )


def _summary(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (configuration, member), frame in case_metrics.groupby(
        ["configuration", "member"], sort=False
    ):
        row: dict[str, Any] = {
            "configuration": configuration,
            "member": member,
        }
        for metric in ("rmse", "mae", "bias", "acc"):
            row[metric] = float(frame[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    source: Path,
    output: Path,
    seeds: Sequence[int],
    *,
    device: str,
    batch_size: int,
    smoke: bool,
) -> Mapping[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    (output / "metrics").mkdir(parents=True)
    (output / "models").mkdir(parents=True)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if any(year >= 2020 for year in bias.VALIDATION_YEARS):
        raise RuntimeError("validation contract unexpectedly includes 2020+")

    prepared, _, preparation = bias.prepare_data()
    shared = prepared.shared
    bias.validate_quarantined_splits(
        shared.initializations, shared.train_indices, shared.validation_indices
    )
    train_indices = shared.train_indices[:32] if smoke else shared.train_indices
    if np.any(
        np.isin(
            bias.compact._calendar_years(shared.initializations[train_indices]),
            bias.VALIDATION_YEARS + bias.QUARANTINED_YEARS,
        )
    ):
        raise RuntimeError("calibration cases escaped the training-year block")

    coefficient_rows: list[Mapping[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    calibrated_members = []
    uncalibrated_members = []
    reference_members = []
    source_hashes: dict[str, str] = {}
    for seed in seeds:
        print(f"Predicting train-only calibration cases for seed {seed}...", flush=True)
        train_residual = _load_model_prediction(
            source,
            prepared,
            int(seed),
            train_indices,
            device=device,
            batch_size=batch_size,
        )
        train_prediction = _reconstruct(train_residual, prepared, train_indices)
        fits = fit_leadwise_affine(
            train_prediction,
            shared.truth[train_indices],
            shared.weights,
            prepared.bias_scale,
        )
        seed_directory = output / "models" / f"seed_{seed}"
        seed_directory.mkdir()
        for fit in fits:
            coefficient_rows.append({"seed": int(seed), **fit.__dict__})
        coefficients_payload = {
            "seed": int(seed),
            "fit_split": "2002-2017 training cases only",
            "in_sample_calibration_diagnostic": True,
            "fits": [fit.__dict__ for fit in fits],
        }
        (seed_directory / "affine_coefficients.json").write_text(
            json.dumps(coefficients_payload, indent=2) + "\n", encoding="utf-8"
        )

        uncalibrated = _existing_validation_prediction(
            source, prepared, SOURCE_CONFIGURATION, int(seed)
        )
        calibrated = apply_leadwise_affine(uncalibrated, fits)
        reference = _existing_validation_prediction(
            source, prepared, REFERENCE_CONFIGURATION, int(seed)
        )
        np.save(seed_directory / "validation_prediction_calibrated.npy", calibrated)
        calibrated_members.append(calibrated)
        uncalibrated_members.append(uncalibrated)
        reference_members.append(reference)
        member = f"seed_{seed}"
        case_frames.extend(
            (
                _validation_metrics(
                    calibrated, prepared, CALIBRATED_CONFIGURATION, member
                ),
                _validation_metrics(
                    uncalibrated, prepared, SOURCE_CONFIGURATION, member
                ),
                _validation_metrics(
                    reference, prepared, REFERENCE_CONFIGURATION, member
                ),
            )
        )
        checkpoint = (
            source
            / "models"
            / SOURCE_CONFIGURATION
            / f"seed_{seed}"
            / "checkpoints"
            / "best.pt"
        )
        source_hashes[str(checkpoint.relative_to(source))] = sha256_file(checkpoint)

    # A physical-space mean is appropriate because each member has its own
    # fitted physical transform.
    ensembles = {
        CALIBRATED_CONFIGURATION: np.mean(
            calibrated_members, axis=0, dtype=np.float64
        ).astype(np.float32),
        SOURCE_CONFIGURATION: _existing_validation_ensemble_prediction(
            source, prepared, SOURCE_CONFIGURATION
        ),
        REFERENCE_CONFIGURATION: _existing_validation_ensemble_prediction(
            source, prepared, REFERENCE_CONFIGURATION
        ),
    }
    for configuration, prediction_array in ensembles.items():
        np.save(output / "models" / f"{configuration}_validation_ensemble.npy", prediction_array)
        case_frames.append(
            _validation_metrics(
                prediction_array, prepared, configuration, "ensemble"
            )
        )
    raw = shared.raw_fuxi[shared.validation_indices]
    case_frames.append(_validation_metrics(raw, prepared, "raw_fuxi", "deterministic"))

    case_metrics = pd.concat(case_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    summaries = _summary(case_metrics)
    case_metrics.to_csv(output / "metrics" / "validation_case_metrics.csv", index=False)
    coefficients.to_csv(output / "metrics" / "affine_coefficients.csv", index=False)
    summaries.to_csv(output / "metrics" / "physical_summary.csv", index=False)
    bias.intensity_strata_metrics(
        ensembles[CALIBRATED_CONFIGURATION],
        shared.truth[shared.validation_indices],
        shared.weights,
        configuration=CALIBRATED_CONFIGURATION,
    ).to_csv(output / "metrics" / "calibrated_intensity_strata.csv", index=False)

    calibrated_candidate = bias.BiasCandidate(
        CALIBRATED_CONFIGURATION,
        "Recentered bias-aware + train affine",
        "physical_recentered",
        "bias_aware",
        bias.BIAS_AWARE_LOSS,
    )
    # The generic sweep selector defines improvement relative to its reference
    # candidate but also expects the raw baseline in ``case_metrics``.  Reuse
    # that tested implementation here, while keeping this diagnostic's
    # parameter count and objective provenance explicit.
    ranking_candidates = (
        bias.CANDIDATE_BY_NAME[REFERENCE_CONFIGURATION],
        calibrated_candidate,
    )
    records = pd.DataFrame(
        [
            {
                "configuration": candidate.name,
                "parameter_count": 323017 + (
                    12 if candidate.name == CALIBRATED_CONFIGURATION else 0
                ),
                "best_validation_objective": np.nan,
                "elapsed_seconds": np.nan,
            }
            for candidate in ranking_candidates
        ]
    )
    seed_guards = bias.build_seed_physical_guards(
        case_metrics,
        ranking_candidates,
        reference_configuration=REFERENCE_CONFIGURATION,
    )
    ranking = bias.build_physical_ranking(
        records,
        case_metrics,
        ranking_candidates,
        reference_configuration=REFERENCE_CONFIGURATION,
        seed_guards=seed_guards,
    )
    seed_guards.to_csv(output / "metrics" / "seed_physical_guards.csv", index=False)
    ranking.to_csv(output / "metrics" / "strict_screening_ranking.csv", index=False)
    candidate_row = ranking.loc[
        ranking.configuration.eq(CALIBRATED_CONFIGURATION)
    ].iloc[0]
    promising = bool(candidate_row.qualifies) and not smoke
    disposition = (
        "promising_requires_blocked_oof_confirmation"
        if promising
        else "not_promoted"
    )
    manifest = {
        "created_utc": utc_now(),
        "status": "complete",
        "disposition": disposition,
        "scientific_status": (
            "in-sample train-calibration diagnostic; not independent evidence"
        ),
        "source": str(source),
        "source_configuration": SOURCE_CONFIGURATION,
        "reference_configuration": REFERENCE_CONFIGURATION,
        "calibrated_configuration": CALIBRATED_CONFIGURATION,
        "seeds": [int(seed) for seed in seeds],
        "smoke": bool(smoke),
        "device": device,
        "calibration_case_count": int(len(train_indices)),
        "calibration_years": list(bias.TRAIN_YEARS),
        "evaluation_years": list(bias.VALIDATION_YEARS),
        "quarantined_years": list(bias.QUARANTINED_YEARS),
        "quarantined_predictions_or_metrics_produced": False,
        "bounds": {
            "slope": list(SLOPE_BOUNDS),
            "intercept_as_fraction_of_train_imd_lead_mean": list(
                INTERCEPT_SCALE_BOUNDS
            ),
        },
        "objective": {
            "smooth_l1_beta_normalized_units": SMOOTH_L1_BETA,
            "bias_penalty": BIAS_PENALTY,
            "regularization": REGULARIZATION,
            "target_scale": "train-only area/case-weighted IMD mean per lead",
        },
        "candidate_passed_strict_diagnostic_guards": bool(candidate_row.qualifies),
        "diagnostic_guard_interpretation": (
            "The generic strict guards require >=20% pooled and lead-wise "
            "absolute-bias improvement versus the established neural control; "
            "absolute bias must also be no worse than raw FuXi."
        ),
        "source_checkpoint_sha256": source_hashes,
        "preparation_metadata": preparation["metadata"],
        "required_next_step_if_promising": (
            "fit coefficients from blocked out-of-fold 2002-2017 predictions, "
            "freeze them, then evaluate the development block once"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(summaries.to_string(index=False), flush=True)
    print(ranking[["configuration", "pooled_rmse", "pooled_mae", "pooled_bias", "pooled_acc", "qualifies"]].to_string(index=False), flush=True)
    print(f"disposition={disposition}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size < 1:
        raise ValueError("batch-size must be positive")
    seeds = parse_seeds(arguments.seeds)
    if arguments.smoke:
        seeds = seeds[:1]
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run(
        arguments.source.resolve(),
        arguments.output.resolve(),
        seeds,
        device=arguments.device,
        batch_size=arguments.batch_size,
        smoke=arguments.smoke,
    )


if __name__ == "__main__":
    main()
