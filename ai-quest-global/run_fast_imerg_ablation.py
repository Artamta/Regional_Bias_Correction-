#!/usr/bin/env python3
"""Score fast FuXi/IMERG checkpoint, ensemble, and calibration ablations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_uniform_blend import (
    aggregate_rps,
    blend_probabilities,
    fit_uniform_blend_alpha,
)
from evaluate import _base_spatial_weight
from predict import load_checkpoint_model, load_prepared, run_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not all(character.isalnum() or character in "_-" for character in name):
        raise argparse.ArgumentTypeError("checkpoint name contains unsafe characters")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return name, path


def _ensemble(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ensemble must be NAME=MEMBER1,MEMBER2,...")
    name, raw_members = value.split("=", 1)
    if not name or not all(character.isalnum() or character in "_-" for character in name):
        raise argparse.ArgumentTypeError("ensemble name contains unsafe characters")
    members = [member.strip() for member in raw_members.split(",") if member.strip()]
    if len(members) < 2:
        raise argparse.ArgumentTypeError("ensemble requires at least two members")
    if len(members) != len(set(members)):
        raise argparse.ArgumentTypeError("ensemble members must be unique")
    return name, members


def _predict(path: Path, prepared, device: str, batch_size: int) -> tuple[np.ndarray, int]:
    model, payload = load_checkpoint_model(
        path, device=device, in_channels=int(prepared.features.shape[2])
    )
    state = payload["model_state"]
    parameters = int(sum(value.numel() for value in state.values()))
    output = np.empty_like(prepared.p0, dtype=np.float32)
    for start in range(0, prepared.n_cases, batch_size):
        stop = min(start + batch_size, prepared.n_cases)
        output[start:stop] = run_model(
            model,
            prepared.features[start:stop],
            prepared.p0[start:stop],
            device=device,
        )
    return output, parameters


def _score(probability: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = [
        aggregate_rps(probability[:, lead : lead + 1], target[:, lead : lead + 1], weight).rps
        for lead in range(2)
    ]
    values.append(aggregate_rps(probability, target, weight).rps)
    return np.asarray(values, dtype=np.float64)


def _calibrate(
    fit_probability: np.ndarray,
    fit_target: np.ndarray,
    evaluation_probability: np.ndarray,
    weight: np.ndarray,
    scope: str,
) -> tuple[np.ndarray, list[float]]:
    if scope == "shared":
        alpha = fit_uniform_blend_alpha(fit_probability, fit_target, weight)
        return blend_probabilities(evaluation_probability, alpha, category_axis=2), [alpha, alpha]
    alphas = [
        fit_uniform_blend_alpha(
            fit_probability[:, lead : lead + 1], fit_target[:, lead : lead + 1], weight
        )
        for lead in range(2)
    ]
    result = np.empty_like(evaluation_probability)
    for lead, alpha in enumerate(alphas):
        result[:, lead : lead + 1] = blend_probabilities(
            evaluation_probability[:, lead : lead + 1], alpha, category_axis=2
        )
    return result, alphas


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fit = load_prepared(args.cache, require_target=True, years=args.fit_years)
    evaluation = load_prepared(
        args.cache, require_target=True, years=args.evaluation_years
    )
    assert fit.target is not None and evaluation.target is not None
    weight = _base_spatial_weight(weighting="area", land_fraction=None)
    uniform_fit = np.full_like(fit.p0, 0.2)
    uniform_evaluation = np.full_like(evaluation.p0, 0.2)
    fit_forecasts: dict[str, np.ndarray] = {"uniform": uniform_fit, "p0": fit.p0}
    evaluation_forecasts: dict[str, np.ndarray] = {
        "uniform": uniform_evaluation,
        "p0": evaluation.p0,
    }
    parameters: dict[str, int] = {"uniform": 0, "p0": 0}
    checkpoints: dict[str, Path] = dict(args.checkpoint)
    for name, path in checkpoints.items():
        fit_forecasts[name], parameters[name] = _predict(
            path, fit, args.device, args.batch_size
        )
        evaluation_forecasts[name], _ = _predict(
            path, evaluation, args.device, args.batch_size
        )
    ensembles: dict[str, list[str]] = dict(args.ensemble)
    if args.ensemble_member:
        if "seed_ensemble" in ensembles:
            raise ValueError("seed_ensemble is defined twice")
        ensembles["seed_ensemble"] = args.ensemble_member
    for ensemble_name, members in ensembles.items():
        if ensemble_name in fit_forecasts:
            raise ValueError(f"ensemble name is already in use: {ensemble_name}")
        missing = [name for name in members if name not in checkpoints]
        if missing:
            raise ValueError(f"ensemble members are not named checkpoints: {missing}")
        if len(members) < 2:
            raise ValueError("ensemble requires at least two checkpoint names")
        fit_forecasts[ensemble_name] = np.mean(
            [fit_forecasts[name] for name in members], axis=0
        ).astype(np.float32)
        evaluation_forecasts[ensemble_name] = np.mean(
            [evaluation_forecasts[name] for name in members], axis=0
        ).astype(np.float32)
        parameters[ensemble_name] = sum(
            parameters[name] for name in members
        )

    rows: list[dict[str, object]] = []
    alphas: dict[str, dict[str, list[float]]] = {}
    for split_name, forecasts, truth in (
        ("fit", fit_forecasts, fit.target),
        ("evaluation", evaluation_forecasts, evaluation.target),
    ):
        uniform_score = _score(forecasts["uniform"], truth, weight)
        p0_score = _score(forecasts["p0"], truth, weight)
        for name, probability in forecasts.items():
            variants = [("none", probability, [1.0, 1.0])]
            if name != "uniform":
                for scope in ("shared", "by_lead"):
                    calibrated, coefficients = _calibrate(
                        fit_forecasts[name], fit.target, probability, weight, scope
                    )
                    variants.append((scope, calibrated, coefficients))
                    alphas.setdefault(name, {})[scope] = coefficients
            for calibration, candidate, coefficients in variants:
                result = _score(candidate, truth, weight)
                for lead_index, lead_name in enumerate(("D19-25", "D26-32", "pooled")):
                    rows.append(
                        {
                            "split": split_name,
                            "system": name,
                            "calibration": calibration,
                            "lead": lead_name,
                            "parameters": parameters[name],
                            "alpha": (
                                float(np.mean(coefficients))
                                if lead_index == 2
                                else float(coefficients[lead_index])
                            ),
                            "rps": float(result[lead_index]),
                            "relative_improvement_vs_p0_percent": float(
                                100.0 * (p0_score[lead_index] - result[lead_index])
                                / p0_score[lead_index]
                            ),
                            "rpss_vs_uniform": float(
                                1.0 - result[lead_index] / uniform_score[lead_index]
                            ),
                        }
                    )

    fields = list(rows[0])
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "exploratory_not_official_ai_weather_quest_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "IMERG Final V07B",
        "weighting": "global cosine-area; no official land mask",
        "fit_years": args.fit_years,
        "evaluation_years": args.evaluation_years,
        "fit_cases": fit.n_cases,
        "evaluation_cases": evaluation.n_cases,
        "cache": str(args.cache.resolve()),
        "cache_sha256": _sha256(args.cache),
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path), "parameters": parameters[name]}
            for name, path in checkpoints.items()
        },
        "ensembles": ensembles,
        "calibration_alphas": alphas,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "comparison.csv")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--checkpoint", action="append", type=_checkpoint, required=True)
    result.add_argument(
        "--ensemble-member",
        action="append",
        help="named checkpoint to include in seed_ensemble; repeat at least twice",
    )
    result.add_argument(
        "--ensemble",
        action="append",
        type=_ensemble,
        default=[],
        metavar="NAME=MEMBER1,MEMBER2,...",
        help="named checkpoint ensemble; repeat for multiple ensembles",
    )
    result.add_argument("--fit-years", type=int, nargs="+", default=[2019])
    result.add_argument("--evaluation-years", type=int, nargs="+", default=[2020])
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--batch-size", type=int, default=2)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
