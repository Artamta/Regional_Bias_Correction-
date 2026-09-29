#!/usr/bin/env python3
"""Score controls, enforce promotion rules, freeze one candidate, and optionally test."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from debias import SeasonalDebiasPlusPlus, apply_uniform_gamma, fit_uniform_gamma
from precip_contract import SPLIT, contract_sha256
from precip_model import PrecipQuestAdapter
from train_precip import CacheCases, evaluate, spatial_weight


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(group, years: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(group["init_yyyymmdd"][:], dtype=np.int32)
    return np.flatnonzero(np.isin(values // 10000, years))


def target_day_of_year(group, indices: np.ndarray) -> np.ndarray:
    """Return zero-based target-start day of year for both Quest leads."""

    initializations = pd.to_datetime(
        np.asarray(group["init_yyyymmdd"].oindex[indices], dtype=np.int32).astype(str),
        format="%Y%m%d",
    )
    return np.stack(
        [
            (initializations + pd.Timedelta(days=offset)).dayofyear.to_numpy() - 1
            for offset in (18, 25)
        ],
        axis=1,
    ).astype(np.int16)


def load_truth(group, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(group["target"].oindex[indices], dtype=np.int8),
        np.asarray(group["target_valid"].oindex[indices], dtype=bool),
    )


def load_p0(group, indices: np.ndarray) -> np.ndarray:
    return np.asarray(group["p0"].oindex[indices], dtype=np.float32)


def score(
    probabilities: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=np.float64)
    forecast_cdf = np.cumsum(probability, axis=2)[:, :, :4]
    observed_cdf = np.stack(
        [(target <= boundary) for boundary in range(4)], axis=2
    ).astype(np.float64)
    full_weight = valid[:, :, None] * weight[None, None, None]
    numerator = ((forecast_cdf - observed_cdf) ** 2 * full_weight).sum(
        axis=(0, 2, 3, 4)
    )
    denominator = full_weight.sum(axis=(0, 2, 3, 4))
    if np.any(denominator <= 0):
        raise RuntimeError("a lead has no valid land-area weight")
    return (numerator / denominator).astype(np.float64)


def fit_debias_streaming(group, indices: np.ndarray) -> SeasonalDebiasPlusPlus:
    height, width = group["p0"].shape[-2:]
    debias = SeasonalDebiasPlusPlus.empty(height, width)
    target_doys = target_day_of_year(group, indices)
    for position, index in enumerate(indices):
        p0 = np.asarray(group["p0"][int(index)], dtype=np.float32)
        target = np.asarray(group["target"][int(index)], dtype=np.int8)
        valid = np.asarray(group["target_valid"][int(index)], dtype=bool)
        debias.add_case(p0, target, valid, target_doys[position])
        if (position + 1) % 100 == 0:
            print(f"fit Debias++ {position + 1}/{len(indices)}", flush=True)
    return debias


def _neural_metadata(run: Path) -> dict[str, object]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    if summary["configuration_sha256"] != checkpoint["configuration_sha256"]:
        raise ValueError(f"summary/checkpoint configuration mismatch in {run}")
    if summary.get("status") != "complete" or summary.get("smoke_only", False):
        raise ValueError(f"smoke or incomplete run is not selectable: {run}")
    return {
        "run": run,
        "variant": checkpoint["train_config"]["variant"],
        "width": int(checkpoint["train_config"]["width"]),
        "seed": int(checkpoint["train_config"]["seed"]),
        "parameters": int(summary["parameters"]),
        "configuration_sha256": summary["configuration_sha256"],
    }


def _neural_validation(run: Path, expected_cases: int) -> np.ndarray:
    path = run / "predictions" / "validation.npy"
    values = np.load(path, mmap_mode="r")
    if values.shape != (expected_cases, 2, 5, 121, 240):
        raise ValueError(f"{path} has invalid shape {values.shape}")
    return np.asarray(values, dtype=np.float32)


def _relative_improvement(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (baseline - candidate) / baseline


def _pooled_relative_improvement(candidate: np.ndarray, baseline: np.ndarray) -> float:
    return float((baseline.mean() - candidate.mean()) / baseline.mean())


def _row(name: str, parameters: int, validation: np.ndarray, uniform: np.ndarray):
    return {
        "candidate": name,
        "parameters": parameters,
        "validation_rps_d19_25": float(validation[0]),
        "validation_rps_d26_32": float(validation[1]),
        "validation_rps_pooled": float(validation.mean()),
        "validation_rpss_d19_25": float(1.0 - validation[0] / uniform[0]),
        "validation_rpss_d26_32": float(1.0 - validation[1] / uniform[1]),
        "validation_rpss_pooled": float(1.0 - validation.mean() / uniform.mean()),
        "test_rps_d19_25": "",
        "test_rps_d26_32": "",
        "test_rps_pooled": "",
        "test_rpss_d19_25": "",
        "test_rpss_d26_32": "",
        "test_rpss_pooled": "",
    }


def _write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_rps(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["candidate"]) for row in rows]
    values = [float(row["validation_rps_pooled"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
    order = np.argsort(values)[::-1]
    axis.barh(np.asarray(names)[order], np.asarray(values)[order], color="#315c91")
    axis.set_xlabel("Validation RPS (lower is better)")
    axis.set_title("AI Weather Quest precipitation controls and ablations")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_reliability(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    weight: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 11)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for lead, axis in enumerate(axes):
        x_values, y_values = [], []
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected_weight = np.zeros_like(probability[:, lead], dtype=np.float64)
            in_bin = (probability[:, lead] >= lower) & (
                (probability[:, lead] < upper) if upper < 1 else (probability[:, lead] <= upper)
            )
            selected_weight[:] = (
                valid[:, lead, None] * weight[None, None] * in_bin
            )
            denominator = selected_weight.sum()
            if denominator <= 0:
                continue
            observation = np.stack(
                [(target[:, lead] == category) for category in range(5)], axis=1
            )
            x_values.append(float((selected_weight * probability[:, lead]).sum() / denominator))
            y_values.append(float((selected_weight * observation).sum() / denominator))
        axis.plot([0, 1], [0, 1], "--", color="0.5")
        axis.plot(x_values, y_values, marker="o")
        axis.set_title(("D19-25", "D26-32")[lead])
        axis.set_xlabel("Forecast probability")
    axes[0].set_ylabel("Observed frequency")
    figure.suptitle("Validation reliability of frozen candidate")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _predict_neural_split(
    cache: Path,
    metadata: dict[str, object],
    years: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    run = Path(metadata["run"])
    checkpoint = torch.load(
        run / "checkpoints" / "best.pt", map_location=device, weights_only=False
    )
    model = PrecipQuestAdapter(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dataset = CacheCases(cache, years, str(metadata["variant"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, prediction = evaluate(
        model, loader, device, spatial_weight(cache).to(device), save_predictions=True
    )
    assert prediction is not None
    return prediction


def select(args: argparse.Namespace) -> None:
    import zarr

    args.output_dir.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(args.cache), mode="r")
    if group.attrs.get("status") != "complete" or group.attrs.get("contract_sha256") != contract_sha256():
        raise ValueError("cache is not the complete frozen precipitation contract")
    train_indices = split_indices(group, SPLIT.train_years)
    validation_indices = split_indices(group, SPLIT.validation_years)
    expected_split_counts = (
        104 * len(SPLIT.train_years),
        104 * len(SPLIT.validation_years),
    )
    if (len(train_indices), len(validation_indices)) != expected_split_counts:
        raise RuntimeError("cache split counts do not match the frozen plan")
    validation_target, validation_valid = load_truth(group, validation_indices)
    validation_doys = target_day_of_year(group, validation_indices)
    p0 = load_p0(group, validation_indices)
    latitude = np.asarray(group["latitude"][:], dtype=np.float64)
    land = np.asarray(group["land_fraction"][:], dtype=np.float64)
    weight = np.maximum(np.cos(np.deg2rad(latitude)), 0.0)[:, None] * (land >= 0.5)
    uniform_probability = np.full_like(p0, 0.2)
    uniform_rps = score(uniform_probability, validation_target, validation_valid, weight)
    p0_rps = score(p0, validation_target, validation_valid, weight)
    p0_gamma = fit_uniform_gamma(p0, validation_target, validation_valid, weight)
    calibrated_p0 = apply_uniform_gamma(p0, p0_gamma)
    calibrated_p0_rps = score(calibrated_p0, validation_target, validation_valid, weight)

    debias = fit_debias_streaming(group, train_indices)
    span_scores: dict[str, list[float]] = {}
    selected_spans = np.full(2, -1, dtype=np.int16)
    best_span_score = np.full(2, np.inf, dtype=np.float64)
    debias_probability = np.empty_like(p0)
    for span in (14, 28, 35):
        candidate_probability = debias.predict(
            p0, validation_doys, np.asarray([span, span], dtype=np.int16)
        )
        candidate_score = score(
            candidate_probability, validation_target, validation_valid, weight
        )
        span_scores[str(span)] = candidate_score.tolist()
        for lead in range(2):
            if candidate_score[lead] < best_span_score[lead]:
                best_span_score[lead] = candidate_score[lead]
                selected_spans[lead] = span
                debias_probability[:, lead] = candidate_probability[:, lead]
        del candidate_probability
    debias_rps = score(debias_probability, validation_target, validation_valid, weight)
    debias_gamma = fit_uniform_gamma(
        debias_probability, validation_target, validation_valid, weight
    )
    calibrated_debias = apply_uniform_gamma(debias_probability, debias_gamma)
    calibrated_debias_rps = score(
        calibrated_debias, validation_target, validation_valid, weight
    )
    np.savez_compressed(
        args.output_dir / "debias_plus_plus.npz",
        error_sum=debias.error_sum,
        valid_count=debias.valid_count,
        clip=np.asarray(debias.clip),
        selected_span_by_lead=selected_spans,
        gamma=debias_gamma,
        cache_contract_sha256=np.asarray(contract_sha256()),
    )

    rows = [
        _row("uniform", 0, uniform_rps, uniform_rps),
        _row("raw_p0", 0, p0_rps, uniform_rps),
        _row("calibrated_p0", 2, calibrated_p0_rps, uniform_rps),
        _row("debias_plus_plus", int(debias.error_sum.size), debias_rps, uniform_rps),
        _row(
            "calibrated_debias_plus_plus",
            int(debias.error_sum.size + 2),
            calibrated_debias_rps,
            uniform_rps,
        ),
    ]
    probabilities: dict[str, np.ndarray] = {
        "uniform": uniform_probability,
        "raw_p0": p0,
        "calibrated_p0": calibrated_p0,
        "debias_plus_plus": debias_probability,
        "calibrated_debias_plus_plus": calibrated_debias,
    }

    neural = [_neural_metadata(path) for path in args.neural_run]
    neural_by_key = {
        (str(item["variant"]), int(item["width"]), int(item["seed"])): item
        for item in neural
    }
    for item in neural:
        name = f"neural_{item['variant']}_w{item['width']}_seed{item['seed']}"
        prediction = _neural_validation(Path(item["run"]), len(validation_indices))
        probabilities[name] = prediction
        rows.append(
            _row(name, int(item["parameters"]), score(prediction, validation_target, validation_valid, weight), uniform_rps)
        )

    best_non_neural = min(rows[:5], key=lambda row: float(row["validation_rps_pooled"]))
    promotion: dict[str, object] = {"best_non_neural": best_non_neural["candidate"]}
    chosen_neural: str | None = None
    target_key = ("target_only", 16, 42)
    tp_key = ("tp_only", 16, 42)
    full_key = ("full", 16, 42)
    allowed_keys = [key for key in (target_key, tp_key) if key in neural_by_key]
    if full_key in neural_by_key and tp_key in neural_by_key:
        tp_name = "neural_tp_only_w16_seed42"
        full_name = "neural_full_w16_seed42"
        tp_score = score(probabilities[tp_name], validation_target, validation_valid, weight)
        full_score = score(probabilities[full_name], validation_target, validation_valid, weight)
        improvement = _relative_improvement(full_score, tp_score)
        pooled_improvement = _pooled_relative_improvement(full_score, tp_score)
        full_advances = pooled_improvement >= 0.0025 and np.all(full_score <= tp_score)
        promotion["full_vs_tp_only"] = {
            "relative_improvement_by_lead": improvement.tolist(),
            "pooled_relative_improvement": pooled_improvement,
            "advances": bool(full_advances),
        }
        if full_advances:
            allowed_keys.append(full_key)
    if allowed_keys:
        allowed_names = [f"neural_{key[0]}_w{key[1]}_seed{key[2]}" for key in allowed_keys]
        chosen_neural = min(
            allowed_names,
            key=lambda name: float(score(probabilities[name], validation_target, validation_valid, weight).mean()),
        )

    if chosen_neural:
        base_score = score(
            probabilities[chosen_neural], validation_target, validation_valid, weight
        )
        nonneural_score = np.asarray(
            [
                best_non_neural["validation_rps_d19_25"],
                best_non_neural["validation_rps_d26_32"],
            ],
            dtype=float,
        )
        width16_improvement = _relative_improvement(base_score, nonneural_score)
        width24_allowed = (
            _pooled_relative_improvement(base_score, nonneural_score) >= 0.005
            and np.all(width16_improvement >= 0.005)
        )
        promotion["width24_precondition"] = bool(width24_allowed)
        # Variant names containing an underscore are handled from metadata.
        chosen_item = next(
            item
            for item in neural
            if f"neural_{item['variant']}_w{item['width']}_seed{item['seed']}" == chosen_neural
        )
        width24_key = (str(chosen_item["variant"]), 24, 42)
        if width24_allowed and width24_key in neural_by_key:
            width24_name = f"neural_{width24_key[0]}_w24_seed42"
            width24_score = score(
                probabilities[width24_name], validation_target, validation_valid, weight
            )
            retain = (
                _pooled_relative_improvement(width24_score, base_score) >= 0.002
                and np.all(width24_score <= base_score)
            )
            promotion["width24_retain"] = bool(retain)
            if retain:
                chosen_neural = width24_name

    if chosen_neural:
        chosen_item = next(
            item
            for item in neural
            if f"neural_{item['variant']}_w{item['width']}_seed{item['seed']}" == chosen_neural
        )
        seed_names = [
            f"neural_{chosen_item['variant']}_w{chosen_item['width']}_seed{seed}"
            for seed in (42, 43, 44)
        ]
        available = [name for name in seed_names if name in probabilities]
        if len(available) == 3:
            beats = [
                np.all(score(probabilities[name], validation_target, validation_valid, weight) < p0_rps)
                for name in available
            ]
            ensemble = np.mean([probabilities[name] for name in available], axis=0)
            ensemble_rps = score(ensemble, validation_target, validation_valid, weight)
            nonneural_rps = np.asarray(
                [best_non_neural["validation_rps_d19_25"], best_non_neural["validation_rps_d26_32"]]
            )
            retain_ensemble = sum(beats) >= 2 and float(ensemble_rps.mean()) < float(nonneural_rps.mean())
            promotion["three_seed_ensemble"] = {
                "seeds_beating_p0_both_leads": int(sum(beats)),
                "beats_best_non_neural": bool(ensemble_rps.mean() < nonneural_rps.mean()),
                "retained": bool(retain_ensemble),
            }
            if retain_ensemble:
                chosen_neural = f"neural_{chosen_item['variant']}_w{chosen_item['width']}_ensemble"
                probabilities[chosen_neural] = ensemble
                rows.append(_row(chosen_neural, 3 * int(chosen_item["parameters"]), ensemble_rps, uniform_rps))

    calibrated_neural_name = None
    neural_gamma = None
    if chosen_neural:
        neural_gamma = fit_uniform_gamma(
            probabilities[chosen_neural], validation_target, validation_valid, weight
        )
        calibrated_neural_name = f"calibrated_{chosen_neural}"
        probabilities[calibrated_neural_name] = apply_uniform_gamma(
            probabilities[chosen_neural], neural_gamma
        )
        neural_rps = score(
            probabilities[calibrated_neural_name], validation_target, validation_valid, weight
        )
        parameters = next(
            int(row["parameters"]) for row in rows if row["candidate"] == chosen_neural
        )
        rows.append(_row(calibrated_neural_name, parameters + 2, neural_rps, uniform_rps))

    eligible = [str(best_non_neural["candidate"])]
    if calibrated_neural_name:
        eligible.append(calibrated_neural_name)
    winner = min(
        eligible,
        key=lambda name: float(score(probabilities[name], validation_target, validation_valid, weight).mean()),
    )
    winner_probability = probabilities[winner]
    selection_payload = {
        "status": "frozen_from_validation",
        "winner": winner,
        "eligible_candidates": eligible,
        "promotion": promotion,
        "p0_gamma_by_lead": p0_gamma.tolist(),
        "debias_gamma_by_lead": debias_gamma.tolist(),
        "debias_span_by_lead": selected_spans.tolist(),
        "debias_validation_rps_by_span": span_scores,
        "neural_gamma_by_lead": None if neural_gamma is None else neural_gamma.tolist(),
        "cache_contract_sha256": contract_sha256(),
        "neural_configurations": [
            {key: (str(value) if isinstance(value, Path) else value) for key, value in item.items()}
            for item in neural
        ],
        "frozen_utc": utc_now(),
    }
    encoded = json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    selection_payload["selection_sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output_dir / "frozen_selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n"
    )
    _write_comparison(args.output_dir / "comparison.csv", rows)
    _plot_rps(rows, args.output_dir / "validation_rps.png")
    _plot_reliability(
        winner_probability,
        validation_target,
        validation_valid,
        weight,
        args.output_dir / "validation_reliability.png",
    )

    hashes = {
        "cache_contract_sha256": contract_sha256(),
        "selection_sha256": selection_payload["selection_sha256"],
        "input_files": {
            str(Path(item["run"]) / "checkpoints" / "best.pt"): sha256_file(
                Path(item["run"]) / "checkpoints" / "best.pt"
            )
            for item in neural
        },
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )

    if args.open_frozen_test:
        _open_test(
            args,
            group,
            selection_payload,
            rows,
            debias,
            neural,
            p0_gamma,
            debias_gamma,
            selected_spans,
            neural_gamma,
            weight,
        )
    print(json.dumps(selection_payload, indent=2))


def _open_test(
    args: argparse.Namespace,
    group,
    selection: dict[str, object],
    rows: list[dict[str, object]],
    debias: SeasonalDebiasPlusPlus,
    neural: list[dict[str, object]],
    p0_gamma: np.ndarray,
    debias_gamma: np.ndarray,
    debias_spans: np.ndarray,
    neural_gamma: np.ndarray | None,
    weight: np.ndarray,
) -> None:
    sentinel = args.output_dir / "frozen_test_opened.json"
    if sentinel.exists():
        raise RuntimeError("frozen test has already been opened for this selection")
    sentinel.write_text(
        json.dumps(
            {
                "status": "opening_irreversible",
                "selection_sha256": selection["selection_sha256"],
                "opened_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    test_indices = split_indices(group, SPLIT.test_years)
    expected_test_cases = 104 * len(SPLIT.test_years)
    if len(test_indices) != expected_test_cases:
        raise RuntimeError(
            f"frozen test must contain exactly {expected_test_cases} cases"
        )
    target, valid = load_truth(group, test_indices)
    p0 = load_p0(group, test_indices)
    target_doys = target_day_of_year(group, test_indices)
    uniform = np.full_like(p0, 0.2)
    test_probabilities = {
        "uniform": uniform,
        "raw_p0": p0,
        "calibrated_p0": apply_uniform_gamma(p0, p0_gamma),
        "debias_plus_plus": debias.predict(p0, target_doys, debias_spans),
    }
    test_probabilities["calibrated_debias_plus_plus"] = apply_uniform_gamma(
        test_probabilities["debias_plus_plus"], debias_gamma
    )
    winner = str(selection["winner"])
    if "neural" in winner:
        selected_configs = [
            item
            for item in neural
            if str(item["variant"]) in winner and f"w{item['width']}" in winner
        ]
        if "ensemble" not in winner:
            selected_configs = [item for item in selected_configs if f"seed{item['seed']}" in winner]
        predictions = [
            _predict_neural_split(
                args.cache,
                item,
                SPLIT.test_years,
                torch.device(args.device),
                args.batch_size,
            )
            for item in selected_configs
        ]
        neural_prediction = np.mean(predictions, axis=0)
        if winner.startswith("calibrated_"):
            assert neural_gamma is not None
            neural_prediction = apply_uniform_gamma(neural_prediction, neural_gamma)
        test_probabilities[winner] = neural_prediction
    uniform_rps = score(uniform, target, valid, weight)
    for row in rows:
        name = str(row["candidate"])
        if name not in test_probabilities:
            continue
        result = score(test_probabilities[name], target, valid, weight)
        row["test_rps_d19_25"] = float(result[0])
        row["test_rps_d26_32"] = float(result[1])
        row["test_rps_pooled"] = float(result.mean())
        row["test_rpss_d19_25"] = float(1 - result[0] / uniform_rps[0])
        row["test_rpss_d26_32"] = float(1 - result[1] / uniform_rps[1])
        row["test_rpss_pooled"] = float(1 - result.mean() / uniform_rps.mean())
    _write_comparison(args.output_dir / "comparison.csv", rows)
    sentinel.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_sha256": selection["selection_sha256"],
                "opened_utc": utc_now(),
                "test_cases": expected_test_cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--neural-run", type=Path, action="append", default=[])
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--open-frozen-test", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=2)
    return result


if __name__ == "__main__":
    select(parser().parse_args())
