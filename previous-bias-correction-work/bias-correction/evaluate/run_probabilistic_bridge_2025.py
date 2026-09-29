#!/usr/bin/env python3
"""Frozen 2025 forecast-only inference for Probabilistic Correction v1.

This isolated prospective runner applies the already-selected seed-43
location-and-spread adapter to the 35 initialization dates frozen by the
deterministic 2025 protocol.  It never fits, tunes, or selects a model and it
does not open IMD 2025 observations.  A later, separately documented scoring
step may join its forecasts to the authorized IMD evaluation fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import india_s2s_probabilistic_bridge as bridge
import india_s2s_probabilistic_bridge_run as inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_probabilistic_bridge"
EXPECTED_SHAPE = (50, 6, 27, 27)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selection_dates(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["evaluation_schedule"]["initializations"]
    result = np.asarray(values, dtype="datetime64[D]")
    if result.shape != (35,) or result.min() != np.datetime64("2025-06-02"):
        raise ValueError("expected the frozen 35-case 2025 deterministic cohort")
    if result.max() != np.datetime64("2025-09-29") or np.unique(result).size != 35:
        raise ValueError("frozen 2025 initialization cohort changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if OUTPUT_ROOT.resolve() not in output.parents or output.exists():
        raise FileExistsError("output must be a fresh directory under resultsv3/india_s2s_probabilistic_bridge")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")

    selected = selection_dates(args.selection.resolve())
    if args.max_cases is not None:
        if not 1 <= args.max_cases <= selected.size:
            raise ValueError("max cases must be in 1..35")
        selected = selected[: args.max_cases]
    output.mkdir(parents=True)
    parent = inference.load_frozen_parent(args.parent_manifest.resolve())
    context = inference._load_context_artifact(output, parent)
    model = inference.load_selected_model(parent, torch.device("cuda"))
    loader = inference._forecast_loader()
    dataset = loader.open_forecast_dataset(
        model="fuxi_s2s", variable="tp", year=2025, grid="common_1p5",
        experiment_id=bridge.FUXI_EXPERIMENT_ID,
    )
    try:
        variable = dataset["forecast_weekly_mean"]
        if variable.dims != ("init", "member", "lead_week", "latitude", "longitude"):
            raise ValueError("2025 FuXi member dimensions changed")
        if variable.shape[1:] != EXPECTED_SHAPE:
            raise ValueError(f"2025 FuXi member shape changed: {variable.shape}")
        all_inits = np.asarray(dataset.init.values, dtype="datetime64[D]")
        positions = np.searchsorted(all_inits, selected)
        if np.any(positions >= all_inits.size) or not np.array_equal(all_inits[positions], selected):
            raise ValueError("one or more frozen starts are absent from the 2025 archive")
        frozen_context = inference.build_frozen_context(selected, context)
        count = selected.size
        delta = np.empty((count, 6, 27, 27), dtype=np.float32)
        log_spread = np.empty_like(delta)
        raw_mean = np.empty_like(delta)
        corrected_mean = np.empty_like(delta)
        digest = hashlib.sha256()
        for begin in range(0, count, args.batch_size):
            end = min(begin + args.batch_size, count)
            raw = np.asarray(variable.isel(init=positions[begin:end]).load().values, dtype=np.float32)
            if raw.shape != (end - begin, *EXPECTED_SHAPE) or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError("invalid raw 2025 FuXi member batch")
            inference._content_digest_update(digest, raw)
            fields = np.stack([inference.calibration.context_for_case(frozen_context, index) for index in range(begin, end)])
            batch_delta, batch_spread = inference.infer_adjustments(
                model, raw, fields, device=torch.device("cuda"), use_amp=True
            )
            corrected = inference.reconstruct_corrected_members(raw, batch_delta, batch_spread)
            delta[begin:end] = batch_delta
            log_spread[begin:end] = batch_spread
            raw_mean[begin:end] = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
            corrected_mean[begin:end] = corrected.mean(axis=1, dtype=np.float64).astype(np.float32)
    finally:
        dataset.close()

    shard = output / "adjustments_2025.npz"
    np.savez_compressed(
        shard, initializations=selected, delta_log_location=delta, log_spread=log_spread,
        raw_ensemble_mean=raw_mean, corrected_ensemble_mean=corrected_mean,
        latitude=context.latitude, longitude=context.longitude,
        member_count=np.asarray(50, dtype=np.int16), selected_seed=np.asarray(43, dtype=np.int16),
    )
    manifest = {
        "experiment": "india_s2s_probabilistic_bridge_independent_2025_inference_v1",
        "status": "complete" if selected.size == 35 else "smoke_complete",
        "scientific_status": "frozen forecast-only 2025 prospective inference; no IMD 2025 observations opened",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_receipt": parent.receipt,
        "selection_manifest": str(args.selection.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection.resolve()),
        "initialization_dates": [np.datetime_as_string(value, unit="D") for value in selected],
        "initialization_dates_sha256": bridge.initialization_dates_sha256(selected),
        "selected_seed": 43,
        "member_count": 50,
        "raw_fuxi_2025_member_content_sha256": digest.hexdigest(),
        "storage_contract": "corrected members are exactly reconstructible from raw 50-member FuXi archive plus delta_log_location and log_spread",
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
        "artifact_sha256": {"adjustments_2025.npz": sha256_file(shard)},
    }
    write_json(output / "manifest.json", manifest)
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
