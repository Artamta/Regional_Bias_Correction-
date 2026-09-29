#!/usr/bin/env python3
"""Validation-only heavy-rain loss screen for the FuXi-to-IMD adapter.

This is a fixed, one-factor follow-up to the completed anchor/loss factorial.
It keeps the proven width-24 architecture, current 0.75/0.20/0.05 objective,
blocked years, preprocessing, and verification contract unchanged.  The only
experimental factor is an opt-in Smooth-L1 weight of 2, 3, or 5 for verifying
IMD rainfall of at least 10 mm/day.  Both the established log anchor and the
train-only physical-recentered anchor are screened.  No prediction or metric
is produced for 2020 onward.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT

for path in (SOURCE_ROOT, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_bias_aware_validation_sweep as bias  # noqa: E402


RESULTS_ROOT = HERE / "results" / "fuxi_imd_tail_weight_validation_sweep"
REFERENCE_CONFIGURATION = bias.REFERENCE_CONFIGURATION
THRESHOLD_MM_DAY = 10.0
MULTIPLIERS = (2.0, 3.0, 5.0)


def _tail_candidate(anchor: str, multiplier: float) -> bias.BiasCandidate:
    anchor_label = "Log" if anchor == "log_anchor" else "Physical-recentered"
    prefix = "log" if anchor == "log_anchor" else "recentered"
    value = int(multiplier)
    return bias.BiasCandidate(
        name=f"{prefix}_anchor_tail_weight_{value}",
        label=f"{anchor_label} anchor + heavy-rain weight {value}",
        anchor_kind=anchor,
        loss_kind=f"current_tail_weight_{value}",
        loss_coefficients=bias.CURRENT_LOSS,
        heavy_rain_threshold_mm_day=THRESHOLD_MM_DAY,
        heavy_rain_multiplier=float(multiplier),
    )


CANDIDATES = (
    bias.CANDIDATE_BY_NAME[REFERENCE_CONFIGURATION],
    *(_tail_candidate("log_anchor", value) for value in MULTIPLIERS),
    *(_tail_candidate("physical_recentered", value) for value in MULTIPLIERS),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_candidates(names: str | None) -> tuple[bias.BiasCandidate, ...]:
    if not names:
        return CANDIDATES
    requested = tuple(value.strip() for value in names.split(",") if value.strip())
    unknown = sorted(set(requested) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("configuration names must be unique")
    return tuple(CANDIDATE_BY_NAME[name] for name in requested)


def output_files(output: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def copy_sources(output: Path) -> None:
    sources = (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imd_bias_aware_validation_sweep.py",
        SOURCE_ROOT / "fuxi_imd_compact_validation_sweep.py",
        HERE / "slurm" / "run_imd_tail_weight_validation_sweep.sbatch",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
    )
    for source in sources:
        if source.exists():
            shutil.copy2(source, output / "code" / source.name)


def _save_anchor_artifacts(
    output: Path,
    prepared: bias.PreparedBiasExperiment,
    preparation: dict[str, Any],
) -> None:
    current = preparation["current_correction"]
    recentered = preparation["recentered_fit"]
    np.savez_compressed(
        output / "models" / "anchor_parameters.npz",
        current_lead_month_residual=current.lead_month_residual,
        current_shrinkage=np.float32(current.shrinkage),
        recentered_lead_month_residual=recentered.correction.lead_month_residual,
        recenter_scalar_by_lead_month=recentered.scalar_by_lead_month,
        log_anchor_target_scale=prepared.anchors["log_anchor"].target_scale,
        recentered_anchor_target_scale=prepared.anchors[
            "physical_recentered"
        ].target_scale,
    )
    preparation["recenter_diagnostics"].to_csv(
        output / "metrics" / "anchor_training_recenter_diagnostics.csv",
        index=False,
    )


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
        "purpose": "validation-only fixed heavy-rain Smooth-L1 weight screen",
        "train_years": list(bias.TRAIN_YEARS),
        "validation_years": list(bias.VALIDATION_YEARS),
        "quarantined_years": list(bias.QUARANTINED_YEARS),
        "test_predictions_created": False,
        "screening_stage": "one_seed" if len(seeds) == 1 else "confirmation",
        "fixed_heavy_rain_threshold_mm_day": THRESHOLD_MM_DAY,
        "predeclared_heavy_rain_multipliers": list(MULTIPLIERS),
        "model": asdict(bias.MODEL_SPEC),
        "candidates": [asdict(candidate) for candidate in candidates],
        "seeds": list(seeds),
        "reference_configuration": REFERENCE_CONFIGURATION,
        "smoke": bool(args.smoke),
        "ranking_uses_physical_metrics_only": True,
        "objective_values_comparable_across_configurations": False,
        "promotion_rule": (
            "promote at most one non-reference candidate after the fixed-grid "
            "screen, using the existing physical bias/RMSE/MAE/ACC guards"
        ),
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    try:
        prepared, normalization, preparation = bias.prepare_data()
        (output / "normalization.json").write_text(
            json.dumps(bias.compact._json_safe(normalization), indent=2) + "\n",
            encoding="utf-8",
        )
        _save_anchor_artifacts(output, prepared, dict(preparation))
        manifest.update(preparation["metadata"])
        workers = args.workers
        if workers <= 0:
            workers = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        manifest["workers"] = int(workers)
        (output / "manifest.json").write_text(
            json.dumps(bias.compact._json_safe(manifest), indent=2) + "\n",
            encoding="utf-8",
        )

        bias.run_parallel(
            candidates,
            seeds,
            prepared,
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
        seed_guards.to_csv(
            output / "metrics" / "seed_physical_guards.csv", index=False
        )
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
            output / "metrics" / "paired_physical_deltas_vs_control.csv",
            index=False,
        )

        status, selected = bias.select_configuration(
            ranking, reference_configuration=REFERENCE_CONFIGURATION
        )
        selection = {
            "selection_status": status,
            "selected_configuration": str(selected.configuration),
            "selected_qualifies": bool(selected.qualifies),
            "top_ranked_configuration": str(ranking.iloc[0].configuration),
            "confirmation_required": len(seeds) == 1
            and str(selected.configuration) != REFERENCE_CONFIGURATION,
        }
        (output / "metrics" / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
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
        print(f"PASS: tail-weight validation screen complete: {output}", flush=True)
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
