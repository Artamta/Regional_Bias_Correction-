#!/usr/bin/env python3
"""Validation-only width ablation for the all-season FuXi adapter.

The four capacity arms change only the member-encoder and spatial-backbone
widths of the frozen ``location_spread`` calibrator.  A separate, approximately
parameter-matched ``summary_only`` control tests whether a learned member-set
representation helps; it never competes in capacity selection.  Every arm uses
the same FuXi member cache, IMD target, 2002--2017 training split, 2018--2019
validation split, finite-ensemble CRPS loss, optimizer, scheduler, member
subsampling, and optimization seeds as the canonical adapter.  No 2020--2021
development score is computed and the sealed 2025 control is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import fuxi_allseason_ensemble_calibration as base
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_capacity_ablation_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv2/fuxi_allseason_capacity_ablation"
PLAN_PATH = PROJECT_ROOT / "plan/CAPACITY_PBC_STUDY_20260822.md"
SLURM_PATH = PROJECT_ROOT / "slurm/run_allseason_capacity_ablation.sbatch"
EXPECTED_CACHE_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
)

BASE_CANDIDATE = "base_42k"
MIN_PROMOTION_SKILL_PCT = 0.5
YEARWISE_CRPS_RATIO_MAX = 1.0
PARSIMONY_TIE_RELATIVE_TOLERANCE = 0.0025
MIN_MATCHED_SEED_IMPROVEMENTS = 2


class CapacityAblationError(RuntimeError):
    """Raised when the frozen capacity-ablation contract is violated."""


@dataclass(frozen=True)
class CapacityCandidate:
    """One width-only arm of the controlled experiment."""

    name: str
    member_hidden_channels: int
    backbone_channels: int
    expected_parameter_count: int
    label: str
    mode: str = "location_spread"
    role: str = "width_candidate"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("capacity candidate name cannot be empty")
        if (
            min(
                self.member_hidden_channels,
                self.backbone_channels,
                self.expected_parameter_count,
            )
            < 1
        ):
            raise ValueError("capacity widths and parameter count must be positive")
        if self.mode not in {"location_spread", "summary_only"}:
            raise ValueError("capacity arm has an unsupported model mode")


CANDIDATES = (
    CapacityCandidate("small_20k", 4, 16, 19_618, "Small · 19.6k"),
    CapacityCandidate("base_42k", 8, 24, 42_434, "Current · 42.4k"),
    CapacityCandidate("medium_158k", 16, 48, 157_570, "Medium · 157.6k"),
    CapacityCandidate("large_294k", 32, 64, 293_762, "Large · 293.8k"),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
CANDIDATE_NAMES = tuple(CANDIDATE_BY_NAME)
SUMMARY_CONTROL = CapacityCandidate(
    "summary_matched_43k",
    8,
    26,
    43_058,
    "Summary-only control · 43.1k",
    mode="summary_only",
    role="parameter_matched_summary_control",
)
EXPERIMENT_ARMS = (*CANDIDATES, SUMMARY_CONTROL)
ARM_BY_NAME = {candidate.name: candidate for candidate in EXPERIMENT_ARMS}
ARM_NAMES = tuple(ARM_BY_NAME)
CANDIDATE_COLORS = {
    "small_20k": "#56B4E9",
    "base_42k": "#009E73",
    "medium_158k": "#E69F00",
    "large_294k": "#D55E00",
    "summary_matched_43k": "#CC79A7",
}


@dataclass(frozen=True)
class CapacityTrainingRun:
    candidate: str
    seed: int
    member_hidden_channels: int
    backbone_channels: int
    parameter_count: int
    best_epoch: int
    stopped_epoch: int
    stopping_reason: str
    best_validation_crps: float
    elapsed_seconds: float
    checkpoint: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file inside an owned staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_model(candidate: CapacityCandidate) -> EnsembleLocationSpreadCalibrator:
    """Construct the exact adapter while changing widths only."""

    model = EnsembleLocationSpreadCalibrator(
        context_channels=7,
        member_hidden_channels=candidate.member_hidden_channels,
        backbone_channels=candidate.backbone_channels,
        mode=candidate.mode,
        dropout=0.05,
        max_abs_log_spread=2.0,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != candidate.expected_parameter_count:
        raise CapacityAblationError(
            f"{candidate.name} has {parameter_count:,} parameters; "
            f"frozen contract expects {candidate.expected_parameter_count:,}"
        )
    return model


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def train_one_candidate(
    candidate: CapacityCandidate,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: base.ContextBundle,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    weights: np.ndarray,
    run_directory: Path,
    *,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    member_subsample: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[pd.DataFrame, CapacityTrainingRun]:
    """Train one width arm using the canonical CRPS optimization recipe."""

    base.set_deterministic_seed(seed)
    checkpoint_path = run_directory / "checkpoints" / "best.pt"
    model = build_model(candidate).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loader = base.make_loader(
        base.EnsembleCaseDataset(members, truth, context, train_indices),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    validation_loader = base.make_loader(
        base.EnsembleCaseDataset(members, truth, context, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    spatial_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
        min_lr=1.0e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    history: list[dict[str, Any]] = []
    start_time = time.monotonic()

    initial_validation = base.validation_crps(
        model, validation_loader, spatial_weights, device, use_amp=use_amp
    )
    history.append(
        {
            "candidate": candidate.name,
            "seed": seed,
            "member_hidden_channels": candidate.member_hidden_channels,
            "backbone_channels": candidate.backbone_channels,
            "parameter_count": parameter_count,
            "epoch": 0,
            "train_crps": np.nan,
            "validation_crps": initial_validation,
            "learning_rate": learning_rate,
            "is_best": True,
        }
    )
    best_validation = initial_validation
    best_epoch = 0
    stale_epochs = 0
    stopping_reason = "max_epochs"

    def save_checkpoint(epoch: int, validation_crps: float) -> None:
        _save_checkpoint_atomic(
            checkpoint_path,
            {
                "experiment": EXPERIMENT,
                "candidate": asdict(candidate),
                "seed": seed,
                "epoch": epoch,
                "validation_crps": validation_crps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
        )

    save_checkpoint(0, initial_validation)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_members, batch_context, batch_truth in train_loader:
            batch_members = batch_members.to(device, non_blocking=True)
            batch_context = batch_context.to(device, non_blocking=True)
            batch_truth = batch_truth.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp and device.type == "cuda",
            ):
                corrected, _, _ = base._call_model(
                    model,
                    batch_members,
                    batch_context,
                    member_subsample=member_subsample,
                )
                loss = base._crps_loss(corrected, batch_truth, spatial_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite CRPS for {candidate.name}, seed {seed}, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch_members.shape[0])
            train_sum += float(loss.detach().cpu()) * count
            train_count += count
        if train_count == 0:
            raise CapacityAblationError("training loader is empty")
        train_crps = train_sum / train_count
        current_validation = base.validation_crps(
            model, validation_loader, spatial_weights, device, use_amp=use_amp
        )
        scheduler.step(current_validation)
        improved = current_validation < best_validation - 1.0e-6
        if improved:
            best_validation = current_validation
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(epoch, current_validation)
        else:
            stale_epochs += 1
        history.append(
            {
                "candidate": candidate.name,
                "seed": seed,
                "member_hidden_channels": candidate.member_hidden_channels,
                "backbone_channels": candidate.backbone_channels,
                "parameter_count": parameter_count,
                "epoch": epoch,
                "train_crps": train_crps,
                "validation_crps": current_validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"[{candidate.name} seed={seed}] epoch={epoch:03d} "
            f"train_CRPS={train_crps:.6f} val_CRPS={current_validation:.6f} "
            f"best={best_validation:.6f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= patience:
            stopping_reason = "early_stopping_patience"
            break

    record = CapacityTrainingRun(
        candidate=candidate.name,
        seed=seed,
        member_hidden_channels=candidate.member_hidden_channels,
        backbone_channels=candidate.backbone_channels,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        stopped_epoch=int(history[-1]["epoch"]),
        stopping_reason=stopping_reason,
        best_validation_crps=float(best_validation),
        elapsed_seconds=float(time.monotonic() - start_time),
        checkpoint=str(checkpoint_path),
    )
    return pd.DataFrame(history), record


def load_checkpoint_model(
    checkpoint_path: Path,
    candidate: CapacityCandidate,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    """Restore a locally produced checkpoint after checking its identity."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    identity = checkpoint.get("candidate")
    if (
        checkpoint.get("experiment") != EXPERIMENT
        or identity != asdict(candidate)
        or int(checkpoint.get("seed", -1)) != seed
    ):
        raise CapacityAblationError(
            f"checkpoint identity mismatch for {candidate.name}, seed {seed}"
        )
    model = build_model(candidate).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validation_case_crps(
    candidate: CapacityCandidate,
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    validation_indices: np.ndarray,
    delta_log_location: np.ndarray,
    log_spread: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """Score one restored checkpoint on validation without materializing it all."""

    selected = np.asarray(validation_indices, dtype=np.int64)
    starts = np.asarray(initializations, dtype="datetime64[D]")[selected]
    years = pd.DatetimeIndex(starts).year.to_numpy()
    if not set(years).issubset(set(base.VALIDATION_YEARS)):
        raise CapacityAblationError("validation scorer received a non-validation year")
    expected = (len(selected), 6, 27, 27)
    if delta_log_location.shape != expected or log_spread.shape != expected:
        raise ValueError(
            f"adjustment fields must both have shape {expected}; got "
            f"{delta_log_location.shape} and {log_spread.shape}"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), chunk_size):
        stop = min(start + chunk_size, len(selected))
        source_indices = selected[start:stop]
        raw = np.asarray(members[source_indices], dtype=np.float32)
        target = np.asarray(truth[source_indices], dtype=np.float32)
        spread_factor = np.exp(np.clip(log_spread[start:stop], -2.0, 2.0)).astype(
            np.float32
        )
        corrected = base.apply_affine_log_calibration(
            raw, delta_log_location[start:stop], spread_factor
        )
        crps = base._weighted_field_mean(
            base.numpy_ensemble_crps(corrected, target), weights
        )
        if crps.shape != (stop - start, 6):
            raise CapacityAblationError(
                f"unexpected validation CRPS shape {crps.shape}"
            )
        for local in range(stop - start):
            position = start + local
            for lead in range(6):
                rows.append(
                    {
                        "split": "validation",
                        "candidate": candidate.name,
                        "seed": seed,
                        "init": np.datetime_as_string(starts[position], unit="D"),
                        "year": int(years[position]),
                        "lead_week": lead + 1,
                        "crps": float(crps[local, lead]),
                    }
                )
        del raw, target, spread_factor, corrected, crps
    result = pd.DataFrame(rows)
    if len(result) != len(selected) * 6:
        raise CapacityAblationError(
            "validation scoring did not retain every case and lead"
        )
    return result


def select_capacity(
    validation_case_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = base.SEEDS,
    expected_years: Sequence[int] = base.VALIDATION_YEARS,
) -> dict[str, Any]:
    """Apply the predeclared validation-only promotion and parsimony rule."""

    required = {"split", "candidate", "seed", "init", "year", "lead_week", "crps"}
    missing = sorted(required - set(validation_case_metrics.columns))
    if missing:
        raise ValueError(f"validation case metrics lack columns: {missing}")
    if set(validation_case_metrics.split) != {"validation"}:
        raise ValueError("capacity selection may use validation rows only")
    if set(validation_case_metrics.candidate) != set(ARM_NAMES):
        raise ValueError(
            "validation metrics must contain every frozen capacity arm and summary control"
        )
    if not np.isfinite(validation_case_metrics.crps.to_numpy(dtype=np.float64)).all():
        raise ValueError("validation CRPS values must be finite")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    expected_year_set = {int(year) for year in expected_years}
    if not expected_seed_set or not expected_year_set:
        raise ValueError("selection requires at least one seed and validation year")
    keys = ["seed", "init", "year", "lead_week"]
    reference_keys: pd.DataFrame | None = None
    for candidate_name, rows in validation_case_metrics.groupby("candidate"):
        observed_seeds = {int(seed) for seed in rows.seed}
        observed_years = {int(year) for year in rows.year}
        if observed_seeds != expected_seed_set:
            raise ValueError(
                f"candidate {candidate_name!r} has seeds {sorted(observed_seeds)}; "
                f"expected {sorted(expected_seed_set)}"
            )
        if observed_years != expected_year_set:
            raise ValueError(
                f"candidate {candidate_name!r} has years {sorted(observed_years)}; "
                f"expected {sorted(expected_year_set)}"
            )
        candidate_keys = rows[keys].sort_values(keys).reset_index(drop=True)
        if candidate_keys.duplicated().any():
            raise ValueError(
                f"candidate {candidate_name!r} has duplicate case/lead rows"
            )
        if reference_keys is None:
            reference_keys = candidate_keys
        elif not candidate_keys.equals(reference_keys):
            raise ValueError(
                "capacity candidates do not share identical validation cases"
            )

    pooled = validation_case_metrics.groupby("candidate").crps.mean()
    by_year = validation_case_metrics.groupby(["candidate", "year"]).crps.mean()
    by_seed = validation_case_metrics.groupby(["candidate", "seed"]).crps.mean()
    base_pooled = float(pooled[BASE_CANDIDATE])
    seed_guard_required = min(MIN_MATCHED_SEED_IMPROVEMENTS, len(expected_seed_set))
    all_records: list[dict[str, Any]] = []
    for candidate in EXPERIMENT_ARMS:
        candidate_pooled = float(pooled[candidate.name])
        pooled_skill = 100.0 * (base_pooled - candidate_pooled) / base_pooled
        year_rows = []
        year_guard = True
        for year in sorted(expected_year_set):
            candidate_crps = float(by_year.loc[(candidate.name, year)])
            base_crps = float(by_year.loc[(BASE_CANDIDATE, year)])
            ratio = candidate_crps / base_crps
            passes = bool(ratio <= YEARWISE_CRPS_RATIO_MAX + 1.0e-12)
            year_guard = year_guard and passes
            year_rows.append(
                {
                    "year": year,
                    "mean_validation_crps": candidate_crps,
                    "base_validation_crps": base_crps,
                    "crps_ratio_vs_base": ratio,
                    "skill_pct_vs_base": 100.0
                    * (base_crps - candidate_crps)
                    / base_crps,
                    "noninferiority_guard": passes,
                }
            )
        seed_rows = []
        seed_guard_passes = 0
        for seed in sorted(expected_seed_set):
            candidate_seed_crps = float(by_seed.loc[(candidate.name, int(seed))])
            base_seed_crps = float(by_seed.loc[(BASE_CANDIDATE, int(seed))])
            improves_base = bool(candidate_seed_crps < base_seed_crps - 1.0e-12)
            seed_guard_passes += int(improves_base)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "mean_validation_crps": candidate_seed_crps,
                    "base_validation_crps": base_seed_crps,
                    "skill_pct_vs_base": 100.0
                    * (base_seed_crps - candidate_seed_crps)
                    / base_seed_crps,
                    "improves_base": improves_base,
                }
            )
        seed_guard = bool(seed_guard_passes >= seed_guard_required)
        promotion = bool(
            candidate.role == "width_candidate"
            and candidate.name != BASE_CANDIDATE
            and pooled_skill >= MIN_PROMOTION_SKILL_PCT
            and year_guard
            and seed_guard
        )
        all_records.append(
            {
                "candidate": candidate.name,
                "role": candidate.role,
                "mode": candidate.mode,
                "member_encoder_used": candidate.mode != "summary_only",
                "member_hidden_channels": candidate.member_hidden_channels,
                "backbone_channels": candidate.backbone_channels,
                "parameter_count": candidate.expected_parameter_count,
                "mean_validation_crps": candidate_pooled,
                "crps_skill_pct_vs_base": pooled_skill,
                "pooled_minimum_improvement_guard": bool(
                    pooled_skill >= MIN_PROMOTION_SKILL_PCT
                ),
                "all_years_noninferior_guard": year_guard,
                "matched_seed_improvement_passes": seed_guard_passes,
                "matched_seed_improvement_required": seed_guard_required,
                "matched_seed_guard": seed_guard,
                "eligible_promotion": promotion,
                "validation_by_year": year_rows,
                "validation_by_seed": seed_rows,
            }
        )

    records = [record for record in all_records if record["role"] == "width_candidate"]
    summary_control = next(
        record
        for record in all_records
        if record["role"] == "parameter_matched_summary_control"
    )

    eligible = [record for record in records if record["eligible_promotion"]]
    if not eligible:
        winner = next(
            record for record in records if record["candidate"] == BASE_CANDIDATE
        )
        reason = (
            "no width variant improved pooled validation CRPS by at least 0.5% "
            "while remaining non-inferior in both validation years and improving "
            f"at least {seed_guard_required} of {len(expected_seed_set)} matched seeds; "
            "base retained"
        )
    else:
        best_crps = min(float(record["mean_validation_crps"]) for record in eligible)
        tied = [
            record
            for record in eligible
            if float(record["mean_validation_crps"])
            <= best_crps * (1.0 + PARSIMONY_TIE_RELATIVE_TOLERANCE)
        ]
        winner = min(tied, key=lambda record: int(record["parameter_count"]))
        reason = (
            "best guard-eligible pooled validation CRPS; the smaller network wins "
            "within a 0.25% practical tie"
        )
    return {
        "status": "validation_selection_locked",
        "selected_candidate": str(winner["candidate"]),
        "selected_parameter_count": int(winner["parameter_count"]),
        "reason": reason,
        "test_metrics_consulted": False,
        "rules": {
            "selection_metric": "mean area-weighted finite-ensemble CRPS over validation cases, leads, and optimization seeds",
            "base_candidate": BASE_CANDIDATE,
            "minimum_pooled_crps_skill_pct_vs_base": MIN_PROMOTION_SKILL_PCT,
            "yearwise_crps_ratio_max_vs_base": YEARWISE_CRPS_RATIO_MAX,
            "minimum_matched_seed_improvements": seed_guard_required,
            "required_validation_years": sorted(expected_year_set),
            "parsimony_tie_relative_tolerance": PARSIMONY_TIE_RELATIVE_TOLERANCE,
            "parameters_are_never_averaged_across_seeds": True,
        },
        "validation_candidates": records,
        "parameter_matched_summary_control": summary_control,
    }


def validation_summary_frames(
    validation_case_metrics: pd.DataFrame,
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_records = [
        *selection["validation_candidates"],
        selection["parameter_matched_summary_control"],
    ]
    records = pd.DataFrame(all_records)
    summary = records.drop(columns=["validation_by_year", "validation_by_seed"])
    year_rows = [
        {"candidate": record["candidate"], **year}
        for record in all_records
        for year in record["validation_by_year"]
    ]
    seed_summary = (
        validation_case_metrics.groupby(["candidate", "seed"], as_index=False)
        .crps.mean()
        .rename(columns={"crps": "mean_validation_crps"})
        .sort_values(["candidate", "seed"])
    )
    return summary, pd.DataFrame(year_rows), seed_summary


def source_snapshot(output: Path) -> dict[str, str]:
    """Freeze all executable and launch-contract sources before training."""

    sources = {
        "src/fuxi_allseason_capacity_ablation.py": Path(__file__).resolve(),
        "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "slurm/run_allseason_capacity_ablation.sbatch": SLURM_PATH,
        "plan/CAPACITY_PBC_STUDY_20260822.md": PLAN_PATH,
    }
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"frozen capacity source is missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = sha256_file(destination)
    return checksums


def output_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative not in {"manifest.json", "failure.json"}:
            checksums[relative] = sha256_file(path)
    return checksums


def plot_training_history(history: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(8.8, 5.0), constrained_layout=True)
    for axis, candidate in zip(axes.ravel(), EXPERIMENT_ARMS):
        selected = history.loc[history.candidate == candidate.name]
        for seed, rows in selected.groupby("seed"):
            rows = rows.sort_values("epoch")
            trained = rows.loc[rows.epoch > 0]
            axis.plot(
                trained.epoch,
                trained.train_crps,
                color="0.65",
                linewidth=0.9,
                label=f"train · seed {seed}",
            )
            axis.plot(
                rows.epoch,
                rows.validation_crps,
                color=CANDIDATE_COLORS[candidate.name],
                linewidth=1.2,
                label=f"validation · seed {seed}",
            )
        axis.set_title(
            f"{candidate.label} ({candidate.expected_parameter_count:,} params)"
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Area-weighted ensemble CRPS")
        axis.grid(alpha=0.2)
        if not selected.empty:
            axis.legend(frameon=False, fontsize=6, ncol=2)
    axes.ravel()[-1].axis("off")
    figure.suptitle(
        "Capacity ablation training histories"
        + (" · plumbing smoke only" if smoke else " · 2018–2019 validation"),
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_capacity_curve(
    summary: pd.DataFrame,
    selected_candidate: str,
    output: Path,
    *,
    smoke: bool,
) -> None:
    ordered = summary.sort_values("parameter_count")
    figure, axis = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    axis.plot(
        ordered.parameter_count,
        ordered.mean_validation_crps,
        color="0.35",
        marker="o",
        linewidth=1.2,
    )
    for row in ordered.itertuples(index=False):
        axis.scatter(
            row.parameter_count,
            row.mean_validation_crps,
            s=55 if row.candidate == selected_candidate else 32,
            color=CANDIDATE_COLORS[row.candidate],
            edgecolor="black" if row.candidate == selected_candidate else "none",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            row.candidate,
            (row.parameter_count, row.mean_validation_crps),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axvline(
        42_434, color=CANDIDATE_COLORS[BASE_CANDIDATE], linestyle="--", alpha=0.5
    )
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Mean validation CRPS (lower is better)")
    axis.set_title(
        "Validation-only width ablation" + (" · smoke, non-scientific" if smoke else "")
    )
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def build_readme(selection: Mapping[str, Any], *, smoke: bool) -> str:
    selected = selection["selected_candidate"]
    status = "NON-SCIENTIFIC PLUMBING SMOKE" if smoke else "VALIDATION-ONLY SCREEN"
    return "\n".join(
        [
            "# FuXi all-season adapter capacity ablation",
            "",
            f"**Status:** {status}",
            "",
            f"Validation rule selected `{selected}`. {selection['reason']}",
            "",
            "Only the member-encoder and backbone widths change across capacity arms. A "
            "separate 43,058-parameter summary-only control is reported but cannot win the "
            "capacity selection. Data, temporal splits, CRPS objective, optimizer, scheduler, "
            "seeds, and member sampling are fixed.",
            "",
            "Selection uses 2018–2019 only. No 2020–2021 development metric is computed, "
            "and this workflow contains no route to the sealed 2025 target.",
            "",
            "## Main artifacts",
            "",
            "- `selection.json`: predeclared promotion guards and locked choice.",
            "- `metrics/validation_case_metrics.csv`: per-seed case/lead CRPS evidence.",
            "- `metrics/validation_summary.csv`: pooled capacity comparison.",
            "- `metrics/validation_by_year.csv`: 2018 and 2019 guard evidence.",
            "- `history/training_history.csv`: actual train/validation CRPS curves.",
            "- `models/*/seed_*/checkpoints/best.pt`: restored best validation checkpoints.",
            "- `manifest.json`: frozen contract, provenance, software, and artifact hashes.",
            "",
        ]
    )


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    """Run the controlled four-arm screen and publish validation evidence only."""

    started_at = time.monotonic()
    snapshot_checksums = source_snapshot(output)
    candidate_names = base._parse_names(
        args.candidates, CANDIDATE_NAMES, "capacity candidates"
    )
    seeds = base._parse_seeds(args.seeds)
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"canonical {EXPERIMENT} must run on CUDA, got {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading verified member cache {args.cache}...", flush=True)
    cache = base.load_member_cache(Path(args.cache), allow_partial=args.smoke)
    provenance = base.cache_provenance(cache)
    if provenance.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT:
        raise CapacityAblationError("member cache source fingerprint is not canonical")
    if not args.smoke and provenance.get("data_sha256") != EXPECTED_CACHE_SHA256:
        raise CapacityAblationError("full member cache SHA-256 is not canonical")

    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(indices) for name, indices in splits.as_dict().items()}
    if args.smoke and cache.members.shape[0] == 64:
        expected_smoke = {"train": 32, "validation": 16, "test": 16, "embargo": 0}
        if split_counts != expected_smoke:
            raise CapacityAblationError(
                f"stratified smoke split {split_counts}, expected {expected_smoke}"
            )
    train_indices = splits.train
    validation_indices = splits.validation
    if args.smoke:
        train_indices = base.select_evenly(train_indices, 32)
        validation_indices = base.select_evenly(validation_indices, 16)
    if min(len(train_indices), len(validation_indices)) == 0:
        raise CapacityAblationError("training and validation splits must be nonempty")
    validation_years = set(
        pd.DatetimeIndex(cache.initializations[validation_indices]).year.to_numpy()
    )
    if validation_years != set(base.VALIDATION_YEARS):
        raise CapacityAblationError(
            f"selected validation years are {sorted(validation_years)}, "
            f"expected {list(base.VALIDATION_YEARS)}"
        )
    print(
        f"Effective cases: train={len(train_indices)}, "
        f"validation={len(validation_indices)}; development/test data will not be scored",
        flush=True,
    )
    observations = base.load_imd_observations(cache)
    context = base.build_context_bundle(cache, observations, train_indices)

    evaluation_directory = output / "evaluation"
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    normalized_weights = observations.weights / observations.weights.sum(
        dtype=np.float64
    )
    scoring_support_path = evaluation_directory / "scoring_support.npz"
    np.savez_compressed(
        scoring_support_path,
        latitude=cache.latitude.astype(np.float64),
        longitude=cache.longitude.astype(np.float64),
        observation_fraction=observations.observation_fraction.astype(np.float32),
        support_mask=observations.weights > 0.0,
        scoring_weight_km2_fraction=observations.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights.astype(np.float64),
    )

    histories: list[pd.DataFrame] = []
    training_runs: list[CapacityTrainingRun] = []
    validation_frames: list[pd.DataFrame] = []
    models_directory = output / "models"
    arm_names = (*candidate_names, SUMMARY_CONTROL.name)
    for candidate_name in arm_names:
        candidate = ARM_BY_NAME[candidate_name]
        for seed in seeds:
            print(f"Training {candidate.name}, seed {seed}...", flush=True)
            run_directory = models_directory / candidate.name / f"seed_{seed}"
            history, record = train_one_candidate(
                candidate,
                seed,
                cache.members,
                observations.weekly_truth,
                context,
                train_indices,
                validation_indices,
                observations.weights,
                run_directory,
                device=device,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                member_subsample=args.member_subsample,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            histories.append(history)
            training_runs.append(record)
            model = load_checkpoint_model(
                Path(record.checkpoint), candidate, seed, device
            )
            delta, log_spread = base.predict_adjustments(
                model,
                cache.members,
                observations.weekly_truth,
                context,
                validation_indices,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            validation_frames.append(
                validation_case_crps(
                    candidate,
                    seed,
                    cache.members,
                    observations.weekly_truth,
                    cache.initializations,
                    validation_indices,
                    delta,
                    log_spread,
                    observations.weights,
                    chunk_size=args.evaluation_batch_size,
                )
            )
            del model, delta, log_spread
            if device.type == "cuda":
                torch.cuda.empty_cache()

    history_frame = pd.concat(histories, ignore_index=True)
    validation_case_metrics = pd.concat(validation_frames, ignore_index=True)
    selection = select_capacity(validation_case_metrics, expected_seeds=seeds)
    selection["written_utc"] = utc_now()
    selection["scientific_selection"] = not args.smoke
    summary, by_year, by_seed = validation_summary_frames(
        validation_case_metrics, selection
    )
    history_directory = output / "history"
    metrics_directory = output / "metrics"
    figures_directory = output / "figures"
    history_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(history_directory / "training_history.csv", index=False)
    validation_case_metrics.to_csv(
        metrics_directory / "validation_case_metrics.csv", index=False
    )
    summary.to_csv(metrics_directory / "validation_summary.csv", index=False)
    by_year.to_csv(metrics_directory / "validation_by_year.csv", index=False)
    by_seed.to_csv(metrics_directory / "validation_by_seed.csv", index=False)
    write_json(output / "selection.json", selection)
    plot_training_history(
        history_frame, figures_directory / "training_loss_curves", smoke=args.smoke
    )
    plot_capacity_curve(
        summary,
        str(selection["selected_candidate"]),
        figures_directory / "validation_capacity_curve",
        smoke=args.smoke,
    )
    (output / "README.md").write_text(
        build_readme(selection, smoke=args.smoke), encoding="utf-8"
    )

    run_records: list[dict[str, Any]] = []
    for record in training_runs:
        values = asdict(record)
        checkpoint_path = Path(record.checkpoint)
        values["checkpoint_sha256"] = sha256_file(checkpoint_path)
        values["checkpoint"] = str(checkpoint_path.relative_to(output))
        run_records.append(values)
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "validation-only capacity screen; no development or sealed-test metrics"
        ),
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "forecast": "FuXi native reforecast weekly TP; all seasons; 51 members",
            "region": "39N-0N, 60E-99E, 27x27 India box",
            "target": "IMD weekly mean precipitation, mm day-1",
            "train_years": list(base.TRAIN_YEARS),
            "validation_years": list(base.VALIDATION_YEARS),
            "selection_data": "2018-2019 validation only",
            "test_metrics_consulted": False,
            "development_years_not_scored": list(base.TEST_YEARS),
            "sealed_unopened_years": list(base.SEALED_YEARS),
            "sealed_2025_target_opened": False,
            "changed_factor": "member_hidden_channels and backbone_channels only",
            "fixed_mode": "location_spread",
            "primary_loss": "area-weighted empirical finite-ensemble CRPS",
            "statistical_unit": "initialization with all members and six leads grouped",
        },
        "candidates": [asdict(CANDIDATE_BY_NAME[name]) for name in candidate_names],
        "controls": [asdict(SUMMARY_CONTROL)],
        "seeds": list(seeds),
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train_indices),
            "validation": len(validation_indices),
        },
        "selection": selection,
        "training": {
            "batch_size": args.batch_size,
            "member_subsample": args.member_subsample,
            "full_members_for_validation": 51,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "automatic_mixed_precision": not args.no_amp,
            "device": str(device),
            "objective": "area-weighted empirical finite-ensemble CRPS",
            "optimizer": "torch.optim.AdamW",
            "scheduler": "ReduceLROnPlateau(factor=0.5,min_lr=1e-6)",
            "early_stopping_metric": "full-51-member validation CRPS",
            "gradient_clip_max_norm": 5.0,
            "runs": run_records,
        },
        "evaluation": {
            "scope": "validation only",
            "case_metric": "area-weighted finite-ensemble CRPS",
            "scoring_support_artifact": "evaluation/scoring_support.npz",
            "scoring_support_sha256": sha256_file(scoring_support_path),
            "support_cells": int(np.count_nonzero(observations.weights > 0.0)),
        },
        "cache": provenance,
        "observation_stores": list(observations.source_stores),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "source_snapshot_sha256": snapshot_checksums,
    }
    manifest["artifact_sha256"] = output_checksums(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the controlled all-season probabilistic adapter width ablation."
    )
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--seeds", default=None, help="default: 42 for --smoke, otherwise 42,43,44"
    )
    parser.add_argument("--candidates", default=",".join(CANDIDATE_NAMES))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--member-subsample", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds is None:
        args.seeds = "42" if args.smoke else "42,43,44"
    if args.max_epochs is None:
        args.max_epochs = 2 if args.smoke else 100
    if args.patience is None:
        args.patience = 1 if args.smoke else 15
    candidates = base._parse_names(
        args.candidates, CANDIDATE_NAMES, "capacity candidates"
    )
    seeds = base._parse_seeds(args.seeds)
    if candidates != CANDIDATE_NAMES:
        raise ValueError(
            f"canonical capacity candidates must be {CANDIDATE_NAMES} in order"
        )
    expected_seeds = (42,) if args.smoke else base.SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"canonical {'smoke' if args.smoke else 'full'} seeds must be {expected_seeds}"
        )
    positive = (
        "max_epochs",
        "patience",
        "batch_size",
        "member_subsample",
        "evaluation_batch_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.member_subsample > 51:
        raise ValueError("--member-subsample cannot exceed 51")
    fixed = {
        "max_epochs": (args.max_epochs, 2 if args.smoke else 100),
        "patience": (args.patience, 1 if args.smoke else 15),
        "batch_size": (args.batch_size, 8),
        "member_subsample": (args.member_subsample, 16),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical run settings differ: {mismatch}")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --learning-rate 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical run requires --weight-decay 0.0001")
    if args.no_amp:
        raise ValueError("canonical GPU run requires automatic mixed precision")
    if args.device == "cpu":
        raise ValueError("canonical run requires CUDA")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested_output = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {requested_output}"
        )
    staging = (
        requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
        os.replace(staging, requested_output)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "requested_output": str(requested_output),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(
        f"PASS: completed {'smoke' if args.smoke else 'full'} capacity run at "
        f"{requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
