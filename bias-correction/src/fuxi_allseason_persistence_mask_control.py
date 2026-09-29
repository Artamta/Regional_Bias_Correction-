#!/usr/bin/env python3
"""Hash-bound mask-only addendum for the all-season persistence experiment.

This program trains exactly one nonselectable 45k arm.  Its two recent-rainfall
channels are forced to zero and its two availability masks are identical to the
frozen persistence candidate.  The original three-arm output is never edited:
it is fully audited, reproduced on validation, and then combined with the new
control only at the score/selection layer.

No development initialization is placed in a dataset, reconstructed, scored,
or consulted by this driver.  A separate downstream evaluator is required for
the reused 2020--2021 comparison with PBC.
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
import numpy as np
import pandas as pd
import torch

import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_pbc_baseline as pbc_driver
import fuxi_allseason_persistence_augmented as v1
from fuxi_pbc_core import (
    build_daily_issue_time_lags,
    calendar_fields,
    fit_calendar_quantiles,
    is_valid_cdf_for_thresholds,
    observation_cdf,
)
from fuxi_persistence_context import (
    ARMS as V1_ARMS,
    BASE_ARM,
    PERSISTENCE_LAG_ARM,
    ZERO_LAG_ARM,
    PersistenceContextBundle,
    build_persistence_context_bundle,
)
from fuxi_persistence_mask_control import (
    MASK_ONLY_ADDED_CHANNEL_NAMES,
    MASK_ONLY_ARM,
    MASK_ONLY_CONTEXT_CHANNEL_COUNT,
    MaskOnlyCaseDataset,
)
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_allseason_persistence_mask_control_v1"
RECEIPT_SCHEMA = "fuxi_persistence_mask_control_slurm_gate_receipt_v1"
JOINT_SELECTION_STATUS = "joint_validation_selection_locked"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv2/fuxi_allseason_persistence_mask_control"
ADDENDUM_PLAN_PATH = (
    PROJECT_ROOT / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
)
ADDENDUM_PLAN_SHA256 = (
    "c429ee96bfd9b9fc1ddc8fa88da653eeb4923ea037d516d8d6a42071dd3c3ba0"
)
DEFAULT_PBC_ROOT = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z"
)
DEFAULT_PBC_MANIFEST = DEFAULT_PBC_ROOT / "manifest.json"
DEFAULT_PBC_FIT = DEFAULT_PBC_ROOT / "models/pbc_fit.npz"
EXPECTED_PBC_MANIFEST_SHA256 = (
    "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6"
)
EXPECTED_PBC_FIT_SHA256 = (
    "0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36"
)
EXPECTED_OBSERVATION_BUNDLE_SHA256 = (
    "3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49"
)
EXPECTED_LAG_BUNDLE_SHA256 = (
    "9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba"
)
EXPECTED_PARAMETER_COUNT = 45_026
MIN_RPS_SKILL_PCT = 0.5
MIN_MATCHED_SEED_IMPROVEMENTS = 2
SOURCE_PATHS = {
    "src/fuxi_allseason_persistence_mask_control.py": Path(__file__).resolve(),
    "src/fuxi_persistence_mask_control.py": PROJECT_ROOT
    / "src/fuxi_persistence_mask_control.py",
    "tests/test_allseason_persistence_mask_control.py": PROJECT_ROOT
    / "tests/test_allseason_persistence_mask_control.py",
    "src/fuxi_allseason_persistence_augmented.py": PROJECT_ROOT
    / "src/fuxi_allseason_persistence_augmented.py",
    "src/fuxi_persistence_context.py": PROJECT_ROOT
    / "src/fuxi_persistence_context.py",
    "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
    / "src/fuxi_allseason_ensemble_calibration.py",
    "src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
    / "src/fuxi_ensemble_calibration_core.py",
    "src/fuxi_allseason_pbc_baseline.py": PROJECT_ROOT
    / "src/fuxi_allseason_pbc_baseline.py",
    "src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
    "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
    / "src/fuxi_allseason_member_cache.py",
    "src/project_paths.py": PROJECT_ROOT / "src/project_paths.py",
    "slurm/run_allseason_persistence_mask_control.sbatch": PROJECT_ROOT
    / "slurm/run_allseason_persistence_mask_control.sbatch",
    "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md": ADDENDUM_PLAN_PATH,
    "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": PROJECT_ROOT
    / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
}


class MaskControlError(RuntimeError):
    """Raised when an addendum input or frozen scientific contract fails."""


@dataclass(frozen=True)
class MaskTrainingRun:
    arm: str
    seed: int
    context_channels: int
    parameter_count: int
    initial_state_sha256: str
    training_rng_state_sha256: str
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def is_digest(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaskControlError(message)


def require_finite_json(value: object, location: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{location}[{index}]")


def array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if first.dtype.hasobject:
        return bool(np.array_equal(first, second))
    first_bytes = np.ascontiguousarray(first).view(np.uint8)
    second_bytes = np.ascontiguousarray(second).view(np.uint8)
    return bool(np.array_equal(first_bytes, second_bytes))


def array_bundle_provenance(
    arrays: Mapping[str, np.ndarray], *, bundle_name: str
) -> dict[str, Any]:
    return dict(pbc_driver.array_bundle_provenance(arrays, bundle_name=bundle_name))


def new_mask_model(seed: int) -> torch.nn.Module:
    model = v1.build_seeded_model(PERSISTENCE_LAG_ARM, int(seed))
    count = sum(parameter.numel() for parameter in model.parameters())
    require(count == EXPECTED_PARAMETER_COUNT, "mask-only model parameter count changed")
    return model


@torch.no_grad()
def validation_crps(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    weights: torch.Tensor,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    return v1.validation_crps(model, loader, weights, device, use_amp=use_amp)


def train_mask_only(
    seed: int,
    members: np.ndarray,
    truth: np.ndarray,
    context: PersistenceContextBundle,
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
) -> tuple[pd.DataFrame, MaskTrainingRun]:
    """Train the mask-only arm with the byte-matched v1 stochastic recipe."""

    model = new_mask_model(seed).to(device)
    initial_state = v1.model_state_sha256(model)
    checkpoint_path = run_directory / "checkpoints" / "best.pt"
    train_data = MaskOnlyCaseDataset(members, truth, context, train_indices)
    validation_data = MaskOnlyCaseDataset(
        members, truth, context, validation_indices
    )
    train_loader = base.make_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    validation_loader = base.make_loader(
        validation_data,
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
    started = time.monotonic()
    initial_validation = validation_crps(
        model, validation_loader, spatial_weights, device, use_amp=use_amp
    )
    history: list[dict[str, Any]] = [
        {
            "arm": MASK_ONLY_ARM,
            "seed": int(seed),
            "epoch": 0,
            "train_crps": np.nan,
            "validation_crps": float(initial_validation),
            "learning_rate": learning_rate,
            "is_best": True,
        }
    ]

    base.set_deterministic_seed(int(seed))
    training_rng_receipt = v1.training_rng_state_sha256(device)

    def persist(epoch: int, score: float) -> None:
        save_checkpoint(
            checkpoint_path,
            {
                "experiment": EXPERIMENT,
                "arm": MASK_ONLY_ARM,
                "model_engine_arm": PERSISTENCE_LAG_ARM,
                "seed": int(seed),
                "epoch": int(epoch),
                "validation_crps": float(score),
                "context_channels": MASK_ONLY_CONTEXT_CHANNEL_COUNT,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "initial_state_sha256": initial_state,
                "training_rng_state_sha256": training_rng_receipt,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
        )

    best_loss = float(initial_validation)
    best_epoch = 0
    stale_epochs = 0
    stopping_reason = "max_epochs"
    persist(0, best_loss)
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
                    f"non-finite mask-only CRPS for seed {seed}, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch_members.shape[0])
            train_sum += float(loss.detach().cpu()) * count
            train_count += count
        current = validation_crps(
            model, validation_loader, spatial_weights, device, use_amp=use_amp
        )
        scheduler.step(current)
        improved = current < best_loss - 1.0e-6
        if improved:
            best_loss = float(current)
            best_epoch = epoch
            stale_epochs = 0
            persist(epoch, best_loss)
        else:
            stale_epochs += 1
        history.append(
            {
                "arm": MASK_ONLY_ARM,
                "seed": int(seed),
                "epoch": epoch,
                "train_crps": train_sum / train_count,
                "validation_crps": float(current),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        print(
            f"[{MASK_ONLY_ARM} seed={seed}] epoch={epoch:03d} "
            f"train_CRPS={train_sum / train_count:.6f} "
            f"val_CRPS={current:.6f} best={best_loss:.6f}@{best_epoch}",
            flush=True,
        )
        if stale_epochs >= patience:
            stopping_reason = "early_stopping_patience"
            break
    record = MaskTrainingRun(
        arm=MASK_ONLY_ARM,
        seed=int(seed),
        context_channels=MASK_ONLY_CONTEXT_CHANNEL_COUNT,
        parameter_count=EXPECTED_PARAMETER_COUNT,
        initial_state_sha256=initial_state,
        training_rng_state_sha256=training_rng_receipt,
        best_epoch=best_epoch,
        stopped_epoch=int(history[-1]["epoch"]),
        stopping_reason=stopping_reason,
        best_validation_crps=best_loss,
        elapsed_seconds=float(time.monotonic() - started),
        checkpoint=str(checkpoint_path),
    )
    del model
    return pd.DataFrame(history), record


def load_mask_checkpoint(
    path: Path, seed: int, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    require(checkpoint.get("experiment") == EXPERIMENT, "mask checkpoint experiment")
    require(checkpoint.get("arm") == MASK_ONLY_ARM, "mask checkpoint arm")
    require(
        checkpoint.get("model_engine_arm") == PERSISTENCE_LAG_ARM,
        "mask checkpoint engine arm",
    )
    require(int(checkpoint.get("seed", -1)) == int(seed), "mask checkpoint seed")
    require(
        int(checkpoint.get("context_channels", -1))
        == MASK_ONLY_CONTEXT_CHANNEL_COUNT,
        "mask checkpoint context width",
    )
    require(
        int(checkpoint.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "mask checkpoint parameter count",
    )
    require(
        is_digest(checkpoint.get("training_rng_state_sha256")),
        "mask checkpoint training RNG receipt",
    )
    model = new_mask_model(seed).to(device)
    require(
        checkpoint.get("initial_state_sha256") == v1.model_state_sha256(model),
        "mask checkpoint initial-state receipt",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(
        all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
        "mask checkpoint contains non-finite model tensors",
    )
    model.eval()
    return model


@torch.no_grad()
def predict_mask_adjustments(
    model: torch.nn.Module,
    members: np.ndarray,
    truth: np.ndarray,
    context: PersistenceContextBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    dataset = MaskOnlyCaseDataset(members, truth, context, selected)
    loader = base.make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    expected = (len(selected), 6, 27, 27)
    delta = np.empty(expected, dtype=np.float32)
    log_spread = np.empty(expected, dtype=np.float32)
    cursor = 0
    model.eval()
    for batch_members, batch_context, _ in loader:
        batch_members = batch_members.to(device, non_blocking=True)
        batch_context = batch_context.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            _, batch_delta, batch_scale = base._call_model(
                model, batch_members, batch_context, member_subsample=None
            )
        count = int(batch_members.shape[0])
        delta[cursor : cursor + count] = batch_delta.float().cpu().numpy()
        log_spread[cursor : cursor + count] = np.log(
            batch_scale.float().cpu().numpy()
        ).astype(np.float32)
        cursor += count
    require(cursor == len(selected), "mask adjustment inference dropped cases")
    require(
        np.isfinite(delta).all() and np.isfinite(log_spread).all(),
        "mask adjustment fields are non-finite",
    )
    return delta, log_spread


def _skill_pct(candidate: float, reference: float) -> float:
    require(
        np.isfinite(candidate) and np.isfinite(reference) and reference > 0.0,
        "joint selection scores must be finite with a positive reference",
    )
    return 100.0 * (reference - candidate) / reference


def _validate_mask_metrics(
    parent_metrics: pd.DataFrame,
    mask_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_initializations: Sequence[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "split",
        "arm",
        "seed",
        "init",
        "year",
        "lead_week",
        "crps",
        "quintile_rps",
        "score_contract",
    }
    for label, frame in (("parent", parent_metrics), ("mask", mask_metrics)):
        missing = sorted(required - set(frame.columns))
        require(not missing, f"{label} validation metrics lack columns: {missing}")
    parent = parent_metrics.copy()
    mask = mask_metrics.copy()
    require(set(parent.arm) == set(V1_ARMS), "parent frame must contain exact v1 arms")
    require(set(mask.arm) == {MASK_ONLY_ARM}, "mask frame contains another arm")
    require(set(parent.split) == {"validation"}, "parent frame is not validation-only")
    require(set(mask.split) == {"validation"}, "mask frame is not validation-only")
    require(
        set(parent.score_contract) == set(mask.score_contract) == {v1.SCORING_CONTRACT_VERSION},
        "joint score-contract mismatch",
    )
    expected_seed_set = {int(seed) for seed in expected_seeds}
    require(expected_seed_set, "joint selector requires seeds")
    expected_dates = pd.to_datetime(
        pd.Series(np.asarray(expected_initializations)), errors="raise", utc=True
    ).dt.strftime("%Y-%m-%d")
    expected_date_set = set(expected_dates.tolist())
    require(
        len(expected_date_set) == len(expected_dates) and expected_date_set,
        "expected validation initialization inventory is invalid",
    )
    require(
        set(pd.to_datetime(expected_dates).dt.year) == set(base.VALIDATION_YEARS),
        "expected initialization inventory lacks the frozen validation years",
    )
    expected_keys = {
        (seed, date, lead)
        for seed in expected_seed_set
        for date in expected_date_set
        for lead in range(1, 7)
    }
    for label, frame in (("parent", parent), ("mask", mask)):
        for column in ("seed", "year", "lead_week"):
            values = pd.to_numeric(frame[column], errors="raise")
            require(
                np.isfinite(values.to_numpy(dtype=np.float64)).all()
                and np.equal(
                    values.to_numpy(dtype=np.float64),
                    values.to_numpy(dtype=np.int64),
                ).all(),
                f"{label} {column} is not finite integer data",
            )
            frame[column] = values.astype(np.int64)
        dates = pd.to_datetime(frame.init, errors="raise", utc=True)
        frame["init"] = dates.dt.strftime("%Y-%m-%d")
        require(
            np.array_equal(
                dates.dt.year.to_numpy(dtype=np.int64),
                frame.year.to_numpy(dtype=np.int64),
            ),
            f"{label} initialization/year mismatch",
        )
        scores = frame[["crps", "quintile_rps"]].to_numpy(dtype=np.float64)
        require(
            np.isfinite(scores).all() and np.all(scores >= 0.0),
            f"{label} validation score is invalid",
        )
        for evidence in (
            "cdf_shape_matches_thresholds",
            "forecast_cdf_finite_where_threshold_defined",
            "observed_cdf_finite_where_threshold_defined",
            "forecast_cdf_valid_for_thresholds",
            "observed_cdf_valid_for_thresholds",
        ):
            require(evidence in frame, f"{label} lacks CDF evidence {evidence}")
            values = frame[evidence]
            if values.dtype == np.bool_:
                passes = bool(values.all())
            else:
                passes = bool(
                    values.astype(str).str.strip().str.lower().eq("true").all()
                )
            require(passes, f"{label} CDF evidence failed: {evidence}")
    for arm in V1_ARMS:
        rows = parent.loc[parent.arm == arm]
        keys = set(zip(rows.seed, rows.init, rows.lead_week))
        require(keys == expected_keys and len(rows) == len(expected_keys), f"v1 keys {arm}")
    mask_keys = set(zip(mask.seed, mask.init, mask.lead_week))
    require(
        mask_keys == expected_keys and len(mask) == len(expected_keys),
        "mask-only validation keys differ from v1",
    )
    return parent, mask


def select_joint_arm(
    parent_metrics: pd.DataFrame,
    mask_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = base.SEEDS,
    expected_initializations: Sequence[Any],
) -> dict[str, Any]:
    """Retain every v1 gate and add the frozen mask-only attribution guards."""

    parent, mask = _validate_mask_metrics(
        parent_metrics,
        mask_metrics,
        expected_seeds=expected_seeds,
        expected_initializations=expected_initializations,
    )
    original = v1.select_persistence_arm(
        parent,
        expected_seeds=expected_seeds,
        expected_initializations=expected_initializations,
    )
    candidate = parent.loc[parent.arm == PERSISTENCE_LAG_ARM]
    candidate_pooled = candidate[["crps", "quintile_rps"]].mean()
    mask_pooled = mask[["crps", "quintile_rps"]].mean()
    candidate_late = float(candidate.loc[candidate.lead_week >= 2, "quintile_rps"].mean())
    mask_late = float(mask.loc[mask.lead_week >= 2, "quintile_rps"].mean())
    candidate_year = candidate.groupby("year")["quintile_rps"].mean()
    mask_year = mask.groupby("year")["quintile_rps"].mean()
    candidate_seed = candidate.groupby("seed")["quintile_rps"].mean()
    mask_seed = mask.groupby("seed")["quintile_rps"].mean()

    pooled_skill = _skill_pct(
        float(candidate_pooled.quintile_rps), float(mask_pooled.quintile_rps)
    )
    late_skill = _skill_pct(candidate_late, mask_late)
    year_rows: list[dict[str, Any]] = []
    all_years_noninferior = True
    for year in sorted({int(value) for value in mask.year}):
        candidate_value = float(candidate_year.loc[year])
        mask_value = float(mask_year.loc[year])
        passes = candidate_value <= mask_value + 1.0e-12
        all_years_noninferior = all_years_noninferior and passes
        year_rows.append(
            {
                "year": year,
                "candidate_quintile_rps": candidate_value,
                "reference_quintile_rps": mask_value,
                "rps_skill_pct": _skill_pct(candidate_value, mask_value),
                "noninferior": bool(passes),
            }
        )
    seed_rows: list[dict[str, Any]] = []
    improved_seeds = 0
    for seed in sorted({int(value) for value in mask.seed}):
        candidate_value = float(candidate_seed.loc[seed])
        mask_value = float(mask_seed.loc[seed])
        improves = candidate_value < mask_value - 1.0e-12
        improved_seeds += int(improves)
        seed_rows.append(
            {
                "seed": seed,
                "candidate_quintile_rps": candidate_value,
                "reference_quintile_rps": mask_value,
                "rps_skill_pct": _skill_pct(candidate_value, mask_value),
                "improves": bool(improves),
            }
        )
    required_seed_improvements = min(
        MIN_MATCHED_SEED_IMPROVEMENTS, len({int(seed) for seed in expected_seeds})
    )
    mask_comparison: dict[str, Any] = {
        "reference_arm": MASK_ONLY_ARM,
        "candidate_mean_validation_crps": float(candidate_pooled.crps),
        "reference_mean_validation_crps": float(mask_pooled.crps),
        "candidate_mean_validation_quintile_rps": float(
            candidate_pooled.quintile_rps
        ),
        "reference_mean_validation_quintile_rps": float(mask_pooled.quintile_rps),
        "candidate_mean_w2_w6_quintile_rps": candidate_late,
        "reference_mean_w2_w6_quintile_rps": mask_late,
        "pooled_rps_skill_pct": pooled_skill,
        "w2_w6_rps_skill_pct": late_skill,
        "pooled_rps_minimum_improvement_guard": bool(
            pooled_skill >= MIN_RPS_SKILL_PCT - 1.0e-12
        ),
        "w2_w6_rps_minimum_improvement_guard": bool(
            late_skill >= MIN_RPS_SKILL_PCT - 1.0e-12
        ),
        "pooled_crps_noninferiority_guard": bool(
            float(candidate_pooled.crps) <= float(mask_pooled.crps) + 1.0e-12
        ),
        "all_years_rps_noninferiority_guard": bool(all_years_noninferior),
        "matched_seed_rps_improvement_passes": int(improved_seeds),
        "matched_seed_rps_improvement_required": int(required_seed_improvements),
        "matched_seed_rps_guard": bool(
            improved_seeds >= required_seed_improvements
        ),
        "validation_by_year": year_rows,
        "validation_by_seed": seed_rows,
    }
    mask_comparison["passes"] = bool(
        mask_comparison["pooled_rps_minimum_improvement_guard"]
        and mask_comparison["w2_w6_rps_minimum_improvement_guard"]
        and mask_comparison["pooled_crps_noninferiority_guard"]
        and mask_comparison["all_years_rps_noninferiority_guard"]
        and mask_comparison["matched_seed_rps_guard"]
    )
    original_passes = bool(original["candidate_promoted"])
    promoted = bool(original_passes and mask_comparison["passes"])
    selected = PERSISTENCE_LAG_ARM if promoted else BASE_ARM
    if promoted:
        reason = (
            "persistence candidate passed every frozen v1 and active-mask "
            "validation guard"
        )
    elif not original_passes:
        reason = "persistence candidate failed at least one immutable v1 guard"
    else:
        reason = "persistence candidate failed at least one mask-only attribution guard"
    mask_mean = {
        "mean_validation_crps": float(mask_pooled.crps),
        "mean_validation_quintile_rps": float(mask_pooled.quintile_rps),
        "mean_w2_w6_validation_quintile_rps": mask_late,
        "role": "active_mask_attribution_control_nonselectable",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    return {
        "status": JOINT_SELECTION_STATUS,
        "selected_arm": selected,
        "candidate_arm": PERSISTENCE_LAG_ARM,
        "candidate_promoted": promoted,
        "reason": reason,
        "test_metrics_consulted": False,
        "development_indices_accessed": False,
        "mask_only_selectable": False,
        "original_three_arm_gates": {
            "passes": original_passes,
            "selected_arm": original["selected_arm"],
            "candidate_promoted": original["candidate_promoted"],
            "zero_lag_control_reproduction": original[
                "zero_lag_control_reproduction"
            ],
            "candidate_comparisons": original["candidate_comparisons"],
            "rules": original["rules"],
        },
        "mask_only_comparison": mask_comparison,
        "rules": {
            "all_original_v1_gates_required": True,
            "minimum_pooled_rps_skill_pct_vs_mask_only": MIN_RPS_SKILL_PCT,
            "minimum_w2_w6_rps_skill_pct_vs_mask_only": MIN_RPS_SKILL_PCT,
            "pooled_crps_ratio_max_vs_mask_only": 1.0,
            "yearwise_rps_ratio_max_vs_mask_only": 1.0,
            "minimum_matched_seed_rps_improvements_vs_mask_only": (
                required_seed_improvements
            ),
            "parameters_or_predictions_averaged_across_seeds": False,
            "development_may_override_selection": False,
        },
        "validation_arm_means": {
            **original["validation_arm_means"],
            MASK_ONLY_ARM: mask_mean,
        },
    }


def _stable_original_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in selection.items()
        if key not in {"written_utc", "scientific_selection"}
    }


def compare_validation_frames(
    expected: pd.DataFrame, reproduced: pd.DataFrame
) -> dict[str, Any]:
    """Prove reproduced parent scores agree with the immutable persisted CSV."""

    keys = ["arm", "seed", "init", "year", "lead_week"]
    left = expected.sort_values(keys).reset_index(drop=True)
    right = reproduced.sort_values(keys).reset_index(drop=True)
    require(len(left) == len(right), "reproduced parent metric row count differs")
    require(
        left[keys].astype(str).equals(right[keys].astype(str)),
        "reproduced parent metric keys differ",
    )
    numeric_delta: dict[str, float] = {}
    for column in ("crps", "quintile_rps"):
        first = left[column].to_numpy(dtype=np.float64)
        second = right[column].to_numpy(dtype=np.float64)
        maximum = float(np.max(np.abs(first - second))) if first.size else 0.0
        require(maximum <= 1.0e-12, f"reproduced parent {column} differs")
        numeric_delta[column] = maximum
    for column in (
        "split",
        "score_contract",
        "cdf_shape_matches_thresholds",
        "forecast_cdf_finite_where_threshold_defined",
        "observed_cdf_finite_where_threshold_defined",
        "forecast_cdf_valid_for_thresholds",
        "observed_cdf_valid_for_thresholds",
    ):
        require(column in left and column in right, f"missing parent metric column {column}")
        require(
            left[column].astype(str).str.lower().equals(
                right[column].astype(str).str.lower()
            ),
            f"reproduced parent metric evidence differs for {column}",
        )
    return {
        "persisted_rows": int(len(left)),
        "reproduced_rows": int(len(right)),
        "maximum_absolute_score_difference": numeric_delta,
        "passes": True,
    }


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"JSON object required: {path}")
    require_finite_json(value, str(path))
    return value


def verify_artifact_inventory(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    recorded = manifest.get("artifact_sha256")
    require(isinstance(recorded, dict) and recorded, "parent artifact hash map missing")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and str(path.relative_to(root))
        not in {"manifest.json", "slurm_gate_receipt.json"}
    }
    require(set(recorded) == actual, "parent artifact inventory changed")
    result: dict[str, str] = {}
    for relative, digest in recorded.items():
        require(is_digest(digest), f"invalid parent artifact digest: {relative}")
        path = root / relative
        actual_digest = sha256_file(path)
        require(actual_digest == digest, f"parent artifact changed: {relative}")
        result[str(relative)] = actual_digest
    return result


def mapping_sha256(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(values), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_parent_envelope(
    manifest_path: Path,
    receipt_path: Path,
    *,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    manifest_file = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    require(manifest_file.name == "manifest.json", "parent manifest filename")
    root = manifest_file.parent
    require(receipt_file == root / "slurm_gate_receipt.json", "parent receipt path")
    manifest = load_json(manifest_file)
    receipt = load_json(receipt_file)
    mode = "smoke" if smoke else "full"
    expected_seeds = [42] if smoke else [42, 43, 44]
    expected_counts = (
        {"train": 32, "validation": 16}
        if smoke
        else {"train": 1652, "validation": 196}
    )
    require(manifest.get("experiment") == v1.EXPERIMENT, "parent experiment")
    require(manifest.get("status") == "complete", "parent completion status")
    require(manifest.get("mode") == mode, "parent mode")
    require(manifest.get("smoke") is smoke, "parent smoke flag")
    require(manifest.get("seeds") == expected_seeds, "parent seed inventory")
    require(
        manifest.get("split_counts_selected") == expected_counts,
        "parent selected split counts",
    )
    require(
        Path(str(manifest.get("output_path", ""))).resolve() == root,
        "parent output identity",
    )
    require(
        receipt.get("schema_version") == "fuxi_persistence_slurm_gate_receipt_v1",
        "parent receipt schema",
    )
    require(receipt.get("experiment") == v1.EXPERIMENT, "parent receipt experiment")
    require(receipt.get("status") == "post_run_audit_passed", "parent receipt status")
    require(receipt.get("mode") == mode, "parent receipt mode")
    require(receipt.get("output_path") == str(root), "parent receipt output")
    require(receipt.get("manifest_path") == str(manifest_file), "parent receipt manifest")
    require(str(receipt.get("job_id", "")).isdigit(), "parent Slurm job identity")
    require(bool(receipt.get("node")), "parent Slurm node identity")
    require(receipt.get("partition") == "gpu_prio", "parent Slurm partition")
    require(bool(receipt.get("gpu")), "parent GPU receipt")
    parent_manifest_sha = sha256_file(manifest_file)
    require(
        receipt.get("manifest_sha256") == parent_manifest_sha,
        "parent receipt manifest digest",
    )
    require(
        receipt.get("source_snapshot_sha256")
        == manifest.get("source_snapshot_sha256"),
        "parent receipt source binding",
    )
    require(
        receipt.get("source_binding_sha256")
        == mapping_sha256(manifest.get("source_snapshot_sha256", {})),
        "parent receipt source-map digest",
    )
    require(
        receipt.get("persistence_lag_sha256") == EXPECTED_LAG_BUNDLE_SHA256,
        "parent receipt lag bundle",
    )
    require(
        receipt.get("observation_bundle_sha256")
        == EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "parent receipt observation bundle",
    )
    require(
        manifest.get("persistence_lag_provenance", {}).get("sha256")
        == EXPECTED_LAG_BUNDLE_SHA256,
        "parent manifest lag bundle",
    )
    require(
        manifest.get("observation_bundle_provenance", {}).get("sha256")
        == EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "parent manifest observation bundle",
    )
    parent_selection = manifest.get("selection", {})
    require(
        isinstance(parent_selection, dict)
        and parent_selection.get("status") == "validation_selection_locked",
        "parent selection lock",
    )
    require(
        parent_selection.get("scientific_selection") is (not smoke),
        "parent selection scientific flag",
    )
    require(parent_selection.get("test_metrics_consulted") is False, "parent test firewall")
    require(parent_selection.get("zero_lag_selectable") is False, "parent zero gate")
    require(
        manifest.get("temporal_evidence", {}).get("development_indices_accessed")
        is False,
        "parent development-index firewall",
    )
    contract = manifest.get("contract", {})
    require(contract.get("train_years") == list(base.TRAIN_YEARS), "parent train years")
    require(
        contract.get("validation_years") == list(base.VALIDATION_YEARS),
        "parent validation years",
    )
    require(contract.get("test_metrics_consulted") is False, "parent test contract")
    require(
        contract.get("development_years_not_scored") == list(base.TEST_YEARS),
        "parent development contract",
    )
    require(
        contract.get("sealed_unopened_years") == list(base.SEALED_YEARS),
        "parent sealed-year contract",
    )
    training = manifest.get("training", {})
    expected_epochs = 2 if smoke else 100
    expected_patience = 1 if smoke else 15
    expected_training = {
        "batch_size": 8,
        "member_subsample": 16,
        "full_members_for_validation": 51,
        "max_epochs": expected_epochs,
        "patience": expected_patience,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "automatic_mixed_precision": True,
        "optimizer": "torch.optim.AdamW",
        "scheduler": "ReduceLROnPlateau(factor=0.5,min_lr=1e-6)",
        "gradient_clip_max_norm": 5.0,
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected, f"parent training contract: {key}")
    require(
        training.get("same_seed_45k_initial_states_exact") is True,
        "parent common 45k initialization",
    )
    require(
        training.get("same_seed_stochastic_training_streams_exact_across_all_arms")
        is True,
        "parent common stochastic stream",
    )
    runs = training.get("runs", [])
    require(isinstance(runs, list), "parent run inventory")
    expected_run_keys = {(arm, seed) for arm in V1_ARMS for seed in expected_seeds}
    run_keys: set[tuple[str, int]] = set()
    initial_45k: dict[tuple[str, int], str] = {}
    rng_receipts: dict[tuple[str, int], str] = {}
    for row in runs:
        require(isinstance(row, dict), "parent run record")
        arm = str(row.get("arm"))
        seed = int(row.get("seed", -1))
        require((arm, seed) in expected_run_keys, "parent run identity")
        require((arm, seed) not in run_keys, "duplicate parent run")
        expected_checkpoint = f"models/{arm}/seed_{seed}/checkpoints/best.pt"
        expected_adjustment = f"models/{arm}/seed_{seed}/validation_adjustments.npz"
        require(row.get("checkpoint") == expected_checkpoint, "parent checkpoint path")
        require(
            row.get("validation_adjustment") == expected_adjustment,
            "parent adjustment path",
        )
        for key in (
            "checkpoint_sha256",
            "validation_adjustment_sha256",
            "initial_state_sha256",
            "training_rng_state_sha256",
        ):
            require(is_digest(row.get(key)), f"parent run digest {key}")
        rng_receipts[(arm, seed)] = str(row["training_rng_state_sha256"])
        if arm != BASE_ARM:
            initial_45k[(arm, seed)] = str(row["initial_state_sha256"])
        run_keys.add((arm, seed))
    require(run_keys == expected_run_keys, "parent complete arm/seed grid")
    for seed in expected_seeds:
        require(
            initial_45k[(ZERO_LAG_ARM, seed)]
            == initial_45k[(PERSISTENCE_LAG_ARM, seed)],
            f"parent common 45k initialization seed {seed}",
        )
        require(
            len({rng_receipts[(arm, seed)] for arm in V1_ARMS}) == 1,
            f"parent common RNG stream seed {seed}",
        )
    require(
        training.get("training_rng_receipts_by_seed")
        == {
            str(seed): rng_receipts[(BASE_ARM, seed)]
            for seed in expected_seeds
        },
        "parent RNG receipt map",
    )
    require(
        receipt.get("cuda_device") == manifest.get("software", {}).get("cuda_device"),
        "parent GPU model receipt agreement",
    )
    require(
        receipt.get("cache_sha256") == v1.EXPECTED_CACHE_SHA256,
        "parent cache receipt",
    )
    plan_binding = receipt.get("plan_binding", {})
    require(
        plan_binding.get("expected_sha256")
        == plan_binding.get("snapshot_sha256")
        == plan_binding.get("live_sha256")
        == manifest.get("source_snapshot_sha256", {}).get(
            "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
        ),
        "parent plan binding",
    )
    selection_path = root / "selection.json"
    require(load_json(selection_path) == parent_selection, "parent selection artifact")
    artifact_hashes = verify_artifact_inventory(root, manifest)

    source_hashes = manifest.get("source_snapshot_sha256", {})
    require(isinstance(source_hashes, dict) and source_hashes, "parent source binding")
    live_sources = {
        "code/src/fuxi_allseason_persistence_augmented.py": PROJECT_ROOT
        / "src/fuxi_allseason_persistence_augmented.py",
        "code/src/fuxi_persistence_context.py": PROJECT_ROOT
        / "src/fuxi_persistence_context.py",
        "code/src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "code/src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "code/src/fuxi_allseason_pbc_baseline.py": PROJECT_ROOT
        / "src/fuxi_allseason_pbc_baseline.py",
        "code/src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "code/src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
        "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": PROJECT_ROOT
        / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
        "code/slurm/run_allseason_persistence_augmented.sbatch": PROJECT_ROOT
        / "slurm/run_allseason_persistence_augmented.sbatch",
    }
    require(set(source_hashes) == set(live_sources), "parent source inventory")
    for relative, live in live_sources.items():
        require(live.is_file(), f"parent live source missing: {live}")
        require(
            sha256_file(live) == source_hashes[relative],
            f"parent source changed after run: {live}",
        )
    return manifest, receipt, root, artifact_hashes


def audit_accepted_pbc(
    manifest_path: Path, fit_path: Path
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    fit_file = Path(fit_path).resolve()
    require(
        sha256_file(manifest_file) == EXPECTED_PBC_MANIFEST_SHA256,
        "accepted PBC manifest digest",
    )
    manifest = load_json(manifest_file)
    require(
        manifest.get("experiment") == "fuxi_allseason_pbc_baseline_v2"
        and manifest.get("status") == "complete"
        and manifest.get("mode") == "full",
        "accepted PBC manifest identity",
    )
    require(
        manifest.get("observation_bundle_provenance", {}).get("sha256")
        == EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "accepted PBC observation bundle",
    )
    require(
        manifest.get("persistence_lag_provenance", {}).get("sha256")
        == EXPECTED_LAG_BUNDLE_SHA256,
        "accepted PBC lag bundle",
    )
    require(fit_file == manifest_file.parent / "models/pbc_fit.npz", "PBC fit path")
    require(sha256_file(fit_file) == EXPECTED_PBC_FIT_SHA256, "accepted PBC fit digest")
    require(
        manifest.get("artifact_sha256", {}).get("models/pbc_fit.npz")
        == EXPECTED_PBC_FIT_SHA256,
        "accepted PBC fit artifact receipt",
    )
    return {
        "manifest_path": str(manifest_file),
        "manifest_sha256": EXPECTED_PBC_MANIFEST_SHA256,
        "fit_path": str(fit_file),
        "fit_sha256": EXPECTED_PBC_FIT_SHA256,
        "observation_bundle_sha256": EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "persistence_lag_sha256": EXPECTED_LAG_BUNDLE_SHA256,
    }


def audit_mask_smoke_parent(
    manifest_path: Path,
    receipt_path: Path,
    *,
    current_source_snapshot: Mapping[str, str],
    expected_original_v1_smoke_manifest_sha256: str,
    expected_original_v1_smoke_receipt_sha256: str,
    accepted_pbc: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    root = manifest_file.parent
    require(manifest_file.name == "manifest.json", "mask smoke manifest filename")
    require(receipt_file == root / "slurm_gate_receipt.json", "mask smoke receipt path")
    manifest = load_json(manifest_file)
    receipt = load_json(receipt_file)
    require(
        manifest.get("schema_version")
        == "fuxi_persistence_mask_control_manifest_v1",
        "mask smoke manifest schema",
    )
    require(manifest.get("experiment") == EXPERIMENT, "mask smoke experiment")
    require(manifest.get("status") == "complete", "mask smoke completion")
    require(manifest.get("mode") == "smoke", "mask smoke mode")
    require(manifest.get("smoke") is True, "mask smoke flag")
    require(manifest.get("seeds") == [42], "mask smoke seed")
    require(
        manifest.get("split_counts_selected") == {"train": 32, "validation": 16},
        "mask smoke selected counts",
    )
    require(
        Path(str(manifest.get("output_path", ""))).resolve() == root,
        "mask smoke output identity",
    )
    manifest_sha = sha256_file(manifest_file)
    require(
        receipt.get("schema_version") == RECEIPT_SCHEMA,
        "mask smoke receipt schema",
    )
    require(receipt.get("experiment") == EXPERIMENT, "mask smoke receipt experiment")
    require(receipt.get("status") == "post_run_audit_passed", "mask smoke receipt status")
    require(receipt.get("mode") == "smoke", "mask smoke receipt mode")
    require(receipt.get("output_path") == str(root), "mask smoke receipt output")
    require(receipt.get("manifest_path") == str(manifest_file), "mask smoke receipt manifest")
    require(receipt.get("manifest_sha256") == manifest_sha, "mask smoke manifest digest")
    require(str(receipt.get("job_id", "")).isdigit(), "mask smoke Slurm job")
    require(bool(receipt.get("node")), "mask smoke Slurm node")
    require(receipt.get("partition") == "gpu_prio", "mask smoke partition")
    require(bool(receipt.get("gpu")), "mask smoke GPU receipt")
    require(
        receipt.get("mask_smoke_manifest_path") is None
        and receipt.get("mask_smoke_manifest_sha256") is None,
        "mask smoke receipt must not have a mask-smoke parent",
    )
    source_hashes = manifest.get("source_snapshot_sha256", {})
    require(
        source_hashes == dict(current_source_snapshot),
        "full mask source bundle differs from mask smoke",
    )
    require(
        manifest.get("source_binding_sha256") == mapping_sha256(source_hashes),
        "mask smoke source binding",
    )
    require(
        receipt.get("source_snapshot_sha256") == source_hashes
        and receipt.get("source_binding_sha256") == mapping_sha256(source_hashes),
        "mask smoke receipt source binding",
    )
    plan = manifest.get("addendum_plan", {})
    require(
        plan.get("expected_sha256")
        == plan.get("live_sha256")
        == plan.get("snapshot_sha256")
        == ADDENDUM_PLAN_SHA256,
        "mask smoke addendum plan binding",
    )
    original = manifest.get("original_v1", {})
    require(original.get("mode") == "smoke", "mask smoke original-v1 mode")
    require(
        original.get("manifest_sha256")
        == expected_original_v1_smoke_manifest_sha256,
        "mask smoke original-v1 manifest binding",
    )
    require(
        original.get("receipt_sha256")
        == expected_original_v1_smoke_receipt_sha256,
        "mask smoke original-v1 receipt binding",
    )
    require(original.get("smoke_parent") is None, "mask smoke v1 ancestry")
    require(
        manifest.get("accepted_pbc") == dict(accepted_pbc),
        "mask smoke accepted-PBC binding",
    )
    require(
        manifest.get("cache", {}).get("actual_data_sha256")
        == v1.EXPECTED_CACHE_SHA256,
        "mask smoke cache binding",
    )
    require(
        manifest.get("observation_bundle_provenance", {}).get("sha256")
        == EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "mask smoke observation binding",
    )
    require(
        manifest.get("persistence_lag_provenance", {}).get("sha256")
        == EXPECTED_LAG_BUNDLE_SHA256,
        "mask smoke lag binding",
    )
    selection = manifest.get("joint_selection", {})
    require(selection.get("status") == JOINT_SELECTION_STATUS, "mask smoke selection lock")
    require(selection.get("scientific_selection") is False, "mask smoke scientific flag")
    require(selection.get("mask_only_selectable") is False, "mask smoke selectability")
    require(selection.get("test_metrics_consulted") is False, "mask smoke test firewall")
    require(
        selection.get("development_indices_accessed") is False,
        "mask smoke development firewall",
    )
    require(
        load_json(root / str(manifest.get("joint_selection_artifact"))) == selection,
        "mask smoke selection artifact",
    )
    require(
        manifest.get("joint_selection_sha256")
        == sha256_file(root / str(manifest.get("joint_selection_artifact"))),
        "mask smoke selection digest",
    )
    runs = manifest.get("training", {}).get("runs", [])
    require(len(runs) == 1, "mask smoke run count")
    row = runs[0]
    require(
        row.get("arm") == MASK_ONLY_ARM
        and row.get("seed") == 42
        and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT,
        "mask smoke run identity",
    )
    for path_key, digest_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("validation_adjustment", "validation_adjustment_sha256"),
    ):
        artifact = root / str(row.get(path_key))
        require(
            sha256_file(artifact) == row.get(digest_key),
            f"mask smoke run artifact {path_key}",
        )
    mask_metrics = root / str(
        manifest.get("evaluation", {}).get("mask_validation_case_metrics_artifact")
    )
    require(
        sha256_file(mask_metrics)
        == manifest.get("evaluation", {}).get("mask_validation_case_metrics_sha256"),
        "mask smoke validation metrics",
    )
    require(
        manifest.get("hardware_software_match", {}).get("matches_original") is True,
        "mask smoke hardware/software match",
    )
    artifacts = verify_artifact_inventory(root, manifest)
    require(
        manifest.get("artifact_binding_sha256") == mapping_sha256(artifacts),
        "mask smoke artifact-map binding",
    )
    for relative, digest in source_hashes.items():
        require(
            artifacts.get(str(relative)) == digest
            and sha256_file(root / str(relative)) == digest,
            f"mask smoke source is not artifact-bound: {relative}",
        )
    return {
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha,
        "receipt_path": str(receipt_file),
        "receipt_sha256": sha256_file(receipt_file),
        "output_path": str(root),
        "source_snapshot_sha256": source_hashes,
        "source_binding_sha256": mapping_sha256(source_hashes),
        "artifact_sha256": artifacts,
        "artifact_binding_sha256": mapping_sha256(artifacts),
        "original_v1_manifest_sha256": original["manifest_sha256"],
        "original_v1_receipt_sha256": original["receipt_sha256"],
        "accepted_pbc": manifest["accepted_pbc"],
        "external_receipt": receipt,
    }


def audit_matching_environment(
    parent_manifest: Mapping[str, Any],
    parent_receipt: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    require(device.type == "cuda", "mask-control canonical run requires CUDA")
    require(
        parent_manifest.get("training", {}).get("automatic_mixed_precision") is True,
        "original v1 did not use the required AMP mode",
    )
    current_node = os.environ.get("SLURMD_NODENAME")
    require(bool(current_node), "SLURMD_NODENAME is required for hardware matching")
    require(
        current_node == parent_receipt.get("node"),
        "mask-control run must use the original v1 physical node",
    )
    base.set_deterministic_seed(0)
    require(
        torch.backends.cudnn.deterministic is True
        and torch.backends.cudnn.benchmark is False,
        "deterministic cuDNN recipe is not active",
    )
    parent = parent_manifest.get("software", {})
    current = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "matplotlib": importlib.metadata.version("matplotlib"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device),
    }
    for key in (
        "python",
        "platform",
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "cuda_device",
    ):
        require(current[key] == parent.get(key), f"parent/current software mismatch: {key}")
    require(current["cuda_available"] is True, "current CUDA unavailable")
    require(
        parent_receipt.get("cuda_device") == current["cuda_device"],
        "parent receipt/current GPU model mismatch",
    )
    command = parent_manifest.get("command_line", [])
    require(
        isinstance(command, list)
        and command
        and Path(str(command[0])).resolve() == Path(sys.executable).resolve(),
        "parent/current Python executable mismatch",
    )
    current["python_executable"] = sys.executable
    current["parent_python_executable"] = str(command[0])
    current["torch_cuda_runtime"] = torch.version.cuda
    current["automatic_mixed_precision"] = True
    current["cudnn_deterministic"] = True
    current["cudnn_benchmark"] = False
    current["parent_gpu_receipt"] = parent_receipt.get("gpu")
    current["parent_node"] = parent_receipt.get("node")
    current["current_node"] = current_node
    current["same_physical_node"] = True
    current["parent_cuda_device"] = parent.get("cuda_device")
    current["same_python_executable"] = True
    current["same_gpu_model"] = True
    current["same_python_and_recorded_library_versions"] = True
    current["same_amp_mode"] = bool(
        parent_manifest.get("training", {}).get("automatic_mixed_precision") is True
    )
    current["same_deterministic_recipe"] = True
    current["fields_compared_exactly"] = [
        "python",
        "platform",
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "cuda_device",
        "python_executable",
        "automatic_mixed_precision",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ]
    current["matches_original"] = True
    return current


def source_snapshot(output: Path) -> dict[str, str]:
    require(
        sha256_file(ADDENDUM_PLAN_PATH) == ADDENDUM_PLAN_SHA256,
        "frozen addendum plan digest changed",
    )
    result: dict[str, str] = {}
    for relative, source in SOURCE_PATHS.items():
        require(source.is_file(), f"addendum source missing: {source}")
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        result[str(destination.relative_to(output))] = sha256_file(destination)
    return result


def output_checksums(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative not in {"manifest.json", "failure.json"}:
            result[relative] = sha256_file(path)
    return result


def base_context_provenance(
    context: base.ContextBundle,
    weekly_climatology: np.ndarray,
) -> dict[str, Any]:
    return array_bundle_provenance(
        {
            "weekly_climatology": np.asarray(weekly_climatology),
            "normalized_climatology": np.asarray(context.normalized_climatology),
            "climatology_mean_by_lead": np.asarray(context.climatology_mean_by_lead),
            "climatology_std_by_lead": np.asarray(context.climatology_std_by_lead),
            "latitude_scaled": np.asarray(context.latitude_scaled),
            "longitude_scaled": np.asarray(context.longitude_scaled),
            "season_sin": np.asarray(context.season_sin),
            "season_cos": np.asarray(context.season_cos),
            "lead_scaled": np.asarray(context.lead_scaled),
            "support": np.asarray(context.support),
        },
        bundle_name="full_base_context_and_weekly_climatology",
    )


def audit_threshold_artifact(
    parent_root: Path,
    parent_manifest: Mapping[str, Any],
    observations: base.ObservationBundle,
    initializations: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    support: np.ndarray,
    destination: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    relative = parent_manifest.get("categorical_validation", {}).get(
        "threshold_fit_artifact"
    )
    require(relative == "evaluation/validation_quintile_fit.npz", "parent threshold path")
    parent_path = parent_root / str(relative)
    expected_digest = parent_manifest.get("categorical_validation", {}).get(
        "threshold_fit_sha256"
    )
    require(sha256_file(parent_path) == expected_digest, "parent threshold digest")
    with np.load(parent_path, allow_pickle=False) as archive:
        parent_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "levels",
        "thresholds",
        "empirical_cdf",
        "support",
        "fit_indices",
        "sample_count_by_day",
        "validation_initializations",
        "validation_thresholds",
    }
    require(set(parent_arrays) == required, "parent threshold artifact schema")
    quantiles = fit_calendar_quantiles(
        observations.weekly_truth,
        initializations,
        train_indices,
        v1.QUINTILE_LEVELS,
        support,
        window_radius_days=(v1.CALENDAR_WINDOW_DAYS - 1) // 2,
        minimum_samples=v1.MINIMUM_CALENDAR_SAMPLES,
    )
    validation_starts = np.asarray(initializations)[validation_indices]
    rebuilt = {
        "levels": quantiles.levels,
        "thresholds": quantiles.thresholds,
        "empirical_cdf": quantiles.empirical_cdf,
        "support": quantiles.support,
        "fit_indices": quantiles.fit_indices,
        "sample_count_by_day": quantiles.sample_count_by_day,
        "validation_initializations": validation_starts,
        "validation_thresholds": calendar_fields(quantiles, validation_starts, 6),
    }
    for name in sorted(required):
        require(array_equal(parent_arrays[name], rebuilt[name]), f"rebuilt threshold {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_path, destination)
    require(sha256_file(destination) == expected_digest, "copied threshold digest")
    thresholds = parent_arrays["validation_thresholds"]
    truth = np.asarray(observations.weekly_truth[validation_indices], dtype=np.float32)
    observed = observation_cdf(truth, thresholds)
    require(
        is_valid_cdf_for_thresholds(observed, thresholds, axis=2),
        "rebuilt validation observation CDF",
    )
    return thresholds, observed, {
        "parent_artifact": str(parent_path),
        "copied_artifact": "evaluation/validation_quintile_fit.npz",
        "sha256": str(expected_digest),
        "all_arrays_rebuilt_byte_exact": True,
        "fit_indices": train_indices.tolist(),
        "validation_initializations": [
            np.datetime_as_string(value, unit="D") for value in validation_starts
        ],
    }


def audit_scoring_support_artifact(
    parent_root: Path,
    parent_manifest: Mapping[str, Any],
    cache: base.MemberCache,
    observations: base.ObservationBundle,
    destination: Path,
) -> dict[str, Any]:
    evaluation = parent_manifest.get("evaluation", {})
    relative = evaluation.get("scoring_support_artifact")
    require(relative == "evaluation/scoring_support.npz", "parent support path")
    expected_digest = evaluation.get("scoring_support_sha256")
    require(is_digest(expected_digest), "parent support digest")
    parent_path = parent_root / str(relative)
    require(sha256_file(parent_path) == expected_digest, "parent support artifact")
    with np.load(parent_path, allow_pickle=False) as archive:
        parent_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    support = np.asarray(observations.weights > 0.0, dtype=np.bool_)
    normalized = observations.weights / observations.weights.sum(dtype=np.float64)
    rebuilt = {
        "latitude": cache.latitude.astype(np.float64),
        "longitude": cache.longitude.astype(np.float64),
        "observation_fraction": observations.observation_fraction.astype(np.float32),
        "support_mask": support,
        "scoring_weight_km2_fraction": observations.weights.astype(np.float64),
        "normalized_scoring_weight": normalized.astype(np.float64),
    }
    require(set(parent_arrays) == set(rebuilt), "parent support artifact schema")
    for name in sorted(rebuilt):
        require(array_equal(parent_arrays[name], rebuilt[name]), f"rebuilt support {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_path, destination)
    require(sha256_file(destination) == expected_digest, "copied support digest")
    return {
        "parent_artifact": str(parent_path),
        "copied_artifact": "evaluation/scoring_support.npz",
        "sha256": str(expected_digest),
        "all_arrays_rebuilt_byte_exact": True,
        "support_cells": int(np.count_nonzero(support)),
    }


def audit_persistence_normalization_artifact(
    parent_root: Path,
    context: PersistenceContextBundle,
    destination: Path,
) -> dict[str, Any]:
    parent_path = parent_root / "evaluation/persistence_normalization.npz"
    with np.load(parent_path, allow_pickle=False) as archive:
        parent_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rebuilt = {
        "lag_log1p_mean_by_lag": context.lag_log1p_mean_by_lag,
        "lag_log1p_std_by_lag": context.lag_log1p_std_by_lag,
        "normalization_fit_indices": context.normalization_fit_indices,
        "channel_names": np.asarray(
            (
                "lag_1week_log1p_z",
                "lag_2week_log1p_z",
                "lag_1week_available",
                "lag_2week_available",
            )
        ),
    }
    require(set(parent_arrays) == set(rebuilt), "parent normalization schema")
    for name in sorted(rebuilt):
        require(
            array_equal(parent_arrays[name], np.asarray(rebuilt[name])),
            f"rebuilt normalization {name}",
        )
    parent_digest = sha256_file(parent_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_path, destination)
    require(sha256_file(destination) == parent_digest, "copied normalization digest")
    return {
        "parent_artifact": str(parent_path),
        "copied_artifact": "evaluation/persistence_normalization.npz",
        "sha256": parent_digest,
        "all_arrays_rebuilt_byte_exact": True,
    }


def _parent_run_map(
    parent_manifest: Mapping[str, Any], expected_seeds: Sequence[int]
) -> dict[tuple[str, int], Mapping[str, Any]]:
    expected = {(arm, int(seed)) for arm in V1_ARMS for seed in expected_seeds}
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in parent_manifest.get("training", {}).get("runs", []):
        require(isinstance(row, dict), "parent training run record")
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        require(key in expected and key not in result, "parent training run grid")
        result[key] = row
    require(set(result) == expected, "parent training run inventory")
    return result


def reproduce_parent_validation(
    parent_root: Path,
    parent_manifest: Mapping[str, Any],
    cache: base.MemberCache,
    observations: base.ObservationBundle,
    context: PersistenceContextBundle,
    validation_indices: np.ndarray,
    thresholds: np.ndarray,
    observed_cdf: np.ndarray,
    *,
    seeds: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Re-infer and re-score every immutable v1 validation checkpoint."""

    persisted_path = parent_root / "metrics/validation_case_metrics.csv"
    persisted = pd.read_csv(persisted_path)
    runs = _parent_run_map(parent_manifest, seeds)
    starts = np.asarray(cache.initializations)[validation_indices]
    reproduced_frames: list[pd.DataFrame] = []
    run_evidence: list[dict[str, Any]] = []
    for arm in V1_ARMS:
        for seed in seeds:
            row = runs[(arm, int(seed))]
            checkpoint = parent_root / str(row["checkpoint"])
            adjustment = parent_root / str(row["validation_adjustment"])
            require(
                sha256_file(checkpoint) == row.get("checkpoint_sha256"),
                f"parent checkpoint changed {arm}/{seed}",
            )
            require(
                sha256_file(adjustment) == row.get("validation_adjustment_sha256"),
                f"parent adjustment changed {arm}/{seed}",
            )
            base.set_deterministic_seed(int(seed))
            expected_rng = v1.training_rng_state_sha256(device)
            require(
                expected_rng == row.get("training_rng_state_sha256"),
                f"reconstructed training RNG differs {arm}/{seed}",
            )
            model = v1.load_checkpoint_model(checkpoint, arm, int(seed), device)
            delta, log_spread = v1.predict_adjustments(
                model,
                arm,
                cache.members,
                observations.weekly_truth,
                context,
                validation_indices,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                use_amp=use_amp,
            )
            rebuilt_arrays = {
                "initializations": starts,
                "delta_log_location": delta,
                "log_spread": log_spread,
                "spread_factor": np.exp(np.clip(log_spread, -2.0, 2.0)).astype(
                    np.float32
                ),
                "arm": np.asarray(arm),
                "seed": np.asarray(seed, dtype=np.int64),
            }
            with np.load(adjustment, allow_pickle=False) as archive:
                require(
                    set(archive.files) == set(rebuilt_arrays),
                    f"parent adjustment schema {arm}/{seed}",
                )
                for name, values in rebuilt_arrays.items():
                    require(
                        array_equal(np.asarray(archive[name]), np.asarray(values)),
                        f"reproduced adjustment differs {arm}/{seed}/{name}",
                    )
            reproduced_frames.append(
                v1.validation_case_metrics(
                    arm,
                    int(seed),
                    cache.members,
                    observations.weekly_truth,
                    cache.initializations,
                    validation_indices,
                    delta,
                    log_spread,
                    observations.weights,
                    thresholds,
                    observed_cdf,
                    chunk_size=chunk_size,
                )
            )
            run_evidence.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "validation_adjustment": str(adjustment),
                    "validation_adjustment_sha256": str(
                        row["validation_adjustment_sha256"]
                    ),
                    "initial_state_sha256": str(row["initial_state_sha256"]),
                    "training_rng_state_sha256": str(
                        row["training_rng_state_sha256"]
                    ),
                    "all_adjustment_arrays_reproduced_byte_exact": True,
                }
            )
            del model, delta, log_spread
            if device.type == "cuda":
                torch.cuda.empty_cache()
    reproduced = pd.concat(reproduced_frames, ignore_index=True)
    score_evidence = compare_validation_frames(persisted, reproduced)
    original_recomputed = v1.select_persistence_arm(
        persisted,
        expected_seeds=seeds,
        expected_initializations=starts,
    )
    require(
        _stable_original_selection(original_recomputed)
        == _stable_original_selection(parent_manifest.get("selection", {})),
        "recomputed original selector differs from immutable parent",
    )
    return persisted, reproduced, {
        "status": "all_original_validation_artifacts_reproduced",
        "parent_validation_case_metrics": str(persisted_path),
        "parent_validation_case_metrics_sha256": sha256_file(persisted_path),
        "runs": run_evidence,
        "all_available_parent_checkpoints_reproduced": True,
        "all_adjustment_arrays_reproduced_byte_exact": True,
        "persisted_score_reproduction": score_evidence,
        "original_selection_recomputed_exact": True,
    }


def categorical_validity_evidence(frame: pd.DataFrame, row_count: int) -> dict[str, Any]:
    columns = [
        "cdf_shape_matches_thresholds",
        "forecast_cdf_finite_where_threshold_defined",
        "observed_cdf_finite_where_threshold_defined",
        "forecast_cdf_valid_for_thresholds",
        "observed_cdf_valid_for_thresholds",
    ]
    require(len(frame) == row_count, "mask validation metric row count")
    for column in columns:
        require(column in frame and bool(frame[column].all()), f"mask CDF evidence {column}")
    require(
        np.isfinite(frame[["crps", "quintile_rps"]].to_numpy(dtype=np.float64)).all(),
        "mask validation scores are non-finite",
    )
    return {
        "evidence_columns": columns,
        "validation_metric_rows": int(len(frame)),
        "every_scoring_chunk_shape_matched_thresholds": bool(
            frame[columns[0]].all()
        ),
        "every_forecast_cdf_finite_where_threshold_defined": bool(
            frame[columns[1]].all()
        ),
        "every_observed_cdf_finite_where_threshold_defined": bool(
            frame[columns[2]].all()
        ),
        "every_forecast_cdf_valid_for_thresholds": bool(frame[columns[3]].all()),
        "every_observed_cdf_valid_for_thresholds": bool(frame[columns[4]].all()),
        "every_validation_score_finite": True,
    }


def build_readme(selection: Mapping[str, Any], *, smoke: bool) -> str:
    status = "non-scientific smoke" if smoke else "frozen full validation"
    return "\n".join(
        [
            "# FuXi persistence active-mask control",
            "",
            f"Status: {status} addendum.",
            "",
            "This output trains only the nonselectable `mask_only_45k` arm. Its ",
            "rainfall positions are exact zero and its two issue-safe availability ",
            "masks match the immutable persistence candidate.",
            "",
            f"Joint validation selected `{selection['selected_arm']}`. ",
            str(selection["reason"]),
            "",
            "No 2020+ initialization was placed in a dataset, scored, or used for ",
            "selection. A separate hash-bound evaluator is required for any PBC ",
            "comparison; the mask-only arm is never selectable.",
            "",
        ]
    )


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    started_at = time.monotonic()
    snapshot = source_snapshot(output)
    seeds = base._parse_seeds(args.seeds)
    device = base.resolve_device(args.device)
    require(device.type == "cuda", f"canonical {EXPERIMENT} requires CUDA")

    parent_manifest, parent_receipt, parent_root, parent_artifacts = (
        audit_parent_envelope(
            args.original_v1_manifest,
            args.original_v1_receipt,
            smoke=bool(args.smoke),
        )
    )
    requested_output = Path(args.output).resolve()
    try:
        requested_output.relative_to(parent_root)
    except ValueError:
        pass
    else:
        raise MaskControlError("mask output may not modify or nest in original v1")

    smoke_parent_binding: dict[str, Any] | None = None
    if args.smoke:
        require(
            parent_receipt.get("smoke_manifest_path") is None
            and parent_receipt.get("smoke_manifest_sha256") is None,
            "original v1 smoke unexpectedly has a smoke parent",
        )
    else:
        smoke_manifest_path = Path(
            str(parent_receipt.get("smoke_manifest_path", ""))
        ).resolve()
        require(
            sha256_file(smoke_manifest_path)
            == parent_receipt.get("smoke_manifest_sha256"),
            "original v1 smoke-parent digest",
        )
        smoke_receipt_path = smoke_manifest_path.parent / "slurm_gate_receipt.json"
        smoke_manifest, smoke_receipt, smoke_root, smoke_artifacts = (
            audit_parent_envelope(
                smoke_manifest_path,
                smoke_receipt_path,
                smoke=True,
            )
        )
        require(
            smoke_manifest.get("source_snapshot_sha256")
            == parent_manifest.get("source_snapshot_sha256"),
            "original full and smoke sources differ",
        )
        smoke_parent_binding = {
            "manifest_path": str(smoke_manifest_path),
            "manifest_sha256": sha256_file(smoke_manifest_path),
            "receipt_path": str(smoke_receipt_path),
            "receipt_sha256": sha256_file(smoke_receipt_path),
            "output_path": str(smoke_root),
            "source_snapshot_sha256": smoke_manifest["source_snapshot_sha256"],
            "artifact_sha256": smoke_artifacts,
            "receipt": smoke_receipt,
        }

    pbc_binding = audit_accepted_pbc(args.pbc_manifest, args.pbc_fit)
    mask_smoke_binding: dict[str, Any] | None = None
    if not args.smoke:
        require(smoke_parent_binding is not None, "missing original-v1 smoke ancestry")
        mask_smoke_binding = audit_mask_smoke_parent(
            args.mask_smoke_manifest,
            args.mask_smoke_receipt,
            current_source_snapshot=snapshot,
            expected_original_v1_smoke_manifest_sha256=str(
                smoke_parent_binding["manifest_sha256"]
            ),
            expected_original_v1_smoke_receipt_sha256=str(
                smoke_parent_binding["receipt_sha256"]
            ),
            accepted_pbc=pbc_binding,
        )
        try:
            requested_output.relative_to(
                Path(str(mask_smoke_binding["output_path"])).resolve()
            )
        except ValueError:
            pass
        else:
            raise MaskControlError("full output may not modify or nest in mask smoke")
    environment = audit_matching_environment(parent_manifest, parent_receipt, device)
    print(
        f"CUDA device: {torch.cuda.get_device_name(device)}; original v1 job "
        f"{parent_receipt['job_id']} on {parent_receipt['node']}",
        flush=True,
    )

    cache = base.load_member_cache(Path(args.cache), allow_partial=False)
    provenance = base.cache_provenance(cache)
    require(
        Path(cache.members_path).resolve() == Path(args.cache).resolve(),
        "loaded cache path differs from requested cache",
    )
    require(
        Path(str(parent_manifest.get("cache", {}).get("data_file", ""))).resolve()
        == Path(cache.members_path).resolve(),
        "mask and original v1 cache paths differ",
    )
    require(provenance == parent_manifest.get("cache"), "cache provenance differs from v1")
    require(
        provenance.get("data_sha256") == v1.EXPECTED_CACHE_SHA256,
        "canonical cache digest receipt",
    )
    actual_cache_sha = sha256_file(Path(cache.members_path))
    require(actual_cache_sha == v1.EXPECTED_CACHE_SHA256, "cache bytes changed")

    splits = base.make_split_indices(cache.initializations)
    split_counts = {name: len(values) for name, values in splits.as_dict().items()}
    require(split_counts == base.EXPECTED_COUNTS, "canonical archive split counts")
    train_indices = splits.train
    validation_indices = splits.validation
    if args.smoke:
        train_indices = base.select_evenly(train_indices, 32)
        validation_indices = base.select_evenly(validation_indices, 16)
    expected_selected = {"train": len(train_indices), "validation": len(validation_indices)}
    require(
        expected_selected == parent_manifest.get("split_counts_selected"),
        "selected split counts differ from original v1",
    )
    inventories = {
        "train": [
            np.datetime_as_string(value, unit="D")
            for value in cache.initializations[train_indices]
        ],
        "validation": [
            np.datetime_as_string(value, unit="D")
            for value in cache.initializations[validation_indices]
        ],
    }
    require(
        inventories == parent_manifest.get("retained_initializations"),
        "selected initialization inventory differs from original v1",
    )
    require(
        set(pd.DatetimeIndex(cache.initializations[train_indices]).year)
        == set(base.TRAIN_YEARS),
        "training inventory years",
    )
    require(
        set(pd.DatetimeIndex(cache.initializations[validation_indices]).year)
        == set(base.VALIDATION_YEARS),
        "validation inventory years",
    )

    observations = base.load_imd_observations(cache)
    require(
        not any("2025" in str(path) for path in observations.source_stores),
        "observation loader opened sealed 2025",
    )
    observation_provenance = array_bundle_provenance(
        {
            "weekly_truth": observations.weekly_truth,
            "observation_fraction": observations.observation_fraction,
            "weights": observations.weights,
        },
        bundle_name="imd_weekly_truth_fraction_and_scoring_weights",
    )
    require(
        observation_provenance == parent_manifest.get("observation_bundle_provenance"),
        "observation bundle differs from original v1",
    )
    require(
        observation_provenance["sha256"] == EXPECTED_OBSERVATION_BUNDLE_SHA256,
        "observation bundle differs from accepted PBC",
    )
    daily_dates, daily_values = pbc_driver.load_daily_imd_for_lags(
        cache, observations.source_stores
    )
    lags = build_daily_issue_time_lags(
        cache.initializations, daily_dates, daily_values, lag_weeks=(1, 2)
    )
    del daily_dates, daily_values
    lag_provenance = dict(pbc_driver.issue_time_lag_provenance(lags))
    support = observations.weights > 0.0
    available_values = np.asarray(lags.values)[lags.source_indices >= 0][:, support]
    require(
        np.isfinite(available_values).all() and np.all(available_values >= 0.0),
        "lag values invalid on scoring support",
    )
    lag_provenance["available_supported_values_finite_nonnegative"] = True
    require(
        lag_provenance == parent_manifest.get("persistence_lag_provenance"),
        "issue-time lag bundle differs from original v1",
    )
    require(
        lag_provenance["sha256"] == EXPECTED_LAG_BUNDLE_SHA256,
        "issue-time lag bundle differs from accepted PBC",
    )
    temporal_evidence = v1.assert_validation_temporal_contract(
        cache.initializations, train_indices, validation_indices, lags
    )
    temporal_evidence["lag_windows"] = (
        "issue-7 through issue-1 and issue-14 through issue-8; every end < issue"
    )
    temporal_evidence["datasets_scored_or_selected_indices"] = {
        "train": train_indices.tolist(),
        "validation": validation_indices.tolist(),
        "development_or_later": [],
    }
    temporal_evidence["full_observation_and_lag_bundles_materialized_for_hashing"] = True

    base_context = base.build_context_bundle(cache, observations, train_indices)
    context = build_persistence_context_bundle(
        base_context,
        cache.initializations,
        lags,
        train_indices,
        observations.weights,
    )
    context_provenance = array_bundle_provenance(
        {
            "normalized_lag_log1p": context.normalized_lag_log1p,
            "lag_available": context.lag_available,
            "lag_log1p_mean_by_lag": context.lag_log1p_mean_by_lag,
            "lag_log1p_std_by_lag": context.lag_log1p_std_by_lag,
            "normalization_fit_indices": context.normalization_fit_indices,
        },
        bundle_name="train_normalized_persistence_context",
    )
    require(
        context_provenance == parent_manifest.get("persistence_context_provenance"),
        "persistence context differs from original v1",
    )
    full_base_context_provenance = base_context_provenance(
        base_context, observations.weekly_climatology
    )
    mask_only_context_provenance = array_bundle_provenance(
        {
            "forced_zero_normalized_rain": np.zeros_like(
                context.normalized_lag_log1p, dtype=np.float32
            ),
            "support_masked_lag_available": (
                context.lag_available[..., None, None] & support[None, None]
            ).astype(np.float32),
            "added_channel_names": np.asarray(MASK_ONLY_ADDED_CHANNEL_NAMES),
        },
        bundle_name="mask_only_added_issue_time_fields",
    )

    evaluation = output / "evaluation"
    metrics = output / "metrics"
    history_directory = output / "history"
    models = output / "models"
    for directory in (evaluation, metrics, history_directory, models):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(evaluation / "observation_bundle_provenance.json", observation_provenance)
    write_json(evaluation / "persistence_lag_provenance.json", lag_provenance)
    write_json(evaluation / "persistence_context_provenance.json", context_provenance)
    write_json(
        evaluation / "base_context_and_weekly_climatology_provenance.json",
        full_base_context_provenance,
    )
    write_json(
        evaluation / "mask_only_context_provenance.json",
        mask_only_context_provenance,
    )
    scoring_support_evidence = audit_scoring_support_artifact(
        parent_root,
        parent_manifest,
        cache,
        observations,
        evaluation / "scoring_support.npz",
    )
    normalization_evidence = audit_persistence_normalization_artifact(
        parent_root, context, evaluation / "persistence_normalization.npz"
    )
    thresholds, observed_cdf, threshold_evidence = audit_threshold_artifact(
        parent_root,
        parent_manifest,
        observations,
        cache.initializations,
        train_indices,
        validation_indices,
        support,
        evaluation / "validation_quintile_fit.npz",
    )

    parent_metrics, reproduced_metrics, reproduction = reproduce_parent_validation(
        parent_root,
        parent_manifest,
        cache,
        observations,
        context,
        validation_indices,
        thresholds,
        observed_cdf,
        seeds=seeds,
        device=device,
        batch_size=args.evaluation_batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        chunk_size=args.cdf_chunk_size,
    )
    copied_parent_metrics = metrics / "original_validation_case_metrics.csv"
    shutil.copy2(parent_root / "metrics/validation_case_metrics.csv", copied_parent_metrics)
    reproduced_metrics_path = metrics / "original_validation_case_metrics_reproduced.csv"
    reproduced_metrics.to_csv(reproduced_metrics_path, index=False)
    reproduction.update(
        {
            "copied_parent_validation_case_metrics": str(
                copied_parent_metrics.relative_to(output)
            ),
            "copied_parent_validation_case_metrics_sha256": sha256_file(
                copied_parent_metrics
            ),
            "reproduced_validation_case_metrics": str(
                reproduced_metrics_path.relative_to(output)
            ),
            "reproduced_validation_case_metrics_sha256": sha256_file(
                reproduced_metrics_path
            ),
        }
    )

    parent_runs = _parent_run_map(parent_manifest, seeds)
    histories: list[pd.DataFrame] = []
    mask_frames: list[pd.DataFrame] = []
    mask_records: list[MaskTrainingRun] = []
    starts = np.asarray(cache.initializations)[validation_indices]
    for seed in seeds:
        print(f"Training {MASK_ONLY_ARM}, seed {seed}...", flush=True)
        run_directory = models / MASK_ONLY_ARM / f"seed_{seed}"
        frame, record = train_mask_only(
            int(seed),
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
        for parent_arm in V1_ARMS:
            require(
                record.training_rng_state_sha256
                == parent_runs[(parent_arm, int(seed))].get(
                    "training_rng_state_sha256"
                ),
                f"mask stochastic stream differs from {parent_arm}/{seed}",
            )
        for parent_arm in (ZERO_LAG_ARM, PERSISTENCE_LAG_ARM):
            require(
                record.initial_state_sha256
                == parent_runs[(parent_arm, int(seed))].get("initial_state_sha256"),
                f"mask initial state differs from {parent_arm}/{seed}",
            )
        model = load_mask_checkpoint(Path(record.checkpoint), int(seed), device)
        delta, log_spread = predict_mask_adjustments(
            model,
            cache.members,
            observations.weekly_truth,
            context,
            validation_indices,
            device=device,
            batch_size=args.evaluation_batch_size,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
        )
        adjustment = run_directory / "validation_adjustments.npz"
        np.savez_compressed(
            adjustment,
            initializations=starts,
            delta_log_location=delta,
            log_spread=log_spread,
            spread_factor=np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32),
            arm=np.asarray(MASK_ONLY_ARM),
            seed=np.int64(seed),
        )
        mask_frames.append(
            v1.validation_case_metrics(
                MASK_ONLY_ARM,
                int(seed),
                cache.members,
                observations.weekly_truth,
                cache.initializations,
                validation_indices,
                delta,
                log_spread,
                observations.weights,
                thresholds,
                observed_cdf,
                chunk_size=args.cdf_chunk_size,
            )
        )
        histories.append(frame)
        mask_records.append(record)
        del model, delta, log_spread
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history_frame = pd.concat(histories, ignore_index=True)
    mask_metrics = pd.concat(mask_frames, ignore_index=True)
    categorical_evidence = categorical_validity_evidence(
        mask_metrics, len(seeds) * len(validation_indices) * 6
    )
    categorical_evidence.update(
        {
            "threshold_and_observed_shapes_equal": bool(
                thresholds.shape == observed_cdf.shape
            ),
            "thresholds_finite_on_scoring_support": bool(
                np.isfinite(thresholds[..., support]).all()
            ),
            "observed_cdf_finite_on_scoring_support": bool(
                np.isfinite(observed_cdf[..., support]).all()
            ),
            "observed_cdf_valid_for_thresholds": bool(
                is_valid_cdf_for_thresholds(observed_cdf, thresholds, axis=2)
            ),
        }
    )
    require(
        all(
            bool(value)
            for key, value in categorical_evidence.items()
            if key not in {"evidence_columns", "validation_metric_rows"}
        ),
        "mask categorical validation evidence failed",
    )
    joint_selection = select_joint_arm(
        parent_metrics,
        mask_metrics,
        expected_seeds=seeds,
        expected_initializations=starts,
    )
    joint_selection["written_utc"] = utc_now()
    joint_selection["scientific_selection"] = not args.smoke

    history_path = history_directory / "training_history.csv"
    mask_metrics_path = metrics / "mask_validation_case_metrics.csv"
    joint_metrics_path = metrics / "joint_validation_case_metrics.csv"
    history_frame.to_csv(history_path, index=False)
    mask_metrics.to_csv(mask_metrics_path, index=False)
    pd.concat((parent_metrics, mask_metrics), ignore_index=True).to_csv(
        joint_metrics_path, index=False
    )
    persisted_mask_metrics = pd.read_csv(mask_metrics_path)
    persisted_selection = select_joint_arm(
        pd.read_csv(copied_parent_metrics),
        persisted_mask_metrics,
        expected_seeds=seeds,
        expected_initializations=starts,
    )
    require(
        persisted_selection
        == {
            key: value
            for key, value in joint_selection.items()
            if key not in {"written_utc", "scientific_selection"}
        },
        "persisted metric artifacts do not reproduce joint selection",
    )
    combined = pd.concat((parent_metrics, mask_metrics), ignore_index=True)
    summary, by_year, by_seed, by_lead = v1.validation_summary_frames(
        combined, joint_selection
    )
    summary.to_csv(metrics / "joint_validation_summary.csv", index=False)
    by_year.to_csv(metrics / "joint_validation_by_year.csv", index=False)
    by_seed.to_csv(metrics / "joint_validation_by_seed.csv", index=False)
    by_lead.to_csv(metrics / "joint_validation_by_lead.csv", index=False)
    joint_selection_path = output / "joint_selection.json"
    write_json(joint_selection_path, joint_selection)
    (output / "README.md").write_text(
        build_readme(joint_selection, smoke=args.smoke), encoding="utf-8"
    )

    run_records: list[dict[str, Any]] = []
    for record in mask_records:
        values = asdict(record)
        checkpoint = Path(record.checkpoint)
        adjustment = checkpoint.parent.parent / "validation_adjustments.npz"
        values.update(
            {
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_adjustment": str(adjustment.relative_to(output)),
                "validation_adjustment_sha256": sha256_file(adjustment),
                "same_seed_initial_state_as_original_45k_arms": True,
                "same_seed_training_rng_as_all_original_arms": True,
            }
        )
        run_records.append(values)

    parent_manifest_path = Path(args.original_v1_manifest).resolve()
    parent_receipt_path = Path(args.original_v1_receipt).resolve()
    original_binding = {
        "experiment": v1.EXPERIMENT,
        "mode": parent_manifest["mode"],
        "output_path": str(parent_root),
        "manifest_path": str(parent_manifest_path),
        "manifest_sha256": sha256_file(parent_manifest_path),
        "receipt_path": str(parent_receipt_path),
        "receipt_sha256": sha256_file(parent_receipt_path),
        "receipt_schema": "fuxi_persistence_slurm_gate_receipt_v1",
        "selection_path": str(parent_root / "selection.json"),
        "selection_sha256": sha256_file(parent_root / "selection.json"),
        "validation_case_metrics_path": str(
            parent_root / "metrics/validation_case_metrics.csv"
        ),
        "validation_case_metrics_sha256": sha256_file(
            parent_root / "metrics/validation_case_metrics.csv"
        ),
        "source_snapshot_sha256": parent_manifest["source_snapshot_sha256"],
        "source_binding_sha256": mapping_sha256(
            parent_manifest["source_snapshot_sha256"]
        ),
        "artifact_sha256": parent_artifacts,
        "artifact_binding_sha256": mapping_sha256(parent_artifacts),
        "external_receipt": parent_receipt,
        "smoke_parent": smoke_parent_binding,
    }
    addendum_plan = {
        "path": str(ADDENDUM_PLAN_PATH),
        "expected_sha256": ADDENDUM_PLAN_SHA256,
        "live_sha256": sha256_file(ADDENDUM_PLAN_PATH),
        "snapshot_path": "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md",
        "snapshot_sha256": snapshot[
            "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
        ],
    }
    manifest: dict[str, Any] = {
        "schema_version": "fuxi_persistence_mask_control_manifest_v1",
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "frozen validation-only four-arm attribution selection"
        ),
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "output_path": str(requested_output),
        "command_line": [sys.executable, *sys.argv],
        "addendum_plan": addendum_plan,
        "mask_smoke_parent": mask_smoke_binding,
        "original_v1": original_binding,
        "accepted_pbc": pbc_binding,
        "contract": {
            "forecast": "FuXi native weekly TP; all seasons; 51 members",
            "target": "IMD weekly mean precipitation, mm day-1",
            "train_years": list(base.TRAIN_YEARS),
            "validation_years": list(base.VALIDATION_YEARS),
            "selection_data": "2018-2019 validation only",
            "test_metrics_consulted": False,
            "development_years_not_scored": list(base.TEST_YEARS),
            "sealed_unopened_years": list(base.SEALED_YEARS),
            "sealed_2025_target_opened": False,
            "added_channels": list(MASK_ONLY_ADDED_CHANNEL_NAMES),
            "added_channel_values": ["zero", "zero", "true_mask", "true_mask"],
            "categorical_score_contract": v1.SCORING_CONTRACT_VERSION,
            "statistical_unit": "initialization with all members and all six leads grouped",
            "mask_only_selectable": False,
        },
        "arms": [
            {
                "name": MASK_ONLY_ARM,
                "role": "active_mask_attribution_control_nonselectable",
                "context_channels": MASK_ONLY_CONTEXT_CHANNEL_COUNT,
                "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
                "selectable": False,
            }
        ],
        "seeds": list(seeds),
        "split_counts_archive": split_counts,
        "split_counts_selected": expected_selected,
        "retained_initializations": inventories,
        "joint_selection": joint_selection,
        "joint_selection_artifact": str(joint_selection_path.relative_to(output)),
        "joint_selection_sha256": sha256_file(joint_selection_path),
        "validation_reproduction": reproduction,
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
            "optimizer": "torch.optim.AdamW",
            "scheduler": "ReduceLROnPlateau(factor=0.5,min_lr=1e-6)",
            "gradient_clip_max_norm": 5.0,
            "same_hyperparameters_as_original_v1": True,
            "same_seed_initial_state_as_original_45k_arms": True,
            "same_seed_stochastic_training_stream_as_all_original_arms": True,
            "runs": run_records,
        },
        "categorical_validation": {
            "levels": v1.QUINTILE_LEVELS.tolist(),
            "calendar_window_days": v1.CALENDAR_WINDOW_DAYS,
            "minimum_calendar_samples": v1.MINIMUM_CALENDAR_SAMPLES,
            "threshold_fit_indices": train_indices.tolist(),
            "threshold_fit_artifact": "evaluation/validation_quintile_fit.npz",
            "threshold_fit_sha256": sha256_file(
                evaluation / "validation_quintile_fit.npz"
            ),
            "score_contract": v1.SCORING_CONTRACT_VERSION,
            "finite_and_cdf_validity_evidence": categorical_evidence,
            "original_threshold_rebuild": threshold_evidence,
        },
        "evaluation": {
            "scope": "validation only",
            "mask_validation_case_metrics_artifact": str(
                mask_metrics_path.relative_to(output)
            ),
            "mask_validation_case_metrics_sha256": sha256_file(mask_metrics_path),
            "joint_validation_case_metrics_artifact": str(
                joint_metrics_path.relative_to(output)
            ),
            "joint_validation_case_metrics_sha256": sha256_file(joint_metrics_path),
            "scoring_support_artifact": "evaluation/scoring_support.npz",
            "scoring_support_sha256": sha256_file(
                evaluation / "scoring_support.npz"
            ),
            "support_cells": int(np.count_nonzero(support)),
            "persisted_metrics_recompute_joint_selection_exact": True,
        },
        "scoring_support_rebuild": scoring_support_evidence,
        "persistence_normalization_rebuild": normalization_evidence,
        "temporal_evidence": temporal_evidence,
        "persistence_lag_provenance": lag_provenance,
        "observation_bundle_provenance": observation_provenance,
        "persistence_context_provenance": context_provenance,
        "base_context_and_weekly_climatology_provenance": (
            full_base_context_provenance
        ),
        "mask_only_context_provenance": mask_only_context_provenance,
        "cache": {**provenance, "actual_data_sha256": actual_cache_sha},
        "observation_stores": list(observations.source_stores),
        "hardware_software_match": environment,
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "source_snapshot_sha256": snapshot,
        "source_binding_sha256": mapping_sha256(snapshot),
    }
    require_finite_json(manifest)
    manifest["artifact_sha256"] = output_checksums(output)
    manifest["artifact_binding_sha256"] = mapping_sha256(
        manifest["artifact_sha256"]
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the validation-only active-mask persistence addendum."
    )
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--original-v1-manifest", type=Path, required=True)
    parser.add_argument("--original-v1-receipt", type=Path, required=True)
    parser.add_argument("--mask-smoke-manifest", type=Path, default=None)
    parser.add_argument("--mask-smoke-receipt", type=Path, default=None)
    parser.add_argument("--pbc-manifest", type=Path, default=DEFAULT_PBC_MANIFEST)
    parser.add_argument("--pbc-fit", type=Path, default=DEFAULT_PBC_FIT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--member-subsample", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--cdf-chunk-size", type=int, default=8)
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
    seeds = base._parse_seeds(args.seeds)
    expected_seeds = (42,) if args.smoke else base.SEEDS
    if seeds != expected_seeds:
        raise ValueError(f"canonical seeds must be {expected_seeds}")
    fixed = {
        "max_epochs": (args.max_epochs, 2 if args.smoke else 100),
        "patience": (args.patience, 1 if args.smoke else 15),
        "batch_size": (args.batch_size, 8),
        "member_subsample": (args.member_subsample, 16),
        "evaluation_batch_size": (args.evaluation_batch_size, 8),
        "cdf_chunk_size": (args.cdf_chunk_size, 8),
        "num_workers": (args.num_workers, 0),
    }
    mismatch = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed.items()
        if actual != expected
    }
    if mismatch:
        raise ValueError(f"canonical settings differ: {mismatch}")
    if not math.isclose(args.learning_rate, 2.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical learning rate is 0.0002")
    if not math.isclose(args.weight_decay, 1.0e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("canonical weight decay is 0.0001")
    if args.no_amp or args.device == "cpu":
        raise ValueError("canonical mask-control run requires CUDA AMP")
    for name in (
        "cache",
        "original_v1_manifest",
        "original_v1_receipt",
        "pbc_manifest",
        "pbc_fit",
    ):
        path = Path(getattr(args, name))
        if not path.is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be absolute")
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.smoke:
        if args.mask_smoke_manifest is not None or args.mask_smoke_receipt is not None:
            raise ValueError("smoke mode may not accept a mask-smoke parent")
    else:
        if args.mask_smoke_manifest is None or args.mask_smoke_receipt is None:
            raise ValueError(
                "full mode requires --mask-smoke-manifest and --mask-smoke-receipt"
            )
        for name in ("mask_smoke_manifest", "mask_smoke_receipt"):
            path = Path(getattr(args, name))
            if not path.is_absolute():
                raise ValueError(f"--{name.replace('_', '-')} must be absolute")
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.output is not None and not Path(args.output).is_absolute():
        raise ValueError("--output must be absolute")
    if args.output is not None:
        requested = Path(args.output).resolve()
        protected = [
            Path(args.original_v1_manifest).resolve().parent,
            Path(args.pbc_manifest).resolve().parent,
        ]
        if args.mask_smoke_manifest is not None:
            protected.append(Path(args.mask_smoke_manifest).resolve().parent)
        for root in protected:
            try:
                requested.relative_to(root)
            except ValueError:
                continue
            raise ValueError(f"--output may not modify or nest in immutable input {root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested_output = (default_output() if args.output is None else args.output).resolve()
    args.output = requested_output
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if requested_output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {requested_output}")
    staging = requested_output.parent / f".{requested_output.name}.incomplete-{os.getpid()}"
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
        f"PASS: completed {'smoke' if args.smoke else 'full'} mask-control run at "
        f"{requested_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
