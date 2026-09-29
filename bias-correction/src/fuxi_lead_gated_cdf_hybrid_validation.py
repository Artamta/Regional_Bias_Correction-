#!/usr/bin/env python3
"""One-shot validation of a deployable neural-CDF/Persistence++ hybrid.

W1 is the equal-weight pool of three frozen neural seed CDFs.  W2--W6 are
frozen Persistence++.  The evaluator uses only purged 2018--2019 validation
cases and cannot access the sealed 2025 target.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import zarr

import fuxi_allseason_ensemble_calibration as neural
import fuxi_allseason_operational_categorical_comparison as operational_pbc
import fuxi_allseason_pbc_baseline as pbc_driver
import fuxi_pbc_core as pbc
from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_lead_gated_cdf_hybrid_validation_v1"
POLICY = "neural_seed_pool_w1__persistence_w2_w6"
PLAN_PATH = PROJECT_ROOT / "plan/LEAD_GATED_CDF_POOL_VALIDATION_20260824.md"
PLAN_SHA256 = "8b3f7c5e5a35d896fe8084900e028c5c1a6c17a041e9370b227692146af79dec"
SOURCE_PATH = PROJECT_ROOT / "src/fuxi_lead_gated_cdf_hybrid_validation.py"
TEST_PATH = PROJECT_ROOT / "tests/test_fuxi_lead_gated_cdf_hybrid_validation.py"

NEURAL_ROOT = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_ensemble_calibration/"
    "full_publication_20260822T115253Z"
)
NEURAL_MANIFEST_PATH = NEURAL_ROOT / "manifest.json"
NEURAL_MANIFEST_SHA256 = (
    "94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91"
)
PBC_ROOT = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z"
)
PBC_MANIFEST_PATH = PBC_ROOT / "manifest.json"
PBC_MANIFEST_SHA256 = (
    "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6"
)
ADJUSTMENT_ROOT = (
    PROJECT_ROOT
    / "resultsv2/fuxi_allseason_persistence_augmented/full_20260822T233542Z"
)
ADJUSTMENT_MANIFEST_PATH = ADJUSTMENT_ROOT / "manifest.json"
ADJUSTMENT_MANIFEST_SHA256 = (
    "da4be561bca3c0ffa1a14fcf8f28f8a77e05bb6a60240edc4a48aed65948d642"
)
ADJUSTMENT_SELECTION_PATH = ADJUSTMENT_ROOT / "selection.json"
ADJUSTMENT_SELECTION_SHA256 = (
    "bdc284b0a27e1914c7fd9926a356086288e4edd1ffaf50f24df7cf17c51f1e21"
)

CACHE_DATA_SHA256 = (
    "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70"
)
CACHE_METADATA_SHA256 = (
    "b2cd07dd540cd96ee1bc7ef5df2c59cefffac78122aa56bc42beeb4c3242580a"
)
CACHE_MANIFEST_SHA256 = (
    "4e05cdc8fcbe609e151beb627bd94ee21fd87a5a00efc3a14fdaf804b4ccd0d8"
)
PBC_FIT_SHA256 = (
    "0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36"
)
PBC_SUPPORT_SHA256 = (
    "6149d5d58a9e46b0b8faf4cd71a7bb633f1c3eba0d6fb1bb207f9f5edf590898"
)
OBSERVATION_BUNDLE_SHA256 = (
    "3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49"
)
LAG_BUNDLE_SHA256 = (
    "9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba"
)

SEEDS = (42, 43, 44)
ADJUSTMENT_SHA256 = {
    42: "cc72ba9a927a5ef69f6698ca4be9906182fbd859301fe384811c6e2026670647",
    43: "8164ebefc57567a7c2e919109bb2fcf5582cdfc242b4920af26dbeab8ce82f18",
    44: "bbfb9f4a645108a536c550e1ed5ddfffddb3989ba932cd941195ddcc669b6a4b",
}
PARENT_CHECKPOINT_SHA256 = {
    42: "965e3598ce0f2a3afb679ae0fee26d8c96d2893f8f64ecb64f6b5bb6739a8e0a",
    43: "20ed0c8ba5b7304413b24c4116957f49ee48d8cc8a83d7c3f179cbfc3adcb54c",
    44: "3d1ed90aae4e05e816f09b8b7ecb5e50e2fbeaf0a4299d3b271df571c7785bf1",
}
CANONICAL_CHECKPOINT_SHA256 = {
    42: "56e2a5dd9ad9acc4d7d68bef25c3920e3354fc37358f0ed571f1d89c11878212",
    43: "5c107653ad702d31155f53add9ad91621e3653ec2e9c99ce232feccd609650af",
    44: "8c4ac7d05a41f3187ba468f3129b6a0a8ebcd9aec7d2fbaef05015510bc96615",
}

RAW_METHOD = "raw_fuxi_categorical"
NEURAL_POOL_METHOD = "neural_seed_cdf_pool"
PERSISTENCE_METHOD = "persistence_plus_plus"
COMBINED_METHOD = "pbc_combined"
HYBRID_METHOD = POLICY
METHODS = (
    RAW_METHOD,
    NEURAL_POOL_METHOD,
    PERSISTENCE_METHOD,
    COMBINED_METHOD,
    HYBRID_METHOD,
)
BASELINES = (PERSISTENCE_METHOD, COMBINED_METHOD)
EXPECTED_CASES = 196
EXPECTED_YEAR_COUNTS = {2018: 104, 2019: 92}
LEAD_COUNT = 6
GRID_SHAPE = (27, 27)
CDF_CHUNK_SIZE = 8
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_824
MINIMUM_REDUCTION = 0.005
SCORE_CONTRACT = "normalized_informative_positive_cut_v2"


class HybridValidationError(RuntimeError):
    """Raised when a frozen input, CDF, or selection contract moves."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HybridValidationError(message)


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_bound_manifest(
    path: Path, expected_sha256: str, expected_experiment: str
) -> dict[str, Any]:
    _require(path.is_file(), f"missing input manifest: {path}")
    _require(sha256_file(path) == expected_sha256, f"input manifest changed: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridValidationError(f"invalid input manifest {path}: {error}") from error
    _require(isinstance(payload, dict), f"input manifest is not an object: {path}")
    _require(payload.get("experiment") == expected_experiment, f"experiment moved: {path}")
    _require(payload.get("status") == "complete", f"input is incomplete: {path}")
    contract = payload.get("contract", {})
    _require(
        isinstance(contract, dict) and contract.get("sealed_2025_target_opened") is False,
        f"input does not prove 2025 remained sealed: {path}",
    )
    return payload


def _validate_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    relative: str,
    expected_sha256: str,
) -> Path:
    inventory = manifest.get("artifact_sha256", {})
    _require(
        isinstance(inventory, dict) and inventory.get(relative) == expected_sha256,
        f"manifest no longer binds {relative}",
    )
    path = root / relative
    _require(path.is_file(), f"missing frozen artifact: {path}")
    _require(sha256_file(path) == expected_sha256, f"frozen artifact changed: {path}")
    return path


def validate_frozen_inputs() -> dict[str, Any]:
    """Validate every parent and artifact before forecast reconstruction."""

    neural_manifest = _load_bound_manifest(
        NEURAL_MANIFEST_PATH,
        NEURAL_MANIFEST_SHA256,
        "fuxi_allseason_ensemble_calibration_v1",
    )
    pbc_manifest = _load_bound_manifest(
        PBC_MANIFEST_PATH, PBC_MANIFEST_SHA256, "fuxi_allseason_pbc_baseline_v2"
    )
    adjustment_manifest = _load_bound_manifest(
        ADJUSTMENT_MANIFEST_PATH,
        ADJUSTMENT_MANIFEST_SHA256,
        "fuxi_allseason_persistence_augmented_v1",
    )
    _require(
        sha256_file(ADJUSTMENT_SELECTION_PATH) == ADJUSTMENT_SELECTION_SHA256,
        "persistence experiment selection receipt changed",
    )
    selection = json.loads(ADJUSTMENT_SELECTION_PATH.read_text(encoding="utf-8"))
    _require(
        selection.get("status") == "validation_selection_locked"
        and selection.get("selected_arm") == "base_42k"
        and selection.get("candidate_promoted") is False
        and selection.get("test_metrics_consulted") is False,
        "base neural validation selection contract changed",
    )
    _validate_artifact(PBC_ROOT, pbc_manifest, "models/pbc_fit.npz", PBC_FIT_SHA256)
    _validate_artifact(
        PBC_ROOT,
        pbc_manifest,
        "evaluation/scoring_support.npz",
        PBC_SUPPORT_SHA256,
    )
    for seed in SEEDS:
        _validate_artifact(
            NEURAL_ROOT,
            neural_manifest,
            f"models/location_spread/seed_{seed}/best.pt",
            CANONICAL_CHECKPOINT_SHA256[seed],
        )
        _validate_artifact(
            ADJUSTMENT_ROOT,
            adjustment_manifest,
            f"models/base_42k/seed_{seed}/checkpoints/best.pt",
            PARENT_CHECKPOINT_SHA256[seed],
        )
        _validate_artifact(
            ADJUSTMENT_ROOT,
            adjustment_manifest,
            f"models/base_42k/seed_{seed}/validation_adjustments.npz",
            ADJUSTMENT_SHA256[seed],
        )
    cache_info = neural_manifest.get("cache", {})
    _require(isinstance(cache_info, dict), "neural cache receipt is absent")
    expected_cache = {
        "data_sha256": CACHE_DATA_SHA256,
        "metadata_sha256": CACHE_METADATA_SHA256,
        "manifest_sha256": CACHE_MANIFEST_SHA256,
    }
    _require(
        all(cache_info.get(key) == value for key, value in expected_cache.items()),
        "neural cache receipt changed",
    )
    cache_path = Path(str(cache_info.get("data_file", ""))).resolve()
    metadata_path = Path(str(cache_info.get("metadata_file", ""))).resolve()
    manifest_path = Path(str(cache_info.get("manifest_file", ""))).resolve()
    _require(
        cache_path.is_file() and metadata_path.is_file() and manifest_path.is_file(),
        "canonical member cache files are absent",
    )
    _require(sha256_file(metadata_path) == CACHE_METADATA_SHA256, "cache metadata changed")
    _require(sha256_file(manifest_path) == CACHE_MANIFEST_SHA256, "cache manifest changed")
    return {
        "neural_manifest": neural_manifest,
        "pbc_manifest": pbc_manifest,
        "adjustment_manifest": adjustment_manifest,
        "cache_path": cache_path,
        "cache_declared_sha256": CACHE_DATA_SHA256,
    }


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        _require(torch.is_tensor(tensor), f"checkpoint state is not tensor-valued: {name}")
        array = tensor.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(array.view(np.uint8)))
    return digest.hexdigest()


def verify_checkpoint_identity() -> dict[str, Any]:
    """Prove wrapper-different base checkpoints contain the accepted tensors."""

    receipts: dict[str, Any] = {}
    for seed in SEEDS:
        canonical_path = NEURAL_ROOT / f"models/location_spread/seed_{seed}/best.pt"
        parent_path = (
            ADJUSTMENT_ROOT / f"models/base_42k/seed_{seed}/checkpoints/best.pt"
        )
        canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        _require(
            canonical.get("configuration") == "location_spread"
            and parent.get("arm") == "base_42k"
            and canonical.get("seed") == parent.get("seed") == seed,
            f"checkpoint metadata changed for seed {seed}",
        )
        left = canonical.get("model_state_dict")
        right = parent.get("model_state_dict")
        _require(isinstance(left, dict) and isinstance(right, dict), "missing model state")
        _require(set(left) == set(right), f"state keys differ for seed {seed}")
        unequal = [name for name in left if not torch.equal(left[name], right[name])]
        _require(not unequal, f"state tensors differ for seed {seed}: {unequal[:3]}")
        left_digest = _state_dict_digest(left)
        right_digest = _state_dict_digest(right)
        _require(left_digest == right_digest, f"state digest differs for seed {seed}")
        receipts[str(seed)] = {
            "canonical_checkpoint": str(canonical_path),
            "canonical_checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256[seed],
            "adjustment_parent_checkpoint": str(parent_path),
            "adjustment_parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256[seed],
            "model_state_sha256": left_digest,
            "tensor_count": len(left),
            "tensor_identity": True,
            "epoch": int(canonical["epoch"]),
            "validation_crps": float(canonical["validation_crps"]),
        }
        del canonical, parent, left, right
    return receipts


def _load_adjustment(
    seed: int, validation_initializations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path = ADJUSTMENT_ROOT / f"models/base_42k/seed_{seed}/validation_adjustments.npz"
    with np.load(path, allow_pickle=False) as archive:
        starts = np.asarray(archive["initializations"], dtype="datetime64[D]")
        delta = np.asarray(archive["delta_log_location"], dtype=np.float32)
        log_spread = np.asarray(archive["log_spread"], dtype=np.float32)
        spread = np.asarray(archive["spread_factor"], dtype=np.float32)
        arm = str(np.asarray(archive["arm"]).item())
        stored_seed = int(np.asarray(archive["seed"]).item())
    expected = (EXPECTED_CASES, LEAD_COUNT, *GRID_SHAPE)
    _require(np.array_equal(starts, validation_initializations), f"seed {seed} starts moved")
    _require(
        delta.shape == log_spread.shape == spread.shape == expected,
        f"seed {seed} adjustment geometry moved",
    )
    reconstructed = np.exp(np.clip(log_spread, -2.0, 2.0)).astype(np.float32)
    _require(np.array_equal(spread, reconstructed), f"seed {seed} spread receipt moved")
    _require(
        arm == "base_42k" and stored_seed == seed,
        f"seed {seed} adjustment metadata moved",
    )
    _require(
        np.isfinite(delta).all() and np.isfinite(spread).all() and np.all(spread > 0.0),
        f"seed {seed} adjustment fields are invalid",
    )
    return delta, spread


def cdf_validity(
    values: np.ndarray, thresholds: np.ndarray, support: np.ndarray, name: str
) -> dict[str, Any]:
    expected = (EXPECTED_CASES, LEAD_COUNT, 4, *GRID_SHAPE)
    _require(values.shape == thresholds.shape == expected, f"{name} CDF geometry moved")
    supported = values[..., support]
    unsupported = values[..., ~support]
    evidence = {
        "shape": list(values.shape),
        "valid_for_physical_thresholds": bool(
            pbc.is_valid_cdf_for_thresholds(values, thresholds, axis=2)
        ),
        "supported_values_finite": bool(np.isfinite(supported).all()),
        "supported_values_bounded": bool(
            np.all((supported >= 0.0) & (supported <= 1.0))
        ),
        "unsupported_values_nan": bool(np.isnan(unsupported).all()),
    }
    _require(all(value for key, value in evidence.items() if key != "shape"), f"invalid {name} CDF")
    return evidence


def reconstruct_neural_pool(
    members: np.ndarray,
    validation_indices: np.ndarray,
    validation_initializations: np.ndarray,
    thresholds: np.ndarray,
    support: np.ndarray,
    *,
    chunk_size: int = CDF_CHUNK_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct each seed CDF and return their exact equal-weight pool."""

    _require(chunk_size >= 1, "CDF chunk size must be positive")
    accumulator = np.zeros(thresholds.shape, dtype=np.float64)
    validity: dict[str, Any] = {}
    selected = np.asarray(validation_indices, dtype=np.int64)
    for seed in SEEDS:
        delta, spread = _load_adjustment(seed, validation_initializations)
        seed_cdf = np.empty(thresholds.shape, dtype=np.float32)
        for start in range(0, EXPECTED_CASES, chunk_size):
            stop = min(start + chunk_size, EXPECTED_CASES)
            raw = np.asarray(members[selected[start:stop]], dtype=np.float32)
            corrected = neural.apply_affine_log_calibration(
                raw, delta[start:stop], spread[start:stop]
            )
            seed_cdf[start:stop] = pbc.ensemble_cdf(
                corrected, thresholds[start:stop], chunk_size=stop - start
            )
            del raw, corrected
        validity[f"neural_seed_{seed}"] = cdf_validity(
            seed_cdf, thresholds, support, f"neural seed {seed}"
        )
        accumulator += seed_cdf.astype(np.float64)
        del seed_cdf, delta, spread
        gc.collect()
    pooled = (accumulator / float(len(SEEDS))).astype(np.float32)
    validity[NEURAL_POOL_METHOD] = cdf_validity(
        pooled, thresholds, support, NEURAL_POOL_METHOD
    )
    return pooled, validity


def assemble_hybrid(neural_pool: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    _require(neural_pool.shape == persistence.shape, "hybrid components differ in shape")
    result = np.asarray(persistence, dtype=np.float32).copy()
    result[:, 0] = np.asarray(neural_pool[:, 0], dtype=np.float32)
    return result


def score_methods(
    methods: Mapping[str, np.ndarray],
    observed: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    initializations: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    for method in METHODS:
        _require(method in methods, f"missing score method: {method}")
        scores = pbc.weighted_spatial_mean(
            pbc.ranked_probability_score(methods[method], observed, thresholds),
            weights,
        )
        _require(
            scores.shape == (EXPECTED_CASES, LEAD_COUNT)
            and np.isfinite(scores).all()
            and np.all((scores >= 0.0) & (scores <= 1.0)),
            f"invalid RPS for {method}",
        )
        for case, initialization in enumerate(initializations):
            for lead in range(LEAD_COUNT):
                records.append(
                    {
                        "split": "validation_reused",
                        "score_contract": SCORE_CONTRACT,
                        "policy": POLICY,
                        "method": method,
                        "initialization": np.datetime_as_string(initialization, unit="D"),
                        "year": int(years[case]),
                        "lead_week": lead + 1,
                        "rps": float(scores[case, lead]),
                    }
                )
    result = pd.DataFrame.from_records(records)
    _require(
        len(result) == len(METHODS) * EXPECTED_CASES * LEAD_COUNT,
        "case score table is incomplete",
    )
    return result


def _bootstrap_draws(initializations: np.ndarray) -> tuple[np.ndarray, ...]:
    """Resample years, then circular blocks, retaining each source year's size."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    years = pd.DatetimeIndex(starts).year.to_numpy()
    counts = pd.Series(years).value_counts().sort_index()
    actual = {int(year): int(count) for year, count in counts.items()}
    _require(actual == EXPECTED_YEAR_COUNTS, f"validation year counts moved: {actual}")
    source_years = tuple(EXPECTED_YEAR_COUNTS)
    groups = {year: np.flatnonzero(years == year) for year in source_years}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_years = rng.choice(
        np.asarray(source_years, dtype=np.int64),
        size=(BOOTSTRAP_DRAWS, len(source_years)),
        replace=True,
    )
    output: list[np.ndarray] = []
    for sampled in sampled_years:
        segments: list[np.ndarray] = []
        for sampled_year in sampled:
            group = groups[int(sampled_year)]
            effective_block = min(BOOTSTRAP_BLOCK_LENGTH, len(group))
            block_count = int(math.ceil(len(group) / effective_block))
            starts_in_year = rng.integers(0, len(group), size=block_count)
            offsets = np.arange(effective_block, dtype=np.int64)
            local = (
                (starts_in_year[:, None] + offsets[None]) % len(group)
            ).reshape(-1)
            segments.append(group[local[: len(group)]])
        output.append(np.concatenate(segments).astype(np.int64, copy=False))
    lengths = {len(draw) for draw in output}
    _require(lengths == {184, 196, 208}, f"bootstrap draw lengths moved: {lengths}")
    return tuple(output)


def summarize_and_compare(
    case_scores: pd.DataFrame, initializations: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = (
        case_scores.groupby("method", as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values("method")
    )
    weekwise = (
        case_scores.groupby(["method", "lead_week"], as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values(["method", "lead_week"])
    )
    yearwise = (
        case_scores.groupby(["method", "year"], as_index=False)
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
        .sort_values(["method", "year"])
    )
    per_initialization = case_scores.groupby(
        ["initialization", "method"], as_index=False
    ).agg(rps=("rps", "mean"))
    labels = [np.datetime_as_string(value, unit="D") for value in initializations]
    pivot = per_initialization.pivot(
        index="initialization", columns="method", values="rps"
    ).reindex(index=labels, columns=METHODS)
    _require(not pivot.isna().any().any(), "paired initialization scores are incomplete")
    draws = _bootstrap_draws(initializations)
    values = pivot.to_numpy(dtype=np.float64)
    indices = {method: index for index, method in enumerate(METHODS)}
    records: list[dict[str, Any]] = []
    for baseline in BASELINES:
        candidate_values = values[:, indices[HYBRID_METHOD]]
        baseline_values = values[:, indices[baseline]]
        effects = np.asarray(
            [
                1.0
                - np.mean(candidate_values[draw], dtype=np.float64)
                / np.mean(baseline_values[draw], dtype=np.float64)
                for draw in draws
            ],
            dtype=np.float64,
        )
        point = 1.0 - candidate_values.mean() / baseline_values.mean()
        records.append(
            {
                "method": HYBRID_METHOD,
                "baseline": baseline,
                "method_rps": float(candidate_values.mean()),
                "baseline_rps": float(baseline_values.mean()),
                "rps_reduction_fraction": float(point),
                "ci_lower_95": float(np.quantile(effects, 0.025)),
                "ci_upper_95": float(np.quantile(effects, 0.975)),
                "bootstrap_probability_improvement": float(np.mean(effects > 0.0)),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "block_length_initializations": BOOTSTRAP_BLOCK_LENGTH,
                "year_clusters": 2,
            }
        )
    return pooled, weekwise, yearwise, pd.DataFrame.from_records(records)


def selection_decision(
    bootstrap: pd.DataFrame, yearwise: pd.DataFrame, *, cdf_checks_pass: bool
) -> dict[str, Any]:
    expected = {(HYBRID_METHOD, baseline) for baseline in BASELINES}
    actual = set(zip(bootstrap.method, bootstrap.baseline, strict=True))
    _require(actual == expected and len(bootstrap) == len(expected), "contrast set moved")
    numerical = bootstrap[
        ["rps_reduction_fraction", "ci_lower_95", "ci_upper_95"]
    ].to_numpy(dtype=np.float64)
    _require(np.isfinite(numerical).all(), "contrast values are non-finite")
    minimum_gain = bool((bootstrap.rps_reduction_fraction >= MINIMUM_REDUCTION).all())
    interval_gate = bool((bootstrap.ci_lower_95 > 0.0).all())
    indexed = yearwise.set_index(["method", "year"])
    year_records: list[dict[str, Any]] = []
    year_gate = True
    for baseline in BASELINES:
        for year in EXPECTED_YEAR_COUNTS:
            candidate = float(indexed.loc[(HYBRID_METHOD, year), "rps"])
            reference = float(indexed.loc[(baseline, year), "rps"])
            reduction = 1.0 - candidate / reference
            lower = candidate < reference
            year_gate = year_gate and lower
            year_records.append(
                {
                    "year": year,
                    "baseline": baseline,
                    "method_rps": candidate,
                    "baseline_rps": reference,
                    "rps_reduction_fraction": reduction,
                    "hybrid_lower": lower,
                }
            )
    passes = bool(minimum_gain and interval_gate and year_gate and cdf_checks_pass)
    return {
        "status": "promoted" if passes else "rejected",
        "candidate_promoted": passes,
        "selected_method": HYBRID_METHOD if passes else PERSISTENCE_METHOD,
        "reason": (
            "all frozen validation gates passed"
            if passes
            else "at least one frozen validation gate failed; no rescue search permitted"
        ),
        "gates": {
            "minimum_0p5_percent_pooled_reduction_vs_both": minimum_gain,
            "lower_in_each_validation_year_vs_both": bool(year_gate),
            "paired_95_percent_intervals_above_zero_vs_both": interval_gate,
            "all_cdf_and_provenance_checks_pass": bool(cdf_checks_pass),
        },
        "rules": {
            "minimum_relative_rps_reduction": MINIMUM_REDUCTION,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_block_length_initializations": BOOTSTRAP_BLOCK_LENGTH,
            "candidate_count": 1,
            "rescue_search_permitted": False,
        },
        "pooled_comparisons": bootstrap.to_dict(orient="records"),
        "yearwise_comparisons": year_records,
        "scientific_status": (
            "reused 2018-2019 validation screen; not an independent superiority test"
        ),
        "independent_superiority_claim_permitted": False,
        "sealed_2025_target_opened": False,
    }


def build_readme(
    pooled: pd.DataFrame, bootstrap: pd.DataFrame, decision: Mapping[str, Any]
) -> str:
    scores = pooled.set_index("method").rps
    lines = [
        "# One-shot deployable neural--PBC CDF hybrid validation",
        "",
        f"Candidate: `{POLICY}`.",
        "",
        f"Decision: **{decision['status']}**. {decision['reason']}",
        "",
        "The W1 forecast is the equal-weight pool of three frozen neural seed CDFs;",
        "W2--W6 are frozen Persistence++. This is one valid probability forecast,",
        "not an average of scores.",
        "",
        "## Reused 2018--2019 validation RPS",
        "",
        "| Method | Pooled RPS |",
        "|---|---:|",
    ]
    for method in METHODS:
        lines.append(f"| {method} | {scores.loc[method]:.6f} |")
    lines.extend(
        [
            "",
            "## Frozen paired gates",
            "",
            "| Baseline | Reduction | 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.baseline} | {100.0 * row.rps_reduction_fraction:.3f}% | "
            f"[{100.0 * row.ci_lower_95:.3f}, {100.0 * row.ci_upper_95:.3f}]% |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is a reused validation screen. Both the candidate idea and parts of",
            "the component selection have prior development exposure. Passing does not",
            "establish an independent NeurIPS superiority result, and it does not mean",
            "the neural adapter alone beats PBC. The sealed 2025 target was not opened.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> Mapping[str, Any]:
    output = Path(output).resolve()
    _require(sha256_file(PLAN_PATH) == PLAN_SHA256, "frozen plan SHA-256 changed")
    _require(not output.exists(), f"fresh output path required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        inputs = validate_frozen_inputs()
        checkpoint_identity = verify_checkpoint_identity()
        cache = neural.load_member_cache(inputs["cache_path"], allow_partial=False)
        splits = neural.make_split_indices(cache.initializations)
        validation = np.asarray(splits.validation, dtype=np.int64)
        validation_initializations = np.asarray(
            cache.initializations[validation], dtype="datetime64[D]"
        )
        _require(len(validation) == EXPECTED_CASES, "validation case count moved")
        _require(
            not np.any(pd.DatetimeIndex(cache.initializations).year == 2025),
            "sealed 2025 initialization entered member cache",
        )
        observations = neural.load_imd_observations(cache)
        _require(
            all("2025" not in Path(path).name for path in observations.source_stores),
            "observation loader opened sealed 2025",
        )
        observation_provenance = pbc_driver.array_bundle_provenance(
            {
                "weekly_truth": observations.weekly_truth,
                "observation_fraction": observations.observation_fraction,
                "weights": observations.weights,
            },
            bundle_name="imd_weekly_truth_fraction_and_scoring_weights",
        )
        _require(
            observation_provenance["sha256"] == OBSERVATION_BUNDLE_SHA256,
            "IMD observation bundle changed",
        )
        daily_dates, daily_values = pbc_driver.load_daily_imd_for_lags(
            cache, observations.source_stores
        )
        lags = pbc.build_daily_issue_time_lags(
            cache.initializations, daily_dates, daily_values, lag_weeks=(1, 2)
        )
        del daily_dates, daily_values
        lag_provenance = pbc_driver.issue_time_lag_provenance(lags)
        _require(lag_provenance["sha256"] == LAG_BUNDLE_SHA256, "lag bundle changed")
        temporal_evidence = pbc_driver.assert_temporal_contract(
            cache.initializations, splits, lags
        )
        _require(np.all(lags.available[validation]), "validation lags are incomplete")

        latitude, longitude, support, weights, support_digest = (
            operational_pbc.load_pbc_static_support(
                PBC_ROOT, inputs["pbc_manifest"]
            )
        )
        _require(support_digest == PBC_SUPPORT_SHA256, "PBC support digest moved")
        _require(
            np.array_equal(latitude, cache.latitude)
            and np.array_equal(longitude, cache.longitude)
            and np.array_equal(weights, observations.weights)
            and np.array_equal(support, observations.weights > 0.0),
            "PBC and neural scoring support differ",
        )
        frozen_pbc = operational_pbc.load_frozen_pbc(
            PBC_ROOT,
            inputs["pbc_manifest"],
            support,
            cache.initializations[splits.train],
        )
        family = frozen_pbc.quintile
        thresholds = pbc.calendar_fields(
            family.model, validation_initializations, LEAD_COUNT
        )
        empirical = pbc.calendar_fields(
            family.model, validation_initializations, LEAD_COUNT, empirical=True
        )
        raw_cdf = pbc.ensemble_cdf(
            cache.members,
            thresholds,
            chunk_size=CDF_CHUNK_SIZE,
            case_indices=validation,
        )
        truth = np.asarray(observations.weekly_truth[validation], dtype=np.float32)
        observed = pbc.observation_cdf(truth, thresholds)
        lag_cdf = pbc.lag_observation_cdf(lags, family.model, validation)
        pbc_methods = operational_pbc.apply_frozen_pbc(
            raw_cdf,
            family,
            validation_initializations,
            thresholds,
            empirical,
            lag_cdf,
        )
        neural_pool, validity = reconstruct_neural_pool(
            cache.members,
            validation,
            validation_initializations,
            thresholds,
            support,
        )
        hybrid = assemble_hybrid(neural_pool, pbc_methods[PERSISTENCE_METHOD])
        methods = {
            RAW_METHOD: raw_cdf,
            NEURAL_POOL_METHOD: neural_pool,
            PERSISTENCE_METHOD: pbc_methods[PERSISTENCE_METHOD],
            COMBINED_METHOD: pbc_methods[COMBINED_METHOD],
            HYBRID_METHOD: hybrid,
        }
        validity["observed_cdf"] = cdf_validity(
            observed, thresholds, support, "observed"
        )
        for method, values in methods.items():
            validity[method] = cdf_validity(values, thresholds, support, method)
        cdf_checks_pass = all(
            all(value for key, value in evidence.items() if key != "shape")
            for evidence in validity.values()
        )
        cdf_content_receipts = {
            name: pbc_driver.array_bundle_provenance(
                {name: values}, bundle_name=f"{name}_cdf_content"
            )
            for name, values in {
                "thresholds": thresholds,
                "observed_cdf": observed,
                **methods,
            }.items()
        }
        case_scores = score_methods(
            methods,
            observed,
            thresholds,
            weights,
            validation_initializations,
        )
        pooled, weekwise, yearwise, bootstrap = summarize_and_compare(
            case_scores, validation_initializations
        )
        decision = selection_decision(
            bootstrap, yearwise, cdf_checks_pass=cdf_checks_pass
        )

        metrics = stage / "metrics"
        evaluation = stage / "evaluation"
        metrics.mkdir()
        evaluation.mkdir()
        case_scores.to_csv(metrics / "validation_case_scores.csv", index=False)
        pooled.to_csv(metrics / "validation_pooled_rps.csv", index=False)
        weekwise.to_csv(metrics / "validation_weekwise_rps.csv", index=False)
        yearwise.to_csv(metrics / "validation_yearwise_rps.csv", index=False)
        bootstrap.to_csv(metrics / "paired_bootstrap.csv", index=False)
        _write_json(evaluation / "cdf_validity.json", validity)
        _write_json(
            evaluation / "cdf_array_content_receipts.json", cdf_content_receipts
        )
        _write_json(evaluation / "observation_bundle_provenance.json", observation_provenance)
        _write_json(evaluation / "lag_bundle_provenance.json", lag_provenance)
        _write_json(evaluation / "checkpoint_tensor_identity.json", checkpoint_identity)
        _write_json(stage / "selection.json", decision)
        selection_sha256 = sha256_file(stage / "selection.json")
        (stage / "README.md").write_text(
            build_readme(pooled, bootstrap, decision), encoding="utf-8"
        )
        source_paths = {
            "plan/LEAD_GATED_CDF_POOL_VALIDATION_20260824.md": PLAN_PATH,
            "src/fuxi_lead_gated_cdf_hybrid_validation.py": SOURCE_PATH,
            "src/fuxi_allseason_ensemble_calibration.py": Path(neural.__file__).resolve(),
            "src/fuxi_allseason_operational_categorical_comparison.py": Path(
                operational_pbc.__file__
            ).resolve(),
            "src/fuxi_allseason_pbc_baseline.py": Path(pbc_driver.__file__).resolve(),
            "src/fuxi_pbc_core.py": Path(pbc.__file__).resolve(),
            "src/fuxi_allseason_member_cache.py": (
                PROJECT_ROOT / "src/fuxi_allseason_member_cache.py"
            ),
            "src/fuxi_ensemble_calibration_core.py": (
                PROJECT_ROOT / "src/fuxi_ensemble_calibration_core.py"
            ),
            "src/project_paths.py": PROJECT_ROOT / "src/project_paths.py",
        }
        if TEST_PATH.is_file():
            source_paths[
                "tests/test_fuxi_lead_gated_cdf_hybrid_validation.py"
            ] = TEST_PATH
        source_snapshot: dict[str, str] = {}
        for relative, source in source_paths.items():
            _require(source.is_file(), f"source snapshot input is missing: {source}")
            destination = stage / "code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = sha256_file(destination)
            _require(digest == sha256_file(source), f"source snapshot copy moved: {relative}")
            source_snapshot[relative] = digest
        artifacts = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in sorted(item for item in stage.rglob("*") if item.is_file())
        }
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": utc_now(),
            "mode": "full_validation_only",
            "output_path": str(output),
            "policy": {
                "name": POLICY,
                "W1": "equal-weight pool of frozen neural CDF seeds 42,43,44",
                "W2_W6": "frozen projected Persistence++",
                "seed_weights": {str(seed): 1.0 / 3.0 for seed in SEEDS},
                "candidate_count": 1,
                "trained_or_refit": False,
            },
            "contract": {
                "selection_data": "purged 2018-2019 validation only",
                "validation_reused": True,
                "score": SCORE_CONTRACT,
                "sealed_2025_target_opened": False,
                "development_2020_2024_scored": False,
                "observations_2020_2022_opened_only_to_reproduce_parent_bundle": True,
                "member_cache_rows_outside_validation_opened_for_integrity_verification": True,
                "forecast_indices_outside_validation_used_or_scored": False,
                "independent_superiority_claim_permitted": False,
            },
            "selection_path": "selection.json",
            "selection_sha256": selection_sha256,
            "selection": decision,
            "input_manifests": {
                "neural": {
                    "path": str(NEURAL_MANIFEST_PATH),
                    "sha256": NEURAL_MANIFEST_SHA256,
                },
                "pbc": {
                    "path": str(PBC_MANIFEST_PATH),
                    "sha256": PBC_MANIFEST_SHA256,
                },
                "validation_adjustments": {
                    "path": str(ADJUSTMENT_MANIFEST_PATH),
                    "sha256": ADJUSTMENT_MANIFEST_SHA256,
                },
            },
            "cache": {
                "path": str(inputs["cache_path"]),
                "declared_data_sha256": CACHE_DATA_SHA256,
                "canonical_loader_verify_true": True,
            },
            "observation_bundle": observation_provenance,
            "lag_bundle": lag_provenance,
            "temporal_evidence": temporal_evidence,
            "checkpoint_tensor_identity": checkpoint_identity,
            "cdf_validity": validity,
            "cdf_array_content_receipts": cdf_content_receipts,
            "software": {
                "python_executable": sys.executable,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "xarray": neural.xr.__version__,
                "zarr": getattr(zarr, "__version__", "unknown"),
                "matplotlib": neural.matplotlib.__version__,
                "neural_checkpoint_inference_used": False,
                "forecast_scoring_backend": "NumPy CPU",
            },
            "source_snapshot_sha256": source_snapshot,
            "artifact_sha256": artifacts,
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args.output)
    print(f"complete: {manifest['output_path']}")
    print(f"selection: {manifest['selection']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
