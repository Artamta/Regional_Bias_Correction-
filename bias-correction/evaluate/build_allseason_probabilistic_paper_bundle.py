#!/usr/bin/env python3
"""Build a receipt-gated, venue-neutral all-season paper evidence bundle.

This program is deliberately a *derived reporting* step.  It never modifies a
source run, never discovers year stores, and never reads forecasts or targets.
Every source is supplied explicitly, its manifest digest is pinned, every
manifest-declared artifact is re-hashed, and every available Slurm gate receipt
is checked before a CSV is read.  The completed bundle is installed atomically
under ``presentation/deliverables``.

The two future inputs (the shared categorical comparison and the operational-
era retrospective) must be accompanied by explicit expected manifest hashes.
That makes the command usable as soon as those jobs finish without weakening
the evidence firewall in advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


BUNDLE_EXPERIMENT = "fuxi_allseason_probabilistic_paper_bundle_v1"
BUNDLE_CONTRACT_VERSION = "allseason_probabilistic_paper_bundle_contract_v1"

KNOWN_MANIFEST_SHA256 = {
    "neural": "94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91",
    "capacity": "2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c",
    "capacity_development": "883aa0b2955b658b9fd164ac839103a4e80a33da2cdbd5a22b6b8fbf4515ec60",
    "hybrid_loss": "a3e77e4cf4e6485a756d99b68fec102fece37ef096725e496fe4a27819828f5e",
    "pbc": "c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6",
}

EXPERIMENT_BY_ROLE = {
    "neural": "fuxi_allseason_ensemble_calibration_v1",
    "capacity": "fuxi_allseason_capacity_ablation_v1",
    "capacity_development": "fuxi_allseason_capacity_development_evaluation_v1",
    "hybrid_loss": "fuxi_allseason_hybrid_loss_ablation_v1",
    "pbc": "fuxi_allseason_pbc_baseline_v2",
    "categorical": "fuxi_allseason_categorical_comparison_v2",
    "operational": "fuxi_allseason_operational_era_audit_v1",
}

EXTERNAL_GATE_RECEIPT_ROLES = {
    "capacity_development",
    "pbc",
    "categorical",
    "operational",
}

RECEIPT_AUDIT_VERSION = {
    "capacity_development": "capacity_dev_postrun_v1",
    "pbc": "pbc_postrun_v2",
    "categorical": "categorical_comparison_postrun_v2",
    "operational": "operational_era_postrun_v3",
}

PRIMARY_CATEGORICAL_CONTRACT = "normalized_informative_positive_cut_v2"
PRIMARY_CATEGORICAL_METRIC = (
    "primary_normalized_informative_positive_cut_rps"
)

ROLE_ORDER = (
    "neural",
    "capacity",
    "capacity_development",
    "hybrid_loss",
    "pbc",
    "categorical",
    "operational",
)

ROLE_LABELS = {
    "neural": "Frozen neural development evaluation",
    "capacity": "Validation-only capacity selection",
    "capacity_development": "Post-selection development evaluation",
    "hybrid_loss": "Post-hoc loss ablation",
    "pbc": "Static PBC development baseline",
    "categorical": "Shared categorical development comparison",
    "operational": "Later operational-era retrospective audit",
}

YEAR_TOKEN = re.compile(r"(?:^|[^0-9])2025(?:[^0-9]|$)")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
YEAR_STORE_SUFFIXES = {
    ".zarr",
    ".nc",
    ".nc4",
    ".grib",
    ".grib2",
    ".npy",
    ".npz",
    ".parquet",
    ".h5",
    ".hdf5",
}


class BundleContractError(RuntimeError):
    """Raised when source evidence or derived output violates the contract."""


@dataclass(frozen=True)
class VerifiedSource:
    """A fully checked immutable source run."""

    role: str
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    artifact_sha256: Mapping[str, str]
    gate_receipt: Mapping[str, Any] | None
    gate_receipt_sha256: str | None
    receipt_policy: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleContractError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.temporary")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path_mentions_forbidden_year(path: Path) -> bool:
    return any(YEAR_TOKEN.search(part) is not None for part in path.parts)


def _store_path_targets_forbidden_year(path: Path) -> bool:
    """Identify a 2025 store leaf without rejecting an archive range label.

    The operational archive parent is legitimately named with its coverage
    range (``...2020_2025_ens50``), while the paths actually opened terminate
    in ``2022.zarr``, ``2023.zarr``, or ``2024.zarr``.  Treating every year
    token in every parent component as an opened year creates a false positive.
    We instead reject an exact year directory, a date-prefixed component, or a
    recognized store/file component whose own name identifies 2025.  The
    manifest's explicit opened-store whitelist and false 2025-open booleans
    remain mandatory independent gates.
    """

    for part in path.parts:
        lowered = part.lower()
        if lowered == "2025" or lowered.startswith("2025-"):
            return True
        if Path(lowered).suffix in YEAR_STORE_SUFFIXES and YEAR_TOKEN.search(
            lowered
        ):
            return True
    return False


def _normalise_manifest_path(value: Path) -> Path:
    unresolved = Path(value)
    _require(
        not _path_mentions_forbidden_year(unresolved),
        f"refusing a source path containing the sealed year: {unresolved}",
    )
    candidate = unresolved / "manifest.json" if unresolved.is_dir() else unresolved
    resolved = candidate.resolve(strict=True)
    _require(
        not _path_mentions_forbidden_year(resolved),
        f"resolved source path contains the sealed year: {resolved}",
    )
    _require(
        resolved.is_file() and resolved.name == "manifest.json",
        f"source must be a run directory or manifest.json: {value}",
    )
    run_name = resolved.parent.name.lower()
    _require(
        "smoke" not in run_name and "incomplete" not in run_name,
        f"smoke/incomplete source paths cannot become paper evidence: {resolved}",
    )
    return resolved


def _safe_artifact(root: Path, relative: str) -> Path:
    item = Path(relative)
    _require(
        bool(relative)
        and not item.is_absolute()
        and ".." not in item.parts
        and not _path_mentions_forbidden_year(item),
        f"unsafe or sealed-year artifact path: {relative!r}",
    )
    candidate = root / item
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise BundleContractError(
            f"artifact escapes its immutable run: {relative!r}"
        ) from error
    _require(not candidate.is_symlink(), f"artifact may not be a symlink: {relative}")
    return candidate


def _iter_year_fields(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_prefix = (*prefix, name)
            if "year" in name.lower():
                found.append((".".join(child_prefix), child))
            found.extend(_iter_year_fields(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_iter_year_fields(child, (*prefix, str(index))))
    return found


def _contains_integer_2025(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 2025
    if isinstance(value, float):
        return value == 2025.0
    if isinstance(value, str):
        return value.strip() == "2025" or value.startswith("2025-")
    if isinstance(value, Mapping):
        return any(
            _contains_integer_2025(key) or _contains_integer_2025(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_integer_2025(child) for child in value)
    return False


def _verify_sealed_year_contract(role: str, manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("contract", {})
    _require(isinstance(contract, Mapping), f"{role}: contract is absent")
    _require(
        contract.get("sealed_2025_target_opened") is False,
        f"{role}: manifest does not prove that the sealed target stayed unopened",
    )
    if "final_2025_store_opened" in contract:
        _require(
            contract.get("final_2025_store_opened") is False,
            f"{role}: final-year forecast store was opened",
        )

    for field, value in _iter_year_fields(manifest):
        field_lower = field.lower()
        if "sealed_unopened_years" in field_lower:
            _require(
                _contains_integer_2025(value),
                f"{role}: sealed-year declaration does not contain 2025",
            )
            continue
        if field_lower.endswith("2025_target_opened") or field_lower.endswith(
            "2025_store_opened"
        ):
            continue
        _require(
            not _contains_integer_2025(value),
            f"{role}: non-seal year field references 2025: {field}",
        )

    path_key_fragments = (
        "opened_store",
        "forecast_store",
        "verification_store",
        "observation_store",
        "target_store",
        "store_paths",
    )

    def inspect_paths(value: Any, prefix: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                child_prefix = (*prefix, str(key))
                if any(fragment in key_text for fragment in path_key_fragments):
                    values = child if isinstance(child, list) else [child]
                    for path_value in values:
                        if isinstance(path_value, str):
                            _require(
                                not _store_path_targets_forbidden_year(
                                    Path(path_value)
                                ),
                                f"{role}: opened/source store path references 2025 at "
                                f"{'.'.join(child_prefix)}",
                            )
                inspect_paths(child, child_prefix)
        elif isinstance(value, list):
            for child in value:
                inspect_paths(child, prefix)

    inspect_paths(manifest)


def _verify_gate_receipt(
    role: str,
    root: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    receipt_path = root / "slurm_gate_receipt.json"
    _require(receipt_path.is_file(), f"{role}: missing Slurm gate receipt")
    _require(not receipt_path.is_symlink(), f"{role}: gate receipt is a symlink")
    receipt = _read_json(receipt_path)
    expected = {
        "experiment": EXPERIMENT_BY_ROLE[role],
        "gate_status": "passed",
        "post_run_audit_version": RECEIPT_AUDIT_VERSION[role],
        "manifest_sha256": manifest_sha256,
    }
    # The categorical post-run V2 receipt predates the explicit receipt-level
    # ``mode`` field; its bound manifest already proves full/non-smoke mode.
    if role != "categorical":
        expected["mode"] = "full"
    for key, wanted in expected.items():
        _require(
            receipt.get(key) == wanted,
            f"{role}: Slurm receipt field {key!r} differs: "
            f"{receipt.get(key)!r} != {wanted!r}",
        )
    for key in ("job_id", "node", "partition", "completed_utc"):
        _require(
            isinstance(receipt.get(key), str) and bool(receipt[key].strip()),
            f"{role}: Slurm receipt lacks {key}",
        )
    if role in {"pbc", "categorical"}:
        _require(
            receipt.get("scoring_contract_version")
            == PRIMARY_CATEGORICAL_CONTRACT,
            f"{role}: Slurm receipt scoring contract differs",
        )
    if role == "operational":
        semantic = root / "evaluation/postflight_semantic_audit.json"
        _require(semantic.is_file(), "operational: missing postflight semantic audit")
        _require(
            receipt.get("postflight_semantic_audit_sha256")
            == sha256_file(semantic),
            "operational: Slurm receipt does not bind the semantic audit",
        )
    return receipt, sha256_file(receipt_path)


def _verify_embedded_slurm(role: str, manifest: Mapping[str, Any]) -> str:
    slurm = manifest.get("slurm")
    _require(isinstance(slurm, Mapping), f"{role}: embedded Slurm record is absent")
    for key in ("job_id", "node", "partition"):
        value = slurm.get(key)
        _require(
            value is not None and bool(str(value).strip()),
            f"{role}: embedded Slurm record lacks {key}",
        )
    return "embedded_manifest_slurm_record"


def verify_source(
    role: str,
    source: Path,
    *,
    expected_manifest_sha256: str,
) -> VerifiedSource:
    """Verify a full source run without opening forecast or target data."""

    _require(role in ROLE_ORDER, f"unknown source role: {role}")
    _require(
        HEX64.fullmatch(expected_manifest_sha256) is not None,
        f"{role}: expected manifest SHA-256 must be 64 lowercase hex characters",
    )
    manifest_path = _normalise_manifest_path(source)
    root = manifest_path.parent
    manifest_sha = sha256_file(manifest_path)
    _require(
        manifest_sha == expected_manifest_sha256,
        f"{role}: manifest SHA-256 differs: {manifest_sha}",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("experiment") == EXPERIMENT_BY_ROLE[role],
        f"{role}: unexpected experiment {manifest.get('experiment')!r}",
    )
    _require(manifest.get("status") == "complete", f"{role}: run is incomplete")
    _require(manifest.get("mode") == "full", f"{role}: run is not full mode")
    _require(manifest.get("smoke") is False, f"{role}: smoke evidence is forbidden")
    _require(
        isinstance(manifest.get("scientific_status"), str)
        and bool(manifest["scientific_status"].strip()),
        f"{role}: scientific-status label is absent",
    )
    if role == "operational":
        _require(
            manifest.get("canonical") is True
            and manifest.get("scientific_eligible") is True,
            "operational: full audit is not marked canonical/scientifically eligible",
        )
    if role in {"pbc", "categorical"}:
        _require(
            manifest.get("scoring_contract_version")
            == PRIMARY_CATEGORICAL_CONTRACT,
            f"{role}: stale categorical scoring contract",
        )
    _verify_sealed_year_contract(role, manifest)

    artifacts = manifest.get("artifact_sha256")
    _require(
        isinstance(artifacts, Mapping) and bool(artifacts),
        f"{role}: artifact SHA-256 inventory is empty",
    )
    checked: dict[str, str] = {}
    for relative, expected in sorted(artifacts.items()):
        _require(
            isinstance(relative, str)
            and isinstance(expected, str)
            and HEX64.fullmatch(expected) is not None,
            f"{role}: malformed artifact receipt for {relative!r}",
        )
        artifact = _safe_artifact(root, relative)
        _require(artifact.is_file(), f"{role}: missing artifact {relative}")
        actual = sha256_file(artifact)
        _require(actual == expected, f"{role}: artifact checksum differs: {relative}")
        checked[relative] = actual

    receipt: Mapping[str, Any] | None = None
    receipt_sha: str | None = None
    if role in EXTERNAL_GATE_RECEIPT_ROLES:
        receipt, receipt_sha = _verify_gate_receipt(
            role, root, manifest, manifest_sha
        )
        receipt_policy = "external_slurm_gate_receipt"
    elif role in {"neural", "hybrid_loss"}:
        receipt_policy = _verify_embedded_slurm(role, manifest)
    else:
        # The validation-only capacity workflow predates external gate receipts.
        # Its exact manifest is instead bound by the later gate-receipted
        # post-selection evaluator; verify_source_relations enforces that link.
        _require(
            role == "capacity" and manifest.get("selection") is not None,
            f"{role}: no accepted receipt policy",
        )
        receipt_policy = "canonical_manifest_transitively_bound_by_capacity_development_gate"

    return VerifiedSource(
        role=role,
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        artifact_sha256=checked,
        gate_receipt=receipt,
        gate_receipt_sha256=receipt_sha,
        receipt_policy=receipt_policy,
    )


def verify_source_relations(sources: Mapping[str, VerifiedSource]) -> None:
    _require(set(sources) == set(ROLE_ORDER), "all seven source roles are required")
    neural = sources["neural"]
    capacity = sources["capacity"]
    development = sources["capacity_development"]
    pbc = sources["pbc"]
    categorical = sources["categorical"]
    operational = sources["operational"]

    dev_capacity = development.manifest.get("capacity_receipt", {})
    _require(
        isinstance(dev_capacity, Mapping)
        and dev_capacity.get("sha256") == capacity.manifest_sha256
        and dev_capacity.get("artifact_inventory_verified") is True,
        "capacity-development run does not bind the verified capacity screen",
    )
    cat_inputs = categorical.manifest.get("input_receipts", {})
    _require(isinstance(cat_inputs, Mapping), "categorical input receipts are absent")
    _require(
        cat_inputs.get("neural_manifest", {}).get("sha256")
        == neural.manifest_sha256,
        "categorical run does not bind the frozen neural manifest",
    )
    _require(
        cat_inputs.get("pbc_manifest", {}).get("sha256") == pbc.manifest_sha256,
        "categorical run does not bind the PBC V2 manifest",
    )
    _require(
        categorical.gate_receipt is not None
        and categorical.gate_receipt.get("neural_manifest_sha256")
        == neural.manifest_sha256
        and categorical.gate_receipt.get("pbc_manifest_sha256")
        == pbc.manifest_sha256,
        "categorical Slurm receipt does not bind both immutable inputs",
    )
    op_capacity = operational.manifest.get("capacity_receipt", {})
    _require(
        isinstance(op_capacity, Mapping)
        and op_capacity.get("sha256") == capacity.manifest_sha256
        and op_capacity.get("artifact_inventory_verified") is True,
        "operational-era audit does not bind the verified capacity selection",
    )


def _verified_csv(
    source: VerifiedSource,
    relative: str,
    *,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    _require(
        relative in source.artifact_sha256,
        f"{source.role}: unreceipted CSV requested: {relative}",
    )
    path = _safe_artifact(source.root, relative)
    frame = pd.read_csv(path)
    _require(not frame.empty, f"{source.role}: empty CSV: {relative}")
    _require(
        not any(str(column).startswith("Unnamed:") for column in frame.columns),
        f"{source.role}: unnamed CSV column: {relative}",
    )
    missing = set(required_columns) - set(frame.columns)
    _require(
        not missing,
        f"{source.role}: {relative} lacks columns {sorted(missing)}",
    )
    for column in frame.columns:
        if any(token in str(column).lower() for token in ("date", "init", "time")):
            values = frame[column].dropna().astype(str)
            _require(
                not values.str.startswith("2025-").any(),
                f"{source.role}: {relative} contains sealed-year rows in {column}",
            )
    return frame


def _finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy()
    _require(np.isfinite(values).all(), f"{label}: non-finite numeric value")


def build_continuous_skill_table(
    sources: Mapping[str, VerifiedSource],
) -> pd.DataFrame:
    development = _verified_csv(
        sources["capacity_development"],
        "metrics/paired_block_bootstrap.csv",
        required_columns=(
            "comparison_scope",
            "optimization_seed",
            "method",
            "baseline",
            "lead_scope",
            "metric",
            "effect",
            "ci_lower",
            "ci_upper",
            "paired_initializations",
            "n_resamples",
            "block_length_initializations",
            "bootstrap",
        ),
    )
    chosen_dev = development.loc[
        development.comparison_scope.eq("mean_seed_scores_vs_raw")
        & development.optimization_seed.eq("mean_of_seed_scores")
        & development.method.eq("base_42k")
        & development.baseline.eq("raw_fuxi")
        & development.metric.eq("crps")
    ].copy()
    _require(
        set(chosen_dev.lead_scope) == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"}
        and len(chosen_dev) == 7,
        "development CRPS bootstrap does not contain exactly W1-W6 and W1-W6 lead rows",
    )

    operational = _verified_csv(
        sources["operational"],
        "metrics/paired_two_stage_bootstrap.csv",
        required_columns=(
            "comparison_scope",
            "optimization_seed",
            "method",
            "baseline",
            "lead_scope",
            "metric",
            "effect",
            "ci_lower_2p5",
            "ci_upper_97p5",
            "bootstrap_probability_effect_positive",
            "paired_initializations",
            "source_year_clusters",
            "n_resamples",
            "block_length_initializations",
            "bootstrap",
        ),
    )
    chosen_op = operational.loc[
        operational.comparison_scope.eq("paired_seed_score_average_vs_raw")
        & operational.optimization_seed.eq("mean_of_seed_scores")
        & operational.method.eq("base_42k")
        & operational.baseline.eq("raw_fuxi")
        & operational.metric.eq("crps")
    ].copy()
    _require(
        set(chosen_op.lead_scope) == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"}
        and len(chosen_op) == 7,
        "operational-era CRPS bootstrap does not contain exactly W1-W6 and W1-W6 lead rows",
    )

    rows: list[dict[str, Any]] = []
    for frame, period, source_role, lower, upper, probability, year_count in (
        (
            chosen_dev,
            "development_2020_2021_reused",
            "capacity_development",
            "ci_lower",
            "ci_upper",
            None,
            2,
        ),
        (
            chosen_op,
            "operational_era_2022_2024_retrospective",
            "operational",
            "ci_lower_2p5",
            "ci_upper_97p5",
            "bootstrap_probability_effect_positive",
            3,
        ),
    ):
        for row in frame.itertuples(index=False):
            lead_week = 0 if row.lead_scope == "W1-W6" else int(str(row.lead_scope)[1:])
            record = {
                "evidence_period": period,
                "source_role": source_role,
                "scientific_status": sources[source_role].manifest["scientific_status"],
                "method": row.method,
                "baseline": row.baseline,
                "lead_scope": row.lead_scope,
                "lead_week": lead_week,
                "crps_skill_pct_vs_raw": float(row.effect),
                "ci_lower_95_pct": float(getattr(row, lower)),
                "ci_upper_95_pct": float(getattr(row, upper)),
                "bootstrap_probability_improvement": (
                    np.nan if probability is None else float(getattr(row, probability))
                ),
                "paired_initializations": int(row.paired_initializations),
                "source_year_clusters": year_count,
                "bootstrap_resamples": int(row.n_resamples),
                "block_length_initializations": int(row.block_length_initializations),
                "bootstrap_scheme": str(row.bootstrap),
            }
            rows.append(record)
    result = pd.DataFrame(rows).sort_values(["evidence_period", "lead_week"])
    _finite(
        result,
        ("crps_skill_pct_vs_raw", "ci_lower_95_pct", "ci_upper_95_pct"),
        "continuous skill table",
    )
    _require(
        (result.ci_lower_95_pct <= result.ci_upper_95_pct).all(),
        "continuous skill confidence intervals are unordered",
    )
    return result.reset_index(drop=True)


def build_categorical_table(
    sources: Mapping[str, VerifiedSource],
) -> pd.DataFrame:
    categorical = _verified_csv(
        sources["categorical"],
        "metrics/pooled_metrics.csv",
        required_columns=(
            "method",
            "method_label",
            "score_contract",
            "rps",
            "training_empirical_climatology_rps",
            "rpss_vs_training_empirical_climatology",
            "paper_nominal_cut_q80_positive_rps",
            "paper_fixed_nominal_climatology_rps",
            "paper_nominal_cut_rpss_vs_fixed_nominal_climatology",
            "n_initializations",
        ),
    ).copy()
    expected_methods = set(sources["categorical"].manifest.get("methods", []))
    _require(
        len(categorical) == len(expected_methods) == 7
        and set(categorical.method) == expected_methods
        and not categorical.method.duplicated().any(),
        "categorical pooled table does not contain exactly the seven shared methods",
    )
    _require(
        set(categorical.score_contract) == {PRIMARY_CATEGORICAL_CONTRACT},
        "categorical pooled table mixes scoring contracts",
    )
    _finite(
        categorical,
        (
            "rps",
            "training_empirical_climatology_rps",
            "rpss_vs_training_empirical_climatology",
            "paper_nominal_cut_q80_positive_rps",
            "paper_fixed_nominal_climatology_rps",
            "paper_nominal_cut_rpss_vs_fixed_nominal_climatology",
        ),
        "categorical pooled table",
    )
    _require(
        np.allclose(
            categorical.rpss_vs_training_empirical_climatology,
            1.0
            - categorical.rps / categorical.training_empirical_climatology_rps,
            rtol=1.0e-10,
            atol=1.0e-10,
        ),
        "categorical empirical-RPSS identity differs",
    )

    bootstrap = _verified_csv(
        sources["categorical"],
        "metrics/paired_block_bootstrap.csv",
        required_columns=(
            "metric",
            "lead_scope",
            "method",
            "baseline",
            "rps_reduction_fraction",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_probability_improvement",
        ),
    )
    versus_raw = bootstrap.loc[
        bootstrap.metric.eq(PRIMARY_CATEGORICAL_METRIC)
        & bootstrap.lead_scope.eq("W1-W6")
        & bootstrap.baseline.eq("raw_fuxi_categorical")
    ].copy()

    # Debias++ and Persistence++ predate the shared comparator's comparison
    # pair list.  Their same-contract paired CIs are taken from the independently
    # gate-receipted PBC V2 run; all other methods use the shared comparator.
    pbc_bootstrap = _verified_csv(
        sources["pbc"],
        "metrics/paired_block_bootstrap.csv",
        required_columns=(
            "score_contract",
            "lead_scope",
            "method",
            "baseline",
            "rps_reduction_fraction",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_probability_improvement",
        ),
    )
    pbc_raw = pbc_bootstrap.loc[
        pbc_bootstrap.score_contract.eq(PRIMARY_CATEGORICAL_CONTRACT)
        & pbc_bootstrap.lead_scope.eq("W1-W6")
        & pbc_bootstrap.baseline.eq("raw_fuxi_categorical")
        & pbc_bootstrap.method.isin(("debias_plus_plus", "persistence_plus_plus"))
    ].copy()
    _require(
        set(pbc_raw.method) == {"debias_plus_plus", "persistence_plus_plus"},
        "PBC component CIs against raw are incomplete",
    )

    ci_by_method: dict[str, dict[str, Any]] = {}
    for row in versus_raw.itertuples(index=False):
        ci_by_method[str(row.method)] = {
            "rps_reduction_pct_vs_raw": 100.0 * float(row.rps_reduction_fraction),
            "ci_lower_95_pct": 100.0 * float(row.ci_lower_95),
            "ci_upper_95_pct": 100.0 * float(row.ci_upper_95),
            "bootstrap_probability_improvement": float(
                row.bootstrap_probability_improvement
            ),
            "ci_source_role": "categorical",
        }
    for row in pbc_raw.itertuples(index=False):
        ci_by_method[str(row.method)] = {
            "rps_reduction_pct_vs_raw": 100.0 * float(row.rps_reduction_fraction),
            "ci_lower_95_pct": 100.0 * float(row.ci_lower_95),
            "ci_upper_95_pct": 100.0 * float(row.ci_upper_95),
            "bootstrap_probability_improvement": float(
                row.bootstrap_probability_improvement
            ),
            "ci_source_role": "pbc",
        }

    records: list[dict[str, Any]] = []
    for row in categorical.itertuples(index=False):
        method = str(row.method)
        uncertainty = ci_by_method.get(method)
        if method == "raw_fuxi_categorical":
            uncertainty = {
                "rps_reduction_pct_vs_raw": 0.0,
                "ci_lower_95_pct": np.nan,
                "ci_upper_95_pct": np.nan,
                "bootstrap_probability_improvement": np.nan,
                "ci_source_role": "not_applicable_reference",
            }
        _require(
            uncertainty is not None,
            f"categorical method has no paired raw comparison: {method}",
        )
        records.append(
            {
                "method": method,
                "method_label": str(row.method_label),
                "scientific_status": sources["categorical"].manifest[
                    "scientific_status"
                ],
                "score_contract": str(row.score_contract),
                "primary_rps": float(row.rps),
                "training_empirical_climatology_rps": float(
                    row.training_empirical_climatology_rps
                ),
                "primary_rpss_vs_training_empirical_climatology": float(
                    row.rpss_vs_training_empirical_climatology
                ),
                "rps_reduction_pct_vs_raw": uncertainty[
                    "rps_reduction_pct_vs_raw"
                ],
                "ci_lower_95_pct": uncertainty["ci_lower_95_pct"],
                "ci_upper_95_pct": uncertainty["ci_upper_95_pct"],
                "bootstrap_probability_improvement": uncertainty[
                    "bootstrap_probability_improvement"
                ],
                "ci_source_role": uncertainty["ci_source_role"],
                "secondary_paper_formula_rps": float(
                    row.paper_nominal_cut_q80_positive_rps
                ),
                "secondary_paper_formula_reference_rps": float(
                    row.paper_fixed_nominal_climatology_rps
                ),
                "secondary_paper_formula_rpss": float(
                    row.paper_nominal_cut_rpss_vs_fixed_nominal_climatology
                ),
                "initializations": int(row.n_initializations),
                "secondary_status": (
                    "formula-aligned equality-aware sensitivity; not a rolling-protocol reproduction"
                ),
            }
        )
    result = pd.DataFrame(records).sort_values("primary_rps").reset_index(drop=True)
    return result


def build_categorical_lead_tables(
    sources: Mapping[str, VerifiedSource],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build shared-support lead scores and the guarded neural/PBC contrast."""

    source = sources["categorical"]
    weekwise = _verified_csv(
        source,
        "metrics/weekwise_metrics.csv",
        required_columns=(
            "lead_week",
            "method",
            "method_label",
            "score_contract",
            "rps",
            "training_empirical_climatology_rps",
            "rpss_vs_training_empirical_climatology",
            "n_initializations",
        ),
    ).copy()
    methods = tuple(source.manifest.get("methods", ()))
    expected_keys = {
        (lead, method) for lead in range(1, 7) for method in methods
    }
    actual_keys = set(
        zip(
            pd.to_numeric(weekwise.lead_week, errors="raise").astype(int),
            weekwise.method.astype(str),
            strict=True,
        )
    )
    _require(
        len(methods) == 7
        and len(weekwise) == 42
        and actual_keys == expected_keys
        and not weekwise.duplicated(["lead_week", "method"]).any(),
        "categorical weekwise table is not the complete 7-method x 6-lead comparison",
    )
    _require(
        set(weekwise.score_contract) == {PRIMARY_CATEGORICAL_CONTRACT},
        "categorical weekwise table mixes scoring contracts",
    )
    _finite(
        weekwise,
        ("lead_week", "rps", "training_empirical_climatology_rps", "rpss_vs_training_empirical_climatology"),
        "categorical weekwise table",
    )
    _require(
        np.allclose(
            weekwise.rpss_vs_training_empirical_climatology,
            1.0 - weekwise.rps / weekwise.training_empirical_climatology_rps,
            rtol=1.0e-10,
            atol=1.0e-10,
        ),
        "categorical weekwise empirical-RPSS identity differs",
    )

    comparator_bootstrap = _verified_csv(
        source,
        "metrics/paired_block_bootstrap.csv",
        required_columns=(
            "metric",
            "lead_scope",
            "lead_week",
            "method",
            "baseline",
            "method_score",
            "baseline_score",
            "rps_reduction_fraction",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_probability_improvement",
            "bootstrap_samples",
            "block_length_initializations",
            "bootstrap_scheme",
            "test_year_count",
        ),
    )
    pbc_bootstrap = _verified_csv(
        sources["pbc"],
        "metrics/paired_block_bootstrap.csv",
        required_columns=(
            "score_contract",
            "lead_scope",
            "lead_week",
            "method",
            "baseline",
            "rps_reduction_fraction",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_probability_improvement",
            "bootstrap_samples",
            "block_length_initializations",
            "bootstrap_scheme",
            "test_year_count",
        ),
    )

    comparator_raw = comparator_bootstrap.loc[
        comparator_bootstrap.metric.eq(PRIMARY_CATEGORICAL_METRIC)
        & comparator_bootstrap.lead_scope.isin([f"W{lead}" for lead in range(1, 7)])
        & comparator_bootstrap.baseline.eq("raw_fuxi_categorical")
    ].copy()
    pbc_raw = pbc_bootstrap.loc[
        pbc_bootstrap.score_contract.eq(PRIMARY_CATEGORICAL_CONTRACT)
        & pbc_bootstrap.lead_scope.isin([f"W{lead}" for lead in range(1, 7)])
        & pbc_bootstrap.baseline.eq("raw_fuxi_categorical")
        & pbc_bootstrap.method.isin(("debias_plus_plus", "persistence_plus_plus"))
    ].copy()
    expected_comparator_methods = {
        "moment_calibration",
        "summary_only",
        "location_spread",
        "pbc_combined",
    }
    _require(
        len(comparator_raw) == 6 * len(expected_comparator_methods)
        and set(comparator_raw.method) == expected_comparator_methods
        and not comparator_raw.duplicated(["lead_week", "method"]).any(),
        "shared comparator lacks all leadwise raw comparisons",
    )
    _require(
        len(pbc_raw) == 12
        and set(pbc_raw.method) == {"debias_plus_plus", "persistence_plus_plus"}
        and not pbc_raw.duplicated(["lead_week", "method"]).any(),
        "PBC V2 lacks all leadwise component-vs-raw comparisons",
    )

    uncertainty: dict[tuple[int, str], dict[str, Any]] = {}
    for frame, role in ((comparator_raw, "categorical"), (pbc_raw, "pbc")):
        for row in frame.itertuples(index=False):
            key = (int(row.lead_week), str(row.method))
            uncertainty[key] = {
                "effect": 100.0 * float(row.rps_reduction_fraction),
                "lower": 100.0 * float(row.ci_lower_95),
                "upper": 100.0 * float(row.ci_upper_95),
                "probability": float(row.bootstrap_probability_improvement),
                "source_role": role,
                "bootstrap_samples": int(row.bootstrap_samples),
                "block_length": int(row.block_length_initializations),
                "bootstrap_scheme": str(row.bootstrap_scheme),
                "test_year_count": int(row.test_year_count),
            }

    lead_records: list[dict[str, Any]] = []
    for lead in range(1, 7):
        selected = weekwise.loc[weekwise.lead_week.eq(lead)]
        raw = selected.loc[selected.method.eq("raw_fuxi_categorical")].iloc[0]
        for row in selected.itertuples(index=False):
            method = str(row.method)
            point = 100.0 * (1.0 - float(row.rps) / float(raw.rps))
            if method == "raw_fuxi_categorical":
                receipt = {
                    "effect": 0.0,
                    "lower": np.nan,
                    "upper": np.nan,
                    "probability": np.nan,
                    "source_role": "not_applicable_reference",
                    "bootstrap_samples": np.nan,
                    "block_length": np.nan,
                    "bootstrap_scheme": "not_applicable_reference",
                    "test_year_count": 2,
                }
            else:
                receipt = uncertainty.get((lead, method))
                _require(receipt is not None, f"missing leadwise raw CI for W{lead}/{method}")
                _require(
                    math.isclose(point, float(receipt["effect"]), rel_tol=0.0, abs_tol=2.0e-10),
                    f"leadwise raw effect differs from shared RPS scores for W{lead}/{method}",
                )
            lead_records.append(
                {
                    "lead_week": lead,
                    "method": method,
                    "method_label": str(row.method_label),
                    "scientific_status": source.manifest["scientific_status"],
                    "score_contract": str(row.score_contract),
                    "primary_rps": float(row.rps),
                    "raw_fuxi_rps": float(raw.rps),
                    "training_empirical_climatology_rps": float(
                        row.training_empirical_climatology_rps
                    ),
                    "primary_rpss_vs_training_empirical_climatology": float(
                        row.rpss_vs_training_empirical_climatology
                    ),
                    "rps_skill_pct_vs_raw": point,
                    "ci_lower_95_pct": receipt["lower"],
                    "ci_upper_95_pct": receipt["upper"],
                    "bootstrap_probability_improvement": receipt["probability"],
                    "ci_source_role": receipt["source_role"],
                    "bootstrap_samples": receipt["bootstrap_samples"],
                    "block_length_initializations": receipt["block_length"],
                    "bootstrap_scheme": receipt["bootstrap_scheme"],
                    "test_year_count": receipt["test_year_count"],
                    "initializations": int(row.n_initializations),
                }
            )
    lead_table = pd.DataFrame(lead_records).sort_values(
        ["lead_week", "primary_rps"]
    ).reset_index(drop=True)

    pairwise = comparator_bootstrap.loc[
        comparator_bootstrap.metric.eq(PRIMARY_CATEGORICAL_METRIC)
        & comparator_bootstrap.method.eq("location_spread")
        & comparator_bootstrap.baseline.eq("pbc_combined")
        & comparator_bootstrap.lead_scope.isin(
            ["W1-W6", *[f"W{lead}" for lead in range(1, 7)]]
        )
    ].copy()
    _require(
        len(pairwise) == 7
        and set(pairwise.lead_scope)
        == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"}
        and not pairwise.lead_scope.duplicated().any(),
        "location-spread vs combined-PBC paired comparison is incomplete",
    )
    pair_records: list[dict[str, Any]] = []
    inference_status = str(
        source.manifest.get("evaluation", {}).get(
            "bootstrap_inference_status",
            "exploratory paired uncertainty; two year clusters do not create an independent test",
        )
    )
    _require(
        "exploratory" in inference_status.lower()
        and "two year" in inference_status.lower(),
        "categorical manifest lacks the exploratory two-year inference caveat",
    )
    for row in pairwise.itertuples(index=False):
        lower = 100.0 * float(row.ci_lower_95)
        upper = 100.0 * float(row.ci_upper_95)
        if lower > 0.0:
            direction = "location_spread_lower_rps"
        elif upper < 0.0:
            direction = "pbc_combined_lower_rps"
        else:
            direction = "interval_crosses_zero"
        pair_records.append(
            {
                "lead_scope": str(row.lead_scope),
                "lead_week": int(row.lead_week),
                "method": "location_spread",
                "baseline": "pbc_combined",
                "effect_definition": "100 * (1 - location_spread_RPS / pbc_combined_RPS); positive favors location_spread",
                "location_spread_rps": float(row.method_score),
                "pbc_combined_rps": float(row.baseline_score),
                "rps_reduction_pct": 100.0 * float(row.rps_reduction_fraction),
                "ci_lower_95_pct": lower,
                "ci_upper_95_pct": upper,
                "bootstrap_probability_location_spread_improves": float(
                    row.bootstrap_probability_improvement
                ),
                "interval_interpretation": direction,
                "evidence_label": (
                    "reused-development exploratory paired comparison with two year clusters"
                ),
                "inference_status": inference_status,
                "bootstrap_samples": int(row.bootstrap_samples),
                "block_length_initializations": int(
                    row.block_length_initializations
                ),
                "bootstrap_scheme": str(row.bootstrap_scheme),
                "test_year_count": int(row.test_year_count),
            }
        )
    pair_table = pd.DataFrame(pair_records).sort_values("lead_week").reset_index(
        drop=True
    )
    return lead_table, pair_table


def build_capacity_table(sources: Mapping[str, VerifiedSource]) -> pd.DataFrame:
    frame = _verified_csv(
        sources["capacity"],
        "metrics/validation_summary.csv",
        required_columns=(
            "candidate",
            "role",
            "mode",
            "member_encoder_used",
            "parameter_count",
            "mean_validation_crps",
            "crps_skill_pct_vs_base",
            "all_years_noninferior_guard",
            "matched_seed_improvement_passes",
            "eligible_promotion",
        ),
    ).copy()
    expected = {row.get("name") for row in sources["capacity"].manifest.get("candidates", [])}
    expected.update(
        row.get("name") for row in sources["capacity"].manifest.get("controls", [])
    )
    expected.discard(None)
    _require(
        len(frame) == len(expected) == 5 and set(frame.candidate) == expected,
        "capacity table does not contain the complete frozen screen",
    )
    selection = sources["capacity"].manifest.get("selection", {})
    selected = selection.get("selected_candidate")
    _require(selected in expected, "capacity selection is absent from candidate table")
    _finite(
        frame,
        ("parameter_count", "mean_validation_crps", "crps_skill_pct_vs_base"),
        "capacity table",
    )
    result = frame[
        [
            "candidate",
            "role",
            "mode",
            "member_encoder_used",
            "parameter_count",
            "mean_validation_crps",
            "crps_skill_pct_vs_base",
            "all_years_noninferior_guard",
            "matched_seed_improvement_passes",
            "eligible_promotion",
        ]
    ].copy()
    result.insert(1, "selected_by_validation", result.candidate.eq(selected))
    result.insert(
        2,
        "scientific_status",
        sources["capacity"].manifest["scientific_status"],
    )
    result["selection_reason"] = str(selection.get("reason", ""))
    return result.sort_values("parameter_count").reset_index(drop=True)


def build_loss_table(sources: Mapping[str, VerifiedSource]) -> pd.DataFrame:
    validation = _verified_csv(
        sources["hybrid_loss"],
        "metrics/validation_profile_summary.csv",
        required_columns=(
            "profile",
            "alpha_mse",
            "mean_validation_crps",
            "mean_validation_rmse",
            "mean_validation_coverage_error",
            "crps_ratio_vs_control",
            "eligible_joint_candidate",
        ),
    ).copy()
    pooled = _verified_csv(
        sources["hybrid_loss"],
        "metrics/pooled_metrics.csv",
        required_columns=(
            "method",
            "crps",
            "rmse",
            "crps_skill_pct_vs_raw",
            "rmse_skill_pct_vs_raw",
            "case_count",
        ),
    ).copy()
    profiles = tuple(sources["hybrid_loss"].manifest.get("profiles", ()))
    _require(
        len(validation) == len(profiles) == 5
        and set(validation.profile) == set(profiles),
        "hybrid-loss validation table does not contain all frozen profiles",
    )
    selected = sources["hybrid_loss"].manifest.get("selection", {}).get(
        "selected_profile"
    )
    _require(selected in profiles, "hybrid-loss selected profile is absent")
    pooled = pooled.loc[pooled.method.isin(profiles)]
    _require(
        len(pooled) == len(profiles) and not pooled.method.duplicated().any(),
        "hybrid-loss development table does not contain all profiles",
    )
    result = validation.merge(
        pooled[
            [
                "method",
                "crps",
                "rmse",
                "crps_skill_pct_vs_raw",
                "rmse_skill_pct_vs_raw",
                "case_count",
            ]
        ],
        left_on="profile",
        right_on="method",
        validate="one_to_one",
    ).drop(columns="method")
    _finite(
        result,
        (
            "alpha_mse",
            "mean_validation_crps",
            "mean_validation_rmse",
            "crps_ratio_vs_control",
            "crps",
            "rmse",
        ),
        "hybrid-loss table",
    )
    result.insert(1, "selected_by_validation", result.profile.eq(selected))
    result.insert(
        2,
        "scientific_status",
        sources["hybrid_loss"].manifest["scientific_status"],
    )
    result = result.rename(
        columns={
            "crps": "development_crps",
            "rmse": "development_rmse",
            "crps_skill_pct_vs_raw": "development_crps_skill_pct_vs_raw",
            "rmse_skill_pct_vs_raw": "development_rmse_skill_pct_vs_raw",
            "case_count": "development_initializations",
        }
    )
    return result.sort_values("alpha_mse").reset_index(drop=True)


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_continuous_skill(table: pd.DataFrame, stem: Path) -> None:
    _paper_style()
    fig, axis = plt.subplots(figsize=(7.2, 3.65))
    palette = {
        "development_2020_2021_reused": "#0072B2",
        "operational_era_2022_2024_retrospective": "#D55E00",
    }
    labels = {
        "development_2020_2021_reused": "2020–2021 reused development",
        "operational_era_2022_2024_retrospective": "2022–2024 retrospective audit",
    }
    offsets = {
        "development_2020_2021_reused": -0.045,
        "operational_era_2022_2024_retrospective": 0.045,
    }
    for period in labels:
        selected = table.loc[
            table.evidence_period.eq(period) & table.lead_week.between(1, 6)
        ].sort_values("lead_week")
        _require(len(selected) == 6, f"figure 1 lacks six leads for {period}")
        x = selected.lead_week.to_numpy(dtype=float) + offsets[period]
        y = selected.crps_skill_pct_vs_raw.to_numpy(dtype=float)
        lower = y - selected.ci_lower_95_pct.to_numpy(dtype=float)
        upper = selected.ci_upper_95_pct.to_numpy(dtype=float) - y
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((lower, upper)),
            color=palette[period],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2.5,
            label=labels[period],
        )
    axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(1, 7), [f"W{week}" for week in range(1, 7)])
    axis.set_xlabel("Lead week")
    axis.set_ylabel("CRPS skill vs raw FuXi (%)")
    axis.set_title("Locked adapter: development and later-era retrospective evidence")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    _save_figure(fig, stem)


def plot_categorical(
    table: pd.DataFrame, lead_table: pd.DataFrame, stem: Path
) -> None:
    _paper_style()
    ordered = table.sort_values("primary_rps").reset_index(drop=True)
    short_labels = {
        "raw_fuxi_categorical": "Raw FuXi",
        "moment_calibration": "Moment",
        "summary_only": "Summary neural",
        "location_spread": "Location+spread",
        "debias_plus_plus": "Debias++",
        "persistence_plus_plus": "Persistence++",
        "pbc_combined": "Combined PBC",
    }
    method_styles = {
        "raw_fuxi_categorical": ("#555555", "--", "o", 1.2, 4.0),
        "moment_calibration": ("#999999", ":", "v", 1.1, 3.8),
        "summary_only": ("#CC79A7", "--", "^", 1.2, 4.0),
        "location_spread": ("#0072B2", "-", "o", 2.2, 5.0),
        "debias_plus_plus": ("#8C6BB1", ":", "P", 1.3, 4.2),
        "persistence_plus_plus": ("#E69F00", "-", "s", 2.2, 5.0),
        "pbc_combined": ("#009E73", "-", "D", 2.2, 4.7),
    }
    bar_colors = [method_styles[str(method)][0] for method in ordered.method]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.15), gridspec_kw={"width_ratios": [0.9, 1.35]})
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered.primary_rps, color=bar_colors, height=0.72)
    axes[0].set_yticks(
        y, [short_labels[str(method)] for method in ordered.method]
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Pooled primary RPS\n(lower is better)")
    axes[0].set_title("a  Pooled ranking")
    axes[0].grid(axis="x", color="0.92", linewidth=0.6)

    for method in short_labels:
        selected = lead_table.loc[lead_table.method.eq(method)].sort_values(
            "lead_week"
        )
        _require(len(selected) == 6, f"categorical figure lacks six leads for {method}")
        color, linestyle, marker, width, marker_size = method_styles[method]
        axes[1].plot(
            selected.lead_week,
            selected.rps_skill_pct_vs_raw,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=marker_size,
            linewidth=width,
            alpha=1.0 if width > 2.0 else 0.82,
            label=short_labels[method],
        )
    axes[1].axhline(0.0, color="0.45", linewidth=0.7)
    axes[1].set_xticks(range(1, 7), [f"W{week}" for week in range(1, 7)])
    axes[1].set_xlabel("Lead week")
    axes[1].set_ylabel("Primary RPS skill vs raw FuXi (%)")
    axes[1].set_title("b  Lead-dependent skill")
    axes[1].grid(axis="y", color="0.92", linewidth=0.6)
    skill_values = lead_table.rps_skill_pct_vs_raw.to_numpy(dtype=float)
    axes[1].set_ylim(
        min(-1.5, float(np.min(skill_values)) - 2.0),
        float(np.max(skill_values)) + 10.0,
    )
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        columnspacing=0.9,
        handlelength=2.2,
    )
    fig.suptitle(
        "Equality-aware shared comparison (2020–2021 reused development)",
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, stem)


def plot_capacity_and_loss(
    capacity: pd.DataFrame, loss: pd.DataFrame, stem: Path
) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65))
    capacity = capacity.sort_values("parameter_count")
    capacity_colors = [
        "#D55E00" if selected else "#0072B2"
        for selected in capacity.selected_by_validation
    ]
    axes[0].scatter(
        capacity.parameter_count,
        capacity.crps_skill_pct_vs_base,
        color=capacity_colors,
        s=34,
        zorder=3,
    )
    axes[0].axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (log scale)")
    axes[0].set_ylabel("Validation CRPS skill vs base (%)")
    axes[0].set_title("Capacity screen (2018–2019 only)")
    capacity_labels = {
        "small_20k": "Small 20k",
        "base_42k": "Base 42k",
        "summary_matched_43k": "Summary 43k",
        "medium_158k": "Medium 158k",
        "large_294k": "Large 294k",
    }
    capacity_offsets = {
        "small_20k": (3, 4),
        "base_42k": (3, 4),
        "summary_matched_43k": (3, 4),
        "medium_158k": (3, 4),
        "large_294k": (-3, 4),
    }
    for row in capacity.itertuples(index=False):
        candidate = str(row.candidate)
        axes[0].annotate(
            capacity_labels[candidate],
            (float(row.parameter_count), float(row.crps_skill_pct_vs_base)),
            xytext=capacity_offsets[candidate],
            textcoords="offset points",
            fontsize=7,
            ha="right" if candidate == "large_294k" else "left",
        )

    loss = loss.sort_values("alpha_mse")
    loss_skill = 100.0 * (1.0 - loss.crps_ratio_vs_control.to_numpy(dtype=float))
    loss_colors = [
        "#D55E00" if selected else "#009E73"
        for selected in loss.selected_by_validation
    ]
    axes[1].plot(
        loss.alpha_mse,
        loss_skill,
        color="#009E73",
        linewidth=1.2,
        zorder=2,
    )
    axes[1].scatter(loss.alpha_mse, loss_skill, color=loss_colors, s=34, zorder=3)
    axes[1].axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("MSE mixture weight, α")
    axes[1].set_ylabel("Validation CRPS skill vs CRPS-only (%)")
    axes[1].set_title("Loss screen (2018–2019 only)")
    loss_labels = {
        "crps_only": "CRPS only",
        "hybrid_010": "α=.10",
        "hybrid_025": "α=.25",
        "hybrid_050": "α=.50",
        "mse_only": "MSE only",
    }
    loss_offsets = {
        "crps_only": (3, 7),
        "hybrid_010": (3, -13),
        "hybrid_025": (3, 7),
        "hybrid_050": (3, 6),
        "mse_only": (3, 5),
    }
    for row, skill in zip(loss.itertuples(index=False), loss_skill, strict=True):
        profile = str(row.profile)
        axes[1].annotate(
            loss_labels[profile],
            (float(row.alpha_mse), float(skill)),
            xytext=loss_offsets[profile],
            textcoords="offset points",
            fontsize=7,
        )
    for axis in axes:
        axis.grid(axis="y", color="0.92", linewidth=0.6)
    fig.suptitle("Frozen architecture and objective sensitivity")
    fig.tight_layout()
    _save_figure(fig, stem)


def _source_registry(sources: Mapping[str, VerifiedSource]) -> dict[str, Any]:
    return {
        "contract_version": BUNDLE_CONTRACT_VERSION,
        "sources": {
            role: {
                "label": ROLE_LABELS[role],
                "path": str(sources[role].manifest_path),
                "experiment": sources[role].manifest["experiment"],
                "manifest_sha256": sources[role].manifest_sha256,
                "scientific_status": sources[role].manifest["scientific_status"],
                "artifact_count_verified": len(sources[role].artifact_sha256),
                "receipt_policy": sources[role].receipt_policy,
                "slurm_gate_receipt_sha256": sources[role].gate_receipt_sha256,
            }
            for role in ROLE_ORDER
        },
        "cross_source_bindings_verified": {
            "capacity_to_postselection_development": True,
            "capacity_to_operational_era": True,
            "neural_and_pbc_to_shared_categorical": True,
        },
        "sealed_year": {
            "year": 2025,
            "forecast_or_target_opened_by_builder": False,
            "builder_reads_only_manifest_declared_reporting_artifacts": True,
        },
    }


def _claim_boundaries(
    sources: Mapping[str, VerifiedSource],
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    categorical_pairwise: pd.DataFrame,
    capacity: pd.DataFrame,
    loss: pd.DataFrame,
) -> dict[str, Any]:
    pooled_continuous = continuous.loc[continuous.lead_week.eq(0)].set_index(
        "evidence_period"
    )
    categorical_best = categorical.sort_values("primary_rps").iloc[0]
    selected_capacity = capacity.loc[capacity.selected_by_validation].iloc[0]
    selected_loss = loss.loc[loss.selected_by_validation].iloc[0]
    pairwise_leads = categorical_pairwise.loc[
        categorical_pairwise.lead_week.between(1, 6)
    ].sort_values("lead_week")
    _require(len(pairwise_leads) == 6, "claim guard lacks six pairwise lead rows")

    continuous_claims = []
    for period, row in pooled_continuous.iterrows():
        interval_positive = float(row.ci_lower_95_pct) > 0.0
        continuous_claims.append(
            {
                "evidence_period": period,
                "effect_crps_skill_pct_vs_raw": float(row.crps_skill_pct_vs_raw),
                "ci_95_pct": [
                    float(row.ci_lower_95_pct),
                    float(row.ci_upper_95_pct),
                ],
                "interval_excludes_zero_in_improvement_direction": interval_positive,
                "allowed_claim": (
                    "paired CRPS improvement in this explicitly labelled retrospective period"
                    if interval_positive
                    else "point estimate only; interval does not establish improvement"
                ),
            }
        )

    return {
        "contract_version": BUNDLE_CONTRACT_VERSION,
        "status": "derived_claim_guardrails_from_verified_sources",
        "numeric_claims": {
            "continuous_crps": continuous_claims,
            "categorical_primary_descriptive_best": {
                "method": str(categorical_best.method),
                "primary_rps": float(categorical_best.primary_rps),
                "primary_rpss_vs_training_empirical_climatology": float(
                    categorical_best.primary_rpss_vs_training_empirical_climatology
                ),
                "allowed_claim": (
                    "lowest point-estimate primary RPS among the seven methods on the shared reused-development comparison"
                ),
                "not_allowed": "universal superiority or independent-test language",
            },
            "location_spread_vs_combined_pbc_by_lead": {
                "effect_definition": str(pairwise_leads.iloc[0].effect_definition),
                "evidence_label": str(pairwise_leads.iloc[0].evidence_label),
                "inference_status": str(pairwise_leads.iloc[0].inference_status),
                "lead_results": [
                    {
                        "lead_week": int(row.lead_week),
                        "effect_pct": float(row.rps_reduction_pct),
                        "ci_95_pct": [
                            float(row.ci_lower_95_pct),
                            float(row.ci_upper_95_pct),
                        ],
                        "interval_interpretation": str(
                            row.interval_interpretation
                        ),
                    }
                    for row in pairwise_leads.itertuples(index=False)
                ],
                "allowed_claim": (
                    "lead-dependent reused-development result: report each paired interval; do not collapse it into universal neural or PBC superiority"
                ),
                "not_allowed": (
                    "independent-test, prospective, or broad all-lead superiority language"
                ),
            },
            "capacity_selection": {
                "selected_candidate": str(selected_capacity.candidate),
                "parameter_count": int(selected_capacity.parameter_count),
                "selection_reason": str(selected_capacity.selection_reason),
                "allowed_claim": "validation-selected architecture retained under the frozen guard rules",
            },
            "loss_selection": {
                "selected_profile": str(selected_loss.profile),
                "alpha_mse": float(selected_loss.alpha_mse),
                "allowed_claim": "validation-selected loss profile within this frozen ablation",
            },
        },
        "mandatory_boundaries": [
            {
                "id": "development_reuse",
                "rule": "2020-2021 is reused development evidence, not an untouched final test",
            },
            {
                "id": "later_era_retrospective",
                "rule": "2022-2024 is a no-retrain retrospective operational-era audit, not a prospective operational trial",
            },
            {
                "id": "sealed_final_year",
                "rule": "2025 remains unopened and cannot be described as evaluated",
            },
            {
                "id": "geography",
                "rule": "results concern the 27x27 India-domain target; no whole-world performance claim",
            },
            {
                "id": "categorical_protocol",
                "rule": "the equality-aware static PBC comparison is not a reproduction of the rolling Guan et al. protocol",
            },
            {
                "id": "extremes",
                "rule": "upper-tail diagnostics are exploratory/descriptive and do not support an extremes headline",
            },
            {
                "id": "bias",
                "rule": "do not claim universal signed-bias improvement from CRPS, RMSE, ACC, or categorical score gains",
            },
            {
                "id": "seed_aggregation",
                "rule": "headline neural results average per-seed scores; parameters, predictions, and adjustment fields are not averaged",
            },
        ],
        "source_scientific_status": {
            role: sources[role].manifest["scientific_status"] for role in ROLE_ORDER
        },
    }


def _readme_text(sources: Mapping[str, VerifiedSource]) -> str:
    status_lines = [
        f"- **{ROLE_LABELS[role]}:** {sources[role].manifest['scientific_status']}"
        for role in ROLE_ORDER
    ]
    return "\n".join(
        [
            "# All-season probabilistic calibration: verified paper evidence bundle",
            "",
            "This is a venue-neutral, derived reporting bundle. It contains tables, figures, provenance, and claim boundaries; it is not submission prose. Every input manifest, every manifest-declared artifact, and every applicable Slurm gate receipt was verified before these outputs were produced.",
            "",
            "## Evidence labels",
            "",
            *status_lines,
            "",
            "## Contents",
            "",
            "- `tables/continuous_crps_skill_by_lead.csv`: paired neural-adapter CRPS skill and 95% intervals for reused development and the later retrospective audit.",
            "- `tables/categorical_rps_comparison.csv` and `tables/categorical_rps_by_lead.csv`: pooled and weekwise seven-method equality-aware RPS/RPSS comparisons on identical 2020–2021 cases, thresholds, and support.",
            "- `tables/categorical_location_vs_combined_by_lead.csv`: paired location+spread neural versus combined-PBC effects and intervals, explicitly labelled as exploratory two-year reused-development inference.",
            "- `tables/capacity_ablation.csv` and `tables/loss_ablation.csv`: validation-locked architecture and objective screens, with their original evidence labels preserved.",
            "- `figures/`: three compact figures, each supplied as vector PDF and 300-dpi PNG.",
            "- `claim_boundaries.json`: machine-readable allowed interpretations and mandatory limitations.",
            "- `source_receipts.json`: immutable source hashes, receipt policies, and cross-source bindings.",
            "- `manifest.json`: SHA-256 inventory of every other file in this derived bundle.",
            "",
            "## Non-negotiable scope",
            "",
            "The 2020–2021 evidence is reused development evidence. The 2022–2024 evidence is a no-retrain retrospective audit, not a prospective operational trial. The India-domain results do not establish whole-world performance. The sealed 2025 forecast/target was not opened by this builder or by the accepted source contracts.",
            "",
        ]
    )


def _artifact_inventory(output: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative == "manifest.json":
            continue
        inventory[relative] = sha256_file(path)
    return inventory


def _normalise_for_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _normalise_for_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_for_json(child) for child in value]
    return value


def build_bundle(
    source_paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str],
    output: Path,
    *,
    allowed_output_root: Path,
) -> Path:
    """Verify seven sources and atomically install a fresh derived bundle."""

    _require(set(source_paths) == set(ROLE_ORDER), "seven explicit source paths are required")
    _require(set(expected_hashes) == set(ROLE_ORDER), "seven manifest hashes are required")
    output = Path(output).resolve(strict=False)
    allowed = Path(allowed_output_root).resolve(strict=False)
    try:
        output.relative_to(allowed)
    except ValueError as error:
        raise BundleContractError(
            f"output must be under the deliverables root {allowed}: {output}"
        ) from error
    _require(output != allowed, "output must be a fresh child of the deliverables root")
    _require(not _path_mentions_forbidden_year(output), "output path may not contain 2025")
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    _require(not staging.exists(), f"staging path already exists: {staging}")

    sources: dict[str, VerifiedSource] = {}
    # Verification happens before staging is created: a failed receipt leaves no
    # apparent bundle and no partial deliverable.
    for role in ROLE_ORDER:
        sources[role] = verify_source(
            role,
            source_paths[role],
            expected_manifest_sha256=expected_hashes[role],
        )
    verify_source_relations(sources)

    continuous = build_continuous_skill_table(sources)
    categorical = build_categorical_table(sources)
    categorical_by_lead, categorical_pairwise = build_categorical_lead_tables(
        sources
    )
    capacity = build_capacity_table(sources)
    loss = build_loss_table(sources)

    staging.mkdir()
    try:
        tables = staging / "tables"
        figures = staging / "figures"
        tables.mkdir()
        figures.mkdir()
        for frame, filename in (
            (continuous, "continuous_crps_skill_by_lead.csv"),
            (categorical, "categorical_rps_comparison.csv"),
            (categorical_by_lead, "categorical_rps_by_lead.csv"),
            (
                categorical_pairwise,
                "categorical_location_vs_combined_by_lead.csv",
            ),
            (capacity, "capacity_ablation.csv"),
            (loss, "loss_ablation.csv"),
        ):
            frame.to_csv(
                tables / filename,
                index=False,
                lineterminator="\n",
                float_format="%.12g",
            )

        plot_continuous_skill(
            continuous, figures / "continuous_crps_skill_by_lead"
        )
        plot_categorical(
            categorical,
            categorical_by_lead,
            figures / "categorical_rps_comparison",
        )
        plot_capacity_and_loss(
            capacity, loss, figures / "capacity_and_loss_ablation"
        )

        (staging / "README.md").write_text(
            _readme_text(sources), encoding="utf-8"
        )
        _write_json(staging / "source_receipts.json", _source_registry(sources))
        _write_json(
            staging / "claim_boundaries.json",
            _normalise_for_json(
                _claim_boundaries(
                    sources,
                    continuous,
                    categorical,
                    categorical_pairwise,
                    capacity,
                    loss,
                )
            ),
        )

        artifact_sha256 = _artifact_inventory(staging)
        expected_outputs = {
            "README.md",
            "source_receipts.json",
            "claim_boundaries.json",
            "tables/continuous_crps_skill_by_lead.csv",
            "tables/categorical_rps_comparison.csv",
            "tables/categorical_rps_by_lead.csv",
            "tables/categorical_location_vs_combined_by_lead.csv",
            "tables/capacity_ablation.csv",
            "tables/loss_ablation.csv",
            "figures/continuous_crps_skill_by_lead.pdf",
            "figures/continuous_crps_skill_by_lead.png",
            "figures/categorical_rps_comparison.pdf",
            "figures/categorical_rps_comparison.png",
            "figures/capacity_and_loss_ablation.pdf",
            "figures/capacity_and_loss_ablation.png",
        }
        _require(
            set(artifact_sha256) == expected_outputs,
            "derived bundle output inventory differs from the contract",
        )
        manifest = {
            "experiment": BUNDLE_EXPERIMENT,
            "contract_version": BUNDLE_CONTRACT_VERSION,
            "status": "complete",
            "mode": "derived_full",
            "smoke": False,
            "scientific_status": (
                "venue-neutral reporting bundle derived from verified validation, reused-development, and retrospective-audit evidence; not an untouched final test"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output),
            "contract": {
                "source_runs_modified": False,
                "forecast_or_target_arrays_opened": False,
                "year_store_discovery_used": False,
                "sealed_2025_target_opened": False,
                "sealed_unopened_years": [2025],
                "submission_prose_generated": False,
                "tables_derived_only_from_receipted_csv_artifacts": True,
                "figures_derived_only_from_bundle_tables": True,
            },
            "source_manifest_sha256": {
                role: sources[role].manifest_sha256 for role in ROLE_ORDER
            },
            "source_scientific_status": {
                role: sources[role].manifest["scientific_status"]
                for role in ROLE_ORDER
            },
            "tables": [
                "tables/continuous_crps_skill_by_lead.csv",
                "tables/categorical_rps_comparison.csv",
                "tables/categorical_rps_by_lead.csv",
                "tables/categorical_location_vs_combined_by_lead.csv",
                "tables/capacity_ablation.csv",
                "tables/loss_ablation.csv",
            ],
            "figures": [
                "figures/continuous_crps_skill_by_lead",
                "figures/categorical_rps_comparison",
                "figures/capacity_and_loss_ablation",
            ],
            "artifact_sha256": artifact_sha256,
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        # Recheck every output immediately before the atomic rename.
        _require(
            _artifact_inventory(staging) == artifact_sha256,
            "derived artifact changed before atomic installation",
        )
        os.replace(staging, output)
    except Exception:
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(
            f".{output.name}.incomplete-"
        ):
            shutil.rmtree(staging)
        raise
    return output


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        PROJECT_ROOT
        / "presentation/deliverables"
        / f"fuxi_allseason_probabilistic_paper_{stamp}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the receipt-gated all-season probabilistic paper evidence bundle."
    )
    parser.add_argument("--neural-full", type=Path, required=True)
    parser.add_argument("--capacity-full", type=Path, required=True)
    parser.add_argument("--capacity-development-full", type=Path, required=True)
    parser.add_argument("--hybrid-loss-full", type=Path, required=True)
    parser.add_argument("--pbc-full", type=Path, required=True)
    parser.add_argument("--categorical-full", type=Path, required=True)
    parser.add_argument("--operational-full", type=Path, required=True)
    parser.add_argument(
        "--categorical-manifest-sha256",
        required=True,
        help="Exact SHA-256 printed only after the categorical full-run audit passes.",
    )
    parser.add_argument(
        "--operational-manifest-sha256",
        required=True,
        help="Exact SHA-256 printed only after the operational full-run audit passes.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_paths = {
        "neural": args.neural_full,
        "capacity": args.capacity_full,
        "capacity_development": args.capacity_development_full,
        "hybrid_loss": args.hybrid_loss_full,
        "pbc": args.pbc_full,
        "categorical": args.categorical_full,
        "operational": args.operational_full,
    }
    expected_hashes = {
        **KNOWN_MANIFEST_SHA256,
        "categorical": args.categorical_manifest_sha256,
        "operational": args.operational_manifest_sha256,
    }
    output = default_output() if args.output is None else args.output
    completed = build_bundle(
        source_paths,
        expected_hashes,
        output,
        allowed_output_root=PROJECT_ROOT / "presentation/deliverables",
    )
    manifest_sha = sha256_file(completed / "manifest.json")
    print(f"PASS: {completed}")
    print(f"manifest_sha256={manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
