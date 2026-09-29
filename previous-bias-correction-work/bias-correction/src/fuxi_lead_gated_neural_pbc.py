#!/usr/bin/env python3
"""Receipt-gated score-only evaluation of one fixed lead router.

The policy uses the accepted neural adapter at W1 and accepted Persistence++
at W2--W6.  It reads only immutable case-level quintile-RPS tables, so it
cannot train, tune, reconstruct forecasts, or access the sealed 2025 target.

The policy was motivated after both input cohorts had already been inspected.
Its output is therefore a post-hoc diagnostic even when paired intervals are
above zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPERIMENT = "fuxi_lead_gated_neural_pbc_v1"
POLICY = "neural_w1_persistence_w2_w6"
POLICY_LABEL = "Neural W1 + Persistence++ W2--W6"
PLAN_PATH = PROJECT_ROOT / "plan/LEAD_GATED_NEURAL_PBC_20260823.md"
PLAN_SHA256 = "0d784a50afa534fbefa92c3f96cae6a2045e9d7518ab927a9901ea50ac6766c7"
SOURCE_PATH = PROJECT_ROOT / "src/fuxi_lead_gated_neural_pbc.py"
TEST_PATH = PROJECT_ROOT / "tests/test_fuxi_lead_gated_neural_pbc.py"

HYBRID_METHOD = POLICY
NEURAL_METHOD = "neural_adapter"
PERSISTENCE_METHOD = "persistence_plus_plus"
COMBINED_METHOD = "pbc_combined"
RAW_METHOD = "raw_fuxi_categorical"
METHOD_ORDER = (
    RAW_METHOD,
    NEURAL_METHOD,
    PERSISTENCE_METHOD,
    COMBINED_METHOD,
    HYBRID_METHOD,
)
CONTRASTS = (
    PERSISTENCE_METHOD,
    COMBINED_METHOD,
    NEURAL_METHOD,
    RAW_METHOD,
)
LEADS = tuple(range(1, 7))
BLOCK_LENGTH = 13


class LeadGateError(RuntimeError):
    """Raised when an immutable input or score-pairing contract moves."""


@dataclass(frozen=True)
class CohortSpec:
    name: str
    evidence_label: str
    manifest_path: Path
    manifest_sha256: str
    case_path: Path
    case_sha256: str
    case_artifact_key: str
    neural_source_method: str
    expected_cases: int
    expected_year_counts: Mapping[int, int]
    bootstrap_draws: int
    bootstrap_seed: int
    bootstrap_kind: str
    require_canonical: bool


COHORTS = (
    CohortSpec(
        name="reused_development_2020_2021",
        evidence_label=(
            "post-hoc reused 2020-2021 development evidence; not independent"
        ),
        manifest_path=(
            PROJECT_ROOT
            / "resultsv2/fuxi_allseason_categorical_comparison_v2/"
            "full_20260822T183010Z/manifest.json"
        ),
        manifest_sha256=(
            "44ca7130e626bfadc71d311d0002dbb39b67aebbf16d0dfb2802999b6be4bdbc"
        ),
        case_path=(
            PROJECT_ROOT
            / "resultsv2/fuxi_allseason_categorical_comparison_v2/"
            "full_20260822T183010Z/metrics/quintile_case_scores.csv"
        ),
        case_sha256=(
            "a6085cc530b2a9662e2b59a79f65f712186cd08467f8e88881c07ae359344047"
        ),
        case_artifact_key="metrics/quintile_case_scores.csv",
        neural_source_method="location_spread",
        expected_cases=208,
        expected_year_counts={2020: 104, 2021: 104},
        bootstrap_draws=2_000,
        bootstrap_seed=20_260_823,
        bootstrap_kind="equal_year_two_stage_circular_blocks",
        require_canonical=False,
    ),
    CohortSpec(
        name="later_retrospective_2022_2024",
        evidence_label=(
            "post-hoc 2022-2024 operational-era retrospective; "
            "no retraining or selection"
        ),
        manifest_path=(
            PROJECT_ROOT
            / "resultsv2/fuxi_allseason_operational_categorical_comparison/"
            "full_20260822T203000Z/manifest.json"
        ),
        manifest_sha256=(
            "cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d"
        ),
        case_path=(
            PROJECT_ROOT
            / "resultsv2/fuxi_allseason_operational_categorical_comparison/"
            "full_20260822T203000Z/metrics/quintile_case_scores.csv"
        ),
        case_sha256=(
            "b2bf3e748b37f2946ea4dbadd1b33720ebc20bc25930c5e9592a4fee1f017ee7"
        ),
        case_artifact_key="metrics/quintile_case_scores.csv",
        neural_source_method="base_42k",
        expected_cases=296,
        expected_year_counts={2022: 104, 2023: 104, 2024: 88},
        bootstrap_draws=10_000,
        bootstrap_seed=20_260_824,
        bootstrap_kind="unequal_year_two_stage_circular_blocks",
        require_canonical=True,
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LeadGateError(message)


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
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeadGateError(f"invalid {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def validate_source_manifest(spec: CohortSpec) -> Mapping[str, Any]:
    """Validate the exact accepted source manifest and sealed-year boundary."""

    _require(spec.manifest_path.is_file(), f"missing {spec.name} manifest")
    _require(
        sha256_file(spec.manifest_path) == spec.manifest_sha256,
        f"{spec.name} manifest SHA-256 changed",
    )
    manifest = _load_json(spec.manifest_path, label=f"{spec.name} manifest")
    _require(manifest.get("status") == "complete", f"{spec.name} is incomplete")
    if spec.require_canonical:
        _require(manifest.get("canonical") is True, f"{spec.name} is not canonical")
        _require(
            manifest.get("scientific_eligible") is True,
            f"{spec.name} is not scientifically eligible",
        )
    contract = manifest.get("contract", {})
    _require(
        isinstance(contract, dict)
        and contract.get("sealed_2025_target_opened") is False,
        f"{spec.name} does not prove the 2025 target remained sealed",
    )
    artifacts = manifest.get("artifact_sha256", {})
    _require(
        isinstance(artifacts, dict)
        and artifacts.get(spec.case_artifact_key) == spec.case_sha256,
        f"{spec.name} manifest does not bind the accepted case scores",
    )
    _require(spec.case_path.is_file(), f"missing {spec.name} case scores")
    _require(
        sha256_file(spec.case_path) == spec.case_sha256,
        f"{spec.name} case-score SHA-256 changed",
    )
    return manifest


def build_policy_case_scores(spec: CohortSpec) -> pd.DataFrame:
    """Build exact case/lead policy scores from accepted paired score rows."""

    validate_source_manifest(spec)
    frame = pd.read_csv(spec.case_path)
    if "family" in frame.columns:
        frame = frame.loc[frame.family.eq("quintile")].copy()
    required = {
        "method",
        "initialization",
        "lead_week",
        "rps",
        "score_contract",
        "seed",
    }
    _require(required.issubset(frame), f"{spec.name} case columns changed")
    source_methods = {
        spec.neural_source_method: NEURAL_METHOD,
        PERSISTENCE_METHOD: PERSISTENCE_METHOD,
        COMBINED_METHOD: COMBINED_METHOD,
        RAW_METHOD: RAW_METHOD,
    }
    selected = frame.loc[frame.method.isin(source_methods)].copy()
    selected["source_method"] = selected.method.astype(str)
    selected["method"] = selected.method.map(source_methods)
    _require(
        set(selected.method) == set(source_methods.values()),
        f"{spec.name} source methods are incomplete",
    )
    selected["initialization"] = selected.initialization.astype(str)
    selected["lead_week"] = selected.lead_week.astype(int)
    selected["year"] = pd.to_datetime(selected.initialization).dt.year.astype(int)
    _require(
        set(selected.lead_week.unique()) == set(LEADS),
        f"{spec.name} lead set changed",
    )
    _require(
        np.isfinite(selected.rps.to_numpy(dtype=np.float64)).all()
        and selected.rps.between(0.0, 1.0).all(),
        f"{spec.name} RPS values are invalid",
    )
    _require(
        selected.score_contract.nunique() == 1,
        f"{spec.name} has multiple score contracts",
    )
    _require(
        not selected.duplicated(["method", "initialization", "lead_week"]).any(),
        f"{spec.name} source rows are not uniquely paired",
    )
    initializations = sorted(selected.initialization.unique().tolist())
    _require(
        len(initializations) == spec.expected_cases,
        f"{spec.name} case count changed",
    )
    year_counts = (
        pd.Series(pd.to_datetime(initializations).year).value_counts().sort_index()
    )
    actual_year_counts = {int(year): int(count) for year, count in year_counts.items()}
    _require(
        actual_year_counts == dict(spec.expected_year_counts),
        f"{spec.name} year counts changed: {actual_year_counts}",
    )
    expected_keys = {
        (initialization, lead)
        for initialization in initializations
        for lead in LEADS
    }
    for method in source_methods.values():
        rows = selected.loc[selected.method.eq(method)]
        actual_keys = set(zip(rows.initialization, rows.lead_week, strict=True))
        _require(
            actual_keys == expected_keys,
            f"{spec.name}/{method} case support differs",
        )
    neural_seeds = set(selected.loc[selected.method.eq(NEURAL_METHOD), "seed"])
    _require(
        neural_seeds == {"mean_of_seed_scores_42_43_44"},
        f"{spec.name} neural score aggregation changed",
    )
    for method in (PERSISTENCE_METHOD, COMBINED_METHOD, RAW_METHOD):
        _require(
            set(selected.loc[selected.method.eq(method), "seed"])
            == {"not_applicable"},
            f"{spec.name}/{method} seed contract changed",
        )

    hybrid = pd.concat(
        [
            selected.loc[
                selected.method.eq(NEURAL_METHOD) & selected.lead_week.eq(1)
            ],
            selected.loc[
                selected.method.eq(PERSISTENCE_METHOD)
                & selected.lead_week.between(2, 6)
            ],
        ],
        ignore_index=True,
    ).copy()
    hybrid["component_method"] = hybrid.method
    hybrid["method"] = HYBRID_METHOD
    hybrid["seed"] = np.where(
        hybrid.lead_week.eq(1),
        "mean_of_seed_scores_42_43_44",
        "not_applicable",
    )
    _require(
        len(hybrid) == spec.expected_cases * len(LEADS)
        and not hybrid.duplicated(["initialization", "lead_week"]).any(),
        f"{spec.name} hybrid policy rows are incomplete",
    )
    selected["component_method"] = selected.method
    result = pd.concat([selected, hybrid], ignore_index=True)
    result["cohort"] = spec.name
    result["evidence_label"] = spec.evidence_label
    result["policy"] = POLICY
    result["policy_post_hoc"] = True
    columns = [
        "cohort",
        "evidence_label",
        "policy",
        "policy_post_hoc",
        "method",
        "component_method",
        "source_method",
        "seed",
        "score_contract",
        "initialization",
        "year",
        "lead_week",
        "rps",
    ]
    return result.loc[:, columns].sort_values(
        ["method", "initialization", "lead_week"]
    ).reset_index(drop=True)


def _equal_year_draws(
    dates: np.ndarray, *, draws: int, block_length: int, seed: int
) -> tuple[tuple[np.ndarray, ...], Mapping[str, Any]]:
    years = pd.DatetimeIndex(dates).year.to_numpy()
    source_years = tuple(int(value) for value in np.sort(np.unique(years)))
    groups = [np.flatnonzero(years == year) for year in source_years]
    _require(
        len(source_years) == 2 and len({len(group) for group in groups}) == 1,
        "development bootstrap requires two equal-sized year groups",
    )
    rng = np.random.default_rng(seed)
    cases_per_year = len(groups[0])
    effective_block = min(block_length, cases_per_year)
    block_count = int(math.ceil(cases_per_year / effective_block))
    offsets = np.arange(effective_block, dtype=np.int64)
    output: list[np.ndarray] = []
    repeated = 0
    for _ in range(draws):
        sampled_years = rng.integers(0, len(groups), size=len(groups))
        repeated += int(np.unique(sampled_years).size < len(groups))
        segments: list[np.ndarray] = []
        for year_slot in sampled_years:
            group = groups[int(year_slot)]
            starts = rng.integers(0, cases_per_year, size=block_count)
            local = ((starts[:, None] + offsets[None]) % cases_per_year).reshape(-1)
            segments.append(group[local[:cases_per_year]])
        output.append(np.concatenate(segments).astype(np.int64, copy=False))
    diagnostics = {
        "source_years": list(source_years),
        "year_sizes": {str(year): len(group) for year, group in zip(source_years, groups)},
        "draws": draws,
        "block_length_initializations": block_length,
        "seed": seed,
        "draws_with_repeated_year_cluster": repeated,
        "minimum_initializations_per_draw": len(dates),
        "maximum_initializations_per_draw": len(dates),
        "scheme": "sample years then circular blocks within sampled years",
    }
    return tuple(output), diagnostics


def _unequal_year_draws(
    dates: np.ndarray, *, draws: int, block_length: int, seed: int
) -> tuple[tuple[np.ndarray, ...], Mapping[str, Any]]:
    years = pd.DatetimeIndex(dates).year.to_numpy()
    source_years = tuple(int(value) for value in np.sort(np.unique(years)))
    _require(
        source_years == (2022, 2023, 2024),
        f"later bootstrap years changed: {source_years}",
    )
    groups = {year: np.flatnonzero(years == year) for year in source_years}
    rng = np.random.default_rng(seed)
    sampled_years = rng.choice(
        np.asarray(source_years, dtype=np.int64),
        size=(draws, len(source_years)),
        replace=True,
    )
    output: list[np.ndarray] = []
    for sampled in sampled_years:
        segments: list[np.ndarray] = []
        for sampled_year in sampled:
            group = groups[int(sampled_year)]
            effective_block = min(block_length, len(group))
            block_count = int(math.ceil(len(group) / effective_block))
            starts = rng.integers(0, len(group), size=block_count)
            offsets = np.arange(effective_block, dtype=np.int64)
            local = ((starts[:, None] + offsets[None]) % len(group)).reshape(-1)
            segments.append(group[local[: len(group)]])
        output.append(np.concatenate(segments).astype(np.int64, copy=False))
    lengths = np.asarray([len(draw) for draw in output], dtype=np.int64)
    diagnostics = {
        "source_years": list(source_years),
        "year_sizes": {str(year): len(groups[year]) for year in source_years},
        "draws": draws,
        "block_length_initializations": block_length,
        "seed": seed,
        "draws_with_repeated_year_cluster": int(
            np.sum([np.unique(row).size < len(row) for row in sampled_years])
        ),
        "minimum_initializations_per_draw": int(lengths.min()),
        "maximum_initializations_per_draw": int(lengths.max()),
        "scheme": "sample years then circular blocks within sampled years",
    }
    return tuple(output), diagnostics


def paired_bootstrap(
    frame: pd.DataFrame, spec: CohortSpec
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Compute paired pooled policy effects with initialization-block draws."""

    score_contracts = frame.score_contract.unique().tolist()
    _require(
        len(score_contracts) == 1,
        f"{spec.name} bootstrap requires one cohort-specific score contract",
    )
    score_contract = str(score_contracts[0])
    pooled = frame.groupby(["initialization", "method"], as_index=False).agg(
        score=("rps", "mean")
    )
    labels = sorted(pooled.initialization.unique().tolist())
    pivot = pooled.pivot(index="initialization", columns="method", values="score")
    pivot = pivot.reindex(index=labels, columns=METHOD_ORDER)
    _require(not pivot.isna().any().any(), f"{spec.name} pooled pairing is incomplete")
    dates = np.asarray(labels, dtype="datetime64[D]")
    if spec.bootstrap_kind == "equal_year_two_stage_circular_blocks":
        draws, diagnostics = _equal_year_draws(
            dates,
            draws=spec.bootstrap_draws,
            block_length=BLOCK_LENGTH,
            seed=spec.bootstrap_seed,
        )
    elif spec.bootstrap_kind == "unequal_year_two_stage_circular_blocks":
        draws, diagnostics = _unequal_year_draws(
            dates,
            draws=spec.bootstrap_draws,
            block_length=BLOCK_LENGTH,
            seed=spec.bootstrap_seed,
        )
    else:
        raise LeadGateError(f"unknown bootstrap kind: {spec.bootstrap_kind}")
    values = pivot.to_numpy(dtype=np.float64)
    method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    hybrid_index = method_index[HYBRID_METHOD]
    point_scores = np.mean(values, axis=0, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for baseline in CONTRASTS:
        baseline_index = method_index[baseline]
        effects = np.empty(len(draws), dtype=np.float64)
        for index, draw in enumerate(draws):
            sampled = values[draw]
            effects[index] = 1.0 - (
                np.mean(sampled[:, hybrid_index], dtype=np.float64)
                / np.mean(sampled[:, baseline_index], dtype=np.float64)
            )
        point = 1.0 - point_scores[hybrid_index] / point_scores[baseline_index]
        records.append(
            {
                "cohort": spec.name,
                "evidence_label": spec.evidence_label,
                "score_contract": score_contract,
                "policy": POLICY,
                "method": HYBRID_METHOD,
                "baseline": baseline,
                "method_rps": float(point_scores[hybrid_index]),
                "baseline_rps": float(point_scores[baseline_index]),
                "rps_reduction_fraction": float(point),
                "ci_lower_95": float(np.quantile(effects, 0.025)),
                "ci_upper_95": float(np.quantile(effects, 0.975)),
                "bootstrap_probability_improvement": float(np.mean(effects > 0.0)),
                "bootstrap_draws": len(draws),
                "bootstrap_seed": spec.bootstrap_seed,
                "block_length_initializations": BLOCK_LENGTH,
                "source_year_clusters": len(spec.expected_year_counts),
                "post_hoc_exploratory": True,
            }
        )
    return pd.DataFrame.from_records(records), diagnostics


def aggregate_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = (
        frame.groupby(
            ["cohort", "evidence_label", "score_contract", "method"],
            as_index=False,
        )
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
    )
    weekwise = (
        frame.groupby(
            [
                "cohort",
                "evidence_label",
                "score_contract",
                "method",
                "lead_week",
            ],
            as_index=False,
        )
        .agg(rps=("rps", "mean"), n_initializations=("initialization", "nunique"))
    )
    return pooled, weekwise


def acceptance_decision(bootstrap: pd.DataFrame) -> Mapping[str, Any]:
    pbc = bootstrap.loc[
        bootstrap.baseline.isin((PERSISTENCE_METHOD, COMBINED_METHOD))
    ].copy()
    expected_pairs = {
        (spec.name, baseline)
        for spec in COHORTS
        for baseline in (PERSISTENCE_METHOD, COMBINED_METHOD)
    }
    actual_pairs = set(zip(pbc.cohort, pbc.baseline, strict=True))
    _require(
        len(pbc) == len(expected_pairs)
        and not pbc.duplicated(["cohort", "baseline"]).any()
        and actual_pairs == expected_pairs,
        "PBC acceptance contrast pairs are incomplete or duplicated",
    )
    numerical = pbc[
        ["rps_reduction_fraction", "ci_lower_95", "ci_upper_95"]
    ].to_numpy(dtype=np.float64)
    _require(np.isfinite(numerical).all(), "PBC acceptance contrasts are non-finite")
    lower_scores = bool((pbc.rps_reduction_fraction > 0.0).all())
    intervals_above_zero = bool((pbc.ci_lower_95 > 0.0).all())
    success = lower_scores and intervals_above_zero
    return {
        "status": "descriptive_success" if success else "negative_or_unresolved",
        "passes": success,
        "rule": (
            "lower pooled RPS than Persistence++ and combined PBC in both cohorts, "
            "with all four paired 95% intervals above zero"
        ),
        "lower_score_gate": lower_scores,
        "paired_interval_gate": intervals_above_zero,
        "independent_superiority_claim_permitted": False,
        "reason": (
            "the gate was fixed after the input cohorts' leadwise results were known"
        ),
    }


def build_readme(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "# One-shot lead-gated neural--PBC diagnostic",
        "",
        f"Policy: `{POLICY}` (neural adapter at W1, Persistence++ at W2--W6).",
        "",
        "This is a score-only, hash-bound derivation from two already inspected",
        "cohorts. It is post-hoc exploratory evidence, not an independent test.",
        "No forecast, probability, parameter, model, or 2025 target was opened.",
        "",
        f"Acceptance status: **{decision['status']}**.",
        "",
        "## Pooled quintile RPS",
        "",
        "Absolute scores are interpreted only within a cohort because the later",
        "audit uses its own dynamic-support score contract.",
        "",
        "| Cohort | Contract | Hybrid | Persistence++ | Combined PBC | Neural | Raw FuXi |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for spec in COHORTS:
        rows = pooled.loc[pooled.cohort.eq(spec.name)].set_index("method")
        contract = str(rows.score_contract.iloc[0])
        lines.append(
            f"| {spec.name} | `{contract}` | "
            f"{rows.loc[HYBRID_METHOD, 'rps']:.6f} | "
            f"{rows.loc[PERSISTENCE_METHOD, 'rps']:.6f} | "
            f"{rows.loc[COMBINED_METHOD, 'rps']:.6f} | "
            f"{rows.loc[NEURAL_METHOD, 'rps']:.6f} | "
            f"{rows.loc[RAW_METHOD, 'rps']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired pooled effects",
            "",
            "Positive reduction favors the hybrid.",
            "",
            "| Cohort | Baseline | Reduction | 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.baseline} | "
            f"{100.0 * row.rps_reduction_fraction:.2f}% | "
            f"[{100.0 * row.ci_lower_95:.2f}, {100.0 * row.ci_upper_95:.2f}]% |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The permitted conclusion is that this fixed post-hoc lead policy has",
            "lower pooled RPS in both already-inspected cohorts. It must not be",
            "called independent evidence that the policy beats PBC. A separate",
            "one-time sealed-year protocol would be required for that claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> Mapping[str, Any]:
    """Build a fresh immutable diagnostic bundle."""

    output = Path(output).resolve()
    _require(sha256_file(PLAN_PATH) == PLAN_SHA256, "frozen plan SHA-256 changed")
    _require(not output.exists(), f"fresh output path required: {output}")
    for spec in COHORTS:
        input_root = spec.manifest_path.parent.resolve()
        try:
            output.relative_to(input_root)
        except ValueError:
            continue
        raise LeadGateError(f"output may not nest in immutable input: {input_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        case_frames: list[pd.DataFrame] = []
        bootstrap_frames: list[pd.DataFrame] = []
        bootstrap_diagnostics: dict[str, Any] = {}
        source_inputs: dict[str, Any] = {}
        for spec in COHORTS:
            manifest = validate_source_manifest(spec)
            frame = build_policy_case_scores(spec)
            paired, diagnostics = paired_bootstrap(frame, spec)
            case_frames.append(frame)
            bootstrap_frames.append(paired)
            bootstrap_diagnostics[spec.name] = diagnostics
            source_inputs[spec.name] = {
                "manifest": str(spec.manifest_path),
                "manifest_sha256": spec.manifest_sha256,
                "case_scores": str(spec.case_path),
                "case_scores_sha256": spec.case_sha256,
                "scientific_status": manifest.get("scientific_status"),
            }
        case_scores = pd.concat(case_frames, ignore_index=True)
        bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
        pooled, weekwise = aggregate_scores(case_scores)
        decision = acceptance_decision(bootstrap)

        metrics = stage / "metrics"
        metrics.mkdir()
        case_scores.to_csv(metrics / "case_scores.csv", index=False)
        pooled.to_csv(metrics / "pooled_rps.csv", index=False)
        weekwise.to_csv(metrics / "weekwise_rps.csv", index=False)
        bootstrap.to_csv(metrics / "paired_bootstrap.csv", index=False)
        (stage / "README.md").write_text(
            build_readme(pooled, bootstrap, decision), encoding="utf-8"
        )
        artifact_paths = (
            "README.md",
            "metrics/case_scores.csv",
            "metrics/pooled_rps.csv",
            "metrics/weekwise_rps.csv",
            "metrics/paired_bootstrap.csv",
        )
        artifact_hashes = {
            relative: sha256_file(stage / relative) for relative in artifact_paths
        }
        source_snapshot = {
            "plan/LEAD_GATED_NEURAL_PBC_20260823.md": sha256_file(PLAN_PATH),
            "src/fuxi_lead_gated_neural_pbc.py": sha256_file(SOURCE_PATH),
        }
        if TEST_PATH.is_file():
            source_snapshot["tests/test_fuxi_lead_gated_neural_pbc.py"] = sha256_file(
                TEST_PATH
            )
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": utc_now(),
            "mode": "full_score_only",
            "policy": {
                "name": POLICY,
                "label": POLICY_LABEL,
                "W1": NEURAL_METHOD,
                "W2_W6": PERSISTENCE_METHOD,
                "fitted_blend": False,
                "alternative_gate_search": False,
                "probabilities_or_forecasts_reconstructed": False,
                "neural_seed_handling": (
                    "accepted arithmetic mean of separately computed seed scores"
                ),
            },
            "scientific_status": (
                "post-hoc exploratory lead-router diagnostic on already inspected "
                "2020-2021 and 2022-2024 cohorts"
            ),
            "contract": {
                "sealed_2025_target_opened": False,
                "training_or_tuning_performed": False,
                "forecast_probability_arrays_opened": False,
                "only_accepted_case_score_tables_opened": True,
                "independent_superiority_claim_permitted": False,
                "score": "normalized informative-positive-cut quintile RPS",
            },
            "source_inputs": source_inputs,
            "bootstrap": bootstrap_diagnostics,
            "acceptance": decision,
            "source_snapshot_sha256": source_snapshot,
            "artifact_sha256": artifact_hashes,
            "output_path": str(output),
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
    print(f"acceptance: {manifest['acceptance']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
