#!/usr/bin/env bash
#SBATCH --job-name=fuxi-catcmp
#SBATCH --partition=gpu_prio
# CPU-only evaluator routed through the available S2S partition.
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CANONICAL_PBC_MANIFEST=${WORK_ROOT}/resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json
EXPECTED_PBC_MANIFEST_SHA256=c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6
NEURAL_MANIFEST=${WORK_ROOT}/resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z/manifest.json
EXPECTED_NEURAL_MANIFEST_SHA256=94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91

usage() {
    cat >&2 <<EOF
usage:
  sbatch slurm/evaluate_allseason_categorical_comparison.sbatch \\
    /absolute/path/completed-pbc-v2-full/manifest.json \\
    /absolute/path/fresh-comparison-output

Canonical completed PBC manifest:
  ${CANONICAL_PBC_MANIFEST}

The manifest path may point to an immutable relocation, but its SHA-256 must
equal ${EXPECTED_PBC_MANIFEST_SHA256} and its V2 Slurm receipt must pass.
EOF
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi
if [[ $1 != /* || $2 != /* ]]; then
    echo "PBC manifest and output must be absolute paths" >&2
    exit 2
fi
if [[ ! -f $1 || $(basename "$1") != manifest.json ]]; then
    echo "completed PBC V2 manifest is unavailable: $1" >&2
    exit 2
fi
if [[ ! -f ${NEURAL_MANIFEST} ]]; then
    echo "immutable neural manifest is unavailable: ${NEURAL_MANIFEST}" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/src/fuxi_allseason_categorical_comparison.py ]]; then
    echo "categorical comparison evaluator is unavailable" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/plan/CATEGORICAL_COMPARISON_20260822.md ]]; then
    echo "frozen categorical comparison protocol is unavailable" >&2
    exit 2
fi

PBC_MANIFEST=$(readlink -f "$1")
OUTPUT=$(readlink -m "$2")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "invalid or missing Slurm job id: ${SLURM_JOB_ID:-unset}" >&2
    exit 2
fi

JOB_TMP=$(mktemp -d "/tmp/fuxi-catcmp-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    if [[ -d ${JOB_TMP} && ${JOB_TMP} == /tmp/fuxi-catcmp-* ]]; then
        rm -rf -- "${JOB_TMP}"
    fi
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${WORK_ROOT}"

"${PYTHON_BIN}" - \
    "${PBC_MANIFEST}" \
    "${EXPECTED_PBC_MANIFEST_SHA256}" \
    "${NEURAL_MANIFEST}" \
    "${EXPECTED_NEURAL_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pbc_path = Path(sys.argv[1]).resolve()
expected_pbc_sha = sys.argv[2]
neural_path = Path(sys.argv[3]).resolve()
expected_neural_sha = sys.argv[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"categorical comparison preflight failed: {message}")


require(sha256(pbc_path) == expected_pbc_sha, "canonical PBC manifest SHA-256")
pbc = json.loads(pbc_path.read_text(encoding="utf-8"))
require(pbc.get("experiment") == "fuxi_allseason_pbc_baseline_v2", "PBC experiment")
require(pbc.get("status") == "complete", "PBC status")
require(pbc.get("mode") == "full" and pbc.get("smoke") is False, "PBC full mode")
require(
    pbc.get("scoring_contract_version") == "normalized_informative_positive_cut_v2",
    "PBC scoring contract",
)
require(
    pbc.get("split_counts_selected")
    == {"train": 1652, "validation": 196, "test": 208},
    "PBC split counts",
)
require(pbc.get("contract", {}).get("sealed_2025_target_opened") is False, "PBC 2025 seal")

receipt_path = pbc_path.parent / "slurm_gate_receipt.json"
require(receipt_path.is_file(), "PBC Slurm receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
for key, expected in {
    "experiment": "fuxi_allseason_pbc_baseline_v2",
    "gate_status": "passed",
    "mode": "full",
    "post_run_audit_version": "pbc_postrun_v2",
    "scoring_contract_version": "normalized_informative_positive_cut_v2",
    "manifest_sha256": expected_pbc_sha,
}.items():
    require(receipt.get(key) == expected, f"PBC receipt {key}")
require(
    receipt.get("persistence_lag_sha256")
    == pbc.get("persistence_lag_provenance", {}).get("sha256"),
    "PBC persistence provenance",
)
require(
    receipt.get("observation_bundle_sha256")
    == pbc.get("observation_bundle_provenance", {}).get("sha256"),
    "PBC observation provenance",
)

required_pbc = (
    "code/src/fuxi_pbc_core.py",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/scoring_support.npz",
    "models/pbc_fit.npz",
    "metrics/probability_bias.csv",
    "metrics/quintile_case_scores.csv",
    "metrics/semidecile_extreme_metrics.csv",
    "metrics/weekwise_metrics.csv",
)
for relative in required_pbc:
    artifact = pbc_path.parent / relative
    expected = pbc.get("artifact_sha256", {}).get(relative)
    require(artifact.is_file() and isinstance(expected, str), f"PBC artifact {relative}")
    require(sha256(artifact) == expected, f"PBC artifact SHA-256 {relative}")

require(sha256(neural_path) == expected_neural_sha, "immutable neural manifest SHA-256")
neural = json.loads(neural_path.read_text(encoding="utf-8"))
require(neural.get("experiment") == "fuxi_allseason_ensemble_calibration_v1", "neural experiment")
require(neural.get("status") == "complete", "neural status")
require(neural.get("mode") == "full" and neural.get("smoke") is False, "neural full mode")
require(neural.get("seeds") == [42, 43, 44], "neural seeds")
require(
    {"summary_only", "location_spread"}.issubset(neural.get("configurations", [])),
    "neural configurations",
)
require(
    neural.get("split_counts_selected")
    == {"train": 1652, "validation": 196, "test": 208},
    "neural split counts",
)
require(neural.get("contract", {}).get("sealed_2025_target_opened") is False, "neural 2025 seal")
require(
    neural.get("cache", {}).get("data_sha256")
    == pbc.get("cache", {}).get("data_sha256"),
    "shared member-cache SHA-256",
)
required_neural = [
    "code/src/fuxi_allseason_ensemble_calibration.py",
    "evaluation/scoring_support.npz",
    "metrics/case_metrics.csv",
    "metrics/seed_case_metrics.csv",
    "models/moment_calibration_fit.npz",
]
required_neural.extend(
    f"models/{configuration}/seed_{seed}/test_adjustments.npz"
    for configuration in ("summary_only", "location_spread")
    for seed in (42, 43, 44)
)
for relative in required_neural:
    artifact = neural_path.parent / relative
    expected = neural.get("artifact_sha256", {}).get(relative)
    require(artifact.is_file() and isinstance(expected, str), f"neural artifact {relative}")
    require(sha256(artifact) == expected, f"neural artifact SHA-256 {relative}")

print(f"accepted completed PBC V2 full: {pbc_path}")
print(f"accepted immutable neural full: {neural_path}")
PY

echo "CPU categorical comparison node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK}"
srun --cpu-bind=cores "${PYTHON_BIN}" -u \
    src/fuxi_allseason_categorical_comparison.py \
    --pbc-manifest "${PBC_MANIFEST}" \
    --neural-manifest "${NEURAL_MANIFEST}" \
    --output "${OUTPUT}" \
    --bootstrap-samples 2000 \
    --block-length 13 \
    --cdf-chunk-size 8

"${PYTHON_BIN}" - \
    "${OUTPUT}/manifest.json" \
    "${EXPECTED_PBC_MANIFEST_SHA256}" \
    "${EXPECTED_NEURAL_MANIFEST_SHA256}" \
    "${WORK_ROOT}" \
    "${SLURM_JOB_ID}" \
    "${SLURMD_NODENAME:-unknown}" \
    "${SLURM_JOB_PARTITION:-gpu_prio}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

manifest_path = Path(sys.argv[1]).resolve()
expected_pbc_sha = sys.argv[2]
expected_neural_sha = sys.argv[3]
work_root = Path(sys.argv[4])
job_id, node, partition = sys.argv[5:8]
root = manifest_path.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"categorical comparison postflight failed: {message}")


def within_two_float64_ulps(left: float, right: float) -> bool:
    """Allow only CSV reaggregation drift of at most two float64 steps."""

    first = np.float64(left)
    second = np.float64(right)
    tolerance = 2.0 * max(abs(np.spacing(first)), abs(np.spacing(second)))
    return bool(abs(first - second) <= tolerance)


require(manifest_path.is_file(), "manifest exists")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require(manifest.get("experiment") == "fuxi_allseason_categorical_comparison_v2", "experiment")
require(manifest.get("status") == "complete", "status")
require(manifest.get("mode") == "full" and manifest.get("smoke") is False, "full mode")
require(
    manifest.get("scoring_contract_version") == "normalized_informative_positive_cut_v2",
    "scoring contract",
)
require(manifest.get("contract", {}).get("test_years") == [2020, 2021], "test years")
require(manifest.get("contract", {}).get("sealed_2025_target_opened") is False, "2025 seal")
require(manifest.get("contract", {}).get("retraining_performed") is False, "no retraining")
require(
    manifest.get("split_counts_selected")
    == {"train": 1652, "validation": 196, "test": 208},
    "split counts",
)
evaluation = manifest.get("evaluation", {})
require(evaluation.get("bootstrap_samples") == 2000, "manifest bootstrap samples")
require(
    evaluation.get("bootstrap_block_length_initializations") == 13,
    "manifest bootstrap block length",
)
require(evaluation.get("bootstrap_seed") == 20260822, "manifest bootstrap seed")
require(
    evaluation.get("bootstrap_scheme")
    == "two_stage_year_resample_then_within_year_circular_blocks",
    "manifest bootstrap scheme",
)
inference_status = evaluation.get("bootstrap_inference_status", "")
require(
    "exploratory" in inference_status and "two year clusters" in inference_status,
    "manifest two-year bootstrap caveat",
)
require(
    evaluation.get("probability_bias_metric")
    == (
        "generic quintile and semidecile CDF-bias curves on ordinary supported "
        "CDF geometry; separate from and not labeled positive-q95 event bias"
    ),
    "manifest generic CDF probability-bias status",
)
methods = [
    "raw_fuxi_categorical",
    "moment_calibration",
    "summary_only",
    "location_spread",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
]
require(manifest.get("methods") == methods, "methods")
seed_rule = manifest.get("seed_aggregation", {})
for key in ("forecast_averaging", "parameter_averaging", "forecast_arrays_written", "learned_parameters_written"):
    require(seed_rule.get(key) is False, f"seed aggregation {key}")
gates = manifest.get("input_hash_gates", {})
for key in (
    "frozen_neural_manifest_match",
    "pbc_manifest_bound_to_successful_v2_full_slurm_receipt",
    "pbc_daily_issue_time_lag_contract_verified",
    "pbc_observation_bundle_recomputed_exactly",
    "member_cache_streamed_sha256_verified",
    "live_pbc_core_matches_completed_pbc_snapshot",
    "neural_reconstruction_helpers_match_frozen_snapshot",
):
    require(gates.get(key) is True, f"input gate {key}")
receipts = manifest.get("input_receipts", {})
require(receipts.get("pbc_manifest", {}).get("sha256") == expected_pbc_sha, "PBC input SHA")
require(receipts.get("neural_manifest", {}).get("sha256") == expected_neural_sha, "neural input SHA")
require(
    receipts.get("pbc_slurm_gate_receipt", {}).get("post_run_audit_version")
    == "pbc_postrun_v2",
    "PBC receipt audit version",
)

required = (
    "README.md",
    "receipts/input_receipts.json",
    "metrics/neural_seed_quintile_case_scores.csv",
    "metrics/neural_seed_upper_extreme_case_scores.csv",
    "metrics/neural_seed_probability_bias.csv",
    "metrics/quintile_case_scores.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/seasonal_weekwise_metrics.csv",
    "metrics/pooled_metrics.csv",
    "metrics/upper_extreme_metrics.csv",
    "metrics/probability_bias.csv",
    "metrics/paired_block_bootstrap.csv",
    "code/src/fuxi_allseason_categorical_comparison.py",
    "code/src/fuxi_pbc_core.py",
    "code/slurm/evaluate_allseason_categorical_comparison.sbatch",
    "code/plan/CATEGORICAL_COMPARISON_20260822.md",
)
artifact_hashes = manifest.get("artifact_sha256", {})
for relative in required:
    path = root / relative
    require(path.is_file(), f"artifact {relative}")
    require(artifact_hashes.get(relative) == sha256(path), f"artifact SHA-256 {relative}")
actual_inventory = {
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file() and path.name != "manifest.json"
}
require(actual_inventory == set(artifact_hashes), "exact artifact inventory")
for relative, expected in artifact_hashes.items():
    require(sha256(root / relative) == expected, f"exhaustive artifact SHA-256 {relative}")

# A passed receipt must describe the exact source bundle that executed.  Reject
# any drift between live sources, the run's copied snapshots, the manifest's
# source map, and the exhaustive artifact inventory before publishing it.
source_live_paths = {
    "code/src/fuxi_allseason_categorical_comparison.py": work_root
    / "src/fuxi_allseason_categorical_comparison.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root
    / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_allseason_member_cache.py": work_root
    / "src/fuxi_allseason_member_cache.py",
    "code/src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_pbc_baseline.py": work_root
    / "src/fuxi_allseason_pbc_baseline.py",
    "code/slurm/evaluate_allseason_categorical_comparison.sbatch": work_root
    / "slurm/evaluate_allseason_categorical_comparison.sbatch",
    "code/plan/CATEGORICAL_COMPARISON_20260822.md": work_root
    / "plan/CATEGORICAL_COMPARISON_20260822.md",
}
source_snapshot = manifest.get("source_snapshot_sha256", {})
require(
    set(source_snapshot) == set(source_live_paths),
    "exact source-snapshot inventory",
)
live_source_hashes = {}
for relative, live_path in source_live_paths.items():
    snapshot_path = root / relative
    require(live_path.is_file(), f"live source exists: {relative}")
    snapshot_sha = sha256(snapshot_path)
    live_sha = sha256(live_path)
    require(source_snapshot.get(relative) == snapshot_sha, f"manifest source snapshot: {relative}")
    require(artifact_hashes.get(relative) == snapshot_sha, f"artifact/source snapshot: {relative}")
    require(live_sha == snapshot_sha, f"mid-run live source drift: {relative}")
    live_source_hashes[relative] = live_sha
source_snapshot_bundle_sha256 = hashlib.sha256(
    json.dumps(
        live_source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

case_scores = pd.read_csv(root / "metrics/quintile_case_scores.csv")
require(len(case_scores) == 7 * 208 * 6, "case-score row count")
require(set(case_scores.method) == set(methods), "case-score methods")
require(set(case_scores.lead_week.astype(int)) == set(range(1, 7)), "case-score leads")
require(case_scores.initialization.nunique() == 208, "case-score initializations")
require(not case_scores.duplicated(["method", "initialization", "lead_week"]).any(), "case-score pairing")
require(
    set(case_scores.score_contract) == {"normalized_informative_positive_cut_v2"},
    "case-score contract",
)
require(case_scores.rps.between(0.0, 1.0).all(), "primary RPS bounds")
require(case_scores.paper_nominal_cut_q80_positive_rps.between(0.0, 4.0).all(), "paper RPS bounds")
require((case_scores.paper_q80_retained_scoring_weight > 0.0).all(), "q80 retained weight")
for column in (
    "nominal_climatology_rps",
    "training_empirical_climatology_rps",
    "paper_fixed_nominal_climatology_rps",
):
    require(np.isfinite(case_scores[column]).all() and (case_scores[column] > 0.0).all(), column)
require(
    np.allclose(
        case_scores.rpss_vs_training_empirical_climatology_case,
        1.0 - case_scores.rps / case_scores.training_empirical_climatology_rps,
        rtol=1e-10,
        atol=1e-10,
    ),
    "primary RPSS identity",
)
require(
    np.allclose(
        case_scores.paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case,
        1.0
        - case_scores.paper_nominal_cut_q80_positive_rps
        / case_scores.paper_fixed_nominal_climatology_rps,
        rtol=1e-10,
        atol=1e-10,
    ),
    "paper RPSS identity",
)
for column in ("initialization", "verification_start", "verification_midpoint", "verification_end"):
    require(not case_scores[column].astype(str).str.startswith("2025-").any(), f"2025 in {column}")

seed_scores = pd.read_csv(root / "metrics/neural_seed_quintile_case_scores.csv")
require(len(seed_scores) == 2 * 3 * 208 * 6, "per-seed row count")
require(set(seed_scores.method) == {"summary_only", "location_spread"}, "per-seed methods")
require(set(seed_scores.seed.astype(int)) == {42, 43, 44}, "per-seed seeds")
require(not seed_scores.duplicated(["method", "seed", "initialization", "lead_week"]).any(), "per-seed pairing")
score_columns = ["rps", "paper_nominal_cut_q80_positive_rps"]
reference_columns = [
    "nominal_climatology_rps",
    "training_empirical_climatology_rps",
    "paper_fixed_nominal_climatology_rps",
    "paper_q80_retained_scoring_weight",
    "paper_q80_total_supported_scoring_weight",
    "paper_q80_retained_scoring_weight_fraction",
]
seed_reference_spread = seed_scores.groupby(
    ["method", "initialization", "lead_week"]
)[reference_columns].agg(lambda values: float(values.max() - values.min()))
require(np.allclose(seed_reference_spread, 0.0, rtol=0.0, atol=1e-10), "per-seed reference/support identity")
seed_means = seed_scores.groupby(
    ["method", "initialization", "lead_week"], as_index=False
).agg(
    **{column: (column, "mean") for column in score_columns},
    **{column: (column, "first") for column in reference_columns},
)
stored_neural = case_scores.loc[
    case_scores.method.isin(["summary_only", "location_spread"]),
    ["method", "initialization", "lead_week", *score_columns, *reference_columns],
]
joined_neural = stored_neural.merge(
    seed_means,
    on=["method", "initialization", "lead_week"],
    suffixes=("_stored", "_seed_mean"),
    validate="one_to_one",
)
require(len(joined_neural) == 2 * 208 * 6, "averaged-neural paired rows")
for column in score_columns:
    require(
        np.allclose(
            joined_neural[f"{column}_stored"],
            joined_neural[f"{column}_seed_mean"],
            rtol=0.0,
            atol=1e-12,
        ),
        f"neural arithmetic mean of per-seed scores: {column}",
    )
for column in reference_columns:
    require(
        np.allclose(
            joined_neural[f"{column}_stored"],
            joined_neural[f"{column}_seed_mean"],
            rtol=0.0,
            atol=1e-10,
        ),
        f"neural threshold-derived reference/support: {column}",
    )

pooled = pd.read_csv(root / "metrics/pooled_metrics.csv")
weekwise = pd.read_csv(root / "metrics/weekwise_metrics.csv")
require(len(pooled) == 7 and set(pooled.method) == set(methods), "pooled rows")
require(len(weekwise) == 42 and set(weekwise.method) == set(methods), "weekwise rows")


def recompute_quintile_aggregate(source: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    prepared = source.assign(
        _paper_score_sum=(
            source.paper_nominal_cut_q80_positive_rps
            * source.paper_q80_retained_scoring_weight
        ),
        _paper_reference_sum=(
            source.paper_fixed_nominal_climatology_rps
            * source.paper_q80_retained_scoring_weight
        ),
    )
    result = prepared.groupby([*group_columns, "method"], as_index=False).agg(
        rps=("rps", "mean"),
        nominal_climatology_rps=("nominal_climatology_rps", "mean"),
        training_empirical_climatology_rps=(
            "training_empirical_climatology_rps",
            "mean",
        ),
        paper_q80_retained_scoring_weight=(
            "paper_q80_retained_scoring_weight",
            "sum",
        ),
        _paper_score_sum=("_paper_score_sum", "sum"),
        _paper_reference_sum=("_paper_reference_sum", "sum"),
        n_initializations=("initialization", "nunique"),
    )
    result["paper_nominal_cut_q80_positive_rps"] = (
        result._paper_score_sum / result.paper_q80_retained_scoring_weight
    )
    result["paper_fixed_nominal_climatology_rps"] = (
        result._paper_reference_sum / result.paper_q80_retained_scoring_weight
    )
    return result.drop(columns=["_paper_score_sum", "_paper_reference_sum"])


aggregate_columns = [
    "rps",
    "nominal_climatology_rps",
    "training_empirical_climatology_rps",
    "paper_q80_retained_scoring_weight",
    "paper_nominal_cut_q80_positive_rps",
    "paper_fixed_nominal_climatology_rps",
    "n_initializations",
]
for stored, groups, label in (
    (pooled, [], "pooled"),
    (weekwise, ["lead_week"], "weekwise"),
):
    recomputed = recompute_quintile_aggregate(case_scores, groups)
    joined = stored.merge(
        recomputed,
        on=[*groups, "method"],
        suffixes=("_stored", "_recomputed"),
        validate="one_to_one",
    )
    require(len(joined) == len(stored), f"{label} aggregate keys")
    for column in aggregate_columns:
        require(
            np.allclose(
                joined[f"{column}_stored"],
                joined[f"{column}_recomputed"],
                rtol=0.0,
                atol=1e-8 if column == "paper_q80_retained_scoring_weight" else 1e-12,
            ),
            f"{label} aggregate identity: {column}",
        )
for frame, label in ((pooled, "pooled"), (weekwise, "weekwise")):
    require(
        np.allclose(
            frame.paper_nominal_cut_rpss_vs_fixed_nominal_climatology,
            1.0
            - frame.paper_nominal_cut_q80_positive_rps
            / frame.paper_fixed_nominal_climatology_rps,
            rtol=1e-10,
            atol=1e-10,
        ),
        f"{label} retained-weight paper identity",
    )

upper = pd.read_csv(root / "metrics/upper_extreme_metrics.csv")
require(set(upper.event) == {"upper_5pct_informative_positive_q95"}, "upper event")
require(set(upper.status) == {"scored_on_positive_q95_cells_only"}, "upper status")
require(set(upper.method) == set(methods), "upper methods")
require(
    np.allclose(
        upper.bss_vs_training_empirical_climatology,
        1.0 - upper.brier_score / upper.training_empirical_climatology_brier_score,
        rtol=1e-10,
        atol=1e-10,
    ),
    "upper empirical BSS identity",
)
seed_upper = pd.read_csv(root / "metrics/neural_seed_upper_extreme_case_scores.csv")
upper_value_columns = [
    "brier_score",
    "nominal_climatology_brier_score",
    "training_empirical_climatology_brier_score",
]
require(len(seed_upper) == 2 * 3 * 208 * 6, "per-seed upper row count")
upper_reference_spread = seed_upper.groupby(
    ["method", "initialization", "lead_week"]
)[upper_value_columns[1:]].agg(lambda values: float(values.max() - values.min()))
require(np.allclose(upper_reference_spread, 0.0, rtol=0.0, atol=1e-12), "upper per-seed references")
mean_upper_cases = seed_upper.groupby(
    ["method", "initialization", "lead_week", "season"], as_index=False
).agg(
    brier_score=("brier_score", "mean"),
    nominal_climatology_brier_score=(
        "nominal_climatology_brier_score",
        "first",
    ),
    training_empirical_climatology_brier_score=(
        "training_empirical_climatology_brier_score",
        "first",
    ),
)
upper_records = []
for (method, lead), rows in mean_upper_cases.groupby(["method", "lead_week"]):
    for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
        chosen = rows if season == "ALL" else rows.loc[rows.season == season]
        if chosen.empty:
            continue
        upper_records.append(
            {
                "method": method,
                "lead_week": int(lead),
                "season": season,
                **{column: float(chosen[column].mean()) for column in upper_value_columns},
            }
        )
recomputed_upper = pd.DataFrame.from_records(upper_records)
stored_neural_upper = upper.loc[
    upper.method.isin(["summary_only", "location_spread"])
]
joined_upper = stored_neural_upper.merge(
    recomputed_upper,
    on=["method", "lead_week", "season"],
    suffixes=("_stored", "_recomputed"),
    validate="one_to_one",
)
require(len(joined_upper) == len(stored_neural_upper), "neural upper aggregate keys")
for column in upper_value_columns:
    require(
        np.allclose(
            joined_upper[f"{column}_stored"],
            joined_upper[f"{column}_recomputed"],
            rtol=0.0,
            atol=1e-12,
        ),
        f"neural upper per-seed score aggregation: {column}",
    )

probability_bias = pd.read_csv(root / "metrics/probability_bias.csv")
seed_bias = pd.read_csv(root / "metrics/neural_seed_probability_bias.csv")
bias_keys = [
    "family",
    "method",
    "lead_week",
    "season",
    "nominal_cumulative_probability",
]
bias_values = ["probability_bias", "mean_absolute_case_probability_bias"]
mean_seed_bias = seed_bias.groupby(bias_keys, as_index=False)[bias_values].mean()
stored_neural_bias = probability_bias.loc[
    probability_bias.method.isin(["summary_only", "location_spread"])
]
joined_bias = stored_neural_bias.merge(
    mean_seed_bias,
    on=bias_keys,
    suffixes=("_stored", "_seed_mean"),
    validate="one_to_one",
)
require(len(joined_bias) == len(stored_neural_bias), "neural probability-bias keys")
for column in bias_values:
    require(
        np.allclose(
            joined_bias[f"{column}_stored"],
            joined_bias[f"{column}_seed_mean"],
            rtol=0.0,
            atol=1e-12,
        ),
        f"neural probability-bias seed aggregation: {column}",
    )

bootstrap = pd.read_csv(root / "metrics/paired_block_bootstrap.csv")
require(len(bootstrap) == 15 * 7 * 2, "bootstrap row count")
require(
    set(bootstrap.bootstrap_scheme)
    == {"two_stage_year_resample_then_within_year_circular_blocks"},
    "bootstrap scheme",
)
require(set(bootstrap.test_year_count.astype(int)) == {2}, "bootstrap years")
require(set(bootstrap.initializations_per_test_year.astype(int)) == {104}, "bootstrap year counts")
require(set(bootstrap.bootstrap_samples.astype(int)) == {2000}, "bootstrap samples")
require(set(bootstrap.block_length_initializations.astype(int)) == {13}, "bootstrap block length")
require(set(bootstrap.bootstrap_seed.astype(int)) == {20260822}, "bootstrap seed")
require(
    set(bootstrap.lead_scope) == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"},
    "bootstrap lead scopes",
)
require(
    set(bootstrap.metric)
    == {
        "primary_normalized_informative_positive_cut_rps",
        "secondary_guan_nominal_cut_q80_positive_rps",
    },
    "bootstrap metrics",
)
require(np.isfinite(bootstrap[["ci_lower_95", "ci_upper_95"]]).all().all(), "bootstrap CIs")
require((bootstrap.ci_lower_95 <= bootstrap.ci_upper_95).all(), "ordered bootstrap CIs")
require(
    bootstrap.bootstrap_probability_improvement.between(0.0, 1.0).all(),
    "bootstrap improvement-probability bounds",
)
expected_pairs = {
    (method, baseline)
    for method in ("moment_calibration", "summary_only", "location_spread")
    for baseline in (
        "raw_fuxi_categorical",
        "debias_plus_plus",
        "persistence_plus_plus",
        "pbc_combined",
    )
} | {
    ("pbc_combined", "raw_fuxi_categorical"),
    ("pbc_combined", "debias_plus_plus"),
    ("pbc_combined", "persistence_plus_plus"),
}
for (metric, scope), rows in bootstrap.groupby(["metric", "lead_scope"]):
    require(
        set(zip(rows.method, rows.baseline)) == expected_pairs,
        f"bootstrap comparison pairs {metric}/{scope}",
    )
    require(
        not rows.duplicated(["method", "baseline"]).any(),
        f"bootstrap duplicate comparisons {metric}/{scope}",
    )


# Independently rebuild the frozen two-stage draw geometry from the case CSV.
# This deliberately does not import the evaluator's bootstrap implementation.
bootstrap_seed = 20260822
bootstrap_samples = 2000
bootstrap_block_length = 13
case_order = np.asarray(
    sorted(case_scores.initialization.unique()), dtype="datetime64[D]"
)
case_labels = [np.datetime_as_string(value, unit="D") for value in case_order]
case_years = pd.DatetimeIndex(case_order).year.to_numpy()
year_groups = [
    np.flatnonzero(case_years == year) for year in np.sort(np.unique(case_years))
]
require(
    len(year_groups) == 2 and {len(group) for group in year_groups} == {104},
    "bootstrap source year geometry",
)
random = np.random.default_rng(bootstrap_seed)
block_count = int(np.ceil(len(year_groups[0]) / bootstrap_block_length))
block_offsets = np.arange(bootstrap_block_length, dtype=np.int64)
draws = np.empty((bootstrap_samples, len(case_order)), dtype=np.int64)
for draw_index in range(bootstrap_samples):
    sampled_years = random.integers(0, len(year_groups), size=len(year_groups))
    cursor = 0
    for sampled_year in sampled_years:
        group = year_groups[int(sampled_year)]
        starts = random.integers(0, len(group), size=block_count)
        local = (
            (starts[:, None] + block_offsets[None]) % len(group)
        ).reshape(-1)
        chosen = group[local[: len(group)]]
        draws[draw_index, cursor : cursor + len(group)] = chosen
        cursor += len(group)


def paired_scores_and_draws(
    metric: str, scope: str
) -> dict[str, tuple[float, np.ndarray, float]]:
    selected = case_scores
    if scope != "W1-W6":
        selected = selected.loc[selected.lead_week == int(scope[1:])]
    if metric == "primary_normalized_informative_positive_cut_rps":
        if scope == "W1-W6":
            prepared = selected.groupby(["initialization", "method"], as_index=False).agg(
                score=("rps", "mean")
            )
        else:
            prepared = selected.rename(columns={"rps": "score"})
        pivot = prepared.pivot(
            index="initialization", columns="method", values="score"
        ).reindex(case_labels)
        require(not pivot.isna().any().any(), f"primary bootstrap pairing {scope}")
        return {
            method: (
                float(pivot[method].mean()),
                np.mean(pivot[method].to_numpy(dtype=np.float64)[draws], axis=1),
                float("nan"),
            )
            for method in methods
        }
    prepared = selected.assign(
        numerator=(
            selected.paper_nominal_cut_q80_positive_rps
            * selected.paper_q80_retained_scoring_weight
        ),
        score=selected.paper_nominal_cut_q80_positive_rps,
        weight=selected.paper_q80_retained_scoring_weight,
    )
    if scope == "W1-W6":
        prepared = prepared.groupby(
            ["initialization", "method"], as_index=False
        ).agg(
            numerator=("numerator", "sum"),
            weight=("weight", "sum"),
        )
        prepared["score"] = prepared.numerator / prepared.weight
    score_pivot = prepared.pivot(
        index="initialization", columns="method", values="score"
    ).reindex(case_labels)
    weight_pivot = prepared.pivot(
        index="initialization", columns="method", values="weight"
    ).reindex(case_labels)
    require(
        not score_pivot.isna().any().any() and not weight_pivot.isna().any().any(),
        f"paper bootstrap pairing {scope}",
    )
    weight_matrix = weight_pivot.to_numpy(dtype=np.float64)
    require(
        np.allclose(weight_matrix, weight_matrix[:, :1], rtol=0.0, atol=1e-8),
        f"paper bootstrap common support {scope}",
    )
    return {
        method: (
            float(
                np.sum(
                    score_pivot[method].to_numpy(dtype=np.float64)
                    * weight_pivot[method].to_numpy(dtype=np.float64)
                )
                / np.sum(weight_pivot[method].to_numpy(dtype=np.float64))
            ),
            np.sum(
                score_pivot[method].to_numpy(dtype=np.float64)[draws]
                * weight_pivot[method].to_numpy(dtype=np.float64)[draws],
                axis=1,
            )
            / np.sum(
                weight_pivot[method].to_numpy(dtype=np.float64)[draws], axis=1
            ),
            float(np.sum(weight_pivot[method].to_numpy(dtype=np.float64))),
        )
        for method in methods
    }


bootstrap_cache = {
    (metric, scope): paired_scores_and_draws(metric, scope)
    for metric in bootstrap.metric.unique()
    for scope in bootstrap.lead_scope.unique()
}
for row in bootstrap.itertuples(index=False):
    scores = bootstrap_cache[(row.metric, row.lead_scope)]
    method_score, method_draw, retained_weight = scores[row.method]
    baseline_score, baseline_draw, _ = scores[row.baseline]
    effect = 1.0 - method_score / baseline_score
    reductions = 1.0 - method_draw / baseline_draw
    ci_lower, ci_upper = np.quantile(reductions, (0.025, 0.975))
    improvement_probability = float(np.mean(reductions > 0.0))
    require(np.isclose(row.method_score, method_score, rtol=0.0, atol=1e-12), "bootstrap method point score")
    require(np.isclose(row.baseline_score, baseline_score, rtol=0.0, atol=1e-12), "bootstrap baseline point score")
    require(np.isclose(row.rps_reduction_fraction, effect, rtol=0.0, atol=1e-12), "bootstrap point effect")
    require(np.isclose(row.ci_lower_95, ci_lower, rtol=0.0, atol=1e-12), "bootstrap lower CI")
    require(np.isclose(row.ci_upper_95, ci_upper, rtol=0.0, atol=1e-12), "bootstrap upper CI")
    require(
        np.isclose(
            row.bootstrap_probability_improvement,
            improvement_probability,
            rtol=0.0,
            atol=1e-12,
        ),
        "bootstrap improvement probability",
    )
    if row.metric.startswith("primary"):
        require(pd.isna(row.paper_retained_scoring_weight), "primary bootstrap retained weight")
    else:
        require(
            within_two_float64_ulps(
                row.paper_retained_scoring_weight,
                retained_weight,
            ),
            "paper bootstrap retained weight",
        )
    require(
        (effect > 0.0) == (method_score < baseline_score)
        and (effect < 0.0) == (method_score > baseline_score),
        "bootstrap point direction/sign",
    )

# Outputs are deliberately score-only. Source snapshots are text; no forecast
# or learned-parameter array is allowed in the published comparison directory.
unexpected_arrays = [
    path
    for path in root.rglob("*")
    if path.is_file() and path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".ckpt"}
]
require(not unexpected_arrays, f"unexpected saved arrays: {unexpected_arrays}")

receipt_payload = {
    "experiment": "fuxi_allseason_categorical_comparison_v2",
    "gate_status": "passed",
    "post_run_audit_version": "categorical_comparison_postrun_v2",
    "scoring_contract_version": "normalized_informative_positive_cut_v2",
    "manifest_sha256": sha256(manifest_path),
    "pbc_manifest_sha256": expected_pbc_sha,
    "neural_manifest_sha256": expected_neural_sha,
    "bootstrap_seed": 20260822,
    "driver_sha256": live_source_hashes[
        "code/src/fuxi_allseason_categorical_comparison.py"
    ],
    "launcher_sha256": live_source_hashes[
        "code/slurm/evaluate_allseason_categorical_comparison.sbatch"
    ],
    "source_snapshot_bundle_sha256": source_snapshot_bundle_sha256,
    "source_snapshot_sha256": live_source_hashes,
    "job_id": job_id,
    "node": node,
    "partition": partition,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
}
receipt_path = root / "slurm_gate_receipt.json"
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".slurm_gate_receipt.json.", suffix=".temporary", dir=root
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)
finally:
    temporary.unlink(missing_ok=True)

print(f"PASS: receipt-gated categorical comparison at {root}")
PY
