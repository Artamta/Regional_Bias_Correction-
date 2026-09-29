#!/usr/bin/env bash
#SBATCH --job-name=fuxi-persist
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch slurm/run_allseason_persistence_augmented.sbatch \
    smoke /absolute/path/full-cache.npy /absolute/path/fresh-smoke-output

  sbatch slurm/run_allseason_persistence_augmented.sbatch \
    full /absolute/path/full-cache.npy /absolute/path/fresh-full-output \
    /absolute/path/completed-smoke/manifest.json

Both modes require the canonical full 2,080-case cache. Full mode additionally
requires the successful smoke manifest produced by this exact source/plan bundle.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

MODE=$1
case ${MODE} in
    smoke)
        if [[ $# -ne 3 ]]; then
            usage
            exit 2
        fi
        ;;
    full)
        if [[ $# -ne 4 ]]; then
            usage
            exit 2
        fi
        ;;
    *)
        echo "mode must be smoke or full: ${MODE}" >&2
        usage
        exit 2
        ;;
esac

CACHE_INPUT=$2
OUTPUT_INPUT=$3
if [[ ${CACHE_INPUT} != /* || ${OUTPUT_INPUT} != /* ]]; then
    echo "cache and output must be explicit absolute paths" >&2
    exit 2
fi
if [[ ! -f ${CACHE_INPUT} ]]; then
    echo "member cache does not exist: ${CACHE_INPUT}" >&2
    exit 2
fi

SMOKE_MANIFEST=
if [[ ${MODE} == full ]]; then
    SMOKE_INPUT=$4
    if [[ ${SMOKE_INPUT} != /* ]]; then
        echo "completed smoke manifest must be an absolute path" >&2
        exit 2
    fi
    if [[ ! -f ${SMOKE_INPUT} ]]; then
        echo "completed smoke manifest does not exist: ${SMOKE_INPUT}" >&2
        exit 2
    fi
    SMOKE_MANIFEST=$(readlink -f "${SMOKE_INPUT}")
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE=$(readlink -f "${CACHE_INPUT}")
OUTPUT=$(readlink -m "${OUTPUT_INPUT}")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
OUTPUT_NAME=$(basename "${OUTPUT}")
ARMS=base_42k,zero_lag_45k,persistence_lag12_45k
EXPECTED_CACHE_SHA256=2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70
EXPECTED_SOURCE_FINGERPRINT=655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae
EXPECTED_PLAN_SHA256=2a1c6d16957cbceedc0bcf62cb2a7518ca1e5e161f8df36aab0f600e28302699

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi

REQUIRED_SOURCES=(
    src/fuxi_allseason_persistence_augmented.py
    src/fuxi_persistence_context.py
    src/fuxi_allseason_ensemble_calibration.py
    src/fuxi_ensemble_calibration_core.py
    src/fuxi_allseason_pbc_baseline.py
    src/fuxi_pbc_core.py
    src/fuxi_allseason_member_cache.py
    slurm/run_allseason_persistence_augmented.sbatch
    plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md
)
for relative in "${REQUIRED_SOURCES[@]}"; do
    if [[ ! -f ${WORK_ROOT}/${relative} ]]; then
        echo "required frozen source/plan is unavailable: ${relative}" >&2
        exit 2
    fi
done
ACTUAL_PLAN_SHA256=$(
    "${PYTHON_BIN}" - "${WORK_ROOT}/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
if [[ ${ACTUAL_PLAN_SHA256} != "${EXPECTED_PLAN_SHA256}" ]]; then
    echo "frozen pre-score plan SHA-256 differs: ${ACTUAL_PLAN_SHA256}" >&2
    exit 2
fi
if ! command -v srun >/dev/null 2>&1; then
    echo "srun is unavailable" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable" >&2
    exit 2
fi

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}"
if [[ -e ${OUTPUT} ]]; then
    echo "output appeared during preflight; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi

JOB_ID=${SLURM_JOB_ID:?this launcher must be submitted with sbatch}
JOB_TMP=$(mktemp -d "/tmp/fuxi-persistence-${JOB_ID}.XXXXXX")
OUTPUT_LOCK=${OUTPUT_PARENT}/.${OUTPUT_NAME}.persistence-launch.lock
LOCK_HELD=false
cleanup() {
    rm -rf -- "${JOB_TMP}"
    if [[ ${LOCK_HELD} == true ]]; then
        rmdir -- "${OUTPUT_LOCK}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if ! mkdir -- "${OUTPUT_LOCK}"; then
    echo "another launch owns this output path: ${OUTPUT}" >&2
    exit 2
fi
LOCK_HELD=true

cd "${WORK_ROOT}"
"${PYTHON_BIN}" src/fuxi_allseason_member_cache.py verify --output "${CACHE}"
"${PYTHON_BIN}" - "${CACHE}" "${EXPECTED_CACHE_SHA256}" \
    "${EXPECTED_SOURCE_FINGERPRINT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
expected_cache_sha = sys.argv[2]
expected_source_fingerprint = sys.argv[3]
metadata_path = cache.with_suffix("").with_suffix(".metadata.json")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"cache contract failed: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

require(metadata_path.is_file(), f"missing metadata {metadata_path}")
with metadata_path.open("r", encoding="utf-8") as stream:
    metadata = json.load(stream)
data_sha256 = sha256_file(cache)
require(data_sha256 == expected_cache_sha, "canonical full-cache SHA-256")
require(metadata.get("data_sha256") == data_sha256, "metadata data SHA-256")
require(
    metadata.get("source_fingerprint") == expected_source_fingerprint,
    "source fingerprint",
)
require(metadata.get("scope") == "full_archive", "full-cache scope")
require(metadata.get("full_archive") is True, "full-cache archive flag")
require(metadata.get("shape") == [2080, 51, 6, 27, 27], "full-cache shape")
require(metadata.get("initialization_count") == 2080, "initialization count")
require(metadata.get("member_labels") == list(range(51)), "member labels")
require(metadata.get("lead_week_labels") == list(range(1, 7)), "lead-week labels")
require(
    metadata.get("dims") == ["init", "member", "lead_week", "lat", "lon"],
    "dimension order",
)
require(metadata.get("output_units") == "mm day-1", "output units")
print(f"accepted canonical full cache: {cache}")
PY

EXPECTED_LAG_SHA256=
EXPECTED_OBSERVATION_SHA256=
SMOKE_MANIFEST_SHA256=
if [[ ${MODE} == full ]]; then
    PROVENANCE_RECEIPT=$(
        "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${WORK_ROOT}" "${CACHE}" \
            "${OUTPUT}" "${EXPECTED_CACHE_SHA256}" \
            "${EXPECTED_SOURCE_FINGERPRINT}" "${EXPECTED_PLAN_SHA256}" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
work_root = Path(sys.argv[2]).resolve()
full_cache = Path(sys.argv[3]).resolve()
requested_output = Path(sys.argv[4]).resolve()
expected_cache_sha = sys.argv[5]
expected_source_fingerprint = sys.argv[6]
expected_plan_sha = sys.argv[7]
smoke_root = path.parent
arms = ["base_42k", "zero_lag_45k", "persistence_lag12_45k"]
parameter_counts = {
    "base_42k": 42_434,
    "zero_lag_45k": 45_026,
    "persistence_lag12_45k": 45_026,
}
context_channels = {"base_42k": 7, "zero_lag_45k": 11, "persistence_lag12_45k": 11}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"smoke-to-full gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def require_finite_json(value: object, location: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{location}[{index}]")

require(path.name == "manifest.json", "smoke receipt must be manifest.json")
try:
    requested_output.relative_to(smoke_root)
except ValueError:
    pass
else:
    require(False, "full output must not modify or nest inside the smoke output")

with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
receipt_path = smoke_root / "slurm_gate_receipt.json"
require(receipt_path.is_file(), "missing Slurm post-audit receipt")
with receipt_path.open("r", encoding="utf-8") as stream:
    gate_receipt = json.load(stream)
require_finite_json(manifest, "manifest")
require(manifest.get("experiment") == "fuxi_allseason_persistence_augmented_v1", "experiment")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == "smoke" and manifest.get("smoke") is True, "smoke mode")
require(Path(manifest.get("output_path", "")).resolve() == smoke_root, "smoke output identity")
require(manifest.get("seeds") == [42], "smoke seed")
require(
    manifest.get("split_counts_archive")
    == {"train": 1652, "validation": 196, "test": 208, "embargo": 24},
    "archive split counts",
)
require(
    manifest.get("split_counts_selected") == {"train": 32, "validation": 16},
    "smoke split counts",
)

arm_rows = manifest.get("arms", [])
require([row.get("name") for row in arm_rows] == arms, "arm order")
for row in arm_rows:
    arm = row.get("name")
    require(row.get("expected_parameter_count") == parameter_counts[arm], f"parameter count {arm}")
    require(row.get("context_channels") == context_channels[arm], f"context channels {arm}")
    require(row.get("selectable") is (arm != "zero_lag_45k"), f"selectability {arm}")

contract = manifest.get("contract", {})
require(contract.get("train_years") == list(range(2002, 2018)), "training years")
require(contract.get("validation_years") == [2018, 2019], "validation years")
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("development_years_not_scored") == [2020, 2021], "development firewall")
require(contract.get("sealed_unopened_years") == [2025], "sealed years")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(
    contract.get("context_order_added")
    == [
        "lag_1week_log1p_z",
        "lag_2week_log1p_z",
        "lag_1week_available",
        "lag_2week_available",
    ],
    "persistence context order",
)

training = manifest.get("training", {})
require(training.get("batch_size") == 8, "batch size")
require(training.get("member_subsample") == 16, "member subsample")
require(training.get("full_members_for_validation") == 51, "full-member validation")
require(training.get("max_epochs") == 2 and training.get("patience") == 1, "smoke epochs/patience")
require(training.get("learning_rate") == 0.0002, "learning rate")
require(training.get("weight_decay") == 0.0001, "weight decay")
require(training.get("automatic_mixed_precision") is True, "AMP")
require(str(training.get("device", "")).startswith("cuda"), "CUDA training device")
require(training.get("same_seed_45k_initial_states_exact") is True, "shared 45k initialization receipt")
require(
    training.get("same_seed_stochastic_training_streams_exact_across_all_arms")
    is True,
    "shared stochastic training streams receipt",
)
runs = training.get("runs", [])
require(len(runs) == 3, "smoke run count")
run_keys = set()
initial_45k = {}
rng_by_arm = {}
for row in runs:
    arm = row.get("arm")
    seed = row.get("seed")
    require(arm in parameter_counts and seed == 42, "training run identity")
    require(row.get("parameter_count") == parameter_counts[arm], f"run parameter count {arm}")
    require(row.get("context_channels") == context_channels[arm], f"run context channels {arm}")
    require(math.isfinite(float(row.get("best_validation_crps", float("nan")))), f"finite validation CRPS {arm}")
    expected_checkpoint = f"models/{arm}/seed_42/checkpoints/best.pt"
    expected_adjustment = f"models/{arm}/seed_42/validation_adjustments.npz"
    require(row.get("checkpoint") == expected_checkpoint, f"checkpoint path {arm}")
    require(row.get("validation_adjustment") == expected_adjustment, f"adjustment path {arm}")
    require(is_digest(row.get("checkpoint_sha256")), f"checkpoint digest {arm}")
    require(is_digest(row.get("validation_adjustment_sha256")), f"adjustment digest {arm}")
    require(is_digest(row.get("training_rng_state_sha256")), f"training RNG digest {arm}")
    rng_by_arm[arm] = row.get("training_rng_state_sha256")
    if arm != "base_42k":
        initial_45k[arm] = row.get("initial_state_sha256")
    run_keys.add((arm, seed))
require(run_keys == {(arm, 42) for arm in arms}, "complete smoke arm/seed grid")
require(initial_45k.get("zero_lag_45k") == initial_45k.get("persistence_lag12_45k"), "exact shared 45k initialization")
require(len(set(rng_by_arm.values())) == 1, "exact same-seed RNG receipt across all arms")
require(
    training.get("training_rng_receipts_by_seed") == {"42": rng_by_arm["base_42k"]},
    "manifest per-seed RNG receipt",
)

selection = manifest.get("selection", {})
require(selection.get("status") == "validation_selection_locked", "selection lock")
require(selection.get("scientific_selection") is False, "smoke is non-scientific")
require(selection.get("test_metrics_consulted") is False, "selection test firewall")
require(selection.get("zero_lag_selectable") is False, "zero-lag selection firewall")
require(selection.get("selected_arm") in {"base_42k", "persistence_lag12_45k"}, "selected arm")
require(set(selection.get("candidate_comparisons", {})) == {"base_42k", "zero_lag_45k"}, "candidate comparisons")
with (smoke_root / "selection.json").open("r", encoding="utf-8") as stream:
    require(json.load(stream) == selection, "selection artifact/manifest agreement")

software = manifest.get("software", {})
require(software.get("cuda_available") is True, "CUDA receipt")
require(bool(software.get("cuda_device")), "CUDA device receipt")
require(
    gate_receipt.get("schema_version") == "fuxi_persistence_slurm_gate_receipt_v1",
    "Slurm receipt schema",
)
require(gate_receipt.get("experiment") == manifest.get("experiment"), "Slurm receipt experiment")
require(gate_receipt.get("status") == "post_run_audit_passed", "Slurm receipt status")
require(gate_receipt.get("mode") == "smoke", "Slurm receipt mode")
require(gate_receipt.get("output_path") == str(smoke_root), "Slurm receipt output")
require(gate_receipt.get("manifest_sha256") == sha256_file(path), "Slurm receipt manifest SHA-256")
require(str(gate_receipt.get("job_id", "")).isdigit(), "Slurm receipt job")
require(bool(gate_receipt.get("node")) and gate_receipt.get("node") != "unknown", "Slurm receipt node")
require(gate_receipt.get("partition") == "gpu_prio", "Slurm receipt partition")
require(
    gate_receipt.get("node") not in {"cn2", "cn3", "cn4", "cn15", "cn16", "cn17"},
    "Slurm receipt excluded node",
)
require(bool(gate_receipt.get("gpu")), "Slurm receipt GPU")
require(gate_receipt.get("cuda_device") == software.get("cuda_device"), "Slurm/CUDA GPU agreement")
require(gate_receipt.get("cache_sha256") == expected_cache_sha, "Slurm receipt cache")
require(gate_receipt.get("smoke_manifest_sha256") is None, "smoke receipt parent field")
temporal = manifest.get("temporal_evidence", {})
require(
    temporal.get("lag_windows")
    == "issue-7 through issue-1 and issue-14 through issue-8; every end < issue",
    "issue-time lag window receipt",
)
require(temporal.get("development_indices_accessed") is False, "development-index firewall")
categorical = manifest.get("categorical_validation", {})
cdf_evidence = categorical.get("finite_and_cdf_validity_evidence", {})
expected_cdf_columns = [
    "cdf_shape_matches_thresholds",
    "forecast_cdf_finite_where_threshold_defined",
    "observed_cdf_finite_where_threshold_defined",
    "forecast_cdf_valid_for_thresholds",
    "observed_cdf_valid_for_thresholds",
]
require(cdf_evidence.get("evidence_columns") == expected_cdf_columns, "CDF evidence columns")
require(cdf_evidence.get("validation_metric_rows") == 3 * 16 * 6, "CDF evidence row count")
for key in (
    "threshold_and_observed_shapes_equal",
    "thresholds_finite_on_scoring_support",
    "observed_cdf_finite_on_scoring_support",
    "observed_cdf_valid_for_thresholds",
    "every_scoring_chunk_shape_matched_thresholds",
    "every_forecast_cdf_finite_where_threshold_defined",
    "every_observed_cdf_finite_where_threshold_defined",
    "every_forecast_cdf_valid_for_thresholds",
    "every_observed_cdf_valid_for_thresholds",
    "every_validation_score_finite",
):
    require(cdf_evidence.get(key) is True, f"categorical validity evidence {key}")

lag = manifest.get("persistence_lag_provenance", {})
observation = manifest.get("observation_bundle_provenance", {})
context = manifest.get("persistence_context_provenance", {})
for name, record in (("lag", lag), ("observation", observation), ("context", context)):
    require(is_digest(record.get("sha256")), f"{name} provenance digest")
    require(record.get("schema_version") == "named_c_order_array_bundle_sha256_v1", f"{name} provenance schema")
require(lag.get("total_case_count") == 2080, "lag case count")
require(lag.get("available_supported_values_finite_nonnegative") is True, "finite nonnegative lag values")
require(
    context.get("bundle_name") == "train_normalized_persistence_context",
    "context provenance bundle",
)
provenance_files = {
    "persistence_lag_provenance": "evaluation/persistence_lag_provenance.json",
    "observation_bundle_provenance": "evaluation/observation_bundle_provenance.json",
    "persistence_context_provenance": "evaluation/persistence_context_provenance.json",
}
for key, relative in provenance_files.items():
    with (smoke_root / relative).open("r", encoding="utf-8") as stream:
        require(json.load(stream) == manifest.get(key), f"{key} artifact agreement")

cache_record = manifest.get("cache", {})
require(cache_record.get("data_sha256") == expected_cache_sha, "smoke cache SHA-256")
require(cache_record.get("source_fingerprint") == expected_source_fingerprint, "smoke source fingerprint")
require(cache_record.get("scope") == "full_archive", "smoke full-cache scope")
require(sha256_file(full_cache) == expected_cache_sha, "current full-cache SHA-256")

required = {
    "selection.json",
    "README.md",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_normalization.npz",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/persistence_context_provenance.json",
    "evaluation/validation_quintile_fit.npz",
    "history/training_history.csv",
    "metrics/validation_case_metrics.csv",
    "metrics/validation_summary.csv",
    "metrics/validation_by_year.csv",
    "metrics/validation_by_seed.csv",
    "metrics/validation_by_lead.csv",
    "figures/training_loss_curves.png",
    "figures/training_loss_curves.pdf",
    "figures/validation_by_lead.png",
    "figures/validation_by_lead.pdf",
}
for arm in arms:
    required.add(f"models/{arm}/seed_42/checkpoints/best.pt")
    required.add(f"models/{arm}/seed_42/validation_adjustments.npz")

artifact_hashes = manifest.get("artifact_sha256", {})
require(isinstance(artifact_hashes, dict), "artifact hash map")
actual_artifacts = {
    str(candidate.relative_to(smoke_root))
    for candidate in smoke_root.rglob("*")
    if candidate.is_file()
    and str(candidate.relative_to(smoke_root))
    not in {"manifest.json", "slurm_gate_receipt.json"}
}
require(set(artifact_hashes) == actual_artifacts, "complete smoke artifact hash inventory")
require("slurm_gate_receipt.json" not in artifact_hashes, "external Slurm receipt is not driver-hashed")
require(required.issubset(actual_artifacts), "required smoke artifact inventory")
for relative, digest in artifact_hashes.items():
    require(is_digest(digest), f"artifact digest schema {relative}")
    require(sha256_file(smoke_root / relative) == digest, f"artifact hash {relative}")

source_map = {
    "code/src/fuxi_allseason_persistence_augmented.py": work_root / "src/fuxi_allseason_persistence_augmented.py",
    "code/src/fuxi_persistence_context.py": work_root / "src/fuxi_persistence_context.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_pbc_baseline.py": work_root / "src/fuxi_allseason_pbc_baseline.py",
    "code/src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_allseason_persistence_augmented.sbatch": work_root / "slurm/run_allseason_persistence_augmented.sbatch",
    "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
}
source_hashes = manifest.get("source_snapshot_sha256", {})
require(set(source_hashes) == set(source_map), "source/plan snapshot inventory")
for relative, current in source_map.items():
    digest = source_hashes.get(relative)
    require(is_digest(digest), f"source digest {relative}")
    require(artifact_hashes.get(relative) == digest, f"source/artifact hash {relative}")
    require(sha256_file(smoke_root / relative) == digest, f"frozen source hash {relative}")
    require(sha256_file(current) == digest, f"current source differs from smoke: {current}")
require(
    source_hashes.get("code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md")
    == expected_plan_sha,
    "frozen pre-score plan SHA-256",
)
require(gate_receipt.get("source_snapshot_sha256") == source_hashes, "Slurm receipt source bindings")
expected_source_binding = hashlib.sha256(
    json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
require(gate_receipt.get("source_binding_sha256") == expected_source_binding, "Slurm receipt source digest")
require(gate_receipt.get("persistence_lag_sha256") == lag.get("sha256"), "Slurm receipt lag digest")
require(
    gate_receipt.get("observation_bundle_sha256") == observation.get("sha256"),
    "Slurm receipt observation digest",
)
plan_binding = gate_receipt.get("plan_binding", {})
require(plan_binding.get("expected_sha256") == expected_plan_sha, "Slurm receipt expected plan")
require(plan_binding.get("snapshot_sha256") == expected_plan_sha, "Slurm receipt snapshot plan")
require(plan_binding.get("live_sha256") == expected_plan_sha, "Slurm receipt live plan")

metrics_path = smoke_root / "metrics/validation_case_metrics.csv"
with metrics_path.open("r", encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
require(len(rows) == 3 * 16 * 6, "validation metric row count")
keys_by_arm = {}
for arm in arms:
    arm_rows = [row for row in rows if row.get("arm") == arm]
    require(len(arm_rows) == 16 * 6, f"validation rows {arm}")
    require({int(row["seed"]) for row in arm_rows} == {42}, f"validation seed {arm}")
    require({int(row["year"]) for row in arm_rows} == {2018, 2019}, f"validation years {arm}")
    require({int(row["lead_week"]) for row in arm_rows} == set(range(1, 7)), f"validation leads {arm}")
    for row in arm_rows:
        for field in ("crps", "quintile_rps"):
            score = float(row[field])
            require(math.isfinite(score) and score >= 0.0, f"finite {field} {arm}")
        for field in expected_cdf_columns:
            require(row.get(field, "").strip().lower() == "true", f"CSV CDF evidence {field} {arm}")
    keys_by_arm[arm] = {
        (row["seed"], row["init"], row["year"], row["lead_week"])
        for row in arm_rows
    }
require(all(keys_by_arm[arm] == keys_by_arm[arms[0]] for arm in arms[1:]), "matched validation cases")

print(lag["sha256"], observation["sha256"], sha256_file(path))
PY
    )
    read -r EXPECTED_LAG_SHA256 EXPECTED_OBSERVATION_SHA256 \
        SMOKE_MANIFEST_SHA256 <<<"${PROVENANCE_RECEIPT}"
    if [[ ! ${EXPECTED_LAG_SHA256} =~ ^[0-9a-f]{64}$ \
        || ! ${EXPECTED_OBSERVATION_SHA256} =~ ^[0-9a-f]{64}$ \
        || ! ${SMOKE_MANIFEST_SHA256} =~ ^[0-9a-f]{64}$ ]]; then
        echo "smoke-to-full gate returned invalid provenance hashes" >&2
        exit 2
    fi
    echo "accepted completed, artifact-audited smoke manifest: ${SMOKE_MANIFEST}"
    echo "bound persistence lag SHA-256: ${EXPECTED_LAG_SHA256}"
    echo "bound observation bundle SHA-256: ${EXPECTED_OBSERVATION_SHA256}"
    echo "bound smoke manifest SHA-256: ${SMOKE_MANIFEST_SHA256}"
fi

mkdir -p "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "experiment=fuxi_allseason_persistence_augmented_v1 mode=${MODE}"
echo "job=${JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi
GPU_RECEIPT=$(
    nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
        --format=csv,noheader,nounits
)
if [[ -z ${GPU_RECEIPT} || ${GPU_RECEIPT} == *$'\n'* ]]; then
    echo "expected exactly one nvidia-smi GPU receipt" >&2
    exit 2
fi
"${PYTHON_BIN}" - <<'PY'
import sys

import matplotlib
import numpy
import pandas
import torch
import xarray

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
probe = torch.ones(1, device="cuda")
if not bool(torch.isfinite(probe).all()):
    raise SystemExit("CUDA tensor probe is non-finite")
print("sys.executable:", sys.executable)
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Xarray:", xarray.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -e ${OUTPUT} ]]; then
    echo "output appeared before launch; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi

COMMON_ARGUMENTS=(
    --cache "${CACHE}"
    --output "${OUTPUT}"
    --device cuda
    --arms "${ARMS}"
    --batch-size 8
    --member-subsample 16
    --learning-rate 0.0002
    --weight-decay 0.0001
    --num-workers 0
    --evaluation-batch-size 8
    --cdf-chunk-size 8
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_persistence_augmented.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42 \
        --max-epochs 2 \
        --patience 1 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_persistence_augmented.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42,43,44 \
        --max-epochs 100 \
        --patience 15 \
        --expected-persistence-lag-sha256 "${EXPECTED_LAG_SHA256}" \
        --expected-observation-bundle-sha256 "${EXPECTED_OBSERVATION_SHA256}"
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${CACHE}" "${WORK_ROOT}" \
    "${EXPECTED_LAG_SHA256}" "${EXPECTED_OBSERVATION_SHA256}" \
    "${EXPECTED_CACHE_SHA256}" "${EXPECTED_SOURCE_FINGERPRINT}" \
    "${EXPECTED_PLAN_SHA256}" "${SMOKE_MANIFEST}" \
    "${SMOKE_MANIFEST_SHA256}" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
cache = Path(sys.argv[3]).resolve()
work_root = Path(sys.argv[4]).resolve()
expected_lag_sha = sys.argv[5]
expected_observation_sha = sys.argv[6]
expected_cache_sha = sys.argv[7]
expected_source_fingerprint = sys.argv[8]
expected_plan_sha = sys.argv[9]
smoke_manifest_argument = sys.argv[10]
expected_smoke_manifest_sha = sys.argv[11]
arms = ["base_42k", "zero_lag_45k", "persistence_lag12_45k"]
parameter_counts = {
    "base_42k": 42_434,
    "zero_lag_45k": 45_026,
    "persistence_lag12_45k": 45_026,
}
context_channels = {"base_42k": 7, "zero_lag_45k": 11, "persistence_lag12_45k": 11}
expected_seeds = [42] if mode == "smoke" else [42, 43, 44]
expected_selected_counts = (
    {"train": 32, "validation": 16}
    if mode == "smoke"
    else {"train": 1652, "validation": 196}
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"post-run gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def require_finite_json(value: object, location: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{location}[{index}]")

required = {
    "selection.json",
    "README.md",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_normalization.npz",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/persistence_context_provenance.json",
    "evaluation/validation_quintile_fit.npz",
    "history/training_history.csv",
    "metrics/validation_case_metrics.csv",
    "metrics/validation_summary.csv",
    "metrics/validation_by_year.csv",
    "metrics/validation_by_seed.csv",
    "metrics/validation_by_lead.csv",
    "figures/training_loss_curves.png",
    "figures/training_loss_curves.pdf",
    "figures/validation_by_lead.png",
    "figures/validation_by_lead.pdf",
}
for arm in arms:
    for seed in expected_seeds:
        required.add(f"models/{arm}/seed_{seed}/checkpoints/best.pt")
        required.add(f"models/{arm}/seed_{seed}/validation_adjustments.npz")
missing = sorted(relative for relative in required if not (root / relative).is_file())
require(not missing, f"missing required artifacts: {missing}")
require((root / "manifest.json").is_file(), "missing manifest.json")
require(not (root / "failure.json").exists(), "failure receipt present")
require(not (root / "slurm_gate_receipt.json").exists(), "Slurm receipt already exists before audit")

with (root / "manifest.json").open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
with (root / "selection.json").open("r", encoding="utf-8") as stream:
    selection_artifact = json.load(stream)
require_finite_json(manifest, "manifest")
require(manifest.get("experiment") == "fuxi_allseason_persistence_augmented_v1", "experiment")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == mode, "mode")
require(manifest.get("smoke") is (mode == "smoke"), "smoke flag")
require(Path(manifest.get("output_path", "")).resolve() == root, "output identity")
require(manifest.get("seeds") == expected_seeds, "seed contract")
require(
    manifest.get("split_counts_archive")
    == {"train": 1652, "validation": 196, "test": 208, "embargo": 24},
    "archive split counts",
)
require(manifest.get("split_counts_selected") == expected_selected_counts, "selected split counts")

arm_rows = manifest.get("arms", [])
require([row.get("name") for row in arm_rows] == arms, "arm order")
for row in arm_rows:
    arm = row.get("name")
    require(row.get("expected_parameter_count") == parameter_counts[arm], f"parameter count {arm}")
    require(row.get("context_channels") == context_channels[arm], f"context channels {arm}")
    require(row.get("selectable") is (arm != "zero_lag_45k"), f"selectability {arm}")

contract = manifest.get("contract", {})
require(contract.get("train_years") == list(range(2002, 2018)), "training years")
require(contract.get("validation_years") == [2018, 2019], "validation years")
require(contract.get("selection_data") == "2018-2019 validation only", "selection data")
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("development_years_not_scored") == [2020, 2021], "development firewall")
require(contract.get("sealed_unopened_years") == [2025], "sealed years")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(
    contract.get("lag_windows") == "issue-7..issue-1 and issue-14..issue-8 inclusive",
    "lag windows",
)
require(
    contract.get("lag_transform")
    == "log1p then per-lag train-only area-weighted scalar standardization",
    "lag normalization",
)
require(
    contract.get("context_order_added")
    == [
        "lag_1week_log1p_z",
        "lag_2week_log1p_z",
        "lag_1week_available",
        "lag_2week_available",
    ],
    "persistence context order",
)
require(contract.get("categorical_score_contract") == "normalized_informative_positive_cut_v2", "score contract")

training = manifest.get("training", {})
require(training.get("batch_size") == 8, "batch size")
require(training.get("member_subsample") == 16, "member subsample")
require(training.get("full_members_for_validation") == 51, "full-member validation")
require(training.get("max_epochs") == (2 if mode == "smoke" else 100), "max epochs")
require(training.get("patience") == (1 if mode == "smoke" else 15), "patience")
require(training.get("learning_rate") == 0.0002, "learning rate")
require(training.get("weight_decay") == 0.0001, "weight decay")
require(training.get("automatic_mixed_precision") is True, "AMP")
require(str(training.get("device", "")).startswith("cuda"), "CUDA training device")
require(training.get("same_seed_45k_initial_states_exact") is True, "shared 45k initialization receipt")
require(
    training.get("same_seed_stochastic_training_streams_exact_across_all_arms")
    is True,
    "shared stochastic training streams receipt",
)
runs = training.get("runs", [])
require(len(runs) == len(arms) * len(expected_seeds), "training run count")
run_keys = set()
initial_45k = {}
rng_receipts = {}
for row in runs:
    arm = row.get("arm")
    seed = row.get("seed")
    require(arm in parameter_counts and seed in expected_seeds, "training run identity")
    require(row.get("parameter_count") == parameter_counts[arm], f"run parameter count {arm}/{seed}")
    require(row.get("context_channels") == context_channels[arm], f"run context channels {arm}/{seed}")
    require(math.isfinite(float(row.get("best_validation_crps", float("nan")))), f"finite validation CRPS {arm}/{seed}")
    checkpoint = f"models/{arm}/seed_{seed}/checkpoints/best.pt"
    adjustment = f"models/{arm}/seed_{seed}/validation_adjustments.npz"
    require(row.get("checkpoint") == checkpoint, f"checkpoint path {arm}/{seed}")
    require(row.get("validation_adjustment") == adjustment, f"adjustment path {arm}/{seed}")
    require(row.get("checkpoint_sha256") == sha256_file(root / checkpoint), f"checkpoint receipt {arm}/{seed}")
    require(row.get("validation_adjustment_sha256") == sha256_file(root / adjustment), f"adjustment receipt {arm}/{seed}")
    require(is_digest(row.get("training_rng_state_sha256")), f"training RNG digest {arm}/{seed}")
    rng_receipts[(arm, seed)] = row.get("training_rng_state_sha256")
    if arm != "base_42k":
        initial_45k[(arm, seed)] = row.get("initial_state_sha256")
    run_keys.add((arm, seed))
require(run_keys == {(arm, seed) for arm in arms for seed in expected_seeds}, "complete arm/seed grid")
for seed in expected_seeds:
    require(
        initial_45k.get(("zero_lag_45k", seed))
        == initial_45k.get(("persistence_lag12_45k", seed)),
        f"exact shared 45k initialization seed {seed}",
    )
    require(
        len({rng_receipts[(arm, seed)] for arm in arms}) == 1,
        f"exact same-seed RNG receipt across all arms seed {seed}",
    )
require(
    training.get("training_rng_receipts_by_seed")
    == {str(seed): rng_receipts[("base_42k", seed)] for seed in expected_seeds},
    "manifest per-seed RNG receipts",
)

selection = manifest.get("selection", {})
require(selection_artifact == selection, "selection artifact/manifest agreement")
require(selection.get("status") == "validation_selection_locked", "selection lock")
require(selection.get("scientific_selection") is (mode == "full"), "scientific selection flag")
require(selection.get("test_metrics_consulted") is False, "selection test firewall")
require(selection.get("zero_lag_selectable") is False, "zero-lag selection firewall")
require(selection.get("selected_arm") in {"base_42k", "persistence_lag12_45k"}, "selected arm")
require(set(selection.get("candidate_comparisons", {})) == {"base_42k", "zero_lag_45k"}, "candidate comparisons")
require(
    selection.get("rules", {}).get("minimum_matched_seed_rps_improvements_vs_each_control")
    == min(2, len(expected_seeds)),
    "matched-seed promotion rule",
)

software = manifest.get("software", {})
require(software.get("cuda_available") is True, "CUDA receipt")
require(bool(software.get("cuda_device")), "CUDA device receipt")
temporal = manifest.get("temporal_evidence", {})
require(
    temporal.get("lag_windows")
    == "issue-7 through issue-1 and issue-14 through issue-8; every end < issue",
    "issue-time lag window receipt",
)
require(temporal.get("development_indices_accessed") is False, "development-index firewall")
categorical = manifest.get("categorical_validation", {})
cdf_evidence = categorical.get("finite_and_cdf_validity_evidence", {})
expected_cdf_columns = [
    "cdf_shape_matches_thresholds",
    "forecast_cdf_finite_where_threshold_defined",
    "observed_cdf_finite_where_threshold_defined",
    "forecast_cdf_valid_for_thresholds",
    "observed_cdf_valid_for_thresholds",
]
require(cdf_evidence.get("evidence_columns") == expected_cdf_columns, "CDF evidence columns")
require(
    cdf_evidence.get("validation_metric_rows")
    == len(arms) * len(expected_seeds) * expected_selected_counts["validation"] * 6,
    "CDF evidence row count",
)
for key in (
    "threshold_and_observed_shapes_equal",
    "thresholds_finite_on_scoring_support",
    "observed_cdf_finite_on_scoring_support",
    "observed_cdf_valid_for_thresholds",
    "every_scoring_chunk_shape_matched_thresholds",
    "every_forecast_cdf_finite_where_threshold_defined",
    "every_observed_cdf_finite_where_threshold_defined",
    "every_forecast_cdf_valid_for_thresholds",
    "every_observed_cdf_valid_for_thresholds",
    "every_validation_score_finite",
):
    require(cdf_evidence.get(key) is True, f"categorical validity evidence {key}")

lag = manifest.get("persistence_lag_provenance", {})
observation = manifest.get("observation_bundle_provenance", {})
context = manifest.get("persistence_context_provenance", {})
for name, record in (("lag", lag), ("observation", observation), ("context", context)):
    require(is_digest(record.get("sha256")), f"{name} provenance digest")
    require(record.get("schema_version") == "named_c_order_array_bundle_sha256_v1", f"{name} provenance schema")
require(lag.get("total_case_count") == 2080, "lag case count")
require(lag.get("available_supported_values_finite_nonnegative") is True, "finite nonnegative lag values")
require(context.get("bundle_name") == "train_normalized_persistence_context", "context provenance bundle")
if mode == "full":
    require(lag.get("sha256") == expected_lag_sha, "smoke-bound lag provenance")
    require(observation.get("sha256") == expected_observation_sha, "smoke-bound observation provenance")

provenance_files = {
    "persistence_lag_provenance": "evaluation/persistence_lag_provenance.json",
    "observation_bundle_provenance": "evaluation/observation_bundle_provenance.json",
    "persistence_context_provenance": "evaluation/persistence_context_provenance.json",
}
for key, relative in provenance_files.items():
    with (root / relative).open("r", encoding="utf-8") as stream:
        require(json.load(stream) == manifest.get(key), f"{key} artifact agreement")

with np.load(root / "evaluation/persistence_normalization.npz", allow_pickle=False) as bundle:
    require(set(bundle.files) == {"lag_log1p_mean_by_lag", "lag_log1p_std_by_lag", "normalization_fit_indices", "channel_names"}, "normalization artifact schema")
    means = np.asarray(bundle["lag_log1p_mean_by_lag"])
    stds = np.asarray(bundle["lag_log1p_std_by_lag"])
    fit_indices = np.asarray(bundle["normalization_fit_indices"])
    channel_names = np.asarray(bundle["channel_names"]).tolist()
require(means.shape == (2,) and np.isfinite(means).all(), "finite per-lag means")
require(stds.shape == (2,) and np.isfinite(stds).all() and np.all(stds > 0.0), "positive finite per-lag standard deviations")
require(fit_indices.shape == (expected_selected_counts["train"],), "normalization train-index count")
require(np.unique(fit_indices).size == fit_indices.size, "unique normalization train indices")
require(
    channel_names
    == ["lag_1week_log1p_z", "lag_2week_log1p_z", "lag_1week_available", "lag_2week_available"],
    "normalization channel order",
)

with np.load(root / "evaluation/scoring_support.npz", allow_pickle=False) as support_bundle:
    support = np.asarray(support_bundle["support_mask"])
    normalized_weights = np.asarray(support_bundle["normalized_scoring_weight"])
require(support.shape == (27, 27) and support.dtype == np.bool_, "scoring support schema")
require(int(np.count_nonzero(support)) == 171, "scoring support cells")
require(np.isfinite(normalized_weights).all(), "finite scoring weights")
require(math.isclose(float(normalized_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-10), "normalized scoring-weight sum")
require(manifest.get("evaluation", {}).get("support_cells") == 171, "manifest support cells")

metrics_path = root / "metrics/validation_case_metrics.csv"
with metrics_path.open("r", encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
expected_metric_rows = len(arms) * len(expected_seeds) * expected_selected_counts["validation"] * 6
require(len(rows) == expected_metric_rows, "validation metric row count")
keys_by_arm = {}
for arm in arms:
    arm_rows = [row for row in rows if row.get("arm") == arm]
    expected_arm_rows = len(expected_seeds) * expected_selected_counts["validation"] * 6
    require(len(arm_rows) == expected_arm_rows, f"validation rows {arm}")
    require({int(row["seed"]) for row in arm_rows} == set(expected_seeds), f"validation seeds {arm}")
    require({int(row["year"]) for row in arm_rows} == {2018, 2019}, f"validation years {arm}")
    require({int(row["lead_week"]) for row in arm_rows} == set(range(1, 7)), f"validation leads {arm}")
    require({row["score_contract"] for row in arm_rows} == {"normalized_informative_positive_cut_v2"}, f"score contract {arm}")
    for row in arm_rows:
        require(row.get("split") == "validation", f"validation split {arm}")
        for field in ("crps", "quintile_rps"):
            score = float(row[field])
            require(math.isfinite(score) and score >= 0.0, f"finite {field} {arm}")
        for field in expected_cdf_columns:
            require(row.get(field, "").strip().lower() == "true", f"CSV CDF evidence {field} {arm}")
    keys_by_arm[arm] = {
        (row["seed"], row["init"], row["year"], row["lead_week"])
        for row in arm_rows
    }
require(all(keys_by_arm[arm] == keys_by_arm[arms[0]] for arm in arms[1:]), "matched validation cases")

artifact_hashes = manifest.get("artifact_sha256", {})
require(isinstance(artifact_hashes, dict), "artifact hash map")
actual_artifacts = {
    str(candidate.relative_to(root))
    for candidate in root.rglob("*")
    if candidate.is_file() and candidate.name != "manifest.json"
}
require(set(artifact_hashes) == actual_artifacts, "complete artifact hash inventory")
require(required.issubset(actual_artifacts), "required artifact inventory")
for relative, digest in artifact_hashes.items():
    require(is_digest(digest), f"artifact digest schema {relative}")
    require(sha256_file(root / relative) == digest, f"artifact hash {relative}")

source_map = {
    "code/src/fuxi_allseason_persistence_augmented.py": work_root / "src/fuxi_allseason_persistence_augmented.py",
    "code/src/fuxi_persistence_context.py": work_root / "src/fuxi_persistence_context.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_pbc_baseline.py": work_root / "src/fuxi_allseason_pbc_baseline.py",
    "code/src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_allseason_persistence_augmented.sbatch": work_root / "slurm/run_allseason_persistence_augmented.sbatch",
    "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
}
source_hashes = manifest.get("source_snapshot_sha256", {})
require(set(source_hashes) == set(source_map), "source/plan snapshot inventory")
for relative, current in source_map.items():
    digest = source_hashes.get(relative)
    require(is_digest(digest), f"source digest {relative}")
    require(artifact_hashes.get(relative) == digest, f"source/artifact hash {relative}")
    require(sha256_file(root / relative) == digest, f"frozen source hash {relative}")
    require(sha256_file(current) == digest, f"current source changed during run: {current}")
require(
    source_hashes.get("code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md")
    == expected_plan_sha,
    "frozen pre-score plan SHA-256",
)
if mode == "full":
    smoke_manifest_path = Path(smoke_manifest_argument).resolve()
    require(smoke_manifest_path.is_file(), "smoke manifest still exists")
    require(
        sha256_file(smoke_manifest_path) == expected_smoke_manifest_sha,
        "smoke manifest changed during full run",
    )
    with smoke_manifest_path.open("r", encoding="utf-8") as stream:
        smoke_manifest = json.load(stream)
    require(
        source_hashes == smoke_manifest.get("source_snapshot_sha256"),
        "full source snapshot differs from smoke",
    )
else:
    require(not smoke_manifest_argument and not expected_smoke_manifest_sha, "smoke has no parent manifest")

cache_record = manifest.get("cache", {})
require(cache_record.get("data_sha256") == expected_cache_sha, "manifest cache SHA-256")
require(cache_record.get("source_fingerprint") == expected_source_fingerprint, "manifest source fingerprint")
require(cache_record.get("scope") == "full_archive", "manifest full-cache scope")
require(sha256_file(cache) == expected_cache_sha, "post-run cache SHA-256")
metadata_path = cache.with_suffix("").with_suffix(".metadata.json")
require(cache_record.get("metadata_sha256") == sha256_file(metadata_path), "cache metadata SHA-256")

print(root / "manifest.json")
PY

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${JOB_ID}" \
    "${SLURMD_NODENAME:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" \
    "${GPU_RECEIPT}" "${WORK_ROOT}" "${EXPECTED_PLAN_SHA256}" \
    "${SMOKE_MANIFEST}" "${SMOKE_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
job_id = sys.argv[3]
node = sys.argv[4]
partition = sys.argv[5]
gpu = sys.argv[6]
work_root = Path(sys.argv[7]).resolve()
expected_plan_sha = sys.argv[8]
smoke_manifest_argument = sys.argv[9]
expected_smoke_manifest_sha = sys.argv[10]
manifest_path = root / "manifest.json"
receipt_path = root / "slurm_gate_receipt.json"

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Slurm receipt failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

require(mode in {"smoke", "full"}, "mode")
require(job_id.isdigit(), "numeric Slurm job ID")
require(bool(node) and node != "unknown", "Slurm node")
require(partition == "gpu_prio", "Slurm gpu_prio partition")
require(node not in {"cn2", "cn3", "cn4", "cn15", "cn16", "cn17"}, "excluded GPU node")
require(bool(gpu), "GPU identity")
require(manifest_path.is_file(), "manifest")
require(not receipt_path.exists(), "refusing to overwrite Slurm receipt")
with manifest_path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
require(manifest.get("status") == "complete", "manifest completion")
require(manifest.get("mode") == mode, "manifest mode")
source_hashes = manifest.get("source_snapshot_sha256", {})
require(bool(source_hashes), "source bindings")
plan_relative = "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
live_plan = work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
live_plan_sha = sha256_file(live_plan)
require(live_plan_sha == expected_plan_sha, "live frozen plan SHA-256")
require(source_hashes.get(plan_relative) == expected_plan_sha, "snapshot frozen plan SHA-256")

smoke_manifest_path = None
smoke_manifest_sha = None
if mode == "full":
    require(bool(smoke_manifest_argument), "full smoke-manifest path")
    require(len(expected_smoke_manifest_sha) == 64, "full smoke-manifest digest")
    smoke_manifest = Path(smoke_manifest_argument).resolve()
    require(smoke_manifest.is_file(), "full smoke manifest still exists")
    require(sha256_file(smoke_manifest) == expected_smoke_manifest_sha, "smoke manifest changed during full run")
    smoke_manifest_path = str(smoke_manifest)
    smoke_manifest_sha = expected_smoke_manifest_sha
else:
    require(not smoke_manifest_argument and not expected_smoke_manifest_sha, "smoke has no parent manifest")

source_binding_sha = hashlib.sha256(
    json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
receipt = {
    "schema_version": "fuxi_persistence_slurm_gate_receipt_v1",
    "experiment": "fuxi_allseason_persistence_augmented_v1",
    "status": "post_run_audit_passed",
    "mode": mode,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_path": str(root),
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "job_id": job_id,
    "node": node,
    "partition": partition,
    "gpu": gpu,
    "cuda_device": manifest.get("software", {}).get("cuda_device"),
    "cache_sha256": manifest.get("cache", {}).get("data_sha256"),
    "source_snapshot_sha256": source_hashes,
    "source_binding_sha256": source_binding_sha,
    "plan_binding": {
        "path": str(live_plan),
        "expected_sha256": expected_plan_sha,
        "snapshot_sha256": source_hashes[plan_relative],
        "live_sha256": live_plan_sha,
    },
    "persistence_lag_sha256": manifest.get("persistence_lag_provenance", {}).get("sha256"),
    "observation_bundle_sha256": manifest.get("observation_bundle_provenance", {}).get("sha256"),
    "smoke_manifest_path": smoke_manifest_path,
    "smoke_manifest_sha256": smoke_manifest_sha,
}
with receipt_path.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(receipt_path)
PY

echo "complete and post-run audited: ${OUTPUT}"
