#!/usr/bin/env bash
#SBATCH --job-name=fuxi-capacity
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
  sbatch slurm/run_allseason_capacity_ablation.sbatch \
    smoke /absolute/path/cache.npy /absolute/path/fresh-smoke-output

  sbatch slurm/run_allseason_capacity_ablation.sbatch \
    full /absolute/path/full-cache.npy /absolute/path/fresh-full-output \
    /absolute/path/completed-smoke/manifest.json
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

if [[ $2 != /* || $3 != /* ]]; then
    echo "cache and output must be absolute paths" >&2
    exit 2
fi
if [[ ${MODE} == full && $4 != /* ]]; then
    echo "completed smoke manifest must be an absolute path" >&2
    exit 2
fi
if [[ ! -f $2 ]]; then
    echo "member cache does not exist: $2" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE=$(readlink -f "$2")
OUTPUT=$(readlink -m "$3")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
JOB_TMP=/tmp/fuxi-capacity-${SLURM_JOB_ID}
CANDIDATES=small_20k,base_42k,medium_158k,large_294k

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/src/fuxi_allseason_capacity_ablation.py ]]; then
    echo "capacity driver is unavailable" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/plan/CAPACITY_PBC_STUDY_20260822.md ]]; then
    echo "frozen capacity/PBC plan is unavailable" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi

cd "${WORK_ROOT}"
"${PYTHON_BIN}" src/fuxi_allseason_member_cache.py verify --output "${CACHE}"
"${PYTHON_BIN}" - "${CACHE}" "${MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
mode = sys.argv[2]
metadata_path = cache.with_suffix("").with_suffix(".metadata.json")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"cache contract failed: {message}")

require(metadata_path.is_file(), f"missing metadata {metadata_path}")
with metadata_path.open("r", encoding="utf-8") as stream:
    metadata = json.load(stream)

expected_fingerprint = (
    "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
)
require(metadata.get("source_fingerprint") == expected_fingerprint, "source fingerprint")
require(metadata.get("member_labels") == list(range(51)), "member labels")
require(metadata.get("lead_week_labels") == list(range(1, 7)), "lead-week labels")
require(
    metadata.get("dims") == ["init", "member", "lead_week", "lat", "lon"],
    "dimension order",
)
require(metadata.get("output_units") == "mm day-1", "output units")

digest = hashlib.sha256()
with cache.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
data_sha256 = digest.hexdigest()
require(metadata.get("data_sha256") == data_sha256, "data SHA-256")

is_full = (
    metadata.get("scope") == "full_archive"
    and metadata.get("full_archive") is True
    and metadata.get("shape") == [2080, 51, 6, 27, 27]
    and metadata.get("initialization_count") == 2080
)
is_smoke = (
    metadata.get("scope")
    == "stratified_32_train_16_validation_16_test_non_scientific"
    and metadata.get("full_archive") is False
    and metadata.get("shape") == [64, 51, 6, 27, 27]
    and metadata.get("initialization_count") == 64
)
if mode == "full":
    require(is_full, "full mode requires the canonical full cache")
    require(
        data_sha256
        == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
        "canonical full-cache SHA-256",
    )
else:
    require(is_full or is_smoke, "smoke mode requires a canonical full or stratified cache")
print(f"accepted verified {mode} cache: {cache}")
PY

if [[ ${MODE} == full ]]; then
    if [[ ! -f $4 ]]; then
        echo "completed smoke manifest does not exist: $4" >&2
        exit 2
    fi
    SMOKE_MANIFEST=$(readlink -f "$4")
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${WORK_ROOT}" "${CACHE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
work_root = Path(sys.argv[2])
full_cache = Path(sys.argv[3])
candidates = ["small_20k", "base_42k", "medium_158k", "large_294k"]
arms = [*candidates, "summary_matched_43k"]
parameter_counts = {
    "small_20k": 19_618,
    "base_42k": 42_434,
    "medium_158k": 157_570,
    "large_294k": 293_762,
    "summary_matched_43k": 43_058,
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"smoke-to-full gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
require(manifest.get("experiment") == "fuxi_allseason_capacity_ablation_v1", "experiment")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == "smoke" and manifest.get("smoke") is True, "smoke mode")
require([row.get("name") for row in manifest.get("candidates", [])] == candidates, "candidate order")
require([row.get("name") for row in manifest.get("controls", [])] == ["summary_matched_43k"], "summary control")
require(manifest.get("seeds") == [42], "smoke seed")
require(manifest.get("split_counts_selected") == {"train": 32, "validation": 16}, "smoke split counts")
contract = manifest.get("contract", {})
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(manifest.get("training", {}).get("full_members_for_validation") == 51, "full-member validation")
require(manifest.get("training", {}).get("automatic_mixed_precision") is True, "AMP")

required = [
    "selection.json",
    "README.md",
    "evaluation/scoring_support.npz",
    "history/training_history.csv",
    "metrics/validation_case_metrics.csv",
    "metrics/validation_summary.csv",
    "metrics/validation_by_year.csv",
    "metrics/validation_by_seed.csv",
    "figures/training_loss_curves.png",
    "figures/training_loss_curves.pdf",
    "figures/validation_capacity_curve.png",
    "figures/validation_capacity_curve.pdf",
]
required.extend(
    f"models/{arm}/seed_42/checkpoints/best.pt" for arm in arms
)
artifact_hashes = manifest.get("artifact_sha256", {})
for relative in required:
    candidate = path.parent / relative
    require(candidate.is_file(), f"missing smoke artifact {relative}")
    require(artifact_hashes.get(relative) == sha256_file(candidate), f"artifact hash {relative}")

current_sources = {
    "code/src/fuxi_allseason_capacity_ablation.py": work_root / "src/fuxi_allseason_capacity_ablation.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_allseason_capacity_ablation.sbatch": work_root / "slurm/run_allseason_capacity_ablation.sbatch",
    "code/plan/CAPACITY_PBC_STUDY_20260822.md": work_root / "plan/CAPACITY_PBC_STUDY_20260822.md",
}
source_hashes = manifest.get("source_snapshot_sha256", {})
require(set(source_hashes) == set(current_sources), "source snapshot inventory")
for relative, current in current_sources.items():
    frozen_hash = source_hashes.get(relative)
    require(isinstance(frozen_hash, str) and bool(frozen_hash), f"source hash {relative}")
    require(artifact_hashes.get(relative) == frozen_hash, f"source/artifact hash {relative}")
    require(sha256_file(path.parent / relative) == frozen_hash, f"frozen source {relative}")
    require(sha256_file(current) == frozen_hash, f"current source differs from smoke: {current}")

full_metadata_path = full_cache.with_suffix("").with_suffix(".metadata.json")
with full_metadata_path.open("r", encoding="utf-8") as stream:
    full_metadata = json.load(stream)
require(full_metadata.get("scope") == "full_archive", "full-cache scope")
require(full_metadata.get("full_archive") is True, "full-cache archive flag")
require(full_metadata.get("shape") == [2080, 51, 6, 27, 27], "full-cache shape")
require(full_metadata.get("initialization_count") == 2080, "full-cache initialization count")
require(
    full_metadata.get("source_fingerprint")
    == manifest.get("cache", {}).get("source_fingerprint"),
    "smoke/full source fingerprint agreement",
)
require(
    full_metadata.get("data_sha256")
    == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
    "canonical full-cache SHA-256",
)
for row in [*manifest.get("candidates", []), *manifest.get("controls", [])]:
    name = row.get("name")
    require(name in parameter_counts, f"unknown arm {name}")
    require(row.get("expected_parameter_count") == parameter_counts[name], f"parameter count {name}")
print(f"accepted completed capacity smoke manifest: {path}")
PY
fi

cleanup() {
    rm -rf -- "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p \
    "${WORK_ROOT}/logs" \
    "${OUTPUT_PARENT}" \
    "${JOB_TMP}/mpl" \
    "${JOB_TMP}/xdg" \
    "${JOB_TMP}/cuda"

export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "experiment=fuxi_allseason_capacity_ablation_v1 mode=${MODE}"
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
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

COMMON_ARGUMENTS=(
    --cache "${CACHE}"
    --output "${OUTPUT}"
    --device cuda
    --candidates "${CANDIDATES}"
    --batch-size 8
    --member-subsample 16
    --learning-rate 0.0002
    --weight-decay 0.0001
    --evaluation-batch-size 8
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_capacity_ablation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42 \
        --max-epochs 2 \
        --patience 1 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_capacity_ablation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42,43,44 \
        --max-epochs 100 \
        --patience 15
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${CACHE}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
cache = Path(sys.argv[3])
candidates = ["small_20k", "base_42k", "medium_158k", "large_294k"]
arms = [*candidates, "summary_matched_43k"]
parameter_counts = {
    "small_20k": 19_618,
    "base_42k": 42_434,
    "medium_158k": 157_570,
    "large_294k": 293_762,
    "summary_matched_43k": 43_058,
}
expected_seeds = [42] if mode == "smoke" else [42, 43, 44]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"post-run gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

required = [
    "manifest.json",
    "selection.json",
    "README.md",
    "evaluation/scoring_support.npz",
    "history/training_history.csv",
    "metrics/validation_case_metrics.csv",
    "metrics/validation_summary.csv",
    "metrics/validation_by_year.csv",
    "metrics/validation_by_seed.csv",
    "figures/training_loss_curves.png",
    "figures/training_loss_curves.pdf",
    "figures/validation_capacity_curve.png",
    "figures/validation_capacity_curve.pdf",
]
required.extend(
    f"models/{arm}/seed_{seed}/checkpoints/best.pt"
    for arm in arms
    for seed in expected_seeds
)
missing = [relative for relative in required if not (root / relative).is_file()]
require(not missing, f"missing required artifacts: {missing}")

with (root / "manifest.json").open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
with (root / "selection.json").open("r", encoding="utf-8") as stream:
    selection = json.load(stream)
require(manifest.get("experiment") == "fuxi_allseason_capacity_ablation_v1", "experiment")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == mode and manifest.get("smoke") is (mode == "smoke"), "mode")
require([row.get("name") for row in manifest.get("candidates", [])] == candidates, "candidate order")
require([row.get("name") for row in manifest.get("controls", [])] == ["summary_matched_43k"], "summary control")
require(manifest.get("seeds") == expected_seeds, "seed contract")
expected_counts = (
    {"train": 32, "validation": 16}
    if mode == "smoke"
    else {"train": 1652, "validation": 196}
)
require(manifest.get("split_counts_selected") == expected_counts, "selected split counts")
contract = manifest.get("contract", {})
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
training = manifest.get("training", {})
require(training.get("full_members_for_validation") == 51, "full-member validation")
require(training.get("automatic_mixed_precision") is True, "AMP")
require(len(training.get("runs", [])) == len(arms) * len(expected_seeds), "training run count")

run_keys = set()
for row in training.get("runs", []):
    name = row.get("candidate")
    seed = row.get("seed")
    require(name in parameter_counts and seed in expected_seeds, "training run identity")
    require(row.get("parameter_count") == parameter_counts[name], f"run parameter count {name}")
    require(math.isfinite(float(row.get("best_validation_crps", float("nan")))), f"finite validation CRPS {name}/{seed}")
    run_keys.add((name, seed))
require(run_keys == {(arm, seed) for arm in arms for seed in expected_seeds}, "complete arm/seed grid")

require(selection.get("status") == "validation_selection_locked", "selection lock")
require(selection.get("test_metrics_consulted") is False, "selection test firewall")
require(selection.get("selected_candidate") in candidates, "selected candidate")
selected_record = next(
    row for row in selection.get("validation_candidates", [])
    if row.get("candidate") == selection.get("selected_candidate")
)
require(selection.get("selected_parameter_count") == selected_record.get("parameter_count"), "selection parameter count")
rules = selection.get("rules", {})
require(
    rules.get("minimum_matched_seed_improvements") == min(2, len(expected_seeds)),
    "matched-seed rule",
)
if selection.get("selected_candidate") != "base_42k":
    require(selected_record.get("eligible_promotion") is True, "selected wider candidate eligibility")
    require(selected_record.get("matched_seed_guard") is True, "selected matched-seed guard")
    require(
        selected_record.get("matched_seed_improvement_passes", 0)
        >= selected_record.get("matched_seed_improvement_required", len(expected_seeds) + 1),
        "selected matched-seed count",
    )
summary_control = selection.get("parameter_matched_summary_control", {})
require(summary_control.get("candidate") == "summary_matched_43k", "selection summary control")
require(summary_control.get("eligible_promotion") is False, "summary control ineligibility")

artifact_hashes = manifest.get("artifact_sha256", {})
for relative in required[1:]:
    require(artifact_hashes.get(relative) == sha256_file(root / relative), f"artifact hash {relative}")
source_hashes = manifest.get("source_snapshot_sha256", {})
require(bool(source_hashes), "source snapshot")
for relative, digest in source_hashes.items():
    require(artifact_hashes.get(relative) == digest, f"source/artifact hash {relative}")
    require((root / relative).is_file(), f"frozen source {relative}")
    require(sha256_file(root / relative) == digest, f"frozen source hash {relative}")
require(manifest.get("cache", {}).get("data_sha256") == sha256_file(cache), "cache hash")
if mode == "full":
    require(
        manifest.get("cache", {}).get("data_sha256")
        == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
        "canonical full-cache SHA-256",
    )
    summary_rows = manifest.get("selection", {}).get("validation_candidates", [])
    base_row = next(row for row in summary_rows if row.get("candidate") == "base_42k")
    base_crps = float(base_row.get("mean_validation_crps", float("nan")))
    canonical_base_crps = 1.334153408298687
    require(math.isfinite(base_crps), "finite base validation CRPS")
    require(
        abs(base_crps - canonical_base_crps) / canonical_base_crps <= 0.005,
        "base_42k does not reproduce V1 validation CRPS within 0.5%",
    )
print(root / "manifest.json")
PY

echo "complete: ${OUTPUT}"
