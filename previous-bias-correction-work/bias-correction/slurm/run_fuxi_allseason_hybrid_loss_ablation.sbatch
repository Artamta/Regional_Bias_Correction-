#!/usr/bin/env bash
#SBATCH --job-name=fuxi-hybrid-loss
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
  sbatch run_fuxi_allseason_hybrid_loss_ablation.sbatch \
    smoke /absolute/path/smoke-cache.npy /absolute/path/fresh-smoke-output

  sbatch run_fuxi_allseason_hybrid_loss_ablation.sbatch \
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

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE=$(readlink -f "$2")
OUTPUT=$(readlink -m "$3")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
JOB_TMP=/tmp/fuxi-hybrid-loss-${SLURM_JOB_ID}
PROFILES=crps_only,hybrid_010,hybrid_025,hybrid_050,mse_only

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${CACHE} ]]; then
    echo "member cache does not exist: ${CACHE}" >&2
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
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"cache contract failed: {message}")

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

hasher = hashlib.sha256()
with cache.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(block)
digest = hasher.hexdigest()
require(metadata.get("data_sha256") == digest, "data SHA-256")
if mode == "smoke":
    require(
        metadata.get("scope")
        == "stratified_32_train_16_validation_16_test_non_scientific",
        "smoke scope",
    )
    require(metadata.get("full_archive") is False, "smoke full_archive flag")
    require(metadata.get("initialization_count") == 64, "smoke initialization count")
    require(metadata.get("shape") == [64, 51, 6, 27, 27], "smoke shape")
else:
    require(metadata.get("scope") == "full_archive", "full scope")
    require(metadata.get("full_archive") is True, "full_archive flag")
    require(metadata.get("initialization_count") == 2080, "full initialization count")
    require(metadata.get("shape") == [2080, 51, 6, 27, 27], "full shape")
    require(
        digest == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
        "canonical full-cache SHA-256",
    )
print(f"accepted verified {mode} cache: {cache}")
PY

if [[ ${MODE} == full ]]; then
    SMOKE_MANIFEST=$(readlink -f "$4")
    if [[ ! -f ${SMOKE_MANIFEST} ]]; then
        echo "completed smoke manifest does not exist: ${SMOKE_MANIFEST}" >&2
        exit 2
    fi
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${WORK_ROOT}" "${CACHE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
work_root = Path(sys.argv[2])
full_cache = Path(sys.argv[3])
manifest = json.loads(path.read_text(encoding="utf-8"))
profiles = ["crps_only", "hybrid_010", "hybrid_025", "hybrid_050", "mse_only"]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"smoke-to-full gate failed: {message}")

require(
    manifest.get("experiment") == "fuxi_allseason_hybrid_loss_ablation_v1",
    "experiment identity",
)
require(manifest.get("status") == "complete", "smoke completion status")
require(manifest.get("mode") == "smoke", "smoke mode")
require(manifest.get("profiles") == profiles, "profile order")
require(manifest.get("seeds") == [42], "smoke seed")
require(
    manifest.get("cache", {}).get("source_fingerprint")
    == "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae",
    "cache source fingerprint",
)

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

required = (
    "selection.json",
    "history/training_history.csv",
    "metrics/pooled_metrics.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/case_metrics.csv",
)
artifact_hashes = manifest["artifact_sha256"]
for relative in required:
    candidate = path.parent / relative
    require(candidate.is_file(), f"missing smoke artifact {relative}")
    require(
        artifact_hashes.get(relative) == sha256_file(candidate),
        f"smoke artifact hash {relative}",
    )

sources = {
    "code/src/fuxi_allseason_hybrid_loss_ablation.py": work_root
    / "src/fuxi_allseason_hybrid_loss_ablation.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root
    / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root
    / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root
    / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_fuxi_allseason_hybrid_loss_ablation.sbatch": work_root
    / "slurm/run_fuxi_allseason_hybrid_loss_ablation.sbatch",
    "code/plan/HYBRID_LOSS_ABLATION_20260822.md": work_root
    / "plan/HYBRID_LOSS_ABLATION_20260822.md",
}
source_hashes = manifest["source_snapshot_sha256"]
for relative, current in sources.items():
    frozen_hash = source_hashes.get(relative)
    require(isinstance(frozen_hash, str) and bool(frozen_hash), f"source hash {relative}")
    require(artifact_hashes.get(relative) == frozen_hash, f"artifact/source hash {relative}")
    require(
        sha256_file(path.parent / relative) == frozen_hash,
        f"frozen source content {relative}",
    )
    require(
        sha256_file(current) == frozen_hash,
        f"current source differs from completed smoke: {current}",
    )

full_metadata_path = full_cache.with_suffix("").with_suffix(".metadata.json")
full_metadata = json.loads(full_metadata_path.read_text(encoding="utf-8"))
require(full_metadata.get("scope") == "full_archive", "full-cache scope")
require(full_metadata.get("full_archive") is True, "full-cache archive flag")
require(full_metadata.get("shape") == [2080, 51, 6, 27, 27], "full-cache shape")
require(full_metadata.get("initialization_count") == 2080, "full-cache initialization count")
require(
    full_metadata.get("source_fingerprint")
    == manifest.get("cache", {}).get("source_fingerprint"),
    "smoke/full source fingerprint agreement",
)
print(f"accepted completed V2 smoke manifest: {path}")
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

echo "experiment=fuxi_allseason_hybrid_loss_ablation_v1 mode=${MODE}"
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import sys

import numpy
import torch

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("sys.executable:", sys.executable)
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

COMMON_ARGUMENTS=(
    --cache "${CACHE}"
    --output "${OUTPUT}"
    --device cuda
    --profiles "${PROFILES}"
    --batch-size 8
    --member-subsample 16
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_hybrid_loss_ablation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42 \
        --max-epochs 2 \
        --patience 1 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_hybrid_loss_ablation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42,43,44 \
        --max-epochs 100 \
        --patience 15
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${CACHE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
cache = Path(sys.argv[3])
profiles = ["crps_only", "hybrid_010", "hybrid_025", "hybrid_050", "mse_only"]
required = (
    "manifest.json",
    "selection.json",
    "history/training_history.csv",
    "metrics/pooled_metrics.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/case_metrics.csv",
)
missing = [relative for relative in required if not (root / relative).is_file()]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"post-run gate failed: {message}")

require(not missing, f"missing required artifacts: {missing}")

manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
require(
    manifest.get("experiment") == "fuxi_allseason_hybrid_loss_ablation_v1",
    "experiment identity",
)
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == mode, "run mode")
require(manifest.get("profiles") == profiles, "profile order")
expected_seeds = [42] if mode == "smoke" else [42, 43, 44]
require(manifest.get("seeds") == expected_seeds, "seed contract")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

require(
    manifest.get("cache", {}).get("data_sha256") == sha256_file(cache),
    "cache data SHA-256",
)
require(
    manifest.get("cache", {}).get("source_fingerprint")
    == "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae",
    "cache source fingerprint",
)
if mode == "full":
    require(
        manifest.get("cache", {}).get("data_sha256")
        == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
        "canonical full-cache SHA-256",
    )
    selected = manifest.get("selected_profile")
    if selected is None:
        selected = manifest["selection"]["selected_profile"]
    require(selected == selection.get("selected_profile"), "selection agreement")
    require(selected in profiles[:-1], "selected arm eligibility")

artifact_hashes = manifest["artifact_sha256"]
for relative in required[1:]:
    require(
        artifact_hashes.get(relative) == sha256_file(root / relative),
        f"artifact hash {relative}",
    )
require(bool(manifest.get("source_snapshot_sha256")), "source snapshot hashes")
print(root / "manifest.json")
PY

echo "complete: ${OUTPUT}"
