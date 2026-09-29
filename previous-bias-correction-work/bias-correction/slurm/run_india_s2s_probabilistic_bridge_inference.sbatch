#!/usr/bin/env bash
#SBATCH --job-name=fuxi-bridge-inf
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch run_india_s2s_probabilistic_bridge_inference.sbatch \
    smoke /absolute/path/aligned-parent/manifest.json /absolute/path/fresh-smoke-output

  sbatch run_india_s2s_probabilistic_bridge_inference.sbatch \
    full /absolute/path/aligned-parent/manifest.json /absolute/path/fresh-full-output \
    /absolute/path/completed-inference-smoke/manifest.json
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

MODE=$1
case ${MODE} in
    smoke)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        ;;
    full)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        ;;
    *)
        echo "mode must be smoke or full: ${MODE}" >&2
        usage
        exit 2
        ;;
esac

if [[ $2 != /* || $3 != /* ]]; then
    echo "parent manifest and output must be absolute paths" >&2
    exit 2
fi
if [[ ${MODE} == full && $4 != /* ]]; then
    echo "smoke manifest must be an absolute path" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
PARENT_MANIFEST=$(readlink -f "$2")
OUTPUT=$(readlink -m "$3")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
JOB_TMP=/tmp/fuxi-bridge-inf-${SLURM_JOB_ID}

case ${OUTPUT} in
    ${WORK_ROOT}/resultsv3/india_s2s_probabilistic_bridge/*) ;;
    *)
        echo "bridge inference output must be under resultsv3: ${OUTPUT}" >&2
        exit 2
        ;;
esac

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${PARENT_MANIFEST} ]]; then
    echo "aligned parent manifest is absent: ${PARENT_MANIFEST}" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi

if [[ ${MODE} == full ]]; then
    SMOKE_MANIFEST=$(readlink -f "$4")
    if [[ ! -f ${SMOKE_MANIFEST} ]]; then
        echo "completed inference smoke manifest is absent: ${SMOKE_MANIFEST}" >&2
        exit 2
    fi
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${PARENT_MANIFEST}" "${WORK_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke_path = Path(sys.argv[1])
parent_path = Path(sys.argv[2])
work_root = Path(sys.argv[3])
smoke = json.loads(smoke_path.read_text())
parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
assert smoke["experiment"] == "india_s2s_probabilistic_bridge_inference_v1"
assert smoke["status"] == "smoke_complete"
assert smoke["mode"] == "bounded_smoke"
assert smoke["inference_selection"]["selected_count"] == 5
assert smoke["inference_selection"]["last"] == "2024-12-30"
assert smoke["parent_receipt"]["parent_manifest_sha256"] == parent_sha
assert smoke["parent_receipt"]["selected_seed"] == 43
assert smoke["execution"]["resolved_device"] == "cuda"
assert smoke["execution"]["cuda_available"] is True
assert smoke["verification_truth_opened"] is False
assert smoke["sealed_2025_target_opened"] is False
assert smoke["observation_access"]["2025_observation_opened"] is False

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

root = smoke_path.parent
for relative, expected in smoke["artifact_sha256"].items():
    assert sha256_file(root / relative) == expected, relative

live = {
    "india_s2s_probabilistic_bridge_run.py": work_root / "src/india_s2s_probabilistic_bridge_run.py",
    "india_s2s_probabilistic_bridge.py": work_root / "src/india_s2s_probabilistic_bridge.py",
    "fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
}
snapshots = smoke["source_snapshot_sha256"]
for name, path in live.items():
    matches = [digest for relative, digest in snapshots.items() if Path(relative).name == name]
    assert len(matches) == 1, name
    assert sha256_file(path) == matches[0], name
print(f"accepted completed CUDA inference smoke: {smoke_path}")
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

cd "${WORK_ROOT}"

echo "experiment=india_s2s_probabilistic_bridge_inference_v1 mode=${MODE}"
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
"${PYTHON_BIN}" --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

ARGUMENTS=(
    inference
    --parent-manifest "${PARENT_MANIFEST}"
    --output "${OUTPUT}"
    --device cuda
    --batch-size 8
)
if [[ ${MODE} == smoke ]]; then
    ARGUMENTS+=(--smoke)
fi

srun "${PYTHON_BIN}" src/india_s2s_probabilistic_bridge_run.py "${ARGUMENTS[@]}"

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["experiment"] == "india_s2s_probabilistic_bridge_inference_v1"
assert manifest["status"] == ("smoke_complete" if mode == "smoke" else "complete")
assert manifest["mode"] == ("bounded_smoke" if mode == "smoke" else "full")
expected_count = 5 if mode == "smoke" else 517
expected_scoreable = 4 if mode == "smoke" else 505
assert manifest["inference_selection"]["selected_count"] == expected_count
assert manifest["inference_selection"]["scoreable_count"] == expected_scoreable
assert sum(item["case_count"] for item in manifest["inference_shards"]) == expected_count
assert sum(item["scoreable_case_count"] for item in manifest["inference_shards"]) == expected_scoreable
assert manifest["execution"]["resolved_device"] == "cuda"
assert manifest["execution"]["cuda_available"] is True
assert manifest["parent_receipt"]["selected_seed"] == 43
assert manifest["verification_truth_opened"] is False
assert manifest["sealed_2025_target_opened"] is False
assert manifest["observation_access"]["2025_observation_opened"] is False

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for relative, expected in manifest["artifact_sha256"].items():
    candidate = root / relative
    assert candidate.is_file(), relative
    assert sha256_file(candidate) == expected, relative
print(root / "manifest.json")
PY

echo "complete: ${OUTPUT}"
