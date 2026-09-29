#!/usr/bin/env bash
#SBATCH --job-name=fuxi-ens-cal-v2
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
  sbatch run_fuxi_allseason_ensemble_calibration.sbatch \
    smoke /absolute/path/cache.npy /absolute/path/fresh-smoke-output

  sbatch run_fuxi_allseason_ensemble_calibration.sbatch \
    full /absolute/path/cache.npy /absolute/path/fresh-full-output \
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
JOB_TMP=/tmp/fuxi-ens-cal-${SLURM_JOB_ID}

case ${OUTPUT} in
    ${WORK_ROOT}/resultsv3/fuxi_allseason_ensemble_calibration/*) ;;
    *)
        echo "corrected calibration output must be under resultsv3: ${OUTPUT}" >&2
        exit 2
        ;;
esac

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
manifest = json.loads(path.read_text())
assert manifest["experiment"] == "fuxi_allseason_ensemble_calibration_v2_aligned"
assert manifest["status"] == "complete"
assert manifest["mode"] == "smoke"
assert manifest["smoke"] is True
assert manifest["configurations"] == ["location_spread"]
assert manifest["seeds"] == [42]
assert manifest["methods"] == [
    "raw_fuxi",
    "moment_calibration",
    "location_spread",
]
assert manifest["split_counts_selected"] == {
    "train": 32,
    "validation": 16,
    "test": 16,
}
assert set(manifest["split_counts_archive"]) == {"train", "validation", "test", "embargo"}
assert manifest["contract"]["all_reported_ensemble_metrics_use_member_count"] == 51
assert manifest["contract"]["sealed_2025_target_opened"] is False
assert manifest["contract"]["revision"] == "imd_end_labelled_init_plus_1_through_42_v2"
assert manifest["contract"]["target_day_offsets"] == list(range(1, 43))
assert manifest["contract"]["initialization_day_included_in_target"] is False
assert manifest["contract"]["legacy_v1_zero_offset_checkpoint_loaded"] is False
assert manifest["deployment_selection"]["selected_seed"] == 42
assert manifest["deployment_selection"]["test_metrics_used_for_selection"] is False
assert manifest["seed_aggregation"]["parameter_averaging"] is False
assert manifest["seed_aggregation"]["prediction_averaging"] is False
assert manifest["training"]["device"] == "cuda"
assert manifest["training"]["full_members_for_validation_and_evaluation"] == 51
assert manifest["training"]["automatic_mixed_precision"] is True
assert manifest["software"]["cuda_available"] is True
assert manifest["software"]["cuda_device"]

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

required = (
    "history/training_history.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/case_metrics.csv",
    "metrics/seed_weekwise_metrics.csv",
    "metrics/seed_variability_by_week.csv",
    "metrics/reliability_bins.csv",
    "evaluation/scoring_support.npz",
    "figures/training_loss_curves.png",
    "figures/weekwise_metrics.png",
    "models/deployable_selection.json",
)
artifact_hashes = manifest["artifact_sha256"]
for relative in required:
    candidate = path.parent / relative
    assert candidate.is_file(), candidate
    assert artifact_hashes[relative] == sha256_file(candidate), relative

sources = {
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root
    / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root
    / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root
    / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_fuxi_allseason_ensemble_calibration.sbatch": work_root
    / "slurm/run_fuxi_allseason_ensemble_calibration.sbatch",
    "code/slurm/build_fuxi_allseason_member_cache.sbatch": work_root
    / "slurm/build_fuxi_allseason_member_cache.sbatch",
    "code/plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md": work_root
    / "plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md",
}
for relative, current in sources.items():
    frozen_hash = manifest["source_snapshot_sha256"][relative]
    assert artifact_hashes[relative] == frozen_hash
    assert sha256_file(path.parent / relative) == frozen_hash
    assert sha256_file(current) == frozen_hash, f"current source differs from passed smoke: {current}"

full_metadata_path = full_cache.with_suffix("").with_suffix(".metadata.json")
full_metadata = json.loads(full_metadata_path.read_text())
assert full_metadata["scope"] == "full_archive"
assert full_metadata["full_archive"] is True
assert full_metadata["shape"] == [2080, 51, 6, 27, 27]
assert full_metadata["initialization_count"] == 2080
smoke_fingerprint = manifest["cache"]["source_fingerprint"]
assert smoke_fingerprint
assert full_metadata["source_fingerprint"] == smoke_fingerprint, (
    "smoke and full cache derive from different native FuXi sources"
)
print(f"accepted completed smoke manifest: {path}")
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

echo "experiment=fuxi_allseason_ensemble_calibration_v2_aligned mode=${MODE}"
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" src/fuxi_allseason_member_cache.py verify --output "${CACHE}"

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
    --configs location_spread
    --batch-size 8
    --member-subsample 16
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_ensemble_calibration.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42 \
        --max-epochs 2 \
        --patience 1 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_ensemble_calibration.py \
        "${COMMON_ARGUMENTS[@]}" \
        --seeds 42,43,44 \
        --max-epochs 100 \
        --patience 15
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
required = (
    "manifest.json",
    "history/training_history.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/case_metrics.csv",
    "metrics/seed_variability_by_week.csv",
    "evaluation/scoring_support.npz",
    "figures/training_loss_curves.png",
    "figures/weekwise_metrics.png",
    "models/deployable_selection.json",
)
missing = [relative for relative in required if not (root / relative).is_file()]
assert not missing, f"missing required artifacts: {missing}"
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["experiment"] == "fuxi_allseason_ensemble_calibration_v2_aligned"
assert manifest["status"] == "complete"
assert manifest["mode"] == mode
assert manifest["smoke"] is (mode == "smoke")
assert manifest["configurations"] == ["location_spread"]
assert manifest["contract"]["all_reported_ensemble_metrics_use_member_count"] == 51
assert manifest["contract"]["sealed_2025_target_opened"] is False
assert manifest["contract"]["revision"] == "imd_end_labelled_init_plus_1_through_42_v2"
assert manifest["contract"]["target_day_offsets"] == list(range(1, 43))
assert manifest["contract"]["initialization_day_included_in_target"] is False
assert manifest["contract"]["legacy_v1_zero_offset_checkpoint_loaded"] is False
assert manifest["seed_aggregation"]["parameter_averaging"] is False
assert manifest["seed_aggregation"]["prediction_averaging"] is False
expected_seeds = [42] if mode == "smoke" else [42, 43, 44]
assert manifest["seeds"] == expected_seeds
selection = manifest["deployment_selection"]
assert selection["configuration"] == "location_spread"
assert selection["selected_seed"] in expected_seeds
assert selection["test_metrics_used_for_selection"] is False
assert selection["prediction_averaging"] is False
assert selection["parameter_averaging"] is False
assert [item["seed"] for item in selection["candidates"]] == expected_seeds
if mode == "smoke":
    assert manifest["split_counts_selected"] == {
        "train": 32,
        "validation": 16,
        "test": 16,
    }
else:
    expected_full = {"train": 1652, "validation": 196, "test": 208, "embargo": 24}
    assert manifest["split_counts_archive"] == expected_full
    assert manifest["split_counts_selected"] == {
        key: expected_full[key] for key in ("train", "validation", "test")
    }
for relative in required[1:]:
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert manifest["artifact_sha256"][relative] == digest, relative
print(root / "manifest.json")
PY

echo "complete: ${OUTPUT}"
