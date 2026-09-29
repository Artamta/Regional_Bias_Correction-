#!/usr/bin/env bash
#SBATCH --job-name=imd-2025-final
#SBATCH --partition=GPU-AI_prio
# Defense in depth: cn3 is MIG; cn2, cn15, and cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

EVAL_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
EVAL_JOB_TMP=/tmp/imd-2025-final-${SLURM_JOB_ID}

cleanup() {
    rm -rf "${EVAL_JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${EVAL_ROOT}/logs" "${EVAL_JOB_TMP}/mpl" "${EVAL_JOB_TMP}/xdg" "${EVAL_JOB_TMP}/cuda"
export PYTHONPATH=${EVAL_ROOT}/src:${EVAL_ROOT}/evaluate:${EVAL_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${EVAL_JOB_TMP}/mpl
export XDG_CACHE_HOME=${EVAL_JOB_TMP}/xdg
export CUDA_CACHE_PATH=${EVAL_JOB_TMP}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

cd "${EVAL_ROOT}"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

# Required arguments are intentionally explicit, for example:
#   --validation-run RUN --selection-manifest FROZEN.json --output FRESH_DIR
# The Python preflight and one-time access ledger run before any 2025 store opens.
srun "${PYTHON_BIN}" evaluate/evaluate_independent_2025_control.py --require-cuda "$@"
