#!/usr/bin/env bash
#SBATCH --job-name=rawid-2025-final
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
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
FINAL_TMP=$(mktemp -d /tmp/rawid-2025-final.XXXXXX)

cleanup() {
    rm -rf "${FINAL_TMP}"
}
trap cleanup EXIT

mkdir -p "${EVAL_ROOT}/logs" "${FINAL_TMP}/cuda"
export PYTHONPATH=${EVAL_ROOT}/src:${EVAL_ROOT}/evaluate:${EVAL_ROOT}/hpc_compat:${NEURAL_SRC}
export CUDA_CACHE_PATH=${FINAL_TMP}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${EVAL_ROOT}"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU; final access not consumed")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

# The only caller-set paths are the frozen selection and exact user approval.
# Preflight proof, three locators, output, ledger, and attempt status are all
# derived from the authenticated selection. Do not submit without approval.
srun "${PYTHON_BIN}" evaluate/evaluate_raw_identity_2025.py --require-cuda "$@"
