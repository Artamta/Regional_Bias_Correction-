#!/usr/bin/env bash
#SBATCH --job-name=fuxi-full-late
#SBATCH --partition=gpu_prio
# Known avoid-list: cn3 is MIG; cn2, cn15, and cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python

export PYTHONPATH=${ROOT}/src:${ROOT}/evaluate:${ROOT}/hpc_compat
export MPLCONFIGDIR=/tmp/fuxi-full-late-${SLURM_JOB_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

srun "${PYTHON_BIN}" src/fuxi_imerg_full_archive_latelead.py
