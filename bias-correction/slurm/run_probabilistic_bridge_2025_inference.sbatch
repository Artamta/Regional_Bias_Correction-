#!/usr/bin/env bash
#SBATCH --job-name=prob-2025-inf
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
[[ $# -eq 4 || $# -eq 5 ]] || { echo 'usage: <parent-manifest> <selection> <output> <batch-size> [max-cases]' >&2; exit 2; }
ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON=/home/raj.ayush/.conda/envs/s2s_core/bin/python
TMP_ROOT=$(mktemp -d /tmp/prob-2025-inf.XXXXXX)
trap 'rm -rf "${TMP_ROOT}"' EXIT
export PYTHONPATH=${ROOT}/src:${ROOT}/evaluate:${ROOT}/hpc_compat
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_CACHE_PATH=${TMP_ROOT}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd ${ROOT}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
args=(--parent-manifest "$1" --selection "$2" --output "$3" --batch-size "$4")
if [[ $# -eq 5 ]]; then args+=(--max-cases "$5"); fi
srun "${PYTHON}" evaluate/run_probabilistic_bridge_2025.py "${args[@]}"
