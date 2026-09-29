#!/usr/bin/env bash
#SBATCH --job-name=quest-tp-train
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON_BIN=/home/raj.ayush/.conda/envs/fuxi/bin/python
JOB_MPL_DIR=/tmp/ai-quest-global-mpl-${SLURM_JOB_ID}

mkdir -p "${JOB_MPL_DIR}"
export MPLCONFIGDIR=${JOB_MPL_DIR}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
srun "${PYTHON_BIN}" train.py --device cuda "$@"
