#!/usr/bin/env bash
#SBATCH --job-name=quest-tp-cache
#SBATCH --partition=gpu
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON_BIN=/home/raj.ayush/.conda/envs/fuxi/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export BLOSC_NTHREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
srun "${PYTHON_BIN}" prepare_data.py build-cache "$@"
