#!/usr/bin/env bash
#SBATCH --job-name=fuxi-phys-post-final
#SBATCH --partition=gpu_prio
# CPU-only finalization; cn3 is MIG and cn2/cn15/cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

POST_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
POST_JOB_TMP=/tmp/fuxi-phys-post-final-${SLURM_JOB_ID}

cleanup() {
    rm -rf "${POST_JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${POST_ROOT}/logs" "${POST_JOB_TMP}"
export TMPDIR=${POST_JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${POST_ROOT}"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpu_requested=no"
srun "${PYTHON_BIN}" src/fuxi_physical_postselection_cache.py finalize "$@"
