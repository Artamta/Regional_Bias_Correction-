#!/usr/bin/env bash
#SBATCH --job-name=fuxi-phys-finalize
#SBATCH --partition=gpu_prio
# Known avoid-list: cn3 is MIG; cn2, cn15, and cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

SWEEP_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
JOB_TMP=/tmp/fuxi-phys-finalize-${SLURM_JOB_ID}

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${SWEEP_ROOT}/logs" "${JOB_TMP}"
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${SWEEP_ROOT}"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
srun "${PYTHON_BIN}" src/fuxi_physical_feature_cache.py finalize
