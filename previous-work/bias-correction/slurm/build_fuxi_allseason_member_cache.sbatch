#!/usr/bin/env bash
#SBATCH --job-name=fuxi-ens-cache
#SBATCH --partition=gpu
# CPU-only job on the generic partition; retain the full S2S avoid-list.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --array=0-259%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.err

set -euo pipefail
umask 027

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 {pilot|build|finalize}" >&2
    exit 2
fi

MODE=$1
if [[ ${MODE} != pilot && ${MODE} != build && ${MODE} != finalize ]]; then
    echo "mode must be pilot, build, or finalize: ${MODE}" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
SOURCE_STORE=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_global_2002_2021.zarr
PARTS_DIR=${WORK_ROOT}/cache/fuxi_tp_members_weekly_2002_2021_allseason_v1.parts
OUTPUT=${WORK_ROOT}/cache/fuxi_tp_members_weekly_2002_2021_allseason_v1.npy
PILOT_ROOT=${WORK_ROOT}/cache/smoke/fuxi_allseason_member_cache_${SLURM_JOB_ID}
PILOT_PARTS=${PILOT_ROOT}/parts
PILOT_OUTPUT=${PILOT_ROOT}/cache.npy
TASK_COUNT=260
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
JOB_TMP=/tmp/fuxi-ens-cache-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${TASK_INDEX}

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d ${SOURCE_STORE} ]]; then
    echo "source Zarr does not exist: ${SOURCE_STORE}" >&2
    exit 2
fi
if [[ ${MODE} == build && ${ARRAY_COUNT} -ne ${TASK_COUNT} ]]; then
    echo "build requires the declared ${TASK_COUNT}-task array; got ${ARRAY_COUNT}" >&2
    exit 2
fi
if [[ ${MODE} != build && ${ARRAY_COUNT} -ne 1 ]]; then
    echo "${MODE} must be submitted with --array=0" >&2
    exit 2
fi

cleanup() {
    rm -rf -- "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/cache" "${JOB_TMP}"
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLOSC_NTHREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${WORK_ROOT}"

echo "mode=${MODE} job=${SLURM_JOB_ID} host=$(hostname) task=${TASK_INDEX}/${ARRAY_COUNT}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

if [[ ${MODE} == pilot ]]; then
    "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        inventory \
        --smoke
    srun "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        build \
        --parts-dir "${PILOT_PARTS}" \
        --task-index 0 \
        --task-count 1 \
        --smoke
    srun "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        finalize \
        --parts-dir "${PILOT_PARTS}" \
        --output "${PILOT_OUTPUT}" \
        --smoke
    "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        verify \
        --output "${PILOT_OUTPUT}"
    echo "complete non-scientific pilot cache: ${PILOT_OUTPUT}"
elif [[ ${MODE} == build ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        build \
        --parts-dir "${PARTS_DIR}" \
        --task-index "${TASK_INDEX}" \
        --task-count "${TASK_COUNT}"
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        finalize \
        --parts-dir "${PARTS_DIR}" \
        --output "${OUTPUT}"
    "${PYTHON_BIN}" src/fuxi_allseason_member_cache.py \
        --source-store "${SOURCE_STORE}" \
        verify \
        --output "${OUTPUT}"
    echo "complete cache: ${OUTPUT}"
fi
