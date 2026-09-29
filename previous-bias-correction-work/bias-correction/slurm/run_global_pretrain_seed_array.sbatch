#!/usr/bin/env bash
#SBATCH --job-name=global-pretrain
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch $0 COMPLETED_SMOKE_MANIFEST FRESH_PRETRAIN_ROOT" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python

SMOKE_PARENT=$(readlink -f "$(dirname "$1")")
SMOKE_MANIFEST=${SMOKE_PARENT}/$(basename "$1")
PRETRAIN_PARENT=$(readlink -f "$(dirname "$2")")
PRETRAIN_ROOT=${PRETRAIN_PARENT}/$(basename "$2")
SEEDS=(42 43 44)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
RUN_NAME=full_seed${SEED}
RUN_DIR=${PRETRAIN_ROOT}/${RUN_NAME}
JOB_TMP=/tmp/global-pretrain-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

if [[ ! -f "${SMOKE_MANIFEST}" ]]; then
    echo "completed smoke manifest does not exist: ${SMOKE_MANIFEST}" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "fresh seed run directory required: ${RUN_DIR}" >&2
    exit 2
fi

"${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; m=json.loads(Path(sys.argv[1]).read_text()); assert m["status"]=="complete_smoke"; assert m["smoke"] is True; assert m["scientific_eligible"] is False; assert m["test_predictions_created"] is False; assert m["promotion"]["selected_system"]=="none_smoke_is_non_scientific"' \
    "${SMOKE_MANIFEST}"

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p \
    "${WORK_ROOT}/logs" \
    "${JOB_TMP}/mpl" \
    "${JOB_TMP}/xdg" \
    "${JOB_TMP}/cuda"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${WORK_ROOT}"

echo "job=${SLURM_JOB_ID} array_job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "starting global pretraining seed=${SEED}: ${RUN_DIR}"

srun "${PYTHON_BIN}" src/fuxi_global_patch_pretraining.py \
    --mode full \
    --output-root "${PRETRAIN_ROOT}" \
    --run-name "${RUN_NAME}" \
    --seed "${SEED}" \
    --device cuda \
    --num-workers 4

"${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); expected=int(sys.argv[2]); m=json.loads((root/"manifest.json").read_text()); assert m["status"]=="complete"; assert m["smoke"] is False; assert m["scientific_eligible"] is False; assert m["test_predictions_created"] is False; assert m["contract"]["seed"]==expected; assert (root/"checkpoints"/"best.pt").is_file(); print(root/"manifest.json")' \
    "${RUN_DIR}" "${SEED}"

