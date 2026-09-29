#!/usr/bin/env bash
#SBATCH --job-name=global-india-smoke
#SBATCH --partition=gpu
# cn3 is MIG; the other listed generic-GPU nodes have measured launch,
# network, or step reliability failures in this workspace.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch $0 FRESH_PRETRAIN_ROOT FRESH_COMPARISON_ROOT" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python

PRETRAIN_PARENT=$(readlink -f "$(dirname "$1")")
COMPARISON_PARENT=$(readlink -f "$(dirname "$2")")
PRETRAIN_ROOT=${PRETRAIN_PARENT}/$(basename "$1")
COMPARISON_ROOT=${COMPARISON_PARENT}/$(basename "$2")
PRETRAIN_RUN=${PRETRAIN_ROOT}/smoke_seed42
PRETRAIN_CHECKPOINT=${PRETRAIN_RUN}/checkpoints/best.pt
JOB_TMP=/tmp/global-india-smoke-${SLURM_JOB_ID}

if [[ -e "${PRETRAIN_ROOT}" ]]; then
    echo "fresh pretraining root required: ${PRETRAIN_ROOT}" >&2
    exit 2
fi
if [[ -e "${COMPARISON_ROOT}" ]]; then
    echo "fresh comparison root required: ${COMPARISON_ROOT}" >&2
    exit 2
fi

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

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "starting global-patch execution smoke: ${PRETRAIN_RUN}"
srun "${PYTHON_BIN}" src/fuxi_global_patch_pretraining.py \
    --mode smoke \
    --output-root "${PRETRAIN_ROOT}" \
    --run-name smoke_seed42 \
    --seed 42 \
    --device cuda \
    --num-workers 2

"${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); m=json.loads((root/"manifest.json").read_text()); assert m["status"]=="complete"; assert m["smoke"] is True; assert m["scientific_eligible"] is False; assert m["test_predictions_created"] is False; assert (root/"checkpoints"/"best.pt").is_file()' \
    "${PRETRAIN_RUN}"

echo "global smoke passed; starting matched India execution smoke"
srun "${PYTHON_BIN}" src/fuxi_imd_global_pretraining_comparison.py \
    --smoke \
    --pretrained-checkpoint "${PRETRAIN_CHECKPOINT}" \
    --output-root "${COMPARISON_ROOT}" \
    --device cuda:0

"${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); runs=[p for p in root.iterdir() if p.is_dir() and (p/"manifest.json").is_file()]; assert len(runs)==1, runs; m=json.loads((runs[0]/"manifest.json").read_text()); assert m["status"]=="complete_smoke"; assert m["smoke"] is True; assert m["scientific_eligible"] is False; assert m["test_predictions_created"] is False; assert m["promotion"]["selected_system"]=="none_smoke_is_non_scientific"; print(runs[0]/"manifest.json")' \
    "${COMPARISON_ROOT}"

echo "smoke complete; inspect both manifests and this GPU log before submitting the full launcher"
