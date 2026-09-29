#!/bin/bash
# Two-case CUDA/data/checkpoint smoke test. Artifacts are explicitly non-selectable.

#SBATCH --job-name=quest-pr-smoke
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_smoke_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_smoke_%j.err

set -euo pipefail
: "${QUEST_PRECIP_CACHE:?export the finalized 17/2/1 precipitation cache path}"
: "${QUEST_RUN_TAG:?export a unique immutable smoke tag}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=${QUEST_PYTHON:-/home/raj.ayush/.conda/envs/fuxi/bin/python}
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/runs/smoke
mkdir -p "${ROOT}/logs"
export MPLCONFIGDIR=/tmp/quest-pr-mpl-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON}" -u "${ROOT}/train_precip.py" \
  --cache "${QUEST_PRECIP_CACHE}" \
  --run-dir "${RUN_ROOT}/${QUEST_RUN_TAG}_full_w16_seed42" \
  --variant full --width 16 --seed 42 \
  --max-epochs 1 --patience 1 --batch-size 2 --effective-batch-size 2 \
  --smoke-cases 2 --device cuda
