#!/bin/bash
# Full candidate/seeds job; values are explicit submit-time environment inputs.

#SBATCH --job-name=quest-pr-cand
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_candidate_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_candidate_%j.err

set -euo pipefail
: "${QUEST_PRECIP_CACHE:?export the finalized precipitation cache path}"
: "${QUEST_VARIANT:?export target_only, tp_only, or full after validation screening}"
: "${QUEST_WIDTH:?export 16 or validation-authorized 24}"
: "${QUEST_SEED:?export 42, 43, or 44 according to the promotion ladder}"
: "${QUEST_RUN_TAG:?export a unique immutable run tag}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=${QUEST_PYTHON:-/home/raj.ayush/.conda/envs/fuxi/bin/python}
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/runs
mkdir -p "${ROOT}/logs"
export MPLCONFIGDIR=/tmp/quest-pr-mpl-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON}" -u "${ROOT}/train_precip.py" \
  --cache "${QUEST_PRECIP_CACHE}" \
  --run-dir "${RUN_ROOT}/${QUEST_RUN_TAG}_${QUEST_VARIANT}_w${QUEST_WIDTH}_seed${QUEST_SEED}" \
  --variant "${QUEST_VARIANT}" --width "${QUEST_WIDTH}" --seed "${QUEST_SEED}" \
  --max-epochs 30 --patience 6 --batch-size 2 --effective-batch-size 8 --device cuda
