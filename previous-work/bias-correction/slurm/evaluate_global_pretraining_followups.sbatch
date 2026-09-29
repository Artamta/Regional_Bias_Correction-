#!/usr/bin/env bash
#SBATCH --job-name=global-followups
# The local iiser partition is restricted to the `limit` account.  This
# account must use the general GPU-node partition even for a CPU-only task.
#SBATCH --partition=gpu
# No GPU is requested; avoid the cluster's known unsafe/unreliable nodes.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch --dependency=afterok:COMPARISON_JOB_ID $0 COMPLETED_COMPARISON_ROOT FRESH_OUTPUT" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
SOURCE_ROOT=$(readlink -f "$1")
OUTPUT=$(readlink -m "$2")
JOB_TMP=/tmp/global-followups-${SLURM_JOB_ID}

if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo "completed comparison root does not exist: ${SOURCE_ROOT}" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "fresh follow-up output required: ${OUTPUT}" >&2
    exit 2
fi

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs" "${JOB_TMP}"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat:${NEURAL_SRC}
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${WORK_ROOT}"

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "source=${SOURCE_ROOT}"
echo "output=${OUTPUT}"

srun "${PYTHON_BIN}" evaluate/evaluate_global_pretraining_followups.py \
    --source-root "${SOURCE_ROOT}" \
    --output "${OUTPUT}" \
    --bootstrap-draws 10000 \
    --block-length 13 \
    --bootstrap-seed 20260822

"${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); assert m["status"]=="complete_posthoc_validation_only"; assert m["canonical_artifact"] is True; assert m["execution_tier"]=="canonical_10000draw_block13_seed20260822"; assert m["scientific_eligible"] is False; assert m["raw_archives_opened"] is False; assert m["later_year_predictions_created"] is False; assert m["later_year_metrics_computed"] is False; assert m["data"]["initialization_years"]==[2018,2019]; assert m["bootstrap"]["method"]=="paired year-stratified circular moving blocks"; assert m["bootstrap"]["draws"]==10000; assert m["bootstrap"]["block_length_initializations"]==13; assert m["bootstrap"]["seed"]==20260822; assert m["bootstrap"]["equal_marginal_date_inclusion"] is True; assert abs(m["bootstrap"]["mean_date_multiplicity"]-1.0)<1e-12; print(p/"manifest.json")' \
    "${OUTPUT}"

echo "post-hoc validation diagnostics complete; no 2020+ data were opened"
