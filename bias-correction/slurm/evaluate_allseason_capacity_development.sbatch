#!/usr/bin/env bash
#SBATCH --job-name=fuxi-cap-dev
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch slurm/evaluate_allseason_capacity_development.sbatch \
    smoke /absolute/path/completed-capacity-full/manifest.json \
    /absolute/path/fresh-development-smoke

  sbatch slurm/evaluate_allseason_capacity_development.sbatch \
    full /absolute/path/completed-capacity-full/manifest.json \
    /absolute/path/fresh-development-full \
    /absolute/path/completed-development-smoke/manifest.json
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

MODE=$1
case ${MODE} in
    smoke)
        if [[ $# -ne 3 ]]; then
            usage
            exit 2
        fi
        ;;
    full)
        if [[ $# -ne 4 ]]; then
            usage
            exit 2
        fi
        ;;
    *)
        echo "mode must be smoke or full: ${MODE}" >&2
        usage
        exit 2
        ;;
esac

if [[ $2 != /* || $3 != /* ]]; then
    echo "capacity manifest and output must be absolute paths" >&2
    exit 2
fi
if [[ ${MODE} == full && $4 != /* ]]; then
    echo "completed evaluation smoke manifest must be absolute" >&2
    exit 2
fi
if [[ ! -f $2 || $(basename "$2") != manifest.json ]]; then
    echo "completed capacity manifest is unavailable: $2" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CAPACITY_MANIFEST=$(readlink -f "$2")
OUTPUT=$(readlink -m "$3")
OUTPUT_PARENT=$(dirname "${OUTPUT}")

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/src/fuxi_allseason_capacity_development_evaluation.py ]]; then
    echo "development evaluator is unavailable" >&2
    exit 2
fi
if [[ ! -f ${WORK_ROOT}/src/fuxi_allseason_capacity_ablation.py ]]; then
    echo "frozen capacity driver is unavailable" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "invalid or missing Slurm job id: ${SLURM_JOB_ID:-unset}" >&2
    exit 2
fi
JOB_TMP=$(mktemp -d "/tmp/fuxi-cap-dev-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    if [[ -d ${JOB_TMP} && ${JOB_TMP} == /tmp/fuxi-cap-dev-* ]]; then
        rm -rf -- "${JOB_TMP}"
    fi
}
trap cleanup EXIT

mkdir -p \
    "${WORK_ROOT}/logs" \
    "${OUTPUT_PARENT}" \
    "${JOB_TMP}/mpl" \
    "${JOB_TMP}/xdg" \
    "${JOB_TMP}/cuda"

export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${WORK_ROOT}"

# Cheap pre-allocation receipt check. The evaluator repeats this check and
# streams every declared SHA-256 before opening development forecasts.
"${PYTHON_BIN}" - "${CAPACITY_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"capacity preflight failed: {message}")

require(manifest.get("experiment") == "fuxi_allseason_capacity_ablation_v1", "experiment")
require(manifest.get("status") == "complete", "status")
require(manifest.get("mode") == "full" and manifest.get("smoke") is False, "full receipt")
require(manifest.get("seeds") == [42, 43, 44], "seeds")
require(
    manifest.get("split_counts_selected") == {"train": 1652, "validation": 196},
    "capacity split counts",
)
contract = manifest.get("contract", {})
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
selection = manifest.get("selection", {})
require(selection.get("status") == "validation_selection_locked", "selection lock")
require(
    selection.get("selected_candidate")
    in ["small_20k", "base_42k", "medium_158k", "large_294k"],
    "selected candidate",
)
print(f"accepted capacity preflight: {path}")
PY

if [[ ${MODE} == full ]]; then
    if [[ ! -f $4 ]]; then
        echo "completed evaluation smoke manifest is unavailable: $4" >&2
        exit 2
    fi
    SMOKE_MANIFEST=$(readlink -f "$4")
    SMOKE_GATE_RECEIPT=$(dirname "${SMOKE_MANIFEST}")/slurm_gate_receipt.json
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${CAPACITY_MANIFEST}" "${WORK_ROOT}" \
        "${SMOKE_GATE_RECEIPT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke_path = Path(sys.argv[1])
capacity_path = Path(sys.argv[2])
work_root = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"development smoke-to-full gate failed: {message}")

with smoke_path.open("r", encoding="utf-8") as stream:
    smoke = json.load(stream)
require(receipt_path.is_file(), f"missing successful Slurm gate receipt {receipt_path}")
with receipt_path.open("r", encoding="utf-8") as stream:
    receipt = json.load(stream)
require(
    receipt.get("experiment")
    == "fuxi_allseason_capacity_development_evaluation_v1",
    "gate receipt experiment",
)
require(receipt.get("gate_status") == "passed", "gate receipt status")
require(receipt.get("mode") == "smoke", "gate receipt mode")
require(
    receipt.get("manifest_sha256") == sha256_file(smoke_path),
    "gate receipt manifest binding",
)
require(
    receipt.get("post_run_audit_version") == "capacity_dev_postrun_v1",
    "gate receipt audit version",
)
require(
    smoke.get("experiment")
    == "fuxi_allseason_capacity_development_evaluation_v1",
    "experiment",
)
require(smoke.get("status") == "complete", "status")
require(smoke.get("mode") == "smoke" and smoke.get("smoke") is True, "mode")
require(
    smoke.get("capacity_receipt", {}).get("sha256") == sha256_file(capacity_path),
    "capacity receipt changed",
)
contract = smoke.get("contract", {})
require(contract.get("workflow_can_train") is False, "training firewall")
require(contract.get("workflow_can_select") is False, "selection firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
required = [
    "selection_status.json",
    "metrics/weekwise_core_metrics.csv",
    "metrics/paired_block_bootstrap.csv",
    "figures/weekwise_core_metrics.png",
    "figures/bootstrap_effects.png",
]
hashes = smoke.get("artifact_sha256", {})
for relative in required:
    candidate = smoke_path.parent / relative
    require(candidate.is_file(), f"missing smoke artifact {relative}")
    require(hashes.get(relative) == sha256_file(candidate), f"artifact hash {relative}")
source_hashes = smoke.get("source_snapshot_sha256", {})
current = {
    "code/src/fuxi_allseason_capacity_development_evaluation.py": work_root
    / "src/fuxi_allseason_capacity_development_evaluation.py",
    "code/slurm/evaluate_allseason_capacity_development.sbatch": work_root
    / "slurm/evaluate_allseason_capacity_development.sbatch",
}
for relative, path in current.items():
    require(source_hashes.get(relative) == sha256_file(path), f"current source {relative}")
print(f"accepted development smoke receipt: {smoke_path}")
PY
fi

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "experiment=fuxi_allseason_capacity_development_evaluation_v1 mode=${MODE}"
"${PYTHON_BIN}" --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import sys
import matplotlib
import numpy
import pandas
import torch
import xarray

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("sys.executable:", sys.executable)
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Xarray:", xarray.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

COMMON_ARGUMENTS=(
    --capacity-manifest "${CAPACITY_MANIFEST}"
    --output "${OUTPUT}"
    --device cuda
    --batch-size 8
    --evaluation-batch-size 8
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_capacity_development_evaluation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --bootstrap-samples 100 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_capacity_development_evaluation.py \
        "${COMMON_ARGUMENTS[@]}" \
        --bootstrap-samples 2000
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${CAPACITY_MANIFEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
capacity_manifest = Path(sys.argv[3])

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"development post-run gate failed: {message}")

required = [
    "manifest.json",
    "selection_status.json",
    "README.md",
    "inputs/capacity_manifest.json",
    "inputs/capacity_selection.json",
    "metrics/case_metrics.csv",
    "metrics/seed_case_metrics.csv",
    "metrics/weekwise_core_metrics.csv",
    "metrics/seed_weekwise_metrics.csv",
    "metrics/selected_vs_base_weekwise.csv",
    "metrics/paired_block_bootstrap.csv",
    "metrics/seed_paired_block_bootstrap.csv",
    "metrics/matched_seed_selected_vs_base_bootstrap.csv",
    "figures/weekwise_core_metrics.png",
    "figures/weekwise_core_metrics.pdf",
    "figures/bootstrap_effects.png",
    "figures/bootstrap_effects.pdf",
]
missing = [relative for relative in required if not (root / relative).is_file()]
require(not missing, f"missing artifacts: {missing}")
with (root / "manifest.json").open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
with (root / "selection_status.json").open("r", encoding="utf-8") as stream:
    selection = json.load(stream)
require(
    manifest.get("experiment")
    == "fuxi_allseason_capacity_development_evaluation_v1",
    "experiment",
)
require(manifest.get("status") == "complete", "status")
require(manifest.get("mode") == mode and manifest.get("smoke") is (mode == "smoke"), "mode")
require(
    manifest.get("capacity_receipt", {}).get("sha256")
    == sha256_file(capacity_manifest),
    "capacity receipt hash",
)
contract = manifest.get("contract", {})
require(contract.get("workflow_can_train") is False, "training firewall")
require(contract.get("workflow_can_select") is False, "selection firewall")
require(contract.get("development_metrics_used_for_selection") is False, "selection contamination")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(manifest.get("seeds") == [42, 43, 44], "seed contract")
seed_handling = manifest.get("seed_handling", {})
require(seed_handling.get("parameters_averaged") is False, "parameter averaging")
require(seed_handling.get("predictions_averaged") is False, "prediction averaging")
require(selection.get("duplicate_selected_alias_created") is False, "duplicate selected alias")
expected_count = 16 if mode == "smoke" else 208
require(
    manifest.get("split_counts_evaluated") == {"development": expected_count},
    "development count",
)
selected = selection.get("selected_candidate")
expected_methods = (
    ["raw_fuxi", "base_42k"]
    if selected == "base_42k"
    else ["raw_fuxi", "base_42k", selected]
)
require(manifest.get("methods") == expected_methods, "method order")
expected_checkpoint_count = 3 if selected == "base_42k" else 6
require(len(manifest.get("input_checkpoints", [])) == expected_checkpoint_count, "checkpoint count")
hashes = manifest.get("artifact_sha256", {})
for relative in required[1:]:
    require(hashes.get(relative) == sha256_file(root / relative), f"artifact hash {relative}")
print(root / "manifest.json")
PY

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${SLURM_JOB_ID}" \
    "${SLURMD_NODENAME:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
job = sys.argv[3]
node = sys.argv[4]
partition = sys.argv[5]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = root / "manifest.json"
if not manifest.is_file():
    raise SystemExit("cannot issue Slurm gate receipt without manifest.json")
payload = {
    "experiment": "fuxi_allseason_capacity_development_evaluation_v1",
    "gate_status": "passed",
    "mode": mode,
    "post_run_audit_version": "capacity_dev_postrun_v1",
    "manifest_sha256": sha256_file(manifest),
    "launcher_sha256": sha256_file(
        Path.cwd() / "slurm/evaluate_allseason_capacity_development.sbatch"
    ),
    "driver_sha256": sha256_file(
        Path.cwd() / "src/fuxi_allseason_capacity_development_evaluation.py"
    ),
    "job_id": job,
    "node": node,
    "partition": partition,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
}
destination = root / "slurm_gate_receipt.json"
temporary = destination.with_name(destination.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, destination)
print(destination)
PY

echo "complete: ${OUTPUT}"
