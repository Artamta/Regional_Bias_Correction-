#!/usr/bin/env bash
#SBATCH --job-name=fuxi-phys-nn
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch slurm/run_allseason_physical_context.sbatch \
    smoke /absolute/member-cache.npy /absolute/physical-cache.npy \
    /absolute/fresh-smoke-output

  sbatch slurm/run_allseason_physical_context.sbatch \
    full /absolute/member-cache.npy /absolute/physical-cache.npy \
    /absolute/fresh-full-output /absolute/completed-smoke/manifest.json

Both modes require the canonical full 2,080-case caches. Full mode requires
the audited smoke produced by this exact source bundle and these exact caches.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi
MODE=$1
case ${MODE} in
    smoke) [[ $# -eq 4 ]] || { usage; exit 2; } ;;
    full) [[ $# -eq 5 ]] || { usage; exit 2; } ;;
    *) echo "mode must be smoke or full: ${MODE}" >&2; usage; exit 2 ;;
esac
for value in "$2" "$3" "$4"; do
    [[ ${value} == /* ]] || { echo "arguments must be absolute: ${value}" >&2; exit 2; }
done
if [[ ${MODE} == full && $5 != /* ]]; then
    echo "completed smoke manifest must be absolute" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
MEMBER_CACHE=$(readlink -f "$2")
PHYSICAL_CACHE=$(readlink -f "$3")
OUTPUT=$(readlink -m "$4")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
OUTPUT_NAME=$(basename "${OUTPUT}")
SMOKE_MANIFEST=
if [[ ${MODE} == full ]]; then
    SMOKE_MANIFEST=$(readlink -f "$5")
fi

PLAN=${WORK_ROOT}/plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md
EXPECTED_PLAN_SHA256=98c5af8cca4bd273c4271d6ef70ff97741552ca606dd9faaa241e665a632c7df
EXPECTED_MEMBER_DATA_SHA256=2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70
EXPECTED_MEMBER_SOURCE_FINGERPRINT=655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae
EXPECTED_PHYSICAL_SOURCE_FINGERPRINT=7e3bdc0ab304fafb54556558b8fa2872c6d2add71361ee8dbba328c6d01757bb
EXPECTED_FEATURE_CONTRACT_SHA256=b9c0c725c3a0b800e0fe26d871b8440bfde7416ccde3fac9bf5bcb7cba198ba4
ARMS=base_42k,zero_physical_49k,physical_context_49k

REQUIRED_SOURCES=(
    src/fuxi_allseason_physical_context.py
    src/fuxi_allseason_physical_context_cache.py
    src/fuxi_allseason_member_cache.py
    src/fuxi_physical_feature_cache.py
    src/fuxi_allseason_ensemble_calibration.py
    src/fuxi_ensemble_calibration_core.py
    src/fuxi_persistence_context.py
    src/fuxi_allseason_persistence_augmented.py
    src/fuxi_lead_gated_cdf_hybrid_validation.py
    src/fuxi_allseason_operational_categorical_comparison.py
    src/fuxi_allseason_pbc_baseline.py
    src/fuxi_pbc_core.py
    tests/test_allseason_physical_context.py
    tests/test_allseason_physical_context_cache.py
    plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md
    slurm/build_fuxi_allseason_physical_context_cache.sbatch
    slurm/run_allseason_physical_context.sbatch
)

[[ -x ${PYTHON_BIN} ]] || { echo "missing s2s_core Python" >&2; exit 2; }
[[ -f ${MEMBER_CACHE} && -f ${PHYSICAL_CACHE} ]] || {
    echo "canonical member and physical caches must exist" >&2; exit 2;
}
[[ ${MODE} != full || -f ${SMOKE_MANIFEST} ]] || {
    echo "missing smoke manifest: ${SMOKE_MANIFEST}" >&2; exit 2;
}
[[ ! -e ${OUTPUT} ]] || { echo "fresh output required: ${OUTPUT}" >&2; exit 2; }
for relative in "${REQUIRED_SOURCES[@]}"; do
    [[ -f ${WORK_ROOT}/${relative} ]] || { echo "missing source: ${relative}" >&2; exit 2; }
done
[[ $(sha256sum "${PLAN}" | awk '{print $1}') == ${EXPECTED_PLAN_SHA256} ]] || {
    echo "frozen plan SHA-256 moved" >&2; exit 2;
}
command -v srun >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 || {
    echo "srun and nvidia-smi are required" >&2; exit 2;
}

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}"
JOB_ID=${SLURM_JOB_ID:?submit this launcher with sbatch}
JOB_TMP=$(mktemp -d "/tmp/fuxi-phys-nn-${JOB_ID}.XXXXXX")
OUTPUT_LOCK=${OUTPUT_PARENT}/.${OUTPUT_NAME}.physical-context-launch.lock
LOCK_HELD=false
cleanup() {
    rm -rf -- "${JOB_TMP}"
    if [[ ${LOCK_HELD} == true ]]; then rmdir -- "${OUTPUT_LOCK}" 2>/dev/null || true; fi
}
trap cleanup EXIT
mkdir -- "${OUTPUT_LOCK}" || { echo "output launch lock exists" >&2; exit 2; }
LOCK_HELD=true

mkdir -p "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export TMPDIR=${JOB_TMP}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
cd "${WORK_ROOT}"

"${PYTHON_BIN}" src/fuxi_allseason_member_cache.py verify --output "${MEMBER_CACHE}"
"${PYTHON_BIN}" src/fuxi_allseason_physical_context_cache.py verify \
    --output "${PHYSICAL_CACHE}"

CACHE_RECEIPT=$(
    "${PYTHON_BIN}" - "${MEMBER_CACHE}" "${PHYSICAL_CACHE}" \
        "${EXPECTED_MEMBER_DATA_SHA256}" "${EXPECTED_MEMBER_SOURCE_FINGERPRINT}" \
        "${EXPECTED_PHYSICAL_SOURCE_FINGERPRINT}" \
        "${EXPECTED_FEATURE_CONTRACT_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

member, physical = map(lambda value: Path(value).resolve(), sys.argv[1:3])
member_sha, member_source, physical_source, feature_contract = sys.argv[3:7]

def require(condition, message):
    if not condition:
        raise SystemExit(f"cache gate failed: {message}")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sidecars(path):
    base = path.with_suffix("")
    return tuple(base.with_name(base.name + suffix) for suffix in (".metadata.json", ".manifest.json"))

member_meta_path, member_manifest_path = sidecars(member)
physical_meta_path, physical_manifest_path = sidecars(physical)
for path in (member_meta_path, member_manifest_path, physical_meta_path, physical_manifest_path):
    require(path.is_file(), f"missing sidecar {path}")
member_meta = json.loads(member_meta_path.read_text(encoding="utf-8"))
physical_meta = json.loads(physical_meta_path.read_text(encoding="utf-8"))
require(member_meta.get("data_sha256") == member_sha, "member SHA")
require(member_meta.get("source_fingerprint") == member_source, "member source")
require(member_meta.get("shape") == [2080, 51, 6, 27, 27], "member shape")
require(member_meta.get("scope") == "full_archive" and member_meta.get("full_archive") is True, "member scope")
require(physical_meta.get("source_fingerprint") == physical_source, "physical source")
require(physical_meta.get("feature_contract_sha256") == feature_contract, "feature contract")
require(physical_meta.get("shape") == [2080, 6, 10, 27, 27], "physical shape")
require(physical_meta.get("scope") == "full_archive" and physical_meta.get("full_archive") is True, "physical scope")
require(physical_meta.get("normalization") == "none", "physical normalization")
require(physical_meta.get("dims") == ["init", "lead_week", "feature", "lat", "lon"], "physical dims")
require(physical_meta.get("initializations") == member_meta.get("initializations"), "initializations")
require(physical_meta.get("latitude") == member_meta.get("latitude"), "latitude")
require(physical_meta.get("longitude") == member_meta.get("longitude"), "longitude")
physical_sha = physical_meta.get("data_sha256")
require(isinstance(physical_sha, str) and len(physical_sha) == 64, "physical SHA")
print(
    physical_sha,
    sha256_file(physical_meta_path),
    sha256_file(physical_manifest_path),
    sha256_file(member_meta_path),
    sha256_file(member_manifest_path),
)
PY
)
read -r PHYSICAL_DATA_SHA256 PHYSICAL_METADATA_SHA256 \
    PHYSICAL_MANIFEST_SHA256 MEMBER_METADATA_SHA256 MEMBER_MANIFEST_SHA256 \
    <<<"${CACHE_RECEIPT}"
for digest in "${PHYSICAL_DATA_SHA256}" "${PHYSICAL_METADATA_SHA256}" \
    "${PHYSICAL_MANIFEST_SHA256}" "${MEMBER_METADATA_SHA256}" \
    "${MEMBER_MANIFEST_SHA256}"; do
    [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || { echo "invalid cache receipt" >&2; exit 2; }
done

SMOKE_MANIFEST_SHA256=
SMOKE_GATE_RECEIPT_SHA256=
if [[ ${MODE} == full ]]; then
    SMOKE_RECEIPT=$(
        "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${WORK_ROOT}" \
            "${MEMBER_CACHE}" "${EXPECTED_MEMBER_DATA_SHA256}" \
            "${EXPECTED_MEMBER_SOURCE_FINGERPRINT}" "${PHYSICAL_DATA_SHA256}" \
            "${EXPECTED_PHYSICAL_SOURCE_FINGERPRINT}" \
            "${EXPECTED_FEATURE_CONTRACT_SHA256}" "${EXPECTED_PLAN_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
root, work_root = manifest_path.parent, Path(sys.argv[2]).resolve()
member_cache = Path(sys.argv[3]).resolve()
member_sha, member_source, physical_sha, physical_source, feature_contract, plan_sha = sys.argv[4:10]
arms = ["base_42k", "zero_physical_49k", "physical_context_49k"]

def require(condition, message):
    if not condition:
        raise SystemExit(f"smoke-to-full gate failed: {message}")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

gate_path = root / "slurm_gate_receipt.json"
require(manifest_path.name == "manifest.json" and gate_path.is_file(), "smoke receipts")
require(not (root / "failure.json").exists(), "failure receipt")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
require(manifest.get("experiment") == "fuxi_allseason_physical_context_v1", "experiment")
require(manifest.get("status") == "complete" and manifest.get("mode") == "smoke", "mode/status")
require(manifest.get("seeds") == [42], "seed")
require(manifest.get("split_counts_selected") == {"train": 32, "validation": 16}, "splits")
require([row.get("name") for row in manifest.get("arms", [])] == arms, "arms")
training = manifest.get("training", {})
require(training.get("max_epochs") == 2 and training.get("patience") == 1, "epochs")
require({(row.get("arm"), row.get("seed")) for row in training.get("runs", [])} == {(arm, 42) for arm in arms}, "run grid")
selection = manifest.get("selection", {})
require(selection.get("scientific_selection") is False, "non-scientific smoke")
require(selection.get("test_metrics_consulted") is False, "test firewall")
require(manifest.get("post_lock_benchmark", {}).get("status") == "skipped_non_scientific_smoke", "post-lock skip")
require(not (root / "post_lock_superiority.json").exists(), "superiority artifact")
contract = manifest.get("contract", {})
require(contract.get("sealed_2025_target_opened") is False, "2025 firewall")
member, physical = manifest.get("member_cache", {}), manifest.get("physical_cache", {})
require(Path(member.get("data_file", "")).resolve() == member_cache, "member path")
require(member.get("data_sha256") == member_sha and member.get("source_fingerprint") == member_source, "member identity")
require(physical.get("cache_sha256") == physical_sha, "physical SHA")
require(physical.get("source_fingerprint") == physical_source, "physical source")
require(physical.get("feature_contract_sha256") == feature_contract, "feature contract")
artifacts = manifest.get("artifact_sha256", {})
actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name not in {"manifest.json", "slurm_gate_receipt.json"}}
require(isinstance(artifacts, dict) and set(artifacts) == actual, "artifact inventory")
for relative, digest in artifacts.items():
    require(sha256_file(root / relative) == digest, f"artifact {relative}")
sources = manifest.get("source_snapshot_sha256", {})
require(sources, "source snapshot")
for snapshot, digest in sources.items():
    require(snapshot.startswith("code/") and artifacts.get(snapshot) == digest, f"source binding {snapshot}")
    relative = snapshot.removeprefix("code/")
    require(sha256_file(root / snapshot) == digest, f"snapshot {relative}")
    require(sha256_file(work_root / relative) == digest, f"live source {relative}")
require(sources.get("code/plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md") == plan_sha, "plan")
manifest_digest, gate_digest = sha256_file(manifest_path), sha256_file(gate_path)
require(gate.get("schema_version") == "fuxi_allseason_physical_context_slurm_gate_v1", "gate schema")
require(gate.get("status") == "post_run_audit_passed" and gate.get("mode") == "smoke", "gate status")
require(gate.get("manifest_sha256") == manifest_digest, "gate manifest")
require(gate.get("artifact_sha256") == artifacts and gate.get("source_snapshot_sha256") == sources, "gate bindings")
require(gate.get("physical_cache", {}).get("data_sha256") == physical_sha, "gate physical")
print(manifest_digest, gate_digest)
PY
    )
    read -r SMOKE_MANIFEST_SHA256 SMOKE_GATE_RECEIPT_SHA256 <<<"${SMOKE_RECEIPT}"
    [[ ${SMOKE_MANIFEST_SHA256} =~ ^[0-9a-f]{64}$ && ${SMOKE_GATE_RECEIPT_SHA256} =~ ^[0-9a-f]{64}$ ]] || {
        echo "invalid smoke receipt hashes" >&2; exit 2;
    }
    echo "accepted audited smoke: ${SMOKE_MANIFEST}"
fi

PARTITION=${SLURM_JOB_PARTITION:-unknown}
NODE=${SLURMD_NODENAME:-unknown}
[[ ${PARTITION} == gpu_prio ]] || { echo "gpu_prio allocation required" >&2; exit 2; }
case ${NODE} in
    cn2|cn3|cn4|cn15|cn16|cn17) echo "blacklisted node: ${NODE}" >&2; exit 2 ;;
esac

echo "experiment=fuxi_allseason_physical_context_v1 mode=${MODE}"
echo "job=${JOB_ID} node=${NODE} partition=${PARTITION}"
"${PYTHON_BIN}" --version
nvidia-smi
GPU_RECEIPT=$(nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader,nounits)
[[ -n ${GPU_RECEIPT} && ${GPU_RECEIPT} != *$'\n'* ]] || {
    echo "expected exactly one visible GPU" >&2; exit 2;
}
"${PYTHON_BIN}" - "${GPU_RECEIPT}" <<'PY'
import csv
import io
import sys
import torch

rows = list(csv.reader(io.StringIO(sys.argv[1])))
if len(rows) != 1 or len(rows[0]) != 5:
    raise SystemExit("invalid GPU receipt")
name, memory_mib = rows[0][2].strip(), int(rows[0][3].strip())
if "MIG" in name.upper() or memory_mib < 20_000:
    raise SystemExit(f"MIG/undersized GPU: {name}, {memory_mib} MiB")
if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"wrong Python: {sys.executable}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA GPU required")
assert bool(torch.isfinite(torch.ones(1, device="cuda")).all())
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
PY

COMMON_ARGUMENTS=(
    --cache "${MEMBER_CACHE}"
    --physical-cache "${PHYSICAL_CACHE}"
    --output "${OUTPUT}"
    --device cuda
    --arms "${ARMS}"
    --batch-size 8
    --member-subsample 16
    --learning-rate 0.0002
    --weight-decay 0.0001
    --num-workers 0
    --evaluation-batch-size 8
    --cdf-chunk-size 8
    --expected-physical-source-fingerprint "${EXPECTED_PHYSICAL_SOURCE_FINGERPRINT}"
)
if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_physical_context.py \
        "${COMMON_ARGUMENTS[@]}" --seeds 42 --max-epochs 2 --patience 1 --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_physical_context.py \
        "${COMMON_ARGUMENTS[@]}" --seeds 42,43,44 --max-epochs 100 --patience 15
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${WORK_ROOT}" \
    "${MEMBER_CACHE}" "${PHYSICAL_CACHE}" \
    "${EXPECTED_MEMBER_DATA_SHA256}" "${EXPECTED_MEMBER_SOURCE_FINGERPRINT}" \
    "${PHYSICAL_DATA_SHA256}" "${PHYSICAL_METADATA_SHA256}" \
    "${PHYSICAL_MANIFEST_SHA256}" "${EXPECTED_PHYSICAL_SOURCE_FINGERPRINT}" \
    "${EXPECTED_FEATURE_CONTRACT_SHA256}" "${EXPECTED_PLAN_SHA256}" \
    "${JOB_ID}" "${PARTITION}" "${NODE}" "${GPU_RECEIPT}" \
    "${SMOKE_MANIFEST}" "${SMOKE_MANIFEST_SHA256}" "${SMOKE_GATE_RECEIPT_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

root, mode, work_root = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
member_cache, physical_cache = Path(sys.argv[4]).resolve(), Path(sys.argv[5]).resolve()
member_sha, member_source = sys.argv[6:8]
physical_sha, physical_metadata_sha, physical_manifest_sha = sys.argv[8:11]
physical_source, feature_contract, plan_sha = sys.argv[11:14]
job_id, partition, node, gpu_receipt = sys.argv[14:18]
smoke_manifest, smoke_manifest_sha, smoke_gate_sha = sys.argv[18:21]
arms = ["base_42k", "zero_physical_49k", "physical_context_49k"]
seeds = [42] if mode == "smoke" else [42, 43, 44]

def require(condition, message):
    if not condition:
        raise SystemExit(f"post-run audit failed: {message}")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest_path, selection_path = root / "manifest.json", root / "selection.json"
gate_path = root / "slurm_gate_receipt.json"
require(manifest_path.is_file() and selection_path.is_file(), "core artifacts")
require(not gate_path.exists() and not (root / "failure.json").exists(), "failure/pre-existing receipt")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
require(manifest.get("experiment") == "fuxi_allseason_physical_context_v1", "experiment")
require(manifest.get("status") == "complete" and manifest.get("mode") == mode, "status/mode")
require(manifest.get("seeds") == seeds, "seeds")
expected_selected = {"train": 32, "validation": 16} if mode == "smoke" else {"train": 1652, "validation": 196}
require(manifest.get("split_counts_selected") == expected_selected, "selected splits")
arm_rows = manifest.get("arms", [])
require([row.get("name") for row in arm_rows] == arms, "arms")
require([row.get("context_channels") for row in arm_rows] == [7, 17, 17], "widths")
require([row.get("parameter_count") for row in arm_rows] == [42434, 48914, 48914], "parameters")
training = manifest.get("training", {})
require(training.get("max_epochs") == (2 if mode == "smoke" else 100), "epochs")
require(training.get("patience") == (1 if mode == "smoke" else 15), "patience")
require(training.get("automatic_mixed_precision") is True, "AMP")
runs = training.get("runs", [])
require({(row.get("arm"), row.get("seed")) for row in runs} == {(arm, seed) for arm in arms for seed in seeds}, "run grid")
for seed in seeds:
    rows = {row["arm"]: row for row in runs if row["seed"] == seed}
    require(rows[arms[1]].get("initial_state_sha256") == rows[arms[2]].get("initial_state_sha256"), f"wide init {seed}")
    require(len({row.get("training_rng_state_sha256") for row in rows.values()}) == 1, f"matched RNG {seed}")
require(selection == manifest.get("selection"), "selection copy")
require(selection.get("scientific_selection") is (mode == "full"), "selection scope")
require(selection.get("test_metrics_consulted") is False, "test firewall")
require(selection.get("zero_physical_selectable") is False, "zero-control firewall")
require(selection.get("selected_arm") in {arms[0], arms[2]}, "selected arm")
contract = manifest.get("contract", {})
require(contract.get("development_years_not_scored") == [2020, 2021], "development firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 firewall")
if mode == "smoke":
    require(manifest.get("post_lock_benchmark", {}).get("status") == "skipped_non_scientific_smoke", "smoke post-lock")
    require(not (root / "post_lock_superiority.json").exists(), "smoke superiority")
else:
    require(manifest.get("post_lock_benchmark", {}).get("status") == "complete", "full post-lock")
    require((root / "post_lock_superiority.json").is_file(), "full superiority")
member, physical = manifest.get("member_cache", {}), manifest.get("physical_cache", {})
require(Path(member.get("data_file", "")).resolve() == member_cache, "member path")
require(member.get("data_sha256") == member_sha and member.get("source_fingerprint") == member_source, "member identity")
require(physical.get("cache_sha256") == physical_sha, "physical SHA")
require(physical.get("source_fingerprint") == physical_source, "physical source")
require(physical.get("feature_contract_sha256") == feature_contract, "feature contract")
artifacts = manifest.get("artifact_sha256", {})
actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name not in {"manifest.json", "slurm_gate_receipt.json"}}
require(isinstance(artifacts, dict) and set(artifacts) == actual, "artifact inventory")
for relative, digest in artifacts.items():
    require(sha256_file(root / relative) == digest, f"artifact {relative}")
sources = manifest.get("source_snapshot_sha256", {})
require(sources, "source snapshot")
for snapshot, digest in sources.items():
    require(snapshot.startswith("code/") and artifacts.get(snapshot) == digest, f"source {snapshot}")
    relative = snapshot.removeprefix("code/")
    require(sha256_file(root / snapshot) == digest, f"snapshot {relative}")
    require(sha256_file(work_root / relative) == digest, f"live source {relative}")
require(sources.get("code/plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md") == plan_sha, "plan")
receipt = {
    "schema_version": "fuxi_allseason_physical_context_slurm_gate_v1",
    "status": "post_run_audit_passed",
    "mode": mode,
    "job_id": job_id,
    "partition": partition,
    "node": node,
    "gpu_receipt": gpu_receipt,
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "artifact_sha256": artifacts,
    "source_snapshot_sha256": sources,
    "member_cache": {"path": str(member_cache), "data_sha256": member_sha, "source_fingerprint": member_source},
    "physical_cache": {
        "path": str(physical_cache), "data_sha256": physical_sha,
        "metadata_sha256": physical_metadata_sha, "manifest_sha256": physical_manifest_sha,
        "source_fingerprint": physical_source, "feature_contract_sha256": feature_contract,
    },
    "firewalls": {
        "development_targets_scored": False, "sealed_2025_target_opened": False,
        "post_lock_used_for_selection": False, "post_lock_skipped_for_smoke": mode == "smoke",
    },
    "smoke_parent": None if mode == "smoke" else {
        "manifest_path": smoke_manifest, "manifest_sha256": smoke_manifest_sha,
        "gate_receipt_sha256": smoke_gate_sha,
    },
}
handle, temporary = tempfile.mkstemp(prefix=".slurm_gate_receipt.", suffix=".tmp", dir=root)
try:
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o640)
    os.replace(temporary, gate_path)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
print(gate_path)
PY

echo "complete ${MODE} physical-context run: ${OUTPUT}"
