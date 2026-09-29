#!/usr/bin/env bash
#SBATCH --job-name=fuxi-opcat
#SBATCH --partition=gpu_prio
# CPU-only score/reconstruction audit routed through the available S2S partition.
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
DRIVER=${WORK_ROOT}/src/fuxi_allseason_operational_categorical_comparison.py
PROTOCOL=${WORK_ROOT}/plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md
LAUNCHER=${WORK_ROOT}/slurm/evaluate_allseason_operational_categorical_comparison.sbatch
CONTRACT_TEST=${WORK_ROOT}/tests/test_allseason_operational_categorical_comparison.py
PBC_MANIFEST=${WORK_ROOT}/resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json
EXPECTED_PBC_MANIFEST_SHA256=c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6
EXPECTED_PBC_RECEIPT_SHA256=1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2
EXPECTED_PBC_FIT_SHA256=0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36
OPERATIONAL_ROOT=${WORK_ROOT}/resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z
OPERATIONAL_MANIFEST=${OPERATIONAL_ROOT}/manifest.json
OPERATIONAL_RECEIPT=${OPERATIONAL_ROOT}/slurm_gate_receipt.json
EXPECTED_OPERATIONAL_MANIFEST_SHA256=7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad
EXPECTED_OPERATIONAL_RECEIPT_SHA256=65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd

usage() {
    cat >&2 <<EOF
usage:
  sbatch $0 smoke OUTPUT
  sbatch $0 full  OUTPUT SUCCESSFUL_SMOKE_MANIFEST

All supplied paths must be absolute. The accepted operational V3 manifest,
receipt, and their exact SHA-256 identities are internal immutable constants.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 2
fi
MODE=$1
OUTPUT_RAW=$2
SMOKE_MANIFEST_RAW=${3:-}

if [[ ${MODE} != smoke && ${MODE} != full ]]; then
    echo "mode must be smoke or full: ${MODE}" >&2
    exit 2
fi
if [[ ${MODE} == smoke && $# -ne 2 ]]; then
    usage
    exit 2
fi
if [[ ${MODE} == full && $# -ne 3 ]]; then
    usage
    exit 2
fi
for value in "${OUTPUT_RAW}"; do
    if [[ ${value} != /* ]]; then
        echo "all output/input paths must be absolute: ${value}" >&2
        exit 2
    fi
done
if [[ ${MODE} == full && ${SMOKE_MANIFEST_RAW} != /* ]]; then
    echo "full mode requires an absolute successful-smoke manifest" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} || ! -f ${DRIVER} || ! -f ${PROTOCOL} || ! -f ${LAUNCHER} || ! -f ${CONTRACT_TEST} ]]; then
    echo "required Python/source/protocol/launcher/test is unavailable" >&2
    exit 2
fi
if [[ ! -f ${PBC_MANIFEST} ]]; then
    echo "canonical PBC manifest is unavailable: ${PBC_MANIFEST}" >&2
    exit 2
fi
if [[ ! -f ${OPERATIONAL_MANIFEST} || ! -f ${OPERATIONAL_RECEIPT} ]]; then
    echo "operational full manifest/receipt is unavailable" >&2
    exit 2
fi
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "invalid or missing Slurm job id: ${SLURM_JOB_ID:-unset}" >&2
    exit 2
fi

OUTPUT=$(readlink -m "${OUTPUT_RAW}")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
OUTPUT_NAME=$(basename "${OUTPUT}")
if [[ ${OUTPUT} == / || -z ${OUTPUT_NAME} || ${OUTPUT_NAME} == . ]]; then
    echo "refusing unsafe publication output: ${OUTPUT}" >&2
    exit 2
fi
PUBLISH_STAGING=${OUTPUT_PARENT}/.${OUTPUT_NAME}.gate-incomplete-${SLURM_JOB_ID}
SMOKE_MANIFEST=
ACCEPTED_SMOKE_MANIFEST_SHA256=-
if [[ ${MODE} == full ]]; then
    if [[ ! -f ${SMOKE_MANIFEST_RAW} ]]; then
        echo "successful-smoke manifest is unavailable: ${SMOKE_MANIFEST_RAW}" >&2
        exit 2
    fi
    SMOKE_MANIFEST=$(readlink -f "${SMOKE_MANIFEST_RAW}")
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi
if [[ -e ${PUBLISH_STAGING} ]]; then
    echo "fresh launcher gate-staging path required: ${PUBLISH_STAGING}" >&2
    exit 2
fi

JOB_TMP=$(mktemp -d "/tmp/fuxi-opcat-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    # Launcher gate staging is deliberately excluded and retained on failure.
    if [[ -d ${JOB_TMP} && ${JOB_TMP} == /tmp/fuxi-opcat-* ]]; then
        rm -rf -- "${JOB_TMP}"
    fi
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat:${WORK_ROOT}/../neural_adapter/src
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "${WORK_ROOT}"

# Immutable parent preflight. This validates receipts, every declared parent
# artifact, and byte identity of all reused runtime sources without opening a
# forecast or observation store.
PREFLIGHT_ARGS=(
    --pbc-manifest "${PBC_MANIFEST}"
    --preflight-only
)
if [[ ${MODE} == smoke ]]; then
    PREFLIGHT_ARGS+=(--smoke)
fi
"${PYTHON_BIN}" "${DRIVER}" "${PREFLIGHT_ARGS[@]}"

if [[ ${MODE} == full ]]; then
    SMOKE_RECEIPT=$(dirname "${SMOKE_MANIFEST}")/slurm_gate_receipt.json
    "${PYTHON_BIN}" - \
        "${SMOKE_MANIFEST}" "${SMOKE_RECEIPT}" \
        "${EXPECTED_OPERATIONAL_MANIFEST_SHA256}" \
        "${EXPECTED_OPERATIONAL_RECEIPT_SHA256}" \
        "${EXPECTED_PBC_MANIFEST_SHA256}" \
        "${EXPECTED_PBC_RECEIPT_SHA256}" \
        "${EXPECTED_PBC_FIT_SHA256}" \
        "${DRIVER}" "${PROTOCOL}" "${LAUNCHER}" "${CONTRACT_TEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke_path = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
expected_op_manifest = sys.argv[3]
expected_op_receipt = sys.argv[4]
expected_pbc = sys.argv[5]
expected_pbc_receipt = sys.argv[6]
expected_pbc_fit = sys.argv[7]
driver = Path(sys.argv[8]).resolve()
protocol = Path(sys.argv[9]).resolve()
launcher = Path(sys.argv[10]).resolve()
contract_test = Path(sys.argv[11]).resolve()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"full-gate rejection: {message}")

require(smoke_path.is_file() and receipt_path.is_file(), "smoke manifest/receipt")
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
require(smoke.get("experiment") == "fuxi_allseason_operational_categorical_comparison_v1", "smoke experiment")
require(smoke.get("status") == "complete" and smoke.get("mode") == "smoke" and smoke.get("smoke") is True, "smoke status")
require(smoke.get("canonical") is False and smoke.get("scientific_eligible") is False, "smoke evidence flags")
require(smoke.get("frozen_pbc", {}).get("manifest_sha256") == expected_pbc, "smoke PBC parent")
require(smoke.get("frozen_pbc", {}).get("receipt_sha256") == expected_pbc_receipt, "smoke PBC receipt")
require(smoke.get("frozen_pbc", {}).get("fit_sha256") == expected_pbc_fit, "smoke PBC fit")
require(smoke.get("operational_parent", {}).get("manifest_sha256") == expected_op_manifest, "smoke operational manifest")
require(smoke.get("operational_parent", {}).get("receipt_sha256") == expected_op_receipt, "smoke operational receipt")
for key, expected in {
    "experiment": "fuxi_allseason_operational_categorical_comparison_v1",
    "gate_status": "passed",
    "mode": "smoke",
    "audit_contract_version": "operational_categorical_contract_v1",
    "post_run_audit_version": "operational_categorical_postrun_v1",
    "manifest_sha256": sha256_file(smoke_path),
}.items():
    require(receipt.get(key) == expected, f"smoke receipt {key}")
semantic_path = smoke_path.parent / "evaluation/postflight_semantic_audit.json"
require(semantic_path.is_file(), "smoke semantic audit")
require(sha256_file(semantic_path) == receipt.get("postflight_semantic_audit_sha256"), "smoke semantic hash")
semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
require(
    semantic.get("status") == "passed"
    and semantic.get("mode") == "smoke"
    and semantic.get("post_run_audit_version") == "operational_categorical_postrun_v1",
    "smoke semantic status",
)
inventory = smoke.get("artifact_sha256", {})
require(isinstance(inventory, dict) and inventory, "smoke artifact inventory")
actual_files = {
    str(path.relative_to(smoke_path.parent))
    for path in smoke_path.parent.rglob("*")
    if path.is_file()
    and str(path.relative_to(smoke_path.parent))
    not in {
        "manifest.json",
        "failure.json",
        "slurm_gate_receipt.json",
        "evaluation/postflight_semantic_audit.json",
    }
}
require(actual_files == set(inventory), "complete smoke artifact inventory")
for relative, expected in inventory.items():
    path = smoke_path.parent / relative
    require(path.is_file() and sha256_file(path) == expected, f"smoke artifact {relative}")
live = {
    "code/src/fuxi_allseason_operational_categorical_comparison.py": driver,
    "code/src/fuxi_allseason_operational_era_audit.py": Path.cwd() / "src/fuxi_allseason_operational_era_audit.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": Path.cwd() / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_pbc_core.py": Path.cwd() / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_capacity_ablation.py": Path.cwd() / "src/fuxi_allseason_capacity_ablation.py",
    "code/src/fuxi_allseason_capacity_development_evaluation.py": Path.cwd() / "src/fuxi_allseason_capacity_development_evaluation.py",
    "code/src/fuxi_allseason_member_cache.py": Path.cwd() / "src/fuxi_allseason_member_cache.py",
    "code/src/fuxi_ensemble_calibration_core.py": Path.cwd() / "src/fuxi_ensemble_calibration_core.py",
    "code/plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md": protocol,
    "code/slurm/evaluate_allseason_operational_categorical_comparison.sbatch": launcher,
    "code/tests/test_allseason_operational_categorical_comparison.py": contract_test,
}
snapshot = smoke.get("source_snapshot_sha256", {})
require(set(snapshot) == set(live), "complete smoke source snapshot map")
for relative, path in live.items():
    expected = snapshot.get(relative)
    frozen = smoke_path.parent / relative
    require(inventory.get(relative) == expected, f"smoke source is not artifact-gated: {relative}")
    require(frozen.is_file() and sha256_file(frozen) == expected, f"smoke source snapshot bytes: {relative}")
    require(sha256_file(path) == expected, f"source changed after smoke: {relative}")
print("PASS: successful source-identical categorical smoke gates full run")
PY
    ACCEPTED_SMOKE_MANIFEST_SHA256=$(sha256sum "${SMOKE_MANIFEST}" | awk '{print $1}')
    if [[ ! ${ACCEPTED_SMOKE_MANIFEST_SHA256} =~ ^[0-9a-f]{64}$ ]]; then
        echo "could not freeze the accepted smoke manifest identity" >&2
        exit 1
    fi
fi

RUN_ARGS=(
    --pbc-manifest "${PBC_MANIFEST}"
    --output "${PUBLISH_STAGING}"
    --publication-output "${OUTPUT}"
    --cdf-chunk-size 8
)
if [[ ${MODE} == smoke ]]; then
    RUN_ARGS+=(--smoke --bootstrap-draws 200)
else
    RUN_ARGS+=(--bootstrap-draws 10000)
fi
srun "${PYTHON_BIN}" "${DRIVER}" "${RUN_ARGS[@]}"

# Rebuild score-table relationships and check the complete immutable
# artifact/source inventory inside launcher staging before publication.
SMOKE_POSTFLIGHT_MANIFEST=-
if [[ ${MODE} == full ]]; then
    SMOKE_POSTFLIGHT_MANIFEST=${SMOKE_MANIFEST}
fi
"${PYTHON_BIN}" - \
    "${PUBLISH_STAGING}" "${OUTPUT}" "${MODE}" \
    "${SMOKE_POSTFLIGHT_MANIFEST}" "${ACCEPTED_SMOKE_MANIFEST_SHA256}" \
    "${EXPECTED_OPERATIONAL_MANIFEST_SHA256}" \
    "${EXPECTED_OPERATIONAL_RECEIPT_SHA256}" \
    "${EXPECTED_PBC_MANIFEST_SHA256}" \
    "${EXPECTED_PBC_RECEIPT_SHA256}" \
    "${EXPECTED_PBC_FIT_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fuxi_allseason_operational_categorical_comparison import (
    AUDIT_CONTRACT_VERSION,
    EXPERIMENT,
    POSTRUN_AUDIT_VERSION,
    validate_output_semantics,
)

root = Path(sys.argv[1]).resolve()
expected_output = str(Path(sys.argv[2]).resolve())
mode = sys.argv[3]
smoke_manifest_argument = sys.argv[4]
accepted_smoke_manifest_sha256 = sys.argv[5]
expected_op_manifest = sys.argv[6]
expected_op_receipt = sys.argv[7]
expected_pbc_manifest = sys.argv[8]
expected_pbc_receipt = sys.argv[9]
expected_pbc_fit = sys.argv[10]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"post-run rejection: {message}")

required = (
    "manifest.json",
    "README.md",
    "metrics/neural_seed_quintile_case_scores.csv",
    "metrics/neural_seed_semidecile_case_scores.csv",
    "metrics/neural_seed_upper_case_scores.csv",
    "metrics/neural_seed_probability_bias.csv",
    "metrics/quintile_case_scores.csv",
    "metrics/semidecile_case_scores.csv",
    "metrics/upper_q95_case_scores.csv",
    "metrics/probability_bias.csv",
    "metrics/weekwise_rps.csv",
    "metrics/seasonal_weekwise_rps.csv",
    "metrics/pooled_rps.csv",
    "metrics/upper_q95_summary.csv",
    "metrics/paired_two_stage_bootstrap.csv",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_lags.npz",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/late_2021_lag_only_receipt.json",
    "evaluation/bootstrap_design.json",
    "receipts/input_receipts.json",
)
require(all((root / relative).is_file() for relative in required), "required artifacts")
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require(manifest.get("experiment") == EXPERIMENT, "experiment")
require(manifest.get("audit_contract_version") == AUDIT_CONTRACT_VERSION, "audit contract")
require(manifest.get("status") == "complete" and manifest.get("mode") == mode, "status/mode")
require(manifest.get("smoke") is (mode == "smoke"), "smoke flag")
require(manifest.get("canonical") is (mode == "full"), "canonical flag")
require(manifest.get("scientific_eligible") is (mode == "full"), "scientific flag")
require(manifest.get("output_path") == expected_output, "requested publication path")
require(
    manifest.get("publication", {}).get("requested_output_path") == expected_output
    and manifest.get("publication", {}).get("driver_output_path") == str(root)
    and manifest.get("publication", {}).get("launcher_semantic_and_receipt_gate_required_before_publication") is True,
    "launcher staging/publication contract",
)
require(manifest.get("operational_parent", {}).get("manifest_sha256") == expected_op_manifest, "operational manifest parent")
require(manifest.get("operational_parent", {}).get("receipt_sha256") == expected_op_receipt, "operational receipt parent")
require(manifest.get("frozen_pbc", {}).get("manifest_sha256") == expected_pbc_manifest, "PBC manifest parent")
require(manifest.get("frozen_pbc", {}).get("receipt_sha256") == expected_pbc_receipt, "PBC receipt parent")
require(manifest.get("frozen_pbc", {}).get("fit_sha256") == expected_pbc_fit, "PBC fit parent")

inventory = manifest.get("artifact_sha256", {})
require(isinstance(inventory, dict) and inventory, "artifact inventory")
actual_files = {
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file()
    and str(path.relative_to(root))
    not in {"manifest.json", "failure.json", "slurm_gate_receipt.json"}
}
require(actual_files == set(inventory), "complete artifact inventory")
for relative, expected in inventory.items():
    path = root / relative
    require(path.is_file() and sha256_file(path) == expected, f"artifact hash {relative}")

source_hashes = manifest.get("source_snapshot_sha256", {})
live_sources = {
    "code/src/fuxi_allseason_operational_categorical_comparison.py": Path.cwd() / "src/fuxi_allseason_operational_categorical_comparison.py",
    "code/src/fuxi_allseason_operational_era_audit.py": Path.cwd() / "src/fuxi_allseason_operational_era_audit.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": Path.cwd() / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_pbc_core.py": Path.cwd() / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_capacity_ablation.py": Path.cwd() / "src/fuxi_allseason_capacity_ablation.py",
    "code/src/fuxi_allseason_capacity_development_evaluation.py": Path.cwd() / "src/fuxi_allseason_capacity_development_evaluation.py",
    "code/src/fuxi_allseason_member_cache.py": Path.cwd() / "src/fuxi_allseason_member_cache.py",
    "code/src/fuxi_ensemble_calibration_core.py": Path.cwd() / "src/fuxi_ensemble_calibration_core.py",
    "code/plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md": Path.cwd() / "plan/OPERATIONAL_ERA_CATEGORICAL_COMPARISON_20260823.md",
    "code/slurm/evaluate_allseason_operational_categorical_comparison.sbatch": Path.cwd() / "slurm/evaluate_allseason_operational_categorical_comparison.sbatch",
    "code/tests/test_allseason_operational_categorical_comparison.py": Path.cwd() / "tests/test_allseason_operational_categorical_comparison.py",
}
require(set(source_hashes) == set(live_sources), "source snapshot inventory")
for relative, live in live_sources.items():
    expected = source_hashes.get(relative)
    snapshot_path = root / relative
    require(inventory.get(relative) == expected, f"source is not artifact-gated {relative}")
    require(snapshot_path.is_file() and sha256_file(snapshot_path) == expected, f"source snapshot bytes {relative}")
    require(sha256_file(live) == expected, f"live source changed during run {relative}")

smoke_manifest_sha256 = None
if mode == "full":
    smoke_path = Path(smoke_manifest_argument).resolve()
    require(smoke_path.is_file(), "accepted smoke manifest is absent at full postflight")
    require(
        sha256_file(smoke_path) == accepted_smoke_manifest_sha256,
        "accepted smoke manifest changed during the full run",
    )
    smoke_manifest = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(
        smoke_manifest.get("source_snapshot_sha256") == source_hashes,
        "full source snapshot map differs from accepted smoke",
    )
    smoke_manifest_sha256 = sha256_file(smoke_path)
else:
    require(
        smoke_manifest_argument == "-" and accepted_smoke_manifest_sha256 == "-",
        "smoke postflight received a smoke parent",
    )

semantic = validate_output_semantics(root, mode, manifest)
semantic.update(
    {
        "manifest_sha256": sha256_file(manifest_path),
        "source_snapshot_sha256": source_hashes,
        "accepted_smoke_manifest_sha256": smoke_manifest_sha256,
        "smoke_full_exact_source_snapshot_identity": mode == "full",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
)
destination = root / "evaluation/postflight_semantic_audit.json"
temporary = destination.with_name(destination.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(semantic, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, destination)
print(destination)
PY

"${PYTHON_BIN}" - \
    "${PUBLISH_STAGING}" "${MODE}" "${SLURM_JOB_ID}" \
    "${SLURMD_NODENAME:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" \
    "${DRIVER}" "${LAUNCHER}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
job = sys.argv[3]
node = sys.argv[4]
partition = sys.argv[5]
driver = Path(sys.argv[6]).resolve()
launcher = Path(sys.argv[7]).resolve()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = root / "manifest.json"
semantic = root / "evaluation/postflight_semantic_audit.json"
if not manifest.is_file() or not semantic.is_file():
    raise SystemExit("cannot issue categorical gate receipt without manifest and semantic audit")
payload = {
    "experiment": "fuxi_allseason_operational_categorical_comparison_v1",
    "gate_status": "passed",
    "mode": mode,
    "audit_contract_version": "operational_categorical_contract_v1",
    "post_run_audit_version": "operational_categorical_postrun_v1",
    "manifest_sha256": sha256_file(manifest),
    "postflight_semantic_audit_sha256": sha256_file(semantic),
    "source_snapshot_sha256": json.loads(manifest.read_text(encoding="utf-8")).get("source_snapshot_sha256", {}),
    "launcher_sha256": sha256_file(launcher),
    "driver_sha256": sha256_file(driver),
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

if [[ -e ${OUTPUT} ]]; then
    echo "publication target appeared before the final atomic rename: ${OUTPUT}" >&2
    exit 1
fi
if [[ ! -d ${PUBLISH_STAGING} || ! -f ${PUBLISH_STAGING}/slurm_gate_receipt.json ]]; then
    echo "complete gate staging is unavailable; retained path: ${PUBLISH_STAGING}" >&2
    exit 1
fi
mv -T -- "${PUBLISH_STAGING}" "${OUTPUT}"

echo "PASS: ${MODE} operational-era categorical comparison accepted at ${OUTPUT}"
