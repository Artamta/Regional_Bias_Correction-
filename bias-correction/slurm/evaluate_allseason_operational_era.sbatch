#!/usr/bin/env bash
#SBATCH --job-name=fuxi-operaudit
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: sbatch $0 smoke|full /absolute/fresh-output [successful-smoke-manifest-for-full]" >&2
    exit 2
fi

MODE=$1
OUTPUT=$2
SMOKE_MANIFEST=${3:-}
if [[ ${MODE} != smoke && ${MODE} != full ]]; then
    echo "mode must be smoke or full: ${MODE}" >&2
    exit 2
fi
if [[ ${OUTPUT} != /* ]]; then
    echo "output must be absolute: ${OUTPUT}" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi
if [[ ${MODE} == smoke && -n ${SMOKE_MANIFEST} ]]; then
    echo "smoke mode does not accept a gate manifest" >&2
    exit 2
fi
if [[ ${MODE} == full && ${SMOKE_MANIFEST} != /* ]]; then
    echo "full mode requires an absolute successful-smoke manifest" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CAPACITY_MANIFEST=${WORK_ROOT}/resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z/manifest.json
DRIVER=${WORK_ROOT}/src/fuxi_allseason_operational_era_audit.py
LAUNCHER=${WORK_ROOT}/slurm/evaluate_allseason_operational_era.sbatch
JOB_TMP=/tmp/fuxi-operaudit-${SLURM_JOB_ID}

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

# Read-only model/path preflight. It builds only literal whitelisted year paths
# and does not open any forecast or observation year store.
"${PYTHON_BIN}" "${DRIVER}" \
    --capacity-manifest "${CAPACITY_MANIFEST}" \
    --preflight-only

if [[ ${MODE} == full ]]; then
    SMOKE_RECEIPT=$(dirname "${SMOKE_MANIFEST}")/slurm_gate_receipt.json
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${SMOKE_RECEIPT}" \
        "${CAPACITY_MANIFEST}" "${DRIVER}" "${LAUNCHER}" "${WORK_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke_path = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
capacity_path = Path(sys.argv[3]).resolve()
driver_path = Path(sys.argv[4]).resolve()
launcher_path = Path(sys.argv[5]).resolve()
work_root = Path(sys.argv[6]).resolve()
semantic_path = smoke_path.parent / "evaluation/postflight_semantic_audit.json"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"full-gate rejection: {message}")

require(smoke_path.name == "manifest.json" and smoke_path.is_file(), "smoke manifest")
require(receipt_path.is_file(), "atomic smoke Slurm receipt")
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
require(smoke.get("experiment") == "fuxi_allseason_operational_era_audit_v1", "experiment")
require(smoke.get("audit_contract_version") == "operational_era_audit_contract_v3", "audit contract")
require(smoke.get("status") == "complete", "smoke status")
require(smoke.get("mode") == "smoke" and smoke.get("smoke") is True, "smoke mode")
require(smoke.get("canonical") is False and smoke.get("scientific_eligible") is False, "smoke eligibility")
require(smoke.get("capacity_receipt", {}).get("sha256") == sha256_file(capacity_path), "capacity receipt")
contract = smoke.get("contract", {})
require(contract.get("workflow_can_train") is False, "training firewall")
require(contract.get("workflow_can_select") is False, "selection firewall")
require(contract.get("final_2025_store_opened") is False, "2025 firewall")
require(smoke.get("cases", {}).get("evaluated_counts") == {"2022": 4, "2023": 4, "2024": 4}, "smoke case counts")
require(receipt.get("experiment") == smoke.get("experiment"), "receipt experiment")
require(receipt.get("gate_status") == "passed" and receipt.get("mode") == "smoke", "receipt status")
require(receipt.get("post_run_audit_version") == "operational_era_postrun_v3", "receipt audit version")
require(receipt.get("manifest_sha256") == sha256_file(smoke_path), "receipt manifest binding")
require(receipt.get("driver_sha256") == sha256_file(driver_path), "driver changed after smoke")
require(receipt.get("launcher_sha256") == sha256_file(launcher_path), "launcher changed after smoke")
require(semantic_path.is_file(), "smoke semantic-audit receipt")
semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
require(semantic.get("status") == "passed" and semantic.get("mode") == "smoke", "semantic status")
require(semantic.get("post_run_audit_version") == "operational_era_postrun_v3", "semantic version")
require(receipt.get("postflight_semantic_audit_sha256") == sha256_file(semantic_path), "semantic receipt binding")
for relative, expected in smoke.get("artifact_sha256", {}).items():
    artifact = smoke_path.parent / relative
    require(artifact.is_file() and sha256_file(artifact) == expected, f"smoke artifact {relative}")
source_hashes = smoke.get("source_snapshot_sha256", {})
expected_source_keys = {
    "code/src/fuxi_allseason_operational_era_audit.py",
    "code/src/fuxi_allseason_capacity_development_evaluation.py",
    "code/src/fuxi_allseason_capacity_ablation.py",
    "code/src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py",
    "code/plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md",
    "code/slurm/evaluate_allseason_operational_era.sbatch",
}
require(set(source_hashes) == expected_source_keys, "complete smoke source snapshot")
require(receipt.get("source_snapshot_sha256") == source_hashes, "receipt source-snapshot binding")
for relative, expected in source_hashes.items():
    parts = Path(relative).parts
    require(parts and parts[0] == "code" and ".." not in parts, f"safe source path {relative}")
    snapshot = smoke_path.parent / relative
    live = work_root.joinpath(*parts[1:])
    require(snapshot.is_file() and sha256_file(snapshot) == expected, f"snapshot source {relative}")
    require(live.is_file() and sha256_file(live) == expected, f"live source changed after smoke {relative}")
print(f"accepted smoke gate: {smoke_path}")
PY
fi

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown} mode=${MODE}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

EXTRA_ARGS=()
if [[ ${MODE} == smoke ]]; then
    EXTRA_ARGS+=(--smoke)
fi

srun "${PYTHON_BIN}" "${DRIVER}" \
    --capacity-manifest "${CAPACITY_MANIFEST}" \
    --output "${OUTPUT}" \
    --device cuda \
    --batch-size 8 \
    --evaluation-batch-size 8 \
    --num-workers 0 \
    "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${CAPACITY_MANIFEST}" <<'PY'
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fuxi_allseason_operational_era_audit import (
    AUDIT_CONTRACT_VERSION,
    EXPECTED_AFFECTED_CASE_LEADS,
    EXPECTED_COVERAGE_DEVIATIONS,
    EXPECTED_FULL_MINIMUM_WEIGHT_RATIO,
    EXPECTED_NORMALIZATION_MEAN,
    EXPECTED_NORMALIZATION_MEAN_SHA256,
    EXPECTED_NORMALIZATION_STD,
    EXPECTED_NORMALIZATION_STD_SHA256,
    EXPECTED_SMOKE_AFFECTED_CASE_LEADS,
    validate_output_semantics,
)

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
capacity_path = Path(sys.argv[3]).resolve()

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
    "metrics/case_metrics.csv",
    "metrics/seed_case_metrics.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/pooled_metrics.csv",
    "metrics/yearwise_pooled_metrics.csv",
    "metrics/year_weekwise_metrics.csv",
    "metrics/verification_season_weekwise_metrics.csv",
    "metrics/seed_weekwise_metrics.csv",
    "metrics/seed_variability_by_week.csv",
    "metrics/weekwise_reliability_summary.csv",
    "metrics/paired_two_stage_bootstrap.csv",
    "metrics/rank_histograms.csv",
    "metrics/seed_rank_histograms.csv",
    "metrics/reliability_bins.csv",
    "metrics/seed_reliability_bins.csv",
    "evaluation/input_content_receipts.json",
    "evaluation/normalization_receipt.json",
    "evaluation/lead_week_diagnostics.json",
    "evaluation/bootstrap_design.json",
    "evaluation/scoring_support.npz",
    "evaluation/evaluated_initializations.csv",
)
require(all((root / relative).is_file() for relative in required), "required artifacts")
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require(manifest.get("experiment") == "fuxi_allseason_operational_era_audit_v1", "experiment")
require(manifest.get("audit_contract_version") == AUDIT_CONTRACT_VERSION, "audit contract")
require(manifest.get("status") == "complete", "status")
require(manifest.get("mode") == mode and manifest.get("smoke") is (mode == "smoke"), "mode")
require(manifest.get("canonical") is (mode == "full"), "canonical flag")
require(manifest.get("scientific_eligible") is (mode == "full"), "scientific eligibility")
require(manifest.get("capacity_receipt", {}).get("sha256") == sha256_file(capacity_path), "capacity hash")
require(manifest.get("methods") == ["raw_fuxi", "base_42k"], "method firewall")
require(manifest.get("seeds") == [42, 43, 44], "seed firewall")
require(len(manifest.get("input_checkpoints", [])) == 3, "checkpoint count")
require(manifest.get("member_count_shift") == {
    "adapter_accepts_variable_member_set_size": True,
    "training_validation_hindcast_members": 51,
    "member_count_domain_shift_explicit": True,
    "operational_audit_members": 50,
}, "member-count shift")
contract = manifest.get("contract", {})
require(contract.get("workflow_can_train") is False, "training firewall")
require(contract.get("workflow_can_finetune") is False, "fine-tuning firewall")
require(contract.get("workflow_can_select") is False, "selection firewall")
require(contract.get("audit_metrics_used_for_selection") is False, "audit selection contamination")
require(contract.get("forecast_year_store_whitelist") == [2022, 2023, 2024], "forecast whitelist")
require(contract.get("verification_year_store_whitelist") == [2022, 2023, 2024], "truth whitelist")
require(contract.get("year_store_discovery_or_globbing_used") is False, "store discovery")
require(contract.get("final_2025_store_opened") is False, "2025 forecast firewall")
require(contract.get("sealed_2025_target_opened") is False, "2025 truth firewall")
require(contract.get("maximum_verification_date") == "2024-12-29", "verification maximum")
require(contract.get("stored_ensemble_mean_gate") == (
    "np.nanmean(member_axis,dtype=float64).astype(float32) must match "
    "the stored float32 product bitwise"
), "stored ensemble-mean contract")
for value in manifest.get("opened_store_paths_exact", []):
    require("2025.zarr" not in Path(value).parts, f"forbidden store {value}")
cases = manifest.get("cases", {})
require(cases.get("stored_counts") == {"2022": 104, "2023": 104, "2024": 100}, "stored counts")
require(cases.get("full_eligible_counts") == {"2022": 104, "2023": 104, "2024": 88}, "eligible counts")
expected_evaluated = {"2022": 4, "2023": 4, "2024": 4} if mode == "smoke" else {"2022": 104, "2023": 104, "2024": 88}
expected_total = 12 if mode == "smoke" else 296
require(cases.get("evaluated_counts") == expected_evaluated, "evaluated counts")
require(cases.get("evaluated_total") == expected_total, "evaluated total")
require(cases.get("excluded_late_2024_count") == 12, "late-2024 exclusions")
lead_diagnostics = manifest.get("lead_week_diagnostics", {})
content_receipts = json.loads((root / "evaluation/input_content_receipts.json").read_text(encoding="utf-8"))
forecast_receipts = content_receipts.get("forecast_inputs", {})
for year, case_count in expected_evaluated.items():
    diagnostics = lead_diagnostics.get(year, {})
    require(diagnostics.get("streamed_in_bounded_chunks") is True, f"{year} daily streaming")
    require(diagnostics.get("chunk_case_limit") == 8, f"{year} daily chunk bound")
    require(diagnostics.get("checked_initialization_count") == case_count, f"{year} daily checked cases")
    require(diagnostics.get("checked_member_count_per_initialization") == 50, f"{year} daily checked members")
    require(diagnostics.get("checked_lead_day_count") == 42, f"{year} daily checked leads")
    require(diagnostics.get("checked_lead_week_count") == 6, f"{year} weekly checked leads")
    require(diagnostics.get("checked_case_member_week_blocks") == case_count * 50 * 6, f"{year} checked case/member/week blocks")
    require(diagnostics.get("checked_weekly_grid_values") == case_count * 50 * 6 * 27 * 27, f"{year} checked weekly grid values")
    require(float(diagnostics.get("maximum_weekly_vs_daily_absolute_difference_mm_day", 1.0)) <= 2.0e-6, f"{year} weekly/daily maximum")
    require(diagnostics.get("stored_ensemble_mean_reduction_contract") == "np.nanmean(member_axis,dtype=float64).astype(float32)", f"{year} member-mean reduction")
    require(diagnostics.get("stored_ensemble_mean_exact_post_float32_cast_identity") is True, f"{year} member-mean exact identity")
    require(diagnostics.get("stored_ensemble_mean_bitwise_mismatch_count") == 0, f"{year} member-mean mismatch count")
    require(diagnostics.get("checked_member_mean_grid_values") == case_count * 6 * 27 * 27, f"{year} checked member-mean grid values")
    pre_round_maximum = float(diagnostics.get("maximum_pre_round_member_mean_absolute_difference_mm_day", -1.0))
    pre_round_ulp_fraction = float(diagnostics.get("maximum_pre_round_member_mean_fraction_of_float32_ulp", -1.0))
    require(math.isfinite(pre_round_maximum) and pre_round_maximum >= 0.0, f"{year} member-mean pre-round diagnostic")
    require(math.isfinite(pre_round_ulp_fraction) and 0.0 <= pre_round_ulp_fraction <= 0.5 + 1.0e-12, f"{year} member-mean ULP diagnostic")
    daily_receipt = forecast_receipts.get(f"operational_{year}.forecast_daily_selected_stream", {})
    require(daily_receipt.get("dtype") == "<f4", f"{year} streamed daily dtype")
    require(daily_receipt.get("shape") == [case_count, 50, 42, 27, 27], f"{year} streamed daily shape")
    require(len(str(daily_receipt.get("sha256", ""))) == 64, f"{year} streamed daily hash")
    receipt_validation = daily_receipt.get("validation", {})
    require(
        receipt_validation
        and all(diagnostics.get(key) == value for key, value in receipt_validation.items()),
        f"{year} content/manifest daily-validation identity",
    )
bootstrap = manifest.get("evaluation", {}).get("bootstrap", {})
require(bootstrap.get("two_stage") is True, "two-stage bootstrap")
require(bootstrap.get("block_length_initializations") == 13, "block length")
require(bootstrap.get("draws") == (200 if mode == "smoke" else 10_000), "bootstrap draws")
evaluation = manifest.get("evaluation", {})
require(evaluation.get("frozen_model_context_support_cells") == 171, "frozen context support")
require(evaluation.get("dynamic_scoring_support_cells_minimum") == 171, "dynamic support minimum")
require(evaluation.get("dynamic_scoring_support_cells_maximum") == 171, "dynamic support maximum")
minimum_weight_ratio = float(evaluation.get("dynamic_to_frozen_weight_ratio_minimum", -1.0))
require(math.isclose(minimum_weight_ratio, EXPECTED_FULL_MINIMUM_WEIGHT_RATIO, rel_tol=0.0, abs_tol=1.0e-12), "exact dynamic weight ratio")
coverage = evaluation.get("dynamic_observation_coverage_amendment", {})
require(coverage.get("frozen_before_forecast_scores") is True, "coverage amendment timing")
require(coverage.get("contract_version") == AUDIT_CONTRACT_VERSION, "coverage contract version")
require(coverage.get("affected_dates") == ["2023-10-12"], "coverage affected dates")
require(coverage.get("daily_deviation_count") == len(EXPECTED_COVERAGE_DEVIATIONS), "daily coverage deviation count")
require(coverage.get("reduced_coverage_cell_count") == len(EXPECTED_COVERAGE_DEVIATIONS), "reduced coverage cells")
require(coverage.get("zero_daily_coverage_cell_count") == 2, "zero daily coverage cells")
records = coverage.get("deviation_records", [])
actual_cells = [
    (row.get("latitude_index"), row.get("longitude_index"), row.get("latitude"), row.get("longitude"))
    for row in records
]
expected_cells = [(row[0], row[1], row[2], row[3]) for row in EXPECTED_COVERAGE_DEVIATIONS]
require(actual_cells == expected_cells, "coverage cell identities")
expected_affected = EXPECTED_SMOKE_AFFECTED_CASE_LEADS if mode == "smoke" else EXPECTED_AFFECTED_CASE_LEADS
actual_affected = tuple(
    (row.get("initialization"), row.get("lead_week"), row.get("day_within_week"))
    for row in coverage.get("affected_case_leads", [])
)
require(actual_affected == expected_affected, "affected case/lead identities")
require(coverage.get("affected_case_lead_count") == len(expected_affected), "affected case/lead count")
require(coverage.get("expected_affected_case_lead_count") == len(expected_affected), "expected affected case/lead count")
require(math.isclose(float(coverage.get("minimum_dynamic_to_frozen_weight_ratio", -1.0)), EXPECTED_FULL_MINIMUM_WEIGHT_RATIO, rel_tol=0.0, abs_tol=1.0e-12), "coverage receipt weight ratio")
require(coverage.get("exact_identity_gate_passed") is True, "daily coverage identity gate")
require(coverage.get("exact_case_lead_gate_passed") is True, "case/lead coverage identity gate")
require(coverage.get("raw_and_calibrated_weights_identical") is True, "coverage pairing")
normalization = manifest.get("normalization", {})
require(normalization.get("climatology_mean_by_lead") == EXPECTED_NORMALIZATION_MEAN.tolist(), "frozen normalization means")
require(normalization.get("climatology_std_by_lead") == EXPECTED_NORMALIZATION_STD.tolist(), "frozen normalization stds")
require(normalization.get("mean_sha256") == EXPECTED_NORMALIZATION_MEAN_SHA256, "frozen normalization mean hash")
require(normalization.get("std_sha256") == EXPECTED_NORMALIZATION_STD_SHA256, "frozen normalization std hash")
require(normalization.get("expected_mean_sha256") == EXPECTED_NORMALIZATION_MEAN_SHA256, "expected normalization mean hash")
require(normalization.get("expected_std_sha256") == EXPECTED_NORMALIZATION_STD_SHA256, "expected normalization std hash")
require(normalization.get("exact_frozen_value_and_hash_gate_passed") is True, "normalization value/hash gate")
normalization_receipt = json.loads((root / "evaluation/normalization_receipt.json").read_text(encoding="utf-8"))
require(normalization_receipt == normalization, "normalization artifact/manifest identity")
seed_handling = manifest.get("seed_handling", {})
require(seed_handling.get("parameters_averaged") is False, "parameter averaging")
require(seed_handling.get("adjustment_fields_averaged") is False, "adjustment averaging")
require(seed_handling.get("predictions_averaged") is False, "prediction averaging")
hashes = manifest.get("artifact_sha256", {})
for relative in required[1:]:
    require(hashes.get(relative) == sha256_file(root / relative), f"artifact hash {relative}")
for relative, expected in hashes.items():
    artifact = root / relative
    require(artifact.is_file() and sha256_file(artifact) == expected, f"artifact inventory {relative}")
source_hashes = manifest.get("source_snapshot_sha256", {})
live_sources = {
    "code/src/fuxi_allseason_operational_era_audit.py": Path.cwd() / "src/fuxi_allseason_operational_era_audit.py",
    "code/src/fuxi_allseason_capacity_development_evaluation.py": Path.cwd() / "src/fuxi_allseason_capacity_development_evaluation.py",
    "code/src/fuxi_allseason_capacity_ablation.py": Path.cwd() / "src/fuxi_allseason_capacity_ablation.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": Path.cwd() / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": Path.cwd() / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py": Path.cwd() / "src/fuxi_allseason_member_cache.py",
    "code/slurm/evaluate_allseason_operational_era.sbatch": Path.cwd() / "slurm/evaluate_allseason_operational_era.sbatch",
    "code/plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md": Path.cwd() / "plan/OPERATIONAL_ERA_ENSEMBLE_AUDIT_20260822.md",
}
require(set(source_hashes) == set(live_sources), "complete source-snapshot inventory")
for relative, live in live_sources.items():
    require(source_hashes.get(relative) == sha256_file(live), f"live source changed during run {relative}")
semantic = validate_output_semantics(root, mode, manifest)
semantic.update({
    "manifest_sha256": sha256_file(manifest_path),
    "source_snapshot_sha256": source_hashes,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
})
semantic_path = root / "evaluation/postflight_semantic_audit.json"
temporary = semantic_path.with_name(semantic_path.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(semantic, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, semantic_path)
print(semantic_path)
PY

# Issue the Slurm receipt only after the independent post-run assertions pass.
"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${SLURM_JOB_ID}" \
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
if not manifest.is_file():
    raise SystemExit("cannot issue gate receipt without manifest.json")
manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
semantic = root / "evaluation/postflight_semantic_audit.json"
if not semantic.is_file():
    raise SystemExit("cannot issue gate receipt without semantic-audit receipt")
payload = {
    "experiment": "fuxi_allseason_operational_era_audit_v1",
    "gate_status": "passed",
    "mode": mode,
    "audit_contract_version": "operational_era_audit_contract_v3",
    "post_run_audit_version": "operational_era_postrun_v3",
    "manifest_sha256": sha256_file(manifest),
    "postflight_semantic_audit_sha256": sha256_file(semantic),
    "source_snapshot_sha256": manifest_payload.get("source_snapshot_sha256", {}),
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

echo "complete: ${OUTPUT}"
