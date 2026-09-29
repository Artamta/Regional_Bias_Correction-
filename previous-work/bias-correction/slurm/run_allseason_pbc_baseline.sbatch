#!/usr/bin/env bash
#SBATCH --job-name=fuxi-pbc
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch slurm/run_allseason_pbc_baseline.sbatch \
    smoke /absolute/path/full-cache.npy /absolute/path/fresh-smoke-output

  sbatch slurm/run_allseason_pbc_baseline.sbatch \
    full /absolute/path/full-cache.npy /absolute/path/fresh-full-output \
    /absolute/path/completed-smoke/manifest.json

Persistence++ needs exact init-7/init-14 observations. Therefore both modes
require the canonical full cache; smoke bounds computation internally to
32 train, 16 validation, and 16 retrospective-development cases.
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
    echo "cache and output must be absolute paths" >&2
    exit 2
fi
EXPECTED_LAG_SHA256=
EXPECTED_OBSERVATION_SHA256=
if [[ ${MODE} == full ]]; then
    if [[ $4 != /* ]]; then
        echo "completed smoke manifest must be an absolute path" >&2
        exit 2
    fi
fi
if [[ ! -f $2 ]]; then
    echo "member cache does not exist: $2" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE=$(readlink -f "$2")
OUTPUT=$(readlink -m "$3")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
JOB_TOKEN=${SLURM_JOB_ID:-manual}
JOB_TMP=/tmp/fuxi-pbc-${JOB_TOKEN}

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
for required_source in \
    src/fuxi_allseason_pbc_baseline.py \
    src/fuxi_pbc_core.py \
    src/fuxi_allseason_ensemble_calibration.py \
    src/fuxi_allseason_member_cache.py \
    plan/CAPACITY_PBC_STUDY_20260822.md; do
    if [[ ! -f ${WORK_ROOT}/${required_source} ]]; then
        echo "required frozen source is unavailable: ${required_source}" >&2
        exit 2
    fi
done
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi

cleanup() {
    rm -rf -- "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p \
    "${WORK_ROOT}/logs" \
    "${OUTPUT_PARENT}" \
    "${JOB_TMP}/mpl" \
    "${JOB_TMP}/xdg"

export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${WORK_ROOT}"

echo "experiment=fuxi_allseason_pbc_baseline_v3_aligned mode=${MODE}"
echo "job=${JOB_TOKEN} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import sys

import matplotlib
import numpy
import pandas
import torch
import xarray

import fuxi_allseason_pbc_baseline
import fuxi_pbc_core

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
print("CUDA preflight only:", torch.version.cuda)
print("GPU preflight only:", torch.cuda.get_device_name(0))
print("PBC runtime: analytic NumPy/CPU; the GPU is not a scientific dependency")
PY

"${PYTHON_BIN}" src/fuxi_allseason_member_cache.py verify --output "${CACHE}"
"${PYTHON_BIN}" - "${CACHE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
metadata_path = cache.with_suffix("").with_suffix(".metadata.json")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"cache contract failed: {message}")

require(metadata_path.is_file(), f"missing metadata {metadata_path}")
with metadata_path.open("r", encoding="utf-8") as stream:
    metadata = json.load(stream)

require(metadata.get("scope") == "full_archive", "PBC smoke and full require full scope")
require(metadata.get("full_archive") is True, "full-archive flag")
require(metadata.get("shape") == [2080, 51, 6, 27, 27], "canonical shape")
require(metadata.get("initialization_count") == 2080, "initialization count")
require(
    metadata.get("dims") == ["init", "member", "lead_week", "lat", "lon"],
    "dimension order",
)
require(metadata.get("member_labels") == list(range(51)), "member labels")
require(metadata.get("lead_week_labels") == list(range(1, 7)), "lead labels")
require(metadata.get("output_units") == "mm day-1", "output units")
require(
    metadata.get("source_fingerprint")
    == "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae",
    "native-source fingerprint",
)
initializations = metadata.get("initializations", [])
require(len(initializations) == 2080, "initialization inventory")
require(not any(str(value).startswith("2025-") for value in initializations), "2025 seal")

digest = hashlib.sha256()
with cache.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
data_sha256 = digest.hexdigest()
require(
    data_sha256
    == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
    "canonical full-cache SHA-256",
)
require(metadata.get("data_sha256") == data_sha256, "metadata/data SHA-256")
print(f"accepted canonical full PBC cache: {cache}")
PY

if [[ ${MODE} == full ]]; then
    if [[ ! -f $4 ]]; then
        echo "completed smoke manifest does not exist: $4" >&2
        exit 2
    fi
    SMOKE_MANIFEST=$(readlink -f "$4")
    SMOKE_GATE_RECEIPT=$(dirname "${SMOKE_MANIFEST}")/slurm_gate_receipt.json
    "${PYTHON_BIN}" - "${SMOKE_MANIFEST}" "${WORK_ROOT}" "${SMOKE_GATE_RECEIPT}" <<'PY'
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

path = Path(sys.argv[1])
work_root = Path(sys.argv[2])
receipt_path = Path(sys.argv[3])
methods = [
    "raw_fuxi_categorical",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"smoke-to-full gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def close_sequence(actual, expected, message: str) -> None:
    require(len(actual) == len(expected), f"{message} length")
    require(
        all(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(actual, expected, strict=True)),
        message,
    )

with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
require(receipt_path.is_file(), f"missing successful Slurm gate receipt {receipt_path}")
with receipt_path.open("r", encoding="utf-8") as stream:
    receipt = json.load(stream)
require(receipt.get("experiment") == "fuxi_allseason_pbc_baseline_v3_aligned", "gate receipt experiment")
require(receipt.get("gate_status") == "passed", "gate receipt status")
require(receipt.get("mode") == "smoke", "gate receipt mode")
require(receipt.get("manifest_sha256") == sha256_file(path), "gate receipt manifest binding")
require(receipt.get("post_run_audit_version") == "pbc_postrun_v3_aligned", "gate receipt audit version")
require(manifest.get("experiment") == "fuxi_allseason_pbc_baseline_v3_aligned", "experiment")
require(manifest.get("scoring_contract_version") == "normalized_informative_positive_cut_v2", "scoring contract")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == "smoke" and manifest.get("smoke") is True, "smoke mode")
require("non-scientific" in manifest.get("scientific_status", "").lower(), "smoke scientific label")
require(manifest.get("methods") == methods, "method order")
require(
    manifest.get("split_counts_archive")
    == {"train": 1652, "validation": 196, "test": 208, "embargo": 24},
    "archive split counts",
)
require(
    manifest.get("split_counts_selected")
    == {"train": 32, "validation": 16, "test": 16},
    "smoke split counts",
)
contract = manifest.get("contract", {})
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(contract.get("sealed_unopened_years") == [2025], "sealed years")
require(contract.get("target_day_offsets") == list(range(1, 43)), "aligned target offsets")
require(
    contract.get("alignment_contract_version")
    == "imd_end_label_plus_1_through_plus_42_v1",
    "alignment contract",
)
require(contract.get("train_years") == list(range(2002, 2018)), "training years")
require(contract.get("validation_years") == [2018, 2019], "validation years")
require(contract.get("test_years") == [2020, 2021], "development years")
require("strict" in contract.get("cdf_definition", "").lower(), "strict CDF")
require("init-7" in contract.get("persistence_lags", ""), "one-week lag")
require("init-14" in contract.get("persistence_lags", ""), "two-week lag")
require("not independent" in contract.get("test_reuse_warning", "").lower(), "reuse warning")
require("static" in contract.get("paper_protocol_divergence", "").lower(), "paper divergence")

temporal = manifest.get("temporal_evidence", {})
require("complete daily imd" in temporal.get("persistence_lag_source", "").lower(), "daily lag source")
require(temporal.get("persistence_usable_training_cases") == 1648, "daily-lag training availability")
require(temporal.get("persistence_usable_validation_cases") == 196, "daily-lag validation availability")
require(temporal.get("persistence_usable_development_cases") == 208, "daily-lag development availability")

quantiles = manifest.get("quantile_definitions", {})
close_sequence(quantiles.get("quintile_levels", []), [0.2, 0.4, 0.6, 0.8], "quintiles")
close_sequence(
    quantiles.get("semidecile_levels", []),
    [index / 20 for index in range(1, 20)],
    "semideciles",
)
require(quantiles.get("calendar_window_days") == 31, "calendar window")
require(quantiles.get("minimum_samples_with_nearest_calendar_fallback") == 8, "calendar samples")
require("training empirical" in quantiles.get("primary_tie_aware_reference", "").lower(), "tie-aware headline")
require("projected" in quantiles.get("nominal_reference_sensitivity", "").lower(), "projected nominal sensitivity")
require("strictly positive" in quantiles.get("effective_category_rule", "").lower(), "effective cuts")
require("not scored" in quantiles.get("lower_5pct_status", "").lower(), "lower-tail degeneracy")
lower_zero_fraction = float(
    quantiles.get("lower_5pct_zero_threshold_fraction", math.nan)
)
require(
    math.isfinite(lower_zero_fraction) and 0.0 < lower_zero_fraction < 1.0,
    "partially degenerate lower-5% zero-threshold fraction",
)

fitting = manifest.get("fitting", {})
require(fitting.get("debias_candidate_half_spans_days") == [14, 28, 35], "Debias spans")
for key in ("quintile_selected_half_span_by_lead", "semidecile_selected_half_span_by_lead"):
    selected = fitting.get(key, [])
    require(len(selected) == 6 and set(selected) <= {14, 28, 35}, key)
require(math.isclose(float(fitting.get("persistence_ridge", math.nan)), 1.0e-3), "ridge")
require(math.isclose(float(fitting.get("blend_debias_weight", math.nan)), 0.5), "blend")
require(
    fitting.get("persistence_feature_names")
    == [
        "intercept",
        "training_empirical_climatology_cdf",
        "lag_1week_indicator",
        "lag_2week_indicator",
        "raw_fuxi_cdf",
    ],
    "Persistence++ features",
)
for family in ("quintile", "semidecile"):
    validity = fitting.get("cdf_validity", {}).get(family, {})
    require(set(validity) == set(methods), f"{family} CDF methods")
    for method, evidence in validity.items():
        for key in (
            "valid_probability_cdf",
            "valid_effective_threshold_cdf",
            "all_supported_values_finite",
            "all_supported_values_bounded_0_1",
            "all_supported_rows_nondecreasing",
            "all_identical_threshold_cdfs_exactly_equal",
            "all_zero_threshold_cdfs_exactly_zero",
            "all_unsupported_values_nan",
        ):
            require(evidence.get(key) is True, f"{family}/{method}/{key}")
        require(evidence.get("supported_cdf_rows") == 16 * 6 * 171, f"{family}/{method} rows")
    equality = fitting.get("threshold_equality_contract", {}).get(family, {})
    require(
        set(equality)
        == {
            "raw_fuxi_categorical",
            "observed_cdf",
            "training_empirical_climatology",
            "projected_nominal_climatology",
            "debias_plus_plus",
            "persistence_plus_plus",
            "pbc_combined",
        },
        f"{family} equality sources",
    )
    for source, evidence in equality.items():
        require(evidence.get("post_equality_violation_count") == 0, f"{family}/{source} tied CDF equality")
        require(evidence.get("post_zero_anchor_violation_count") == 0, f"{family}/{source} q0 anchor")

evaluation = manifest.get("evaluation", {})
require("informative-positive-cut" in evaluation.get("primary_metric", "").lower(), "primary metric")
require("projected" in evaluation.get("projected_nominal_representation_sensitivity", "").lower(), "projected sensitivity")
require("q80>0" in evaluation.get("paper_formula_sensitivity", "").lower(), "paper-formula sensitivity")
require("distinct strictly positive" in evaluation.get("rps_definition", "").lower(), "RPS definition")
require(evaluation.get("support_cells") == 171, "support cells")
require(evaluation.get("bootstrap_samples") == 100, "smoke bootstrap samples")
require(evaluation.get("bootstrap_block_length_initializations") == 13, "bootstrap block")
require("two-stage" in evaluation.get("bootstrap_scheme", "").lower(), "bootstrap scheme")
for family in ("quintile", "semidecile"):
    support_evidence = evaluation.get("effective_cut_support", {}).get(family, {})
    require(support_evidence.get("all_case_leads_have_positive_scoring_weight") is True, f"{family} effective support")
require(
    evaluation.get("paper_nominal_cut_support", {}).get(
        "all_case_leads_have_positive_q80_scoring_weight"
    )
    is True,
    "paper q80 support",
)
require(
    evaluation.get("informative_upper_tail_support", {}).get(
        "all_case_leads_have_positive_q95_scoring_weight"
    )
    is True,
    "upper q95 support",
)

lag = manifest.get("persistence_lag_provenance", {})
require(lag.get("schema_version") == "named_c_order_array_bundle_sha256_v1", "lag provenance schema")
require(lag.get("bundle_name") == "derived_persistence_issue_time_lags", "lag provenance bundle")
require(len(lag.get("sha256", "")) == 64, "lag provenance SHA-256")
require(lag.get("total_case_count") == 2080, "lag provenance case count")
require(lag.get("available_supported_values_finite_nonnegative") is True, "lag value validity")
observation = manifest.get("observation_bundle_provenance", {})
require(observation.get("schema_version") == "named_c_order_array_bundle_sha256_v1", "observation provenance schema")
require(observation.get("bundle_name") == "imd_weekly_truth_fraction_and_scoring_weights", "observation provenance bundle")
require(len(observation.get("sha256", "")) == 64, "observation provenance SHA-256")
require(receipt.get("scoring_contract_version") == manifest.get("scoring_contract_version"), "receipt scoring contract")
require(receipt.get("persistence_lag_sha256") == lag.get("sha256"), "receipt lag binding")
require(receipt.get("observation_bundle_sha256") == observation.get("sha256"), "receipt observation binding")

retained = manifest.get("retained_initializations", {})
require({key: len(retained.get(key, [])) for key in ("train", "validation", "test")}
        == {"train": 32, "validation": 16, "test": 16}, "retained dates")
for values in retained.values():
    require(not any(date.fromisoformat(value).year == 2025 for value in values), "retained 2025 date")
require(not any("2025" in str(store) for store in manifest.get("observation_stores", [])), "2025 observation store")

required = [
    "README.md",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/threshold_equality_case_lead_grid.npz",
    "models/pbc_fit.npz",
    "metrics/quintile_case_scores.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/seasonal_weekwise_metrics.csv",
    "metrics/component_ablation_metrics.csv",
    "metrics/paired_block_bootstrap.csv",
    "metrics/semidecile_extreme_metrics.csv",
    "metrics/probability_bias.csv",
    "metrics/threshold_diagnostics.csv",
    "metrics/threshold_equality_summary.csv",
    "metrics/effective_cut_support.csv",
    "metrics/paper_nominal_cut_support.csv",
    "metrics/informative_upper_tail_support.csv",
]
artifact_hashes = manifest.get("artifact_sha256", {})
for relative in required:
    candidate = path.parent / relative
    require(candidate.is_file(), f"missing smoke artifact {relative}")
    require(artifact_hashes.get(relative) == sha256_file(candidate), f"artifact hash {relative}")
with (path.parent / "evaluation/persistence_lag_provenance.json").open(
    "r", encoding="utf-8"
) as stream:
    require(json.load(stream) == lag, "lag provenance artifact/manifest identity")
with (path.parent / "evaluation/observation_bundle_provenance.json").open(
    "r", encoding="utf-8"
) as stream:
    require(json.load(stream) == observation, "observation provenance artifact/manifest identity")

current_sources = {
    "code/src/fuxi_allseason_pbc_baseline.py": work_root / "src/fuxi_allseason_pbc_baseline.py",
    "code/src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_allseason_pbc_baseline.sbatch": work_root / "slurm/run_allseason_pbc_baseline.sbatch",
    "code/plan/CAPACITY_PBC_STUDY_20260822.md": work_root / "plan/CAPACITY_PBC_STUDY_20260822.md",
}
source_hashes = manifest.get("source_snapshot_sha256", {})
require(set(source_hashes) == set(current_sources), "source snapshot inventory")
for relative, current in current_sources.items():
    frozen_hash = source_hashes.get(relative)
    require(isinstance(frozen_hash, str) and bool(frozen_hash), f"source hash {relative}")
    require(artifact_hashes.get(relative) == frozen_hash, f"source/artifact hash {relative}")
    require(sha256_file(path.parent / relative) == frozen_hash, f"frozen source {relative}")
    require(sha256_file(current) == frozen_hash, f"current source differs from smoke: {current}")
require(
    manifest.get("cache", {}).get("data_sha256")
    == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
    "smoke cache identity",
)
require(
    manifest.get("cache", {}).get("source_fingerprint")
    == "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae",
    "smoke source fingerprint",
)
print(f"accepted completed PBC smoke manifest: {path}")
PY
    EXPECTED_LAG_SHA256=$("${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["persistence_lag_provenance"]["sha256"])' \
        "${SMOKE_MANIFEST}")
    EXPECTED_OBSERVATION_SHA256=$("${PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["observation_bundle_provenance"]["sha256"])' \
        "${SMOKE_MANIFEST}")
fi

COMMON_ARGUMENTS=(
    --cache "${CACHE}"
    --output "${OUTPUT}"
    --calendar-window-days 31
    --minimum-calendar-samples 8
    --debias-spans 14,28,35
    --ridge 0.001
    --cdf-chunk-size 16
)

if [[ ${MODE} == smoke ]]; then
    srun "${PYTHON_BIN}" src/fuxi_allseason_pbc_baseline.py \
        "${COMMON_ARGUMENTS[@]}" \
        --bootstrap-samples 100 \
        --smoke
else
    srun "${PYTHON_BIN}" src/fuxi_allseason_pbc_baseline.py \
        "${COMMON_ARGUMENTS[@]}" \
        --expected-persistence-lag-sha256 "${EXPECTED_LAG_SHA256}" \
        --expected-observation-bundle-sha256 "${EXPECTED_OBSERVATION_SHA256}" \
        --bootstrap-samples 2000
fi

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" <<'PY'
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
mode = sys.argv[2]
methods = [
    "raw_fuxi_categorical",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
]
comparisons = {
    ("pbc_combined", "raw_fuxi_categorical"),
    ("debias_plus_plus", "raw_fuxi_categorical"),
    ("persistence_plus_plus", "raw_fuxi_categorical"),
    ("pbc_combined", "debias_plus_plus"),
    ("pbc_combined", "persistence_plus_plus"),
}
test_count = 16 if mode == "smoke" else 208
bootstrap_samples = 100 if mode == "smoke" else 2000
selected_counts = (
    {"train": 32, "validation": 16, "test": 16}
    if mode == "smoke"
    else {"train": 1652, "validation": 196, "test": 208}
)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"post-run gate failed: {message}")

def sha256_file(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def close_sequence(actual, expected, message: str) -> None:
    require(len(actual) == len(expected), f"{message} length")
    require(
        all(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(actual, expected, strict=True)),
        message,
    )

def finite_columns(frame: pd.DataFrame, columns, label: str) -> None:
    for column in columns:
        require(column in frame.columns, f"{label} missing column {column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"{label} non-finite {column}")

required = [
    "manifest.json",
    "README.md",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/threshold_equality_case_lead_grid.npz",
    "models/pbc_fit.npz",
    "metrics/quintile_case_scores.csv",
    "metrics/weekwise_metrics.csv",
    "metrics/seasonal_weekwise_metrics.csv",
    "metrics/component_ablation_metrics.csv",
    "metrics/paired_block_bootstrap.csv",
    "metrics/semidecile_extreme_metrics.csv",
    "metrics/probability_bias.csv",
    "metrics/threshold_diagnostics.csv",
    "metrics/threshold_equality_summary.csv",
    "metrics/effective_cut_support.csv",
    "metrics/paper_nominal_cut_support.csv",
    "metrics/informative_upper_tail_support.csv",
]
missing = [relative for relative in required if not (root / relative).is_file()]
require(not missing, f"missing required artifacts: {missing}")

with (root / "manifest.json").open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
require(manifest.get("experiment") == "fuxi_allseason_pbc_baseline_v3_aligned", "experiment")
require(manifest.get("scoring_contract_version") == "normalized_informative_positive_cut_v2", "scoring contract")
require(manifest.get("status") == "complete", "completion status")
require(manifest.get("mode") == mode and manifest.get("smoke") is (mode == "smoke"), "mode")
if mode == "smoke":
    require("non-scientific" in manifest.get("scientific_status", "").lower(), "smoke label")
else:
    status = manifest.get("scientific_status", "").lower()
    require("retrospective" in status and "not an untouched" in status, "retrospective label")
require(manifest.get("methods") == methods, "method order")
require(
    manifest.get("split_counts_archive")
    == {"train": 1652, "validation": 196, "test": 208, "embargo": 24},
    "archive split counts",
)
require(manifest.get("split_counts_selected") == selected_counts, "selected split counts")

contract = manifest.get("contract", {})
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
require(contract.get("sealed_unopened_years") == [2025], "sealed years")
require(contract.get("target_day_offsets") == list(range(1, 43)), "aligned target offsets")
require(
    contract.get("alignment_contract_version")
    == "imd_end_label_plus_1_through_plus_42_v1",
    "alignment contract",
)
require(contract.get("train_years") == list(range(2002, 2018)), "training years")
require(contract.get("validation_years") == [2018, 2019], "validation years")
require(contract.get("test_years") == [2020, 2021], "development years")
require("strict" in contract.get("cdf_definition", "").lower(), "strict CDF")
require("init-7" in contract.get("persistence_lags", ""), "one-week lag")
require("init-14" in contract.get("persistence_lags", ""), "two-week lag")
require("not independent" in contract.get("test_reuse_warning", "").lower(), "reuse warning")
require("static" in contract.get("paper_protocol_divergence", "").lower(), "paper divergence")
require(not any("2025" in str(store) for store in manifest.get("observation_stores", [])), "2025 observation store")

temporal = manifest.get("temporal_evidence", {})
require(date.fromisoformat(temporal.get("training_latest_outcome_end", "9999-12-31")) < date(2018, 1, 1), "training purge")
require(date.fromisoformat(temporal.get("validation_latest_outcome_end", "9999-12-31")) < date(2020, 1, 1), "validation purge")
latest_test = date.fromisoformat(temporal.get("test_latest_initialization", "9999-12-31"))
require(date(2020, 1, 1) <= latest_test < date(2022, 1, 1), "development period")
require("complete daily imd" in temporal.get("persistence_lag_source", "").lower(), "daily lag source")
require(temporal.get("persistence_usable_training_cases") == 1648, "daily-lag training availability")
require(temporal.get("persistence_usable_validation_cases") == 196, "daily-lag validation availability")
require(temporal.get("persistence_usable_development_cases") == 208, "daily-lag development availability")

retained = manifest.get("retained_initializations", {})
require({key: len(retained.get(key, [])) for key in ("train", "validation", "test")}
        == selected_counts, "retained date counts")
expected_years = {
    "train": set(range(2002, 2018)),
    "validation": {2018, 2019},
    "test": {2020, 2021},
}
for split, values in retained.items():
    parsed = [date.fromisoformat(value) for value in values]
    require(len(parsed) == len(set(parsed)), f"duplicate retained dates in {split}")
    require(set(value.year for value in parsed) <= expected_years[split], f"retained years {split}")
    require(not any(value.year == 2025 for value in parsed), f"retained 2025 date {split}")

quantiles = manifest.get("quantile_definitions", {})
close_sequence(quantiles.get("quintile_levels", []), [0.2, 0.4, 0.6, 0.8], "quintiles")
close_sequence(
    quantiles.get("semidecile_levels", []),
    [index / 20 for index in range(1, 20)],
    "semideciles",
)
require(quantiles.get("calendar_window_days") == 31, "calendar window")
require(quantiles.get("minimum_samples_with_nearest_calendar_fallback") == 8, "calendar samples")
require("training empirical" in quantiles.get("primary_tie_aware_reference", "").lower(), "tie-aware headline")
require("projected" in quantiles.get("nominal_reference_sensitivity", "").lower(), "projected nominal sensitivity")
require("strictly positive" in quantiles.get("effective_category_rule", "").lower(), "effective cuts")
require("not scored" in quantiles.get("lower_5pct_status", "").lower(), "lower-tail status")
lower_zero_fraction = float(
    quantiles.get("lower_5pct_zero_threshold_fraction", math.nan)
)
require(
    math.isfinite(lower_zero_fraction) and 0.0 < lower_zero_fraction < 1.0,
    "partially degenerate lower-5% zero-threshold fraction",
)

fitting = manifest.get("fitting", {})
require(fitting.get("debias_candidate_half_spans_days") == [14, 28, 35], "Debias candidates")
for key in ("quintile_selected_half_span_by_lead", "semidecile_selected_half_span_by_lead"):
    selected = fitting.get(key, [])
    require(len(selected) == 6 and set(selected) <= {14, 28, 35}, key)
for family in ("quintile", "semidecile"):
    values_by_span = fitting.get(f"{family}_validation_rps_by_span", {})
    require(set(values_by_span) == {"14", "28", "35"}, f"{family} validation spans")
    for span, values in values_by_span.items():
        require(len(values) == 6, f"{family}/{span} validation lead count")
        require(all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values), f"{family}/{span} validation RPS")
require(math.isclose(float(fitting.get("persistence_ridge", math.nan)), 1.0e-3), "ridge")
require(math.isclose(float(fitting.get("blend_debias_weight", math.nan)), 0.5), "blend")
require(
    fitting.get("persistence_feature_names")
    == [
        "intercept",
        "training_empirical_climatology_cdf",
        "lag_1week_indicator",
        "lag_2week_indicator",
        "raw_fuxi_cdf",
    ],
    "Persistence++ features",
)
require(int(fitting.get("persistence_usable_training_cases", 0)) >= 6, "usable Persistence++ training cases")
for family in ("quintile", "semidecile"):
    validity = fitting.get("cdf_validity", {}).get(family, {})
    require(set(validity) == set(methods), f"{family} CDF methods")
    for method, evidence in validity.items():
        for key in (
            "valid_probability_cdf",
            "valid_effective_threshold_cdf",
            "all_supported_values_finite",
            "all_supported_values_bounded_0_1",
            "all_supported_rows_nondecreasing",
            "all_identical_threshold_cdfs_exactly_equal",
            "all_zero_threshold_cdfs_exactly_zero",
            "all_unsupported_values_nan",
        ):
            require(evidence.get(key) is True, f"{family}/{method}/{key}")
        require(evidence.get("supported_cdf_rows") == test_count * 6 * 171, f"{family}/{method} rows")
    projection = fitting.get("projection_diagnostics", {}).get(family, {})
    require(set(projection) == {"debias_plus_plus", "persistence_plus_plus"}, f"{family} projection methods")
    for method, evidence in projection.items():
        for key, value in evidence.items():
            require(math.isfinite(float(value)), f"{family}/{method} finite projection {key}")
        require(0.0 <= float(evidence.get("violating_fraction", -1.0)) <= 1.0, f"{family}/{method} violation fraction")
        require(float(evidence.get("finite_cdf_rows", 0.0)) > 0.0, f"{family}/{method} finite rows")
    equality = fitting.get("threshold_equality_contract", {}).get(family, {})
    require(
        set(equality)
        == {
            "raw_fuxi_categorical",
            "observed_cdf",
            "training_empirical_climatology",
            "projected_nominal_climatology",
            "debias_plus_plus",
            "persistence_plus_plus",
            "pbc_combined",
        },
        f"{family} equality sources",
    )
    for source, evidence in equality.items():
        require(evidence.get("post_equality_violation_count") == 0, f"{family}/{source} tie equality")
        require(evidence.get("post_zero_anchor_violation_count") == 0, f"{family}/{source} q0 anchor")

evaluation = manifest.get("evaluation", {})
require("informative-positive-cut" in evaluation.get("primary_metric", "").lower(), "primary metric")
require("projected" in evaluation.get("projected_nominal_representation_sensitivity", "").lower(), "projected sensitivity")
require("q80>0" in evaluation.get("paper_formula_sensitivity", "").lower(), "paper-formula sensitivity")
require("distinct strictly positive" in evaluation.get("rps_definition", "").lower(), "RPS definition")
require(evaluation.get("support_cells") == 171, "support cells")
require(evaluation.get("bootstrap_samples") == bootstrap_samples, "bootstrap samples")
require(evaluation.get("bootstrap_block_length_initializations") == 13, "bootstrap block")
require("two-stage" in evaluation.get("bootstrap_scheme", "").lower(), "bootstrap scheme")
for family in ("quintile", "semidecile"):
    require(
        evaluation.get("effective_cut_support", {}).get(family, {}).get(
            "all_case_leads_have_positive_scoring_weight"
        )
        is True,
        f"{family} effective support",
    )
require(evaluation.get("paper_nominal_cut_support", {}).get("all_case_leads_have_positive_q80_scoring_weight") is True, "paper q80 support")
require(evaluation.get("informative_upper_tail_support", {}).get("all_case_leads_have_positive_q95_scoring_weight") is True, "upper q95 support")

lag = manifest.get("persistence_lag_provenance", {})
observation = manifest.get("observation_bundle_provenance", {})
for label, receipt, bundle in (
    ("lag", lag, "derived_persistence_issue_time_lags"),
    ("observation", observation, "imd_weekly_truth_fraction_and_scoring_weights"),
):
    require(receipt.get("schema_version") == "named_c_order_array_bundle_sha256_v1", f"{label} provenance schema")
    require(receipt.get("bundle_name") == bundle, f"{label} provenance bundle")
    require(len(receipt.get("sha256", "")) == 64, f"{label} provenance SHA-256")
require(lag.get("available_supported_values_finite_nonnegative") is True, "lag value validity")

artifact_hashes = manifest.get("artifact_sha256", {})
for relative in required[1:]:
    require(artifact_hashes.get(relative) == sha256_file(root / relative), f"artifact hash {relative}")
for relative, receipt in (
    ("evaluation/persistence_lag_provenance.json", lag),
    ("evaluation/observation_bundle_provenance.json", observation),
):
    with (root / relative).open("r", encoding="utf-8") as stream:
        require(json.load(stream) == receipt, f"{relative} manifest identity")
current_sources = {
    "code/src/fuxi_allseason_pbc_baseline.py": Path.cwd() / "src/fuxi_allseason_pbc_baseline.py",
    "code/src/fuxi_pbc_core.py": Path.cwd() / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": Path.cwd() / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_allseason_member_cache.py": Path.cwd() / "src/fuxi_allseason_member_cache.py",
    "code/slurm/run_allseason_pbc_baseline.sbatch": Path.cwd() / "slurm/run_allseason_pbc_baseline.sbatch",
    "code/plan/CAPACITY_PBC_STUDY_20260822.md": Path.cwd() / "plan/CAPACITY_PBC_STUDY_20260822.md",
}
source_hashes = manifest.get("source_snapshot_sha256", {})
require(set(source_hashes) == set(current_sources), "source snapshot inventory")
for relative, current in current_sources.items():
    digest = source_hashes.get(relative)
    require(artifact_hashes.get(relative) == digest, f"source/artifact hash {relative}")
    require((root / relative).is_file(), f"frozen source {relative}")
    require(sha256_file(root / relative) == digest, f"frozen source hash {relative}")
    require(sha256_file(current) == digest, f"current source hash {relative}")
cache = manifest.get("cache", {})
require(
    cache.get("data_sha256")
    == "2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70",
    "canonical cache identity",
)
require(
    cache.get("source_fingerprint")
    == "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae",
    "native-source fingerprint",
)
require(cache.get("scope") == "full_archive", "cache scope")

support_path = root / "evaluation/scoring_support.npz"
with np.load(support_path, allow_pickle=False) as support:
    require(
        set(support.files)
        == {
            "latitude",
            "longitude",
            "observation_fraction",
            "support_mask",
            "scoring_weight",
            "normalized_scoring_weight",
        },
        "scoring-support inventory",
    )
    mask = support["support_mask"]
    weights = support["scoring_weight"]
    normalized = support["normalized_scoring_weight"]
    require(mask.shape == weights.shape == normalized.shape == (27, 27), "support shape")
    require(int(np.count_nonzero(mask)) == 171, "support-cell count")
    require(np.isfinite(weights).all() and np.all(weights >= 0.0), "scoring weights")
    require(np.all(weights[~mask] == 0.0), "unsupported weights")
    require(math.isclose(float(normalized.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-10), "normalized weights")

model_path = root / "models/pbc_fit.npz"
with np.load(model_path, allow_pickle=False) as model:
    for family, levels in (
        ("quintile", [0.2, 0.4, 0.6, 0.8]),
        ("semidecile", [index / 20 for index in range(1, 20)]),
    ):
        cuts = len(levels)
        close_sequence(model[f"{family}_levels"].tolist(), levels, f"{family} model levels")
        require(model[f"{family}_thresholds_mm_day"].shape == (366, cuts, 27, 27), f"{family} threshold shape")
        require(model[f"{family}_training_empirical_strict_cdf"].shape == (366, cuts, 27, 27), f"{family} empirical shape")
        require(model[f"{family}_fit_indices"].size == selected_counts["train"], f"{family} fit scope")
        selected = model[f"{family}_selected_debias_span_by_lead"]
        require(selected.shape == (6,) and set(selected.tolist()) <= {14, 28, 35}, f"{family} model spans")
        require(model[f"{family}_persistence_coefficients"].shape == (6, cuts, 27, 27, 5), f"{family} coefficient shape")
        require(int(model[f"{family}_unique_fit_window_count"]) > 0, f"{family} unique windows")
        require(int(model[f"{family}_duplicate_fit_window_count"]) >= 0, f"{family} duplicate windows")
        for span in (14, 28, 35):
            require(f"{family}_debias_correction_span_{span}" in model.files, f"{family} correction span {span}")

case_scores = pd.read_csv(root / "metrics/quintile_case_scores.csv")
require(len(case_scores) == test_count * 6 * len(methods), "case-score row count")
require(set(case_scores.method) == set(methods), "case-score methods")
require(set(case_scores.lead_week.astype(int)) == set(range(1, 7)), "case-score leads")
require(case_scores.initialization.nunique() == test_count, "case-score initializations")
require(not case_scores.duplicated(["method", "initialization", "lead_week"]).any(), "duplicate case-score rows")
require(set(case_scores.score_contract) == {"normalized_informative_positive_cut_v2"}, "case score contract")
finite_columns(
    case_scores,
    [
        "rps",
        "nominal_climatology_rps",
        "training_empirical_climatology_rps",
        "rpss_vs_nominal_climatology_case",
        "rpss_vs_training_empirical_climatology_case",
        "paper_nominal_cut_q80_positive_rps",
        "paper_fixed_nominal_climatology_rps",
        "paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case",
        "paper_q80_retained_scoring_weight",
        "paper_q80_total_supported_scoring_weight",
        "paper_q80_retained_scoring_weight_fraction",
    ],
    "case scores",
)
require((case_scores.rps >= 0.0).all(), "negative RPS")
require(case_scores.rps.between(0.0, 1.0).all(), "normalized RPS bounds")
require(case_scores.paper_nominal_cut_q80_positive_rps.between(0.0, 4.0).all(), "paper RPS bounds")
require((case_scores.paper_q80_retained_scoring_weight > 0.0).all(), "paper retained weight")
require((case_scores.nominal_climatology_rps > 0.0).all(), "nominal reference")
require((case_scores.training_empirical_climatology_rps > 0.0).all(), "tie-aware reference")
require(
    np.allclose(
        case_scores.rpss_vs_training_empirical_climatology_case,
        1.0 - case_scores.rps / case_scores.training_empirical_climatology_rps,
        rtol=1.0e-10,
        atol=1.0e-10,
    ),
    "case tie-aware RPSS identity",
)
require(
    np.allclose(
        case_scores.rpss_vs_nominal_climatology_case,
        1.0 - case_scores.rps / case_scores.nominal_climatology_rps,
        rtol=1.0e-10,
        atol=1.0e-10,
    ),
    "case nominal RPSS identity",
)
require(
    np.allclose(
        case_scores.paper_nominal_cut_rpss_vs_fixed_nominal_climatology_case,
        1.0
        - case_scores.paper_nominal_cut_q80_positive_rps
        / case_scores.paper_fixed_nominal_climatology_rps,
        rtol=1.0e-10,
        atol=1.0e-10,
    ),
    "case paper-formula RPSS identity",
)
for column in ("initialization", "verification_start", "verification_midpoint", "verification_end"):
    require(not case_scores[column].astype(str).str.startswith("2025-").any(), f"2025 in {column}")
reference_spread = case_scores.groupby(["initialization", "lead_week"])[
    ["nominal_climatology_rps", "training_empirical_climatology_rps"]
].agg(lambda values: float(values.max() - values.min()))
require(np.allclose(reference_spread.to_numpy(), 0.0, atol=1.0e-12), "method-dependent references")

for filename in ("weekwise_metrics.csv", "seasonal_weekwise_metrics.csv", "component_ablation_metrics.csv"):
    frame = pd.read_csv(root / "metrics" / filename)
    require(set(frame.method) == set(methods), f"{filename} methods")
    require(set(frame.lead_week.astype(int)) == set(range(1, 7)), f"{filename} leads")
    finite_columns(
        frame,
        [
            "rps",
            "nominal_climatology_rps",
            "training_empirical_climatology_rps",
            "rpss_vs_nominal_climatology",
            "rpss_vs_training_empirical_climatology",
            "paper_nominal_cut_q80_positive_rps",
            "paper_fixed_nominal_climatology_rps",
            "paper_nominal_cut_rpss_vs_fixed_nominal_climatology",
            "paper_q80_retained_scoring_weight",
            "n_initializations",
        ],
        filename,
    )
    require(
        np.allclose(
            frame.rpss_vs_training_empirical_climatology,
            1.0 - frame.rps / frame.training_empirical_climatology_rps,
            rtol=1.0e-10,
            atol=1.0e-10,
        ),
        f"{filename} tie-aware RPSS identity",
    )
    require(
        np.allclose(
            frame.paper_nominal_cut_rpss_vs_fixed_nominal_climatology,
            1.0
            - frame.paper_nominal_cut_q80_positive_rps
            / frame.paper_fixed_nominal_climatology_rps,
            rtol=1.0e-10,
            atol=1.0e-10,
        ),
        f"{filename} paper-formula RPSS identity",
    )
weekwise = pd.read_csv(root / "metrics/weekwise_metrics.csv")
component = pd.read_csv(root / "metrics/component_ablation_metrics.csv")
require(len(weekwise) == 24 and len(component) == 24, "weekwise/component row count")
expected_roles = {
    "raw_fuxi_categorical": "uncorrected_reference",
    "debias_plus_plus": "projected_debias_component",
    "persistence_plus_plus": "projected_persistence_component",
    "pbc_combined": "equal_weight_projected_components",
}
require(
    all(row.ablation_role == expected_roles[row.method] for row in component.itertuples()),
    "component roles",
)

bootstrap = pd.read_csv(root / "metrics/paired_block_bootstrap.csv")
require(len(bootstrap) == 35, "bootstrap row count")
require(set(bootstrap.lead_scope) == {"W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"}, "bootstrap scopes")
for scope, rows in bootstrap.groupby("lead_scope"):
    require(set(zip(rows.method, rows.baseline)) == comparisons, f"bootstrap comparisons {scope}")
finite_columns(
    bootstrap,
    [
        "rps_reduction_fraction",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_probability_improvement",
        "bootstrap_samples",
        "block_length_initializations",
    ],
    "bootstrap",
)
require((bootstrap.bootstrap_samples == bootstrap_samples).all(), "bootstrap sample contract")
require((bootstrap.block_length_initializations == 13).all(), "bootstrap block contract")
require(set(bootstrap.score_contract) == {"normalized_informative_positive_cut_v2"}, "bootstrap score contract")
require(set(bootstrap.bootstrap_scheme) == {"two_stage_year_resample_then_within_year_circular_blocks"}, "bootstrap scheme")
require((bootstrap.ci_lower_95 <= bootstrap.ci_upper_95).all(), "bootstrap interval order")
require(bootstrap.bootstrap_probability_improvement.between(0.0, 1.0).all(), "bootstrap probabilities")

extremes = pd.read_csv(root / "metrics/semidecile_extreme_metrics.csv")
lower = extremes.loc[extremes.event == "lower_5pct"]
upper = extremes.loc[extremes.event == "upper_5pct_informative_positive_q95"]
require(len(lower) == 1, "lower-tail diagnostic row")
require(
    lower.iloc[0].status.startswith("not_scored_partially_degenerate"),
    "lower-tail status",
)
require(set(upper.method) == set(methods), "upper-tail methods")
require(set(upper.lead_week.astype(int)) == set(range(1, 7)), "upper-tail leads")
finite_columns(
    upper,
    [
        "brier_score",
        "nominal_climatology_brier_score",
        "training_empirical_climatology_brier_score",
        "bss_vs_nominal_climatology",
        "bss_vs_training_empirical_climatology",
        "n_initializations",
    ],
    "upper tail",
)
require((upper.brier_score >= 0.0).all(), "negative Brier score")
require((upper.loc[upper.season == "ALL", "n_initializations"] == test_count).all(), "upper-tail case count")

bias = pd.read_csv(root / "metrics/probability_bias.csv")
require(set(bias.family) == {"quintile", "semidecile"}, "bias families")
require(set(bias.method) == set(methods), "bias methods")
require(set(bias.lead_week.astype(int)) == set(range(1, 7)), "bias leads")
finite_columns(bias, ["nominal_cumulative_probability", "probability_bias", "mean_absolute_case_probability_bias"], "probability bias")
require(bias.probability_bias.between(-1.0, 1.0).all(), "probability-bias range")
require(bias.mean_absolute_case_probability_bias.between(0.0, 1.0).all(), "absolute probability-bias range")

diagnostics = pd.read_csv(root / "metrics/threshold_diagnostics.csv")
require(len(diagnostics) == 23, "threshold diagnostic row count")
require((diagnostics.family == "quintile").sum() == 4, "quintile diagnostics")
require((diagnostics.family == "semidecile").sum() == 19, "semidecile diagnostics")
finite_columns(
    diagnostics,
    [
        "nominal_cumulative_probability",
        "zero_threshold_fraction",
        "duplicate_previous_threshold_fraction",
        "mean_training_empirical_strict_cdf_probability",
        "min_training_empirical_strict_cdf_probability",
        "max_training_empirical_strict_cdf_probability",
        "minimum_threshold_mm_day",
        "median_threshold_mm_day",
        "maximum_threshold_mm_day",
        "minimum_calendar_training_samples",
        "unique_training_verification_windows",
        "deduplicated_forecast_lead_replicates",
    ],
    "threshold diagnostics",
)
require(diagnostics.zero_threshold_fraction.between(0.0, 1.0).all(), "zero-threshold fractions")
require(diagnostics.duplicate_previous_threshold_fraction.between(0.0, 1.0).all(), "duplicate-threshold fractions")
require((diagnostics.minimum_calendar_training_samples >= 8).all(), "calendar sample minimum")
require((diagnostics.unique_training_verification_windows > 0).all(), "unique training windows")
require((diagnostics.deduplicated_forecast_lead_replicates >= 0).all(), "deduplicated repeats")
require((diagnostics.duplicate_previous_threshold_fraction > 0.0).any(), "duplicate thresholds not diagnosed")

equality = pd.read_csv(root / "metrics/threshold_equality_summary.csv")
require(set(equality.aggregation_scope) == {"test_date_lead", "lead_grid"}, "equality scopes")
require((equality.post_equality_violation_count == 0).all(), "post equality violations")
require((equality.post_zero_anchor_violation_count == 0).all(), "post q0 violations")
with np.load(root / "evaluation/threshold_equality_case_lead_grid.npz", allow_pickle=False) as exact:
    require(exact["initializations"].shape == (test_count,), "equality initialization shape")
    require(exact["support_mask"].shape == (27, 27), "equality support shape")
    post_equality = [exact[name] for name in exact.files if name.endswith("post_equality_violation_count")]
    post_zero = [exact[name] for name in exact.files if name.endswith("post_zero_anchor_violation_count")]
    require(bool(post_equality) and bool(post_zero), "equality diagnostic arrays")
    for values in [*post_equality, *post_zero]:
        require(values.shape == (test_count, 6, 27, 27), "equality diagnostic shape")
        require(np.count_nonzero(values) == 0, "nonzero post equality diagnostic")

effective = pd.read_csv(root / "metrics/effective_cut_support.csv")
require(len(effective) == 2 * test_count * 6, "effective-support rows")
require(set(effective.family) == {"quintile", "semidecile"}, "effective-support families")
require(effective.scorable_weight_fraction.between(0.0, 1.0, inclusive="right").all(), "effective-support weights")
paper_support = pd.read_csv(root / "metrics/paper_nominal_cut_support.csv")
require(len(paper_support) == test_count * 6, "paper-support rows")
require((paper_support.retained_scoring_weight > 0.0).all(), "paper-support retained weight")
require(paper_support.retained_scoring_weight_fraction.between(0.0, 1.0, inclusive="right").all(), "paper-support fractions")
upper_support = pd.read_csv(root / "metrics/informative_upper_tail_support.csv")
require(len(upper_support) == test_count * 6, "upper-support rows")
require(upper_support.retained_scoring_weight_fraction.between(0.0, 1.0, inclusive="right").all(), "upper-support fractions")

readme = (root / "README.md").read_text(encoding="utf-8").lower()
require("scientific headline is rpss against the training-empirical" in readme, "README headline")
require("paper-formula sensitivity" in readme, "README paper sensitivity")
require("inspired baseline" in readme and "not an untouched final test" in readme, "README limitations")
if mode == "smoke":
    require("non-scientific smoke" in readme, "README smoke warning")

pooled = case_scores.groupby("method").rps.mean()
references = case_scores.drop_duplicates(["initialization", "lead_week"])
tie_reference = float(references.training_empirical_climatology_rps.mean())
nominal_reference = float(references.nominal_climatology_rps.mean())
combined_rps = float(pooled["pbc_combined"])
headline_rpss = 1.0 - combined_rps / tie_reference
nominal_sensitivity_rpss = 1.0 - combined_rps / nominal_reference
paper_combined = case_scores.loc[case_scores.method == "pbc_combined"]
paper_weight = paper_combined.paper_q80_retained_scoring_weight.to_numpy(dtype=np.float64)
paper_score = float(
    np.sum(paper_combined.paper_nominal_cut_q80_positive_rps * paper_weight)
    / np.sum(paper_weight)
)
paper_reference = float(
    np.sum(paper_combined.paper_fixed_nominal_climatology_rps * paper_weight)
    / np.sum(paper_weight)
)
paper_formula_rpss = 1.0 - paper_score / paper_reference
pooled_rows = bootstrap.loc[bootstrap.lead_scope == "W1-W6"]

def comparison_row(method: str, baseline: str):
    selected = pooled_rows.loc[
        (pooled_rows.method == method) & (pooled_rows.baseline == baseline)
    ]
    require(len(selected) == 1, f"pooled bootstrap {method} vs {baseline}")
    return selected.iloc[0]

raw_row = comparison_row("pbc_combined", "raw_fuxi_categorical")
direct_reduction = 1.0 - combined_rps / float(pooled["raw_fuxi_categorical"])
require(math.isclose(float(raw_row.rps_reduction_fraction), direct_reduction, rel_tol=0.0, abs_tol=1.0e-12), "pooled bootstrap point estimate")
competitive_with_raw = bool(headline_rpss > 0.0 and float(raw_row.ci_lower_95) > 0.0)
better_than_components = bool(
    float(comparison_row("pbc_combined", "debias_plus_plus").ci_lower_95) > 0.0
    and float(comparison_row("pbc_combined", "persistence_plus_plus").ci_lower_95) > 0.0
)
assessment = {
    "mode": mode,
    "claim_allowed": "none_non_scientific_smoke" if mode == "smoke" else (
        "competitive_with_raw" if competitive_with_raw else "transparent_baseline_or_negative_result"
    ),
    "better_than_both_components_claim_allowed": bool(mode == "full" and better_than_components),
    "headline_tie_aware_rpss": headline_rpss,
    "projected_nominal_representation_sensitivity_rpss": nominal_sensitivity_rpss,
    "paper_formula_q80_positive_rpss": paper_formula_rpss,
    "pbc_vs_raw_pooled_rps_reduction_fraction": float(raw_row.rps_reduction_fraction),
    "pbc_vs_raw_paired_ci_lower_95": float(raw_row.ci_lower_95),
    "pbc_vs_raw_paired_ci_upper_95": float(raw_row.ci_upper_95),
    "development_period_is_retrospective_not_independent": True,
}
print("PBC language-gate assessment:")
print(json.dumps(assessment, indent=2, sort_keys=True))
print(root / "manifest.json")
PY

"${PYTHON_BIN}" - "${OUTPUT}" "${MODE}" "${JOB_TOKEN}" \
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
with manifest.open("r", encoding="utf-8") as stream:
    manifest_payload = json.load(stream)
payload = {
    "experiment": "fuxi_allseason_pbc_baseline_v3_aligned",
    "gate_status": "passed",
    "mode": mode,
    "post_run_audit_version": "pbc_postrun_v3_aligned",
    "scoring_contract_version": manifest_payload["scoring_contract_version"],
    "persistence_lag_sha256": manifest_payload["persistence_lag_provenance"]["sha256"],
    "observation_bundle_sha256": manifest_payload["observation_bundle_provenance"]["sha256"],
    "manifest_sha256": sha256_file(manifest),
    "launcher_sha256": sha256_file(Path.cwd() / "slurm/run_allseason_pbc_baseline.sbatch"),
    "driver_sha256": sha256_file(Path.cwd() / "src/fuxi_allseason_pbc_baseline.py"),
    "core_sha256": sha256_file(Path.cwd() / "src/fuxi_pbc_core.py"),
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
