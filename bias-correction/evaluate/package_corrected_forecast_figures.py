#!/usr/bin/env python3
"""Package the revised Corrected Forecast presentation figures.

Only completed, checksummed presentation outputs are copied.  The packager
does not open model inputs, observations, forecasts, targets, or checkpoints
and it does not recompute a metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SCOPE_WORDS = ("exploratory", "reused", "not independent")
EXPECTED_CONFIGURATION = "physical_full_compact"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return value


def _scope_is_honest(manifest: Mapping[str, Any]) -> bool:
    scope = str(manifest.get("evaluation_scope", "")).lower()
    return all(word in scope for word in EXPECTED_SCOPE_WORDS)


def _verify_manifest_artifacts(
    directory: Path,
    manifest: Mapping[str, Any],
    field: str,
) -> None:
    artifacts = manifest.get(field)
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"manifest lacks {field}: {directory}")
    for name, expected in artifacts.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def validate_sources(
    spatial: Path,
    acc_curve: Path,
    paired_acc: Path,
    jjas: Path,
    anomaly: Path,
) -> dict[str, dict[str, str]]:
    paths = {
        "transposed_spatial": spatial / "transposed_spatial_composites_manifest.json",
        "jjas_acc_curve": acc_curve / "manifest.json",
        "paired_acc": paired_acc / "manifest.json",
        "jjas_month_lead": jjas / "diagnostic_manifest.json",
        "anomaly_maps": anomaly / "anomaly_spatial_skill_manifest.json",
    }
    manifests = {name: read_json(path) for name, path in paths.items()}
    if not all(_scope_is_honest(manifest) for manifest in manifests.values()):
        raise ValueError("all revised sources must say exploratory/reused and not independent")

    spatial_manifest = manifests["transposed_spatial"]
    if spatial_manifest.get("status") != "complete":
        raise ValueError("transposed spatial figures are incomplete")
    layout = spatial_manifest.get("layout", {})
    expected_columns = [
        "IMD Observation",
        "Raw FuXi-S2S",
        "Corrected Forecast",
        "IMD − Raw FuXi",
        "IMD − Corrected Forecast",
    ]
    if layout.get("columns") != expected_columns:
        raise ValueError("transposed spatial columns differ from the requested order")
    if spatial_manifest.get("visual_interpolation_used_for_metrics") is not False:
        raise ValueError("interpolation entered spatial metrics")

    curve_manifest = manifests["jjas_acc_curve"]
    if curve_manifest.get("method_labels", {}).get("corrected") != "Corrected Forecast":
        raise ValueError("ACC curve lacks the Corrected Forecast label")
    if curve_manifest.get("imd_role") != "verification reference; IMD self-curve deliberately omitted":
        raise ValueError("ACC curve mishandles the IMD reference")
    if any(
        curve_manifest.get(key) is not False
        for key in ("source_arrays_opened", "metrics_recomputed", "model_refit")
    ):
        raise ValueError("ACC curve violated the read-only result-table contract")

    paired_manifest = manifests["paired_acc"]
    if paired_manifest.get("genuine_independent_confirmation") is not False:
        raise ValueError("paired ACC scope is not exploratory")
    if paired_manifest.get("source_arrays_opened") is not False:
        raise ValueError("paired ACC opened source arrays")

    jjas_manifest = manifests["jjas_month_lead"]
    if jjas_manifest.get("status") != "complete":
        raise ValueError("JJAS month-lead diagnostic is incomplete")
    if jjas_manifest.get("selected_configuration") != EXPECTED_CONFIGURATION:
        raise ValueError("unexpected corrected configuration")
    if jjas_manifest.get("method_labels", {}).get("corrected") != "Corrected Forecast":
        raise ValueError("JJAS diagnostic lacks the Corrected Forecast label")
    uncertainty = jjas_manifest.get("uncertainty", {})
    if uncertainty.get("p_values_computed") is not False:
        raise ValueError("p-values are outside the revised figure contract")
    if uncertainty.get("significance_claimed") is not False:
        raise ValueError("significance claims are outside the revised figure contract")

    anomaly_manifest = manifests["anomaly_maps"]
    if anomaly_manifest.get("status") != "complete":
        raise ValueError("anomaly figures are incomplete")
    if anomaly_manifest.get("display_name") != "Corrected Forecast":
        raise ValueError("anomaly figures lack the Corrected Forecast label")
    if anomaly_manifest.get("spatial_interpolation") is not False:
        raise ValueError("anomaly maps must stay on the native grid")
    if "training-only" not in str(anomaly_manifest.get("climatology", "")):
        raise ValueError("anomaly climatology is not explicitly training-only")
    warning = anomaly_manifest.get("displayed_map_correlation", {}).get("warning")
    if warning != "descriptive r_map; not mean per-case ACC":
        raise ValueError("composite-map correlation is not distinguished from ACC")

    _verify_manifest_artifacts(spatial, spatial_manifest, "artifacts")
    _verify_manifest_artifacts(acc_curve, curve_manifest, "figures")
    _verify_manifest_artifacts(paired_acc, paired_manifest, "figures")
    _verify_manifest_artifacts(jjas, jjas_manifest, "artifacts")
    _verify_manifest_artifacts(anomaly, anomaly_manifest, "artifacts")

    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _story() -> str:
    return """# Corrected Forecast — revised figure story

## Slide order

1. **Native-grid rainfall composites:** lead weeks run downward; IMD Observation, Raw FuXi-S2S, Corrected Forecast, IMD−Raw, and IMD−Corrected run across.
2. **Interpolated view:** the same 70-start composites, smoothed for display only. Metrics remain on the native 1.5° grid.
3. **JJAS-average ACC curve:** Raw FuXi-S2S, training-only log-bias, and Corrected Forecast are verified against IMD. IMD is the reference, not a fourth forecast curve; an IMD self-correlation curve would be exactly 1 and misleading.
4. **Paired ACC cases:** 329/420 initialization × lead cases favor Corrected Forecast; pooled ΔACC is +0.106.
5. **JJAS month × lead:** ACC and RMSE improve in 22/24 point-estimate cells; national absolute mean bias improves in only 6/24.
6. **JJAS uncertainty:** improvements are clearest early in the season and weaken toward September W5–W6.
7. **Weekwise anomaly maps:** anomalies are relative to the case/lead-matched fixed 2002–2017 training-only IMD climatology. Corrected Forecast follows the IMD anomaly structure more closely than Raw FuXi-S2S.
8. **Compact anomaly slide:** all 70 starts × six leads summarized. Composite-map correlation is 0.385 for Corrected Forecast versus 0.025 for Raw FuXi-S2S.

## Exact ACC result

- Mean ACC W1–W6: Raw FuXi-S2S = 0.253, training-only log-bias = 0.305, Corrected Forecast = 0.359.
- Corrected Forecast minus Raw FuXi-S2S: +0.139, +0.118, +0.121, +0.094, +0.073, +0.092 for W1–W6.
- All-week paired ΔACC = +0.106 with descriptive 95% percentile interval [+0.044, +0.149].
- W1–W4 intervals are wholly above zero; W5–W6 include zero.

## Important interpretation

- The ACC curve is the quantitative skill result: it averages 70 case-wise spatial correlations at each lead.
- The anomaly-map `r_map` values describe correlations between displayed composite maps. They are visual evidence, not substitutes for the case-wise ACC.
- These 2020–2021 hindcasts are exploratory/reused and not independent confirmation.
- Do not claim that all metrics improve: absolute national-mean bias worsens overall.
"""


def _atomic_write(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package(
    spatial: Path,
    acc_curve: Path,
    paired_acc: Path,
    jjas: Path,
    anomaly: Path,
    output: Path,
) -> Path:
    paths = [Path(value).expanduser().resolve() for value in (
        spatial, acc_curve, paired_acc, jjas, anomaly
    )]
    spatial, acc_curve, paired_acc, jjas, anomaly = paths
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = validate_sources(spatial, acc_curve, paired_acc, jjas, anomaly)

    mapping = (
        (spatial, "01_week_rows_spatial_composites_native_grid", "01_rainfall_native"),
        (spatial, "02_week_rows_spatial_composites_interpolated_display_only", "02_rainfall_interpolated_display_only"),
        (acc_curve, "jjas_average_acc_against_imd_by_lead", "03_jjas_average_acc_curve"),
        (paired_acc, "01_paired_case_acc_raw_vs_corrected_exploratory_2020_2021", "04_paired_case_acc"),
        (jjas, "jjas_month_lead_improvement_tradeoffs", "05_jjas_month_lead_tradeoffs"),
        (jjas, "jjas_month_lead_paired_uncertainty", "06_jjas_month_lead_uncertainty"),
        (anomaly, "07_weekwise_imd_referenced_anomaly_spatial_skill", "07_weekwise_anomaly_maps"),
        (anomaly, "08_jjas_mean_imd_referenced_anomaly_spatial_skill", "08_compact_jjas_anomaly_map"),
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        artifacts: dict[str, str] = {}
        for source, stem, target in mapping:
            for suffix in (".png", ".pdf"):
                source_path = source / f"{stem}{suffix}"
                target_path = temporary / f"{target}{suffix}"
                shutil.copy2(source_path, target_path)
                artifacts[target_path.name] = sha256_file(target_path)

        summary_source = jjas / "jjas_initialization_month_by_lead_summary.csv"
        summary_target = temporary / summary_source.name
        shutil.copy2(summary_source, summary_target)
        artifacts[summary_target.name] = sha256_file(summary_target)

        notes = temporary / "STORY_AND_SLIDE_ORDER.md"
        _atomic_write(notes, _story())
        artifacts[notes.name] = sha256_file(notes)
        manifest = {
            "schema_name": "fuxi_imd_corrected_forecast_revised_figures",
            "schema_version": 1,
            "status": "complete",
            "display_name": "Corrected Forecast",
            "evaluation_scope": (
                "2020-2021 exploratory/reused locked hindcasts; "
                "not independent confirmation"
            ),
            "cases": 70,
            "lead_weeks": [1, 2, 3, 4, 5, 6],
            "source_or_target_arrays_opened_by_packager": False,
            "metrics_recomputed_by_packager": False,
            "independent_confirmation_claimed": False,
            "statistical_significance_claimed": False,
            "sources": sources,
            "artifacts": dict(sorted(artifacts.items())),
        }
        _atomic_write(
            temporary / "MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spatial_directory", type=Path)
    parser.add_argument("acc_curve_directory", type=Path)
    parser.add_argument("paired_acc_directory", type=Path)
    parser.add_argument("jjas_directory", type=Path)
    parser.add_argument("anomaly_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = package(
        args.spatial_directory,
        args.acc_curve_directory,
        args.paired_acc_directory,
        args.jjas_directory,
        args.anomaly_directory,
        args.output_directory,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
