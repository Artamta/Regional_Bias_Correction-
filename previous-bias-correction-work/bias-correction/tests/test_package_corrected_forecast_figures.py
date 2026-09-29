"""Contract checks for the revised Corrected Forecast package."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate"
    / "package_corrected_forecast_figures.py"
)
SPEC = importlib.util.spec_from_file_location("package_corrected_forecast_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
package_figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_figures)

ROOT = Path(__file__).resolve().parents[1] / "presentation"


def test_real_revised_sources_have_locked_semantics() -> None:
    sources = package_figures.validate_sources(
        ROOT / "clear_story_spatial_transposed_20260812_v1",
        ROOT / "generated" / "jjas_average_acc_curve_exploratory_2020_2021_v1",
        ROOT / "generated" / "locked_exploratory_acc_corrected_forecast_2020_2021_v1",
        ROOT / "derived" / "jjas_month_lead_corrected_forecast_final_20260812",
        ROOT / "anomaly_spatial_skill_20260812_v2",
    )
    assert set(sources) == {
        "transposed_spatial",
        "jjas_acc_curve",
        "paired_acc",
        "jjas_month_lead",
        "anomaly_maps",
    }


def test_story_distinguishes_acc_from_composite_map_correlation() -> None:
    story = package_figures._story()
    assert "Corrected Forecast" in story
    assert "IMD is the reference" in story
    assert "not substitutes for the case-wise ACC" in story
    assert "not independent confirmation" in story
