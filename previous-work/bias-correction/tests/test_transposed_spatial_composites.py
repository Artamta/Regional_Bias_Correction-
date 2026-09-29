"""Focused contract tests for the week-row spatial composites."""

from __future__ import annotations

import sys
from pathlib import Path


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import plot_transposed_spatial_composites as composites  # noqa: E402


def test_transposed_layout_has_leads_down_and_requested_fields_across() -> None:
    contract = composites.layout_contract()
    assert contract["rows"] == [f"WEEK {lead}" for lead in range(1, 7)]
    assert contract["columns"] == [
        "IMD Observation",
        "Raw FuXi-S2S",
        "Corrected Forecast",
        "IMD − Raw FuXi",
        "IMD − Corrected Forecast",
    ]
    assert contract["difference_sign"].startswith("IMD minus forecast")


def test_corrected_forecast_is_the_only_corrected_display_name() -> None:
    assert composites.COLUMN_TITLES[2] == "Corrected Forecast"
    assert composites.COLUMN_TITLES[4] == "IMD − Corrected Forecast"
    assert all("Frozen" not in value for value in composites.COLUMN_TITLES)
    assert all("corrected" not in value for value in composites.COLUMN_TITLES)


def test_locked_composite_scope_is_explicit() -> None:
    assert composites.CASE_COUNT == 70
    assert composites.LEAD_LABELS == (
        "WEEK 1",
        "WEEK 2",
        "WEEK 3",
        "WEEK 4",
        "WEEK 5",
        "WEEK 6",
    )
