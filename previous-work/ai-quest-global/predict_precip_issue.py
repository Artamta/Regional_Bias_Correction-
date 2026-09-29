#!/usr/bin/env python3
"""Create the local 20260813 precipitation forecast from the frozen candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from build_precip_cache import FUXI_CHANNELS, open_era5
from debias import SeasonalDebiasPlusPlus, apply_uniform_gamma
from precip_contract import (
    GRID_SHAPE,
    ISSUE_DATE,
    MEMBER_QUANTILES,
    TARGET_WINDOWS,
    build_model_inputs,
    climatology_boundaries,
    climatology_sample_starts,
    contract_sha256,
    jeffreys_probabilities,
    member_category_counts,
    weekly_fuxi_fields,
)
from precip_model import PrecipQuestAdapter


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_operational_config(path: Path, issue: pd.Timestamp) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("members", -1)) != 51 or int(config.get("lead_days", -1)) != 42:
        raise ValueError("Quest operational config must specify 51 members and 42 days")
    if config.get("output_contract") != "quest_global_multivariable_v1":
        raise ValueError("Quest operational config must retain the global six-field output")
    if tuple(config.get("retained_variables", ())) != FUXI_CHANNELS:
        raise ValueError("Quest operational config has the wrong retained variables")
    if config.get("input", {}).get("builder") != "gfs_daily_proxy":
        raise ValueError("the 20260813 run must disclose the experimental GFS proxy")
    temporal = config.get("temporal_alignment", {})
    if temporal.get("issue_hour_utc") != 0 or max(temporal.get("input_day_offsets", [1])) > 0:
        raise ValueError("operational inputs violate the Thursday 00 UTC cutoff")
    calendar = Path(config["calendar"])
    if not calendar.is_absolute():
        calendar = Path("/home/raj.ayush/s2s/s2s_anlysis") / calendar
    frame = pd.read_csv(calendar)
    if len(frame) != 1 or pd.Timestamp(frame.iloc[0]["init_date"]).normalize() != issue:
        raise ValueError("operational calendar does not match the issue date")
    return config


def _weekly_era5(era5: xr.DataArray, start: pd.Timestamp) -> np.ndarray:
    from precip_contract import aggregate_era5_six_hour_week

    return np.asarray(aggregate_era5_six_hour_week(era5, start).values, dtype=np.float32)


def _official_climatology_boundaries(path: Path) -> np.ndarray:
    """Load one AI-WQ precipitation climatology file as ``[4,121,240]``."""

    with xr.open_dataarray(path) as source:
        field = source.squeeze(drop=True)
        rename = {}
        for candidates, canonical in (
            (("latitude", "lat"), "lat"),
            (("longitude", "lon"), "lon"),
        ):
            found = next((name for name in candidates if name in field.dims), None)
            if found is None:
                raise ValueError(f"official climatology {path} lacks {canonical}")
            if found != canonical:
                rename[found] = canonical
        if rename:
            field = field.rename(rename)
        category_dims = [name for name in field.dims if name not in {"lat", "lon"}]
        if len(category_dims) != 1 or field.sizes[category_dims[0]] != 4:
            raise ValueError(
                f"official climatology {path} must contain four boundaries"
            )
        field = field.assign_coords(lon=np.mod(field.lon.values, 360.0)).sortby("lon")
        field = field.sortby("lat", ascending=False).transpose(
            category_dims[0], "lat", "lon"
        )
        expected_lat = np.linspace(90.0, -90.0, GRID_SHAPE[0])
        expected_lon = np.arange(GRID_SHAPE[1]) * 1.5
        if not np.allclose(field.lat.values, expected_lat, atol=1.0e-6):
            raise ValueError(f"official climatology {path} has the wrong latitude grid")
        if not np.allclose(field.lon.values, expected_lon, atol=1.0e-6):
            raise ValueError(f"official climatology {path} has the wrong longitude grid")
        values = np.asarray(field.values, dtype=np.float32)
    if values.shape != (4, *GRID_SHAPE) or not np.isfinite(values).all():
        raise ValueError(f"official climatology {path} is not a finite Quest-grid field")
    if np.any(values < -1.0e-6) or np.any(np.diff(values, axis=0) < -1.0e-6):
        raise ValueError(f"official climatology {path} has invalid ordered boundaries")
    return np.maximum(values, 0.0)


def _prepare_inputs(args: argparse.Namespace):
    import zarr

    issue = pd.Timestamp(args.issue_date).normalize()
    if issue.strftime("%Y%m%d") != ISSUE_DATE or issue.dayofweek != 3:
        raise ValueError("this frozen workflow accepts only Thursday issue 20260813")
    _validate_operational_config(args.operational_config, issue)
    with xr.open_dataset(args.forecast, chunks=None) as dataset:
        required_dims = {"member": 51, "lead_day": 42, "lat": 121, "lon": 240}
        for name, size in required_dims.items():
            if dataset.sizes.get(name) != size:
                raise ValueError(f"operational forecast {name}={dataset.sizes.get(name)}, expected {size}")
        if not set(FUXI_CHANNELS).issubset({str(item) for item in dataset.channel.values}):
            raise ValueError("operational forecast lacks required TP/physical fields")
        daily = {
            variable: np.asarray(dataset.forecast.sel(channel=variable).values, dtype=np.float32)
            for variable in FUXI_CHANNELS
        }
    context, target_members = weekly_fuxi_fields(daily)
    target_quantiles = np.moveaxis(
        np.quantile(np.log1p(target_members), MEMBER_QUANTILES, axis=1), 0, 1
    ).astype(np.float32)
    thresholds = np.empty((2, 4, *GRID_SHAPE), dtype=np.float32)
    counts = np.empty((2, 5, *GRID_SHAPE), dtype=np.uint8)
    official_paths = (args.climatology_period_1, args.climatology_period_2)
    if any(path is not None for path in official_paths):
        if not all(path is not None for path in official_paths):
            raise ValueError("both official climatology-period files are required")
        for lead, path in enumerate(official_paths):
            assert path is not None
            thresholds[lead] = _official_climatology_boundaries(path)
            counts[lead] = member_category_counts(target_members[lead], thresholds[lead])
    else:
        era5 = open_era5(args.era5_store, args.era5_variable)
        for lead, (start_day, _end_day) in enumerate(TARGET_WINDOWS):
            valid_start = issue + pd.Timedelta(days=start_day - 1)
            samples = np.stack(
                [_weekly_era5(era5, item) for item in climatology_sample_starts(valid_start)]
            )
            thresholds[lead] = climatology_boundaries(samples)
            counts[lead] = member_category_counts(target_members[lead], thresholds[lead])
    p0 = jeffreys_probabilities(counts)
    cache = zarr.open_group(str(args.cache), mode="r")
    if cache.attrs.get("status") != "complete" or cache.attrs.get("contract_sha256") != contract_sha256():
        raise ValueError("training cache is not the frozen complete contract")
    context_x, target_x = build_model_inputs(
        context,
        target_quantiles,
        p0,
        issue,
        np.asarray(cache["land_fraction"][:], dtype=np.float32),
        np.asarray(cache.attrs["context_mean"], dtype=np.float32),
        np.asarray(cache.attrs["context_std"], dtype=np.float32),
        np.asarray(cache.attrs["target_quantile_mean"], dtype=np.float32),
        np.asarray(cache.attrs["target_quantile_std"], dtype=np.float32),
    )
    return context_x, target_x, counts, p0, thresholds


def _run_neural(
    selection: dict,
    winner: str,
    context_x: np.ndarray,
    target_x: np.ndarray,
    p0: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    matching = [
        item
        for item in selection["neural_configurations"]
        if str(item["variant"]) in winner and f"w{item['width']}" in winner
    ]
    if "ensemble" not in winner:
        matching = [item for item in matching if f"seed{item['seed']}" in winner]
    if not matching:
        raise RuntimeError("frozen neural winner has no matching checkpoint")
    outputs = []
    context = torch.from_numpy(context_x[None]).to(device)
    target = torch.from_numpy(target_x[None]).to(device)
    anchor = torch.from_numpy(p0[None]).to(device)
    for item in matching:
        checkpoint_path = Path(item["run"]) / "checkpoints" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["configuration_sha256"] != item["configuration_sha256"]:
            raise ValueError("neural checkpoint does not match frozen configuration hash")
        model = PrecipQuestAdapter(**checkpoint["model_kwargs"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        with torch.no_grad():
            outputs.append(model(context, target, anchor)[0].float().cpu().numpy())
    probability = np.mean(outputs, axis=0)
    if winner.startswith("calibrated_"):
        gamma = np.asarray(selection["neural_gamma_by_lead"], dtype=np.float32)
        probability = apply_uniform_gamma(probability[None], gamma)[0]
    return probability


def predict(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_from_validation":
        raise ValueError("candidate selection has not been frozen from validation")
    payload = dict(selection)
    recorded_hash = payload.pop("selection_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != recorded_hash:
        raise ValueError("frozen selection hash is invalid")

    context_x, target_x, counts, p0, thresholds = _prepare_inputs(args)
    winner = str(selection["winner"])
    if winner == "uniform":
        probability = np.full_like(p0, 0.2)
    elif winner == "raw_p0":
        probability = p0
    elif winner == "calibrated_p0":
        probability = apply_uniform_gamma(
            p0[None], np.asarray(selection["p0_gamma_by_lead"], dtype=np.float32)
        )[0]
    elif "debias_plus_plus" in winner:
        with np.load(args.debias, allow_pickle=False) as archive:
            if str(archive["cache_contract_sha256"].item()) != contract_sha256():
                raise ValueError("Debias++ artifact has a stale cache contract")
            debias = SeasonalDebiasPlusPlus(
                np.asarray(archive["error_sum"], dtype=np.float32),
                np.asarray(archive["valid_count"], dtype=np.uint16),
                float(archive["clip"]),
            )
            selected_spans = np.asarray(
                archive["selected_span_by_lead"], dtype=np.int16
            )
        issue = pd.Timestamp(ISSUE_DATE)
        target_doys = np.asarray(
            [[
                (issue + pd.Timedelta(days=18)).dayofyear - 1,
                (issue + pd.Timedelta(days=25)).dayofyear - 1,
            ]],
            dtype=np.int16,
        )
        if selected_spans.tolist() != selection["debias_span_by_lead"]:
            raise ValueError("Debias++ artifact does not match the frozen spans")
        probability = debias.predict(p0[None], target_doys, selected_spans)[0]
        if winner.startswith("calibrated_"):
            probability = apply_uniform_gamma(
                probability[None],
                np.asarray(selection["debias_gamma_by_lead"], dtype=np.float32),
            )[0]
    elif "neural" in winner:
        probability = _run_neural(
            selection,
            winner,
            context_x,
            target_x,
            p0,
            torch.device(args.device),
        )
    else:
        raise ValueError(f"unsupported frozen winner {winner!r}")
    if probability.shape != (2, 5, *GRID_SHAPE):
        raise RuntimeError("candidate returned the wrong probability shape")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("candidate returned invalid probabilities")
    if float(np.max(np.abs(probability.sum(axis=1) - 1.0))) > 1.0e-6:
        raise RuntimeError("candidate probabilities do not sum to one within 1e-6")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probabilities=probability.astype(np.float32),
        p0=p0.astype(np.float32),
        category_counts=counts,
        thresholds=thresholds,
        context_x=context_x.astype(np.float16),
        target_x=target_x.astype(np.float16),
        issue_date=np.asarray(ISSUE_DATE),
        winner=np.asarray(winner),
        selection_sha256=np.asarray(recorded_hash),
        cache_contract_sha256=np.asarray(contract_sha256()),
    )
    manifest = {
        "status": "local_forecast_complete",
        "issue_date": ISSUE_DATE,
        "winner": winner,
        "selection_sha256": recorded_hash,
        "selection_file": str(args.selection),
        "selection_file_sha256": sha256_file(args.selection),
        "cache_contract_sha256": contract_sha256(),
        "initialization": "experimental GFS-proxy FuXi-S2S",
        "input_cutoff": "2026-08-13T00:00:00Z",
        "members": 51,
        "lead_days": 42,
        "source_forecast": str(args.forecast),
        "source_forecast_sha256": sha256_file(args.forecast),
        "operational_config": str(args.operational_config),
        "operational_config_sha256": sha256_file(args.operational_config),
        "climatology": (
            {
                "kind": args.climatology_kind,
                "period_1": str(args.climatology_period_1),
                "period_1_sha256": sha256_file(args.climatology_period_1),
                "period_2": str(args.climatology_period_2),
                "period_2_sha256": sha256_file(args.climatology_period_2),
            }
            if args.climatology_period_1 is not None
            else {
                "kind": "locally_reconstructed_from_era5_store",
                "era5_store": str(args.era5_store),
            }
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "created_utc": utc_now(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def emergency_uniform(args: argparse.Namespace) -> None:
    if args.issue_date != ISSUE_DATE:
        raise ValueError("emergency fallback is frozen to issue 20260813")
    if args.forecast.exists():
        raise RuntimeError("live FuXi forecast exists; emergency uniform fallback is not allowed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probability = np.full((2, 5, *GRID_SHAPE), 0.2, dtype=np.float32)
    np.savez_compressed(
        args.output,
        probabilities=probability,
        issue_date=np.asarray(ISSUE_DATE),
        winner=np.asarray("emergency_uniform_no_live_fuxi"),
    )
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "status": "emergency_uniform",
                "reason": args.reason,
                "issue_date": ISSUE_DATE,
                "wrong_date_or_regional_data_used": False,
                "output_sha256": sha256_file(args.output),
                "created_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--issue-date", default=ISSUE_DATE)
    common.add_argument("--forecast", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    predict_parser = sub.add_parser("predict", parents=[common])
    predict_parser.add_argument("--operational-config", type=Path, required=True)
    predict_parser.add_argument("--era5-store", type=Path, required=True)
    predict_parser.add_argument("--era5-variable", default="total_precipitation_6hr")
    predict_parser.add_argument("--climatology-period-1", type=Path)
    predict_parser.add_argument("--climatology-period-2", type=Path)
    predict_parser.add_argument(
        "--climatology-kind",
        choices=("official_ai_wq_downloads", "public_era5_lagged_proxy"),
        default="official_ai_wq_downloads",
    )
    predict_parser.add_argument("--cache", type=Path, required=True)
    predict_parser.add_argument("--selection", type=Path, required=True)
    predict_parser.add_argument("--debias", type=Path, required=True)
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.set_defaults(function=predict)
    fallback = sub.add_parser("emergency-uniform", parents=[common])
    fallback.add_argument("--reason", required=True)
    fallback.set_defaults(function=emergency_uniform)
    return result


if __name__ == "__main__":
    namespace = parser().parse_args()
    namespace.function(namespace)
