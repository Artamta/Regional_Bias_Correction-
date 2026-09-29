"""Run the untrained three-variable temporal model on a synthetic global case.

This is an executable contract prototype, not a forecast and not a submission.
It proves that the joint 38-channel model can preserve three distinct anchors
on the native grid and export the required six local variable/period files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data import QUEST_WINDOWS, weekly_valid_dates
from model import QuestTemporalProbUNet
from multivariable import (
    MULTIVARIABLE_FEATURE_NAMES,
    MULTIVARIABLE_INPUT_CHANNELS,
    QUEST_VARIABLES,
)
from predict import (
    FUXI_PERMISSION_WARNING,
    LATITUDE,
    LONGITUDE,
    canonical_date,
    export_all_variables,
)


@dataclass(frozen=True)
class PrototypeArtifacts:
    """Files written by one synthetic prototype run."""

    forecasts: tuple[Path, ...]
    checkpoint: Path
    manifest: Path


def synthetic_anchor() -> np.ndarray:
    """Return distinct valid anchors with shape ``[3,2,5,121,240]``."""

    category_probabilities = np.asarray(
        [
            [[0.10, 0.15, 0.20, 0.25, 0.30], [0.14, 0.17, 0.20, 0.23, 0.26]],
            [[0.22, 0.21, 0.20, 0.19, 0.18], [0.18, 0.19, 0.20, 0.21, 0.22]],
            [[0.30, 0.25, 0.20, 0.15, 0.10], [0.26, 0.23, 0.20, 0.17, 0.14]],
        ],
        dtype=np.float32,
    )
    return np.broadcast_to(
        category_probabilities[:, :, :, None, None],
        (3, 2, 5, LATITUDE.size, LONGITUDE.size),
    ).copy()


def synthetic_features(anchor: np.ndarray, init_date: str) -> np.ndarray:
    """Return a self-consistent synthetic ``[2,38,121,240]`` feature tensor."""

    expected = (3, 2, 5, LATITUDE.size, LONGITUDE.size)
    values = np.asarray(anchor, dtype=np.float32)
    if values.shape != expected:
        raise ValueError(f"anchor must have shape {expected}; got {values.shape}")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("anchor must be finite and strictly positive")
    if not np.allclose(values.sum(axis=2), 1.0, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("anchor probabilities must sum to one")

    features = np.zeros(
        (2, MULTIVARIABLE_INPUT_CHANNELS, LATITUDE.size, LONGITUDE.size),
        dtype=np.float32,
    )
    feature_index = {
        name: index for index, name in enumerate(MULTIVARIABLE_FEATURE_NAMES)
    }
    for variable_index, variable in enumerate(QUEST_VARIABLES):
        start = feature_index[f"{variable}_log_p_q1"]
        features[:, start : start + 5] = np.log(values[variable_index])

    latitude_radians = np.deg2rad(LATITUDE.astype(np.float64))
    longitude_radians = np.deg2rad(LONGITUDE.astype(np.float64))
    features[:, feature_index["sin_lat"]] = np.sin(latitude_radians)[:, None]
    features[:, feature_index["cos_lat"]] = np.cos(latitude_radians)[:, None]
    features[:, feature_index["sin_lon"]] = np.sin(longitude_radians)[None, :]
    features[:, feature_index["cos_lon"]] = np.cos(longitude_radians)[None, :]
    starts = [weekly_valid_dates(init_date, window)[0] for window in QUEST_WINDOWS]
    phases = np.asarray(
        [2.0 * np.pi * (valid.dayofyear - 1) / 365.2425 for valid in starts]
    )
    features[:, feature_index["sin_doy"]] = np.sin(phases)[:, None, None]
    features[:, feature_index["cos_doy"]] = np.cos(phases)[:, None, None]
    features[:, feature_index["lead_flag"]] = np.asarray([-1.0, 1.0])[
        :, None, None
    ]
    # A real cache must use the official land-fraction field. Ones are explicit
    # synthetic placeholders and are never represented as real predictors.
    features[:, feature_index["land_fraction"]] = 1.0
    return features


def _portable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }


def run_prototype(
    output_dir: str | Path,
    *,
    init_date: str,
    device: str = "cpu",
    base_channels: int = 16,
    attention_heads: int = 4,
) -> PrototypeArtifacts:
    """Run identity inference and write six NetCDFs plus an untrained checkpoint."""

    resolved_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else device
    selected_device = torch.device(resolved_device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")

    canonical_init_date = canonical_date(init_date)
    anchor = synthetic_anchor()
    features = synthetic_features(anchor, canonical_init_date)
    model_config: dict[str, Any] = {
        "in_channels": MULTIVARIABLE_INPUT_CHANNELS,
        "variables": list(QUEST_VARIABLES),
        "base_channels": int(base_channels),
        "dropout": 0.0,
        "attention_heads": int(attention_heads),
        "attention_dropout": 0.0,
    }
    model = QuestTemporalProbUNet(
        in_channels=model_config["in_channels"],
        variables=tuple(model_config["variables"]),
        base_channels=model_config["base_channels"],
        dropout=model_config["dropout"],
        attention_heads=model_config["attention_heads"],
        attention_dropout=model_config["attention_dropout"],
    ).to(selected_device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(
            torch.from_numpy(features[None]).to(selected_device),
            torch.from_numpy(anchor[None]).to(selected_device),
        ).cpu().numpy()[0]
    identity_error = float(np.max(np.abs(probabilities - anchor)))
    if identity_error > 1.0e-6:
        raise RuntimeError(
            f"zero-head identity check failed (maximum error {identity_error})"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    forecasts = export_all_variables(
        probabilities,
        destination,
        init_date=canonical_init_date,
        variables=QUEST_VARIABLES,
        model_name="QuestTemporalProbUNet-untrained-prototype",
    )
    checkpoint = destination / "prototype_untrained.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_name": "QuestTemporalProbUNet",
            "model_state": _portable_state_dict(model),
            "model_config": model_config,
            "feature_contract": {
                "name": "quest_multivariable_38_channel_v1",
                "feature_names": list(MULTIVARIABLE_FEATURE_NAMES),
                "variable_order": list(QUEST_VARIABLES),
            },
            "metadata": {
                "trained": False,
                "data": "synthetic",
                "submission_ready": False,
                "permission_notice": FUXI_PERMISSION_WARNING,
            },
        },
        checkpoint,
    )
    manifest = destination / "prototype_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "synthetic_untrained_contract_prototype",
                "submission_ready": False,
                "init_date": canonical_init_date,
                "variables": list(QUEST_VARIABLES),
                "lead_windows": list(QUEST_WINDOWS),
                "feature_contract": "quest_multivariable_38_channel_v1",
                "feature_count": MULTIVARIABLE_INPUT_CHANNELS,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "maximum_anchor_identity_error": identity_error,
                "forecast_files": [path.name for path in forecasts],
                "checkpoint": checkpoint.name,
                "limitations": [
                    "synthetic inputs and untrained weights",
                    "attention mixes only D19-25 and D26-32, not all six weeks",
                    "real training requires a new three-variable cache and train-only normalization",
                    FUXI_PERMISSION_WARNING,
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PrototypeArtifacts(forecasts, checkpoint, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--init-date", default=date.today().isoformat())
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_prototype(
        args.output_dir,
        init_date=args.init_date,
        device=args.device,
        base_channels=args.base_channels,
        attention_heads=args.attention_heads,
    )
    print("synthetic prototype complete; these files are not submission-ready")
    for path in (*artifacts.forecasts, artifacts.checkpoint, artifacts.manifest):
        print(path)


if __name__ == "__main__":
    main()
