"""Run the trained lead-adaptive FuXi–IMERG post-processor.

Example
-------
from trained_spatiotemporal_model import predict_fuxi_shards

prediction = predict_fuxi_shards(["20200602.nc"])
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr

from trained_model import FuXiImergModel


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent if HERE.name == "tools" else HERE
LOCAL_BUNDLE = HERE / "lead_adaptive_ensemble.pt"
DEFAULT_BUNDLE = (
    LOCAL_BUNDLE
    if LOCAL_BUNDLE.is_file()
    else REPOSITORY_ROOT
    / "trained_model/fuxi_imerg_spatiotemporal_2014_2018_v1/lead_adaptive_ensemble.pt"
)


def _model_classes(bundle_path: Path):
    frozen_code = bundle_path.parent / "code"
    live_code = REPOSITORY_ROOT.parent / "neural_adapter/src"
    code = frozen_code if frozen_code.is_dir() else live_code
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from fuxi_adapter.models import ResidualUNet, TemporalAttentionUNet

    return ResidualUNet, TemporalAttentionUNet


class LeadAdaptiveFuXiImergModel(FuXiImergModel):
    """Three-seed spatial/temporal ensemble selected by lead week."""

    def __init__(self, bundle=DEFAULT_BUNDLE, device=None):
        self.bundle_path = Path(bundle).resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError(self.bundle_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu":
            torch.set_num_threads(min(8, os.cpu_count() or 1))

        saved = torch.load(self.bundle_path, map_location="cpu", weights_only=True)
        if saved["format_version"] != "fuxi-imerg-lead-adaptive-v1":
            raise ValueError("unsupported trained-model bundle")
        self.latitude = saved["latitude"].numpy().astype(np.float64)
        self.longitude = saved["longitude"].numpy().astype(np.float64)
        self.support = saved["support"].numpy().astype(bool)
        self.daily_climatology = saved["daily_imerg_climatology"].numpy().astype(
            np.float32
        )
        self.normalization = saved["normalization"]
        self.target_scale = np.asarray(
            self.normalization["target_rms_by_lead"], dtype=np.float32
        )
        self.lead_sources = tuple(saved["lead_sources"])
        if self.lead_sources != (
            "spatial",
            "spatial",
            "spatial",
            "spatial",
            "temporal",
            "temporal",
        ):
            raise ValueError("unexpected lead-selection contract")

        ResidualUNet, TemporalAttentionUNet = _model_classes(self.bundle_path)
        self.spatial_models = []
        self.temporal_models = []
        self.seeds = []
        for item in saved["models"]:
            spatial = ResidualUNet(9, base_channels=16, dropout=0.1)
            temporal = TemporalAttentionUNet(
                9, base_channels=16, dropout=0.1, max_leads=6
            )
            spatial.load_state_dict(item["spatial_state_dict"], strict=True)
            temporal.load_state_dict(item["temporal_state_dict"], strict=True)
            self.spatial_models.append(spatial.to(self.device).eval())
            self.temporal_models.append(temporal.to(self.device).eval())
            self.seeds.append(int(item["seed"]))

    def _predict_models(self, models, inputs, mean, batch_size):
        predictions = []
        with torch.inference_mode():
            for model in models:
                parts = []
                for start in range(0, len(inputs), batch_size):
                    batch = torch.from_numpy(inputs[start : start + batch_size]).to(
                        self.device
                    )
                    parts.append(model(batch).float().cpu().numpy())
                residual = np.concatenate(parts).astype(np.float32)
                log_rain = np.log1p(mean) + residual * self.target_scale[
                    None, :, None, None
                ]
                rainfall = np.expm1(np.clip(log_rain, 0.0, 20.0))
                rainfall[..., ~self.support] = np.nan
                predictions.append(rainfall.astype(np.float32))
        return np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)

    def predict(
        self,
        fuxi_mean_mm_day: np.ndarray,
        fuxi_spread_mm_day: np.ndarray,
        initializations,
        batch_size=32,
    ) -> np.ndarray:
        mean = np.asarray(fuxi_mean_mm_day, dtype=np.float32)
        inputs = self.make_inputs(mean, fuxi_spread_mm_day, initializations)
        spatial = self._predict_models(
            self.spatial_models, inputs, mean, batch_size
        )
        temporal = self._predict_models(
            self.temporal_models, inputs, mean, batch_size
        )
        result = spatial.copy()
        result[:, 4:] = temporal[:, 4:]
        return result


def predict_fuxi_shards(
    paths: Sequence[str | Path], bundle=DEFAULT_BUNDLE, device=None
) -> xr.DataArray:
    """Post-process native FuXi NetCDF shards into weekly rainfall."""

    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("provide at least one FuXi shard")
    daily = []
    initializations = []
    latitude = longitude = None
    for path in paths:
        with xr.open_dataset(path) as dataset:
            if dataset.tp.attrs.get("units") != "mm h-1":
                raise ValueError(f"unexpected TP units in {path}")
            values = np.asarray(dataset.tp.load(), dtype=np.float32)
            if values.shape[1:] != (42, 27, 27):
                raise ValueError(f"unexpected FuXi shape in {path}: {values.shape}")
            daily.append(values)
            initializations.append(
                np.asarray(
                    dataset.forecast_reference_time.values, dtype="datetime64[D]"
                ).reshape(-1)[0]
            )
            if latitude is None:
                latitude = dataset.latitude.values
                longitude = dataset.longitude.values

    trained = LeadAdaptiveFuXiImergModel(bundle=bundle, device=device)
    if not np.array_equal(latitude, trained.latitude) or not np.array_equal(
        longitude, trained.longitude
    ):
        raise ValueError("FuXi shard grid differs from the trained-model grid")
    initializations = np.asarray(initializations, dtype="datetime64[D]")
    prediction = trained.predict_from_members(
        np.stack(daily), initializations, units="mm h-1"
    )
    return xr.DataArray(
        prediction,
        dims=("init", "lead_week", "latitude", "longitude"),
        coords={
            "init": initializations,
            "lead_week": np.arange(1, 7),
            "latitude": trained.latitude,
            "longitude": trained.longitude,
        },
        name="tp",
        attrs={
            "units": "mm day-1",
            "method": "validation-selected spatial/spatiotemporal U-Net ensemble",
            "lead_selection": "W1-W4 spatial; W5-W6 spatiotemporal",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
        },
    )


if __name__ == "__main__":
    model = LeadAdaptiveFuXiImergModel()
    print(f"Loaded seeds {model.seeds} on {model.device}")
