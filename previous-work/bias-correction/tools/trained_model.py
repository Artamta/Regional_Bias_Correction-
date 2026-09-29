"""Load and run the trained FuXi–IMERG residual U-Net ensemble.

Example
-------
from trained_model import predict_fuxi_shards

prediction = predict_fuxi_shards([
    "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
    "native_reforecast_jjas_2002_2021/shards/20200602.nc"
])
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent if HERE.name == "tools" else HERE
LOCAL_BUNDLE = HERE / "residual_unet_ensemble.pt"
DEFAULT_BUNDLE = (
    LOCAL_BUNDLE
    if LOCAL_BUNDLE.is_file()
    else REPOSITORY_ROOT
    / "trained_model"
    / "fuxi_imerg_2014_2018_v1"
    / "residual_unet_ensemble.pt"
)


def _model_class(bundle_path: Path):
    sys.dont_write_bytecode = True
    frozen_code = bundle_path.parent / "code"
    live_code = REPOSITORY_ROOT.parent / "neural_adapter" / "src"
    code = frozen_code if frozen_code.is_dir() else live_code
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from fuxi_adapter.models import ResidualUNet

    return ResidualUNet


def _calendar_positions(dates: np.ndarray) -> np.ndarray:
    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    flat = np.asarray(dates, dtype="datetime64[D]").reshape(-1)
    positions = np.asarray(
        [lookup[pd.Timestamp(value).strftime("%m-%d")] for value in flat],
        dtype=np.int16,
    )
    return positions.reshape(np.asarray(dates).shape)


class FuXiImergModel:
    """Three-seed deterministic post-processor for weekly FuXi rainfall."""

    def __init__(self, bundle=DEFAULT_BUNDLE, device=None):
        self.bundle_path = Path(bundle).resolve()
        if not self.bundle_path.is_file():
            raise FileNotFoundError(self.bundle_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu":
            torch.set_num_threads(min(8, os.cpu_count() or 1))

        saved = torch.load(
            self.bundle_path, map_location="cpu", weights_only=True
        )
        if saved["format_version"] != "fuxi-imerg-residual-unet-v1":
            raise ValueError("unsupported trained-model bundle")

        self.latitude = saved["latitude"].numpy().astype(np.float64)
        self.longitude = saved["longitude"].numpy().astype(np.float64)
        self.support = saved["support"].numpy().astype(bool)
        self.daily_climatology = (
            saved["daily_imerg_climatology"].numpy().astype(np.float32)
        )
        self.normalization = saved["normalization"]
        self.target_scale = np.asarray(
            self.normalization["target_rms_by_lead"], dtype=np.float32
        )
        self.seeds = [int(item["seed"]) for item in saved["models"]]

        ResidualUNet = _model_class(self.bundle_path)
        self.models = []
        for item in saved["models"]:
            model = ResidualUNet(
                in_channels=9, base_channels=16, dropout=0.1
            )
            model.load_state_dict(item["model_state_dict"], strict=True)
            model.to(self.device).eval()
            self.models.append(model)

    def weekly_climatology(self, initializations) -> np.ndarray:
        initializations = np.asarray(initializations, dtype="datetime64[D]")
        offsets = np.arange(42, dtype="timedelta64[D]").reshape(1, 6, 7)
        valid_dates = initializations[:, None, None] + offsets
        daily = self.daily_climatology[_calendar_positions(valid_dates)]
        return np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)

    def make_inputs(
        self,
        fuxi_mean_mm_day: np.ndarray,
        fuxi_spread_mm_day: np.ndarray,
        initializations,
    ) -> np.ndarray:
        mean = np.asarray(fuxi_mean_mm_day, dtype=np.float32)
        spread = np.asarray(fuxi_spread_mm_day, dtype=np.float32)
        if mean.shape != spread.shape or mean.ndim != 4:
            raise ValueError(
                "mean and spread must match [case, 6, latitude, longitude]"
            )
        if mean.shape[1:] != (6, 27, 27):
            raise ValueError(f"expected [case, 6, 27, 27], got {mean.shape}")
        if np.any(mean < 0.0) or np.any(spread < 0.0):
            raise ValueError("FuXi mean and spread must be nonnegative")

        initializations = np.asarray(initializations, dtype="datetime64[D]")
        if initializations.shape != (len(mean),):
            raise ValueError("one initialization date is required per case")
        climatology = self.weekly_climatology(initializations)

        fields = {
            "log_fuxi_mean": np.log1p(mean).astype(np.float32),
            "log_fuxi_spread": np.log1p(spread).astype(np.float32),
            "log_imerg_climatology": np.log1p(climatology).astype(np.float32),
        }
        channels = []
        for name, values in fields.items():
            statistics = self.normalization[name]
            centre = np.asarray(statistics["mean_by_lead"], dtype=np.float32)
            scale = np.asarray(statistics["std_by_lead"], dtype=np.float32)
            normalized = (
                values - centre[None, :, None, None]
            ) / scale[None, :, None, None]
            channels.append(
                np.where(
                    self.support[None, None] & np.isfinite(normalized),
                    normalized,
                    0.0,
                ).astype(np.float32)
            )

        cases, leads, height, width = mean.shape
        latitude = self.latitude.astype(np.float32)
        longitude = self.longitude.astype(np.float32)
        latitude = 2.0 * (latitude - latitude.min()) / (
            latitude.max() - latitude.min()
        ) - 1.0
        longitude = 2.0 * (longitude - longitude.min()) / (
            longitude.max() - longitude.min()
        ) - 1.0
        lat_grid = np.broadcast_to(
            latitude[None, None, :, None], (cases, leads, height, width)
        )
        lon_grid = np.broadcast_to(
            longitude[None, None, None, :], (cases, leads, height, width)
        )

        midpoints = initializations[:, None] + (
            7 * np.arange(6)[None, :] + 3
        ).astype("timedelta64[D]")
        midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
        day_of_year = (midpoint_index.dayofyear.to_numpy() - 1).reshape(
            cases, leads
        )
        angle = 2.0 * np.pi * day_of_year / 365.2425
        season_sin = np.broadcast_to(
            np.sin(angle)[:, :, None, None], (cases, leads, height, width)
        )
        season_cos = np.broadcast_to(
            np.cos(angle)[:, :, None, None], (cases, leads, height, width)
        )
        lead_grid = np.broadcast_to(
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, :, None, None],
            (cases, leads, height, width),
        )
        support_grid = np.broadcast_to(
            self.support[None, None], (cases, leads, height, width)
        ).astype(np.float32)
        channels.extend(
            [lat_grid, lon_grid, season_sin, season_cos, lead_grid, support_grid]
        )
        inputs = np.stack(channels, axis=2).astype(np.float32)
        if not np.isfinite(inputs).all():
            raise ValueError("model inputs contain non-finite values")
        return inputs

    def predict(
        self,
        fuxi_mean_mm_day: np.ndarray,
        fuxi_spread_mm_day: np.ndarray,
        initializations,
        batch_size=32,
    ) -> np.ndarray:
        mean = np.asarray(fuxi_mean_mm_day, dtype=np.float32)
        inputs = self.make_inputs(mean, fuxi_spread_mm_day, initializations)
        predictions = []
        with torch.inference_mode():
            for model in self.models:
                residual_parts = []
                for start in range(0, len(inputs), batch_size):
                    batch = torch.from_numpy(inputs[start : start + batch_size]).to(
                        self.device
                    )
                    residual_parts.append(model(batch).float().cpu().numpy())
                residual = np.concatenate(residual_parts).astype(np.float32)
                log_rain = np.log1p(mean) + residual * self.target_scale[
                    None, :, None, None
                ]
                rainfall = np.expm1(np.clip(log_rain, 0.0, 20.0))
                rainfall[..., ~self.support] = np.nan
                predictions.append(rainfall.astype(np.float32))
        return np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)

    def predict_from_members(
        self,
        daily_members: np.ndarray,
        initializations,
        units="mm h-1",
        batch_size=32,
    ) -> np.ndarray:
        values = np.asarray(daily_members, dtype=np.float32)
        if values.ndim == 4:
            values = values[None]
        if values.ndim != 5 or values.shape[2:] != (42, 27, 27):
            raise ValueError(
                "daily_members must be [case, member, 42, 27, 27]"
            )
        if units == "mm h-1":
            values = values * np.float32(24.0)
        elif units != "mm day-1":
            raise ValueError("units must be 'mm h-1' or 'mm day-1'")
        member_weekly = values.reshape(
            values.shape[0], values.shape[1], 6, 7, 27, 27
        ).mean(axis=3, dtype=np.float64)
        mean = member_weekly.mean(axis=1, dtype=np.float64).astype(np.float32)
        spread = member_weekly.std(axis=1, ddof=0).astype(np.float32)
        return self.predict(mean, spread, initializations, batch_size=batch_size)


def predict_fuxi_shards(
    paths: Sequence[str | Path], bundle=DEFAULT_BUNDLE, device=None
) -> xr.DataArray:
    """Predict post-processed weekly rainfall from native FuXi NetCDF shards."""

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
            values = np.asarray(dataset.tp.load().values, dtype=np.float32)
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

    trained = FuXiImergModel(bundle=bundle, device=device)
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
            "method": "three-seed FuXi–IMERG residual U-Net ensemble",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
        },
    )


if __name__ == "__main__":
    model = FuXiImergModel()
    print(f"Loaded seeds {model.seeds} on {model.device}")
