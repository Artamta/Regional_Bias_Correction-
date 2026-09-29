"""Configuration for the global FuXi rainfall calibration experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Large data stay on /storage; code and small launch files stay here."""

    fuxi_store: Path = Path(
        "/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/"
        "native_reforecast_global_2002_2021.zarr"
    )
    data_root: Path = Path(
        "/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global"
    )
    era5_root: Path = data_root / "era5"
    cache_store: Path = data_root / "cache" / "fuxi_tp_2017_2021.zarr"
    runs_root: Path = data_root / "runs"
    forecasts_root: Path = data_root / "local_forecasts"


@dataclass(frozen=True)
class Experiment:
    """One small, fixed experiment; edit here instead of hunting for constants."""

    train_years: tuple[int, ...] = (2017, 2018)
    validation_years: tuple[int, ...] = (2019,)
    test_years: tuple[int, ...] = (2020, 2021)
    lead_windows: tuple[tuple[int, int], ...] = ((19, 25), (26, 32))
    feature_names: tuple[str, ...] = (
        "log_p_q1",
        "log_p_q2",
        "log_p_q3",
        "log_p_q4",
        "log_p_q5",
        "tp_q10",
        "tp_q25",
        "tp_q50",
        "tp_q75",
        "tp_q90",
        "sin_lat",
        "cos_lat",
        "sin_lon",
        "cos_lon",
        "sin_doy",
        "cos_doy",
        "lead_flag",
        "land_fraction",
    )
    seed: int = 42
    base_channels: int = 16
    dropout: float = 0.10
    batch_size: int = 8
    max_epochs: int = 60
    patience: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    correction_penalty: float = 1.0e-4
    gradient_clip: float = 1.0

    @property
    def in_channels(self) -> int:
        return len(self.feature_names)


PATHS = Paths()
EXPERIMENT = Experiment()


def serializable_config() -> dict:
    """Return the complete configuration without Path objects."""

    values = {"paths": asdict(PATHS), "experiment": asdict(EXPERIMENT)}
    values["paths"] = {key: str(value) for key, value in values["paths"].items()}
    return values
