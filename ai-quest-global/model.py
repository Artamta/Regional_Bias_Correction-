"""Compact probabilistic residual U-Nets for global Quest fields.

The precipitation control shares one spatial correction operator across the
two forecast periods.  The multi-variable prototype adds explicit two-period
bottleneck attention and separate heads for precipitation, temperature, and
pressure.  Every head starts at zero, so a new model reproduces its valid
ensemble anchors.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PROBABILITY_CATEGORIES = 5
FORECAST_PERIODS = 2
ANCHOR_EPSILON = 1.0e-8


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest GroupNorm group count that divides ``channels``."""

    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _GlobalConv3x3(nn.Module):
    """A 3x3 convolution with periodic longitude and bounded latitude."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=0,
            bias=False,
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        # Latitude must not wrap from one pole to the other. Longitude is a
        # periodic coordinate, including at the two latitude-padding rows.
        fields = F.pad(fields, (0, 0, 1, 1), mode="replicate")
        fields = F.pad(fields, (1, 1, 0, 0), mode="circular")
        return self.conv(fields)


class _ConvBlock(nn.Module):
    """Two padded convolutions with GroupNorm, SiLU, and spatial dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            _GlobalConv3x3(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            _GlobalConv3x3(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.block(fields)


class TPProbUNet(nn.Module):
    """Predict calibrated global TP quintile probabilities.

    Parameters
    ----------
    in_channels:
        Predictor channels per forecast period. The approved feature contract
        contains 18 channels.
    base_channels:
        First encoder width. The default produces widths 16, 32, and 64.
    dropout:
        Spatial dropout probability in every convolutional block.
    period_attention:
        If true, mix the two forecast periods with bottleneck attention at
        each spatial location. The default is false so legacy checkpoints and
        independent-period behavior remain unchanged.
    attention_heads:
        Number of heads used only when ``period_attention`` is enabled.
    attention_dropout:
        Transformer dropout used only when ``period_attention`` is enabled.
    Notes
    -----
    Inputs have shape ``[batch, 2, 18, latitude, longitude]``. Forecast periods
    are folded into the batch dimension, so every period uses exactly the same
    U-Net weights. With optional period attention, the two bottleneck tokens at
    each spatial location are mixed before decoding. The five output categories
    are ordered from the lowest to the highest precipitation quintile.
    """

    def __init__(
        self,
        in_channels: int = 18,
        base_channels: int = 16,
        dropout: float = 0.10,
        period_attention: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)
        self.period_attention_enabled = bool(period_attention)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)

        width_1 = self.base_channels
        width_2 = 2 * self.base_channels
        width_3 = 4 * self.base_channels
        if self.period_attention_enabled and width_3 % self.attention_heads != 0:
            raise ValueError(
                "attention_heads must divide the bottleneck channel count "
                f"({width_3})"
            )

        self.encoder_1 = _ConvBlock(self.in_channels, width_1, self.dropout)
        self.encoder_2 = _ConvBlock(width_1, width_2, self.dropout)
        self.bottleneck = _ConvBlock(width_2, width_3, self.dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Do not register even empty attention modules in the default branch:
        # its state-dict keys and random initialization sequence must remain
        # compatible with every pre-attention TPProbUNet checkpoint.
        if self.period_attention_enabled:
            self.period_position = nn.Parameter(
                torch.empty(1, FORECAST_PERIODS, width_3)
            )
            nn.init.normal_(self.period_position, mean=0.0, std=0.02)
            self.period_attention = nn.TransformerEncoderLayer(
                d_model=width_3,
                nhead=self.attention_heads,
                dim_feedforward=2 * width_3,
                dropout=self.attention_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.up_2 = nn.Conv2d(width_3, width_2, kernel_size=1)
        self.decoder_2 = _ConvBlock(width_2 + width_2, width_2, self.dropout)
        self.up_1 = nn.Conv2d(width_2, width_1, kernel_size=1)
        self.decoder_1 = _ConvBlock(width_1 + width_1, width_1, self.dropout)

        self.correction_head = nn.Conv2d(
            width_1, PROBABILITY_CATEGORIES, kernel_size=1
        )
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _mix_periods(
        self, encoded: torch.Tensor, *, batch_size: int, periods: int
    ) -> torch.Tensor:
        """Mix the two lead tokens independently at every bottleneck cell."""

        _, channels, height, width = encoded.shape
        tokens = encoded.reshape(batch_size, periods, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, periods, channels
        )
        tokens = self.period_attention(tokens + self.period_position[:, :periods])
        return (
            tokens.reshape(batch_size, height, width, periods, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * periods, channels, height, width)
        )

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, period, channel, latitude, longitude]"
            )
        if x.shape[1] != FORECAST_PERIODS:
            raise ValueError(f"x must contain exactly {FORECAST_PERIODS} periods")
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[2]} channels; expected {self.in_channels}"
            )
        if x.shape[-2] < 4 or x.shape[-1] < 4:
            raise ValueError("latitude and longitude dimensions must be at least 4")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

    def _validate_inputs(self, x: torch.Tensor, p0: torch.Tensor) -> None:
        self._validate_x(x)
        expected_anchor_shape = (
            x.shape[0],
            FORECAST_PERIODS,
            PROBABILITY_CATEGORIES,
            x.shape[-2],
            x.shape[-1],
        )
        if tuple(p0.shape) != expected_anchor_shape:
            raise ValueError(
                f"p0 must have shape {expected_anchor_shape}; got {tuple(p0.shape)}"
            )
        if not p0.is_floating_point():
            raise TypeError("p0 must be a floating-point tensor")

    def _correction_logits(self, x: torch.Tensor) -> torch.Tensor:
        batch, periods, channels, height, width = x.shape
        fields = x.reshape(batch * periods, channels, height, width)

        skip_1 = self.encoder_1(fields)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))
        if self.period_attention_enabled:
            encoded = self._mix_periods(
                encoded, batch_size=batch, periods=periods
            )

        decoded = F.interpolate(
            encoded,
            size=skip_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, skip_2), dim=1))

        decoded = F.interpolate(
            decoded,
            size=skip_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_1(decoded)
        decoded = self.decoder_1(torch.cat((decoded, skip_1), dim=1))

        correction = self.correction_head(decoded)
        return correction.reshape(
            batch, periods, PROBABILITY_CATEGORIES, height, width
        )

    def forward_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Return the five additive correction logits for each period and cell."""

        self._validate_x(x)
        return self._correction_logits(x)

    def forward(
        self,
        x: torch.Tensor,
        p0: torch.Tensor,
    ) -> torch.Tensor:
        """Return calibrated quintile probabilities with the same grid as ``p0``."""

        self._validate_inputs(x, p0)
        correction = self._correction_logits(x)
        anchor_logits = torch.log(p0.clamp_min(ANCHOR_EPSILON))
        logits = anchor_logits + correction
        return torch.softmax(logits, dim=2)


class QuestTemporalProbUNet(nn.Module):
    """Joint probabilistic adapter for the three global Quest variables.

    The spatial U-Net is shared across variables and forecast periods.  A
    Transformer layer mixes the two official lead periods independently at
    every bottleneck grid location, following the lead-mixing pattern used by
    the India bias-correction experiments.  Separate zero-initialized heads
    produce five correction logits for each requested variable.

    Inputs have shape ``[batch, 2, channel, latitude, longitude]``.  Anchors and
    outputs have shape
    ``[batch, variable, 2, 5, latitude, longitude]``.  As in ``TPProbUNet``, a
    newly constructed model is an exact identity adapter for every variable.
    """

    def __init__(
        self,
        in_channels: int = 38,
        variables: tuple[str, ...] = ("pr", "tas", "mslp"),
        base_channels: int = 16,
        dropout: float = 0.10,
        attention_heads: int = 4,
        attention_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("variables must be a non-empty sequence of unique names")
        if any(not name or not name.isidentifier() for name in variables):
            raise ValueError("variable names must be non-empty Python identifiers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if attention_heads < 1:
            raise ValueError("attention_heads must be positive")

        self.in_channels = int(in_channels)
        self.variables = tuple(str(name) for name in variables)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)

        width_1 = self.base_channels
        width_2 = 2 * self.base_channels
        width_3 = 4 * self.base_channels
        if width_3 % self.attention_heads != 0:
            raise ValueError(
                "attention_heads must divide the bottleneck channel count "
                f"({width_3})"
            )

        self.encoder_1 = _ConvBlock(self.in_channels, width_1, self.dropout)
        self.encoder_2 = _ConvBlock(width_1, width_2, self.dropout)
        self.bottleneck = _ConvBlock(width_2, width_3, self.dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.period_position = nn.Parameter(
            torch.empty(1, FORECAST_PERIODS, width_3)
        )
        nn.init.normal_(self.period_position, mean=0.0, std=0.02)
        self.period_attention = nn.TransformerEncoderLayer(
            d_model=width_3,
            nhead=self.attention_heads,
            dim_feedforward=2 * width_3,
            dropout=self.attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.up_2 = nn.Conv2d(width_3, width_2, kernel_size=1)
        self.decoder_2 = _ConvBlock(width_2 + width_2, width_2, self.dropout)
        self.up_1 = nn.Conv2d(width_2, width_1, kernel_size=1)
        self.decoder_1 = _ConvBlock(width_1 + width_1, width_1, self.dropout)

        self.correction_heads = nn.ModuleDict(
            {
                variable: nn.Conv2d(
                    width_1, PROBABILITY_CATEGORIES, kernel_size=1
                )
                for variable in self.variables
            }
        )
        for head in self.correction_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, period, channel, latitude, longitude]"
            )
        if x.shape[1] != FORECAST_PERIODS:
            raise ValueError(f"x must contain exactly {FORECAST_PERIODS} periods")
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[2]} channels; expected {self.in_channels}"
            )
        if x.shape[-2] < 4 or x.shape[-1] < 4:
            raise ValueError("latitude and longitude dimensions must be at least 4")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

    def _validate_inputs(self, x: torch.Tensor, p0: torch.Tensor) -> None:
        self._validate_x(x)
        expected = (
            x.shape[0],
            len(self.variables),
            FORECAST_PERIODS,
            PROBABILITY_CATEGORIES,
            x.shape[-2],
            x.shape[-1],
        )
        if tuple(p0.shape) != expected:
            raise ValueError(f"p0 must have shape {expected}; got {tuple(p0.shape)}")
        if not p0.is_floating_point():
            raise TypeError("p0 must be a floating-point tensor")

    def _mix_periods(
        self, encoded: torch.Tensor, *, batch_size: int, periods: int
    ) -> torch.Tensor:
        _, channels, height, width = encoded.shape
        tokens = encoded.reshape(batch_size, periods, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, periods, channels
        )
        tokens = self.period_attention(tokens + self.period_position[:, :periods])
        return (
            tokens.reshape(batch_size, height, width, periods, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * periods, channels, height, width)
        )

    def _decoded_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, periods, channels, height, width = x.shape
        fields = x.reshape(batch * periods, channels, height, width)

        skip_1 = self.encoder_1(fields)
        skip_2 = self.encoder_2(self.pool(skip_1))
        encoded = self.bottleneck(self.pool(skip_2))
        encoded = self._mix_periods(
            encoded, batch_size=batch, periods=periods
        )

        decoded = F.interpolate(
            encoded,
            size=skip_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, skip_2), dim=1))

        decoded = F.interpolate(
            decoded,
            size=skip_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.up_1(decoded)
        return self.decoder_1(torch.cat((decoded, skip_1), dim=1))

    def _correction_logits(self, x: torch.Tensor) -> torch.Tensor:
        batch, periods, _, height, width = x.shape
        decoded = self._decoded_features(x)
        per_variable = [
            self.correction_heads[variable](decoded).reshape(
                batch,
                periods,
                PROBABILITY_CATEGORIES,
                height,
                width,
            )
            for variable in self.variables
        ]
        return torch.stack(per_variable, dim=1)

    def forward_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, variable, period, 5, lat, lon]`` corrections."""

        self._validate_x(x)
        return self._correction_logits(x)

    def forward(self, x: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for all configured variables."""

        self._validate_inputs(x, p0)
        correction = self._correction_logits(x)
        anchor_logits = torch.log(p0.clamp_min(ANCHOR_EPSILON))
        return torch.softmax(anchor_logits + correction, dim=3)


__all__ = ["QuestTemporalProbUNet", "TPProbUNet"]
