"""
Implementation of ``nn.Modules`` for N-Beats model.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear(input_size, output_size, bias=True, dropout: int = None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


def linspace(backcast_length: int, forecast_length: int, centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
        dropout=0.1,
    ):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        fc_stack = [
            nn.Linear(backcast_length, units),
            nn.ReLU(),
        ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class NBEATSSeasonalBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim=None,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
        min_period=1,
        dropout=0.1,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        self.min_period = min_period

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=False)

        p1, p2 = (thetas_dim // 2, thetas_dim // 2) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1)
        s1_b = torch.tensor(
            [np.cos(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32
        )  # H/2-1
        s2_b = torch.tensor(
            [np.sin(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32
        )
        self.register_buffer("S_backcast", torch.cat([s1_b, s2_b]))

        s1_f = torch.tensor(
            [np.cos(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32
        )  # H/2-1
        s2_f = torch.tensor(
            [np.sin(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32
        )
        self.register_buffer("S_forecast", torch.cat([s1_f, s2_f]))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        amplitudes_backward = self.theta_b_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = self.theta_f_fc(x)
        forecast = amplitudes_forward.mm(self.S_forecast)

        return backcast, forecast

    def get_frequencies(self, n):
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)


class NBEATSEnhancedSeasonalBlock(NBEATSBlock):
    """
    Model spectrum of frequencies as mixed Gaussian distribution.
    """

    def __init__(
        self,
        units,
        thetas_dim=32,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
    ):

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
        )
        assert (
            self.thetas_dim % 4 == 0
        ), "Theta dimensions must be dividable by 4 for amplitude, width, frequency and phase for each gaussian"

        backcast_linspace = torch.arange(backcast_length, dtype=torch.float)
        forecast_linspace = torch.arange(backcast_length, backcast_length + forecast_length, dtype=torch.float)
        frequencies = 1.0 / torch.arange(2.0, (backcast_length + forecast_length) * 2 + 1, dtype=torch.float)

        cos_b = torch.cos(2 * np.pi * frequencies.unsqueeze(-1) * backcast_linspace)
        sin_b = torch.sin(2 * np.pi * frequencies.unsqueeze(-1) * backcast_linspace)
        self.register_buffer("s_backcast", torch.cat([sin_b, cos_b]))

        sin_f = torch.sin(2 * np.pi * frequencies.unsqueeze(-1) * forecast_linspace)
        cos_f = torch.cos(2 * np.pi * frequencies.unsqueeze(-1) * forecast_linspace)
        self.register_buffer("s_forecast", torch.cat([sin_f, cos_f]))
        self.register_buffer("frequencies", frequencies)
        self.frequency_norm = nn.LayerNorm(self.thetas_dim // 4)
        self.width_norm = nn.LayerNorm(self.thetas_dim // 4)
        self.softplus = nn.Softplus(beta=10)
        self.sigmoid = nn.Sigmoid()

    def calculate_amplitudes(self, coeff):
        params = coeff.view(
            coeff.size(0), -1, 4
        )  # n_samples x n_gaussians x (amplitudes_sin, amplitudes_cos, width, frequency)

        # extract parameters
        center_frequencies = 1.0 / (
            1.0
            + self.sigmoid(self.frequency_norm(params[..., 3])).unsqueeze(-1)
            * (self.backcast_length + self.forecast_length)
        )
        gaussian_sin_amplitudes = params[..., 0].unsqueeze(-1)
        gaussian_cos_amplitudes = params[..., 1].unsqueeze(-1)
        inverse_widths = center_frequencies / self.softplus(self.width_norm(params[..., 2]).unsqueeze(-1)).clamp(
            min=1e-6
        )

        # calculate amplitudes
        exponent = torch.pow((self.frequencies - center_frequencies) * inverse_widths, 2) / 2
        normalized_amplitudes = F.softmax(exponent, dim=-1)
        amplitude_sin = (normalized_amplitudes * gaussian_sin_amplitudes).sum(1)
        amplitude_cos = (normalized_amplitudes * gaussian_cos_amplitudes).sum(1)
        amplitudes = torch.cat([amplitude_sin, amplitude_cos], dim=-1)

        return amplitudes

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)

        # backwards
        coeff_backward = self.theta_b_fc(x)
        amplitudes_backward = self.calculate_amplitudes(coeff_backward)
        backcast = amplitudes_backward.mm(self.s_backcast)

        # forwards
        if self.share_thetas:
            amplitudes_forward = amplitudes_backward
        else:
            coeff_forward = self.theta_f_fc(x)
            amplitudes_forward = self.calculate_amplitudes(coeff_forward)
        forecast = amplitudes_forward.mm(self.s_forecast)

        return backcast, forecast


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
        norm = np.sqrt(forecast_length / thetas_dim)  # ensure range of predictions is comparable to input

        coefficients = torch.tensor([backcast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_backcast", coefficients * norm)

        coefficients = torch.tensor([forecast_linspace ** i for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer("T_forecast", coefficients * norm)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        coefficients_backward = self.theta_b_fc(x)
        if self.share_thetas:
            coefficients_forward = coefficients_backward
        else:
            coefficients_forward = self.theta_f_fc(x)
        backcast = coefficients_backward.mm(self.T_backcast)
        forecast = coefficients_forward.mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
