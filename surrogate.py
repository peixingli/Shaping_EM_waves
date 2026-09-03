"""Forward surrogate for the programmable cavity."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoltageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(1, 512)
        self.positional_encoding = nn.Parameter(torch.randn(23, 512))
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, voltage):
        tokens = self.input_projection(voltage.unsqueeze(-1))
        tokens = tokens + self.positional_encoding
        return self.encoder(tokens).mean(dim=1)


class CavitySurrogate(nn.Module):
    n_voltages = 23
    n_frequencies = 101
    n_receivers = 44

    def __init__(self):
        super().__init__()
        self.grid_dim = 256

        self.branch = VoltageEncoder()
        self.freq_embedding = nn.Embedding(self.n_frequencies, 256)
        self.register_buffer(
            "normalized_frequencies",
            torch.linspace(-1.0, 1.0, self.n_frequencies).reshape(-1, 1),
        )
        self.frequency_coordinate_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 256),
        )
        self.recv_embedding = nn.Embedding(self.n_receivers, 256)
        self.trunk_mlp = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
        self.branch_to_grid = nn.Linear(512, 256)

        receiver_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.receiver_encoder = nn.TransformerEncoder(receiver_layer, num_layers=2)

        frequency_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.freq_encoder = nn.TransformerEncoder(frequency_layer, num_layers=1)
        self.readout_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

        self.register_buffer(
            "_all_freq_idx", torch.arange(self.n_frequencies), persistent=False
        )
        self.register_buffer(
            "_all_recv_idx", torch.arange(self.n_receivers), persistent=False
        )

    def _readout(self, grid, branch):
        first = self.readout_mlp[0]
        grid_weight = first.weight[:, : self.grid_dim]
        branch_weight = first.weight[:, self.grid_dim :]
        hidden = F.linear(grid, grid_weight, first.bias)
        hidden = hidden + F.linear(branch, branch_weight)[:, None, None, :]
        return self.readout_mlp[2](F.relu(hidden)).squeeze(-1)

    def forward(self, voltage):
        branch = self.branch(voltage)
        frequency = self.freq_embedding(self._all_freq_idx)
        frequency = frequency + self.frequency_coordinate_mlp(
            self.normalized_frequencies
        )
        receiver = self.recv_embedding(self._all_recv_idx)

        grid = self.trunk_mlp(frequency[:, None, :] + receiver[None, :, :])
        grid = grid.unsqueeze(0) + self.branch_to_grid(branch)[:, None, None, :]

        batch, n_frequency, n_receiver, width = grid.shape
        grid = grid.permute(0, 2, 1, 3).reshape(
            batch * n_receiver, n_frequency, width
        )
        grid = self.freq_encoder(grid)
        grid = grid.reshape(batch, n_receiver, n_frequency, width)
        grid = grid.permute(0, 2, 1, 3)
        grid = grid.reshape(batch * n_frequency, n_receiver, width)
        grid = self.receiver_encoder(grid)
        grid = grid.reshape(batch, n_frequency, n_receiver, width)

        log_response = self._readout(grid, branch)
        return F.log_softmax(log_response.flatten(1), dim=1).view_as(log_response)


def build_surrogate(
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> CavitySurrogate:
    model = CavitySurrogate()
    if checkpoint is not None:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
    return model.to(device)
