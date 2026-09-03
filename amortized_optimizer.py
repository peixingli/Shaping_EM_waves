"""Gradient-conditioned amortized optimizer for inverse design."""

import math
from pathlib import Path

import torch
import torch.nn as nn

from .objectives import InverseObjective


class VoltageMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mid", torch.tensor(14.0))
        self.register_buffer("amplitude", torch.tensor(5.88))

    def forward(self, phase):
        return self.mid + self.amplitude * torch.sin(phase)


class CoordinateUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, -4.0]))

    @staticmethod
    def _normalize(value):
        scale = value.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        return torch.tanh(value / scale)

    def forward(self, phase, gradient, momentum, step):
        batch, dimensions = phase.shape
        progress = torch.full_like(phase, step / 9.0)
        features = torch.stack(
            (
                torch.sin(phase),
                self._normalize(gradient),
                self._normalize(momentum),
                progress,
            ),
            dim=-1,
        )
        output = self.net(features.reshape(batch * dimensions, 4))
        output = output.reshape(batch, dimensions, 2)
        local_step = 0.2 * torch.tanh(output[..., 0])
        jump = 0.9 * math.pi * torch.sigmoid(output[..., 1])
        return local_step - jump * gradient.sign()


class AmortizedOptimizer(nn.Module):
    n_voltages = 23
    steps = 10
    momentum_decay = 0.9

    def __init__(self):
        super().__init__()
        self.mapper = VoltageMap()
        self.updater = CoordinateUpdate()

    def initial_phase(self, batch_size, device):
        sample = torch.rand(batch_size, self.n_voltages, device=device)
        return torch.asin(2.0 * sample - 1.0)

    def voltages(self, phase):
        return self.mapper(phase)

    def forward(self, phase, gradient, momentum, step):
        return self.updater(phase, gradient, momentum, step)

    def solve(self, surrogate, objective: InverseObjective, request):
        device = next(surrogate.parameters()).device
        request = objective.prepare(request, device)
        batch_size = objective.batch_size(request)
        phase = self.initial_phase(batch_size, device).requires_grad_(True)
        momentum = torch.zeros_like(phase)

        self.eval()
        surrogate.eval()
        objective.to(device)
        with torch.enable_grad():
            for step in range(self.steps):
                prediction = surrogate(self.voltages(phase))
                loss = objective(prediction, request)
                gradient = torch.autograd.grad(loss, phase)[0].detach()
                momentum = (
                    self.momentum_decay * momentum
                    + (1.0 - self.momentum_decay) * gradient
                )
                update = self(phase, gradient, momentum, step)
                phase = (phase + update).detach().requires_grad_(True)

        voltage = self.voltages(phase).detach()
        with torch.no_grad():
            prediction = surrogate(voltage)
            loss = objective(prediction, request)
        return voltage, prediction, loss


def build_optimizer(
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> AmortizedOptimizer:
    model = AmortizedOptimizer()
    if checkpoint is not None:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    return model.to(device)
