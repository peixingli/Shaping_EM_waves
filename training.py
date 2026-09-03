"""Training losses, schedules, and routines."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset

from .amortized_optimizer import AmortizedOptimizer
from .objectives import InverseObjective


@dataclass(frozen=True)
class SurrogateTraining:
    epochs: int = 20
    learning_rate: float = 5e-4
    warmup_epochs: int = 2


@dataclass(frozen=True)
class OptimizerTraining:
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    truncation_steps: int = 5


class SurrogateLoss(nn.Module):
    def forward(self, log_prediction, target):
        peak = target.amax(dim=(1, 2), keepdim=True)
        target = target.clamp_min(peak * 1e-3)
        target = target / target.sum(dim=(1, 2), keepdim=True)
        return F.mse_loss(log_prediction, target.log())


class ObjectiveDataset(Dataset):
    def __init__(self, objective: InverseObjective, size, seed=42):
        self.objective = objective
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        return self.objective.sample(generator)


def _cosine_schedule(adam, warmup_epochs, total_epochs):
    def scale(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(adam, scale)


def train_surrogate(
    model,
    data_loader,
    device="cpu",
    config=SurrogateTraining(),
):
    model.to(device).train().requires_grad_(True)
    objective = SurrogateLoss()
    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _cosine_schedule(adam, config.warmup_epochs, config.epochs)
    history = []

    for _ in range(config.epochs):
        total = 0.0
        samples = 0
        for voltage, target in data_loader:
            voltage = voltage.to(device)
            target = target.to(device)
            adam.zero_grad(set_to_none=True)
            loss = objective(model(voltage), target)
            loss.backward()
            adam.step()
            total += float(loss.detach()) * voltage.shape[0]
            samples += voltage.shape[0]
        scheduler.step()
        history.append(total / samples)
    return history


def train_amortized_optimizer(
    model: AmortizedOptimizer,
    surrogate,
    objective: InverseObjective,
    data_loader,
    device="cpu",
    config=OptimizerTraining(),
):
    model.to(device).train()
    surrogate.to(device).eval().requires_grad_(False)
    objective.to(device)
    adam = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []

    for _ in range(config.epochs):
        total = 0.0
        samples = 0
        for request in data_loader:
            request = objective.prepare(request, device)
            batch_size = objective.batch_size(request)
            phase = model.initial_phase(batch_size, device).requires_grad_(True)
            momentum = torch.zeros_like(phase)

            for step in range(model.steps):
                prediction = surrogate(model.voltages(phase))
                loss = objective(prediction, request)
                gradient = torch.autograd.grad(loss, phase)[0].detach()
                momentum = (
                    model.momentum_decay * momentum
                    + (1.0 - model.momentum_decay) * gradient
                )
                phase = phase + model(phase, gradient, momentum.detach(), step)
                truncate = config.truncation_steps and (
                    (step + 1) % config.truncation_steps == 0
                    and step + 1 < model.steps
                )
                if truncate:
                    phase = phase.detach().requires_grad_(True)
                    momentum = momentum.detach()

            prediction = surrogate(model.voltages(phase))
            loss = objective(prediction, request)
            adam.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            adam.step()

            total += float(loss.detach()) * batch_size
            samples += batch_size
        history.append(total / samples)
    return history
