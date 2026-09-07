"""Inverse-design objectives on a frequency-by-44-receiver grid."""

from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .dataset import frequency_indices
else:
    from dataset import frequency_indices


class InverseObjective(nn.Module, ABC):
    n_receivers = 44

    def __init__(self, n_frequencies=201):
        super().__init__()
        self.n_frequencies = len(frequency_indices(n_frequencies))

    @staticmethod
    def prepare(request, device):
        return {name: value.to(device) for name, value in request.items()}

    @staticmethod
    def batch_size(request):
        if not request:
            raise ValueError("An objective request cannot be empty")
        return next(iter(request.values())).shape[0]

    def _validate_indices(self, response, coordinates, name):
        if response.ndim != 3 or response.shape[1:] != (
            self.n_frequencies, self.n_receivers
        ):
            raise ValueError("Response grid must match the objective frequency count and 44 ports")
        if (
            coordinates.ndim != 3
            or coordinates.shape[0] != response.shape[0]
            or coordinates.shape[2] != 2
            or coordinates.shape[0] == 0
        ):
            raise ValueError(f"{name} must have shape (batch, count, 2) with a nonempty batch")
        if coordinates.dtype != torch.int64:
            raise TypeError(f"{name} must contain torch.int64 coordinates")
        if (
            (coordinates < 0).any()
            or (coordinates[..., 0] >= self.n_frequencies).any()
            or (coordinates[..., 1] >= self.n_receivers).any()
        ):
            raise ValueError(f"{name} contains a coordinate outside the response grid")
        flat = coordinates[..., 0] * self.n_receivers + coordinates[..., 1]
        ordered = flat.sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError(f"{name} contains duplicate coordinates")
        return flat

    @staticmethod
    def gather(response, index):
        n_receivers = response.shape[2]
        flat_index = index[..., 0] * n_receivers + index[..., 1]
        return response.flatten(1).gather(1, flat_index)

    @staticmethod
    def to_db(log_response):
        return log_response.float() * (10.0 / math.log(10.0))

    @abstractmethod
    def sample(self, generator):
        pass

    @abstractmethod
    def per_sample(self, log_response, request):
        pass

    def forward(self, log_response, request):
        request = self.prepare(request, log_response.device)
        return self.per_sample(log_response, request).mean()


class HighLowObjective(InverseObjective):
    def sample(self, generator):
        flat = torch.randperm(
            self.n_frequencies * self.n_receivers,
            generator=generator,
        )[:4]
        coordinates = torch.stack(
            (flat // self.n_receivers, flat % self.n_receivers), dim=1
        )
        return {
            "high_index": coordinates[:2],
            "low_index": coordinates[2:],
        }

    def per_sample(self, log_response, request):
        high_index = request["high_index"]
        low_index = request["low_index"]
        high_flat = self._validate_indices(log_response, high_index, "high_index")
        low_flat = self._validate_indices(log_response, low_index, "low_index")
        count = high_index.shape[1] + low_index.shape[1]
        if count == 0:
            raise ValueError("Provide at least one high or low objective point")
        if count >= self.n_frequencies * self.n_receivers:
            raise ValueError("Leave at least one grid point outside the objectives for background comparison")
        ordered = torch.cat((high_flat, low_flat), dim=1).sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("A coordinate cannot be both a high and a low objective")
        response_db = self.to_db(log_response)
        batch, _, n_receivers = response_db.shape
        high = self.gather(response_db, high_index) if high_index.shape[1] else None
        low = self.gather(response_db, low_index) if low_index.shape[1] else None

        marked = torch.zeros_like(response_db.flatten(1), dtype=torch.bool)
        if high is not None:
            flat_high = high_index[..., 0] * n_receivers + high_index[..., 1]
            marked.scatter_(1, flat_high, True)
        if low is not None:
            flat_low = low_index[..., 0] * n_receivers + low_index[..., 1]
            marked.scatter_(1, flat_low, True)

        background = response_db.flatten(1)
        background_mean = (background * ~marked).sum(dim=1) / (~marked).sum(dim=1)
        zero = torch.zeros(batch, device=response_db.device)

        high_background = (
            F.relu(10.0 - high + background_mean[:, None]).mean(dim=1)
            if high is not None
            else zero
        )
        low_background = (
            F.relu(10.0 - background_mean[:, None] + low).mean(dim=1)
            if low is not None
            else zero
        )
        direct = (
            F.relu(20.0 - high.mean(dim=1) + low.mean(dim=1))
            if high is not None and low is not None
            else zero
        )
        high_rank = (
            F.relu(torch.quantile(background, 0.8, dim=1)[:, None] - high).mean(dim=1)
            if high is not None
            else zero
        )
        low_rank = (
            F.relu(low - torch.quantile(background, 0.2, dim=1)[:, None]).mean(dim=1)
            if low is not None
            else zero
        )

        terms = 2 * int(high is not None) + 2 * int(low is not None)
        terms += int(high is not None and low is not None)
        total = high_background + low_background + direct + high_rank + low_rank
        return total / terms


class RelativeDifferenceObjective(InverseObjective):
    def sample(self, generator):
        flat = torch.randperm(
            self.n_frequencies * self.n_receivers,
            generator=generator,
        )[:4]
        target_index = torch.stack(
            (flat // self.n_receivers, flat % self.n_receivers), dim=1
        )
        target_difference_db = 20.0 * torch.rand(3, generator=generator) - 10.0
        return {
            "target_index": target_index,
            "target_difference_db": target_difference_db,
        }

    def per_sample(self, log_response, request):
        coordinates = request["target_index"]
        differences = request["target_difference_db"]
        self._validate_indices(log_response, coordinates, "target_index")
        if coordinates.shape[1] < 2:
            raise ValueError("Relative objectives require at least two ordered points")
        if differences.shape != (coordinates.shape[0], coordinates.shape[1] - 1):
            raise ValueError("Provide one difference per consecutive pair in each sample")
        if not torch.isfinite(differences).all():
            raise ValueError("Requested differences must be finite")
        target_response = self.gather(
            self.to_db(log_response), coordinates
        )
        achieved_difference = target_response[:, 1:] - target_response[:, :-1]
        error = achieved_difference - differences
        error = F.relu(error.abs() - 0.5)
        pair_loss = F.smooth_l1_loss(
            error,
            torch.zeros_like(error),
            reduction="none",
        )
        return pair_loss.mean(dim=1)
