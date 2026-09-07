"""Load cavity measurements and prepare training batches."""

from operator import index

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def frequency_indices(n_frequencies=201):
    n_frequencies = index(n_frequencies)
    if not 1 <= n_frequencies <= 201:
        raise ValueError("n_frequencies must be between 1 and 201")
    if n_frequencies == 1:
        return [0]
    return [round(i * 200 / (n_frequencies - 1)) for i in range(n_frequencies)]


def load_dataset(path, n_frequencies=201):
    selected = frequency_indices(n_frequencies)
    with h5py.File(path, "r") as data:
        voltage = np.asarray(data["V"], dtype=np.float32)
        response = np.asarray(data["S21"], dtype=np.float32)

    if voltage.ndim == 2 and voltage.shape[0] == 23 and voltage.shape[1] != 23:
        voltage = voltage.T
    if voltage.ndim != 2 or voltage.shape[1] != 23:
        raise ValueError("V must have shape (samples, 23) or (23, samples)")

    samples = len(voltage)
    if response.shape == (44, 201, samples):
        response = response.transpose(2, 1, 0)
    if response.shape != (samples, 201, 44):
        raise ValueError("S21 must have shape (samples, 201, 44) or (44, 201, samples)")

    response = response[:, selected, :]

    if not np.isfinite(response).all():
        raise ValueError("S21 must contain finite dB values")
    response = response - response.max(axis=(1, 2), keepdims=True)
    response = np.power(10.0, response / 10.0)
    response = response / response.sum(axis=(1, 2), keepdims=True)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(voltage)),
        torch.from_numpy(np.ascontiguousarray(response)),
    )
    dataset.frequency_indices = tuple(selected)
    return dataset


def build_data_loaders(
    path, batch_size=96, train_fraction=0.9, seed=42, n_frequencies=201
):
    dataset = load_dataset(path, n_frequencies=n_frequencies)
    train_size = int(train_fraction * len(dataset))
    if not 0 < train_size < len(dataset):
        raise ValueError("The split must leave at least one sample in each subset")
    training, validation = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation, batch_size=batch_size)
    return train_loader, validation_loader
