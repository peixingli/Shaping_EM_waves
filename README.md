# Codebase for the manuscript "Shaping Electromagnetic Waves in Complex Reverberant Environment with Learning Driven Metasurface"

This package contains the surrogate and the gradient-conditioned amortized optimizer. The surrogate maps 23 voltages to a 201-frequency by 44-receiver response grid and returns log-scale response.

Dependencies required: torch, numpy, h5py

Two inverse-design objectives demonstrated in the manuscript are included. Each uses the same optimizer architecture but is trained and saved as a separate model.

## 1. Setup

Adjust batch sizes to available memory.

```python
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from amortized_optimizer import build_optimizer
from dataset import build_data_loaders
from objectives import HighLowObjective, RelativeDifferenceObjective
from surrogate import build_surrogate
from training import ObjectiveDataset, train_amortized_optimizer, train_surrogate

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
surrogate_batch_size = 4
optimizer_batch_size = 4
n_frequencies = 201
data_path = "path/to/cavity_data.h5"
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
```

## 2. Train the surrogate

`dataset.py` loads HDF5 datasets `V` and `S21`, which are voltage commands and measured S21's in dB. The loader reads the dataset and prepares the training batches.

Surrogate training uses tempered KL divergence, `KL(data_alpha || prediction_alpha)`, averaged over samples. Each distribution is raised to `alpha` and renormalized over the selected frequency-by-port grid. `SurrogateTraining.alpha` defaults to 0.5. Prediction tempering is computed in log space.

```python
train_loader, validation_loader = build_data_loaders(
    data_path, batch_size=surrogate_batch_size,
    n_frequencies=n_frequencies, seed=seed,
)
surrogate = build_surrogate(device=device, n_frequencies=n_frequencies)
history = train_surrogate(surrogate, train_loader, device=device)
torch.save(surrogate.state_dict(), checkpoint_dir / "surrogate.pth")
```

The default split is 90% training and 10% validation, with seed 42.

`n_frequencies` defaults to 201. You may subsample the frequency grid by setting it lower for a faster run.

## 3. Train the amortized optimizers

Coordinates are `(frequency, receiver)` indices, with receiver indices from 0 to 43. The high/low training sampler selects two locations of each type. The relative-difference sampler selects four ordered locations and consecutive differences uniformly from -10 to 10 dB, with a 0.5 dB loss tolerance. `ObjectiveDataset` samples requests, and `train_amortized_optimizer` provides the shared training loop.

Train each optimizer against the trained surrogate, then save its weights.

```python
high_low = HighLowObjective(n_frequencies=n_frequencies)
relative = RelativeDifferenceObjective(n_frequencies=n_frequencies)
training_requests = 1024

for name, objective in (("high_low", high_low), ("relative", relative)):
    requests = ObjectiveDataset(objective, size=training_requests, seed=seed)
    request_loader = DataLoader(
        requests, batch_size=optimizer_batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = build_optimizer(device=device)
    history = train_amortized_optimizer(
        optimizer, surrogate, objective, request_loader, device=device,
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / f"{name}_optimizer.pth")
```

## 4. Specify objectives and run inverse design

Manual requests are not restricted to two points. High/low requests may contain any number of distinct points, including high-only or low-only requests. For an unused group, use `torch.empty((batch_size, 0, 2), dtype=torch.long)`. Relative requests accept any number of distinct ordered points (at least two), with one requested difference per consecutive pair. Requests in a batch must share the same point counts.

```python
surrogate = build_surrogate(
    checkpoint_dir / "surrogate.pth", device=device, n_frequencies=n_frequencies,
)
high_low_optimizer = build_optimizer(
    checkpoint_dir / "high_low_optimizer.pth", device=device,
)
high_low_request = {
    "high_index": torch.tensor([[[50, 20], [60, 30], [70, 35]]]),
    "low_index": torch.tensor([[[20, 3], [30, 10]]]),
}
voltage, response, loss = high_low_optimizer.solve(
    surrogate, high_low, high_low_request,
)

relative_optimizer = build_optimizer(
    checkpoint_dir / "relative_optimizer.pth", device=device,
)
relative_request = {
    "target_index": torch.tensor([[[20, 8], [45, 18], [75, 35]]]),
    "target_difference_db": torch.tensor([[8.0, -3.0]]),
}
voltage, response, loss = relative_optimizer.solve(
    surrogate, relative, relative_request,
)
```
