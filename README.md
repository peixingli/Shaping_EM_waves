# Publication code

This package contains the surrogate and the gradient-conditioned
amortized optimizer. The surrogate maps 23 voltages to a 101-frequency by
44-receiver response grid and returns log-scale response.

Two inverse-design objectives are included. Each uses the same optimizer
architecture but is trained and saved as a separate model.

```python
import torch

from publication_code import (
    HighLowObjective,
    RelativeDifferenceObjective,
    build_optimizer,
    build_surrogate,
)

surrogate = build_surrogate("path/to/surrogate.pth")

high_low = HighLowObjective()
high_low_optimizer = build_optimizer("path/to/high_low_optimizer.pth")
high_low_request = {
    "high_index": torch.tensor([[[50, 20], [60, 30]]]),
    "low_index": torch.tensor([[[20, 3], [30, 10]]]),
}
voltage, response, loss = high_low_optimizer.solve(
    surrogate, high_low, high_low_request
)

relative = RelativeDifferenceObjective()
relative_optimizer = build_optimizer("path/to/relative_optimizer.pth")
relative_request = {
    "target_index": torch.tensor([[[20, 8], [45, 18], [75, 35]]]),
    "target_difference_db": torch.tensor([[8.0, -3.0]]),
}
voltage, response, loss = relative_optimizer.solve(
    surrogate, relative, relative_request
)
```

Coordinates are `(frequency, receiver)` indices, with receiver indices from 0
to 43. The high/low training sampler selects two locations of each type. The
relative-difference sampler selects four ordered locations and consecutive
differences uniformly from -10 to 10 dB, with a 0.5 dB loss tolerance.
`ObjectiveDataset` samples requests, and `train_amortized_optimizer` provides
the shared training loop.
