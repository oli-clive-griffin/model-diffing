import numpy as np
import torch


class FiringTracker:
    def __init__(self, activation_size: int):
        self._activation_size = activation_size
        self.examples_since_fired_A = np.zeros(activation_size, dtype=np.int64)

    def add_batch(self, hidden_BH: torch.Tensor) -> None:
        batch_size = hidden_BH.shape[0]
        firing_A = (hidden_BH.any(dim=0).detach().cpu().numpy())
        self.examples_since_fired_A[firing_A] = 0
        self.examples_since_fired_A[~firing_A] += batch_size
