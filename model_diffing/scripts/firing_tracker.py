import numpy as np
import torch


class FiringTracker:
    def __init__(self, activation_size: int):
        self._activation_size = activation_size
        self.examples_since_fired_A = np.zeros(activation_size, dtype=np.int64)

    def add_batch(self, hidden_BH: torch.Tensor) -> None:
        firing_BA = (hidden_BH.detach().cpu().numpy() > 0)
        assert firing_BA.shape[1] == self._activation_size
        assert firing_BA.dtype == np.bool_, "firing_BA must be a boolean array"
        firing_A = firing_BA.any(axis=0)
        self.examples_since_fired_A[firing_A] = 0
        self.examples_since_fired_A[~firing_A] += firing_BA.shape[0]
