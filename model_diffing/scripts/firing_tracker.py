from typing import Any

import numpy as np


class FiringTracker:
    def __init__(self, activation_size: int):
        self._activation_size = activation_size
        self.steps_since_fired_A = np.zeros(activation_size, dtype=np.int64)

    def add_batch(self, firing_BA: np.ndarray[Any, np.dtype[np.bool_]]) -> None:
        firing_A = firing_BA.any(axis=0)
        self.steps_since_fired_A[firing_A] = 0
        self.steps_since_fired_A[~firing_A] += 1
