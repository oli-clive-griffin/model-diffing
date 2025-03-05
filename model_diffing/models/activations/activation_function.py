from abc import ABC, abstractmethod

import torch

from model_diffing.utils import SaveableModule


class ActivationFunction(SaveableModule, ABC):
    @abstractmethod
    def forward(self, hidden_preact_BH: torch.Tensor) -> torch.Tensor: ...
