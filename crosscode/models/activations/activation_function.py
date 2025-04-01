from abc import ABC, abstractmethod

import torch

from crosscode.saveable_module import SaveableModule


class ActivationFunction(SaveableModule, ABC):
    @abstractmethod
    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor: ...
