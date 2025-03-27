from abc import ABC, abstractmethod

import torch

from model_diffing.saveable_module import SaveableModule


class ActivationFunction(SaveableModule, ABC):
    @abstractmethod
    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor: ...
