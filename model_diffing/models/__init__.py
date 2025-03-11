from .activations import (
    ACTIVATIONS_MAP,
    BatchTopkActivation,
    JumpReLUActivation,
    ReLUActivation,
    TopkActivation,
)
from .crosscoder import AcausalCrosscoder
from model_diffing.utils import SaveableModule
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TModel = TypeVar("TModel", bound=SaveableModule)


class InitStrategy(ABC, Generic[TModel]):
    @abstractmethod
    def init_weights(self, cc: TModel) -> None: ...


__all__ = [
    "AcausalCrosscoder",
    "ACTIVATIONS_MAP",
    "BatchTopkActivation",
    "JumpReLUActivation",
    "ReLUActivation",
    "TopkActivation",
]