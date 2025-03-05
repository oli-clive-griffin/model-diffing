from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from model_diffing.utils import SaveableModule

TModel = TypeVar("TModel", bound=SaveableModule)


class InitStrategy(ABC, Generic[TModel]):
    @abstractmethod
    def init_weights(self, cc: TModel) -> None: ...
