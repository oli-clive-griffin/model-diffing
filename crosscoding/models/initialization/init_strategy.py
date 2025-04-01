from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from crosscoding.saveable_module import SaveableModule

TModel = TypeVar("TModel", bound=SaveableModule)


class InitStrategy(Generic[TModel], ABC):
    @abstractmethod
    def init_weights(self, cc: TModel) -> None: ...
